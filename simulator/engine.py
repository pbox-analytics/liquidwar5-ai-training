"""
Liquid War game engine implemented as batched PyTorch tensor operations.

Core mechanics:
1. Grid map with walls (passable/impassable cells)
2. Fighters on the grid, each belonging to a team (0-5)
3. Cursors (one per team) that the AI moves each tick
4. Gradient propagation: distance field from each cursor spreads
   through the grid via iterative relaxation (like BFS)
5. Fighters follow the gradient toward their team's cursor
6. Combat: when fighters from different teams are adjacent,
   the outnumbered one takes damage
7. Fighters that reach 0 health switch teams (get captured)

All operations are batched across B games simultaneously.
"""

import torch
import torch.nn.functional as F


# Direction offsets: 12 directions matching the C code
# DIR_NNE, DIR_NE, DIR_ENE, DIR_ESE, DIR_SE, DIR_SSE,
# DIR_SSW, DIR_SW, DIR_WSW, DIR_WNW, DIR_NW, DIR_NNW
DIR_DY = torch.tensor([-1, -1,  0,  0,  1,  1,  1,  1,  0,  0, -1, -1])
DIR_DX = torch.tensor([ 0,  1,  1,  1,  1,  0,  0, -1, -1, -1, -1,  0])

# 8-connected neighbor offsets for combat and movement
NEIGHBOR_DY = torch.tensor([-1, -1, -1,  0, 0,  1, 1, 1])
NEIGHBOR_DX = torch.tensor([-1,  0,  1, -1, 1, -1, 0, 1])

MAX_HEALTH = 16384
GRADIENT_INF = 999999


class LiquidWarEngine:
    """Batched Liquid War game engine on GPU.

    Runs B games simultaneously using tensor operations.

    Args:
        batch_size: Number of games to run in parallel.
        height: Grid height in cells.
        width: Grid width in cells.
        num_teams: Number of teams (2-6).
        fighters_per_team: Initial fighters per team.
        device: 'cuda' or 'cpu'.
    """

    def __init__(self, batch_size=256, height=120, width=160,
                 num_teams=6, fighters_per_team=2000,
                 device='cuda'):
        self.B = batch_size
        self.H = height
        self.W = width
        self.T = num_teams
        self.device = device
        self.fighters_per_team = fighters_per_team
        self.tick = 0

        # Pre-compute neighbor offsets on device
        self.dir_dy = DIR_DY.to(device)
        self.dir_dx = DIR_DX.to(device)
        self.nb_dy = NEIGHBOR_DY.to(device)
        self.nb_dx = NEIGHBOR_DX.to(device)

    def reset(self, walls=None):
        """Initialize game state for all B games.

        Args:
            walls: (B, H, W) bool tensor. True = wall (impassable).
                   If None, generates random maps.

        Returns:
            State dict with all tensors.
        """
        B, H, W, T = self.B, self.H, self.W, self.T
        dev = self.device

        # Map: True = wall
        if walls is None:
            walls = self._generate_random_maps()
        self.walls = walls.to(dev)
        self.passable = ~self.walls  # True = passable

        # Team ownership: -1 = empty, 0..T-1 = team
        self.team_grid = torch.full((B, H, W), -1, dtype=torch.int8,
                                    device=dev)

        # Fighter health: 0 = no fighter
        self.health = torch.zeros(B, H, W, dtype=torch.int16, device=dev)

        # Gradient: distance from each team's cursor (lower = closer)
        self.gradient = torch.full((B, T, H, W), GRADIENT_INF,
                                   dtype=torch.int32, device=dev)

        # Cursor positions: (B, T, 2) -> [y, x]
        self.cursor_pos = torch.zeros(B, T, 2, dtype=torch.long,
                                      device=dev)

        # Team alive flags
        self.team_alive = torch.ones(B, T, dtype=torch.bool, device=dev)
        if T < 6:
            self.team_alive[:, T:] = False

        # Place fighters and cursors
        self._place_teams()

        self.tick = 0
        return self._get_state()

    def step(self, cursor_actions=None):
        """Run one game tick for all B games.

        Args:
            cursor_actions: (B, T, 2) delta [dy, dx] for cursor movement.
                           Values in {-1, 0, 1}. None = no movement.

        Returns:
            state: Updated state dict.
            done: (B,) bool tensor, True if game is over.
            info: Dict with per-game stats.
        """
        # 1. Move cursors
        if cursor_actions is not None:
            self._move_cursors(cursor_actions)

        # 2. Update cursor gradient values
        self._seed_gradients()

        # 3. Spread gradient (iterative relaxation)
        self._spread_gradient()

        # 4. Move fighters along gradient
        self._move_fighters()

        # 5. Combat
        self._resolve_combat()

        # 6. Check for eliminated teams
        self._check_eliminations()

        self.tick += 1

        # Game is done when only 1 team remains or tick limit reached
        teams_remaining = self.team_alive.sum(dim=1)
        done = teams_remaining <= 1

        return self._get_state(), done, self._get_info()

    def _generate_random_maps(self):
        """Generate random maps with walls."""
        B, H, W = self.B, self.H, self.W
        # Start with no walls
        walls = torch.zeros(B, H, W, dtype=torch.bool)
        # Add border walls
        walls[:, 0, :] = True
        walls[:, -1, :] = True
        walls[:, :, 0] = True
        walls[:, :, -1] = True
        # Add random interior walls (20% density)
        interior = torch.rand(B, H - 2, W - 2) < 0.15
        walls[:, 1:-1, 1:-1] = interior
        return walls

    def _place_teams(self):
        """Place fighters for each team in different regions of the map."""
        B, H, W, T = self.B, self.H, self.W, self.T

        # Divide map into T vertical strips for initial placement
        strip_w = (W - 2) // T

        for t in range(T):
            if not self.team_alive[0, t]:
                continue

            x_start = 1 + t * strip_w
            x_end = x_start + strip_w
            # Cursor in center of strip
            cy = H // 2
            cx = (x_start + x_end) // 2
            self.cursor_pos[:, t, 0] = cy
            self.cursor_pos[:, t, 1] = cx

            # Place fighters randomly in the strip
            placed = torch.zeros(B, dtype=torch.long, device=self.device)
            target = self.fighters_per_team

            for _ in range(target * 3):  # over-sample to handle walls
                y = torch.randint(2, H - 2, (B,), device=self.device)
                x = torch.randint(x_start, min(x_end, W - 1), (B,),
                                  device=self.device)

                can_place = (self.passable[torch.arange(B), y, x]
                             & (self.team_grid[torch.arange(B), y, x] == -1)
                             & (placed < target))

                batch_idx = torch.where(can_place)[0]
                if len(batch_idx) == 0:
                    continue

                self.team_grid[batch_idx, y[batch_idx],
                               x[batch_idx]] = t
                self.health[batch_idx, y[batch_idx],
                            x[batch_idx]] = MAX_HEALTH
                placed[batch_idx] += 1

    def _move_cursors(self, actions):
        """Move cursors by delta, clamping to passable cells."""
        new_pos = self.cursor_pos + actions
        # Clamp to grid bounds
        new_pos[:, :, 0].clamp_(1, self.H - 2)
        new_pos[:, :, 1].clamp_(1, self.W - 2)
        # Only move if target is passable
        for t in range(self.T):
            ny = new_pos[:, t, 0]
            nx = new_pos[:, t, 1]
            passable = self.passable[torch.arange(self.B), ny, nx]
            alive = self.team_alive[:, t]
            mask = passable & alive
            self.cursor_pos[:, t, 0] = torch.where(
                mask, ny, self.cursor_pos[:, t, 0])
            self.cursor_pos[:, t, 1] = torch.where(
                mask, nx, self.cursor_pos[:, t, 1])

    def _seed_gradients(self):
        """Set gradient = 0 at each cursor position."""
        for t in range(self.T):
            cy = self.cursor_pos[:, t, 0]
            cx = self.cursor_pos[:, t, 1]
            self.gradient[torch.arange(self.B), t, cy, cx] = 0

    def _spread_gradient(self, iterations=3):
        """Spread gradient via iterative relaxation.

        Each iteration, each cell adopts min(current, neighbor + 1)
        for all 8 neighbors. This is equivalent to BFS but parallelizable.
        More iterations = farther spread per tick.
        """
        B, T, H, W = self.B, self.T, self.H, self.W

        # Pad gradient with INF for boundary handling
        for _ in range(iterations):
            padded = F.pad(self.gradient.float(),
                           (1, 1, 1, 1), value=GRADIENT_INF)

            # Check all 8 neighbors, take minimum + 1
            min_neighbor = padded[:, :, 1:-1, 1:-1].clone()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                shifted = padded[:, :, 1 + dy:H + 1 + dy,
                                 1 + dx:W + 1 + dx]
                candidate = shifted + 1
                min_neighbor = torch.minimum(min_neighbor, candidate)

            # Only update passable cells
            passable_mask = self.passable.unsqueeze(1).expand_as(
                self.gradient)
            self.gradient = torch.where(
                passable_mask,
                torch.minimum(self.gradient,
                              min_neighbor.to(torch.int32)),
                self.gradient)

    def _move_fighters(self):
        """Move each fighter toward its team's cursor (lowest gradient)."""
        B, H, W = self.B, self.H, self.W

        has_fighter = self.health > 0
        if not has_fighter.any():
            return

        # For each cell with a fighter, find the neighbor with lowest
        # gradient for that fighter's team
        padded_team = F.pad(self.team_grid.float(), (1, 1, 1, 1),
                            value=-1)
        padded_health = F.pad(self.health.float(), (1, 1, 1, 1),
                              value=0)

        # Build gradient lookup per fighter's team
        # This is the key operation: each fighter follows ITS team gradient
        new_team = self.team_grid.clone()
        new_health = self.health.clone()

        # Process fighters that want to move
        # For simplicity, we move a random 50% each tick to avoid
        # all-move-at-once artifacts (like the C code alternates)
        move_mask = has_fighter & (torch.rand(B, H, W,
                                              device=self.device) < 0.5)

        if not move_mask.any():
            return

        # For each cell with a moving fighter, find best neighbor
        best_dy = torch.zeros(B, H, W, dtype=torch.long,
                              device=self.device)
        best_dx = torch.zeros(B, H, W, dtype=torch.long,
                              device=self.device)
        best_grad = torch.full((B, H, W), GRADIENT_INF,
                               dtype=torch.int32, device=self.device)

        # Get each fighter's team
        fighter_team = self.team_grid.long()  # (B, H, W)
        fighter_team_clamped = fighter_team.clamp(0, self.T - 1)

        # Get this fighter's gradient at current position
        current_grad = self.gradient[
            torch.arange(B).view(B, 1, 1).expand(B, H, W),
            fighter_team_clamped,
            torch.arange(H).view(1, H, 1).expand(B, H, W),
            torch.arange(W).view(1, 1, W).expand(B, H, W),
        ]

        # Check each neighbor direction
        for i in range(8):
            dy = self.nb_dy[i].item()
            dx = self.nb_dx[i].item()

            # Neighbor coordinates (clamped)
            ny = (torch.arange(H, device=self.device) + dy).clamp(0, H - 1)
            nx = (torch.arange(W, device=self.device) + dx).clamp(0, W - 1)

            ny_grid = ny.view(1, H, 1).expand(B, H, W)
            nx_grid = nx.view(1, 1, W).expand(B, H, W)

            # Is neighbor passable and empty?
            nb_passable = self.passable[
                torch.arange(B).view(B, 1, 1).expand(B, H, W),
                ny_grid, nx_grid]
            nb_empty = self.health[
                torch.arange(B).view(B, 1, 1).expand(B, H, W),
                ny_grid, nx_grid] == 0

            # Gradient at neighbor for this fighter's team
            nb_grad = self.gradient[
                torch.arange(B).view(B, 1, 1).expand(B, H, W),
                fighter_team_clamped,
                ny_grid, nx_grid]

            # Is this the best neighbor so far?
            better = nb_passable & nb_empty & (nb_grad < best_grad)
            best_grad = torch.where(better, nb_grad, best_grad)
            best_dy = torch.where(better, torch.tensor(dy, device=self.device),
                                  best_dy)
            best_dx = torch.where(better, torch.tensor(dx, device=self.device),
                                  best_dx)

        # Only move if we found a better position
        should_move = move_mask & (best_grad < current_grad)

        if not should_move.any():
            return

        # Compute target positions
        src_y = torch.arange(H, device=self.device).view(1, H, 1).expand(
            B, H, W)
        src_x = torch.arange(W, device=self.device).view(1, 1, W).expand(
            B, H, W)
        dst_y = (src_y + best_dy).clamp(0, H - 1)
        dst_x = (src_x + best_dx).clamp(0, W - 1)

        # Move fighters: clear source, set destination
        # We need to be careful about conflicts (two fighters moving
        # to same cell). Use scatter with 'first wins' approach.
        move_idx = torch.where(should_move)
        if len(move_idx[0]) == 0:
            return

        b_idx = move_idx[0]
        sy = move_idx[1]
        sx = move_idx[2]
        ty = dst_y[b_idx, sy, sx]
        tx = dst_x[b_idx, sy, sx]

        # Check destination is still empty (handle conflicts)
        still_empty = new_health[b_idx, ty, tx] == 0
        b_move = b_idx[still_empty]
        sy_move = sy[still_empty]
        sx_move = sx[still_empty]
        ty_move = ty[still_empty]
        tx_move = tx[still_empty]

        # Execute moves
        new_team[b_move, ty_move, tx_move] = self.team_grid[
            b_move, sy_move, sx_move]
        new_health[b_move, ty_move, tx_move] = self.health[
            b_move, sy_move, sx_move]
        new_team[b_move, sy_move, sx_move] = -1
        new_health[b_move, sy_move, sx_move] = 0

        self.team_grid = new_team
        self.health = new_health

    def _resolve_combat(self):
        """Adjacent fighters from different teams deal damage."""
        B, H, W, T = self.B, self.H, self.W, self.T

        has_fighter = self.health > 0
        if not has_fighter.any():
            return

        fighter_team = self.team_grid.long()

        # Count friendly and enemy neighbors for each cell
        padded_team = F.pad(self.team_grid.float(), (1, 1, 1, 1),
                            value=-1)
        padded_has = F.pad(has_fighter.float(), (1, 1, 1, 1), value=0)

        enemy_count = torch.zeros(B, H, W, device=self.device)
        friendly_count = torch.zeros(B, H, W, device=self.device)

        for i in range(8):
            dy = self.nb_dy[i].item()
            dx = self.nb_dx[i].item()

            nb_team = padded_team[:, 1 + dy:H + 1 + dy,
                                  1 + dx:W + 1 + dx]
            nb_has = padded_has[:, 1 + dy:H + 1 + dy,
                                1 + dx:W + 1 + dx]

            same_team = (nb_team == self.team_grid.float()) & (nb_has > 0)
            diff_team = (nb_team != self.team_grid.float()) & (nb_has > 0) \
                & (nb_team >= 0)

            friendly_count += same_team.float()
            enemy_count += diff_team.float()

        # Damage based on enemy count, reduced by friendly support
        attack = LW_CONFIG_CURRENT_RULES_ATTACK
        defense = LW_CONFIG_CURRENT_RULES_DEFENSE

        damage = (enemy_count * attack - friendly_count * defense).clamp(min=0)
        damage = damage.to(torch.int16)

        # Apply damage
        in_combat = has_fighter & (enemy_count > 0)
        self.health = torch.where(in_combat,
                                  (self.health - damage).clamp(min=0),
                                  self.health)

        # Fighters at 0 health get captured by the most common
        # enemy neighbor team
        dead = has_fighter & (self.health == 0)
        if dead.any():
            # Find dominant enemy team at each dead cell
            # Simple: assign to team with most adjacent fighters
            team_counts = torch.zeros(B, H, W, T, device=self.device)
            for i in range(8):
                dy = self.nb_dy[i].item()
                dx = self.nb_dx[i].item()
                nb_team = padded_team[:, 1 + dy:H + 1 + dy,
                                      1 + dx:W + 1 + dx].long()
                nb_has = padded_has[:, 1 + dy:H + 1 + dy,
                                    1 + dx:W + 1 + dx]
                valid = (nb_team >= 0) & (nb_team < T) & (nb_has > 0)
                # Scatter count
                nb_team_clamped = nb_team.clamp(0, T - 1)
                team_counts.scatter_add_(
                    3,
                    nb_team_clamped.unsqueeze(-1) * valid.unsqueeze(-1).long(),
                    valid.unsqueeze(-1).float())

            # Set dead fighter's team to zero out own team count
            for t in range(T):
                own_team_mask = dead & (fighter_team == t)
                team_counts[:, :, :, t] = torch.where(
                    own_team_mask, torch.zeros_like(team_counts[:, :, :, t]),
                    team_counts[:, :, :, t])

            capture_team = team_counts.argmax(dim=-1).to(torch.int8)
            has_captor = team_counts.sum(dim=-1) > 0

            # Revive captured fighter with new health
            new_health_val = MAX_HEALTH // 2

            capture_mask = dead & has_captor
            self.team_grid = torch.where(capture_mask, capture_team,
                                         self.team_grid)
            self.health = torch.where(
                capture_mask,
                torch.tensor(new_health_val, dtype=torch.int16,
                             device=self.device),
                self.health)

            # Remove dead fighters with no captor
            remove_mask = dead & ~has_captor
            self.team_grid = torch.where(
                remove_mask, torch.tensor(-1, dtype=torch.int8,
                                          device=self.device),
                self.team_grid)

    def _check_eliminations(self):
        """Check which teams have been eliminated."""
        for t in range(self.T):
            has_fighters = (self.team_grid == t).any(dim=(1, 2))
            self.team_alive[:, t] = has_fighters

    def _get_state(self):
        """Return current state as a dict."""
        return {
            'team_grid': self.team_grid,
            'health': self.health,
            'gradient': self.gradient,
            'cursor_pos': self.cursor_pos,
            'team_alive': self.team_alive,
            'walls': self.walls,
            'tick': self.tick,
        }

    def _get_info(self):
        """Return per-game stats."""
        fighters_per_team = torch.zeros(self.B, self.T,
                                        device=self.device)
        for t in range(self.T):
            fighters_per_team[:, t] = (self.team_grid == t).sum(
                dim=(1, 2)).float()

        total = fighters_per_team.sum(dim=1)
        best = fighters_per_team.max(dim=1)

        return {
            'fighters_per_team': fighters_per_team,
            'total_fighters': total,
            'best_team': best.indices,
            'best_count': best.values,
            'dominance': best.values / total.clamp(min=1),
            'tick': self.tick,
        }

    def get_observation(self):
        """Return an observation tensor suitable for neural network input.

        Returns (B, C, H, W) float tensor with channels:
            0: wall map (0=passable, 1=wall)
            1-T: per-team fighter presence (0 or 1)
            T+1 to 2T: per-team gradient (normalized)
            2T+1 to 3T: per-team health (normalized)
        """
        B, H, W, T = self.B, self.H, self.W, self.T

        channels = []
        channels.append(self.walls.float())

        for t in range(T):
            channels.append((self.team_grid == t).float())

        for t in range(T):
            g = self.gradient[:, t].float()
            g = g / g.max().clamp(min=1)
            channels.append(g)

        for t in range(T):
            h = ((self.team_grid == t).float()
                 * self.health.float() / MAX_HEALTH)
            channels.append(h)

        return torch.stack(channels, dim=1)


# Game rules constants (matching C defaults)
LW_CONFIG_CURRENT_RULES_ATTACK = 30
LW_CONFIG_CURRENT_RULES_DEFENSE = 10
