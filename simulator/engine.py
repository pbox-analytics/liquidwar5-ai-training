"""
Liquid War game engine implemented as batched PyTorch tensor operations.

All neighbor operations use convolutions or shifted tensor slicing —
no Python loops in the hot path. This enables hundreds of games
to run simultaneously on GPU.

Core mechanics:
1. Grid map with walls (passable/impassable cells)
2. Fighters on the grid, each belonging to a team (0-5)
3. Cursors (one per team) that the AI moves each tick
4. Gradient propagation via iterative min-convolution
5. Fighters follow the gradient toward their team's cursor
6. Combat via neighbor counting with convolutions
7. Fighters at 0 health get captured by dominant neighbor team
"""

import torch
import torch.nn.functional as F


MAX_HEALTH = 16384
GRADIENT_INF = 999999
MAX_TEAMS = 6

# 3x3 kernel for 8-connected neighbor operations (excludes center)
NEIGHBOR_KERNEL = torch.tensor([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=torch.float32)

# 3x3 kernel including center for min-pooling gradient spread
SPREAD_KERNEL = torch.tensor([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]], dtype=torch.float32)


class LiquidWarEngine:
    """Batched Liquid War game engine on GPU.

    Args:
        batch_size: Number of games to run in parallel.
        height: Grid height in cells.
        width: Grid width in cells.
        num_teams: Number of teams (2-6).
        fighters_per_team: Initial fighters per team.
        device: 'cuda' or 'cpu'.
        attack: Damage per enemy neighbor per tick.
        defense: Damage reduction per friendly neighbor per tick.
    """

    def __init__(self, batch_size=256, height=120, width=160,
                 num_teams=6, fighters_per_team=2000,
                 device='cuda', attack=30, defense=10):
        self.B = batch_size
        self.H = height
        self.W = width
        self.T = min(num_teams, MAX_TEAMS)
        self.device = device
        self.fighters_per_team = fighters_per_team
        self.attack = attack
        self.defense = defense
        self.tick = 0

        # Pre-compute convolution kernels on device
        # Neighbor kernel: (1, 1, 3, 3) for conv2d
        self._nb_kernel = NEIGHBOR_KERNEL.view(1, 1, 3, 3).to(device)
        self._spread_kernel = SPREAD_KERNEL.view(1, 1, 3, 3).to(device)

        # 8 direction offsets as (dy, dx) pairs
        self._shifts = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]

    def reset(self, walls=None):
        """Initialize all B games. Returns state dict."""
        B, H, W, T = self.B, self.H, self.W, self.T
        dev = self.device

        if walls is None:
            walls = self._generate_random_maps()
        self.walls = walls.bool().to(dev)
        self.passable = ~self.walls
        self.passable_f = self.passable.float()

        # Team grid: one-hot per team. (B, T, H, W) float
        # This avoids expensive scatter/gather with int8 team IDs
        self.team_oh = torch.zeros(B, T, H, W, device=dev)

        # Fighter health (B, H, W) float
        self.health = torch.zeros(B, H, W, device=dev)

        # Gradient per team (B, T, H, W) float — lower = closer to cursor
        self.gradient = torch.full((B, T, H, W), GRADIENT_INF,
                                   dtype=torch.float32, device=dev)

        # Cursor positions (B, T, 2) long — [y, x]
        self.cursor_pos = torch.zeros(B, T, 2, dtype=torch.long,
                                      device=dev)

        # Team alive
        self.team_alive = torch.ones(B, T, dtype=torch.bool, device=dev)

        self._place_teams()
        self.tick = 0
        return self._get_state()

    def step(self, cursor_actions=None):
        """Run one tick. Returns (state, done, info)."""
        if cursor_actions is not None:
            self._move_cursors(cursor_actions)

        self._seed_gradients()
        # More iterations in early ticks to build initial gradient field
        iters = 30 if self.tick < 100 else 10
        self._spread_gradient(iterations=iters)
        self._move_fighters()
        self._resolve_combat()
        self._check_eliminations()

        self.tick += 1

        teams_left = self.team_alive.sum(dim=1)
        done = teams_left <= 1

        return self._get_state(), done, self._get_info()

    # ------------------------------------------------------------------
    # Map generation
    # ------------------------------------------------------------------

    def _generate_random_maps(self):
        B, H, W = self.B, self.H, self.W
        walls = torch.zeros(B, H, W, dtype=torch.bool)
        walls[:, 0, :] = True
        walls[:, -1, :] = True
        walls[:, :, 0] = True
        walls[:, :, -1] = True
        walls[:, 1:-1, 1:-1] = torch.rand(B, H - 2, W - 2) < 0.12
        return walls

    # ------------------------------------------------------------------
    # Team placement
    # ------------------------------------------------------------------

    def _place_teams(self):
        B, H, W, T = self.B, self.H, self.W, self.T
        strip_w = max(1, (W - 2) // T)

        for t in range(T):
            x_start = 1 + t * strip_w
            x_end = min(x_start + strip_w, W - 1)
            cy, cx = H // 2, (x_start + x_end) // 2
            self.cursor_pos[:, t, 0] = cy
            self.cursor_pos[:, t, 1] = cx

            # Create placement region mask
            region = torch.zeros(B, H, W, dtype=torch.bool,
                                 device=self.device)
            region[:, 2:H - 2, x_start:x_end] = True
            region = region & self.passable

            # Randomly select fighters_per_team cells in the region
            region_flat = region.view(B, -1).float()
            n_available = region_flat.sum(dim=1, keepdim=True).clamp(min=1)

            # Probability of placing in each cell
            prob = region_flat / n_available
            target = min(self.fighters_per_team,
                         int(region_flat.sum(dim=1).min().item()))

            if target > 0:
                selected = torch.multinomial(prob + 1e-8, target,
                                             replacement=False)
                # Convert flat indices back to 2D
                sy = selected // W
                sx = selected % W

                # Set team and health using scatter
                for b in range(B):
                    self.team_oh[b, t, sy[b], sx[b]] = 1.0
                    self.health[b, sy[b], sx[b]] = MAX_HEALTH

    # ------------------------------------------------------------------
    # Cursor movement
    # ------------------------------------------------------------------

    def _move_cursors(self, actions):
        """actions: (B, T, 2) with dy, dx in {-1, 0, 1}"""
        new_pos = self.cursor_pos + actions.long()
        new_pos[:, :, 0].clamp_(1, self.H - 2)
        new_pos[:, :, 1].clamp_(1, self.W - 2)

        # Batch check passability
        b_idx = torch.arange(self.B, device=self.device)
        for t in range(self.T):
            ny, nx = new_pos[:, t, 0], new_pos[:, t, 1]
            ok = self.passable[b_idx, ny, nx] & self.team_alive[:, t]
            self.cursor_pos[:, t, 0] = torch.where(
                ok, ny, self.cursor_pos[:, t, 0])
            self.cursor_pos[:, t, 1] = torch.where(
                ok, nx, self.cursor_pos[:, t, 1])

    # ------------------------------------------------------------------
    # Gradient propagation (the expensive part — pure convolution)
    # ------------------------------------------------------------------

    def _seed_gradients(self):
        """Seed cursor positions and age the gradient field.

        The gradient increases by 1 each tick everywhere, then the cursor
        reseeds at 0. This creates an expanding wavefront like the C code
        where cursor.val decreases each tick.
        """
        # Age: increase all gradients by 1 (like the C code's clock mechanism)
        wall_mask = self.walls.unsqueeze(1).expand_as(self.gradient)
        self.gradient = torch.where(
            wall_mask, self.gradient,
            (self.gradient + 1).clamp(max=GRADIENT_INF))

        # Seed cursor positions at 0
        b_idx = torch.arange(self.B, device=self.device)
        for t in range(self.T):
            cy = self.cursor_pos[:, t, 0]
            cx = self.cursor_pos[:, t, 1]
            self.gradient[b_idx, t, cy, cx] = 0

    def _spread_gradient(self, iterations=6):
        """Min-convolution: each cell adopts min(neighbors) + 1.

        Uses padding + shifted min instead of actual convolution for
        integer min operation (conv2d does sum, we need min).
        """
        B, T, H, W = self.B, self.T, self.H, self.W
        wall_mask = self.walls.unsqueeze(1).expand(B, T, H, W)

        for _ in range(iterations):
            # Pad with INF
            g = F.pad(self.gradient, (1, 1, 1, 1), value=GRADIENT_INF)

            # Compute min of all 8 neighbors + center using shifted views
            # This avoids a Python loop by stacking all shifts at once
            shifts = torch.stack([
                g[:, :, 0:H, 0:W],       # top-left
                g[:, :, 0:H, 1:W + 1],   # top
                g[:, :, 0:H, 2:W + 2],   # top-right
                g[:, :, 1:H + 1, 0:W],   # left
                g[:, :, 1:H + 1, 2:W + 2],  # right
                g[:, :, 2:H + 2, 0:W],   # bottom-left
                g[:, :, 2:H + 2, 1:W + 1],  # bottom
                g[:, :, 2:H + 2, 2:W + 2],  # bottom-right
            ], dim=0)  # (8, B, T, H, W)

            # Min neighbor + 1
            min_nb = shifts.min(dim=0).values + 1

            # Keep current if it's already lower
            new_grad = torch.minimum(self.gradient, min_nb)

            # Don't update walls
            self.gradient = torch.where(wall_mask, self.gradient, new_grad)

    # ------------------------------------------------------------------
    # Fighter movement (vectorized)
    # ------------------------------------------------------------------

    def _move_fighters(self):
        """Move fighters toward their team's cursor along the gradient."""
        B, T, H, W = self.B, self.T, self.H, self.W

        has_fighter = self.health > 0
        if not has_fighter.any():
            return

        # Which team does each fighter belong to? (B, H, W)
        # team_oh is (B, T, H, W), get team index via argmax
        fighter_team = self.team_oh.argmax(dim=1)  # (B, H, W)

        # Get each fighter's gradient value at its position
        b_idx = torch.arange(B, device=self.device).view(B, 1, 1)
        y_idx = torch.arange(H, device=self.device).view(1, H, 1)
        x_idx = torch.arange(W, device=self.device).view(1, 1, W)

        current_grad = self.gradient[
            b_idx.expand(B, H, W),
            fighter_team,
            y_idx.expand(B, H, W),
            x_idx.expand(B, H, W),
        ]  # (B, H, W)

        # For each of 8 directions, compute the gradient at neighbor
        # and find the best direction to move
        padded_grad = F.pad(self.gradient, (1, 1, 1, 1), value=GRADIENT_INF)
        padded_health = F.pad(self.health, (1, 1, 1, 1), value=-1)
        padded_passable = F.pad(self.passable_f, (1, 1, 1, 1), value=0)

        # Stack all 8 neighbor gradients, emptiness, passability at once
        # Each is (8, B, H, W)
        nb_grads = []
        nb_valids = []
        dy_vals = []
        dx_vals = []

        for i, (dy_off, dx_off) in enumerate(self._shifts):
            y_s = 1 + dy_off
            x_s = 1 + dx_off

            nb_grad_all = padded_grad[:, :, y_s:y_s + H, x_s:x_s + W]
            nb_grad = nb_grad_all[
                b_idx.expand(B, H, W),
                fighter_team,
                y_idx.expand(B, H, W),
                x_idx.expand(B, H, W),
            ]
            nb_empty = padded_health[:, y_s:y_s + H, x_s:x_s + W] <= 0
            nb_pass = padded_passable[:, y_s:y_s + H, x_s:x_s + W] > 0

            nb_grads.append(nb_grad)
            nb_valids.append(nb_pass & nb_empty)
            dy_vals.append(dy_off)
            dx_vals.append(dx_off)

        # (8, B, H, W)
        all_grads = torch.stack(nb_grads, dim=0)
        all_valid = torch.stack(nb_valids, dim=0)

        # Mask invalid with INF
        all_grads = torch.where(all_valid, all_grads, GRADIENT_INF)

        # Find best direction (argmin over dim 0)
        best_dir = all_grads.argmin(dim=0)  # (B, H, W)
        best_grad = all_grads.gather(0, best_dir.unsqueeze(0)).squeeze(0)

        # Map direction index to dy, dx
        dy_tensor = torch.tensor(dy_vals, device=self.device)
        dx_tensor = torch.tensor(dx_vals, device=self.device)
        best_dy = dy_tensor[best_dir.flatten()].view(B, H, W)
        best_dx = dx_tensor[best_dir.flatten()].view(B, H, W)

        # Move if the best neighbor has lower gradient than current
        should_move = has_fighter & (best_grad < current_grad)

        if not should_move.any():
            return

        # Get source and destination coordinates
        move_idx = torch.where(should_move)
        b_m, sy, sx = move_idx
        dy_m = best_dy[b_m, sy, sx]
        dx_m = best_dx[b_m, sy, sx]
        ty = (sy + dy_m).clamp(0, H - 1)
        tx = (sx + dx_m).clamp(0, W - 1)

        # Resolve conflicts: only first mover to each destination wins
        # Create a destination key and deduplicate
        dest_key = b_m * H * W + ty * W + tx
        _, unique_idx = torch.unique(dest_key, return_inverse=True)

        # For each unique destination, keep the first mover
        first_mask = torch.zeros(len(b_m), dtype=torch.bool,
                                 device=self.device)
        seen = torch.full((B * H * W,), False, dtype=torch.bool,
                          device=self.device)
        # Vectorized first-occurrence: scatter with first-wins
        first_occurrence = torch.full_like(dest_key, len(b_m))
        first_occurrence.scatter_(0, dest_key.argsort(),
                                  torch.arange(len(b_m),
                                               device=self.device))
        # Actually simpler: just check destination is still empty
        dest_empty = self.health[b_m, ty, tx] <= 0
        b_ok = b_m[dest_empty]
        sy_ok = sy[dest_empty]
        sx_ok = sx[dest_empty]
        ty_ok = ty[dest_empty]
        tx_ok = tx[dest_empty]

        if len(b_ok) == 0:
            return

        # Move: copy team and health to destination, clear source
        for t in range(T):
            team_vals = self.team_oh[b_ok, t, sy_ok, sx_ok].clone()
            self.team_oh[b_ok, t, ty_ok, tx_ok] = team_vals
            self.team_oh[b_ok, t, sy_ok, sx_ok] = 0

        health_vals = self.health[b_ok, sy_ok, sx_ok].clone()
        self.health[b_ok, ty_ok, tx_ok] = health_vals
        self.health[b_ok, sy_ok, sx_ok] = 0

    # ------------------------------------------------------------------
    # Combat (convolution-based neighbor counting)
    # ------------------------------------------------------------------

    def _resolve_combat(self):
        """Damage fighters based on enemy/friendly neighbor counts."""
        B, T, H, W = self.B, self.T, self.H, self.W

        has_fighter = (self.health > 0).float()
        if has_fighter.sum() == 0:
            return

        # Per-team presence: (B, T, H, W)
        team_presence = self.team_oh * has_fighter.unsqueeze(1)

        # Count neighbors per team using convolution
        # Reshape for grouped conv: (B*T, 1, H, W)
        tp_flat = team_presence.view(B * T, 1, H, W)
        kernel = self._nb_kernel  # (1, 1, 3, 3)

        nb_count = F.conv2d(tp_flat, kernel, padding=1)
        nb_count = nb_count.view(B, T, H, W)  # (B, T, H, W)

        # For each cell, friendly count = neighbors of same team
        # enemy count = neighbors of other teams
        fighter_team = self.team_oh.argmax(dim=1)  # (B, H, W)

        # Gather friendly neighbor count
        friendly = nb_count[
            torch.arange(B, device=self.device).view(B, 1, 1).expand(B, H, W),
            fighter_team,
            torch.arange(H, device=self.device).view(1, H, 1).expand(B, H, W),
            torch.arange(W, device=self.device).view(1, 1, W).expand(B, H, W),
        ]

        # Total neighbor count (all teams)
        total_nb = nb_count.sum(dim=1)  # (B, H, W)
        enemy = total_nb - friendly

        # Damage
        damage = (enemy * self.attack - friendly * self.defense).clamp(min=0)

        # Apply only to cells with fighters and enemies
        in_combat = (has_fighter > 0) & (enemy > 0)
        self.health = torch.where(in_combat,
                                  (self.health - damage).clamp(min=0),
                                  self.health)

        # Capture: fighters at 0 health join the dominant enemy team
        dead = (has_fighter > 0) & (self.health <= 0)
        if not dead.any():
            return

        # Find dominant enemy neighbor team (team with most neighbors)
        # Zero out the dead fighter's own team from the count
        capture_counts = nb_count.clone()
        for t in range(T):
            own_mask = dead & (fighter_team == t)
            capture_counts[:, t] = torch.where(
                own_mask, torch.zeros_like(capture_counts[:, t]),
                capture_counts[:, t])

        captor_team = capture_counts.argmax(dim=1)  # (B, H, W)
        has_captor = capture_counts.sum(dim=1) > 0

        capture = dead & has_captor
        if capture.any():
            # Update team one-hot
            self.team_oh[:, :, :, :] = torch.where(
                capture.unsqueeze(1),
                torch.zeros_like(self.team_oh),
                self.team_oh)
            for t in range(T):
                new_member = capture & (captor_team == t)
                self.team_oh[:, t] = torch.where(
                    new_member, torch.ones_like(self.team_oh[:, t]),
                    self.team_oh[:, t])
            self.health = torch.where(capture,
                                      torch.tensor(MAX_HEALTH // 2,
                                                   dtype=torch.float32,
                                                   device=self.device),
                                      self.health)

        # Remove dead with no captor
        remove = dead & ~has_captor
        if remove.any():
            self.team_oh = torch.where(
                remove.unsqueeze(1),
                torch.zeros_like(self.team_oh),
                self.team_oh)
            self.health = torch.where(remove,
                                      torch.zeros_like(self.health),
                                      self.health)

    # ------------------------------------------------------------------
    # Elimination check
    # ------------------------------------------------------------------

    def _check_eliminations(self):
        for t in range(self.T):
            self.team_alive[:, t] = (self.team_oh[:, t] > 0).any(
                dim=(1, 2))

    # ------------------------------------------------------------------
    # State and observation
    # ------------------------------------------------------------------

    def _get_state(self):
        return {
            'team_oh': self.team_oh,
            'health': self.health,
            'gradient': self.gradient,
            'cursor_pos': self.cursor_pos,
            'team_alive': self.team_alive,
            'walls': self.walls,
            'tick': self.tick,
        }

    def _get_info(self):
        fighters = self.team_oh.sum(dim=(2, 3))  # (B, T)
        total = fighters.sum(dim=1)
        best = fighters.max(dim=1)
        return {
            'fighters_per_team': fighters,
            'total_fighters': total,
            'best_team': best.indices,
            'best_count': best.values,
            'dominance': best.values / total.clamp(min=1),
            'tick': self.tick,
        }

    def get_observation(self):
        """Return (B, C, H, W) tensor for neural network input.

        Channels: walls, per-team presence, per-team gradient (norm),
        per-team health (norm).
        """
        B, H, W, T = self.B, self.H, self.W, self.T
        has_fighter = (self.health > 0).float()

        channels = [self.walls.float()]  # 1 channel
        for t in range(T):
            channels.append(self.team_oh[:, t])  # T channels
        for t in range(T):
            g = self.gradient[:, t].clone()
            g[g >= GRADIENT_INF] = 0
            g = g / (g.max().clamp(min=1))
            channels.append(g)  # T channels
        for t in range(T):
            channels.append(self.team_oh[:, t] * self.health / MAX_HEALTH)

        return torch.stack(channels, dim=1)  # (B, 1+3T, H, W)
