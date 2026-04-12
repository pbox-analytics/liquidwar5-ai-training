"""
Liquid War game engine — batched PyTorch tensor operations on GPU.

Optimized for throughput: pre-allocated tensors, minimal Python dispatch,
torch.compile-friendly operations. No Python loops in the hot path
except the per-team cursor loop (T<=6 iterations).
"""

import torch
import torch.nn.functional as F

try:
    from simulator.triton_kernels import triton_gradient_spread
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


MAX_HEALTH = 16384
GRADIENT_INF = 999999
MAX_TEAMS = 6


class LiquidWarEngine:
    """Batched Liquid War game engine on GPU.

    Args:
        batch_size: Number of games to run in parallel.
        height: Grid height.
        width: Grid width.
        num_teams: Teams per game (2-6).
        fighters_per_team: Starting fighters per team.
        device: 'cuda' or 'cpu'.
        attack: Damage per enemy neighbor.
        defense: Damage reduction per friendly neighbor.
        grad_iters: Gradient spread iterations per tick.
    """

    def __init__(self, batch_size=256, height=120, width=160,
                 num_teams=6, fighters_per_team=2000,
                 device='cuda', attack=30, defense=10, grad_iters=8):
        self.B = batch_size
        self.H = height
        self.W = width
        self.T = min(num_teams, MAX_TEAMS)
        self.device = device
        self.fighters_per_team = fighters_per_team
        self.attack = attack
        self.defense = defense
        self.grad_iters = grad_iters
        self.tick = 0

        # Neighbor convolution kernel (1, 1, 3, 3)
        self._nb_kernel = torch.tensor(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            dtype=torch.float32, device=device).view(1, 1, 3, 3)

        # Direction offsets
        self._dy = [-1, -1, -1, 0, 0, 1, 1, 1]
        self._dx = [-1, 0, 1, -1, 1, -1, 0, 1]
        self._dy_t = torch.tensor(self._dy, device=device)
        self._dx_t = torch.tensor(self._dx, device=device)

        # Pre-allocate index grids
        self._b_idx = torch.arange(batch_size, device=device).view(
            batch_size, 1, 1)
        self._y_idx = torch.arange(height, device=device).view(1, height, 1)
        self._x_idx = torch.arange(width, device=device).view(1, 1, width)

        # Pre-allocate scratch tensors for gradient spread
        self._grad_padded = torch.zeros(
            batch_size, num_teams, height + 2, width + 2,
            device=device)

    def reset(self, walls=None):
        """Initialize all B games."""
        B, H, W, T = self.B, self.H, self.W, self.T
        dev = self.device

        if walls is None:
            walls = self._generate_random_maps()
        self.walls = walls.bool().to(dev)
        self.passable = ~self.walls
        self.passable_f = self.passable.float()

        # One-hot team presence (B, T, H, W)
        self.team_oh = torch.zeros(B, T, H, W, device=dev)
        # Fighter health (B, H, W)
        self.health = torch.zeros(B, H, W, device=dev)
        # Gradient (B, T, H, W)
        self.gradient = torch.full((B, T, H, W), GRADIENT_INF,
                                   dtype=torch.float32, device=dev)
        # Cursor positions (B, T, 2)
        self.cursor_pos = torch.zeros(B, T, 2, dtype=torch.long, device=dev)
        # Team alive
        self.team_alive = torch.ones(B, T, dtype=torch.bool, device=dev)

        # Wall mask for gradient (expanded once)
        self._wall_grad = self.walls.unsqueeze(1).expand(B, T, H, W)

        self._place_teams()
        self.tick = 0
        return self._get_state()

    def step(self, cursor_actions=None):
        """Run one tick for all B games."""
        if cursor_actions is not None:
            self._move_cursors(cursor_actions)

        self._seed_and_spread_gradient()
        self._move_fighters()
        self._resolve_combat()
        self._check_eliminations()

        self.tick += 1
        teams_left = self.team_alive.sum(dim=1)
        done = teams_left <= 1
        return self._get_state(), done, self._get_info()

    def step_with_ai(self):
        """Run one tick with built-in simple AI (no Python round-trip).

        Computes cursor actions and game step in one call,
        avoiding the overhead of returning to Python between them.
        """
        B, T, H, W = self.B, self.T, self.H, self.W

        # --- Inline AI: move each cursor toward enemy centroid ---
        has_fighter = (self.health > 0).float()
        y_c = self._y_idx.float().expand(B, H, W)
        x_c = self._x_idx.float().expand(B, H, W)

        for t in range(T):
            if not self.team_alive[:, t].any():
                continue

            enemy = (has_fighter - self.team_oh[:, t] * has_fighter).clamp(min=0)
            e_count = enemy.sum(dim=(1, 2)).clamp(min=1)
            e_y = (enemy * y_c).sum(dim=(1, 2)) / e_count
            e_x = (enemy * x_c).sum(dim=(1, 2)) / e_count

            cy = self.cursor_pos[:, t, 0].float()
            cx = self.cursor_pos[:, t, 1].float()
            dy = (e_y - cy).sign().long()
            dx = (e_x - cx).sign().long()

            alive = self.team_alive[:, t]
            new_y = (self.cursor_pos[:, t, 0] + torch.where(alive, dy, torch.zeros_like(dy))).clamp(1, H - 2)
            new_x = (self.cursor_pos[:, t, 1] + torch.where(alive, dx, torch.zeros_like(dx))).clamp(1, W - 2)

            b = torch.arange(B, device=self.device)
            ok = self.passable[b, new_y, new_x] & alive
            self.cursor_pos[:, t, 0] = torch.where(ok, new_y, self.cursor_pos[:, t, 0])
            self.cursor_pos[:, t, 1] = torch.where(ok, new_x, self.cursor_pos[:, t, 1])

        # --- Game step ---
        self._seed_and_spread_gradient()
        self._move_fighters()
        self._resolve_combat()
        self._check_eliminations()

        self.tick += 1
        teams_left = self.team_alive.sum(dim=1)
        done = teams_left <= 1
        return done, self._get_info()

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

            region = torch.zeros(B, H, W, dtype=torch.bool,
                                 device=self.device)
            region[:, 2:H - 2, x_start:x_end] = True
            region = region & self.passable

            region_flat = region.view(B, -1).float()
            n_available = region_flat.sum(dim=1, keepdim=True).clamp(min=1)
            prob = region_flat / n_available
            target = min(self.fighters_per_team,
                         int(region_flat.sum(dim=1).min().item()))

            if target > 0:
                selected = torch.multinomial(prob + 1e-8, target,
                                             replacement=False)
                sy = selected // W
                sx = selected % W
                for b in range(B):
                    self.team_oh[b, t, sy[b], sx[b]] = 1.0
                    self.health[b, sy[b], sx[b]] = MAX_HEALTH

    # ------------------------------------------------------------------
    # Cursor movement
    # ------------------------------------------------------------------

    def _move_cursors(self, actions):
        new_pos = self.cursor_pos + actions.long()
        new_pos[:, :, 0].clamp_(1, self.H - 2)
        new_pos[:, :, 1].clamp_(1, self.W - 2)
        b = torch.arange(self.B, device=self.device)
        for t in range(self.T):
            ny, nx = new_pos[:, t, 0], new_pos[:, t, 1]
            ok = self.passable[b, ny, nx] & self.team_alive[:, t]
            self.cursor_pos[:, t, 0] = torch.where(
                ok, ny, self.cursor_pos[:, t, 0])
            self.cursor_pos[:, t, 1] = torch.where(
                ok, nx, self.cursor_pos[:, t, 1])

    # ------------------------------------------------------------------
    # Combined gradient seed + spread (main bottleneck — optimized)
    # ------------------------------------------------------------------

    def _seed_and_spread_gradient(self):
        """Age gradient, seed cursors, spread via in-place min-shifts."""
        B, T, H, W = self.B, self.T, self.H, self.W

        # Age: +1 everywhere except walls
        self.gradient.add_(1)
        self.gradient[self._wall_grad] = GRADIENT_INF

        # Seed cursor positions
        b_idx = torch.arange(B, device=self.device)
        for t in range(T):
            cy = self.cursor_pos[:, t, 0]
            cx = self.cursor_pos[:, t, 1]
            self.gradient[b_idx, t, cy, cx] = 0

        # Spread gradient
        iters = 40 if self.tick < 30 else self.grad_iters

        if HAS_TRITON and self.device != 'cpu':
            # Triton kernel: fused iterations, no padded buffer
            triton_gradient_spread(self.gradient, self.walls, iterations=iters)
        else:
            # Fallback: Python min-shift spread
            p = self._grad_padded
            for _ in range(iters):
                p.fill_(GRADIENT_INF)
                p[:, :, 1:H + 1, 1:W + 1] = self.gradient
                g = self.gradient
                torch.minimum(g, p[:, :, 0:H, 0:W] + 1, out=g)
                torch.minimum(g, p[:, :, 0:H, 1:W+1] + 1, out=g)
                torch.minimum(g, p[:, :, 0:H, 2:W+2] + 1, out=g)
                torch.minimum(g, p[:, :, 1:H+1, 0:W] + 1, out=g)
                torch.minimum(g, p[:, :, 1:H+1, 2:W+2] + 1, out=g)
                torch.minimum(g, p[:, :, 2:H+2, 0:W] + 1, out=g)
                torch.minimum(g, p[:, :, 2:H+2, 1:W+1] + 1, out=g)
                torch.minimum(g, p[:, :, 2:H+2, 2:W+2] + 1, out=g)
                g[self._wall_grad] = GRADIENT_INF

    # ------------------------------------------------------------------
    # Fighter movement (vectorized)
    # ------------------------------------------------------------------

    def _move_fighters(self):
        B, T, H, W = self.B, self.T, self.H, self.W

        has_fighter = self.health > 0
        if not has_fighter.any():
            return

        fighter_team = self.team_oh.argmax(dim=1)  # (B, H, W)
        b_e = self._b_idx.expand(B, H, W)
        y_e = self._y_idx.expand(B, H, W)
        x_e = self._x_idx.expand(B, H, W)

        current_grad = self.gradient[b_e, fighter_team, y_e, x_e]

        # Pad gradient, health, passable for boundary-safe neighbor access
        gp = F.pad(self.gradient, (1, 1, 1, 1), value=GRADIENT_INF)
        hp = F.pad(self.health, (1, 1, 1, 1), value=-1)
        pp = F.pad(self.passable_f, (1, 1, 1, 1), value=0)

        # Stack all 8 neighbor gradients: (8, B, H, W)
        nb_grads = []
        nb_valids = []
        for i in range(8):
            dy, dx = self._dy[i], self._dx[i]
            ys, xs = 1 + dy, 1 + dx

            # Neighbor gradient for each fighter's team
            ng_all = gp[:, :, ys:ys + H, xs:xs + W]
            ng = ng_all[b_e, fighter_team, y_e, x_e]

            # Valid = passable AND empty
            valid = (pp[:, ys:ys+H, xs:xs+W] > 0) & \
                    (hp[:, ys:ys+H, xs:xs+W] <= 0)

            nb_grads.append(torch.where(valid, ng, GRADIENT_INF))
            nb_valids.append(valid)

        all_grads = torch.stack(nb_grads, dim=0)  # (8, B, H, W)
        best_dir = all_grads.argmin(dim=0)  # (B, H, W)
        best_grad = all_grads.gather(
            0, best_dir.unsqueeze(0)).squeeze(0)

        best_dy = self._dy_t[best_dir.flatten()].view(B, H, W)
        best_dx = self._dx_t[best_dir.flatten()].view(B, H, W)

        should_move = has_fighter & (best_grad < current_grad)
        if not should_move.any():
            return

        move_idx = torch.where(should_move)
        b_m, sy, sx = move_idx
        ty = (sy + best_dy[b_m, sy, sx]).clamp(0, H - 1)
        tx = (sx + best_dx[b_m, sy, sx]).clamp(0, W - 1)

        # Conflict resolution: only move to empty destinations
        dest_empty = self.health[b_m, ty, tx] <= 0
        b_ok = b_m[dest_empty]
        sy_ok = sy[dest_empty]
        sx_ok = sx[dest_empty]
        ty_ok = ty[dest_empty]
        tx_ok = tx[dest_empty]

        if len(b_ok) == 0:
            return

        # Execute moves
        for t in range(T):
            v = self.team_oh[b_ok, t, sy_ok, sx_ok].clone()
            self.team_oh[b_ok, t, ty_ok, tx_ok] = v
            self.team_oh[b_ok, t, sy_ok, sx_ok] = 0

        hv = self.health[b_ok, sy_ok, sx_ok].clone()
        self.health[b_ok, ty_ok, tx_ok] = hv
        self.health[b_ok, sy_ok, sx_ok] = 0

    # ------------------------------------------------------------------
    # Combat (convolution-based)
    # ------------------------------------------------------------------

    def _resolve_combat(self):
        B, T, H, W = self.B, self.T, self.H, self.W

        has_fighter = (self.health > 0).float()
        if has_fighter.sum() == 0:
            return

        team_presence = self.team_oh * has_fighter.unsqueeze(1)

        # Neighbor count per team via conv2d
        tp_flat = team_presence.view(B * T, 1, H, W)
        nb_count = F.conv2d(tp_flat, self._nb_kernel, padding=1)
        nb_count = nb_count.view(B, T, H, W)

        fighter_team = self.team_oh.argmax(dim=1)
        b_e = self._b_idx.expand(B, H, W)
        y_e = self._y_idx.expand(B, H, W)
        x_e = self._x_idx.expand(B, H, W)

        friendly = nb_count[b_e, fighter_team, y_e, x_e]
        total_nb = nb_count.sum(dim=1)
        enemy = total_nb - friendly

        damage = (enemy * self.attack - friendly * self.defense).clamp(min=0)

        in_combat = (has_fighter > 0) & (enemy > 0)
        self.health = torch.where(in_combat,
                                  (self.health - damage).clamp(min=0),
                                  self.health)

        # Capture dead fighters
        dead = (has_fighter > 0) & (self.health <= 0)
        if not dead.any():
            return

        # Zero out dead fighter's own team from capture counts
        capture_counts = nb_count.clone()
        for t in range(T):
            own_dead = dead & (fighter_team == t)
            capture_counts[:, t] = torch.where(
                own_dead, torch.zeros_like(capture_counts[:, t]),
                capture_counts[:, t])

        captor_team = capture_counts.argmax(dim=1)
        has_captor = capture_counts.sum(dim=1) > 0
        capture = dead & has_captor

        if capture.any():
            # Clear old team
            self.team_oh *= (~capture).unsqueeze(1).float()
            # Set new team
            for t in range(T):
                new_member = capture & (captor_team == t)
                self.team_oh[:, t] = torch.where(
                    new_member,
                    torch.ones_like(self.team_oh[:, t]),
                    self.team_oh[:, t])
            self.health = torch.where(
                capture,
                torch.tensor(MAX_HEALTH * 0.5, device=self.device),
                self.health)

        remove = dead & ~has_captor
        if remove.any():
            self.team_oh *= (~remove).unsqueeze(1).float()
            self.health = torch.where(remove, 0.0, self.health)

    # ------------------------------------------------------------------
    # Elimination
    # ------------------------------------------------------------------

    def _check_eliminations(self):
        for t in range(self.T):
            self.team_alive[:, t] = (self.team_oh[:, t] > 0).any(
                dim=(1, 2))

    # ------------------------------------------------------------------
    # State / Info / Observation
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
        fighters = self.team_oh.sum(dim=(2, 3))
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
        """(B, 1+3T, H, W) tensor for neural network input."""
        T = self.T
        has_fighter = (self.health > 0).float()
        channels = [self.walls.float()]
        for t in range(T):
            channels.append(self.team_oh[:, t])
        for t in range(T):
            g = self.gradient[:, t].clone()
            g[g >= GRADIENT_INF] = 0
            gmax = g.amax(dim=(1, 2), keepdim=True).clamp(min=1)
            channels.append(g / gmax)
        for t in range(T):
            channels.append(self.team_oh[:, t] * self.health / MAX_HEALTH)
        return torch.stack(channels, dim=1)
