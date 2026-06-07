"""Faithful GPU Liquid War engine — indexed-particle SoA.

Re-implements the real Liquid War mechanics (extracted from the C source into
``docs/lw-mechanics-spec.md``) on the GPU, batched for RL training. The three
things the old presence-grid sim got wrong, now fixed:

- **Conservation** — fighters are a fixed army (``army_size`` per game). They are
  never created or destroyed; movement only relocates them.
- **Conversion, not deletion** — combat drains a target's health; when it goes
  negative the fighter is rebased positive and **defects to the attacker's
  team**. The total count is invariant.
- **Collision / one-per-cell** — an ``occ`` grid holds the slot index of the
  single fighter on each cell (the GPU analogue of ``PLACE.fighter``). Fighters
  cannot overlap, so they pack instead of collapsing.
- **Persistent gradient** — a per-team distance field seeded at the cursor and
  relaxed every tick, never reset (vs the old capped/aged spread).

The public surface (``reset`` / ``step`` / ``get_observation`` and the
``team_oh`` / ``health`` / ``gradient`` / ``cursor_pos`` / ``team_alive`` /
``walls`` attributes) is preserved — ``team_oh`` and ``health`` are rebuilt as
**derived scatter views** of the SoA each tick — so ``build_egocentric_obs``,
the policy, ``collect_rollout``, eval, and the play server keep working unchanged.

.. note::
   This is an MDP change: the gradient distribution and the conversion reward
   landscape differ from the old engine, so existing ``results/rl`` checkpoints
   are invalid and training restarts from scratch. No RL-code edits are needed.

.. seealso:: ``docs/lw-mechanics-spec.md`` — the extracted C contract.
"""
from __future__ import annotations

import torch

MAX_TEAMS = 6
MAX_FIGHTER_HEALTH = 16384                 # health valid in [0, 16383]
GRAD_INIT = 2_000_000                       # distance-field init / "unreachable"
CURSOR_SEED = 1_000_000                     # gradient seed value at the cursor cell
# Default-config combat coefficients (number_influence=8 => rubber-band off, so
# these are constants; see spec). Held as ints, applied to int health.
ATTACK = 2048
SIDE_ATTACK = ATTACK >> 4                   # 128
DEFENSE = 64
NEW_HEALTH = 4096
GRADIENT_INF = GRAD_INIT                     # back-compat alias (get_observation)
MAX_HEALTH = MAX_FIGHTER_HEALTH - 1          # back-compat alias

# 8-neighbour movement (the C 12-dir refinement is a later parity step; the
# invariants — conservation/conversion/collision — are direction-agnostic).
_DY = [-1, -1, -1, 0, 0, 1, 1, 1]
_DX = [-1, 0, 1, -1, 1, -1, 0, 1]


class LiquidWarEngine:
    """Batched faithful Liquid War engine on GPU (indexed-particle SoA).

    :ivar fx: ``(B, N)`` int fighter x (column) per army slot.
    :ivar fy: ``(B, N)`` int fighter y (row) per army slot.
    :ivar fhealth: ``(B, N)`` int32 health; may go transiently negative pre-rebase.
    :ivar fteam: ``(B, N)`` int8 team id (mutates on conversion).
    :ivar occ: ``(B, H, W)`` int32 — owning slot index per cell, ``-1`` if empty.
    :ivar gradient: ``(B, T, H, W)`` int32 persistent per-team distance field.
    """

    def __init__(self, batch_size: int = 256, height: int = 120, width: int = 160,
                 num_teams: int = 6, fighters_per_team: int = 2000,
                 device: str = 'cuda', attack: int = 30, defense: int = 10,
                 grad_iters: int = 8) -> None:
        """Construct the engine.

        :param grad_iters: number of gradient relaxation sweeps per tick
            (repurposed; was the old fixed-iteration count).
        :param attack: ignored — kept for constructor back-compat (real combat
            uses the spec's constant coefficients).
        :param defense: ignored — kept for constructor back-compat.
        """
        self.B = batch_size
        self.H = height
        self.W = width
        self.T = min(num_teams, MAX_TEAMS)
        self.device = device
        self.fighters_per_team = fighters_per_team
        self.grad_sweeps = max(1, int(grad_iters))
        self.tick = 0

        dev = device
        # Per-game army size (constant for the game). Equal across games here.
        self.N = fighters_per_team * self.T
        # Pre-allocated index grids.
        self._b_idx = torch.arange(batch_size, device=dev).view(batch_size, 1, 1)
        self._y_idx = torch.arange(height, device=dev).view(1, height, 1)
        self._x_idx = torch.arange(width, device=dev).view(1, 1, width)
        self._barangeN = torch.arange(batch_size, device=dev).view(batch_size, 1)
        self._dy_t = torch.tensor(_DY, device=dev)
        self._dx_t = torch.tensor(_DX, device=dev)

    # ------------------------------------------------------------------
    # Reset / placement
    # ------------------------------------------------------------------

    def reset(self, walls: torch.Tensor | None = None) -> dict:
        """Initialise all ``B`` games; returns the public state dict."""
        B, H, W, T, N = self.B, self.H, self.W, self.T, self.N
        dev = self.device

        if walls is None:
            walls = self._generate_random_maps()
        self.walls = walls.bool().to(dev)
        self.passable = ~self.walls

        # SoA fighter buffers.
        self.fx = torch.zeros(B, N, dtype=torch.long, device=dev)
        self.fy = torch.zeros(B, N, dtype=torch.long, device=dev)
        self.fhealth = torch.full((B, N), MAX_HEALTH, dtype=torch.int32, device=dev)
        self.fteam = torch.zeros(B, N, dtype=torch.long, device=dev)
        self.occ = torch.full((B, H, W), -1, dtype=torch.long, device=dev)

        self.cursor_pos = torch.zeros(B, T, 2, dtype=torch.long, device=dev)
        self.cursor_val = torch.full((B, T), CURSOR_SEED, dtype=torch.int32, device=dev)
        self.team_alive = torch.ones(B, T, dtype=torch.bool, device=dev)
        self.gradient = torch.full((B, T, H, W), GRAD_INIT, dtype=torch.int32, device=dev)
        self._wall_grad = self.walls.unsqueeze(1).expand(B, T, H, W)

        self._place_teams()
        self._rebuild_occ()
        self._rebuild_views()
        self.tick = 0
        return self._get_state()

    def _generate_random_maps(self) -> torch.Tensor:
        B, H, W = self.B, self.H, self.W
        walls = torch.zeros(B, H, W, dtype=torch.bool)
        walls[:, 0, :] = True
        walls[:, -1, :] = True
        walls[:, :, 0] = True
        walls[:, :, -1] = True
        walls[:, 1:-1, 1:-1] = torch.rand(B, H - 2, W - 2) < 0.08
        return walls

    def _place_teams(self) -> None:
        """Place ``fighters_per_team`` fighters per team in vertical strips, one
        per (distinct, passable) cell, and seat each cursor at the strip centre."""
        B, H, W, T = self.B, self.H, self.W, self.T
        dev = self.device
        per = self.fighters_per_team
        strip_w = max(1, (W - 2) // T)
        for t in range(T):
            x0 = 1 + t * strip_w
            x1 = min(x0 + strip_w, W - 1)
            self.cursor_pos[:, t, 0] = H // 2
            self.cursor_pos[:, t, 1] = (x0 + x1) // 2
            region = torch.zeros(B, H, W, dtype=torch.bool, device=dev)
            region[:, 2:H - 2, x0:x1] = True
            region &= self.passable
            flat = region.view(B, -1).float()
            avail = int(flat.sum(dim=1).min().item())
            k = min(per, max(avail, 1))
            sel = torch.multinomial(flat + 1e-8, k, replacement=False)   # (B, k)
            sy = sel // W
            sx = sel % W
            base = t * per
            self.fy[:, base:base + k] = sy
            self.fx[:, base:base + k] = sx
            self.fteam[:, base:base + k] = t
            # Any unfilled slots for this team (avail < per): park them on the
            # cursor cell so the count stays exact (rare on real maps).
            if k < per:
                self.fy[:, base + k:base + per] = H // 2
                self.fx[:, base + k:base + per] = (x0 + x1) // 2
                self.fteam[:, base + k:base + per] = t

    # ------------------------------------------------------------------
    # Derived views (so the public interface is unchanged)
    # ------------------------------------------------------------------

    def _rebuild_occ(self) -> None:
        """Rebuild the one-per-cell occupancy grid from the SoA (last-writer per
        cell; placement guarantees distinct cells so there is no real overlap)."""
        B, H, W, N = self.B, self.H, self.W, self.N
        self.occ.fill_(-1)
        flat = (self.fy * W + self.fx)                      # (B, N)
        slots = torch.arange(N, device=self.device).expand(B, N)
        self.occ.view(B, -1).scatter_(1, flat, slots)

    def _rebuild_views(self) -> None:
        """Rebuild ``team_oh`` (B,T,H,W), ``health`` (B,H,W) and ``team_alive``
        as scatter views of the SoA, and the per-team active census."""
        B, H, W, T, N = self.B, self.H, self.W, self.T, self.N
        flat = self.fy * W + self.fx                        # (B, N)
        team_oh = torch.zeros(B, T, H * W, device=self.device)
        idx = self.fteam.unsqueeze(1)                       # placeholder unused
        # presence: scatter 1 into [b, fteam, cell]
        oh = torch.zeros(B, T, H * W, device=self.device)
        b_ar = self._barangeN.expand(B, N)
        oh[b_ar.reshape(-1), self.fteam.reshape(-1), flat.reshape(-1)] = 1.0
        self.team_oh = oh.view(B, T, H, W)
        health = torch.zeros(B, H * W, device=self.device)
        health[b_ar.reshape(-1), flat.reshape(-1)] = self.fhealth.float().reshape(-1).clamp(min=0)
        self.health = health.view(B, H, W)
        # active census per team
        active = torch.zeros(B, T, device=self.device, dtype=torch.long)
        active.scatter_add_(1, self.fteam, torch.ones_like(self.fteam))
        self.active_fighters = active
        self.team_alive = active > 0

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def step(self, cursor_actions: torch.Tensor | None = None):
        """Advance one tick. Returns ``(state, done, info)``."""
        if cursor_actions is not None:
            self._move_cursors(cursor_actions)
        self._seed_and_spread_gradient()
        self._move_fighters()
        self._resolve_combat()
        self._rebuild_views()
        self.tick += 1
        teams_left = self.team_alive.sum(dim=1)
        done = teams_left <= 1
        return self._get_state(), done, self._get_info()

    def _move_cursors(self, actions: torch.Tensor) -> None:
        new = self.cursor_pos + actions.long()
        new[:, :, 0].clamp_(1, self.H - 2)
        new[:, :, 1].clamp_(1, self.W - 2)
        b = torch.arange(self.B, device=self.device)
        for t in range(self.T):
            ny, nx = new[:, t, 0], new[:, t, 1]
            ok = self.passable[b, ny, nx] & self.team_alive[:, t]
            self.cursor_pos[:, t, 0] = torch.where(ok, ny, self.cursor_pos[:, t, 0])
            self.cursor_pos[:, t, 1] = torch.where(ok, nx, self.cursor_pos[:, t, 1])
        # cursor seed decays (val-- when moved or every 13th tick) — keep >0.
        dec = (self.tick % 13 == 0)
        if dec:
            self.cursor_val = (self.cursor_val - 1).clamp(min=1)

    def _seed_and_spread_gradient(self) -> None:
        """Overwrite each cursor cell with its seed, then relax the persistent
        per-team distance field with ``grad_sweeps`` neighbour-min passes."""
        B, T, H, W = self.B, self.T, self.H, self.W
        b = torch.arange(B, device=self.device)
        for t in range(T):
            cy = self.cursor_pos[:, t, 0]
            cx = self.cursor_pos[:, t, 1]
            self.gradient[b, t, cy, cx] = self.cursor_val[:, t]
        g = self.gradient
        wall = self._wall_grad
        for _ in range(self.grad_sweeps):
            p = torch.nn.functional.pad(g, (1, 1, 1, 1), value=GRAD_INIT)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    shifted = p[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                    torch.minimum(g, (shifted + 1).clamp(max=GRAD_INIT), out=g)
            g[wall] = GRAD_INIT

    # ------------------------------------------------------------------
    # Fighter movement — gradient descent + priority-claim collision
    # ------------------------------------------------------------------

    def _best_dir(self):
        """For every fighter, the 8-neighbour direction of steepest gradient
        descent toward its team's cursor, plus that neighbour's (y,x). Returns
        ``(best_dir (B,N), ny (B,N), nx (B,N), cur_grad (B,N))``."""
        B, N, H, W = self.B, self.N, self.H, self.W
        b = self._barangeN.expand(B, N)
        cur = self.gradient[b, self.fteam, self.fy, self.fx]            # (B,N)
        best_g = cur.clone()
        best_dir = torch.full((B, N), -1, dtype=torch.long, device=self.device)
        for i in range(8):
            ny = (self.fy + _DY[i]).clamp(0, H - 1)
            nx = (self.fx + _DX[i]).clamp(0, W - 1)
            ng = self.gradient[b, self.fteam, ny, nx]
            better = ng < best_g
            best_g = torch.where(better, ng, best_g)
            best_dir = torch.where(better, torch.full_like(best_dir, i), best_dir)
        return best_dir, best_g, cur

    def _move_fighters(self) -> None:
        """Move each fighter one step down its gradient into a free cell,
        resolving same-target contention by lowest slot index (deterministic
        analogue of the C's sequential first-free-cell-wins)."""
        B, N, H, W = self.B, self.N, self.H, self.W
        best_dir, best_g, cur = self._best_dir()
        wants = (best_dir >= 0) & (best_g < cur)                       # downhill move
        dy = torch.where(best_dir >= 0, self._dy_t[best_dir.clamp(min=0)], 0)
        dx = torch.where(best_dir >= 0, self._dx_t[best_dir.clamp(min=0)], 0)
        ty = (self.fy + dy).clamp(0, H - 1)
        tx = (self.fx + dx).clamp(0, W - 1)
        tcell = ty * W + tx
        # eligible target: in bounds, passable, currently empty.
        empty = self.occ.view(B, -1).gather(1, tcell) == -1
        passable = self.passable.view(B, -1).gather(1, tcell)
        eligible = wants & empty & passable
        # Contention: for each target cell, the lowest slot index wins.
        slots = torch.arange(N, device=self.device).expand(B, N)
        BIG = N + 1
        claim = torch.full((B, H * W), BIG, dtype=torch.long, device=self.device)
        claim.scatter_reduce_(1, torch.where(eligible, tcell, tcell),
                              torch.where(eligible, slots, torch.full_like(slots, BIG)),
                              reduce='amin', include_self=True)
        winner = claim.gather(1, tcell) == slots
        moves = eligible & winner
        # Commit: vacate old, fill new.
        self.fy = torch.where(moves, ty, self.fy)
        self.fx = torch.where(moves, tx, self.fx)
        self._rebuild_occ()
        # stash for combat: who was blocked + their intended front dir
        self._blocked = wants & ~moves
        self._front_dy, self._front_dx = dy, dx

    # ------------------------------------------------------------------
    # Combat — convert, never delete
    # ------------------------------------------------------------------

    def _resolve_combat(self) -> None:
        """Blocked fighters front-attack the enemy in their intended direction;
        accumulated damage is applied to targets; targets at <0 health rebase and
        defect to an attacker's team. Total fighter count is invariant."""
        B, N, H, W = self.B, self.N, self.H, self.W
        if not bool(self._blocked.any()):
            return
        fy, fx = self.fy, self.fx
        ty = (fy + self._front_dy).clamp(0, H - 1)
        tx = (fx + self._front_dx).clamp(0, W - 1)
        tcell = ty * W + tx
        tgt_slot = self.occ.view(B, -1).gather(1, tcell)              # (B,N) slot at target, -1 empty
        b = self._barangeN.expand(B, N)
        tgt_team = torch.where(tgt_slot >= 0,
                               self.fteam.gather(1, tgt_slot.clamp(min=0)),
                               self.fteam)
        is_enemy = (tgt_slot >= 0) & (tgt_team != self.fteam)
        attack = self._blocked & is_enemy                             # (B,N) attackers
        if not bool(attack.any()):
            return
        # Accumulate damage onto target slots.
        dmg = torch.zeros(B, N, dtype=torch.int32, device=self.device)
        a_tgt = torch.where(attack, tgt_slot, torch.zeros_like(tgt_slot))
        dmg.scatter_add_(1, a_tgt, torch.where(attack, torch.full_like(tgt_slot, ATTACK, dtype=torch.int32),
                                               torch.zeros_like(tgt_slot, dtype=torch.int32)))
        self.fhealth = self.fhealth - dmg
        # Conversion: any slot now <0 defects to a (lowest-slot-index) attacker's team.
        neg = self.fhealth < 0
        if bool(neg.any()):
            # which team converts this target: pick the lowest-index attacker on it.
            BIG = N + 1
            owner = torch.full((B, N), BIG, dtype=torch.long, device=self.device)
            atk_slots = torch.arange(N, device=self.device).expand(B, N)
            owner.scatter_reduce_(1, a_tgt,
                                  torch.where(attack, atk_slots, torch.full_like(atk_slots, BIG)),
                                  reduce='amin', include_self=True)
            has_owner = owner < BIG
            conv = neg & has_owner
            if bool(conv.any()):
                new_team = self.fteam.gather(1, owner.clamp(max=N - 1))
                # rebase health up by NEW_HEALTH until >= 0
                h = self.fhealth
                steps = ((-h + NEW_HEALTH - 1) // NEW_HEALTH).clamp(min=0)
                h = h + steps * NEW_HEALTH
                self.fhealth = torch.where(conv, h, self.fhealth)
                self.fteam = torch.where(conv, new_team, self.fteam)
            # any still-negative-with-no-owner: clamp to 0 (stays own team, alive)
            self.fhealth = self.fhealth.clamp(min=0)
        self.fhealth = self.fhealth.clamp(max=MAX_HEALTH)

    # ------------------------------------------------------------------
    # State / Info / Observation (unchanged public surface)
    # ------------------------------------------------------------------

    def _get_state(self) -> dict:
        return {
            'team_oh': self.team_oh, 'health': self.health,
            'gradient': self.gradient, 'cursor_pos': self.cursor_pos,
            'team_alive': self.team_alive, 'walls': self.walls, 'tick': self.tick,
        }

    def _get_info(self) -> dict:
        fighters = self.active_fighters.float()
        total = fighters.sum(dim=1)
        best = fighters.max(dim=1)
        return {
            'fighters_per_team': fighters, 'total_fighters': total,
            'best_team': best.indices, 'best_count': best.values,
            'dominance': best.values / total.clamp(min=1), 'tick': self.tick,
        }

    def get_observation(self) -> torch.Tensor:
        """``(B, 1+3T, H, W)`` — walls, per-team presence, per-team normalised
        gradient, per-team health. Unchanged layout for the policy."""
        T = self.T
        channels = [self.walls.float()]
        for t in range(T):
            channels.append(self.team_oh[:, t])
        for t in range(T):
            g = self.gradient[:, t].float().clone()
            g[g >= GRAD_INIT] = 0
            gmax = g.amax(dim=(1, 2), keepdim=True).clamp(min=1)
            channels.append(g / gmax)
        for t in range(T):
            channels.append(self.team_oh[:, t] * self.health / MAX_HEALTH)
        return torch.stack(channels, dim=1)
