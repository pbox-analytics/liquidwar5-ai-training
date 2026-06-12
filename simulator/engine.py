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

import random

import torch

MAX_TEAMS = 6

#: Map archetype names; index = the archetype id ``_gen_one_map`` draws. The play
#: server's map picker (force one) and telemetry (log which one) both use these.
MAP_NAMES = ("Open", "Barrier", "Pillars", "Scatter", "Rooms", "Corners")
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
        # MOMENTUM alignment table (9, 8): row = a fighter's PREVIOUS heading
        # (direction index 0-7; row 8 = was stationary), col = candidate
        # direction. Keeping the same heading earns +6, a 45° turn +3, a 90°
        # turn 0, a reversal (135-180°) -4 — in gradient-field units (one
        # neighbour step costs 10 orthogonal / 14 diagonal), scaled by the
        # ``_inertia`` knob and SUBTRACTED from the candidate score in
        # :meth:`_move_fighters` (lower score = better). Row 8 is all zeros: a
        # fighter at rest has no heading to keep. Built once here; per tick
        # it costs a single (B,N) -> (B,N,8) gather.
        tab = torch.zeros(9, 8, device=dev)
        for p in range(8):
            pn = (_DY[p] * _DY[p] + _DX[p] * _DX[p]) ** 0.5
            for c in range(8):
                cn = (_DY[c] * _DY[c] + _DX[c] * _DX[c]) ** 0.5
                cos = (_DY[p] * _DY[c] + _DX[p] * _DX[c]) / (pn * cn)
                tab[p, c] = (6.0 if cos > 0.92 else
                             3.0 if cos > 0.5 else
                             0.0 if cos > -0.5 else -4.0)
        self._mom_tab = tab

    # ------------------------------------------------------------------
    # Reset / placement
    # ------------------------------------------------------------------

    def reset(self, walls: torch.Tensor | None = None) -> dict:
        """Initialise all ``B`` games; returns the public state dict."""
        # NEVER free a captured graph's memory pool with replays in flight: a
        # mid-game reset (map switch) lands ~16ms after a replay was enqueued,
        # and dropping self._graph while its kernels still run had them writing
        # freed pool memory -> CUDA illegal access poisoning the whole context.
        if getattr(self, "_graph", None) is not None:
            torch.cuda.synchronize()
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
        # Per-fighter LAST TAKEN heading (0-7 = direction index into _DY/_DX,
        # 8 = stood still) feeding the discrete momentum bias in
        # :meth:`_move_fighters`. Updated IN PLACE (``copy_``) at the end of
        # every movement sub-step — never rebound, so its address is stable
        # and a captured CUDA graph reads/writes it like ``cursor_val`` /
        # ``gradient`` (no ``_persist`` round-trip needed).
        self._fdir = torch.full((B, N), 8, dtype=torch.long, device=dev)

        self.cursor_pos = torch.zeros(B, T, 2, dtype=torch.long, device=dev)
        self.cursor_val = torch.full((B, T), CURSOR_SEED, dtype=torch.int32, device=dev)
        self.team_alive = torch.ones(B, T, dtype=torch.bool, device=dev)
        # float32, not int32: the relax sweeps now run as pooling ops (float
        # only). Every value is an exact integer <= GRAD_INIT + 14 < 2^24, so
        # float32 represents the field exactly — bit-identical distances.
        self.gradient = torch.full((B, T, H, W), GRAD_INIT, dtype=torch.float32, device=dev)
        self._wall_grad = self.walls.unsqueeze(1).expand(B, T, H, W)

        # Cross-team well SLOTS, one per team (Doom gravity / Maelstrom current),
        # written IN-PLACE by the play server each tick. str==0 / horizon==0 means
        # "off" — the loops always run T fixed iterations so the kernel sequence
        # is static (CUDA-graph-safe); a zeroed slot is a no-op force. Gated by
        # ``_wells_enabled`` (play only) so training skips the loops entirely.
        self._doom_pos = torch.zeros(B, T, 2, device=dev)
        self._doom_str = torch.zeros(B, T, device=dev)
        self._doom_range = torch.ones(B, T, device=dev)
        self._doom_horizon = torch.zeros(B, T, device=dev)
        self._doom_cap = torch.zeros(B, T, device=dev)
        self._vortex_pos = torch.zeros(B, T, 2, device=dev)
        self._vortex_str = torch.zeros(B, T, device=dev)
        self._vortex_range = torch.ones(B, T, device=dev)
        self._vortex_sign = torch.ones(B, T, device=dev)
        self._vortex_rad = torch.zeros(B, T, device=dev)
        # In-graph animation clock: python ``self.tick`` is invisible to a replayed
        # graph (it would bake in as a constant), so every tensor expression that
        # animates over time reads this GPU scalar instead, incremented inside the
        # captured step.
        self._tick_f = torch.zeros((), device=dev)
        self._graph = None              # captured CUDA graph (per game; reset() invalidates)
        self._fixed_sweeps = None       # gradient sweeps/tick under graph capture (else converge+early-out)
        self._persist = {}              # cross-tick state buffers (graph replay round-trip)
        self._place_teams()
        self._rebuild_occ()
        self._rebuild_views()
        self.tick = 0
        return self._get_state()

    def _generate_random_maps(self) -> torch.Tensor:
        """Procedural map generator — each ``reset`` draws a random ARCHETYPE with
        randomized parameters, always point-symmetric (180° rotation, so both
        teams get identical terrain) and connectivity-checked (no walled-off
        pockets where fighters would strand). Archetypes: open arena, central
        barrier (gapped), pillars, scattered blocks, four rooms, quadrant blocks.
        The random parameters yield far more than 100 distinct maps; every game is
        a fresh one. The gradient flood-fill routes the army around whatever walls
        appear, so no per-map movement code is needed.

        :returns: ``(B, H, W)`` bool wall grid (same map across the batch).
        """
        B, H, W = self.B, self.H, self.W
        w = self._gen_one_map()
        for _ in range(11):                              # redraw until connected
            if self._is_connected(w):
                break
            w = self._gen_one_map()
        return w.unsqueeze(0).expand(B, H, W).contiguous()

    def _gen_one_map(self) -> torch.Tensor:
        """Draw one random point-symmetric wall layout (no connectivity guarantee
        — :meth:`_generate_random_maps` retries). Returns an ``(H, W)`` bool grid.

        :returns: ``(H, W)`` bool wall grid on ``self.device``.
        """
        H, W, dev = self.H, self.W, self.device
        w = torch.zeros(H, W, dtype=torch.bool, device=dev)
        w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = True       # solid border

        def box(y0: float, y1: float, x0: float, x1: float) -> None:
            """Fill a rectangle and its 180° rotation (keeps the map fair)."""
            iy0, iy1 = max(1, int(y0)), min(H - 1, int(y1))
            ix0, ix1 = max(1, int(x0)), min(W - 1, int(x1))
            if iy1 > iy0 and ix1 > ix0:
                w[iy0:iy1, ix0:ix1] = True
                w[H - iy1:H - iy0, W - ix1:W - ix0] = True

        th = max(2, round(H / 48))
        # ``_map_choice`` (set by the play server's map picker) forces an archetype;
        # None -> a random one each game.
        choice = getattr(self, "_map_choice", None)
        arch = choice if choice is not None else random.randint(0, 5)
        self._last_arch = arch                               # remember archetype, for telemetry
        if arch == 0:                                        # open arena (border only)
            pass
        elif arch == 1:                                      # central barrier with gaps
            gaps = random.randint(1, 3)
            if random.random() < 0.5:                        # vertical barrier
                cx = W // 2
                for a, b in self._gapped(H, gaps):
                    box(a, b, cx - th, cx + th + 1)
            else:                                            # horizontal barrier
                cy = H // 2
                for a, b in self._gapped(W, gaps):
                    box(cy - th, cy + th + 1, a, b)
        elif arch == 2:                                      # a few pillars
            for _ in range(random.randint(2, 5)):
                ph, pw = random.randint(H // 10, H // 3), random.randint(W // 12, W // 4)
                py, px = random.randint(3, max(4, H - 3 - ph)), random.randint(3, max(4, W // 2 - pw))
                box(py, py + ph, px, px + pw)
        elif arch == 3:                                      # scattered blocks
            for _ in range(random.randint(8, 18)):
                s = random.randint(2 * th, max(2 * th + 1, H // 9))
                py, px = random.randint(3, max(4, H - 3 - s)), random.randint(3, max(4, W // 2 - s))
                box(py, py + s, px, px + s)
        elif arch == 4:                                      # four rooms (cross + doorways)
            cy, cx = H // 2, W // 2
            dh, dw = random.randint(H // 12, H // 6), random.randint(W // 12, W // 6)
            w[cy - th:cy + th, :] = True
            w[:, cx - th:cx + th] = True
            for dx in (W // 4, 3 * W // 4):                  # doorways through the horizontal wall
                w[cy - th:cy + th, dx - dw:dx + dw] = False
            for dy in (H // 4, 3 * H // 4):                  # doorways through the vertical wall
                w[dy - dh:dy + dh, cx - th:cx + th] = False
            w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = True   # re-seal the border
        else:                                                # walled corners, open center
            bh, bw = random.randint(H // 6, H // 3), random.randint(W // 8, W // 4)
            box(2, 2 + bh, 2, 2 + bw)                        # top-left (+ rot bottom-right)
            box(2, 2 + bh, W - 2 - bw, W - 2)                # top-right (+ rot bottom-left)
        return w

    def _gapped(self, length: int, gaps: int) -> list[tuple[int, int]]:
        """Wall segments spanning ``[1, length-1]`` broken by ``gaps`` random
        chokepoint gaps, so a barrier never fully seals the arena.

        :param length: Span to fill (the barrier's long axis).
        :param gaps: Number of chokepoint gaps to leave.
        :returns: List of ``(start, end)`` wall segments along the span.
        """
        gw = max(self.H // 14, 6)                            # gap (chokepoint) width
        lo, hi = length // 6, max(length // 6 + 1, 5 * length // 6)
        cuts = sorted(random.sample(range(lo, hi), min(gaps, hi - lo)))
        segs: list[tuple[int, int]] = []
        a = 1
        for c in cuts:
            if c - gw // 2 > a:
                segs.append((a, c - gw // 2))
            a = c + gw // 2
        if length - 1 > a:
            segs.append((a, length - 1))
        return segs

    def _is_connected(self, w: torch.Tensor) -> bool:
        """True if the passable area is essentially one region (>=85% reachable
        from a seed via 4-connectivity) — rejects walled-off pockets.

        :param w: ``(H, W)`` bool wall grid.
        :returns: Whether the open area is one connected region.
        """
        H, W = self.H, self.W
        passable = ~w
        total = int(passable.sum())
        if total == 0:
            return False
        seed = int(passable.view(-1).nonzero()[0])
        reach = torch.zeros_like(w)
        reach.view(-1)[seed] = True
        thresh, last = 0.85 * total, 0
        for it in range(H + W):                              # iterative 4-conn flood
            nxt = reach.clone()
            nxt[1:, :] |= reach[:-1, :]
            nxt[:-1, :] |= reach[1:, :]
            nxt[:, 1:] |= reach[:, :-1]
            nxt[:, :-1] |= reach[:, 1:]
            reach = nxt & passable
            if it % 24 == 23:                                # .sum() syncs -> check periodically
                s = int(reach.sum())
                if s >= thresh:                              # connected enough -> done early
                    return True
                if s == last:                                # flood stalled -> walled-off pocket
                    break
                last = s
        return int(reach.sum()) >= thresh

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
            region = torch.zeros(B, H, W, dtype=torch.bool, device=dev)
            region[:, 2:H - 2, x0:x1] = True
            region &= self.passable
            # Seat the cursor at the strip centre — but if the map dropped a wall
            # block there, snap to the nearest passable cell, else the cursor is
            # stuck inside a wall and the army can't gather on it. (Walls are the
            # same across the batch, so cell from ``[0]`` applies to all.)
            cy0, cx0 = H // 2, (x0 + x1) // 2
            if not bool(self.passable[0, cy0, cx0]):
                ys, xs = torch.where(region[0])
                if ys.numel() == 0:                          # whole strip walled (rare) -> any open cell
                    ys, xs = torch.where(self.passable[0])
                if ys.numel():
                    j = int(((ys - cy0) ** 2 + (xs - cx0) ** 2).argmin())
                    cy0, cx0 = int(ys[j]), int(xs[j])
            self.cursor_pos[:, t, 0] = cy0
            self.cursor_pos[:, t, 1] = cx0
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
            # (passable) cursor cell so the count stays exact — never in a wall now.
            if k < per:
                self.fy[:, base + k:base + per] = cy0
                self.fx[:, base + k:base + per] = cx0
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
        # GPU scalar, not python 1.0: a python-scalar RHS in an index-put makes
        # an unpinned CPU->CUDA copy, which CUDA graph capture forbids
        if not hasattr(self, "_one_f"):
            self._one_f = torch.ones((), device=self.device)
        oh[b_ar.reshape(-1), self.fteam.reshape(-1), flat.reshape(-1)] = self._one_f
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
            self._move_cursors(cursor_actions)        # eager (python fast path at B=1)
        if not self._graph_step():
            self._step_body()
        self.tick += 1
        teams_left = self.team_alive.sum(dim=1)
        done = teams_left <= 1
        return self._get_state(), done, self._get_info()

    def _step_body(self) -> None:
        """Everything between cursor input and state readout — the capturable
        region: NO GPU->CPU syncs, NO data-dependent python control flow, and
        every cross-tick tensor folded back into a stable buffer at the end."""
        self._seed_and_spread_gradient()
        # Units advance multiple cells/tick (scaled to grid width, matching the
        # cursor) so the army keeps pace instead of crawling behind a fast cursor.
        # Each sub-step is a move + combat resolve down the same per-tick gradient;
        # the army still trails because it's chasing the cursor's lead.
        if not hasattr(self, "unit_speed"):
            self.unit_speed = max(1, round(self.W / 96))
        for _ in range(self.unit_speed):
            self._move_fighters()
            self._resolve_combat()
        # BLACK HOLE event-horizon capture (Doom): enemy fighters that reach a well's
        # core DEFECT to the well's team — the hole devours what its gravity pulls in, so
        # the strip steals the enemy army instead of just relocating it onto you. Convert,
        # not delete (count invariant). horizon==0 (slot off) grabs nothing.
        if getattr(self, "_wells_enabled", False):
            # the Maelstrom counter-current / Wall brace also slow the DEVOUR
            # (same shield as the move-phase well forces, recomputed per tick)
            B2, N2 = self.B, self.N
            o_pos = self._vortex_pos.gather(1, self.fteam.unsqueeze(-1).expand(B2, N2, 2))
            o_str = self._vortex_str.gather(1, self.fteam)
            o_R = self._vortex_range.gather(1, self.fteam).clamp(min=1.0)
            oy = o_pos[..., 0] - self.fy
            ox = o_pos[..., 1] - self.fx
            o_d = (oy * oy + ox * ox).sqrt()
            o_fall = (((1.5 * o_R - o_d) / (0.3 * o_R)).clamp(0.0, 1.0)
                      * (o_str > 0).float())
            wall_t = getattr(self, "_wall", None)
            brace = ((wall_t.abs().sum(-1).gather(1, self.fteam) > 0).float()
                     if wall_t is not None else torch.zeros(B2, N2, device=self.device))
            # the DEVOUR shield is harder than the force shield (95% vs 85%):
            # a parked Doom's horizon overlaps the storm's near arc and the
            # rotation cycles fighters through it — at 85% that still ate the
            # shell over ~30s; at 95% the storm holds while contact combat
            # stays an honest fight
            cap_shield = (1.0 - 0.95 * o_fall) * (1.0 - 0.45 * brace)
            for t in range(self.T):
                R_h = self._doom_horizon[:, t:t + 1]
                dy = self._doom_pos[:, t, 0:1] - self.fy.float()
                dx = self._doom_pos[:, t, 1:2] - self.fx.float()
                grab = ((dy * dy + dx * dx) <= R_h * R_h) & (self.fteam != t)
                grab = grab & (torch.rand(self.B, self.N, device=self.device)
                               < self._doom_cap[:, t:t + 1] * cap_shield)  # devour GRADUALLY — a drain you can fight
                # DIGESTION LIMIT: a well can only swallow so fast. Without
                # this, a blob overlapping the horizon converted hundreds per
                # tick and every conversion grew the wielder -> the snowball
                # that tore through every stance. Cap expected conversions to
                # ~6 + 0.1% of the wielder's mass per tick (probabilistic
                # thinning — vectorized, graph-safe).
                k_digest = 6.0 + 0.001 * self.active_fighters[:, t:t + 1].float()
                g_cnt = grab.float().sum(1, keepdim=True).clamp(min=1.0)
                grab = grab & (torch.rand(self.B, self.N, device=self.device)
                               < (k_digest / g_cnt).clamp(max=1.0))
                self.fteam = torch.where(grab, torch.full_like(self.fteam, t), self.fteam)
                self.fhealth = torch.where(grab, torch.full_like(self.fhealth, NEW_HEALTH), self.fhealth)
        self._rebuild_views()
        # Fold every cross-tick tensor back into its persistent buffer: a replayed
        # graph reads its INPUTS from fixed addresses, so state produced at pool
        # addresses must round-trip into those input buffers each tick (eager mode
        # pays 6 tiny copies for the same code path).
        for name in ("fy", "fx", "fteam", "fhealth", "fvy", "fvx"):
            cur = getattr(self, name)
            buf = self._persist.get(name) if hasattr(self, "_persist") else None
            if buf is None or buf.data_ptr() == cur.data_ptr():
                if not hasattr(self, "_persist"):
                    self._persist = {}
                self._persist[name] = cur
            else:
                buf.copy_(cur)
                setattr(self, name, buf)
        self._tick_f += 1

    def _graph_step(self) -> bool:
        """CUDA-graph fast path: capture ``_step_body`` once per game, then replay
        it as a single unit — at B=1 the tick is launch-overhead-bound (~4k tiny
        kernels), so replay roughly halves the tick. Captures only after the cold
        flood has converged (fixed in-graph sweep count tracks a warm field).
        Returns True when it ran the tick."""
        if not (getattr(self, "_cuda_graph", False) and self.B == 1
                and self.tick >= 70 and self.gradient.is_cuda):
            return False
        if self._graph is None:
            # A failed capture corrupts python state: the body's attribute
            # rebindings run while the recorded kernels never execute, leaving
            # self.fy etc. pointing at garbage pool tensors. Snapshot the
            # bindings so the except path can restore a consistent engine.
            saved = {k: getattr(self, k, None)
                     for k in ("fy", "fx", "fteam", "fhealth", "fvy", "fvx",
                               "team_oh", "health", "team_alive", "active_fighters")}
            saved_persist = dict(self._persist)
            try:
                # in-graph gradient mode: fixed sweeps (a warm field needs
                # cursor_speed + margin), no convergence sync
                self._fixed_sweeps = getattr(self, "cursor_speed", max(1, round(self.W / 96))) + 4
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):                       # warmup (allocator/cublas state)
                        self._step_body()
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._step_body()
                self._graph = g
            except Exception as exc:                          # any capture failure -> eager forever
                print(f"[engine] CUDA graph capture failed ({exc!r}); staying eager", flush=True)
                torch.cuda.synchronize()
                for k, v in saved.items():
                    if v is not None:
                        setattr(self, k, v)
                self._persist = saved_persist
                self._cuda_graph = False
                self._fixed_sweeps = None
                self._step_body()
                return True
        self._graph.replay()
        return True

    def _move_cursors(self, actions: torch.Tensor) -> None:
        # Cursor speed scales with grid width so the on-screen feel stays constant
        # as the map grows: 1 cell/tick suited the original ~110-wide grid; a
        # 288-wide grid needs ~3. actions are unit directions (±1).
        if not hasattr(self, "cursor_speed"):
            self.cursor_speed = max(1, round(self.W / 96))
        if self.B == 1:
            # PLAY fast path: stepping <=MAX_TEAMS cursors through the slide
            # logic below costs ~180 kernel launches (~2ms/tick) for a few
            # integers of real work. Walls are static per game, so mirror
            # ``passable`` to CPU once (identity-keyed: reset() reassigns it)
            # and do the whole thing in python ints — two tiny syncs replace
            # the launch storm. Same semantics as the tensor path.
            if getattr(self, "_pass_cpu_src", None) is not self.passable:
                self._pass_cpu = self.passable[0].tolist()      # (H,W) nested bools
                self._pass_cpu_src = self.passable
            pas = self._pass_cpu
            adir = actions[0].long().clamp(-1, 1).tolist()
            cur = self.cursor_pos[0].tolist()
            alive = self.team_alive[0].tolist()
            # optional per-team speeds (rooms: each Doom holder slows ITSELF)
            speeds = getattr(self, "_cursor_speed_t", None)
            moved_l = [False] * self.T
            cost_l = [0] * self.T
            for t in range(self.T):
                dy0, dx0 = int(adir[t][0]), int(adir[t][1])
                if not alive[t] or (dy0 == 0 and dx0 == 0):
                    continue
                cy, cx = int(cur[t][0]), int(cur[t][1])
                for _ in range(int(speeds[t]) if speeds is not None else self.cursor_speed):
                    for dy, dx in ((dy0, dx0), (dy0, 0), (0, dx0)):
                        ny = min(max(cy + dy, 1), self.H - 2)
                        nx = min(max(cx + dx, 1), self.W - 2)
                        if (ny != cy or nx != cx) and pas[ny][nx]:
                            cost_l[t] += 14 if (dy != 0 and dx != 0) else 10
                            cy, cx = ny, nx
                            moved_l[t] = True
                            break
                cur[t] = [cy, cx]
            self.cursor_pos[0] = torch.as_tensor(cur, dtype=self.cursor_pos.dtype,
                                                 device=self.device)
            decay = 10 if (self.tick % 13 == 0) else 0
            dec = torch.as_tensor([[c if m else decay for m, c in zip(moved_l, cost_l)]],
                                  dtype=torch.int32, device=self.device)
            self.cursor_val.sub_(dec).clamp_(min=1)   # in-place: graph input buffer
            return
        b = torch.arange(self.B, device=self.device)
        adir = actions.long().clamp(-1, 1)
        moved = torch.zeros(self.B, self.T, dtype=torch.bool, device=self.device)
        cost = torch.zeros(self.B, self.T, dtype=torch.int32, device=self.device)
        # Step one cell at a time up to cursor_speed, stopping at walls — so the
        # cursor can't slide through a maze barrier on a multi-cell move. Each
        # sub-step tries the full (possibly diagonal) move first, then SLIDES
        # along each single axis if a wall blocks it — so holding two directions
        # against a barrier glides the cursor along it toward a gap instead of
        # pinning it (no more releasing a key to line up with the opening).
        bt = b.unsqueeze(1).expand(self.B, self.T)                    # (B,T) batch index
        zero = torch.zeros_like(adir[:, :, 0])
        # optional per-(B,T) speeds (training parity: Doom holders glide slow,
        # same dial as play's per-seat list) — substeps beyond a team's speed
        # are masked off; the loop still runs cursor_speed (the max) times
        spd_bt = getattr(self, "_cursor_speed_bt", None)
        for k in range(self.cursor_speed):
            oy = self.cursor_pos[:, :, 0].clone()                     # (B,T) substep snapshot
            ox = self.cursor_pos[:, :, 1].clone()
            done = torch.zeros(self.B, self.T, dtype=torch.bool, device=self.device)
            allow = None if spd_bt is None else (spd_bt > k)
            for dy, dx in ((adir[:, :, 0], adir[:, :, 1]),
                           (adir[:, :, 0], zero), (zero, adir[:, :, 1])):
                ny = (oy + dy).clamp(1, self.H - 2)
                nx = (ox + dx).clamp(1, self.W - 2)
                ok = (~done & self.passable[bt, ny, nx] & self.team_alive
                      & ((ny != oy) | (nx != ox)))
                if allow is not None:
                    ok = ok & allow
                self.cursor_pos[:, :, 0] = torch.where(ok, ny, self.cursor_pos[:, :, 0])
                self.cursor_pos[:, :, 1] = torch.where(ok, nx, self.cursor_pos[:, :, 1])
                # Seed-decay bookkeeping below needs the OCTILE cost of the
                # step actually taken (a blocked diagonal that slides moves
                # orthogonally -> 10, not 14).
                step_c = ((dy != 0) & (dx != 0)).to(torch.int32) * 4 + 10
                cost += torch.where(ok, step_c, torch.zeros_like(step_c))
                done |= ok
            moved |= done
        # Seed decays whenever the cursor MOVES (or every 13th tick) so the
        # current cursor cell dominates the persistent field and fighters blob
        # around it rather than smearing toward stale old positions. The decay
        # must equal the cursor's per-tick OCTILE distance (orthogonal 10,
        # diagonal 14 — matching the gradient step costs below), else a moving
        # cursor leaves stale low values and the army smears.
        decay = 10 if (self.tick % 13 == 0) else 0
        dec = torch.where(moved, cost, torch.full_like(cost, decay))
        self.cursor_val.sub_(dec).clamp_(min=1)   # in-place: graph input buffer

    def _seed_and_spread_gradient(self) -> None:
        """Overwrite each cursor cell with its seed, then relax the persistent
        per-team distance field with ``grad_sweeps`` neighbour-min passes."""
        B, T, H, W = self.B, self.T, self.H, self.W
        b = torch.arange(B, device=self.device)
        for t in range(T):
            cy = self.cursor_pos[:, t, 0]
            cx = self.cursor_pos[:, t, 1]
            self.gradient[b, t, cy, cx] = self.cursor_val[:, t].float()
        g = self.gradient
        wall = self._wall_grad
        # Complete flood-fill: relax to CONVERGENCE (a full geodesic distance
        # field) rather than a fixed ``grad_sweeps`` cap. Only a converged field
        # reaches EVERY reachable cell, so fighters flow toward the cursor from
        # anywhere on the map and pathfind around walls. The old capped relax
        # (4 sweeps in training, 24 in play) left the far half of the army with
        # no gradient at all — which read as "blobs just converge" and is the
        # core gameplay break. Persistence + the per-move cursor-seed decrement
        # make the converged field track a MOVING cursor exactly: the decrement
        # offsets the cursor's displacement each tick, so no stale low values
        # survive. Hard-capped at 2*(H+W) (the longest possible geodesic path)
        # as a safety bound; the early-out keeps steady state cheap — a warm
        # field reconverges in a couple of sweeps once it has flooded once.
        # Per-tick sweep cap (set by the play server via ``_grad_cap``) spreads a
        # cold field's full flood over a few frames instead of one ~88ms stutter at
        # game start; the field persists, so the convergence accumulates tick to
        # tick. Default None -> full convergence each tick (training MDP unchanged).
        cap = getattr(self, "_grad_cap", None)
        # Each sweep used to be 8 shifted-slice minimums (~26 kernels) + a
        # clone + a torch.equal GPU->CPU sync. Now 3 pooling ops: a separable
        # cross-min (1x3 then 3x1 of THAT would over-reach; two independent
        # pools whose min is the 4-orthogonal+center min) for the +10 step and
        # one full 3x3 min for the +14 diagonal step. Including the center /
        # orthogonal cells in a candidate set is harmless: min(g, g+10) = g,
        # and the orthogonal cells' +14 candidates can never beat their own
        # +10 ones — the relaxed field is identical to the 8-slice version.
        # The convergence check (one sync + clone) runs every 4th sweep only.
        mp = torch.nn.functional.max_pool2d
        gv = g.view(B * T, 1, H, W)
        # Under CUDA-graph capture/replay (``_fixed_sweeps`` set) the sweep
        # count is FIXED — no clone, no torch.equal sync, no data-dependent
        # break (none are capturable). A warm field only needs to track the
        # cursor's <= cursor_speed cells/tick; capture begins after the cold
        # flood has converged eagerly.
        fixed = getattr(self, "_fixed_sweeps", None)
        n_sweeps = fixed or (min(2 * (H + W), cap) if cap else 2 * (H + W))
        wallv = wall.reshape(B * T, 1, H, W)
        prev = None
        for i in range(n_sweeps):
            if fixed is None and i % 4 == 0:
                prev = gv.clone()
            ng = -gv
            orth = torch.maximum(mp(ng, (1, 3), stride=1, padding=(0, 1)),
                                 mp(ng, (3, 1), stride=1, padding=(1, 0)))
            diag = mp(ng, 3, stride=1, padding=1)
            gv = torch.minimum(gv, torch.minimum((10 - orth), (14 - diag)).clamp(max=GRAD_INIT))
            gv = torch.where(wallv, torch.full_like(gv, GRAD_INIT), gv)
            if fixed is None and i % 4 == 3 and torch.equal(gv, prev):
                break
        # copy_ back, never rebind: ``self.gradient`` is a graph INPUT read at a
        # fixed address by the replayed kernels — the relaxed field must land in
        # that same storage or replays would relax a frozen snapshot forever.
        self.gradient.copy_(gv.view(B, T, H, W))

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
        """Move each fighter one step toward its cursor, trying its downhill
        neighbours in steepest-first PRIORITY ORDER (not just the single best),
        so a fighter whose best cell is taken REROUTES into its next-best free
        cell instead of stalling. Without this the whole army funnels onto one
        path and jams — a thin line reaches the cursor while the bulk deadlocks
        behind it. Candidates are retried over up to 8 rounds with occupancy
        rebuilt between rounds, so cells freed by movers open up for followers
        and the mass flows + spreads like liquid. Same-cell contention in a
        round is won by the lowest slot index (deterministic analogue of the
        C engine's sequential first-free-cell-wins)."""
        B, N, H, W = self.B, self.N, self.H, self.W
        b = self._barangeN.expand(B, N)
        cur = self.gradient[b, self.fteam, self.fy, self.fx]           # (B,N)
        BIGG = GRAD_INIT * 4
        ng = torch.empty(B, N, 8, dtype=self.gradient.dtype, device=self.device)
        ncell = torch.empty(B, N, 8, dtype=torch.long, device=self.device)
        for i in range(8):
            ny = (self.fy + _DY[i]).clamp(0, H - 1)
            nx = (self.fx + _DX[i]).clamp(0, W - 1)
            ng[:, :, i] = self.gradient[b, self.fteam, ny, nx]
            ncell[:, :, i] = ny * W + nx
        downhill = ng < cur.unsqueeze(-1)                              # strictly toward the cursor
        # IDLE JITTER as a Dictyostelium-style TRAVELING WAVE. A uniform random
        # twinkle felt flat; real slime-mold / social-amoeba colonies pulse in
        # rings that ripple OUTWARD from the attractant. So restlessness is gated
        # by a wave riding the distance field: fighters on a moving crest may step
        # onto an EQUAL-gradient neighbour (same ring -> they shuffle tangentially
        # without dispersing), and the crest travels outward over time -> the
        # settled mass undulates like a living colony. A 3% random base keeps a
        # little life off the crests. Strict-downhill flow is untouched (lower
        # cells sort first) so a moving cursor still pulls the mass in. Per-fighter
        # direction jitter (0..6) breaks lockstep -> independent-looking units.
        # (k=0.25 -> ~25-cell wavelength; w=0.15 -> crest ~0.6 cells/tick outward.)
        # Phase rides EUCLIDEAN distance to the cursor, not the octile gradient:
        # octile iso-contours are octagons, so gradient-phased crests read as
        # angular CHEVRONS sweeping the army; Euclidean rings are round, so the
        # undulation reads as organic ripples. (The radial vector is also what
        # the swirl/push/burst terms below need — computed once here.)
        cpos = self.cursor_pos[b, self.fteam]                          # (B,N,2) own-cursor pos
        ry = (cpos[..., 0] - self.fy).float(); rx = (cpos[..., 1] - self.fx).float()
        rn = (ry * ry + rx * rx).sqrt().clamp(min=1.0)                 # cells from the cursor
        phase = rn * 0.25 - self._tick_f * 0.15                           # crest travels outward
        # Broad crest (sin>0 => ~half the phase) so the wave band is wide enough
        # to read as a rolling EDGE RIPPLE: boundary fighters (the only ones with
        # empty cells to extend into) bulge outward as the crest sweeps past, then
        # the downhill pull retracts them -> the silhouette undulates like a
        # living membrane. Interior fighters can't bulge (no empty neighbour) so
        # the body stays dense; only the rim ripples.
        restless = ((torch.sin(phase) > 0.0)
                    | (torch.rand(B, N, device=self.device) < 0.07)).unsqueeze(-1)
        # +12 tolerance (not exact-equal): with octile costs neighbours sit at
        # cur±10/14, so a restless fighter steps onto the next ring out and the
        # downhill pull tugs it back -> a visible shimmer/undulation, not a freeze.
        # GATHER-BURST move: a per-team ``_burst`` (-1 gather inward / +1 burst
        # outward), set by the play server over a short two-phase window. The burst
        # phase relaxes the gate so the mass can flow OUTWARD a long way (a
        # shockwave); the radial score term below sets the direction.
        burst = getattr(self, "_burst", None)
        burst_f = burst.gather(1, self.fteam) if burst is not None else None   # (B,N) or None
        movable = downhill | (restless & (ng <= cur.unsqueeze(-1) + 12))
        if burst_f is not None:
            movable = movable | ((burst_f > 0).unsqueeze(-1) & (ng <= cur.unsqueeze(-1) + 36))
        # ACCRETION RING: a per-team ``_ring`` target ORBIT RADIUS (cells; 0 =
        # off). Fighters are biased toward the ring from BOTH sides — inward when
        # outside it, outward when inside — so with the swirl providing the orbit
        # the team forms a spinning annulus with an open centre (Doom's visible
        # black-hole disk) instead of a packed blob. The score term needs the
        # radial direction, added with the burst term below.
        ringk = getattr(self, "_ring", None)
        ring_f = ringk.gather(1, self.fteam) if ringk is not None else None    # (B,N) radius or 0
        if ring_f is not None:
            movable = movable | ((ring_f > 0).unsqueeze(-1) & (ng <= cur.unsqueeze(-1) + 36))
        # DRILL move: a per-team thrust direction ``_drill`` (dy,dx); the team
        # pierces FORWARD along it with concentrated speed, regardless of gradient.
        drill = getattr(self, "_drill", None)
        drill_fwd = None
        if drill is not None:
            dd = drill.gather(1, self.fteam.unsqueeze(-1).expand(B, N, 2))     # (B,N,2) per-fighter thrust dir
            drill_fwd = dd[..., 0:1] * self._dy_t + dd[..., 1:2] * self._dx_t   # (B,N,8) forward alignment
            on = (dd.abs().sum(-1, keepdim=True) > 0)                           # (B,N,1) is this team drilling?
            movable = movable | (on & (drill_fwd > 0) & (ng <= cur.unsqueeze(-1) + 44))
        # WALL stance: a per-team facing ``_wall`` (dy,dx); the team collapses onto
        # the line through the cursor PERPENDICULAR to the facing -> a dense shield
        # bar pointed at the threat. (The score term needs the radial, added below.)
        wall = getattr(self, "_wall", None)
        wdd = None
        if wall is not None:
            wdd = wall.gather(1, self.fteam.unsqueeze(-1).expand(B, N, 2))      # (B,N,2) facing
            won = (wdd.abs().sum(-1, keepdim=True) > 0)
            movable = movable | (won & (ng <= cur.unsqueeze(-1) + 30))
        # TIDAL WAVE (Pulse tide mode): the standing-wave machinery made
        # DIRECTIONAL — the phase travels along ``_tide``'s (dy,dx) heading, so
        # the army advances in rolling crests (a marching front), not radial
        # rings. Per-team (B,T,2); zero = off. (Score term added below with the
        # other wave biases.)
        tide = getattr(self, "_tide", None)
        tdd = None
        if tide is not None:
            tdd = tide.gather(1, self.fteam.unsqueeze(-1).expand(B, N, 2))      # (B,N,2) wave heading
            t_on = (tdd.abs().sum(-1, keepdim=True) > 0)
            movable = movable | (t_on & (ng <= cur.unsqueeze(-1) + 30))
        jitter = torch.randint(0, 7, ng.shape, device=self.device)
        # MOMENTUM / INERTIA: bias each candidate's score toward the fighter's
        # velocity, so a moving mass carries weight — it overshoots, banks around
        # corners, and head-on clashes become collisions (momentum vs momentum) —
        # instead of snapping to the gradient every tick. The bias (~±VEL_W) is
        # comparable to one octile step (10/14), so it can pull a fighter slightly
        # off the steepest line in its heading, but the gradient still dominates.
        if not hasattr(self, "fvy") or self.fvy.shape != (B, N):
            self.fvy = torch.zeros(B, N, device=self.device)
            self.fvx = torch.zeros(B, N, device=self.device)
        if not hasattr(self, "_fdir") or self._fdir.shape != (B, N):    # lazy, like fvy (pre-reset safety)
            self._fdir = torch.full((B, N), 8, dtype=torch.long, device=self.device)
        VEL_W = 8.0
        align = self.fvy.unsqueeze(-1) * self._dy_t + self.fvx.unsqueeze(-1) * self._dx_t  # (B,N,8)
        # DISCRETE HEADING INERTIA: on top of the smooth velocity carry above,
        # each fighter remembers the direction it ACTUALLY took last sub-step
        # (``_fdir``; 8 = stood still) and its candidate scores are biased
        # toward continuing it — same heading +6*mom, 45° turn +3*mom, 90°
        # neutral, reversal -4*mom (``_mom_tab``, see __init__). At the
        # default ``_inertia`` 0.35 the max bonus (~2.1) sits well under one
        # octile step (10/14) and under every stance-shaping weight (14-26),
        # so an attacking swarm carries weight through turns — it arcs instead
        # of snapping to the new gradient — while Doom rings / walls / drills
        # keep their forms and the gradient still decides where the mass ends
        # up. Movement CHOICE only: ``movable`` (where a fighter MAY step) and
        # combat/conversion are untouched. ``_inertia = 0`` switches it off.
        mom = getattr(self, "_inertia", 0.35)
        mom_bias = self._mom_tab[self._fdir] * mom if mom else None     # (B,N,8) single gather, graph-safe
        # SWIRL: a tangential bias so units spiral INTO the cursor along curved,
        # magnetized field-lines instead of straight radial columns. The inward
        # gradient (~10-14/step) still dominates SWIRL_W, so they converge — just
        # on a curve, not a beeline. (ry/rx/rn computed with the wave phase above.)
        SWIRL_W = 8.0
        # Per-fighter, re-randomised swirl-strength jitter breaks the COHERENT
        # spiral arms — which on an 8-direction grid read as angular spokes (a
        # pinwheel) — into a smooth churn; the average orbit is unchanged.
        swirl_jit = 0.3 + 1.4 * torch.rand(B, N, 1, device=self.device)
        # Spin sign per team: the player can flip the swarm's orbit CW/CCW (or 0 to
        # stop it). Default +1 when unset. ``_spin`` is (B, T).
        spin = getattr(self, "_spin", None)
        spin_f = spin.gather(1, self.fteam).unsqueeze(-1) if spin is not None else 1.0
        swirl = spin_f * SWIRL_W * swirl_jit * ((-rx / rn).unsqueeze(-1) * self._dy_t + (ry / rn).unsqueeze(-1) * self._dx_t)
        # ATOM (figure-8 / binary star): ``_fig8`` is a per-team MODE — 1 puts
        # the lobes on the FIXED x-axis with counter-rotating halves (the mass
        # loops in two lobes crossing at the center = a lemniscate); 2 puts
        # them on a slowly ROTATING axis, co-rotating with extra separation and
        # lobe cohesion -> two distinct discs orbiting their barycenter (the
        # cursor), like a close binary star.
        fig8 = getattr(self, "_fig8", None)
        if fig8 is not None:
            f8 = fig8.gather(1, self.fteam)                            # (B,N) 0 off / 1 lemniscate / 2 binary
            on8 = f8 > 0
            is_bin = (f8 > 1.5).float()
            bphi = self._tick_f * 0.035                                # binary axis angle (slow orbit)
            axy = is_bin * torch.sin(bphi)                             # lobe axis: rotating for binary,
            axx = is_bin * torch.cos(bphi) + (1.0 - is_bin)            #   the x-axis for the lemniscate
            R8 = 8.0 + 6.0 * is_bin                                    # binary lobes sit wider apart
            s = torch.sign(-(ry * axy + rx * axx))                     # which side of the axis plane
            lry = ry + s * R8 * axy                                    # radial to the per-side lobe centre
            lrx = rx + s * R8 * axx
            lrn = (lrx * lrx + lry * lry).sqrt().clamp(min=1.0)
            tang = (-lrx / lrn).unsqueeze(-1) * self._dy_t + (lry / lrn).unsqueeze(-1) * self._dx_t
            sense = is_bin + (1.0 - is_bin) * s                        # binary co-rotates; lemniscate counter-rotates
            # binary lobe COHESION folded into the swirl term (swirl enters the
            # score negated): pull toward the assigned lobe centre so the two
            # discs stay distinct instead of merging across the barycenter
            l_align = (lry / lrn).unsqueeze(-1) * self._dy_t + (lrx / lrn).unsqueeze(-1) * self._dx_t
            swirl8 = (spin_f * SWIRL_W * swirl_jit * sense.unsqueeze(-1) * tang
                      + 9.0 * is_bin.unsqueeze(-1) * l_align)
            swirl = torch.where(on8.unsqueeze(-1), swirl8, swirl)
        # PERISTALTIC EDGE PUSH: a wave-modulated OUTWARD bias (rides the same
        # traveling wave as the restless gate). On a crest the rim extends outward
        # (a pseudopod bulge, up to unit_speed cells); off-crest the inward
        # gradient retracts it -> the silhouette undulates like a living membrane.
        # Interior fighters can't extend (neighbours occupied), so only the edge
        # ripples; the body stays dense.
        PUSH_W = 14.0
        push = (PUSH_W * torch.sin(phase)).unsqueeze(-1)               # (B,N,1), oscillates ±
        if ring_f is not None:
            # the rim ripple (±14) is comparable to the ring bias (15/26) and
            # segments the accretion disk into ripple bands — damp it hard for
            # teams holding a ring formation so the disk stays a solid annulus
            push = push * (1.0 - 0.8 * (ring_f > 0).float()).unsqueeze(-1)
        out_align = (-ry / rn).unsqueeze(-1) * self._dy_t + (-rx / rn).unsqueeze(-1) * self._dx_t
        score = ng.float() + jitter.float() - VEL_W * align - swirl - push * out_align
        if mom_bias is not None:
            score = score - mom_bias                                   # heading inertia (movement choice only)
        if burst_f is not None:                                       # gather (-1) inward / burst (+1) outward
            score = score - 15.0 * burst_f.unsqueeze(-1) * out_align
        if ring_f is not None:                                        # accretion ring: settle on the orbit radius
            # Ring shape is a per-team dial ``_ring_ecc`` (0..1, default 1):
            # at 1, an OBLATE (Gargantua) disk — the target radius is
            # angle-dependent, pinched vertically (0.78x) and stretched along
            # the equator (1.28x) — so Doom's spinning disk reads as the
            # edge-on silhouette. At 0, a CIRCULAR annulus (Maelstrom's
            # whirlpool), so the two stances don't share a silhouette.
            ecc_f = (self._ring_ecc.gather(1, self.fteam)
                     if hasattr(self, "_ring_ecc") else torch.ones_like(ring_f))
            ell = ring_f * (1.0 - 0.22 * ecc_f + 0.5 * ecc_f * (rx / rn) ** 2)
            rbias = ((ell - rn) / 4.0).clamp(-1.0, 1.0) * (ring_f > 0).float()
            # Outward (inside the ring) needs MORE weight than inward: it fights
            # the gradient's 10-14/step pull toward the cursor, else stragglers
            # pool in the centre and the hole never opens.
            rw = torch.where(rbias > 0, 26.0, 15.0)
            score = score - (rw * rbias).unsqueeze(-1) * out_align
            # GARGANTUA BLADE: fighters well OUTSIDE the disk also flatten toward
            # the cursor's equator row and stream in along it — so the formation
            # reads as the edge-on accretion blade feeding an orbiting halo (the
            # Interstellar silhouette), not a plain donut. Scaled by ``ecc_f``:
            # a circular ring (Maelstrom) wants stragglers spiraling in on the
            # swirl, not streaming along an equator it doesn't have.
            far = ((rn > ell * 1.25) & (ring_f > 0)).float() * ecc_f
            eq_align = torch.sign(ry).unsqueeze(-1) * self._dy_t      # toward the equator line
            score = score - 15.0 * far.unsqueeze(-1) * eq_align
        # CHLADNI RESONANCE (Pulse's cymatic modes): standing-wave nodal patterns,
        # like sand on a vibrating plate. ``_node_l`` sets a radial wavelength —
        # fighters drift to the nodal RADII (concentric standing rings, slowly
        # breathing). ``_node_m`` adds an angular mode — m nodal diameters, so the
        # mass gathers into an m-petal star. Both are per-team (B,T), 0 = off.
        # (no .any() guards here: at B=1 a guard's GPU->CPU sync costs more than
        # the handful of 16k-element kernels it would skip — run unconditionally,
        # the *_on masks zero the bias when the knob is off)
        nodel = getattr(self, "_node_l", None)
        if nodel is not None:
            l_f = nodel.gather(1, self.fteam)                          # (B,N) wavelength or 0
            l_on = l_f > 0
            movable = movable | (l_on.unsqueeze(-1) & (ng <= cur.unsqueeze(-1) + 24))
            # ``_node_v``: per-team ring BREATHE speed (rad/tick; was a fixed
            # 0.02 — the play server dials it up for a more energetic Pulse)
            v_f = (self._node_v.gather(1, self.fteam)
                   if hasattr(self, "_node_v") else 0.02)
            rad = torch.sin(rn * 6.2832 / l_f.clamp(min=1.0) - self._tick_f * v_f)
            # outward (rad>0) must out-weigh the gradient's 10-14/step inward
            # pull or the standing rings never separate (same as _ring above)
            rw_n = torch.where(rad > 0, 26.0, 15.0) * l_on.float()
            score = score - (rw_n * rad).unsqueeze(-1) * out_align
        nodem = getattr(self, "_node_m", None)
        if nodem is not None:
            m_f = nodem.gather(1, self.fteam)                          # (B,N) petal count or 0
            m_on = m_f > 0
            movable = movable | (m_on.unsqueeze(-1) & (ng <= cur.unsqueeze(-1) + 24))
            theta = torch.atan2(-ry, -rx)                              # fighter angle about the cursor
            tang_n = (-rx / rn).unsqueeze(-1) * self._dy_t + (ry / rn).unsqueeze(-1) * self._dx_t
            # Generalized angular mode — m nodal diameters, optionally SPIRALED
            # by a radial pitch (``_node_k``: galaxy arms wind with radius) and
            # ROTATED over time (``_node_w``: the pattern sweeps — sawblade
            # teeth). Static Chladni star = k=0, w≈0. Weight 16 so the pattern
            # survives the swirl churn of a fast-spinning team.
            k_f = self._node_k.gather(1, self.fteam) if hasattr(self, "_node_k") else 0.0
            w_f = self._node_w.gather(1, self.fteam) if hasattr(self, "_node_w") else 0.0
            ang = torch.sin(m_f * theta + k_f * rn + w_f * self._tick_f)
            score = score - 16.0 * (m_on.float() * ang).unsqueeze(-1) * tang_n
        if tdd is not None:                                           # TIDAL WAVE: rolling directional crests
            t_along = self.fy.float() * tdd[..., 0] + self.fx.float() * tdd[..., 1]
            t_ph = torch.sin(t_along * 0.30 - self._tick_f * 0.22)    # crest sweeps ALONG the heading
            t_align = tdd[..., 0:1] * self._dy_t + tdd[..., 1:2] * self._dx_t
            t_on_f = (tdd.abs().sum(-1) > 0).float()
            # a crest must out-weigh the gradient's 10-14/step (like the ring /
            # node terms) to push the mass forward; the trough lets the
            # gradient regroup it -> the army marches in waves
            score = score - 22.0 * (t_on_f * t_ph).unsqueeze(-1) * t_align
        if drill_fwd is not None:                                     # DRILL: pierce forward, concentrated
            # |_drill| encodes ADVANCE SPEED (drill mode): a small magnitude weakens
            # the forward bias so the spin/squeeze dominate -> the fast (high-spin)
            # mode bores slowly; a unit magnitude advances fast.
            score = score - 16.0 * drill_fwd
            # Lateral squeeze onto the thrust axis -> a NARROW piercing column. Uses
            # the UNIT thrust dir so squeeze strength is independent of advance speed.
            ndd = dd / dd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            perp_dot = self._dy_t * ndd[..., 1:2] - self._dx_t * ndd[..., 0:1]    # (B,N,8) candidate · perp
            lateral = (-ry * ndd[..., 1] + rx * ndd[..., 0])                      # (B,N) offset · perp
            # ENDER'S-GAME HELIX: the squeeze targets a traveling-SINE centreline
            # (phase advances along the thrust axis and with time) instead of a
            # straight line — the 2D projection of a rotating drill bit. The
            # column visibly corkscrews as it advances; twist direction follows
            # the team's spin sign (Q/E).
            along = (-ry) * ndd[..., 0] + (-rx) * ndd[..., 1]                     # (B,N) offset · thrust
            tw = torch.sign(spin_f.squeeze(-1)) if spin is not None else 1.0
            # omega 0.12/tick (~1s per twist at 60Hz): fighters can only step ~1
            # cell/tick laterally, so a faster wave sweeps by before the column
            # can track it and the helix never materializes.
            helix = 4.5 * torch.sin(along * 0.35 - tw * self._tick_f * 0.12)
            score = score + 16.0 * torch.sign(lateral - helix).unsqueeze(-1) * perp_dot
        if wdd is not None:                                           # WALL: collapse onto the cursor's perp line
            fwd_comp = (-ry) * wdd[..., 0] + (-rx) * wdd[..., 1]      # how far ahead/behind that line
            c_face = wdd[..., 0:1] * self._dy_t + wdd[..., 1:2] * self._dx_t
            # 20 (was 14): the collapse must dominate the wave/swirl noise so the
            # bar packs DENSE — with the server's stronger inward burst it reads
            # as a solid column, not a loose picket line.
            score = score + 20.0 * torch.sign(fwd_comp).unsqueeze(-1) * c_face
        # WELL COUNTERPLAY: a team's OWN active Maelstrom is a COUNTER-CURRENT —
        # whirlpool vs black hole: inside its radius your fighters are shielded
        # from enemy well forces (up to 75% at the eye, squared falloff like the
        # offensive current). A held Wall BRACES: dug-in fighters resist 45% of
        # the drag. Both feed ``well_shield`` (B,N), multiplied into every
        # enemy-well force below — so Maelstrom is the answer to Doom, and the
        # policy can learn the matchup (same physics in training).
        well_shield = None
        if getattr(self, "_wells_enabled", False):
            o_pos = self._vortex_pos.gather(1, self.fteam.unsqueeze(-1).expand(B, N, 2))
            o_str = self._vortex_str.gather(1, self.fteam)
            o_R = self._vortex_range.gather(1, self.fteam).clamp(min=1.0)
            oy = o_pos[..., 0] - self.fy
            ox = o_pos[..., 1] - self.fx
            on2 = oy * oy + ox * ox
            # PLATEAU, not a slope: full 85% protection across the WHOLE
            # storm (out to 1.2R, fading to zero by 1.5R). The two slope
            # attempts both failed the same way — at close range Doom's
            # falloff nears 1 while any mid-shell shield gap leaves the
            # rim losing pull-vs-gradient, and stripping runs away as the
            # victim drifts off the shield. Inside the plateau a 3x Doom's
            # point-blank pull (~96) drops to ~14 ~= the gradient, and the
            # reel finishes the argument.
            o_d = on2.sqrt()
            o_fall = (((1.5 * o_R - o_d) / (0.3 * o_R)).clamp(0.0, 1.0)
                      * (o_str > 0).float())
            brace = (wdd.abs().sum(-1) > 0).float() if wdd is not None else torch.zeros(B, N, device=self.device)
            well_shield = (1.0 - 0.85 * o_fall) * (1.0 - 0.45 * brace)
        # BLACK HOLE (Doom): a cross-team gravity well at a team's cursor that drags
        # the OTHER teams' fighters in (overriding their own gradient) so they get pulled
        # into the singularity and devastated. One SLOT per team (play server writes
        # them in-place; str==0 = off = no-op force, fixed kernel sequence).
        if getattr(self, "_wells_enabled", False):
            for t in range(self.T):
                bh_str = self._doom_str[:, t:t + 1]                    # (B,1) broadcasts over N
                bh_R = self._doom_range[:, t:t + 1]                    # FINITE reach -> distant forces escape
                bhy = self._doom_pos[:, t, 0:1] - self.fy              # (B,N) toward the well
                bhx = self._doom_pos[:, t, 1:2] - self.fx
                bhn = (bhy * bhy + bhx * bhx).sqrt().clamp(min=1.0)
                falloff = (bh_R * bh_R) / (bhn * bhn + bh_R * bh_R)    # 1 at the well, ->0 far (no map-wide vacuum)
                is_en = (self.fteam != t)                              # (B,N) not the well's own team
                in_range = (bhn < bh_R * 2.5) & (bh_str > 0)           # only nearby enemies get caught/dragged
                pull = (bh_str * falloff * is_en.float() * well_shield).unsqueeze(-1)  # (B,N,1) shield-damped
                bh_align = (bhy / bhn).unsqueeze(-1) * self._dy_t + (bhx / bhn).unsqueeze(-1) * self._dx_t
                score = score - pull * bh_align                        # near enemies sucked in; distant ones free
                # the well can only PRY fighters off-gradient (movable relax)
                # where their shield is weak — inside a defender's storm
                # plateau Doom keeps the score bias but loses permission to
                # move them, which is what stripping-on-approach actually was
                movable = movable | ((is_en & in_range & (well_shield > 0.5)).unsqueeze(-1)
                                     & (ng <= cur.unsqueeze(-1) + 40))
        # WHIRLPOOL (Maelstrom): a cross-team CURRENT at one team's cursor — the
        # rotational counterpart of Doom's gravity well. Where Doom drags enemies
        # RADIALLY into a singularity and captures them, the maelstrom is
        # vorticity: nearby enemy fighters are swept TANGENTIALLY off their own
        # gradient and entrained into orbit around the well, plus a radial
        # component ``_vortex_rad`` (>0 undertow: spiral them inward / <0 ejecta:
        # fling them outward / 0 pure shear). No capture — entrained enemies
        # circle through the owner's spinning rim and are ground down by ordinary
        # adjacency combat. Disruption and area-denial, not a devour. One SLOT per
        # team, same scheme as the Doom slots above.
        if getattr(self, "_wells_enabled", False):
            for t in range(self.T):
                wp_str = self._vortex_str[:, t:t + 1]                  # (B,1) broadcasts over N
                wp_R = self._vortex_range[:, t:t + 1]                  # FINITE reach, like Doom's
                wp_sgn = self._vortex_sign[:, t:t + 1]                 # current direction (owner's Q/E)
                wp_rad = self._vortex_rad[:, t:t + 1]
                wy = self._vortex_pos[:, t, 0:1] - self.fy             # (B,N) toward the well
                wx = self._vortex_pos[:, t, 1:2] - self.fx
                wn = (wy * wy + wx * wx).sqrt().clamp(min=1.0)
                wfall = (wp_R * wp_R) / (wn * wn + wp_R * wp_R)        # 1 at the well, ->0 far
                # SQUARED falloff (Doom keeps the flat one): the current is a
                # LOCAL hazard — at d=R it's 25% strength, not 50% — so it bends
                # nearby flow instead of vacuuming units off the enemy blob from
                # across the arena (which read as an unfightable tractor beam)
                wfall = wfall * wfall
                w_en = (self.fteam != t)                               # only the OTHER teams feel the current
                w_in = (wn < wp_R * 2.5) & (wp_str > 0)
                drag = (wp_str * wfall * w_en.float() * well_shield).unsqueeze(-1)  # (B,N,1) shield-damped
                # tangential alignment matches the own-team swirl convention, so
                # +1 entrains enemies in the same sense as the owner's spin
                w_tan = wp_sgn.unsqueeze(-1) * ((-wx / wn).unsqueeze(-1) * self._dy_t + (wy / wn).unsqueeze(-1) * self._dx_t)
                w_radial = (wy / wn).unsqueeze(-1) * self._dy_t + (wx / wn).unsqueeze(-1) * self._dx_t
                score = score - drag * (w_tan + wp_rad.unsqueeze(-1) * w_radial)  # bend their flow; their gradient still fights
                movable = movable | ((w_en & w_in & (well_shield > 0.5)).unsqueeze(-1)
                                     & (ng <= cur.unsqueeze(-1) + 40))
                # THE REEL: the storm recovers its OWN strays — fighters of the
                # wielding team beyond ~0.8R feel a ramping pull back into the
                # circulation, so an enemy Doom must out-muscle your current for
                # every fighter it tries to strip off the shell.
                w_own = (self.fteam == t).float() * (wp_str > 0).float()
                # full reel strength AT the rim (ramp 0.6R -> 1.0R), not a
                # gentle slope that only matters two radii out
                reel = (16.0 * w_own * ((wn - 0.6 * wp_R) / (0.4 * wp_R)).clamp(0.0, 1.0)).unsqueeze(-1)
                score = score - reel * w_radial
                movable = movable | (((self.fteam == t) & (wp_str > 0) & (wn > 0.8 * wp_R)).unsqueeze(-1)
                                     & (ng <= cur.unsqueeze(-1) + 30))
        order = torch.where(movable, score, score.new_full((), float(BIGG))).argsort(dim=-1)
        ncell_s = ncell.gather(-1, order)                              # cells, best-first
        down_s = movable.gather(-1, order)
        slots = torch.arange(N, device=self.device).expand(B, N)
        BIGN = N + 1
        moved = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        attacking = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        front = torch.zeros(B, N, dtype=torch.long, device=self.device)
        taken = torch.full((B, N), 8, dtype=torch.long, device=self.device)  # ACTUAL heading this sub-step (8 = none)
        # (no early-break/.any() guards in this loop: each is a GPU->CPU sync,
        # and there are rounds x unit_speed sub-steps of them per tick — at
        # B=1 the syncs cost far more than letting an empty round's no-op
        # kernels run. Same lesson as the stance-bias guards above.)
        # 4 rounds, not all 8 candidate ranks: moves overwhelmingly resolve in
        # rounds 0-2, and a fighter whose top-4 are all blocked simply waits —
        # it retries next sub-step (unit_speed of them per tick), so the rare
        # deep reroute costs 1/unit_speed of a tick, while every round costs
        # ~10 kernels x unit_speed per tick for everyone.
        for k in range(4):                                             # priority rounds
            active = down_s[:, :, k] & ~moved & ~attacking
            kcell = ncell_s[:, :, k]
            occ_slot = self.occ.view(B, -1).gather(1, kcell)
            occ_team = torch.where(occ_slot >= 0,
                                   self.fteam.gather(1, occ_slot.clamp(min=0)), self.fteam)
            passable = self.passable.view(B, -1).gather(1, kcell)
            is_empty = (occ_slot == -1) & passable
            is_enemy = (occ_slot >= 0) & (occ_team != self.fteam)
            # MOVE into the best free cell (lowest-slot wins contention).
            elig = active & is_empty
            claim = torch.full((B, H * W), BIGN, dtype=torch.long, device=self.device)
            claim.scatter_reduce_(1, kcell,
                                  torch.where(elig, slots, slots.new_full((), BIGN)),
                                  reduce='amin', include_self=True)
            kmoves = elig & (claim.gather(1, kcell) == slots)
            self.fy = torch.where(kmoves, kcell // W, self.fy)
            self.fx = torch.where(kmoves, kcell % W, self.fx)
            moved = moved | kmoves
            taken = torch.where(kmoves, order[:, :, k], taken)         # record the heading actually taken
            # ENEMY at this (best available) down-gradient cell -> ATTACK here,
            # do NOT reroute past it. A teammate, by contrast, leaves the fighter
            # active so it tries the next candidate (reroute). This is what makes
            # contact compound into a takeover instead of sliding by.
            atk = active & is_enemy & ~kmoves
            front = torch.where(atk & ~attacking, order[:, :, k], front)
            attacking = attacking | atk
            self._rebuild_occ()                                        # followers see freed cells
        # COORDINATED ROTATION: the priority loop above only resolves chains that
        # terminate in an EMPTY cell — a rim fighter steps into open space, freeing
        # its cell for the follower behind it, one ring inward per tick. A dense,
        # swirling CORE has no such empty seed: every fighter's best (tangential)
        # cell holds a same-team neighbour, and that neighbour is blocked the same
        # way, so the whole interior deadlocks and only the rim ripples. This pass
        # closes the gap — it lets a fighter FOLLOW a same-team occupant that is
        # itself vacating this tick, which (for a ring of such followers) resolves
        # as a simultaneous rotation cycle. Conservation + one-per-cell are kept:
        # see :meth:`_resolve_rotation`.
        rmoved = self._resolve_rotation(ncell_s, down_s, order, moved, attacking)
        moved = moved | rmoved
        # Rotation movers stepped to their FIRST movable rank — the same rule
        # _resolve_rotation used to pick its target cell — so that rank's
        # original direction index is the heading they actually took.
        rot_dir = order.gather(-1, torch.argmax(down_s.to(torch.int8),
                                                dim=-1, keepdim=True)).squeeze(-1)
        taken = torch.where(rmoved, rot_dir, taken)
        # combat: attackers push into the down-gradient enemy they committed to.
        self._blocked = attacking
        self._front_dy = torch.where(attacking, self._dy_t[front], front.new_zeros(()))
        self._front_dx = torch.where(attacking, self._dx_t[front], front.new_zeros(()))
        # every fighter's facing = its steepest direction toward its own cursor;
        # the combat phase reads the DEFENDER's facing to tell a back-attack
        # (defender facing away) from a defended head-on clash.
        self._facing = order[:, :, 0]
        # Persist the heading ACTUALLY taken this sub-step (blocked fighters
        # and committed attackers read as stationary — they took no step).
        # ``copy_``, never rebind: graph I/O buffer at a fixed address. A
        # fighter CONVERTED later this tick keeps its pre-conversion heading —
        # harmless: it re-chooses next sub-step under its new team's gradient
        # and the stale heading is at most a one-substep ~2.1-point nudge.
        self._fdir.copy_(taken)
        # Carry velocity toward the chosen heading with high inertia (MOM); a
        # settled fighter (nothing movable) coasts to rest. This is what gives the
        # mass weight — momentum persists ~1/(1-MOM) ticks after the gradient shifts.
        MOM = 0.88
        best = order[:, :, 0]
        any_mov = movable.any(-1)
        self.fvy = MOM * self.fvy + (1 - MOM) * torch.where(any_mov, self._dy_t[best].float(), self.fvy.new_zeros(()))
        self.fvx = MOM * self.fvx + (1 - MOM) * torch.where(any_mov, self._dx_t[best].float(), self.fvx.new_zeros(()))
        # WALL SLOSH: deflect velocity off adjacent walls — strip the component
        # heading INTO a wall (keep the tangential) and bleed a little energy, so a
        # mass hitting a barrier slides/sloshes ALONG it (fluid) instead of stopping
        # dead. ``ncell`` holds each fighter's 8 (clamped) neighbour cells.
        wall_n = self.walls.view(B, -1).gather(1, ncell.reshape(B, -1)).reshape(B, N, 8).float()
        wny = (wall_n * self._dy_t).sum(-1)                           # net direction toward walls
        wnx = (wall_n * self._dx_t).sum(-1)
        wmag = (wny * wny + wnx * wnx).sqrt()
        hit = wmag > 0
        inv = 1.0 / wmag.clamp(min=1e-6)
        wyh, wxh = wny * inv, wnx * inv                               # unit into-wall normal
        into = (self.fvy * wyh + self.fvx * wxh).clamp(min=0)         # velocity heading INTO the wall
        self.fvy = torch.where(hit, (self.fvy - into * wyh) * 0.88, self.fvy)
        self.fvx = torch.where(hit, (self.fvx - into * wxh) * 0.88, self.fvx)

    def _resolve_rotation(self, ncell_s: torch.Tensor, down_s: torch.Tensor,
                          order: torch.Tensor, moved: torch.Tensor,
                          attacking: torch.Tensor) -> torch.Tensor:
        """Let a dense core ROTATE by following same-team occupants that vacate.

        The priority loop in :meth:`_move_fighters` only advances a fighter into an
        **empty** cell, so it resolves movement chains that terminate in open space
        but never a closed rotation cycle: a ring of same-team fighters each wanting
        the next cell, with no empty seed, deadlocks (only the rim, which borders
        empty space, ever moves). This pass adds the missing move — a fighter may
        follow a same-team occupant **that is itself vacating this tick** — so such a
        ring resolves as a single simultaneous rotation.

        It is a self-consistent simultaneous permutation, found by batched iterated
        relaxation (no python loop over fighters):

        - Each still-active fighter proposes its single best movable candidate cell
          ``tgt`` (the rim already took empty cells, so for the core this is a
          same-team-occupied cell). Enemy / wall / non-movable candidates never
          propose — combat and the gradient flow are untouched.
        - Same-cell contention is broken by lowest slot index (the loop's rule),
          guaranteeing **one mover per target cell** → one-per-cell preserved.
        - A proposer is *cleared* iff its target is empty **or** occupied by a
          same-team fighter that is itself a cleared mover (it will vacate). Starting
          from "all contention winners cleared", any proposer whose target's occupant
          is a fighter that stays put has its clearance **revoked**; revocation
          propagates until a fixpoint. What survives = chains into empty cells +
          closed rotation cycles. Each cleared mover both claims exactly one target
          and frees exactly its own (distinct) source cell, so total count is
          invariant and no cell ends double-occupied.

        :param ncell_s: ``(B, N, 8)`` candidate cells, best-first (post-argsort).
        :param down_s: ``(B, N, 8)`` movable mask aligned with ``ncell_s``.
        :param order: ``(B, N, 8)`` argsort mapping rank -> original slot.
        :param moved: ``(B, N)`` bool — fighters that already moved this tick.
        :param attacking: ``(B, N)`` bool — fighters committed to an attack.
        :returns: ``(B, N)`` bool mask of fighters relocated by this rotation pass.

        .. note::
           Invariant-preserving by construction; see the relaxation argument above.
           Pairs with the priority loop — together they cover empty-seeded chains
           (there) and closed cycles (here).
        """
        B, N, H, W = self.B, self.N, self.H, self.W
        device = self.device
        # Per-fighter best MOVABLE candidate (rank order already encodes the swirl
        # bias). ``down_s`` is False past a fighter's real candidates, so masked
        # fighters get no proposal.
        any_cand = down_s.any(-1)
        first_rank = torch.argmax(down_s.to(torch.int8), dim=-1)            # (B,N) first True rank
        br = first_rank.unsqueeze(-1)
        tgt = ncell_s.gather(-1, br).squeeze(-1)                            # (B,N) best movable cell
        # Eligible proposers: still active (not moved, not attacking) with a real
        # movable candidate, whose best cell is PASSABLE and NOT an enemy. (Empty or
        # same-team occupied — both are valid rotation targets; enemies go to combat
        # via the priority loop, so we must not steal them here.)
        active = any_cand & ~moved & ~attacking
        passable = self.passable.view(B, -1).gather(1, tgt)
        occ_slot = self.occ.view(B, -1).gather(1, tgt)
        occ_team = torch.where(occ_slot >= 0,
                               self.fteam.gather(1, occ_slot.clamp(min=0)), self.fteam)
        is_enemy = (occ_slot >= 0) & (occ_team != self.fteam)
        propose = active & passable & ~is_enemy
        # (no propose.any() early-out: the GPU->CPU sync costs more per
        # sub-step than running the no-op kernels below on an empty mask)
        slots = torch.arange(N, device=device).expand(B, N)
        BIGN = N + 1
        # Contention: at most one proposer (lowest slot) per target cell.
        claim = torch.full((B, H * W), BIGN, dtype=torch.long, device=device)
        claim.scatter_reduce_(1, tgt,
                              torch.where(propose, slots, slots.new_full((), BIGN)),
                              reduce='amin', include_self=True)
        winner = propose & (claim.gather(1, tgt) == slots)                 # (B,N) one per target
        tgt_empty = (occ_slot == -1)
        src = (self.fy * W + self.fx)                                      # (B,N) each fighter's own cell
        # ``cleared`` starts optimistic (every contention winner) and is MONOTONICALLY
        # SHRUNK to a fixpoint: a winner survives only if its target is empty OR its
        # target's occupant is itself a surviving cleared mover (it will vacate). We
        # recompute from ``winner`` against a ``movers_grid`` built from the current
        # ``cleared`` — which only shrinks — so the iteration is monotone and reaches
        # a self-consistent fixpoint. ``movers_grid[b, c]`` is True iff a cleared
        # mover currently sits on cell ``c`` (one-per-cell ⇒ exactly its occupant).
        # The cap bounds the longest revocation chain we propagate; the FINAL
        # consistency filter below guarantees no unjustified move is applied even if
        # an (atypically long) chain hasn't fully converged.
        cleared = winner.clone()

        def _vacated(active_set: torch.Tensor) -> torch.Tensor:
            """``(B,N)`` mask: winner whose target is empty or vacated by a mover."""
            grid = torch.zeros(B, H * W, dtype=torch.bool, device=device)
            grid.scatter_(1, src, active_set)
            return winner & (tgt_empty | (grid.gather(1, tgt) & (occ_slot >= 0)))

        # A closed rotation cycle is self-consistent immediately (one pass); chains
        # into empty cells converge in chain-length passes. A small cap keeps this
        # cheap (60fps budget). Only a TRUE fixpoint (``_vacated(cleared) ==
        # cleared``) is self-consistent — every survivor justified by a survivor —
        # and therefore safe to apply. If the cap is hit without convergence (an
        # atypically long chain), we fall back to the unconditionally-safe subset
        # (pure empty-target moves); the rest resolves over the next ticks. This
        # makes the one-per-cell invariant impossible to violate.
        # (Fixed-count shrink + a BRANCHLESS per-batch fixpoint select — the old
        # per-pass converged check was a GPU->CPU sync per iteration per
        # sub-step, which cost more than the extra no-op passes it skipped.)
        for _ in range(6):                                                # monotone shrink
            cleared = _vacated(cleared)
        at_fixpoint = (_vacated(cleared) == cleared).all(dim=1, keepdim=True)  # (B,1), stays on GPU
        cleared = torch.where(at_fixpoint, cleared, winner & tgt_empty)   # safe fallback per batch row
        # Apply the simultaneous permutation. Each cleared mover relocates to ``tgt``;
        # its source cell is freed by whichever cleared mover follows it (or stays
        # empty). One-per-cell holds: distinct winners hold distinct targets.
        self.fy = torch.where(cleared, tgt // W, self.fy)
        self.fx = torch.where(cleared, tgt % W, self.fx)
        self._rebuild_occ()
        return cleared

    # ------------------------------------------------------------------
    # Combat — convert, never delete
    # ------------------------------------------------------------------

    def _resolve_combat(self) -> None:
        """Blocked fighters front-attack the enemy in their intended direction;
        accumulated damage is applied to targets; targets at <0 health rebase and
        defect to an attacker's team. Total fighter count is invariant."""
        B, N, H, W = self.B, self.N, self.H, self.W
        # (no .any() guards anywhere in here: each is a GPU->CPU sync per
        # sub-step; at B=1 they cost more than the no-op kernels they skip)
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
        # Accumulate DIRECTIONAL damage onto target slots. A defender hit while
        # facing AWAY (its back to the attacker — the attacker pushes the same
        # way the defender is moving) barely resists -> full ATTACK -> converts
        # fast, so the attacker overtakes and SPREADS through the line. A head-on
        # or side clash -> defender resists -> SIDE_ATTACK (a slow grind).
        dmg = torch.zeros(B, N, dtype=torch.int32, device=self.device)
        a_tgt = torch.where(attack, tgt_slot, torch.zeros_like(tgt_slot))
        def_facing = self._facing.gather(1, tgt_slot.clamp(min=0))
        align = (self._front_dy * self._dy_t[def_facing]
                 + self._front_dx * self._dx_t[def_facing])             # dot(attack dir, defender facing)
        hit = torch.where(align > 0, ATTACK, SIDE_ATTACK).float()
        # Pulse / surge: a per-team damage multiplier (default absent = 1x). The
        # play server sets ``_surge`` for the human team during a Pulse so the
        # army's contact briefly overwhelms — a peristaltic burst.
        surge = getattr(self, "_surge", None)
        if surge is not None:
            hit = hit * surge.gather(1, self.fteam)
        # MOMENTUM PIERCE (the inertia rule): a fast-moving attacker hits harder, so
        # a charging mass — and especially the Drill, which builds forward speed —
        # punches through, while a standing blob hits soft. Kept well under the 16x
        # back-attack bonus so direction stays the bigger prize.
        att_speed = (self.fvy ** 2 + self.fvx ** 2).sqrt()           # (B,N) attacker speed ~[0,1.4]
        hit = (hit * (1.0 + 1.5 * att_speed)).to(torch.int32)
        dmg.scatter_add_(1, a_tgt, torch.where(attack, hit,
                                               torch.zeros_like(tgt_slot, dtype=torch.int32)))
        self.fhealth = self.fhealth - dmg
        # Conversion: any slot now <0 defects to a (lowest-slot-index) attacker's team.
        neg = self.fhealth < 0
        # which team converts this target: pick the lowest-index attacker on it.
        BIG = N + 1
        owner = torch.full((B, N), BIG, dtype=torch.long, device=self.device)
        atk_slots = torch.arange(N, device=self.device).expand(B, N)
        owner.scatter_reduce_(1, a_tgt,
                              torch.where(attack, atk_slots, torch.full_like(atk_slots, BIG)),
                              reduce='amin', include_self=True)
        has_owner = owner < BIG
        conv = neg & has_owner
        new_team = self.fteam.gather(1, owner.clamp(max=N - 1))
        # rebase health up by NEW_HEALTH until >= 0
        h = self.fhealth
        steps = ((-h + NEW_HEALTH - 1) // NEW_HEALTH).clamp(min=0)
        h = h + steps * NEW_HEALTH
        self.fhealth = torch.where(conv, h, self.fhealth)
        self.fteam = torch.where(conv, new_team, self.fteam)
        # any still-negative-with-no-owner: clamp to 0 (stays own team, alive)
        self.fhealth = self.fhealth.clamp(min=0, max=MAX_HEALTH)

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
