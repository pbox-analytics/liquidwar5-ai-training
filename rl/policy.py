"""Shared convolutional cursor policy for Liquid War self-play PPO.

ONE network controls every team (shared weights). For each team we build a
team-centric observation: the acting team's channels are placed in canonical
slots and the other teams are pooled into "enemy" channels, so the same network
plays any team and any team-count.

Engine observation (simulator/engine.py:get_observation) is
(B, 1+3T, H, W): channel 0 = walls, then T team-presence, T normalized
gradients, T team-health. We re-pack that, per acting team, into a fixed
6-channel egocentric observation:
    [walls, own_presence, own_gradient, own_health,
     enemy_presence_sum, enemy_health_sum]
so the network input width is independent of T.

Action space per team = 9 discrete cursor moves (8 directions + stay),
decoded to the (dy, dx) in {-1,0,1}^2 that engine.step expects.
"""

import torch
import torch.nn as nn


# 9 cursor moves -> (dy, dx), row-major over {-1,0,1}^2. Index 4 = (0,0) = stay.
# (Dead teams are forced to stay by zeroing dydx in act(), not by an index.)
MOVE_DYDX = torch.tensor(
    [[-1, -1], [-1, 0], [-1, 1],
     [0, -1],  [0, 0],  [0, 1],
     [1, -1],  [1, 0],  [1, 1]],
    dtype=torch.long,
)
NUM_MOVES = 9
EGO_CHANNELS = 6
#: FLAT ACTION SPACE: every (stance, re-tap mode) pair the play server offers a
#: human is its own action, so the policy can both USE and FACE all of them —
#: comet, lattice/nova/tide, binary, Doom charge levels, wall orientation,
#: drill gears, maelstrom modes. ``apply_stances`` maps a flat action onto the
#: engine's per-team knobs with the SAME dials as the play server's key
#: handling, and (when ``engine._wells_enabled``) casts the real cross-team
#: Doom well / Maelstrom current.
ACTIONS = (
    ("Swarm", "cloud"), ("Swarm", "comet"),
    ("Spin", "vortex"), ("Spin", "sawblade"), ("Spin", "galaxy"),
    ("Drill", "slow"), ("Drill", "med"), ("Drill", "fast"),
    ("Wall", "horiz"), ("Wall", "vert"),
    ("Pulse", "wave"), ("Pulse", "rings"), ("Pulse", "star"),
    ("Pulse", "lattice"), ("Pulse", "nova"), ("Pulse", "tide"),
    ("Doom", "1x"), ("Doom", "2x"), ("Doom", "3x"),
    ("Maelstrom", "undertow"), ("Maelstrom", "ejecta"), ("Maelstrom", "shear"),
    ("Atom", "orbital"), ("Atom", "binary"),
    ("Classic", ""),
)
NUM_STANCES = len(ACTIONS)        # 25 — the name survives because it sizes the head
#: Legacy 8-stance checkpoints: base stance id -> the flat action it meant
#: (Swarm cloud, Spin vortex, Drill med, Wall horiz, Pulse wave, Doom 1x,
#: Maelstrom undertow, Atom orbital). The play server maps old policies through
#: this so they keep working until a flat-action best.pt beats them.
LEGACY_ACTION = (0, 2, 6, 8, 10, 16, 19, 22)

# Per-action knob tables (rows = ACTIONS). Columns follow the play server's
# stance blocks exactly; 0 = knob off. Tick-phased modes (Pulse wave / nova)
# are patched at apply time.
#          spin burst surge fig8 nl  nm   nk    nw    nv   adv  wy  wx tide rin  ecc dLvl mRad mOn
_KNOBS = [
    (0.5,  0.15, 1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Swarm cloud
    (0.35, -0.25, 1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.85, 0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Swarm comet
    (1.7,  -0.4, 1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Spin vortex
    (1.6,  -0.45, 1.0, 0, 0,  8,  0.0,  0.4,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Spin sawblade
    (1.1,  0.35, 1.0, 0, 0,  3,  0.25, -0.05, 0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Spin galaxy
    (0.3,  0.0,  1.0, 0, 0,  0,  0.0,  0.0,  0.0,  1.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Drill slow
    (0.7,  0.0,  2.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.62, 0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Drill med
    (1.5,  0.0,  4.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.34, 0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Drill fast
    (0.0,  -0.9, 1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  1, 0, 0, 0.0,  0, 0, 0.0,  0),  # Wall horiz
    (0.0,  -0.9, 1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 1, 0, 0.0,  0, 0, 0.0,  0),  # Wall vert
    (0.0,  0.0,  1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Pulse wave (tick-phased)
    (0.0,  0.0,  3.0, 0, 12, 0,  0.0,  0.0,  0.05, 0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Pulse rings
    (0.0,  0.0,  3.0, 0, 16, 6,  0.0,  0.05, 0.05, 0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Pulse star
    (0.0,  0.0,  2.5, 0, 16, 8,  0.15, 0.03, 0.05, 0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Pulse lattice
    (0.0,  0.0,  1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Pulse nova (tick-phased)
    (0.0,  0.0,  2.5, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 1, 0.0,  0, 0, 0.0,  0),  # Pulse tide
    (1.2,  0.0,  1.5, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 6.7,  1, 1, 0.0,  0),  # Doom 1x
    (1.8,  0.0,  1.75, 0, 0, 0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 9.0,  1, 2, 0.0,  0),  # Doom 2x
    (2.4,  0.0,  2.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 11.3, 1, 3, 0.0,  0),  # Doom 3x
    (2.0,  0.6,  1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.30, 1),  # Maelstrom undertow
    (2.0,  0.6,  1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, -0.7, 1),  # Maelstrom ejecta
    (2.0,  0.6,  1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  1),  # Maelstrom shear
    (1.8,  0.4,  1.0, 1, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Atom orbital
    (1.6,  0.45, 1.0, 2, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Atom binary
    (0.0,  0.0,  1.0, 0, 0,  0,  0.0,  0.0,  0.0,  0.0,  0, 0, 0, 0.0,  0, 0, 0.0,  0),  # Classic
]
_KNOB_NAMES = ("spin", "burst", "surge", "fig8", "nl", "nm", "nk", "nw", "nv",
               "adv", "wy", "wx", "tide", "rin", "ecc", "dlvl", "mrad", "mon")
_TABS_CACHE = {}


def _tables(dev):
    """Per-device cache of the (25,) knob lookup tensors."""
    tabs = _TABS_CACHE.get(dev)
    if tabs is None:
        cols = list(zip(*_KNOBS))
        tabs = {name: torch.tensor(col, dtype=torch.float32, device=dev)
                for name, col in zip(_KNOB_NAMES, cols)}
        _TABS_CACHE[dev] = tabs
    return tabs


def build_egocentric_obs(obs, num_teams):
    """Re-pack engine obs into per-team egocentric views.

    Args:
        obs: (B, 1+3T, H, W) from engine.get_observation().
        num_teams: T.

    Returns:
        (B, T, EGO_CHANNELS, H, W) — one 6-channel view per team.
    """
    B, C, H, W = obs.shape
    T = num_teams
    assert C == 1 + 3 * T, f"expected 1+3T={1+3*T} channels, got {C}"

    walls = obs[:, 0:1]                       # (B,1,H,W)
    presence = obs[:, 1:1 + T]                # (B,T,H,W)
    gradient = obs[:, 1 + T:1 + 2 * T]        # (B,T,H,W)
    health = obs[:, 1 + 2 * T:1 + 3 * T]      # (B,T,H,W)

    presence_sum = presence.sum(dim=1, keepdim=True)   # (B,1,H,W)
    health_sum = health.sum(dim=1, keepdim=True)       # (B,1,H,W)

    views = []
    for t in range(T):
        own_p = presence[:, t:t + 1]
        own_g = gradient[:, t:t + 1]
        own_h = health[:, t:t + 1]
        enemy_p = (presence_sum - own_p).clamp(min=0)
        enemy_h = (health_sum - own_h).clamp(min=0)
        views.append(torch.cat(
            [walls, own_p, own_g, own_h, enemy_p, enemy_h], dim=1))
    return torch.stack(views, dim=1)         # (B,T,EGO,H,W)


class CursorPolicy(nn.Module):
    """Conv actor-critic. Input (N, EGO_CHANNELS, H, W) -> move logits + value.

    Global-pooled so it is map-size agnostic. Applied to N = B*T flattened
    team-views; caller reshapes back to (B, T, ...).
    """

    def __init__(self, ego_channels=EGO_CHANNELS, hidden=64,
                 num_moves=NUM_MOVES, num_stances=NUM_STANCES):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ego_channels, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, stride=2), nn.ReLU(),
        )
        # Global average + max pool concatenated -> 2*hidden feature vector.
        self.actor = nn.Sequential(                       # cursor-move head
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_moves),
        )
        self.stance_head = nn.Sequential(                 # tactical-stance head
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_stances),
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        """x: (N, EGO, H, W) -> (move_logits (N,moves), stance_logits (N,stances), value (N,))."""
        h = self.body(x)
        gap = h.mean(dim=(2, 3))
        gmp = h.amax(dim=(2, 3))
        feat = torch.cat([gap, gmp], dim=1)
        move_logits = self.actor(feat)
        stance_logits = self.stance_head(feat)
        value = self.critic(feat).squeeze(-1)
        return move_logits, stance_logits, value


def act(policy, obs, num_teams, team_alive=None, deterministic=False,
        held_stance=None, decide=True):
    """Sample cursor actions for all teams from the shared policy.

    Args:
        policy: CursorPolicy.
        obs: (B, 1+3T, H, W) engine observation.
        num_teams: T.
        team_alive: (B, T) bool — dead teams emit "stay" with zero logprob.
        deterministic: argmax instead of sampling (eval).
        held_stance: (B, T) long — STICKY STANCES (training): the stance held
            since the last decision tick. When ``decide`` is False the held
            stance is reused and the joint logprob/entropy carry the MOVE term
            only — a stance that lasts one tick can never form a sawblade or
            charge a nova, so per-tick resampling starves the stance head of
            credit (the entropy bonus then pins it at uniform). Decisions
            happen every K ticks; ppo_update masks the stance terms to match.
        decide: this tick is a stance-decision tick (always True in play/eval).

    Returns:
        actions: (B, T, 2) long in {-1,0,1} for engine.step.
        stance:  (B, T) long in 0..NUM_STANCES-1 (feed to :func:`apply_stances`).
        logprob: (B, T) joint log-prob of (move, stance) (0 for dead teams).
        value:   (B, T) critic value.
        entropy: (B, T) joint policy entropy.
    """
    B = obs.shape[0]
    T = num_teams
    ego = build_egocentric_obs(obs, T)            # (B,T,EGO,H,W)
    flat = ego.reshape(B * T, EGO_CHANNELS, *ego.shape[-2:])
    move_logits, stance_logits, value = policy(flat)       # (B*T,9), (B*T,5), (B*T,)
    mdist = torch.distributions.Categorical(logits=move_logits)
    sdist = torch.distributions.Categorical(logits=stance_logits)
    if deterministic:
        move = move_logits.argmax(dim=-1)
        stance = stance_logits.argmax(dim=-1)
    else:
        move = mdist.sample()
        stance = sdist.sample()
    if held_stance is not None and not decide:               # sticky: hold, move-only credit
        stance = held_stance.reshape(-1)
        logprob = mdist.log_prob(move)
        entropy = mdist.entropy()
    else:
        logprob = mdist.log_prob(move) + sdist.log_prob(stance)   # joint (independent heads)
        entropy = mdist.entropy() + sdist.entropy()

    move = move.view(B, T)
    stance = stance.view(B, T)
    logprob = logprob.view(B, T)
    value = value.view(B, T)
    entropy = entropy.view(B, T)

    dydx = MOVE_DYDX.to(obs.device)[move]         # (B,T,2)

    if team_alive is not None:
        dead = ~team_alive
        # Dead teams: force "stay" + Swarm, zero logprob/value/entropy.
        dydx = torch.where(dead.unsqueeze(-1), torch.zeros_like(dydx), dydx)
        stance = torch.where(dead, torch.zeros_like(stance), stance)
        logprob = torch.where(dead, torch.zeros_like(logprob), logprob)
        value = torch.where(dead, torch.zeros_like(value), value)
        entropy = torch.where(dead, torch.zeros_like(entropy), entropy)

    return dydx, stance, logprob, value, entropy


def apply_stances(engine, action, dydx, team_start=0, human_teams=None):
    """Set the engine's per-team knobs from each team's chosen FLAT action —
    the exact stance+mode mapping the play server applies from player keys,
    vectorized over all (B, T) teams via the per-action knob tables. Also
    casts the cross-team Doom/Maelstrom wells (``engine._wells_enabled``).
    Call right before ``engine.step``.

    :param engine: the :class:`LiquidWarEngine` being driven.
    :param action: ``(B, T)`` long in 0..NUM_STANCES-1 (flat stance-mode id).
    :param dydx: ``(B, T, 2)`` the cursor move, reused as the Drill / Wall /
        comet / tide aim.
    :param team_start: only write knobs for teams ``>= team_start`` (so the
        play server can stance the AI opponents 1.. while leaving the human's
        team-0 knobs, already set from keys, intact). 0 = all teams (training).
    :param human_teams: room play — the set of seats held by HUMANS; the AI
        drives exactly the complement (overrides ``team_start``). The human
        seats' knobs and well slots are never touched.
    """
    B, T = action.shape
    dev = action.device
    tabs = _tables(dev)
    a = action.clamp(0, NUM_STANCES - 1)
    g = lambda name: tabs[name][a]                                # (B,T) lookups
    spin, burst, surge, fig8 = g("spin"), g("burst"), g("surge"), g("fig8")
    node_l, node_m, node_k, node_w, node_v = g("nl"), g("nm"), g("nk"), g("nw"), g("nv")
    ring_rin, ring_ecc = g("rin"), g("ecc")
    doom_lvl, mael_rad, mael_on = g("dlvl"), g("mrad"), g("mon")
    aim = dydx.float()
    drill = aim * g("adv").unsqueeze(-1)                          # comet + drill gears
    wall = torch.stack([g("wy"), g("wx")], dim=-1)
    tide = aim * g("tide").unsqueeze(-1)
    # tick-phased modes (same clocks as the play server's stance blocks)
    tick = int(engine.tick)
    wv = (a == 10)                                                # Pulse wave: traveling rings
    ringph = float(torch.sin(torch.as_tensor(tick * 0.33)))
    burst = torch.where(wv, torch.full_like(burst, 1.0 if ringph > 0 else -0.6), burst)
    surge = torch.where(wv, torch.full_like(surge, 4.0 if ringph > 0.5 else 1.0), surge)
    nv_m = (a == 14)                                              # Pulse nova: charge -> detonate
    if tick % 144 < 108:
        burst = torch.where(nv_m, torch.full_like(burst, -0.9), burst)
        spin = torch.where(nv_m, torch.full_like(spin, 0.6), spin)
    else:
        burst = torch.where(nv_m, torch.full_like(burst, 1.0), burst)
        surge = torch.where(nv_m, torch.full_like(surge, 5.0), surge)
    # mass-scaled pieces: Doom's accretion-disk radius + both wells' dials
    af = engine.active_fighters.float()                           # (B,T) live mass
    frac = (af / max(1.0, engine.fighters_per_team)).clamp(0.0, 1.0)
    blob_r = (af / 3.14159).sqrt()
    ring = torch.where(ring_rin > 0,
                       0.5 * (ring_rin + (ring_rin ** 2 + af / 3.14159).sqrt()),
                       torch.zeros_like(ring_rin))
    if human_teams is None and team_start <= 0:                 # training: drive all teams
        engine._spin, engine._burst, engine._drill, engine._wall, engine._surge = spin, burst, drill, wall, surge
        engine._fig8 = fig8
        engine._node_l, engine._node_m, engine._node_k = node_l, node_m, node_k
        engine._node_w, engine._node_v = node_w, node_v
        engine._tide = tide
        engine._ring, engine._ring_ecc = ring, ring_ecc
        cols = list(range(T))                                    # wells: every team
    else:                                                        # play: ONLY the AI-held seats; humans keep their key-set knobs
        if engine._surge is None:
            engine._surge = torch.ones(B, T, device=dev)
        cols = ([t for t in range(T) if t not in human_teams] if human_teams is not None
                else list(range(team_start, T)))
        if cols:
            for name, val in (("_spin", spin), ("_burst", burst), ("_surge", surge),
                              ("_fig8", fig8), ("_node_l", node_l), ("_node_m", node_m),
                              ("_node_k", node_k), ("_node_w", node_w), ("_node_v", node_v),
                              ("_ring", ring), ("_ring_ecc", ring_ecc),
                              ("_drill", drill), ("_wall", wall), ("_tide", tide)):
                getattr(engine, name)[:, cols] = val[:, cols]
    if getattr(engine, "_wells_enabled", False) and cols:
        # REAL cross-team wells, mass-scaled like the play server's key
        # handling (Doom charge level from the action; Maelstrom mode sets the
        # radial component). In-place slot writes on the AI-held seats only —
        # human seats' slots, written from their keys, are never touched.
        cpos = engine.cursor_pos.float()
        d_on = (doom_lvl > 0).float()
        engine._doom_pos[:, cols] = cpos[:, cols]
        engine._doom_str[:, cols] = (doom_lvl * 24.0 * frac ** 1.5)[:, cols]
        engine._doom_range[:, cols] = (2.2 * ring).clamp(min=70.0)[:, cols]
        engine._doom_horizon[:, cols] = (ring_rin * 1.25 * d_on)[:, cols]   # the rendered hole, not blob-scaled (no snowball)
        engine._doom_cap[:, cols] = (0.09 * frac.sqrt() * d_on)[:, cols]
        engine._vortex_pos[:, cols] = cpos[:, cols]
        engine._vortex_str[:, cols] = (22.0 * frac.sqrt() * mael_on)[:, cols]
        engine._vortex_range[:, cols] = (1.5 * blob_r).clamp(min=60.0)[:, cols]
        engine._vortex_sign[:, cols] = 1.0
        engine._vortex_rad[:, cols] = mael_rad[:, cols]
