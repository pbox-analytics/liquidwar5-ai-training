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
#: Tactical stances the policy can hold (mirror the play server / engine knobs):
#: 0 Swarm / 1 Spin / 2 Drill / 3 Wall / 4 Pulse / 5 Doom / 6 Maelstrom / 7 Atom.
#: The stance head picks one per team per tick; ``apply_stances`` maps it onto the
#: engine's per-team knobs. (Doom in training is self-collapse only — the cross-team
#: gravity well is a play-server-only effect.)
NUM_STANCES = 8


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


def act(policy, obs, num_teams, team_alive=None, deterministic=False):
    """Sample cursor actions for all teams from the shared policy.

    Args:
        policy: CursorPolicy.
        obs: (B, 1+3T, H, W) engine observation.
        num_teams: T.
        team_alive: (B, T) bool — dead teams emit "stay" with zero logprob.
        deterministic: argmax instead of sampling (eval).

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


def apply_stances(engine, stance, dydx, team_start=0):
    """Set the engine's per-team knobs from each team's chosen stance — the exact
    Swarm/Spin/Drill/Wall/Pulse mapping the play server applies from player keys,
    but vectorized over all (B, T) teams. Call right before ``engine.step``.

    :param engine: the :class:`LiquidWarEngine` being driven.
    :param stance: ``(B, T)`` long in 0..4 (the chosen stance per team).
    :param dydx: ``(B, T, 2)`` the cursor move, reused as the Drill/Wall aim.
    :param team_start: only write knobs for teams ``>= team_start`` (so the play
        server can stance the AI opponents 1.. while leaving the human's team 0
        knobs, already set from keys, intact). Default 0 = all teams (training).
    """
    B, T = stance.shape
    dev = stance.device
    spin = torch.zeros(B, T, device=dev)
    burst = torch.zeros(B, T, device=dev)
    drill = torch.zeros(B, T, 2, device=dev)
    wall = torch.zeros(B, T, 2, device=dev)
    surge = torch.ones(B, T, device=dev)
    fig8 = torch.zeros(B, T, device=dev)
    aim = dydx.float()
    sw, sp, dr, wl, pu = (stance == 0), (stance == 1), (stance == 2), (stance == 3), (stance == 4)
    spin = torch.where(sw, torch.full_like(spin, 0.5), spin)      # Swarm: loose orbit
    burst = torch.where(sw, torch.full_like(burst, 0.15), burst)
    spin = torch.where(sp, torch.full_like(spin, 1.7), spin)      # Spin: tight fast vortex
    burst = torch.where(sp, torch.full_like(burst, -0.4), burst)
    spin = torch.where(dr, torch.full_like(spin, 0.5), spin)      # Drill: pierce (medium mode)
    drill = torch.where(dr.unsqueeze(-1), aim * 0.62, drill)
    surge = torch.where(dr, torch.full_like(surge, 2.0), surge)
    wall = torch.where(wl.unsqueeze(-1), aim, wall)              # Wall: shield across the aim
    ring = float(torch.sin(torch.as_tensor(float(engine.tick) * 0.33)))   # Pulse: concentric rings
    burst = torch.where(pu, torch.full_like(burst, 1.0 if ring > 0 else -0.6), burst)
    surge = torch.where(pu, torch.full_like(surge, 4.0 if ring > 0.5 else 1.0), surge)
    dm, ml, at = (stance == 5), (stance == 6), (stance == 7)
    spin = torch.where(dm, torch.full_like(spin, 0.25), spin)     # Doom: violent implosion +
    burst = torch.where(dm, torch.full_like(burst, -6.5), burst)  #   devastation (self-collapse only here;
    surge = torch.where(dm, torch.full_like(surge, 6.0), surge)   #   the cross-team well is play-only)
    spin = torch.where(ml, torch.full_like(spin, 2.0), spin)      # Maelstrom: fast wide orbiting shell
    burst = torch.where(ml, torch.full_like(burst, 0.6), burst)   #   (the cross-team whirlpool current is play-only)
    spin = torch.where(at, torch.full_like(spin, 1.8), spin)      # Atom: figure-8 orbitals (engine _fig8)
    burst = torch.where(at, torch.full_like(burst, 0.4), burst)
    fig8 = torch.where(at, torch.full_like(fig8, 1.0), fig8)
    if team_start <= 0:                                          # training: drive all teams
        engine._spin, engine._burst, engine._drill, engine._wall, engine._surge = spin, burst, drill, wall, surge
        engine._fig8 = fig8
    else:                                                        # play: ONLY the AI opponents 1..; keep team 0 (human)
        if engine._surge is None:
            engine._surge = torch.ones(B, T, device=dev)
        if getattr(engine, "_fig8", None) is None:
            engine._fig8 = torch.zeros(B, T, device=dev)
        engine._spin[:, team_start:] = spin[:, team_start:]
        engine._burst[:, team_start:] = burst[:, team_start:]
        engine._drill[:, team_start:] = drill[:, team_start:]
        engine._wall[:, team_start:] = wall[:, team_start:]
        engine._surge[:, team_start:] = surge[:, team_start:]
        engine._fig8[:, team_start:] = fig8[:, team_start:]
