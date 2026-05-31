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
import torch.nn.functional as F


# 9 cursor moves -> (dy, dx). Index 8 = stay.
MOVE_DYDX = torch.tensor(
    [[-1, -1], [-1, 0], [-1, 1],
     [0, -1],  [0, 0],  [0, 1],
     [1, -1],  [1, 0],  [1, 1]],
    dtype=torch.long,
)
NUM_MOVES = 9
EGO_CHANNELS = 6


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

    def __init__(self, ego_channels=EGO_CHANNELS, hidden=64, num_moves=NUM_MOVES):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ego_channels, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, stride=2), nn.ReLU(),
        )
        # Global average + max pool concatenated -> 2*hidden feature vector.
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_moves),
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        """x: (N, EGO_CHANNELS, H, W) -> (logits (N, num_moves), value (N,))."""
        h = self.body(x)
        gap = h.mean(dim=(2, 3))
        gmp = h.amax(dim=(2, 3))
        feat = torch.cat([gap, gmp], dim=1)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value


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
        logprob: (B, T) log-prob of the chosen move (0 for dead teams).
        value:   (B, T) critic value.
        entropy: (B, T) policy entropy.
    """
    B = obs.shape[0]
    T = num_teams
    ego = build_egocentric_obs(obs, T)            # (B,T,EGO,H,W)
    flat = ego.reshape(B * T, EGO_CHANNELS, *ego.shape[-2:])
    logits, value = policy(flat)                   # (B*T, 9), (B*T,)
    dist = torch.distributions.Categorical(logits=logits)
    if deterministic:
        move = logits.argmax(dim=-1)
    else:
        move = dist.sample()
    logprob = dist.log_prob(move)
    entropy = dist.entropy()

    move = move.view(B, T)
    logprob = logprob.view(B, T)
    value = value.view(B, T)
    entropy = entropy.view(B, T)

    dydx = MOVE_DYDX.to(obs.device)[move]         # (B,T,2)

    if team_alive is not None:
        dead = ~team_alive
        # Dead teams: force "stay" (index 8 -> (0,0)) and zero logprob.
        dydx = torch.where(dead.unsqueeze(-1), torch.zeros_like(dydx), dydx)
        logprob = torch.where(dead, torch.zeros_like(logprob), logprob)
        value = torch.where(dead, torch.zeros_like(value), value)
        entropy = torch.where(dead, torch.zeros_like(entropy), entropy)

    return dydx, logprob, value, entropy
