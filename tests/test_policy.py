"""CursorPolicy + act() contract.

Pins the documented shapes of ``act()`` across team counts, that the play
server's stance-mask / stance-temp governor kwargs behave (a -inf mask on the
Doom actions 16-18 makes argmax never pick Doom), and that the legacy-checkpoint
head-size mapping the play server relies on is sane.
"""
from __future__ import annotations

import pytest
import torch

from rl import policy
from rl.policy import (ACTIONS, EGO_CHANNELS, LEGACY_ACTION, NUM_MOVES,
                       NUM_STANCES, CursorPolicy, act)


def _obs(B, T, H=18, W=24):
    """A synthetic engine-style observation: (B, 1+3T, H, W)."""
    return torch.randn(B, 1 + 3 * T, H, W)


def test_policy_constructs_and_forward_shapes():
    p = CursorPolicy()
    p.eval()
    x = torch.randn(5, EGO_CHANNELS, 18, 24)
    move_logits, stance_logits, value = p(x)
    assert move_logits.shape == (5, NUM_MOVES)
    assert stance_logits.shape == (5, NUM_STANCES)
    assert value.shape == (5,)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 6])
def test_act_shapes_for_team_counts(T):
    B = 2
    p = CursorPolicy()
    p.eval()
    obs = _obs(B, T)
    team_alive = torch.ones(B, T, dtype=torch.bool)
    dydx, stance, logprob, value, entropy = act(
        p, obs, T, team_alive=team_alive, deterministic=True)
    assert dydx.shape == (B, T, 2)
    assert dydx.dtype == torch.long
    assert stance.shape == (B, T)
    assert logprob.shape == (B, T)
    assert value.shape == (B, T)
    assert entropy.shape == (B, T)
    # dydx is a unit move in {-1,0,1}
    assert int(dydx.min()) >= -1 and int(dydx.max()) <= 1
    # stance in range
    assert int(stance.min()) >= 0 and int(stance.max()) < NUM_STANCES


def test_dead_teams_forced_to_stay_and_zeroed():
    B, T = 2, 3
    p = CursorPolicy()
    p.eval()
    obs = _obs(B, T)
    team_alive = torch.ones(B, T, dtype=torch.bool)
    team_alive[0, 1] = False                      # kill one team in game 0
    with torch.no_grad():
        dydx, stance, logprob, value, entropy = act(
            p, obs, T, team_alive=team_alive, deterministic=True)
    # dead team: stay (0,0), zero stance/logprob/value/entropy
    assert torch.equal(dydx[0, 1], torch.zeros(2, dtype=torch.long))
    assert int(stance[0, 1]) == 0
    assert float(logprob[0, 1]) == 0.0
    assert float(value[0, 1]) == 0.0
    assert float(entropy[0, 1]) == 0.0


def test_stance_mask_blocks_doom_under_argmax():
    """The play server's Doom-uptime governor passes an additive -inf mask on
    the Doom actions (16, 17, 18). Under deterministic argmax that must make the
    stance head NEVER choose Doom, for every team-view.
    """
    B, T = 4, 3
    p = CursorPolicy()
    p.eval()
    obs = _obs(B, T)
    team_alive = torch.ones(B, T, dtype=torch.bool)
    doom_actions = [i for i, (s, _m) in enumerate(ACTIONS) if s == "Doom"]
    assert doom_actions == [16, 17, 18], (
        f"Doom actions moved to {doom_actions}; update the governor mask test")

    # additive mask broadcastable to (B*T, NUM_STANCES): -inf on Doom rows.
    mask = torch.zeros(B, T, NUM_STANCES)
    for a in doom_actions:
        mask[..., a] = float("-inf")

    dydx, stance, logprob, value, entropy = act(
        p, obs, T, team_alive=team_alive, deterministic=True, stance_mask=mask)
    assert not torch.isin(stance, torch.tensor(doom_actions)).any(), (
        "argmax picked a Doom stance despite a -inf mask on 16-18")
    # masking must not produce NaN logprob/value
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(value).all()


def test_stance_mask_blocks_doom_under_stance_temp_sampling():
    """``stance_temp`` > 0 samples the stance head (used in play to show the
    true mixture) — even then a -inf mask must give Doom probability 0, so the
    governor cannot be defeated by the temperature path."""
    torch.manual_seed(0)
    B, T = 6, 2
    p = CursorPolicy()
    p.eval()
    obs = _obs(B, T)
    team_alive = torch.ones(B, T, dtype=torch.bool)
    doom_actions = torch.tensor([16, 17, 18])
    mask = torch.zeros(B, T, NUM_STANCES)
    mask[..., 16:19] = float("-inf")

    # Sample many times: Doom must never appear.
    for _ in range(40):
        _d, stance, _lp, _v, _e = act(
            p, obs, T, team_alive=team_alive, deterministic=True,
            stance_temp=1.5, stance_mask=mask)
        assert not torch.isin(stance, doom_actions).any(), (
            "stance_temp sampling chose Doom despite -inf mask")


def test_held_stance_move_only_credit_path():
    """When ``decide=False`` and a held stance is supplied, act() reuses the held
    stance and the joint logprob carries only the MOVE term (the sticky-stance
    training path). Shapes must still match and the returned stance must equal
    the held one for live teams."""
    B, T = 2, 3
    p = CursorPolicy()
    p.eval()
    obs = _obs(B, T)
    team_alive = torch.ones(B, T, dtype=torch.bool)
    held = torch.full((B, T), 5)                  # Drill slow, say
    dydx, stance, logprob, value, entropy = act(
        p, obs, T, team_alive=team_alive, deterministic=True,
        held_stance=held, decide=False)
    assert stance.shape == (B, T)
    assert torch.equal(stance, held), "held stance not reused when decide=False"
    assert logprob.shape == (B, T)


# --------------------------------------------------------------------------
# legacy-checkpoint head-size mapping (the play server's _load_policy logic)
# --------------------------------------------------------------------------

def test_legacy_action_table_is_sane():
    """LEGACY_ACTION maps each legacy base-stance id to a flat action; every
    entry must index a real action, and there must be one per legacy stance."""
    assert len(LEGACY_ACTION) == 8, "legacy era had 8 base stances"
    for i, a in enumerate(LEGACY_ACTION):
        assert 0 <= a < NUM_STANCES, f"LEGACY_ACTION[{i}]={a} out of range"
    # the documented mapping names: Swarm cloud, Spin vortex, Drill med, Wall
    # horiz, Pulse wave, Doom 1x, Maelstrom undertow, Atom orbital.
    expected = [
        ("Swarm", "cloud"), ("Spin", "vortex"), ("Drill", "med"),
        ("Wall", "horiz"), ("Pulse", "wave"), ("Doom", "1x"),
        ("Maelstrom", "undertow"), ("Atom", "orbital"),
    ]
    assert [ACTIONS[a] for a in LEGACY_ACTION] == expected, (
        "LEGACY_ACTION no longer maps to its documented stance names")


def test_legacy_head_sized_policy_loads_into_full_head():
    """The server sizes a policy's stance head from the checkpoint
    (``CursorPolicy(num_stances=n_act)``) then maps actions through
    LEGACY_ACTION. Simulate: a legacy 8-stance policy's weights must load into
    an 8-head CursorPolicy, and act() must only ever emit ids < 8 (the legacy
    head), which LEGACY_ACTION can remap into the flat space.
    """
    legacy_n = len(LEGACY_ACTION)                 # 8
    legacy_policy = CursorPolicy(num_stances=legacy_n)
    sd = legacy_policy.state_dict()
    # round-trip: a fresh policy sized from the checkpoint head accepts it
    reloaded = CursorPolicy(num_stances=sd["stance_head.2.weight"].shape[0])
    reloaded.load_state_dict(sd)
    reloaded.eval()
    assert sd["stance_head.2.weight"].shape[0] == legacy_n

    B, T = 2, 2
    obs = _obs(B, T)
    _d, stance, _lp, _v, _e = act(
        reloaded, obs, T, team_alive=torch.ones(B, T, dtype=torch.bool),
        deterministic=True)
    assert int(stance.max()) < legacy_n, "legacy head emitted an out-of-head id"
    # and remapping through LEGACY_ACTION lands inside the flat action space
    remap = torch.tensor(LEGACY_ACTION)[stance]
    assert int(remap.min()) >= 0 and int(remap.max()) < NUM_STANCES


def test_full_width_head_matches_num_stances():
    """A current full-width checkpoint has a stance head of NUM_STANCES — the
    server's ``n_act == NUM_STANCES`` (full head, raw mapping) branch."""
    p = CursorPolicy()
    assert p.state_dict()["stance_head.2.weight"].shape[0] == NUM_STANCES
