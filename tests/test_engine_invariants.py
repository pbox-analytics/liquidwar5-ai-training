"""Engine invariants over many ticks of random play.

These guard the properties the faithful engine docstring promises and the
specialist panel said were only ever checked ad hoc:

* **one-per-cell occupancy** — no two living fighters share ``fy*W + fx``;
* **total fighter count conserved** — a fixed army, never created/destroyed;
* **per-team counts change ONLY via conversion** — the sum of per-team deltas is
  zero each tick (a conversion recolours one fighter), so nothing spawns/vanishes;
* **valid ranges** — ``fteam`` in ``[0, T)``, ``fhealth`` in ``[0, MAX_HEALTH]``;
* **the ``occ`` grid agrees with the (fy, fx) SoA**.

Each runs with wells ON and OFF and with discrete-heading inertia 0 and 0.12,
because those branches take different movement/conversion code paths.

A note on one-per-cell: the engine's placement (``_place_teams``) guarantees a
distinct cell per fighter *when the strip has room*; if a wall-heavy map leaves a
team's strip with fewer passable cells than ``fighters_per_team`` it deliberately
PARKS the overflow on the cursor cell (engine.py:366-371) to keep the count
exact. So on a cramped board reset can start with stacked fighters by design. The
real engine guarantee is therefore two-fold and tested as such:

* on a **spacious** board, placement is strictly one-per-cell and movement keeps
  it that way (``test_one_per_cell_strict``);
* on **any** board, movement NEVER increases overlap beyond the reset baseline —
  fighters pack, they never collapse onto each other
  (the ``overlap <= reset_overlap`` check in the matrix tests).
"""
from __future__ import annotations

import itertools

import pytest
import torch

from simulator.engine import MAX_HEALTH
from tests.conftest import make_engine

# (wells_enabled, inertia) matrix. Wells need the per-team well slots (reset()
# always allocates them); we just flip the gate. Inertia is the _inertia knob.
_MATRIX = list(itertools.product([False, True], [0.0, 0.12]))
_TICKS = 100          # per matrix case; x8 cases + the strict run = "a few hundred"
_B = 3
_T = 3
_PER = 22


def _random_actions(B, T, gen):
    return torch.randint(-1, 2, (B, T, 2), generator=gen)


def _overlap_per_game(e):
    """For each game, how many fighters share a cell with an earlier slot
    (0 == strict one-per-cell)."""
    flat = (e.fy * e.W + e.fx)                    # (B, N)
    return [int(flat[b].numel() - torch.unique(flat[b]).numel())
            for b in range(e.B)]


def _build_engine(wells, inertia, height=16, width=22, teams=_T, per=_PER):
    e = make_engine(batch_size=_B, height=height, width=width, num_teams=teams,
                    fighters_per_team=per, grad_iters=4)
    e.reset()
    e._inertia = inertia
    e._wells_enabled = wells
    # Cap gradient relaxation sweeps/tick (the play server's _grad_cap knob).
    # The invariants are independent of how converged the distance field is, and
    # the persistent field keeps accumulating tick to tick — this just bounds the
    # per-tick cost so a few hundred ticks stays a few seconds on CPU.
    e._grad_cap = 10
    return e


def _check_invariants(e, total_per_game, reset_overlap):
    B, W = e.B, e.W
    # (d) ranges first — cheap, and a corrupt team id would break later checks.
    assert e.fteam.min().item() >= 0
    assert e.fteam.max().item() < e.T, "fteam out of [0, T)"
    assert e.fhealth.min().item() >= 0, "health went negative post-tick"
    assert e.fhealth.max().item() <= MAX_HEALTH, "health exceeded MAX_HEALTH"

    # (b) conservation: every game keeps exactly its starting army.
    counts = e.active_fighters.sum(dim=1)         # (B,)
    assert torch.equal(counts, total_per_game), (
        f"total fighters changed: {counts.tolist()} != {total_per_game.tolist()}")
    assert int(counts.min()) == e.N, "active census lost slots"

    # (a) one-per-cell: movement must never INCREASE overlap beyond what reset
    # placement created. (reset_overlap is 0 on a board with room; on a cramped
    # map the engine parks overflow on the cursor cell by design — but a fighter
    # never spontaneously collapses onto another mid-game.)
    overlap = _overlap_per_game(e)
    for b in range(B):
        assert overlap[b] <= reset_overlap[b], (
            f"game {b}: overlap rose to {overlap[b]} from a reset baseline of "
            f"{reset_overlap[b]} — movement collapsed distinct fighters onto a cell")

    # (e) occ grid agrees with the (fy,fx) SoA: occ[b, fy, fx] points back at a
    # slot sitting on that exact cell (last-writer per cell; with one-per-cell
    # that is THE slot), and every non-empty occ entry indexes a real slot.
    flat = (e.fy * W + e.fx)
    occ_flat = e.occ.view(B, -1)
    for b in range(B):
        owner = occ_flat[b].gather(0, flat[b])    # slot occ says owns each cell
        owner_cell = flat[b].gather(0, owner)     # the cell that owner sits on
        assert torch.equal(owner_cell, flat[b]), (
            f"game {b}: occ grid disagrees with (fy,fx) SoA")
        nonempty = occ_flat[b][occ_flat[b] >= 0]
        assert int(nonempty.min()) >= 0 and int(nonempty.max()) < e.N


@pytest.mark.parametrize("wells,inertia", _MATRIX,
                         ids=[f"wells={w}-inertia={i}" for w, i in _MATRIX])
def test_invariants_over_random_play(wells, inertia):
    e = _build_engine(wells, inertia)
    total_per_game = e.active_fighters.sum(dim=1).clone()    # (B,) fixed army
    reset_overlap = _overlap_per_game(e)                     # placement baseline
    gen = torch.Generator().manual_seed(7 + int(wells) * 2 + int(inertia * 100))

    # sanity: starting state already satisfies everything
    _check_invariants(e, total_per_game, reset_overlap)

    for _ in range(_TICKS):
        e.step(_random_actions(e.B, e.T, gen))
    _check_invariants(e, total_per_game, reset_overlap)


def test_one_per_cell_strict():
    """On a SPACIOUS board (every team's strip has far more passable cells than
    fighters), placement is strictly one-per-cell and movement keeps it that
    way — the pure occupancy guarantee, with no reset-time overflow stacking to
    confound it. Verified overlap-free across 80 map seeds offline.
    """
    e = make_engine(batch_size=_B, height=20, width=32, num_teams=2,
                    fighters_per_team=24, grad_iters=4)
    e.reset()
    e._grad_cap = 10
    reset_overlap = _overlap_per_game(e)
    assert reset_overlap == [0] * e.B, (
        f"spacious board should place one-per-cell, got overlaps {reset_overlap}")

    gen = torch.Generator().manual_seed(7)
    for i in range(_TICKS):
        e.step(_random_actions(e.B, e.T, gen))
        # strict: zero overlap — no two living fighters share a cell. Sampled
        # every few ticks (torch.unique per game) to keep the loop cheap; a
        # collapse persists in the SoA until movement clears it, so sampling
        # plus the final check catches any without the per-tick cost.
        if i % 5 == 0:
            assert _overlap_per_game(e) == [0] * e.B, "movement created a cell overlap"
    assert _overlap_per_game(e) == [0] * e.B, "movement created a cell overlap"


@pytest.mark.parametrize("wells,inertia", _MATRIX,
                         ids=[f"wells={w}-inertia={i}" for w, i in _MATRIX])
def test_team_counts_change_only_by_conversion(wells, inertia):
    """Per-team census may shift, but only by RECOLOURING: each tick the
    per-team deltas sum to zero (a conversion moves +1 to the winner / -1 from
    the loser) and the GRAND total is invariant — i.e. no spawn/vanish sneaks
    in. A spawn would raise some team's count with no matching fall; a vanish,
    the reverse.
    """
    e = _build_engine(wells, inertia)
    gen = torch.Generator().manual_seed(99 + int(wells) + int(inertia * 10))
    prev = e.active_fighters.clone()              # (B, T)
    total = prev.sum(dim=1)

    for _ in range(70):
        e.step(_random_actions(e.B, e.T, gen))
        cur = e.active_fighters
        assert torch.equal(cur.sum(dim=1), total), "grand total drifted"
        delta = cur - prev
        assert torch.equal(delta.sum(dim=1), torch.zeros_like(total)), (
            "per-team deltas did not sum to zero -> a count changed by something "
            "other than conversion")
        prev = cur.clone()
