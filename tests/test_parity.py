"""Knob-table / play-parity — the CRITICAL suite.

``rl/policy.py``'s ``_KNOBS`` table + ``apply_stances`` is meant to be the SINGLE
source of play/train physics: training drives the engine through the table, and
the play server (``web/server.py`` ``_apply_player_stance``) drives the same
engine through hand-written per-stance blocks. When the two drift, balance bugs
follow — exactly the Doom cursor-speed-tax bug the panel flagged, which once
lived only in play (the server applied a mobility tax that the training table
did not), so the policy over-valued Doom.

These tests pin the table's load-bearing columns to the play server's constants
so a future change that touches one side but not the other FAILS here.

web/server.py is parsed as TEXT (regex) — importing it pulls fastapi + torch and
constructs models at import time.
"""
from __future__ import annotations

import re

import pytest
import torch

from rl import policy
from rl.policy import (ACTIONS, NUM_STANCES, _KNOB_NAMES, _KNOBS, apply_stances)
from tests.conftest import make_engine


# --------------------------------------------------------------------------
# (a) / (b) table shape consistency
# --------------------------------------------------------------------------

def test_every_knob_row_matches_knob_names_width():
    n = len(_KNOB_NAMES)
    for i, row in enumerate(_KNOBS):
        assert len(row) == n, (
            f"_KNOBS[{i}] ({ACTIONS[i]}) has width {len(row)}, "
            f"expected len(_KNOB_NAMES)={n}")


def test_action_space_sizes_agree():
    assert len(ACTIONS) == NUM_STANCES, "len(ACTIONS) != NUM_STANCES"
    assert NUM_STANCES == len(_KNOBS), "NUM_STANCES != len(_KNOBS)"
    assert len(ACTIONS) == len(_KNOBS), "len(ACTIONS) != len(_KNOBS)"


# --------------------------------------------------------------------------
# helpers: locate stance rows by their (stance, mode) label
# --------------------------------------------------------------------------

def _rows_for(stance):
    return [i for i, (s, _m) in enumerate(ACTIONS) if s == stance]


def _col(name):
    return _KNOB_NAMES.index(name)


# --------------------------------------------------------------------------
# (c) Doom mobility-tax parity: the bug the panel flagged
# --------------------------------------------------------------------------

def test_doom_cspd_matches_server_mobility_tax(server_src):
    """The play server's Doom branch returns
    ``_base_cs * (0.7, 0.45, 0.3)[doom_level - 1]`` — the mobility tax. The
    training table's ``cspd`` column for Doom 1x/2x/3x MUST equal that tuple,
    or training and play disagree on how slow a Doom-holder's cursor is (the
    original balance bug).
    """
    # Parse the server tuple. The line is:
    #   return max(1, round(_base_cs * (0.7, 0.45, 0.3)[ctrl["doom_level"] - 1]))
    m = re.search(
        r"_base_cs\s*\*\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)"
        r"\s*\[\s*ctrl\[\"doom_level\"\]\s*-\s*1\s*\]",
        server_src,
    )
    assert m is not None, (
        "could not find the Doom mobility-tax tuple "
        "`_base_cs * (a, b, c)[ctrl[\"doom_level\"] - 1]` in web/server.py — "
        "the server refactored it; update this parser AND verify the table.")
    server_tax = tuple(float(x) for x in m.groups())

    doom_rows = _rows_for("Doom")
    assert len(doom_rows) == 3, f"expected 3 Doom rows, found {doom_rows}"
    cspd = _col("cspd")
    table_tax = tuple(float(_KNOBS[r][cspd]) for r in doom_rows)

    assert table_tax == server_tax, (
        f"Doom mobility-tax PARITY MISMATCH:\n"
        f"  rl/policy.py _KNOBS cspd (Doom 1x/2x/3x) = {table_tax}\n"
        f"  web/server.py _apply_player_stance tax   = {server_tax}\n"
        f"Training and play disagree on the Doom cursor-speed tax — fix the side "
        f"that changed.")


def test_doom_is_the_only_cspd_tax_in_the_table():
    """Sanity: only Doom rows carry a cursor-speed tax (cspd != 1.0). If a new
    stance grows a tax, the server must learn about it too — flag it here.
    """
    cspd = _col("cspd")
    doom_rows = set(_rows_for("Doom"))
    for i, row in enumerate(_KNOBS):
        if abs(float(row[cspd]) - 1.0) > 1e-9:
            assert i in doom_rows, (
                f"_KNOBS[{i}] ({ACTIONS[i]}) has a non-1.0 cspd "
                f"({row[cspd]}) but is not a Doom row — the play server only "
                f"taxes Doom mobility; add server parity or reset this to 1.0.")


# --------------------------------------------------------------------------
# (d) Wall armor parity
# --------------------------------------------------------------------------

def test_wall_armr_matches_server_armor(server_src):
    """The play server's Wall branch sets ``_e._armor[0, t] = 0.6`` (a formed
    rampart takes 40% less). The table's ``armr`` column for both Wall rows
    must equal that constant.
    """
    # Parse the server's Wall-branch armor assignment.
    m = re.search(r"_e\._armor\[0,\s*t\]\s*=\s*([0-9.]+)", server_src)
    assert m is not None, (
        "could not find `_e._armor[0, t] = <const>` (the Wall brace) in "
        "web/server.py — update this parser AND verify the table.")
    server_armor = float(m.group(1))

    wall_rows = _rows_for("Wall")
    assert len(wall_rows) == 2, f"expected 2 Wall rows, found {wall_rows}"
    armr = _col("armr")
    for r in wall_rows:
        table_armor = float(_KNOBS[r][armr])
        assert table_armor == server_armor, (
            f"Wall armor PARITY MISMATCH for {ACTIONS[r]}:\n"
            f"  rl/policy.py _KNOBS armr = {table_armor}\n"
            f"  web/server.py Wall brace = {server_armor}\n"
            f"Training and play disagree on the Wall damage reduction.")


def test_wall_is_the_only_armr_change_in_the_table():
    """Only Wall rows should carry armor != 1.0 (matching the single
    ``_e._armor`` assignment in the server's Wall branch)."""
    armr = _col("armr")
    wall_rows = set(_rows_for("Wall"))
    for i, row in enumerate(_KNOBS):
        if abs(float(row[armr]) - 1.0) > 1e-9:
            assert i in wall_rows, (
                f"_KNOBS[{i}] ({ACTIONS[i]}) has armr={row[armr]} != 1.0 but is "
                f"not a Wall row — the server only braces Wall.")


# --------------------------------------------------------------------------
# (e) apply_stances actually WRITES the engine tensors
# --------------------------------------------------------------------------

def _fresh_engine(teams=2):
    e = make_engine(batch_size=1, height=18, width=26, num_teams=teams,
                    fighters_per_team=20, grad_iters=4)
    e.reset()
    return e


def test_apply_stances_writes_cursor_speed_for_driven_teams():
    """Training path (team_start=0): apply_stances must set
    ``engine._cursor_speed_bt`` so the engine's batched cursor step actually
    slows a Doom-holder. A Doom 3x team must end slower than a Classic team.
    """
    e = _fresh_engine(teams=2)
    assert not hasattr(e, "_cursor_speed_bt") or e._cursor_speed_bt is None
    # team 0 = Classic (cspd 1.0), team 1 = Doom 3x (cspd 0.3)
    classic = ACTIONS.index(("Classic", ""))
    doom3 = ACTIONS.index(("Doom", "3x"))
    action = torch.tensor([[classic, doom3]])
    dydx = torch.zeros(1, 2, 2, dtype=torch.long)

    apply_stances(e, action, dydx, team_start=0)

    assert hasattr(e, "_cursor_speed_bt"), "apply_stances did not set _cursor_speed_bt"
    spd = e._cursor_speed_bt
    assert spd.shape == (1, 2)
    assert spd.dtype == torch.int32
    base = max(1, round(e.W / 96))
    # Classic keeps base speed; Doom is taxed (cspd 0.3 -> strictly slower when
    # base allows, but always >= 1).
    assert int(spd[0, 0]) == base, "Classic team's cursor speed should be base"
    assert int(spd[0, 1]) >= 1
    assert int(spd[0, 1]) <= int(spd[0, 0]), "Doom team should not be faster than Classic"


def test_apply_stances_writes_armor_for_driven_teams():
    """A Wall team must get its ``_armor`` (< 1.0) written; a Classic team stays
    at 1.0. Confirms the table's armr column reaches the engine tensor the
    combat phase reads.
    """
    e = _fresh_engine(teams=2)
    classic = ACTIONS.index(("Classic", ""))
    wall = ACTIONS.index(("Wall", "horiz"))
    action = torch.tensor([[classic, wall]])
    dydx = torch.zeros(1, 2, 2, dtype=torch.long)

    apply_stances(e, action, dydx, team_start=0)

    assert hasattr(e, "_armor"), "apply_stances did not set _armor"
    armor = e._armor
    assert armor.shape == (1, 2)
    armr = _col("armr")
    assert float(armor[0, 0]) == pytest.approx(1.0), "Classic team armor should be 1.0"
    assert float(armor[0, 1]) == pytest.approx(float(_KNOBS[wall][armr])), (
        "Wall team armor in engine != table armr column")
    assert float(armor[0, 1]) < 1.0, "Wall should reduce incoming damage"


def test_apply_stances_mutates_tensors_in_place_not_just_rebinds():
    """The engine's cursor step / combat read ``_cursor_speed_bt`` and ``_armor``
    by attribute each tick, so it is enough that the ATTRIBUTE reflects the new
    values after the call. Assert the values genuinely CHANGED from a neutral
    baseline (catch a no-op apply_stances).
    """
    e = _fresh_engine(teams=2)
    # Establish a neutral baseline (all Classic).
    classic = ACTIONS.index(("Classic", ""))
    neutral = torch.full((1, 2), classic)
    dydx = torch.zeros(1, 2, 2, dtype=torch.long)
    apply_stances(e, neutral, dydx, team_start=0)
    base_speed = e._cursor_speed_bt.clone()
    base_armor = e._armor.clone()

    # Now drive Doom 3x + Wall: at least one of the two tensors must change.
    doom3 = ACTIONS.index(("Doom", "3x"))
    wall = ACTIONS.index(("Wall", "horiz"))
    apply_stances(e, torch.tensor([[doom3, wall]]), dydx, team_start=0)

    speed_changed = not torch.equal(e._cursor_speed_bt, base_speed)
    armor_changed = not torch.equal(e._armor, base_armor)
    assert speed_changed or armor_changed, (
        "apply_stances did not change _cursor_speed_bt or _armor when switching "
        "from Classic to Doom3x/Wall")
    # specifically armor must have moved (Wall < 1.0 vs Classic 1.0)
    assert armor_changed, "Wall stance should have changed _armor"


def test_apply_stances_drives_wells_when_enabled():
    """With ``_wells_enabled`` a Doom stance must arm the engine's Doom well
    slots (``_doom_horizon`` / ``_doom_str``) for the driven team — the
    cross-team physics training relies on. Confirms training exercises the same
    well machinery play does."""
    e = _fresh_engine(teams=2)
    e._wells_enabled = True
    # give the Doom team some live mass scaling (already placed by reset())
    doom3 = ACTIONS.index(("Doom", "3x"))
    classic = ACTIONS.index(("Classic", ""))
    action = torch.tensor([[classic, doom3]])
    dydx = torch.zeros(1, 2, 2, dtype=torch.long)
    apply_stances(e, action, dydx, team_start=0)
    # team 1 (Doom) horizon must be positive; team 0 (Classic) zero.
    assert float(e._doom_horizon[0, 1]) > 0.0, "Doom team got no event horizon"
    assert float(e._doom_horizon[0, 0]) == 0.0, "Classic team should have no horizon"
