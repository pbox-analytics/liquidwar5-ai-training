# Tests

Automated checks for the GPU Liquid War engine and the RL cursor policy. The
project previously had no tests; everything was verified ad hoc. The specialist
panel flagged that train/play physics parity was "assumed, not checked" — which
caused the Doom cursor-speed-tax balance bug (the mobility tax once existed only
in the play server, not the training knob table, so the policy over-valued Doom).
This suite locks that parity down and pins the engine's core invariants.

## Running

CPU-only, no GPU required:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
    uv run --with pytest --with numpy python -m pytest tests/ -q
```

The whole suite runs in well under 30s on CPU (tiny boards, few hundred ticks).

## What is covered

### `test_engine_invariants.py`
A few hundred ticks of random cursor actions at B=3, small board, run under the
matrix {wells off/on} x {`_inertia` 0 / 0.12}:

- **one-per-cell occupancy** — on a spacious board, no two living fighters ever
  share `fy*W + fx` (`test_one_per_cell_strict`); on any board, movement never
  *increases* overlap beyond the reset baseline. (When a wall-heavy map leaves a
  team's strip with fewer passable cells than `fighters_per_team`, the engine
  deliberately parks the overflow on the cursor cell to keep the count exact —
  documented behavior — so the strict zero-overlap assertion uses a board with
  room and the matrix tests assert the "never collapses" guarantee.)
- **conservation** — the total fighter count is the fixed starting army;
- **conversion-only team counts** — per-team census deltas sum to zero each tick
  (a conversion recolours; nothing spawns or vanishes);
- **valid ranges** — `fteam` in `[0, T)`, `fhealth` in `[0, MAX_HEALTH]`;
- **occ agrees with the SoA** — `occ[b, fy, fx]` resolves back to a slot on that
  exact cell, and every non-empty `occ` entry indexes a real slot.

### `test_parity.py` (the critical one)
`rl/policy.py`'s `_KNOBS` + `apply_stances` is the single source of play/train
physics. These assert it stays in sync with the play server:

- every `_KNOBS` row width == `len(_KNOB_NAMES)`;
- `len(ACTIONS) == NUM_STANCES == len(_KNOBS)`;
- the Doom rows' `cspd` column == the server's mobility tax tuple
  `_base_cs * (0.7, 0.45, 0.3)[doom_level-1]` (parsed from `web/server.py` by
  regex — the server is **not** imported, it pulls fastapi + torch);
- the Wall rows' `armr` column == the server's `_e._armor` Wall-brace constant;
- `apply_stances` actually writes `engine._cursor_speed_bt` and `engine._armor`
  (and arms the Doom well slots when `_wells_enabled`).

A future balance change that touches play but not the table (or vice-versa)
**fails here**.

### `test_policy.py`
`CursorPolicy` constructs and forwards the documented shapes; `act()` returns the
documented `(B,T,...)` shapes for several team counts; dead teams are forced to
stay and zeroed; the `stance_mask` / `stance_temp` governor kwargs make a -inf
mask on the Doom actions (16-18) keep argmax **and** temperature-sampling off
Doom; the held-stance move-only credit path works; and the legacy-checkpoint
head-size mapping (`LEGACY_ACTION`, variable `num_stances`) is sane.

## Constraints honoured

- Only files under `tests/` are created. `simulator/engine.py`, `web/server.py`,
  `rl/policy.py`, and `web/static/*` are **read-only** here (another engineer is
  editing the engine).
- `web/server.py` is parsed as **text** (regex on the specific tuple/constant),
  never imported.
