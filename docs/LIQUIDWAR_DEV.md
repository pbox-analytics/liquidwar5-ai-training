# Liquid War — development log

How the GPU-native Liquid War clone got from "the browser version is broken" to
"plays the way real Liquid War should." The engine (`simulator/engine.py`) is the
**single source of truth** — it is *both* the RL training environment and the
playable game served at `web/server.py`. No C-engine bridge, no fidelity gap: you
play the exact engine the policy trains in.

Deploy: `scripts/run-play.sh` → http://192.168.1.226:8099 (RTX PRO 6000, GPU-direct).
Controls: **arrows / WASD** move the cursor, **SPACE** = Pulse, **T** = toggle trails.

---

## 1. The core fix — real pathfinding

The browser game looked broken because the gradient was a *capped/aged* spread,
not a flood-fill: armies just converged into blobs near the cursor and stalled.

- **Complete flood-fill gradient** — a per-team geodesic distance field, seeded at
  the cursor and relaxed to convergence each tick (not capped). The army pathfinds
  the *whole* map.
- **Octile distance** (orthogonal 10, diagonal 14 ≈ √2·10), not Chebyshev — so the
  blob is **round**, not an angular square.

## 2. Movement — a layered "living organism" model

Each fighter, each tick, scores its 8 neighbour cells and moves to the best open
one. The score is a sum of forces — that layering is what makes it feel alive.
All weights are constants near the top of `_move_fighters`:

| force | what it does | knob |
|---|---|---|
| **gradient** | pulls down-field toward the cursor (the base attraction) | (octile cost) |
| **momentum / inertia** | a per-fighter velocity blended with the gradient → weight, overshoot, banking; head-on masses collide | `VEL_W=8`, `MOM=0.88` |
| **swirl** | a tangential bias → the army **spirals into** the cursor on curved, magnetised field-lines and **orbits** it as a churning swarm | `SWIRL_W=11` |
| **edge push** | a wave-modulated *outward* bias → the rim extends pseudopods on each crest then retracts → an **undulating membrane** | `PUSH_W=14` |
| **traveling wave** | *Dictyostelium* cAMP-style restlessness gated by `sin(dist·k − tick·ω)` → idle undulation + the edge ripple | `> 0.0` crest, `0.07` base |
| **jitter** | small per-fighter noise → independent-looking units, no lockstep | `randint(0,7)` |

Units advance at **`unit_speed`** cells/tick (grid-scaled, matches the cursor) so
the army keeps pace instead of crawling behind — each tick sub-steps move+combat.

## 3. Combat — convert, don't delete

- **Conversion, not deletion** — combat drains a target's health; when it goes
  negative the fighter **defects to the attacker's team**. Total count is invariant
  (verified every change: 16000 fighters in, 16000 out).
- **Directional** — a back-attack (defender facing away) lands full `ATTACK` and the
  attacker **overtakes & spreads**; a defended head-on clash lands `SIDE_ATTACK` and
  grinds. So flanking a committed army snowballs.
- **One-per-cell** `occ` grid — fighters pack, never overlap/collapse.

## 4. Special move — Pulse (SPACE)

A peristaltic surge: the human team deals `6×` on contact for ~0.3s, ~3s cooldown
(`PULSE_DUR/CD/MULT` in `web/server.py`). With momentum it's a burst you slam into
the line. (Backlog of more slime-mold moves — Rally, Pheromone Tube, Split — in
`docs/POTENTIAL_FEATURES.md`.)

## 5. Rendering / FX (client-side, `web/static/index.html`)

- **Glowing cursors** — halo + pulsate, player bigger, flares on Pulse.
- **Motion trails** (the "shader") — each frame fades instead of clearing, so the
  swirl reads as **flow-streaks** and the rim as a **shimmering undulating edge**.
  `T` toggles; the fade alpha sets trail length.
- **Collision particles** — real sparks spawn at the army-vs-army contact line,
  **fly out, arc under gravity, slow, and fade** — white-hot at birth, cooling to
  each team's colour. They stack bright where the fighting is fiercest and streak
  with the trails.

## 6. Hitting 60fps

Profiling beat guessing — the engine was never the cap (8ms of a 16.7ms budget):

1. The loop slept a *full* `dt` *after* the work → fixed to sleep the remainder. 35→48.
2. The per-frame sleep overshot ~1ms with no correction → **absolute-deadline
   scheduler** (claws back drift). 48→58.
3. `asyncio.sleep` can't beat the event loop's ~1ms granularity → **over-target the
   tick to 63Hz** so the overshoot lands on a true 60. 58→**60**.

HUD metrics (counts/flood/spread) each force a GPU→CPU sync, so they're cached at
~10Hz; render-essential fields stay per-frame.

## 7. Maps — procedural generator

`_generate_random_maps` draws a random **archetype** with randomized parameters
each game — *open arena, central barrier (gapped), pillars, scattered blocks, four
rooms, walled corners* — always **point-symmetric** (180° rotation → fair) and
**connectivity-checked** (4-conn flood ≥85%, redraw if a walled-off pocket). Verified
**100/100 connected, 76+ distinct** per 100 draws → far more than 100 maps; every
game is fresh. The flood-fill routes the army around whatever walls appear, so the
flow goes organic for free and chokepoints/flanking become real strategy. The cursor
steps cell-by-cell so it can't slide through a barrier.

## 8. Settings (where they live)

- **Movement feel** — constants in `_move_fighters` / `_move_cursors` (`VEL_W`,
  `MOM`, `SWIRL_W`, `PUSH_W`, wave/jitter, `cursor_speed`, `unit_speed`).
- **Play size / rate** — `web/server.py` defaults: grid **192×288**, **8000**
  fighters/team, **60Hz** (deploy over-targets 63). Env-overridable: `LW_PLAY_H/W/
  FIGHTERS`, `LW_TICK_HZ`.
- **Combat / Pulse** — constants in the engine / `web/server.py`.

## 9. Next

- **Play test** the 100+ maps — find archetypes/params that play badly, tune.
- **Retrain the policy** on the corrected engine + maps — the original project goal;
  the current opponent is the heuristic, and old checkpoints learned in the broken
  world (see the engine module docstring).
- `docs/POTENTIAL_FEATURES.md` — slime-mold special moves + fluid-clash backlog.
