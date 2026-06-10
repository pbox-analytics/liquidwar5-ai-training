# Liquid War — development log

How the GPU-native Liquid War clone got from "the browser version is broken" to
"plays the way real Liquid War should." The engine (`simulator/engine.py`) is the
**single source of truth** — it is *both* the RL training environment and the
playable game served at `web/server.py`. No C-engine bridge, no fidelity gap: you
play the exact engine the policy trains in.

Deploy: `scripts/run-play.sh` → http://192.168.1.133:8099 (pandora-storm, RTX 5090 Laptop, GPU-direct; moved off the RTX PRO 6000 2026-06-09).
Controls: **arrows / WASD** move the cursor, **1–8** hold a stance (Swarm / Spin /
Drill / Wall / Pulse / Doom / Maelstrom / Atom), **Q/E** spin direction, **T** trails.

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
| **swirl** | a tangential bias → the army **spirals into** the cursor on curved, magnetised field-lines and **orbits** it as a churning swarm | `SWIRL_W=8` |
| **edge push** | a wave-modulated *outward* bias → the rim extends pseudopods on each crest then retracts → an **undulating membrane** | `PUSH_W=14` |
| **traveling wave** | *Dictyostelium* cAMP-style restlessness gated by `sin(dist·k − tick·ω)`, dist = **Euclidean** cells to the cursor (octile-gradient phase made the crests angular chevrons; Euclidean rings are round) → idle undulation + the edge ripple | `> 0.0` crest, `0.07` base |
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

## 4. Stances — the tactical system

The blob has **eight held stances** on the number row (hold, not one-shot). Each is
just a *preset of the per-team engine knobs* — the integration surface read in
`_move_fighters` via `getattr`, so adding a stance is a knob preset, never a rewrite.
The AI policy drives the **same** knobs via `rl/policy.apply_stances`.

| key | stance | feel | knobs |
|---|---|---|---|
| **1** | 🐝 Swarm | loose orbiting electron-cloud | `_spin 0.5`, `_burst 0.15` |
| **2** | 🌀 Spin | tight fast vortex (Q/E = direction) | `_spin 1.7`, `_burst -0.4` |
| **3** | ➤ Drill | Ender's-Game piercing column; tap to rev **slow→med→fast** (faster spin grinds harder but advances slower) | `_drill` (aim×advance), `_spin`, `_surge` |
| **4** | 🛡 Wall | dense shield bar across the cursor, perpendicular to the threat | `_wall` (facing) |
| **5** | 💥 Pulse | concentric rings + `6×` damage waves on the crest | `_burst` rhythm, `_surge` |
| **6** | 🕳 Doom | black-hole implosion (see below) | `_burst -6.5`, `_blackhole_*`, `_surge 6` |
| **7** | 🌪 Maelstrom | fast wide orbiting shell / whirlpool | `_spin 2.0`, `_burst 0.6` |
| **8** | ⚛ Atom | figure-8 electron orbitals | `_spin 1.8`, `_fig8 1` |

Knobs: `_spin` (swirl mult, signed) · `_burst` (radial in/out) · `_drill` (thrust
dir + advance speed) · `_wall` (shield facing) · `_surge` (damage mult) · `_fig8`
(figure-8 flag) · `_blackhole_pos/_team/_str/_range` (Doom's well). **Atom's
figure-8**: the swirl orbits two lobe-centres offset ±R in x and counter-rotates
across the cursor's vertical axis, so the halves trace a ∞ instead of a flat spin.

### Doom — the Interstellar black hole

Doom is the finisher, modelled on *Gargantua* — and the **army itself forms the
black hole**. The engine's per-team `_ring` knob holds a target orbit radius:
fighters are biased onto it from both sides (outward needs more weight than the
gradient's 10–14/step inward pull, else stragglers pool and the hole never
opens), so the team becomes a **spinning annulus with an open black centre**.
The radius is **mass-scaled** by the play server (solve π(r_out²−r_in²)=mass for
the band centred on the target with its inner edge at the rendered horizon —
fighters pack one-per-cell, so a fixed small radius would saturate back into a
solid blob). The swirl spins the disk (charge-scaled `_spin` 1.2/1.8/2.4, tap 6:
1x→2x→3x), and fighters far outside the ring flatten onto the cursor's equator
row and stream in along it — the **edge-on blade** of the Interstellar
silhouette. The client adds only the physics of light, no painted geometry:
**gravitational lensing** (the units' own glow bends around the hole), a soft
shadow zone, **amber heat grading** near the horizon (mote shader + composite
re-colour the units and their infall trails — the accretion fire is made of the
army), and a pure-black **event horizon**, all ramping in over ~1.5s and growing
with charge, so 3x *looks* like what it is.

- **Singularity** — `_burst -6.5` (near-zero spin) violently implodes your *own* mass
  to the cursor point.
- **Cross-team gravity well** — `_blackhole_pos` at your cursor drags *enemy* fighters
  in too (a real black hole pulls everyone). Pull is **∝ your mass**
  (`_blackhole_str = 34 · current/initial fighters`) with a **finite reach** (falloff
  `R²/(d²+R²)`, range ~55) so distant/dispersed forces escape — no map-wide vacuum.
  What it swallows meets the `6×` tidal surge.
- **Counter:** strip its mass — Drill the dense core, or stay dispersed and convert
  its small perimeter. Since pull ∝ mass, every fighter you take off it *weakens* the
  well, so it unravels as you fight it; or kite beyond the reach. Net: a finisher for
  a winning army, useless as a comeback button.

(Backlog of more slime-mold moves — Rally, Pheromone Tube, Split — in
`docs/POTENTIAL_FEATURES.md`.)

## 5. Rendering — the WebGL2 mote pipeline (client, `web/static/gl.js`)

The army is drawn as individual **motes** — **a mote is one fighter rendered as a
single particle** — not a grid of lit cells. The server streams a stride-sampled set
of fighter positions (`int16`) + team (`pos_b64`/`pteam_b64`/`pn` in `state()`); the
client animates them. Since v0.02 the renderer is **WebGL2** (`gl.js`; the Canvas2D
original is kept at `index-2d.html` and is the automatic fallback). GL was chosen as
the mobile-ready layer — a phone GPU handles this pipeline easily, mobile Canvas2D
does not.

Per `requestAnimationFrame` (decoupled from the ~45–63fps server):

1. **Army pass** — one instanced draw of 16k **capsule quads**. The vertex shader
   does the **quadratic-arc glide** (3 ring-buffered snapshots prevprev→prev→cur,
   control = prev + `curve`·incoming-velocity) and stretches each capsule along its
   per-tick velocity (the **streak**). Per-mote individuality comes from a static
   random table: size/brightness jitter, ~12% **sparkle** motes lifted toward white,
   and a density-gated **twinkle** so the packed core shimmers like a living mass.
   A CPU 8×8 binning pass uploads local density per mote: the deep core renders
   darker/smaller (volume shading), the sparse rim more translucent — this is what
   killed the "flat paint blob" look. Sparks + conversion flashes join this pass as
   additive point sprites so they bloom. **Premultiplied normal blending, not
   additive** — dense armies stay solid *team colour* (the old additive lesson).
2. **Trail pass** — a persistent framebuffer: decay-multiply, add the army layer.
   Motion history glows and fades with proper colour (toggle **T**, `trail` slider).
3. **Bloom** — army layer downsampled ¼, two separable gaussian passes, composited
   additively (`glow` slider). Real bloom, not the old downscale-upscale trick.
4. **Composite** — one fullscreen pass: vignetted/dithered background, **rim-lit
   beveled walls** (the wall mask ships as a 2-channel texture — crisp + box-blurred;
   the blurred ramp gives the shader a wide gradient for the bevel and the soft
   contact shadow that grounds the slabs), then trail + army + bloom, finished with
   a soft filmic clip (`1-exp(-1.8x)`).
5. **Cursors** — ring/disc point sprites drawn crisp on top (halo + pulsate).

Everything renders at **native device pixels** (CSS size × devicePixelRatio, capped
2×) — no more grid-resolution buffer with pixelated upscale.

### Live Visuals sliders

A **Visuals** row under the board exposes the render dials as live sliders (no
redeploy): **streak** (tail length, 0 = dots) · **curve** (arc bend) · **size** (mote
thickness) · **trail** (fade length) · **opacity**. They write `pTail/pCurve/pWidth/
pTrail/pAlpha`, which `frame()` reads each tick — find a feel, then bake the numbers in.

> The dense **core looks static** because it physically is: packed one-per-cell, the
> swirl has no empty cell to flow into, so only the rim moves. Letting a packed core
> rotate in place is an *engine* (move-resolution) change, not a render dial.

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
  `MOM`, `SWIRL_W=8`, `PUSH_W`, wave/jitter, `cursor_speed`, `unit_speed`).
- **Stance knobs** — the per-stance presets in `web/server.py`'s game loop
  (`_spin/_burst/_drill/_wall/_surge/_fig8`, Doom's `_blackhole_*`); the AI mirror is
  `rl/policy.apply_stances`.
- **Play size / rate** — `web/server.py` defaults: grid **384×576** (doubled again for
  room; ~40–50fps at this size vs a policy opponent), **8000** fighters/team, **60Hz**
  (deploy over-targets 63). Env-overridable: `LW_PLAY_H/W/FIGHTERS`, `LW_TICK_HZ`.
- **Visual feel** — the in-game **Visuals** sliders (live: streak/curve/size/trail/
  opacity/glow), or the `pTail/pCurve/pWidth/pTrail/pAlpha/pGlow` defaults in
  `web/static/index.html`; deeper look constants (twinkle, density shading, wall
  bevel, tonemap) live in the shaders in `web/static/gl.js`.
- **Combat / Pulse** — constants in the engine / `web/server.py`.

## 9. Next

- **Play test** the 100+ maps — find archetypes/params that play badly, tune.
- **Retrain the policy** on the corrected engine + maps — the original project goal;
  the current opponent is the heuristic, and old checkpoints learned in the broken
  world (see the engine module docstring).
- `docs/POTENTIAL_FEATURES.md` — slime-mold special moves + fluid-clash backlog.
