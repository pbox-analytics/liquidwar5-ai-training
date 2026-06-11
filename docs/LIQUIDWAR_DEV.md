# Liquid War — development log

How the GPU-native Liquid War clone got from "the browser version is broken" to
"plays the way real Liquid War should." The engine (`simulator/engine.py`) is the
**single source of truth** — it is *both* the RL training environment and the
playable game served at `web/server.py`. No C-engine bridge, no fidelity gap: you
play the exact engine the policy trains in.

Deploy: `scripts/run-play.sh` → http://192.168.1.133:8099 (pandora-storm, RTX 5090 Laptop, GPU-direct; moved off the RTX PRO 6000 2026-06-09 — the PRO 6000 now trains the opponent, §13).
Controls: **arrows / WASD** move the cursor (or hold the mouse), **1–9** hold a
stance (Swarm / Spin / Drill / Wall / Pulse / Doom / Maelstrom / Atom / Classic;
re-tap to cycle a stance's modes, §12), **Q/E** spin direction, **T** trails,
**F** fullscreen, gamepad supported (§15). LAN multiplayer: open the same
`/?room=<name>` link on two machines, or use the 🔗 invite button (§14).

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
| **2** | 🌀 Spin | 3 forms, tap to cycle (Q/E = direction): **vortex** (tight fast spin), **sawblade** (dense disc + 8 rotating teeth — `_node_m 8`, `_node_w` sweeps the pattern), **galaxy** (wide slow swirl, 3 spiral arms — `_node_k` winds them with radius) | `_spin`, `_burst`; `_node_m/k/w` |
| **3** | ➤ Drill | Ender's-Game piercing column that **corkscrews** — the lateral squeeze targets a traveling-sine centreline (the 2D projection of a rotating bit; twist follows Q/E); tap to rev **slow→med→fast** | `_drill` (aim×advance), `_spin`, `_surge` |
| **4** | 🛡 Wall | DENSE shield column across the cursor (collapse 20 + burst −0.9 — solid, not a picket line); tap to flip horizontal/vertical | `_wall` (facing), `_burst` |
| **5** | 💥 Pulse | 3 modes, tap to cycle: **wave** (traveling rings + `4×` crest damage), **rings** (cymatic standing rings — Chladni circular mode, `_node_l`), **star** (6-petal nodal-diameter figure, `_node_m`) | `_burst`/`_surge`; `_node_l`, `_node_m` |
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
opens), so the team becomes a **spinning OBLATE disk with an open black centre**
— the target radius is angle-dependent (pinched vertically, stretched ~1.3× along
the equator), and the rim ripple is damped for ring teams so the disk stays solid.
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
  a winning army, useless as a comeback button. (Rebalanced 2026-06-10 — full-map
  range made it inescapable, i.e. an auto-win: range is now ~2.2× the disk radius,
  the devour horizon ~0.9× the mass radius (was 1.5×), capture 0.12/tick (was 0.18),
  and the tidal surge scales 4/5/6× with charge instead of a flat 6×.)

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

*(2026-06-10: superseded in part by §10 — the engine eventually DID become the
cap at the 384×576/8000 scale, and the fix was CUDA-graphing the whole tick.)*

Profiling beat guessing — at the original scale the engine was never the cap (8ms of a 16.7ms budget):

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

## 10. The CUDA-graph tick (2026-06-10)

At 384×576 with 8000 fighters/team the tick had crept to ~29ms — half the 60fps
budget gone before serving a frame. Profiling showed the engine is
**kernel-launch bound at B=1**: ~3,800 tiny CUDA kernels per tick, only ~11.5ms
of actual GPU work; the rest was launch overhead and hidden GPU→CPU syncs.
The fix, in order of payoff:

1. **Strip every `.any()`/`.item()` sync from the hot path** (priority rounds,
   rotation fixpoint, combat, capture). Each guard cost a blocking sync per
   sub-step; the no-op kernels they "saved" are cheaper. The rotation fixpoint
   became a fixed-count shrink + a branchless on-GPU select.
2. **Cheaper algorithms where they were free**: priority rounds 8→4 (a blocked
   fighter retries next sub-step anyway), the gradient relax as 3 pooling ops
   per sweep instead of 8 shifted-slice minimums (bit-identical field, float32
   carries exact integers), B=1 cursor stepping in plain python over a CPU
   walls mirror (~180 launches/tick gone, verified bit-identical).
3. **CUDA-graph capture of the whole tick** (`_graph_step`): record the kernel
   sequence once per game (at tick 70, after the cold flood converges), then
   replay it as a single unit. Engine: 20 → **11.2ms — the hardware floor**.

The capturable-tick contract (anything inside `_step_body` must honor it):
no data-dependent python control flow or CPU syncs; every cross-tick tensor
round-trips through persistent buffers (a replay reads inputs at fixed
addresses); time-varying scalars are GPU tensors (`_tick_f` — a python
`self.tick` would bake into the graph as a constant); per-team effect dials are
**slot tensors written in-place** (zero = off, so the kernel sequence is
static). Kill switch: `LW_CUDA_GRAPH=0`. Training (B>1) keeps the eager path.
Caveat discovered along the way: the engine is inherently nondeterministic on
CUDA (argsort ties), so refactors are validated by invariants (one-per-cell,
conservation, health bounds), never by same-seed replay.

With the policy inference cached every 2nd tick, the live server holds
**60+fps**, and ~58fps with two players in a room.

## 11. Cross-team wells — Doom & Maelstrom as real weapons

Doom's gravity well and Maelstrom's whirlpool current are the only effects that
act on the *enemy*. Both live in per-team **slot tensors**
(`_doom_*` / `_vortex_*`, one slot per team, in-place writes, str==0 = off) so
any number of wielders coexist — dueling Dooms included — and the CUDA graph
stays valid. Balance history (all server-tunable dials in the stance blocks):

- **Maelstrom** = Doom rotated 90°: tangential entrainment (enemies near the
  well are swept into orbit through your storm-cloud), undertow/ejecta/shear
  radial modes. Nerfed from a cross-arena tractor beam to a local hazard:
  squared falloff (25% at R), str `22·√frac`, reach 1.5× blob radius.
- **Doom** can no longer be an unkillable last stand: capture rate scales with
  the wielder's mass (`0.12·√frac`) and the horizon floor dropped 14 → the
  rendered hole radius, so a whittled army's well stops out-eating the blob
  consuming it.
- **Parity**: the AI casts the same wells with the same dials — before
  2026-06-10 its Doom was a cosmetic self-collapse and the human duelled with
  superpowers the opponent lacked.

## 12. The mode system — every stance got a re-tap

Re-tapping a held stance's key cycles its modes (the HUD pill shows the mode):

| key | stance | modes |
|-----|--------|-------|
| 1 | Swarm | cloud → **comet** (drill machinery aimed along your recent cursor motion) |
| 2 | Spin | vortex → sawblade → galaxy |
| 3 | Drill | slow → med → fast |
| 4 | Wall | horizontal → vertical |
| 5 | Pulse | wave → rings → star → **lattice** (superposed Chladni modes) → **nova** (~2.3s charge then detonation, 5× surge) → **tide** (directional traveling crests — new engine `_tide` bias) |
| 6 | Doom | 1x → 2x → 3x charge |
| 7 | Maelstrom | undertow → ejecta → shear |
| 8 | Atom | orbital → **binary star** (lobes on a rotating axis, co-rotating with cohesion: two discs orbiting their barycenter) |
| 9 | Classic | — (no knobs at all: the original gradient-following blob) |

Maelstrom renders a whirlpool refraction shader (rotational ray-bending that
composes with Doom's lensing); the AI's casts are visible via `ai_doom` /
`ai_mael` HUD markers.

## 13. Training the opponent — what we're doing now, and why

**The goal:** an opponent that earns its difficulty — it should know every
weapon the human has, use them when they pay, and know how to fight against
them. Two structural problems blocked that, and 2026-06-10 removed both:

1. **Crash fragility.** Earlier runs died and lost everything (no resume, no
   optimizer checkpointing, numbered saves every 50 updates). Now:
   `--resume <run-dir>` continues a run in place — `latest.pt` + `optim.pt` +
   `meta.json` refresh every 10 updates (atomic tmp+rename), so the k8s pod can
   crash or be re-deployed and the run continues within 10 updates. The same
   mechanism doubles as a **mid-run hyperparameter swap**: push a chart change,
   the pod restarts, training resumes with the new flag.
2. **The action-space gap.** The policy used to pick from 8 base stances while
   humans played 25 stance-mode combinations, and its Doom/Maelstrom were
   training-time no-ops. Now the policy's stance head is the **flat 25-action
   space** (`rl/policy.py:ACTIONS` — every stance×mode pair, including Doom
   charge levels and wall orientation), `apply_stances` maps actions through
   per-action knob tables carrying the *exact* play-server dials, and
   `--wells` casts the real cross-team physics in training. Self-play means it
   learns to use *and* defend against everything simultaneously. Legacy
   8-action checkpoints still load in play via `LEGACY_ACTION` mapping.

**The pipeline** (gitops, ArgoCD watches `main`):
`gitops-containerize` branch (working code) → kaniko in-cluster build
(`k8s/kaniko-build-job-gpu.yaml`, bump job name + `TRAINING_REF` + tag) →
`charts/liquidwar-gpu-trainer` values on `main` (image tag, `replicaCount: 1`,
`config.*` hyperparameters) → trainer Deployment on **pandoras-box's RTX PRO
6000** (96 GB; freed by parking `comfyui-pbox-{0,1,2}` via project-homer's
`gpus.yaml` mode toggle — flip replicas, run `scripts/regen-gpu-values.py`,
push both repos). Checkpoints land on the pbox NFS
(`/mnt/dlred1/datalake/liquidwar/results/rl/<run>/`); promote a winning
`best.pt` to `/tmp/lwgood/rl/best/policy.pt` on pstorm to upgrade the live
opponent. **Batch ≤128**: 256 OOMs even 96 GB (ppo_update materializes
whole-minibatch egocentric views — a single 28.7 GiB alloc in backward).

**Current run** (`wells-090`): 5000 updates, batch 128, teams 2, `--wells`,
flat 25-action head. Watch the `[eval] win-rate vs heuristic` lines —
`best.pt` is selected by that, not by self-play return. Known issue being
tuned: with 25 actions the default entropy bonus (`--ent-coef 0.01`) pinned
the joint entropy near uniform (~5.3 of max 5.42) for 1000+ updates with
whipsawing evals; dropped to 0.003 mid-run (via the resume mechanism) to let
the policy commit. If it still won't sharpen, anneal further (0.001).

## 14. Multiplayer rooms & the network protocol

`/?room=<name>` shares one game: the first client creates the room (their
mode/opponent/teams win), later clients take the next free seat, leavers hand
their seat back to the AI mid-game. One server game loop per room; every seat
holds stances/modes independently (`_apply_player_stance`), including per-seat
Doom cursor slowdown. The 🔗 invite button mints a room and copies the join
link; toolbar chips show human seats ("you" / "P2").

The wire (designed so a phone on weak WiFi is a fine guest):

- **Binary mote channel**: `u8 type | u8 hasGrid | u16 pn | u32 tick | pos |
  teams | grid?`. Positions are int16 **keyframes** every 30 ticks and int8
  **deltas** between (fighters move ≤6 cells/tick — always fits a byte).
  Delta frames are ~48KB vs ~110KB of the old base64-in-JSON; decode is a
  typed-array add loop, no `atob`/JSON on the hot path.
- **The cell grid ships at ~5Hz** (plus game start/end) inside the blob — the
  clients only need it for the wall mask and cosmetic contact sparks; motes
  carry the motion (the renderer interpolates between frames).
- **HUD JSON per seat** (small): cursors, counts, seats, your stance pill.
- **Latest-frame-wins send queues**: the room loop never awaits a client
  send — each player has a 1-deep queue + sender task, so a slow consumer
  drops stale frames instead of throttling the room (verified: a 6fps-reader
  guest left the room loop at 63fps and a fast peer at full rate).

Headroom if internet play ever matters: permessage-deflate (the grid is highly
compressible), 30Hz frames for remote clients, client-side cursor echo.

## 15. Big screen & PWA

**F** (or ⛶) sends the canvas fullscreen — chrome disappears, works on any
TV browser. Gamepad: stick/d-pad moves, LB/RB cycle the held stance, A re-taps
(cycles its mode), X/B set spin, Y trails, Start = new game. The page is an
installable PWA (`manifest.json` + icons + a minimal service worker) — the
install prompt needs HTTPS, e.g. `tailscale serve` on pstorm.

## 16. Next

- **Watch `wells-090`** converge (entropy should fall, evals stabilize); promote
  its `best.pt` to `/tmp/lwgood` when it beats the current opponent in play.
  Consider auto-promotion with an eval gate.
- **League training** — anchor against 2-3 frozen past checkpoints, not just the
  heuristic, to punish one-trick strategies (~20 lines in `collect_rollout`).
- **Internet play** — deflate + 30Hz remote frames + cursor echo (see §14).
- **Telemetry** — the trainer's Kafka publishing is wired but off; the cluster
  has Kafka + Grafana. A `/metrics` endpoint on the play server would close the
  loop.
- `docs/POTENTIAL_FEATURES.md` — slime-mold special moves + fluid-clash backlog.
