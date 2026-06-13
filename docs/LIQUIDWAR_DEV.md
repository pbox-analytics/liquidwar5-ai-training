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
  radial modes. Local hazard, not a tractor beam: squared falloff (25% at R),
  str `28·√frac`, reach 1.5× blob radius. Its bite is the **spinning rim**,
  not the current: `_surge 1.7` (§27) — raw current strength only feeds the
  enemy well, so impact came from rim damage.
- **Doom** can no longer be an unkillable last stand OR an inescapable tractor
  beam: capture rate scales with wielder mass (`0.09·√frac`), the horizon is
  the rendered hole radius, **squared falloff**, range floor **56**, and a
  per-level strength ladder `24/40/52` that concentrates power near the well
  (§26). A committed kite escapes; a stationary blob still bleeds ~31%.
- **Parity**: the AI casts the same wells with the same dials — before
  2026-06-10 its Doom was a cosmetic self-collapse and the human duelled with
  superpowers the opponent lacked.
- **Counterplay** (`well_shield` + the reel in the engine): your own active
  **Maelstrom is the counter-current** — your fighters shed up to 85% of enemy
  well forces *and* the event-horizon devour (linear falloff sized to cover
  the whole storm shell, not just the eye), and the storm **reels back its own
  strays** beyond ~0.8R, so a Doom must out-muscle your current for every
  fighter it strips. A held **Wall braces** for a further 45%. Measured:
  vs a 2x Doom, undefended loses ~25% of its army; a Maelstrom defender wins
  the engagement outright. Same physics in training, so the policy can learn
  the matchup.

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
composes with Doom's lensing) PLUS a per-mode look (§27 — undertow/ejecta/shear
read at a glance); the AI's casts are visible via `ai_doom` / `ai_mael` HUD
markers (the mael marker carries its mode). The `_KNOBS` table that drives all
this (`rl/policy.py`) is the single source of play/train parity — 20 columns
as of 2026-06-12 (`...mon, cspd, armr`: Maelstrom-on, Doom mobility tax, Wall
armor).

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

## 17. Lobby v2 — gather, pick, start (2026-06-11)

Named rooms now open in a LOBBY phase instead of dropping joiners into a
running game. The card floats over a live all-AI battle (free attract mode).

- **Phase machine** (`Room.phase`): `lobby` -> (host `{start:true}`) ->
  `_start_match()` -> `play`. Result card's 🏠 Lobby button (`{lobby:true}`)
  goes back. Solo `~solo-*` rooms are always `play` (instant game, as ever).
- **Colors are cosmetic and travel with the player**: `Player.color` is a
  palette index (0-5), picked in the lobby (`{color:n}`, conflicts rejected
  server-side), persisted client-side as `lw-color` and passed at connect.
  `st.colors` broadcasts the per-TEAM palette map; the client rewrites
  COLORS/RGB in place and calls `R.setPalette` — every shader/fx site keys
  off team index and just follows.
- **Host dial** (`{ai:n}`, host-only): how many AI seats join at START.
  `_start_match()` rebuilds the GameSession when `humans + ai != engine.T`
  (seats compact to 0..n-1, colors/wins follow, host crown remaps).
- **Live identity**: `{name:"..."}` renames mid-lobby; `{loadout:[a,b,c]}`
  shows each player's 3-stance kit on the lobby card.
- **Rejoin grace**: a name that left <2 min ago gets its seat+color back, no
  fair-join restart (`Room.recent`).
- Lobby seats up to 6 regardless of the attract engine's T; START sizes the
  real match. `/rooms` now reports `phase` so the browser can say "in lobby".

## 18. AI guardrails at the play server (2026-06-11)

Born of the perma-Doom-3x checkpoint (see §19), kept as policy-agnostic
safety nets (all env-tunable, `web/server.py:_ai_dydx`):

- `LW_AI_DOOM_BUDGET` (600 ticks) / `LW_AI_DOOM_COOLDOWN` (360): a seat may
  hold Doom that long, then actions 16-18 are masked -inf and argmax falls to
  its best non-Doom plan. Legacy small-head checkpoints get a post-act
  Classic override instead. 0 disables.
- `LW_AI_STANCE_TEMP` (0): >0 samples the stance head at that temperature
  once per ~45 ticks and holds (training's K_HOLD cadence) — shows the true
  mixture instead of the argmax mode. Inert on a collapsed head.
- **Opponent rotation**: `opponent=latest` re-rolls per match — 65% the live
  champion, 35% a random `rl/archive/*.pt` generation. Three eras play three
  different games; the HUD opponent field names the one you got.
- **Doom mobility tax is in the knob table now** (`_KNOBS` `cspd` column):
  `apply_stances` writes per-seat speeds in play AND `_cursor_speed_bt`
  (batched) in training — the tax finally exists in the trainer's world.

## 19. Training postmortem — why the AI never left Doom (2026-06-11)

A DS+MLE+playtester panel measured the deployed policy: stance head is a
delta function on Doom-3x (P=1.0000, entropy 0.0 nats; T=2.0 sampling still
picks it 240/240). Root causes, all verified against code + live logs:

1. **Dead-air rollouts**: engine `done` is latched, the terminal +1/-1
   re-applied every tick, `_reset_done_games` a no-op below all-done — ~99%
   of buffer transitions were post-victory idle. ret saturated at 0.993
   while eval skill fell 0.938 -> 0.354 (wells-100 was actively degrading).
2. **Stance gradient starvation**: K_HOLD masking puts stance terms on ~2.2%
   of samples; ent-coef 0.001 averaged over ALL samples = effective 2.2e-5.
   The logged `ent` is move-head dominated — collapse was invisible.
3. **Warm-start fossil**: wells-080/090 lineage learned Doom under pre-nerf
   physics; the prior survived every warm start. Policy now: any run across
   a balance change resets the stance head (`--reset-stance-head`).
4. **Nothing punished Doom**: anchored opponents had wells ZEROED (Doom ate
   free food 1/3 of games); eval never set `_wells_enabled` and best.pt
   gated on a wells-blind 48-game eval frozen at a lucky 1.000 since ~1899.
5. **Horizon mismatch**: (gamma·lam)^45 = 0.06 — only instant-payoff stances
   (Doom conversions) fit inside the credit window; nova can't even finish a
   charge in one hold.
6. **Train/play gaps**: play shows argmax-mode every 2 ticks vs training's
   sampled 45-tick holds; the Doom cursor tax existed only in play (fixed,
   see §18).

The wells-110 program (rl/ changes landed 2026-06-11): newly-done-only
terminal rewards + finished-game exclusion + batch reset >30% done;
`--stance-ent-coef` (decision-normalized); `--reset-stance-head` +
`--stance-eps` mixture exploration; eval vs a counter-capable pool (neutral /
scripted-Maelstrom / scripted-Doom) with wells ON, best gated on pool mean,
resume decays the carried best; anchored thirds now script Maelstrom/Doom
opponents so both "Doom loses into Maelstrom" and "Maelstrom defends" appear
in the data; stance telemetry (per-update histogram, separate entropies,
frac_deadair) so collapse can never hide again.

## 20. Perf review adoption (2026-06-11)

Live measurements (panel): clean solo big board 47-56fps eager; the fps-6
outages were co-tenant CPU starvation (host load 48). Landed:

- run-play.sh: `--cap-add=SYS_NICE --cpu-shares=4096 --cpuset-cpus=0-5`,
  uvicorn `--ws-ping-*` (reaps half-open phones in ~40s); server.py:
  `torch.set_num_threads(2)` + `os.nice(-10)`. Batch jobs on pstorm should
  run `nice -n 19`, ideally `taskset -c 6-23`.
- frame_blob: send-counter cadence (`_fseq`) + frozen-tick cached-blob
  early-out (countdown/result-hold frames cost ~0 GPU and ~0 bytes new);
  state() HUD block gated on the tick ADVANCING. The wire seq also lets the
  client drop mis-based deltas after a queue drop (no more mote smear).
- Eager sweep cap after the cold flood (`_fixed_sweeps = cursor_speed+4` at
  tick>=80): -516 launches and -12 sync stalls/tick, same fixed count the
  graph used.
- Client: walls texture re-uploads only when the mask changes (was 3-6ms at
  5Hz); trails no longer wiped per grid frame; HUD DOM writes gated to ~10Hz
  + #teams rebuilds only on content change.
- Still queued (medium/large): dead-peer reaping + attract throttle, the
  _move_fighters launch diet (5610 of 6761 launches/tick), grid RLE + team
  diffs (~32 -> ~16 Mbit/s), engine-pool path back to CUDA graphs, batched
  same-geometry rooms.

## 21. Anti-camp escort (2026-06-12)

The small-map lineage pins its cursor into the play board's corner (measured
93% occupancy of the corner CELL at 384x576 — OOD scale, the move argmax
degenerates to a constant heading; the curriculum that should have fixed it
trained on dead air, §19). Guard (`web/server.py` `_ai_dydx`):
`LW_AI_CAMP_BUDGET` (120 ticks in a 12% corner box) hands the seat to the
heuristic coach for `LW_AI_CAMP_ESCORT` (900 ticks), then the policy resumes.
Measured corner time 0.93 -> 0.23. Stances stay policy-chosen (the Doom
governor composes on top). 0 disables. Joins the §18 guardrail family —
all band-aids on the wells-100 fossil until wells-110 lands.

## 22. The synthwave score (2026-06-12)

`why did we pick an organ key to play when the units fight?` -> the whole
music half of `audio.js` was re-voiced from "Pompeii orchestral" to
**deep-space synthwave at 112 BPM** (`more electronic` / `upbeat with a good
beat`). The adaptive ARCHITECTURE is unchanged — one Phrygian progression,
two voicings crossfaded on combat intensity, stance signatures, the
win/lose cadences, all the SFX — only the instruments changed:

- PEACE voice (was pipe organ): warm analog pad — detuned saw pair + sub-
  octave square per chord tone through a slow-wandering shared lowpass
  (0.08Hz, 700-1400Hz) + two-tap chorus.
- WAR voice (was strings): a 5-saw SUPERSAW wall, lowpass opening
  900->4500Hz with intensity, SIDECHAIN-PUMPED by every kick.
- DRUM MACHINE on the grid: pitch-drop kick (four-on-the-floor), noise
  snare on the backbeat, 7kHz hats opening to shimmered 16ths. `taiko()`
  itself is byte-identical — countdown / nova / cadence still use it.
- New mono BASSLINE (eighth-note root, kick-pumped), arpeggiator ostinato
  (resonant filter env), PWM portamento lead, tanh-warmed sub for phones.
- GROOVE FLOOR 0.32 (`drive = max(intensity, doom?0.3, 0.32)`): the kit /
  bass / arp run four-on-the-floor even at peace so a beat ALWAYS plays;
  snare / 16th-hats / theme / double-time chords still gate higher, and the
  pad->supersaw crossfade tracks RAW intensity so calm still sounds calm.

The honest limit: this is composed BLIND (the model can't hear the output).
Every "the music sounds off" has been the only feedback loop. The planned
next step is generated stems — see §28.

## 23. Cursor visibility & kinetic feel (2026-06-12)

`cursors get lost in the fight` / `make the cursor look cooler` / `make
switching stance energetic` — a batch of game-feel fixes, client + server:

- CURSOR CONTRAST (`gl.js` drawFx `over` path + `index.html`): a
  REVERSE-SUBTRACT pass draws dark shadow pucks + a contrast edge UNDER
  bigger, brighter rings — a colored ring inside a same-colored bloomed
  blob was invisible. (Subtractive, not premult-over: black is a no-op in
  additive, and over-blend white-screened on the SwiftShader test backend.)
- SONAR PING: your cursor emits an expanding ring every ~1.8s — motion
  pulls the eye back even when contrast can't.
- COMET TAIL: per-team cursor path history (`curTrail[]`) burns behind a
  moving cursor in team colour (white-hot core on yours) — a move order is
  visible at a glance.
- STANCE MORPH (server `_apply_player_stance(morph=)` + client shockwave):
  any stance OR mode change kicks a 16-tick flare-open -> whip-spin ->
  snap-into-form transient, plus a client particle burst + shockwave ring
  from the cursor. A formation change is now an EVENT.
- FOLLOW-LUNGE + COHESION (server): a moving cursor tightens `_burst` so the
  swarm CHASES your hand and settles when you stop; loose/neutral forms get
  a constant -0.18 inward bias so the army clusters near the cursor without
  deforming engineered silhouettes (Wall / Maelstrom / Atom).
- CLASH IMPACT (client): armies meeting OUT OF CALM (conv>=8 while the combat
  EMA is low, edge-triggered, 4s cooldown) fire a white/amber shockwave +
  `taiko` at the contact centroid.
- NaN HARDENING (`gl.js`): safe-normalize in the cilia mix + an isnan scrub
  in the composite + finite-guards on every well uniform — one poisoned
  value used to white the whole RGBA8 canvas for a frame.

## 24. Heading inertia — momentum in the swarm (2026-06-12)

`do we still consider momentum and inertia for the swarm attacking?` — we
didn't: fighters were memoryless gradient-descenders (the `fvy/fvx` EMA only
fed combat pierce, not steering), so attacks turned on a dime. Added
(`simulator/engine.py`): a persistent per-fighter `_fdir` (0-7 heading,
8 = stationary) and a `_mom_tab` (9x8) alignment bias added to the candidate
score — same heading +6*mom, 45deg +3, reversal -4, in gradient-field units,
`_inertia` knob. Graph-safe (one gather + one sub/sub-step, `_fdir.copy_`
in place, no syncs). DEFAULT 0.12, NOT the agent's 0.35: live A/B showed
0.35 snowballed routs (brawls collapsed in ~220 ticks — the 5-second-match
regression); 0.12 keeps the weight while fronts hold (>1500 ticks). Verified:
invariants hold over 1500 ticks at 0 and 0.12; bit-identical to HEAD at 0;
turning fraction down ~17%; no measurable perf cost. Mirrored in training
via the trainer-image ref bump (play/train physics parity).

## 25. The balance-sweep method (2026-06-12)

Balance complaints ("Doom inescapable", "Maelstrom gets overwhelmed") are now
settled EMPIRICALLY, not by intuition. Pattern: a `/tmp/lw_*_testbed.py`
script builds a controlled 1v1 on an open arena (full-mass attacker vs a
held stance, scripted kite / charge / counter scenarios), reports kept-mass
and geometry after N ticks; a Workflow fans 4-5 parameter configs across
parallel CPU sims, and a judge agent picks the config meeting the design
bars. Run from the repo root with `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv
run --with numpy python /tmp/lw_*_testbed.py <args>` (script files get /tmp
as sys.path[0], hence PYTHONPATH). CAVEAT learned: a judge agent went rogue
and re-ran experiments instead of ruling — read the workflow `journal.jsonl`
and judge the raw numbers yourself if the verdict stalls.

## 26. Doom kiting rebalance (2026-06-12)

`units cannot escape it when trying to kite` + `bigger penalty for 2x/3x`.
The 5-config sweep (§25) found the culprit was NOT the falloff shape but the
**range floor of 70 cells** — it guaranteed a grip zone where escape was
0-7% in every rfloor=70 variant. Shipped (engine `_doom_fall_sq` default ON
+ server/policy dials):

- SQUARED falloff (was flat) — crosses the ~10/cell escape gradient at ~1.3R
  instead of holding pull to ~2R.
- Range floor 70 -> **56** — THE fix: converts "survives in the grip" into
  33-69% clean escapes for a committed runner.
- Strength LADDER 24/48/72 -> **24/40/52** — charge concentrates power NEAR
  the well, not as extra reach.
- Steeper cursor tax: 2x 0.5->0.45, 3x 0.35->0.3.
- Offense intact: a stationary engaged blob still loses ~31% point-blank
  (69% kept vs 68% baseline) — Doom stays a close-range finisher.

## 27. Stance impact pass (2026-06-12)

Four stance-feel asks, each measured:

- MAELSTROM `none of these are impactful` — the sweep's surprise: the storm
  already crushed naive charges but LOST `ejecta-vs-Doom2x` (83% vs 118%),
  the matchup it's designed to win. Raw strength made it WORSE (entrainment
  feeds the enemy well). Fix: the spinning rim got TEETH — `_surge` 1.0 ->
  **1.7** (it was the only aggressive stance with no damage bonus), current
  22 -> 28, shell 0.6 -> 0.5 (denser grinder). Flips the matchup to 106/92.
- WALL `defensive buff + more visible structure` — new engine `_armor` knob
  (the defender-side mirror of `_surge`, a per-team incoming-damage
  multiplier in the combat damage path). Wall takes **40% less** (`_armor
  0.6`) + crisper bar (facing 1.0 -> 1.25). Drill's 4x still cracks it
  (eff 2.4x) — counters survive.
- SWARM WEB (was cloud) `form a web-like structure` + `units travel through
  it, in and out, around` — the cloud mode became a living SPIDER WEB built
  from the Chladni machinery: node_l 14 (concentric rings) x node_m 12
  (straight radial spokes, k=0 — distinct from Pulse-lattice's spiral). The
  node terms are TRAVELING waves, so node_v 0.07 migrates the rings (units
  ride them in/out), node_w 0.09 sweeps the spokes around, and spin 0.7
  streams units along the threads — a swarming current through a fixed net
  (measured: web persists across frames with ~25/255 inter-frame mote flow).
  Keeps the 1.2x bite. Reuses node knobs = NO new action slot (the 25-action
  head is fixed). Labels: re-tap cycles web -> comet now.
- SWARM COMET `heavy head, small tail` — head pack -0.25->-0.45, punch
  0.85->0.95, twist 0.35->0.2.
- MAELSTROM MODE VISUALS `no visible difference` — the composite shader now
  renders three storms: undertow drinks cool rings INWARD to a dark eye,
  ejecta blasts warm rings OUT with a burning rim, shear whips silver spokes
  with no radial motion. AI marker carries the mode (`_ai_mael[3]`).

The KNOB TABLE (`rl/policy.py` `_KNOBS`) is now 20 columns — added `cspd`
(Doom mobility tax, §18) and `armr` (Wall armor). Every play dial is mirrored
there so training and play share the physics; the trainer-image ref bumps
with each balance change.

## 28. Music generation — the plan (pending go-ahead)

The procedural score (§22) has hit its ceiling (composed blind). The agreed
next step, NOT yet started (awaiting an explicit "go"): host a music-gen
model on pstorm (5090 has ~15GB headroom beside lwplay) — MusicGen-medium
first (ungated, runs in the trainer image), Stable Audio Open later for
seamless loops (gated — needs an HF token). Pipeline: generate ~4 takes per
stem (calm pad / mid groove / battle wall / overtime, + win/lose stingers)
at 112 BPM D-minor; spectral analysis pre-filters takes (the substitute
ears); a listening page lets Wolfgang pick winners; loop + encode to OGG;
rewrite `audio.js`'s music half into a STEM MIXER (crossfade real loops by
the same intensity signals), keeping procedural SFX + the engine as fallback.
What helps most: the go-ahead, Wolfgang's ear on the rating page, and one
reference track (MusicGen has a melody-conditioned variant).

## 29. The stance balance matrix (2026-06-13)

First systematic balance read (`/tmp/lw_matrix_testbed.py` + a 36-pair sweep):
a symmetric mirror duel, both cursors charging the enemy centroid, each
holding one canonical stance, mass ratio after ~440 ticks. PROTOCOL CAVEAT:
"both charge" is the aggressor's test — defensive/spread stances (Wall,
Swarm-web) play their worst case and need a hold-protocol follow-up before
tuning. Tier list (wins of 8):

- **Doom 7-0-1** — dominant head-on; ONLY Maelstrom holds it (draw — the §27
  rim buff worked, it's the intended check). Head-on dominance is acceptable
  *because kiting is the out* (§26): you can refuse the standup fight.
- **Drill 6-2**, **Maelstrom 5-2-1** — strong.
- **Spin / Pulse 4-3-1** — mid, and a clean RPS triangle: Spin > Drill >
  Pulse > Spin.
- **Wall 3-4-1**, **Swarm 1-6-1** — weak, but BOTH protocol-confounded
  (defensive / spread); don't nerf/buff off this protocol alone.
- **Classic 0-4-4** — baseline (draws the gentle, loses to the aggressive).
- **Atom 0-8 — DEAD** (and it's compact, so NOT a protocol artifact). See §30.

## 30. Atom is mechanically anti-brawl — flagged for redesign (2026-06-13)

The matrix found Atom loses every head-on duel (0-8, even to do-nothing
Classic). Two intuitive buffs were tried and EMPIRICALLY REJECTED: (a) damage
surge (1.3/1.5) — *backfired*, Atom went 57%→0% vs Classic, because speeding
the boundary conversion churn helps the DENSER opponent win the exchange;
(b) tighter lobes (negative burst) — no help. Root cause is mechanical: the
fig-8 keeps Atom's units perpetually ORBITING, never consolidating into a
fighting mass, so any dense formation envelops the thin moving front. This
needs a redesign (a combat role for the orbit, or a denser fightable form),
not a knob — reverted to original. The matrix's value here was catching that
both "obvious" buffs ship a WORSE stance. Open task.

## 31. Doubled board + practice mode + test suite (2026-06-13)

- **Board doubled** (§ commit): 384×576 → 544×816 (2× area, same 8000
  fighters) for maneuver room. Cost ~17→47ms tick (~21fps); the client's
  60fps interpolation masks it, and the launch-diet/CUDA-graph work recovers
  it. `LW_PLAY_H/W` + `LW_GRAD_SWEEPS` env-tunable. (Gradient sweeps proven
  NOT the bottleneck — 12→4 changed nothing; it's the movement/collision
  machinery scaling with cells.)
- **Practice mode** (`opponent=practice`): a forgiving sandbox — the dummy
  holds a rotating LESSON stance (Classic → Doom → Wall → Maelstrom → Drill →
  Pulse → web) with a coach line ("Enemy holds DOOM — hold Maelstrom"); no
  win/loss/streak/gauntlet recorded, ~4s between rounds. Server `GameSession.
  practice`/`_lesson` + `LESSONS`; client `#practiceTip` banner, result card
  suppressed.
- **Test suite** (`tests/`, 32 passing, CPU): engine invariants
  (conservation, one-per-cell, occ↔SoA) across {wells, inertia} settings;
  the **train/play parity** tests that lock the `_KNOBS` table to the server
  dials (Doom `cspd` tax, Wall `armr`) so a one-sided balance change FAILS
  CI; `act()` shape/mask/legacy coverage. Run: `CUDA_VISIBLE_DEVICES=""
  PYTHONPATH=. uv run --with pytest --with numpy python -m pytest tests/ -q`.
- **Launch-diet agent FAILED**: its worktree branched from a stale commit
  (475-line dense-grid prototype, not the 1450-line real engine) — discarded,
  not merged. Worktree agents need a base from the working branch HEAD.

## 32. Black-out mode + the rest of the queue (2026-06-13)

- **Black-out mode** (toggle: B / 🌑 sheet button, persisted): the doubled
  board zoomed out and made barriers dim. Lean into it — the world goes near-
  black and the UNITS are the only light: the composite shader spills the
  army bloom (`amb`) onto nearby walls + floor, so barriers are revealed
  exactly where the action is and fade to black far away. `uBlackout` uniform;
  bloom auto-boosts +0.6 when on. Normal-mode wall rim also brightened
  (0.13→0.17) so barriers read on the big board either way.
- **Launch diet** (§9): _move_fighters 8-dir candidate loop → one vectorized
  gather, bit-exact, ~47→36ms tick.
- **Grid RLE** (§16): cell grid → (i8,u16) run triples (gflag=2), ~5x on
  battle grids, raw fallback, decoded before the mote-sync early-returns
  (drop-resilient).
- **King of the Hill** (§12) and **CB palette/accessibility** (§15) earlier
  this session.
- REMAINING: CUDA-graph restore (the real big-board fps fix — needs the
  teardown race reproduced offline first; risky) and music (on hold per
  request). Engine nondeterminism note: CPU runs are NOT bit-reproducible at
  fixed seed (scatter tie-breaks) — verify engine refactors with invariants,
  not hash equality.
