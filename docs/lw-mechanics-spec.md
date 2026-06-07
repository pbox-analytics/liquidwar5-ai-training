# Liquid War — faithful mechanics spec (the implementation contract)

Extracted from the C source (`/home/wolfgang/repo/liquidwar5-ai`) by the `lw-mechanics-port` workflow (2026-06-07). This is the source of truth for `simulator/engine.py`.

## Data model

## Data model (fixed-size, no allocation during play)

**Global scalars**
- `GLOBAL_CLOCK: int` — monotonic tick counter. Init = 2 (lwtime.c reset_time). `++` once per tick at the very end of `logic()`.
- `PLAYING_TEAMS: int` — count of alive teams; shrinks by 1 each time a team is eliminated, and team ids are COMPACTED to stay dense in [0, PLAYING_TEAMS).
- `CURRENT_ARMY_SIZE: int` — total fighter count. CONSTANT for the whole game (always an exact multiple of the initial PLAYING_TEAMS). Fighters never spawn or die.
- `ACTIVE_FIGHTERS[NB_TEAMS]: int` — per-team census, ZEROED and RE-COUNTED every tick inside move_fighters.
- Constants: `NB_TEAMS=6`, `NB_DIRS=12`, `MAX_FIGHTER_HEALTH=16384` (valid health [0,16383]), `AREA_START_GRADIENT=2000000`, `CURSOR_START_GRADIENT=1000000`, `SIDE_ATTACK_FACTOR=4` (side dmg = attack>>4 = /16), `NB_SENS_MOVE=2`, `NB_TRY_MOVE=5`.

**FIGHTER** (one per army slot; the only mutable per-agent state):
`{ short x; short y; short health; char team; char last_dir; }`
- `x,y`: integer cell coords into CURRENT_AREA (kept in sync with the grid pointer).
- `health` in [0,16383].
- `team` in [0, PLAYING_TEAMS). READ THIS, never infer from array index (conversion changes it).
- `last_dir`: VESTIGIAL — zeroed at init, never read/written by the sim. Drop on GPU.
Layout: `CURRENT_ARMY[CURRENT_ARMY_SIZE]`, initially interleaved by team (slot i belongs to team i % PLAYING_TEAMS; team t's k-th fighter at index t + k*PLAYING_TEAMS), but team mutates at runtime.

**PLACE** (the occupancy/collision grid — single source of truth for "who is where"):
`{ MESH* mesh; FIGHTER* fighter; }`, grid `CURRENT_AREA[CURRENT_AREA_W * CURRENT_AREA_H]`, indexed `y*CURRENT_AREA_W + x`.
- `mesh == NULL` ⇒ WALL / off-board (impassable).
- `fighter == NULL` ⇒ empty cell; else points at the single occupying fighter.
INVARIANT: each fighter is pointed at by exactly one PLACE.fighter, and that PLACE's (x,y) equals the fighter's (x,y). At most one fighter per cell — there is no overlap possible by construction.

**MESH** (navigation-graph node, one per passable cell-square):
`{ short x; short y; MESH_SIDE side; MESH_INFO info[NB_TEAMS]; void* link[NB_DIRS]; }`
- `link[dir]`: MESH* neighbor in direction dir, or NULL (wall/edge). NULL links are how walls block both gradient spread and movement.
- `side`: bitfield `{ int decal_for_dir:8; int size:24; }`. `side.size` is the cell's edge length (power of two 1,2,4,8,16 from quadtree merge) and is the GRADIENT EDGE WEIGHT.
- Array form `CURRENT_MESH[CURRENT_MESH_SIZE]` in RASTER (row-major: index = y*w + x ascending) order — this ordering is load-bearing for the in-place spread.

**MESH_INFO** (per-cell, per-team): `{ MESH_UPDATE update; MESH_STATE state; }`
- **MESH_STATE** bitfield `{ int dir:8; int grad:24; }`: `grad` = per-team approximate distance-to-cursor (24-bit SIGNED, can be negative via decremented seed; stays within ±2000000 in practice). `dir` = cached gradient-descent direction (0..11) last computed by a fighter on this cell this tick.
- **MESH_UPDATE** UNION `{ int time; struct { short x; short y; } cursor; }` — `time` and `cursor` ALIAS the same int. Semantics:
  - `time >= 0` ⇒ the union currently holds packed cursor coords ⇒ this team's cursor is on/at this cell ⇒ fighters use get_close_dir (home directly on cursor.x/cursor.y).
  - `time == -1` ⇒ stale sentinel (cursor just left / invalidate).
  - `time < 0` (other) ⇒ `-time` is the GLOBAL_CLOCK at which `dir` was last computed (freshness token).
  Writing `cursor.x/.y` deliberately clobbers `time` to a non-negative value (cursor coords are small positive ints) — this is the signal, not a bug.

**CURSOR** `CURRENT_CURSOR[NB_TEAMS]`: `{ int val; int x; int y; int active; int team; int key_state; int control_type; int from_network; ... }`
- `val`: the gradient SEED value injected at the cursor's cell. Init = AREA_START_GRADIENT/2 = 1000000. Decremented over time (see movement).
- `x,y`: integer cell position. `key_state`: 4-bit mask UP=1,RIGHT=2,DOWN=4,LEFT=8.

**Direction enum (the 12-dir fighter/mesh scheme), clockwise from just-east-of-north:**
DIR_NNE=0, DIR_NE=1, DIR_ENE=2, DIR_ESE=3, DIR_SE=4, DIR_SSE=5, DIR_SSW=6, DIR_SW=7, DIR_WSW=8, DIR_WNW=9, DIR_NW=10, DIR_NNW=11.
Unit vectors (+Y is screen-DOWN/south): `X_REF[12]={0,1,1,1,1,0,0,-1,-1,-1,-1,0}`, `Y_REF[12]={-1,-1,0,0,1,1,1,1,0,0,-1,-1}`.
NOTE: cursor movement uses a SEPARATE 8-dir scheme (1=N..8=NW) — do not conflate.

## Per-tick sequence

1. 1. move_all_cursors() — Outer loop runs (cursor_increase_speed+1) times (default 0 ⇒ once). Each iteration polls input / runs AI to set each active cursor's key_state, then calls move_cursor(i) for each active cursor. move_cursor: invalidate old cell (old_mesh.info[team].update.time = -1); derive 8-dir from key_state (opposing keys cancel ⇒ may be 0=no move); try move_if_free(dir), then dir-1, then dir+1 (45deg wall-slide); if still blocked run up to LW_MOVE_SIDE_LIMIT=10 side-step probes (dir-2/dir+2 scouting). A step succeeds only into a non-wall cell (target PLACE.mesh != NULL). After moving, write new_mesh.info[team].update.cursor.x/.y = cursor.x/.y (aliases time ⇒ marks cursor-present). Then decrement seed: if (moved || GLOBAL_CLOCK % 13 == 0) cursor.val--.
2. 2. apply_all_cursor() — For each active cursor: m = CURRENT_AREA[cursor.y*W + cursor.x].mesh; if m != NULL set m.info[cursor.team].state.grad = cursor.val (OVERWRITE, not min). Exactly ONE cell per team is seeded — the cell under the cursor. This is the single low-distance SOURCE the spread propagates from.
3. 3. spread_single_gradient() — ONE directional relaxation sweep over the entire mesh (NOT iterated to convergence). dir = (GLOBAL_CLOCK * 7) % 12. If dir in {2,3,4,5,6,7} ('down/forward' group) iterate CURRENT_MESH index 0 -> SIZE-1; if dir in {8,9,10,11,0,1} ('up/backward' group) iterate SIZE-1 -> 0. For each cell pos and each team i in [0,PLAYING_TEAMS): t = pos.link[dir]; if t != NULL: ng = pos.info[i].grad + pos.side.size; if t.info[i].grad > ng: t.info[i].grad = ng. PUSH relaxation, edge weight = SOURCE cell's side.size, IN-PLACE on a single buffer. Because the array is raster-ordered, a single forward/backward pass propagates the wavefront many cells in the swept direction this tick. gcd(7,12)=1 ⇒ all 12 dirs visited once per 12 ticks.
4. 4. move_fighters() — (a) Build cpu_influence[]; compute per-team attack[]/defense[]/new_health[] coefficients (see combat); then ACTIVE_FIGHTERS[i]=0 for all teams. (b) start=(GLOBAL_CLOCK/6)%12; table=(GLOBAL_CLOCK/3)%2; sens=0. (c) Iterate fighters f = CURRENT_ARMY[0..SIZE-1] IN ARRAY ORDER, SEQUENTIAL and IN-PLACE: ACTIVE_FIGHTERS[f.team]++; advance start (mod 12); pick dir from the cell's cached/recomputed gradient-descent dir; try to MOVE into the first of 5 ranked candidate cells that is non-wall AND empty (commit immediately, mutating the grid); if all 5 blocked, resolve COMBAT (attack enemy / heal ally) against candidates 0,1,2. Earlier fighters' moves/conversions are visible to later fighters in the same tick.
5. 5. check_loose_team() — Scan i in [0,PLAYING_TEAMS); the FIRST team with ACTIVE_FIGHTERS[i]==0 is eliminated (only one per tick). eliminate_team: PLAYING_TEAMS--; deactivate that cursor; COMPACT all team indices > eliminated down by 1 across cursors, fighters, ACTIVE_FIGHTERS[], COLOR_FIRST_ENTRY[], and every mesh cell's info[j>=eliminated]=info[j+1]. Team ids stay dense.
6. 6. GLOBAL_CLOCK++ — always, once. (In the original it runs even under PAUSE while the 5 sim steps are gated by !PAUSE_ON; headless has no pause, so all 6 are unconditional.)
7. Loop cadence (headless/training): repeat logic() while PLAYING_TEAMS >= 2 and TIME_LEFT > 0, with a ~24000-tick cap and an early dominance break (every 500 ticks after 2000, stop if a team holds >=60% of all fighters or <=1 team alive). Winner = team with max ACTIVE_FIGHTERS. display() is a pure render side-channel; it NEVER mutates sim state (check_loose was deliberately moved into logic() to avoid network desync).

## gradient

## Gradient algorithm (per-team Dijkstra-like distance field, relaxed one direction per tick)

**Nature.** Each team owns an independent scalar field `info[team].state.grad` over the mesh graph. Lower grad = closer to that team's cursor. It is a PERSISTENT, APPROXIMATE distance field — there is NO per-tick reset and NO convergence loop. It lags the cursor by many ticks; that lag is core to Liquid War's feel and MUST be reproduced.

**Three (and only three) writers of grad:**
1. `reset_game_area()` — ONCE at startup: for every cell i and every team k, `grad = AREA_START_GRADIENT (2000000)`. This is the "far/unknown" ceiling. Never re-run per tick.
2. `apply_all_cursor()` — each tick, BEFORE spread: the single cell under each cursor gets `grad = cursor.val` (OVERWRITE). cursor.val starts at 1000000 and only decreases, so the seed stays strictly below the 2000000 ceiling ⇒ reachable cells relax below unreachable ones.
3. `spread_single_gradient()` — each tick, ONE directional relaxation.

**Per-tick relaxation (the spread):**
```
dir = (GLOBAL_CLOCK * 7) % 12
if dir in {2,3,4,5,6,7}:  sweep CURRENT_MESH ascending  (index 0 -> SIZE-1)   # 'down/forward'
if dir in {8,9,10,11,0,1}: sweep CURRENT_MESH descending (index SIZE-1 -> 0)  # 'up/backward'
for pos in sweep_order:
    for i in 0 .. PLAYING_TEAMS-1:           # NOTE: PLAYING_TEAMS, not NB_TEAMS
        t = pos.link[dir]
        if t != NULL:
            ng = pos.info[i].grad + pos.side.size   # weight = SOURCE cell edge length
            if t.info[i].grad > ng:
                t.info[i].grad = ng              # PUSH min into the neighbor
```
- This is Bellman-Ford with a good ordering: because the mesh array is raster-ordered, the in-place ascending sweep lets an eastward/southward improvement chain forward across many cells in ONE pass; the descending sweep does the same for westward/northward dirs. So a value can travel far in a single tick along the swept monotone direction.
- gcd(7,12)=1 ⇒ over any 12 consecutive ticks all 12 directions are relaxed exactly once. There is no cap on iterations — propagation distance is bounded only by the in-place sweep + how many ticks pass, NOT by a fixed iteration count.
- Walls = NULL links, so the field naturally routes around them; unreached cells stay at 2000000.
- `cursor.val` decrement (in cursor movement): when the cursor moves 1 cell every point is up to 1 unit farther, so instead of incrementing the whole field it is cheaper to decrement the single seed; the every-13th-tick decrement even when stationary breaks lost-fighter cycling.

**GPU parity warning.** A naive parallel / double-buffered relaxation does NOT reproduce the single-pass long-distance propagation and will diverge. To match exactly you must replicate the SEQUENTIAL in-place sweep in the prescribed array order (ascending for down-group dirs, descending for up-group dirs). The ASM `boost_gradient_*` path is a bit-equivalent optimization of this same C math — implement the C semantics.

**Fighter consumption.** Fighters read this field via get_main_dir (steepest descent): scan the 12 links from `start`, pick the neighbor with strictly-minimum grad; ties broken by scan direction (sens) and start offset; if no neighbor improves, fallback dir = GLOBAL_CLOCK % NB_TEAMS (range 0..5 — a quirk that can never return dirs 6..11; reproduce verbatim).

## movement

## Fighter movement + collision (sequential, in-place, first-free-cell)

**Per-fighter direction selection** (cell p = CURRENT_AREA[f.y*W + f.x], mi = p.mesh.info[team]):
- if `mi.update.time >= 0` (cursor present on this cell): `dir = get_close_dir(...)` — home straight at the stored cursor coords. code_dir bitmask = (cursor.y<f.y ?1:0)|(cursor.x>f.x ?2:0)|(cursor.y>f.y ?4:0)|(cursor.x<f.x ?8:0); dir = code_dir ? LOCAL_DIR[(code_dir-1)*2 + sens] : start. LOCAL_DIR (sens0/sens1): 1(N)->NNE/NNW, 2(E)->ESE/ENE, 3(NE)->NE/NE, 4(S)->SSW/SSE, 6(SE)->SE/SE, 8(W)->WNW/WSW, 9(NW)->NW/NW, 12(SW)->SW/SW; codes 5,7,10,11,13,14,15 are impossible.
- else if `(-mi.update.time) < GLOBAL_CLOCK` (cached dir stale): `dir = get_main_dir(...)` (gradient steepest descent, above) and set `mi.update.time = -GLOBAL_CLOCK` to cache for the rest of this tick.
- else: reuse the cached `mi.state.dir` (another fighter on this same cell already computed it this tick — a deliberate per-cell-per-tick recompute cache).
`sens` starts at 0 each tick and post-increments (mod 2) ONLY in the two recompute branches, so handedness alternates across recomputing fighters, not all fighters. `start` = (GLOBAL_CLOCK/6)%12, advanced (mod 12) once per fighter. `table` = (GLOBAL_CLOCK/3)%2 selects the candidate table.

**The 5-try fallback (collision rule).** For (table k, primary dir) take the 5 ranked candidate directions FIGHTER_MOVE_DIR[k][dir][0..4] (k=0 and k=1 differ only at the four cardinal-ish rows 1,4,7,10 where slot1/slot2 swap to flip handedness). Candidate cell j = p + OFFSET[k][dir][j], where OFFSET = dx + dy*CURRENT_AREA_W and (dx,dy)=(X_REF[cand],Y_REF[cand]). Evaluate j=0..4 IN ORDER:
- A cell is MOVABLE iff `p_j.mesh != NULL` (in-map) AND `p_j.fighter == NULL` (empty).
- Move into the FIRST movable candidate: `p_j.fighter = f; p.fighter = NULL; f.x += dx; f.y += dy;` then STOP (no combat).
- The occupancy check reads the LIVE grid, so a move committed earlier this tick is visible. Two fighters can never occupy a cell — collision is structurally impossible because of the empty-cell check + immediate in-place commit.
- ONLY if all 5 candidates are non-movable does control fall through to combat.

**Sequentiality is load-bearing.** Fighters are processed strictly in array order, mutating PLACE.fighter as they go; an earlier fighter can vacate/fill a cell a later fighter then sees. A faithful GPU port must reproduce these sequential semantics (serial pass, or a conflict-resolution scheme that yields identical first-free-cell-wins ordering). The parallel ASM `boost_move_fighters` path is an optional optimization — implement the C reference.

## Conservation in movement
Movement only relocates a fighter from one PLACE to another (and zeroes the old) — it never creates or destroys a FIGHTER struct. CURRENT_ARMY_SIZE is fixed at create_army time (battle_room * fill% / PLAYING_TEAMS, floored ≥1, then * PLAYING_TEAMS). No code path appends to or shrinks CURRENT_ARMY during play.

## combat

## Combat: health drain + team conversion (NEVER deletion)

Reached ONLY when a fighter found no free move cell (all 5 candidates blocked). Examine the SAME ranked candidates p0,p1,p2 in priority order; EXACTLY ONE of A/B/C/D fires (if/else chain). `team = f.team` (the acting fighter); attack/defense/new_health are the ACTING team's coefficients. The attacker's own health is never changed; only the TARGET is mutated. Candidates p3,p4 are never used for combat.

- A. FRONT ATTACK (p0 enemy): if p0.mesh && p0.fighter && p0.fighter.team != team ⇒ `p0.fighter.health -= attack[team]`.
- B. SIDE ATTACK slot1 (p1 enemy): else if p1 is enemy ⇒ `p1.fighter.health -= attack[team] >> 4` (/16).
- C. SIDE ATTACK slot2 (p2 enemy): else if p2 is enemy ⇒ same with p2, >>4.
- D. HEAL (p0 ally): else if p0.fighter.team == team ⇒ `p0.fighter.health += defense[team]`, clamped to MAX_FIGHTER_HEALTH-1 (16383).

**CONVERSION (the core — no fighter is ever removed):** after an attack subtracts health, if `target.health < 0`:
```
while (target.health < 0) target.health += new_health[team]   # rebase up by ATTACKER team's new_health
target.fighter.team = team                                     # defect to attacker's team
```
Post-conversion health = original_negative + n*new_health[team], n = smallest integer making it ≥0 (lands in [0,new_health), plus any carry). The fighter slot persists; only its `team` and `health` change. This is the ONLY way ownership shifts, and it is why the army count is conserved. (Dissolve particles / 30th-kill screen shake are cosmetic — omit on headless.)

**Per-team coefficients (computed once per tick, before the fighter loop):**
```
cpu_influence[i] = (cursor[i] is CPU && active) ? cpu_advantage : 0      # default 0
coef = ACTIVE_FIGHTERS[i]*PLAYING_TEAMS - CURRENT_ARMY_SIZE              # population balance
coef *= 256; coef /= CURRENT_ARMY_SIZE; if (coef > 256) coef = 256
coef *= (number_influence-8)^2; coef /= 64
if (number_influence < 8) coef = -coef
if (coef < 0) coef /= 2
coef += 256                                                             # baseline 256 == 1.0
attack[i]     = (coef * fixsqrt(fixsqrt(1 << (fighter_attack     + cpu_influence[i])))) / (256*8)    # /2048
defense[i]    = (coef * fixsqrt(fixsqrt(1 << (fighter_defense    + cpu_influence[i])))) / (256*256)  # /65536
new_health[i] = (coef * fixsqrt(fixsqrt(1 << (fighter_new_health + cpu_influence[i])))) / (256*4)    # /1024
# each clamped: if >= MAX_FIGHTER_HEALTH -> MAX-1; if < 1 -> 1
```
- All arithmetic is INTEGER (truncating) division. The rubber-band makes outnumbered teams hit harder / heal more.
- `fixsqrt(x)` is a 16.16 fixed-point sqrt: x<=0 -> 0, else `(int)(65536 * sqrt(x/65536.0))`. The nested `fixsqrt(fixsqrt(1<<n))` is a 4th-root in 16.16 space — `1<<n` is a raw int REINTERPRETED as 16.16 fixed (e.g. 1<<8 = 256 reads as 256/65536 ≈ 0.0039). Implement the nested fixed-point op EXACTLY; do not collapse to a real 4th-root. Use double sqrt for bit parity.
- Defaults (config.c): fighter_attack=fighter_defense=fighter_new_health=8, number_influence=8 (⇒ middle term 0 ⇒ coef=256 exactly), cpu_advantage=0, cursor_increase_speed=0.

## conservationInvariant

## Why total fighter count is strictly constant

CURRENT_ARMY_SIZE is computed ONCE in create_army (battle_room * fill% / PLAYING_TEAMS, floored ≥1, then * the initial PLAYING_TEAMS so it is an exact multiple) and is never changed during play. The FIGHTER array is a fixed-size buffer; nothing appends to it or removes from it after placement.

Every per-tick operation conserves the count by construction:
- MOVEMENT relocates a fighter between PLACE cells (sets new PLACE.fighter, clears old) — it neither creates nor destroys a struct, and the empty-cell precondition guarantees at most one fighter per cell.
- COMBAT only mutates the TARGET fighter's `health` and (on lethal) `team`. A fighter whose health goes negative is CONVERTED (team reassigned + health rebased positive by repeatedly adding new_health), NOT deleted. The slot, and therefore the total, is untouched.
- HEALING only raises an ally's health (clamped).
- ELIMINATION (check_loose_team) removes a TEAM from play (PLAYING_TEAMS--), not fighters — by the time a team is eliminated it already has ACTIVE_FIGHTERS==0, so there are no fighters of that team left to remove; the remaining fighters are unaffected except for the team-index compaction.
- ACTIVE_FIGHTERS[] is a derived census recomputed from scratch every tick (zeroed, then ++ as each of the CURRENT_ARMY_SIZE fighters is visited) — so sum(ACTIVE_FIGHTERS) == CURRENT_ARMY_SIZE every tick, by definition. This is the testable conservation invariant: at the end of any tick, the total over all teams equals the constant army size. The game is therefore a pure REDISTRIBUTION of a fixed population among teams.

## maps

## Maps: 1-bit bitmap -> connectivity-validated -> magnified -> mesh graph -> team placement

**Map model.** A map is fundamentally a 1-bit bicolor bitmap: each pixel is WALL or PLAYABLE. There is NO stored teams/areas layer — playability is DERIVED by flood-fill at load, and team spawns are computed at runtime from fixed fractional anchors.

**Load pipeline (src/map.c):**
1. Threshold any source BMP: pixel is LIGHT(passable=2) iff `6R+3G+1B > 315` (R,G,B are Allegro 0..63), else DARK(wall=0).
2. Crop to the tight bounding box of all DARK pixels; reject if width or height < MINI_SIDE_SIZE(4).
3. check_if_playable: seal the border to DARK; seed at the first LIGHT pixel with x>0,y>0; flood-fill 8-connected (forward spread_color_down + backward spread_color_up to a fixpoint) marking the connected component PLAYABLE_AREA(1). Reject if the component has < MINI_PLAYABLE_AREA(1024) cells. Any LIGHT pixel NOT in the seed component stays LIGHT and is later painted as WALL — only ONE connected component is playable.
4. RLE-encode (row-major): token l>0 = l wall pixels, l<0 = (-l) playable pixels, runs capped at 127, 0 terminator. Stored blob = 8-byte header (size:int32 LE, w:int16 LE, h:int16 LE) + 16-byte system name + 32-byte readable name + RLE body (body offset 56).

**Game-start build (lw_map_create_bicolor + mesh.c):**
5. Decode RLE to a w*h DARK/LIGHT bitmap, re-run check_if_playable, paint MESH_FG=1 (wall) / MESH_BG=2 (passable).
6. Magnify by integer `zoom = max(((min_w-1)/w)+1, ((min_h-1)/h)+1)`, nearest-neighbour, so CURRENT_AREA_W/H = w*zoom, h*zoom (min_w/h from rules.min_map_res).
7. create_first_mesher: cell.used = (pixel != MESH_FG). Border cells and walls are unused. group_mesher merges 2x2 blocks of equal-size passable squares iteratively for sizes 1,2,4,8 (stops at MESH_MAX_ELEM_SIZE=16) — a quadtree compaction (pure optimization; passability semantics = pixel != MESH_FG). mesher_to_mesh emits CURRENT_MESH in raster order with .x,.y,.side.size and 12 directional links. Initial info[j].state.dir = (i+j) % 12.
8. create_game_area: for each mesh element, for the side.size x side.size block of pixels it covers, set CURRENT_AREA[(my+y)*W + mx+x].mesh = &that mesh. A cell is passable iff PLACE.mesh != NULL.

**Army sizing (create_army):** battle_room = sum of side.size^2 over mesh (total passable cells). CURRENT_ARMY_SIZE = battle_room * fill_table[rules.fighter_number] / 100, then /PLAYING_TEAMS (floor, min 1), then *PLAYING_TEAMS. fill_table[33] = {1,2,3,4,5,6,8,9,10,12,14,16,18,20,22,24,25,27,29,31,33,36,40,45,50,55,60,65,70,75,80,90,99}.

**Team placement (place_team / add_fighter):** up to 6 teams, anchors at x in {W/6, W/2, 5W/6} and y in {H/4, 3H/4} (integer div), part index 0..5 = (W/6,H/4),(W/2,H/4),(5W/6,H/4),(W/6,3H/4),(W/2,3H/4),(5W/6,3H/4). Per team = CURRENT_ARMY_SIZE/PLAYING_TEAMS fighters, each spawned at health=MAX-1=16383. Placement spirals outward in a clockwise expanding box from the anchor (top edge, ++x_max; right edge, ++y_max; bottom edge, --x_min; left edge, --y_min; bounded to [1,W-2]/[1,H-2]); each visited passable, empty cell gets a fighter (add_fighter: requires mesh!=NULL && fighter==NULL). Walls/occupied cells are skipped, spiral continues.

**Map generators (utils/lwmapgen, standalone tool).** Source is an 8-bit grayscale BMP (palette pal[i]=i/4; walls color 0, background 255) drawn over a num_row x num_col SECTION grid. func table: 0 rand_func (picks 1..12 uniformly), 1 big_quad, 2 boxes, 3 bubbles, 4 circles, 5 circuit, 6 hole, 7 lines, 8 rand_box, 9 rand_poly, 10 rand_poly_cut, 11 street, 12 worms. There is NO literal perfect-maze generator — the maze-LIKE ones are `circuit` (wells joined by L-shaped pipe corridors) and `street` (main roads + random branches). All randomness is glibc rand()/srand(time); bit-exact reproduction needs the same rand() sequence. map_size[6] = {128x95,160x120,256x190,320x240,512x380,640x480} (default idx 3). For a re-implementation, treating the map as a fixed pre-validated wall/passable bitmap + the derived mesh graph is sufficient — the generators only matter for reproducing specific procedural maps.

## What the OLD sim got wrong

- Presence grid drops overlapping fighters. The current GPU sim uses a one-fighter-per-cell PRESENCE grid as its state, which structurally loses any fighters that would share a cell. In real LW there is never overlap to begin with — PLACE.fighter holds at most one fighter and movement only ever targets an EMPTY cell (mesh!=NULL && fighter==NULL), committed in-place during a sequential pass. The fix is not 'allow overlap' but to model each fighter as an explicit indexed slot in a fixed-size CURRENT_ARMY[CURRENT_ARMY_SIZE] buffer with a strict one-fighter-per-cell occupancy grid, and resolve moves so first-free-cell-wins (serial order, or a conflict resolution that reproduces it). Using a presence grid as the sole state both drops fighters AND breaks the conservation invariant (sum of per-team counts must always equal the constant CURRENT_ARMY_SIZE).
- Combat DELETES instead of CONVERTING. The current sim removes a fighter on lethal combat; the real engine NEVER removes a fighter. On health<0 the target is reassigned to the attacker's team and its health is rebased positive via `while(h<0) h += new_health[attacker_team]`. Deleting violates conservation (count must stay constant) and removes the entire team-flip game mechanic (Liquid War is won by CONVERTING the enemy population, not killing it). Must change to: subtract attack[team] (or attack>>4 for the two side slots), and on negative, rebase-and-reassign team in the same slot.
- Gradient is capped at ~8-40 iterations instead of a full persistent distance field. The current sim runs a bounded fixed-iteration relaxation per tick, producing a truncated/local field. Real LW does the OPPOSITE in two ways that BOTH matter: (a) it is a PERSISTENT field — initialized to 2000000 ONCE at startup, never reset, only seeded at the single cursor cell each tick and relaxed incrementally; (b) each tick does exactly ONE directional sweep (dir = (GLOBAL_CLOCK*7)%12) but that sweep is an IN-PLACE, raster-ordered Bellman-Ford pass that propagates the wavefront MANY cells in the swept direction in a single tick — there is no iteration cap. Capping iterations and/or resetting the field each tick both diverge from the original. Fix: maintain a persistent per-team grad buffer; seed only the cursor cell (overwrite with cursor.val); do one direction's in-place sweep per tick in the prescribed ascending/descending array order with edge weight = source cell side.size; never reset.
- Likely-missing: per-tick seeding via cursor.val (overwrite, not min) and the slow cursor.val DECREMENT (val-- when moved or every 13th tick). Without the decrementing seed below the 2000000 ceiling, reachable cells won't relax below unreachable ones and fighters won't descend correctly / will form lost-fighter loops.
- Likely-missing: the union-aliased cursor-present mode (update.time>=0 ⇒ get_close_dir direct homing) and the get_main_dir fallback dir = GLOBAL_CLOCK%NB_TEAMS (0..5) quirk. If the GPU sim only ever does grad descent and ignores the cursor-on-cell direct-homing branch and the stale-dir cache, near-cursor behavior and tie-breaking will diverge.
- Likely-missing: the population-balance combat coefficients with exact 16.16 nested fixsqrt and integer truncation, plus side-attack >>4 and the heal-on-blocked-ally branch. Approximating attack/defense as constants or using real-valued sqrt changes conversion dynamics and breaks bit-parity; only candidates 0/1/2 attack, only candidate 0 heals.
- Likely-missing: team-index COMPACTION on elimination (PLAYING_TEAMS--, shift all team ids > eliminated down by one across fighters, cursors, ACTIVE_FIGHTERS, and every mesh cell's info[]). If the GPU sim keeps fixed team ids without compaction, it must mask eliminated teams everywhere the original iterates [0,PLAYING_TEAMS) and must match the shifted indexing; otherwise multi-team games diverge after the first elimination.
- Likely-missing: sequential, order-dependent fighter processing. If the GPU sim updates all fighters from a single snapshot in parallel (read old grid, write new), it will NOT match the original where an earlier fighter's move/conversion is visible to later fighters in the SAME tick. This affects both collision (who gets the contested empty cell) and combat (a fighter can be attacked and converted partway through the scan). Bit-exact parity requires reproducing the serial in-place semantics.

## Chosen GPU design — Design B (indexed-particle SoA with scatter-derived public views, direct in-place engine replacement)

All three designs correctly diagnose and fix the three spec-mandated errors (presence-grid drops fighters -> indexed slots / one-per-cell occupancy; combat deletes -> convert-on-negative with health rebase; capped/reset gradient -> persistent never-reset cursor-seeded distance field). The decision therefore comes down to faithfulness per the priority order, then GPU-batchability, then tractability against the REAL consumer surface I verified in the repo.

Designs A and B share an essentially identical representation: a fixed CURRENT_ARMY[N] structure-of-arrays (fx, fy, fhealth int32, fteam int8) as the single source of truth, an int32 occ grid holding the owning slot index (the GPU analogue of PLACE.fighter), a persistent (B,T,H,W) int32 gradient initialized to 2_000_000 once and never reset, integer 16.16 nested-fixsqrt combat coefficients, scatter_reduce(amin)-by-slot-index priority-claim collision (a deterministic, faithful analogue of first-free-cell-wins), and — crucially — team_oh/health rebuilt every tick as DERIVED scatter views so the entire downstream stack reads unchanged. This is the maximally faithful tractable port: it reproduces conservation (sum(ACTIVE_FIGHTERS)==N is structurally true because moves only relocate a slot and combat only flips a slot's team), the convert-not-delete mechanic, and a true persistent flood-fill field.

I verified the consumer surface in /home/wolfgang/repo/liquidwar5-ai-training: web/server.py:108-112 does oh.argmax(0) and oh.sum(0)>0 to color exactly one team per cell and oh.sum(dim=(1,2)).long() for integer fighter counts; ppo.py:24 and eval.py:116 compute team share via team_oh.sum(dim=(2,3)); the class is constructed by name LiquidWarEngine at four sites (train.py:100, web/server.py:67, eval.py:101, batch_runner.py:30) with a grad_iters kwarg. A and B both keep binary one-hot presence via scatter, so server/eval/ppo need ZERO edits. Design C's count-field model abandons one-fighter-per-cell (the spec's explicit anti-pattern in simErrors[0]: 'The fix is NOT allow overlap'), makes team_oh fractional (breaks the argmax-one-color render and integer fighter counts at server.py:108/119 and forces a lockstep front-end change), collapses the 12-dir scheme + FIGHTER_MOVE_DIR/sens/start/table tie-break tables away, and replaces 16.16 fixsqrt with floats. That is three independent faithfulness sacrifices the spec calls out by name, so C scores lowest on the top-priority axis despite being the cleanest to batch.

B wins over A on a narrow but real margin: B is a direct in-place engine swap with explicit backward-compat handling of the verified-required grad_iters constructor kwarg (repurposed as the K-pass wavefront budget so train.py:103 / batch_runner.py:37 keep working), and it gives the most concrete two-tier gradient implementation. A is functionally the same engine but proposes a faithful=True flag / LiquidWarEngineV2 to keep the old presence engine alive for A/B parity testing. That A/B-parity idea is the single best thing A adds — but it is a development scaffold, not a design difference, and it is better grafted onto B than chosen as the winning architecture (keeping a permanently-divergent presence engine around invites the exact 'old checkpoint on new MDP looks random' confusion both designs warn about). On the three scored axes B and A tie on faithfulness and batchability; B edges tractability by being a single drop-in with the kwarg-compat spelled out, and grafting A's parity harness as a throwaway test fixture captures A's only real advantage.

### representation

Per-game-batched, fixed-size, no per-tick allocation. Let B=batch, N=CURRENT_ARMY_SIZE (constant for a run, exact multiple of initial T), H,W grid, T<=6.

FIGHTER slots (the single source of truth, SoA):
  fx        : (B,N) int16  — x cell coord
  fy        : (B,N) int16  — y cell coord
  fhealth   : (B,N) int32  — [0,16383]; int32 (not int16) so it can transiently go negative before conversion rebase
  fteam     : (B,N) int8   — [0,PLAYING_TEAMS); READ this, never index%T
  (last_dir dropped per spec — vestigial)

OCCUPANCY grid (replaces presence bool — holds the owning slot index, the GPU analogue of PLACE.fighter):
  occ       : (B,H,W) int32 — slot index occupying the cell, or -1 = empty
  walls     : (B,H,W) bool  — mesh==NULL (impassable); passable = ~walls
  side_size : (B,H,W) int16 — gradient EDGE WEIGHT per cell (quadtree side.size; =1 if you skip the quadtree merge and treat every passable pixel as a size-1 mesh node, which is the faithful-enough default)

GRADIENT (persistent, never reset):
  gradient  : (B,T,H,W) int32 — per-team grad field; init AREA_START_GRADIENT=2_000_000 everywhere ONCE at reset; walls held at the ceiling
  grad_dir  : (B,T,H,W) int8  — cached steepest-descent dir per cell (mesh state.dir), recomputed lazily per tick
  grad_stamp: (B,T,H,W) int32 — freshness token (-GLOBAL_CLOCK) so a 2nd fighter on a cell reuses the cached dir this tick

CURSOR:
  cursor_pos: (B,T,2) int64  — (y,x); KEEP this name/shape (web/eval read it)
  cursor_val: (B,T)   int32  — gradient seed, init 1_000_000, decremented
  cursor_present: (B,T,H,W) bool — the update.time>=0 / get_close_dir flag (one true cell per team)

CENSUS / scalars (per batch element):
  active_fighters: (B,T) int32 — re-zeroed and recounted every tick
  team_alive: (B,T) bool
  playing_teams: (B,) int32
  GLOBAL_CLOCK: (B,) int32 (or a shared python int if all games tick in lockstep — they do, so a scalar int is fine)

DERIVED VIEWS rebuilt at tick end (so the old readers/obs keep working):
  team_oh (B,T,H,W) float = scatter one-hot of fteam at (fy,fx)
  health  (B,H,W)  float  = scatter fhealth at (fy,fx)
Both come straight from the slot tensors via index_put_ on flattened (b*H*W) indices.

Direction tables as buffers: X_REF,Y_REF (12,) int8; FIGHTER_MOVE_DIR (2,12,5) int8; OFFSET derived from X_REF/Y_REF; LOCAL_DIR (15,2) int8; all registered once, device-resident.

### collisionResolution

The C is strictly sequential: fighters processed in array order, each commits in-place, later fighters see earlier moves; first-free-cell-wins. A naive batched 'all read old grid, all write' breaks this two ways: (1) two fighters pick the same empty target, (2) a fighter moves into a cell another just vacated. To reproduce first-free-cell-wins deterministically WITHOUT a Triton scan, use an ITERATED PRIORITY-CLAIM scheme over the 5 candidate ranks, with the original array index as the tie-break priority (lower slot index = processed earlier in C = wins):

For each fighter compute its 5 ranked candidate cells from FIGHTER_MOVE_DIR[table][dir] + OFFSET. Then loop rank r=0..4 (5 iterations, python loop, batched over B*N):
  - A fighter is still UNRESOLVED if it hasn't moved yet and hasn't been blocked-out of all ranks.
  - candidate cell c_r is ELIGIBLE iff passable AND occ[c_r]==-1 (empty in the CURRENT, progressively-updated occ) AND no earlier-rank candidate of this same fighter was eligible (enforced by processing ranks in order and dropping a fighter once it claims).
  - Among all unresolved fighters whose rank-r candidate is the SAME empty cell, the winner is the one with the SMALLEST slot index (scatter_reduce 'amin' of slot-index keyed by flattened target cell -> winner-per-cell map; a fighter wins iff winner_of[its target]==its own index). This min-index tie-break exactly reproduces 'earlier fighter in array order got the cell first'.
  - Winners commit: occ[target]=idx (scatter), occ[source]=-1, update fx/fy; mark resolved. Crucially, because occ is mutated between ranks AND the engine still must let a rank-0 winner free a cell that another fighter wants at rank-0 in the SAME pass, run the 5-rank loop INSIDE an outer 'sweep' loop that repeats until no fighter moves (fixpoint). In practice 2-3 outer sweeps converge because vacated cells only open new rank-0 options locally.

This is NOT bit-identical to the C serial order in pathological chains (A vacates a cell B wanted, but B already lost it to C at a lower index in the same sub-pass) — the min-index claim resolves the dominant ordering correctly but a fully bit-exact port needs the serial scan. Tradeoff stated under risks. The min-index priority is deterministic and B-batched: same seed -> same result, which is what RL training requires (reproducibility), even if it diverges from the C reference by a few contested cells per tick. occ is int32 holding slot index, so 'who is here' and 'who vacated' are both O(1) gathers; collision is structurally impossible (one slot per occ cell by construction of the scatter_reduce-amin claim).

### combatVectorization

Combat fires ONLY for fighters that found no free cell after all 5 ranks/sweeps (the 'unresolved & has-a-fighter-neighbor' set). Acting on the SAME ranked candidates p0,p1,p2, exactly one of A/B/C/D per fighter (if/else chain), using the ACTING team's coefficients.

Per-team coefficients (computed once/tick, BEFORE the loop, fully vectorized over (B,T)):
  coef = active_fighters*playing_teams - N ; coef*=256; coef//=N (integer trunc); clamp<=256
  coef *= (number_influence-8)**2; coef//=64; if number_influence<8: coef=-coef; if coef<0: coef//=2; coef+=256
  attack = (coef * fixsqrt(fixsqrt(1<<(fighter_attack+cpu_inf)))) // 2048
  defense= (coef * fixsqrt(fixsqrt(1<<(fighter_defense+cpu_inf)))) // 65536
  new_health=(coef* fixsqrt(fixsqrt(1<<(fighter_new_health+cpu_inf)))) // 1024
  clamp each to [1, 16383].
  fixsqrt implemented as torch op on int64: x<=0 ->0 else (65536.0*sqrt(x.double()/65536.0)).floor().long() — double for bit parity, no Triton.
  With defaults (8,8,8, number_influence=8) coef==256 exactly so these reduce to constants per tick unless population diverges; still compute the full formula so the rubber-band engages as games progress.

The drain itself, batched: for each combatant fighter f gather candidate slots p0,p1,p2 (slot index at each candidate cell, or -1). Compute the if/else chain as MUTUALLY EXCLUSIVE masks over the combatant set:
  is_enemy(p) = occ_slot(p)!=-1 & fteam[target]!=fteam[f]
  A = is_enemy(p0): dmg=attack[f.team] applied to p0
  B = ~A & is_enemy(p1): dmg=attack>>4 to p1
  C = ~A&~B & is_enemy(p2): dmg=attack>>4 to p2
  D = ~A&~B&~C & p0_is_ally: heal p0 by defense[team], clamp 16383
Because two attackers can target the SAME victim slot this tick, accumulate damage with scatter_reduce(sum) keyed by target slot index (the C applies them serially; summed total drain is order-independent for the health subtraction; only conversion ownership is order-sensitive — see below), then apply to fhealth in one shot.

CONVERSION (no deletion): after drain, for any slot with fhealth<0: rebase while(h<0) h+=new_health[attacker_team] — vectorize as n=ceil((-h)/nh); h += n*nh (matches the while loop exactly since each add is nh); set fteam=attacker_team. The attacker team for a multiply-hit victim must be deterministic: pick the attacker with the smallest slot index among those who hit it (scatter_reduce amin of attacker-slot keyed by victim slot), reproducing 'last writer in serial order' approximately — note C uses the LAST attacker to push it negative, so for strict parity track the attacker whose cumulative drain crossed zero; the min-index pick is the deterministic approximation (risk noted). Count is conserved: slot persists, only team+health change.

### gradientImpl

Full persistent flood-fill distance field, batched, NO Triton, reproducing the single in-place raster sweep. gradient (B,T,H,W) int32, init 2_000_000 once at reset, never reset. Per tick: dir=(clock*7)%12; for the 4 'diagonal'/'cardinal' dirs the neighbor is link[dir] = cell + (X_REF[dir],Y_REF[dir]); edge weight = SOURCE side_size.

The faithful part is the SEQUENTIAL in-place chain along the swept axis. I implement it as a batched 1-D scan along the sweep's monotone axis using a python loop over LINES (rows or cols), not over cells:

- For 'down/forward' dirs (ascending raster, dy>=0 dominant e.g. SE/SSE/S): the improvement chains along increasing y (and x). Do a loop over y=0..H-1 (and within a row, vectorize x; for dirs with x-propagation also loop x in the swept x-direction). At step y: g[:, :, y, :] = min(g[:, :, y, :], shift_along_x(g[:, :, y-prev, :]) + side_size), masked by ~wall and by link existence (target not wall). Because we march y in order and write in place, row y sees the just-updated row y-1 — exactly the raster forward chaining that lets a value travel many cells in one tick.
- For 'up/backward' dirs (descending), march y=H-1..0 (and reversed x). Same recurrence, opposite direction.
- A pure-diagonal dir propagates along BOTH axes per step, so the inner is a 1-D cumulative-min-with-offset along x done with a python x-loop OR a logarithmic doubling (Hillis-Steele) min-scan; the simple x-loop is correct and is what the C does. The whole thing is O(H+W) python-iteration steps per tick, each a (B,T,W) or (B,T,H) vectorized torch.minimum — cheap, GPU-resident, no kernel.

Walls/NULL-links: precompute, per dir, a boolean link_exists mask (target cell passable AND source passable) and AND it into the relaxation; unreached cells stay at 2_000_000. This is Bellman-Ford with the prescribed good ordering, matching the spec's 'single pass propagates many cells'. Over 12 ticks all 12 dirs run once (gcd(7,12)=1).

Fighter consumption (get_main_dir / get_close_dir): when computing per-fighter primary dir in move_fighters, gather the 12 neighbor grads at the fighter cell, pick strict-min (ties by sens/start offset), fallback dir=clock%6 when none improve. If cursor_present at the cell, override with get_close_dir homing (code_dir bitmask -> LOCAL_DIR[(code-1)*2+sens]). Cache into grad_dir/grad_stamp so a 2nd fighter on the cell reuses it (the stamp==-clock check).

### blastRadius

TRAINING RESET (hard): existing policy checkpoints in results/rl are INVALID and training restarts from scratch. Reasons: (1) gradient channel now carries true persistent distances (scale 0..2e6 pre-norm) vs the old capped aged counter, so the own_gradient/enemy obs channel distribution shifts; (2) conversion-not-deletion changes the entire reward landscape (population is redistributed, not destroyed) — the optimal policy is genuinely different (convert enemies, don't just kill). This closes task #28's sibling work but means re-running rl.train --updates from 0. The good news: build_egocentric_obs, CursorPolicy, act(), collect_rollout, compute_gae, ppo_update need NO code edits — only retraining.

POLICY OBS / build_egocentric_obs: code unchanged; semantics improved (presence is now exactly one-hot-per-cell, health conserved). No interface break.

PLAY SERVER (web/server.py): reads engine.team_oh[0], engine.health[0], engine.walls[0], engine.cursor_pos[0], engine.team_alive[0] and calls get_observation()+act(). ALL preserved as derived views -> server keeps working with no changes, and renders correctly (no more vanishing fighters; team flips now show as recolor). Must reload the new checkpoint.

eval.py + batch_runner.py: read engine.health, team_oh, cursor_pos, team_alive — all preserved. _heuristic_dydx / inline AI unaffected. eval comparisons against old checkpoints are meaningless (different obs scale) — re-baseline.

NEW internal attributes (fx,fy,fhealth,fteam,occ,cursor_val,cursor_present,grad_dir,grad_stamp,active_fighters,playing_teams) are additive; nothing reads them externally today. Land as LiquidWarEngineV2 behind a flag so the presence-grid engine remains for parity A/B until the new one is verified in-prod (per repo rule: tests passing != shipped; build+deploy+browser-verify the web server before declaring done).

## Implementation plan

1. Add the SoA particle buffers to LiquidWarEngine.reset(): fx,fy (B,N) int16; fhealth (B,N) int32 (NOT int16 — must hold transient negatives before rebase); fteam (B,N) int8. N = fighters_per_team * initial_T, captured once as self.army_size (B,). Replace the multinomial _place_teams scatter with the spec's spiral placement (anchors at x in {W/6,W/2,5W/6}, y in {H/4,3H/4}) writing into the slot arrays at health=16383; build occ_idx (B,H,W) int32 init -1 via scatter of arange(N) at (fy,fx).
2. Register direction tables as device buffers: X_REF/Y_REF (12,) int8, FIGHTER_MOVE_DIR (2,12,5) int8, OFFSET = X_REF + Y_REF*W, LOCAL_DIR (15,2) int8. Replace the current 8-dir _dy/_dx scheme with the full 12-dir scheme; reproduce get_main_dir steepest-descent (strict-min over the 12 links, tie-break by sens/start) and the fallback dir = GLOBAL_CLOCK % 6 quirk verbatim.
3. Rewrite _seed_and_spread_gradient: drop the .add_(1) aging and the iters=40/grad_iters reset loop. Init gradient to 2_000_000 ONCE in reset, never reset. Per tick: overwrite the single cursor cell with cursor_val (apply_all_cursor), then ONE directional sweep with dir=(tick*7)%12 — ascending raster shifted-min for dir in {2..7}, descending for {8..11,0,1}, edge weight = source side_size (default 1), mask where source-or-target is wall. Repurpose grad_iters as the K wavefront-pass budget along the swept axis. Add a fixed-map unit test against a CPU Dijkstra to confirm long-distance single-pass propagation and the many-tick lag.
4. Add cursor_val (B,T) int32 init 1_000_000 and the decrement rule (val -= moved | (tick%13==0)) into _move_cursors, plus a cursor_present (B,T,H,W) bool flag set/cleared on cursor move; in direction selection, override get_main_dir with get_close_dir (code_dir bitmask -> LOCAL_DIR[(code-1)*2+sens]) wherever cursor_present is set. Keep the (B,T,2) dydx in {-1,0,1} step() contract unchanged.
5. Rewrite _move_fighters as the rank-loop priority-claim: for rank r in 0..4 compute each unmoved fighter's candidate cell from FIGHTER_MOVE_DIR[table][dir]+OFFSET (table=(tick//3)%2, start=(tick//6)%12); eligible = passable & occ_idx==-1; elect winner per target cell via scatter_reduce(amin) of slot index; commit winners (occ_idx[target]=idx, occ_idx[source]=-1, update fx/fy), repeat across ranks. Document the amin tie-break as the deterministic non-bit-exact analogue of C serial order.
6. Rewrite _resolve_combat for the blocked-on-all-5 set only: compute per-team attack/defense/new_health once/tick from the integer rubber-band chain with nested fixsqrt (torch double sqrt floored to int64, kept nested — do NOT collapse to **0.25), clamp each to [1,16383]. Apply the A/B/C/D if-else (front attack p0, side >>4 on p1/p2, heal p0 ally) as mutually-exclusive masks; accumulate damage per target via scatter_reduce(sum); on fhealth<0 rebase n=ceil(-h/new_health) then fteam=attacker_team (min-index attacker). Delete the old delete-and-remove path entirely.
7. Rebuild team_oh (B,T,H,W) and health (B,H,W) as derived scatter views at tick end (index_put_ from fteam/fhealth into flattened b*H*W indices) so get_observation, ppo._team_share, eval winner=argmax, and web/server's oh.argmax(0)/oh.sum(0) all keep working with zero edits. Keep get_observation's channel layout (walls, T presence, T normalized gradient with 2_000_000 ceiling mapped to 0, T team-health) identical.
8. Implement _check_eliminations with dense-id compaction: first team with ACTIVE_FIGHTERS==0 per game eliminated (only one/tick); playing_teams--; deactivate cursor; shift team ids > eliminated down by 1 across fteam, cursor_pos, cursor_val, team_alive, ACTIVE_FIGHTERS, AND gradient[:,j]=gradient[:,j+1] / cursor_present. Add a 3+team elimination integration test asserting dense ids and sum(ACTIVE_FIGHTERS)==army_size after each elimination.
9. Add a CPU serial-reference parity harness (B=1, N<=64, fixed map + fixed clock sequence) that literally serially ports the C spread/move/combat, and a conservation test asserting count == army_size every tick over a long random rollout. Accept and document the bounded cell-divergence from C as an explicit non-goal (RL needs determinism + invariants, not C bit-parity).
10. Keep grad_iters as the accepted (repurposed) kwarg and accept legacy attack/defense kwargs as ignored-with-warning so train.py:103, batch_runner.py:37, eval.py:101, web/server.py:67 keep constructing LiquidWarEngine unchanged. After tests pass, build+deploy the trainer + liquidwar-play images and browser-verify web/server.py with a freshly trained checkpoint before declaring shipped (tests passing != shipped); update PROJECT_TRACKER (closes Clone P2 / WS task #31) and the liquidwar memory note, and flag that all results/rl checkpoints are invalidated by the MDP change so an in-flight run isn't silently wasted.

## Grafted ideas

- From A: build a CPU serial-reference parity harness — a literal sequential port of spread_single_gradient + move_fighters + resolve_combat on tiny (B=1, N<=64) fixed maps and a fixed GLOBAL_CLOCK sequence — and assert the batched engine matches it within a documented cell-divergence tolerance. Keep it as a throwaway test fixture (pytest), NOT as a permanently-shipped second engine, so you get A's parity-testing benefit without A's maintenance drag of a divergent presence engine.
- From A/B: land the swap so the OLD presence engine is removed from the import path but preserved in git history; do not keep a faithful=False mode wired into train.py/server.py, to avoid the 'old checkpoint on new MDP looks random' footgun both designs warn about.
- From C: keep an explicit per-cell side_size (B,H,W) int field in the representation even though the faithful default is side_size=1. This is the open/closed seam for adding the MESH quadtree edge-weights later without touching the gradient relaxation code — graft C's 'keep the channel' framing onto B's default-1.
- From C: adopt its conv2d-based neighbor census ONLY for the per-team ACTIVE_FIGHTERS recount and the combat coefficient inputs (population balance), where a count is genuinely all that's needed — but keep B's indexed-slot occ_idx for the collision/conversion path where identity matters. This isolates the cheap batched reduction from the identity-preserving slot logic.
- From B: repurpose the existing grad_iters constructor kwarg as the K wavefront-passes budget (verified required by train.py:103 and batch_runner.py:37) rather than adding a new kwarg, and accept the legacy attack/defense kwargs as ignored-with-deprecation-warning so no call site breaks.