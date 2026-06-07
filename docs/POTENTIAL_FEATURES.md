# Potential Features — slime-mold & fluid-dynamics backlog

Ideas for the GPU Liquid War clone, inspired by how *Physarum* slime mold and
*Dictyostelium* social amoebae move and communicate with their networks. The
engine is a per-team distance field (the gradient) + a flowing body of indexed
particles, so most of these map onto either **a wave the army broadcasts** or
**structure it writes into the field for its body to read** — emergent, not
scripted.

## Already built (for context)
- **Complete flood-fill gradient** — army pathfinds the whole map (octile distance → round blob).
- **Multi-candidate movement** — reroute around teammates, push *into* enemies (no stalemate).
- **Directional combat** — back-attacks (defender facing away) overtake & spread; head-on clashes grind.
- **Per-fighter jitter** — independent-looking units (no lockstep).
- **Dictyostelium traveling wave** — idle undulation + edge ripple rolling outward from the cursor.
- **Pulse / Surge** (SPACE) — peristaltic burst: human team deals 6× on contact for ~0.3s, ~3s cooldown.
- **Momentum / inertia** — velocity field blended with the gradient → weight, overshoot, collisions on clash.

---

## Proposed special moves (slime-mold network communication)

### Rally  (Dictyostelium cAMP aggregation)
- **Biology:** amoebae emit cAMP waves that pull the colony inward to a point.
- **Game:** emit an aggregation signal → all your units rush to the cursor fast (defensive snap, or pack a concentrated punch).
- **Engine hook:** briefly steepen the gradient pull / suppress idle-jitter for the human team.

### Pheromone Tube  (the slime trail / adaptive tube network)
- **Biology:** slime mold lays trails and thickens high-flow tubes into an optimized network.
- **Game:** lay a **highway** between two points — units flow ~2× along it (fast flanks/repositioning). Optionally self-reinforcing: routes used a lot get faster.
- **Engine hook:** a trail tensor that *subtracts* a bonus from the gradient on marked cells (a carved low-cost channel); decays over time.

### Split / Fragment  (plasmodium division)
- **Biology:** the plasmodium can split into independent fragments and re-merge.
- **Game:** drop a **second cursor** → the army splits between them → pincer from two sides, then merge.
- **Engine hook:** per-team gets 2 gradient seeds; the field is `min` of both → the body divides naturally.

---

## Proposed movement / feel

### Amoeboid pseudopod reach
- **Biology:** an amoeba extends a tendril toward an attractant, then the body flows in behind it (cytoplasmic streaming).
- **Game:** when the cursor jumps, briefly boost movement along the leading edge so a tendril shoots out and the mass follows — instead of the whole blob sliding uniformly.
- **Engine hook:** transient outward-mobility boost for boundary fighters nearest the new cursor heading.

### Fluid clash dynamics (extends momentum)
- With momentum in, push further: a **pressure / shockwave** at the contact line when two fast masses collide; the denser/faster side visibly punches through.
- **Engine hook:** density-aware push-back in the combat phase, scaled by relative momentum.

---

## Tuning knobs worth exposing (env, like `LW_PLAY_*`)
- momentum strength `VEL_W`, inertia `MOM`
- Pulse `PULSE_DUR / PULSE_CD / PULSE_MULT`
- idle-wave `k / ω / threshold`, jitter rate
- grid size + fighter count (already `LW_PLAY_H/W/FIGHTERS`)

## Notes
- Every move should stay **emergent** (a field the body reads) rather than scripted animation — that's what keeps it feeling organic.
- All of these sit *on top* of the momentum body, so momentum was the right foundation to build first.
