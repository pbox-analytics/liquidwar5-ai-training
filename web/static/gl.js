"use strict";
// WebGL2 renderer for Liquid War — replaces the Canvas2D mote pipeline.
//
// Why GL: per-mote individuality (size/brightness/twinkle computed in the
// vertex shader), real multi-pass bloom, native-device-pixel rendering, and
// framebuffer trails — none of which Canvas2D could do at 16k motes / 60fps,
// and all of which a phone GPU handles easily (this is the mobile-ready layer).
//
// Pipeline per rAF:
//   1. army FBO   — instanced capsule quads (3-snapshot quadratic-arc glide,
//                   velocity streak = capsule length) + fx points (sparks,
//                   conversion flashes), premultiplied alpha.
//   2. trail FBO  — decay-multiply, then add the army layer (history glow).
//   3. bloom      — downsample army 1/4, two separable gaussian passes.
//   4. composite  — background (vignette + noise) + rim-lit beveled walls
//                   (from the grid texture) + trail + army + bloom, soft
//                   filmic clip. One fullscreen pass.
//   5. cursors    — ring/disc point sprites straight to screen (crisp, on top).
//
// Coordinates: grid space (x: 0..W, y: 0..H, y down) -> clip with a y flip.
// All offscreen targets share that flip, so only the walls texture (row 0 =
// grid top) needs a v-flip at sample time.
const LWGL = (() => {

const MOTE_VS = `#version 300 es
precision highp float;
layout(location=1) in vec2 aPA;     // (y,x) int16 grid coords — prevprev
layout(location=2) in vec2 aPB;     // prev
layout(location=3) in vec2 aPC;     // cur
layout(location=4) in float aTeam;
layout(location=5) in float aDens;  // local 8x8-bin density, 0..255
layout(location=6) in vec4 aRnd;    // static per-mote randoms, 0..1
uniform vec2 uGrid;                 // (W, H)
uniform float uF, uCurve, uTail, uSize, uAlpha, uTime, uTeams;
uniform vec4 uHole;                 // Doom: x, y, horizon radius, strength — the UNITS are the accretion fire
uniform float uHoleSpin;
uniform vec3 uColors[6];
uniform vec2 uCurs[6];              // per-team cursor (x,y) — the army's "heart"
uniform float uBeat[6];             // heartbeat rate; SIGN = wave direction (Doom beats inward)
uniform vec4 uFlush[6];             // combat flush: x, y, age (s), amplitude
uniform float uCursSpd[6];          // cursor speed (cells/s) -> bow-wave wake
uniform float uLife;                // master dial for the whole organic layer
out vec2 vUV;                       // capsule-local coords (grid units)
out vec2 vHL;                       // (halfLen, halfWidth)
out vec4 vColor;                    // premultiplied
void main() {
  // corner of the quad from gl_VertexID: (-1,-1) (1,-1) (-1,1) (1,1) strip
  vec2 cor = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, float(gl_VertexID >> 1) * 2.0 - 1.0);
  if (aTeam >= uTeams) {            // dead / out-of-game mote: degenerate quad
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); vUV = vec2(0.0); vHL = vec2(1.0); vColor = vec4(0.0);
    return;
  }
  vec2 A = vec2(aPA.y, aPA.x), B = vec2(aPB.y, aPB.x), C = vec2(aPC.y, aPC.x);
  vec2 K = B + (B - A) * uCurve;                       // control = prev + curve*incoming velocity
  vec2 pos = mix(mix(B, K, uF), mix(K, C, uF), uF);    // quadratic arc glide
  // Streak along the CENTRAL-DIFFERENCE velocity (A->C), not the raw last step
  // (B->C): per-tick steps snap to the 8 grid directions, so raw streaks render
  // everything at hard 45-degree angles; averaging two steps halves the angular
  // quantization and turning swirls read as curves.
  vec2 vel = (C - A) * 0.5;
  float vl = min(length(vel), 8.0);                    // teleport guard (new game / conversion warp)
  // --- LIFE IN THE STILL MASS: three animations that only emerge as a mote
  // comes to rest (gated by 1-speed), so flow keeps its streaks and a parked
  // army reads as a living, simmering organism instead of frozen pixels.
  float still = 1.0 - smoothstep(0.25, 1.5, vl);
  float dnp = clamp(aDens / 22.0, 0.0, 1.0);
  // 1. HEARTBEAT: a pulse radiates out from the team's cursor through the
  //    packed body — both a brightness wave (used below) and a subtle radial
  //    breathing displacement, strongest in the dense core.
  int ti = int(aTeam + 0.5);
  vec2 heart = uCurs[ti];
  float hd = max(distance(pos, heart), 1e-3);
  vec2 outw = (pos - heart) / hd;
  // rate + direction follow the held STANCE (uBeat sign: Doom beats inward)
  float beat = sin(hd * 0.30 - uTime * uBeat[ti]);
  pos += outw * (0.22 * uLife * still * dnp * beat);
  // 2. MICRO-CONVECTION: parked motes simmer on tiny personal orbits.
  float mfreq = 0.8 + 2.2 * aRnd.y;
  pos += uLife * still * 0.20 * vec2(sin(uTime * mfreq + aRnd.w * 6.2832),
                                     cos(uTime * mfreq * 1.3 + aRnd.z * 6.2832));
  // 4. CURSOR WAKE: a moving cursor parts its own parked army — a local
  //    bow-wave displacement that dies with distance and cursor speed.
  pos += outw * (exp(-hd * hd / 40.0) * min(uCursSpd[ti] * 0.05, 1.3) * still * uLife);
  vec2 dir = vl > 1e-4 ? normalize(vel) : vec2(1.0, 0.0);
  vec2 perp = vec2(-dir.y, dir.x);
  float dn = clamp(aDens / 22.0, 0.0, 1.0);            // normalized local density
  // per-mote size jitter; packed core motes shrink a touch so points stay readable
  float halfW = 0.5 * uSize * (0.75 + 0.5 * aRnd.x) * (1.0 - 0.25 * smoothstep(0.5, 1.0, dn));
  float halfL = min(vl * uTail * 0.5, 5.0);            // velocity streak length
  // 5. EDGE CILIA: resting RIM motes grow tiny waving filaments pointing
  //    outward — the parked blob gets a living, anemone-like membrane.
  float rim = (1.0 - smoothstep(0.12, 0.45, dnp)) * step(0.02, dnp);
  float cil = still * rim * uLife;
  if (cil > 0.01) {
    // SAFE normalize: dir ~= -outw at cil ~0.5 mixes to ~zero — normalize(0)
    // is NaN, and one NaN vertex paints white on drivers that store NaN as 255
    vec2 m = mix(dir, outw, cil);
    float ml = length(m);
    if (ml > 1e-4) dir = m / ml;
    perp = vec2(-dir.y, dir.x);
    halfL += cil * 0.9 * (0.45 + 0.55 * sin(uTime * (2.0 + 3.0 * aRnd.z) + aRnd.w * 6.2832));
  }
  vec2 world = pos - dir * halfL + dir * cor.x * (halfL + halfW) + perp * cor.y * halfW;
  gl_Position = vec4(world.x / uGrid.x * 2.0 - 1.0, 1.0 - 2.0 * world.y / uGrid.y, 0.0, 1.0);
  vUV = vec2(cor.x * (halfL + halfW), cor.y * halfW);
  vHL = vec2(halfL, halfW);
  // colour: team base, per-mote brightness jitter, density-gated twinkle (the
  // packed core shimmers like a living mass), darker deep core (volume), ~12%
  // sparkle motes lifted toward white.
  vec3 base = uColors[int(aTeam + 0.5)];
  float tw = 0.9 + 0.28 * sin(uTime * (1.5 + 3.0 * aRnd.z) + aRnd.w * 6.2832);
  float sparkle = step(0.88, aRnd.x);
  float bright = (0.95 + 0.4 * aRnd.y) * mix(1.0, tw, 0.25 + 0.55 * dn);
  bright *= mix(1.0, 0.78, smoothstep(0.5, 1.0, dn));
  bright *= 1.0 + 0.18 * uLife * still * dn * beat;    // the heartbeat GLOWS as it travels
  vec3 colr = mix(base, vec3(1.0), 0.45 * sparkle) * (bright + 0.8 * sparkle * max(tw - 0.9, 0.0));
  // 3. LAVA DRIFT: slow warm patches wander through the resting body — a
  //    low-frequency spatial wave nudging colour temperature (subtle).
  float lava = 0.5 + 0.5 * sin(pos.x * 0.045 + pos.y * 0.06 + uTime * 0.45);
  colr *= mix(vec3(1.0), vec3(1.10, 0.97, 0.87), 0.35 * uLife * still * lava);
  // 6. COMBAT FLUSH: when this team is bleeding conversions, a red-shifted
  //    shockwave radiates through its body FROM the wound (the front line).
  vec4 fl = uFlush[ti];
  if (fl.w > 0.001) {
    float fd = distance(pos, fl.xy);
    float ring = exp(-pow((fd - fl.z * 45.0) / 9.0, 2.0)) * fl.w * uLife;
    colr = mix(colr, vec3(1.25, 0.38, 0.30), min(ring * 0.6, 0.75));
  }
  // Doom: motes near the horizon ARE the accretion disk — they heat toward
  // amber-white, doppler-beamed (the approaching side burns brighter).
  if (uHole.w > 0.001) {
    float hdist = distance(pos, uHole.xy);
    float heat = min(uHole.w, 1.3) * exp(-pow((hdist - uHole.z * 1.15) / (uHole.z * 0.6), 2.0));
    float dop = 1.0 + 0.7 * clamp(-(pos.x - uHole.x) * uHoleSpin / max(uHole.z, 1.0), -1.0, 1.0);
    colr = mix(colr, vec3(1.0, 0.82, 0.55) * (1.2 * dop), min(heat * 0.85, 0.9));
  }
  float a = uAlpha * mix(0.6, 1.0, smoothstep(0.02, 0.3, dn));   // sparse rim more translucent
  vColor = vec4(colr * a, a);
}`;

const MOTE_FS = `#version 300 es
precision mediump float;
in vec2 vUV; in vec2 vHL; in vec4 vColor;
out vec4 o;
void main() {
  // capsule SDF in grid units; soft edge ~ half the width
  float d = length(vec2(max(abs(vUV.x) - vHL.x, 0.0), vUV.y)) - vHL.y;
  float a = clamp(-d / (vHL.y * 0.55 + 0.01), 0.0, 1.0);
  o = vColor * a;
}`;

// fx points: sparks, conversion flashes, cursor rings. shape 0 = soft disc, 1 = ring.
const FX_VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;    // grid (x, y)
layout(location=1) in float aSize;  // diameter, grid units
layout(location=2) in float aShape;
layout(location=3) in vec4 aColor;  // straight alpha
uniform vec2 uGrid;
uniform float uScale;               // device px per grid cell
out vec4 vColor; out float vShape;
void main() {
  gl_Position = vec4(aPos.x / uGrid.x * 2.0 - 1.0, 1.0 - 2.0 * aPos.y / uGrid.y, 0.0, 1.0);
  gl_PointSize = max(aSize * uScale, 1.5);
  vColor = vec4(aColor.rgb * aColor.a, aColor.a);
  vShape = aShape;
}`;

const FX_FS = `#version 300 es
precision mediump float;
in vec4 vColor; in float vShape;
out vec4 o;
void main() {
  float r = length(gl_PointCoord * 2.0 - 1.0);
  float a = vShape > 0.5 ? smoothstep(0.16, 0.02, abs(r - 0.72))   // ring
                         : pow(max(1.0 - r, 0.0), 1.8);            // soft disc
  o = vColor * a;
}`;

const FS_VS = `#version 300 es
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}`;

const FLAT_FS = `#version 300 es
precision mediump float;
uniform vec4 uColor;
out vec4 o;
void main() { o = uColor; }`;

const BLIT_FS = `#version 300 es
precision mediump float;
uniform sampler2D uTex; uniform vec2 uRes;
out vec4 o;
void main() { o = texture(uTex, gl_FragCoord.xy / uRes); }`;

const BLUR_FS = `#version 300 es
precision mediump float;
uniform sampler2D uTex; uniform vec2 uRes; uniform vec2 uDir;   // (1,0) or (0,1), texel units
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / uRes, px = uDir / uRes;
  o = texture(uTex, uv) * 0.227
    + (texture(uTex, uv + px * 1.385) + texture(uTex, uv - px * 1.385)) * 0.316
    + (texture(uTex, uv + px * 3.231) + texture(uTex, uv - px * 3.231)) * 0.070;
}`;

const COMP_FS = `#version 300 es
precision highp float;
uniform sampler2D uWalls, uArmy, uTrail, uBloom;
uniform vec2 uRes, uGrid;
uniform float uBloomK, uTrailK;
uniform float uBlackout;             // 0 = normal, 1 = black-out (units are the only light)
uniform vec4 uHole;                 // Doom black hole: x, y (grid), horizon radius (cells), strength (0=off)
uniform vec2 uHoleT;                // (time, spin direction)
uniform vec4 uWhirl;                // Maelstrom: x, y (grid), radius (cells), strength (0=off)
uniform float uWhirlDir;            // current direction (the owner's Q/E)
uniform float uWhirlMode;           // 0 undertow / 1 ejecta / 2 shear — three different storms
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  // --- Doom: Gargantua-style gravitational lensing. Light from the army/trail
  // layers is sampled along rays BENT toward the hole, so the scene visibly
  // warps and smears around it; matter (walls/bg) is sampled straight.
  vec2 uvg = vec2(uv.x * uGrid.x, (1.0 - uv.y) * uGrid.y);   // this fragment in grid coords
  vec2 hd = uvg - uHole.xy;
  float hr = length(hd);
  vec2 uvL = uv;
  if (uHole.w > 0.001) {
    float rh = uHole.z;
    float pull = uHole.w * rh * rh * 2.6 / (hr * hr + rh * rh * 0.6);   // bend, strongest near the horizon
    vec2 dg = (hd / max(hr, 1e-3)) * min(pull, hr * 0.85);              // never pull past the centre
    uvL = uv - vec2(dg.x / uGrid.x, -dg.y / uGrid.y);
  }
  // --- Maelstrom: whirlpool refraction. Light from the army/trail layers is
  // sampled along ROTATED rays about the well (rotation decays with radius),
  // so the scene visibly swirls into the current — the water analog of the
  // Doom lensing above (the two compose).
  vec2 wd = uvg - uWhirl.xy;
  float wr = length(wd);
  if (uWhirl.w > 0.001) {
    float wR = uWhirl.z;
    float wang = uWhirl.w * uWhirlDir * 1.1 * exp(-(wr * wr) / (wR * wR));
    float cs = cos(wang), sn = sin(wang);
    vec2 rwd = vec2(cs * wd.x - sn * wd.y, sn * wd.x + cs * wd.y) - wd;
    uvL -= vec2(rwd.x / uGrid.x, -rwd.y / uGrid.y);
  }
  // wall texture: R = crisp mask, G = blurred mask (wide ramp for bevel + shadow)
  vec2 wm = texture(uWalls, vec2(uv.x, 1.0 - uv.y)).rg;     // v-flip: row 0 = top
  // background: gentle centre lift, vignette, blue-noise dither to kill banding
  float dC = distance(uv, vec2(0.5));
  vec3 col = mix(vec3(0.022, 0.028, 0.052), vec3(0.008, 0.010, 0.020), smoothstep(0.2, 0.85, dC));
  col += (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) * 0.012;
  // walls: soft contact shadow outside, wide beveled rim-light inside the edge
  float wall = smoothstep(0.42, 0.58, wm.r);
  float soft = wm.g;
  col *= 1.0 - 0.55 * (1.0 - wall) * smoothstep(0.06, 0.5, soft);
  vec2 g = vec2(dFdx(soft), dFdy(soft));
  float lit = 0.5 + 0.5 * dot(normalize(g + vec2(1e-5)), normalize(vec2(-0.5, 0.86)));
  float bevel = smoothstep(0.5, 0.95, soft) * (1.0 - smoothstep(0.95, 1.0, soft));
  vec3 wallCol = vec3(0.052, 0.066, 0.108) + vec3(0.02, 0.03, 0.05) * soft;
  wallCol += vec3(0.17, 0.23, 0.38) * bevel * lit;          // brighter rim — barriers read on the zoomed-out board
  // BLACK-OUT MODE: the world goes dark and the UNITS are the only light —
  // their bloom (the blurred army glow, amb below) spills onto nearby walls
  // and floor, so barriers are revealed exactly where the action is. Far-off
  // walls fade to black until a swarm approaches. Lean into the dark.
  vec3 amb = texture(uBloom, uvL).rgb;                      // unit ambient light field
  if (uBlackout > 0.0) {
    col = mix(col, col * 0.10, uBlackout);                  // open floor goes near-black
    vec3 wDark = vec3(0.014, 0.018, 0.032)                  // wall in shadow
               + amb * 3.2                                  // lit by nearby units
               + vec3(0.10, 0.14, 0.24) * bevel * lit * (0.3 + 6.0 * amb.g);  // rim catches the glow
    wallCol = mix(wallCol, wDark, uBlackout);
  }
  col = mix(col, wallCol, wall);
  col += texture(uTrail, uvL).rgb * uTrailK;                // motion history glow (lensed)
  vec4 ar = texture(uArmy, uvL);                            // crisp current frame, premult over (lensed)
  col = ar.rgb + col * (1.0 - ar.a);
  col += amb * uBloomK;                                     // glow (lensed; reuses amb)
  // --- Doom: the hole itself. A doppler-bright accretion band (one side
  // blue-shifted brighter, like Gargantua), slow spiral arms feeding it, a
  // hot thin photon ring, and an event horizon that swallows ALL light.
  // Doom: NO painted portrait — the units themselves form Gargantua (engine
  // ring + blade formation; mote shader heats them amber near the horizon).
  // Here only the physics of light: a soft shadow zone, and the event horizon
  // swallowing everything that crosses it. The lensing above bends the UNITS'
  // light into the halo arcs.
  if (uHole.w > 0.001) {
    float rh = uHole.z;
    col *= 1.0 - 0.45 * min(uHole.w, 1.0) * exp(-pow(hr / (rh * 1.2), 2.0));
    // heat GRADING, not painting: whatever light exists near the horizon —
    // orbiting units and their infall trails — is re-coloured toward amber,
    // so the accretion fire is made of the army itself.
    float hot = min(uHole.w, 1.0) * exp(-pow((hr - rh * 1.2) / (rh * 0.8), 2.0));
    col *= mix(vec3(1.0), vec3(1.45, 1.02, 0.55), hot);
    col *= smoothstep(rh * 0.85, rh * 1.02, hr);            // the horizon: pure black
  }
  // --- Maelstrom: cool spiraling shimmer riding the refraction — ripple
  // rings drifting inward, slight darkening toward the eye of the storm.
  if (uWhirl.w > 0.001) {
    float wR = uWhirl.z;
    float wmask = min(uWhirl.w, 1.0) * exp(-(wr * wr) / (wR * wR * 1.3));
    // THE MODES READ DIFFERENTLY AT A GLANCE:
    //  undertow — cool rings DRIFTING INWARD, dark swallowing eye
    //  ejecta   — warm rings BLASTING OUTWARD, bright violent rim
    //  shear    — no radial motion at all: silver spokes whipping around
    if (uWhirlMode < 0.5) {                  // undertow
      float rip = 0.5 + 0.5 * sin(wr * 0.7 - uHoleT.x * 6.0);
      col += vec3(0.04, 0.13, 0.20) * rip * wmask;
      col *= 1.0 - 0.30 * wmask;             // the eye pulls light DOWN
    } else if (uWhirlMode < 1.5) {           // ejecta
      float rip = 0.5 + 0.5 * sin(wr * 0.7 + uHoleT.x * 9.0);
      col += vec3(0.22, 0.15, 0.07) * rip * wmask;
      float rim = exp(-pow((wr - wR * 0.85) / (wR * 0.22), 2.0)) * min(uWhirl.w, 1.0);
      col += vec3(0.35, 0.22, 0.10) * rim;   // the spray edge burns
    } else {                                 // shear
      float ang = atan(wd.y, wd.x);
      float spoke = 0.5 + 0.5 * sin(ang * 7.0 - uWhirlDir * uHoleT.x * 5.0);
      col += vec3(0.16, 0.18, 0.22) * spoke * spoke * wmask;
      col *= 1.0 - 0.10 * wmask;
    }
  }
  col = 1.0 - exp(-col * 1.8);                              // soft filmic clip (no harsh saturate)
  col *= 1.0 - 0.30 * smoothstep(0.55, 1.05, dC);           // final vignette
  // NaN scrub: a poisoned uniform/sample must fail BLACK for one frame, not
  // lock the canvas white (RGBA8 stores NaN as 255 on some drivers)
  if (isnan(col.x + col.y + col.z)) col = vec3(0.0);
  o = vec4(col, 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error("shader: " + gl.getShaderInfoLog(s) + "\n" + src.split("\n").map((l, i) => (i + 1) + ": " + l).join("\n"));
  return s;
}
function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("link: " + gl.getProgramInfoLog(p));
  return p;
}
function hex2rgb(h) { return [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255]; }

function create(canvas, teamColors) {
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false, premultipliedAlpha: true });
  if (!gl) throw new Error("WebGL2 unavailable");

  const progs = {
    mote: program(gl, MOTE_VS, MOTE_FS),
    fx:   program(gl, FX_VS, FX_FS),
    flat: program(gl, FS_VS, FLAT_FS),
    blit: program(gl, FS_VS, BLIT_FS),
    blur: program(gl, FS_VS, BLUR_FS),
    comp: program(gl, FS_VS, COMP_FS),
  };
  const U = {};                                   // uniform location cache: U[prog][name]
  for (const [k, p] of Object.entries(progs)) {
    U[k] = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const inf = gl.getActiveUniform(p, i); U[k][inf.name.replace("[0]", "")] = gl.getUniformLocation(p, inf.name); }
  }
  const COLORS = new Float32Array(18);
  teamColors.slice(0, 6).forEach((c, i) => COLORS.set(hex2rgb(c), i * 3));
  // lobby color picks re-skin the armies live: uColors uploads every render,
  // so rewriting the array is the whole job
  function setPalette(hexList) {
    hexList.slice(0, 6).forEach((c, i) => COLORS.set(hex2rgb(c), i * 3));
  }

  // ---- mote buffers: a 3-buffer ring of int16 positions + per-frame team/density,
  // plus a static per-mote random table.
  const CAP = 20000;
  const posBufs = [gl.createBuffer(), gl.createBuffer(), gl.createBuffer()];
  const teamBuf = gl.createBuffer(), densBuf = gl.createBuffer(), rndBuf = gl.createBuffer();
  const rnd = new Uint8Array(CAP * 4);
  for (let i = 0; i < rnd.length; i++) rnd[i] = (Math.random() * 256) | 0;
  gl.bindBuffer(gl.ARRAY_BUFFER, rndBuf); gl.bufferData(gl.ARRAY_BUFFER, rnd, gl.STATIC_DRAW);
  let ring = 0, nMotes = 0, framesPushed = 0;
  const densScratch = new Uint8Array(CAP);
  let binCount = new Uint16Array(0);

  // ---- fx points (dynamic)
  const fxBuf = gl.createBuffer();
  const FXF = 8;                                  // floats per point: x,y,size,shape,r,g,b,a

  // ---- offscreen targets
  function makeTarget(w, h, filter) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fbo, w, h };
  }
  let army = null, trail = null, bloomA = null, bloomB = null;
  let vw = 0, vh = 0;

  // walls texture (grid-resolution mask, bilinear so the composite shader gets
  // smooth 0..1 gradients to bevel)
  const wallTex = gl.createTexture();
  let gridW = 0, gridH = 0;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(2, Math.round(canvas.clientHeight * dpr));
    if (w === vw && h === vh) return;
    vw = canvas.width = w; vh = canvas.height = h;
    army = makeTarget(w, h, gl.LINEAR);
    trail = makeTarget(w, h, gl.LINEAR);
    const bw = Math.max(2, w >> 2), bh = Math.max(2, h >> 2);
    bloomA = makeTarget(bw, bh, gl.LINEAR);
    bloomB = makeTarget(bw, bh, gl.LINEAR);
    clearTrails();
  }

  function clearTrails() {
    if (!trail) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, trail.fbo);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  }

  function setGrid(W, H, wallMask) {              // wallMask: Uint8Array W*H, 255 = wall
    gridW = W; gridH = H;
    // G channel: the mask box-blurred 3x (radius 1) -> a ~3-cell ramp at every
    // edge, so the composite shader gets a wide gradient to bevel/shadow with
    // (the crisp mask alone transitions in <1 device px and the bevel vanishes).
    let soft = Float32Array.from(wallMask), tmp = new Float32Array(soft.length);
    for (let pass = 0; pass < 3; pass++) {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {   // horizontal
        const i = y * W + x;
        tmp[i] = (soft[Math.max(i - 1, y * W)] + soft[i] + soft[Math.min(i + 1, y * W + W - 1)]) / 3;
      }
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {   // vertical
        const i = y * W + x;
        soft[i] = (tmp[y > 0 ? i - W : i] + tmp[i] + tmp[y < H - 1 ? i + W : i]) / 3;
      }
    }
    const rg = new Uint8Array(W * H * 2);
    for (let i = 0; i < wallMask.length; i++) { rg[2 * i] = wallMask[i]; rg[2 * i + 1] = soft[i]; }
    gl.bindTexture(gl.TEXTURE_2D, wallTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, W, H, 0, gl.RG, gl.UNSIGNED_BYTE, rg);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // (no clearTrails here: the client re-uploads walls only on map change
    // and clears trails explicitly there — a same-map re-upload mid-fight
    // must not wipe the motion history)
  }

  function pushFrame(pos, team, n, snap) {        // pos Int16Array (y,x)*n, team Uint8Array n
    n = Math.min(n, CAP);
    nMotes = n;
    ring = (ring + 1) % 3;
    gl.bindBuffer(gl.ARRAY_BUFFER, posBufs[ring]);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
    if (framesPushed === 0 || snap) {             // first frame / teleport SNAP: fill the whole ring
      for (const b of posBufs) { gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW); }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, teamBuf);
    gl.bufferData(gl.ARRAY_BUFFER, team, gl.DYNAMIC_DRAW);
    // local density: bin current positions into 8x8-cell buckets (volume shading)
    const gw = (gridW >> 3) + 1, gh = (gridH >> 3) + 1;
    if (binCount.length < gw * gh) binCount = new Uint16Array(gw * gh);
    else binCount.fill(0);
    for (let i = 0; i < n; i++) {
      const gy = pos[2 * i] >> 3, gx = pos[2 * i + 1] >> 3;
      if (gx >= 0 && gy >= 0 && gx < gw && gy < gh) binCount[gy * gw + gx]++;
    }
    for (let i = 0; i < n; i++) {
      const gy = pos[2 * i] >> 3, gx = pos[2 * i + 1] >> 3;
      const c = (gx >= 0 && gy >= 0 && gx < gw && gy < gh) ? binCount[gy * gw + gx] : 0;
      densScratch[i] = c > 255 ? 255 : c;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, densBuf);
    gl.bufferData(gl.ARRAY_BUFFER, densScratch.subarray(0, n), gl.DYNAMIC_DRAW);
    framesPushed++;
  }

  function bindMoteAttribs() {
    // ring roles: cur = ring, prev = ring-1, prevprev = ring-2
    const cur = posBufs[ring], prev = posBufs[(ring + 2) % 3], pp = posBufs[(ring + 1) % 3];
    for (const [loc, buf] of [[1, pp], [2, prev], [3, cur]]) {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.SHORT, false, 4, 0);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, teamBuf);
    gl.enableVertexAttribArray(4); gl.vertexAttribPointer(4, 1, gl.UNSIGNED_BYTE, false, 1, 0); gl.vertexAttribDivisor(4, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, densBuf);
    gl.enableVertexAttribArray(5); gl.vertexAttribPointer(5, 1, gl.UNSIGNED_BYTE, false, 1, 0); gl.vertexAttribDivisor(5, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, rndBuf);
    gl.enableVertexAttribArray(6); gl.vertexAttribPointer(6, 4, gl.UNSIGNED_BYTE, true, 4, 0); gl.vertexAttribDivisor(6, 1);
  }

  function drawFx(points, count, targetW, targetH, over) {
    if (!count) return;
    gl.useProgram(progs.fx);
    gl.uniform2f(U.fx.uGrid, gridW, gridH);
    gl.uniform1f(U.fx.uScale, targetW / gridW);
    gl.bindBuffer(gl.ARRAY_BUFFER, fxBuf);
    gl.bufferData(gl.ARRAY_BUFFER, points.subarray(0, count * FXF), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, FXF * 4, 0);  gl.vertexAttribDivisor(0, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, FXF * 4, 8);  gl.vertexAttribDivisor(1, 0);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, FXF * 4, 12); gl.vertexAttribDivisor(2, 0);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 4, gl.FLOAT, false, FXF * 4, 16); gl.vertexAttribDivisor(3, 0);
    // additive for the hot stuff; REVERSE-SUBTRACT for darkeners (the cursor
    // shadow pucks): dst - src clamps at black — a true shadow with no alpha
    // semantics, robust where premult-over onto the alpha:false backbuffer
    // white-screened (SwiftShader). Pass color = the GRAY LEVEL to subtract.
    if (over) gl.blendEquation(gl.FUNC_REVERSE_SUBTRACT);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArrays(gl.POINTS, 0, count);
    if (over) gl.blendEquation(gl.FUNC_ADD);
    for (const l of [0, 1, 2, 3]) gl.disableVertexAttribArray(l);
  }

  function fullscreen() { gl.drawArrays(gl.TRIANGLES, 0, 3); }

  // opts: {f, time, tail, curve, size, alpha, trailFade, trailVis, bloom, teams,
  //        fx: Float32Array, fxCount, cursorFx: Float32Array, cursorFxCount}
  function render(o) {
    resize();
    if (!framesPushed || !gridW) return;
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);

    // 1 — army layer
    gl.bindFramebuffer(gl.FRAMEBUFFER, army.fbo);
    gl.viewport(0, 0, army.w, army.h);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(progs.mote);
    bindMoteAttribs();
    gl.uniform2f(U.mote.uGrid, gridW, gridH);
    gl.uniform1f(U.mote.uF, o.f); gl.uniform1f(U.mote.uCurve, o.curve);
    gl.uniform1f(U.mote.uTail, o.tail); gl.uniform1f(U.mote.uSize, o.size);
    gl.uniform1f(U.mote.uAlpha, o.alpha); gl.uniform1f(U.mote.uTime, o.time);
    gl.uniform1f(U.mote.uTeams, o.teams);
    const mh = o.hole || { x: 0, y: 0, r: 1, a: 0, spin: 1 };
    gl.uniform4f(U.mote.uHole, mh.x, mh.y, mh.r, mh.a);
    gl.uniform1f(U.mote.uHoleSpin, mh.spin);
    const curs = new Float32Array(12);
    if (o.cursors) for (let t = 0; t < Math.min(6, o.cursors.length); t++) {
      curs[t * 2] = o.cursors[t][1] + 0.5; curs[t * 2 + 1] = o.cursors[t][0] + 0.5;  // (y,x) -> (x,y)
    }
    gl.uniform2fv(U.mote.uCurs, curs);
    gl.uniform1fv(U.mote.uBeat, o.beats || new Float32Array([2.6, 2.6, 2.6, 2.6, 2.6, 2.6]));
    gl.uniform4fv(U.mote.uFlush, o.flush || new Float32Array(24));
    gl.uniform1fv(U.mote.uCursSpd, o.cursSpd || new Float32Array(6));
    gl.uniform1f(U.mote.uLife, o.life === undefined ? 1.0 : o.life);
    gl.uniform3fv(U.mote.uColors, COLORS);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);  // premult over: dense army stays team colour
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, nMotes);
    for (const l of [1, 2, 3, 4, 5, 6]) { gl.disableVertexAttribArray(l); gl.vertexAttribDivisor(l, 0); }
    drawFx(o.fx, o.fxCount, army.w, army.h);       // sparks + flashes glow (and bloom) with the army

    // 2 — trail accumulation: decay, then add this frame's army
    gl.bindFramebuffer(gl.FRAMEBUFFER, trail.fbo);
    gl.viewport(0, 0, trail.w, trail.h);
    gl.useProgram(progs.flat);
    gl.uniform4f(U.flat.uColor, 0, 0, 0, o.trailFade);
    gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA); // dst *= (1 - fade)
    fullscreen();
    gl.useProgram(progs.blit);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, army.tex);
    gl.uniform1i(U.blit.uTex, 0); gl.uniform2f(U.blit.uRes, trail.w, trail.h);
    gl.blendFunc(gl.ONE, gl.ONE);
    fullscreen();

    // 3 — bloom: downsample 1/4 then separable gaussian (two ping-pong passes)
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, bloomA.fbo);
    gl.viewport(0, 0, bloomA.w, bloomA.h);
    gl.useProgram(progs.blit);
    gl.bindTexture(gl.TEXTURE_2D, army.tex);
    gl.uniform1i(U.blit.uTex, 0); gl.uniform2f(U.blit.uRes, bloomA.w, bloomA.h);
    fullscreen();
    gl.useProgram(progs.blur);
    gl.uniform1i(U.blur.uTex, 0); gl.uniform2f(U.blur.uRes, bloomA.w, bloomA.h);
    for (const [src, dst, dx, dy] of [[bloomA, bloomB, 1.7, 0], [bloomB, bloomA, 0, 1.7],
                                      [bloomA, bloomB, 3.4, 0], [bloomB, bloomA, 0, 3.4]]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
      gl.bindTexture(gl.TEXTURE_2D, src.tex);
      gl.uniform2f(U.blur.uDir, dx, dy);
      fullscreen();
    }

    // 4 — composite to screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, vw, vh);
    gl.useProgram(progs.comp);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, wallTex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, army.tex);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, trail.tex);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, bloomA.tex);
    gl.uniform1i(U.comp.uWalls, 0); gl.uniform1i(U.comp.uArmy, 1);
    gl.uniform1i(U.comp.uTrail, 2); gl.uniform1i(U.comp.uBloom, 3);
    gl.uniform2f(U.comp.uRes, vw, vh);
    gl.uniform2f(U.comp.uGrid, gridW, gridH);
    gl.uniform1f(U.comp.uBloomK, o.bloom); gl.uniform1f(U.comp.uTrailK, o.trailVis);
    gl.uniform1f(U.comp.uBlackout, o.blackout || 0);
    const fin = (v) => Number.isFinite(v) ? v : 0;   // one NaN uniform whites EVERY pixel
    const hole = o.hole || { x: 0, y: 0, r: 1, a: 0, spin: 1 };
    gl.uniform4f(U.comp.uHole, fin(hole.x), fin(hole.y), fin(hole.r) || 1, fin(hole.a));
    gl.uniform2f(U.comp.uHoleT, fin(o.time), fin(hole.spin) || 1);
    const wh = o.whirl || { x: 0, y: 0, r: 1, a: 0, dir: 1 };
    gl.uniform4f(U.comp.uWhirl, fin(wh.x), fin(wh.y), fin(wh.r) || 1, fin(wh.a));
    gl.uniform1f(U.comp.uWhirlDir, fin(wh.dir) || 1);
    gl.uniform1f(U.comp.uWhirlMode, fin(wh.mode));
    fullscreen();
    gl.activeTexture(gl.TEXTURE0);

    // 5 — cursors on top (crisp, not bloomed): first the dark backdrops
    // (over-blend, so they DIM the chaos beneath), then the bright marks
    gl.enable(gl.BLEND);
    if (o.cursorBg) drawFx(o.cursorBg, o.cursorBgCount, vw, vh, true);
    drawFx(o.cursorFx, o.cursorFxCount, vw, vh);
  }

  return { setGrid, pushFrame, render, clearTrails, resize, setPalette };
}

return { create };
})();
