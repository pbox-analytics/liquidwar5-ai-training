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
uniform vec3 uColors[6];
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
  vec2 vel = C - B;
  float vl = min(length(vel), 8.0);                    // teleport guard (new game / conversion warp)
  vec2 dir = vl > 1e-4 ? normalize(vel) : vec2(1.0, 0.0);
  vec2 perp = vec2(-dir.y, dir.x);
  float dn = clamp(aDens / 22.0, 0.0, 1.0);            // normalized local density
  // per-mote size jitter; packed core motes shrink a touch so points stay readable
  float halfW = 0.5 * uSize * (0.75 + 0.5 * aRnd.x) * (1.0 - 0.25 * smoothstep(0.5, 1.0, dn));
  float halfL = min(vl * uTail * 0.5, 5.0);            // velocity streak length
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
  vec3 colr = mix(base, vec3(1.0), 0.45 * sparkle) * (bright + 0.8 * sparkle * max(tw - 0.9, 0.0));
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
uniform vec2 uRes;
uniform float uBloomK, uTrailK;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
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
  wallCol += vec3(0.13, 0.18, 0.30) * bevel * lit;
  col = mix(col, wallCol, wall);
  col += texture(uTrail, uv).rgb * uTrailK;                 // motion history glow
  vec4 ar = texture(uArmy, uv);                             // crisp current frame, premult over
  col = ar.rgb + col * (1.0 - ar.a);
  col += texture(uBloom, uv).rgb * uBloomK;                 // glow
  col = 1.0 - exp(-col * 1.8);                              // soft filmic clip (no harsh saturate)
  col *= 1.0 - 0.30 * smoothstep(0.55, 1.05, dC);           // final vignette
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
    clearTrails();
  }

  function pushFrame(pos, team, n) {              // pos Int16Array (y,x)*n, team Uint8Array n
    n = Math.min(n, CAP);
    nMotes = n;
    ring = (ring + 1) % 3;
    gl.bindBuffer(gl.ARRAY_BUFFER, posBufs[ring]);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
    if (framesPushed === 0) {                     // first frame: fill the whole ring
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

  function drawFx(points, count, targetW, targetH) {
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
    gl.blendFunc(gl.ONE, gl.ONE);                 // additive: they're hot
    gl.drawArrays(gl.POINTS, 0, count);
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
    gl.uniform1f(U.comp.uBloomK, o.bloom); gl.uniform1f(U.comp.uTrailK, o.trailVis);
    fullscreen();
    gl.activeTexture(gl.TEXTURE0);

    // 5 — cursors on top (crisp, not bloomed)
    gl.enable(gl.BLEND);
    drawFx(o.cursorFx, o.cursorFxCount, vw, vh);
  }

  return { setGrid, pushFrame, render, clearTrails, resize };
}

return { create };
})();
