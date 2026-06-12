"use strict";
// Procedural audio for Liquid War — no samples, everything synthesized.
//
// MUSIC: an "Interstellar organ x bioluminescent ambient" hybrid —
//   - organ pads: additive sine harmonics, 4s attacks, slow chord drift
//     through an Aeolian progression (voice-led, never jumps)
//   - tectonic sub: a slow-swelling low triangle an octave below the root
//   - sparkles: sparse pentatonic bell-plucks through delay (the
//     bioluminescent layer; density falls as combat rises)
//   - generated impulse-response reverb (procedural cathedral)
// ADAPTIVE: combat intensity opens the filter, speeds the chord clock and
// thins the sparkles; LOSING detunes a tension voice in; Doom (either side)
// adds a pulsing 36.7Hz drone; Maelstrom adds a swirling band-passed wash.
// SFX: conversion thuds scaled by bleed size, tiny blips for skirmish,
// per-stance tap notes, victory/defeat stingers.
const LWAUDIO = (() => {
  let ctx = null, master, musicBus, sfxBus;
  let musicVol = 0.5, sfxVol = 0.7, enabled = false;
  let chordTimer = 0, sparkTimer = 0, chordIdx = 0, voices = [];
  let sub = null, doomDrone = null, maelWash = null, tension = null;
  let sig = { intensity: 0, losing: 0, doom: 0, mael: 0 };
  let lastThud = 0, lastBlip = 0;

  // Aeolian drift in A: rich, dark, never resolves hard.
  const CHORDS = [               // semitone offsets from A2 (110 Hz)
    [0, 7, 12, 15],              // Am  (A C E)
    [-4, 3, 12, 19],             // F   (F A C)
    [-2, 5, 14, 17],             // G   (G B D)
    [3, 10, 15, 22],             // C   (C E G)
    [-5, 2, 10, 14],             // Em over E
    [0, 7, 12, 17],              // Asus
  ];
  const PENTA = [0, 3, 5, 7, 10, 12, 15, 19, 24];   // A minor pentatonic
  const f = (semi, base = 110) => base * Math.pow(2, semi / 12);

  function makeVerb() {
    const len = ctx.sampleRate * 2.8, ir = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let c = 0; c < 2; c++) {
      const d = ir.getChannelData(c);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 2.6);
    }
    const v = ctx.createConvolver(); v.buffer = ir; return v;
  }

  function init() {
    if (ctx) { ctx.resume(); return; }
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    master = ctx.createDynamicsCompressor();
    master.threshold.value = -18; master.ratio.value = 6;
    master.connect(ctx.destination);
    const verb = makeVerb();
    const verbGain = ctx.createGain(); verbGain.gain.value = 0.5;
    verb.connect(verbGain).connect(master);
    musicBus = ctx.createGain(); musicBus.gain.value = musicVol;
    musicBus.connect(master); musicBus.connect(verb);
    sfxBus = ctx.createGain(); sfxBus.gain.value = sfxVol;
    sfxBus.connect(master); sfxBus.connect(verb);
    startSub(); startDoom(); startMael();
  }

  // --- organ voice: additive harmonics with cathedral attack ---
  function organ(freq, gain, attack = 4, dur = 22) {
    const g = ctx.createGain(); g.gain.value = 0;
    const lp = ctx.createBiquadFilter(); lp.type = "lowpass";
    lp.frequency.value = 700 + 2400 * sig.intensity;
    g.connect(lp).connect(musicBus);
    const t = ctx.currentTime;
    g.gain.linearRampToValueAtTime(gain, t + attack);
    g.gain.setTargetAtTime(0, t + dur - 7, 2.4);
    const parts = [[1, 1], [2, 0.42], [3, 0.21], [4, 0.1], [0.5, 0.25]];
    const oscs = parts.map(([m, a]) => {
      const o = ctx.createOscillator(); o.type = "sine";
      o.frequency.value = freq * m;
      o.detune.value = (Math.random() - 0.5) * 7;
      const og = ctx.createGain(); og.gain.value = a;
      o.connect(og).connect(g); o.start(t); o.stop(t + dur + 1);
      return o;
    });
    return { g, oscs };
  }

  function nextChord() {
    chordIdx = (chordIdx + (Math.random() < 0.75 ? 1 : CHORDS.length - 1)) % CHORDS.length;
    const semis = CHORDS[chordIdx];
    voices = semis.map((s, i) => organ(f(s), i === 0 ? 0.16 : 0.10 - i * 0.015));
    // losing: a minor-second tension voice bleeds in with the deficit
    if (sig.losing > 0.15) {
      tension = organ(f(semis[0] + 1) / 2, 0.05 * Math.min(1, sig.losing * 2.5), 6);
    }
  }

  function startSub() {
    const o = ctx.createOscillator(); o.type = "triangle"; o.frequency.value = 55;
    const g = ctx.createGain(); g.gain.value = 0;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.045;
    const lg = ctx.createGain(); lg.gain.value = 0.05;
    lfo.connect(lg).connect(g.gain);
    o.connect(g).connect(musicBus); o.start(); lfo.start();
    sub = { o, g };
  }

  function startDoom() {                       // 36.7Hz = D1: the well's voice
    const o = ctx.createOscillator(); o.frequency.value = 36.7;
    const o2 = ctx.createOscillator(); o2.frequency.value = 36.95;   // beat ~0.25Hz
    const g = ctx.createGain(); g.gain.value = 0;
    const trem = ctx.createOscillator(); trem.frequency.value = 0.8; // the heartbeat
    const tg = ctx.createGain(); tg.gain.value = 0.04;
    trem.connect(tg).connect(g.gain);
    o.connect(g); o2.connect(g); g.connect(musicBus);
    o.start(); o2.start(); trem.start();
    doomDrone = g;
  }

  function startMael() {                       // swirling band-passed wash
    const len = ctx.sampleRate * 2, buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    const src = ctx.createBufferSource(); src.buffer = buf; src.loop = true;
    const bp = ctx.createBiquadFilter(); bp.type = "bandpass"; bp.Q.value = 2.2;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.11;
    const lg = ctx.createGain(); lg.gain.value = 320;
    bp.frequency.value = 420; lfo.connect(lg).connect(bp.frequency);
    const g = ctx.createGain(); g.gain.value = 0;
    src.connect(bp).connect(g).connect(musicBus);
    src.start(); lfo.start();
    maelWash = g;
  }

  function sparkle() {                          // bioluminescent bell-pluck
    const semi = PENTA[(Math.random() * PENTA.length) | 0];
    const o = ctx.createOscillator(); o.type = "sine";
    o.frequency.value = f(semi, 440) * (Math.random() < 0.3 ? 2 : 1);
    const g = ctx.createGain();
    const t = ctx.currentTime, amp = 0.05 + 0.04 * Math.random();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(amp, t + 0.02);
    g.gain.setTargetAtTime(0, t + 0.03, 0.5);
    const dl = ctx.createDelay(); dl.delayTime.value = 0.31;
    const fb = ctx.createGain(); fb.gain.value = 0.35;
    o.connect(g); g.connect(musicBus);
    g.connect(dl); dl.connect(fb).connect(dl); dl.connect(musicBus);
    o.start(t); o.stop(t + 4);
  }

  // ---- SFX ----
  function thud(strength) {                     // a front bleeding: tectonic hit
    const t = ctx.currentTime;
    if (t - lastThud < 0.18) return;
    lastThud = t;
    const o = ctx.createOscillator(); o.type = "sine";
    o.frequency.setValueAtTime(150, t);
    o.frequency.exponentialRampToValueAtTime(48, t + 0.28);
    const g = ctx.createGain();
    g.gain.setValueAtTime(Math.min(0.5, 0.1 + strength * 0.4), t);
    g.gain.setTargetAtTime(0, t + 0.05, 0.12);
    o.connect(g).connect(sfxBus); o.start(t); o.stop(t + 0.8);
  }
  function blip() {                             // skirmish crackle
    const t = ctx.currentTime;
    if (t - lastBlip < 0.09) return;
    lastBlip = t;
    const o = ctx.createOscillator(); o.type = "triangle";
    o.frequency.value = 900 + Math.random() * 1600;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.025, t);
    g.gain.setTargetAtTime(0, t + 0.01, 0.05);
    o.connect(g).connect(sfxBus); o.start(t); o.stop(t + 0.3);
  }
  function stanceTap(i) {                       // per-stance confirmation note
    if (!ctx) return;
    const t = ctx.currentTime;
    const o = ctx.createOscillator(); o.type = "sine";
    o.frequency.value = f(PENTA[i % PENTA.length], 330);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.12, t);
    g.gain.setTargetAtTime(0, t + 0.04, 0.18);
    o.connect(g).connect(sfxBus); o.start(t); o.stop(t + 1);
  }
  function stinger(won) {
    if (!ctx) return;
    const base = won ? [0, 7, 12, 19] : [0, -3, -8];
    base.forEach((s, i) => {
      const o = ctx.createOscillator(); o.type = "sine";
      o.frequency.value = f(s, won ? 220 : 165);
      const g = ctx.createGain();
      const t = ctx.currentTime + i * (won ? 0.12 : 0.3);
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.14, t + 0.1);
      g.gain.setTargetAtTime(0, t + 0.4, won ? 0.5 : 1.4);
      o.connect(g).connect(sfxBus); o.start(t); o.stop(t + 5);
    });
  }

  // ---- per-frame driver (call from the render loop) ----
  let lastT = 0, wasDone = false;
  function update(s) {
    if (!ctx || !enabled) return;
    sig = s;
    const now = ctx.currentTime, dt = Math.min(0.1, now - lastT || 0.016);
    lastT = now;
    chordTimer -= dt; sparkTimer -= dt;
    if (chordTimer <= 0) {
      nextChord();
      chordTimer = 22 - 10 * sig.intensity + Math.random() * 4;
    }
    if (sparkTimer <= 0) {
      if (Math.random() < 0.8 - 0.5 * sig.intensity) sparkle();
      sparkTimer = 1.2 + Math.random() * 3.5 + 3 * sig.intensity;
    }
    sub.g.gain.setTargetAtTime(0.10 + 0.10 * sig.intensity, now, 1.5);
    doomDrone.gain.setTargetAtTime(sig.doom ? 0.20 : 0, now, sig.doom ? 1.2 : 0.6);
    maelWash.gain.setTargetAtTime(sig.mael ? 0.12 : 0, now, 0.8);
    if (s.done && !wasDone) stinger(s.won);
    wasDone = s.done;
  }

  function setEnabled(v) {
    enabled = v;
    if (v) init(); else if (ctx) ctx.suspend();
    if (v && ctx) ctx.resume();
  }
  function setVolumes(m, sx) {
    musicVol = m; sfxVol = sx;
    if (musicBus) musicBus.gain.value = m;
    if (sfxBus) sfxBus.gain.value = sx;
  }
  return { update, thud, blip, stanceTap, setEnabled, setVolumes,
           get enabled() { return enabled; } };
})();
