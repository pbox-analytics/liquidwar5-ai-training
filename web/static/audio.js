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

  // PHRYGIAN drift in A (E.S. Posthumus register): the b2 — Bb against A —
  // is the dread interval, voiced in open fifths for ceremonial weight.
  const CHORDS = [               // semitone offsets from A2 (110 Hz)
    [0, 7, 12, 19],              // A5   (open fifths — the home drone)
    [1, 8, 13, 20],              // Bb5  (the b2: pure menace)
    [-2, 5, 10, 17],             // Gm
    [-4, 3, 8, 15],              // Fm-ish shade
    [0, 6, 12, 18],              // A dim color (tritone gleam)
    [-5, 2, 7, 14],              // Em over E
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
    startSub(); startDoom(); startMael(); startFlow();
  }

  // --- organ voice: additive harmonics with cathedral attack ---
  function organ(freq, gain, attack = 4, dur = 22) {
    const g = ctx.createGain(); g.gain.value = 0;
    const lp = ctx.createBiquadFilter(); lp.type = "lowpass";
    lp.frequency.value = 460 + 2100 * sig.intensity;   // darker at rest
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

  // CHOIR: a detuned saw pair through "ahh" formant bandpasses — the
  // E.S. Posthumus chorus, rising out of the organ as battle builds.
  function choir(freq, gain, dur = 22) {
    const t = ctx.currentTime;
    const g = ctx.createGain(); g.gain.value = 0;
    g.gain.linearRampToValueAtTime(gain, t + 5);
    g.gain.setTargetAtTime(0, t + dur - 7, 2.5);
    const mix = ctx.createGain(); mix.gain.value = 0;
    for (const det of [-6, 5]) {
      const o = ctx.createOscillator(); o.type = "sawtooth";
      o.frequency.value = freq; o.detune.value = det;
      o.connect(mix); o.start(t); o.stop(t + dur + 1);
    }
    mix.gain.value = 1;
    for (const [ff, fq, fa] of [[700, 9, 1.0], [1080, 11, 0.55], [2500, 14, 0.18]]) {
      const bp = ctx.createBiquadFilter(); bp.type = "bandpass";
      bp.frequency.value = ff; bp.Q.value = fq;
      const fg = ctx.createGain(); fg.gain.value = fa;
      mix.connect(bp).connect(fg).connect(g);
    }
    g.connect(musicBus);
    return g;
  }

  let activeGains = [];                        // every sounding pad/choir voice
  function retire(g) {                         // graceful handover, no tail-stack
    const now = ctx.currentTime;
    g.gain.cancelScheduledValues(now);
    g.gain.setValueAtTime(g.gain.value, now);
    g.gain.setTargetAtTime(0, now, 1.1);
  }
  function nextChord() {
    // HANDOVER: the grid changes chords every 2 bars (~5s) but voices used to
    // live ~20s — three or four chords rang at once and combat made the
    // pileup audibly sour. Old voices bow out as the new chord enters.
    activeGains.forEach(retire);
    activeGains = [];
    chordIdx = (chordIdx + (Math.random() < 0.75 ? 1 : CHORDS.length - 1)) % CHORDS.length;
    const semis = CHORDS[chordIdx];
    voices = semis.map((s, i) => organ(f(s), i === 0 ? 0.16 : 0.10 - i * 0.015));
    activeGains.push(...voices.map(v => v.g));
    // the chorus swells with the war: barely-there at peace, full voice in battle
    const ch = 0.035 + 0.11 * sig.intensity + 0.05 * (sig.doom ? 1 : 0);
    activeGains.push(choir(f(semis[0]) * 2, ch));
    if (sig.intensity > 0.35) activeGains.push(choir(f(semis[1]) * 2, ch * 0.7));
    // losing: a minor-second tension voice bleeds in with the deficit
    if (sig.losing > 0.1) {
      tension = organ(f(semis[0] + 1) / 2, 0.07 * Math.min(1, sig.losing * 2.5), 5);
      activeGains.push(tension.g);
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

  // TAIKO: ceremonial hits — a pitch-dropping low sine + a skin-noise burst.
  // Silent at peace; the pattern wakes and quickens as the battle builds
  // (or while a Doom well is open — dread has a drum).
  function taiko(strength, low = 1) {
    const t = ctx.currentTime;
    const o = ctx.createOscillator(); o.type = "sine";
    o.frequency.setValueAtTime(66 * low, t);
    o.frequency.exponentialRampToValueAtTime(34 * low, t + 0.32);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.22 * strength, t);
    g.gain.setTargetAtTime(0, t + 0.02, 0.21);
    o.connect(g).connect(musicBus); o.start(t); o.stop(t + 1.2);
    const len = ctx.sampleRate * 0.12, nb = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = nb.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3);
    const src = ctx.createBufferSource(); src.buffer = nb;
    const bp = ctx.createBiquadFilter(); bp.type = "bandpass";
    bp.frequency.value = 190; bp.Q.value = 1.1;
    const ng = ctx.createGain(); ng.gain.value = 0.5 * strength;
    src.connect(bp).connect(ng).connect(musicBus); src.start(t);
  }

  // FLOW: the sound of the swarm itself moving — a looping noise bed whose
  // loudness AND brightness track the army's real aggregate speed (fed per
  // frame from the delta decode). Still army = silence; a full-flood charge
  // rushes like water over stone.
  let flowG = null, flowLP = null, flowLP2 = null, flowV = 0;
  function startFlow() {
    // WATER, not static: brown-ish noise (deep integration kills the fizz)
    // through TWO cascaded lowpasses (steep rolloff), with motion INSIDE the
    // sound — a ~0.8Hz wash and a slow surge LFO so it breathes like a
    // current instead of hissing like a detuned radio.
    const len = ctx.sampleRate * 3, buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    let last = 0;
    for (let i = 0; i < len; i++) {
      last = 0.985 * last + 0.07 * (Math.random() * 2 - 1);   // much deeper red
      d[i] = last * 3.2;
    }
    const src = ctx.createBufferSource(); src.buffer = buf; src.loop = true;
    flowLP = ctx.createBiquadFilter(); flowLP.type = "lowpass"; flowLP.frequency.value = 300;
    flowLP2 = ctx.createBiquadFilter(); flowLP2.type = "lowpass"; flowLP2.frequency.value = 600;
    flowG = ctx.createGain(); flowG.gain.value = 0;
    const wash = ctx.createGain(); wash.gain.value = 1;
    const lfo1 = ctx.createOscillator(); lfo1.frequency.value = 0.8;     // the wash
    const l1g = ctx.createGain(); l1g.gain.value = 0.25;
    const lfo2 = ctx.createOscillator(); lfo2.frequency.value = 0.13;    // the surge
    const l2g = ctx.createGain(); l2g.gain.value = 0.18;
    lfo1.connect(l1g).connect(wash.gain);
    lfo2.connect(l2g).connect(wash.gain);
    src.connect(flowLP).connect(flowLP2).connect(wash).connect(flowG).connect(sfxBus);
    src.start(); lfo1.start(); lfo2.start();
  }
  function setFlow(v, mass = 1) {              // v: avg mote speed; mass: your army 0..1
    if (!ctx || !enabled || !flowG) { flowV = v; return; }
    flowV = v;
    const now = ctx.currentTime;
    // MENACE SCALES WITH MASS: a full flood is louder AND deeper than a
    // scouting party moving at the same speed
    flowG.gain.setTargetAtTime(0.12 * Math.pow(v, 1.4) * (0.3 + 0.7 * mass), now, 0.25);
    const deep = 1.1 - 0.45 * mass;
    flowLP.frequency.setTargetAtTime((180 + 380 * v) * deep, now, 0.3);
    flowLP2.frequency.setTargetAtTime((340 + 560 * v) * deep, now, 0.3);
  }

  // ---- SFX ----
  function thud(strength) {                     // a front bleeding: tectonic hit
    if (!ctx || !enabled) return;
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
    if (!ctx || !enabled) return;
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

  // ---- THE GRID (the Pompeii engine, energetic cut): 104 BPM, a 16th-note
  // lookahead clock. The ostinato GALLOPS (DUM-dumdum per beat), taiko gets a
  // backbeat, the theme rides on top at full battle, and the harmonic rhythm
  // doubles at climax. At rest only the drone + sparkles remain.
  const BPM = 104, SPB = 60 / BPM, STEP = SPB / 4;      // 16th-note steps
  let gridStep = 0, nextNote = 0, melPos = 0, melNext = 0;
  // pitch per BEAT over a 2-bar (8-beat) cycle: root bar, the b2 bar
  const OSTP = [0, 0, 12, 0, 1, 1, 13, 1];
  // the theme (semitone, beats): rises, aches on the b2, falls home
  const THEME = [[12, 2], [13, 1], [12, 1], [8, 2], [10, 2],
                 [12, 2], [15, 1], [13, 1], [12, 3], [8, 1],
                 [10, 2], [7, 2], [8, 3], [-99, 1]];

  function spicc(semi, when, gain) {           // ostinato voice: struck string
    const o = ctx.createOscillator(); o.type = "sawtooth";
    o.frequency.value = f(semi, 220);
    const bp = ctx.createBiquadFilter(); bp.type = "bandpass";
    bp.frequency.value = 950; bp.Q.value = 1.4;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, when);
    g.gain.linearRampToValueAtTime(gain, when + 0.01);
    g.gain.setTargetAtTime(0, when + 0.025, 0.055);
    o.connect(bp).connect(g).connect(musicBus);
    o.start(when); o.stop(when + 0.4);
  }
  function lead(semi, when, dur, gain) {       // the theme: bowed, breathing
    if (semi === -99) return;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, when);
    g.gain.linearRampToValueAtTime(gain, when + 0.15);
    g.gain.setTargetAtTime(0, when + dur - 0.12, 0.22);
    const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 1900;
    g.connect(lp).connect(musicBus);
    for (const det of [-5, 4]) {
      const o = ctx.createOscillator(); o.type = "sawtooth";
      o.frequency.value = f(semi, 440); o.detune.value = det;
      const vib = ctx.createOscillator(); vib.frequency.value = 5.4;
      const vg = ctx.createGain(); vg.gain.value = 5;
      vib.connect(vg).connect(o.detune);
      o.connect(g); o.start(when); o.stop(when + dur + 1);
      vib.start(when); vib.stop(when + dur + 1);
    }
  }
  function taikoAt(when, strength, low) {      // grid-scheduled taiko
    setTimeout(() => enabled && taiko(strength, low),
               Math.max(0, (when - ctx.currentTime) * 1000 - 5));
  }

  let lastT = 0, wasDone = false;
  function update(s) {
    if (!ctx || !enabled) return;
    sig = s;
    const now = ctx.currentTime, dt = Math.min(0.1, now - lastT || 0.016);
    lastT = now;
    if (nextNote < now) nextNote = now + 0.05;
    const drive = Math.max(sig.intensity, sig.doom ? 0.3 : 0);
    while (nextNote < now + 0.25) {            // standard WebAudio lookahead
      const st16 = gridStep % 32;              // 2-bar cycle in 16ths
      const beat = (st16 / 4) | 0, sub16 = st16 % 4;
      const bar = (gridStep / 16) | 0;
      // harmonic rhythm doubles at climax: chords every bar, else every 2
      if (st16 === 0 || (drive > 0.55 && st16 === 16)) {
        if ((bar % 2 === 0) || drive > 0.55) nextChord();
      }
      // the GALLOP: hit on 1, and-a (sub16 0, 2, 3) — rest on the e
      if (drive > 0.05 && sub16 !== 1) {
        const accent = sub16 === 0;
        spicc(OSTP[beat], nextNote, (accent ? 1 : 0.62) * (0.055 + 0.105 * drive));
        if (drive > 0.5 && accent) spicc(OSTP[beat] + 12, nextNote, 0.04 + 0.05 * drive);
      }
      // TAIKO: downbeat boom + backbeat + pickup; relentless at climax
      if (drive > 0.1) {
        if (st16 === 0) taikoAt(nextNote, Math.min(1, 0.5 + drive), 1);
        if (st16 === 16) taikoAt(nextNote, 0.6 * Math.min(1, 0.5 + drive), 1.2);
        if (drive > 0.35 && (st16 === 8 || st16 === 24)) taikoAt(nextNote, 0.4 * drive, 1.5);
        if (drive > 0.6 && st16 === 30) taikoAt(nextNote, 0.5 * drive, 1.4);
      }
      // the THEME at full battle, riding the pulse
      if (drive > 0.4 && gridStep >= melNext) {
        const [semi, beats] = THEME[melPos % THEME.length];
        lead(semi, nextNote, beats * SPB, 0.075 + 0.05 * drive);
        melNext = gridStep + beats * 4;
        melPos++;
      }
      gridStep++;
      nextNote += STEP;
    }
    sparkTimer -= dt;
    if (sparkTimer <= 0) {
      if (Math.random() < 0.8 - 0.9 * drive) sparkle();        // calm only
      sparkTimer = 1.2 + Math.random() * 3.5 + 4 * drive;
    }
    sub.g.gain.setTargetAtTime(0.13 + 0.11 * sig.intensity, now, 1.5);
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
  return { update, thud, blip, stanceTap, setFlow, setEnabled, setVolumes,
           get enabled() { return enabled; },
           get debug() { return { state: ctx && ctx.state, beat: gridBeat, music: musicBus && musicBus.gain.value }; } };
})();
