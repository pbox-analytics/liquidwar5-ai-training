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
  let ctx = null, master, musicBus, sfxBus, padBus, echoIn;
  let coda = 0;                                // >now = match resolution playing; grid muted
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
    // FILTERED reverb send: the 34-150Hz pileup (sub + doom drone + taiko +
    // thuds) was smearing through the 2.8s cathedral and pumping the master
    // compressor — lows stay dry, the air gets the space
    const sendHP = ctx.createBiquadFilter(); sendHP.type = "highpass";
    sendHP.frequency.value = 260; sendHP.Q.value = 0.7;
    const sendLP = ctx.createBiquadFilter(); sendLP.type = "lowpass";
    sendLP.frequency.value = 5500;
    sendHP.connect(sendLP).connect(verb).connect(master);
    musicBus = ctx.createGain(); musicBus.gain.value = musicVol;
    const mSend = ctx.createGain(); mSend.gain.value = 0.5;
    musicBus.connect(master); musicBus.connect(mSend).connect(sendHP);
    sfxBus = ctx.createGain(); sfxBus.gain.value = sfxVol;
    const sSend = ctx.createGain(); sSend.gain.value = 0.25;
    sfxBus.connect(master); sfxBus.connect(sSend).connect(sendHP);
    // PAD BUS: organ + choir ride here so big drum moments can duck them
    padBus = ctx.createGain(); padBus.gain.value = 1.0;
    padBus.connect(musicBus);
    // shared ping-pong echo for the sparkles (one pair of delays forever —
    // the per-pluck feedback DelayNode leaked live graph nodes for hours)
    echoIn = ctx.createGain(); echoIn.gain.value = 1.0;
    const dA = ctx.createDelay(); dA.delayTime.value = 0.31;
    const dB = ctx.createDelay(); dB.delayTime.value = 0.43;
    const fA = ctx.createGain(); fA.gain.value = 0.3;
    const fB = ctx.createGain(); fB.gain.value = 0.3;
    echoIn.connect(dA); dA.connect(musicBus); dA.connect(fA).connect(dB);
    dB.connect(musicBus); dB.connect(fB).connect(dA);
    startSub(); startDoom(); startMael(); startFlow();
  }

  // --- organ voice: additive harmonics with cathedral attack ---
  function organ(freq, gain, attack = 4, dur = 22, noSub = false) {
    const g = ctx.createGain(); g.gain.value = 0;
    const lp = ctx.createBiquadFilter(); lp.type = "lowpass";
    lp.frequency.value = 460 + 2100 * sig.intensity;   // darker at rest
    g.connect(lp).connect(padBus);
    const t = ctx.currentTime;
    g.gain.linearRampToValueAtTime(gain, t + attack);
    g.gain.setTargetAtTime(0, t + dur - 7, 2.4);
    // the dedicated sub OWNS the bottom octave — the lowest chord voice
    // dropping its 0.5x partial un-muds the floor
    const parts = noSub ? [[1, 1], [2, 0.42], [3, 0.21], [4, 0.1]]
                        : [[1, 1], [2, 0.42], [3, 0.21], [4, 0.1], [0.5, 0.25]];
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
  function choir(freq, gain, dur = 22) {       // (dur in seconds)
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
    g.connect(padBus);
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
    voices = semis.map((s, i) => organ(f(s), i === 0 ? 0.16 : 0.10 - i * 0.015, 4, 22, i === 0));
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
    // ENEMY DOOM, audible: 36.7Hz is silent on phone speakers. The approach
    // layer is a beating 110Hz pair + a dark saw, opening and LOUDENING with
    // proximity, its heartbeat tremolo ACCELERATING 0.8 -> 3.5Hz as the hole
    // nears your cursor — fear you can hear with your eyes shut.
    const e1 = ctx.createOscillator(); e1.frequency.value = 110;
    const e2 = ctx.createOscillator(); e2.frequency.value = 110.7;
    const e3 = ctx.createOscillator(); e3.type = "sawtooth"; e3.frequency.value = 73.4;
    enemyLP = ctx.createBiquadFilter(); enemyLP.type = "lowpass"; enemyLP.frequency.value = 200;
    enemyG = ctx.createGain(); enemyG.gain.value = 0;
    enemyTrem = ctx.createOscillator(); enemyTrem.frequency.value = 0.8;
    const etg = ctx.createGain(); etg.gain.value = 0.05;
    enemyTrem.connect(etg).connect(enemyG.gain);
    e1.connect(enemyLP); e2.connect(enemyLP); e3.connect(enemyLP);
    enemyLP.connect(enemyG).connect(musicBus);
    e1.start(); e2.start(); e3.start(); enemyTrem.start();
    // THE CLASH: holding the correct counter (Maelstrom into enemy Doom) is a
    // tritone shimmer through a wobbling band — audible strain that HOLDS
    const c1 = ctx.createOscillator(); c1.type = "sawtooth"; c1.frequency.value = f(6, 220);
    const c2 = ctx.createOscillator(); c2.type = "sawtooth"; c2.frequency.value = f(6, 220) * 1.006;
    const cbp = ctx.createBiquadFilter(); cbp.type = "bandpass"; cbp.frequency.value = 1100; cbp.Q.value = 6;
    const cwob = ctx.createOscillator(); cwob.frequency.value = 0.9;
    const cwg = ctx.createGain(); cwg.gain.value = 350;
    cwob.connect(cwg).connect(cbp.frequency);
    clashG = ctx.createGain(); clashG.gain.value = 0;
    c1.connect(cbp); c2.connect(cbp); cbp.connect(clashG).connect(musicBus);
    c1.start(); c2.start(); cwob.start();
  }
  let enemyG = null, enemyLP = null, enemyTrem = null, clashG = null;
  let novaRiser = null;
  function novaCharge(on) {                    // the riser teaches the detonation by ear
    const now = ctx.currentTime;
    if (on && !novaRiser) {
      const o = ctx.createOscillator(); o.type = "sawtooth";
      o.frequency.setValueAtTime(f(12, 220), now);
      o.frequency.linearRampToValueAtTime(f(24, 220), now + 1.8);
      const lp = ctx.createBiquadFilter(); lp.type = "lowpass";
      lp.frequency.setValueAtTime(600, now);
      lp.frequency.linearRampToValueAtTime(4000, now + 1.8);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, now);
      g.gain.linearRampToValueAtTime(0.09, now + 1.6);
      o.connect(lp).connect(g).connect(sfxBus);
      o.start(now);
      novaRiser = { o, g };
    } else if (!on && novaRiser) {
      novaRiser.g.gain.setTargetAtTime(0, now, 0.02);     // cut...
      novaRiser.o.stop(now + 0.3);
      novaRiser = null;
      taiko(1.0, 0.7); thud(1.0); duck(0.6);              // ...and DETONATE
    }
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
    const pan = ctx.createStereoPanner();
    pan.pan.value = (Math.random() * 1.4 - 0.7);         // bioluminescence is wide
    o.connect(g); g.connect(pan).connect(musicBus); g.connect(echoIn);
    o.start(t); o.stop(t + 4);
  }

  // TAIKO: ceremonial hits — a pitch-dropping low sine + a skin-noise burst.
  // Silent at peace; the pattern wakes and quickens as the battle builds
  // (or while a Doom well is open — dread has a drum).
  function duck(amount = 0.68) {               // drums get a pocket in the pads
    const t = ctx.currentTime;
    padBus.gain.setTargetAtTime(amount, t, 0.015);
    padBus.gain.setTargetAtTime(1.0, t + 0.09, 0.22);
  }
  function taiko(strength, low = 1, when = null) {
    const t = when ?? ctx.currentTime;
    if (strength > 0.6) duck();
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
  // MATCH RESOLUTION: the score finishes the story instead of stopping.
  // win: one great drum + an open-fifth home cadence with the choir cresting;
  // lose: the dread interval (b2) sinking home, low and drumless;
  // another human won: the win cadence, softer — one fanfare per room.
  function resolveEnd(outcome) {
    if (!ctx || !enabled) return;
    const now = ctx.currentTime;
    coda = now + 7;
    activeGains.forEach(retire); activeGains = [];
    const scale = outcome === "human" ? 0.6 : 1.0;
    if (outcome === "lose") {
      for (const [semi, base, gn] of [[1, 55, 0.12], [1, 110, 0.10]]) {
        const o = ctx.createOscillator(); o.type = "sine";
        o.frequency.setValueAtTime(f(semi, base), now);
        o.frequency.setTargetAtTime(f(0, base), now + 0.8, 1.1);  // b2 sinks home
        const g = ctx.createGain();
        g.gain.setValueAtTime(0, now);
        g.gain.linearRampToValueAtTime(gn, now + 0.5);
        g.gain.setTargetAtTime(0, now + 4.5, 1.6);
        const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 500;
        o.connect(g).connect(lp).connect(musicBus);
        o.start(now); o.stop(now + 10);
      }
      return;
    }
    taiko(1.0 * scale, 0.8);
    const cadence = [[-12, 0.18], [0, 0.14], [7, 0.10], [12, 0.08], [19, 0.06]];
    for (const [semi, gn] of cadence) {
      const v = organ(f(semi), gn * scale, 0.4, 10, semi === -12);
      activeGains.push(v.g);
    }
    activeGains.push(choir(f(12) * 2, 0.16 * scale, 10));
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
    const F = (ctx.sampleRate * 0.25) | 0;     // seam crossfade: kill the loop thump
    for (let i = 0; i < F; i++) {
      const w = i / F;
      d[len - F + i] = (1 - w) * d[len - F + i] + w * d[i];
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
    if (strength > 0.5) duck(0.72);
    const o = ctx.createOscillator(); o.type = "sine";
    o.frequency.setValueAtTime(150, t);
    o.frequency.exponentialRampToValueAtTime(48, t + 0.28);
    const g = ctx.createGain();
    g.gain.setValueAtTime(Math.min(0.35, 0.08 + strength * 0.27), t);
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
    if (!ctx || !enabled) return;
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
    const pan = ctx.createStereoPanner();                // alternating 16ths sit L/R
    pan.pan.value = (gridStep & 1) ? 0.25 : -0.25;
    o.connect(bp).connect(g).connect(pan).connect(musicBus);
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
      const inCoda = now < coda;                 // resolution playing: war machine rests
      if (!inCoda && (st16 === 0 || (drive > 0.55 && st16 === 16))) {
        if ((bar % 2 === 0) || drive > 0.55) nextChord();
      }
      // the GALLOP: hit on 1, and-a (sub16 0, 2, 3) — rest on the e
      if (!inCoda && drive > 0.05 && sub16 !== 1) {
        const accent = sub16 === 0;
        spicc(OSTP[beat], nextNote, (accent ? 1 : 0.62) * (0.055 + 0.105 * drive));
        if (drive > 0.5 && accent) spicc(OSTP[beat] + 12, nextNote, 0.04 + 0.05 * drive);
      }
      // TAIKO on the grid, sample-accurate (setTimeout flammed 20-50ms on phones)
      if (!inCoda && drive > 0.1) {
        if (st16 === 0) taiko(Math.min(1, 0.5 + drive), 1, nextNote);
        if (st16 === 16) taiko(0.6 * Math.min(1, 0.5 + drive), 1.2, nextNote);
        if (drive > 0.35 && (st16 === 8 || st16 === 24)) taiko(0.4 * drive, 1.5, nextNote);
        if (drive > 0.6 && st16 === 30) taiko(0.5 * drive, 1.4, nextNote);
      }
      // the THEME at full battle, riding the pulse
      if (!inCoda && drive > 0.4 && gridStep >= melNext) {
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
    // YOUR doom: the sub drone scales with charge level; ENEMY doom: the
    // audible approach layer + accelerating heartbeat; the counter CLASHES
    const lvl = sig.doomLvl || 0;
    doomDrone.gain.setTargetAtTime(lvl ? [0.14, 0.19, 0.24][lvl - 1] : (sig.doom ? 0.12 : 0), now, 1.0);
    const prox = sig.doomProx || 0;
    enemyG.gain.setTargetAtTime(prox > 0.02 ? 0.05 + 0.13 * prox : 0, now, 0.5);
    enemyLP.frequency.setTargetAtTime(200 + 700 * prox, now, 0.5);
    enemyTrem.frequency.setTargetAtTime(0.8 + 2.7 * prox, now, 0.4);
    clashG.gain.setTargetAtTime(0.06 * Math.min(sig.mael ? 1 : 0, prox > 0.1 ? 1 : 0) * Math.max(prox, 0.4), now, 0.4);
    maelWash.gain.setTargetAtTime(sig.mael ? 0.12 : 0, now, 0.8);
    novaCharge(sig.nova === "charge");
    if (s.done && !wasDone) resolveEnd(s.outcome || (s.won ? "win" : "lose"));
    wasDone = s.done;
  }

  function setEnabled(v) {
    enabled = v;
    if (v) init(); else if (ctx) ctx.suspend();
    if (v && ctx) ctx.resume();
  }
  function setVolumes(m, sx) {
    musicVol = m; sfxVol = sx;
    if (!ctx) return;
    musicBus.gain.setTargetAtTime(m, ctx.currentTime, 0.04);
    sfxBus.gain.setTargetAtTime(sx, ctx.currentTime, 0.04);
  }
  return { update, thud, blip, stanceTap, setFlow, taiko, setEnabled, setVolumes,
           get enabled() { return enabled; },
           get debug() { return { state: ctx && ctx.state, beat: gridStep, music: musicBus && musicBus.gain.value }; } };
})();
