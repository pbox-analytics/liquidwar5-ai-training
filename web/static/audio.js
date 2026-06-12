"use strict";
// Procedural audio for Liquid War — no samples, everything synthesized.
//
// MUSIC: deep-space synthwave. ONE chord progression, TWO voicings,
// crossfaded on combat — analog-electronic, still cinematic, still Phrygian:
//   - PEACE voice: warm analog pads (detuned saw pair + sub-octave square
//     per chord tone, 3s attacks, slow chord drift) breathing through a
//     slow-wandering lowpass and a soft two-tap chorus — calm, spacious,
//     and OUT of the fight: fully silent by intensity ~0.45. The pad is a
//     sanctuary instrument, never a combat one.
//   - WAR voice: a supersaw wall — FIVE detuned saws per chord tone through
//     a lowpass that opens with the battle, 30ms synth attacks, octave-
//     doubled bass, sidechain-pumped by every kick so it breathes with the
//     112 BPM grid instead of droning
//   - DRUM MACHINE on the grid: pitch-drop kick (four-on-the-floor at full
//     drive), noise snare on the backbeat, 7kHz hat ticks opening to
//     shimmered 16ths; taiko() remains the ceremonial voice (countdown,
//     nova, cadences)
//   - BASSLINE: an eighth-note mono saw+square pulse on the chord root,
//     pumped with the kick — the electronic floor the orchestra never had
//   - tectonic sub: a slow-swelling low triangle, tanh-warmed for phones
//   - sparkles: sparse pentatonic sequencer blips through ping-pong delay
//     (the bioluminescent layer; density falls as combat rises)
//   - generated impulse-response reverb (procedural cathedral)
// ADAPTIVE: combat intensity crossfades pad->supersaw (equal-power, ~2s),
// brightens and raises the choir, speeds the chord clock and thins the
// sparkles; LOSING detunes a tension voice into BOTH voicings; Doom (either
// side) adds a pulsing 36.7Hz drone; Maelstrom adds a swirling wash.
// STANCE SIGNATURES: each held formation hums its own continuous texture
// (wingbeat noise, rotating sweep, machine grind, ...) at identity-not-noise
// gain UNDER the score; 0.6s crossfade on stance change; Doom and Maelstrom
// reuse the existing drone/wash machinery; Classic is intentional silence.
// SFX: conversion thuds scaled by bleed size, tiny blips for skirmish,
// per-stance tap notes (timbre keyed to the slot), victory/defeat stingers.
const LWAUDIO = (() => {
  let ctx = null, master, musicBus, sfxBus, padBus, peaceBus, warBus, echoIn;
  let coda = 0;                                // >now = match resolution playing; grid muted
  let musicVol = 0.5, sfxVol = 0.7, enabled = false;
  let chordTimer = 0, sparkTimer = 0, chordIdx = 0, voices = [];
  let sub = null, doomDrone = null, maelWash = null, tension = null;
  let maelWhirlPan = null, maelWhirlLP = null; // held-Maelstrom whirl depths
  let stanceBus = null, stanceSig = null, stanceCur = "", smodeCur = "";
  let sig = { intensity: 0, losing: 0, doom: 0, mael: 0 };
  let lastThud = 0, lastBlip = 0;
  let padIn = null, warPump = null, bassPump = null;   // synthwave plumbing
  let leadBus = null, leadLastF = 0, noiseBuf = null;  // (built in init)

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
    // PAD BUS: chords + choir ride here so big drum moments can duck them
    padBus = ctx.createGain(); padBus.gain.value = 1.0;
    padBus.connect(musicBus);
    // TWO VOICINGS, ONE PROGRESSION: update() equal-power crossfades these
    // on intensity — pad at peace, supersaw at war, never both at full
    peaceBus = ctx.createGain(); peaceBus.gain.value = 1.0;
    warBus = ctx.createGain(); warBus.gain.value = 0.0;
    peaceBus.connect(padBus);
    // SIDECHAIN: every kick punches a pocket in the saw wall and the
    // bassline through these (see pump()); duck() handles the rest of the bed
    warPump = ctx.createGain(); warPump.gain.value = 1.0;
    warBus.connect(warPump).connect(padBus);
    bassPump = ctx.createGain(); bassPump.gain.value = 1.0;
    bassPump.connect(musicBus);
    // PAD CHAIN, shared by every pad voice: one slow-wandering lowpass
    // (700-1400Hz — the analog filter drifting in its sleep) into a two-tap
    // modulated chorus. Shared on purpose: per-voice LFOs would pile up.
    padIn = ctx.createGain(); padIn.gain.value = 1.0;
    const padLP = ctx.createBiquadFilter(); padLP.type = "lowpass";
    padLP.frequency.value = 1050; padLP.Q.value = 0.6;
    const wander = ctx.createOscillator(); wander.frequency.value = 0.08;
    const wg = ctx.createGain(); wg.gain.value = 350;
    wander.connect(wg).connect(padLP.frequency); wander.start();
    padIn.connect(padLP); padLP.connect(peaceBus);       // dry leg
    for (const [dt, rate] of [[0.012, 0.61], [0.017, 0.47]]) {
      const dl = ctx.createDelay(0.05); dl.delayTime.value = dt;
      const lfo = ctx.createOscillator(); lfo.frequency.value = rate;
      const lg = ctx.createGain(); lg.gain.value = 0.003; // ±3ms shimmer
      lfo.connect(lg).connect(dl.delayTime); lfo.start();
      const cg = ctx.createGain(); cg.gain.value = 0.5;   // subtle, under the dry
      padLP.connect(dl).connect(cg).connect(peaceBus);
    }
    // LEAD BUS: same volume law as the music, but a drier send — the theme
    // sits forward instead of dissolving into the cathedral
    leadBus = ctx.createGain(); leadBus.gain.value = musicVol;
    leadBus.connect(master);
    const lSend = ctx.createGain(); lSend.gain.value = 0.18;
    leadBus.connect(lSend).connect(sendHP);
    // shared noise for the kit: ONE buffer, many sources — per-hit buffers
    // at 16th-note hat rates would churn the GC
    noiseBuf = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
    const nd = noiseBuf.getChannelData(0);
    for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
    // STANCE BUS: the per-formation signature textures (identity, not noise)
    stanceBus = ctx.createGain(); stanceBus.gain.value = 1.0;
    stanceBus.connect(musicBus);
    // shared ping-pong echo for the sparkles (one pair of delays forever —
    // the per-pluck feedback DelayNode leaked live graph nodes for hours);
    // a touch hotter now: the blips are sequencer hits, the echo is the room
    echoIn = ctx.createGain(); echoIn.gain.value = 1.25;
    const dA = ctx.createDelay(); dA.delayTime.value = 0.31;
    const dB = ctx.createDelay(); dB.delayTime.value = 0.43;
    const fA = ctx.createGain(); fA.gain.value = 0.3;
    const fB = ctx.createGain(); fB.gain.value = 0.3;
    echoIn.connect(dA); dA.connect(musicBus); dA.connect(fA).connect(dB);
    dB.connect(musicBus); dB.connect(fB).connect(dA);
    startSub(); startDoom(); startMael(); startFlow();
  }

  // --- analog pad (PEACE): a detuned saw pair + one sub-octave square per
  // chord tone, breathing through the shared wandering lowpass + chorus
  // built in init — warm analog mass where the organ used to live, with the
  // same cathedral patience (slow attack stays: sanctuary)
  function pad(freq, gain, attack = 3, dur = 22, noSub = false) {
    const g = ctx.createGain(); g.gain.value = 0;
    g.connect(padIn);
    const t = ctx.currentTime;
    g.gain.linearRampToValueAtTime(gain, t + attack);
    g.gain.setTargetAtTime(0, t + dur - 7, 2.4);
    // the dedicated sub OWNS the bottom octave — the lowest chord voice
    // dropping its sub-octave square un-muds the floor
    const parts = noSub ? [["sawtooth", 1, -7, 0.5], ["sawtooth", 1, 7, 0.5]]
                        : [["sawtooth", 1, -7, 0.5], ["sawtooth", 1, 7, 0.5],
                           ["square", 0.5, 0, 0.3]];
    const oscs = parts.map(([type, m, det, a]) => {
      const o = ctx.createOscillator(); o.type = type;
      o.frequency.value = freq * m;
      o.detune.value = det + (Math.random() - 0.5) * 4;
      const og = ctx.createGain(); og.gain.value = a;
      o.connect(og).connect(g); o.start(t); o.stop(t + dur + 1);
      return o;
    });
    return { g, oscs };
  }

  // --- supersaw (WAR): per chord tone, FIVE detuned saws through a lowpass
  // that opens with the battle — 30ms synth attack, shorter release than the
  // pad. The old half-bar re-articulation is gone: the sidechain pump (see
  // pump()) now carves the pocket on every kick, so the wall PULSES on the
  // 112 BPM grid instead of droning. THIS is what plays when units fight;
  // the pad stays home.
  function supersaw(freq, gain, when, dur = 22) {
    const t = when ?? ctx.currentTime;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain * 0.7, t + 0.03);  // snap, not bow swell
    g.gain.setTargetAtTime(0, t + dur - 9, 1.3);
    const lp = ctx.createBiquadFilter(); lp.type = "lowpass";
    lp.frequency.value = 900 + 3600 * sig.intensity;   // 900-4.5k with the war
    lp.Q.value = 0.8;
    g.connect(lp).connect(warBus);
    for (const det of [-18, -9, 0, 9, 18]) {           // the stack, not a soloist
      const o = ctx.createOscillator(); o.type = "sawtooth";
      o.frequency.value = freq;
      o.detune.value = det + (Math.random() - 0.5) * 3;
      o.connect(g); o.start(t); o.stop(t + dur + 1);
    }
    return g;
  }

  // CHOIR: a detuned saw pair through "ahh" formant bandpasses — the
  // E.S. Posthumus chorus, rising out of the pad as battle builds.
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
      // battle BRIGHTENS the chorus: the upper formants open with intensity
      const fg = ctx.createGain();
      fg.gain.value = fa * (ff > 900 ? 1 + 1.1 * sig.intensity : 1);
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
  function nextChord(when) {
    // HANDOVER: the grid changes chords every 2 bars (~5s) but voices used to
    // live ~20s — three or four chords rang at once and combat made the
    // pileup audibly sour. Old voices bow out as the new chord enters.
    // BOTH voicings retire through here, so chords never stack on either side.
    activeGains.forEach(retire);
    activeGains = [];
    chordIdx = (chordIdx + (Math.random() < 0.75 ? 1 : CHORDS.length - 1)) % CHORDS.length;
    const semis = CHORDS[chordIdx];
    // PEACE voicing: the analog pad, as ever
    voices = semis.map((s, i) => pad(f(s), i === 0 ? 0.16 : 0.10 - i * 0.015, 3, 22, i === 0));
    activeGains.push(...voices.map(v => v.g));
    // WAR voicing of the SAME chord: supersaws an octave up, on the grid,
    // plus the octave-doubled bass note for the floor
    const t = when ?? ctx.currentTime;
    activeGains.push(...semis.map((s, i) => supersaw(f(s) * 2, i === 0 ? 0.10 : 0.07 - i * 0.008, t)));
    activeGains.push(supersaw(f(semis[0]), 0.09, t));
    // the chorus swells with the war: barely-there at peace, full voice in battle
    const ch = 0.04 + 0.13 * sig.intensity + 0.05 * (sig.doom ? 1 : 0);
    activeGains.push(choir(f(semis[0]) * 2, ch));
    if (sig.intensity > 0.35) activeGains.push(choir(f(semis[1]) * 2, ch * 0.7));
    // losing: a minor-second tension voice bleeds into BOTH voicings
    if (sig.losing > 0.1) {
      const tg = 0.07 * Math.min(1, sig.losing * 2.5);
      tension = pad(f(semis[0] + 1) / 2, tg, 5);
      activeGains.push(tension.g);
      activeGains.push(supersaw(f(semis[0] + 1), tg, t));
    }
  }

  function startSub() {
    const o = ctx.createOscillator(); o.type = "triangle"; o.frequency.value = 55;
    // gentle tanh saturation: a whisper of odd harmonics so the sub READS
    // on phone speakers instead of vanishing below their cones
    const ws = ctx.createWaveShaper();
    const curve = new Float32Array(257);
    for (let i = 0; i < 257; i++) curve[i] = Math.tanh(1.6 * (i / 128 - 1));
    ws.curve = curve;
    const g = ctx.createGain(); g.gain.value = 0;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.045;
    const lg = ctx.createGain(); lg.gain.value = 0.05;
    lfo.connect(lg).connect(g.gain);
    o.connect(ws).connect(g).connect(musicBus); o.start(); lfo.start();
    sub = { o, g };
  }

  function startDoom() {                       // 36.7Hz = D1: the well's voice
    const o = ctx.createOscillator(); o.frequency.value = 36.7;
    const o2 = ctx.createOscillator(); o2.frequency.value = 36.95;   // beat ~0.25Hz
    const g = ctx.createGain(); g.gain.value = 0;
    const trem = ctx.createOscillator(); trem.frequency.value = 0.8; // the heartbeat
    // tremolo depth is GATED with the drone (update scales it): a fixed
    // depth summed into g.gain kept a faint 36.7Hz pulse alive at idle
    const tg = ctx.createGain(); tg.gain.value = 0;
    trem.connect(tg).connect(g.gain);
    o.connect(g); o2.connect(g); g.connect(musicBus);
    o.start(); o2.start(); trem.start();
    doomDrone = g; doomTremG = tg;
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
    const etg = ctx.createGain(); etg.gain.value = 0;    // gated with the layer (see update)
    enemyTrem.connect(etg).connect(enemyG.gain);
    enemyTremG = etg;
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
  let doomTremG = null, enemyTremG = null;
  let novaRiser = null;
  function novaCycle(nova) {                   // the riser teaches the detonation by ear
    const now = ctx.currentTime;
    if (nova === "charge" && !novaRiser) {
      const o = ctx.createOscillator(); o.type = "sawtooth";
      o.frequency.setValueAtTime(f(12, 220), now);
      o.frequency.linearRampToValueAtTime(f(24, 220), now + 1.8);
      const lp = ctx.createBiquadFilter(); lp.type = "lowpass";
      lp.frequency.setValueAtTime(600, now);
      lp.frequency.linearRampToValueAtTime(4000, now + 1.8);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, now);
      g.gain.linearRampToValueAtTime(0.09, now + 1.6);
      // SELF-TERMINATING: if the charge signal freezes (hidden tab, dropped
      // frames) the saw must not drone at full gain forever
      g.gain.setTargetAtTime(0, now + 2.4, 0.3);
      o.connect(lp).connect(g).connect(sfxBus);
      o.start(now); o.stop(now + 4);
      novaRiser = { o, g };
    } else if (nova !== "charge" && novaRiser) {
      novaRiser.g.gain.setTargetAtTime(0, now, 0.02);     // cut...
      try { novaRiser.o.stop(now + 0.3); } catch (e) {}
      novaRiser = null;
      // ...and DETONATE — but ONLY on the real boom phase: aborting the
      // charge (stance switch, mode cycle, match end) gets no false payoff
      if (nova === "boom") { taiko(1.0, 0.7); thud(1.0); duck(0.6); }
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
    // WHIRL (held Maelstrom): one slow 0.25Hz LFO leans the wash around the
    // head AND breathes a lowpass on top — both depths sit at 0 until the
    // stance is held, so the plain mael signal sounds exactly as before
    const lpW = ctx.createBiquadFilter(); lpW.type = "lowpass"; lpW.frequency.value = 2600;
    const pan = ctx.createStereoPanner();
    const whirl = ctx.createOscillator(); whirl.frequency.value = 0.25;
    maelWhirlPan = ctx.createGain(); maelWhirlPan.gain.value = 0;
    maelWhirlLP = ctx.createGain(); maelWhirlLP.gain.value = 0;
    whirl.connect(maelWhirlPan).connect(pan.pan);
    whirl.connect(maelWhirlLP).connect(lpW.frequency);
    const g = ctx.createGain(); g.gain.value = 0;
    src.connect(bp).connect(lpW).connect(g).connect(pan).connect(musicBus);
    src.start(); lfo.start(); whirl.start();
    maelWash = g;
  }

  // ---- STANCE SIGNATURES: one continuous texture per held formation,
  // sitting UNDER the score (identity, not noise — gains 0.05-0.08 on the
  // music bus). 0.6s crossfade old->new on stance change; retired graphs are
  // stopped + disconnected after the fade so churn never leaks nodes.
  // Doom and Maelstrom reuse the existing drone/wash machinery (see update);
  // Classic is intentional silence — the original game's voice.
  function mkSig(level) {                      // fade-in handle for a signature
    const g = ctx.createGain();
    const t = ctx.currentTime;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(level, t + 0.6);
    g.connect(stanceBus);
    return { g, nodes: [] };                   // nodes: everything with a .stop()
  }
  function retireSig(h) {
    if (!h) return;
    const now = ctx.currentTime;
    h.g.gain.cancelScheduledValues(now);
    h.g.gain.setValueAtTime(h.g.gain.value, now);
    h.g.gain.linearRampToValueAtTime(0, now + 0.6);
    setTimeout(() => {                         // stop AFTER the fade — no clicks
      for (const n of h.nodes) { try { n.stop(); } catch (e) {} }
      try { h.g.disconnect(); } catch (e) {}
    }, 900);
  }
  function noiseSrc(secs = 1) {                // small white-noise loop for textures
    const len = (ctx.sampleRate * secs) | 0, buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    const src = ctx.createBufferSource(); src.buffer = buf; src.loop = true;
    return src;
  }
  const SIGS = {
    Swarm() {                                  // airy shimmer — a thousand wingbeats
      const h = mkSig(0.05);
      const src = noiseSrc();
      const bp = ctx.createBiquadFilter(); bp.type = "bandpass";
      bp.frequency.value = 4500; bp.Q.value = 1.2;
      const fl = ctx.createGain(); fl.gain.value = 0.6;
      const lfo = ctx.createOscillator(); lfo.frequency.value = 6;   // the flutter
      const lg = ctx.createGain(); lg.gain.value = 0.35;
      lfo.connect(lg).connect(fl.gain);
      src.connect(bp).connect(fl).connect(h.g);
      src.start(); lfo.start(); h.nodes.push(src, lfo);
      return h;
    },
    Spin() {                                   // rotating sweep — the blade going by
      const h = mkSig(0.05);
      const o = ctx.createOscillator(); o.type = "sawtooth"; o.frequency.value = f(0, 220);
      const bp = ctx.createBiquadFilter(); bp.type = "bandpass";
      bp.frequency.value = 1000; bp.Q.value = 5;
      const pan = ctx.createStereoPanner();
      const lfo = ctx.createOscillator(); lfo.frequency.value = 0.4; // one rev ~2.5s
      const fg = ctx.createGain(); fg.gain.value = 620;
      const pg = ctx.createGain(); pg.gain.value = 0.8;
      lfo.connect(fg).connect(bp.frequency);   // sweep + pan in phase = rotation
      lfo.connect(pg).connect(pan.pan);
      o.connect(bp).connect(pan).connect(h.g);
      o.start(); lfo.start(); h.nodes.push(o, lfo);
      return h;
    },
    Drill() {                                  // low machine grind, gated at 8Hz
      const h = mkSig(0.07);
      const ws = ctx.createWaveShaper();       // the grit
      const curve = new Float32Array(257);
      for (let i = 0; i < 257; i++) curve[i] = Math.tanh(2.5 * (i / 128 - 1));
      ws.curve = curve;
      const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 260;
      const gate = ctx.createGain(); gate.gain.value = 0.5;
      const lfo = ctx.createOscillator(); lfo.type = "square"; lfo.frequency.value = 8;
      const lg = ctx.createGain(); lg.gain.value = 0.45;             // gates ~0.05..0.95
      lfo.connect(lg).connect(gate.gain);
      for (const [type, fr] of [["square", 55], ["sawtooth", 55.6]]) {
        const o = ctx.createOscillator(); o.type = type; o.frequency.value = fr;
        o.connect(ws); o.start(); h.nodes.push(o);
      }
      ws.connect(lp).connect(gate).connect(h.g);
      lfo.start(); h.nodes.push(lfo);
      return h;
    },
    Wall() {                                   // granite stillness — D2+A2, barely breathing
      const h = mkSig(0.05);
      const br = ctx.createGain(); br.gain.value = 0.8;
      const lfo = ctx.createOscillator(); lfo.frequency.value = 0.1; // the breath
      const lg = ctx.createGain(); lg.gain.value = 0.2;
      lfo.connect(lg).connect(br.gain);
      for (const fr of [73.42, 110]) {         // perfect fifth, both in the Phrygian bed
        const o = ctx.createOscillator(); o.type = "sine"; o.frequency.value = fr;
        o.connect(br); o.start(); h.nodes.push(o);
      }
      br.connect(h.g); lfo.start(); h.nodes.push(lfo);
      return h;
    },
    Pulse() {                                  // empty vessel: the GRID delivers the ticks
      return mkSig(0.08);
    },
    Atom() {                                   // 220 vs 223Hz: orbital interference, ears apart
      const h = mkSig(0.05);
      for (const [fr, p] of [[220, -0.6], [223, 0.6]]) {
        const o = ctx.createOscillator(); o.type = "sine"; o.frequency.value = fr;
        const og = ctx.createGain(); og.gain.value = 0.5;
        const pan = ctx.createStereoPanner(); pan.pan.value = p;
        o.connect(og).connect(pan).connect(h.g);
        o.start(); h.nodes.push(o);
      }
      return h;
    },
    // Doom / Maelstrom: no graph here — update() drives the existing drone
    // and wash machinery off stanceCur. Classic: silence, on purpose.
  };
  function pulseTick(when, dest, thump) {      // the Pulse clock-throb, grid-locked
    const o = ctx.createOscillator(); o.type = "sine"; o.frequency.value = 1760;
    const bp = ctx.createBiquadFilter(); bp.type = "bandpass";
    bp.frequency.value = 1760; bp.Q.value = 9;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.6, when);
    g.gain.setTargetAtTime(0, when + 0.004, 0.02);
    o.connect(bp).connect(g).connect(dest);
    o.start(when); o.stop(when + 0.25);
    if (!thump) return;                        // the faint sub only rides the beat
    const o2 = ctx.createOscillator(); o2.type = "sine"; o2.frequency.value = 55;
    const g2 = ctx.createGain();
    g2.gain.setValueAtTime(0.5, when);
    g2.gain.setTargetAtTime(0, when + 0.02, 0.05);
    o2.connect(g2).connect(dest); o2.start(when); o2.stop(when + 0.4);
  }

  function sparkle() {                          // bioluminescent sequencer blip
    const semi = PENTA[(Math.random() * PENTA.length) | 0];
    const o = ctx.createOscillator(); o.type = "sine";
    o.frequency.value = f(semi, 440) * (Math.random() < 0.3 ? 2 : 1);
    const g = ctx.createGain();
    const t = ctx.currentTime, amp = 0.05 + 0.04 * Math.random();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(amp, t + 0.004);      // crisp: a blip, not a bloom
    g.gain.setTargetAtTime(0, t + 0.012, 0.3);
    const pan = ctx.createStereoPanner();
    pan.pan.value = (Math.random() * 1.4 - 0.7);         // bioluminescence is wide
    o.connect(g); g.connect(pan).connect(musicBus); g.connect(echoIn);
    o.start(t); o.stop(t + 2.5);
  }

  // DRUMS. taiko() stays the ceremonial hit (countdown, nova, the win drum —
  // external callers depend on it); the GRID now runs a synthesized kit:
  // kick/snare/hats below, every kick pumping the whole bed.
  function duck(amount = 0.68, when = null) {  // drums get a pocket in the pads
    if (!ctx) return;
    const t = when ?? ctx.currentTime;         // honor the grid's lookahead — the
    padBus.gain.setTargetAtTime(amount, t, 0.015);   // pocket used to open ~250ms early
    padBus.gain.setTargetAtTime(1.0, t + 0.09, 0.22);
  }
  function pump(when) {                        // SIDECHAIN: the kick punches a
    if (!warPump) return;                      // deeper hole in saw wall + bass
    for (const p of [warPump, bassPump]) {
      p.gain.setTargetAtTime(0.32, when, 0.008);
      p.gain.setTargetAtTime(1.0, when + 0.05, 0.12);  // ~120ms recovery: the pump
    }
  }
  function kick(when, vel = 1) {               // the floor: pitch-drop sine + click
    duck(0.75, when); pump(when);              // the whole bed breathes on the kick
    const o = ctx.createOscillator(); o.type = "sine";
    o.frequency.setValueAtTime(150, when);
    o.frequency.exponentialRampToValueAtTime(48, when + 0.12);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.30 * vel, when);
    g.gain.setTargetAtTime(0, when + 0.03, 0.09);
    o.connect(g).connect(musicBus); o.start(when); o.stop(when + 0.6);
    const src = ctx.createBufferSource(); src.buffer = noiseBuf;  // 2ms beater click
    const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 2500;
    const cg = ctx.createGain();
    cg.gain.setValueAtTime(0.12 * vel, when);
    cg.gain.setTargetAtTime(0, when + 0.002, 0.004);
    src.connect(hp).connect(cg).connect(musicBus); src.start(when); src.stop(when + 0.05);
  }
  function snare(when, vel = 1) {              // backbeat: noise burst + 180Hz body
    const src = ctx.createBufferSource(); src.buffer = noiseBuf;
    const bp = ctx.createBiquadFilter(); bp.type = "bandpass";
    bp.frequency.value = 1800; bp.Q.value = 1;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.18 * vel, when);
    g.gain.setTargetAtTime(0, when + 0.01, 0.055);
    src.connect(bp).connect(g).connect(musicBus); src.start(when); src.stop(when + 0.35);
    const o = ctx.createOscillator(); o.type = "triangle"; o.frequency.value = 180;
    const og = ctx.createGain();
    og.gain.setValueAtTime(0.09 * vel, when);
    og.gain.setTargetAtTime(0, when + 0.005, 0.035);
    o.connect(og).connect(musicBus); o.start(when); o.stop(when + 0.25);
  }
  function hat(when, vel = 1, open = false) {  // the clock: 7kHz noise ticks
    const src = ctx.createBufferSource(); src.buffer = noiseBuf;
    const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 7000;
    const g = ctx.createGain();
    g.gain.setValueAtTime((open ? 0.08 : 0.05) * vel, when);
    g.gain.setTargetAtTime(0, when + 0.004, open ? 0.11 : 0.014);
    src.connect(hp).connect(g).connect(musicBus);
    src.start(when); src.stop(when + (open ? 0.8 : 0.12));
  }
  function taiko(strength, low = 1, when = null) {
    if (!ctx || !enabled) return;              // public API: must survive pre-enable calls
    const t = when ?? ctx.currentTime;
    if (strength > 0.6) duck(0.68, t);
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
    // cadences belong to the PAD: snap the crossfade home for the coda
    // (update() holds it there while the resolution plays)
    peaceBus.gain.setTargetAtTime(1, now, 0.15);
    warBus.gain.setTargetAtTime(0, now, 0.15);
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
      const v = pad(f(semi), gn * scale, 0.4, 10, semi === -12);
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
  // per-stance confirmation note: same pentatonic bed, but each stance
  // answers in its OWN voice — waveform + register keyed to the slot
  // (Swarm..Classic), so the hand learns the sound before the eye finds the
  // label. Classic keeps the original plain sine: purity.
  const TAP_TYPE = ["sine", "sawtooth", "square", "triangle", "sine",
                    "sawtooth", "triangle", "sine", "sine"];
  const TAP_OCT  = [1, 1, 0.5, 0.5, 2, 0.25, 1, 2, 1];
  function stanceTap(i) {
    if (!ctx || !enabled) return;
    const t = ctx.currentTime;
    const o = ctx.createOscillator(); o.type = TAP_TYPE[i % 9];
    o.frequency.value = f(PENTA[i % PENTA.length], 330) * TAP_OCT[i % 9];
    const lp = ctx.createBiquadFilter(); lp.type = "lowpass";
    lp.frequency.value = o.type === "sine" ? 4000 : 1400;  // keep rough waves soft
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.12, t);
    g.gain.setTargetAtTime(0, t + 0.04, 0.18);
    o.connect(lp).connect(g).connect(sfxBus); o.start(t); o.stop(t + 1);
    if (i % 9 === 7) {                          // Atom: the beat pair, even in the tap
      const o2 = ctx.createOscillator(); o2.type = "sine";
      o2.frequency.value = o.frequency.value + 3;
      o2.connect(lp); o2.start(t); o2.stop(t + 1);
    }
  }
  // (the old stinger() is gone — resolveEnd() is the match-resolution voice)

  // ---- THE GRID (the synthwave engine): 112 BPM, a 16th-note lookahead
  // clock. The ostinato GALLOPS (DUM-dumdum per beat) as an arp pluck, the
  // kit drives four-on-the-floor, the bassline pulses eighths, the theme
  // rides on top at full battle, and the harmonic rhythm doubles at climax.
  // At rest only the drone + sparkles remain.
  const BPM = 112, SPB = 60 / BPM, STEP = SPB / 4;      // 16th-note steps
  let gridStep = 0, nextNote = 0, melPos = 0, melNext = 0;
  // pitch per BEAT over a 2-bar (8-beat) cycle: root bar, the b2 bar
  const OSTP = [0, 0, 12, 0, 1, 1, 13, 1];
  // the theme (semitone, beats): rises, aches on the b2, falls home
  const THEME = [[12, 2], [13, 1], [12, 1], [8, 2], [10, 2],
                 [12, 2], [15, 1], [13, 1], [12, 3], [8, 1],
                 [10, 2], [7, 2], [8, 3], [-99, 1]];

  function pluck(semi, when, gain, subOct = false) {   // ostinato: arp pluck
    const o = ctx.createOscillator(); o.type = "square";
    o.frequency.value = f(semi, 220);
    // the snap lives in the FILTER: resonant lowpass biting 2000->300 in 80ms
    const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.Q.value = 8;
    lp.frequency.setValueAtTime(2000, when);
    lp.frequency.exponentialRampToValueAtTime(300, when + 0.08);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, when);
    g.gain.linearRampToValueAtTime(gain, when + 0.005);
    g.gain.setTargetAtTime(0, when + 0.03, 0.055);
    const pan = ctx.createStereoPanner();                // alternating 16ths sit L/R
    pan.pan.value = (gridStep & 1) ? 0.25 : -0.25;
    o.connect(lp); lp.connect(g).connect(pan).connect(musicBus);
    o.start(when); o.stop(when + 0.4);
    if (!subOct) return;                       // full drive: a sub-octave saw
    const o2 = ctx.createOscillator(); o2.type = "sawtooth";  // under the pluck,
    o2.frequency.value = f(semi, 220) / 2;                    // half gain, same
    const og = ctx.createGain(); og.gain.value = 0.5;         // filter envelope
    o2.connect(og).connect(lp); o2.start(when); o2.stop(when + 0.4);
  }
  function bass(semi, when, drive) {           // the eighth-note mono pulse:
    let fr = f(semi, 110);                     // chord root, folded into 55-110Hz
    while (fr >= 110) fr /= 2;
    while (fr < 55) fr *= 2;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, when);
    g.gain.linearRampToValueAtTime(0.10, when + 0.006);
    g.gain.setTargetAtTime(0, when + 0.16, 0.045);
    // two cascaded biquads: the 24dB-feel ladder the saw+square growl wants
    const lp1 = ctx.createBiquadFilter(); lp1.type = "lowpass";
    const lp2 = ctx.createBiquadFilter(); lp2.type = "lowpass";
    lp1.frequency.value = lp2.frequency.value = 300 + 500 * drive;
    g.connect(lp1).connect(lp2).connect(bassPump);       // pumped with the kick
    for (const [type, a] of [["sawtooth", 1], ["square", 0.5]]) {
      const o = ctx.createOscillator(); o.type = type; o.frequency.value = fr;
      const og = ctx.createGain(); og.gain.value = a;
      o.connect(og).connect(g); o.start(when); o.stop(when + 0.5);
    }
  }
  function lead(semi, when, dur, gain) {       // the theme: a mono PWM-ish lead
    if (semi === -99) return;
    const fr = f(semi, 440);
    const from = leadLastF || fr; leadLastF = fr;        // mono-synth glide state
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, when);
    g.gain.linearRampToValueAtTime(gain, when + 0.02);
    g.gain.setTargetAtTime(0, when + dur - 0.12, 0.22);
    const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 1900;
    g.connect(lp).connect(leadBus);            // the drier bus: forward, not drowned
    // two detuned squares beat like slow PWM; the quiet saw fills the comb
    for (const [type, det, a] of [["square", -6, 0.5], ["square", 5, 0.5],
                                  ["sawtooth", 0, 0.3]]) {
      const o = ctx.createOscillator(); o.type = type;
      o.frequency.setValueAtTime(from, when);
      o.frequency.setTargetAtTime(fr, when, 0.06);       // ~60ms portamento
      o.detune.value = det;
      const og = ctx.createGain(); og.gain.value = a;
      const vib = ctx.createOscillator(); vib.frequency.value = 5.4;
      const vg = ctx.createGain(); vg.gain.value = 5;
      vib.connect(vg).connect(o.detune);
      o.connect(og).connect(g); o.start(when); o.stop(when + dur + 1);
      vib.start(when); vib.stop(when + dur + 1);
    }
  }
  let lastT = 0, wasDone = false;
  function update(s) {
    if (!ctx || !enabled) return;
    sig = s;
    const now = ctx.currentTime, dt = Math.min(0.1, now - lastT || 0.016);
    lastT = now;
    // a NEW round while the coda still rings (quick Rematch) un-mutes the
    // grid — the war machine used to start the next match silenced
    if (!s.done && wasDone && coda > now) coda = 0;
    // STANCE SIGNATURE handover (stance/smode are OPTIONAL — undefined-safe):
    // retire the old texture, raise the new one, 0.6s crossfade
    const st = s.stance || "";
    smodeCur = s.smode || "";
    if (st !== stanceCur) {
      retireSig(stanceSig);
      stanceCur = st;
      stanceSig = Object.hasOwn(SIGS, st) ? SIGS[st]() : null;  // Doom/Mael/Classic build nothing
    }
    if (nextNote < now) nextNote = now + 0.05;
    // GROOVE FLOOR 0.32: the kit, bass and arp run four-on-the-floor even at
    // peace ("upbeat sound track with a good beat") — battle still escalates
    // everything above it (snare/16th hats/theme/double-time chords gate
    // higher), and the pad->supersaw crossfade keeps tracking RAW intensity
    // so calm still *sounds* calm, just never beatless.
    const drive = Math.max(sig.intensity, sig.doom ? 0.3 : 0, 0.32);
    while (nextNote < now + 0.25) {            // standard WebAudio lookahead
      const st16 = gridStep % 32;              // 2-bar cycle in 16ths
      const beat = (st16 / 4) | 0, sub16 = st16 % 4;
      const bar = (gridStep / 16) | 0;
      // harmonic rhythm doubles at climax: chords every bar, else every 2
      const inCoda = now < coda;                 // resolution playing: war machine rests
      if (!inCoda && (st16 === 0 || (drive > 0.55 && st16 === 16))) {
        if ((bar % 2 === 0) || drive > 0.55) nextChord(nextNote);
      }
      // the GALLOP: hit on 1, and-a (sub16 0, 2, 3) — rest on the e
      if (!inCoda && drive > 0.05 && sub16 !== 1) {
        const accent = sub16 === 0;
        pluck(OSTP[beat], nextNote, (accent ? 1 : 0.62) * (0.055 + 0.105 * drive), drive > 0.5);
        if (drive > 0.5 && accent) pluck(OSTP[beat] + 12, nextNote, 0.04 + 0.05 * drive);
      }
      // THE KIT on the grid, sample-accurate (setTimeout flammed 20-50ms on
      // phones): four-on-the-floor past drive 0.3 (beats 1+3 below), snare
      // answering the backbeat, hats opening from offbeat 8ths to shimmered
      // 16ths, the and-of-4 left ringing. Every kick pumps the whole bed.
      if (!inCoda && drive > 0.1) {
        const inBar = st16 % 16;
        if (sub16 === 0 && (drive > 0.3 || inBar % 8 === 0))
          kick(nextNote, Math.min(1, 0.6 + 0.4 * drive));
        if (drive > 0.45 && (inBar === 4 || inBar === 12))
          snare(nextNote, Math.min(1, 0.5 + 0.6 * drive));
        if (inBar === 14) hat(nextNote, 1, true);             // open: the and-of-4
        else if (drive > 0.6) hat(nextNote, 0.5 + 0.5 * Math.random()); // 16th shimmer
        else if (sub16 === 2) hat(nextNote, 0.9);             // offbeat 8ths
      }
      // BASSLINE: the eighth-note mono pulse under everything — chord root,
      // with the b2 slipping past on the last 8th of every 4th bar
      if (!inCoda && drive > 0.25 && (sub16 === 0 || sub16 === 2)) {
        const passing = bar % 4 === 3 && st16 % 16 === 14;
        bass(CHORDS[chordIdx][0] + (passing ? 1 : 0), nextNote, drive);
      }
      // PULSE signature: the clock-throb rides the same grid, sample-accurate
      // (every 4th 16th; nova mode: the clock RACES at 8ths)
      if (!inCoda && stanceCur === "Pulse" && stanceSig) {
        const every = smodeCur === "nova" ? 2 : 4;
        if (gridStep % every === 0) pulseTick(nextNote, stanceSig.g, sub16 === 0);
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
    // PEACE/WAR crossfade: equal-power on intensity, smoothed (~2s settle)
    // so it breathes — pad fully out by intensity ~0.45, the supersaw owns
    // the battle. Forced home during the coda so cadences play on the pad.
    const war = now < coda ? 0 : Math.min(1, (sig.intensity || 0) / 0.45);
    peaceBus.gain.setTargetAtTime(Math.cos(war * Math.PI / 2), now, 0.5);
    warBus.gain.setTargetAtTime(Math.sin(war * Math.PI / 2), now, 0.5);
    // YOUR doom: the sub drone scales with charge level; ENEMY doom: the
    // audible approach layer + accelerating heartbeat; the counter CLASHES.
    // HOLDING Doom yourself wakes the same drone (max, never summed — one
    // well, one voice, even when both sides are doom-handed)
    const lvl = sig.doomLvl || 0, dtab = [0.14, 0.19, 0.24];
    let dg = lvl ? dtab[lvl - 1] : (sig.doom ? 0.12 : 0);
    if (stanceCur === "Doom") dg = Math.max(dg, lvl ? dtab[lvl - 1] : 0.10);
    doomDrone.gain.setTargetAtTime(dg, now, 1.0);
    doomTremG.gain.setTargetAtTime(dg * 0.3, now, 1.0);  // heartbeat depth gated WITH the drone
    const prox = sig.doomProx || 0;
    const eg = prox > 0.02 ? 0.05 + 0.13 * prox : 0;
    enemyG.gain.setTargetAtTime(eg, now, 0.5);
    enemyTremG.gain.setTargetAtTime(eg * 0.4, now, 0.5);
    enemyLP.frequency.setTargetAtTime(200 + 700 * prox, now, 0.5);
    enemyTrem.frequency.setTargetAtTime(0.8 + 2.7 * prox, now, 0.4);
    // the clash is YOUR counter-play cue: it sings only when YOU hold the
    // Maelstrom against a nearby enemy well (sig.mael also covers AI storms)
    const meMael = stanceCur === "Maelstrom" ? 1 : 0;
    clashG.gain.setTargetAtTime(0.06 * Math.min(meMael, prox > 0.1 ? 1 : 0) * Math.max(prox, 0.4), now, 0.4);
    // HOLDING Maelstrom keeps a quieter wash up and engages the whirl
    const maelHeld = stanceCur === "Maelstrom";
    maelWash.gain.setTargetAtTime(sig.mael ? 0.12 : (maelHeld ? 0.08 : 0), now, 0.8);
    maelWhirlPan.gain.setTargetAtTime(maelHeld ? 0.7 : 0, now, 0.4);
    maelWhirlLP.gain.setTargetAtTime(maelHeld ? 1500 : 0, now, 0.4);
    novaCycle(sig.nova);
    if (s.done && !wasDone) resolveEnd(s.outcome || (s.won ? "win" : "lose"));
    wasDone = s.done;
  }
  // hidden tab: a muted-by-OS game kept paying for (and sometimes leaking)
  // battle SFX — suspend the whole context, resume on return
  if (typeof document !== "undefined") {
    document.addEventListener("visibilitychange", () => {
      if (!ctx || !enabled) return;
      if (document.hidden) ctx.suspend(); else ctx.resume();
    });
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
    leadBus.gain.setTargetAtTime(m, ctx.currentTime, 0.04);  // lead follows the music
    sfxBus.gain.setTargetAtTime(sx, ctx.currentTime, 0.04);
  }
  return { update, thud, blip, stanceTap, setFlow, taiko, setEnabled, setVolumes,
           get enabled() { return enabled; },
           get debug() { return { state: ctx && ctx.state, beat: gridStep, stance: stanceCur, music: musicBus && musicBus.gain.value }; } };
})();
