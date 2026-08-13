/* ============================================================
   POSTURE.SCAN — UI + capture flow (KinaPT funnel)

   Two-photo scan (frontal + sagittal), fully on-device:
   MediaPipe Pose Landmarker supplies landmarks, 3D world
   landmarks and a segmentation mask; posture.js turns those
   into graded metrics; the alignment simulation mesh-warps the
   user's own photo toward plumb.
   ============================================================ */

import {
  LM, SEV_LABELS, sevBucket, clamp,
  checkQuality, analyseBackContour,
  assessFront, assessSide, scoreOf, verdictFor,
  idealTargets, solveAffine, makeDisplacer
} from './posture.js';

/* ---------- configuration ---------- */

// TODO: replace with the real KinaPT App Store listing before launch.
// Format: https://apps.apple.com/app/id<APP_ID>
const APPSTORE_URL = 'https://apps.apple.com/app/id0000000000';

/* ---------- custom media ----------
   Drop files into /assets and point these at them. Each one is optional:
   while it's null the app uses its built-in fallback (the procedural KINA
   core, and the browser's own voice reading the same script), so nothing
   is ever requested that doesn't exist.

   kinaLoop  — seamless looping video for the intro stage (mp4 or webm)
   kinaPoster— first frame, shown while the video decodes
   voIntro / voFront / voSide — ElevenLabs voiceover, one file per script  */
const ASSETS = {
  kinaLoop: null,   // e.g. 'assets/kina-loop.mp4'
  kinaPoster: null, // e.g. 'assets/kina-poster.jpg'
  voIntro: 'assets/vo-intro.mp3',
  voFront: null,    // e.g. 'assets/vo-front.mp3'
  voSide: null      // e.g. 'assets/vo-side.mp3'
};

const APP_URL = location.origin + location.pathname;

/* ---------- helpers ---------- */
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function showPanel(id) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  $(id).classList.add('active');
  // The full-screen KINA stage belongs to the intro only.
  document.body.classList.toggle('intro', id === 'panel-greet');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
document.body.classList.add('intro');

/* ---------- animated HUD background ---------- */
(function bgFx() {
  const c = $('bg-canvas'), ctx = c.getContext('2d');
  let w, h, t = 0;
  const resize = () => { w = c.width = innerWidth; h = c.height = innerHeight; };
  resize(); addEventListener('resize', resize);
  const dots = Array.from({ length: 40 }, () => ({
    x: Math.random(), y: Math.random(), s: Math.random() * 1.4 + 0.4, v: Math.random() * 0.0006 + 0.0002
  }));
  (function loop() {
    t++;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(36,221,221,0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 48) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y < h; y += 48) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    for (const d of dots) {
      d.y -= d.v; if (d.y < 0) d.y = 1;
      ctx.fillStyle = 'rgba(36,221,221,0.38)';
      ctx.beginPath(); ctx.arc(d.x * w, d.y * h, d.s, 0, 7); ctx.fill();
    }
    const sy = (t * 0.6) % (h + 200) - 100;
    const grad = ctx.createLinearGradient(0, sy - 60, 0, sy + 60);
    grad.addColorStop(0, 'rgba(36,221,221,0)');
    grad.addColorStop(0.5, 'rgba(36,221,221,0.06)');
    grad.addColorStop(1, 'rgba(36,221,221,0)');
    ctx.fillStyle = grad; ctx.fillRect(0, sy - 60, w, 120);
    requestAnimationFrame(loop);
  })();
})();

/* ============================================================
   KINA CORE — procedural HUD animation shown on the intro stage.
   Replaced automatically by assets/kina-loop.(webm|mp4) when that
   file exists, so the screen never looks empty before the custom
   loop is dropped in.
   ============================================================ */
/* Declared here rather than beside the loader below: the core's render loop
   starts during module evaluation and reads them, so `let` further down would
   be in the temporal dead zone on the first frames. */
let modelProgress = 0;    // 0–1 while the pose model downloads
let modelLoading = false;
let activeVO = null;      // voiceover element currently playing

/* Boot runs on the Begin tap, not on load, so the sequence can be scored —
   browsers block audio until the user interacts. */
let bootStart = 0;
let bootRunning = false;
let bootDone = false;

let coreEnergy = 0;       // 0 idle, 1 speaking — drives glow and bar height
let coreAnalyser = null;  // live FFT of the voiceover, when Web Audio is available
let audioCtx = null;

/* Audio must be unlocked from inside the tap handler itself — creating or
   resuming an AudioContext asynchronously afterwards leaves it suspended on
   iOS. Call this synchronously from the Begin gesture. */
function unlockAudio() {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    audioCtx = audioCtx || new AC();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    // iOS 16.4+: spoken content should ignore the ringer switch.
    if (navigator.audioSession) {
      try { navigator.audioSession.type = 'playback'; } catch (e) {}
    }
    startBed(); // the drone runs from the first tap onward
  } catch (e) { audioCtx = null; }
}

/* ============================================================
   SFX — synthesized interface sounds.

   Generated with oscillators and filtered noise rather than shipped as
   files: the page already carries ~18 MB of model and runtime, and a
   library of UI blips would add more for something a few lines of Web
   Audio can produce. Every cue is short and quiet by design — a HUD
   should feel responsive, not chatty.

   Silent until unlockAudio() has run, which happens on the first tap.
   ============================================================ */
const SFX_GAIN = 0.5; // master trim for the whole cue set

function sfxReady() { return voiceOn && audioCtx && audioCtx.state === 'running'; }

/** Pitched cue, optionally sweeping between two frequencies. */
function tone(f0, f1, dur, { type = 'sine', gain = 0.12, delay = 0 } = {}) {
  if (!sfxReady()) return;
  const t0 = audioCtx.currentTime + delay;
  const osc = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(f0, t0);
  if (f1 && f1 !== f0) osc.frequency.exponentialRampToValueAtTime(Math.max(1, f1), t0 + dur);
  // short attack, exponential tail — clicks if we ramp straight to zero
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(gain * SFX_GAIN, t0 + Math.min(0.02, dur * 0.2));
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  osc.connect(g); g.connect(audioCtx.destination);
  osc.start(t0); osc.stop(t0 + dur + 0.02);
}

/** Filtered noise, for whooshes and sweeps. */
function noise(dur, fFrom, fTo, gain = 0.08) {
  if (!sfxReady()) return;
  const t0 = audioCtx.currentTime;
  const frames = Math.ceil(audioCtx.sampleRate * dur);
  const buf = audioCtx.createBuffer(1, frames, audioCtx.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < frames; i++) d[i] = Math.random() * 2 - 1;
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  const bp = audioCtx.createBiquadFilter();
  bp.type = 'bandpass'; bp.Q.value = 1.2;
  bp.frequency.setValueAtTime(fFrom, t0);
  bp.frequency.exponentialRampToValueAtTime(fTo, t0 + dur);
  const g = audioCtx.createGain();
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(gain * SFX_GAIN, t0 + dur * 0.25);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  src.connect(bp); bp.connect(g); g.connect(audioCtx.destination);
  src.start(t0); src.stop(t0 + dur);
}

/* ---- ambient bed ----
   Measuring the reference audio settled the character: zero near-silent
   frames, spectral centroid 440–690 Hz, and 50–70% of all energy below
   300 Hz, with well under one transient per second. So it is a deep
   continuous drone with rare events — not a stream of bright blips. The
   bed below carries that weight; the cues sit low on top of it. */
let bed = null;

function startBed() {
  if (bed || !sfxReady()) return;
  const t0 = audioCtx.currentTime;
  const out = audioCtx.createGain();
  out.gain.setValueAtTime(0.0001, t0);
  out.gain.exponentialRampToValueAtTime(0.055 * SFX_GAIN, t0 + 2.5); // fades in
  out.connect(audioCtx.destination);

  // stacked low partials, slightly detuned so they beat against each other
  const oscs = [[55, 'sine', 1], [82.4, 'sine', 0.55], [110, 'triangle', 0.28]]
    .map(([f, type, lvl]) => {
      const o = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      o.type = type; o.frequency.value = f; g.gain.value = lvl;
      o.connect(g); g.connect(out); o.start(t0);
      return o;
    });

  // low-passed noise for air under the tone
  const frames = audioCtx.sampleRate * 2;
  const nb = audioCtx.createBuffer(1, frames, audioCtx.sampleRate);
  const d = nb.getChannelData(0);
  for (let i = 0; i < frames; i++) d[i] = Math.random() * 2 - 1;
  const nsrc = audioCtx.createBufferSource();
  nsrc.buffer = nb; nsrc.loop = true;
  const lp = audioCtx.createBiquadFilter();
  lp.type = 'lowpass'; lp.frequency.value = 260; lp.Q.value = 0.7;
  const ng = audioCtx.createGain(); ng.gain.value = 0.5;
  nsrc.connect(lp); lp.connect(ng); ng.connect(out);
  nsrc.start(t0);

  // very slow swell so it breathes rather than sitting flat
  const lfo = audioCtx.createOscillator();
  const lfoAmt = audioCtx.createGain();
  lfo.frequency.value = 0.07; lfoAmt.gain.value = 0.018 * SFX_GAIN;
  lfo.connect(lfoAmt); lfoAmt.connect(out.gain); lfo.start(t0);

  bed = { out, oscs, nsrc, lfo };
}

function stopBed() {
  if (!bed) return;
  const t0 = audioCtx.currentTime;
  try {
    bed.out.gain.cancelScheduledValues(t0);
    bed.out.gain.setValueAtTime(Math.max(0.0001, bed.out.gain.value), t0);
    bed.out.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.4);
    bed.oscs.forEach(o => o.stop(t0 + 0.5));
    bed.nsrc.stop(t0 + 0.5);
    bed.lfo.stop(t0 + 0.5);
  } catch (e) {}
  bed = null;
}

/* ---- action bed ----
   Driving pulse for the frantic first phase of the boot, cut dead at the
   reset. Separate from the ambient bed, which is the idle drone. */
let action = null;

function startAction() {
  if (action || !sfxReady()) return;
  const t0 = audioCtx.currentTime;
  const out = audioCtx.createGain();
  out.gain.setValueAtTime(0.0001, t0);
  out.gain.exponentialRampToValueAtTime(0.10 * SFX_GAIN, t0 + 0.5);
  out.connect(audioCtx.destination);

  const drone = audioCtx.createOscillator();
  drone.type = 'sawtooth'; drone.frequency.value = 41.2;
  const dlp = audioCtx.createBiquadFilter();
  dlp.type = 'lowpass'; dlp.frequency.value = 180;
  const dg = audioCtx.createGain(); dg.gain.value = 0.5;
  drone.connect(dlp); dlp.connect(dg); dg.connect(out); drone.start(t0);

  // 16th-note pulse driving the montage
  const pulse = audioCtx.createOscillator();
  pulse.type = 'square'; pulse.frequency.value = 82.4;
  const pg = audioCtx.createGain(); pg.gain.value = 0;
  pulse.connect(pg); pg.connect(out); pulse.start(t0);
  for (let i = 0; i < 80; i++) {
    const at = t0 + i * 0.15;
    pg.gain.setValueAtTime(0.34, at);
    pg.gain.exponentialRampToValueAtTime(0.001, at + 0.11);
  }
  action = { out, drone, pulse };
}

function stopAction(fade = 0.06) {
  if (!action) return;
  const t0 = audioCtx.currentTime;
  try {
    action.out.gain.cancelScheduledValues(t0);
    action.out.gain.setValueAtTime(Math.max(0.0001, action.out.gain.value), t0);
    action.out.gain.exponentialRampToValueAtTime(0.0001, t0 + fade);
    action.drone.stop(t0 + fade + 0.05);
    action.pulse.stop(t0 + fade + 0.05);
  } catch (e) {}
  action = null;
}

/* Cues pitched an octave-plus below the first pass, to sit in the same
   register as the reference rather than chiming over it. */
const SFX = {
  tap:    () => tone(330, 330, 0.05, { type: 'sine', gain: 0.05 }),
  blip:   () => tone(560, 560, 0.06, { type: 'sine', gain: 0.045 }),
  powerUp: () => {
    tone(40, 190, 1.1, { type: 'sine', gain: 0.22 });        // sub sweep
    tone(80, 380, 1.1, { type: 'sine', gain: 0.10 });
    noise(1.0, 90, 700, 0.07);
  },
  whoosh: () => { noise(0.55, 700, 90, 0.09); tone(150, 70, 0.5, { type: 'sine', gain: 0.09 }); },
  lock:   () => { tone(180, 180, 0.09, { type: 'sine', gain: 0.13 });
                  tone(270, 270, 0.12, { type: 'sine', gain: 0.10, delay: 0.08 }); },
  reject: () => { tone(150, 62, 0.5, { type: 'sine', gain: 0.16 });
                  noise(0.35, 320, 80, 0.06); },
  scan:   () => { tone(90, 240, 0.7, { type: 'sine', gain: 0.10 });
                  noise(0.7, 160, 900, 0.05); },
  reveal: () => {                                            // low rising figure
    [131, 165, 196, 262].forEach((f, i) =>
      tone(f, f, 0.42, { type: 'sine', gain: 0.13, delay: i * 0.1 }));
    tone(65, 98, 0.9, { type: 'sine', gain: 0.14 });          // sub underneath
    noise(0.9, 120, 800, 0.05);
  },
  // boot cues
  telemetry: () => tone(880 + Math.random() * 900, 0, 0.035, { type: 'square', gain: 0.03 }),
  swoosh:  () => noise(0.3, 2400, 200, 0.07),
  target:  () => { tone(1400, 1400, 0.06, { type: 'sine', gain: 0.10 });
                   tone(1400, 1400, 0.06, { type: 'sine', gain: 0.08, delay: 0.18 }); },
  staticZap: () => { noise(0.42, 1200, 5000, 0.20); tone(60, 40, 0.3, { type: 'square', gain: 0.10 }); }
};

/** Play a cue and give the core a matching visual kick. */
function sfx(name, kick = 0.5) {
  if (!SFX[name]) return;
  SFX[name]();
  if (sfxReady()) coreEnergy = Math.max(coreEnergy, kick);
}

/* Route a voiceover element through an analyser so the spectrum ring follows
   the real waveform. Only ever done on a RUNNING context: createMediaElement-
   Source detaches the element from the default output, so wiring it up while
   the context is suspended would play the briefing into silence. Each element
   can only be sourced once, so the node is cached alongside it. */
function wireAnalyser(node) {
  if (node.analyser || node.wired) return;
  if (!audioCtx || audioCtx.state !== 'running') return; // leave it on the speaker
  node.wired = true;
  try {
    const src = audioCtx.createMediaElementSource(node.el);
    try {
      const an = audioCtx.createAnalyser();
      an.fftSize = 256;                 // 128 bins — one per pair of bars
      an.smoothingTimeConstant = 0.72;  // damped enough to read as motion, not noise
      src.connect(an);
      an.connect(audioCtx.destination);
      node.analyser = an;
      node.data = new Uint8Array(an.frequencyBinCount);
    } catch (inner) {
      try { src.connect(audioCtx.destination); } catch (e2) {}
    }
  } catch (e) { /* already sourced, or unsupported — audio still plays */ }
}
function detachAnalyser() { coreAnalyser = null; }

/* Preloaded voiceover elements, so a tap plays instantly instead of waiting
   on the network with a fallback timer racing it. */
const voNodes = {};
function getVONode(key, file) {
  if (!file) return null;
  if (!voNodes[key]) {
    const el = new Audio();
    el.preload = 'auto';
    el.playsInline = true;
    el.src = file;
    voNodes[key] = { el, analyser: null, data: null, wired: false };
  }
  return voNodes[key];
}

(function kinaCore() {
  const c = $('kina-core');
  if (!c) return;
  const ctx = c.getContext('2d');
  let t = 0;
  /* Bloom is what separates the reference from crisp vector line-work: every
     element there is blown out and glowing. Rendering the frame again through
     a downscaled blur with additive compositing is the cheap way to get it —
     quarter resolution keeps it affordable on a phone. */
  const glow = document.createElement('canvas');
  const gctx = glow.getContext('2d');
  const bloomOK = typeof gctx.filter === 'string';

  const fit = () => {
    const r = c.getBoundingClientRect();
    const dpr = Math.min(2, devicePixelRatio || 1);
    c.width = Math.max(1, r.width * dpr);
    c.height = Math.max(1, r.height * dpr);
    glow.width = Math.max(1, Math.round(c.width / 4));
    glow.height = Math.max(1, Math.round(c.height / 4));
  };
  fit(); addEventListener('resize', fit);

  function applyBloom(strength = 1) {
    if (!bloomOK || !glow.width) return;
    gctx.clearRect(0, 0, glow.width, glow.height);
    gctx.drawImage(c, 0, 0, glow.width, glow.height);
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = 0.55 * strength;
    ctx.filter = 'blur(5px)';
    ctx.drawImage(glow, 0, 0, c.width, c.height);
    ctx.globalAlpha = 0.30 * strength;
    ctx.filter = 'blur(14px)';       // wider, softer halo
    ctx.drawImage(glow, 0, 0, c.width, c.height);
    ctx.restore();
  }

  const BLUE = '36,221,221';
  const HOT = '168,245,245';
  // Accents sampled from the reference frames (see palette note in index.html).
  const GOLD = '241,220,42';
  const MAGENTA = '252,64,201';

  const arc = (cx, cy, r, from, to, w, alpha, col = BLUE) => {
    ctx.beginPath();
    ctx.arc(cx, cy, r, from, to);
    ctx.strokeStyle = `rgba(${col},${alpha})`;
    ctx.lineWidth = w;
    ctx.stroke();
  };

  // Telemetry strings, refreshed occasionally so the panel feels live.
  const TELEM_L = ['SPINE.MAP', 'JOINT.VEC', 'PLUMB.REF', 'DEPTH.Z', 'MASK.SEG', 'GAIT.IDX'];
  const telemetry = TELEM_L.map(k => ({ k, v: Math.floor(Math.random() * 900 + 100) }));
  setInterval(() => {
    const i = Math.floor(Math.random() * telemetry.length);
    telemetry[i].v = Math.floor(Math.random() * 900 + 100);
  }, 700);

  /* ---- boot sequence ----
     Plays once on load: a point ignites, a shockwave blooms outward, then the
     rings, spectrum, scan modules and telemetry assemble in sequence before
     settling into the steady core. */
  const BOOT_MS = 15000;
  const easeOut = (x) => 1 - Math.pow(1 - x, 3);
  // ramp from 0→1 across a slice of the boot timeline
  const ph = (b, from, to) => easeOut(clamp((b - from) / (to - from), 0, 1));

  const SCAN_MODULES = [
    { label: 'SKELETAL TRACE', at: 0.62 },
    { label: 'PLUMB REFERENCE', at: 0.72 },
    { label: 'SPINAL CURVE MAP', at: 0.82 }
  ];

  /* Status callouts that cycle around the assembly, matching the reference's
     running commentary ("SCANNING FOR USER DEVICES", "RETINA SCAN"...). */
  const CALLOUTS = [
    'SCANNING FOR SUBJECT',
    'JOINT VECTOR LOCK',
    'DEPTH FIELD STABLE',
    'PLUMB LINE ACQUIRED',
    'SEGMENTING SILHOUETTE',
    'CURVE FIT NOMINAL',
    'GRADING CHECKPOINTS',
    'CALIBRATION HELD'
  ];
  let calloutIdx = 0, calloutAt = 0;

  /* Smooth pseudo-random field driving the spectrum bar heights. */
  const wave = (a, time) =>
    0.5 + 0.5 * (
      Math.sin(a * 7 + time * 1.7) * 0.42 +
      Math.sin(a * 13 - time * 2.3) * 0.30 +
      Math.sin(a * 23 + time * 3.1) * 0.18 +
      Math.sin(a * 3 - time * 1.1) * 0.10
    );

  /* ---- display modes ----
     The reference cycles through distinct screens rather than holding one
     assembly: a ring core, an iris/retina scan, a geographic projection with
     pings, and crosshatched data panels. Each collapses toward centre and the
     next builds out of it. While KINA is speaking we stay on the ring, since
     that's the mode whose spectrum is driven by the voice. */
  const MODES = ['ring', 'iris', 'globe', 'panels'];
  const MODE_IN = 0.55, MODE_HOLD = 6.0, MODE_OUT = 0.45;
  const MODE_CYCLE = MODE_IN + MODE_HOLD + MODE_OUT;
  let modeIdx = 0, modeStart = 0, lastSwitch = -9;

  const MODE_LABEL = {
    ring: 'ALIGNMENT CORE', iris: 'OPTICAL CALIBRATION',
    globe: 'REFERENCE FRAME', panels: 'DATA CHANNELS'
  };

  function drawIris(cx, cy, R, k) {
    // concentric lens rings with radial spokes and a hot pupil
    for (let i = 0; i < 6; i++) {
      const rr = R * (0.55 + i * 0.42);
      arc(cx, cy, rr, t * (i % 2 ? -0.3 : 0.3) + i, t * (i % 2 ? -0.3 : 0.3) + i + 5.2,
          Math.max(1, R * 0.03), (0.5 - i * 0.05), i === 2 ? GOLD : BLUE);
    }
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(t * 0.5);
    for (let i = 0; i < 60; i++) {
      ctx.rotate(Math.PI * 2 / 60);
      const long = i % 5 === 0;
      ctx.beginPath();
      ctx.moveTo(R * 1.05, 0); ctx.lineTo(R * (long ? 1.45 : 1.25), 0);
      ctx.strokeStyle = `rgba(${long ? GOLD : BLUE},${long ? 0.6 : 0.25})`;
      ctx.lineWidth = Math.max(1, R * 0.02); ctx.stroke();
    }
    ctx.restore();
    const g2 = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.5);
    g2.addColorStop(0, `rgba(${HOT},0.9)`);
    g2.addColorStop(0.5, `rgba(${MAGENTA},0.5)`);
    g2.addColorStop(1, `rgba(${BLUE},0.05)`);
    ctx.fillStyle = g2;
    ctx.beginPath(); ctx.arc(cx, cy, R * 0.5, 0, Math.PI * 2); ctx.fill();
    // sweeping lens flare bar
    ctx.strokeStyle = `rgba(${HOT},0.35)`;
    ctx.lineWidth = Math.max(1, R * 0.02);
    ctx.beginPath();
    ctx.moveTo(cx - R * 2.4, cy + Math.sin(t) * R * 0.5);
    ctx.lineTo(cx + R * 2.4, cy + Math.sin(t) * R * 0.5);
    ctx.stroke();
  }

  function drawGlobe(cx, cy, R, k) {
    const gr = R * 1.5;
    arc(cx, cy, gr, 0, Math.PI * 2, Math.max(1, R * 0.02), 0.5);
    // latitudes as flattened ellipses
    for (let i = 1; i < 5; i++) {
      const y = -gr + (i / 5) * gr * 2;
      const rx = Math.sqrt(Math.max(0, gr * gr - y * y));
      ctx.beginPath();
      ctx.ellipse(cx, cy + y, rx, rx * 0.22, 0, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${BLUE},0.28)`;
      ctx.lineWidth = Math.max(1, R * 0.012); ctx.stroke();
    }
    // longitudes sweeping as the globe turns
    for (let i = 0; i < 6; i++) {
      const phase = t * 0.4 + i * (Math.PI / 6);
      const rx = Math.abs(Math.cos(phase)) * gr;
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, gr, 0, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${BLUE},0.2)`;
      ctx.lineWidth = Math.max(1, R * 0.01); ctx.stroke();
    }
    // pings on the surface
    for (let i = 0; i < 5; i++) {
      const a = t * 0.4 + i * 1.25;
      const px = cx + Math.cos(a) * gr * 0.72;
      const py = cy + Math.sin(i * 2.1) * gr * 0.55;
      const pulse = (Math.sin(t * 3 + i) + 1) / 2;
      ctx.fillStyle = `rgba(255,92,92,${0.5 + pulse * 0.5})`;
      ctx.beginPath(); ctx.arc(px, py, Math.max(2, R * 0.05), 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = `rgba(255,92,92,${0.5 * (1 - pulse)})`;
      ctx.lineWidth = Math.max(1, R * 0.015);
      ctx.beginPath(); ctx.arc(px, py, R * 0.06 + pulse * R * 0.28, 0, Math.PI * 2); ctx.stroke();
    }
    arc(cx, cy, gr * 1.22, t * 0.6, t * 0.6 + 2.4, Math.max(1.5, R * 0.045), 0.7, GOLD);
  }

  function drawPanels(cx, cy, R, k) {
    // crosshatched data blocks, the reference's "loading" panels
    const pw = R * 4.6, phh = R * 0.62;
    for (let row = 0; row < 3; row++) {
      const py = cy + (row - 1) * R * 1.05;
      const x0 = cx - pw / 2;
      ctx.strokeStyle = `rgba(${GOLD},0.55)`;
      ctx.lineWidth = Math.max(1, R * 0.018);
      ctx.strokeRect(x0, py - phh / 2, pw, phh);
      const cells = 6;
      for (let i = 0; i < cells; i++) {
        const cw = pw / cells, cxx = x0 + i * cw;
        // fill each cell progressively, sweeping left to right
        const on = clamp((t * 0.8 + row * 0.4) % 3 - i * 0.16, 0, 1);
        if (on <= 0) continue;
        ctx.strokeStyle = `rgba(${GOLD},${0.22 + on * 0.35})`;
        ctx.lineWidth = Math.max(1, R * 0.01);
        ctx.beginPath();
        ctx.moveTo(cxx, py - phh / 2); ctx.lineTo(cxx + cw, py + phh / 2);
        ctx.moveTo(cxx + cw, py - phh / 2); ctx.lineTo(cxx, py + phh / 2);
        ctx.stroke();
        ctx.strokeStyle = `rgba(${BLUE},0.3)`;
        ctx.beginPath(); ctx.moveTo(cxx, py - phh / 2); ctx.lineTo(cxx, py + phh / 2); ctx.stroke();
      }
    }
  }

  /* ============================================================
     BOOT SEQUENCE — 15s, scored, per the supplied timing sheet.
       0–2   retina scan / radar / geographic, glitching to hexagons
       2–5   portal flash shattering into particles, warped grids
       5–7   matrix grid, loading bar stretching outward
       7–10  bar dissolves into flowing fibre-optic data
       10–12 black, green crosshair, red grid expanding — silence
       12–15 static burst snapping into the stable radar HUD
     ============================================================ */
  const GREEN = '90,255,140';
  const RED = '237,77,66';

  let particles = null;
  const seg = (bt, a, z) => clamp((bt - a) / (z - a), 0, 1);

  function glitch(W, H, amt) {
    if (amt <= 0.01) return;
    const slices = Math.round(2 + amt * 7);
    for (let i = 0; i < slices; i++) {
      const sy = Math.random() * H;
      const sh = H * (0.01 + Math.random() * 0.07);
      const dx = (Math.random() - 0.5) * W * 0.22 * amt;
      try { ctx.drawImage(c, 0, sy, W, sh, dx, sy, W, sh); } catch (e) {}
      if (Math.random() < 0.5) {
        ctx.fillStyle = `rgba(${Math.random() < 0.5 ? GREEN : RED},${0.05 + Math.random() * 0.12 * amt})`;
        ctx.fillRect(0, sy, W, sh);
      }
    }
  }

  function hexField(W, H, cx, cy, k, spin) {
    const s = Math.min(W, H) * 0.075;
    ctx.strokeStyle = `rgba(${GREEN},${0.18 + 0.4 * k})`;
    ctx.lineWidth = Math.max(1, W * 0.0022);
    for (let row = -1; row * s * 1.5 < H + s; row++) {
      for (let col = -1; col * s * 1.73 < W + s; col++) {
        const hx = col * s * 1.73 + (row % 2 ? s * 0.87 : 0);
        const hy = row * s * 1.5;
        const d = Math.hypot(hx - cx, hy - cy) / Math.max(W, H);
        if (d > k * 1.3) continue;
        ctx.beginPath();
        for (let v = 0; v < 6; v++) {
          const a = spin + v * Math.PI / 3;
          const px = hx + Math.cos(a) * s * 0.55, py = hy + Math.sin(a) * s * 0.55;
          v ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        }
        ctx.closePath(); ctx.stroke();
      }
    }
  }

  function drawBoot(bt, W, H, cx, cy) {
    const M = Math.min(W, H);
    ctx.fillStyle = '#01050a';
    ctx.fillRect(0, 0, W, H);
    const fsz = Math.max(9, W * 0.022);
    ctx.font = `${fsz}px "Consolas", monospace`;
    ctx.textBaseline = 'middle';

    /* ---- 0–2s: retina scan, radar, geographic — frantic ---- */
    if (bt < 2.2) {
      const k = seg(bt, 0, 2);
      ctx.textAlign = 'left';
      ctx.fillStyle = `rgba(${GREEN},${0.5 + 0.5 * Math.sin(bt * 22)})`;
      ctx.fillText('RETINA SCAN', W * 0.06, H * 0.30);
      ctx.textAlign = 'right';
      ctx.fillText('GEOGRAPHIC', W * 0.94, H * 0.30);
      ctx.textAlign = 'left';
      ctx.fillStyle = `rgba(${GREEN},0.4)`;
      ctx.fillText('ENABLING SENSOR ARRAY', W * 0.06, H * 0.36);
      ctx.textAlign = 'right';
      ctx.fillText('SPY SEQUENCE', W * 0.94, H * 0.36);

      // circular radar/eye
      const rr = M * 0.2;
      for (let i = 0; i < 4; i++)
        arc(cx, cy, rr * (0.4 + i * 0.24), bt * (i % 2 ? -3 : 3), bt * (i % 2 ? -3 : 3) + 4.4,
            Math.max(1, M * 0.006), 0.55, GREEN);
      const eye = ctx.createRadialGradient(cx, cy, 0, cx, cy, rr * 0.42);
      eye.addColorStop(0, `rgba(255,255,255,0.9)`);
      eye.addColorStop(0.5, `rgba(${GREEN},0.6)`);
      eye.addColorStop(1, `rgba(${GREEN},0)`);
      ctx.fillStyle = eye;
      ctx.beginPath(); ctx.arc(cx, cy, rr * 0.42, 0, 7); ctx.fill();
      // sweep arm
      ctx.strokeStyle = `rgba(${GREEN},0.8)`;
      ctx.lineWidth = Math.max(1, M * 0.008);
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(bt * 9) * rr, cy + Math.sin(bt * 9) * rr);
      ctx.stroke();

      if (bt > 1.2) hexField(W, H, cx, cy, seg(bt, 1.2, 2.2), bt);
      glitch(W, H, 0.5 + 0.5 * Math.abs(Math.sin(bt * 13)));
      return;
    }

    /* ---- 2–5s: portal flash, particle shatter, warped grids ---- */
    if (bt < 5) {
      const k = seg(bt, 2.2, 5);
      // portal, alive only for a beat before it shatters
      if (bt < 2.75) {
        const pr = M * (0.08 + seg(bt, 2.2, 2.75) * 0.28);
        const pg = ctx.createRadialGradient(cx, cy, 0, cx, cy, pr);
        pg.addColorStop(0, 'rgba(255,255,255,0.95)');
        pg.addColorStop(0.35, `rgba(${MAGENTA},0.8)`);
        pg.addColorStop(0.7, `rgba(${GOLD},0.6)`);
        pg.addColorStop(1, `rgba(${GREEN},0)`);
        ctx.fillStyle = pg;
        ctx.beginPath(); ctx.arc(cx, cy, pr, 0, 7); ctx.fill();
      } else {
        if (!particles) {
          particles = Array.from({ length: 150 }, () => {
            const a = Math.random() * Math.PI * 2, sp = 0.25 + Math.random() * 1.5;
            return { a, sp, r: M * 0.05, s: 1 + Math.random() * 2.5 };
          });
        }
        const age = bt - 2.75;
        for (const p of particles) {
          const d = p.r + p.sp * age * M * 0.55;
          const px = cx + Math.cos(p.a) * d, py = cy + Math.sin(p.a) * d * 0.8;
          ctx.fillStyle = `rgba(${GREEN},${Math.max(0, 0.9 - age * 0.5)})`;
          ctx.fillRect(px, py, p.s * 2, p.s * 2);
        }
      }
      // warped data waves
      ctx.strokeStyle = `rgba(${GREEN},0.35)`;
      ctx.lineWidth = Math.max(1, W * 0.002);
      for (let i = 0; i < 9; i++) {
        ctx.beginPath();
        for (let x = 0; x <= W; x += 12) {
          const y = H * (0.12 + i * 0.095)
            + Math.sin(x * 0.012 + bt * 6 + i) * H * 0.05 * (0.4 + k);
          x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
      }
      hexField(W, H, cx, cy, 1 - k * 0.6, bt * 0.5);
      glitch(W, H, 0.75 * (1 - k * 0.5));
      return;
    }

    /* ---- 5–7s: matrix grid, loading bar stretches outward ---- */
    if (bt < 7) {
      const k = seg(bt, 5, 7);
      const g = M * 0.045;
      ctx.strokeStyle = `rgba(${GREEN},0.16)`;
      ctx.lineWidth = 1;
      for (let x = 0; x < W; x += g) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
      for (let y = 0; y < H; y += g) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
      // falling cells, matrix-ish
      for (let i = 0; i < 40; i++) {
        const cxx = Math.floor((i * 7919) % Math.floor(W / g)) * g;
        const cyy = ((bt * (0.3 + (i % 5) * 0.12) * H * 1.4) + i * 137) % H;
        ctx.fillStyle = `rgba(${GREEN},${0.15 + 0.5 * Math.random()})`;
        ctx.fillRect(cxx, Math.floor(cyy / g) * g, g, g);
      }
      // loading bar stretching from centre outward
      const bw = W * 0.94 * easeOut(k);
      const bh = Math.max(6, H * 0.022);
      ctx.strokeStyle = `rgba(${GREEN},0.9)`;
      ctx.lineWidth = Math.max(1, W * 0.003);
      ctx.strokeRect(cx - bw / 2, cy - bh / 2, bw, bh);
      ctx.fillStyle = `rgba(${GREEN},0.55)`;
      ctx.fillRect(cx - bw / 2 + 2, cy - bh / 2 + 2, (bw - 4) * k, bh - 4);
      ctx.textAlign = 'center';
      ctx.fillStyle = `rgba(${GREEN},0.95)`;
      ctx.font = `600 ${fsz * 1.5}px "Consolas", monospace`;
      ctx.fillText(Math.round(k * 100) + '%', cx, cy - bh * 2.2);
      ctx.font = `${fsz}px "Consolas", monospace`;
      ctx.fillText('loading…', cx, cy + bh * 2.4);
      return;
    }

    /* ---- 7–10s: bar dissolves into flowing fibre-optic data ---- */
    if (bt < 10) {
      const k = seg(bt, 7, 10);
      for (let i = 0; i < 26; i++) {
        const base = H * (0.08 + (i / 26) * 0.84);
        const amp = H * 0.10 * (0.3 + Math.sin(i * 1.7) * 0.7);
        const hot = i % 5 === 0;
        ctx.strokeStyle = hot ? `rgba(${HOT},${0.5 + 0.4 * k})` : `rgba(${GREEN},${0.22 + 0.4 * k})`;
        ctx.lineWidth = Math.max(1, W * (hot ? 0.004 : 0.0022));
        ctx.beginPath();
        for (let x = -20; x <= W + 20; x += 10) {
          const y = base + Math.sin(x * 0.009 + bt * 5 + i * 0.8) * amp * k
                         + Math.sin(x * 0.021 - bt * 3) * amp * 0.35 * k;
          x <= -20 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        // travelling packet along the strand
        const px = ((bt * 0.55 + i * 0.13) % 1) * W;
        const py = base + Math.sin(px * 0.009 + bt * 5 + i * 0.8) * amp * k;
        ctx.fillStyle = `rgba(${HOT},${0.8 * k})`;
        ctx.beginPath(); ctx.arc(px, py, Math.max(1.5, W * 0.004), 0, 7); ctx.fill();
      }
      if (bt < 7.4) glitch(W, H, 0.35 * (1 - seg(bt, 7, 7.4)));
      return;
    }

    /* ---- 10–12s: the reset — black, crosshair, red grid ---- */
    if (bt < 12) {
      const k = seg(bt, 10, 12);
      const ch = M * 0.05;
      ctx.strokeStyle = `rgba(${GREEN},${0.9 * clamp(seg(bt, 10.15, 10.5), 0, 1)})`;
      ctx.lineWidth = Math.max(1, W * 0.003);
      ctx.beginPath();
      ctx.moveTo(cx - ch, cy); ctx.lineTo(cx + ch, cy);
      ctx.moveTo(cx, cy - ch); ctx.lineTo(cx, cy + ch);
      ctx.stroke();
      arc(cx, cy, ch * 0.55, 0, Math.PI * 2, Math.max(1, W * 0.002),
          0.8 * clamp(seg(bt, 10.2, 10.6), 0, 1), GREEN);
      // red grid expanding outward
      const gk = seg(bt, 10.7, 12);
      if (gk > 0) {
        const half = M * 0.06 + easeOut(gk) * M * 0.44;
        ctx.strokeStyle = `rgba(${RED},${0.75 * (1 - gk * 0.25)})`;
        ctx.lineWidth = Math.max(1, W * 0.0035);
        ctx.strokeRect(cx - half, cy - half, half * 2, half * 2);
        ctx.lineWidth = Math.max(1, W * 0.0015);
        ctx.strokeStyle = `rgba(${RED},${0.35 * (1 - gk * 0.3)})`;
        for (let i = 1; i < 4; i++) {
          const o = -half + (i / 4) * half * 2;
          ctx.beginPath();
          ctx.moveTo(cx + o, cy - half); ctx.lineTo(cx + o, cy + half);
          ctx.moveTo(cx - half, cy + o); ctx.lineTo(cx + half, cy + o);
          ctx.stroke();
        }
      }
      return;
    }

    /* ---- 12–15s: static burst snapping into the radar HUD ---- */
    const k = seg(bt, 12, 15);
    if (bt < 12.45) {                                    // grey TV static
      const cell = Math.max(2, W * 0.006);
      for (let y = 0; y < H; y += cell) {
        for (let x = 0; x < W; x += cell) {
          const v = 40 + Math.random() * 180;
          ctx.fillStyle = `rgba(${v},${v},${v},0.85)`;
          ctx.fillRect(x, y, cell, cell);
        }
      }
      return;
    }
    const on = seg(bt, 12.45, 12.8);
    const rr = M * 0.28;
    ctx.globalAlpha = on;
    // rotating radar
    for (let i = 1; i <= 4; i++)
      arc(cx, cy, rr * i / 4, 0, Math.PI * 2, Math.max(1, M * 0.004), 0.35);
    ctx.strokeStyle = `rgba(${BLUE},0.3)`;
    ctx.lineWidth = Math.max(1, M * 0.003);
    for (let i = 0; i < 8; i++) {
      const a = i * Math.PI / 4;
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr); ctx.stroke();
    }
    const sweep = bt * 1.6;
    const sg = ctx.createConicGradient
      ? ctx.createConicGradient(sweep, cx, cy) : null;
    if (sg) {
      sg.addColorStop(0, `rgba(${GREEN},0.45)`);
      sg.addColorStop(0.12, `rgba(${GREEN},0)`);
      sg.addColorStop(1, `rgba(${GREEN},0)`);
      ctx.fillStyle = sg;
      ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = `rgba(${GREEN},0.9)`;
    ctx.lineWidth = Math.max(1, M * 0.006);
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(sweep) * rr, cy + Math.sin(sweep) * rr); ctx.stroke();
    // side data panels
    ctx.strokeStyle = `rgba(${BLUE},0.4)`;
    ctx.lineWidth = Math.max(1, W * 0.002);
    for (let i = 0; i < 3; i++) {
      const py = H * (0.3 + i * 0.16);
      ctx.strokeRect(W * 0.04, py, W * 0.16, H * 0.11);
      ctx.strokeRect(W * 0.80, py, W * 0.16, H * 0.11);
    }
    ctx.globalAlpha = 1;
  }

  (function frame() {
    t += 0.016;

    // Sample the live voiceover spectrum, if one is playing.
    let spec = null;
    if (coreAnalyser) {
      coreAnalyser.an.getByteFrequencyData(coreAnalyser.data);
      spec = coreAnalyser.data;
      let sum = 0;
      for (let i = 0; i < spec.length; i++) sum += spec[i];
      coreEnergy = Math.max(coreEnergy, Math.min(1, (sum / spec.length) / 96));
    }

    const W = c.width, H = c.height;
    const cx = W / 2, cy = H / 2;

    // Before the tap: a quiet holding pattern, so the boot has impact.
    if (!bootRunning && !bootDone) {
      ctx.fillStyle = '#01050a';
      ctx.fillRect(0, 0, W, H);
      const M = Math.min(W, H);
      arc(cx, cy, M * 0.2, t * 0.25, t * 0.25 + 2.2, Math.max(1, M * 0.004), 0.28);
      arc(cx, cy, M * 0.16, -t * 0.18, -t * 0.18 + 1.4, Math.max(1, M * 0.003), 0.2);
      const gl = ctx.createRadialGradient(cx, cy, 0, cx, cy, M * 0.3);
      gl.addColorStop(0, `rgba(${BLUE},${0.10 + Math.sin(t) * 0.03})`);
      gl.addColorStop(1, `rgba(${BLUE},0)`);
      ctx.fillStyle = gl; ctx.fillRect(0, 0, W, H);
      applyBloom(0.7);
      requestAnimationFrame(frame);
      return;
    }

    // The scored 15s sequence owns the screen while it runs.
    if (bootRunning) {
      const bt = (performance.now() - bootStart) / 1000;
      if (bt >= BOOT_MS / 1000) { bootRunning = false; bootDone = true; }
      else {
        drawBoot(bt, W, H, cx, cy);
        applyBloom(1.15);            // the montage is the most blown-out part
        requestAnimationFrame(frame);
        return;
      }
    }

    const b = 1; // boot complete — the steady assembly is fully revealed

    // Sized so the outermost tick ring (2.88 R) clears the frame edges.
    const Rsteady = Math.min(W, H) * 0.150 * (1 + Math.sin(t * 2.1) * 0.012 + coreEnergy * 0.05);
    // The assembly falls in from oversize as it locks on.
    const R = Rsteady * (1 + (1 - ph(b, 0.28, 0.72)) * 0.9);
    const energy = 0.28 + coreEnergy * 0.72;

    ctx.clearRect(0, 0, W, H);

    // ---- ignition flash: a hot point that blooms and fades
    const ignite = 1 - clamp((b - 0.02) / 0.22, 0, 1);
    if (ignite > 0.001) {
      const fr = Math.min(W, H) * (0.02 + (1 - ignite) * 0.55);
      const fg = ctx.createRadialGradient(cx, cy, 0, cx, cy, fr);
      fg.addColorStop(0, `rgba(255,255,255,${0.95 * ignite})`);
      fg.addColorStop(0.25, `rgba(${HOT},${0.8 * ignite})`);
      fg.addColorStop(1, `rgba(${BLUE},0)`);
      ctx.fillStyle = fg;
      ctx.fillRect(0, 0, W, H);
    }

    // ---- shockwave ring travelling outward from the ignition
    const sw = clamp((b - 0.04) / 0.5, 0, 1);
    if (sw > 0 && sw < 1) {
      const sr = easeOut(sw) * Math.min(W, H) * 0.85;
      ctx.beginPath();
      ctx.arc(cx, cy, sr, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${HOT},${(1 - sw) * 0.55})`;
      ctx.lineWidth = Math.max(1, (1 - sw) * Math.min(W, H) * 0.02);
      ctx.stroke();
      // radial streaks chasing the wave
      ctx.lineWidth = Math.max(1, (1 - sw) * 3);
      for (let i = 0; i < 24; i++) {
        const a = (i / 24) * Math.PI * 2 + 0.2;
        ctx.beginPath();
        ctx.strokeStyle = `rgba(${BLUE},${(1 - sw) * 0.3})`;
        ctx.moveTo(cx + Math.cos(a) * sr * 0.72, cy + Math.sin(a) * sr * 0.72);
        ctx.lineTo(cx + Math.cos(a) * sr, cy + Math.sin(a) * sr);
        ctx.stroke();
      }
    }

    // ---- honeycomb backdrop, faint, drifting (present across the reference)
    {
      const hs = Math.min(W, H) * 0.085;
      const drift = (t * 6) % (hs * 1.5);
      ctx.strokeStyle = `rgba(${BLUE},0.055)`;
      ctx.lineWidth = Math.max(1, W * 0.0015);
      for (let row = -1; row * hs * 1.5 - drift < H + hs; row++) {
        for (let col = -1; col * hs * 1.73 < W + hs; col++) {
          const hx = col * hs * 1.73 + (row % 2 ? hs * 0.87 : 0);
          const hy = row * hs * 1.5 - drift;
          ctx.beginPath();
          for (let v = 0; v < 6; v++) {
            const a = v * Math.PI / 3;
            const px = hx + Math.cos(a) * hs * 0.5, py = hy + Math.sin(a) * hs * 0.5;
            v ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
          }
          ctx.closePath(); ctx.stroke();
        }
      }
    }

    // ---- ambient glow behind the assembly
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 4);
    g.addColorStop(0, `rgba(${BLUE},${(0.16 + coreEnergy * 0.16) * ph(b, 0.05, 0.4)})`);
    g.addColorStop(0.4, `rgba(${BLUE},${0.05 * ph(b, 0.05, 0.4)})`);
    g.addColorStop(1, `rgba(${BLUE},0)`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    /* ---- mode cycling ----
       Hold on the ring while KINA speaks so the spectrum keeps tracking her
       voice; otherwise advance through the modes, each collapsing toward
       centre before the next builds out. */
    const speaking = coreEnergy > 0.12 || (activeVO && !activeVO.paused);
    let kVis = 1;
    if (b >= 1) {
      if (speaking) { modeIdx = 0; modeStart = t; }
      else {
        const cyc = t - modeStart;
        if (cyc < MODE_IN) kVis = easeOut(cyc / MODE_IN);
        else if (cyc < MODE_IN + MODE_HOLD) kVis = 1;
        else if (cyc < MODE_CYCLE) kVis = 1 - easeOut((cyc - MODE_IN - MODE_HOLD) / MODE_OUT);
        else {
          modeIdx = (modeIdx + 1) % MODES.length;
          modeStart = t; lastSwitch = t; kVis = 0;
        }
      }
    }
    const mode = MODES[modeIdx];
    // white flash at the instant of the switch
    const swFlash = Math.max(0, 1 - (t - lastSwitch) / 0.22);
    if (swFlash > 0.01) {
      ctx.fillStyle = `rgba(${HOT},${swFlash * 0.16})`;
      ctx.fillRect(0, 0, W, H);
    }

    // Centrepiece is drawn inside a collapse transform so mode changes read
    // as the assembly folding away and the next unfolding in its place.
    ctx.save();
    ctx.globalAlpha = kVis;
    const msc = 0.5 + 0.5 * kVis;
    ctx.translate(cx, cy); ctx.scale(msc, msc); ctx.translate(-cx, -cy);

    if (mode === 'iris') drawIris(cx, cy, R, kVis);
    else if (mode === 'globe') drawGlobe(cx, cy, R, kVis);
    else if (mode === 'panels') drawPanels(cx, cy, R, kVis);
    else {

    // ---- radial spectrum annulus (the signature element)
    const BARS = 132;
    const HALF = BARS / 2;
    const r0 = R * 1.28;

    /* Bar height. With audio playing this is the real FFT, mirrored across the
       vertical axis so the ring reads as symmetric. Bins are mapped with a
       curve and lifted toward the top end, because speech energy piles into
       the low bins and would otherwise leave half the ring flat. */
    const amplitude = (i, a) => {
      const organic = wave(a, t);
      if (!spec) return organic;
      const m = i < HALF ? i : BARS - i;                    // mirror index
      const bin = Math.floor(Math.pow(m / HALF, 1.6) * spec.length * 0.62);
      const raw = spec[Math.min(bin, spec.length - 1)] / 255;
      const lifted = Math.pow(raw, 0.75) * (1 + (m / HALF) * 1.6);
      return Math.min(1, lifted * 0.82 + organic * 0.18);   // never fully dead
    };

    // Bars grow out of nothing as the spectrum comes online.
    const specIn = ph(b, 0.5, 0.88);
    if (specIn > 0.001) {
      // pass 1: soft wide glow, pass 2: crisp cores
      for (const pass of [0, 1]) {
        ctx.lineWidth = pass === 0 ? Math.max(2.5, R * 0.055) : Math.max(1, R * 0.022);
        ctx.lineCap = 'butt';
        for (let i = 0; i < BARS; i++) {
          const a = (i / BARS) * Math.PI * 2 + t * 0.09;
          const amp = amplitude(i, a);
          const len = R * (0.22 + amp * 0.62 * energy) * specIn;
          const alpha = (pass === 0 ? 0.10 + amp * 0.18 : 0.30 + amp * 0.55) * specIn;
          ctx.beginPath();
          ctx.strokeStyle = `rgba(${amp > 0.72 ? HOT : BLUE},${alpha})`;
          ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
          ctx.lineTo(cx + Math.cos(a) * (r0 + len), cy + Math.sin(a) * (r0 + len));
          ctx.stroke();
        }
      }
    }

    // ---- outer segmented ring, slow clockwise (locks in first)
    const outIn = ph(b, 0.26, 0.56);
    for (let i = 0; i < 6; i++) {
      const a = t * 0.32 + i * (Math.PI * 2 / 6);
      arc(cx, cy, R * 2.42, a, a + 0.62 * outIn, Math.max(1.5, R * 0.05), 0.30 * outIn);
    }
    // block markers riding the outer ring
    for (let i = 0; i < 12; i++) {
      if (i / 12 > outIn) continue;
      const a = -t * 0.18 + i * (Math.PI * 2 / 12);
      const rr = R * 2.42;
      const s = Math.max(2, R * 0.055);
      ctx.fillStyle = `rgba(${BLUE},${(i % 3 === 0 ? 0.65 : 0.28) * outIn})`;
      ctx.fillRect(cx + Math.cos(a) * rr - s / 2, cy + Math.sin(a) * rr - s / 2, s, s);
    }

    // ---- lime accent ring with tick segments, riding above the spectrum
    const limeIn = ph(b, 0.44, 0.78);
    if (limeIn > 0.001) {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(t * 0.42);
      const lr = R * 2.08;
      for (let i = 0; i < 40; i++) {
        ctx.rotate(Math.PI * 2 / 40);
        if (i / 40 > limeIn) continue;
        const long = i % 5 === 0;
        ctx.beginPath();
        ctx.moveTo(lr, 0);
        ctx.lineTo(lr + R * (long ? 0.16 : 0.09), 0);
        ctx.strokeStyle = `rgba(${GOLD},${(long ? 0.75 : 0.34) * limeIn})`;
        ctx.lineWidth = Math.max(1, R * 0.028);
        ctx.stroke();
      }
      ctx.restore();
      // a bright partial arc sweeping around that ring
      arc(cx, cy, lr, t * 0.42, t * 0.42 + 1.05 * limeIn,
          Math.max(1.5, R * 0.035), 0.65 * limeIn, GOLD);
    }

    // ---- magenta marker arcs, counter-rotating
    const magIn = ph(b, 0.55, 0.85);
    if (magIn > 0.001) {
      for (let i = 0; i < 2; i++) {
        const a = -t * 0.62 + i * Math.PI;
        arc(cx, cy, R * 1.72, a, a + 0.42 * magIn,
            Math.max(1.5, R * 0.045), 0.62 * magIn, MAGENTA);
      }
    }

    // ---- fine tick ring, counter-rotating (sweeps into existence)
    const tickIn = ph(b, 0.38, 0.7);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-t * 0.14);
    for (let i = 0; i < 72; i++) {
      ctx.rotate(Math.PI * 2 / 72);
      if (i / 72 > tickIn) continue;
      const long = i % 6 === 0;
      ctx.beginPath();
      ctx.moveTo(R * 2.62, 0);
      ctx.lineTo(R * (long ? 2.88 : 2.75), 0);
      ctx.strokeStyle = `rgba(${BLUE},${long ? 0.42 : 0.17})`;
      ctx.lineWidth = Math.max(1, R * 0.012);
      ctx.stroke();
    }
    ctx.restore();

    // ---- dashed inner ring, clockwise
    const dashIn = ph(b, 0.46, 0.74);
    if (dashIn > 0.001) {
      ctx.save();
      ctx.setLineDash([R * 0.14, R * 0.10]);
      arc(cx, cy, R * 1.16, t * 0.9, t * 0.9 + Math.PI * 2 * dashIn,
          Math.max(1, R * 0.02), 0.42 * dashIn);
      ctx.restore();
    }

    // ---- mechanical iris: overlapping blade arcs, slow rotation
    const irisIn = ph(b, 0.34, 0.66);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(t * 0.2 + (1 - irisIn) * 1.2); // unwinds into place
    for (let i = 0; i < 5; i++) {
      const a = i * (Math.PI * 2 / 5);
      ctx.beginPath();
      ctx.arc(Math.cos(a) * R * 0.34, Math.sin(a) * R * 0.34, R * 0.55, a + 0.7, a + 2.5);
      ctx.strokeStyle = `rgba(215,235,245,${(0.20 + coreEnergy * 0.15) * irisIn})`;
      ctx.lineWidth = Math.max(1, R * 0.018);
      ctx.stroke();
    }
    ctx.restore();

    // ---- core disc
    // Blown-out white-hot centre, as in the reference — the pupil clips to
    // white and only falls to cyan well outside it.
    const coreIn = ph(b, 0.10, 0.45);
    const heat = 0.72 + coreEnergy * 0.28;
    const iris = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.02);
    iris.addColorStop(0, `rgba(255,255,255,${0.95 * heat * coreIn})`);
    iris.addColorStop(0.18, `rgba(255,255,255,${0.72 * heat * coreIn})`);
    iris.addColorStop(0.38, `rgba(${HOT},${0.55 * heat * coreIn})`);
    iris.addColorStop(0.68, `rgba(${BLUE},${0.22 * coreIn})`);
    iris.addColorStop(1, `rgba(${BLUE},0.02)`);
    ctx.fillStyle = iris;
    ctx.beginPath(); ctx.arc(cx, cy, R * 1.02, 0, Math.PI * 2); ctx.fill();
    arc(cx, cy, R * 1.02, 0, Math.PI * 2, Math.max(1, R * 0.022), 0.55 * coreIn);

    // dense short spokes radiating out of the hot centre
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-t * 0.35);
    for (let i = 0; i < 90; i++) {
      ctx.rotate(Math.PI * 2 / 90);
      const len = R * (0.16 + 0.10 * Math.abs(Math.sin(i * 1.3 + t * 2)));
      ctx.beginPath();
      ctx.moveTo(R * 0.52, 0); ctx.lineTo(R * 0.52 + len, 0);
      ctx.strokeStyle = `rgba(${HOT},${(0.16 + 0.24 * heat) * coreIn})`;
      ctx.lineWidth = Math.max(1, R * 0.012);
      ctx.stroke();
    }
    ctx.restore();

    // ---- centre ring (no numeric readout — it read as a countdown)
    const readIn = ph(b, 0.55, 0.8);
    if (readIn > 0.001) {
      arc(cx, cy, R * 0.44, t * 0.7, t * 0.7 + 4.6, Math.max(1, R * 0.02), 0.6 * readIn);
      arc(cx, cy, R * 0.26, -t * 1.1, -t * 1.1 + 3.1, Math.max(1, R * 0.018), 0.45 * readIn, GOLD);
    }

    } // end ring mode
    ctx.restore();

    // ---- corner telemetry, revealed row by row
    // The stage is full-viewport, so the top inset clears the page header.
    const pad = Math.max(8, W * 0.022);
    const padTop = pad + H * 0.062;
    const fs = Math.max(7, W * 0.0165);
    ctx.font = `${fs}px "Consolas", monospace`;
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    const telIn = ph(b, 0.62, 1.0);
    telemetry.forEach((row, i) => {
      if (i / telemetry.length > telIn) return;
      ctx.fillStyle = `rgba(${BLUE},0.42)`;
      ctx.fillText(row.k, pad, padTop + i * fs * 1.75);
      ctx.fillStyle = `rgba(${HOT},0.55)`;
      ctx.fillText(String(row.v), pad + fs * 5.6, padTop + i * fs * 1.75);
    });
    ctx.textAlign = 'right';
    ['ALIGNMENT', 'FRONTAL', 'SAGITTAL', 'CONFIDENCE'].forEach((k, i) => {
      if (i / 4 > telIn) return;
      ctx.fillStyle = `rgba(${BLUE},0.34)`;
      ctx.fillText(k, W - pad, padTop + i * fs * 1.75);
    });
    // current mode name, so the cycling reads as deliberate
    if (b >= 1) {
      ctx.textAlign = 'center';
      ctx.fillStyle = `rgba(${GOLD},${0.5 * kVis})`;
      ctx.fillText('［ ' + MODE_LABEL[mode] + ' ］', cx, padTop);
    }

    // ---- scan modules coming online, bottom-left, one after another.
    //      They retire a couple of seconds after boot so they don't sit
    //      behind the control at the bottom of the screen.
    const sinceBoot = bootStart ? (performance.now() - bootStart) / 1000 - BOOT_MS / 1000 : 99;
    const modFade = 1 - clamp((sinceBoot - 2) / 1.8, 0, 1);
    ctx.textAlign = 'left';
    if (modFade > 0.01) SCAN_MODULES.forEach((m, i) => {
      const on = clamp((b - m.at) / 0.1, 0, 1) * modFade;
      if (on <= 0) return;
      // offset clears the status line pinned to the bottom of the stage
      const y = H - pad - fs * 1.75 * (SCAN_MODULES.length - i) - fs * 3.4;
      // brief flash as each module reports in
      const flash = 1 - clamp((b - m.at) / 0.16, 0, 1);
      ctx.fillStyle = `rgba(${flash > 0.05 ? HOT : BLUE},${(0.30 + flash * 0.5) * on})`;
      ctx.fillText('▸ ' + m.label + '  ONLINE', pad, y);
    });

    // ---- ambient detail: scattered crosshair marks (reference has these
    //      sprinkled across the panels as texture)
    const ambIn = ph(b, 0.75, 1.0);
    if (ambIn > 0.001) {
      ctx.strokeStyle = `rgba(${BLUE},${0.22 * ambIn})`;
      ctx.lineWidth = Math.max(1, W * 0.0018);
      const marks = [[0.14, 0.34], [0.11, 0.68], [0.86, 0.4], [0.9, 0.72], [0.3, 0.14], [0.72, 0.87]];
      const ms = Math.max(3, W * 0.009);
      marks.forEach(([mx, my], i) => {
        const x = mx * W, y = my * H;
        const tw = ms * (0.7 + 0.3 * Math.sin(t * 2 + i));
        ctx.beginPath();
        ctx.moveTo(x - tw, y); ctx.lineTo(x + tw, y);
        ctx.moveTo(x, y - tw); ctx.lineTo(x, y + tw);
        ctx.stroke();
      });
    }

    // ---- model download meter, mirroring the reference's "loading… 100%"
    if (modelLoading && modelProgress < 1) {
      const pct = Math.round(modelProgress * 100);
      const barW = W * 0.34, barH = Math.max(3, H * 0.012);
      const bx = cx - barW / 2, by = H * 0.845;
      // dark plate so the readout stays legible over the spectrum ring
      const plW = barW * 1.35, plH = fs * 4.2;
      ctx.fillStyle = 'rgba(3,7,13,0.82)';
      ctx.fillRect(cx - plW / 2, by - plH + barH * 2, plW, plH);
      ctx.strokeStyle = `rgba(${BLUE},0.25)`;
      ctx.lineWidth = Math.max(1, W * 0.0015);
      ctx.strokeRect(cx - plW / 2, by - plH + barH * 2, plW, plH);
      ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      ctx.font = `600 ${fs * 1.6}px "Consolas", monospace`;
      ctx.fillStyle = `rgba(${HOT},0.95)`;
      ctx.fillText(pct + '%', cx, by - fs * 1.5);
      ctx.font = `${fs}px "Consolas", monospace`;
      ctx.fillStyle = `rgba(${BLUE},0.6)`;
      ctx.fillText('loading ai core…', cx, by - fs * 0.3);
      ctx.strokeStyle = `rgba(${BLUE},0.45)`;
      ctx.lineWidth = Math.max(1, W * 0.002);
      ctx.strokeRect(bx, by, barW, barH);
      ctx.fillStyle = `rgba(${GOLD},0.75)`;
      ctx.fillRect(bx + 1, by + 1, Math.max(0, (barW - 2) * modelProgress), barH - 2);
      ctx.textBaseline = 'top';
    }

    // ---- running status callout, mid-right, cycling with a flash on change
    if (b > 0.8) {
      if (t - calloutAt > 2.2) { calloutAt = t; calloutIdx = (calloutIdx + 1) % CALLOUTS.length; }
      const age = t - calloutAt;
      const flash = 1 - clamp(age / 0.28, 0, 1);
      ctx.textAlign = 'right';
      ctx.font = `${fs * 1.05}px "Consolas", monospace`;
      ctx.fillStyle = flash > 0.05 ? `rgba(${HOT},0.95)` : `rgba(${GOLD},0.62)`;
      ctx.fillText('◂ ' + CALLOUTS[calloutIdx], W - pad, cy - fs * 0.6);
      // underline that wipes in with each new line
      const uw = fs * 7 * clamp(age / 0.5, 0, 1);
      ctx.strokeStyle = `rgba(${GOLD},0.4)`;
      ctx.lineWidth = Math.max(1, W * 0.002);
      ctx.beginPath();
      ctx.moveTo(W - pad, cy + fs * 0.9);
      ctx.lineTo(W - pad - uw, cy + fs * 0.9);
      ctx.stroke();
    }

    // ---- frame brackets
    const bl = Math.max(10, W * 0.05) * ph(b, 0.7, 1.0);
    ctx.strokeStyle = `rgba(${BLUE},${0.5 * ph(b, 0.7, 1.0)})`;
    ctx.lineWidth = Math.max(1, W * 0.004);
    [[pad, pad, 1, 1], [W - pad, pad, -1, 1], [pad, H - pad, 1, -1], [W - pad, H - pad, -1, -1]]
      .forEach(([x, y, sx, sy]) => {
        ctx.beginPath();
        ctx.moveTo(x + sx * bl, y); ctx.lineTo(x, y); ctx.lineTo(x, y + sy * bl);
        ctx.stroke();
      });

    applyBloom(0.72);   // lighter than the montage; the steady HUD stays legible
    coreEnergy *= 0.94; // decays unless refreshed by speech
    requestAnimationFrame(frame);
  })();
})();

// Fade the custom loop in over the procedural core once it decodes.
// Nothing is requested unless ASSETS.kinaLoop is configured.
(function kinaLoop() {
  const v = $('kina-loop');
  if (!v || !ASSETS.kinaLoop) return;
  if (ASSETS.kinaPoster) v.poster = ASSETS.kinaPoster;
  v.addEventListener('loadeddata', () => {
    v.classList.add('ready');
    v.play().catch(() => {});
  });
  v.addEventListener('error', () => { /* keep the procedural core */ });
  v.src = ASSETS.kinaLoop;
})();

/* ---------- Jarvis voice ---------- */
let voiceOn = true;
let jarvisVoice = null;

function pickVoice() {
  const vs = speechSynthesis.getVoices();
  if (!vs.length) return;
  jarvisVoice =
    vs.find(v => /en-GB/i.test(v.lang) && /male|daniel|arthur|george/i.test(v.name)) ||
    vs.find(v => /en-GB/i.test(v.lang)) ||
    vs.find(v => /^en/i.test(v.lang)) || vs[0];
}
if ('speechSynthesis' in window) { pickVoice(); speechSynthesis.onvoiceschanged = pickVoice; }

function say(text) {
  if (!voiceOn || !('speechSynthesis' in window)) return;
  try {
    const u = new SpeechSynthesisUtterance(text);
    if (jarvisVoice) u.voice = jarvisVoice;
    u.rate = 1.02; u.pitch = 0.82; u.volume = 0.95;
    if (speechSynthesis.speaking || speechSynthesis.pending) {
      // Android Chrome goes silent if speak() immediately follows cancel()
      speechSynthesis.cancel();
      setTimeout(() => { try { speechSynthesis.speak(u); } catch (e) {} }, 80);
    } else {
      speechSynthesis.speak(u);
    }
  } catch (e) { /* voice is decorative — never block the flow */ }
}

$('mute-btn').addEventListener('click', () => {
  voiceOn = !voiceOn;
  $('mute-btn').textContent = voiceOn ? 'SOUND: ON' : 'SOUND: OFF';
  if (voiceOn) { unlockAudio(); sfx('tap'); }
  else {
    stopBed();
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    if (activeVO) { try { activeVO.pause(); } catch (e) {} }
  }
});

// Every button gets a tap cue, without wiring each one individually.
document.addEventListener('click', (e) => {
  const btn = e.target.closest && e.target.closest('.btn');
  if (btn && btn.id !== 'btn-begin') sfx('tap', 0.25);
}, true);

/* ============================================================
   VOICEOVER — pre-recorded ElevenLabs audio when the file exists,
   otherwise the browser's speech synthesis saying the same words.
   Captions are distributed across the real audio duration weighted
   by line length, so no manual timing table has to be maintained.
   ============================================================ */

/* `pauses[i]` is the silence in seconds that follows line i in the recording
   (the <break> tags in docs/voiceover-scripts.md). Captions are timed on
   speech + trailing silence, so the text holds through the pause instead of
   running ahead of the voice. Keep these in step with the script. */
const VO_LINES = {
  intro: {
    file: ASSETS.voIntro,
    lines: [
      'Good day. I am KINA — posture intelligence, online.',
      "I'll scan you from two angles and grade your alignment out of one hundred.",
      'Stand two or three steps back. I need to see you from head to feet.',
      'Bare feet. Fitted clothing — loose fabric hides your spine from me.',
      'First photo, face me. Second, turn ninety degrees.',
      "And stand how you normally stand. I'll know if you're cheating."
    ],
    pauses: [0.8, 0.7, 0.5, 0.5, 0.7, 0]
  },
  front: {
    file: ASSETS.voFront,
    lines: [
      'Frontal view. Stand square to me, arms relaxed at your sides.',
      'Look straight ahead, and hold still.'
    ],
    pauses: [0.5, 0]
  },
  side: {
    file: ASSETS.voSide,
    lines: [
      'Good. Now turn ninety degrees, so one shoulder faces me.',
      'Arms hanging naturally. Do not correct your posture — I will know.'
    ],
    pauses: [0.5, 0]
  }
};


function stopVO() {
  if (activeVO) {
    try { activeVO.pause(); activeVO.currentTime = 0; } catch (e) {}
    activeVO = null;
  }
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  detachAnalyser();
  if (voWalkId) { cancelAnimationFrame(voWalkId); voWalkId = null; }
  if (voCancel) { const c = voCancel; voCancel = null; c(); }
  const cap = $('vo-caption');
  if (cap) cap.classList.remove('on');
  const stage = $('kina-stage');
  if (stage) stage.classList.remove('speaking');
}

/** Reading pace used when there's no audio to sync against. */
const VO_MS_PER_CHAR = 55;

/**
 * Plays a scripted line and walks its captions.
 *
 * The caption timeline is authoritative and always runs, so the briefing
 * still reads as a briefing when the recording is missing AND the device
 * has no speech voices (headless browsers, some Android builds). Real audio,
 * when present, just retimes the same walk to its true duration.
 */
function playVO(key) {
  const spec = VO_LINES[key];
  if (!spec) return Promise.resolve();

  const cap = $('vo-caption');
  const stage = $('kina-stage');
  const lines = spec.lines;
  // Weight each caption by its spoken length plus the silence that follows it,
  // both expressed in "character equivalents" at the reading pace below.
  const pauses = spec.pauses || [];
  const weights = lines.map((l, i) =>
    Math.max(24, l.length) + ((pauses[i] || 0) * 1000) / VO_MS_PER_CHAR);
  const total = weights.reduce((a, b) => a + b, 0);

  let durationMs = Math.max(2400, total * VO_MS_PER_CHAR);
  let shown = -1;
  const showChunk = (i) => {
    if (i === shown || !cap) return;
    shown = i;
    cap.classList.add('on');
    cap.textContent = lines[i];
  };
  const finish = () => {
    if (cap) cap.classList.remove('on');
    if (stage) stage.classList.remove('speaking');
    activeVO = null;
    coreAnalyser = null;
  };

  if (stage) stage.classList.add('speaking');

  return new Promise((resolve) => {
    let cancelled = false;
    let audio = null;
    let audioLive = false;

    const node = voiceOn ? getVONode(key, spec.file) : null;

    if (node) {
      audio = node.el;
      wireAnalyser(node); // no-op unless the context is already running
      coreAnalyser = node.analyser ? { an: node.analyser, data: node.data } : null;

      const onPlaying = () => {
        audioLive = true;
        activeVO = audio;
        if (isFinite(audio.duration) && audio.duration > 0) durationMs = audio.duration * 1000;
      };
      const useSpeech = () => {
        if (audioLive || cancelled) return;
        try { audio.pause(); } catch (e) {}
        audio = null;
        coreAnalyser = null;
        speakLines(lines); // fire and forget — the caption walk owns the timing
      };
      // Surface a silent failure instead of leaving the user staring at a
      // muted screen wondering whether they broke it.
      audio.addEventListener('playing', () => {
        $('kina-status').textContent = '◈ KINA SPEAKING';
      }, { once: true });
      setTimeout(() => {
        if (!audioLive && !cancelled) {
          $('kina-status').textContent = '◈ NO AUDIO — CHECK RINGER / TAP SKIP';
        }
      }, 2500);
      audio.addEventListener('playing', onPlaying, { once: true });
      audio.addEventListener('error', useSpeech, { once: true });
      try { audio.currentTime = 0; } catch (e) {}
      audio.play().catch(useSpeech);
      // Generous last resort: the file is preloaded, so this should never fire
      // on a working connection. Short timers here used to kill a slow-loading
      // recording mid-buffer.
      setTimeout(useSpeech, 6000);
    } else if (voiceOn) {
      speakLines(lines); // no recording configured — built-in voice
    }

    const t0 = performance.now();
    (function walk() {
      if (cancelled) return;
      const elapsed = audioLive && audio ? audio.currentTime * 1000 : performance.now() - t0;
      const pos = Math.min(1, elapsed / durationMs);
      coreEnergy = Math.max(coreEnergy, 0.85);
      let acc = 0;
      for (let i = 0; i < weights.length; i++) {
        acc += weights[i] / total;
        if (pos <= acc || i === weights.length - 1) { showChunk(i); break; }
      }
      if (pos < 1) { voWalkId = requestAnimationFrame(walk); }
      else { finish(); resolve(); }
    })();

    voCancel = () => { cancelled = true; finish(); resolve(); };
  });
}

let voWalkId = null;
let voCancel = null;

/** Speech-synthesis reading of the same script, chunk by chunk. */
function speakLines(lines, onChunk) {
  if (!voiceOn || !('speechSynthesis' in window)) return Promise.resolve();
  return new Promise((resolve) => {
    let i = 0;
    // keep the core pulsing while synthesis runs — it emits no timeupdate
    const tick = setInterval(() => { coreEnergy = Math.max(coreEnergy, 0.8); }, 120);
    const done = () => { clearInterval(tick); resolve(); };
    const next = () => {
      if (i >= lines.length) return done();
      if (onChunk) onChunk(i);
      try {
        const u = new SpeechSynthesisUtterance(lines[i]);
        if (jarvisVoice) u.voice = jarvisVoice;
        u.rate = 1.02; u.pitch = 0.82; u.volume = 0.95;
        u.onend = () => { i++; next(); };
        u.onerror = () => { i++; next(); };
        speechSynthesis.speak(u);
      } catch (e) { done(); }
    };
    next();
  });
}

/* ---------- typewriter console ---------- */
async function typeLines(el, lines, speed = 14) {
  el.innerHTML = '';
  const cur = document.createElement('span');
  cur.className = 'cursor';
  for (const line of lines) {
    const span = document.createElement('span');
    el.appendChild(span); el.appendChild(cur);
    for (let i = 0; i <= line.length; i++) {
      span.textContent = line.slice(0, i);
      await sleep(speed);
    }
    el.insertBefore(document.createTextNode('\n'), cur);
  }
}

/* ---------- MediaPipe lazy loader ----------
   The model is fetched by hand rather than by URL so its download can be
   metered: it's ~9 MB, the slowest step on a phone, and a real percentage
   is the difference between "loading" and "stuck". */

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('model fetch failed: ' + res.status);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received / total);
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const chunk of chunks) { buf.set(chunk, off); off += chunk.length; }
  return buf;
}

let landmarkerPromise = null;
function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      modelLoading = true;
      const vision = await import('./vendor/mediapipe/vision_bundle.mjs');
      const files = await vision.FilesetResolver.forVisionTasks('./vendor/mediapipe/wasm');
      const modelAssetBuffer = await fetchWithProgress(
        './vendor/models/pose_landmarker_full.task', (p) => { modelProgress = p; });
      modelProgress = 1;
      const lm = await vision.PoseLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetBuffer, delegate: 'GPU' },
        runningMode: 'IMAGE',
        numPoses: 1,
        outputSegmentationMasks: true,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5
      });
      modelLoading = false;
      return lm;
    })().catch(err => { landmarkerPromise = null; modelLoading = false; throw err; });
  }
  return landmarkerPromise;
}

/* ============================================================
   RENDERING — skeleton overlay + scan sweep
   ============================================================ */
const BONES = [
  [7, 8], [11, 12], [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32],
  [11, 13], [12, 14]
];
const KEY_PTS = [0, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

function drawScene(ctx, img, W, H, lms, progress) {
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(img, 0, 0, W, H);
  ctx.fillStyle = 'rgba(4,8,15,0.35)';
  ctx.fillRect(0, 0, W, H);

  if (lms) {
    const P = (i) => ({ x: lms[i].x * W, y: lms[i].y * H });
    ctx.lineWidth = Math.max(2, W / 300);
    for (const [a, b] of BONES) {
      if ((lms[a].visibility ?? 1) < 0.4 || (lms[b].visibility ?? 1) < 0.4) continue;
      const pa = P(a), pb = P(b);
      ctx.strokeStyle = 'rgba(36,221,221,0.88)';
      ctx.shadowColor = '#24dddd'; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
    }
    ctx.shadowBlur = 0;
    for (const i of KEY_PTS) {
      if ((lms[i].visibility ?? 1) < 0.4) continue;
      const pt = P(i);
      ctx.fillStyle = '#24dddd';
      ctx.beginPath(); ctx.arc(pt.x, pt.y, Math.max(3, W / 220), 0, 7); ctx.fill();
      ctx.strokeStyle = 'rgba(36,221,221,0.52)';
      ctx.beginPath(); ctx.arc(pt.x, pt.y, Math.max(7, W / 90), 0, 7); ctx.stroke();
    }
  }

  if (progress < 1) {
    const y = H * progress;
    const grad = ctx.createLinearGradient(0, y - 40, 0, y + 40);
    grad.addColorStop(0, 'rgba(36,221,221,0)');
    grad.addColorStop(0.5, 'rgba(36,221,221,0.38)');
    grad.addColorStop(1, 'rgba(36,221,221,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, y - 40, W, 80);
    ctx.fillStyle = 'rgba(36,221,221,0.92)';
    ctx.fillRect(0, y, W, 2);
  }
}

/* ---------- mesh warp (canvas side of the simulation) ---------- */
function expandFrom(p, cx, cy, px) {
  const dx = p.x - cx, dy = p.y - cy;
  const len = Math.hypot(dx, dy) || 1e-6;
  return { x: p.x + (dx / len) * px, y: p.y + (dy / len) * px };
}

function warpImage(srcCanvas, controls, W, H) {
  const out = document.createElement('canvas');
  out.width = W; out.height = H;
  const ctx = out.getContext('2d');
  ctx.drawImage(srcCanvas, 0, 0); // base layer so edges never show gaps
  if (!controls.length) return out;

  const COLS = 20, ROWS = 26;
  const displace = makeDisplacer(controls, W, H, COLS, ROWS);

  const src = [], dst = [];
  for (let j = 0; j <= ROWS; j++) {
    for (let i = 0; i <= COLS; i++) {
      const x = (i / COLS) * W, y = (j / ROWS) * H;
      src.push({ x, y });
      dst.push(displace(x, y));
    }
  }
  const at = (i, j) => j * (COLS + 1) + i;

  for (let j = 0; j < ROWS; j++) {
    for (let i = 0; i < COLS; i++) {
      const q = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
      for (const [m, n, o] of [[0, 1, 2], [0, 2, 3]]) {
        const s = [src[q[m]], src[q[n]], src[q[o]]];
        const d = [dst[q[m]], dst[q[n]], dst[q[o]]];
        const mat = solveAffine(s[0], s[1], s[2], d[0], d[1], d[2]);
        if (!mat) continue;
        const cx = (d[0].x + d[1].x + d[2].x) / 3, cy = (d[0].y + d[1].y + d[2].y) / 3;
        const de = d.map(pt => expandFrom(pt, cx, cy, 0.7)); // hides seams
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(de[0].x, de[0].y); ctx.lineTo(de[1].x, de[1].y); ctx.lineTo(de[2].x, de[2].y);
        ctx.closePath(); ctx.clip();
        ctx.setTransform(mat[0], mat[1], mat[2], mat[3], mat[4], mat[5]);
        ctx.drawImage(srcCanvas, 0, 0);
        ctx.restore();
      }
    }
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return out;
}

/* ============================================================
   CAPTURE FLOW
   ============================================================ */

const state = { front: null, side: null, want: 'front' };

const VIEW_COPY = {
  front: {
    label: '▸ VIEW 01 · FRONTAL PLANE',
    title: 'Face me, <span class="accent">feet hip-width</span>',
    hint: 'Stand square to the camera. Arms relaxed at your sides, barefoot, looking straight ahead.',
    voice: 'First, the frontal view. Stand square to me, arms relaxed, and look straight ahead.'
  },
  side: {
    label: '▸ VIEW 02 · SAGITTAL PLANE',
    title: 'Now turn <span class="accent">90 degrees</span>',
    hint: 'Turn a full quarter-turn so one shoulder faces the camera. Arms hanging naturally. Don\'t correct your posture.',
    voice: 'Excellent. Now turn ninety degrees so one shoulder faces me. Do not correct your posture — I will know.'
  }
};

const GUIDE_FRONT = `<svg viewBox="0 0 100 220" fill="none" stroke="#24dddd" stroke-width="1.4" stroke-dasharray="4 4">
  <circle cx="50" cy="20" r="12"/><line x1="50" y1="32" x2="50" y2="120"/>
  <line x1="28" y1="52" x2="72" y2="52"/><line x1="28" y1="52" x2="24" y2="105"/>
  <line x1="72" y1="52" x2="76" y2="105"/><line x1="34" y1="118" x2="66" y2="118"/>
  <line x1="38" y1="118" x2="36" y2="205"/><line x1="62" y1="118" x2="64" y2="205"/>
  <line x1="50" y1="0" x2="50" y2="220" stroke-width="0.7" opacity="0.5"/></svg>`;
const GUIDE_SIDE = `<svg viewBox="0 0 100 220" fill="none" stroke="#24dddd" stroke-width="1.4" stroke-dasharray="4 4">
  <circle cx="54" cy="20" r="12"/><line x1="50" y1="32" x2="50" y2="120"/>
  <line x1="50" y1="55" x2="52" y2="105"/><line x1="50" y1="118" x2="50" y2="205"/>
  <line x1="50" y1="0" x2="50" y2="220" stroke-width="0.7" opacity="0.5"/></svg>`;

function renderCaptureUI() {
  const v = state.want;
  const c = VIEW_COPY[v];
  $('capture-label').textContent = c.label;
  $('capture-title').innerHTML = c.title;
  $('capture-hint').textContent = c.hint;
  $('pose-guide').innerHTML = v === 'front' ? GUIDE_FRONT : GUIDE_SIDE;
  $('dot-front').className = 'step-dot ' + (state.front ? 'done' : v === 'front' ? 'current' : '');
  $('dot-side').className = 'step-dot ' + (state.side ? 'done' : v === 'side' ? 'current' : '');
  $('dot-result').className = 'step-dot' + (state.front && state.side ? ' current' : '');
}

function setThumb(view, dataUrl) {
  const el = $('thumb-' + view);
  el.className = 'thumb';
  el.innerHTML = '';
  const im = new Image();
  im.src = dataUrl; im.alt = '';
  const tag = document.createElement('div');
  tag.className = 'tag';
  tag.textContent = view.toUpperCase() + ' ✓';
  el.appendChild(im); el.appendChild(tag);
}

function resetThumb(view) {
  const el = $('thumb-' + view);
  el.className = 'thumb empty';
  el.textContent = view.toUpperCase() + ' — PENDING';
}

const showError = (msg) => { const b = $('err-box'); b.textContent = '⚠ ' + msg; b.style.display = 'block'; };
const hideError = () => { $('err-box').style.display = 'none'; };
const showWarn = (msg) => { const b = $('warn-box'); b.textContent = 'ⓘ ' + msg; b.style.display = 'block'; };
const hideWarn = () => { $('warn-box').style.display = 'none'; };

const BOOT_SEQ_MS = 15000; // must match BOOT_MS inside the core renderer

/* ---------- intro briefing ----------
   Audio needs a user gesture on mobile, so the briefing starts on tap.
   The muted loop autoplays before that; only the voiceover waits. */
let briefingRan = false;

/* Scored 15s boot, then the briefing. Timings mirror drawBoot(). */
const bootTimers = [];
function scheduleBootAudio() {
  const at = (s, fn) => bootTimers.push(setTimeout(fn, s * 1000));
  startAction();                                   // 0:00 music in
  // telemetry chatter and whooshes through the frantic phase
  for (let s = 0.3; s < 9.6; s += 0.34) at(s + Math.random() * 0.12, () => sfx('telemetry', 0.3));
  [0.6, 2.2, 3.4, 4.6, 5.0, 7.0, 8.4].forEach(s => at(s, () => sfx('swoosh', 0.5)));
  at(2.2, () => sfx('powerUp', 0.9));              // portal shatter
  at(10.0, () => { stopAction(0.05); coreEnergy = 0; });  // 0:10 hard cut to silence
  at(10.9, () => sfx('target', 0.4));              // lone targeting beep
  at(12.0, () => sfx('staticZap', 1));             // static crackle
  at(12.6, () => {                                 // "Welcome."
    if (!voiceOn) return;
    startBed();
    speakLines(['Welcome.']);
  });
}
function clearBootAudio() {
  bootTimers.splice(0).forEach(clearTimeout);
  stopAction(0.15);
}

async function runBriefing() {
  if (briefingRan) return;
  briefingRan = true;
  unlockAudio(); // must happen inside the tap, before anything async

  $('btn-begin').classList.add('hidden');
  $('btn-skip-vo').classList.remove('hidden');
  $('kina-status').textContent = '';

  bootStart = performance.now();
  bootRunning = true;
  scheduleBootAudio();
  getLandmarker().catch(() => {});  // model downloads behind the sequence

  await sleep(BOOT_SEQ_MS + 700);   // let "Welcome." land before KINA speaks
  if (!briefingRan) return;         // skipped out mid-sequence
  $('kina-status').textContent = '◈ BRIEFING IN PROGRESS';
  await playVO('intro');
  endBriefing();
}

function endBriefing() {
  clearBootAudio();
  bootRunning = false; bootDone = true;
  stopVO();
  $('btn-skip-vo').classList.add('hidden');
  $('btn-begin').classList.add('hidden');
  $('btn-start').classList.remove('hidden');
  $('kina-status').textContent = '◈ READY WHEN YOU ARE';
}

$('btn-begin').addEventListener('click', runBriefing);
$('btn-skip-vo').addEventListener('click', endBriefing);

$('btn-start').addEventListener('click', () => {
  unlockAudio(); // covers users who skip the briefing entirely
  stopVO();
  sfx('whoosh', 0.6);
  getLandmarker().catch(() => {});
  state.want = 'front';
  renderCaptureUI();
  showPanel('panel-capture');
  playVO('front');
});

/* ---------- upload ---------- */
$('btn-upload').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  const img = new Image();
  img.onload = () => { URL.revokeObjectURL(url); ingest(img); };
  img.onerror = () => { URL.revokeObjectURL(url); showError('Could not read that image. Try a different photo (JPG/PNG).'); };
  img.src = url;
  e.target.value = '';
});

/* ---------- live camera ---------- */
let stream = null;
function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $('video-wrap').classList.add('hidden');
  $('btn-snap').classList.add('hidden');
  $('btn-snap').disabled = false;
  $('btn-cancel-cam').classList.add('hidden');
  $('capture-choices').classList.remove('hidden');
}
$('btn-cancel-cam').addEventListener('click', stopCamera);

$('btn-camera').addEventListener('click', async () => {
  hideError();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('Live camera not available in this browser — use Upload Photo instead.');
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
    });
    $('cam-video').srcObject = stream;
    $('video-wrap').classList.remove('hidden');
    $('btn-snap').classList.remove('hidden');
    $('btn-cancel-cam').classList.remove('hidden');
    $('capture-choices').classList.add('hidden');
    say('Camera online. Line yourself up with the guide, then press capture.');
  } catch (err) {
    showError('Camera access denied. You can still use Upload Photo.');
  }
});

$('btn-snap').addEventListener('click', async () => {
  const video = $('cam-video');
  if (!video.videoWidth || $('btn-snap').disabled) return;
  $('btn-snap').disabled = true;
  $('btn-cancel-cam').classList.add('hidden');
  const cd = $('countdown');
  cd.style.display = 'flex';
  for (const n of ['3', '2', '1']) { cd.textContent = n; say(n); await sleep(900); }
  cd.style.display = 'none';
  const c = document.createElement('canvas');
  c.width = video.videoWidth; c.height = video.videoHeight;
  const cx = c.getContext('2d');
  // Mirror to match the mirrored preview (WYSIWYG). This swaps MediaPipe's
  // anatomical left/right labels — fine while all metrics are symmetric.
  cx.translate(c.width, 0); cx.scale(-1, 1);
  cx.drawImage(video, 0, 0);
  stopCamera();
  const img = new Image();
  img.onload = () => ingest(img);
  img.src = c.toDataURL('image/jpeg', 0.92);
});

/* ---------- per-photo detection, gating and storage ---------- */
async function ingest(img) {
  hideError(); hideWarn();
  showPanel('panel-analyze');
  $('metrics').innerHTML = '';
  const view = state.want;
  $('scan-status').textContent = 'ACQUIRING…';
  sfx('scan', 0.6);
  say(view === 'front' ? 'Frontal image acquired. Analysing.' : 'Sagittal image acquired. Analysing.');

  const canvas = $('scan-canvas');
  const scale = Math.min(1, 1000 / img.width);
  const W = canvas.width = Math.round(img.width * scale);
  const H = canvas.height = Math.round(img.height * scale);
  const ctx = canvas.getContext('2d');
  drawScene(ctx, img, W, H, null, 0);

  let det = null, err = null, settled = false;
  getLandmarker()
    .then(lm => { det = lm.detect(img); })
    .catch(e => { err = e; })
    .finally(() => { settled = true; });

  const t0 = performance.now();
  const SWEEP_MS = 2000, TIMEOUT_MS = 30000;
  const statuses = ['CALIBRATING SENSORS…', 'MAPPING SKELETAL NODES…', 'TRACING BODY OUTLINE…', 'MEASURING JOINT VECTORS…'];
  await new Promise(res => {
    (function frame() {
      const el = performance.now() - t0;
      const pr = Math.min(1, el / SWEEP_MS);
      const lms = det && det.landmarks && det.landmarks[0];
      if (pr < 1) {
        $('scan-status').textContent = statuses[Math.min(statuses.length - 1, Math.floor(pr * statuses.length))];
        drawScene(ctx, img, W, H, lms || null, pr);
        requestAnimationFrame(frame);
      } else if (!settled && el < TIMEOUT_MS) {
        $('scan-status').textContent = 'LOADING AI CORE… FIRST SCAN TAKES A MOMENT';
        drawScene(ctx, img, W, H, lms || null, (el / SWEEP_MS) % 1);
        requestAnimationFrame(frame);
      } else res();
    })();
  });

  const reject = (msg) => {
    sfx('reject', 0.4);
    showPanel('panel-capture'); renderCaptureUI(); showError(msg);
    say('Scan rejected. ' + msg);
  };

  if (!settled) return reject('The AI engine is taking too long to load. Check your connection and retry.');
  if (err) return reject('The AI engine failed to load (network issue). Check your connection and retry.');

  const lms = det && det.landmarks && det.landmarks[0];
  const world = det && det.worldLandmarks && det.worldLandmarks[0];
  if (!lms || !world) return reject('No human detected in frame. Make sure your whole body is visible with decent lighting.');

  const q = checkQuality(lms, world, view);
  if (!q.ok) {
    releaseMasks(det);
    return reject(q.msg);
  }

  // Body outline → spinal curvature (sagittal view only).
  let contour = null;
  if (view === 'side' && det.segmentationMasks && det.segmentationMasks[0]) {
    const m = det.segmentationMasks[0];
    try {
      const facingRight = lms[LM.nose].x > (lms[LM.leftShoulder].x + lms[LM.rightShoulder].x) / 2;
      contour = analyseBackContour(m.getAsUint8Array(), m.width, m.height, lms, facingRight);
    } catch (e) { contour = null; }
  }
  releaseMasks(det);

  drawScene(ctx, img, W, H, lms, 1);
  sfx('lock', 0.7); // view accepted
  state[view] = { img, lms, world, contour, W, H };
  setThumb(view, canvas.toDataURL('image/jpeg', 0.6));

  if (view === 'front') {
    state.want = 'side';
    renderCaptureUI();
    $('scan-status').textContent = 'FRONTAL VIEW LOCKED ✓';
    playVO('side');
    await sleep(900);
    showPanel('panel-capture');
    if (q.warn) showWarn(q.warn);
  } else {
    if (q.warn) showWarn(q.warn);
    await runFullAnalysis();
  }
}

function releaseMasks(det) {
  if (det && det.segmentationMasks) {
    det.segmentationMasks.forEach(m => { try { m.close(); } catch (e) {} });
  }
}

/* ---------- combined analysis ---------- */
let lastResult = null;

async function runFullAnalysis() {
  const f = state.front, s = state.side;
  const frontMetrics = assessFront(f.lms, f.world, f.img.width, f.img.height);
  const sideMetrics = assessSide(s.lms, s.world, s.img.width, s.img.height, s.contour);
  const all = frontMetrics.concat(sideMetrics);

  lastResult = {
    frontMetrics, sideMetrics,
    frontScore: scoreOf(frontMetrics),
    sideScore: scoreOf(sideMetrics),
    score: scoreOf(all)
  };
  updateCta();

  $('scan-status').textContent = `${all.length} CHECKPOINTS LOCKED · GRADING`;
  say(`${all.length} checkpoints locked across both planes. Grading severity now.`);

  const box = $('metrics');
  box.innerHTML = '';
  for (const [title, list] of [['FRONTAL PLANE', frontMetrics], ['SAGITTAL PLANE', sideMetrics]]) {
    const h = document.createElement('div');
    h.className = 'plane-head';
    h.textContent = '▸ ' + title;
    box.appendChild(h);
    for (const m of list) {
      const row = metricRow(m);
      box.appendChild(row);
      await sleep(80);
      row.classList.add('show');
      sfx('blip', 0.35); // one tick per checkpoint locking in
      await sleep(260);
    }
  }
  await sleep(600);
  revealScore(lastResult);
}

function metricRow(m) {
  const b = sevBucket(m.severity);
  const row = document.createElement('div');
  row.className = 'metric-row';
  const name = document.createElement('span');
  name.className = 'metric-name';
  name.textContent = '▸ ' + m.name;
  const badge = document.createElement('span');
  badge.className = 'metric-badge sev-' + b;
  badge.textContent = SEV_LABELS[b];
  row.appendChild(name); row.appendChild(badge);
  row.title = m.tip;
  return row;
}

/* ---------- score panel ---------- */
function revealScore(result) {
  showPanel('panel-score');
  sfx('reveal', 1);
  const v = verdictFor(result.score);
  $('verdict').textContent = v.title;
  $('verdict-sub').textContent = v.sub;
  $('sub-front').textContent = result.frontScore;
  $('sub-side').textContent = result.sideScore;

  const fin = $('metrics-final');
  fin.innerHTML = '';
  for (const [title, list, sub] of [['FRONTAL PLANE', result.frontMetrics, result.frontScore],
                                    ['SAGITTAL PLANE', result.sideMetrics, result.sideScore]]) {
    const h = document.createElement('div');
    h.className = 'plane-head';
    h.textContent = '▸ ' + title;
    const badge = document.createElement('span');
    badge.textContent = sub + '/100';
    h.appendChild(badge);
    fin.appendChild(h);
    for (const m of list) {
      const row = metricRow(m);
      row.classList.add('show');
      fin.appendChild(row);
    }
  }

  const CIRC = 2 * Math.PI * 86;
  const ring = $('ring-fg');
  ring.style.strokeDasharray = CIRC;
  ring.style.strokeDashoffset = CIRC;
  ring.style.stroke = result.score >= 75 ? 'var(--green)' : result.score >= 55 ? 'var(--cyan)'
                    : result.score >= 40 ? 'var(--orange)' : 'var(--red)';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    ring.style.strokeDashoffset = CIRC * (1 - result.score / 100);
  }));
  const numEl = $('score-num');
  const t0 = performance.now();
  (function count() {
    const pr = Math.min(1, (performance.now() - t0) / 1900);
    numEl.textContent = Math.round(result.score * (1 - Math.pow(1 - pr, 3)));
    if (pr < 1) requestAnimationFrame(count);
  })();

  buildSimulation();
  say(v.voice + ' Drag the slider to see your aligned posture. To correct these deviations, I recommend the Kina P T protocol.');
}

/* ---------- before/after alignment simulation ---------- */
function buildSimulation() {
  // The sagittal view shows postural change most clearly.
  const src = state.side || state.front;
  const view = state.side ? 'side' : 'front';
  if (!src) { $('slider-box').classList.add('hidden'); return; }

  const W = src.W, H = src.H;
  const before = $('before-canvas');
  before.width = W; before.height = H;
  before.getContext('2d').drawImage(src.img, 0, 0, W, H);

  const after = $('after-canvas');
  after.width = W; after.height = H;
  const actx = after.getContext('2d');
  try {
    const warped = warpImage(before, idealTargets(src.lms, view, W, H), W, H);
    actx.drawImage(warped, 0, 0);
  } catch (e) {
    actx.drawImage(before, 0, 0); // never break the page over a cosmetic feature
  }

  const range = $('slider-range');
  const apply = (pct) => {
    $('after-layer').style.clipPath = `inset(0 0 0 ${pct}%)`;
    $('slider-handle').style.left = `calc(${pct}% - 1px)`;
  };
  range.addEventListener('input', () => apply(+range.value));
  range.value = 50;
  apply(50);
}

/* ---------- share ---------- */
function buildShareCard(result) {
  const c = $('share-canvas'), ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.fillStyle = '#03070d'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(36,221,221,0.07)';
  for (let x = 0; x < W; x += 54) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = 0; y < H; y += 54) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  ctx.strokeStyle = '#24dddd'; ctx.lineWidth = 3;
  ctx.strokeRect(40, 40, W - 80, H - 80);

  ctx.fillStyle = '#24dddd'; ctx.font = '600 42px monospace'; ctx.textAlign = 'center';
  ctx.fillText('KINA·OS // POSTURE.SCAN', W / 2, 130);
  ctx.fillStyle = '#7fb2c6'; ctx.font = '26px monospace';
  ctx.fillText('AI POSTURE ANALYSIS · 2-VIEW SCAN', W / 2, 176);

  const cx = W / 2, cy = 520, r = 235;
  ctx.strokeStyle = 'rgba(36,221,221,0.17)'; ctx.lineWidth = 32;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.stroke();
  const col = result.score >= 75 ? '#5fe6b0' : result.score >= 55 ? '#24dddd' : result.score >= 40 ? '#ffb347' : '#ed4d42';
  ctx.strokeStyle = col; ctx.lineCap = 'round';
  ctx.shadowColor = col; ctx.shadowBlur = 30;
  ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + (result.score / 100) * Math.PI * 2); ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#fff'; ctx.font = '700 180px monospace';
  ctx.fillText(String(result.score), cx, cy + 52);
  ctx.fillStyle = '#7fb2c6'; ctx.font = '38px monospace';
  ctx.fillText('/ 100', cx, cy + 118);

  const v = verdictFor(result.score);
  ctx.fillStyle = '#fff'; ctx.font = '600 50px sans-serif';
  ctx.fillText(v.title.replace(/^[^\w]+\s*/, ''), W / 2, 862);
  ctx.font = '30px monospace'; ctx.fillStyle = '#7fb2c6';
  ctx.fillText(`FRONT ${result.frontScore}    ·    SIDE ${result.sideScore}`, W / 2, 918);

  ctx.font = '29px monospace'; ctx.textAlign = 'left';
  let y = 990;
  const top = result.frontMetrics.concat(result.sideMetrics)
    .sort((a, b) => b.severity - a.severity).slice(0, 5);
  for (const m of top) {
    const b = sevBucket(m.severity);
    ctx.fillStyle = '#b8dced'; ctx.fillText('▸ ' + m.name, 110, y);
    ctx.fillStyle = ['#5fe6b0', '#24dddd', '#ffb347', '#ed4d42'][b];
    ctx.textAlign = 'right'; ctx.fillText(SEV_LABELS[b], W - 110, y);
    ctx.textAlign = 'left';
    y += 52;
  }

  ctx.textAlign = 'center';
  ctx.fillStyle = '#24dddd'; ctx.font = '600 36px monospace';
  ctx.fillText('Can you beat my score?', W / 2, H - 128);
  ctx.fillStyle = '#7fb2c6'; ctx.font = '29px monospace';
  ctx.fillText(APP_URL.replace(/^https?:\/\//, ''), W / 2, H - 78);
  return c;
}

let sharing = false;
$('btn-share').addEventListener('click', async () => {
  if (!lastResult || sharing) return;
  sharing = true;
  $('btn-share').disabled = true;
  try {
    const card = buildShareCard(lastResult);
    const shareText = `KINA scanned my posture from both angles: ${lastResult.score}/100 😳 Can you beat my score?`;
    const blob = await new Promise(r => card.toBlob(r, 'image/png'));
    const file = blob && new File([blob], 'posture-score.png', { type: 'image/png' });

    if (navigator.share) {
      // Transient failures (sheet already open, lost activation) are no-ops —
      // never fall through to the desktop path on a device with a share sheet.
      try {
        if (file && navigator.canShare && navigator.canShare({ files: [file] })) {
          await navigator.share({ files: [file], text: shareText, title: 'My Posture Score' });
        } else {
          await navigator.share({ text: shareText + ' ' + APP_URL, title: 'My Posture Score' });
        }
      } catch (e) { /* AbortError / NotAllowedError / InvalidStateError */ }
      return;
    }
    if (blob) {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'posture-score.png';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    }
    window.open('https://twitter.com/intent/tweet?text=' + encodeURIComponent(shareText + ' ' + APP_URL), '_blank');
  } finally {
    sharing = false;
    $('btn-share').disabled = false;
  }
});

/* ---------- retry ---------- */
$('btn-retry').addEventListener('click', () => {
  hideError(); hideWarn();
  state.front = null; state.side = null; state.want = 'front';
  resetThumb('front'); resetThumb('side');
  renderCaptureUI();
  showPanel('panel-capture');
  say('Rescanning. Show me your best posture, human.');
});

/* ---------- CTA ----------
   Deep-links to the App Store carrying scan context, so an attribution
   SDK (Branch / AppsFlyer) can hand the score to the app on first open
   and KinaPT can resume where the scan left off. */
function updateCta() {
  let href = APPSTORE_URL;
  try {
    const url = new URL(APPSTORE_URL);
    url.searchParams.set('ct', 'posture-scan');
    url.searchParams.set('pt', 'kinapt');
    if (lastResult) url.searchParams.set('score', String(lastResult.score));
    href = url.toString();
  } catch (e) { /* fall back to the bare listing URL */ }
  $('btn-kinapt').href = href;
}

updateCta();
renderCaptureUI();
// Start buffering the voiceover now so the Begin tap plays instantly.
Object.entries(VO_LINES).forEach(([k, s]) => getVONode(k, s.file));

/* Test seam — exposed only when served locally, so the browser test harness
   can drive the flow without a live camera. Never active on the deployed site. */
if (['localhost', '127.0.0.1'].includes(location.hostname)) {
  window.__scan = {
    state, ingest, runFullAnalysis, buildShareCard, warpImage, getLandmarker, showPanel,
    boot: () => ({ running: bootRunning, done: bootDone }),
    audio: () => ({
      energy: coreEnergy,
      analyser: !!coreAnalyser,
      ctx: audioCtx && audioCtx.state,
      t: activeVO ? activeVO.currentTime : null,
      paused: activeVO ? activeVO.paused : null,
      muted: activeVO ? (activeVO.muted || activeVO.volume === 0) : null
    })
  };
}
