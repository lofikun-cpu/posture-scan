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

const APP_URL = location.origin + location.pathname;

/* ---------- helpers ---------- */
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function showPanel(id) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  $(id).classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

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
    ctx.strokeStyle = 'rgba(33,230,255,0.04)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 48) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y < h; y += 48) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    for (const d of dots) {
      d.y -= d.v; if (d.y < 0) d.y = 1;
      ctx.fillStyle = 'rgba(33,230,255,0.35)';
      ctx.beginPath(); ctx.arc(d.x * w, d.y * h, d.s, 0, 7); ctx.fill();
    }
    const sy = (t * 0.6) % (h + 200) - 100;
    const grad = ctx.createLinearGradient(0, sy - 60, 0, sy + 60);
    grad.addColorStop(0, 'rgba(33,230,255,0)');
    grad.addColorStop(0.5, 'rgba(33,230,255,0.05)');
    grad.addColorStop(1, 'rgba(33,230,255,0)');
    ctx.fillStyle = grad; ctx.fillRect(0, sy - 60, w, 120);
    requestAnimationFrame(loop);
  })();
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
  $('mute-btn').textContent = voiceOn ? 'VOICE: ON' : 'VOICE: OFF';
  if (!voiceOn && 'speechSynthesis' in window) speechSynthesis.cancel();
});

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

/* ---------- MediaPipe lazy loader ---------- */
let landmarkerPromise = null;
function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await import('./vendor/mediapipe/vision_bundle.mjs');
      const files = await vision.FilesetResolver.forVisionTasks('./vendor/mediapipe/wasm');
      return vision.PoseLandmarker.createFromOptions(files, {
        baseOptions: {
          modelAssetPath: './vendor/models/pose_landmarker_full.task',
          delegate: 'GPU'
        },
        runningMode: 'IMAGE',
        numPoses: 1,
        outputSegmentationMasks: true,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5
      });
    })().catch(err => { landmarkerPromise = null; throw err; });
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
      ctx.strokeStyle = 'rgba(33,230,255,0.85)';
      ctx.shadowColor = '#21e6ff'; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
    }
    ctx.shadowBlur = 0;
    for (const i of KEY_PTS) {
      if ((lms[i].visibility ?? 1) < 0.4) continue;
      const pt = P(i);
      ctx.fillStyle = '#21e6ff';
      ctx.beginPath(); ctx.arc(pt.x, pt.y, Math.max(3, W / 220), 0, 7); ctx.fill();
      ctx.strokeStyle = 'rgba(33,230,255,0.5)';
      ctx.beginPath(); ctx.arc(pt.x, pt.y, Math.max(7, W / 90), 0, 7); ctx.stroke();
    }
  }

  if (progress < 1) {
    const y = H * progress;
    const grad = ctx.createLinearGradient(0, y - 40, 0, y + 40);
    grad.addColorStop(0, 'rgba(33,230,255,0)');
    grad.addColorStop(0.5, 'rgba(33,230,255,0.35)');
    grad.addColorStop(1, 'rgba(33,230,255,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, y - 40, W, 80);
    ctx.fillStyle = 'rgba(33,230,255,0.9)';
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

const GUIDE_FRONT = `<svg viewBox="0 0 100 220" fill="none" stroke="#21e6ff" stroke-width="1.4" stroke-dasharray="4 4">
  <circle cx="50" cy="20" r="12"/><line x1="50" y1="32" x2="50" y2="120"/>
  <line x1="28" y1="52" x2="72" y2="52"/><line x1="28" y1="52" x2="24" y2="105"/>
  <line x1="72" y1="52" x2="76" y2="105"/><line x1="34" y1="118" x2="66" y2="118"/>
  <line x1="38" y1="118" x2="36" y2="205"/><line x1="62" y1="118" x2="64" y2="205"/>
  <line x1="50" y1="0" x2="50" y2="220" stroke-width="0.7" opacity="0.5"/></svg>`;
const GUIDE_SIDE = `<svg viewBox="0 0 100 220" fill="none" stroke="#21e6ff" stroke-width="1.4" stroke-dasharray="4 4">
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

const GREET_TEXT =
  'Good day. I am KINA — your personal posture intelligence.\n' +
  'I will scan you from two angles and grade your alignment out of 100.\n' +
  'Very few humans achieve a perfect score. Let us see what you are made of.';

typeLines($('console'), GREET_TEXT.split('\n'));

$('btn-start').addEventListener('click', () => {
  say('Initiating posture scan. ' + VIEW_COPY.front.voice);
  getLandmarker().catch(() => {}); // pre-warm while the user gets into position
  state.want = 'front';
  renderCaptureUI();
  showPanel('panel-capture');
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
  state[view] = { img, lms, world, contour, W, H };
  setThumb(view, canvas.toDataURL('image/jpeg', 0.6));

  if (view === 'front') {
    state.want = 'side';
    renderCaptureUI();
    $('scan-status').textContent = 'FRONTAL VIEW LOCKED ✓';
    say(VIEW_COPY.side.voice);
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
  ctx.fillStyle = '#04080f'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(33,230,255,0.06)';
  for (let x = 0; x < W; x += 54) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = 0; y < H; y += 54) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  ctx.strokeStyle = '#21e6ff'; ctx.lineWidth = 3;
  ctx.strokeRect(40, 40, W - 80, H - 80);

  ctx.fillStyle = '#21e6ff'; ctx.font = '600 42px monospace'; ctx.textAlign = 'center';
  ctx.fillText('KINA·OS // POSTURE.SCAN', W / 2, 130);
  ctx.fillStyle = '#6da8bd'; ctx.font = '26px monospace';
  ctx.fillText('AI POSTURE ANALYSIS · 2-VIEW SCAN', W / 2, 176);

  const cx = W / 2, cy = 520, r = 235;
  ctx.strokeStyle = 'rgba(33,230,255,0.15)'; ctx.lineWidth = 32;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.stroke();
  const col = result.score >= 75 ? '#3dffa0' : result.score >= 55 ? '#21e6ff' : result.score >= 40 ? '#ffb347' : '#ff4d5e';
  ctx.strokeStyle = col; ctx.lineCap = 'round';
  ctx.shadowColor = col; ctx.shadowBlur = 30;
  ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + (result.score / 100) * Math.PI * 2); ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#fff'; ctx.font = '700 180px monospace';
  ctx.fillText(String(result.score), cx, cy + 52);
  ctx.fillStyle = '#6da8bd'; ctx.font = '38px monospace';
  ctx.fillText('/ 100', cx, cy + 118);

  const v = verdictFor(result.score);
  ctx.fillStyle = '#fff'; ctx.font = '600 50px sans-serif';
  ctx.fillText(v.title.replace(/^[^\w]+\s*/, ''), W / 2, 862);
  ctx.font = '30px monospace'; ctx.fillStyle = '#6da8bd';
  ctx.fillText(`FRONT ${result.frontScore}    ·    SIDE ${result.sideScore}`, W / 2, 918);

  ctx.font = '29px monospace'; ctx.textAlign = 'left';
  let y = 990;
  const top = result.frontMetrics.concat(result.sideMetrics)
    .sort((a, b) => b.severity - a.severity).slice(0, 5);
  for (const m of top) {
    const b = sevBucket(m.severity);
    ctx.fillStyle = '#a8d8e8'; ctx.fillText('▸ ' + m.name, 110, y);
    ctx.fillStyle = ['#3dffa0', '#21e6ff', '#ffb347', '#ff4d5e'][b];
    ctx.textAlign = 'right'; ctx.fillText(SEV_LABELS[b], W - 110, y);
    ctx.textAlign = 'left';
    y += 52;
  }

  ctx.textAlign = 'center';
  ctx.fillStyle = '#21e6ff'; ctx.font = '600 36px monospace';
  ctx.fillText('Can you beat my score?', W / 2, H - 128);
  ctx.fillStyle = '#6da8bd'; ctx.font = '29px monospace';
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

/* Test seam — exposed only when served locally, so the browser test harness
   can drive the flow without a live camera. Never active on the deployed site. */
if (['localhost', '127.0.0.1'].includes(location.hostname)) {
  window.__scan = { state, ingest, runFullAnalysis, buildShareCard, warpImage, getLandmarker, showPanel };
}
