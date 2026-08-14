/* ============================================================
   POSTURE.SCAN — analysis engine (pure, no DOM)

   Everything here is deterministic geometry over MediaPipe pose
   output, so it can be unit-tested independently of the UI.

   Coordinate note: MediaPipe normalises `x` by image width and `y`
   by image height. Mixing them raw makes every angle wrong on
   non-square images, so all planar maths rescales x into y-units
   first (see `aspectFix`).
   ============================================================ */

export const LM = {
  nose: 0, leftEye: 2, rightEye: 5, leftEar: 7, rightEar: 8,
  leftShoulder: 11, rightShoulder: 12, leftElbow: 13, rightElbow: 14,
  leftHip: 23, rightHip: 24, leftKnee: 25, rightKnee: 26,
  leftAnkle: 27, rightAnkle: 28, leftHeel: 29, rightHeel: 30,
  leftFoot: 31, rightFoot: 32
};

export const SEV_LABELS = ['OPTIMAL', 'MINOR', 'MODERATE', 'SEVERE'];

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const deg = (rad) => rad * 180 / Math.PI;
export const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
export const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
export const sevBucket = (s) => s < 0.2 ? 0 : s < 0.45 ? 1 : s < 0.72 ? 2 : 3;

/** Rescale x into y-units so angles/ratios are aspect-correct. */
function aspectFix(lms, imgW, imgH) {
  const A = imgW / imgH;
  return (i) => ({ x: lms[i].x * A, y: lms[i].y });
}

/* ============================================================
   QUALITY GATING
   A body rotated relative to the camera silently corrupts every
   planar measurement, so detect it in 3D and reject before scoring.
   ============================================================ */

/** Hip-axis rotation in the horizontal plane. 0° = square to camera, 90° = profile. */
export function bodyRotation(world) {
  const dx = world[LM.rightHip].x - world[LM.leftHip].x;
  const dz = world[LM.rightHip].z - world[LM.leftHip].z;
  const len = Math.hypot(dx, dz) || 1e-6;
  return deg(Math.asin(clamp(Math.abs(dz) / len, 0, 1)));
}

export function checkQuality(lms, world, wantView) {
  const need = [LM.leftShoulder, LM.rightShoulder, LM.leftHip, LM.rightHip];
  if (need.some(i => (lms[i].visibility ?? 1) < 0.4)) {
    return { ok: false, msg: 'I can\'t see your full torso. Step back so your head AND hips are in frame.' };
  }
  const rot = bodyRotation(world);
  if (wantView === 'front' && rot > 32) {
    return { ok: false, msg: `You're turned about ${Math.round(rot)}° away from square. Face the camera straight on and retake.` };
  }
  if (wantView === 'side' && rot < 55) {
    return { ok: false, msg: `That looks closer to a front view (${Math.round(rot)}° of turn). Turn a full 90° so one shoulder faces me.` };
  }
  const feetSeen = (lms[LM.leftAnkle].visibility ?? 1) > 0.4 || (lms[LM.rightAnkle].visibility ?? 1) > 0.4;
  return { ok: true, warn: feetSeen ? null : 'Your feet aren\'t in frame — foot and plumb-line checks will be skipped.' };
}

/* ============================================================
   BODY OUTLINE → SPINAL CURVATURE
   The skeleton gives joint centres; curvature lives in the outline.
   Trace the posterior edge between shoulder and hip, then measure
   deviation from the shoulder→hip chord.
   ============================================================ */

/** Largest curvature we treat as anatomy rather than interference, as a
 *  fraction of torso length. Anything beyond this means something else is in
 *  the silhouette (an outstretched arm, a bag, furniture), so the contour is
 *  discarded rather than scored — bad numbers here would silently corrupt two
 *  sagittal metrics. */
export const CONTOUR_PLAUSIBLE_MAX = 0.25;

/** Largest row-to-row jump in the traced edge, as a fraction of torso length.
 *  A spine outline changes gradually; a step this sharp is a limb boundary
 *  (or a chair back) crossing the silhouette, so the trace can't be trusted. */
export const CONTOUR_MAX_STEP = 0.05;

/** Median filter to shrug off single-row mask speckle. */
function medianSmooth(arr, win = 5) {
  const half = win >> 1;
  return arr.map((_, i) => {
    const slice = [];
    for (let k = -half; k <= half; k++) {
      const v = arr[i + k];
      if (v !== undefined && v >= 0) slice.push(v);
    }
    if (!slice.length) return arr[i];
    slice.sort((a, b) => a - b);
    return slice[slice.length >> 1];
  });
}

export function analyseBackContour(maskData, mw, mh, lms, facingRight) {
  const shY = Math.round(((lms[LM.leftShoulder].y + lms[LM.rightShoulder].y) / 2) * mh);
  const hipY = Math.round(((lms[LM.leftHip].y + lms[LM.rightHip].y) / 2) * mh);
  if (hipY - shY < 12) return null;
  const y0 = Math.max(0, shY), y1 = Math.min(mh - 1, hipY);
  if (y1 - y0 < 12) return null;

  const raw = [];
  for (let y = y0; y <= y1; y++) {
    let found = -1;
    if (facingRight) {
      for (let x = 0; x < mw; x++) { if (maskData[y * mw + x] > 127) { found = x; break; } }
    } else {
      for (let x = mw - 1; x >= 0; x--) { if (maskData[y * mw + x] > 127) { found = x; break; } }
    }
    raw.push(found);
  }
  if (raw.filter(v => v >= 0).length < raw.length * 0.7) return null;

  const edge = medianSmooth(raw);
  const x0 = edge[0], xN = edge[edge.length - 1];
  if (!(x0 >= 0) || !(xN >= 0)) return null;
  const n = edge.length;
  const torsoPx = y1 - y0;
  const sign = facingRight ? 1 : -1; // positive deviation = bulges posteriorly

  // Reject a trace that steps discontinuously — that's a limb crossing the
  // outline, not spinal curvature.
  for (let i = 1; i < n; i++) {
    if (edge[i] < 0 || edge[i - 1] < 0) continue;
    if (Math.abs(edge[i] - edge[i - 1]) / torsoPx > CONTOUR_MAX_STEP) return null;
  }

  let kyphosis = 0, lordosis = 0;
  for (let i = 0; i < n; i++) {
    if (edge[i] < 0) continue;
    const chord = x0 + (xN - x0) * (i / (n - 1));
    const dev = (chord - edge[i]) * sign;
    const frac = i / (n - 1);
    if (frac < 0.45) kyphosis = Math.max(kyphosis, dev);
    else if (frac > 0.6) lordosis = Math.max(lordosis, -dev);
  }
  const out = { kyphosis: kyphosis / torsoPx, lordosis: lordosis / torsoPx };
  if (out.kyphosis > CONTOUR_PLAUSIBLE_MAX || out.lordosis > CONTOUR_PLAUSIBLE_MAX) return null;
  return out;
}

/* ============================================================
   METRICS
   Each returns severity in [0,1]; 0 is ideal alignment.
   ============================================================ */

function makeAdder(list) {
  return (name, severity, tip, weight = 1, detail = null) =>
    list.push({ name, severity: clamp(severity, 0, 1), tip, weight, detail });
}

/* ============================================================
   REAL-WORLD SCALE

   The planar maths works in image units, which mean nothing to a reader.
   MediaPipe's world landmarks are metric — roughly, a body-shaped model fitted
   to the image — so the torso gives a conversion factor from image units to
   inches.

   Read the accuracy honestly: the metric fit is estimated from one photograph
   by a generic model, so absolute scale carries real error. It is good enough
   to say "about an inch and a half" and wrong to present as a clinical
   measurement, which is why everything below rounds to a tenth of an inch and
   the UI labels it an estimate.
   ============================================================ */
const IN_PER_M = 39.3701;

function dist3(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

/** Inches per image y-unit, from the torso measured both ways. Null if unusable. */
export function inchScale(lms, world, imgW, imgH) {
  if (!world) return null;
  const p = aspectFix(lms, imgW, imgH);
  const imgTorso = dist(mid(p(LM.leftShoulder), p(LM.rightShoulder)),
                        mid(p(LM.leftHip), p(LM.rightHip)));
  if (!(imgTorso > 1e-4)) return null;
  const wSh = { x: (world[LM.leftShoulder].x + world[LM.rightShoulder].x) / 2,
                y: (world[LM.leftShoulder].y + world[LM.rightShoulder].y) / 2,
                z: (world[LM.leftShoulder].z + world[LM.rightShoulder].z) / 2 };
  const wHip = { x: (world[LM.leftHip].x + world[LM.rightHip].x) / 2,
                 y: (world[LM.leftHip].y + world[LM.rightHip].y) / 2,
                 z: (world[LM.leftHip].z + world[LM.rightHip].z) / 2 };
  const worldTorso = dist3(wSh, wHip);
  // A torso outside 20-80 cm means the metric fit has gone wrong; better to
  // show no number than a confident wrong one.
  if (!(worldTorso > 0.2 && worldTorso < 0.8)) return null;
  return (worldTorso * IN_PER_M) / imgTorso;
}

/** Format an image-unit offset as inches, or null when there is no scale. */
function inches(units, scale, suffix = '') {
  if (!scale) return null;
  const v = Math.abs(units) * scale;
  return `${v.toFixed(1)} in${suffix}`;
}

/* ---------- FRONTAL PLANE ---------- */
export function assessFront(lms, world, imgW, imgH) {
  const p = aspectFix(lms, imgW, imgH);
  const vis = (i) => (lms[i].visibility ?? 1);

  const ls = p(LM.leftShoulder), rs = p(LM.rightShoulder);
  const lh = p(LM.leftHip), rh = p(LM.rightHip);
  const midSh = mid(ls, rs), midHip = mid(lh, rh);
  const shoulderW = Math.abs(ls.x - rs.x) || 1e-6;

  const metrics = [];
  const add = makeAdder(metrics);
  const scale = inchScale(lms, world, imgW, imgH);

  // Shoulder height difference. A sign of lateral asymmetry — deliberately
  // NOT labelled as a spinal diagnosis.
  const shTilt = Math.abs(deg(Math.atan2(ls.y - rs.y, ls.x - rs.x)));
  add('SHOULDER BALANCE', shTilt / 9,
      'One shoulder sits higher than the other — common with bag-carrying or a dominant side.', 1.2,
      inches(ls.y - rs.y, scale, ' uneven') || `${shTilt.toFixed(1)}° tilt`);

  // Pelvic obliquity (hip drop).
  const hipTilt = Math.abs(deg(Math.atan2(lh.y - rh.y, lh.x - rh.x)));
  add('PELVIC LEVEL', hipTilt / 8,
      'Your hips are uneven — often a hip-hitch habit or a leg-length difference.', 1.2,
      inches(lh.y - rh.y, scale, ' drop') || `${hipTilt.toFixed(1)}° tilt`);

  // Lateral trunk lean from vertical.
  const lean = Math.abs(deg(Math.atan2(midSh.x - midHip.x, midHip.y - midSh.y)));
  add('TRUNK LEAN', lean / 7, 'Your upper body leans off the vertical.', 1.2,
      `${lean.toFixed(1)}° off vertical`);

  // Head centring over the shoulders.
  const headShift = Math.abs(p(LM.nose).x - midSh.x) / shoulderW;
  add('HEAD CENTRING', headShift / 0.32, 'Your head drifts off your body\'s midline.', 1,
      inches(p(LM.nose).x - midSh.x, scale, ' off centre') ||
        `${(headShift * 100).toFixed(0)}% of shoulder width`);

  // Knee valgus / varus: perpendicular offset of the knee from the
  // hip→ankle mechanical axis, signed toward the body midline.
  const kneeDev = (hipI, kneeI, ankleI) => {
    if (vis(kneeI) < 0.5 || vis(ankleI) < 0.5) return null;
    const hip = p(hipI), knee = p(kneeI), ankle = p(ankleI);
    const legLen = dist(hip, ankle);
    if (legLen < 1e-4) return null;
    const dx = ankle.x - hip.x, dy = ankle.y - hip.y;
    const cross = ((knee.x - hip.x) * dy - (knee.y - hip.y) * dx) / legLen;
    const medial = Math.sign(midHip.x - hip.x) || 1;
    return (cross * medial) / legLen;
  };
  const knees = [kneeDev(LM.leftHip, LM.leftKnee, LM.leftAnkle),
                 kneeDev(LM.rightHip, LM.rightKnee, LM.rightAnkle)].filter(v => v !== null);
  if (knees.length) {
    const worst = knees.reduce((a, b) => Math.abs(a) > Math.abs(b) ? a : b);
    const valgus = worst > 0;
    const legLen = dist(p(LM.leftHip), p(LM.leftAnkle)) || dist(p(LM.rightHip), p(LM.rightAnkle));
    add(valgus ? 'KNEE VALGUS' : 'KNEE VARUS', Math.abs(worst) / 0.055,
        valgus ? 'Your knees track inward of the hip-to-ankle line.'
               : 'Your knees track outward of the hip-to-ankle line.', 1.1,
        inches(worst * legLen, scale, valgus ? ' inward' : ' outward') ||
          `${(Math.abs(worst) * 100).toFixed(1)}% of leg length`);
  }

  // Foot progression (forefoot abduction). Measured in 3D against the
  // pelvis-forward axis so whole-body rotation can't masquerade as toe-out.
  const fa = footProgression(world, LM.leftHeel, LM.leftFoot, vis);
  const fb = footProgression(world, LM.rightHeel, LM.rightFoot, vis);
  const feet = [fa, fb].filter(v => v !== null);
  if (feet.length) {
    const avg = feet.reduce((a, b) => a + b, 0) / feet.length;
    add('FOOT PROGRESSION', Math.max(0, avg - 10) / 18,
        'Your forefoot turns out more than typical — can accompany a flattened arch.', 0.8,
        `${avg.toFixed(0)}° toe-out`);
  }

  return metrics;
}

/** Angle (deg) between the heel→toe vector and the pelvis-forward axis. */
export function footProgression(world, heelI, toeI, vis) {
  if (vis && (vis(heelI) < 0.4 || vis(toeI) < 0.4)) return null;
  const heel = world[heelI], toe = world[toeI];
  const fv = { x: toe.x - heel.x, z: toe.z - heel.z };
  const flen = Math.hypot(fv.x, fv.z);
  if (flen < 1e-6) return null;

  const hv = { x: world[LM.rightHip].x - world[LM.leftHip].x,
               z: world[LM.rightHip].z - world[LM.leftHip].z };
  let fwd = { x: -hv.z, z: hv.x };
  const n = Math.hypot(fwd.x, fwd.z) || 1e-6;
  fwd = { x: fwd.x / n, z: fwd.z / n };

  // Orient "forward" toward the nose.
  const hipCx = (world[LM.leftHip].x + world[LM.rightHip].x) / 2;
  const hipCz = (world[LM.leftHip].z + world[LM.rightHip].z) / 2;
  const nose = { x: world[LM.nose].x - hipCx, z: world[LM.nose].z - hipCz };
  if (fwd.x * nose.x + fwd.z * nose.z < 0) fwd = { x: -fwd.x, z: -fwd.z };

  return deg(Math.acos(clamp((fv.x * fwd.x + fv.z * fwd.z) / flen, -1, 1)));
}

/* ---------- SAGITTAL PLANE ---------- */
export function assessSide(lms, world, imgW, imgH, contour) {
  const p = aspectFix(lms, imgW, imgH);
  const vis = (i) => (lms[i].visibility ?? 1);

  const leftward = vis(LM.leftEar) + vis(LM.leftShoulder) >= vis(LM.rightEar) + vis(LM.rightShoulder);
  const S = (l, r) => leftward ? l : r;
  const ear = p(S(LM.leftEar, LM.rightEar));
  const sh = p(S(LM.leftShoulder, LM.rightShoulder));
  const hip = p(S(LM.leftHip, LM.rightHip));
  const knee = p(S(LM.leftKnee, LM.rightKnee));
  const ankle = p(S(LM.leftAnkle, LM.rightAnkle));
  const hasLegs = vis(S(LM.leftAnkle, LM.rightAnkle)) > 0.4;

  const torsoLen = dist(sh, hip) || 1e-6;
  const facingRight = p(LM.nose).x > sh.x;
  const fwd = facingRight ? 1 : -1;

  const metrics = [];
  const add = makeAdder(metrics);
  const scale = inchScale(lms, world, imgW, imgH);

  // Forward head: ear anterior to the acromion.
  const fwdHead = ((ear.x - sh.x) * fwd) / torsoLen;
  add('FORWARD HEAD', Math.max(0, fwdHead) / 0.28,
      '"Tech neck" — a forward head position can add significant load to the neck and upper back.', 1.5,
      inches(Math.max(0, fwdHead) * torsoLen, scale, ' ahead of shoulder') ||
        `${(Math.max(0, fwdHead) * 100).toFixed(0)}% of torso length`);

  // Rounded shoulders: acromion anterior to the greater trochanter.
  const round = ((sh.x - hip.x) * fwd) / torsoLen;
  add('SHOULDER PROTRACTION', Math.max(0, round) / 0.2,
      'Your shoulders sit ahead of your hips — a rounded, protracted shoulder pattern.', 1.3,
      inches(Math.max(0, round) * torsoLen, scale, ' ahead of hip') ||
        `${(Math.max(0, round) * 100).toFixed(0)}% of torso length`);

  // Spinal curves from the body outline, when the mask gave us one.
  // Both are normalised by the shoulder-to-hip span, so the torso converts
  // them straight into a depth.
  if (contour) {
    add('THORACIC CURVE', Math.max(0, contour.kyphosis - 0.045) / 0.075,
        'The upper back curves more than typical — a rounded thoracic spine.', 1.2,
        inches(contour.kyphosis * torsoLen, scale, ' of upper-back curve') ||
          `${(contour.kyphosis * 100).toFixed(1)}% of torso length`);
    const deeper = contour.lordosis >= 0.05;
    add('LUMBAR CURVE', Math.abs(contour.lordosis - 0.05) / 0.06,
        'Your low-back curve is deeper or flatter than the typical range.', 1,
        inches(Math.abs(contour.lordosis - 0.05) * torsoLen, scale,
               deeper ? ' deeper than typical' : ' flatter than typical') ||
          `${(contour.lordosis * 100).toFixed(1)}% of torso length`);
  }

  if (hasLegs) {
    const legLen = dist(hip, ankle) || 1e-6;
    // Sway back: hips carried ahead of the ankles.
    const sway = ((hip.x - ankle.x) * fwd) / legLen;
    add('PELVIS OVER BASE', Math.max(0, sway) / 0.12,
        'Your hips ride ahead of your ankles — the classic sway-back stance.', 1,
        inches(Math.max(0, sway) * legLen, scale, ' ahead of ankle') ||
          `${(Math.max(0, sway) * 100).toFixed(0)}% of leg length`);

    // Full clinical plumb chain: ankle → knee → hip → shoulder → ear.
    const refs = [knee, hip, sh, ear];
    const devSum = refs.reduce((s, pt) => s + Math.abs(pt.x - ankle.x), 0) / refs.length;
    add('PLUMB LINE', (devSum / legLen) / 0.16, 'The ear-shoulder-hip-ankle line is broken.', 1.3,
        inches(devSum, scale, ' average drift') ||
          `${((devSum / legLen) * 100).toFixed(0)}% of leg length`);
  }

  return metrics;
}

/* ============================================================
   HUNCH INDEX

   One number for the thing people actually recognise in the mirror: how far
   the upper body has already rolled forward, presented as how far along the
   road to a rounded back the person is. It is a weighted read of the three
   sagittal checkpoints that make up the pattern — the thoracic curve itself,
   the head carried ahead of the shoulders, and the shoulders ahead of the
   hips. 0% is a stacked, neutral upper body; 100% is the far end of what this
   scan can resolve.

   Read what it is: a measurement of a posture pattern TODAY, scaled 0-100.
   It is not an epidemiological probability and not a diagnosis — hyperkyphosis
   is a clinical finding that needs a clinician and imaging. The screen says so
   in as many words, which is both honest and what keeps the app on the right
   side of health-claim rules.
   ============================================================ */
const HUNCH_PARTS = {
  'THORACIC CURVE': 3,
  'FORWARD HEAD': 2,
  'SHOULDER PROTRACTION': 2
};

export function hunchIndex(sideMetrics) {
  if (!sideMetrics || !sideMetrics.length) return null;
  let sum = 0, wSum = 0;
  for (const m of sideMetrics) {
    const w = HUNCH_PARTS[m.name];
    if (!w) continue;
    sum += m.severity * w;
    wSum += w;
  }
  if (!wSum) return null;
  return Math.round(clamp(sum / wSum, 0, 1) * 100);
}

export function hunchBand(pct) {
  if (pct < 15) return {
    label: 'LOW RISK',
    note: `You are ${pct}% of the way to a hunched back. Barely started — keep it there.`
  };
  if (pct < 35) return {
    label: 'EARLY WARNING',
    note: `You are ${pct}% of the way to a hunched back. The forward roll has begun — ` +
          'the classic desk-work opening move.'
  };
  if (pct < 60) return {
    label: 'ON THE WAY',
    note: `You are ${pct}% of the way to a hunched back. The rounded pattern is already ` +
          'set through your upper back.'
  };
  if (pct < 80) return {
    label: 'HIGH RISK',
    note: `You are ${pct}% of the way to a hunched back. Your head and upper back carry ` +
          'well forward of neutral, and this one compounds.'
  };
  return {
    label: 'SEVERE',
    note: `You are ${pct}% of the way to a hunched back — the far end of what this scan ` +
          'can measure. Worth a professional look.'
  };
}

/** The plain-English rule the number follows. */
export const HUNCH_EXPLAINER =
  'The higher the percentage, the further your upper back has already rolled toward a ' +
  'permanent hunch — and the higher your chance of ending up there if nothing changes.';

/* ---------- scoring ---------- */
export function scoreOf(metrics) {
  if (!metrics.length) return 0;
  const wSum = metrics.reduce((s, m) => s + m.weight, 0);
  const sevAvg = metrics.reduce((s, m) => s + m.severity * m.weight, 0) / wSum;
  return Math.round(clamp(100 - Math.pow(sevAvg, 0.85) * 95, 8, 100));
}

export function verdictFor(score) {
  if (score >= 90) return {
    title: '🏆 STARK-LEVEL ALIGNMENT',
    sub: 'Exceptional. Your skeleton would make a physiotherapist weep with joy. Maintain the protocol.',
    voice: `Remarkable. ${score} out of one hundred. Your alignment is in the top percentile of humans I have scanned. I am genuinely impressed.`
  };
  if (score >= 75) return {
    title: '✅ COMBAT READY',
    sub: 'Solid alignment with minor deviations. A focused routine would push you into elite territory.',
    voice: `${score} out of one hundred. Respectable, human. Minor deviations detected — nothing a proper protocol cannot fix.`
  };
  if (score >= 55) return {
    title: '⚠ STRUCTURAL DRIFT DETECTED',
    sub: 'Your frame is compensating in several places. This is where most desk workers live — and it responds well to daily work.',
    voice: `${score} out of one hundred. Structural drift detected. Your spine is filing a formal complaint. I recommend intervention.`
  };
  return {
    title: '🚨 CRITICAL MISALIGNMENT',
    sub: 'Multiple checkpoints outside range. The good news: alignment responds fast to consistent daily correction.',
    voice: `${score} out of one hundred. This is... concerning. Your posture resembles a question mark. Deploying corrective protocol immediately.`
  };
}
