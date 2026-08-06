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
  return (name, severity, tip, weight = 1) =>
    list.push({ name, severity: clamp(severity, 0, 1), tip, weight });
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

  // Shoulder height difference. A sign of lateral asymmetry — deliberately
  // NOT labelled as a spinal diagnosis.
  const shTilt = Math.abs(deg(Math.atan2(ls.y - rs.y, ls.x - rs.x)));
  add('SHOULDER BALANCE', shTilt / 9,
      'One shoulder sits higher than the other — common with bag-carrying or a dominant side.', 1.2);

  // Pelvic obliquity (hip drop).
  const hipTilt = Math.abs(deg(Math.atan2(lh.y - rh.y, lh.x - rh.x)));
  add('PELVIC LEVEL', hipTilt / 8,
      'Your hips are uneven — often a hip-hitch habit or a leg-length difference.', 1.2);

  // Lateral trunk lean from vertical.
  const lean = Math.abs(deg(Math.atan2(midSh.x - midHip.x, midHip.y - midSh.y)));
  add('TRUNK LEAN', lean / 7, 'Your upper body leans off the vertical.', 1.2);

  // Head centring over the shoulders.
  const headShift = Math.abs(p(LM.nose).x - midSh.x) / shoulderW;
  add('HEAD CENTRING', headShift / 0.32, 'Your head drifts off your body\'s midline.', 1);

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
    add(valgus ? 'KNEE VALGUS' : 'KNEE VARUS', Math.abs(worst) / 0.055,
        valgus ? 'Your knees track inward of the hip-to-ankle line.'
               : 'Your knees track outward of the hip-to-ankle line.', 1.1);
  }

  // Foot progression (forefoot abduction). Measured in 3D against the
  // pelvis-forward axis so whole-body rotation can't masquerade as toe-out.
  const fa = footProgression(world, LM.leftHeel, LM.leftFoot, vis);
  const fb = footProgression(world, LM.rightHeel, LM.rightFoot, vis);
  const feet = [fa, fb].filter(v => v !== null);
  if (feet.length) {
    const avg = feet.reduce((a, b) => a + b, 0) / feet.length;
    add('FOOT PROGRESSION', Math.max(0, avg - 10) / 18,
        'Your forefoot turns out more than typical — can accompany a flattened arch.', 0.8);
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

  // Forward head: ear anterior to the acromion.
  const fwdHead = ((ear.x - sh.x) * fwd) / torsoLen;
  add('FORWARD HEAD', Math.max(0, fwdHead) / 0.28,
      '"Tech neck" — a forward head position can add significant load to the neck and upper back.', 1.5);

  // Rounded shoulders: acromion anterior to the greater trochanter.
  const round = ((sh.x - hip.x) * fwd) / torsoLen;
  add('SHOULDER PROTRACTION', Math.max(0, round) / 0.2,
      'Your shoulders sit ahead of your hips — a rounded, protracted shoulder pattern.', 1.3);

  // Spinal curves from the body outline, when the mask gave us one.
  if (contour) {
    add('THORACIC CURVE', Math.max(0, contour.kyphosis - 0.045) / 0.075,
        'The upper back curves more than typical — a rounded thoracic spine.', 1.2);
    add('LUMBAR CURVE', Math.abs(contour.lordosis - 0.05) / 0.06,
        'Your low-back curve is deeper or flatter than the typical range.', 1);
  }

  if (hasLegs) {
    const legLen = dist(hip, ankle) || 1e-6;
    // Sway back: hips carried ahead of the ankles.
    const sway = ((hip.x - ankle.x) * fwd) / legLen;
    add('PELVIS OVER BASE', Math.max(0, sway) / 0.12,
        'Your hips ride ahead of your ankles — the classic sway-back stance.', 1);

    // Full clinical plumb chain: ankle → knee → hip → shoulder → ear.
    const refs = [knee, hip, sh, ear];
    const devSum = refs.reduce((s, pt) => s + Math.abs(pt.x - ankle.x), 0) / refs.length;
    add('PLUMB LINE', (devSum / legLen) / 0.16, 'The ear-shoulder-hip-ankle line is broken.', 1.3);
  }

  return metrics;
}

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

/* ============================================================
   ALIGNMENT SIMULATION — geometry only
   ============================================================ */

export const WARP_STRENGTH = 0.85; // 1.0 = fully idealised; keep them recognisable

/** Control points {sx,sy,dx,dy} in pixel space moving the body toward plumb. */
export function idealTargets(lms, view, W, H) {
  const P = (i) => ({ x: lms[i].x * W, y: lms[i].y * H });
  const vis = (i) => (lms[i].visibility ?? 1);
  const pts = [];
  const push = (i, tx, ty) => {
    if (vis(i) < 0.4) return;
    const s = P(i);
    pts.push({ sx: s.x, sy: s.y,
               dx: s.x + (tx - s.x) * WARP_STRENGTH,
               dy: s.y + (ty - s.y) * WARP_STRENGTH });
  };

  if (view === 'front') {
    const ls = P(LM.leftShoulder), rs = P(LM.rightShoulder);
    const lh = P(LM.leftHip), rh = P(LM.rightHip);
    const midSh = mid(ls, rs), midHip = mid(lh, rh);
    push(LM.leftShoulder, ls.x, midSh.y);
    push(LM.rightShoulder, rs.x, midSh.y);
    push(LM.leftHip, lh.x, midHip.y);
    push(LM.rightHip, rh.x, midHip.y);
    const nose = P(LM.nose);
    push(LM.nose, midHip.x, nose.y);
    for (const e of [LM.leftEar, LM.rightEar]) {
      if (vis(e) < 0.4) continue;
      const pt = P(e); push(e, pt.x + (midHip.x - nose.x), pt.y);
    }
    for (const [h, k, a] of [[LM.leftHip, LM.leftKnee, LM.leftAnkle],
                             [LM.rightHip, LM.rightKnee, LM.rightAnkle]]) {
      if (vis(k) < 0.4 || vis(a) < 0.4) continue;
      const hp = P(h), kp = P(k), ap = P(a);
      const t = (kp.y - hp.y) / ((ap.y - hp.y) || 1e-6);
      push(k, hp.x + (ap.x - hp.x) * t, kp.y);
    }
  } else {
    const anchorX = (vis(LM.leftAnkle) > 0.4 || vis(LM.rightAnkle) > 0.4)
      ? (vis(LM.leftAnkle) >= vis(LM.rightAnkle) ? P(LM.leftAnkle).x : P(LM.rightAnkle).x)
      : mid(P(LM.leftHip), P(LM.rightHip)).x;
    // Trunk lands on the plumb line.
    for (const i of [LM.leftHip, LM.rightHip, LM.leftShoulder, LM.rightShoulder]) {
      if (vis(i) < 0.4) continue;
      push(i, anchorX, P(i).y);
    }
    // The earlobe is the classic plumb reference, but the face must travel
    // with the head as a rigid unit — pulling the nose onto the line too
    // would flatten the profile.
    const earI = vis(LM.leftEar) >= vis(LM.rightEar) ? LM.leftEar : LM.rightEar;
    if (vis(earI) >= 0.4) {
      const shift = anchorX - P(earI).x;
      for (const i of [LM.leftEar, LM.rightEar, LM.nose]) {
        if (vis(i) < 0.4) continue;
        const pt = P(i);
        push(i, pt.x + shift, pt.y);
      }
    }
  }
  return pts;
}

/** Affine matrix [a,b,c,d,e,f] mapping source triangle → destination triangle. */
export function solveAffine(s0, s1, s2, d0, d1, d2) {
  const den = s0.x * (s2.y - s1.y) + s1.x * (s0.y - s2.y) + s2.x * (s1.y - s0.y);
  if (Math.abs(den) < 1e-9) return null;
  const m = (q0, q1, q2) => [
    (q0 * (s2.y - s1.y) + q1 * (s0.y - s2.y) + q2 * (s1.y - s0.y)) / den,
    (q0 * (s1.x - s2.x) + q1 * (s2.x - s0.x) + q2 * (s0.x - s1.x)) / den,
    (q0 * (s1.x * s2.y - s2.x * s1.y) + q1 * (s2.x * s0.y - s0.x * s2.y)
      + q2 * (s0.x * s1.y - s1.x * s0.y)) / den
  ];
  const [a, c, e] = m(d0.x, d1.x, d2.x);
  const [b, d, f] = m(d0.y, d1.y, d2.y);
  return [a, b, c, d, e, f];
}

/** Smooth, local displacement field from control points (inverse-distance). */
export function makeDisplacer(controls, W, H, cols = 20, rows = 26) {
  const anchors = [];
  for (let i = 0; i <= cols; i++) {
    anchors.push({ sx: (i / cols) * W, sy: 0, dx: (i / cols) * W, dy: 0 });
    anchors.push({ sx: (i / cols) * W, sy: H, dx: (i / cols) * W, dy: H });
  }
  for (let j = 0; j <= rows; j++) {
    anchors.push({ sx: 0, sy: (j / rows) * H, dx: 0, dy: (j / rows) * H });
    anchors.push({ sx: W, sy: (j / rows) * H, dx: W, dy: (j / rows) * H });
  }
  const all = controls.concat(anchors);
  const eps = Math.pow(Math.min(W, H) * 0.45, 2) * 0.02;
  return (x, y) => {
    let wsum = 0, ax = 0, ay = 0;
    for (const c of all) {
      const w = 1 / ((x - c.sx) ** 2 + (y - c.sy) ** 2 + eps);
      wsum += w; ax += w * (c.dx - c.sx); ay += w * (c.dy - c.sy);
    }
    return wsum ? { x: x + ax / wsum, y: y + ay / wsum } : { x, y };
  };
}
