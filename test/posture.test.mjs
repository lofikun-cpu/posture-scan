/* Unit tests for the POSTURE.SCAN analysis engine.
   Landmarks are constructed from a *physical* layout so the same pose can be
   rendered at any aspect ratio — which is how the aspect-correction is tested. */

import {
  LM, sevBucket, bodyRotation, checkQuality, analyseBackContour,
  assessFront, assessSide, scoreOf, footProgression,
  inchScale, hunchIndex, hunchBand
} from '../posture.js';

let pass = 0, fail = 0;
const approx = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;
function check(name, cond, extra = '') {
  if (cond) { pass++; console.log(`  ✓ ${name}`); }
  else { fail++; console.log(`  ✗ ${name} ${extra}`); }
}
function section(t) { console.log(`\n── ${t}`); }

/* ---------- landmark factory ----------
   `phys` is in units where image height = 1, x measured in the same units.
   For an image of aspect A = W/H, normalised x = X_phys / A. */
function buildLms(phys, A, vis = 1) {
  const lms = Array.from({ length: 33 }, () => ({ x: 0.5 / A, y: 0.5, z: 0, visibility: 0.05 }));
  for (const [idx, p] of Object.entries(phys)) {
    lms[idx] = { x: p.x / A, y: p.y, z: 0, visibility: p.v ?? vis };
  }
  return lms;
}
function buildWorld(spec) {
  const w = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 1 }));
  for (const [idx, p] of Object.entries(spec)) w[idx] = { x: p.x, y: p.y ?? 0, z: p.z, visibility: 1 };
  return w;
}

/* Squared-to-camera world hips + an anterior nose (front view). */
const WORLD_FRONT = buildWorld({
  [LM.leftHip]: { x: -0.1, z: 0 }, [LM.rightHip]: { x: 0.1, z: 0 },
  [LM.nose]: { x: 0, z: 0.1 },
  [LM.leftHeel]: { x: -0.08, z: 0 }, [LM.leftFoot]: { x: -0.08, z: 0.15 },
  [LM.rightHeel]: { x: 0.08, z: 0 }, [LM.rightFoot]: { x: 0.08, z: 0.15 }
});
/* Profile world hips (one hip behind the other). */
const WORLD_SIDE = buildWorld({
  [LM.leftHip]: { x: 0, z: -0.1 }, [LM.rightHip]: { x: 0, z: 0.1 },
  [LM.nose]: { x: 0.1, z: 0 }
});

/* ---------- a physically perfect frontal stance ---------- */
function perfectFrontPhys() {
  const cx = 0.28; // horizontal centre in height-units
  return {
    [LM.nose]:          { x: cx,        y: 0.10 },
    [LM.leftEar]:       { x: cx + 0.02, y: 0.095 },
    [LM.rightEar]:      { x: cx - 0.02, y: 0.095 },
    [LM.leftShoulder]:  { x: cx + 0.065, y: 0.20 },
    [LM.rightShoulder]: { x: cx - 0.065, y: 0.20 },
    [LM.leftHip]:       { x: cx + 0.035, y: 0.50 },
    [LM.rightHip]:      { x: cx - 0.035, y: 0.50 },
    [LM.leftKnee]:      { x: cx + 0.0295, y: 0.72 },   // exactly on hip→ankle
    [LM.rightKnee]:     { x: cx - 0.0295, y: 0.72 },
    [LM.leftAnkle]:     { x: cx + 0.025, y: 0.92 },
    [LM.rightAnkle]:    { x: cx - 0.025, y: 0.92 },
    [LM.leftHeel]:      { x: cx + 0.025, y: 0.94 },
    [LM.rightHeel]:     { x: cx - 0.025, y: 0.94 },
    [LM.leftFoot]:      { x: cx + 0.025, y: 0.97 },
    [LM.rightFoot]:     { x: cx - 0.025, y: 0.97 }
  };
}
// knee exactly on the hip→ankle line: t = (0.72-0.50)/(0.92-0.50)
{
  const t = (0.72 - 0.50) / (0.92 - 0.50);
  const kx = 0.035 + (0.025 - 0.035) * t;
  const p = perfectFrontPhys();
  p[LM.leftKnee].x = 0.28 + kx;
  p[LM.rightKnee].x = 0.28 - kx;
  perfectFrontPhys.exact = p;
}

function perfectSidePhys() {
  const cx = 0.28;
  return {
    [LM.nose]:          { x: cx + 0.045, y: 0.10 },  // nose leads → facing +x
    [LM.leftEar]:       { x: cx,         y: 0.095 },
    [LM.rightEar]:      { x: cx,         y: 0.095, v: 0.3 },
    [LM.leftShoulder]:  { x: cx,         y: 0.20 },
    [LM.rightShoulder]: { x: cx,         y: 0.20, v: 0.3 },
    [LM.leftHip]:       { x: cx,         y: 0.50 },
    [LM.rightHip]:      { x: cx,         y: 0.50, v: 0.3 },
    [LM.leftKnee]:      { x: cx,         y: 0.72 },
    [LM.leftAnkle]:     { x: cx,         y: 0.92 }
  };
}

const PORTRAIT = 1080 / 1920;   // 0.5625
const LANDSCAPE = 1920 / 1080;  // 1.7778

/* ============================================================ */
section('Perfect posture scores 100');

{
  const lms = buildLms(perfectFrontPhys.exact, PORTRAIT);
  const m = assessFront(lms, WORLD_FRONT, 1080, 1920);
  const s = scoreOf(m);
  check('front: perfect stance = 100', s === 100, `got ${s}`);
  check('front: every metric OPTIMAL', m.every(x => sevBucket(x.severity) === 0),
        m.filter(x => sevBucket(x.severity) !== 0).map(x => `${x.name}=${x.severity.toFixed(3)}`).join(','));
  check('front: produces 6 checkpoints', m.length === 6, `got ${m.length}: ${m.map(x => x.name)}`);
}
{
  const lms = buildLms(perfectSidePhys(), PORTRAIT);
  // A flat, neutral thoracic contour — chosen to describe a back, not to match
  // whatever the threshold happens to be, so this still means something if the
  // calibration moves.
  const m = assessSide(lms, WORLD_SIDE, 1080, 1920, { kyphosis: 0.035, lordosis: 0.05 });
  const s = scoreOf(m);
  check('side: perfect stance = 100', s === 100, `got ${s}`);
  check('side: every metric OPTIMAL', m.every(x => sevBucket(x.severity) === 0),
        m.filter(x => sevBucket(x.severity) !== 0).map(x => `${x.name}=${x.severity.toFixed(3)}`).join(','));
  check('side: produces 6 checkpoints with contour', m.length === 6, `got ${m.length}`);
}

/* ============================================================ */
section('Aspect-ratio independence (the portrait-photo bug)');

{
  // Same physical pose, tilted shoulders, rendered portrait vs landscape.
  const p = perfectFrontPhys.exact;
  const tilted = JSON.parse(JSON.stringify(p));
  tilted[LM.leftShoulder].y += 0.020;   // drop one shoulder
  tilted[LM.rightShoulder].y -= 0.020;

  const sp = scoreOf(assessFront(buildLms(tilted, PORTRAIT), WORLD_FRONT, 1080, 1920));
  const sl = scoreOf(assessFront(buildLms(tilted, LANDSCAPE), WORLD_FRONT, 1920, 1080));
  check('front: portrait and landscape agree', sp === sl, `portrait=${sp} landscape=${sl}`);

  const q = perfectSidePhys();
  q[LM.leftEar].x += 0.035;             // forward head
  q[LM.nose].x += 0.035;
  const tp = scoreOf(assessSide(buildLms(q, PORTRAIT), WORLD_SIDE, 1080, 1920, null));
  const tl = scoreOf(assessSide(buildLms(q, LANDSCAPE), WORLD_SIDE, 1920, 1080, null));
  check('side: portrait and landscape agree', tp === tl, `portrait=${tp} landscape=${tl}`);
  check('side: forward head actually penalised', tp < 100, `got ${tp}`);
}

/* ============================================================ */
section('Frontal metrics respond correctly');

{
  const base = perfectFrontPhys.exact;
  const withShoulderTilt = (dy) => {
    const p = JSON.parse(JSON.stringify(base));
    p[LM.leftShoulder].y += dy; p[LM.rightShoulder].y -= dy;
    return assessFront(buildLms(p, PORTRAIT), WORLD_FRONT, 1080, 1920)
      .find(m => m.name === 'SHOULDER BALANCE');
  };
  // shoulders span 0.13 height-units; 9° tilt ⇒ Δy = 0.13*tan(9°) = 0.0206 total
  const half = 0.13 * Math.tan(9 * Math.PI / 180) / 2;
  const at9 = withShoulderTilt(half);
  check('shoulder balance: 9° ⇒ severity ≈ 1.0', approx(at9.severity, 1.0, 0.03), `got ${at9.severity.toFixed(3)}`);
  const at0 = withShoulderTilt(0);
  check('shoulder balance: level ⇒ 0', approx(at0.severity, 0, 1e-9), `got ${at0.severity}`);
  check('shoulder balance: monotonic', withShoulderTilt(half / 3).severity < at9.severity);

  // hip drop
  const ph = JSON.parse(JSON.stringify(base));
  ph[LM.leftHip].y += 0.012; ph[LM.rightHip].y -= 0.012;
  const hipM = assessFront(buildLms(ph, PORTRAIT), WORLD_FRONT, 1080, 1920).find(m => m.name === 'PELVIC LEVEL');
  check('pelvic level: drop is detected', hipM.severity > 0.3, `got ${hipM.severity.toFixed(3)}`);

  // knee valgus vs varus — sign must be right
  const pv = JSON.parse(JSON.stringify(base));
  pv[LM.leftKnee].x -= 0.02; pv[LM.rightKnee].x += 0.02;   // both knees toward midline
  const valg = assessFront(buildLms(pv, PORTRAIT), WORLD_FRONT, 1080, 1920).find(m => /KNEE/.test(m.name));
  check('knees inward ⇒ labelled VALGUS', valg.name === 'KNEE VALGUS', `got ${valg.name}`);
  check('knee valgus: severity raised', valg.severity > 0.5, `got ${valg.severity.toFixed(3)}`);

  const pw = JSON.parse(JSON.stringify(base));
  pw[LM.leftKnee].x += 0.02; pw[LM.rightKnee].x -= 0.02;   // both knees outward
  const varus = assessFront(buildLms(pw, PORTRAIT), WORLD_FRONT, 1080, 1920).find(m => /KNEE/.test(m.name));
  check('knees outward ⇒ labelled VARUS', varus.name === 'KNEE VARUS', `got ${varus.name}`);

  // trunk lean
  const pl = JSON.parse(JSON.stringify(base));
  for (const i of [LM.leftShoulder, LM.rightShoulder, LM.nose, LM.leftEar, LM.rightEar]) pl[i].x += 0.05;
  const lean = assessFront(buildLms(pl, PORTRAIT), WORLD_FRONT, 1080, 1920).find(m => m.name === 'TRUNK LEAN');
  check('trunk lean: detected', lean.severity > 0.7, `got ${lean.severity.toFixed(3)}`);
}

/* ============================================================ */
section('Foot progression (forefoot abduction)');

{
  const vis = () => 1;
  const straight = footProgression(WORLD_FRONT, LM.leftHeel, LM.leftFoot, vis);
  check('feet pointing forward ⇒ ~0°', approx(straight, 0, 0.5), `got ${straight?.toFixed(2)}`);

  const wOut = buildWorld({
    [LM.leftHip]: { x: -0.1, z: 0 }, [LM.rightHip]: { x: 0.1, z: 0 },
    [LM.nose]: { x: 0, z: 0.1 },
    [LM.leftHeel]: { x: -0.08, z: 0 }, [LM.leftFoot]: { x: -0.18, z: 0.15 }
  });
  const out = footProgression(wOut, LM.leftHeel, LM.leftFoot, vis);
  check('toe-out ⇒ ~33.7°', approx(out, 33.69, 0.5), `got ${out?.toFixed(2)}`);

  // Whole-body rotation must NOT masquerade as toe-out: rotate hips, nose and
  // feet together by 40° and the measured angle should be unchanged.
  const rot = (p, a) => ({ x: p.x * Math.cos(a) - p.z * Math.sin(a), z: p.x * Math.sin(a) + p.z * Math.cos(a) });
  const a = 40 * Math.PI / 180;
  const wRot = buildWorld({
    [LM.leftHip]: rot({ x: -0.1, z: 0 }, a), [LM.rightHip]: rot({ x: 0.1, z: 0 }, a),
    [LM.nose]: rot({ x: 0, z: 0.1 }, a),
    [LM.leftHeel]: rot({ x: -0.08, z: 0 }, a), [LM.leftFoot]: rot({ x: -0.18, z: 0.15 }, a)
  });
  const rotated = footProgression(wRot, LM.leftHeel, LM.leftFoot, vis);
  check('body rotation does not inflate toe-out', approx(rotated, out, 0.01),
        `upright=${out?.toFixed(2)} rotated=${rotated?.toFixed(2)}`);
}

/* ============================================================ */
section('Sagittal metrics');

{
  const base = perfectSidePhys();
  const fh = JSON.parse(JSON.stringify(base));
  fh[LM.leftEar].x += 0.084;   // 0.28 torso × 0.28 threshold ⇒ severity 1.0
  fh[LM.nose].x += 0.084;
  const m1 = assessSide(buildLms(fh, PORTRAIT), WORLD_SIDE, 1080, 1920, null).find(m => m.name === 'FORWARD HEAD');
  check('forward head: 0.28 of torso ⇒ severity ≈ 1', approx(m1.severity, 1.0, 0.02), `got ${m1.severity.toFixed(3)}`);

  const rs = JSON.parse(JSON.stringify(base));
  for (const i of [LM.leftShoulder, LM.leftEar, LM.nose]) rs[i].x += 0.03;
  const m2 = assessSide(buildLms(rs, PORTRAIT), WORLD_SIDE, 1080, 1920, null)
    .find(m => m.name === 'SHOULDER PROTRACTION');
  check('shoulder protraction: detected', m2.severity > 0.4, `got ${m2.severity.toFixed(3)}`);

  // Posture leaning BACKWARD must not be scored as forward head.
  const back = JSON.parse(JSON.stringify(base));
  back[LM.leftEar].x -= 0.05;
  const m3 = assessSide(buildLms(back, PORTRAIT), WORLD_SIDE, 1080, 1920, null).find(m => m.name === 'FORWARD HEAD');
  check('ear behind shoulder ⇒ forward head stays 0', m3.severity === 0, `got ${m3.severity}`);

  // Facing the other way must give identical results (mirror invariance).
  const mirrored = {};
  for (const [k, v] of Object.entries(fh)) mirrored[k] = { ...v, x: 0.56 - v.x };
  const worldMirror = buildWorld({
    [LM.leftHip]: { x: 0, z: -0.1 }, [LM.rightHip]: { x: 0, z: 0.1 }, [LM.nose]: { x: -0.1, z: 0 }
  });
  const m4 = assessSide(buildLms(mirrored, PORTRAIT), worldMirror, 1080, 1920, null)
    .find(m => m.name === 'FORWARD HEAD');
  check('mirror invariance: same forward-head severity', approx(m4.severity, m1.severity, 0.02),
        `orig=${m1.severity.toFixed(3)} mirrored=${m4.severity.toFixed(3)}`);

  // sway back
  const sw = JSON.parse(JSON.stringify(base));
  for (const i of [LM.leftHip, LM.leftShoulder, LM.leftEar, LM.nose]) sw[i].x += 0.05;
  sw[LM.leftShoulder].x -= 0.05; sw[LM.leftEar].x -= 0.05; sw[LM.nose].x -= 0.05;
  const m5 = assessSide(buildLms(sw, PORTRAIT), WORLD_SIDE, 1080, 1920, null)
    .find(m => m.name === 'PELVIS OVER BASE');
  check('sway back: hips ahead of ankles detected', m5.severity > 0.8, `got ${m5.severity.toFixed(3)}`);
}

/* ============================================================ */
section('Quality gating (3D rotation)');

{
  check('bodyRotation: square hips ⇒ 0°', approx(bodyRotation(WORLD_FRONT), 0, 0.01));
  check('bodyRotation: profile hips ⇒ 90°', approx(bodyRotation(WORLD_SIDE), 90, 0.01));

  const lmsF = buildLms(perfectFrontPhys.exact, PORTRAIT);
  check('front photo, square ⇒ accepted', checkQuality(lmsF, WORLD_FRONT, 'front').ok);
  check('front photo, profile ⇒ rejected', !checkQuality(lmsF, WORLD_SIDE, 'front').ok);

  const w45 = buildWorld({
    [LM.leftHip]: { x: -0.0707, z: -0.0707 }, [LM.rightHip]: { x: 0.0707, z: 0.0707 },
    [LM.nose]: { x: 0, z: 0.1 }
  });
  const r45 = checkQuality(lmsF, w45, 'front');
  check('front photo, 45° turned ⇒ rejected', !r45.ok, JSON.stringify(r45));
  check('rejection message names the angle', /45°/.test(r45.msg || ''), r45.msg);

  check('side photo, profile ⇒ accepted', checkQuality(lmsF, WORLD_SIDE, 'side').ok);
  check('side photo, square ⇒ rejected', !checkQuality(lmsF, WORLD_FRONT, 'side').ok);

  // missing torso
  const noTorso = buildLms(perfectFrontPhys.exact, PORTRAIT);
  noTorso[LM.leftHip].visibility = 0.1;
  check('missing torso ⇒ rejected', !checkQuality(noTorso, WORLD_FRONT, 'front').ok);

  // missing feet ⇒ accepted but warned
  const noFeet = buildLms(perfectFrontPhys.exact, PORTRAIT);
  noFeet[LM.leftAnkle].visibility = 0.1; noFeet[LM.rightAnkle].visibility = 0.1;
  const nf = checkQuality(noFeet, WORLD_FRONT, 'front');
  check('missing feet ⇒ accepted with warning', nf.ok && !!nf.warn, JSON.stringify(nf));
}

/* ============================================================ */
section('Back-contour curvature from the segmentation mask');

{
  const mw = 100, mh = 200;
  const data = new Uint8Array(mw * mh);
  const shY = 50, hipY = 150;           // matches lms below
  const edgeAt = (y) => {
    const frac = (y - shY) / (hipY - shY);
    let e = 40;                          // straight chord baseline
    if (frac > 0.2 && frac < 0.6) e = 40 - 10 * Math.sin((frac - 0.2) / 0.4 * Math.PI);  // kyphotic bulge
    if (frac > 0.65) e = 40 + 8 * Math.sin((frac - 0.65) / 0.35 * Math.PI);              // lordotic hollow
    return Math.round(e);
  };
  for (let y = 0; y < mh; y++) {
    const e = (y >= shY && y <= hipY) ? edgeAt(y) : 40;
    for (let x = e; x < 90; x++) data[y * mw + x] = 255;
  }
  const lms = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));
  lms[LM.leftShoulder] = { x: 0.5, y: shY / mh, visibility: 1 };
  lms[LM.rightShoulder] = { x: 0.5, y: shY / mh, visibility: 1 };
  lms[LM.leftHip] = { x: 0.5, y: hipY / mh, visibility: 1 };
  lms[LM.rightHip] = { x: 0.5, y: hipY / mh, visibility: 1 };

  const c = analyseBackContour(data, mw, mh, lms, true);
  check('contour: returns a result', !!c, String(c));
  check('contour: kyphosis ≈ 0.10', c && approx(c.kyphosis, 0.10, 0.012), `got ${c && c.kyphosis.toFixed(4)}`);
  check('contour: lordosis ≈ 0.08', c && approx(c.lordosis, 0.08, 0.012), `got ${c && c.lordosis.toFixed(4)}`);

  // A perfectly flat back should report ~no curvature.
  const flat = new Uint8Array(mw * mh);
  for (let y = 0; y < mh; y++) for (let x = 40; x < 90; x++) flat[y * mw + x] = 255;
  const cf = analyseBackContour(flat, mw, mh, lms, true);
  check('contour: flat back ⇒ ~0 curvature', cf && cf.kyphosis < 0.005 && cf.lordosis < 0.005, JSON.stringify(cf));

  // Facing the other way must mirror cleanly.
  const mir = new Uint8Array(mw * mh);
  for (let y = 0; y < mh; y++) {
    const e = (y >= shY && y <= hipY) ? edgeAt(y) : 40;
    for (let x = mw - 90; x < mw - e; x++) mir[y * mw + x] = 255;
  }
  const cm = analyseBackContour(mir, mw, mh, lms, false);
  check('contour: mirror invariance', cm && approx(cm.kyphosis, c.kyphosis, 0.02) && approx(cm.lordosis, c.lordosis, 0.02),
        JSON.stringify(cm));

  // Empty mask must not throw or invent numbers.
  const empty = new Uint8Array(mw * mh);
  check('contour: empty mask ⇒ null', analyseBackContour(empty, mw, mh, lms, true) === null);

  // Interference (outstretched arm, bag, chair) must be rejected, not scored.
  const armOut = new Uint8Array(mw * mh);
  for (let y = 0; y < mh; y++) {
    const e = (y >= shY && y <= shY + 25) ? 2 : 40;  // arm juts far out up top
    for (let x = e; x < 90; x++) armOut[y * mw + x] = 255;
  }
  check('contour: arm-contaminated silhouette ⇒ null',
        analyseBackContour(armOut, mw, mh, lms, true) === null,
        JSON.stringify(analyseBackContour(armOut, mw, mh, lms, true)));

  // Single-row speckle must not become a spike in the result.
  const speckle = new Uint8Array(mw * mh);
  for (let y = 0; y < mh; y++) {
    const e = (y >= shY && y <= hipY) ? edgeAt(y) : 40;
    for (let x = e; x < 90; x++) speckle[y * mw + x] = 255;
  }
  speckle[(shY + 30) * mw + 5] = 255;   // one stray pixel far out
  const cs = analyseBackContour(speckle, mw, mh, lms, true);
  check('contour: single stray pixel is filtered out',
        cs && approx(cs.kyphosis, c.kyphosis, 0.015), `clean=${c.kyphosis.toFixed(4)} speckled=${cs && cs.kyphosis.toFixed(4)}`);
}

/* ============================================================ */
/* ============================================================ */
section('Real-world measurements');

{
  /* The default world fixtures leave the shoulders at the origin, which is not
     a body — inchScale is supposed to refuse that rather than invent a number.
     This one is a plausible torso: 0.35 m from shoulder centre to hip centre. */
  const WORLD_SCALED = buildWorld({
    [LM.leftShoulder]: { x: -0.18, y: -0.25, z: 0 },
    [LM.rightShoulder]: { x: 0.18, y: -0.25, z: 0 },
    [LM.leftHip]: { x: -0.1, y: 0.1, z: 0 }, [LM.rightHip]: { x: 0.1, y: 0.1, z: 0 },
    [LM.nose]: { x: 0, y: -0.45, z: 0.1 },
    [LM.leftHeel]: { x: -0.08, z: 0 }, [LM.leftFoot]: { x: -0.08, z: 0.15 },
    [LM.rightHeel]: { x: 0.08, z: 0 }, [LM.rightFoot]: { x: 0.08, z: 0.15 }
  });
  const lms = buildLms(perfectFrontPhys.exact, PORTRAIT);

  const scale = inchScale(lms, WORLD_SCALED, 1080, 1920);
  // image torso is 0.30 height-units; 0.35 m is 13.78 in
  check('scale: converts image units to inches', approx(scale, 13.7795 / 0.30, 0.01),
        `got ${scale}`);
  check('scale: refuses a world fit with no torso',
        inchScale(lms, WORLD_FRONT, 1080, 1920) === null);
  check('scale: refuses a missing world', inchScale(lms, null, 1080, 1920) === null);

  // A shoulder raised by 0.02 height-units should read as 0.02 * scale inches.
  const tilt = JSON.parse(JSON.stringify(perfectFrontPhys.exact));
  tilt[LM.leftShoulder].y -= 0.02;
  const m = assessFront(buildLms(tilt, PORTRAIT), WORLD_SCALED, 1080, 1920);
  const sb = m.find(x => x.name === 'SHOULDER BALANCE');
  const expected = (0.02 * scale).toFixed(1);
  check('measurement: shoulder difference reported in inches',
        sb.detail === `${expected} in uneven`, `${sb.detail} (expected ${expected} in uneven)`);
  check('measurement: every frontal checkpoint carries a figure',
        m.every(x => typeof x.detail === 'string' && x.detail.length > 2),
        JSON.stringify(m.map(x => [x.name, x.detail])));

  // Without a usable scale it must still say something, just not in inches.
  const noScale = assessFront(buildLms(tilt, PORTRAIT), WORLD_FRONT, 1080, 1920);
  check('measurement: falls back to angles when scale is unavailable',
        noScale.every(x => x.detail && !x.detail.includes(' in ')),
        JSON.stringify(noScale.map(x => x.detail)));

  const side = assessSide(buildLms(perfectSidePhys(), PORTRAIT), WORLD_SCALED, 1080, 1920,
                          { kyphosis: 0.14, lordosis: 0.05 });
  check('measurement: every sagittal checkpoint carries a figure',
        side.every(x => typeof x.detail === 'string' && x.detail.length > 2),
        JSON.stringify(side.map(x => [x.name, x.detail])));
  const tc = side.find(x => x.name === 'THORACIC CURVE');
  check('measurement: upper-back curve given as a depth in inches',
        /^\d+\.\d in of upper-back curve$/.test(tc.detail), tc.detail);
}

/* ============================================================ */
section('Hunch index');

{
  const flat = assessSide(buildLms(perfectSidePhys(), PORTRAIT), WORLD_SIDE, 1080, 1920,
                          { kyphosis: 0.035, lordosis: 0.05 });
  check('hunch: an upright side view reads 0%', hunchIndex(flat) === 0,
        String(hunchIndex(flat)));

  const rolled = JSON.parse(JSON.stringify(perfectSidePhys()));
  rolled[LM.leftEar].x += 0.10; rolled[LM.nose].x += 0.10;   // head forward
  rolled[LM.leftShoulder].x += 0.07;                          // shoulders forward
  const hi = hunchIndex(assessSide(buildLms(rolled, PORTRAIT), WORLD_SIDE, 1080, 1920,
                                   { kyphosis: 0.16, lordosis: 0.05 }));
  check('hunch: a rounded side view reads high', hi > 70, String(hi));

  check('hunch: rises with the roll', hi > hunchIndex(flat));
  check('hunch: null without a sagittal view', hunchIndex([]) === null);
  check('hunch: ignores checkpoints unrelated to the pattern',
        hunchIndex([{ name: 'PLUMB LINE', severity: 1, weight: 1 }]) === null);
  check('hunch: stays within 0-100', hi >= 0 && hi <= 100, String(hi));
  check('hunch band: reads as a risk level at both ends',
        hunchBand(5).label === 'LOW RISK' && hunchBand(95).label === 'SEVERE',
        `${hunchBand(5).label} / ${hunchBand(95).label}`);
  check('hunch band: rises monotonically through the range',
        ['LOW RISK', 'EARLY WARNING', 'ON THE WAY', 'HIGH RISK', 'SEVERE']
          .every((l, i) => hunchBand([5, 25, 45, 70, 95][i]).label === l),
        [5, 25, 45, 70, 95].map(v => hunchBand(v).label).join(' → '));
}

/* ============================================================ */
section('Calibration against real bodies');

/* The failure this exists to prevent: a person with a visible hunch scored
   62/100 and a 39% risk figure, because every threshold was set wide enough to
   be "fair" and twelve checkpoints averaged the bad ones away.

   Poses below are built in CENTIMETRES and converted, so they can be argued
   with. The fixture torso (shoulder 0.20 → hip 0.50) is 0.30 image units; call
   a real torso 48 cm and 1 cm is 0.00625 units. Shoulders are 38 cm across,
   hips 19 cm — adult averages — because every angle-based metric depends on
   that baseline and a narrow fixture silently changes what an angle means. */
{
  const CM = 0.30 / 48;
  const W_SCALED = buildWorld({
    [LM.leftShoulder]: { x: -0.19, y: -0.25, z: 0 },
    [LM.rightShoulder]: { x: 0.19, y: -0.25, z: 0 },
    [LM.leftHip]: { x: -0.095, y: 0.1, z: 0 }, [LM.rightHip]: { x: 0.095, y: 0.1, z: 0 },
    [LM.nose]: { x: 0, y: -0.45, z: 0.1 }
  });

  const frontPose = ({ shoulder, hip, head, knee, toeOut }) => {
    const cx = 0.28;
    const t = toeOut * Math.PI / 180;
    const phys = {
      [LM.nose]:          { x: cx + head * CM, y: 0.10 },
      [LM.leftEar]:       { x: cx + 0.02, y: 0.095 },
      [LM.rightEar]:      { x: cx - 0.02, y: 0.095 },
      [LM.leftShoulder]:  { x: cx + 19 * CM, y: 0.20 - shoulder * CM },
      [LM.rightShoulder]: { x: cx - 19 * CM, y: 0.20 },
      [LM.leftHip]:       { x: cx + 9.5 * CM, y: 0.50 - hip * CM },
      [LM.rightHip]:      { x: cx - 9.5 * CM, y: 0.50 },
      [LM.leftKnee]:      { x: cx + 8 * CM + knee * CM, y: 0.72 },
      [LM.rightKnee]:     { x: cx - 8 * CM - knee * CM, y: 0.72 },
      [LM.leftAnkle]:     { x: cx + 6.8 * CM, y: 0.92 },
      [LM.rightAnkle]:    { x: cx - 6.8 * CM, y: 0.92 },
      [LM.leftHeel]:      { x: cx + 6.8 * CM, y: 0.94 },
      [LM.rightHeel]:     { x: cx - 6.8 * CM, y: 0.94 },
      [LM.leftFoot]:      { x: cx + 6.8 * CM, y: 0.97 },
      [LM.rightFoot]:     { x: cx - 6.8 * CM, y: 0.97 }
    };
    const w = buildWorld({
      [LM.leftShoulder]: { x: -0.19, y: -0.25, z: 0 },
      [LM.rightShoulder]: { x: 0.19, y: -0.25, z: 0 },
      [LM.leftHip]: { x: -0.095, y: 0.1, z: 0 }, [LM.rightHip]: { x: 0.095, y: 0.1, z: 0 },
      [LM.nose]: { x: 0, y: -0.45, z: 0.1 },
      [LM.leftHeel]: { x: -0.08, z: 0 },
      [LM.leftFoot]: { x: -0.08 - 0.15 * Math.sin(t), z: 0.15 * Math.cos(t) },
      [LM.rightHeel]: { x: 0.08, z: 0 },
      [LM.rightFoot]: { x: 0.08 + 0.15 * Math.sin(t), z: 0.15 * Math.cos(t) }
    });
    return assessFront(buildLms(phys, PORTRAIT), w, 1080, 1920);
  };

  /* `ear` is ahead of the SHOULDER, `shoulder` ahead of the hip, `sway` ahead
     of the ankle — each measured from the part above it, which is what the
     clinical terms mean. Getting this wrong makes a forward head look mild. */
  const sidePose = ({ ear, shoulder, sway, kyphosis }) => {
    const cx = 0.28;
    const shX = cx + (sway + shoulder) * CM;
    const earX = cx + (sway + shoulder + ear) * CM;
    const phys = {
      [LM.nose]:          { x: earX + 0.045, y: 0.10 },
      [LM.leftEar]:       { x: earX, y: 0.095 },
      [LM.rightEar]:      { x: earX, y: 0.095, v: 0.3 },
      [LM.leftShoulder]:  { x: shX, y: 0.20 },
      [LM.rightShoulder]: { x: shX, y: 0.20, v: 0.3 },
      [LM.leftHip]:       { x: cx + sway * CM, y: 0.50 },
      [LM.rightHip]:      { x: cx + sway * CM, y: 0.50, v: 0.3 },
      [LM.leftKnee]:      { x: cx, y: 0.72 },
      [LM.leftAnkle]:     { x: cx, y: 0.92 }
    };
    return assessSide(buildLms(phys, PORTRAIT), WORLD_SIDE, 1080, 1920,
                      { kyphosis, lordosis: 0.05 });
  };

  const CASES = [
    { name: 'textbook',   lo: 92, hi: 100, maxHunch: 12,
      f: { shoulder: 0, hip: 0, head: 0, knee: 0, toeOut: 8 },
      s: { ear: 0.5, shoulder: 0.5, sway: 0, kyphosis: 0.035 } },
    { name: 'minor drift', lo: 80, hi: 96, maxHunch: 30,
      f: { shoulder: 0.5, hip: 0.4, head: 0.6, knee: 0.4, toeOut: 12 },
      s: { ear: 2, shoulder: 1.5, sway: 1, kyphosis: 0.05 } },
    { name: 'desk worker', lo: 40, hi: 70, minHunch: 35,
      f: { shoulder: 1.2, hip: 0.8, head: 1.5, knee: 1, toeOut: 16 },
      s: { ear: 4, shoulder: 3, sway: 2.5, kyphosis: 0.07 } },
    // The one that started this: plainly bad posture, previously scored 62.
    { name: 'visibly bad', lo: 5, hi: 32, minHunch: 80,
      f: { shoulder: 2.5, hip: 1.5, head: 3, knee: 2, toeOut: 20 },
      s: { ear: 7, shoulder: 6, sway: 4, kyphosis: 0.10 } },
    { name: 'severe', lo: 3, hi: 18, minHunch: 90,
      f: { shoulder: 4, hip: 2.5, head: 5, knee: 3.5, toeOut: 26 },
      s: { ear: 11, shoulder: 9, sway: 7, kyphosis: 0.15 } }
  ];

  const scores = [];
  for (const c of CASES) {
    const fm = frontPose(c.f), sm = sidePose(c.s);
    const total = scoreOf(fm.concat(sm));
    const hunch = hunchIndex(sm);
    scores.push(total);
    console.log(`     ${c.name.padEnd(12)} ${String(total).padStart(3)}/100  hunch ${String(hunch).padStart(3)}%`);
    check(`${c.name}: scores ${c.lo}-${c.hi}`, total >= c.lo && total <= c.hi, `got ${total}`);
    if (c.maxHunch !== undefined) {
      check(`${c.name}: risk stays under ${c.maxHunch}%`, hunch <= c.maxHunch, `got ${hunch}`);
    }
    if (c.minHunch !== undefined) {
      check(`${c.name}: risk reads at least ${c.minHunch}%`, hunch >= c.minHunch, `got ${hunch}`);
    }
  }
  check('worse posture always scores lower',
        scores.every((v, i) => i === 0 || v < scores[i - 1]), scores.join(' > '));
  check('a visible hunch cannot pass as respectable', scores[3] < 40, `got ${scores[3]}`);
}

/* ============================================================ */
section('Scoring behaviour');

{
  check('empty metric list ⇒ 0', scoreOf([]) === 0);
  const allBad = [{ severity: 1, weight: 1 }, { severity: 1, weight: 2 }];
  check('all-severe ⇒ floor of 8 or below-mid', scoreOf(allBad) <= 10, `got ${scoreOf(allBad)}`);
  const allGood = [{ severity: 0, weight: 1 }, { severity: 0, weight: 2 }];
  check('all-optimal ⇒ 100', scoreOf(allGood) === 100, `got ${scoreOf(allGood)}`);
  const mixed = scoreOf([{ severity: 0.5, weight: 1 }]);
  check('mid severity lands in the shareable band (40-70)', mixed > 40 && mixed < 70, `got ${mixed}`);
  check('severity buckets', sevBucket(0) === 0 && sevBucket(0.3) === 1 && sevBucket(0.5) === 2 && sevBucket(0.9) === 3);
}

console.log(`\n${'═'.repeat(46)}\n  ${pass} passed, ${fail} failed\n${'═'.repeat(46)}`);
process.exit(fail ? 1 : 0);
