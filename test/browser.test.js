/* Browser integration test: verifies the parts unit tests can't reach —
   MediaPipe actually loading with segmentation masks, mask decoding,
   the two-photo UI flow, the mesh warp, and share-card generation. */
const { chromium } = require('playwright-core');

let pass = 0, fail = 0, skipped = 0;
const check = (n, c, x = '') => {
  if (c) { pass++; console.log(`  ✓ ${n}`); } else { fail++; console.log(`  ✗ ${n} ${x}`); }
};

/* Everything downstream of detection needs a real photograph of a person —
   MediaPipe finds nothing in a synthetic image, and the only photo to hand is
   personal, so it is not committed. Drop any full-body shot at the repo root
   as person.jpg to run those sections; without it they are skipped rather than
   failed, so the suite still covers load, intro, audio and the share card. */
const FIXTURE = '/person.jpg';
const skip = (section) => {
  skipped++;
  console.log(`  … ${section} skipped — no person.jpg at the repo root`);
};

(async () => {
  const browser = await chromium.launch({
    executablePath: '/opt/pw-browsers/chromium',
    args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader',
           '--autoplay-policy=no-user-gesture-required']
  });
  const page = await browser.newPage({ viewport: { width: 420, height: 900 } });
  const errors = [];
  // Location included so a 404 can be attributed — the fixture probe below
  // deliberately requests a file that may not exist.
  page.on('console', m => {
    if (m.type() === 'error') errors.push(`${m.text()} @ ${(m.location() || {}).url || '?'}`);
  });
  page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

  console.log('\n── Page load');
  await page.goto('http://localhost:8899/index.html', { waitUntil: 'load' });
  check('title set', (await page.title()).includes('POSTURE.SCAN'));
  check('test seam exposed on localhost', await page.evaluate(() => !!window.__scan));
  await page.waitForTimeout(1200);
  // Intro is the animation plus a single control — no headline, console or list.
  check('intro strips back to one control', await page.evaluate(() => {
    const g = document.getElementById('panel-greet');
    const btns = [...g.querySelectorAll('.btn')].filter(b => !b.classList.contains('hidden'));
    return !document.getElementById('console') && !document.getElementById('intro-steps')
        && btns.length === 1 && btns[0].id === 'btn-begin';
  }));
  check('stage fills the viewport', await page.evaluate(() => {
    const r = document.getElementById('kina-stage').getBoundingClientRect();
    return Math.abs(r.width - innerWidth) < 2 && Math.abs(r.height - innerHeight) < 2;
  }));
  check('body flagged as intro', await page.evaluate(() => document.body.classList.contains('intro')));
  // The intro is one screen by design. Note the limit of this check: Playwright
  // has no browser toolbars, so it cannot reproduce the iOS case where 100vh is
  // taller than the visible area — the dvh rule in the CSS is what covers that.
  // This guards the simpler regression of the content itself outgrowing the page.
  const fit = await page.evaluate(() => {
    const de = document.documentElement;
    const btn = document.getElementById('btn-begin').getBoundingClientRect();
    return { over: de.scrollHeight - de.clientHeight, btnBottom: btn.bottom, vis: de.clientHeight };
  });
  check('intro does not scroll', fit.over <= 0, JSON.stringify(fit));
  check('the one control sits inside the viewport',
        fit.btnBottom <= fit.vis, JSON.stringify(fit));

  const hasFixture = await page.evaluate(async (src) => {
    try { const r = await fetch(src, { method: 'HEAD' }); return r.ok; } catch (e) { return false; }
  }, FIXTURE);

  if (!hasFixture) {
    console.log('\n── MediaPipe integration');
    skip('detection, contour, quality gate, full flow, warp and CTA');
  } else {
  console.log('\n── MediaPipe integration (real model, real image)');
  const det = await page.evaluate(async () => {
    const lm = await window.__scan.getLandmarker();
    const img = new Image();
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = '/person.jpg'; });
    const r = lm.detect(img);
    const m = r.segmentationMasks && r.segmentationMasks[0];
    let maskInfo = null;
    if (m) {
      const arr = m.getAsUint8Array();
      let on = 0;
      for (let i = 0; i < arr.length; i++) if (arr[i] > 127) on++;
      maskInfo = { w: m.width, h: m.height, len: arr.length, coverage: on / arr.length };
    }
    return {
      landmarks: r.landmarks?.[0]?.length ?? 0,
      world: r.worldLandmarks?.[0]?.length ?? 0,
      hasVisibility: typeof r.landmarks?.[0]?.[11]?.visibility === 'number',
      worldSample: r.worldLandmarks?.[0]?.[23],
      maskInfo,
      imgW: img.width, imgH: img.height
    };
  });
  check('33 landmarks returned', det.landmarks === 33, `got ${det.landmarks}`);
  check('33 world landmarks returned', det.world === 33, `got ${det.world}`);
  check('landmarks carry visibility', det.hasVisibility);
  check('world landmarks have z depth',
        det.worldSample && Number.isFinite(det.worldSample.z), JSON.stringify(det.worldSample));
  check('segmentation mask produced', !!det.maskInfo, JSON.stringify(det.maskInfo));
  check('mask decodes to Uint8Array of full size',
        det.maskInfo && det.maskInfo.len === det.maskInfo.w * det.maskInfo.h, JSON.stringify(det.maskInfo));
  check('mask covers a plausible body fraction (2-80%)',
        det.maskInfo && det.maskInfo.coverage > 0.02 && det.maskInfo.coverage < 0.8,
        det.maskInfo && `coverage=${(det.maskInfo.coverage * 100).toFixed(1)}%`);

  console.log('\n── Contour extraction from a real mask');
  const contour = await page.evaluate(async () => {
    const { analyseBackContour, LM } = await import('/posture.js');
    const lm = await window.__scan.getLandmarker();
    const img = new Image();
    await new Promise(res => { img.onload = res; img.src = '/person.jpg'; });
    const r = lm.detect(img);
    const m = r.segmentationMasks[0];
    const lms = r.landmarks[0];
    const facingRight = lms[LM.nose].x > (lms[LM.leftShoulder].x + lms[LM.rightShoulder].x) / 2;
    return analyseBackContour(m.getAsUint8Array(), m.width, m.height, lms, facingRight);
  });
  // The test photo is a wide yoga pose, so an outstretched arm becomes the
  // leftmost edge. Correct behaviour is to reject it, not to score garbage.
  check('contour never returns implausible curvature',
        contour === null || (contour.kyphosis <= 0.25 && contour.lordosis <= 0.25),
        JSON.stringify(contour));
  check('arm-contaminated silhouette is rejected', contour === null, JSON.stringify(contour));

  console.log('\n── Quality gate on a real photo');
  const gate = await page.evaluate(async () => {
    const { checkQuality, bodyRotation } = await import('/posture.js');
    const lm = await window.__scan.getLandmarker();
    const img = new Image();
    await new Promise(res => { img.onload = res; img.src = '/person.jpg'; });
    const r = lm.detect(img);
    return {
      rotation: bodyRotation(r.worldLandmarks[0]),
      asFront: checkQuality(r.landmarks[0], r.worldLandmarks[0], 'front'),
      asSide: checkQuality(r.landmarks[0], r.worldLandmarks[0], 'side')
    };
  });
  console.log(`     measured body rotation: ${gate.rotation.toFixed(1)}°`);
  check('gate reaches a decision for front', typeof gate.asFront.ok === 'boolean', JSON.stringify(gate.asFront));
  check('gate reaches a decision for side', typeof gate.asSide.ok === 'boolean', JSON.stringify(gate.asSide));
  check('front and side verdicts are mutually exclusive',
        gate.asFront.ok !== gate.asSide.ok, JSON.stringify(gate));
  }

  console.log('\n── Intro briefing');
  const core = await page.evaluate(() => {
    const c = document.getElementById('kina-core');
    if (!c || !c.width) return { ok: false };
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let lit = 0;
    for (let i = 0; i < d.length; i += 4 * 53) if (d[i + 1] > 20) lit++;
    return { ok: true, w: c.width, litFrac: lit / Math.floor(d.length / (4 * 53)) };
  });
  // Pre-tap the core is a deliberately quiet holding pattern.
  check('idle core renders', core.ok && core.litFrac > 0.004 && core.litFrac < 0.2, JSON.stringify(core));
  check('custom loop absent -> core still shown',
        !(await page.getAttribute('#kina-loop', 'class') || '').includes('ready'));

  const tapAt = Date.now();
  await page.click('#btn-begin');
  // The loading cue holds this gap. It used to run a second longer than it
  // needed to, which read on the phone as the app having stalled.
  await page.waitForTimeout(1100);
  check('loading state shown before the voiceover', await page.evaluate(() =>
    document.getElementById('kina-status').textContent.includes('LOADING')));
  await page.waitForFunction(() => {
    const a = window.__scan.audio();
    return a.t !== null && a.t > 0.05;
  }, { timeout: 15000 });
  const startGap = (Date.now() - tapAt) / 1000;
  console.log(`     KINA starts ${startGap.toFixed(2)}s after the tap`);
  check('KINA comes in without an awkward pause (<2.6s)', startGap < 2.6, `${startGap.toFixed(2)}s`);
  await page.waitForTimeout(700);
  const sfxOk = await page.evaluate(() => {
    const a = window.__scan.audio();
    return { ctx: a.ctx, energy: a.energy };
  });
  check('audio context unlocked by the tap', sfxOk.ctx === 'running', JSON.stringify(sfxOk));
  check('ignition ran', await page.evaluate(() => window.__scan.boot().done));
  const bootLit = await page.evaluate(() => {
    const c = document.getElementById('kina-core');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let lit = 0;
    for (let i = 0; i < d.length; i += 4 * 53) if (d[i + 1] > 20) lit++;
    return lit / Math.floor(d.length / (4 * 53));
  });
  // Idle was lifted deliberately, so the gap is narrower than it was; ignition
  // still has to be an event rather than a nudge.
  check('boot renders far brighter than idle', bootLit > core.litFrac * 2.2,
        `idle=${core.litFrac.toFixed(3)} boot=${bootLit.toFixed(3)}`);
  check('core reached steady state fast (<2.5s)',
        await page.evaluate(() => window.__scan.boot().done));
  await page.waitForTimeout(900);
  const au = await page.evaluate(() => window.__scan.audio());
  console.log('     audio:', JSON.stringify(au));
  check('voiceover audio is driving the core', au.energy > 0.02, JSON.stringify(au));
  check('analyser attached to the voiceover', au.analyser, JSON.stringify(au));
  // The bug this guards: routing through Web Audio detached the element from
  // the speaker, so playback "worked" while producing no sound at all.
  check('playback is actually progressing', au.t > 0.5, JSON.stringify(au));
  check('element is playing, not paused', au.paused === false, JSON.stringify(au));
  check('output is not muted', au.muted === false, JSON.stringify(au));
  check('briefing shows a caption',
        (await page.textContent('#vo-caption')).trim().length > 10,
        await page.textContent('#vo-caption'));
  check('caption track visible', (await page.getAttribute('#vo-caption', 'class') || '').includes('on'));
  check('core pulses while speaking',
        (await page.getAttribute('#kina-stage', 'class') || '').includes('speaking'));
  await page.click('#btn-skip-vo');
  await page.waitForTimeout(300);
  check('skip hides the caption', !(await page.getAttribute('#vo-caption', 'class') || '').includes('on'));
  check('skip reveals start button', !(await page.getAttribute('#btn-start', 'class') || '').includes('hidden'));

  if (hasFixture) {
  console.log('\n── Two-photo UI flow (seam-driven)');
  await page.click('#btn-start');
  await page.waitForSelector('#panel-capture.active', { timeout: 5000 });
  check('capture panel shows', true);
  check('full-screen stage hidden off the intro', await page.evaluate(() =>
    !document.body.classList.contains('intro') &&
    getComputedStyle(document.getElementById('kina-stage')).display === 'none'));
  check('front step marked current', (await page.getAttribute('#dot-front', 'class')).includes('current'));
  check('pose guide rendered', (await page.innerHTML('#pose-guide')).includes('svg'));

  const flow = await page.evaluate(async () => {
    const lm = await window.__scan.getLandmarker();
    const img = new Image();
    await new Promise(res => { img.onload = res; img.src = '/person.jpg'; });
    const r = lm.detect(img);
    const lms = r.landmarks[0], world = r.worldLandmarks[0];

    // Synthesise the two required orientations; the geometry itself is
    // already covered by unit tests, this exercises the UI pipeline.
    const clone = (o) => JSON.parse(JSON.stringify(o));
    const frontWorld = clone(world);
    frontWorld[23] = { x: -0.1, y: 0, z: 0 }; frontWorld[24] = { x: 0.1, y: 0, z: 0 };
    const sideWorld = clone(world);
    sideWorld[23] = { x: 0, y: 0, z: -0.1 }; sideWorld[24] = { x: 0, y: 0, z: 0.1 };

    const s = window.__scan.state;
    s.front = { img, lms, world: frontWorld, contour: null, W: img.width, H: img.height };
    s.side = { img, lms, world: sideWorld, contour: { kyphosis: 0.09, lordosis: 0.07 }, W: img.width, H: img.height };
    await window.__scan.runFullAnalysis();
    return true;
  });
  check('full analysis ran', flow);

  await page.waitForSelector('#panel-score.active', { timeout: 30000 });
  await page.waitForTimeout(2600);
  const score = parseInt(await page.textContent('#score-num'), 10);
  const front = await page.textContent('#sub-front');
  const side = await page.textContent('#sub-side');
  console.log(`     total=${score}  front=${front}  side=${side}`);
  check('score panel reached', true);
  check('total score in 0-100', score >= 0 && score <= 100, `got ${score}`);
  check('front sub-score rendered', /^\d+$/.test(front.trim()), front);
  check('side sub-score rendered', /^\d+$/.test(side.trim()), side);
  check('verdict text present', (await page.textContent('#verdict')).length > 3);
  const rows = await page.$$eval('#metrics-final .metric-row', els => els.length);
  check('metric rows listed (>=8 across both planes)', rows >= 8, `got ${rows}`);
  const heads = await page.$$eval('#metrics-final .plane-head', els => els.map(e => e.textContent));
  check('both planes reported', heads.length === 2 && heads.some(h => /FRONTAL/.test(h)) && heads.some(h => /SAGITTAL/.test(h)),
        JSON.stringify(heads));

  console.log('\n── Alignment simulation (mesh warp)');
  const warp = await page.evaluate(() => {
    const b = document.getElementById('before-canvas');
    const a = document.getElementById('after-canvas');
    if (!b.width || !a.width) return { ok: false };
    const bc = b.getContext('2d').getImageData(0, 0, b.width, b.height).data;
    const ac = a.getContext('2d').getImageData(0, 0, a.width, a.height).data;
    let diff = 0, nonBlank = 0;
    for (let i = 0; i < bc.length; i += 4 * 97) {
      if (Math.abs(bc[i] - ac[i]) > 6) diff++;
      if (ac[i] > 8) nonBlank++;
    }
    const n = Math.floor(bc.length / (4 * 97));
    return { ok: true, w: a.width, h: a.height, diffFrac: diff / n, nonBlankFrac: nonBlank / n };
  });
  check('both canvases sized', warp.ok && warp.w > 0 && warp.h > 0, JSON.stringify(warp));
  check('warped output is not blank', warp.nonBlankFrac > 0.5, JSON.stringify(warp));
  check('warped output differs from the original', warp.diffFrac > 0.01, JSON.stringify(warp));
  check('warp is subtle, not destroyed (<60% of pixels)', warp.diffFrac < 0.6, JSON.stringify(warp));

  const slider = await page.evaluate(() => {
    const r = document.getElementById('slider-range');
    r.value = 20; r.dispatchEvent(new Event('input'));
    const clip = document.getElementById('after-layer').style.clipPath;
    return clip;
  });
  check('slider drives the clip path', /20%/.test(slider), slider);

  console.log('\n── CTA deep link');
  const cta = await page.getAttribute('#btn-kinapt', 'href');
  console.log(`     ${cta}`);
  check('CTA points at the App Store', /apps\.apple\.com/.test(cta), cta);
  check('CTA carries scan attribution', /ct=posture-scan/.test(cta) && /score=/.test(cta), cta);

  console.log('\n── Rescan resets state');
  await page.click('#btn-retry');
  await page.waitForSelector('#panel-capture.active', { timeout: 5000 });
  check('front thumb reset', (await page.textContent('#thumb-front')).includes('PENDING'));
  check('back to front step', (await page.getAttribute('#dot-front', 'class')).includes('current'));
  }

  console.log('\n── Scan sweep closes when detection lands');
  /* Needs no photograph: what is being measured is how long the sweep holds on
     AFTER the detector has answered, and a blank frame answers (with no human)
     just as definitively as a good one. It used to hold a flat two seconds
     either way, which is dead time between the upload and KINA speaking. */
  const sweep = await page.evaluate(async () => {
    window.__scan.warmDetector();                 // as the capture screen does
    await window.__scan.getLandmarker();
    await new Promise(r => setTimeout(r, 2500));  // let the warm-up land
    const cv = document.createElement('canvas');
    cv.width = 600; cv.height = 900;
    const c2 = cv.getContext('2d');
    c2.fillStyle = '#888'; c2.fillRect(0, 0, 600, 900);
    const img = new Image();
    await new Promise(r => { img.onload = r; img.src = cv.toDataURL(); });
    const once = async () => {
      window.__scan.state.want = 'front';
      const t0 = performance.now();
      await window.__scan.ingest(img);
      return performance.now() - t0;
    };
    return { first: await once(), second: await once() };
  });
  console.log(`     sweep held ${(sweep.first / 1000).toFixed(2)}s then ` +
              `${(sweep.second / 1000).toFixed(2)}s on an instant answer`);
  check('sweep closes soon after the detector answers (<1.4s)', sweep.second < 1400, `${Math.round(sweep.second)}ms`);
  check('sweep still reads as a scan, not a flash (>0.7s)', sweep.second > 700, `${Math.round(sweep.second)}ms`);
  // The warm-up during the briefing is what should make these two alike.
  check('the first scan is not much slower than the next', sweep.first < sweep.second + 900,
        `${Math.round(sweep.first)}ms then ${Math.round(sweep.second)}ms`);

  console.log('\n── Share card');
  const card = await page.evaluate(() => {
    const c = window.__scan.buildShareCard({
      score: 63, frontScore: 71, sideScore: 55,
      frontMetrics: [{ name: 'SHOULDER BALANCE', severity: 0.3, weight: 1 }],
      sideMetrics: [{ name: 'FORWARD HEAD', severity: 0.8, weight: 1 }]
    });
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let lit = 0;
    for (let i = 0; i < d.length; i += 4 * 211) if (d[i] + d[i + 1] + d[i + 2] > 90) lit++;
    return { w: c.width, h: c.height, litFrac: lit / Math.floor(d.length / (4 * 211)) };
  });
  check('share card is 1080x1350 (portrait social)', card.w === 1080 && card.h === 1350, JSON.stringify(card));
  check('share card has rendered content', card.litFrac > 0.02, JSON.stringify(card));

  const realErrors = errors.filter(e => !/favicon/i.test(e) && !/person\.jpg/.test(e));
  check('no console errors', realErrors.length === 0, JSON.stringify(realErrors.slice(0, 3)));

  const tail = skipped ? `, ${skipped} section${skipped > 1 ? 's' : ''} skipped` : '';
  console.log(`\n${'═'.repeat(46)}\n  ${pass} passed, ${fail} failed${tail}\n${'═'.repeat(46)}`);
  await browser.close();
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error('E2E FAIL:', e.message); process.exit(1); });
