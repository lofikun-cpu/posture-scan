/* Does the core actually move WITH the voice?

   "Moving" was never enough — a sine wave moves. The iOS fallback used to be
   exactly that: speech-shaped motion with no relationship to what KINA was
   saying, which reads on the device as a ring vibrating out of time. So this
   measures correlation against the recording's own loudness curve, not just
   that the number changes.
*/
const { chromium } = require('playwright-core');

let pass = 0, fail = 0;
const check = (n, c, x = '') => {
  if (c) { pass++; console.log(`  ✓ ${n}`); } else { fail++; console.log(`  ✗ ${n} ${x}`); }
};

/** How the bar ring reads: how much neighbouring bins differ, and how far the
    values spread. A smooth ramp scores near zero on the first — which is what
    an interpolation between three band values produced, and why the ring
    expanded as one even annulus on iOS instead of rippling. */
function specStats(frames) {
  const per = frames.map(a => {
    let d = 0;
    for (let i = 1; i < a.length; i++) d += Math.abs(a[i] - a[i - 1]);
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    const sd = Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length);
    return { rough: d / (a.length - 1), sd, mean, bins: a.length };
  });
  const avg = k => per.reduce((s, v) => s + v[k], 0) / per.length;
  return { rough: avg('rough'), sd: avg('sd'), mean: avg('mean'), bins: per[0].bins };
}

/** Pearson correlation — 1 is lockstep, 0 is unrelated. */
function correlate(a, b) {
  const n = Math.min(a.length, b.length);
  const ma = a.reduce((s, v) => s + v, 0) / n;
  const mb = b.reduce((s, v) => s + v, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  return da && db ? num / Math.sqrt(da * db) : 0;
}

const UA = {
  ios: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
  desk: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
};

const seen = {};

(async () => {
  const b = await chromium.launch({
    executablePath: '/opt/pw-browsers/chromium',
    args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader',
           '--autoplay-policy=no-user-gesture-required']
  });

  for (const ios of [true, false]) {
    console.log(`\n── ${ios ? 'iOS' : 'desktop'}`);
    const p = await b.newPage({
      viewport: { width: 390, height: 844 },
      userAgent: ios ? UA.ios : UA.desk,
      isMobile: ios, hasTouch: ios
    });
    p.on('pageerror', e => console.log('PAGEERROR:', e.message));
    await p.goto('http://localhost:8899/index.html', { waitUntil: 'domcontentloaded' });
    await p.waitForTimeout(600);
    await p.click('#btn-begin');
    await p.waitForTimeout(5200);                     // into the voiceover

    // Sample in the page so energy and the recording's own level are read on
    // the same frame — sampling them a round-trip apart would smear the pair.
    const rec = await p.evaluate(async () => {
      const out = [];
      for (let i = 0; i < 40; i++) {
        const a = window.__scan.audio();
        out.push({ e: a.energy, x: a.envAt, t: a.t });
        await new Promise(r => requestAnimationFrame(() => setTimeout(r, 45)));
      }
      return out;
    });
    const specFrames = (await p.evaluate(async () => {
      const out = [];
      for (let f = 0; f < 10; f++) {
        out.push(Array.from(window.__scan.spectrum() || []));
        await new Promise(r => requestAnimationFrame(() => setTimeout(r, 55)));
      }
      return out;
    })).filter(a => a.length);

    const a = await p.evaluate(() => window.__scan.audio());
    const energies = rec.map(r => r.e);
    const spread = Math.max(...energies) - Math.min(...energies);

    console.log(`    vo=${a.t === null ? 'none' : a.t.toFixed(2) + 's'} paused=${a.paused}` +
                ` analyser=${a.analyser} envelope=${a.env}`);
    check('voiceover is playing', a.paused === false, JSON.stringify(a));
    check('core is moving', spread > 0.05, `spread=${spread.toFixed(3)}`);

    if (ios) {
      check('recording was measured up front', a.env === true, JSON.stringify(a));
      const paired = rec.filter(r => typeof r.x === 'number');
      check('envelope available throughout', paired.length > rec.length * 0.8,
            `${paired.length}/${rec.length}`);
      const r = correlate(paired.map(p => p.e), paired.map(p => p.x));
      console.log(`    correlation with the recording: ${r.toFixed(2)}`);
      // The old sine fallback scored near zero here — that is the whole point.
      check('core amplitude tracks the actual audio', r > 0.65, `r=${r.toFixed(2)}`);
    } else {
      check('live analyser in use', a.analyser === true, JSON.stringify(a));
    }

    check('the bars are given a spectrum to read', specFrames.length > 0,
          `${specFrames.length} frames captured`);
    if (specFrames.length) {
      seen[ios ? 'ios' : 'desk'] = specStats(specFrames);
      const st = seen[ios ? 'ios' : 'desk'];
      console.log(`    spectrum: ${st.bins} bins, bin-to-bin ${st.rough.toFixed(1)},` +
                  ` spread ${st.sd.toFixed(1)}, mean ${st.mean.toFixed(1)}`);
    }

    await p.close();
  }

  // The complaint that prompted this: the ring looked different on the phone.
  // It was — one platform was drawing a real spectrum and the other a ramp.
  console.log('\n── iOS against desktop');
  if (seen.ios && seen.desk) {
    check('same number of bins', seen.ios.bins === seen.desk.bins,
          `${seen.ios.bins} vs ${seen.desk.bins}`);
    check('bars vary bin to bin, not as a smooth ramp',
          seen.ios.rough > seen.desk.rough * 0.6,
          `iOS ${seen.ios.rough.toFixed(1)} vs desktop ${seen.desk.rough.toFixed(1)}`);
    check('not jagged either — within half again of desktop',
          seen.ios.rough < seen.desk.rough * 1.6,
          `iOS ${seen.ios.rough.toFixed(1)} vs desktop ${seen.desk.rough.toFixed(1)}`);
    check('sits at a comparable level, so the bars do not clip flat',
          seen.ios.mean < seen.desk.mean * 1.5,
          `iOS ${seen.ios.mean.toFixed(1)} vs desktop ${seen.desk.mean.toFixed(1)}`);
  } else {
    check('both platforms reported a spectrum', false, JSON.stringify(seen));
  }

  await b.close();
  console.log(`\n${'═'.repeat(46)}\n  ${pass} passed, ${fail} failed\n${'═'.repeat(46)}\n`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e.message); process.exit(1); });
