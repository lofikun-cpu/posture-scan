/* Two things the user could hear that they should not.

   1. The live-camera timer. It used to run 3-2-1 in under three seconds, which
      is not enough to put the phone down and step back into a normal stance —
      and the numbers were read aloud by a second voice. The length also lives
      in two places, the constant and the button's label, so the label is
      checked against what the countdown actually does.
   2. That second voice. Every line without a recording used to fall back to the
      browser's built-in speech synthesis: a different speaker, mid-flow,
      announcing "frontal image acquired". This asserts nothing ever reaches
      speechSynthesis.speak, on either build.
*/
const { chromium } = require('playwright-core');

let pass = 0, fail = 0;
const check = (n, c, x = '') => {
  if (c) { pass++; console.log(`  ✓ ${n}`); } else { fail++; console.log(`  ✗ ${n} ${x}`); }
};

(async () => {
  const b = await chromium.launch({
    executablePath: '/opt/pw-browsers/chromium',
    args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader',
           '--autoplay-policy=no-user-gesture-required',
           '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']
  });

  for (const build of ['index', 'brand']) {
    console.log(`\n── ${build}.html`);
    const p = await b.newPage({ viewport: { width: 390, height: 844 } });
    // Record every utterance the page tries to speak, before anything runs.
    await p.addInitScript(() => {
      window.__spoken = [];
      const orig = window.speechSynthesis && window.speechSynthesis.speak;
      if (orig) {
        window.speechSynthesis.speak = function (u) {
          window.__spoken.push(u && u.text);
          return orig.apply(this, arguments);
        };
      }
    });
    await p.goto(`http://localhost:8899/${build}.html`, { waitUntil: 'domcontentloaded' });
    await p.waitForTimeout(700);

    await p.click('#btn-begin');
    await p.waitForFunction(() => { const a = window.__scan.audio(); return a.t !== null && a.t > 1; },
                            { timeout: 20000 });
    await p.click('#btn-skip-vo');
    await p.waitForTimeout(400);
    await p.click('#btn-start');
    await p.waitForSelector('#panel-capture.active', { timeout: 5000 });
    await p.waitForTimeout(600);

    // --- the camera timer
    await p.click('#btn-camera');
    await p.waitForTimeout(1200);
    const camUp = await p.evaluate(() =>
      !document.getElementById('btn-snap').classList.contains('hidden'));
    check('camera opens', camUp);

    if (camUp) {
      const label = await p.textContent('#btn-snap');
      const promised = Number((label.match(/(\d+)\s*s/) || [])[1]);
      const t0 = Date.now();
      await p.click('#btn-snap');
      // Watch the countdown element rather than trusting a constant.
      const seen = await p.evaluate(async () => {
        const cd = document.getElementById('countdown');
        const vals = [];
        for (let i = 0; i < 100; i++) {
          if (getComputedStyle(cd).display === 'none' && vals.length) break;
          const t = cd.textContent.trim();
          if (t && vals[vals.length - 1] !== t) vals.push(t);
          await new Promise(r => setTimeout(r, 100));
        }
        return vals;
      });
      const elapsed = (Date.now() - t0) / 1000;
      console.log(`     button says ${promised}s · counted ${seen.join(',')} over ${elapsed.toFixed(1)}s`);
      check('the button promises the length the timer runs',
            Number(seen[0]) === promised, `label ${promised}s, counted from ${seen[0]}`);
      check('timer counts all the way down to 1',
            seen.length === promised && seen[seen.length - 1] === '1', seen.join(','));
      check('timer takes about that long in real time',
            elapsed > promised - 0.7 && elapsed < promised + 2.5,
            `${elapsed.toFixed(1)}s for a ${promised}s timer`);
    }

    await p.waitForTimeout(1500);
    const spoken = await p.evaluate(() => window.__spoken);
    console.log('     speech-synthesis utterances:', JSON.stringify(spoken));
    check('nothing but KINA ever speaks', spoken.length === 0, JSON.stringify(spoken));

    await p.close();
  }

  await b.close();
  console.log(`\n${'═'.repeat(46)}\n  ${pass} passed, ${fail} failed\n${'═'.repeat(46)}\n`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e.message); process.exit(1); });
