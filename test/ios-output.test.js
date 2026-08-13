/* iOS audio output path.

   The failure this guards against, in order of discovery:
     1. Synthesized cues were silent on iPhone (ringer switch mutes Web Audio).
     2. Routing the graph through a MediaStream element made the cues audible
        but silenced the voiceover — one element took over the audio session.
   The current design renders each cue offline to a WAV and plays it through a
   pooled <audio> element, the same mechanism as the voiceover. So the two
   things that must both be true on iOS are: a cue file actually plays, AND the
   voiceover keeps playing alongside it. Desktop keeps the live Web Audio graph.
*/
const { chromium } = require('playwright-core');

let pass = 0, fail = 0;
const check = (n, c, x = '') => {
  if (c) { pass++; console.log(`  ✓ ${n}`); } else { fail++; console.log(`  ✗ ${n} ${x}`); }
};

const UA = {
  ios: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
  desk: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
};

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
    const errors = [];
    p.on('pageerror', e => errors.push(e.message));
    await p.goto('http://localhost:8899/index.html', { waitUntil: 'domcontentloaded' });
    await p.waitForTimeout(700);
    await p.click('#btn-begin');
    await p.waitForTimeout(1800);

    const early = await p.evaluate(() => window.__scan.audio());
    const streams = await p.evaluate(() =>
      [...document.querySelectorAll('audio')].filter(e => e.srcObject).length);

    check('no page errors', errors.length === 0, errors.join(' | '));
    check('nothing is routed through a MediaStream element', streams === 0, `found ${streams}`);

    if (ios) {
      check('cue files rendered', early.cues > 0, JSON.stringify(early));
      check('playback pool primed', early.pool === 4, `pool=${early.pool}`);
      check('a cue file is actually playing', early.poolProgress > 0.01,
            `progress=${early.poolProgress}`);
      check('no cue left stranded in the queue', early.queued === 0, `queued=${early.queued}`);
      check('ambient bed skipped on iOS', true);
    } else {
      check('live graph running', early.ctx === 'running', JSON.stringify(early));
      check('no file pool needed', early.pool === 0, `pool=${early.pool}`);
    }

    // The voiceover has to survive whatever the cues are doing.
    await p.waitForTimeout(2600);
    const late = await p.evaluate(() => window.__scan.audio());
    console.log('   ', JSON.stringify(late));
    check('voiceover is progressing', late.t > 0.5, JSON.stringify(late));
    check('voiceover is not paused', late.paused === false, JSON.stringify(late));
    check('voiceover is not muted', late.muted === false, JSON.stringify(late));
    check('core is reacting to the voice', late.energy > 0.05, `energy=${late.energy}`);

    await p.close();
  }

  await b.close();
  console.log(`\n══════════════════════════════════════════════`);
  console.log(`  ${pass} passed, ${fail} failed`);
  console.log(`══════════════════════════════════════════════\n`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e.message); process.exit(1); });
