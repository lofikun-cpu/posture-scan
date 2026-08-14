/* Captions must be absent when KINA is audible, and present when he is not. */
const { chromium } = require('playwright-core');
let pass = 0, fail = 0;
const check = (n, c, x = '') => {
  if (c) { pass++; console.log(`  ✓ ${n}`); } else { fail++; console.log(`  ✗ ${n} ${x}`); }
};
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium',
    args: ['--use-gl=swiftshader','--enable-unsafe-swiftshader','--autoplay-policy=no-user-gesture-required'] });
  const capState = (p) => p.evaluate(() => {
    const c = document.getElementById('vo-caption');
    return { on: c.classList.contains('on'), text: c.textContent.trim(),
             visible: getComputedStyle(c).display !== 'none' };
  });

  // 1. sound on, audio plays
  {
    const p = await b.newPage({ viewport: { width: 390, height: 844 } });
    await p.goto('http://localhost:8899/index.html', { waitUntil: 'domcontentloaded' });
    await p.waitForTimeout(700);
    await p.click('#btn-begin');
    await p.waitForTimeout(4500);
    const a = await p.evaluate(() => window.__scan.audio());
    const c = await capState(p);
    console.log(`    voice playing at ${a.t && a.t.toFixed(2)}s ·`, JSON.stringify(c));
    check('voice audible → no caption on screen', !c.visible, JSON.stringify(c));
    check('voice is genuinely playing', a.paused === false && a.t > 1, JSON.stringify(a));
    await p.close();
  }

  // 2. sound muted before starting
  {
    const p = await b.newPage({ viewport: { width: 390, height: 844 } });
    await p.goto('http://localhost:8899/index.html', { waitUntil: 'domcontentloaded' });
    await p.waitForTimeout(700);
    await p.click('#mute-btn');                      // SOUND: OFF
    await p.click('#btn-begin');
    await p.waitForTimeout(4500);
    const c = await capState(p);
    console.log('    muted ·', JSON.stringify(c));
    check('sound off → captions carry the briefing', c.visible && c.text.length > 10,
          JSON.stringify(c));
    await p.close();
  }

  // 3. audio that never starts
  {
    const p = await b.newPage({ viewport: { width: 390, height: 844 } });
    await p.addInitScript(() => {
      const orig = HTMLMediaElement.prototype.play;
      HTMLMediaElement.prototype.play = function () {
        if (this.src && /vo-intro/.test(this.src)) return Promise.reject(new Error('blocked'));
        return orig.apply(this, arguments);
      };
    });
    await p.goto('http://localhost:8899/index.html', { waitUntil: 'domcontentloaded' });
    await p.waitForTimeout(700);
    await p.click('#btn-begin');
    await p.waitForTimeout(5200);
    const c = await capState(p);
    console.log('    audio blocked ·', JSON.stringify(c));
    check('audio fails → captions appear as the fallback', c.visible && c.text.length > 10,
          JSON.stringify(c));
    await p.close();
  }

  await b.close();
  console.log(`\n  ${pass} passed, ${fail} failed\n`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e.message); process.exit(1); });
