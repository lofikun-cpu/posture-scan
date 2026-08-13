/* Reproduce the iOS case: force the context to start suspended so resume()
   resolves after the tap, and confirm cues are held then released. */
const { chromium } = require('playwright-core');
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium',
    args: ['--use-gl=swiftshader','--enable-unsafe-swiftshader'] });  // no autoplay flag
  const p = await b.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2,
    userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
    isMobile: true, hasTouch: true });
  p.on('pageerror', e => console.log('PAGEERROR:', e.message));
  // delay resume() resolution to mimic iOS
  // Report 'suspended' until resume() resolves, exactly as iOS behaves.
  await p.addInitScript(() => {
    const OrigAC = window.AudioContext || window.webkitAudioContext;
    window.AudioContext = function () {
      const ctx = new OrigAC();
      let fake = 'suspended';
      Object.defineProperty(ctx, 'state', { get: () => fake, configurable: true });
      const orig = ctx.resume.bind(ctx);
      ctx.resume = () => new Promise(r =>
        setTimeout(() => orig().then(() => { fake = 'running'; r(); }), 300));
      return ctx;
    };
    window.webkitAudioContext = window.AudioContext;
  });
  await p.goto('http://localhost:8899/index.html', { waitUntil: 'domcontentloaded' });
  await p.waitForTimeout(700);
  await p.click('#btn-begin');
  await p.waitForTimeout(60);
  const during = await p.evaluate(() => window.__scan.audio());
  console.log(`  just after tap : ctx=${during.ctx}  queued=${during.queued}`);
  await p.waitForTimeout(900);
  const after = await p.evaluate(() => window.__scan.audio());
  console.log(`  after resume   : ctx=${after.ctx}  queued=${after.queued}  energy=${after.energy.toFixed(3)}`);
  console.log(after.ctx === 'running' && after.queued === 0
    ? '  ✓ cues were held and released' : '  ✗ cues lost');
  await b.close();
})().catch(e => { console.error(e.message); process.exit(1); });
