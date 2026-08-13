/* Deferred resume() on the live Web Audio path.

   AudioContext.resume() is asynchronous, so a cue fired in the same tick as
   the tap finds the context still suspended and used to be dropped — silence
   where the interface should have answered. Cues raised too early are queued
   and released once the context is running; this forces that window open.

   iOS is deliberately NOT the subject any more: it creates no AudioContext at
   all and plays rendered files instead (see ios-output.test.js). This covers
   everywhere the live graph does run — desktop and Android — where an
   autoplay policy can hold the context suspended past the gesture. */
const { chromium } = require('playwright-core');
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium',
    args: ['--use-gl=swiftshader','--enable-unsafe-swiftshader'] });  // no autoplay flag
  const p = await b.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2,
    userAgent: 'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Mobile Safari/537.36',
    isMobile: true, hasTouch: true });
  p.on('pageerror', e => console.log('PAGEERROR:', e.message));
  // Report 'suspended' until resume() actually resolves, which is the window
  // the queue exists to cover.
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
