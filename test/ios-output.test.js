const { chromium } = require('playwright-core');
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium',
    args: ['--use-gl=swiftshader','--enable-unsafe-swiftshader','--autoplay-policy=no-user-gesture-required'] });
  for (const ios of [true, false]) {
    const p = await b.newPage({ viewport: { width: 390, height: 844 },
      userAgent: ios
        ? 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
        : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
      isMobile: ios, hasTouch: ios });
    p.on('pageerror', e => console.log('PAGEERROR:', e.message));
    await p.goto('http://localhost:8899/index.html', { waitUntil: 'domcontentloaded' });
    await p.waitForTimeout(600);
    await p.click('#btn-begin');
    await p.waitForTimeout(1500);
    const r = await p.evaluate(() => {
      const els = [...document.querySelectorAll('audio')];
      const streamEl = els.find(e => e.srcObject);
      return {
        ctx: window.__scan.audio().ctx,
        energy: window.__scan.audio().energy,
        streamEl: !!streamEl,
        streamPlaying: streamEl ? !streamEl.paused : null,
        audioEls: els.length
      };
    });
    console.log(`  ${ios ? 'iOS ' : 'desk'}  ctx=${r.ctx}  streamOutput=${r.streamEl}  playing=${r.streamPlaying}  energy=${r.energy.toFixed(3)}`);
    await p.close();
  }
  await b.close();
})().catch(e => { console.error(e.message); process.exit(1); });
