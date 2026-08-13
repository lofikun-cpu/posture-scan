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
    await p.waitForTimeout(5200);                     // into the voiceover
    const samples = [];
    for (let i = 0; i < 6; i++) {
      samples.push(await p.evaluate(() => window.__scan.audio().energy));
      await p.waitForTimeout(160);
    }
    const a = await p.evaluate(() => window.__scan.audio());
    const moving = Math.max(...samples) - Math.min(...samples) > 0.05;
    console.log(`  ${ios ? 'iOS ' : 'desk'}  vo=${a.t === null ? 'none' : a.t.toFixed(2)+'s'} paused=${a.paused} analyser=${a.analyser}  energy ${samples.map(s=>s.toFixed(2)).join(' ')}  ${moving ? '✓ core moving' : '✗ core static'}`);
    await p.close();
  }
  await b.close();
})().catch(e => { console.error(e.message); process.exit(1); });
