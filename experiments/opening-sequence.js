/* ============================================================
   PARKED — 15-second scored opening sequence
   ============================================================

   Built to the written timing sheet (retina scan → portal shatter →
   loading bar → data stream → reset → static → radar), with a matching
   audio schedule. Pulled out of the live app because the montage read as
   a run of unrelated full-screen patterns that had nothing in common
   with the KINA core it handed off to.

   Kept here as the starting point for the opening sequence as its own
   project. It is NOT loaded by the app.

   To run it again you need, from app.js: the kinaCore canvas context and
   its `arc()` helper, `clamp`, `easeOut`, the palette constants
   (BLUE/HOT/GOLD/MAGENTA), `sfx()`, `tone()`, `noise()` and `sfxReady()`.
   drawBoot(bt, W, H, cx, cy) expects `bt` in seconds from 0 to 15.
   ============================================================ */

/* ---- action bed ----
   Driving pulse for the frantic first phase of the boot, cut dead at the
   reset. Separate from the ambient bed, which is the idle drone. */
let action = null;

function startAction() {
  if (action || !sfxReady()) return;
  const t0 = audioCtx.currentTime;
  const out = audioCtx.createGain();
  out.gain.setValueAtTime(0.0001, t0);
  out.gain.exponentialRampToValueAtTime(0.10 * SFX_GAIN, t0 + 0.5);
  out.connect(audioCtx.destination);

  const drone = audioCtx.createOscillator();
  drone.type = 'sawtooth'; drone.frequency.value = 41.2;
  const dlp = audioCtx.createBiquadFilter();
  dlp.type = 'lowpass'; dlp.frequency.value = 180;
  const dg = audioCtx.createGain(); dg.gain.value = 0.5;
  drone.connect(dlp); dlp.connect(dg); dg.connect(out); drone.start(t0);

  // 16th-note pulse driving the montage
  const pulse = audioCtx.createOscillator();
  pulse.type = 'square'; pulse.frequency.value = 82.4;
  const pg = audioCtx.createGain(); pg.gain.value = 0;
  pulse.connect(pg); pg.connect(out); pulse.start(t0);
  for (let i = 0; i < 80; i++) {
    const at = t0 + i * 0.15;
    pg.gain.setValueAtTime(0.34, at);
    pg.gain.exponentialRampToValueAtTime(0.001, at + 0.11);
  }
  action = { out, drone, pulse };
}

function stopAction(fade = 0.06) {
  if (!action) return;
  const t0 = audioCtx.currentTime;
  try {
    action.out.gain.cancelScheduledValues(t0);
    action.out.gain.setValueAtTime(Math.max(0.0001, action.out.gain.value), t0);
    action.out.gain.exponentialRampToValueAtTime(0.0001, t0 + fade);
    action.drone.stop(t0 + fade + 0.05);
    action.pulse.stop(t0 + fade + 0.05);
  } catch (e) {}
  action = null;
}


/* Scored 15s boot, then the briefing. Timings mirror drawBoot(). */
const bootTimers = [];
function scheduleBootAudio() {
  const at = (s, fn) => bootTimers.push(setTimeout(fn, s * 1000));
  startAction();                                   // 0:00 music in
  // telemetry chatter and whooshes through the frantic phase
  for (let s = 0.3; s < 9.6; s += 0.34) at(s + Math.random() * 0.12, () => sfx('telemetry', 0.3));
  [0.6, 2.2, 3.4, 4.6, 5.0, 7.0, 8.4].forEach(s => at(s, () => sfx('swoosh', 0.5)));
  at(2.2, () => sfx('powerUp', 0.9));              // portal shatter
  at(10.0, () => { stopAction(0.05); coreEnergy = 0; });  // 0:10 hard cut to silence
  at(10.9, () => sfx('target', 0.4));              // lone targeting beep
  at(12.0, () => sfx('staticZap', 1));             // static crackle
  at(12.6, () => {                                 // "Welcome."
    if (!voiceOn) return;
    startBed();
    speakLines(['Welcome.']);
  });
}
function clearBootAudio() {
  bootTimers.splice(0).forEach(clearTimeout);
  stopAction(0.15);
}


  /* ============================================================
     BOOT SEQUENCE — 15s, scored, per the supplied timing sheet.
       0–2   retina scan / radar / geographic, glitching to hexagons
       2–5   portal flash shattering into particles, warped grids
       5–7   matrix grid, loading bar stretching outward
       7–10  bar dissolves into flowing fibre-optic data
       10–12 black, green crosshair, red grid expanding — silence
       12–15 static burst snapping into the stable radar HUD
     ============================================================ */
  const GREEN = '90,255,140';
  const RED = '237,77,66';

  let particles = null;
  const seg = (bt, a, z) => clamp((bt - a) / (z - a), 0, 1);

  function glitch(W, H, amt) {
    if (amt <= 0.01) return;
    const slices = Math.round(2 + amt * 7);
    for (let i = 0; i < slices; i++) {
      const sy = Math.random() * H;
      const sh = H * (0.01 + Math.random() * 0.07);
      const dx = (Math.random() - 0.5) * W * 0.22 * amt;
      try { ctx.drawImage(c, 0, sy, W, sh, dx, sy, W, sh); } catch (e) {}
      if (Math.random() < 0.5) {
        ctx.fillStyle = `rgba(${Math.random() < 0.5 ? GREEN : RED},${0.05 + Math.random() * 0.12 * amt})`;
        ctx.fillRect(0, sy, W, sh);
      }
    }
  }

  function hexField(W, H, cx, cy, k, spin) {
    const s = Math.min(W, H) * 0.075;
    ctx.strokeStyle = `rgba(${GREEN},${0.18 + 0.4 * k})`;
    ctx.lineWidth = Math.max(1, W * 0.0022);
    for (let row = -1; row * s * 1.5 < H + s; row++) {
      for (let col = -1; col * s * 1.73 < W + s; col++) {
        const hx = col * s * 1.73 + (row % 2 ? s * 0.87 : 0);
        const hy = row * s * 1.5;
        const d = Math.hypot(hx - cx, hy - cy) / Math.max(W, H);
        if (d > k * 1.3) continue;
        ctx.beginPath();
        for (let v = 0; v < 6; v++) {
          const a = spin + v * Math.PI / 3;
          const px = hx + Math.cos(a) * s * 0.55, py = hy + Math.sin(a) * s * 0.55;
          v ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        }
        ctx.closePath(); ctx.stroke();
      }
    }
  }

  function drawBoot(bt, W, H, cx, cy) {
    const M = Math.min(W, H);
    ctx.fillStyle = '#01050a';
    ctx.fillRect(0, 0, W, H);
    const fsz = Math.max(9, W * 0.022);
    ctx.font = `${fsz}px "Consolas", monospace`;
    ctx.textBaseline = 'middle';

    /* ---- 0–2s: retina scan, radar, geographic — frantic ---- */
    if (bt < 2.2) {
      const k = seg(bt, 0, 2);
      ctx.textAlign = 'left';
      ctx.fillStyle = `rgba(${GREEN},${0.5 + 0.5 * Math.sin(bt * 22)})`;
      ctx.fillText('RETINA SCAN', W * 0.06, H * 0.30);
      ctx.textAlign = 'right';
      ctx.fillText('GEOGRAPHIC', W * 0.94, H * 0.30);
      ctx.textAlign = 'left';
      ctx.fillStyle = `rgba(${GREEN},0.4)`;
      ctx.fillText('ENABLING SENSOR ARRAY', W * 0.06, H * 0.36);
      ctx.textAlign = 'right';
      ctx.fillText('SPY SEQUENCE', W * 0.94, H * 0.36);

      // circular radar/eye
      const rr = M * 0.2;
      for (let i = 0; i < 4; i++)
        arc(cx, cy, rr * (0.4 + i * 0.24), bt * (i % 2 ? -3 : 3), bt * (i % 2 ? -3 : 3) + 4.4,
            Math.max(1, M * 0.006), 0.55, GREEN);
      const eye = ctx.createRadialGradient(cx, cy, 0, cx, cy, rr * 0.42);
      eye.addColorStop(0, `rgba(255,255,255,0.9)`);
      eye.addColorStop(0.5, `rgba(${GREEN},0.6)`);
      eye.addColorStop(1, `rgba(${GREEN},0)`);
      ctx.fillStyle = eye;
      ctx.beginPath(); ctx.arc(cx, cy, rr * 0.42, 0, 7); ctx.fill();
      // sweep arm
      ctx.strokeStyle = `rgba(${GREEN},0.8)`;
      ctx.lineWidth = Math.max(1, M * 0.008);
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(bt * 9) * rr, cy + Math.sin(bt * 9) * rr);
      ctx.stroke();

      if (bt > 1.2) hexField(W, H, cx, cy, seg(bt, 1.2, 2.2), bt);
      glitch(W, H, 0.5 + 0.5 * Math.abs(Math.sin(bt * 13)));
      return;
    }

    /* ---- 2–5s: portal flash, particle shatter, warped grids ---- */
    if (bt < 5) {
      const k = seg(bt, 2.2, 5);
      // portal, alive only for a beat before it shatters
      if (bt < 2.75) {
        const pr = M * (0.08 + seg(bt, 2.2, 2.75) * 0.28);
        const pg = ctx.createRadialGradient(cx, cy, 0, cx, cy, pr);
        pg.addColorStop(0, 'rgba(255,255,255,0.95)');
        pg.addColorStop(0.35, `rgba(${MAGENTA},0.8)`);
        pg.addColorStop(0.7, `rgba(${GOLD},0.6)`);
        pg.addColorStop(1, `rgba(${GREEN},0)`);
        ctx.fillStyle = pg;
        ctx.beginPath(); ctx.arc(cx, cy, pr, 0, 7); ctx.fill();
      } else {
        if (!particles) {
          particles = Array.from({ length: 150 }, () => {
            const a = Math.random() * Math.PI * 2, sp = 0.25 + Math.random() * 1.5;
            return { a, sp, r: M * 0.05, s: 1 + Math.random() * 2.5 };
          });
        }
        const age = bt - 2.75;
        for (const p of particles) {
          const d = p.r + p.sp * age * M * 0.55;
          const px = cx + Math.cos(p.a) * d, py = cy + Math.sin(p.a) * d * 0.8;
          ctx.fillStyle = `rgba(${GREEN},${Math.max(0, 0.9 - age * 0.5)})`;
          ctx.fillRect(px, py, p.s * 2, p.s * 2);
        }
      }
      // warped data waves
      ctx.strokeStyle = `rgba(${GREEN},0.35)`;
      ctx.lineWidth = Math.max(1, W * 0.002);
      for (let i = 0; i < 9; i++) {
        ctx.beginPath();
        for (let x = 0; x <= W; x += 12) {
          const y = H * (0.12 + i * 0.095)
            + Math.sin(x * 0.012 + bt * 6 + i) * H * 0.05 * (0.4 + k);
          x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
      }
      hexField(W, H, cx, cy, 1 - k * 0.6, bt * 0.5);
      glitch(W, H, 0.75 * (1 - k * 0.5));
      return;
    }

    /* ---- 5–7s: matrix grid, loading bar stretches outward ---- */
    if (bt < 7) {
      const k = seg(bt, 5, 7);
      const g = M * 0.045;
      ctx.strokeStyle = `rgba(${GREEN},0.16)`;
      ctx.lineWidth = 1;
      for (let x = 0; x < W; x += g) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
      for (let y = 0; y < H; y += g) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
      // falling cells, matrix-ish
      for (let i = 0; i < 40; i++) {
        const cxx = Math.floor((i * 7919) % Math.floor(W / g)) * g;
        const cyy = ((bt * (0.3 + (i % 5) * 0.12) * H * 1.4) + i * 137) % H;
        ctx.fillStyle = `rgba(${GREEN},${0.15 + 0.5 * Math.random()})`;
        ctx.fillRect(cxx, Math.floor(cyy / g) * g, g, g);
      }
      // loading bar stretching from centre outward
      const bw = W * 0.94 * easeOut(k);
      const bh = Math.max(6, H * 0.022);
      ctx.strokeStyle = `rgba(${GREEN},0.9)`;
      ctx.lineWidth = Math.max(1, W * 0.003);
      ctx.strokeRect(cx - bw / 2, cy - bh / 2, bw, bh);
      ctx.fillStyle = `rgba(${GREEN},0.55)`;
      ctx.fillRect(cx - bw / 2 + 2, cy - bh / 2 + 2, (bw - 4) * k, bh - 4);
      ctx.textAlign = 'center';
      ctx.fillStyle = `rgba(${GREEN},0.95)`;
      ctx.font = `600 ${fsz * 1.5}px "Consolas", monospace`;
      ctx.fillText(Math.round(k * 100) + '%', cx, cy - bh * 2.2);
      ctx.font = `${fsz}px "Consolas", monospace`;
      ctx.fillText('loading…', cx, cy + bh * 2.4);
      return;
    }

    /* ---- 7–10s: bar dissolves into flowing fibre-optic data ---- */
    if (bt < 10) {
      const k = seg(bt, 7, 10);
      for (let i = 0; i < 26; i++) {
        const base = H * (0.08 + (i / 26) * 0.84);
        const amp = H * 0.10 * (0.3 + Math.sin(i * 1.7) * 0.7);
        const hot = i % 5 === 0;
        ctx.strokeStyle = hot ? `rgba(${HOT},${0.5 + 0.4 * k})` : `rgba(${GREEN},${0.22 + 0.4 * k})`;
        ctx.lineWidth = Math.max(1, W * (hot ? 0.004 : 0.0022));
        ctx.beginPath();
        for (let x = -20; x <= W + 20; x += 10) {
          const y = base + Math.sin(x * 0.009 + bt * 5 + i * 0.8) * amp * k
                         + Math.sin(x * 0.021 - bt * 3) * amp * 0.35 * k;
          x <= -20 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        // travelling packet along the strand
        const px = ((bt * 0.55 + i * 0.13) % 1) * W;
        const py = base + Math.sin(px * 0.009 + bt * 5 + i * 0.8) * amp * k;
        ctx.fillStyle = `rgba(${HOT},${0.8 * k})`;
        ctx.beginPath(); ctx.arc(px, py, Math.max(1.5, W * 0.004), 0, 7); ctx.fill();
      }
      if (bt < 7.4) glitch(W, H, 0.35 * (1 - seg(bt, 7, 7.4)));
      return;
    }

    /* ---- 10–12s: the reset — black, crosshair, red grid ---- */
    if (bt < 12) {
      const k = seg(bt, 10, 12);
      const ch = M * 0.05;
      ctx.strokeStyle = `rgba(${GREEN},${0.9 * clamp(seg(bt, 10.15, 10.5), 0, 1)})`;
      ctx.lineWidth = Math.max(1, W * 0.003);
      ctx.beginPath();
      ctx.moveTo(cx - ch, cy); ctx.lineTo(cx + ch, cy);
      ctx.moveTo(cx, cy - ch); ctx.lineTo(cx, cy + ch);
      ctx.stroke();
      arc(cx, cy, ch * 0.55, 0, Math.PI * 2, Math.max(1, W * 0.002),
          0.8 * clamp(seg(bt, 10.2, 10.6), 0, 1), GREEN);
      // red grid expanding outward
      const gk = seg(bt, 10.7, 12);
      if (gk > 0) {
        const half = M * 0.06 + easeOut(gk) * M * 0.44;
        ctx.strokeStyle = `rgba(${RED},${0.75 * (1 - gk * 0.25)})`;
        ctx.lineWidth = Math.max(1, W * 0.0035);
        ctx.strokeRect(cx - half, cy - half, half * 2, half * 2);
        ctx.lineWidth = Math.max(1, W * 0.0015);
        ctx.strokeStyle = `rgba(${RED},${0.35 * (1 - gk * 0.3)})`;
        for (let i = 1; i < 4; i++) {
          const o = -half + (i / 4) * half * 2;
          ctx.beginPath();
          ctx.moveTo(cx + o, cy - half); ctx.lineTo(cx + o, cy + half);
          ctx.moveTo(cx - half, cy + o); ctx.lineTo(cx + half, cy + o);
          ctx.stroke();
        }
      }
      return;
    }

    /* ---- 12–15s: static burst snapping into the radar HUD ---- */
    const k = seg(bt, 12, 15);
    if (bt < 12.45) {                                    // grey TV static
      const cell = Math.max(2, W * 0.006);
      for (let y = 0; y < H; y += cell) {
        for (let x = 0; x < W; x += cell) {
          const v = 40 + Math.random() * 180;
          ctx.fillStyle = `rgba(${v},${v},${v},0.85)`;
          ctx.fillRect(x, y, cell, cell);
        }
      }
      return;
    }
    const on = seg(bt, 12.45, 12.8);
    const rr = M * 0.28;
    ctx.globalAlpha = on;
    // rotating radar
    for (let i = 1; i <= 4; i++)
      arc(cx, cy, rr * i / 4, 0, Math.PI * 2, Math.max(1, M * 0.004), 0.35);
    ctx.strokeStyle = `rgba(${BLUE},0.3)`;
    ctx.lineWidth = Math.max(1, M * 0.003);
    for (let i = 0; i < 8; i++) {
      const a = i * Math.PI / 4;
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr); ctx.stroke();
    }
    const sweep = bt * 1.6;
    const sg = ctx.createConicGradient
      ? ctx.createConicGradient(sweep, cx, cy) : null;
    if (sg) {
      sg.addColorStop(0, `rgba(${GREEN},0.45)`);
      sg.addColorStop(0.12, `rgba(${GREEN},0)`);
      sg.addColorStop(1, `rgba(${GREEN},0)`);
      ctx.fillStyle = sg;
      ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = `rgba(${GREEN},0.9)`;
    ctx.lineWidth = Math.max(1, M * 0.006);
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(sweep) * rr, cy + Math.sin(sweep) * rr); ctx.stroke();
    // side data panels
    ctx.strokeStyle = `rgba(${BLUE},0.4)`;
    ctx.lineWidth = Math.max(1, W * 0.002);
    for (let i = 0; i < 3; i++) {
      const py = H * (0.3 + i * 0.16);
      ctx.strokeRect(W * 0.04, py, W * 0.16, H * 0.11);
      ctx.strokeRect(W * 0.80, py, W * 0.16, H * 0.11);
    }
    ctx.globalAlpha = 1;
  }

