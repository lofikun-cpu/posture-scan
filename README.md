# POSTURE.SCAN — viral AI posture analyzer (KinaPT marketing funnel)

A single-file, "Jarvis"-style HUD web app: the user uploads or captures a photo,
an on-device AI grades their posture across multiple checkpoints, delivers a
score out of 100 with a synthesized voice, generates a shareable score card,
and funnels the user to **KinaPT**.

## The 6-step funnel

1. **Greet** — animated boot sequence, typed + spoken instructions
2. **Capture** — live camera (3-2-1 countdown) or photo upload
3. **Analyze** — MediaPipe pose detection, skeleton overlay, scan-line sweep, per-checkpoint severity grading (OPTIMAL / MINOR / MODERATE / SEVERE)
4. **Score** — animated ring, score /100, Jarvis-style verdict
5. **Share** — canvas-generated 1080×1350 score card via Web Share API (native share sheet on mobile; download + tweet-intent fallback on desktop)
6. **CTA** — "Fix My Posture with KinaPT"

## Deploy (the URL you boost traffic to)

Hosted free on **GitHub Pages** — no backend, no keys, no scaling cost:

1. Repo → **Settings → Pages → Deploy from a branch** → pick the branch, root folder
2. Your URL: `https://<user>.github.io/posture-scan/`
3. Optional: add a custom domain (e.g. `scan.kinapt.app`) for brand trust in shares

Before boosting:
- Replace `KINAPT_URL` in `index.html` with the real KinaPT store/landing link (currently `https://kinapt.app`)
- Add an `og:image` (a sample score card) so link previews pop on social

## Architecture — why it's built this way

**Everything runs client-side.** Pose estimation uses Google's free
**MediaPipe Pose Landmarker** (pretrained, ~5&nbsp;MB model, vendored into `/vendor`
so the whole app is served from one origin — no third-party CDN — and runs
in-browser via WASM/WebGPU). The Jarvis voice is the browser's **Web Speech API**. Share
cards are drawn with **Canvas**. There is no server:

- **$0 per user** — boosted traffic spikes cost nothing and can't take the app down
- **Privacy is a feature** — "your photo never leaves your phone" is displayed prominently; that removes the #1 objection to uploading a body photo
- **No datasets needed** — the pose model is pretrained; posture scoring is deterministic geometry (joint angles / plumb-line deviations) on top of its 33 landmarks

## APIs used (all free, no keys)

| API | Purpose |
|---|---|
| MediaPipe Tasks Vision (vendored) | 33-point pose landmarks, on-device |
| Web Speech API (`speechSynthesis`) | Jarvis voice |
| `getUserMedia` | live camera capture |
| Canvas 2D | HUD overlay, skeleton, share card |
| Web Share API (`navigator.share`) | native share sheet with image |

## Scoring model

View is auto-detected (front vs side profile) from shoulder width vs torso length.

- **Front:** head tilt, shoulder level, head centering, pelvic balance, spinal lean
- **Side:** forward head ("tech neck"), neck angle, shoulder-over-hip stack, hip stack, ear-shoulder-hip plumb line

Each checkpoint maps to a 0–1 severity against physio-informed thresholds, then
a weighted, slightly non-linear blend produces the /100 score (mid-range scores
cluster 55–85: low enough to motivate the CTA, high enough to be shareable).

> Not a medical device — wellness screening demo only (stated in the footer).
