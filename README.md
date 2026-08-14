# POSTURE.SCAN — viral AI posture analyzer (KinaPT funnel)

A "Jarvis"-style HUD web app. The user takes **two photos** (front and side),
an on-device AI grades their alignment across 12 biomechanical checkpoints,
delivers a score out of 100 with a synthesized voice, reports **how far each
checkpoint sits from ideal in inches**, headlines a **forward-roll percentage**
for the upper body, generates a shareable score card, and deep-links to
**KinaPT** on the App Store.

No backend. No API keys. No per-user cost.

## The funnel

1. **Greet** — animated boot sequence, typed instructions (voice starts on first tap, per mobile autoplay policy)
2. **Capture front** — live camera with countdown + silhouette guide, or upload
3. **Capture side** — same, with a 90°-turn guide
4. **Analyze** — per-photo quality gating, skeleton overlay, scan sweep, 12 graded checkpoints
5. **Score** — animated ring, total /100 plus separate FRONT and SIDE sub-scores
6. **Simulate** — drag-to-compare slider: their photo warped toward ideal alignment
7. **Share** — 1080×1350 score card via the native share sheet
8. **CTA** — deep link to the KinaPT App Store listing, carrying scan context

## Deploy (the URL you boost traffic to)

Hosted free on **GitHub Pages**:

1. Repo → **Settings → Pages → Deploy from a branch** → pick the branch, root folder
2. Your URL: `https://<user>.github.io/posture-scan/`
3. Optional: a custom domain (e.g. `scan.kinapt.app`) for brand trust in shares

**Before boosting**, edit two constants in `app.js`:
- `APPSTORE_URL` — currently a placeholder `id0000000000`; replace with the real KinaPT listing
- The `og:image` URL in `index.html` if you move to a custom domain

## Architecture

Everything runs client-side. Pose estimation uses Google's **MediaPipe Pose
Landmarker** (`full` model, ~9.4 MB), vendored into `/vendor` so the whole app
is served from one origin — no third-party CDN. The Jarvis voice is the Web
Speech API. Share cards are canvas.

This is deliberate:
- **$0 per user** — boosted traffic spikes cost nothing and can't take the app down
- **Privacy is a feature** — "your photos never leave your phone" is displayed prominently, which removes the main objection to uploading a body photo
- **No dataset needed** — the pose model is pretrained; grading is deterministic geometry over its landmarks

### APIs used (all free, no keys)

| API | Purpose |
|---|---|
| MediaPipe Tasks Vision (vendored) | 33 landmarks, 3D world landmarks, segmentation mask |
| Web Speech API | Jarvis voice |
| `getUserMedia` | live camera capture |
| Canvas 2D | HUD overlay, mesh warp, share card |
| Web Share API | native share sheet with image |

## What's measured

The view is verified in 3D before anything is scored (see *Quality gating*).

**Frontal plane** — shoulder balance, pelvic level (hip drop), trunk lean,
head centring, knee valgus/varus (offset from the hip→ankle mechanical axis),
foot progression (forefoot abduction).

**Sagittal plane** — forward head (ear vs acromion), shoulder protraction
(acromion vs greater trochanter), thoracic curve, lumbar curve, pelvis over
base (sway back), and the full clinical plumb chain ankle → knee → hip →
shoulder → ear.

Each checkpoint maps to a 0–1 severity against physio-informed thresholds,
then a weighted, slightly non-linear blend produces the score. Mid-range
results cluster 55–85: low enough to motivate the CTA, high enough to share.

### Three things worth knowing about the measurements

**Aspect ratio is corrected before any geometry.** MediaPipe normalises `x` by
image width and `y` by image height. Mixing them raw makes every angle wrong on
portrait phone photos — which is essentially all traffic. `posture.js` rescales
x into y-units first, and there's a regression test asserting the same physical
pose scores identically in portrait and landscape.

**Pelvic tilt is not measured, on purpose.** Anterior/posterior pelvic tilt is
the ASIS–PSIS angle, and MediaPipe gives one hip point per side — the line
between them is the pelvis's mediolateral axis, and you can't measure rotation
*about* an axis using only that axis. What's reported instead is what's
genuinely observable: lumbar curve depth from the body outline, and sway back
from hip-over-ankle translation.

**Foot progression is computed in 3D against the pelvis-forward axis**, so
whole-body rotation can't masquerade as forefoot abduction. It still can't see
the medial arch — no pose model can from front or side — so it's labelled as
what it measures.

Metrics are described, never diagnosed. Shoulder asymmetry is reported as
"shoulder balance", not as a spinal condition — that keeps the app out of
medical-device territory and out of ad-moderation trouble.

## Quality gating

Bad input producing a confident score is the fastest way to lose credibility,
so each photo must pass before it's scored. Using the 3D world landmarks, the
app computes hip-axis rotation and rejects a "front" photo turned more than 32°
off square, or a "side" photo under 55° of turn — with a message naming the
measured angle. It also requires the full torso, and warns (rather than fails)
when feet are out of frame. The back-contour trace is discarded if it steps
discontinuously, which is how an outstretched arm or a chair back gets caught
instead of being scored as spinal curvature.

## Measurements

Every checkpoint reports a figure, not just a severity word: "1.8 in ahead of
shoulder", "0.6 in uneven", "17° toe-out". The planar maths works in image
units, so the conversion comes from MediaPipe's world landmarks — a metric body
fit — using the shoulder-to-hip torso as the ruler. That fit is estimated from
a single photograph by a generic model, so absolute scale carries real error:
figures are rounded to a tenth of an inch, labelled as estimates in the UI, and
suppressed entirely (falling back to angles and percentages) when the fitted
torso lands outside 20–80 cm, which means the fit has failed.

## Forward-roll index

One headline percentage for the pattern people recognise in the mirror: a
weighted read of the thoracic curve, the head carried ahead of the shoulders,
and the shoulders ahead of the hips. 0% is a stacked upper body, 100% the far
end of what the scan resolves. It is a posture measurement and is not presented
as a diagnosis — hyperkyphosis is a clinical finding that needs a clinician.

## Tests

```bash
node test/posture.test.mjs     # 63 checks, no dependencies
```

Covers the full analysis engine: perfect-posture baselines, aspect-ratio
independence, metric monotonicity and sign conventions (valgus vs varus,
forward vs backward head), mirror invariance, 3D rotation gating, contour
extraction and its rejection guards, warp geometry, and scoring behaviour.

The browser integration path (MediaPipe loading, mask decoding, two-photo UI
flow, mesh warp, share card) is exercised separately with Playwright against a
local server; `app.js` exposes a `window.__scan` seam on localhost only.

> Not a medical device — wellness screening demo only (stated in the footer).
