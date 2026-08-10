# KINA voiceover scripts (ElevenLabs)

Paste each block into ElevenLabs, export as MP3, save to `/assets` with the
filename shown, then switch it on in the `ASSETS` block at the top of `app.js`:

```js
const ASSETS = {
  kinaLoop:  'assets/kina-loop.mp4',
  kinaPoster:'assets/kina-poster.jpg',
  voIntro:   'assets/vo-intro.mp3',
  voFront:   'assets/vo-front.mp3',
  voSide:    'assets/vo-side.mp3'
};
```

Until a file is switched on, the app reads the same script with the browser's
built-in voice, so nothing breaks and nothing 404s.

**Keep the line breaks.** Captions on screen are split on them, and the app
distributes each line across the real audio duration — so the words on screen
stay in step with the audio automatically, whatever length your export is.

---

## 1 · Intro briefing → `vo-intro.mp3`

Target ~25 seconds. This is the one that decides whether people stay.

```
Good day. I am KINA — posture intelligence, online.
I'll scan you from two angles and grade your alignment out of one hundred.
Stand two or three steps back. I need to see you from head to feet.
Bare feet. Fitted clothing — loose fabric hides your spine from me.
First photo, face me. Second, turn ninety degrees.
And stand how you normally stand. I'll know if you're cheating.
```

## 2 · Front photo prompt → `vo-front.mp3`

Target ~7 seconds.

```
Frontal view. Stand square to me, arms relaxed at your sides.
Look straight ahead, and hold still.
```

## 3 · Side photo prompt → `vo-side.mp3`

Target ~8 seconds.

```
Good. Now turn ninety degrees, so one shoulder faces me.
Arms hanging naturally. Do not correct your posture — I will know.
```

---

## Voice direction

Aim for **calm, dry, unhurried** — the authority comes from restraint, not
volume. A British male voice sells the reference hardest. Slightly slower than
default; KINA should never sound rushed or salesy.

The two deliberate character beats are *"loose fabric hides your spine from me"*
and *"I'll know if you're cheating."* Let them land flat and dry rather than
played for laughs — that's what makes it feel like a system with a personality
instead of a mascot.

---

## One limitation worth planning around

Pre-recorded audio **cannot say the user's score**, because the number is
different every time. The score readout ("Sixty-three out of one hundred…")
therefore stays on the browser's built-in voice unless you handle it another way.

Three options, in order of effort:

1. **Leave it.** The intro and prompts are pre-recorded and sound great; only
   the results line uses the synthetic voice. Simplest, and it's already what
   the app does.
2. **Record four number-free verdict lines** — one per score band (90+, 75+,
   55+, below 55) — phrased to avoid stating the figure, e.g. *"Remarkable.
   Top percentile of humans I have scanned."* The number is on screen anyway,
   so nothing is lost. Best quality-to-effort ratio.
3. **Record all 101 numbers** and stitch them at playback. Highest fidelity,
   considerable work, and probably not worth it.

If you want option 2, the four lines follow the bands already in `verdictFor()`
in `posture.js`, and I can wire them up the same way as the others.
