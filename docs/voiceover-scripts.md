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

**Keep the line breaks and the `<break>` tags exactly as written.** On-screen
captions split on the line breaks, and the app times each caption using its
spoken length *plus* the silence that follows it. Those pause values are
mirrored in `VO_LINES[...].pauses` in `app.js` — **if you change a `<break>`
here, change the matching number there**, or the captions will drift out of
step with the voice.

---

## 1 · Intro briefing → `vo-intro.mp3`

Recorded: **Edmund (British Podcast Host)**, 32.1 s including pauses. This is the
line that decides whether people stay. Paste exactly as-is:

```
Good day. I am KINA — posture intelligence, online. <break time="0.8s" />
I'll scan you from two angles and grade your alignment out of one hundred. <break time="0.7s" />
Stand two or three steps back. I need to see you from head to feet. <break time="0.5s" />
Bare feet. Fitted clothing — loose fabric hides your spine from me. <break time="0.5s" />
First photo, face me. Second, turn ninety degrees. <break time="0.7s" />
And stand how you normally stand. I'll know if you're cheating.
```

App-side pause values: `[0.8, 0.7, 0.5, 0.5, 0.7, 0]`

## 2 · Front photo prompt → `vo-front.mp3`

Target ~8 seconds.

```
Frontal view. Stand square to me, arms relaxed at your sides. <break time="0.5s" />
Look straight ahead, and hold still.
```

App-side pause values: `[0.5, 0]`

## 3 · Side photo prompt → `vo-side.mp3`

Target ~9 seconds.

```
Good. Now turn ninety degrees, so one shoulder faces me. <break time="0.5s" />
Arms hanging naturally. Do not correct your posture — I will know.
```

App-side pause values: `[0.5, 0]`

---

## About the `<break>` tags

`<break time="0.8s" />` is ElevenLabs' documented syntax and gives a clean,
exact silence — far more reliable than hoping punctuation produces a pause.
Ceiling is 3 seconds per tag.

Use them sparingly. ElevenLabs warns that heavy use of break tags can
destabilise some voices, producing artefacts or odd breathing. Six tags across
a 32-second read is comfortably within normal use.

If a voice does misbehave on them, the fallback is punctuation — an ellipsis
gives roughly a beat:

```
Good day. I am KINA — posture intelligence, online...
I'll scan you from two angles and grade your alignment out of one hundred...
```

That's less precise, so re-time the `pauses` arrays by ear if you go that route.

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
