# Step 8: web demo with the rules baseline

> This branch is one step of **[eyeTracker](https://github.com/tegemenozyurek/eyeTracker)**, a real-time driver
> monitoring system that runs in the browser. Every step of the project has its own branch, and its README explains
> what was done in that step. The project overview is the README on [`main`](https://github.com/tegemenozyurek/eyeTracker).
>
> Previous: [Step 7: train eyeTrack0.1](https://github.com/tegemenozyurek/eyeTracker/tree/step-07-train-eyetrack01) ·
> Next: Step 9: one eye-crop function in Python and JavaScript

## What was done

The first live demo, in [`web/`](web/). It runs entirely in the browser; no frame leaves the device.

- **Face tracking:** MediaPipe Face Landmarker (tasks-vision) gives, for every webcam frame, 478 face points, 52
  expression scores (blendshapes such as `eyeBlinkLeft`, `jawOpen`) and the head's 4x4 transformation matrix, from which
  pitch, yaw and roll are computed.
- **The `rules` baseline** ([`web/rules.js`](web/rules.js), no learning, no DOM, so it is testable in Node):

  | signal | rule (provisional thresholds) |
  |---|---|
  | eye closed | MediaPipe eyeBlink score ≥ 0.5 (the definition used in the RLDD features) |
  | PERCLOS | share of time with closed eyes over the last 60 s, judged after 20 s: Attention ≥ 7.5%, Take a break ≥ 15% |
  | microsleep | one closure ≥ 1.5 s: Take a break at once |
  | yawns | jawOpen > 0.5 for ≥ 1.5 s; 3 yawns in 5 minutes: Attention |
  | eyes off the road | head > 25° to the side or > 20° down from the direction calibrated in the first 3 s, or face out of view: Attention after 2 s, Take a break after 4 s |
  | head nods | pitch drops > 12° below its 30 s median and comes back within 2 s (counted, shown) |
  | no flicker | a level steps down only after it has been gone for 4 s |

- **Live panels:** driver state, each eye (open/closed, EAR, closure score), blinks (count, per minute, last duration),
  PERCLOS with its thresholds, yawns and nods, head pose relative to the road, eyes-off-road timer, a 60 s eye-closure
  curve, an alert banner and an optional beep.
- **Model picker** with hover cards: `rules` now; `eyeTrack0.1`, `eyeTrack0.5` and `eyeTrack1` arrive in Steps 11 and 14.
- **Debug view** of the 32x32 eye crops a model would see (simple box crops; Step 9 replaces them with the aligned crop
  shared with training), and a **video-file input** for testing without a camera.

## Why

The `rules` baseline is the classic approach the learned models must beat, and the demo is where every model will be
judged by eye. Building it before the models also fixes the per-frame signals that `eyeTrack1` will learn from.

## Results

- **10 of 10 rule tests pass** (`node web/rules.test.mjs`): normal blinking stays OK (PERCLOS 5%), a 2 s closure gives
  Take a break at once, high PERCLOS gives Take a break, PERCLOS waits for 20 s of data, looking 40° away gives
  Attention after 2 s and Take a break after 4 s, a face out of view counts as off the road, a camera mounted 30° to the
  side is not distraction after calibration, three yawns give Attention, a nod is counted, and an alert does not flicker.
  One test caught a wrong expectation of mine: after a 2 s closure the alert falls back to Attention, not OK, because
  those 2 seconds raise the last minute's PERCLOS above 7.5%.
- **Browser check** on a test video of eight face photos (in the in-app browser on the M4): faces found, eyes reported
  open or closed with their EAR, head pose filled in, about 17 frames per second with 14.9 ms of face tracking per
  frame. A live webcam test is still to be done by a person in front of the camera.

## Try it

```bash
git checkout step-08-web-demo-rules
python3 -m http.server -d web 8010     # then open http://localhost:8010 and press Start camera
node web/rules.test.mjs                # the rule tests
```

Look at the screen for 3 seconds (calibration), then try: close your eyes for 2 s, look to the side for 4 s, yawn
three times.
