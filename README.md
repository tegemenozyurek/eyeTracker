# Step 14: eyeTrack1 in the demo

> This branch is one step of **[eyeTracker](https://github.com/tegemenozyurek/eyeTracker)**, a real-time driver
> monitoring system that runs in the browser. Every step of the project has its own branch, and its README explains
> what was done in that step. The project overview is the README on [`main`](https://github.com/tegemenozyurek/eyeTracker).
>
> Previous: [Step 13: train eyeTrack1](https://github.com/tegemenozyurek/eyeTracker/tree/step-13-train-eyetrack1) ·
> Next: Step 15: benchmark and charts

## What was done

- **Live features identical to training:** [`web/features.js`](web/features.js) and
  [`src/rldd_features.py`](src/rldd_features.py) compute the 13 per-frame features with the formulas of the UTA-RLDD
  feature dataset's own extraction code (blink, squint and look-down blendshapes, jawOpen, 3D EAR and MAR, head pitch,
  yaw and roll from the transformation matrix).
- **Feature check on real faces:** [`scripts/feature_parity.py`](scripts/feature_parity.py) +
  [`tools/feature_check.html`](tools/feature_check.html) compute the features for 12 LFW faces in Python and in the
  browser and compare them, including the other possible reading of the head-pose matrix.
- **ONNX export** of `eyeTrack1` ([`scripts/export_web.py`](scripts/export_web.py)): 17 kB, checked against PyTorch.
- **Demo:** selecting `eyeTrack1` samples the features 5 times per second into a 60 s buffer (as the training windows);
  every 2 s, once the minute is full and the face was found in at least 70% of it, the GRU judges it. A **Driver state**
  panel shows alert / low vigilant / drowsy. `eyeTrack1` replaces the PERCLOS and yawn rules; the microsleep (eyes
  closed 1.5 s) and eyes-off-the-road rules stay as safety nets.
- **End-to-end check:** [`tools/onnx_check.html`](tools/onnx_check.html) feeds real UTA-RLDD windows frame by frame into
  the demo's own `eyeTrack1` code and compares with PyTorch.

## Why

`eyeTrack1` was trained on features someone else extracted. If the browser computed any of them differently (a
different EAR, a transposed rotation matrix), the live model would receive numbers it never saw, and its measured
accuracy would mean nothing.

## Results

| check | result |
|---|---|
| features, browser vs Python, 12 real faces | 10 blendshape / ratio features within 1e-4; pitch, yaw, roll within 0.002° |
| head-pose matrix read the other way round | off by up to 31.7° (pitch), 59.1° (yaw), 16.5° (roll): the column-major reading is right |
| ONNX vs PyTorch, 64 windows | same answer for 100.0%; largest logit difference 1.8e-07 |
| demo's sampling + buffer + normalization + ONNX Runtime Web, 4 real UTA-RLDD windows | same answer as PyTorch for 4 of 4; largest probability difference 1.25e-7 |

Not yet done: a live test by a person in front of a webcam for at least one minute.

## Try it

```bash
git checkout step-14-eyetrack1-demo
python3 -m http.server -d web 8010      # pick eyeTrack1, start the camera, wait one minute
python scripts/feature_parity.py && python -m http.server 8020
# then open http://localhost:8020/tools/feature_check.html and /tools/onnx_check.html
```
