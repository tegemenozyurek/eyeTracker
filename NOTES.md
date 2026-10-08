# Notes

Plain-language notes, one section per step: what was done, why, and what the result means.
Every number here comes from a script in this repo; the script is named next to it.

## Step 0: project brief

The full plan lives in [BRIEF.md](BRIEF.md): a driver monitoring system that runs in the browser and warns
when the driver gets drowsy or looks away from the road. It will have one rule-based baseline (`rules`) and three
trained models that each fix a weakness of the previous one: `eyeTrack0.1` learns open/closed eyes from infrared photos,
`eyeTrack0.5` closes the gap to normal colour webcams, and `eyeTrack1` looks at a whole minute of behaviour instead of one
frame, because drowsiness is a pattern over time, not a single closed eye. The work is split into 16 steps, each
ending with a commit and a short report.

## Step 1: project skeleton

The repo now has its basic structure: a `.gitignore` that keeps datasets, training logs and secrets out of git,
the Python requirements, the MIT license, a short README and an environment check
([`check_env.py`](scripts/check_env.py)). The check confirmed that PyTorch can use the M4's built-in GPU (MPS): a small
eye-sized CNN runs 3.6x faster there than on the CPU (27.0 vs 98.3 ms per batch of 256 crops), and the Kaggle token
is in place with private permissions. Only 71 GB of disk is free, so the full UTA-RLDD dataset (111 GB) could not fit
even if we wanted it; we will download a subset of participants. MediaPipe's Python face landmarker can output
blendshapes and the head-pose matrix, which we need later to compute the same per-frame features in Python
(training) as in the browser (live).

What we reuse from emotionDetecter, instead of reinventing it:
- **one script per step** that prints its numbers and saves a chart; `models/<version>/` and `assets/<version>/` folders
- **the same alignment math in Python and JavaScript** (a similarity transform onto a fixed template), now for eyes
  instead of whole faces
- **webcam-style augmentation** (dim light, blur, motion blur, low resolution, noise, detector jitter) and the same
  degradations as a fixed "simulated webcam" test
- **ONNX export checked against PyTorch**, ONNX Runtime Web in the browser (WebGPU, WASM fallback)
- **smoothing over time** so the live label does not flicker; model picker with hover cards; GitHub Pages deployment
