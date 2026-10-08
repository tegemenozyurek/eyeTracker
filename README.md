# 👁️ eyeTracker

[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Real-time driver monitoring in the browser.** A normal webcam watches the driver's eyes, mouth and head and
detects **drowsiness** (eye closure, PERCLOS, slow blinks, yawning, head nodding) and **distraction**
(looking away from the road), with three alert levels: OK, Attention, Take a break.

> **Work in progress.** Built step by step following [BRIEF.md](BRIEF.md). Plain-language notes for every
> step are in [NOTES.md](NOTES.md). Results, the live demo link and charts will appear here as they are produced.

## Planned models

| version | what it is | trained on |
|---|---|---|
| `rules` | reference baseline, no learning: eye aspect ratio, PERCLOS, yawn and head-pose thresholds | nothing |
| `eyeTrack0.1` | small CNN, eye crop → open / closed | MRL Eye Dataset (infrared) |
| `eyeTrack0.5` | same CNN family + RGB eyes, MediaPipe-aligned crops, webcam augmentations | MRL Eye + CEW |
| `eyeTrack1` | temporal model over per-frame features → alert / low vigilant / drowsy | UTA-RLDD |

## Project layout

```
src/               shared code: models, eye alignment, augmentation, features
scripts/           one script per step; each prints its results and saves a chart
tools/monitor/     live dashboard for training runs and long jobs
models/<version>/  weights, config, training log, test report
assets/<version>/  charts
web/               browser demo
data/              datasets (git-ignored, downloaded by scripts)
runs/<version>/    live metrics read by the monitor (git-ignored)
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

## License

Code is released under the [MIT License](LICENSE). Datasets belong to their creators and are not redistributed here.
