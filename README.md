# Step 15: benchmark and charts

> This branch is one step of **[eyeTracker](https://github.com/tegemenozyurek/eyeTracker)**, a real-time driver
> monitoring system that runs in the browser. Every step of the project has its own branch, and its README explains
> what was done in that step. The project overview is the README on [`main`](https://github.com/tegemenozyurek/eyeTracker).
>
> Previous: [Step 14: eyeTrack1 in the demo](https://github.com/tegemenozyurek/eyeTracker/tree/step-14-eyetrack1-demo) ·
> Next: Step 16: final README and live demo

## What was done

- **[`scripts/benchmark.py`](scripts/benchmark.py)** collects every result from the reports the earlier steps wrote
  (no number retyped) into [`models/benchmark.txt`](models/benchmark.txt) and one chart.
- **[`scripts/browser_benchmark.py`](scripts/browser_benchmark.py) + [`tools/benchmark.html`](tools/benchmark.html)**
  measure each part of the live pipeline in a real browser: the page runs the demo's own code on LFW photos used as
  640x480 video frames and sends its timings to a small local server, which saves them to
  [`models/browser_benchmark.json`](models/browser_benchmark.json).
- **[`scripts/make_showcase.py`](scripts/make_showcase.py)**: the README header image.

## Results

![Benchmark](assets/benchmark.png)

**Eye state** (people never seen in training):

| | `rules` | `eyeTrack0.1` | `eyeTrack0.5` |
|---|---:|---:|---:|
| MRL test (infrared) | – | 97.8% | **98.1%** |
| CEW test (normal camera) | – | 89.3% | **96.7%** |
| simulated webcam, MRL / CEW | – | 60.6% / 56.3% | **78.2% / 84.7%** |
| live pipeline, 716 real eyes | 89.5% | 91.5% | **96.5%** |
| closed eyes caught (live pipeline) | 79.9% | 89.8% | **95.5%** |
| parameters / training time (M4) | – | 295,266 / 7.8 min | 295,266 / 8.2 min |

**Drowsiness over 60 s** (UTA-RLDD, 60 drivers, 5-fold cross-validation):

| | rules (demo) | PERCLOS, tuned | `eyeTrack1` | `eyeTrack1` + personal baseline* |
|---|---:|---:|---:|---:|
| 3 classes, window accuracy | 44.8% | 48.1% | 44.4% | 49.7% |
| 3 classes, macro F1 | 41.7% | 48.3% | 44.4% | 49.9% |
| alert vs drowsy | 68.7% | 65.5% | 66.8% | 72.4% |

\* needs the driver's data from several sessions: an analysis, not deployable. Paper (Ghoddoosian et al. 2019,
Table 2): 54% per blink sequence, 65.2% per video.

**Browser speed** (Chrome 152 on an Apple M4, median of 200 frames):

| part of the pipeline | time per frame |
|---|---:|
| MediaPipe face landmarker (478 points, blendshapes, head pose) | 8.5 ms |
| two aligned 32x32 eye crops | 2.0 ms |
| `eyeTrack0.5`, both eyes (ONNX Runtime Web, WASM) | 3.1 ms |
| 13 `eyeTrack1` features, rules | < 0.1 ms each (below Chrome's 0.1 ms timer resolution) |
| `eyeTrack1`, one 60 s window (runs every 2 s) | 0.2 ms |
| **whole pipeline with `eyeTrack0.5`** | **13.6 ms, up to 74 frames per second** |

The benchmark page was not visible on screen during the run; hidden tabs are slowed down, so these numbers are if
anything pessimistic.

![Showcase](assets/showcase.png)

16 eyes of test people, picked at random with a fixed seed (not hand-picked); `eyeTrack0.5` got all 16 right, a likely
but slightly lucky draw at 96.7 to 98.1% accuracy.

## Try it

```bash
git checkout step-15-benchmark
python scripts/benchmark.py
python scripts/make_showcase.py
python scripts/browser_benchmark.py      # then open http://localhost:8030/tools/benchmark.html in a visible tab
```
