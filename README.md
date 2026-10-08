# Step 1: project skeleton

> This branch is one step of **[eyeTracker](https://github.com/tegemenozyurek/eyeTracker)**, a real-time driver
> monitoring system that runs in the browser. Every step of the project has its own branch, and its README explains
> what was done in that step. The project overview is the README on [`main`](https://github.com/tegemenozyurek/eyeTracker).
>
> Next: [Step 2: training monitor](https://github.com/tegemenozyurek/eyeTracker/tree/step-02-training-monitor)

## What was done

- **Repository rules from the first commit:** a `.gitignore` that keeps datasets (`data/`), training logs (`runs/`),
  credentials and model files out of git. Only the final weights of each model (`models/*/model.pt`) may be committed.
- **Requirements:** PyTorch, MediaPipe, ONNX / ONNX Runtime, kagglehub, NumPy, Matplotlib, Pillow
  ([`requirements.txt`](requirements.txt)), in a Python 3.12 virtual environment.
- **MIT license**, a first README and [`NOTES.md`](NOTES.md), which explains every step in plain language.
- **An environment check**, [`scripts/check_env.py`](scripts/check_env.py), run before any real work.
- **Model names** fixed in the plan: `rules` (baseline without learning), `eyeTrack0.1`, `eyeTrack0.5`, `eyeTrack1`.

## Why

Datasets, logs and credentials must never end up in git. One of the datasets (UTA-RLDD) shows people who did not all
agree to be published, so this mattered before the first line of model code. Checking the hardware first also
showed early what is and is not possible on this laptop.

## Results

From `python scripts/check_env.py`:

| check | result |
|---|---|
| Apple GPU (MPS) | available; a small eye-sized CNN runs 3.6x faster than on the CPU (27.0 vs 98.3 ms per batch of 256 crops) |
| Kaggle token | found, private permissions (`0o600`) |
| free disk | 71 GB of 494 GB, so the full UTA-RLDD (111 GB) cannot fit: a smaller version is needed |
| MediaPipe | the Python face landmarker can output blendshapes and the head-pose matrix |

## Try it

```bash
git checkout step-01-project-skeleton
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```
