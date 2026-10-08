# Step 2: training monitor

> This branch is one step of **[eyeTracker](https://github.com/tegemenozyurek/eyeTracker)**, a real-time driver
> monitoring system that runs in the browser. Every step of the project has its own branch, and its README explains
> what was done in that step. The project overview is the README on [`main`](https://github.com/tegemenozyurek/eyeTracker).
>
> Previous: [Step 1: project skeleton](https://github.com/tegemenozyurek/eyeTracker/tree/step-01-project-skeleton) ·
> Next: [Step 3: download the data](https://github.com/tegemenozyurek/eyeTracker/tree/step-03-download-data)

## What was done

A local dashboard to watch training runs and long jobs live, built **before** any real training.

- **[`src/runlog.py`](src/runlog.py):** every training run and long job appends one JSON line per event to
  `runs/<name>/metrics.jsonl` (training steps, epochs with the confusion matrix, sample predictions, job progress,
  start and end). It also notices when the Stop button was pressed.
- **[`tools/monitor/`](tools/monitor/):** the dashboard, started with one command. A small Python server (standard
  library only) and one HTML page with hand-drawn SVG charts, so it needs no extra packages and works offline. It
  refreshes every 3 seconds and shows:
  - loss and accuracy curves, train vs validation, per step and per epoch
  - epoch, step, learning rate, best validation score, elapsed time and ETA
  - an **overfitting warning** when validation loss rises while training loss keeps falling
  - per-class precision / recall / F1 and the confusion matrix
  - sample predictions with mistakes outlined in red (for UTA-RLDD: feature curves only, never faces)
  - progress bars for downloads and feature extraction, a comparison of runs, and a **Stop** button
- **[`tools/monitor/dummy_run.py`](tools/monitor/dummy_run.py):** a fake training run built to overfit (with drawn
  eyes as sample images) and a fake download job, to test every panel without any data.

## Why

From the first real model on, every run is watched while it happens. Overfitting, a stuck run or a crash are visible
after a few epochs instead of after hours of training.

## Results

From `python tools/monitor/dummy_run.py`:

| test | result |
|---|---|
| overfitting warning | validation loss lowest at epoch 8 (0.1979), then rising to 0.3616 by epoch 15 while training loss kept falling: the warning appeared |
| Stop button | pressed during epoch 9 of a second run: it ended as `stopped` and kept epoch 8 (88.1% validation accuracy) as its best |
| cost of logging | 189 µs per training step, 4.8 ms per epoch: far too small to slow training down |

Bugs found while testing and fixed: clicks in the run list and the comparison checkboxes were lost when the page
refreshed, a chart label was cut off at the edge, and a second monitor on the same port crashed instead of pointing
to the one already running.

## Try it

```bash
git checkout step-02-training-monitor
python tools/monitor/app.py                          # opens http://127.0.0.1:8501
python tools/monitor/dummy_run.py --name _try --delay 0.2   # in a second terminal
```

Watch the validation curve turn upward after epoch 8 and the overfitting warning appear, then try Stop.
