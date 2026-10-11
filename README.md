# Step 17: drowsiness improvements

> This branch is one step of **[eyeTracker](https://github.com/tegemenozyurek/eyeTracker)**, a real-time driver
> monitoring system that runs in the browser. Every step of the project has its own branch, and its README explains
> what was done in that step. The project overview is the README on [`main`](https://github.com/tegemenozyurek/eyeTracker).
>
> Previous: [Step 16: final README](https://github.com/tegemenozyurek/eyeTracker/tree/step-16-final-readme) · An extra step after the 16-step plan.

## What was done

**a) The demo's PERCLOS thresholds, learned from data** ([`scripts/tune_rules.py`](scripts/tune_rules.py)). For each
of the 5 folds, the two thresholds (Attention, Take a break) are fitted on the other 4 folds and scored on the 12
drivers left out. Fitting for the best macro F1 alone gives thresholds that warn half of all alert drivers, so the
demo's pair is fitted for the best macro F1 **with at least 70% of alert windows left OK**.

| PERCLOS thresholds (5-fold CV) | 3 classes, macro F1 | alert windows OK | drowsy windows warned |
|---|---:|---:|---:|
| first, hand-set (7.5% / 15%) | 39.8% | 81.6% | 53.5% |
| best F1 only | 48.3% | 50.2% | 82.1% |
| **best F1, ≥ 70% alert OK** (demo: 4.5% / 11.5%) | **44.7%** | **71.2%** | **66.2%** |

The rule tests now use a realistic normal blink (PERCLOS 2.5%; alert drivers in UTA-RLDD: median 1.7%), and the
gauge marks in the demo are drawn from the thresholds.

**b) Blink channels for `eyeTrack1`** ([`src/blink_channels.py`](src/blink_channels.py) =
`blinkChannels` in [`web/features.js`](web/features.js)): eyes closed now, closed for how long (capped at 3 s), and
eyelid speed, computed inside each 60 s window from features the browser already has.

| check | result |
|---|---|
| Python vs browser, 4 real UTA-RLDD windows | within 4.5e-6 |
| selection on validation folds | 48.6% mean macro F1 (current `eyeTrack1`: 48.1%), so it replaces it |
| **test, 5-fold CV** | **3-class accuracy 44.9%** (was 44.4%), macro F1 44.7% (44.4%), alert vs drowsy 66.9% (66.8%) |
| demo path (buffer + channels + ONNX Runtime Web) vs PyTorch | 4 of 4 real windows, within 7.3e-8 |

## Result, honestly

Learned thresholds clearly help the rules (macro F1 +4.9 points with few false alarms). The blink channels help
`eyeTrack1` only within the noise: on new drivers the learned model stays level with the learned rules. A personal
baseline per driver remains what would really help (Step 13 analysis: 49.7%).

![Benchmark](assets/benchmark.png)

## Try it

```bash
git checkout step-17-drowsiness-tuning
python scripts/tune_rules.py
node web/rules.test.mjs && node web/blink_channels.test.mjs
python scripts/train_eyetrack1.py --person-aug 0.5 --lr 1e-3 --epochs 20 --blink-channels
```
