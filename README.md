# Step 16: final README and cleanup

> This branch is one step of **[eyeTracker](https://github.com/tegemenozyurek/eyeTracker)**, a real-time driver
> monitoring system that runs in the browser. Every step of the project has its own branch, and its README explains
> what was done in that step. The project overview is the README on [`main`](https://github.com/tegemenozyurek/eyeTracker).
>
> Previous: [Step 15: benchmark and charts](https://github.com/tegemenozyurek/eyeTracker/tree/step-15-benchmark) · This is the last step.

## What was done

- **The project README** on [`main`](https://github.com/tegemenozyurek/eyeTracker), in the style of the sister project
  emotionDetecter: badges, the showcase image, how to run the demo, the results of every model side by side, what
  changed between the models, a table of all 16 steps with their branches, how to retrain everything, limitations and
  data credits. Every number in it is read from a report a script wrote.
- **Cleanup:**
  - `kagglehub` and `kagglesdk` removed from `requirements.txt`: the downloads stream from the Kaggle API directly.
  - `scripts/check_env.py` now lists exactly what the code imports.
  - Audit: every Python file compiles, the JavaScript tests pass (10 rule tests, 24 of 24 crop-parity cases), and the
    repository tracks no data, secrets or test leftovers (115 files, 9.3 MB of history).
- **Not done, by choice:** the live GitHub Pages site. The demo runs locally with one command
  (`python3 -m http.server -d web 8010`).

## Why

The README is what an admissions committee reads first. It has to say what the project does, show the results
without overselling them (including that `eyeTrack1` does not beat the rules), and let anyone reproduce everything.

## Try it

```bash
git checkout main
python3 -m http.server -d web 8010      # then open http://localhost:8010
```
