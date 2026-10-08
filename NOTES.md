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

## Step 2: training monitor

Before training anything real, I built a live dashboard ([`tools/monitor/`](tools/monitor/)) so every run can be
watched while it happens instead of judged only at the end. Training scripts write one line per event to
`runs/<name>/metrics.jsonl` through [`src/runlog.py`](src/runlog.py); a small local server reads those files and the
page redraws every 3 seconds: loss and accuracy curves, ETA, per-class precision/recall, a confusion matrix, sample
predictions with mistakes outlined in red, progress bars for downloads, and a comparison of runs. Writing the log is
cheap, 189 µs per training step and 4.8 ms per epoch (measured by [`dummy_run.py`](tools/monitor/dummy_run.py)), so the
monitor cannot slow training down. A fake run that was built to overfit tested it: validation loss bottomed out at
epoch 8 and then rose while training loss kept falling, and the monitor flagged this as overfitting. Pressing Stop on
a second fake run ended it cleanly during epoch 9, keeping the best checkpoint from epoch 8.

**Overfitting, in one sentence:** the model starts memorizing its training examples instead of learning the general
pattern, which shows up as training loss still falling while validation loss (on examples it never trains on) rises.

## Step 3: download the data

[`download_data.py`](scripts/download_data.py) downloaded the three datasets (1.76 GB on disk in total) and checked
them; the job's progress was visible in the training monitor. **MRL Eye** has 84,898 infrared eye crops from 37
people (41,946 closed, 42,952 open); every file name carries the person's ID, which Step 5 needs so that no person
ends up in both training and test data. **CEW** has 4,846 eye patches (2,384 closed, 2,462 open) cut from normal
camera photos, so they look like what a webcam sees rather than an infrared sensor. For **UTA-RLDD** we did not
download the 111 GB of videos: a public Kaggle version (1.33 GB) already contains MediaPipe face features for all
60 participants at 10 frames per second, 1,115,058 frames, with the authors' official 5-fold split; 168 of its
178 videos pass that dataset's own quality check. It contains no images at all, so no driver's face can ever end up
in this project ([`assets/data_overview.png`](assets/data_overview.png)).

**Two honest trade-offs:** no Kaggle copy of CEW has the full face photos for both classes, only the 24x24 eye
patches, so Step 5 decides how to align them with our live eye crops. And because the RLDD version has no images,
`eyeTrack1` will use MediaPipe's own eye-blink scores instead of `eyeTrack0.5`'s output as its eye-closure signal.
`scripts/clean_data.py` lists or deletes downloaded datasets once they are no longer needed.

## Step 4: explore the data

[`explore_data.py`](scripts/explore_data.py) looked at the data before any model sees it (charts in
[`assets/explore/`](assets/explore/)). **MRL Eye is very uneven between people:** the three largest of the 37 people
contribute 32.8% of all images, and the share of closed eyes per person ranges from 2% to 100%. If the same person
appeared in training and test, a model could simply recognize the person and guess their usual eye state, so the
subject-wise split in Step 5 must balance images and classes, not only count people. **MRL also hides shortcuts:**
only 4.8% of the images from the IDS camera show closed eyes, against 58.4% from the RealSense camera, and bad
lighting comes with more closed eyes (55.8%) than good lighting (38.5%). A model could learn "this camera's look
means open" instead of looking at the eyelid, which is exactly the kind of thing the evaluation must check.

**UTA-RLDD shows the signal is real but weak across people.** Per video, the median PERCLOS (share of frames with
the eyes closed) is 2.2% for alert, 4.7% for low vigilant and 9.8% for drowsy drivers, and within the same person the
drowsy video has the higher PERCLOS for 42 of 53 people. But the dots of the three classes overlap heavily between
people, low vigilant sits close to alert, and yawning is rare in every class (median 0). Person 01, the first ID
and not a hand-picked example, does not close the eyes more when drowsy, but holds the head about 5 to 10 degrees
lower the whole time. Two consequences: features should be compared with the same driver's own normal values, and
the 3-class task will be much harder than alert vs drowsy.

**PERCLOS, in one sentence:** the percentage of time the eyes are (almost) closed over a time window, the most
widely used drowsiness measure in driver monitoring research.
