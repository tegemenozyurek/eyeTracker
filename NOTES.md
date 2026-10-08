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
and not a hand-picked example, does not close the eyes more when drowsy (mean eye-closure score 0.220 vs 0.248
when alert), but holds the head lower the whole time (mean pitch -7.9 vs +0.4 degrees). Two consequences: features should be compared with the same driver's own normal values, and
the 3-class task will be much harder than alert vs drowsy.

**PERCLOS, in one sentence:** the percentage of time the eyes are (almost) closed over a time window, the most
widely used drowsiness measure in driver monitoring research.

## Step 5: eye crops and subject-wise splits

[`preprocess.py`](scripts/preprocess.py) turned all 89,744 eyes (84,898 MRL, 4,846 CEW) into 32x32 grayscale crops
and split them into training, validation and test data so that **no person appears in two splits**. This matters
because a model tested on people it has already seen can pass by recognizing faces it memorized, which says nothing
about a new driver. MRL's 37 people were assigned by a seeded search over 20,000 random assignments: 25 people
(66.8% of images) for training, 5 (15.0%) for validation and 7 (18.2%) for testing, with 48% to 50% closed eyes in
every split and every camera present in the test set. CEW has no person IDs for closed eyes, so its eyes are grouped
by source photo (left and right eye of a face stay together) and open eyes by person name. Two problems were caught
and fixed on the way: a file-name pattern silently skipped 140 CEW eyes, and the first split put only 3 people in
the test set, which would have made the test a judgement of 3 individuals.

One camera (Aptina) belongs to only two people, who ended up in validation and test, so the model never trains on
it: a built-in check of how it copes with a camera it has never seen.

## Step 6: define the CNN

The eye model ([`src/model.py`](src/model.py)) is a small convolutional neural network: three blocks that each look
at small 3x3 neighbourhoods of pixels, then halve the image while doubling the number of patterns they track
(32x32 pixels → 16x16 → 8x8 → 4x4), followed by a tiny classifier that outputs two scores, closed and open.
It has 295,266 parameters (1.18 MB), small on purpose because it must run on both eyes of every webcam frame; on the
laptop's CPU both eyes take 0.75 ms ([`model_summary.py`](scripts/model_summary.py)). Before training, it answers
"open" with 53.2% for every eye, which is what an untrained network should do: it has not learned anything yet.

**CNN, in one sentence:** a network that slides small learned filters over the image, so the same edge or curve
detector (an eyelid line, a dark pupil) is recognised wherever it appears.

## Step 7: train eyeTrack0.1

`eyeTrack0.1` is the first real model: the Step 6 CNN trained for 30 epochs on the infrared eyes of the 25 MRL
training people, with no tricks (7.8 minutes on the M4's GPU, [`train.py`](scripts/train.py)). It was watched live in
the training monitor. On the 7 test people it has never seen, it gets **97.8%** right
([`evaluate.py`](scripts/evaluate.py)). Validation accuracy stopped improving at about 99.0% while validation loss rose
from 0.0320 (epoch 10) to 0.0487 (epoch 30): mild overfitting, where the model becomes more confident on its training
eyes without becoming more correct on new ones. The camera shortcut feared in Step 4 did not appear: it catches 98.5%
of the closed eyes from the IDS camera and 100% from the Aptina camera it never trained on (130 and 73 closed eyes, so
small samples).

**The domain gap, measured:** on CEW's normal-camera eyes the same model drops to **89.3%**, 8.5 points lower, and on
the simulated webcam (dim light, motion blur, low resolution, noise and jitter all at once) to 60.6% on MRL and 56.3%
on CEW, barely above the 50% of guessing. Sensor noise alone costs the most (MRL 97.8% → 67.2%). A model that is
excellent on the data it was trained on can be almost useless on a real webcam; that is exactly what `eyeTrack0.5`
must fix.

**Domain gap, in one sentence:** the difference between the kind of images a model was trained on (here: infrared,
sharp) and the kind it meets in use (webcam: visible light, small, noisy), which makes accuracy on the training
kind a poor promise for the real one.
