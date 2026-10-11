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

## Step 8: web demo with the rules baseline

The first live demo ([`web/`](web/)) runs entirely in the browser: MediaPipe's face landmarker finds 478 face points,
52 expression scores ("blendshapes", such as eyeBlink and jawOpen) and the head's rotation in every webcam frame, and
the `rules` baseline ([`web/rules.js`](web/rules.js)) turns them into **OK**, **Attention** or **Take a break**.
The rules are the classic ones from driver-monitoring research: an eye counts as closed when its eyeBlink score is at
least 0.5 (the same definition as in the RLDD features), PERCLOS is measured over the last 60 s, a single closure of
1.5 s (a microsleep) triggers a break at once, three yawns in five minutes give Attention, and looking more than 25
degrees away from the calibrated road direction for 2 s gives Attention, for 4 s a break. An alert only steps down
after it has been gone for 4 s, so the display does not flicker. The thresholds are provisional; Steps 12 and 13 test
them on UTA-RLDD. Ten automated tests with synthetic frames ([`web/rules.test.mjs`](web/rules.test.mjs)) check every
rule, and one of them caught a wrong expectation of mine: after a 2 s closure the alert correctly falls back to
Attention, not OK, because those 2 seconds raise the last minute's PERCLOS above 7.5%.

**EAR (eye aspect ratio), in one sentence:** the eye's height divided by its width, measured from six landmarks
around the eye; it drops towards zero as the eyelid closes (Soukupová and Čech, 2016).

## Step 9: one eye-crop function in Python and JavaScript

A model only works live if the eyes it sees in the browser look like the eyes it was trained on. The eye crop is
now one function written twice, line by line: [`src/eyecrop.py`](src/eyecrop.py) for training and
[`web/eyecrop.js`](web/eyecrop.js) for the browser. It takes the two corners of an eye from MediaPipe, rotates and
scales the image so the corners land on fixed points of a 32x32 crop, and samples the pixels with hand-written code
instead of OpenCV or the browser's canvas, which would resample differently. A parity test
([`web/eyecrop.test.mjs`](web/eyecrop.test.mjs)) runs both on 24 hard cases (tilted, tiny, huge and half-outside eyes,
on a noise image that punishes any difference): all 24 match within 5e-7 gray levels, also when the browser reads
only the small box of pixels around the eye.

**How big should the eye be in the crop?** Measured, not guessed: CEW's open eyes were cut from LFW photos, so
[`calibrate_eye_crop.py`](scripts/calibrate_eye_crop.py) ran MediaPipe on 600 of those photos and tried 84 framings.
The eye spanning 60% of the crop width, vertically centred, is the most similar to the CEW patches (mean correlation
0.801, against 0.763 for my first guess of 70%). On 20 real LFW faces the full browser pipeline and the Python
pipeline find the same eye corners (less than 0.005 px apart) and their crops differ by a mean of 0.12 gray levels out
of 255, only because the browser and Python decode JPEG images slightly differently
([`tools/eyecrop_check.html`](tools/eyecrop_check.html)). The average MRL eye is framed tighter than ours, one more
reason why `eyeTrack0.5` trains with random zoom and shift.

**Parity test, in one sentence:** a test that feeds the same inputs to two implementations that must agree (here
Python for training and JavaScript for the browser) and fails if they differ, so a silent mismatch cannot creep in.

## Step 10: train eyeTrack0.5

`eyeTrack0.5` is the same network as `eyeTrack0.1`, trained differently: on infrared MRL eyes **and** normal-camera
CEW eyes (CEW drawn often enough to make up 30% of every epoch, although it is only 6% of the data), and every
training eye is randomly degraded the way a car webcam would degrade it: mirrored, rotated, zoomed, shifted, darker or
brighter, blurred, lower in resolution, noisy ([`src/augment.py`](src/augment.py), preview in
[`assets/augmentation.png`](assets/augmentation.png)). A first version of the augmentation was too harsh (many dark
infrared eyes became black noise that not even a person could label) and was toned down before training.

**Result** ([`compare_eye_models.py`](scripts/compare_eye_models.py), same 7 + 354 test people for both models): on the
simulated webcam with all effects at once, accuracy rises from 60.6% to **78.2%** on MRL and from 56.3% to **84.7%** on
CEW; on clean CEW eyes from 89.3% to **96.7%**, and clean MRL stays where it was (97.8% → 98.1%). Sensor noise alone
no longer hurts (MRL 67.2% → 95.7%). Training accuracy is now below validation accuracy, because the model trains on
the harder, degraded eyes; validation loss stays flat, so there is no overfitting. The price: on clean MRL eyes it
catches slightly fewer closed eyes (98.2% → 96.9%). One caveat, stated openly: the training augmentation and the
simulated-webcam test use the same kinds of effects, so `eyeTrack0.5` was prepared for that test by design; the real
judge is the live webcam.

**Data augmentation, in one sentence:** showing the model a randomly altered version of each training image every
time, so it learns what stays the same (the eyelid) instead of details that change between cameras (noise, light, blur).

## Step 11: export to ONNX and plug into the demo

Both eye models now run in the browser. [`export_web.py`](scripts/export_web.py) converts them from PyTorch to ONNX,
a format the browser can run with ONNX Runtime Web, and checks that nothing changed on the way: on 512 test eyes the
ONNX models give the same answer as PyTorch every time (largest difference in the raw scores 2.3e-05), and in the
browser itself the same holds for 16 test eyes, with both eyes classified in about 3 ms
([`tools/onnx_check.html`](tools/onnx_check.html)). In the demo, picking `eyeTrack0.1` or `eyeTrack0.5` keeps all the
rules, but "is this eye closed?" is now answered by the CNN on the aligned eye crop instead of MediaPipe's blink score.

**The most meaningful check so far** ([`check_live_pipeline.py`](scripts/check_live_pipeline.py)) runs the whole live
pipeline (MediaPipe, aligned crop, model) on real face photos of test people only: 172 CEW photos with closed eyes and
191 LFW photos with open eyes (716 eyes; MediaPipe missed 5 faces). The rules baseline is right for 89.5% of eyes but
catches only 79.9% of closed eyes; `eyeTrack0.1` reaches 91.5% (89.8% of closed eyes), and `eyeTrack0.5` **96.5%**
(**95.5%** of closed eyes, 97.4% of open eyes). Because Step 9 showed that the browser finds the same eye corners and
cuts the same crops as Python, these numbers carry over to the demo.

**ONNX, in one sentence:** an open file format for trained neural networks, so a model trained in PyTorch can run
somewhere else, here in the browser, without PyTorch.

## Step 12: UTA-RLDD features and windows

[`prepare_rldd.py`](scripts/prepare_rldd.py) turned the 168 quality-checked UTA-RLDD videos (60 people, 29.3 hours)
into what `eyeTrack1` will learn from: 13 numbers per frame, five frames per second (eye-blink, squint and look-down
scores for each eye, jaw opening, eye and mouth openness ratios, and the head's pitch, yaw and roll). Every one of
them is defined exactly as in the dataset's own extraction code, because the browser has to compute the same numbers
live in Step 14. The frames are cut into 8,817 windows of 60 seconds, a new one every 10 seconds, kept only when the
face was found in at least 70% of the frames; each window gets its video's label. The authors' five folds (12 people
each) keep every person in exactly one fold, so a model is always tested on drivers it never trained on.

To compare `eyeTrack1` with exactly what the demo does today, every video was also replayed through the demo's own
rules ([`web/rules.js`](web/rules.js), via [`tools/rules_on_rldd.mjs`](tools/rules_on_rldd.mjs)). The median
PERCLOS per window rises from 1.7% (alert) to 4.0% (low vigilant) and 8.3% (drowsy), but the rules still say "OK" for
44.2% of the drowsy windows and "Take a break" for 8.3% of the alert ones: that is the bar `eyeTrack1` has to clear.

**Window, in one sentence:** a fixed stretch of time (here 60 seconds of frames) that the model looks at as a whole,
because drowsiness shows in how the eyes and head behave over time, not in a single frame.

## Step 13: train eyeTrack1

`eyeTrack1` is a small recurrent network (a GRU, 3,699 parameters, [`src/temporal.py`](src/temporal.py)) that reads one
minute of the 13 face features, frame by frame, and says alert, low vigilant or drowsy. It was tested with 5-fold
cross-validation ([`train_eyetrack1.py`](scripts/train_eyetrack1.py)): each fold's 12 people are judged by a model
that never saw them, so all 60 drivers are tested once.

**What happened, honestly.** The first, larger model memorized the drivers: 98.6% right on its 36 training people,
32.7% (chance) on new ones. Each driver has exactly one video per class, so the network learned "this face, in
this video" instead of drowsiness. A much smaller network with strong regularization and a trick that simulates new
drivers (every training window gets a random shift and scale of each feature) helped; the settings were chosen on the
validation folds only ([`selection.txt`](models/eyeTrack1/selection.txt)). The result on the test people is
**sobering**: 44.4% window accuracy over three classes (chance 33.3%), against 44.8% for the demo's rules and 48.1%
for a single PERCLOS threshold pair fitted on the training folds; alert vs drowsy 66.8% against 68.7% for the rules.
The dataset's authors report 54% per blink sequence and 65.2% per video with blink features (Ghoddoosian et al.
2019, Table 2); `eyeTrack1` is clearly below that. With 36 training people, a learned model cannot yet beat a
well-chosen PERCLOS threshold on new drivers.

**What does help** (an analysis, not usable in the demo): comparing each driver with their own average across all
their recordings lifts the same network to 49.7% (3 classes) and 72.4% (alert vs drowsy), above every rule. Step 4
had already shown why: differences between people are larger than differences between alert and drowsy in one
person. A real car could learn the driver's personal baseline over many drives; this demo cannot.

**Cross-validation, in one sentence:** split the people into five groups, train five times, each time testing on the
group left out, so every person is tested once by a model that never saw them and the result does not depend on one
lucky split.

## Step 14: eyeTrack1 in the demo

`eyeTrack1` now runs in the browser. The demo samples the 13 face features five times per second into a one-minute
buffer, exactly as the training windows were built, and every 2 seconds the GRU (17 kB as ONNX) judges the last
minute; a new "Driver state" panel shows its three probabilities. With `eyeTrack1` selected it replaces the PERCLOS
and yawn rules, while the microsleep and eyes-off-the-road rules stay on as safety nets.

The risky part was making the live features identical to the training features, because they come from a dataset
someone else extracted. [`web/features.js`](web/features.js) and [`src/rldd_features.py`](src/rldd_features.py) copy the
dataset's own formulas, and on 12 real faces ([`tools/feature_check.html`](tools/feature_check.html)) the browser
and Python agree within 1e-4 for every feature and 0.002 degrees for the head angles. That check also settled the
head-pose matrix layout by measurement: read the other way round, the angles would be off by up to 59 degrees. The
demo's whole `eyeTrack1` path (sampling, buffer, normalization, ONNX Runtime Web) gives PyTorch's answer on 4 real
UTA-RLDD windows, within 1.25e-7 ([`tools/onnx_check.html`](tools/onnx_check.html)).

**Feature parity, in one sentence:** making sure the numbers a model receives live are computed exactly like the ones
it was trained on, because a model can only be as right as its inputs are faithful.

## Step 15: benchmark and charts

All results are now in one place, [`models/benchmark.txt`](models/benchmark.txt) and
[`assets/benchmark.png`](assets/benchmark.png), produced by [`benchmark.py`](scripts/benchmark.py) from the reports the
earlier steps wrote, so no number is typed by hand. For the eyes, `eyeTrack0.5` is the clear winner: on real face
photos through the whole live pipeline it gets 96.5% of eyes right and catches 95.5% of closed eyes, against 89.5% and
79.9% for the rules. For drowsiness, no learned model beats a well-chosen PERCLOS threshold on new drivers; only the
personal-baseline analysis does.

Speed was measured in a real browser ([`browser_benchmark.py`](scripts/browser_benchmark.py) with
[`tools/benchmark.html`](tools/benchmark.html), median of 200 frames in Chrome on the M4): MediaPipe's face landmarker
takes 8.5 ms, the two eye crops 2.0 ms, `eyeTrack0.5` 3.1 ms for both eyes, the features and rules less than 0.1 ms,
and `eyeTrack1` 0.2 ms for a full minute. One frame takes 13.6 ms, so the pipeline could handle about 74 frames per
second, more than a webcam delivers. The page was not visible on screen during the run, which only slows a browser
down, so these numbers are if anything pessimistic. The README header image ([`showcase.png`](assets/showcase.png))
shows 16 random eyes of test people with `eyeTrack0.5`'s answer; it got all 16 right, which at 96.7 to 98.1% accuracy is
a likely but slightly lucky draw.
