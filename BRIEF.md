# eyeTracker: Project Brief for Claude Code

**Repo:** https://github.com/tegemenozyurek/eyeTracker.git
**Owner:** Turgut Egemen Özyürek (GitHub: tegemenozyurek)
**Deadline:** 3 days, with **3 trained models**, a live browser demo and a polished README
**Machine:** MacBook Air, Apple M4, PyTorch with the MPS backend

---

## 1. What we are building

A **real-time driver monitoring system** that runs in the browser on a normal webcam. It detects
**drowsiness** (eye closure, PERCLOS, slow blinks, yawning, head nodding) and **distraction**
(looking away from the road) and raises alert levels.

The project will be shown to master's admission committees in Germany (Automotive Software Engineering,
Computer Science, AI). It must look and read like a careful, measured engineering and research project.

**Sister project:** `~/Projects/ai/emotionDetecter` (public: github.com/tegemenozyurek/emotionDetecter).
Reuse its structure, style and as much infrastructure as possible:
- MediaPipe face detection/landmarks in the browser, face tracking between frames, score smoothing
- ONNX export and ONNX Runtime Web (WebGPU with WASM fallback)
- Webcam-style degradations in training (dim light, blur, motion blur, low resolution, noise, detector jitter)
- One script per step that prints results and saves a chart
- `models/<version>/` (weights, config, train log, test report) and `assets/<version>/` (charts)
- README format: badges, live demo link, results table, "what changed between models", "how it was built" table, limitations, acknowledgements
- GitHub Pages deployment via `.github/workflows/pages.yml`

Read the emotionDetecter repo first and copy patterns from it rather than reinventing them.

---

## 2. The 3 models (named eyeTrack0.1, eyeTrack0.5, eyeTrack1)

Plus one **non-ML reference baseline** (`rules`) that is not counted as a model.

| version | what it is | trained on | the point it proves |
|---|---|---|---|
| `rules` | Eye Aspect Ratio (EAR) + PERCLOS + yawn (jawOpen) + head-pose thresholds | nothing | the classic baseline to beat |
| `eyeTrack0.1` | small CNN, eye crop → open/closed | MRL Eye Dataset (infrared) | works in-domain, but measure the drop on RGB webcam-like eyes (domain gap) |
| `eyeTrack0.5` | same CNN family + RGB data + MediaPipe-aligned eye crops + webcam augmentations | MRL Eye + CEW (RGB) | fixing the domain gap: cross-dataset accuracy and live-demo robustness |
| `eyeTrack1` | temporal model (GRU or 1D-CNN) over per-frame features → alert / low-vigilant / drowsy | UTA-RLDD (real drowsiness videos) | real drowsiness, not just "eyes closed" |

**Per-frame features for `eyeTrack1`:** left and right eye-closure probability from `eyeTrack0.5`, MediaPipe blendshapes
(eyeBlinkLeft/Right, jawOpen, and others if useful), head pose (pitch, yaw, roll from the facial
transformation matrix), EAR, plus derived blink duration and blink rate. Use sliding windows (for example
60 s at 5 fps). Compare `eyeTrack1` against `rules` (PERCLOS thresholds) on the same windows.

**Important:** eye crops must be produced by **the exact same alignment and crop function** in training
(Python) and live (JS), as emotionDetecter does for faces. Write a parity test.

---

## 3. Data (do not commit datasets; add them to .gitignore)

| dataset | use | notes |
|---|---|---|
| **MRL Eye Dataset** | `eyeTrack0.1`, `eyeTrack0.5` | Infrared eye crops, open/closed labels. Mirrors on Kaggle. Check the license and cite the original authors |
| **CEW (Closed Eyes In The Wild)** | `eyeTrack0.5` | RGB eye patches, open/closed. Mirrors on Kaggle. Cite the original paper |
| **UTA-RLDD** | `eyeTrack1` | ~30 h, 60 participants, labels alert(0) / low vigilant(5) / drowsy(10). **111 GB in total**, so DO NOT download all of it. First check Kaggle mirrors (smaller, sometimes frame-extracted), otherwise download a subset of participants. Subsample to ~5 fps for feature extraction |

**UTA-RLDD rules (must follow):**
- Cite: Ghoddoosian, Galib, Athitsos, *A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection*, CVPR Workshops 2019 (arXiv:1904.07312).
- Use **subject-wise splits only** (no person in both train and test). The authors recommend 5-fold cross-validation. Use their fold protocol if available, otherwise at least one held-out fold.
- **Never publish any RLDD face image** in the README, assets, or demo. Only 36 of 60 participants consented, and identities must not be revealed. Show charts and numbers only.
- Look up the paper's reported baseline and compare honestly. Quote the exact number with the citation, never from memory.

A Kaggle token is already on this machine (`~/.kaggle`). Keep secrets out of the repo.

---

## 4. Web demo (`web/`)

- Webcam in the browser. MediaPipe Face Landmarker (tasks-vision): 478 landmarks, blendshapes, transformation matrix.
- Live panels: eye state per eye, blink counter and duration, **PERCLOS (rolling 60 s)**, yawn counter,
  head pose, "eyes off road" timer, and a driver state from the selected model.
- Alert levels: OK / Attention / Take a break, with clear, non-flickering UI (reuse emotionDetecter's smoothing).
- Model picker: `rules`, `eyeTrack0.1`, `eyeTrack0.5`, `eyeTrack1`, with hover cards explaining what changed (as in emotionDetecter).
- Debug toggle: show the aligned eye crops that the model receives.
- Privacy line: video never leaves the device.
- Deploy to GitHub Pages and link it at the top of the README.

---

## 5. Evaluation (what goes in the README results table)

- `eyeTrack0.1` vs `eyeTrack0.5`: accuracy and F1 on MRL test, CEW test, and a **simulated webcam** version of both
  (same degradation pipeline as emotionDetecter). Confusion matrices.
- `rules` vs `eyeTrack1`: accuracy, macro-F1, per-class recall on held-out RLDD subjects (3-class), plus
  binary alert-vs-drowsy. Report window length and fps.
- Training time per model on the M4 (MPS), parameter count, browser inference time per frame.
- Honest limitations: self-reported RLDD labels, IR vs RGB gap, glasses and sunglasses, night driving, no real-car test.

---

## 6. Three-day plan (16 steps, stop and report after each)

**Day 1: foundations + first model + first demo**
1. Read emotionDetecter; project skeleton, `.gitignore`, `requirements.txt`, MIT license.
2. Training monitor GUI (section 8), tested with a dummy run before any real training.
3. Download MRL + CEW; start the RLDD subset download in the background (the long pole).
4. Explore data: class balance, sample grid, subject counts.
5. Preprocess: eye crops, **subject-wise splits** (MRL has subject IDs, so no person in both train and test).
6. Define the CNN and print its summary.
7. Train `eyeTrack0.1` (watched live in the monitor); evaluate on MRL test, CEW test and simulated webcam.
8. Web demo v0 with the `rules` baseline (EAR, PERCLOS, yawn, head pose) working live.

**Day 2: second model + temporal features**
9. Eye alignment/crop function in JS identical to Python, with a parity test.
10. Train `eyeTrack0.5` (MRL + CEW, aligned crops, webcam augmentations); benchmark against `eyeTrack0.1`.
11. Export to ONNX, verify against PyTorch, plug into the web demo.
12. RLDD feature extraction at ~5 fps (progress visible in the monitor): per-frame features → windows → subject-wise folds.

**Day 3: third model + polish**
13. Train `eyeTrack1` (GRU or 1D-CNN); evaluate against `rules` on held-out subjects.
14. Export `eyeTrack1` to ONNX and add the driver-state panel to the demo.
15. Benchmark script, charts for all models, showcase image.
16. README (emotionDetecter style), GitHub Pages live, final cleanup and final report.

**If time runs short:** cut scope in this order: (a) the 3-class RLDD task (keep binary alert vs drowsy),
(b) fewer RLDD participants, (c) demo extras. Never cut honest evaluation or the subject-wise split.

---

## 7. Working style

- **Commit after each step** with messages like `Step 4: train eyeTrack0.1 (xx.x% test)`, as in emotionDetecter.
- Every script prints its key numbers and saves a chart in `assets/`.
- File and folder names in **English**.
- **Explain as you go:** after each step, write 3–5 plain-language sentences in `NOTES.md` (what was done,
  why, what the result means). Egemen must be able to explain every model in an interview: overfitting,
  augmentation, domain gap (IR vs RGB), PERCLOS, why subject-wise splits matter, what the GRU sees.
- Do not invent numbers. Every number in the README must come from a script output in the repo.
- Ask before downloading more than ~20 GB or running jobs longer than ~2 hours.

---

## 8. Training monitor GUI (`tools/monitor/`)

A local dashboard to watch training and long jobs live, started with **one command**
(e.g. `python tools/monitor/app.py`, opens at `http://localhost:8501` or similar).

- All training and long scripts append events to `runs/<version>/metrics.jsonl` (step, epoch, train/val loss,
  train/val accuracy, learning rate, time per epoch, best checkpoint, etc.). `runs/` is git-ignored.
- The dashboard auto-refreshes every few seconds and shows:
  - **live loss and accuracy curves** (train vs val) per step and per epoch
  - current epoch/step, elapsed time, **ETA**, learning rate, best val score and when it happened
  - **overfitting warning** when val loss rises while train loss falls
  - per-class precision/recall and a **confusion matrix** refreshed at each epoch end
  - **sample predictions** (eye crops with true vs predicted label, misclassified ones highlighted).
    For RLDD, never show faces: plot feature curves only.
  - **run comparison**: overlay curves of `eyeTrack0.1`, `eyeTrack0.5`, `eyeTrack1`
  - **progress bars** for long non-training jobs (dataset download, RLDD feature extraction)
  - a **Stop training** button (writes a stop file; the trainer saves the best checkpoint and exits cleanly)
- Keep it simple and robust (Streamlit, or a small Python server + one HTML page). It must not slow training down.
