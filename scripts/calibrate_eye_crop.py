"""Step 9: measure how big and where the eye should sit in our 32x32 crop, so that live webcam crops
look like the CEW training patches.

CEW's open eyes were cut from LFW photos, and their file names say which photo (Aaron_Guiel_0001_L.jpg).
For a sample of them this script runs MediaPipe's face landmarker (the model the browser uses) on the
LFW photo, cuts our aligned crop with every candidate framing (eye width as a share of the crop, height
of the eye in the crop), and measures how similar it is to the CEW patch (normalized correlation, 1 = same
picture up to brightness and contrast). The framing with the highest mean similarity is saved.

Note: Kaggle's LFW copy is "deep-funneled" (each photo slightly re-aligned). The framing is measured
relative to the eye corners found in the same photo, so a whole-photo re-alignment does not change it.

Outputs:
  models/eye_crop.json         the crop spec used by training and the live demo (copied to web/)
  assets/eye_crop/calibration.png   similarity for every framing, and CEW patches next to our crops
  assets/eye_crop/mean_eyes.png     average eye of CEW, of our crops, and of MRL
"""
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eyecrop import DEFAULT_SPEC, SPEC_FILE, eye_corners, eye_crop, to_gray  # noqa: E402
from src.runlog import RunLog  # noqa: E402
from src.style import COLORS, setup  # noqa: E402

DATA = ROOT / "data"
LFW = DATA / "lfw" / "lfw-deepfunneled" / "lfw-deepfunneled"
MODEL = DATA / "mediapipe" / "face_landmarker.task"
ASSETS = ROOT / "assets" / "eye_crop"
NAME = re.compile(r"^(.+)_(\d{4})_([LR])\.jpg$")
WIDTHS = np.round(np.arange(0.40, 0.96, 0.05), 2)
HEIGHTS = np.round(np.arange(0.35, 0.66, 0.05), 2)
SAMPLE = 600


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def cew_patch(path, size=32):
    """The CEW patch as preprocess.py turns it into training data (24 px -> 32 px bilinear)."""
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float64)


def landmarks(detector, path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None, None
    h, w = rgb.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])
    return to_gray(rgb), pts


def main():
    setup()
    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.IMAGE, num_faces=1))
    files = sorted((DATA / "cew").rglob("openEyes/*.jpg"))
    rng = np.random.default_rng(0)
    files = [files[i] for i in rng.choice(len(files), min(SAMPLE, len(files)), replace=False)]

    samples, missing, no_face = [], 0, 0
    with RunLog("_calibrate_eye_crop", kind="job", unit="eyes") as log:
        for n, path in enumerate(files, 1):
            m = NAME.match(path.name)
            photo = LFW / m.group(1) / f"{m.group(1)}_{m.group(2)}.jpg"
            if not photo.exists():
                missing += 1
                continue
            gray, pts = landmarks(detector, photo)
            if pts is None:
                no_face += 1
                continue
            samples.append((path, m.group(3), gray, pts, cew_patch(path)))
            log.progress(n, len(files), message=path.name)
    print(f"{len(files)} CEW open eyes sampled: {len(samples)} usable, {missing} LFW photo not found, "
          f"{no_face} without a face found by MediaPipe")

    # Which of MediaPipe's eyes is CEW's "L"? Decide by similarity with the default framing.
    votes = {"right": 0, "left": 0}
    for _, side, gray, pts, patch in samples:
        scores = {eye: ncc(eye_crop(gray, *eye_corners(pts, eye), DEFAULT_SPEC), patch) for eye in votes}
        best = max(scores, key=scores.get)
        votes[best if side == "L" else ("left" if best == "right" else "right")] += 1
    l_is = max(votes, key=votes.get)
    eye_of = {"L": l_is, "R": "left" if l_is == "right" else "right"}
    print(f"CEW 'L' patches match MediaPipe's {l_is} eye in {votes[l_is]} of {len(samples)} cases")

    grid = np.zeros((len(HEIGHTS), len(WIDTHS)))
    for i, cy in enumerate(HEIGHTS):
        for j, w in enumerate(WIDTHS):
            spec = {**DEFAULT_SPEC, "eye_width": float(w), "center_y": float(cy)}
            grid[i, j] = np.mean([ncc(eye_crop(g, *eye_corners(p, eye_of[s]), spec), patch)
                                  for _, s, g, p, patch in samples])
    bi, bj = np.unravel_index(grid.argmax(), grid.shape)
    spec = {**DEFAULT_SPEC, "eye_width": float(WIDTHS[bj]), "center_y": float(HEIGHTS[bi])}
    default_score = grid[list(HEIGHTS).index(0.5), list(WIDTHS).index(0.7)]
    print(f"Best framing: eye width {spec['eye_width']:.2f} of the crop, eye centre at {spec['center_y']:.2f} "
          f"of the height; mean similarity {grid[bi, bj]:.3f} (first guess 0.70 / 0.50: {default_score:.3f})")
    spec["calibration"] = {"cew_open_eyes": len(samples), "mean_ncc": round(float(grid[bi, bj]), 4),
                           "cew_L_is_mediapipe": l_is}
    SPEC_FILE.write_text(json.dumps(spec, indent=2) + "\n")
    shutil.copy(SPEC_FILE, ROOT / "web" / "eye_crop.json")
    print(f"Saved {SPEC_FILE.relative_to(ROOT)} and web/eye_crop.json")

    ASSETS.mkdir(parents=True, exist_ok=True)
    plot_calibration(grid, spec, samples, eye_of)
    plot_means(spec, samples, eye_of)
    print(f"Saved {ASSETS.relative_to(ROOT)}/calibration.png, mean_eyes.png")


def plot_calibration(grid, spec, samples, eye_of, n=8):
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(2, n + 5, width_ratios=[1] * 4 + [0.35] + [0.55] * n, hspace=0.25)  # column 4: gap for the colour bar
    ax = fig.add_subplot(gs[:, :4])
    im = ax.imshow(grid, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(len(WIDTHS)), [f"{w:.2f}" for w in WIDTHS], rotation=60, fontsize=8)
    ax.set_yticks(range(len(HEIGHTS)), [f"{h:.2f}" for h in HEIGHTS], fontsize=8)
    ax.set_xlabel("eye width (share of the crop)")
    ax.set_ylabel("eye centre (share of the height)")
    bi, bj = list(HEIGHTS).index(spec["center_y"]), list(WIDTHS).index(spec["eye_width"])
    ax.scatter([bj], [bi], marker="x", color=COLORS["val"], s=60)
    ax.set_title(f"Mean similarity to CEW (best {grid[bi, bj]:.3f})", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for k, (path, side, g, p, patch) in enumerate(samples[:n]):
        top, bottom = fig.add_subplot(gs[0, 5 + k]), fig.add_subplot(gs[1, 5 + k])
        top.imshow(patch, cmap="gray", vmin=0, vmax=255)
        bottom.imshow(eye_crop(g, *eye_corners(p, eye_of[side]), spec), cmap="gray", vmin=0, vmax=255)
        for a in (top, bottom):
            a.axis("off")
        if k == 0:
            top.set_title("CEW", fontsize=9, loc="left")
            bottom.set_title("ours", fontsize=9, loc="left")
    fig.suptitle("Eye-crop calibration: our MediaPipe-aligned crop vs the CEW patch from the same photo",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(ASSETS / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_means(spec, samples, eye_of):
    eyes = np.load(DATA / "processed" / "eyes32.npz")
    mrl_open = eyes["X"][(eyes["source"] == 0) & (eyes["y"] == 1)].astype(np.float64)
    means = [("CEW open eyes", np.mean([s[4] for s in samples], axis=0)),
             ("our crops, same photos", np.mean([eye_crop(g, *eye_corners(p, eye_of[sd]), spec)
                                                 for _, sd, g, p, _ in samples], axis=0)),
             ("MRL open eyes", mrl_open.mean(axis=0))]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    for ax, (title, img) in zip(axes, means):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Average eye (no single person recognizable)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "mean_eyes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
