"""Step 11: the live pipeline on real face photos of TEST people: does it see open and closed eyes?

The browser demo finds the face with MediaPipe, cuts each eye with the aligned crop (Step 9) and asks
an ONNX eye model. This script does exactly that in Python, on whole face photos:
  open eyes    LFW photos of the people whose CEW open eyes are in the test split
  closed eyes  CEW's own closed-eye face photos whose eyes are in the test split
so no photo here was seen in training. Step 9 showed that Python and the browser find the same eye
corners and cut the same crops, and tools/onnx_check.html that the browser's ONNX models give
PyTorch's answers, so these numbers carry over to the demo.

Three ways to decide "closed", on the same faces:
  rules        MediaPipe's eyeBlink blendshape >= 0.5 (the rules baseline)
  eyeTrack0.1  P(closed) >= 0.5 from web/models/eyeTrack0.1.onnx
  eyeTrack0.5  P(closed) >= 0.5 from web/models/eyeTrack0.5.onnx

Usage:
  python scripts/download_data.py lfw cew_faces      # once
  python scripts/check_live_pipeline.py

Outputs: models/live_pipeline_check.txt, assets/live_pipeline.png
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eyecrop import eye_corners, eye_crop, load_spec, to_gray  # noqa: E402
from src.runlog import RunLog  # noqa: E402
from src.style import COLORS, finish, setup  # noqa: E402

DATA = ROOT / "data"
LFW = DATA / "lfw" / "lfw-deepfunneled" / "lfw-deepfunneled"
CLOSED_FACES = DATA / "cew_faces" / "dataset_B_FacialImages_highResolution"
MODELS = ["eyeTrack0.1", "eyeTrack0.5"]
CLOSED_NAME = re.compile(r"^(closed_eye_\d+)\.\w+_face_\d+\.jpg$")
OPEN_EYE = re.compile(r"^(.+_\d{4})_[LR]\.jpg$")
BLENDSHAPE = {"right": "eyeBlinkRight", "left": "eyeBlinkLeft"}


def test_photos():
    eyes = np.load(DATA / "processed" / "eyes32.npz")
    test_groups = set(eyes["group"][(eyes["split"] == 2) & (eyes["source"] == 1)])
    closed = [p for p in sorted(CLOSED_FACES.glob("*.jpg"))
              if (m := CLOSED_NAME.match(p.name)) and f"cew_closed_{m.group(1)}" in test_groups]
    photos = set()
    for p in sorted((DATA / "cew").rglob("openEyes/*.jpg")):
        m = OPEN_EYE.match(p.name)
        if m and f"cew_open_{m.group(1)[:-5]}" in test_groups:
            photos.add(m.group(1))
    opened = [LFW / name[:-5] / f"{name}.jpg" for name in sorted(photos)]
    return [(p, 0) for p in closed] + [(p, 1) for p in opened if p.exists()]


def main():
    setup()
    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(DATA / "mediapipe" / "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True))
    manifest = json.loads((ROOT / "web" / "models" / "manifest.json").read_text())
    models = {m["version"]: (ort.InferenceSession(str(ROOT / "web" / m["file"])), m["mean"], m["std"])
              for m in manifest["models"] if m["version"] in MODELS}
    spec = load_spec()
    photos = test_photos()
    print(f"Test photos: {sum(y == 0 for _, y in photos)} with closed eyes (CEW), {sum(y == 1 for _, y in photos)} "
          f"with open eyes (LFW)")

    rows, no_face = [], {0: 0, 1: 0}
    with RunLog("_live_pipeline_check", kind="job", unit="photos") as log:
        for n, (path, y) in enumerate(photos, 1):
            rgb = np.asarray(Image.open(path).convert("RGB"))
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            log.progress(n, len(photos), message=path.name)
            if not result.face_landmarks:
                no_face[y] += 1
                continue
            h, w = rgb.shape[:2]
            pts = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])
            bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
            gray = to_gray(rgb)
            crops = np.stack([eye_crop(gray, *eye_corners(pts, eye), spec) for eye in ("right", "left")])
            row = {"y": y, "rules": [bs[BLENDSHAPE[e]] for e in ("right", "left")]}
            for name, (session, mean, std) in models.items():
                x = ((crops / 255 - mean) / std).astype(np.float32)[:, None]
                z = session.run(None, {"input": x})[0]
                row[name] = list(1 / (1 + np.exp(z[:, 1] - z[:, 0])))  # P(closed) per eye
            rows.append(row)

    y = np.repeat([r["y"] for r in rows], 2)  # both eyes of a photo share its label
    lines = [f"Live pipeline on real face photos of test people only (MediaPipe -> aligned crop -> model)",
             f"{len(rows)} faces, {len(y)} eyes; MediaPipe found no face in {no_face[0]} closed-eye and "
             f"{no_face[1]} open-eye photos", "",
             f"{'method':<14}{'accuracy':>10}{'closed eyes caught':>20}{'open eyes right':>17}"]
    results = {}
    for method in ["rules", *MODELS]:
        closed = np.concatenate([r[method] for r in rows]) >= 0.5
        pred = np.where(closed, 0, 1)
        results[method] = (np.mean(pred == y), np.mean(pred[y == 0] == 0), np.mean(pred[y == 1] == 1))
        lines.append(f"{method:<14}{results[method][0]:>10.1%}{results[method][1]:>20.1%}{results[method][2]:>17.1%}")
    report = "\n".join(lines)
    print("\n" + report)
    (ROOT / "models" / "live_pipeline_check.txt").write_text(report + "\n")

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ["accuracy", "closed eyes caught", "open eyes right"]
    x = np.arange(3)
    for i, method in enumerate(results):
        b = ax.bar(x - 0.3 + 0.2 * (i + 0.5), results[method], 0.18, color=COLORS["series"][i], label=method)
        for bar in b:
            ax.annotate(f"{bar.get_height():.0%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=COLORS["muted"])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, ncols=3, loc="lower left")
    ax.set_title(f"Live pipeline on real faces of test people ({len(y)} eyes)", loc="left", fontsize=13)
    finish(ax)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "live_pipeline.png", dpi=150)
    print("\nSaved models/live_pipeline_check.txt and assets/live_pipeline.png")


if __name__ == "__main__":
    main()
