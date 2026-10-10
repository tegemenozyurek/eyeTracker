"""Step 9: compare the full live pipeline in the browser with the training pipeline in Python on real faces.

The parity test (web/eyecrop.test.mjs) proves the crop code is identical. On a real photo two other
things can still differ: MediaPipe's landmarks (the browser and Python run separate builds) and JPEG
decoding. This script crops 20 LFW faces in Python and writes the results; the page
tools/eyecrop_check.html does the same in the browser and prints how far apart the two are.

Usage:
  python scripts/eye_crop_realface.py                   # writes runs/_eyecrop_check/python.json
  python -m http.server 8020                            # from the repo root, then open
                                                        # http://localhost:8020/tools/eyecrop_check.html
"""
import json
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eyecrop import eye_corners, eye_crop, load_spec, to_gray  # noqa: E402

LFW = ROOT / "data" / "lfw" / "lfw-deepfunneled" / "lfw-deepfunneled"
OUT = ROOT / "runs" / "_eyecrop_check" / "python.json"
N = 20


def main():
    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "data" / "mediapipe" / "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_faces=1))
    people = sorted(p for p in LFW.iterdir() if p.is_dir())
    rng = np.random.default_rng(3)
    spec, faces = load_spec(), []
    for person in rng.choice(people, 3 * N, replace=False):
        photo = sorted(person.glob("*.jpg"))[0]
        rgb = np.asarray(Image.open(photo).convert("RGB"))
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            continue
        h, w = rgb.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])
        gray = to_gray(rgb)
        faces.append({"photo": str(photo.relative_to(ROOT)), "eyes": {
            eye: {"corners": [list(map(float, c)) for c in eye_corners(pts, eye)],
                  "crop": np.round(eye_crop(gray, *eye_corners(pts, eye), spec), 3).ravel().tolist()}
            for eye in ("right", "left")}})
        if len(faces) == N:
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"spec": spec, "faces": faces}))
    print(f"Cropped both eyes of {len(faces)} LFW faces in Python -> {OUT.relative_to(ROOT)}")
    print("Now: python -m http.server 8020  (repo root)  and open http://localhost:8020/tools/eyecrop_check.html")


if __name__ == "__main__":
    main()
