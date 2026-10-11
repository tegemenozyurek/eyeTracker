"""Step 14: the 13 eyeTrack1 features computed in Python (src/rldd_features.py, the dataset's definitions) on
real LFW faces, for comparison with the browser (web/features.js via tools/feature_check.html).

Usage:
  python scripts/feature_parity.py          # writes runs/_feature_check/python.json
  python -m http.server 8020                # repo root, then open http://localhost:8020/tools/feature_check.html
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
from src.rldd_features import FEATURES, frame_features  # noqa: E402

LFW = ROOT / "data" / "lfw" / "lfw-deepfunneled" / "lfw-deepfunneled"
OUT = ROOT / "runs" / "_feature_check" / "python.json"


def main(n=12):
    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "data" / "mediapipe" / "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True,
        output_facial_transformation_matrixes=True))
    people = sorted(p for p in LFW.iterdir() if p.is_dir())
    rng = np.random.default_rng(5)
    faces = []
    for person in rng.choice(people, 4 * n, replace=False):
        photo = sorted(person.glob("*.jpg"))[0]
        rgb = np.asarray(Image.open(photo).convert("RGB"))
        r = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not r.face_landmarks:
            continue
        lms = np.array([[p.x, p.y, p.z] for p in r.face_landmarks[0]])
        bs = {c.category_name: c.score for c in r.face_blendshapes[0]}
        faces.append({"photo": str(photo.relative_to(ROOT)),
                      "features": [float(v) for v in frame_features(lms, bs, r.facial_transformation_matrixes[0])]})
        if len(faces) == n:
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"features": FEATURES, "faces": faces}))
    print(f"Computed {len(FEATURES)} features for {len(faces)} LFW faces in Python -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
