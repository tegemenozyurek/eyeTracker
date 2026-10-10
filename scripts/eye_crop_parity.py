"""Step 9: write test cases for the Python / JavaScript eye-crop parity test.

Two small test images (a smooth synthetic pattern and random noise, which punishes any
difference in sampling) and 12 pairs of eye corners each: level, tilted up to 40 degrees,
tiny and large eyes, eyes past the image edge. For every case src/eyecrop.py computes the crop.
web/eyecrop.test.mjs then runs web/eyecrop.js on the same inputs and compares.

Usage:
  python scripts/eye_crop_parity.py      # writes web/eyecrop.fixture.json
  node web/eyecrop.test.mjs              # runs the comparison
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eyecrop import eye_crop, load_spec  # noqa: E402

OUT = ROOT / "web" / "eyecrop.fixture.json"


def images(width=96, height=72, seed=0):
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    smooth = 128 + 60 * np.sin(xs / 7.0) * np.cos(ys / 5.0) + 40 * np.sin((xs + 2 * ys) / 13.0)
    noise = np.random.default_rng(seed).uniform(0, 255, (height, width))
    return {"smooth": np.round(smooth, 3), "noise": np.round(noise, 3)}


def corner_pairs(width=96, height=72, seed=1):
    rng = np.random.default_rng(seed)
    fixed = [((30, 36), (60, 36)),       # level eye in the middle
             ((60, 36), (30, 36)),       # corners given right-to-left
             ((30.3, 40.7), (58.9, 30.2)),  # tilted, sub-pixel corners
             ((44, 30), (52, 31)),       # tiny eye: 8 px wide
             ((5, 20), (90, 50)),        # large and tilted
             ((-10, 10), (12, 4)),       # past the top-left edge
             ((80, 65), (110, 70))]      # past the bottom-right edge
    random = []
    for _ in range(5):
        c1 = rng.uniform([0, 0], [width, height])
        angle = rng.uniform(-np.radians(40), np.radians(40))
        length = rng.uniform(6, 50)
        random.append((tuple(c1), tuple(c1 + length * np.array([np.cos(angle), np.sin(angle)]))))
    return [(tuple(map(float, a)), tuple(map(float, b))) for a, b in fixed + random]


def main():
    spec = load_spec()
    imgs = images()
    cases = []
    for name, img in imgs.items():
        for c1, c2 in corner_pairs():
            crop = eye_crop(img, c1, c2, spec)
            cases.append({"image": name, "c1": c1, "c2": c2, "crop": np.round(crop, 6).tolist()})
    h, w = imgs["smooth"].shape
    OUT.write_text(json.dumps({"spec": spec, "width": w, "height": h,
                               "images": {k: v.ravel().tolist() for k, v in imgs.items()}, "cases": cases}))
    print(f"Wrote {len(cases)} cases ({len(imgs)} images x {len(cases) // len(imgs)} corner pairs) to "
          f"{OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB), crop spec {spec}")


if __name__ == "__main__":
    main()
