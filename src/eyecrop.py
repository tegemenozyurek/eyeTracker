"""The eye crop shared by training (Python) and the live demo (web/eyecrop.js).

Given the two corners of one eye (MediaPipe landmarks), a similarity transform (rotation,
scale, shift) puts the left-in-image corner and the right-in-image corner on fixed points of
a SIZE x SIZE crop, so every eye arrives level, centred and at the same size, whatever the
head tilt or the distance to the camera.

The pixel sampling is written out by hand (bilinear sampling, averaged over K x K points per
output pixel so large eyes are shrunk smoothly) instead of using OpenCV or the browser's canvas,
because those two resample differently. web/eyecrop.js is a line-by-line copy of this file, and
web/eyecrop.test.mjs checks that both give the same crop (scripts/eye_crop_parity.py writes the
test cases).

Coordinates are continuous: pixel (row i, column j) covers [j, j+1) x [i, i+1), its centre is at
(j + 0.5, i + 0.5). Landmarks in pixels use the same convention (MediaPipe x * width).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = ROOT / "models" / "eye_crop.json"

# MediaPipe face-mesh corner landmarks of each eye ("right" / "left" are the person's own eyes).
CORNERS = {"right": (33, 133), "left": (362, 263)}
DEFAULT_SPEC = {"size": 32, "eye_width": 0.7, "center_x": 0.5, "center_y": 0.5, "supersample": 4}


def load_spec():
    return json.loads(SPEC_FILE.read_text()) if SPEC_FILE.exists() else dict(DEFAULT_SPEC)


def to_gray(rgb):
    """(H, W, 3) uint8 RGB -> (H, W) float64 in 0-255, same weights as the browser code."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def crop_transform(c1, c2, spec):
    """Map crop coordinates -> image coordinates. c1, c2: (x, y) eye corners in pixels.
    The corner further left in the image goes to the left template point."""
    if c2[0] < c1[0]:
        c1, c2 = c2, c1
    s = spec["size"]
    w = spec["eye_width"] * s
    p1 = np.array([spec["center_x"] * s - w / 2, spec["center_y"] * s])
    # complex ratio z = (c2 - c1) / (p2 - p1): rotation and scale from crop to image
    a = (c2[0] - c1[0]) / w
    b = (c2[1] - c1[1]) / w
    tx = c1[0] - (a * p1[0] - b * p1[1])
    ty = c1[1] - (b * p1[0] + a * p1[1])
    return a, b, tx, ty


def bilinear(gray, u, v):
    """Sample gray (H, W) at continuous image coordinates (u, v); the image edge is extended outward."""
    h, w = gray.shape
    x = u - 0.5
    y = v - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    xa = np.clip(x0, 0, w - 1)
    xb = np.clip(x0 + 1, 0, w - 1)
    ya = np.clip(y0, 0, h - 1)
    yb = np.clip(y0 + 1, 0, h - 1)
    top = gray[ya, xa] * (1 - fx) + gray[ya, xb] * fx
    bottom = gray[yb, xa] * (1 - fx) + gray[yb, xb] * fx
    return top * (1 - fy) + bottom * fy


def eye_crop(gray, c1, c2, spec=None):
    """gray: (H, W) float64 image in 0-255. c1, c2: eye corners (x, y) in pixels.
    Returns a (size, size) float64 crop in 0-255."""
    spec = spec or load_spec()
    s, k = spec["size"], spec["supersample"]
    a, b, tx, ty = crop_transform(c1, c2, spec)
    sub = (np.arange(k) + 0.5) / k
    ox = (np.arange(s)[:, None] + sub[None, :]).reshape(-1)  # all sample x positions along one row
    X = np.tile(ox, s * k)                                    # row-major over (row, subrow, col, subcol)
    Y = np.repeat(ox, s * k)
    u = a * X - b * Y + tx
    v = b * X + a * Y + ty
    samples = bilinear(gray, u, v).reshape(s, k, s, k)
    return samples.mean(axis=(1, 3))


def eye_corners(landmarks_px, eye):
    """landmarks_px: (478, 2) array of MediaPipe points in pixels. eye: "right" or "left"."""
    i, j = CORNERS[eye]
    return landmarks_px[i], landmarks_px[j]
