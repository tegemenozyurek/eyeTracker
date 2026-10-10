"""The 13 per-frame features of eyeTrack1, from one MediaPipe Face Landmarker result.

Copied from the extraction code of the UTA-RLDD feature dataset (SleepDetective, branch dataset-release,
src/extract.py: frame_features and euler_from_matrix) so that live features (web/features.js, a
line-by-line copy of this file) are the numbers eyeTrack1 was trained on:
  EAR / MAR  3D distances between MediaPipe's normalized landmarks (x, y, z as returned)
  pose       degrees from the 4x4 facial transformation matrix:
             pitch = atan2(r21, r22), yaw = atan2(-r20, hypot(r21, r22)), roll = atan2(r10, r00)
"""
import math

import numpy as np

FEATURES = ["bs_eyeBlinkLeft", "bs_eyeBlinkRight", "bs_eyeSquintLeft", "bs_eyeSquintRight", "bs_eyeLookDownLeft",
            "bs_eyeLookDownRight", "bs_jawOpen", "ear_left", "ear_right", "mar", "pitch", "yaw", "roll"]
RIGHT_EYE_EAR = ((160, 144), (158, 153), (33, 133))
LEFT_EYE_EAR = ((387, 373), (384, 381), (362, 263))
MOUTH_VERTICAL = (13, 14)
MOUTH_HORIZONTAL = (61, 291)


def _dist(a, b):
    return math.dist(a, b)


def frame_features(landmarks, blendshapes, matrix):
    """landmarks: (478, 3) normalized x, y, z. blendshapes: {name: score}. matrix: 4x4 array (row-major, as numpy)."""
    row = {f"bs_{k}": blendshapes[k] for k in ("eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
                                               "eyeLookDownLeft", "eyeLookDownRight", "jawOpen")}
    for side, ((v1a, v1b), (v2a, v2b), (ha, hb)) in (("right", RIGHT_EYE_EAR), ("left", LEFT_EYE_EAR)):
        v = _dist(landmarks[v1a], landmarks[v1b]) + _dist(landmarks[v2a], landmarks[v2b])
        row[f"ear_{side}"] = v / (2 * _dist(landmarks[ha], landmarks[hb]))
    row["mar"] = (_dist(landmarks[MOUTH_VERTICAL[0]], landmarks[MOUTH_VERTICAL[1]])
                  / _dist(landmarks[MOUTH_HORIZONTAL[0]], landmarks[MOUTH_HORIZONTAL[1]]))
    r = np.asarray(matrix)[:3, :3]
    row["pitch"] = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    row["yaw"] = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    row["roll"] = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return [row[f] for f in FEATURES]
