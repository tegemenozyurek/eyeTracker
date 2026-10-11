// The 13 per-frame features of eyeTrack1: a line-by-line copy of src/rldd_features.py, which copies the
// UTA-RLDD feature dataset's own extraction code. Checked against Python on real faces by tools/feature_check.html.
// Pure functions (no DOM).

export const FEATURES = ["bs_eyeBlinkLeft", "bs_eyeBlinkRight", "bs_eyeSquintLeft", "bs_eyeSquintRight", "bs_eyeLookDownLeft",
  "bs_eyeLookDownRight", "bs_jawOpen", "ear_left", "ear_right", "mar", "pitch", "yaw", "roll"];
const RIGHT_EYE_EAR = [[160, 144], [158, 153], [33, 133]];
const LEFT_EYE_EAR = [[387, 373], [384, 381], [362, 263]];
const MOUTH_VERTICAL = [13, 14];
const MOUTH_HORIZONTAL = [61, 291];

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

// landmarks: MediaPipe NormalizedLandmark[] ({x, y, z}); blendshapes: {name: score};
// matrix: the 16 numbers of facialTransformationMatrixes[0].data.
export function frameFeatures(landmarks, blendshapes, matrix) {
  const row = {};
  for (const k of ["eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "eyeLookDownLeft", "eyeLookDownRight", "jawOpen"]) {
    row[`bs_${k}`] = blendshapes[k];
  }
  for (const [side, [[v1a, v1b], [v2a, v2b], [ha, hb]]] of [["right", RIGHT_EYE_EAR], ["left", LEFT_EYE_EAR]]) {
    const v = dist(landmarks[v1a], landmarks[v1b]) + dist(landmarks[v2a], landmarks[v2b]);
    row[`ear_${side}`] = v / (2 * dist(landmarks[ha], landmarks[hb]));
  }
  row.mar = dist(landmarks[MOUTH_VERTICAL[0]], landmarks[MOUTH_VERTICAL[1]])
    / dist(landmarks[MOUTH_HORIZONTAL[0]], landmarks[MOUTH_HORIZONTAL[1]]);
  // r[i][j] of the rotation part. MediaPipe's JS matrix data is column-major: element (row i, column j) = m[j * 4 + i].
  const r = (i, j) => matrix[j * 4 + i];
  const deg = 180 / Math.PI;
  row.pitch = Math.atan2(r(2, 1), r(2, 2)) * deg;
  row.yaw = Math.atan2(-r(2, 0), Math.hypot(r(2, 1), r(2, 2))) * deg;
  row.roll = Math.atan2(r(1, 0), r(0, 0)) * deg;
  return FEATURES.map((f) => row[f]);
}
