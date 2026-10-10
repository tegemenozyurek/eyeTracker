// The eye crop shared by training (src/eyecrop.py) and the live demo. A line-by-line copy of the Python version;
// web/eyecrop.test.mjs checks that both give the same crop. See src/eyecrop.py for the coordinate conventions.
// Pure functions on plain arrays (no DOM), so the same code runs in the browser and in Node.

export const CORNERS = { right: [33, 133], left: [362, 263] };

// rgba: Uint8ClampedArray from getImageData -> Float64Array gray, same weights as Python's to_gray.
export function toGray(rgba, width, height) {
  const gray = new Float64Array(width * height);
  for (let i = 0; i < width * height; i++) {
    gray[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
  }
  return gray;
}

// Crop coordinates -> image coordinates; the corner further left in the image goes to the left template point.
export function cropTransform(c1, c2, spec) {
  if (c2[0] < c1[0]) [c1, c2] = [c2, c1];
  const s = spec.size;
  const w = spec.eye_width * s;
  const p1 = [spec.center_x * s - w / 2, spec.center_y * s];
  const a = (c2[0] - c1[0]) / w;
  const b = (c2[1] - c1[1]) / w;
  const tx = c1[0] - (a * p1[0] - b * p1[1]);
  const ty = c1[1] - (b * p1[0] + a * p1[1]);
  return [a, b, tx, ty];
}

const clamp = (i, n) => (i < 0 ? 0 : i > n - 1 ? n - 1 : i);

// Sample gray (row-major, width x height) at continuous coordinates (u, v); the edge is extended outward.
export function bilinear(gray, width, height, u, v) {
  const x = u - 0.5;
  const y = v - 0.5;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const xa = clamp(x0, width), xb = clamp(x0 + 1, width);
  const ya = clamp(y0, height), yb = clamp(y0 + 1, height);
  const top = gray[ya * width + xa] * (1 - fx) + gray[ya * width + xb] * fx;
  const bottom = gray[yb * width + xa] * (1 - fx) + gray[yb * width + xb] * fx;
  return top * (1 - fy) + bottom * fy;
}

// gray: Float64Array (0-255), c1, c2: eye corners [x, y] in pixels. Returns Float64Array size*size (0-255).
export function eyeCrop(gray, width, height, c1, c2, spec) {
  const s = spec.size, k = spec.supersample;
  const [a, b, tx, ty] = cropTransform(c1, c2, spec);
  const out = new Float64Array(s * s);
  for (let row = 0; row < s; row++) {
    for (let col = 0; col < s; col++) {
      let sum = 0;
      for (let i = 0; i < k; i++) {
        const Y = row + (i + 0.5) / k;
        for (let j = 0; j < k; j++) {
          const X = col + (j + 0.5) / k;
          sum += bilinear(gray, width, height, a * X - b * Y + tx, b * X + a * Y + ty);
        }
      }
      out[row * s + col] = sum / (k * k);
    }
  }
  return out;
}

// The part of the image a crop reads from (with a 2 px margin), so the browser only needs to read those pixels.
export function cropBox(c1, c2, spec) {
  const [a, b, tx, ty] = cropTransform(c1, c2, spec);
  const s = spec.size;
  const pts = [[0, 0], [s, 0], [0, s], [s, s]].map(([X, Y]) => [a * X - b * Y + tx, b * X + a * Y + ty]);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  return [Math.floor(Math.min(...xs)) - 2, Math.floor(Math.min(...ys)) - 2,
    Math.ceil(Math.max(...xs)) + 2, Math.ceil(Math.max(...ys)) + 2];
}
