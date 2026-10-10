// Parity test: the browser's eye crop (web/eyecrop.js) must equal the training crop (src/eyecrop.py).
// Test cases come from scripts/eye_crop_parity.py. Run: node web/eyecrop.test.mjs
import { readFileSync } from "node:fs";
import { cropBox, eyeCrop } from "./eyecrop.js";

const TOLERANCE = 1e-4;  // gray levels (0-255); Python rounds the expected values to 6 decimals
const fx = JSON.parse(readFileSync(new URL("./eyecrop.fixture.json", import.meta.url)));
const images = Object.fromEntries(Object.entries(fx.images).map(([k, v]) => [k, Float64Array.from(v)]));

let worst = 0, failed = 0;
for (const c of fx.cases) {
  const got = eyeCrop(images[c.image], fx.width, fx.height, c.c1, c.c2, fx.spec);
  const expected = c.crop.flat();
  let diff = 0;
  for (let i = 0; i < got.length; i++) diff = Math.max(diff, Math.abs(got[i] - expected[i]));
  worst = Math.max(worst, diff);
  if (diff > TOLERANCE) {
    failed++;
    console.log(`  FAIL ${c.image} c1=${c.c1} c2=${c.c2}: max difference ${diff.toExponential(2)}`);
  }
}
console.log(`${fx.cases.length - failed} of ${fx.cases.length} crops match Python; largest difference ` +
  `${worst.toExponential(2)} gray levels (tolerance ${TOLERANCE})`);

// The browser reads only the box a crop needs (clamped to the image) and shifts the corners into it
// (extractEye in app.js). That must give the same crop as cropping the full image.
let boxWorst = 0, boxFailed = 0;
for (const c of fx.cases) {
  const full = images[c.image], W = fx.width, H = fx.height;
  let [x0, y0, x1, y1] = cropBox(c.c1, c.c2, fx.spec);
  x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(W, x1); y1 = Math.min(H, y1);
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  const sub = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) sub[y * w + x] = full[(y0 + y) * W + (x0 + x)];
  const got = eyeCrop(sub, w, h, [c.c1[0] - x0, c.c1[1] - y0], [c.c2[0] - x0, c.c2[1] - y0], fx.spec);
  const expected = c.crop.flat();
  let diff = 0;
  for (let i = 0; i < got.length; i++) diff = Math.max(diff, Math.abs(got[i] - expected[i]));
  boxWorst = Math.max(boxWorst, diff);
  if (diff > TOLERANCE) boxFailed++;
}
console.log(`${fx.cases.length - boxFailed} of ${fx.cases.length} crops from the browser's small pixel box match too; ` +
  `largest difference ${boxWorst.toExponential(2)}`);
process.exit(failed || boxFailed ? 1 : 0);
