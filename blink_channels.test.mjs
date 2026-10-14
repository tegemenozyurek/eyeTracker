// Parity test: the browser's blink channels (web/features.js) must equal Python's (src/blink_channels.py) on real
// UTA-RLDD windows written by scripts/blink_channels_parity.py. Run: node web/blink_channels.test.mjs
import { readFileSync } from "node:fs";
import { blinkChannels } from "./features.js";

const fx = JSON.parse(readFileSync(new URL("./blink_channels.fixture.json", import.meta.url)));
let worst = 0;
for (const c of fx.cases) {
  const got = blinkChannels(c.rows, fx.fps);
  got.forEach((row, t) => row.forEach((v, k) => { worst = Math.max(worst, Math.abs(v - c.channels[t][k])); }));
}
const ok = worst < 1e-4;
console.log(`${fx.cases.length} real windows: largest difference ${worst.toExponential(2)} ${ok ? "(match)" : "(MISMATCH)"}`);
process.exit(ok ? 0 : 1);
