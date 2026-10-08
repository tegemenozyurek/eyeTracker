// Tests for the rules baseline with synthetic 30 fps frames. Run: node web/rules.test.mjs
import assert from "node:assert/strict";
import { DriverRules, RULES } from "./rules.js";

const FPS = 30;

// Simulate `seconds` of driving; frame(t) returns overrides of a calm, alert, forward-looking driver.
function drive(rules, seconds, frame = () => ({}), start = rules.lastT ?? 0) {
  let state;
  for (let i = 1; i <= seconds * FPS; i++) {
    const t = start + i / FPS;
    state = rules.update({ t, face: true, blinkL: 0.05, blinkR: 0.05, jawOpen: 0.02, yaw: 0, pitch: 0, roll: 0, ...frame(t) });
  }
  return state;
}
// A normal blink: eyes closed for 0.2 s every 4 s.
const blinking = (t) => (t % 4 < 0.2 ? { blinkL: 0.8, blinkR: 0.8 } : {});

const results = [];
function test(name, fn) {
  fn();
  results.push(name);
  console.log(`  ok  ${name}`);
}

test("alert driver blinking normally stays OK, PERCLOS about 5%", () => {
  const r = new DriverRules();
  const s = drive(r, 120, blinking);
  assert.equal(s.levelName, "OK");
  assert.ok(Math.abs(s.perclos - 0.05) < 0.01, `perclos ${s.perclos}`);
  assert.equal(s.blinks, 30);
});

test("eyes closed for 2 s triggers Take a break at once", () => {
  const r = new DriverRules();
  drive(r, 30, blinking);
  const s = drive(r, 2, () => ({ blinkL: 0.9, blinkR: 0.9 }));
  assert.equal(s.levelName, "Take a break");
  assert.ok(s.reasons.some((x) => x.startsWith("eyes closed")));
});

test("the alert stays up for the hold time after eyes open again (no flicker), then steps down", () => {
  const r = new DriverRules();
  drive(r, 30, blinking);
  drive(r, 2, () => ({ blinkL: 0.9, blinkR: 0.9 }));
  assert.equal(drive(r, RULES.holdS - 1).levelName, "Take a break");
  // Down one level, not to OK: the 2 s closure itself pushed the last minute's PERCLOS above 7.5%.
  const s = drive(r, 3);
  assert.equal(s.levelName, "Attention");
  assert.ok(s.reasons.some((x) => x.startsWith("PERCLOS")));
  // Once that closure has left the 60 s window, the driver is OK again.
  assert.equal(drive(r, 60, blinking).levelName, "OK");
});

test("high PERCLOS (eyes closed 20% of the time) gives Take a break", () => {
  const r = new DriverRules();
  const s = drive(r, 60, (t) => (t % 2 < 0.4 ? { blinkL: 0.8, blinkR: 0.8 } : {}));
  assert.ok(s.perclos > 0.18 && s.perclos < 0.22, `perclos ${s.perclos}`);
  assert.equal(s.levelName, "Take a break");
});

test("PERCLOS is not judged before 20 s of observation", () => {
  const r = new DriverRules();
  const s = drive(r, 10, (t) => (t % 2 < 0.6 ? { blinkL: 0.8, blinkR: 0.8 } : {}));
  assert.equal(s.perclosReady, false);
  assert.equal(s.levelName, "OK");
});

test("looking 40 degrees to the side: Attention after 2 s, Take a break after 4 s", () => {
  const r = new DriverRules();
  drive(r, 10, blinking);
  assert.equal(drive(r, 2.5, () => ({ yaw: 40 })).levelName, "Attention");
  assert.equal(drive(r, 2, () => ({ yaw: 40 })).levelName, "Take a break");
});

test("face out of view after calibration counts as eyes off the road", () => {
  const r = new DriverRules();
  drive(r, 10, blinking);
  const s = drive(r, 3, () => ({ face: false }));
  assert.equal(s.offRoad, true);
  assert.equal(s.levelName, "Attention");
});

test("the calibrated direction is the road: a camera mounted 30 degrees to the side is not distraction", () => {
  const r = new DriverRules();
  const s = drive(r, 20, (t) => ({ ...blinking(t), yaw: 30 }));
  assert.equal(s.offRoad, false);
  assert.equal(s.levelName, "OK");
});

test("three yawns within 5 minutes give Attention", () => {
  const r = new DriverRules();
  const s = drive(r, 90, (t) => ({ ...blinking(t), ...(t % 30 > 10 && t % 30 < 13 ? { jawOpen: 0.8 } : {}) }));
  assert.equal(s.yawns, 3);
  assert.equal(s.levelName, "Attention");
});

test("a quick head nod is counted", () => {
  const r = new DriverRules();
  const s = drive(r, 40, (t) => ({ ...blinking(t), ...(t > 35 && t < 36 ? { pitch: -18 } : {}) }));
  assert.equal(s.nods, 1);
});

console.log(`\n${results.length} tests passed`);
