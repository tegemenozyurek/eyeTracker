// Replays UTA-RLDD feature videos through the demo's own rules (web/rules.js), frame by frame as they would run
// live, and records the level and PERCLOS at the end of every window. Called by scripts/prepare_rldd.py.
//   node tools/rules_on_rldd.mjs <input.json> <output.json>
// input:  {videos: [{t: [...], face: [...], blinkL: [...], blinkR: [...], jaw: [...], ends: [frame indices]}]}
// output: {videos: [{level: [...], perclos: [...], perclosReady: [...]}]}  (one value per window end)
//
// Only drowsiness is judged here: head pose is held at 0, so the eyes-off-the-road rule never fires, and frames
// without a detected face are skipped (live, a missing face would count as looking away).
import { readFileSync, writeFileSync } from "node:fs";
import { DriverRules } from "../web/rules.js";

const [input, output] = process.argv.slice(2);
const data = JSON.parse(readFileSync(input, "utf8"));
const out = { videos: [] };
for (const v of data.videos) {
  const rules = new DriverRules();
  const ends = new Set(v.ends);
  const res = { level: [], perclos: [], perclosReady: [] };
  let state = null;
  for (let i = 0; i < v.t.length; i++) {
    if (v.face[i]) {
      state = rules.update({ t: v.t[i], face: true, blinkL: v.blinkL[i], blinkR: v.blinkR[i], jawOpen: v.jaw[i],
        yaw: 0, pitch: 0, roll: 0 });
    }
    if (ends.has(i)) {
      res.level.push(state ? state.level : 0);
      res.perclos.push(state ? state.perclos : 0);
      res.perclosReady.push(state ? state.perclosReady : false);
    }
  }
  out.videos.push(res);
}
writeFileSync(output, JSON.stringify(out));
console.log(`rules replayed on ${data.videos.length} videos`);
