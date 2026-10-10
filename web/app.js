// eyeTracker live: webcam -> MediaPipe Face Landmarker -> per-frame signals -> driver-state model.
// Everything runs in the browser; no frame ever leaves the device.
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";
import { DriverRules, RULES } from "./rules.js";
import { CORNERS, cropBox, eyeCrop, toGray } from "./eyecrop.js";
import { frameFeatures } from "./features.js";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const MEDIAPIPE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// MediaPipe face-mesh indices. "Right" / "left" are the person's own eyes.
const EYE_R = [33, 160, 158, 133, 153, 144];   // corner, top, top, corner, bottom, bottom (for EAR)
const EYE_L = [362, 385, 387, 263, 373, 380];
const EYE_R_RING = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
const EYE_L_RING = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382];
const LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];

// rules first, then the eye models from web/models/manifest.json (scripts/export_web.py), then what is still to come.
const RULES_MODEL = { version: "rules", live: true, tagline: "Baseline, no learning",
  changes: ["Eye closed when MediaPipe's eyeBlink score ≥ 0.5", "PERCLOS over 60 s, long closures, yawns",
            "Head pose vs a calibrated road direction"] };
const COMING = [];

const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries(["video", "overlay", "viewport", "empty", "banner", "camera", "file", "recalibrate", "sound",
  "mesh", "showCrops", "stats", "models", "state", "stateLabel", "stateSub", "eyeL", "eyeR", "eyeLBar", "eyeRBar",
  "blinks", "blinkInfo", "yawns", "yawnInfo", "perclos", "perclosBar", "perclosInfo", "pose", "poseInfo", "offroad",
  "offroadInfo", "spark", "crops", "cropL", "cropR", "temporal", "temporalInfo", "temporalBars"].map((id) => [id, $(id)]));

const state = { landmarker: null, rules: new DriverRules(), source: null, stream: null, loopId: 0, mirrored: true,
  fps: 0, lastFrame: 0, detectMs: 0, history: [], lastLevel: 0, model: "rules", eyeModel: null, inferMs: 0, models: [],
  temporal: null };

// ---------------------------------------------------------------- per-frame signals

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// Eye aspect ratio (Soukupová & Čech 2016): eye height over eye width; drops towards 0 as the eye closes.
function ear(pts, idx) {
  const [p1, p2, p3, p4, p5, p6] = idx.map((i) => pts[i]);
  return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4));
}

// Head rotation in degrees from MediaPipe's 4x4 facial transformation matrix (column-major).
function headPose(m) {
  const r00 = m[0], r10 = m[1], r20 = m[2], r21 = m[6], r22 = m[10];
  const deg = 180 / Math.PI;
  return {
    pitch: Math.atan2(r21, r22) * deg,                       // + chin up, - chin down
    yaw: Math.atan2(-r20, Math.hypot(r21, r22)) * deg,       // left / right
    roll: Math.atan2(r10, r00) * deg,                        // head tilt
  };
}

function signals(result, width, height, t) {
  const lm = result.faceLandmarks?.[0];
  if (!lm) return { t, face: false };
  const pts = lm.map((p) => [p.x * width, p.y * height]);
  const bs = Object.fromEntries(result.faceBlendshapes[0].categories.map((c) => [c.categoryName, c.score]));
  const pose = result.facialTransformationMatrixes?.[0] ? headPose(result.facialTransformationMatrixes[0].data) : { pitch: 0, yaw: 0, roll: 0 };
  // blinkL / blinkR are the eye-closure scores the rules use: MediaPipe's blendshapes, or an eye model's
  // probability of "closed" when one is selected (see classifyEyes).
  const matrix = result.facialTransformationMatrixes?.[0]?.data;
  return { t, face: true, pts, blinkL: bs.eyeBlinkLeft, blinkR: bs.eyeBlinkRight, jawOpen: bs.jawOpen,
    bsBlinkL: bs.eyeBlinkLeft, bsBlinkR: bs.eyeBlinkRight, earL: ear(pts, EYE_L), earR: ear(pts, EYE_R), ...pose,
    feat: matrix ? frameFeatures(lm, bs, matrix) : null };
}

// Both eyes through the selected eye CNN in one call; returns P(closed) for [right, left].
async function classifyEyes(crops) {
  const m = state.eyeModel, n = 32 * 32;
  const data = new Float32Array(2 * n);
  crops.forEach((c, k) => { for (let i = 0; i < n; i++) data[k * n + i] = (c[i] / 255 - m.mean) / m.std; });
  const t0 = performance.now();
  const out = await m.session.run({ input: new ort.Tensor("float32", data, [2, 1, 32, 32]) });
  state.inferMs = 0.9 * state.inferMs + 0.1 * (performance.now() - t0);
  const z = out.logits.data;
  return [0, 1].map((k) => 1 / (1 + Math.exp(z[2 * k + 1] - z[2 * k])));  // softmax over 2 classes, class 0 = closed
}

// ---------------------------------------------------------------- eyeTrack1 (drowsiness over 60 s)

// Frames are sampled 5 times per second into a 60 s buffer (300 frames, as in training: Step 12). Every 2 s,
// once the buffer is full and a face was found in at least 70% of it, the GRU judges the last minute.
// Its answer replaces the PERCLOS and yawn rules; the microsleep and eyes-off-the-road rules stay as safety nets.
const LEVEL_OF_CLASS = [0, 1, 2];  // alert -> OK, low vigilant -> Attention, drowsy -> Take a break

async function temporalStep(f, s) {
  const T = state.temporal;
  if (T.lastSample == null || f.t - T.lastSample >= 1 / T.fps - 0.01) {
    T.lastSample = f.t;
    const ok = f.face && f.feat;
    T.buffer.push({ x: ok ? f.feat : null, face: ok });
    if (T.buffer.length > T.window) T.buffer.shift();
  }
  const faceShare = T.buffer.filter((b) => b.face).length / T.window;
  T.ready = T.buffer.length === T.window && faceShare >= 0.7;
  if (T.ready && f.t - (T.lastRun ?? -1e9) >= 2) {
    T.lastRun = f.t;
    const n = T.features + 1, data = new Float32Array(T.window * n);
    T.buffer.forEach((b, i) => {
      if (!b.face) return;  // missing face: zeros and flag 0, as in training
      for (let k = 0; k < T.features; k++) data[i * n + k] = (b.x[k] - T.mean[k]) / T.std[k];
      data[i * n + T.features] = 1;
    });
    const out = await T.session.run({ input: new ort.Tensor("float32", data, [1, T.window, n]) });
    const z = Array.from(out.logits.data), m = Math.max(...z), e = z.map((v) => Math.exp(v - m));
    const probs = e.map((v) => v / e.reduce((a, b) => a + b, 0));
    T.probs = T.probs ? T.probs.map((p, k) => 0.5 * p + 0.5 * probs[k]) : probs;  // a little smoothing between runs
  }
  if (T.probs) {
    const top = T.probs.indexOf(Math.max(...T.probs));
    const level = LEVEL_OF_CLASS[top];
    if (level > s.level) { s.level = level; s.levelName = ["OK", "Attention", "Take a break"][level]; }
    if (level > 0) s.reasons.push(`eyeTrack1: ${T.classes[top]} ${(T.probs[top] * 100).toFixed(0)}%`);
  }
  s.temporal = { ready: T.ready, filled: T.buffer.length / T.fps, probs: T.probs, classes: T.classes, faceShare };
}

// ---------------------------------------------------------------- loop

async function loop(id) {
  if (id !== state.loopId || !state.source) return;
  const v = state.source;
  if (v.readyState >= 2 && !v.paused) {
    const now = performance.now();
    const t0 = performance.now();
    const result = state.landmarker.detectForVideo(v, now);
    state.detectMs = 0.9 * state.detectMs + 0.1 * (performance.now() - t0);
    const f = signals(result, v.videoWidth, v.videoHeight, now / 1000);
    let crops = null;
    if (f.face && (state.eyeModel || ui.showCrops.checked)) crops = ["right", "left"].map((eye) => extractEye(v, f.pts, eye));
    if (f.face && state.eyeModel) {
      const model = state.eyeModel;
      const [pR, pL] = await classifyEyes(crops);
      if (model !== state.eyeModel) return loop(id);  // the model was switched meanwhile
      Object.assign(f, { blinkR: pR, blinkL: pL });
    }
    const s = state.rules.update(f);
    if (state.temporal) await temporalStep(f, s);
    state.history.push({ t: f.t, closure: f.face ? (f.blinkL + f.blinkR) / 2 : null, level: s.level });
    while (state.history.length && state.history[0].t < f.t - 60) state.history.shift();
    draw(f, v.videoWidth, v.videoHeight);
    render(f, s);
    if (ui.showCrops.checked && crops) renderCrops(crops);
    if (state.lastFrame) state.fps = 0.9 * state.fps + 0.1 * (1000 / Math.max(1, now - state.lastFrame));
    state.lastFrame = now;
    ui.stats.textContent = `${state.fps.toFixed(0)} fps · face tracking ${state.detectMs.toFixed(1)} ms` +
      (state.eyeModel ? ` · ${state.model} ${state.inferMs.toFixed(1)} ms (both eyes)` : " · rules");
  }
  if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(() => loop(id));
  else requestAnimationFrame(() => loop(id));
}

// ---------------------------------------------------------------- drawing

function draw(f, width, height) {
  const rect = ui.overlay.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  ui.overlay.width = Math.round(rect.width * dpr);
  ui.overlay.height = Math.round(rect.height * dpr);
  const ctx = ui.overlay.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!f.face || !ui.mesh.checked) return;
  const scale = Math.min(rect.width / width, rect.height / height);
  const ox = (rect.width - width * scale) / 2, oy = (rect.height - height * scale) / 2;
  const P = ([x, y]) => [ox + (state.mirrored ? width - x : x) * scale, oy + y * scale];
  const ring = (idx, color) => {
    ctx.beginPath();
    idx.forEach((i, k) => { const [x, y] = P(f.pts[i]); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  };
  const closedColor = (b) => (b >= RULES.closedBlink ? "#f43f5e" : "#34d399");
  ring(EYE_R_RING, closedColor(f.blinkR));
  ring(EYE_L_RING, closedColor(f.blinkL));
  ring(LIPS, f.jawOpen > RULES.yawnJaw ? "#f59e0b" : "rgba(148,163,184,.7)");
}

const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;

function render(f, s) {
  ui.state.dataset.level = String(s.level);
  ui.stateLabel.textContent = s.face || s.calibrated ? s.levelName : "No face";
  ui.stateSub.textContent = !s.calibrated ? (s.face ? "Calibrating: look at the road (the screen)…" : "Look at the camera")
    : s.reasons.length ? s.reasons.join(" · ") : `${state.model} · all signals normal`;
  ui.banner.hidden = s.level === 0;
  ui.banner.dataset.level = String(s.level);
  ui.banner.textContent = s.level === 2 ? "⚠ Take a break" : "Attention";
  if (s.level === 2 && state.lastLevel < 2 && ui.sound.checked) beep();
  state.lastLevel = s.level;

  const eye = (b) => (b == null ? "–" : b >= RULES.closedBlink ? "closed" : "open");
  const how = (b) => (state.eyeModel ? ` · ${pct(b)} closed` : ` · blink ${b.toFixed(2)}`);
  ui.eyeL.textContent = f.face ? `${eye(f.blinkL)}${how(f.blinkL)} · EAR ${f.earL.toFixed(2)}` : "–";
  ui.eyeR.textContent = f.face ? `${eye(f.blinkR)}${how(f.blinkR)} · EAR ${f.earR.toFixed(2)}` : "–";
  ui.eyeLBar.style.width = f.face ? pct(f.blinkL, 1) : "0%";
  ui.eyeRBar.style.width = f.face ? pct(f.blinkR, 1) : "0%";
  ui.blinks.textContent = s.blinks;
  ui.blinkInfo.textContent = `${s.blinksPerMin.toFixed(0)}/min` + (s.lastBlinkS != null ? ` · last ${(s.lastBlinkS * 1000).toFixed(0)} ms` : "")
    + (s.longClosures ? ` · ${s.longClosures} long` : "");
  ui.yawns.textContent = s.yawns;
  ui.yawnInfo.textContent = s.yawning ? "yawning…" : `head nods: ${s.nods}`;
  ui.perclos.textContent = s.perclosReady ? pct(s.perclos, 1) : `warming up (${s.observedS.toFixed(0)} / ${RULES.perclosMinS} s)`;
  ui.perclosBar.style.width = `${Math.min(100, (s.perclos / 0.3) * 100).toFixed(1)}%`;
  ui.perclosBar.style.background = !s.perclosReady ? "var(--muted)" : s.perclos >= RULES.perclosBreak ? "var(--break)"
    : s.perclos >= RULES.perclosAttention ? "var(--attention)" : "var(--ok)";
  ui.perclosInfo.textContent = `Attention at ${pct(RULES.perclosAttention, 1)}, break at ${pct(RULES.perclosBreak)}`;
  const road = s.road ?? { pitch: 0, yaw: 0 };
  ui.pose.textContent = f.face ? `${(f.pitch - road.pitch).toFixed(0)}° / ${(f.yaw - road.yaw).toFixed(0)}° / ${f.roll.toFixed(0)}°` : "–";
  ui.poseInfo.textContent = s.calibrated ? "pitch and yaw relative to the calibrated road direction" : "calibrating: absolute angles";
  ui.offroad.textContent = s.offRoad ? `${s.offRoadS.toFixed(1)} s` : "on the road";
  ui.offroadInfo.textContent = `Attention after ${RULES.offRoadAttentionS} s, break after ${RULES.offRoadBreakS} s`;
  drawSpark();
  renderTemporal(s.temporal);
}

function renderTemporal(t) {
  ui.temporal.hidden = !t;
  if (!t) return;
  ui.temporalInfo.textContent = t.probs ? (t.ready ? "last 60 s, updated every 2 s" : `face found in only ${pct(t.faceShare)} of the last minute`)
    : `collecting the first minute: ${t.filled.toFixed(0)} / 60 s`;
  ui.temporalBars.innerHTML = t.classes.map((c, k) => `<div class="tbar"><span>${c}</span><i class="meter"><span style="width:${
    t.probs ? (t.probs[k] * 100).toFixed(1) : 0}%;background:${["var(--ok)", "var(--attention)", "var(--break)"][k]}"></span></i>` +
    `<b>${t.probs ? pct(t.probs[k]) : "–"}</b></div>`).join("");
}

function drawSpark() {
  const c = ui.spark, dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = 60;
  c.width = w * dpr;
  c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const hist = state.history;
  if (hist.length < 2) return;
  const t1 = hist.at(-1).t, X = (t) => w - ((t1 - t) / 60) * w, Y = (v) => h - 4 - v * (h - 8);
  ctx.strokeStyle = "rgba(148,163,184,.35)";
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, Y(RULES.closedBlink)); ctx.lineTo(w, Y(RULES.closedBlink)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#8b5cf6";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let pen = false;
  for (const p of hist) {
    if (p.closure == null) { pen = false; continue; }
    pen ? ctx.lineTo(X(p.t), Y(p.closure)) : ctx.moveTo(X(p.t), Y(p.closure));
    pen = true;
  }
  ctx.stroke();
}

// The aligned 32x32 eye crop shared with training (web/eyecrop.js = src/eyecrop.py, checked by
// web/eyecrop.test.mjs). Only the small box of pixels the crop needs is read from the frame, at 1:1, so the
// crop sees the camera's own pixels; corners are shifted into that box. Returns a Float64Array (0-255).
const work = document.createElement("canvas");
const workCtx = work.getContext("2d", { willReadFrequently: true });

export function extractEye(source, pts, eye) {
  const [i, j] = CORNERS[eye];
  const width = source.videoWidth || source.width, height = source.videoHeight || source.height;
  let [x0, y0, x1, y1] = cropBox(pts[i], pts[j], state.cropSpec);
  x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(width, x1); y1 = Math.min(height, y1);
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  work.width = w;
  work.height = h;
  workCtx.drawImage(source, x0, y0, w, h, 0, 0, w, h);
  const gray = toGray(workCtx.getImageData(0, 0, w, h).data, w, h);
  const shift = ([x, y]) => [x - x0, y - y0];
  return eyeCrop(gray, w, h, shift(pts[i]), shift(pts[j]), state.cropSpec);
}

// Debug view: exactly what a model receives.
function renderCrops(crops) {
  const s = state.cropSpec.size;
  for (const [crop, canvas] of [[crops[0], ui.cropR], [crops[1], ui.cropL]]) {
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(s, s);
    crop.forEach((v, k) => img.data.set([v, v, v, 255], 4 * k));
    ctx.putImageData(img, 0, 0);
  }
}

let audio = null;
function beep() {
  audio ??= new AudioContext();
  const osc = audio.createOscillator(), gain = audio.createGain();
  osc.frequency.value = 880;
  gain.gain.setValueAtTime(0.15, audio.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.5);
  osc.connect(gain).connect(audio.destination);
  osc.start();
  osc.stop(audio.currentTime + 0.5);
}

// ---------------------------------------------------------------- sources

function startSource(mirrored) {
  state.mirrored = mirrored;
  ui.video.classList.toggle("mirrored", mirrored);
  ui.empty.hidden = true;
  ui.recalibrate.disabled = false;
  state.rules.reset();
  state.history = [];
  if (state.temporal) Object.assign(state.temporal, { buffer: [], probs: null, lastSample: null, lastRun: null });
  state.source = ui.video;
  state.loopId++;
  loop(state.loopId);
}

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
  } catch (err) {
    ui.empty.hidden = false;
    ui.empty.innerHTML = `<div class="empty-icon">📷</div><p>Camera not available: ${err.message}</p>` +
      `<p class="muted">Allow camera access in the address bar, or use a video file.</p>`;
    return;
  }
  ui.video.removeAttribute("src");
  ui.video.srcObject = state.stream;
  await ui.video.play();
  ui.viewport.style.aspectRatio = `${ui.video.videoWidth} / ${ui.video.videoHeight}`;
  ui.camera.textContent = "Stop camera";
  startSource(true);
}

function stop() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  state.source = null;
  ui.video.pause();
  ui.video.srcObject = null;
  ui.camera.textContent = "Start camera";
}

async function openVideo(file) {
  if (!file) return;
  stop();
  ui.video.srcObject = null;
  ui.video.src = URL.createObjectURL(file);
  ui.video.loop = true;
  try {
    await ui.video.play();
  } catch {
    // The browser refuses to play video in a background tab; it starts once the tab is visible again.
    document.addEventListener("visibilitychange", () => ui.video.play().catch(() => {}), { once: true });
  }
  await new Promise((resolve) => (ui.video.readyState >= 1 ? resolve() : ui.video.addEventListener("loadedmetadata", resolve, { once: true })));
  ui.viewport.style.aspectRatio = `${ui.video.videoWidth} / ${ui.video.videoHeight}`;
  startSource(false);
}

ui.camera.addEventListener("click", () => (state.stream ? stop() : startCamera()));
ui.file.addEventListener("change", () => openVideo(ui.file.files[0]));
ui.recalibrate.addEventListener("click", () => { state.rules.reset(); state.history = []; });
ui.showCrops.addEventListener("change", () => { ui.crops.hidden = !ui.showCrops.checked; });

// ---------------------------------------------------------------- model picker

const tip = document.createElement("div");
tip.className = "model-tip";
document.body.append(tip);

const tipStats = (m) => (m.kind === "temporal" ? `<ul><li>new drivers (5-fold CV): ${pct(m.cv_acc, 1)} on 3 classes, ` +
  `${pct(m.cv_binary, 1)} alert vs drowsy</li><li>${m.train_windows.toLocaleString("en")} windows of 60 s · ` +
  `${m.params.toLocaleString("en")} parameters</li></ul>` : m.mrl_test == null ? "" : `<ul>` +
  `<li>MRL test (infrared): ${pct(m.mrl_test, 1)} · CEW test (normal camera): ${pct(m.cew_test, 1)}</li>` +
  `<li>simulated webcam, all effects: ${pct(m.webcam_mrl, 1)} / ${pct(m.webcam_cew, 1)}</li>` +
  `<li>${m.train_images.toLocaleString("en")} training eyes · ${m.params.toLocaleString("en")} parameters · ` +
  `${m.train_minutes} min on an M4</li></ul>`);

function buildModels() {
  for (const m of state.models) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = m.version;
    chip.dataset.version = m.version;
    chip.setAttribute("role", "radio");
    chip.setAttribute("aria-checked", String(m.version === state.model));
    chip.disabled = !m.live;
    chip.addEventListener("click", () => selectModel(m.version));
    chip.addEventListener("mouseenter", () => {
      tip.innerHTML = `<b>${m.version}</b> · ${m.tagline}<ul>${m.changes.map((c) => `<li>${c}</li>`).join("")}</ul>` + tipStats(m);
      const r = chip.getBoundingClientRect();
      tip.style.left = `${Math.min(r.left, window.innerWidth - 340)}px`;
      tip.style.top = `${r.bottom + 8}px`;
      tip.classList.add("show");
    });
    chip.addEventListener("mouseleave", () => tip.classList.remove("show"));
    ui.models.append(chip);
  }
}

// rules: MediaPipe's blink scores decide; an eye model: its P(closed) decides. Same rules either way.
async function selectModel(version) {
  const m = state.models.find((x) => x.version === version);
  ui.stats.textContent = `loading ${version}…`;
  state.eyeModel = null;
  state.temporal = null;
  if (m.kind === "temporal") {
    const session = await ort.InferenceSession.create(m.file, { executionProviders: ["wasm"] });
    state.temporal = { session, mean: m.mean, std: m.std, fps: m.fps, window: m.window, features: m.features.length,
      classes: m.classes, buffer: [], probs: null };
  } else if (m.file) {
    const session = await ort.InferenceSession.create(m.file, { executionProviders: ["wasm"] });
    state.eyeModel = { ...m, session };
  }
  state.model = version;
  // eyeTrack1 takes over drowsiness: PERCLOS and yawn rules off, microsleep and eyes-off-the-road stay
  state.rules = new DriverRules(m.kind === "temporal" ? { ...RULES, perclosAttention: Infinity, perclosBreak: Infinity,
    yawnsAttention: Infinity } : RULES);  // never mix closure signals from two models in one PERCLOS window
  state.history = [];
  state.inferMs = 0;
  for (const chip of ui.models.children) chip.setAttribute("aria-checked", String(chip.dataset.version === version));
  ui.stats.textContent = `${version} ready`;
}

// ---------------------------------------------------------------- start

async function init() {
  const manifest = await (await fetch("models/manifest.json")).json();
  state.models = [RULES_MODEL, ...manifest.models.map((m) => ({ ...m, live: true })), ...COMING];
  buildModels();
  state.cropSpec = await (await fetch("eye_crop.json")).json();
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE);
  state.landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: LANDMARKER_MODEL, delegate: "GPU" },
    runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
  });
  await selectModel(manifest.default);
  ui.camera.disabled = false;
}

// For tests: feed a video URL and read the latest driver state.
window.eyeTracker = { state, selectModel, classifyEyes, extractEye, temporalStep, openVideoUrl: async (url) => openVideo(new File([await (await fetch(url)).blob()], "test.mp4", { type: "video/mp4" })) };

init().catch((err) => {
  ui.stats.textContent = `error: ${err.message}`;
  console.error(err);
});
