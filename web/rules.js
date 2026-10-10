// The `rules` baseline: classic driver-monitoring rules on MediaPipe's per-frame face signals.
// No learning. Pure logic (no DOM), so it can be tested in Node with synthetic frames (web/rules.test.mjs).
//
// Every threshold is in RULES below. They are provisional, informed by Step 4 (median PERCLOS per video:
// alert 2.2%, drowsy 9.8%); Steps 12-13 tune and evaluate them on UTA-RLDD.

export const RULES = {
  closedBlink: 0.5,        // eye counts as closed when MediaPipe's eyeBlink score is at least this (as in the RLDD features)
  perclosWindowS: 60,      // PERCLOS = share of time with closed eyes over the last 60 s
  perclosMinS: 20,         // judge PERCLOS only once this much time has been observed
  perclosAttention: 0.075, // PERCLOS levels for Attention / Take a break
  perclosBreak: 0.15,
  longClosureS: 1.5,       // one closure this long (a microsleep) means Take a break at once
  yawnJaw: 0.5,            // jawOpen above this ...
  yawnMinS: 1.5,           // ... for this long counts as a yawn
  yawnWindowS: 300,        // yawns counted over the last 5 minutes
  yawnsAttention: 3,
  calibrateS: 3,           // the first seconds of a face define "looking at the road"
  offRoadYaw: 25,          // degrees away from the calibrated direction that count as eyes off the road
  offRoadPitchDown: 20,
  offRoadAttentionS: 2,    // continuous seconds off the road for Attention / Take a break
  offRoadBreakS: 4,
  nodDropDeg: 12,          // a nod: pitch drops this far below its recent median ...
  nodMaxS: 2,              // ... and comes back within this time
  holdS: 4,                // a level must have been gone this long before the display steps down (no flicker)
};

export const LEVELS = ["OK", "Attention", "Take a break"];

const median = (values) => {
  const s = [...values].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

export class DriverRules {
  constructor(rules = RULES) {
    this.r = rules;
    this.reset();
  }

  reset() {
    this.samples = [];          // {t, closed} for PERCLOS
    this.closedSince = null;    // start time of the current closure
    this.blinks = [];           // {t, duration}
    this.longClosures = 0;
    this.jawSince = null;
    this.yawns = [];            // times
    this.calib = [];            // {yaw, pitch} during calibration
    this.road = null;           // calibrated {yaw, pitch}
    this.offRoadSince = null;
    this.pitchHistory = [];     // {t, pitch} for nod detection
    this.nodStart = null;
    this.nods = 0;
    this.shown = 0;             // displayed level
    this.lastAbove = [0, 0, 0]; // last time each level was triggered
    this.startT = null;
    this.lastT = null;
  }

  // frame: {t (seconds), face (bool), blinkL, blinkR, jawOpen, yaw, pitch, roll}
  update(f) {
    const r = this.r, t = f.t;
    this.startT ??= t;
    const dt = this.lastT == null ? 0 : Math.min(t - this.lastT, 0.5);  // ignore long gaps (tab in background)
    this.lastT = t;
    const reasons = [];

    // --- eyes: closure, blinks, PERCLOS
    let closed = false;
    if (f.face) {
      closed = (f.blinkL + f.blinkR) / 2 >= r.closedBlink;
      this.samples.push({ t, closed, dt });
      if (closed && this.closedSince == null) this.closedSince = t;
      if (!closed && this.closedSince != null) {
        const duration = t - this.closedSince;
        this.blinks.push({ t, duration });
        if (duration >= r.longClosureS) this.longClosures++;
        this.closedSince = null;
      }
    }
    while (this.samples.length && this.samples[0].t < t - r.perclosWindowS) this.samples.shift();
    const observed = this.samples.reduce((a, s) => a + s.dt, 0);
    const perclos = observed > 0 ? this.samples.reduce((a, s) => a + (s.closed ? s.dt : 0), 0) / observed : 0;
    const closureNow = this.closedSince != null ? t - this.closedSince : 0;

    // --- mouth: yawns
    if (f.face && f.jawOpen > r.yawnJaw) {
      this.jawSince ??= t;
    } else if (this.jawSince != null) {
      if (t - this.jawSince >= r.yawnMinS) this.yawns.push(t);
      this.jawSince = null;
    }
    while (this.yawns.length && this.yawns[0] < t - r.yawnWindowS) this.yawns.shift();

    // --- head: calibration, eyes off the road, nods
    let offRoad = false;
    if (f.face) {
      if (!this.road) {
        this.calib.push({ yaw: f.yaw, pitch: f.pitch, t });
        if (t - this.calib[0].t >= r.calibrateS) {
          this.road = { yaw: median(this.calib.map((c) => c.yaw)), pitch: median(this.calib.map((c) => c.pitch)) };
        }
      } else {
        offRoad = Math.abs(f.yaw - this.road.yaw) > r.offRoadYaw || this.road.pitch - f.pitch > r.offRoadPitchDown;
      }
      this.pitchHistory.push({ t, pitch: f.pitch });
      while (this.pitchHistory[0].t < t - 30) this.pitchHistory.shift();
      const base = median(this.pitchHistory.map((p) => p.pitch));
      if (base - f.pitch > r.nodDropDeg) this.nodStart ??= t;
      else if (this.nodStart != null) {
        if (t - this.nodStart <= r.nodMaxS) this.nods++;
        this.nodStart = null;
      }
    } else if (this.road) {
      offRoad = true;  // no face in view after calibration: the driver turned away
    }
    if (offRoad) this.offRoadSince ??= t;
    else this.offRoadSince = null;
    const offRoadS = this.offRoadSince != null ? t - this.offRoadSince : 0;

    // --- decide the level for this frame
    let level = 0;
    const perclosReady = observed >= r.perclosMinS;
    if (closureNow >= r.longClosureS) { level = 2; reasons.push(`eyes closed for ${closureNow.toFixed(1)} s`); }
    if (perclosReady && perclos >= r.perclosBreak) { level = 2; reasons.push(`PERCLOS ${(perclos * 100).toFixed(0)}%`); }
    else if (perclosReady && perclos >= r.perclosAttention) { level = Math.max(level, 1); reasons.push(`PERCLOS ${(perclos * 100).toFixed(0)}%`); }
    if (this.yawns.length >= r.yawnsAttention) { level = Math.max(level, 1); reasons.push(`${this.yawns.length} yawns in 5 min`); }
    if (offRoadS >= r.offRoadBreakS) { level = 2; reasons.push(`eyes off the road for ${offRoadS.toFixed(1)} s`); }
    else if (offRoadS >= r.offRoadAttentionS) { level = Math.max(level, 1); reasons.push(`eyes off the road for ${offRoadS.toFixed(1)} s`); }

    // Step up at once, step down only after the higher level has been quiet for holdS.
    for (let l = 1; l <= level; l++) this.lastAbove[l] = t;
    if (level >= this.shown) this.shown = level;
    else while (this.shown > level && t - this.lastAbove[this.shown] >= r.holdS) this.shown--;

    const minutes = Math.max((t - this.startT) / 60, 1 / 60);
    const recent = this.blinks.filter((b) => b.t >= t - 60);
    return {
      level: this.shown, levelName: LEVELS[this.shown], reasons,
      face: f.face, closed, closureNow,
      blinks: this.blinks.length, blinksPerMin: (recent.length * 60) / Math.min(60, Math.max(t - this.startT, 10)),
      lastBlinkS: this.blinks.at(-1)?.duration ?? null, longClosures: this.longClosures,
      perclos, perclosReady, observedS: observed,
      yawns: this.yawns.length, yawning: this.jawSince != null,
      calibrated: !!this.road, road: this.road, offRoad, offRoadS, nods: this.nods,
      elapsedMin: minutes,
    };
  }
}
