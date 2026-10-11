"""Step 17b: test cases for the Python / JavaScript blink-channel parity test (web/blink_channels.test.mjs):
4 real UTA-RLDD windows (raw features and face flags) with the channels src/blink_channels.py computes."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.blink_channels import blink_channels  # noqa: E402

d = np.load(ROOT / "data" / "processed" / "rldd5.npz")
size, fps = int(d["window"]), int(d["fps"])
picks = [int(np.flatnonzero(d["win_label"] == c)[7]) for c in (0, 1, 2)] + [int(np.flatnonzero(d["win_valid"] < 0.9)[0])]
cases = []
for w in picks:
    start = d["offsets"][d["win_video"][w]] + d["win_start"][w]
    x, face = d["x"][start:start + size].astype(np.float64), d["face"][start:start + size]
    ch = blink_channels(x, face, fps)
    cases.append({"rows": [None if not f else [round(float(v), 6) for v in r] for r, f in zip(x, face)],
                  "channels": np.round(ch, 6).tolist()})
out = ROOT / "web" / "blink_channels.fixture.json"
out.write_text(json.dumps({"fps": fps, "cases": cases}))
print(f"Wrote {len(cases)} windows of {size} frames to {out.relative_to(ROOT)}")
