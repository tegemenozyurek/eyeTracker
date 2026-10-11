"""Step 17a: learn the demo's two PERCLOS thresholds (Attention, Take a break) from UTA-RLDD, with cross-validation.

PERCLOS per window comes from the demo's own rules replayed on every video (Step 12). For each test fold, the
thresholds are fitted on the other four folds and judged on the test fold's 12 people, so the scores below are for
drivers the thresholds never saw. Two ways to fit them:
  best F1          maximize the 3-class macro F1 (alert / low vigilant / drowsy)
  F1, few alarms   the same, but at least 70% of alert windows must stay "OK" in the training folds: a monitor that
                   warns alert drivers half of the time gets switched off
The demo's current hand-set thresholds (7.5% / 15%) are scored the same way. Finally both fits are made once on all
60 drivers; the "few alarms" pair is the one meant for the demo.

Output: models/rules_tuning.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "rldd5.npz"
GRID = np.round(np.arange(0.0, 0.40, 0.005), 3)
MIN_ALERT_OK = 0.70
CURRENT = (0.075, 0.15)


def predict(perclos, a, b):
    return np.where(perclos >= b, 2, np.where(perclos >= a, 1, 0))


def macro_f1(y, p):
    f = []
    for c in range(3):
        tp = np.sum((p == c) & (y == c))
        prec, rec = tp / max(1, np.sum(p == c)), tp / max(1, np.sum(y == c))
        f.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f))


def fit(perclos, y, min_alert_ok=0.0):
    best = (-1, None)
    for a in GRID:
        for b in GRID[GRID > a]:
            p = predict(perclos, a, b)
            if np.mean(p[y == 0] == 0) < min_alert_ok:
                continue
            f = macro_f1(y, p)
            if f > best[0]:
                best = (f, (float(a), float(b)))
    return best[1]


def scores(y, p):
    ad = y != 1
    return {"acc": float(np.mean(p == y)), "f1": macro_f1(y, p),
            "alert_ok": float(np.mean(p[y == 0] == 0)), "drowsy_caught": float(np.mean(p[y == 2] >= 1)),
            "binary_acc": float(np.mean((p[ad] >= 1) == (y[ad] == 2)))}


def main():
    d = np.load(DATA)
    perclos, y, fold = d["rules_perclos"], d["win_label"].astype(np.int64), d["win_fold"]
    methods = {"current 7.5% / 15%": None, "best F1": 0.0, f"F1, >= {MIN_ALERT_OK:.0%} alert OK": MIN_ALERT_OK}
    preds = {m: np.zeros(len(y), np.int64) for m in methods}
    per_fold = {m: [] for m in methods}
    for k in range(1, 6):
        te, tr = fold == k, fold != k
        for m, min_ok in methods.items():
            a, b = CURRENT if min_ok is None else fit(perclos[tr], y[tr], min_ok)
            preds[m][te] = predict(perclos[te], a, b)
            per_fold[m].append((a, b))
    lines = [f"PERCLOS thresholds on UTA-RLDD, 5-fold cross-validation ({len(y):,} windows of 60 s, 60 drivers; "
             "thresholds fitted on 4 folds, scored on the 5th)", "",
             f"{'':<34}" + "".join(f"{m:>24}" for m in methods)]
    res = {m: scores(y, preds[m]) for m in methods}
    for label, k in [("3 classes, window accuracy", "acc"), ("3 classes, macro F1", "f1"),
                     ("alert windows left OK", "alert_ok"), ("drowsy windows with a warning", "drowsy_caught"),
                     ("alert vs drowsy (any warning)", "binary_acc")]:
        lines.append(f"{label:<34}" + "".join(f"{res[m][k]:>24.1%}" for m in methods))
    lines.append(f"{'thresholds per fold':<34}" + "".join(
        f"{' '.join(f'{a:.1%}/{b:.1%}' for a, b in per_fold[m][:2]) + ' ...':>24}" for m in methods))
    final = {m: fit(perclos, y, min_ok) for m, min_ok in methods.items() if min_ok is not None}
    lines += ["", "Fitted once on all 60 drivers:"]
    lines += [f"  {m:<26} Attention at PERCLOS >= {a:.1%}, Take a break at >= {b:.1%}" for m, (a, b) in final.items()]
    report = "\n".join(lines)
    print(report)
    (ROOT / "models" / "rules_tuning.txt").write_text(report + "\n")
    (ROOT / "models" / "rules_tuning.json").write_text(json.dumps(
        {"cv": res, "final": {m: {"attention": a, "break": b} for m, (a, b) in final.items()},
         "min_alert_ok": MIN_ALERT_OK}, indent=2) + "\n")
    print("\nSaved models/rules_tuning.txt and models/rules_tuning.json")


if __name__ == "__main__":
    main()
