"""Step 15: one summary of every model, read from the reports the earlier steps wrote (no number is retyped).

  eye state      models/<v>/test_report.txt, models/<v>/config.json        (Steps 7, 10)
  live pipeline  models/live_pipeline_check.txt                             (Step 11)
  drowsiness     models/eyeTrack1/config.json, models/eyeTrack1_personnorm/cv_report.txt   (Step 13)
  browser speed  models/browser_benchmark.json                              (scripts/browser_benchmark.py)
  paper          Ghoddoosian, Galib, Athitsos, CVPR Workshops 2019, Table 2

Outputs: models/benchmark.txt, assets/benchmark.png
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.style import COLORS, finish, setup  # noqa: E402

M = ROOT / "models"
PAPER = {"window": 0.54, "video": 0.652}


def pct(text, pattern):
    return float(re.search(pattern, text).group(1)) / 100


def eye_models():
    out = {}
    for v in ["eyeTrack0.1", "eyeTrack0.5"]:
        text = (M / v / "test_report.txt").read_text()
        cfg = json.loads((M / v / "config.json").read_text())
        out[v] = {"mrl": pct(text, r"MRL test .*?accuracy ([\d.]+)%"), "cew": pct(text, r"CEW test .*?accuracy ([\d.]+)%"),
                  "webcam_mrl": pct(text, r"all combined\s+([\d.]+)%"), "webcam_cew": pct(text, r"all combined\s+[\d.]+%\s+([\d.]+)%"),
                  "params": cfg["params"], "minutes": cfg["train_seconds"] / 60, "data": cfg["sources"]}
    live = (M / "live_pipeline_check.txt").read_text()
    for v in ["rules", "eyeTrack0.1", "eyeTrack0.5"]:
        m = re.search(rf"^{re.escape(v)}\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", live, re.M)
        out.setdefault(v, {}).update(live_acc=float(m.group(1)) / 100, live_closed=float(m.group(2)) / 100)
    out["_live_eyes"] = int(re.search(r"(\d+) eyes", live).group(1))
    return out


def drowsiness():
    cfg = json.loads((M / "eyeTrack1" / "config.json").read_text())
    cv = cfg["cv"]
    tuned = json.loads((M / "rules_tuning.json").read_text())["cv"]
    rows = {"rules, first thresholds": cv["rules (demo)"],
            "rules, learned (demo)": {**tuned["F1, >= 70% alert OK"], "video_acc": float("nan")},
            "PERCLOS, best F1": cv["rules (PERCLOS tuned)"], "eyeTrack1": cv["eyeTrack1"]}
    text = (M / "eyeTrack1_personnorm" / "cv_report.txt").read_text()
    rows["eyeTrack1 + baseline*"] = {"acc": pct(text, r"3 classes, window accuracy\s+([\d.]+)%"),
                                              "f1": pct(text, r"3 classes, macro F1\s+([\d.]+)%"),
                                              "binary_acc": pct(text, r"alert vs drowsy, window accuracy\s+([\d.]+)%"),
                                              "video_acc": pct(text, r"video accuracy \(vote\)\s+([\d.]+)%")}
    return rows, cfg


def personnorm_params():
    """The analysis run's own parameter count (Step 13 model, without the blink channels)."""
    m = re.search(r"([\d,]+) parameters", (M / "eyeTrack1_personnorm" / "train_log.txt").read_text())
    return m.group(1) if m else "-"


def main():
    setup()
    eyes, (drowsy, cfg) = eye_models(), drowsiness()
    browser = json.loads((M / "browser_benchmark.json").read_text())
    ms = lambda x: "< 0.1 ms" if x < 0.1 else f"{x:.1f} ms"  # Chrome's timer has 0.1 ms steps

    lines = ["EYE STATE (open / closed), people never seen in training", "",
             f"{'':<34}{'rules':>12}{'eyeTrack0.1':>14}{'eyeTrack0.5':>14}"]
    row = lambda label, f: lines.append(f"{label:<34}" + "".join(f"{f(v):>{w}}" for v, w in
                                                                 [('rules', 12), ('eyeTrack0.1', 14), ('eyeTrack0.5', 14)]))
    get = lambda v, k, fmt="{:.1%}": fmt.format(eyes[v][k]) if k in eyes[v] else "-"
    row("trained on", lambda v: {"rules": "nothing", "eyeTrack0.1": "MRL", "eyeTrack0.5": "MRL+CEW, aug"}[v])
    row("MRL test (infrared)", lambda v: get(v, "mrl"))
    row("CEW test (normal camera)", lambda v: get(v, "cew"))
    row("simulated webcam, MRL / CEW", lambda v: "-" if v == "rules" else f"{eyes[v]['webcam_mrl']:.1%} / {eyes[v]['webcam_cew']:.1%}")
    row(f"live pipeline, {eyes['_live_eyes']} real eyes", lambda v: get(v, "live_acc"))
    row("  closed eyes caught", lambda v: get(v, "live_closed"))
    row("parameters", lambda v: "-" if v == "rules" else f"{eyes[v]['params']:,}")
    row("training time (M4)", lambda v: "-" if v == "rules" else f"{eyes[v]['minutes']:.1f} min")
    row("browser, both eyes", lambda v: ms(browser["per_frame_ms"]["rules"]) if v == "rules" else ms(browser["per_frame_ms"][f"{v} (both eyes)"]))

    names = list(drowsy)
    lines += ["", "DROWSINESS over 60 s (UTA-RLDD, 60 drivers, 5-fold cross-validation, people never seen in training)", "",
              f"{'':<34}" + "".join(f"{n:>26}" for n in names)]
    for label, k in [("3 classes, window accuracy", "acc"), ("3 classes, macro F1", "f1"), ("3 classes, video accuracy", "video_acc"),
                     ("alert vs drowsy, window accuracy", "binary_acc")]:
        lines.append(f"{label:<34}" + "".join(f"{'-' if np.isnan(drowsy[n][k]) else f'{drowsy[n][k]:.1%}':>26}" for n in names))
    lines += [f"{'parameters':<34}" + "".join(f"{v:>26}" for v in ["-", "2 thresholds", "2 thresholds", f"{cfg['params']:,}",
                                                                   personnorm_params()]),
              "", "rules, learned: PERCLOS thresholds fitted on 4 folds with >= 70% of alert windows left OK, scored on the 5th"
              " (scripts/tune_rules.py);", "  the demo uses the pair fitted on all 60 drivers (4.5% / 11.5%). Video accuracy was not computed for it.",
              "* compares each driver with their own average over all their recordings: needs data from several sessions,",
              "  so it is an analysis, not a model the demo can run (Step 13 model, without the blink channels).",
              f"Paper (Ghoddoosian et al. 2019, Table 2, HM-LSTM on blink features): {PAPER['window']:.0%} per blink sequence, "
              f"{PAPER['video']:.1%} per video.",
              "", f"BROWSER, time per frame ({browser['browser']}, {browser['cores']} cores, median of {browser['frames']} frames, "
              f"page visible: {browser['visible']})", ""]
    for k, v in browser["per_frame_ms"].items():
        lines.append(f"  {k:<34}{ms(v):>12}")
    lines.append(f"  {'whole pipeline with eyeTrack0.5':<34}{ms(browser['total_ms_eyetrack05']):>12}  "
                 f"(up to {1000 / browser['total_ms_eyetrack05']:.0f} frames per second)")
    report = "\n".join(lines)
    print(report)
    (M / "benchmark.txt").write_text(report + "\n")
    plot(eyes, drowsy, browser)
    print("\nSaved models/benchmark.txt, assets/benchmark.png")


def plot(eyes, drowsy, browser):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [5, 4, 2.4]})
    # 1. eye state
    ax = axes[0]
    groups = [("MRL test", "mrl"), ("CEW test", "cew"), ("webcam sim.\nMRL", "webcam_mrl"), ("webcam sim.\nCEW", "webcam_cew"),
              ("live pipeline\nreal faces", "live_acc")]
    x = np.arange(len(groups))
    for i, v in enumerate(["rules", "eyeTrack0.1", "eyeTrack0.5"]):
        vals = [eyes[v].get(k, np.nan) for _, k in groups]
        b = ax.bar(x + (i - 1) * 0.27, vals, 0.25, color=COLORS["series"][[4, 0, 2][i]], label=v)
        for bar, val in zip(b, vals):
            if not np.isnan(val):
                ax.annotate(f"{val:.0%}", (bar.get_x() + bar.get_width() / 2, val), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7.5, color=COLORS["muted"])
    ax.set_xticks(x, [g for g, _ in groups], fontsize=9)
    ax.set_ylim(0.4, 1.14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, ncols=3, loc="upper left", fontsize=9)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.set_title("Eye state: open or closed (test people)\n", loc="left", fontsize=12)
    ax.text(0, 1.02, "rules need a whole face, so they appear in the live-pipeline test only", transform=ax.transAxes,
            fontsize=8.5, color=COLORS["muted"])
    finish(ax)
    # 2. drowsiness
    ax = axes[1]
    names = list(drowsy)
    groups = [("3 classes\nwindow accuracy", "acc"), ("3 classes\nmacro F1", "f1"), ("alert vs drowsy", "binary_acc")]
    x = np.arange(len(groups))
    colors = [COLORS["series"][4], COLORS["series"][0], COLORS["series"][3], COLORS["series"][2], COLORS["series"][2]]
    for i, n in enumerate(names):
        vals = [drowsy[n][k] for _, k in groups]
        b = ax.bar(x + (i - 2) * 0.16, vals, 0.15, color=colors[i], label=n, hatch="///" if "*" in n else None,
                   edgecolor="white" if "*" in n else None, alpha=0.55 if "*" in n else 1)
        for bar, val in zip(b, vals):
            ax.annotate(f"{val:.0%}", (bar.get_x() + bar.get_width() / 2, val), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7.5, color=COLORS["muted"])
    ax.plot([-0.45, 0.45], [PAPER["window"]] * 2, color=COLORS["text"], ls="--", lw=1.2)
    ax.text(0, PAPER["window"] + 0.012, f"paper {PAPER['window']:.0%}", ha="center", fontsize=8)
    ax.axhline(1 / 3, color=COLORS["muted"], ls=":", lw=0.8)
    ax.set_xticks(x, [g for g, _ in groups], fontsize=9)
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Drowsiness over 60 s (new drivers, 5-fold CV)", loc="left", fontsize=12)
    finish(ax)
    # 3. browser time per frame
    ax = axes[2]
    parts = [("face landmarker", browser["per_frame_ms"]["MediaPipe face landmarker"]),
             ("eye crops", browser["per_frame_ms"]["two aligned eye crops"]),
             ("eyeTrack0.5", browser["per_frame_ms"]["eyeTrack0.5 (both eyes)"]),
             ("features + rules", browser["per_frame_ms"]["13 eyeTrack1 features"] + browser["per_frame_ms"]["rules"])]
    bottom = 0
    for i, (label, v) in enumerate(parts):
        ax.bar(0, v, 0.5, bottom=bottom, color=COLORS["series"][[0, 3, 2, 4][i]],
               label=f"{label} {'< 0.1' if v < 0.1 else f'{v:.1f}'} ms")
        bottom += v
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylabel("ms per frame")
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.6, 1))
    ax.set_title(f"Browser: {bottom:.1f} ms per frame", loc="left", fontsize=12)
    finish(ax)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
