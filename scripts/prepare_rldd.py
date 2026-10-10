"""Step 12: UTA-RLDD per-frame features -> 60 s windows -> subject-wise folds, plus the rules baseline on every window.

Frames
  The 168 videos that pass the dataset's quality check, halved from 10 to 5 frames per second.
  13 features per frame, all computable in the browser from MediaPipe's face landmarker, defined exactly
  as in the dataset's extraction code (SleepDetective, branch dataset-release, src/extract.py):
    eyeBlink, eyeSquint and eyeLookDown blendshapes for each eye, jawOpen,
    EAR per eye and MAR (3D distances between normalized landmarks), pitch, yaw, roll (degrees).
  Frames without a face keep zeros and a mask flag.
Windows
  60 s = 300 frames, a new window every 10 s, kept when a face was found in at least 70% of its frames.
  Label = the video's label (alert, low vigilant, drowsy). Folds = the authors' 5 folds (12 people each),
  so a person is never in two folds.
Rules
  Every video is replayed through the demo's own rules (web/rules.js, run by tools/rules_on_rldd.mjs) as it
  would run live; its level (OK / Attention / Take a break) and PERCLOS at the end of each window are stored,
  so Step 13 compares eyeTrack1 with exactly what the browser does.

Output: data/processed/rldd5.npz, assets/rldd/windows.png, assets/rldd/rules_by_class.png
"""
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.runlog import RunLog  # noqa: E402
from src.style import COLORS, finish, setup  # noqa: E402

DATA = ROOT / "data" / "rldd"
OUT = ROOT / "data" / "processed" / "rldd5.npz"
ASSETS = ROOT / "assets" / "rldd"
FEATURES = ["bs_eyeBlinkLeft", "bs_eyeBlinkRight", "bs_eyeSquintLeft", "bs_eyeSquintRight", "bs_eyeLookDownLeft",
            "bs_eyeLookDownRight", "bs_jawOpen", "ear_left", "ear_right", "mar", "pitch", "yaw", "roll"]
CLASSES = ["alert", "low vigilant", "drowsy"]
LABEL = {0: 0, 5: 1, 10: 2}
FPS, WINDOW_S, STRIDE_S, MIN_FACE = 5, 60, 10, 0.7
LEVELS = ["OK", "Attention", "Take a break"]


def load_videos(log):
    with open(DATA / "metadata.csv") as f:
        meta = [v for v in csv.DictReader(f) if v["passed_verification"].strip().lower() == "true"]
    videos = []
    for n, v in enumerate(meta, 1):
        table = pq.read_table(DATA / v["file_name"], columns=["t_sec", "face_detected", *FEATURES])
        cols = {c: table.column(c).to_numpy(zero_copy_only=False) for c in table.column_names}
        order = np.argsort(cols["t_sec"])[::2]  # 10 fps -> 5 fps: every other frame
        face = cols["face_detected"][order].astype(bool)
        x = np.stack([cols[c][order].astype(np.float64) for c in FEATURES], 1)
        face &= ~np.isnan(x).any(1)
        x[~face] = 0.0
        videos.append({"id": v["video_id"], "subject": int(v["subject_id"]), "fold": int(v["fold"]),
                       "label": LABEL[int(float(v["class_label"]))], "t": cols["t_sec"][order], "x": x.astype(np.float32),
                       "face": face})
        log.progress(n, len(meta), message=v["video_id"])
    return videos


def windows(videos):
    size, step = WINDOW_S * FPS, STRIDE_S * FPS
    rows = []
    for vi, v in enumerate(videos):
        for start in range(0, len(v["t"]) - size + 1, step):
            valid = v["face"][start:start + size].mean()
            if valid >= MIN_FACE:
                rows.append((vi, start, valid))
    return rows


def positions(videos):
    size, step = WINDOW_S * FPS, STRIDE_S * FPS
    return sum((len(v["t"]) - size) // step + 1 for v in videos if len(v["t"]) >= size)


def run_rules(videos, wins):
    size = WINDOW_S * FPS
    ends = [[] for _ in videos]
    for vi, start, _ in wins:
        ends[vi].append(start + size - 1)
    payload = {"videos": [{"t": v["t"].round(3).tolist(), "face": v["face"].astype(int).tolist(),
                           "blinkL": v["x"][:, 0].round(4).tolist(), "blinkR": v["x"][:, 1].round(4).tolist(),
                           "jaw": v["x"][:, 6].round(4).tolist(), "ends": e} for v, e in zip(videos, ends)]}
    tmp = ROOT / "runs" / "_rldd_rules"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "input.json").write_text(json.dumps(payload))
    subprocess.run(["node", str(ROOT / "tools" / "rules_on_rldd.mjs"), str(tmp / "input.json"), str(tmp / "output.json")],
                   check=True)
    out = json.loads((tmp / "output.json").read_text())["videos"]
    level, perclos = [], []
    for v in out:
        level += v["level"]
        perclos += v["perclos"]
    return np.array(level, np.int8), np.array(perclos, np.float32)


def main():
    setup()
    with RunLog("_prepare_rldd", kind="job", unit="videos") as log:
        videos = load_videos(log)
    wins = windows(videos)
    level, perclos = run_rules(videos, wins)
    win_video = np.array([w[0] for w in wins], np.int32)
    win_start = np.array([w[1] for w in wins], np.int32)
    win_label = np.array([videos[i]["label"] for i in win_video], np.int8)
    win_fold = np.array([videos[i]["fold"] for i in win_video], np.int8)
    win_subject = np.array([videos[i]["subject"] for i in win_video], np.int16)

    frames = sum(len(v["t"]) for v in videos)
    print(f"{len(videos)} videos from {len({v['subject'] for v in videos})} people, {frames:,} frames at {FPS} fps "
          f"({frames / FPS / 3600:.1f} h), face found in {np.mean(np.concatenate([v['face'] for v in videos])):.1%}")
    print(f"{len(wins):,} windows of {WINDOW_S} s (every {STRIDE_S} s, face in >= {MIN_FACE:.0%} of frames); "
          f"{positions(videos) - len(wins):,} of {positions(videos):,} positions dropped for too few faces")
    print(f"\n{'fold':<6}{'people':>7}" + "".join(f"{c:>14}" for c in CLASSES))
    for f in sorted(set(win_fold)):
        m = win_fold == f
        print(f"{f:<6}{len(set(win_subject[m])):>7}" + "".join(f"{np.sum(m & (win_label == k)):>14,}" for k in range(3)))
    leaks = [s for s in set(win_subject) if len(set(win_fold[win_subject == s])) > 1]
    print(f"people in more than one fold: {len(leaks)}")

    print("\nRules baseline (web/rules.js) at the end of each window:")
    print(f"{'true class':<14}{'median PERCLOS':>15}" + "".join(f"{l:>14}" for l in LEVELS))
    for k, c in enumerate(CLASSES):
        m = win_label == k
        print(f"{c:<14}{np.median(perclos[m]):>15.1%}" + "".join(f"{np.mean(level[m] == l):>14.1%}" for l in range(3)))

    # per-feature statistics for normalization, from all frames with a face (Step 13 uses training folds only)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(v["t"]) for v in videos])
    np.savez_compressed(
        OUT, x=np.concatenate([v["x"] for v in videos]), face=np.concatenate([v["face"] for v in videos]),
        offsets=offsets, video_id=np.array([v["id"] for v in videos]), video_subject=np.array([v["subject"] for v in videos]),
        video_fold=np.array([v["fold"] for v in videos]), video_label=np.array([v["label"] for v in videos]),
        win_video=win_video, win_start=win_start, win_label=win_label, win_fold=win_fold, win_subject=win_subject,
        win_valid=np.array([w[2] for w in wins], np.float32), rules_level=level, rules_perclos=perclos,
        features=np.array(FEATURES), classes=np.array(CLASSES), fps=FPS, window=WINDOW_S * FPS)
    print(f"\nSaved {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.0f} MB)")
    plots(win_fold, win_label, level, perclos)


def plots(win_fold, win_label, level, perclos):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    folds = sorted(set(win_fold))
    bottom = np.zeros(len(folds))
    for k, c in enumerate(CLASSES):
        vals = np.array([np.sum((win_fold == f) & (win_label == k)) for f in folds])
        ax.bar([f"fold {f}" for f in folds], vals, bottom=bottom, color=COLORS["series"][[0, 3, 1][k]], label=c, width=0.6)
        bottom += vals
    ax.legend(frameon=False, ncols=3)
    ax.set_title("UTA-RLDD: 60 s windows per fold and class (no person in two folds)", loc="left", fontsize=12)
    finish(ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "windows.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    data = [perclos[win_label == k] for k in range(3)]
    axes[0].boxplot(data, showfliers=False, widths=0.5)
    axes[0].set_xticks([1, 2, 3], CLASSES)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_title("PERCLOS per window, by true class", loc="left", fontsize=11)
    for k in range(3):
        shares = [np.mean(level[win_label == k] == l) for l in range(3)]
        left = 0
        for l, s in enumerate(shares):
            axes[1].barh(CLASSES[k], s, left=left, color=[COLORS["series"][2], COLORS["series"][3], COLORS["series"][1]][l],
                         label=LEVELS[l] if k == 0 else None)
            if s > 0.06:
                axes[1].text(left + s / 2, k, f"{s:.0%}", ha="center", va="center", fontsize=9, color="white")
            left += s
    axes[1].invert_yaxis()
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axes[1].set_title("What the rules say at the end of each window", loc="left", fontsize=11)
    for ax in axes:
        finish(ax)
    fig.suptitle("The rules baseline (web/rules.js) replayed on UTA-RLDD", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSETS / "rules_by_class.png", dpi=150)
    plt.close(fig)
    print("Saved assets/rldd/windows.png, assets/rldd/rules_by_class.png")


if __name__ == "__main__":
    main()
