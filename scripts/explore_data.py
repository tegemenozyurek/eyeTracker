"""Step 4: Explore the three datasets before building anything on them.

MRL Eye   how many images each person contributed (this decides how a subject-wise split
          must be balanced), and how glasses, reflections, lighting and sensor are spread
CEW       sample eye patches next to MRL ones: how different infrared and normal-camera eyes look
UTA-RLDD  video length and face-detection rate per video, and whether simple per-video signals
          (eye closure, yawning, head movement) actually differ between alert and drowsy drivers.
          Numbers and curves only: this dataset version contains no images.

Outputs (assets/explore/):
  mrl_subjects.png     images per person, closed vs open
  mrl_attributes.png   glasses / reflections / lighting / sensor / image size
  eye_samples.png      random MRL and CEW eyes, closed and open
  rldd_videos.png      length and face-detection rate of every video
  rldd_features.png    per-video eye closure, yawning and head movement by class
  rldd_timeline.png    one driver's eye closure and head pitch over time, alert vs drowsy
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.style import COLORS, finish, setup  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "assets" / "explore"
MRL_NAME = re.compile(r"s(\d{4})_(\d{5})_(\d)_(\d)_(\d)_(\d)_(\d)_(\d{2})\.png$")
CLASSES = {0: "alert", 5: "low vigilant", 10: "drowsy"}
CLASS_COLOR = {"alert": COLORS["series"][0], "low vigilant": COLORS["series"][3], "drowsy": COLORS["series"][1]}
FPS = 10


# ---------------------------------------------------------------- MRL

def load_mrl():
    rows = []
    for p in sorted((DATA / "mrl").rglob("*.png")):
        m = MRL_NAME.search(p.name)
        if m:
            subj, _, gender, glasses, state, refl, light, sensor = m.groups()
            rows.append(dict(path=p, subject=int(subj), gender=int(gender), glasses=int(glasses), open=int(state),
                             reflections=int(refl), lighting=int(light), sensor=int(sensor)))
    return rows


def mrl_subjects(rows):
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["subject"]][r["open"]] += 1
    order = sorted(per, key=lambda s: -sum(per[s]))
    totals = np.array([sum(per[s]) for s in order])
    closed_share = np.array([per[s][0] / sum(per[s]) for s in order])
    print(f"MRL: {len(rows):,} images from {len(order)} people")
    print(f"  largest 3 people: {totals[:3].sum():,} images = {totals[:3].sum() / totals.sum():.1%} of all images")
    print(f"  smallest person: {totals.min():,}, median: {int(np.median(totals)):,}, largest: {totals.max():,}")
    print(f"  share of closed eyes per person: {closed_share.min():.0%} to {closed_share.max():.0%} "
          f"(median {np.median(closed_share):.0%})")

    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(order))
    closed = [per[s][0] for s in order]
    opened = [per[s][1] for s in order]
    ax.bar(x, closed, color=COLORS["series"][0], label="closed", width=0.8)
    ax.bar(x, opened, bottom=closed, color=COLORS["series"][1], label="open", width=0.8)
    ax.set_xticks(x, [f"{s}" for s in order], fontsize=8)
    ax.set_xlabel("person (MRL subject ID), largest first")
    ax.set_ylabel("images")
    ax.legend(frameon=False)
    ax.set_title(f"MRL Eye: images per person ({len(order)} people). A few people dominate, so the split "
                 "must balance images, not just people", loc="left", fontsize=12)
    finish(ax)
    fig.tight_layout()
    fig.savefig(OUT / "mrl_subjects.png", dpi=150)
    plt.close(fig)


def mrl_attributes(rows):
    attrs = {
        "glasses": {0: "no glasses", 1: "glasses"},
        "reflections": {0: "none", 1: "small", 2: "big"},
        "lighting": {0: "bad", 1: "good"},
        "sensor": {1: "RealSense\n640x480", 2: "IDS\n1280x1024", 3: "Aptina\n752x480"},
    }
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), gridspec_kw={"width_ratios": [2, 3, 2, 3, 4]})
    print("  attributes (share of images, and share of closed eyes within it):")
    for ax, (key, names) in zip(axes, attrs.items()):
        counts = Counter(r[key] for r in rows)
        closed = Counter(r[key] for r in rows if not r["open"])
        values = sorted(names)
        bars = ax.bar([names[v] for v in values], [counts[v] for v in values], color=COLORS["series"][2], width=0.6)
        for v, bar in zip(values, bars):
            ax.annotate(f"{counts[v] / len(rows):.0%}\n{closed[v] / max(counts[v], 1):.0%} closed",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=COLORS["muted"])
            print(f"    {key:<12} {names[v].splitlines()[0]:<12} {counts[v] / len(rows):6.1%}   "
                  f"{closed[v] / max(counts[v], 1):5.1%} closed")
        ax.set_title(key, loc="left", fontsize=11)
        ax.set_ylim(0, max(counts.values()) * 1.3)
        ax.tick_params(axis="x", labelsize=8)
        finish(ax)
    widths = np.array([Image.open(r["path"]).size[0] for r in rows[::20]])  # every 20th image is plenty
    ax = axes[-1]
    ax.hist(widths, bins=40, color=COLORS["series"][4])
    ax.set_title("image width (px, every 20th image)", loc="left", fontsize=11)
    finish(ax)
    print(f"  image width: {widths.min()} to {widths.max()} px, median {int(np.median(widths))} "
          f"(CEW patches are 24 px)")
    fig.suptitle("MRL Eye: recording conditions", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "mrl_attributes.png", dpi=150)
    plt.close(fig)
    return widths


def eye_samples(rows, widths, n=10, seed=4):
    rng = np.random.default_rng(seed)
    cew = sorted((DATA / "cew").rglob("*.jpg"))
    groups = [
        ("MRL closed", [r["path"] for r in rows if not r["open"]]),
        ("MRL open", [r["path"] for r in rows if r["open"]]),
        ("CEW closed", [p for p in cew if "closed" in p.parent.name.lower()]),
        ("CEW open", [p for p in cew if "open" in p.parent.name.lower()]),
    ]
    fig, axes = plt.subplots(len(groups), n, figsize=(n * 1.2, len(groups) * 1.35))
    for row, (label, paths) in enumerate(groups):
        for col, i in enumerate(rng.choice(len(paths), n, replace=False)):
            ax = axes[row, col]
            ax.imshow(Image.open(paths[i]).convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11, labelpad=10)
    fig.suptitle(f"Random eyes: MRL (infrared, {widths.min()}-{widths.max()} px wide) vs CEW (normal camera, 24 px)",
                 x=0.02, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "eye_samples.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- UTA-RLDD

def load_rldd():
    with open(DATA / "rldd" / "metadata.csv") as f:
        videos = list(csv.DictReader(f))
    cols = ["t_sec", "face_detected", "bs_eyeBlinkLeft", "bs_eyeBlinkRight", "bs_jawOpen", "pitch", "yaw", "roll"]
    for v in videos:
        t = pq.read_table(DATA / "rldd" / v["file_name"], columns=cols)
        v["data"] = {c: t.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in cols}
        v["cls"] = CLASSES[int(float(v["class_label"]))]
        v["ok"] = v["passed_verification"].strip().lower() == "true"
    return videos


def rldd_videos(videos):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for cls, color in CLASS_COLOR.items():
        vs = [v for v in videos if v["cls"] == cls]
        minutes = [float(v["duration_sec"]) / 60 for v in vs]
        rate = [float(v["face_detected_rate"]) for v in vs]
        axes[0].hist(minutes, bins=np.arange(6, 17, 0.5), alpha=0.65, color=color, label=cls)
        axes[1].scatter(minutes, rate, s=18, color=color, label=cls)
        print(f"  {cls:<13} {len(vs)} videos, {np.sum(minutes):.0f} min in total, "
              f"median {np.median(minutes):.1f} min, median face-detection rate {np.median(rate):.1%}")
    axes[1].axhline(0.5, color=COLORS["muted"], ls="--", lw=1)
    axes[1].text(0.6, 0.52, "quality gate: face found in ≥ 50% of frames", fontsize=8, color=COLORS["muted"],
                 transform=axes[1].get_yaxis_transform())
    axes[0].set_title("Video length (minutes)", loc="left", fontsize=11)
    axes[1].set_title("Face-detection rate vs length (one dot = one video)", loc="left", fontsize=11)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
        finish(ax)
    fig.suptitle("UTA-RLDD: the 178 videos", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "rldd_videos.png", dpi=150)
    plt.close(fig)


def video_signals(v):
    """Simple per-video signals over the frames where a face was found."""
    d = v["data"]
    face = d["face_detected"] > 0.5
    blink = (d["bs_eyeBlinkLeft"] + d["bs_eyeBlinkRight"])[face] / 2
    return {
        "PERCLOS\n(eyes closed, eyeBlink ≥ 0.5)": np.mean(blink >= 0.5),
        "mean eye-closure score\n(eyeBlink)": np.mean(blink),
        "mouth wide open\n(jawOpen > 0.3)": np.mean(d["bs_jawOpen"][face] > 0.3),
        "head pitch spread\n(std, degrees)": np.nanstd(d["pitch"][face]),
    }


def rldd_features(videos):
    good = [v for v in videos if v["ok"]]
    signals = {v["video_id"]: video_signals(v) for v in good}
    names = list(next(iter(signals.values())))
    fig, axes = plt.subplots(1, len(names), figsize=(15, 4.2))
    rng = np.random.default_rng(0)
    print(f"\n  per-video signals ({len(good)} videos that pass the quality check), median per class:")
    for ax, name in zip(axes, names):
        for i, cls in enumerate(CLASS_COLOR):
            vals = np.array([signals[v["video_id"]][name] for v in good if v["cls"] == cls])
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=14, alpha=0.75, color=CLASS_COLOR[cls])
            ax.plot([i - 0.28, i + 0.28], [np.median(vals)] * 2, color=COLORS["text"], lw=2)
        ax.set_xticks(range(3), list(CLASS_COLOR), fontsize=9)
        ax.set_title(name, loc="left", fontsize=10)
        finish(ax)
        meds = "  ".join(f"{cls} {np.median([signals[v['video_id']][name] for v in good if v['cls'] == cls]):.3f}"
                         for cls in CLASS_COLOR)
        print(f"    {name.splitlines()[0]:<24} {meds}")

    # The same person alert vs drowsy: does the signal move in the expected direction?
    by_subject = defaultdict(dict)
    for v in good:
        by_subject[v["subject_id"]][v["cls"]] = signals[v["video_id"]]
    pairs = [s for s in by_subject.values() if "alert" in s and "drowsy" in s]
    print(f"\n  same person, drowsy video vs alert video ({len(pairs)} people with both):")
    for name in names:
        up = sum(p["drowsy"][name] > p["alert"][name] for p in pairs)
        print(f"    {name.splitlines()[0]:<24} higher when drowsy for {up} of {len(pairs)} people")
    fig.suptitle("UTA-RLDD: per-video signals by class (one dot = one video, line = median). "
                 "Classes overlap a lot between people", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "rldd_features.png", dpi=150)
    plt.close(fig)


def rldd_timeline(videos):
    """One person who has a good alert and a good drowsy video: 30 s rolling averages."""
    good = {(v["subject_id"], v["cls"]): v for v in videos if v["ok"]}
    subject = sorted(s for s, c in good if c == "alert" and (s, "drowsy") in good)[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    win = 30 * FPS
    for cls in ("alert", "drowsy"):
        d = good[(subject, cls)]["data"]
        minutes = d["t_sec"] / 60
        blink = (d["bs_eyeBlinkLeft"] + d["bs_eyeBlinkRight"]) / 2
        for ax, series in ((axes[0], blink), (axes[1], d["pitch"])):
            valid = ~np.isnan(series)
            filled = np.where(valid, series, 0.0)
            smooth = np.convolve(filled, np.ones(win), "same") / np.maximum(np.convolve(valid, np.ones(win), "same"), 1)
            ax.plot(minutes, smooth, color=CLASS_COLOR[cls], lw=1.6, label=f"{cls} video")
    axes[0].set_title(f"Person {subject}: eye-closure score (eyeBlink, 30 s average)", loc="left", fontsize=11)
    axes[1].set_title("Head pitch (degrees, 30 s average)", loc="left", fontsize=11)
    axes[1].set_xlabel("minutes into the video")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
        finish(ax)
    fig.suptitle("UTA-RLDD: one driver, alert vs drowsy (features only, no images)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "rldd_timeline.png", dpi=150)
    plt.close(fig)
    print(f"\n  timeline chart: person {subject}")


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_mrl()
    mrl_subjects(rows)
    widths = mrl_attributes(rows)
    eye_samples(rows, widths)
    print("\nUTA-RLDD:")
    videos = load_rldd()
    rldd_videos(videos)
    rldd_features(videos)
    rldd_timeline(videos)
    print(f"\nSaved 6 charts to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
