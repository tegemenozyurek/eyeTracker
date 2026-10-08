"""Step 5: Eye crops and subject-wise splits for MRL Eye and CEW.

Crops
  Every eye becomes a 32x32 grayscale image (uint8). MRL crops (52-282 px) are shrunk by
  averaging pixel blocks (PIL BOX), the same way the live demo will shrink webcam eyes;
  CEW patches (24 px) are enlarged bilinearly. Non-square MRL crops are centre-cropped first.

Splits: no person appears in two splits
  MRL   37 people. A seeded search over many random assignments picks the one whose
        train / val / test shares are closest to 70 / 15 / 15% of the IMAGES, with a closed-eye
        share close to the overall one in every split, every camera present in test, and at
        least 10 / 5 / 7 people per split.
  CEW   open eyes: grouped by person (file names are LFW names, e.g. George_W_Bush_0003_L);
        closed eyes: grouped by source photo (closed_eye_0001...). Left and right eye of a face
        always stay together. Groups are shuffled into 70 / 15 / 15%.

Output
  data/processed/eyes32.npz   X, y (0 closed, 1 open), source (0 MRL, 1 CEW), split (0 train,
                              1 val, 2 test), group, and the MRL recording conditions
                              (sensor, lighting, glasses, reflections; -1 for CEW)
  assets/splits.png           images and closed-eye share per split and dataset
  assets/crops.png            what the model will see: 32x32 crops per dataset and class
"""
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.runlog import RunLog  # noqa: E402
from src.style import COLORS, finish, setup  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "processed" / "eyes32.npz"
ASSETS = ROOT / "assets"
SIZE = 32
SEED = 42
TARGET = np.array([0.70, 0.15, 0.15])
MIN_PEOPLE = np.array([10, 5, 7])  # MRL people per split: results must not hinge on 2-3 individuals
SPLITS = ["train", "val", "test"]
MRL_NAME = re.compile(r"s(\d{4})_(\d{5})_(\d)_(\d)_(\d)_(\d)_(\d)_(\d{2})\.png$")
CEW_OPEN = re.compile(r"^(.+)_\d{4}_[LR]\.jpg$")
CEW_CLOSED = re.compile(r"^(closed_eye_\d+)\.\w+_face_\d+_[LR]\.jpg$")  # source photos were .jpg, .JPG, .BMP or .png


def to_crop(path):
    """Any eye image -> SIZE x SIZE uint8: centre square, then box-average down or bilinear up."""
    img = Image.open(path).convert("L")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
    method = Image.Resampling.BOX if side >= SIZE else Image.Resampling.BILINEAR
    return np.asarray(img.resize((SIZE, SIZE), method), dtype=np.uint8), w != h


def load(log):
    items = []
    for p in sorted((DATA / "mrl").rglob("*.png")):
        m = MRL_NAME.search(p.name)
        if m:
            subj, _, _, glasses, state, refl, light, sensor = m.groups()
            items.append(dict(path=p, y=int(state), source=0, group=f"mrl_{subj}",
                              sensor=int(sensor), lighting=int(light), glasses=int(glasses), reflections=int(refl)))
    skipped = []
    for p in sorted((DATA / "cew").rglob("*.jpg")):
        is_open = "open" in p.parent.name.lower()
        m = (CEW_OPEN if is_open else CEW_CLOSED).match(p.name)
        if not m:
            skipped.append(p.name)
            continue
        items.append(dict(path=p, y=int(is_open), source=1, group=f"cew_{'open' if is_open else 'closed'}_{m.group(1)}",
                          sensor=-1, lighting=-1, glasses=-1, reflections=-1))
    X = np.zeros((len(items), SIZE, SIZE), np.uint8)
    not_square = 0
    t0 = time.time()
    for i, it in enumerate(items):
        X[i], ns = to_crop(it["path"])
        not_square += ns
        if i % 500 == 0 or i == len(items) - 1:
            log.progress(i + 1, len(items), message=f"cropping {it['path'].parent.name}/{it['path'].name}")
    print(f"Loaded {len(items):,} eyes in {time.time() - t0:.0f} s ({not_square:,} non-square MRL crops centre-cropped; "
          f"{len(skipped)} CEW files with unexpected names skipped)")
    return items, X


def split_mrl(items, trials=20000):
    """Search random person -> split assignments; keep the most balanced one."""
    per = defaultdict(lambda: np.zeros(2))
    sensors = defaultdict(set)
    for it in items:
        if it["source"] == 0:
            per[it["group"]][it["y"]] += 1
            sensors[it["group"]].add(it["sensor"])
    people = sorted(per)
    counts = np.array([per[p] for p in people])           # (people, [closed, open])
    total, closed_share = counts.sum(), counts[:, 0].sum() / counts.sum()
    all_sensors = set().union(*sensors.values())
    rng = np.random.default_rng(SEED)
    best = (np.inf, None)
    for _ in range(trials):
        assign = rng.choice(3, len(people), p=TARGET)
        size = np.array([counts[assign == s].sum() for s in range(3)])
        if (size == 0).any():
            continue
        share = size / total
        closed = np.array([counts[assign == s, 0].sum() for s in range(3)]) / size
        test_sensors = set().union(*(sensors[p] for p, a in zip(people, assign) if a == 2))
        if test_sensors != all_sensors or (np.bincount(assign, minlength=3) < MIN_PEOPLE).any():
            continue  # every camera must be tested; enough people in every split
        score = np.abs(share - TARGET).sum() + np.abs(closed - closed_share).sum()
        if score < best[0]:
            best = (score, assign)
    return dict(zip(people, best[1]))


def split_cew(items):
    groups = sorted({it["group"] for it in items if it["source"] == 1})
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(groups))
    cut = np.cumsum(TARGET)[:2] * len(groups)
    return {groups[i]: int(np.searchsorted(cut, rank, side="right")) for rank, i in enumerate(order)}


def report(items, split, group):
    y, source = np.array([it["y"] for it in items]), np.array([it["source"] for it in items])
    print(f"\n{'':<6}{'':<6}{'images':>8} {'share':>7} {'closed':>7} {'people/groups':>14}")
    for src, name in [(0, "MRL"), (1, "CEW")]:
        n_src = np.sum(source == src)
        for s in range(3):
            mask = (source == src) & (split == s)
            print(f"{name:<6}{SPLITS[s]:<6}{mask.sum():>8,} {mask.sum() / n_src:>7.1%} {1 - y[mask].mean():>7.1%} "
                  f"{len(set(group[mask])):>14,}")
    leaks = [g for g in set(group) if len(set(split[group == g])) > 1]
    print(f"\nPeople / groups that appear in more than one split: {len(leaks)}")
    mrl_test = sorted({g for g in set(group[(source == 0) & (split == 2)])})
    print(f"MRL test people: {', '.join(g.removeprefix('mrl_') for g in mrl_test)}")
    sensors = np.array([it["sensor"] for it in items])
    for code, name in [(1, "RealSense"), (2, "IDS"), (3, "Aptina")]:
        m = (source == 0) & (sensors == code)
        print(f"  {name:<9} images per split: " + " / ".join(f"{np.sum(m & (split == s)):,}" for s in range(3)))
    return leaks


def plot_splits(items, split):
    y, source = np.array([it["y"] for it in items]), np.array([it["source"] for it in items])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (src, name) in zip(axes, [(0, "MRL Eye"), (1, "CEW")]):
        closed = [np.sum((source == src) & (split == s) & (y == 0)) for s in range(3)]
        opened = [np.sum((source == src) & (split == s) & (y == 1)) for s in range(3)]
        ax.bar(SPLITS, closed, color=COLORS["series"][0], label="closed", width=0.55)
        ax.bar(SPLITS, opened, bottom=closed, color=COLORS["series"][1], label="open", width=0.55)
        for i in range(3):
            n = closed[i] + opened[i]
            ax.annotate(f"{n:,}\n{closed[i] / n:.0%} closed", (i, n), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=COLORS["muted"])
        ax.set_ylim(0, max(c + o for c, o in zip(closed, opened)) * 1.25)
        ax.set_title(f"{name}: no person in two splits", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        finish(ax)
    fig.suptitle("Subject-wise splits: images per split", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(ASSETS / "splits.png", dpi=150)
    plt.close(fig)


def plot_crops(items, X, n=12, seed=1):
    rng = np.random.default_rng(seed)
    y, source = np.array([it["y"] for it in items]), np.array([it["source"] for it in items])
    rows = [("MRL closed", (source == 0) & (y == 0)), ("MRL open", (source == 0) & (y == 1)),
            ("CEW closed", (source == 1) & (y == 0)), ("CEW open", (source == 1) & (y == 1))]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 0.95, len(rows) * 1.05))
    for r, (label, mask) in enumerate(rows):
        for c, i in enumerate(rng.choice(np.flatnonzero(mask), n, replace=False)):
            axes[r, c].imshow(X[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10, labelpad=8)
    fig.suptitle(f"What the model sees: {SIZE}x{SIZE} grayscale eye crops", x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "crops.png", dpi=150)
    plt.close(fig)


def main():
    setup()
    with RunLog("_preprocess", kind="job", unit="eyes") as log:
        items, X = load(log)
    assign = {**split_mrl(items), **split_cew(items)}
    group = np.array([it["group"] for it in items])
    split = np.array([assign[g] for g in group], dtype=np.uint8)
    leaks = report(items, split, group)
    assert not leaks, "a person or group appears in more than one split"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    col = lambda key, dtype: np.array([it[key] for it in items], dtype=dtype)
    np.savez_compressed(OUT, X=X, y=col("y", np.uint8), source=col("source", np.uint8), split=split, group=group,
                        sensor=col("sensor", np.int8), lighting=col("lighting", np.int8),
                        glasses=col("glasses", np.int8), reflections=col("reflections", np.int8))
    print(f"\nSaved {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.0f} MB)")
    plot_splits(items, split)
    plot_crops(items, X)
    print("Saved assets/splits.png, assets/crops.png")


if __name__ == "__main__":
    main()
