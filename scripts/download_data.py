"""Step 3: Download the three datasets from Kaggle into data/ and check what arrived.

  mrl    MRL Eye Dataset: infrared eye crops, open / closed. File names keep the original
         MRL fields (subject id, eye state, glasses, ...), which the subject-wise split needs.
  cew    Closed Eyes In The Wild: the official 24x24 grayscale eye patches, open / closed, cut from
         normal-camera photos (not infrared).
  rldd   UTA-RLDD Face Features: MediaPipe features per frame (10 fps) for all 60 participants,
         official 5-fold split, labels alert / low vigilant / drowsy. No video and no images.

Each dataset is streamed straight from the Kaggle API as a zip (progress visible in the
training monitor as the job "_download"), unpacked into data/<name>/ and the zip is deleted,
so nothing stays in hidden caches. Requires the Kaggle token in ~/.kaggle/access_token.

Usage:
  python scripts/download_data.py              # all three
  python scripts/download_data.py mrl cew      # some of them
  python scripts/download_data.py lfw          # LFW photos, only for the eye-crop calibration (Step 9)

Outputs: data/mrl/, data/cew/, data/rldd/  and  assets/data_overview.png
"""
import csv
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.runlog import RunLog  # noqa: E402
from src.style import COLORS, finish, setup  # noqa: E402

DATA = ROOT / "data"
ASSETS = ROOT / "assets"
TOKEN = Path.home() / ".kaggle" / "access_token"
DATASETS = {
    "mrl": "imadeddinedjerarda/mrl-eye-dataset",
    "cew": "faisal7/cew-dataset",
    "rldd": "abdulrahmankhengari/uta-rldd-face-features",
    "lfw": "jessicali9530/lfw-dataset",  # Step 9 only: the photos CEW's open eyes were cut from
}
DEFAULT = ["mrl", "cew", "rldd"]
# MRL file name: s0001_00001_0_0_0_0_0_01.png = subject, image, gender, glasses, eye state, reflections, lighting, sensor
MRL_NAME = re.compile(r"s(\d{4})_(\d{5})_(\d)_(\d)_(\d)_(\d)_(\d)_(\d{2})\.png$")
RLDD_CLASSES = {0: "alert", 5: "low vigilant", 10: "drowsy"}


def download(name, ref, log):
    """Stream the dataset zip into data/<name>.zip with byte progress, unpack it, delete the zip."""
    target = DATA / name
    if target.exists() and any(target.iterdir()):
        log.message(f"{name}: already in {target.relative_to(ROOT)}, skipped")
        return
    zip_path = DATA / f"{name}.zip"
    request = urllib.request.Request(f"https://www.kaggle.com/api/v1/datasets/download/{ref}",
                                     headers={"Authorization": f"Bearer {TOKEN.read_text().strip()}"})
    t0 = time.time()
    with urllib.request.urlopen(request) as response, open(zip_path, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            log.progress(done // (1 << 20), max(total // (1 << 20), done // (1 << 20)),
                         message=f"{name}: {done / 1e6:,.0f} of {total / 1e6:,.0f} MB")
    log.message(f"{name}: downloaded {done / 1e6:,.0f} MB in {time.time() - t0:.0f} s")
    print(f"  {name}: downloaded {done / 1e6:,.0f} MB in {time.time() - t0:.0f} s", flush=True)

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        for i, m in enumerate(members, 1):
            path = (target / m.filename).resolve()
            if not path.is_relative_to(target.resolve()):  # never write outside data/<name>/
                raise ValueError(f"unsafe path in zip: {m.filename}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            log.progress(i, len(members), message=f"{name}: unpacking {i:,} of {len(members):,} files")
    zip_path.unlink()


def check_mrl():
    files = sorted((DATA / "mrl").rglob("*.png"))
    rows = [(p, MRL_NAME.search(p.name)) for p in files]
    parsed = [(p, m) for p, m in rows if m]
    state = Counter("open" if m.group(5) == "1" else "closed" for _, m in parsed)
    folder_agrees = sum(("open" in p.parent.name.lower()) == (m.group(5) == "1") for p, m in parsed)
    subjects = Counter(m.group(1) for _, m in parsed)
    glasses = sum(m.group(4) == "1" for _, m in parsed)
    print(f"\nMRL Eye: {len(files):,} images, {len(parsed):,} with a parseable MRL file name")
    print(f"  closed {state['closed']:,} / open {state['open']:,}")
    print(f"  eye state in the file name matches the folder for {folder_agrees:,} of {len(parsed):,}")
    print(f"  {len(subjects)} subjects, {min(subjects.values()):,} to {max(subjects.values()):,} images each")
    print(f"  wearing glasses: {glasses:,} ({glasses / len(parsed):.1%})")
    return {"closed": state["closed"], "open": state["open"]}


def check_cew():
    files = [p for p in (DATA / "cew").rglob("*.jpg")]
    closed = sum("closed" in p.parent.name.lower() for p in files)
    opened = sum("open" in p.parent.name.lower() for p in files)
    print(f"\nCEW: {len(files):,} eye patches: closed {closed:,} / open {opened:,}")
    return {"closed": closed, "open": opened}


def check_rldd():
    root = DATA / "rldd"
    with open(next(root.rglob("metadata.csv"))) as f:
        videos = list(csv.DictReader(f))
    parquets = sorted(root.rglob("features/*.parquet"))
    frames = sum(pq.ParquetFile(p).metadata.num_rows for p in parquets)
    columns = pq.ParquetFile(parquets[0]).schema_arrow.names
    label_key = next(k for k in videos[0] if "label" in k)
    ok_key = next(k for k in videos[0] if "verif" in k)
    by_class = Counter(RLDD_CLASSES.get(int(float(v[label_key])), v[label_key]) for v in videos)
    passed = [v for v in videos if v[ok_key].strip().lower() in ("true", "1")]
    folds = Counter(v["fold"] for v in videos)
    subjects = {v["subject_id"] for v in videos}
    print(f"\nUTA-RLDD features: {len(videos)} videos from {len(subjects)} subjects, {len(parquets)} feature files")
    print(f"  videos per class: " + ", ".join(f"{c} {by_class[c]}" for c in RLDD_CLASSES.values()))
    print(f"  videos per fold: " + ", ".join(f"fold {k} {folds[k]}" for k in sorted(folds)))
    print(f"  passed the dataset's quality check: {len(passed)} of {len(videos)}")
    print(f"  {frames:,} frames, {len(columns)} columns per frame "
          f"({sum(c.startswith('bs_') for c in columns)} blendshapes)")
    return {c: by_class[c] for c in RLDD_CLASSES.values()}


def plot_overview(counts):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2, 2, 3]})
    titles = {"mrl": "MRL Eye (infrared eye crops)", "cew": "CEW (24x24 eye patches, normal cameras)",
              "rldd": "UTA-RLDD (videos per class)"}
    for ax, name in zip(axes, ["mrl", "cew", "rldd"]):
        c = counts.get(name)
        if not c:
            ax.set_visible(False)
            continue
        bars = ax.bar(list(c), list(c.values()), color=COLORS["series"][:len(c)], width=0.6)
        for bar in bars:
            ax.annotate(f"{int(bar.get_height()):,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=COLORS["muted"])
        ax.set_title(titles[name], loc="left", fontsize=11)
        finish(ax)
    fig.suptitle("Downloaded data: samples per class", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "data_overview.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {(ASSETS / 'data_overview.png').relative_to(ROOT)}")


def main():
    names = sys.argv[1:] or DEFAULT
    DATA.mkdir(exist_ok=True)
    with RunLog("_download", kind="job", unit="MB / files", datasets=names) as log:
        for name in names:
            print(f"{name}: {DATASETS[name]}", flush=True)
            download(name, DATASETS[name], log)
    checks = {"mrl": check_mrl, "cew": check_cew, "rldd": check_rldd}
    counts = {name: checks[name]() for name in names if name in checks}
    sizes = {n: sum(f.stat().st_size for f in (DATA / n).rglob("*") if f.is_file()) for n in names}
    print("\nDisk use: " + ", ".join(f"{n} {s / 1e6:,.0f} MB" for n, s in sizes.items())
          + f"  (total {sum(sizes.values()) / 1e9:.2f} GB)")
    if counts:
        plot_overview(counts)


if __name__ == "__main__":
    main()
