"""Evaluate an eye-state model on the held-out TEST people (the first time it sees them).

  MRL test        7 people the model never trained on (infrared, like its training data)
  CEW test        normal-camera eyes: the domain gap from infrared to webcam-like images
  simulated webcam both test sets with dim light, motion blur, low resolution, sensor noise and
                  detector jitter (src/webcam.py), each alone and all combined
  MRL conditions  accuracy per camera, lighting, glasses and reflections (Step 4 found that eye
                  state is tied to the camera: does the model cheat with it?)

Usage:
  python scripts/evaluate.py --model eyeTrack0.1

Outputs:
  models/<model>/test_report.txt         every number printed here
  assets/<model>/confusion_matrices.png  MRL test and CEW test
  assets/<model>/webcam_conditions.png   accuracy per simulated webcam condition
  assets/<model>/predictions.png         test eyes with true vs predicted label (mistakes in red)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import CLASSES, EyeCNN, get_device  # noqa: E402
from src.style import COLORS, finish, setup  # noqa: E402
from src.webcam import CONDITIONS, degrade  # noqa: E402

DATA = ROOT / "data" / "processed" / "eyes32.npz"
TEST = 2
BLUES = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#2a78d6", "#0d2f5c"])
MRL_CONDITIONS = {
    "sensor": {1: "RealSense camera", 2: "IDS camera", 3: "Aptina camera (never trained on)"},
    "lighting": {0: "bad lighting", 1: "good lighting"},
    "glasses": {0: "no glasses", 1: "glasses"},
    "reflections": {0: "no reflections", 1: "small reflections", 2: "big reflections"},
}


def load_model(name, device):
    folder = ROOT / "models" / name
    config = json.loads((folder / "config.json").read_text())
    model = EyeCNN()
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
    return model.to(device).eval(), config


@torch.no_grad()
def predict(model, config, images, device, condition="clean", batch=2048):
    x = torch.from_numpy(images).float().div(255).unsqueeze(1)
    x = degrade(x, condition)  # identical degradations for every model (fixed seed)
    probs = [torch.softmax(model(((x[i:i + batch] - config["mean"]) / config["std"]).to(device)), 1).cpu()
             for i in range(0, len(x), batch)]
    return torch.cat(probs).numpy()


def metrics(y, pred):
    cm = np.zeros((2, 2), np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return {"acc": tp.sum() / cm.sum(), "f1": f1.mean(), "recall": recall, "precision": precision, "cm": cm}


def plot_confusions(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, m) in zip(axes, results.items()):
        norm = m["cm"] / m["cm"].sum(1, keepdims=True)
        ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j]:.1%}\n{m['cm'][i, j]:,}", ha="center", va="center", fontsize=10,
                        color="white" if norm[i, j] > 0.5 else COLORS["text"])
        ax.set_xticks([0, 1], CLASSES)
        ax.set_yticks([0, 1], CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{name}: {m['acc']:.1%} accuracy", loc="left", fontsize=11)
    fig.suptitle("Confusion matrices (test people, row-normalized)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_webcam(webcam, name, path):
    conds = list(CONDITIONS)
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for i, (test, color) in enumerate([("MRL test", COLORS["series"][0]), ("CEW test", COLORS["series"][1])]):
        bars = ax.bar(x + (i - 0.5) * 0.38, [webcam[test][c] for c in conds], 0.36, color=color, label=test)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.0%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=COLORS["muted"])
    ax.axhline(0.5, color=COLORS["muted"], ls="--", lw=1)
    ax.text(len(conds) - 0.5, 0.51, "chance", ha="right", fontsize=8, color=COLORS["muted"])
    ax.set_xticks(x, conds)
    ax.set_ylim(0.3, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, ncols=2, loc="lower left")
    ax.set_title(f"{name}: accuracy under simulated webcam conditions", loc="left", fontsize=13)
    finish(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(images, y, probs, title, path, n=24, seed=7):
    rng = np.random.default_rng(seed)
    pred = probs.argmax(1)
    wrong = np.flatnonzero(pred != y)
    k = min(8, len(wrong))  # show some mistakes, the rest at random
    idx = np.concatenate([rng.choice(wrong, k, replace=False), rng.choice(len(y), n - k, replace=False)])
    cols = 12
    fig, axes = plt.subplots(n // cols, cols, figsize=(cols * 1.15, n // cols * 1.85))
    for ax, i in zip(axes.ravel(), idx):
        ok = pred[i] == y[i]
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{'✓' if ok else '✗'} {CLASSES[pred[i]]} {probs[i].max():.0%}\ntrue: {CLASSES[y[i]]}",
                     fontsize=7.5, color=COLORS["text"] if ok else COLORS["bad"])
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    setup()
    device = get_device()
    model, config = load_model(args.model, device)
    data = np.load(DATA)
    plot_dir = ROOT / "assets" / args.model
    plot_dir.mkdir(parents=True, exist_ok=True)
    tests = {"MRL test": (data["split"] == TEST) & (data["source"] == 0),
             "CEW test": (data["split"] == TEST) & (data["source"] == 1)}

    lines = [f"Model: {args.model} | trained on {config['sources']} | augment={config['augment']} | "
             f"{config['params']:,} params | best epoch {config['best_epoch']} | "
             f"training time {config['train_seconds'] / 60:.1f} min", ""]
    clean, webcam, probs_by_test = {}, {}, {}
    for test, mask in tests.items():
        images, y = data["X"][mask], data["y"][mask].astype(np.int64)
        webcam[test] = {}
        for cond in CONDITIONS:
            probs = predict(model, config, images, device, cond)
            m = metrics(y, probs.argmax(1))
            webcam[test][cond] = m["acc"]
            if cond == "clean":
                clean[test], probs_by_test[test] = m, probs
        m = clean[test]
        lines.append(f"{test} ({len(y):,} eyes): accuracy {m['acc']:.1%}, macro F1 {m['f1']:.1%}, "
                     f"recall closed {m['recall'][0]:.1%}, recall open {m['recall'][1]:.1%}")
    lines += ["", f"{'simulated webcam':<18}" + "".join(f"{t:>12}" for t in tests)]
    lines += [f"{c:<18}" + "".join(f"{webcam[t][c]:>12.1%}" for t in tests) for c in CONDITIONS]

    mrl = tests["MRL test"]
    pred_mrl = probs_by_test["MRL test"].argmax(1)
    y_mrl = data["y"][mrl]
    lines += ["", "MRL test by recording condition (accuracy alone flatters conditions with few closed eyes,\n"
              "so the share of closed eyes the model catches is shown too):"]
    for key, names in MRL_CONDITIONS.items():
        values = data[key][mrl]
        for code, label in names.items():
            sel = values == code
            if sel.any():
                closed = sel & (y_mrl == 0)
                recall = f"{np.mean(pred_mrl[closed] == 0):6.1%}" if closed.any() else "     –"
                lines.append(f"  {label:<34} {sel.sum():>6,} eyes  accuracy {np.mean(pred_mrl[sel] == y_mrl[sel]):6.1%}"
                             f"  closed eyes caught {recall} of {closed.sum():>5,}")
    report = "\n".join(lines)
    print(report)
    (ROOT / "models" / args.model / "test_report.txt").write_text(report + "\n")

    plot_confusions(clean, plot_dir / "confusion_matrices.png")
    plot_webcam(webcam, args.model, plot_dir / "webcam_conditions.png")
    cew = tests["CEW test"]
    plot_predictions(data["X"][cew], data["y"][cew].astype(np.int64), probs_by_test["CEW test"],
                     f"{args.model} on CEW test eyes (normal camera): ✓ correct, ✗ wrong", plot_dir / "predictions.png")
    print(f"\nSaved models/{args.model}/test_report.txt and assets/{args.model}/"
          "{confusion_matrices,webcam_conditions,predictions}.png")


if __name__ == "__main__":
    main()
