"""Step 10: compare eye-state models on exactly the same test eyes and simulated webcam conditions.

Usage:
  python scripts/compare_eye_models.py eyeTrack0.1 eyeTrack0.5

Outputs:
  models/eye_models_comparison.txt     the printed table
  assets/eye_models_comparison.png     clean tests and simulated webcam, one bar per model
  assets/eye_models_webcam.png         accuracy per webcam condition, MRL and CEW test
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import get_device  # noqa: E402
from src.style import COLORS, finish, setup  # noqa: E402
from src.webcam import CONDITIONS  # noqa: E402

_spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def main():
    names = sys.argv[1:] or ["eyeTrack0.1", "eyeTrack0.5"]
    setup()
    device = get_device()
    data = np.load(ev.DATA)
    tests = {"MRL test": (data["split"] == ev.TEST) & (data["source"] == 0),
             "CEW test": (data["split"] == ev.TEST) & (data["source"] == 1)}
    results = {}
    for name in names:
        model, config = ev.load_model(name, device)
        r = {"config": config}
        for test, mask in tests.items():
            images, y = data["X"][mask], data["y"][mask].astype(np.int64)
            for cond in CONDITIONS:
                m = ev.metrics(y, ev.predict(model, config, images, device, cond).argmax(1))
                r[(test, cond)] = m
        results[name] = r

    lines = [f"{'':<26}" + "".join(f"{n:>14}" for n in names)]
    row = lambda label, f: lines.append(f"{label:<26}" + "".join(f"{f(results[n]):>14}" for n in names))
    row("trained on", lambda r: r["config"]["sources"] + (" + aug" if r["config"]["augment"] != "none" else ""))
    row("training time", lambda r: f"{r['config']['train_seconds'] / 60:.1f} min")
    for test in tests:
        row(f"{test} accuracy", lambda r, t=test: f"{r[(t, 'clean')]['acc']:.1%}")
        row(f"{test} macro F1", lambda r, t=test: f"{r[(t, 'clean')]['f1']:.1%}")
        row(f"{test} closed caught", lambda r, t=test: f"{r[(t, 'clean')]['recall'][0]:.1%}")
    lines.append("")
    lines.append("simulated webcam:")
    for test in tests:
        for cond in CONDITIONS:
            row(f"  {test[:3]} {cond}", lambda r, t=test, c=cond: f"{r[(t, c)]['acc']:.1%}")
    table = "\n".join(lines)
    print(table)
    (ROOT / "models" / "eye_models_comparison.txt").write_text(table + "\n")

    groups = [("MRL test", ("MRL test", "clean")), ("CEW test", ("CEW test", "clean")),
              ("MRL webcam\n(all effects)", ("MRL test", "all combined")),
              ("CEW webcam\n(all effects)", ("CEW test", "all combined"))]
    bars(results, names, groups, "Eye-state models on the same test people", ROOT / "assets" / "eye_models_comparison.png")
    groups = [(f"{t[:3]}\n{c}", (t, c)) for t in tests for c in CONDITIONS if c != "clean"]
    bars(results, names, groups, "Simulated webcam, one effect at a time and all combined", ROOT / "assets" / "eye_models_webcam.png",
         width=15)
    print("\nSaved models/eye_models_comparison.txt, assets/eye_models_comparison.png, assets/eye_models_webcam.png")


def bars(results, names, groups, title, path, width=10):
    x = np.arange(len(groups))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(width, 4.4))
    for i, name in enumerate(names):
        vals = [results[name][key]["acc"] for _, key in groups]
        b = ax.bar(x - 0.4 + w * (i + 0.5), vals, w - 0.02, color=COLORS["series"][i], label=name)
        for bar in b:
            ax.annotate(f"{bar.get_height():.0%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=COLORS["muted"])
    ax.axhline(0.5, color=COLORS["muted"], ls="--", lw=1)
    ax.text(len(groups) - 0.5, 0.51, "guessing", ha="right", fontsize=8, color=COLORS["muted"])
    ax.set_xticks(x, [g for g, _ in groups], fontsize=9)
    ax.set_ylim(0.3, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, ncols=len(names), loc="lower left")
    ax.set_title(title, loc="left", fontsize=13)
    finish(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
