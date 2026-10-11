"""Step 15: the README's header image: eyes of TEST people (never seen in training) with eyeTrack0.5's
probability that the eye is closed. Infrared eyes (MRL) on top, normal-camera eyes (CEW) below; half open, half
closed, picked at random with a fixed seed (not hand-picked). No UTA-RLDD image exists in this project.

Output: assets/showcase.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import EyeCNN  # noqa: E402
from src.style import COLORS, setup  # noqa: E402

MODEL = "eyeTrack0.5"
PER_ROW = 8


def main(seed=11):
    setup()
    data = np.load(ROOT / "data" / "processed" / "eyes32.npz")
    cfg = json.loads((ROOT / "models" / MODEL / "config.json").read_text())
    model = EyeCNN()
    model.load_state_dict(torch.load(ROOT / "models" / MODEL / "model.pt", map_location="cpu"))
    model.eval()
    rng = np.random.default_rng(seed)
    rows = []
    for source, name in [(0, "MRL · infrared"), (1, "CEW · normal camera")]:
        picks = []
        for y in (0, 1):
            pool = np.flatnonzero((data["split"] == 2) & (data["source"] == source) & (data["y"] == y))
            picks += list(rng.choice(pool, PER_ROW // 2, replace=False))
        rng.shuffle(picks)
        x = (torch.from_numpy(data["X"][picks]).float().div(255).unsqueeze(1) - cfg["mean"]) / cfg["std"]
        with torch.no_grad():
            p_closed = torch.softmax(model(x), 1)[:, 0].numpy()
        rows.append((name, picks, p_closed))

    fig, axes = plt.subplots(2, PER_ROW, figsize=(PER_ROW * 1.45 + 1.4, 4.9))
    wrong = 0
    for r, (name, picks, probs) in enumerate(rows):
        for c, (i, p) in enumerate(zip(picks, probs)):
            truth, pred = int(data["y"][i]), int(p < 0.5)
            wrong += truth != pred
            ax = axes[r, c]
            ax.imshow(data["X"][i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(COLORS["bad"] if truth != pred else COLORS["grid"])
                spine.set_linewidth(2.5 if truth != pred else 1)
            ax.set_title("closed" if truth == 0 else "open", fontsize=9, color=COLORS["muted"], pad=4)
            color = COLORS["series"][1] if p >= 0.5 else COLORS["series"][2]
            ax.set_xlabel(f"P(closed) {p:.0%}", fontsize=9, color=color, labelpad=9)
            strip = ax.inset_axes([0, -0.1, 1, 0.06])  # a thin probability bar right under the eye
            strip.barh([0], [1], color=COLORS["grid"], height=1)
            strip.barh([0], [p], color=color, height=1)
            strip.set_xlim(0, 1)
            strip.axis("off")
        axes[r, 0].set_ylabel(name, fontsize=10, labelpad=8)
    fig.suptitle(f"{MODEL} on eyes of people it never saw: probability that the eye is closed "
                 f"({wrong} of {2 * PER_ROW} wrong, outlined in red)", x=0.01, ha="left", fontsize=12)
    out = ROOT / "assets" / "showcase.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out.relative_to(ROOT)}: {2 * PER_ROW} random test eyes, {wrong} wrong")


if __name__ == "__main__":
    main()
