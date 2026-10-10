"""Step 10: show what the webcam-style augmentation (src/augment.py) does to a few training eyes.

Output: assets/augmentation.png (original eye on the left, 9 random versions next to it)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.augment import webcam  # noqa: E402
from src.style import setup  # noqa: E402


def main(versions=9):
    setup()
    torch.manual_seed(0)
    data = np.load(ROOT / "data" / "processed" / "eyes32.npz")
    rng = np.random.default_rng(2)
    train = data["split"] == 0
    picks = [("MRL open", (data["source"] == 0) & (data["y"] == 1)), ("MRL closed", (data["source"] == 0) & (data["y"] == 0)),
             ("CEW open", (data["source"] == 1) & (data["y"] == 1)), ("CEW closed", (data["source"] == 1) & (data["y"] == 0))]
    fig, axes = plt.subplots(len(picks), versions + 1, figsize=((versions + 1) * 1.05, len(picks) * 1.2))
    for r, (label, mask) in enumerate(picks):
        eye = torch.from_numpy(data["X"][rng.choice(np.flatnonzero(train & mask))]).float().div(255)[None, None]
        variants = [eye[0, 0]] + list(webcam(eye.repeat(versions, 1, 1, 1))[:, 0])
        for c, img in enumerate(variants):
            axes[r, c].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title("original" if c == 0 else f"#{c}", fontsize=9)
        axes[r, 0].text(-0.15, 0.5, label, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=10)
    fig.suptitle("Webcam-style augmentation: mirror, rotation, zoom, shift, lighting, blur, low resolution, noise",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out = ROOT / "assets" / "augmentation.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
