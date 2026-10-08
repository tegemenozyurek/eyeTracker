"""Step 6: Build the eye CNN, print a layer-by-layer summary, run it once on real eye crops
to check the wiring, and time it (it must be fast enough to run on both eyes of every frame).

Outputs: assets/model_summary.png (output size and parameters per block)
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import CLASSES, EyeCNN, get_device  # noqa: E402
from src.style import COLORS, finish, setup  # noqa: E402

DATA = ROOT / "data" / "processed" / "eyes32.npz"


def summary(model):
    rows = []

    def hook(module, _inputs, output):
        rows.append((module.__class__.__name__, tuple(output.shape[1:]),
                     sum(p.numel() for p in module.parameters(recurse=False))))

    handles = [m.register_forward_hook(hook) for m in model.modules() if not list(m.children())]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 32, 32))
    for h in handles:
        h.remove()
    print(f"{'#':>2}  {'layer':<18} {'output shape':<16} {'params':>8}")
    print("-" * 48)
    for i, (name, shape, params) in enumerate(rows, 1):
        print(f"{i:>2}  {name:<18} {str(shape):<16} {params:>8,}")
    print("-" * 48)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total:,}  ({total * 4 / 1e6:.2f} MB as float32)")
    return total


@torch.no_grad()
def ms_per_call(model, device, batch, repeats=50):
    model = model.to(device).eval()
    x = torch.randn(batch, 1, 32, 32, device=device)
    sync = torch.mps.synchronize if device.type == "mps" else (lambda: None)
    for _ in range(5):
        model(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x)
    sync()
    return (time.perf_counter() - t0) / repeats * 1000


def plot_blocks(model):
    """Output size and parameters of the three blocks and the head."""
    names, params, shapes = [], [], []
    x = torch.zeros(1, 1, 32, 32)
    for i, block in enumerate(model.features, 1):
        x = block(x)
        names.append(f"block {i}")
        params.append(sum(p.numel() for p in block.parameters()))
        shapes.append("x".join(map(str, x.shape[1:])))
    names.append("head")
    params.append(sum(p.numel() for p in model.classifier.parameters()))
    shapes.append("2 logits")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.bar(names, params, color=COLORS["series"][0], width=0.55)
    for bar, shape, p in zip(bars, shapes, params):
        ax.annotate(f"{p:,} params\nout: {shape}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=COLORS["muted"])
    ax.set_ylim(0, max(params) * 1.3)
    ax.set_title(f"EyeCNN: {sum(p.numel() for p in model.parameters()):,} parameters, input 1x32x32",
                 loc="left", fontsize=12)
    finish(ax)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "model_summary.png", dpi=150)
    plt.close(fig)


def main():
    setup()
    torch.manual_seed(0)
    model = EyeCNN()
    summary(model)

    data = np.load(DATA)
    mrl_val = (data["split"] == 1) & (data["source"] == 0)
    rng = np.random.default_rng(0)  # 4 closed + 4 open eyes
    val = np.concatenate([rng.choice(np.flatnonzero(mrl_val & (data["y"] == c)), 4, replace=False) for c in (0, 1)])
    x = torch.from_numpy(data["X"][val]).float().div(255).unsqueeze(1)
    device = get_device()
    with torch.no_grad():
        probs = torch.softmax(model.to(device).eval()(x.to(device)), 1).cpu().numpy()
    print(f"\nDevice: {device} | input {tuple(x.shape)} -> output {probs.shape}")
    print("UNTRAINED model on 8 MRL validation eyes:")
    for true, p in zip(data["y"][val], probs):
        print(f"  true {CLASSES[true]:<6}  predicted {CLASSES[p.argmax()]:<6} {p.max():.1%}")
    print(f"(chance = {1 / len(CLASSES):.0%}: it has learned nothing yet)")

    print("\nSpeed (forward pass only):")
    for device_name in ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]:
        for batch in (2, 256):
            ms = ms_per_call(model, torch.device(device_name), batch)
            print(f"  {device_name.upper():<4} batch {batch:>3}: {ms:6.2f} ms ({ms / batch * 1000:,.0f} µs per eye)")
    plot_blocks(model.cpu())
    print("\nSaved assets/model_summary.png")


if __name__ == "__main__":
    main()
