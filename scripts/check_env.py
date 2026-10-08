"""Step 1: Check that this machine is ready for the project.

Prints library versions, whether PyTorch can use the Apple GPU (MPS) and how much
faster it is than the CPU on a small CNN, whether the Kaggle token is in place
(never its contents), and how much disk space is left for the datasets.

Usage:
  python scripts/check_env.py
"""
import importlib
import platform
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["torch", "numpy", "matplotlib", "PIL", "cv2", "mediapipe", "onnx", "onnxruntime", "kagglehub"]
KAGGLE_TOKENS = [Path.home() / ".kaggle" / "access_token", Path.home() / ".kaggle" / "kaggle.json"]


def versions():
    print(f"Python {platform.python_version()} on {platform.machine()} ({platform.system()} {platform.release()})")
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"  {name:<12} {getattr(module, '__version__', 'installed')}")
        except ImportError as err:
            print(f"  {name:<12} MISSING ({err})")


def small_cnn():
    """Roughly the size of the eye-state CNN planned for Step 6."""
    block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(1, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(128, 2))


@torch.no_grad()
def ms_per_batch(device, batch=256, size=64, repeats=20):
    model = small_cnn().to(device).eval()
    x = torch.randn(batch, 1, size, size, device=device)
    sync = torch.mps.synchronize if device.type == "mps" else (lambda: None)
    for _ in range(3):  # warm-up: first calls compile kernels
        model(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x)
    sync()
    return (time.perf_counter() - t0) / repeats * 1000


def gpu():
    mps = torch.backends.mps.is_available()
    print(f"\nApple GPU (MPS) available: {mps}")
    cpu_ms = ms_per_batch(torch.device("cpu"))
    print(f"  small CNN, 256 eye crops of 64x64:  CPU {cpu_ms:.1f} ms per batch")
    if mps:
        mps_ms = ms_per_batch(torch.device("mps"))
        print(f"  {'':<36}  MPS {mps_ms:.1f} ms per batch ({cpu_ms / mps_ms:.1f}x faster)")


def kaggle():
    found = [p for p in KAGGLE_TOKENS if p.exists()]
    print(f"\nKaggle token: {'found at ' + str(found[0]).replace(str(Path.home()), '~') if found else 'MISSING'}")
    if found:
        mode = found[0].stat().st_mode & 0o777
        print(f"  permissions {oct(mode)} ({'private, good' if mode & 0o077 == 0 else 'readable by others: run chmod 600'})")


def disk():
    usage = shutil.disk_usage(ROOT)
    print(f"\nDisk space on the volume holding {ROOT.name}/: {usage.free / 1e9:.0f} GB free of {usage.total / 1e9:.0f} GB")
    print(f"  ffmpeg: {shutil.which('ffmpeg') or 'not found'}")


def main():
    versions()
    gpu()
    kaggle()
    disk()
    print(f"\nRunning inside the project's .venv: {Path(sys.prefix) == ROOT / '.venv'}")


if __name__ == "__main__":
    main()
