"""Step 11: export the eye models for the browser demo (web/): PyTorch -> ONNX, checked against PyTorch.

For every model:
  - export to web/models/<version>.onnx (batch size free, so both eyes run in one call)
  - run 512 test eyes through PyTorch and through ONNX Runtime: the answers must agree
  - read its numbers from models/<version>/config.json and test_report.txt for the demo's hover cards
Also writes web/models/onnx_check.json: 16 test eyes with PyTorch's probabilities, so the browser can
check its own ONNX Runtime Web results (tools/onnx_check.html).

Usage:
  python scripts/export_web.py                       # eyeTrack0.1 and eyeTrack0.5

Outputs: web/models/<version>.onnx, web/models/manifest.json, web/models/onnx_check.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import CLASSES, EyeCNN  # noqa: E402
from src.temporal import CLASSES as STATES, DrowsinessGRU  # noqa: E402

OUT = ROOT / "web" / "models"
DATA = ROOT / "data" / "processed" / "eyes32.npz"
VERSIONS = ["eyeTrack0.1", "eyeTrack0.5"]
DESCRIPTIONS = {
    "eyeTrack0.1": {"tagline": "Eye CNN, infrared only",
                    "changes": ["Learns open / closed eyes from 56,704 infrared MRL eyes",
                                "No augmentation: the baseline", "Expect trouble on a normal webcam (domain gap)"]},
    "eyeTrack0.5": {"tagline": "Eye CNN for webcams",
                    "changes": ["Adds normal-camera eyes (CEW)", "Webcam-style augmentation: blur, noise, dim light, low resolution",
                                "Eyes cut with the same aligned crop as this demo"]},
}


def report_numbers(version):
    text = (ROOT / "models" / version / "test_report.txt").read_text()
    grab = lambda pattern: float(re.search(pattern, text).group(1)) / 100
    return {"mrl_test": grab(r"MRL test .*?accuracy ([\d.]+)%"), "cew_test": grab(r"CEW test .*?accuracy ([\d.]+)%"),
            "webcam_mrl": grab(r"all combined\s+([\d.]+)%"), "webcam_cew": grab(r"all combined\s+[\d.]+%\s+([\d.]+)%")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA)
    test = np.flatnonzero(data["split"] == 2)
    rng = np.random.default_rng(0)
    sample = rng.choice(test, 512, replace=False)
    entries, check = [], {"images": data["X"][sample[:16]].reshape(16, -1).tolist(),
                          "labels": [CLASSES[i] for i in data["y"][sample[:16]]], "models": {}}
    for v in VERSIONS:
        folder = ROOT / "models" / v
        config = json.loads((folder / "config.json").read_text())
        model = EyeCNN()
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
        model.eval()
        path = OUT / f"{v}.onnx"
        torch.onnx.export(model, torch.zeros(1, 1, 32, 32), str(path), dynamo=False, opset_version=17,
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})

        x = (torch.from_numpy(data["X"][sample]).float().div(255).unsqueeze(1) - config["mean"]) / config["std"]
        with torch.no_grad():
            expected = model(x).numpy()
        got = ort.InferenceSession(str(path)).run(None, {"input": x.numpy()})[0]
        diff = float(np.abs(expected - got).max())
        same = float(np.mean(expected.argmax(1) == got.argmax(1)))
        assert diff < 1e-3, f"{v}: ONNX differs from PyTorch by {diff}"
        probs = torch.softmax(torch.from_numpy(expected[:16]), 1).numpy()
        check["models"][v] = {"mean": config["mean"], "std": config["std"], "probs": np.round(probs, 6).tolist()}

        entries.append({"version": v, "file": f"models/{v}.onnx", "size": 32, "mean": config["mean"], "std": config["std"],
                        "classes": CLASSES, "params": config["params"], "train_images": config["train_images"],
                        "train_minutes": round(config["train_seconds"] / 60, 1), **DESCRIPTIONS[v], **report_numbers(v)})
        print(f"{v:<12} -> {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.2f} MB); 512 test eyes: largest logit "
              f"difference vs PyTorch {diff:.1e}, same answer for {same:.1%}")

    (OUT / "onnx_check.json").write_text(json.dumps(check))
    entries.append(export_temporal(data))
    (OUT / "manifest.json").write_text(json.dumps({"default": VERSIONS[-1], "models": entries}, indent=2) + "\n")
    print(f"Saved {(OUT / 'manifest.json').relative_to(ROOT)} and {(OUT / 'onnx_check.json').relative_to(ROOT)}")



def export_temporal(_):
    """eyeTrack1: 60 s of 13 features + face flag -> alert / low vigilant / drowsy."""
    v = "eyeTrack1"
    folder = ROOT / "models" / v
    config = json.loads((folder / "config.json").read_text())
    model = DrowsinessGRU(hidden=config["hidden"], dropout=config["dropout"], proj=config["proj"])
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
    model.eval()
    path = OUT / f"{v}.onnx"
    window, n = config["window"], len(config["features"]) + 1
    torch.onnx.export(model, torch.zeros(1, window, n), str(path), dynamo=False, opset_version=17,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    x = torch.randn(64, window, n)
    x[..., -1] = (torch.rand(64, window) > 0.1).float()
    with torch.no_grad():
        expected = model(x).numpy()
    got = ort.InferenceSession(str(path)).run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(expected - got).max())
    assert diff < 1e-3, f"{v}: ONNX differs from PyTorch by {diff}"
    # real UTA-RLDD windows (raw features, one per class + one more) for the browser check (tools/onnx_check.html)
    d = np.load(ROOT / "data" / "processed" / "rldd5.npz")
    pick = [int(np.flatnonzero(d["win_label"] == c)[len(np.flatnonzero(d["win_label"] == c)) // 2]) for c in (0, 1, 2)]
    pick.append(int(np.flatnonzero(d["win_valid"] < 0.95)[0]))  # one with missing faces
    raw, mask = [], []
    for w in pick:
        start = d["offsets"][d["win_video"][w]] + d["win_start"][w]
        raw.append(d["x"][start:start + window])
        mask.append(d["face"][start:start + window])
    raw, mask = np.stack(raw), np.stack(mask)
    mean, std = np.array(config["mean"], np.float32), np.array(config["std"], np.float32)
    z = ((raw - mean) / std) * mask[..., None]
    z = np.concatenate([z, mask[..., None]], -1).astype(np.float32)
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(z)), 1).numpy()
    check = json.loads((OUT / "onnx_check.json").read_text()) if (OUT / "onnx_check.json").exists() else {}
    check["temporal"] = {"raw": np.round(raw, 6).tolist(), "face": mask.astype(int).tolist(), "probs": probs.tolist(),
                         "labels": [STATES[int(d["win_label"][w])] for w in pick]}
    (OUT / "onnx_check.json").write_text(json.dumps(check))
    cv = config["cv"]["eyeTrack1"]
    print(f"{v:<12} -> {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB); 64 random windows: largest logit "
          f"difference vs PyTorch {diff:.1e}, same answer for {np.mean(expected.argmax(1) == got.argmax(1)):.1%}")
    return {"version": v, "kind": "temporal", "file": f"models/{v}.onnx", "features": config["features"],
            "mean": config["mean"], "std": config["std"], "fps": config["fps"], "window": window, "classes": STATES,
            "params": config["params"], "train_windows": config["windows"], "cv_acc": cv["acc"], "cv_f1": cv["f1"],
            "cv_binary": cv["binary_acc"], "train_minutes": config["final_minutes"],
            "tagline": "Drowsiness over 60 s (UTA-RLDD)",
            "changes": ["A GRU reads one minute of 13 face features, 5 times per second",
                        "Trained on 60 real drivers; tested on people it never saw (5-fold)",
                        "Level with the rules on new drivers: an honest result, see the README"]}


if __name__ == "__main__":
    main()
