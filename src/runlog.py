"""Live log of a training run or long job, read by the training monitor (tools/monitor/).

Every event is one JSON object appended as a line to runs/<name>/metrics.jsonl.
Appending a line is cheap (no plotting, no network), so logging never slows
training down; the monitor is a separate process that only reads the file.

    with RunLog("eyeTrack0.1", kind="train", config=vars(args), classes=["closed", "open"],
                epochs=40, steps_per_epoch=steps) as log:
        for epoch in ...:
            for step in ...:
                log.step(epoch, step, loss=..., acc=..., lr=...)
                if log.stop_requested():   # the monitor's Stop button
                    ...
            log.epoch(epoch, train_loss=..., train_acc=..., val_loss=..., val_acc=..., confusion=cm, best=True)
            log.samples(epoch, items)      # eye crops with true / predicted label
        log.end("finished", best_epoch=..., best_val_acc=...)

Long jobs (downloads, feature extraction) use kind="job" and log.progress(done, total).
"""
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _to_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):  # torch tensors
        return value.tolist()
    return str(value)


def _png_base64(image):
    """uint8 (H, W) or (H, W, 3) array, or floats in 0-1 -> base64 PNG."""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RunLog:
    def __init__(self, name, kind="train", step_every=10, **info):
        """name: run folder under runs/. kind: "train" or "job".
        step_every: write only every n-th step event (epoch events are always written).
        info: anything the monitor should know (config, classes, epochs, steps_per_epoch, total, unit ...)."""
        self.name, self.kind, self.step_every = name, kind, step_every
        self.dir = RUNS / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "metrics.jsonl"
        self.stop_file = self.dir / "STOP"
        if self.path.exists():  # keep the previous attempt next to the new one
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.path.stat().st_mtime))
            self.path.rename(self.dir / f"metrics.{stamp}.jsonl")
        for stale in (self.stop_file, self.dir / "samples.json"):
            stale.unlink(missing_ok=True)
        self.file = open(self.path, "a", buffering=1)  # line-buffered: each event is on disk at once
        self.ended = False
        self._last_progress = 0.0
        self.write("start", kind=kind, pid=os.getpid(), **info)

    def write(self, type_, **fields):
        self.file.write(json.dumps({"type": type_, "time": round(time.time(), 3), **fields}, default=_to_json) + "\n")

    def step(self, epoch, step, global_step=None, **metrics):
        """One optimizer step. epoch counts from 1, step from 0 within the epoch."""
        n = step if global_step is None else global_step
        if n % self.step_every == 0:
            self.write("step", epoch=epoch, step=step, global_step=global_step, **metrics)

    def epoch(self, epoch, confusion=None, **metrics):
        """End of an epoch: train/val metrics, lr, epoch_time, best (bool), confusion matrix (rows = true class)."""
        if confusion is not None:
            cm = np.asarray(confusion, dtype=np.int64)
            tp = np.diag(cm)
            metrics["confusion"] = cm
            metrics["precision"] = tp / np.maximum(cm.sum(0), 1)
            metrics["recall"] = tp / np.maximum(cm.sum(1), 1)
        self.write("epoch", epoch=epoch, **metrics)

    def samples(self, epoch, items):
        """Sample predictions shown in the monitor, newest epoch only (overwrites samples.json).
        items: dicts with "true", "pred", "conf" and either "image" (an eye crop) or
        "series" ({feature name: list of values}, for RLDD windows: never faces)."""
        out = []
        for item in items:
            entry = {k: v for k, v in item.items() if k != "image"}
            if "image" in item:
                entry["image"] = _png_base64(item["image"])
            out.append(entry)
        tmp = self.dir / "samples.json.tmp"
        tmp.write_text(json.dumps({"epoch": epoch, "items": out}, default=_to_json))
        os.replace(tmp, self.dir / "samples.json")  # the monitor never sees a half-written file
        self.write("samples", epoch=epoch, n=len(out))

    def progress(self, done, total, message="", force=False):
        """Progress of a long job; written at most twice per second unless forced or finished."""
        now = time.time()
        if force or done >= total or now - self._last_progress >= 0.5:
            self._last_progress = now
            self.write("progress", done=done, total=total, message=message)

    def message(self, text):
        self.write("message", text=text)

    def stop_requested(self):
        """True once the monitor's Stop button was pressed (it creates runs/<name>/STOP)."""
        return self.stop_file.exists()

    def end(self, status="finished", **summary):
        """status: finished, stopped or failed."""
        if not self.ended:
            self.ended = True
            self.write("end", status=status, **summary)
            self.stop_file.unlink(missing_ok=True)
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is KeyboardInterrupt:
            self.end("stopped", error="interrupted (Ctrl+C)")
        elif exc_type is not None:
            self.end("failed", error=f"{exc_type.__name__}: {exc}")
        else:
            self.end("finished")
        return False
