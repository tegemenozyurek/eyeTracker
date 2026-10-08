"""Fake training run and fake long job, to test the training monitor without data or a GPU.

The fake training behaves like a real one that overfits: training loss keeps
falling, validation loss reaches its minimum around epoch 7 and then rises, so
the monitor's overfitting warning should appear. Sample predictions are drawn
eyes (open: ellipse with an iris, closed: a curved line), not real data.

Usage:
  python tools/monitor/dummy_run.py                         # training run "_dummy"
  python tools/monitor/dummy_run.py --name _dummy2 --delay 0.2   # slower, to try the Stop button
  python tools/monitor/dummy_run.py --job                   # progress-bar job "_dummy_job"
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.runlog import RunLog  # noqa: E402

CLASSES = ["closed", "open"]


def draw_eye(is_open, rng, size=48):
    """A synthetic 48x48 grayscale eye: skin background, open = ellipse + iris, closed = eyelid line."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2 + rng.uniform(-3, 3), size / 2 + rng.uniform(-3, 3)
    img = np.full((size, size), rng.uniform(150, 200), np.float32)
    if is_open:
        eye = ((xs - cx) / 17) ** 2 + ((ys - cy) / 9) ** 2 < 1
        img[eye] = 235
        img[(xs - cx) ** 2 + (ys - cy) ** 2 < 36] = 60
        img[(xs - cx) ** 2 + (ys - cy) ** 2 < 9] = 15
    else:
        lid = np.abs(ys - (cy + 0.012 * (xs - cx) ** 2)) < 1.4
        img[lid & (np.abs(xs - cx) < 17)] = 55
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def curves(epoch, rng):
    """Loss / accuracy that overfit: validation loss is lowest around epoch 7, then rises."""
    train_loss = 0.65 * math.exp(-0.28 * epoch) + 0.03 + rng.normal(0, 0.005)
    val_loss = 0.18 + 0.5 * math.exp(-0.45 * epoch) + 0.012 * max(0, epoch - 7) ** 1.3 + rng.normal(0, 0.006)
    return train_loss, val_loss, 1 - 0.55 * train_loss, 1 - 0.6 * val_loss


def fake_training(args, rng):
    log = RunLog(args.name, kind="train", classes=CLASSES, epochs=args.epochs, steps_per_epoch=args.steps,
                 config=vars(args), step_every=2)
    step_seconds, step_calls, epoch_seconds, epoch_calls = 0.0, 0, 0.0, 0
    best, best_epoch, status = -1.0, 0, "finished"
    t_run = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, val_loss, train_acc, val_acc = curves(epoch, rng)
        lr = 1e-3 * 0.5 ** (epoch // 6)
        for step in range(args.steps):
            time.sleep(args.delay)  # stands in for a forward/backward pass
            progress = (step + 1) / args.steps
            loss = train_loss + 0.15 * math.exp(-0.28 * epoch) * (1 - progress) + rng.normal(0, 0.03)
            t = time.perf_counter()
            log.step(epoch, step, global_step=(epoch - 1) * args.steps + step, loss=loss,
                     acc=min(1, 1 - 0.55 * loss + rng.normal(0, 0.02)), lr=lr)
            stop = log.stop_requested()
            step_seconds += time.perf_counter() - t
            step_calls += 1
            if stop:
                break
        if stop:
            log.message(f"stop requested during epoch {epoch}: keeping the checkpoint from epoch {best_epoch}")
            status = "stopped"
            break

        # validation: 1,000 fake eyes, each class recognized with roughly val_acc
        y = rng.integers(0, 2, 1000)
        correct = rng.random(1000) < val_acc + np.where(y == 0, -0.03, 0.03)
        pred = np.where(correct, y, 1 - y)
        cm = np.zeros((2, 2), np.int64)
        np.add.at(cm, (y, pred), 1)
        is_best = val_acc > best
        if is_best:
            best, best_epoch = val_acc, epoch
        items = []
        for i in range(16):
            true = int(y[i])
            items.append({"image": draw_eye(true == 1, rng), "true": CLASSES[true], "pred": CLASSES[int(pred[i])],
                          "conf": float(rng.uniform(0.55, 0.99))})
        t = time.perf_counter()
        log.epoch(epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc, lr=lr,
                  epoch_time=time.time() - t0, best=is_best, confusion=cm)
        log.samples(epoch, items)
        epoch_seconds += time.perf_counter() - t
        epoch_calls += 1
        print(f"epoch {epoch:>2}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.1%}"
              f"{'  * best' if is_best else ''}", flush=True)
    log.end(status, best_epoch=best_epoch, best_val_acc=best)
    size = log.path.stat().st_size
    print(f"\nRun {status} after {time.time() - t_run:.1f} s; best val acc {best:.1%} at epoch {best_epoch}")
    print(f"Logger cost per training step: {step_seconds * 1e6 / step_calls:.0f} µs "
          f"(step event + stop-file check, mean of {step_calls} steps)")
    print(f"Logger cost per epoch: {epoch_seconds * 1e3 / epoch_calls:.1f} ms "
          f"(epoch event with confusion matrix + 16 sample images encoded, mean of {epoch_calls} epochs)")
    print(f"metrics.jsonl: {size / 1024:.1f} KB  ->  {log.path.relative_to(ROOT)}")


def fake_job(args, rng):
    total = 120
    with RunLog(args.name, kind="job", total=total, unit="files") as log:
        for i in range(1, total + 1):
            time.sleep(args.delay * rng.uniform(0.5, 1.5))
            log.progress(i, total, message=f"participant_{(i - 1) // 4 + 1:02d}/video_{(i - 1) % 4 + 1}.mp4")
            if i % 40 == 0:
                log.message(f"{i} of {total} files downloaded")
    print(f"Job finished: {total} fake files -> {log.path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int, default=40, help="steps per epoch")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds per fake step")
    parser.add_argument("--job", action="store_true", help="fake a long download instead of training")
    args = parser.parse_args()
    args.name = args.name or ("_dummy_job" if args.job else "_dummy")
    rng = np.random.default_rng(0)
    (fake_job if args.job else fake_training)(args, rng)


if __name__ == "__main__":
    main()
