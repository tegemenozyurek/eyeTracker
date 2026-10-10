"""Train an eye-state model (closed / open) on the 32x32 crops from Step 5.

Each epoch the model sees every training eye once, in shuffled mini-batches. For every batch:
predict -> measure how wrong (cross-entropy loss) -> backpropagate -> nudge the weights.
After each epoch it is checked on the validation people and the best epoch is kept.
Everything is logged live to runs/<name>/metrics.jsonl for the training monitor, whose
Stop button ends the run cleanly (the best checkpoint so far is kept).

Usage:
  python scripts/train.py --name eyeTrack0.1                       # MRL only, no augmentation
  python scripts/train.py --name eyeTrack0.5 --sources mrl,cew --augment webcam --cew-share 0.3

Outputs:
  models/<name>/model.pt            weights of the epoch with the best validation accuracy
  models/<name>/config.json         settings, normalization, parameter count, training time
  models/<name>/history.csv         loss / accuracy per epoch
  models/<name>/train_log.txt       the printed log
  assets/<name>/training_curves.png loss and accuracy, train vs validation
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import CLASSES, EyeCNN, get_device  # noqa: E402
from src.runlog import RunLog  # noqa: E402
from src.style import COLORS, finish, setup  # noqa: E402

DATA = ROOT / "data" / "processed" / "eyes32.npz"
SOURCES = {"mrl": 0, "cew": 1}
TRAIN, VAL = 0, 1


def augment_batch(x, kind):
    if kind == "none":
        return x
    from src.augment import augment  # Step 10
    return augment(x, kind)


@torch.no_grad()
def evaluate(model, X, y, mean, std, batch=2048):
    model.eval()
    logits = torch.cat([model((X[i:i + batch].float().div(255).unsqueeze(1) - mean) / std)
                        for i in range(0, len(X), batch)])
    loss = F.cross_entropy(logits, y).item()
    pred = logits.argmax(1)
    cm = torch.zeros(2, 2, dtype=torch.long, device=y.device)
    cm.index_put_((y, pred), torch.ones_like(y), accumulate=True)
    return loss, (pred == y).float().mean().item(), cm.cpu().numpy(), logits.softmax(1)


def plot_history(history, best_epoch, path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (key, title) in zip(axes, [("loss", "Loss (lower is better)"), ("acc", "Accuracy (higher is better)")]):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], color=COLORS["train"], lw=2, label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], color=COLORS["val"], lw=2, label="validation")
        ax.axvline(best_epoch, color=COLORS["muted"], lw=1, ls="--")
        ax.text(best_epoch, ax.get_ylim()[1], f" best epoch {best_epoch}", va="top", fontsize=9, color=COLORS["muted"])
        if key == "acc":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("epoch")
        ax.legend(frameon=False)
        finish(ax)
    fig.suptitle("Training curves", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="model version, e.g. eyeTrack0.1 -> models/eyeTrack0.1/")
    parser.add_argument("--sources", default="mrl", help="training data: mrl, or mrl,cew")
    parser.add_argument("--augment", default="none", help="none, or webcam (Step 10)")
    parser.add_argument("--cew-share", type=float, default=0.0,
                        help="share of CEW eyes in each epoch (drawn with replacement); 0 = natural share")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    data = np.load(DATA)
    sources = [SOURCES[s] for s in args.sources.split(",")]
    use = np.isin(data["source"], sources)
    tr, va = np.flatnonzero(use & (data["split"] == TRAIN)), np.flatnonzero(use & (data["split"] == VAL))
    pixels = data["X"][tr].astype(np.float32) / 255
    mean, std = float(pixels.mean()), float(pixels.std())

    X_tr, y_tr = torch.from_numpy(data["X"][tr]).to(device), torch.from_numpy(data["y"][tr].astype(np.int64)).to(device)
    X_va, y_va = torch.from_numpy(data["X"][va]).to(device), torch.from_numpy(data["y"][va].astype(np.int64)).to(device)
    src_tr, src_va = data["source"][tr], data["source"][va]
    names = {v: k for k, v in SOURCES.items()}
    if args.cew_share > 0:  # each sample's chance of being drawn, so CEW fills cew_share of every epoch
        share = np.where(src_tr == SOURCES["cew"], args.cew_share, 1 - args.cew_share)
        draw_weights = torch.from_numpy(share / np.bincount(src_tr)[src_tr]).float().to(device)

    out_dir, plot_dir = ROOT / "models" / args.name, ROOT / "assets" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train_log.txt", "w")

    def log(line=""):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    model = EyeCNN().to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(tr) / args.batch_size)
    total, warmup = args.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    config = {**vars(args), "input": "eye32", "size": 32, "mean": mean, "std": std, "classes": CLASSES,
              "params": params, "train_images": len(tr), "val_images": len(va), "device": str(device)}

    runlog = RunLog(args.name, kind="train", classes=CLASSES, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                    config=config)
    log(f"Model {args.name} | EyeCNN {params:,} params | training on {device} | {len(tr):,} train / {len(va):,} val "
        f"eyes ({args.sources}) | {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}, augment={args.augment}, "
        f"cew_share={args.cew_share}\n")
    header = "".join(f" {'val_' + names[s]:>8}" for s in sources) if len(sources) > 1 else ""
    log(f"{'epoch':>5} {'train_loss':>10} {'train_acc':>9} {'val_loss':>9} {'val_acc':>8} {'lr':>8} {'time':>6}{header}")
    if len(sources) > 1:
        log("(val_acc = mean of the per-dataset validation accuracies, so the small CEW set counts as much as MRL)")

    history, best_acc, best_epoch, status = [], -1.0, 0, "finished"
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            perm = (torch.multinomial(draw_weights, len(tr), replacement=True) if args.cew_share > 0
                    else torch.randperm(len(tr), device=device))
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), device=device)
            for step in range(steps_per_epoch):
                idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
                x = augment_batch(X_tr[idx].float().div(255).unsqueeze(1), args.augment)
                logits = model((x - mean) / std)
                loss = F.cross_entropy(logits, y_tr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                loss_sum += loss.detach() * len(idx)
                correct += (logits.argmax(1) == y_tr[idx]).sum()
                global_step = (epoch - 1) * steps_per_epoch + step
                if global_step % runlog.step_every == 0:  # .item() waits for the GPU, so only when logging
                    runlog.step(epoch, step, global_step=global_step, loss=loss.item(),
                                acc=(logits.argmax(1) == y_tr[idx]).float().mean().item(), lr=sched.get_last_lr()[0])
                if runlog.stop_requested():
                    raise KeyboardInterrupt("stop button")
            train_loss, train_acc = (loss_sum / len(tr)).item(), (correct / len(tr)).item()
            val_loss, val_acc, cm, probs = evaluate(model, X_va, y_va, mean, std)
            hits = (probs.argmax(1) == y_va).cpu().numpy()
            per_source = {f"val_{names[s]}": float(hits[src_va == s].mean()) for s in sources}
            if len(sources) > 1:
                val_acc = float(np.mean(list(per_source.values())))  # each dataset counts equally
            lr = opt.param_groups[0]["lr"]
            best = val_acc > best_acc
            if best:
                best_acc, best_epoch = val_acc, epoch
                torch.save(model.state_dict(), out_dir / "model.pt")
            seconds = time.time() - t0
            history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss,
                                val_acc=val_acc, lr=lr, time=seconds, **per_source))
            runlog.epoch(epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc, lr=lr,
                         epoch_time=seconds, best=best, confusion=cm, **per_source)
            # sample predictions: 12 mistakes (if any) and 12 random validation eyes
            pred = probs.argmax(1).cpu().numpy()
            yv = y_va.cpu().numpy()
            wrong = np.flatnonzero(pred != yv)
            pick = np.concatenate([rng.choice(wrong, min(12, len(wrong)), replace=False),
                                   rng.choice(len(yv), 24 - min(12, len(wrong)), replace=False)])
            runlog.samples(epoch, [{"image": data["X"][va[i]], "true": CLASSES[yv[i]], "pred": CLASSES[pred[i]],
                                    "conf": float(probs[i].max())} for i in pick])
            extra = "".join(f" {v:>8.1%}" for v in per_source.values()) if len(sources) > 1 else ""
            log(f"{epoch:>5} {train_loss:>10.4f} {train_acc:>9.1%} {val_loss:>9.4f} {val_acc:>8.1%} {lr:>8.1e} "
                f"{seconds:>5.1f}s{extra}{'  * saved' if best else ''}")
    except KeyboardInterrupt:
        status = "stopped"
        log(f"\nStopped during epoch {len(history) + 1}; keeping the best checkpoint (epoch {best_epoch}).")

    train_seconds = time.time() - t_start
    config.update(best_epoch=best_epoch, best_val_acc=best_acc, epochs_done=len(history),
                  train_seconds=round(train_seconds, 1), status=status)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if history:
        with open(out_dir / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        setup()
        plot_history(history, best_epoch, plot_dir / "training_curves.png")
    log(f"\nBest validation accuracy: {best_acc:.1%} at epoch {best_epoch}")
    log(f"Training time: {train_seconds / 60:.1f} min on {device}")
    log(f"Saved {out_dir.relative_to(ROOT)}/ and {plot_dir.relative_to(ROOT)}/training_curves.png")
    runlog.end(status, best_epoch=best_epoch, best_val_acc=best_acc)
    log_file.close()


if __name__ == "__main__":
    main()
