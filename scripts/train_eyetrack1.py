"""Step 13: train eyeTrack1 with 5-fold cross-validation on UTA-RLDD and compare it with the rules baseline.

For each fold k (12 people) as the test set:
  validation = the next fold (k + 1), used only to pick the best epoch
  training   = the other 3 folds (36 people)
  normalization statistics come from the training folds only.
Every person is tested exactly once, by a model that never saw them. The same windows are judged by
  rules        the demo's own rules (web/rules.js, replayed in Step 12): OK / Attention / Take a break
  rules tuned  PERCLOS alone, with its two thresholds fitted on the 4 non-test folds (a fairer, data-driven rule)
and compared with the numbers reported by the dataset's authors (Ghoddoosian et al., CVPR-W 2019, Table 2).

Finally one model is trained on all 60 people (for the demo) for the median best number of epochs.

Usage:
  python scripts/train_eyetrack1.py

Outputs:
  models/eyeTrack1/{model.pt, config.json, cv_report.txt, train_log.txt}
  assets/eyeTrack1/{comparison.png, confusion.png, training_curves.png}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.runlog import RunLog  # noqa: E402
from src.style import COLORS, finish, setup  # noqa: E402
from src.temporal import CLASSES, DrowsinessGRU  # noqa: E402

DATA = ROOT / "data" / "processed" / "rldd5.npz"
OUT = ROOT / "models" / "eyeTrack1"
ASSETS = ROOT / "assets" / "eyeTrack1"
PAPER = {"window": 0.54, "video": 0.652, "human_video": 0.578}  # Ghoddoosian et al. 2019, Table 2: BSA, VA, human VA


def build_windows(d):
    """All windows as one array (N, 300, 13) plus face masks (N, 300)."""
    size = int(d["window"])
    starts = d["offsets"][d["win_video"]] + d["win_start"]
    idx = starts[:, None] + np.arange(size)[None, :]
    return d["x"][idx], d["face"][idx]


def normalize(X, M, train_mask, subjects=None):
    """z-score with statistics of the training windows' face frames; missing faces become 0; append the face flag.
    subjects: if given, each person's features are first centred on that person's own average over all their
    windows (label-free, but it needs the person's data from all three sessions: an analysis, not deployable)."""
    if subjects is not None:
        X = X.copy()
        for s in np.unique(subjects):
            m = subjects == s
            X[m] = X[m] - X[m][M[m]].mean(0)
    frames = X[train_mask][M[train_mask]]
    mean, std = frames.mean(0), frames.std(0) + 1e-6
    Z = ((X - mean) / std) * M[..., None]
    return np.concatenate([Z, M[..., None].astype(np.float32)], -1).astype(np.float32), mean, std


def macro_f1(y, p, n=3):
    f = []
    for c in range(n):
        tp = np.sum((p == c) & (y == c))
        prec, rec = tp / max(1, np.sum(p == c)), tp / max(1, np.sum(y == c))
        f.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f))


def train_one(name, Z, y, tr, va, epochs, args, device, log_lines):
    torch.manual_seed(args.seed)
    model = DrowsinessGRU(hidden=args.hidden, dropout=args.dropout, proj=args.proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr, ytr = torch.from_numpy(Z[tr]).to(device), torch.from_numpy(y[tr]).long().to(device)
    Xva = torch.from_numpy(Z[va]).to(device) if va is not None else None
    steps = int(np.ceil(len(tr) / args.batch_size))
    log = RunLog(name, kind="train", classes=CLASSES, epochs=epochs, steps_per_epoch=steps, step_every=5,
                 config={**vars(args), "train_windows": len(tr), "val_windows": 0 if va is None else len(va)})
    best, best_state, best_epoch, history = -1, None, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(tr), device=device)
        loss_sum, correct = 0.0, 0
        for s in range(steps):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            x = Xtr[idx]
            x = x + torch.randn_like(x) * args.noise * x[..., -1:]  # small jitter on frames with a face
            if args.person_aug > 0:  # a new driver has a different 'normal': random offset and scale per window
                n, _, f = x.shape
                off = torch.randn(n, 1, f, device=x.device) * args.person_aug
                scale = torch.exp(torch.randn(n, 1, f, device=x.device) * args.person_aug / 2)
                off[..., -1] = 0
                scale[..., -1] = 1
                x = (x * scale + off) * x[..., -1:] + x * (1 - x[..., -1:])  # only frames with a face
            logits = model(x)
            loss = F.cross_entropy(logits, ytr[idx], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == ytr[idx]).sum().item()
            log.step(epoch, s, global_step=(epoch - 1) * steps + s, loss=loss.item(), lr=sched.get_last_lr()[0])
            if log.stop_requested():
                break
        sched.step()
        row = {"epoch": epoch, "train_loss": loss_sum / len(tr), "train_acc": correct / len(tr), "epoch_time": time.time() - t0}
        if va is not None:
            probs = predict(model, Xva)
            pred = probs.argmax(1)
            row.update(val_loss=float(F.nll_loss(torch.log(torch.from_numpy(probs) + 1e-9), torch.from_numpy(y[va]).long())),
                       val_acc=float(np.mean(pred == y[va])), val_f1=macro_f1(y[va], pred))
            cm = np.zeros((3, 3), np.int64)
            np.add.at(cm, (y[va], pred), 1)
            is_best = row["val_f1"] > best
            if is_best:
                best, best_epoch = row["val_f1"], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            log.epoch(epoch, train_loss=row["train_loss"], train_acc=row["train_acc"], val_loss=row["val_loss"],
                      val_acc=row["val_acc"], val_f1=row["val_f1"], lr=sched.get_last_lr()[0], epoch_time=row["epoch_time"],
                      best=is_best, confusion=cm)
            pick = np.random.default_rng(epoch).choice(len(va), 12, replace=False)
            log.samples(epoch, [{"series": {"eye closure (eyeBlink)": Z[va[i], ::5, 0].round(2).tolist(),
                                            "head pitch": Z[va[i], ::5, 10].round(2).tolist()},
                                 "true": CLASSES[y[va[i]]], "pred": CLASSES[pred[i]], "conf": float(probs[i].max())}
                                for i in pick])
            line = (f"  {name:<18} epoch {epoch:>2}  train loss {row['train_loss']:.3f} acc {row['train_acc']:.1%}  "
                    f"val loss {row['val_loss']:.3f} acc {row['val_acc']:.1%} F1 {row['val_f1']:.1%}"
                    f"{'  * best' if is_best else ''}  {row['epoch_time']:.1f}s")
        else:
            log.epoch(epoch, train_loss=row["train_loss"], train_acc=row["train_acc"], lr=sched.get_last_lr()[0],
                      epoch_time=row["epoch_time"])
            line = f"  {name:<18} epoch {epoch:>2}  train loss {row['train_loss']:.3f} acc {row['train_acc']:.1%}  {row['epoch_time']:.1f}s"
        print(line, flush=True)
        log_lines.append(line)
        history.append(row)
        if log.stop_requested():
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    log.end("finished", best_epoch=best_epoch, best_val_f1=best)
    return model, best_epoch, history


@torch.no_grad()
def predict(model, X, batch=512):
    model.eval()
    return torch.cat([F.softmax(model(X[i:i + batch]), 1).cpu() for i in range(0, len(X), batch)]).numpy()


def fit_perclos_thresholds(perclos, y):
    """Two PERCLOS thresholds (alert < a <= low vigilant < b <= drowsy) maximizing macro F1."""
    grid = np.round(np.arange(0.0, 0.40, 0.005), 3)
    best = (-1, 0, 0)
    for a in grid:
        for b in grid[grid > a]:
            p = np.where(perclos >= b, 2, np.where(perclos >= a, 1, 0))
            f = macro_f1(y, p)
            if f > best[0]:
                best = (f, a, b)
    return best[1], best[2]


def summarize(name, y, pred, videos, probs=None):
    """Window accuracy, macro F1, per-class recall, binary alert-vs-drowsy, video accuracy."""
    r = {"acc": float(np.mean(pred == y)), "f1": macro_f1(y, pred),
         "recall": [float(np.mean(pred[y == c] == c)) for c in range(3)]}
    ad = y != 1
    if probs is not None:
        bin_pred = np.where(probs[ad, 2] > probs[ad, 0], 2, 0)
        vid_scores = {v: probs[videos == v].mean(0) for v in np.unique(videos)}
        vid_pred = {v: int(s.argmax()) for v, s in vid_scores.items()}
    else:
        bin_pred = np.where(pred[ad] >= 1, 2, 0)  # rules: any warning counts as drowsy
        vid_pred = {v: int(np.bincount(pred[videos == v], minlength=3).argmax()) for v in np.unique(videos)}
    r["binary_acc"] = float(np.mean(bin_pred == y[ad]))
    r["binary_drowsy_recall"] = float(np.mean(bin_pred[y[ad] == 2] == 2))
    r["binary_alert_recall"] = float(np.mean(bin_pred[y[ad] == 0] == 0))
    vid_true = {v: int(y[videos == v][0]) for v in vid_pred}
    r["video_acc"] = float(np.mean([vid_pred[v] == vid_true[v] for v in vid_pred]))
    r["cm"] = np.zeros((3, 3), np.int64)
    np.add.at(r["cm"], (y, pred), 1)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu or mps")
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--proj", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--subject-norm", action="store_true", help="centre each person on their own average (analysis)")
    parser.add_argument("--person-aug", type=float, default=0.0,
                        help="std of the random per-window feature offset (and log-scale / 2) simulating a new driver")
    parser.add_argument("--select", action="store_true",
                        help="selection run: report validation scores only, never touch the test folds")
    parser.add_argument("--tag", default="", help="suffix for run names and outputs of experiments")
    parser.add_argument("--no-final", action="store_true", help="cross-validation only, no model for the demo")
    args = parser.parse_args()
    setup()
    device = torch.device(args.device)
    d = np.load(DATA)
    X, M = build_windows(d)
    y, fold, video = d["win_label"].astype(np.int64), d["win_fold"], d["win_video"]
    global OUT, ASSETS
    OUT, ASSETS = OUT.with_name(f"eyeTrack1{args.tag}"), ASSETS.with_name(f"eyeTrack1{args.tag}")
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    log_lines = [f"eyeTrack1 | {len(y):,} windows of {int(d['window'])} frames ({int(d['window']) // int(d['fps'])} s at "
                 f"{int(d['fps'])} fps), {X.shape[-1]} features + face flag | {args.epochs} epochs, batch {args.batch_size}, "
                 f"lr {args.lr}, device {device}"]
    print(log_lines[0])
    params = sum(p.numel() for p in DrowsinessGRU(hidden=args.hidden, dropout=args.dropout, proj=args.proj).parameters())
    subjects = d["win_subject"] if args.subject_norm else None
    name = f"eyeTrack1{args.tag}"
    print(f"{name}: {params:,} parameters, hidden {args.hidden}, dropout {args.dropout}, weight decay {args.weight_decay}, "
          f"noise {args.noise}, subject norm {args.subject_norm}")
    val_scores = []

    probs_all = np.zeros((len(y), 3), np.float32)
    tuned_pred = np.zeros(len(y), np.int64)
    best_epochs, histories, t_start = [], {}, time.time()
    for k in range(1, 6):
        val_fold = k % 5 + 1
        te, va = np.flatnonzero(fold == k), np.flatnonzero(fold == val_fold)
        tr = np.flatnonzero((fold != k) & (fold != val_fold))
        Z, _, _ = normalize(X, M, np.isin(np.arange(len(y)), tr), subjects)
        model, best_epoch, hist = train_one(f"{name}_fold{k}", Z, y, tr, va, args.epochs, args, device, log_lines)
        val_scores.append(max(h["val_f1"] for h in hist))
        if args.select:
            print(f"fold {k}: best validation macro F1 {val_scores[-1]:.1%} at epoch {best_epoch}", flush=True)
            best_epochs.append(best_epoch)
            continue
        probs_all[te] = predict(model, torch.from_numpy(Z[te]).to(device))
        a, b = fit_perclos_thresholds(d["rules_perclos"][fold != k], y[fold != k])
        tuned_pred[te] = np.where(d["rules_perclos"][te] >= b, 2, np.where(d["rules_perclos"][te] >= a, 1, 0))
        best_epochs.append(best_epoch)
        histories[k] = hist
        acc = np.mean(probs_all[te].argmax(1) == y[te])
        line = (f"fold {k}: test people {sorted(set(d['win_subject'][te].tolist()))}, best epoch {best_epoch}, "
                f"test window accuracy {acc:.1%}; tuned PERCLOS thresholds {a:.1%} / {b:.1%}")
        print(line, flush=True)
        log_lines.append(line)
    cv_minutes = (time.time() - t_start) / 60
    if args.select:
        print(f"\nSELECTION {name}: mean best validation macro F1 {np.mean(val_scores):.1%} "
              f"(folds {[round(v, 3) for v in val_scores]}), best epochs {best_epochs}, {cv_minutes:.1f} min")
        return

    results = {name: summarize(name, y, probs_all.argmax(1), video, probs_all),
               "rules (demo)": summarize("rules", y, d["rules_level"].astype(np.int64), video),
               "rules (PERCLOS tuned)": summarize("tuned", y, tuned_pred, video)}
    lines = ["", f"5-fold cross-validation on UTA-RLDD: {len(y):,} windows of 60 s, 168 videos, 60 people; "
                 f"every window judged by a model that never saw its person",
             f"{'':<34}" + "".join(f"{n:>22}" for n in results)]
    def row(label, f):
        lines.append(f"{label:<34}" + "".join(f"{f(r):>22}" for r in results.values()))
    row("3 classes, window accuracy", lambda r: f"{r['acc']:.1%}")
    row("3 classes, macro F1", lambda r: f"{r['f1']:.1%}")
    for c in range(3):
        row(f"  recall {CLASSES[c]}", lambda r, c=c: f"{r['recall'][c]:.1%}")
    row("3 classes, video accuracy (vote)", lambda r: f"{r['video_acc']:.1%}")
    row("alert vs drowsy, window accuracy", lambda r: f"{r['binary_acc']:.1%}")
    row("  drowsy windows caught", lambda r: f"{r['binary_drowsy_recall']:.1%}")
    row("  alert windows left alone", lambda r: f"{r['binary_alert_recall']:.1%}")
    lines += ["", "Reported by the dataset's authors (Ghoddoosian, Galib, Athitsos, CVPR Workshops 2019, Table 2; "
                  "blink features, 5 folds):",
              f"  HM-LSTM: blink-sequence accuracy {PAPER['window']:.0%}, video accuracy {PAPER['video']:.1%}; "
              f"human judgment: video accuracy {PAPER['human_video']:.1%}",
              "  Not identical setups: they use 180 videos and blink sequences, we use 168 quality-checked videos and",
              "  60 s windows of MediaPipe features; window accuracy is compared with their sequence accuracy (BSA)."]
    report = "\n".join(lines)
    print(report)

    if args.no_final:
        (OUT / "cv_report.txt").write_text(report.lstrip() + f"\nCross-validation took {cv_minutes:.1f} min.\n")
        (OUT / "train_log.txt").write_text("\n".join(log_lines) + "\n")
        plots(results, histories)
        return

    # final model on all 60 people for the demo
    final_epochs = int(np.median(best_epochs))
    Z, mean, std = normalize(X, M, np.ones(len(y), bool), subjects)
    t0 = time.time()
    model, _, _ = train_one(name, Z, y, np.arange(len(y)), None, final_epochs, args, device, log_lines)
    final_minutes = (time.time() - t0) / 60
    torch.save(model.state_dict(), OUT / "model.pt")
    config = {**vars(args), "features": [str(f) for f in d["features"]], "fps": int(d["fps"]), "window": int(d["window"]),
              "mean": mean.tolist(), "std": std.tolist(), "classes": CLASSES, "params": params,
              "cv_best_epochs": best_epochs, "final_epochs": final_epochs, "cv_minutes": round(cv_minutes, 1),
              "final_minutes": round(final_minutes, 1), "windows": int(len(y)),
              "cv": {n: {k: v for k, v in r.items() if k != "cm"} for n, r in results.items()}}
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    extra = (f"\nParameters: {params:,}. Cross-validation took {cv_minutes:.1f} min; the final model (all 60 people, "
             f"{final_epochs} epochs = median best epoch of the folds {best_epochs}) took {final_minutes:.1f} min on {device}.")
    print(extra)
    (OUT / "cv_report.txt").write_text(report.lstrip() + "\n" + extra + "\n")
    (OUT / "train_log.txt").write_text("\n".join(log_lines) + "\n")
    plots(results, histories)


def plots(results, histories):
    names = list(results)
    groups = [("3 classes\nwindow accuracy", "acc"), ("3 classes\nmacro F1", "f1"), ("3 classes\nvideo accuracy", "video_acc"),
              ("alert vs drowsy\nwindow accuracy", "binary_acc"), ("drowsy windows\ncaught", "binary_drowsy_recall")]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(groups))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [results[n][k] for _, k in groups]
        b = ax.bar(x - 0.4 + w * (i + 0.5), vals, w - 0.02, color=COLORS["series"][[2, 0, 4][i]], label=n)
        for bar in b:
            ax.annotate(f"{bar.get_height():.0%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=COLORS["muted"])
    for gx, key, label in [(0, "window", "paper HM-LSTM (sequences)"), (2, "video", "paper HM-LSTM (videos)")]:
        ax.plot([gx - 0.42, gx + 0.42], [PAPER[key]] * 2, color=COLORS["text"], lw=1.5, ls="--")
        ax.text(gx, PAPER[key] + 0.015, f"{label} {PAPER[key]:.0%}" if key == "window" else f"{label} {PAPER[key]:.1%}",
                ha="center", fontsize=8)
    ax.axhline(1 / 3, color=COLORS["muted"], lw=0.8, ls=":")
    ax.set_xticks(x, [g for g, _ in groups], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, ncols=3, loc="upper left")
    ax.set_title("UTA-RLDD, 5-fold cross-validation (people never seen in training)", loc="left", fontsize=13)
    finish(ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, n in zip(axes, [list(results)[0], "rules (demo)"]):
        cm = results[n]["cm"]
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else COLORS["text"])
        ticks = ["OK", "Attention", "Take a break"] if n == "rules (demo)" else CLASSES
        ax.set_xticks(range(3), ticks, fontsize=9)
        ax.set_yticks(range(3), CLASSES, fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(n, loc="left", fontsize=11)
        ax.tick_params(length=0)
    fig.suptitle("Windows by true class (rows) and prediction (columns), all 5 test folds", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "confusion.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for ax, (k, hist) in zip(axes, histories.items()):
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, [h["train_acc"] for h in hist], color=COLORS["train"], label="train")
        ax.plot(ep, [h["val_acc"] for h in hist], color=COLORS["val"], label="validation")
        ax.set_title(f"test fold {k}", loc="left", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        finish(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{list(results)[0]} accuracy per epoch (validation = another fold's people)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved {ASSETS.relative_to(ROOT)}/comparison.png, confusion.png, training_curves.png")


if __name__ == "__main__":
    main()
