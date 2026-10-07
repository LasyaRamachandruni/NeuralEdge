#!/usr/bin/env python3
"""Subject-wise cross-validated FP32 training.

For every fold the test subjects are never seen during training *or* model
selection: a few of the remaining subjects are held out as a validation set for
early stopping, and the test fold is only scored once at the end.

Outputs
  artifacts/<dataset>_<model>_fold<k>.pt   best checkpoint of each fold
  artifacts/<dataset>_<model>_fp32.pt      copy of the fold with the best *validation* F1
                                           (used by export / QAT / pruning)
  <log_dir>/<dataset>_<model>_cv.json      per-fold + pooled metrics, config, environment
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from neuraledge.dataio import SubjectWindowDataset, load_meta

MODEL_MODULES = {"resnet1d_tiny": "models.resnet1d_tiny", "mobilenet1d": "models.mobilenet1d"}


def load_model(model_name: str, num_classes: int, in_ch: int = 1) -> nn.Module:
    if model_name not in MODEL_MODULES:
        raise ValueError(model_name)
    mod = importlib.import_module(MODEL_MODULES[model_name])
    return mod.create_model(in_ch=in_ch, num_classes=num_classes)


def make_folds(subjects: list[int], n_folds: int, seed: int) -> list[list[int]]:
    """Split subjects into disjoint test groups. n_folds <= 0 means leave-one-subject-out."""
    subjects = list(subjects)
    if n_folds <= 0 or n_folds >= len(subjects):
        return [[s] for s in subjects]
    order = np.random.default_rng(seed).permutation(subjects)
    return [sorted(int(s) for s in chunk) for chunk in np.array_split(order, n_folds)]


def split_train_val(train_subjects: list[int], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    order = [int(s) for s in np.random.default_rng(seed).permutation(train_subjects)]
    n_val = max(1, int(round(val_frac * len(order))))
    if n_val >= len(order):
        raise ValueError("need at least 2 non-test subjects per fold (one to train, one to validate)")
    return sorted(order[n_val:]), sorted(order[:n_val])


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
        gts.append(yb.numpy())
    if not preds:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(preds), np.concatenate(gts)


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def train_one_fold(model, train_loader, val_loader, device, epochs, lr, patience,
                   num_classes, weights=None, log=print):
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    labels = list(range(num_classes))
    best_f1, best_epoch, best_state, history = -1.0, -1, None, []
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            n += len(yb)
        preds, gts = predict(model, val_loader, device)
        val_f1 = float(f1_score(gts, preds, labels=labels, average="macro", zero_division=0))
        history.append({"epoch": epoch + 1, "train_loss": total / max(n, 1), "val_macro_f1": val_f1})
        log(f"  epoch {epoch + 1:3d}  train_loss {total / max(n, 1):.4f}  val_macro_f1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif patience and epoch + 1 - best_epoch >= patience:
            log(f"  early stop (no val improvement for {patience} epochs)")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_macro_f1": best_f1, "best_epoch": best_epoch, "history": history}


def metrics(gts: np.ndarray, preds: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    per_class = f1_score(gts, preds, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(gts)),
        "macro_f1": float(f1_score(gts, preds, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gts, preds)),
        "cohen_kappa": float(cohen_kappa_score(gts, preds, labels=labels)),
        "per_class_f1": {c: float(f) for c, f in zip(classes, per_class)},
        "confusion_matrix": confusion_matrix(gts, preds, labels=labels).tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=str, default="sleepedf")
    ap.add_argument("--data_root", type=str, required=True,
                    help="directory containing <dataset>/meta.json")
    ap.add_argument("--model", type=str, default="resnet1d_tiny", choices=sorted(MODEL_MODULES))
    ap.add_argument("--num_classes", type=int, default=None, help="default: from meta.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--early_stop_patience", type=int, default=5,
                    help="stop when val macro-F1 has not improved for this many epochs (0 = off)")
    ap.add_argument("--folds", type=int, default=5, help="subject-wise folds; 0 = leave-one-subject-out")
    ap.add_argument("--max_folds", type=int, default=None, help="only run the first N folds (quick checks)")
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="fraction of non-test subjects held out for early stopping")
    ap.add_argument("--class_weights", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--num_threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--log_dir", type=str, default="results")
    ap.add_argument("--quiet", action="store_true")
    return ap


def run_cv(args) -> dict:
    log = (lambda *a, **k: None) if args.quiet else print
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root) / args.dataset
    meta = load_meta(data_root)
    classes = meta["classes"]
    num_classes = args.num_classes or len(classes)
    if num_classes != len(classes):
        raise ValueError(f"--num_classes {num_classes} but {data_root} has {len(classes)} classes {classes}")
    in_ch = len(meta["channels"])
    if meta.get("synthetic"):
        log("WARNING: training on SYNTHETIC data -- metrics below are meaningless smoke-test numbers.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = make_folds(meta["subjects"], args.folds, args.seed)
    scheme = "leave-one-subject-out" if all(len(f) == 1 for f in folds) else f"{len(folds)}-fold subject-wise"
    if args.max_folds:
        folds = folds[: args.max_folds]
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    fold_results, all_preds, all_gts = [], [], []
    t0 = time.time()
    for k, test_subj in enumerate(folds):
        rest = [s for s in meta["subjects"] if s not in test_subj]
        train_subj, val_subj = split_train_val(rest, args.val_frac, args.seed + k)
        log(f"fold {k}: train={train_subj} val={val_subj} test={test_subj}")
        train_ds = SubjectWindowDataset(data_root, train_subj)
        val_ds = SubjectWindowDataset(data_root, val_subj)
        test_ds = SubjectWindowDataset(data_root, test_subj)
        g = torch.Generator().manual_seed(args.seed + k)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, generator=g)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

        torch.manual_seed(args.seed + k)
        model = load_model(args.model, num_classes, in_ch).to(device)
        weights = class_weights(train_ds.labels, num_classes) if args.class_weights == "balanced" else None
        fit = train_one_fold(model, train_loader, val_loader, device, args.epochs, args.lr,
                             args.early_stop_patience, num_classes, weights, log)
        preds, gts = predict(model, test_loader, device)
        m = metrics(gts, preds, classes)
        ckpt = art_dir / f"{args.dataset}_{args.model}_fold{k}.pt"
        torch.save(model.state_dict(), ckpt)
        log(f"fold {k}: test macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  -> {ckpt}")
        fold_results.append({"fold": k, "train_subjects": train_subj, "val_subjects": val_subj,
                             "test_subjects": test_subj, "checkpoint": str(ckpt), **fit, "test": m})
        all_preds.append(preds)
        all_gts.append(gts)

    fold_f1 = np.array([f["test"]["macro_f1"] for f in fold_results])
    best = max(fold_results, key=lambda f: f["best_val_macro_f1"])
    deploy_ckpt = art_dir / f"{args.dataset}_{args.model}_fp32.pt"
    shutil.copyfile(best["checkpoint"], deploy_ckpt)

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "synthetic_data": bool(meta.get("synthetic")),
        "data_source": meta.get("source"),
        "classes": classes,
        "channels": meta["channels"],
        "n_subjects": len(meta["subjects"]),
        "cv": {"scheme": scheme, "n_folds_run": len(folds)},
        "macro_f1_mean": float(fold_f1.mean()),
        "macro_f1_std": float(fold_f1.std(ddof=0)),
        "pooled": metrics(np.concatenate(all_gts), np.concatenate(all_preds), classes),
        "deploy_checkpoint": {"path": str(deploy_ckpt), "fold": best["fold"],
                              "selected_by": "best validation macro-F1 (not test)"},
        "folds": fold_results,
        "config": dict(vars(args)),
        "environment": {"torch": torch.__version__, "device": str(device),
                        "python": platform.python_version(), "platform": platform.platform()},
        "wall_time_sec": round(time.time() - t0, 1),
    }
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{args.dataset}_{args.model}_cv.json"
    out.write_text(json.dumps(results, indent=2))
    tag = " (SYNTHETIC DATA)" if results["synthetic_data"] else ""
    print(f"CV macro-F1{tag}: {results['macro_f1_mean']:.3f} ± {results['macro_f1_std']:.3f} "
          f"over {len(folds)} folds; pooled per-class F1: "
          + ", ".join(f"{c}={f:.3f}" for c, f in results["pooled"]["per_class_f1"].items()))
    print(f"Wrote {out}; deploy checkpoint {deploy_ckpt} (fold {best['fold']})")
    return results


def main():
    run_cv(build_parser().parse_args())


if __name__ == "__main__":
    main()
