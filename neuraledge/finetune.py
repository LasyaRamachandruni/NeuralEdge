"""Helpers shared by qat_train.py and prune.py.

Both start from the deploy checkpoint written by train.py and reuse *that fold's*
subject split (read from results/<dataset>_<model>_cv.json), so the compressed
model is fine-tuned on the same training subjects and scored on the same unseen
test subjects as the FP32 model it is compared against.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from neuraledge.dataio import SubjectWindowDataset, load_meta


def deploy_split(log_dir: str, dataset: str, model: str) -> dict:
    cv_path = Path(log_dir) / f"{dataset}_{model}_cv.json"
    if not cv_path.exists():
        raise FileNotFoundError(f"{cv_path} not found; run train.py first")
    cv = json.loads(cv_path.read_text())
    fold = cv["folds"][cv["deploy_checkpoint"]["fold"]]
    return {"fold": fold["fold"], "train": fold["train_subjects"], "val": fold["val_subjects"],
            "test": fold["test_subjects"], "checkpoint": cv["deploy_checkpoint"]["path"]}


def loaders(data_root: Path, split: dict, batch_size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(SubjectWindowDataset(data_root, split["train"]), batch_size=batch_size,
                       shuffle=True, generator=g)
    test = DataLoader(SubjectWindowDataset(data_root, split["test"]), batch_size=batch_size)
    return train, test


def finetune(model: nn.Module, loader: DataLoader, epochs: int, lr: float, max_batches: int | None = None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for i, (xb, yb) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()


@torch.no_grad()
def macro_f1(model, loader: DataLoader, num_classes: int) -> float:
    if hasattr(model, "eval"):
        model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        preds.append(model(xb).argmax(1).numpy())
        gts.append(yb.numpy())
    return float(f1_score(np.concatenate(gts), np.concatenate(preds), labels=list(range(num_classes)),
                          average="macro", zero_division=0))


def dataset_info(data_root: Path) -> tuple[dict, int, int, int]:
    meta = load_meta(data_root)
    return meta, len(meta["classes"]), len(meta["channels"]), int(meta["n_samples"])
