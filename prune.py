#!/usr/bin/env python3
import argparse
import importlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import NPYSleepDataset


def l1_channel_prune(module: nn.Conv1d, prune_ratio: float) -> None:
    with torch.no_grad():
        weight = module.weight.detach().abs().sum(dim=(1, 2))
        k = int(prune_ratio * weight.numel())
        if k <= 0:
            return
        thresh = torch.topk(weight, k, largest=False).values.max()
        mask = weight > thresh
        module.weight.mul_(mask[:, None, None])


def apply_pruning(model: nn.Module, ratio: float):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) and m.groups == 1:
            l1_channel_prune(m, ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet1d_tiny")
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--prune_ratio", type=float, default=0.3)
    ap.add_argument("--finetune_epochs", type=int, default=2)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root) / args.dataset
    import pandas as pd
    df = pd.read_csv(data_root / "meta.csv")
    subjects = df["subject"].tolist()

    mod = importlib.import_module(f"models.{args.model}")
    model = mod.create_model(1, args.num_classes)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    apply_pruning(model, args.prune_ratio)

    train_ds = NPYSleepDataset(data_root, subjects[:-1])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.finetune_epochs):
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print("Saved pruned model to", args.output)


if __name__ == "__main__":
    main()

