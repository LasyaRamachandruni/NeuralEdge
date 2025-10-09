#!/usr/bin/env python3
import argparse
import importlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert,
)

from train import NPYSleepDataset
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet1d_tiny")
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
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
    model.train()

    model.fuse_modules = getattr(model, "fuse_modules", None)
    qconfig = get_default_qat_qconfig("fbgemm")
    model.qconfig = qconfig
    prepared = prepare_qat(model)

    # One-fold quick fine-tune
    train_ds = NPYSleepDataset(data_root, subjects[:-1])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(prepared.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = prepared(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    quantized = convert(prepared.eval())
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(quantized.state_dict(), args.output)
    print("Saved QAT model to", args.output)


if __name__ == "__main__":
    main()

