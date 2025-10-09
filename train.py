#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

import importlib


class NPYSleepDataset(Dataset):
    def __init__(self, root: Path, subjects: list[int]):
        self.samples = []
        for s in subjects:
            x = np.load(root / f"subject{s:02d}_x.npy")
            y = np.load(root / f"subject{s:02d}_y.npy")
            for i in range(len(x)):
                self.samples.append((x[i], y[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.tensor(int(y)).long()
        return x, y


def load_model(model_name: str, num_classes: int):
    if model_name == "resnet1d_tiny":
        mod = importlib.import_module("models.resnet1d_tiny")
    elif model_name == "mobilenet1d":
        mod = importlib.import_module("models.mobilenet1d")
    else:
        raise ValueError(model_name)
    return mod.create_model(in_ch=1, num_classes=num_classes)


def train_one_fold(model, train_loader, val_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1 = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        # val
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                gts.append(yb.numpy())
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        f1 = f1_score(gts, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="sleepedf")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet1d_tiny")
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log_dir", type=str, default="results")
    args = ap.parse_args()

    data_root = Path(args.data_root) / args.dataset
    meta = np.load(data_root / "meta.csv", allow_pickle=True)
    num_subjects = 0
    try:
        import pandas as pd
        df = pd.read_csv(data_root / "meta.csv")
        num_subjects = len(df)
    except Exception:
        num_subjects = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f1s = []
    for val_subj in range(num_subjects):
        train_subj = [s for s in range(num_subjects) if s != val_subj]
        train_ds = NPYSleepDataset(data_root, train_subj)
        val_ds = NPYSleepDataset(data_root, [val_subj])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = load_model(args.model, args.num_classes).to(device)
        best_f1 = train_one_fold(model, train_loader, val_loader, device, args.epochs, args.lr)
        f1s.append(best_f1)

    print(f"CV macro-F1: mean={np.mean(f1s):.3f} std={np.std(f1s):.3f}")
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"artifacts/{args.dataset}_{args.model}_fp32.pt")


if __name__ == "__main__":
    main()

