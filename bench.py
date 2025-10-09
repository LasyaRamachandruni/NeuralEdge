#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import importlib
from sklearn.metrics import f1_score


def load_subject(root: Path, subj: int):
    x = np.load(root / f"subject{subj:02d}_x.npy")
    y = np.load(root / f"subject{subj:02d}_y.npy")
    return x, y


def evaluate_pytorch(model, x: np.ndarray, batch: int = 1):
    model.eval()
    latencies = []
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i : i + batch]).float().unsqueeze(1)
            t0 = time.perf_counter()
            logits = model(xb)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    return np.array(latencies), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet1d_tiny")
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--mode", type=str, choices=["host", "jetson", "pi"], required=True)
    ap.add_argument("--results", type=str, required=True)
    ap.add_argument("--trt_engine", type=str, default=None)
    ap.add_argument("--tflite_model", type=str, default=None)
    ap.add_argument("--power_log", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root) / args.dataset
    df = pd.read_csv(data_root / "meta.csv")
    subjects = df["subject"].tolist()

    # Host/PyTorch path
    if args.mode == "host":
        mod = importlib.import_module(f"models.{args.model}")
        model = mod.create_model(1, args.num_classes)
        # try to load checkpoint if provided or default exists
        ckpt = args.checkpoint
        if ckpt is None:
            maybe = Path("artifacts") / f"{args.dataset}_{args.model}_fp32.pt"
            if maybe.exists():
                ckpt = str(maybe)
        if ckpt and Path(ckpt).exists():
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)
        model.eval()
        all_f1, all_lat = [], []
        for s in subjects:
            x, y = load_subject(data_root, s)
            lat, preds = evaluate_pytorch(model, x)
            all_lat.extend(lat)
            all_f1.append(f1_score(y, preds, average="macro"))
        p50 = float(np.percentile(all_lat, 50))
        p90 = float(np.percentile(all_lat, 90))
        row = {
            "dataset": args.dataset,
            "variant": f"{args.model}_fp32",
            "p50_ms": p50,
            "p90_ms": p90,
            "macro_f1": float(np.mean(all_f1)),
        }
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            df_out = pd.read_csv(out)
            df_out = pd.concat([df_out, pd.DataFrame([row])], ignore_index=True)
        else:
            df_out = pd.DataFrame([row])
        df_out.to_csv(out, index=False)
        print("Saved results to", out)
    elif args.mode == "jetson":
        # Placeholder: in-container run should use TensorRT Python bindings to measure inference-only latency
        print("Jetson TRT benchmarking should be executed in jetson container.")
        if args.power_log and Path(args.power_log).exists():
            watts = pd.read_csv(args.power_log, header=None)
            print("Power log captured.")
    elif args.mode == "pi":
        print("Raspberry Pi TFLite benchmarking should be executed in pi container.")


if __name__ == "__main__":
    main()

