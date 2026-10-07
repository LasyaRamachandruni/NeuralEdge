#!/usr/bin/env python3
"""ONNX Runtime CPU latency of each architecture with RANDOM (untrained) weights.

Latency of these CNNs does not depend on the weight values, so this measures the
architectures before any real-data training. It says nothing about accuracy.

  python scripts/measure_arch_latency.py --out results/cpu_latency_architecture.json
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import bench_onnx_cpu  # noqa: E402
from export_onnx import export  # noqa: E402
from train import MODEL_MODULES, load_model  # noqa: E402

SHAPES = {"sleepedf": (1, 3000, 5), "wesad": (3, 256, 2)}  # channels, samples, classes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/cpu_latency_architecture.json")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()
    torch.manual_seed(0)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted(MODEL_MODULES):
            for ds, (ch, n, k) in SHAPES.items():
                model = load_model(name, k, ch)
                path = Path(tmp) / f"{name}_{ds}.onnx"
                export(model, path, ch, n)
                r = bench_onnx_cpu(path, args.threads, 1, args.warmup, args.iters)
                r.update(model=name, input=ds, params=sum(p.numel() for p in model.parameters()),
                         weights="random init (architecture only, no accuracy meaning)")
                rows.append(r)
                print(f"{name:14s} {ds:9s} 1x{ch}x{n}  params {r['params']:7,d}  "
                      f"p50 {r['p50_ms']:.3f} ms  p90 {r['p90_ms']:.3f} ms  p99 {r['p99_ms']:.3f} ms")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
