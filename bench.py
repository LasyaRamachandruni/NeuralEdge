#!/usr/bin/env python3
"""Inference latency benchmarking.

Implemented
  onnx-cpu   ONNX Runtime on the CPU of whatever machine runs this script. Latency
             depends only on the architecture and input shape, not on the weights,
             so it can be measured before real-data training. Runs as-is on a
             Raspberry Pi 4 (aarch64 onnxruntime wheel) -- see docs/edge_benchmarking.md.

Not implemented yet (TODO)
  jetson     TensorRT engine latency + tegrastats power on a Jetson Nano
  pi         TFLite (int8) latency on a Raspberry Pi 4
  Both print the on-device commands to use instead and exit with status 2.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TODO_MSG = {
    "jetson": (
        "Jetson TensorRT benchmarking is not implemented in bench.py yet and nothing has been "
        "measured on a Jetson. On the device, build an engine and time it with trtexec:\n"
        "  /usr/src/tensorrt/bin/trtexec --onnx=artifacts/sleepedf_resnet1d_tiny.onnx --fp16 "
        "--shapes=input:1x1x3000 --iterations=1000 --avgRuns=100\n"
        "and log power with: tegrastats --interval 200 --logfile tegrastats.log\n"
        "See docs/edge_benchmarking.md."
    ),
    "pi": (
        "Raspberry Pi TFLite benchmarking is not implemented in bench.py yet and nothing has been "
        "measured on a Pi. On the Pi you can already run the ONNX Runtime path:\n"
        "  python3 bench.py --mode onnx-cpu --onnx artifacts/sleepedf_resnet1d_tiny.onnx --threads 4\n"
        "See docs/edge_benchmarking.md."
    ),
}


def cpu_description() -> str:
    try:
        info = {}
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                info.setdefault(key.strip().lower(), val.strip())
        # x86: "model name"; Raspberry Pi: "Model" (board) / "Hardware"
        for key in ("model name", "model", "hardware"):
            if key == "model" and info.get(key, "").isdigit():
                continue
            if info.get(key):
                return info[key]
    except OSError:
        pass
    return platform.processor() or platform.machine()


def bench_onnx_cpu(onnx_path: Path, threads: int = 1, batch: int = 1, warmup: int = 50,
                   iters: int = 500, seed: int = 0) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [batch if not isinstance(d, int) else d for d in inp.shape]
    shape[0] = batch
    x = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    lat = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        lat[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "runtime": f"onnxruntime {ort.__version__} CPUExecutionProvider",
        "cpu": cpu_description(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "threads": threads,
        "batch": batch,
        "input_shape": shape,
        "warmup": warmup,
        "iters": iters,
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p99_ms": float(np.percentile(lat, 99)),
        "mean_ms": float(lat.mean()),
        "model_bytes": Path(onnx_path).stat().st_size,
    }


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat))
        if not exists:
            w.writeheader()
        w.writerow(flat)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["onnx-cpu", "jetson", "pi"], required=True)
    ap.add_argument("--onnx", type=str, help="ONNX model (onnx-cpu)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--label", type=str, default="", help="free text stored with the result, "
                    "e.g. 'random weights, architecture only' or 'sleepedf fold 2 checkpoint'")
    ap.add_argument("--json_out", type=str, default=None)
    ap.add_argument("--results", type=str, default=None, help="CSV to append a summary row to")
    args = ap.parse_args(argv)

    if args.mode in TODO_MSG:
        print("TODO:", TODO_MSG[args.mode])
        raise SystemExit(2)
    if not args.onnx:
        ap.error("--onnx is required for --mode onnx-cpu")

    res = {"model": Path(args.onnx).name, "label": args.label,
           "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           **bench_onnx_cpu(Path(args.onnx), args.threads, args.batch, args.warmup, args.iters)}
    print(f"{res['model']} on {res['cpu']} ({res['threads']} thread(s), batch {res['batch']}): "
          f"p50 {res['p50_ms']:.3f} ms  p90 {res['p90_ms']:.3f} ms  p99 {res['p99_ms']:.3f} ms")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(res, indent=2))
    if args.results:
        append_csv(Path(args.results), res)
    return res


if __name__ == "__main__":
    main()
