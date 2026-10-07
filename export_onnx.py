#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX and check it against PyTorch with ONNX Runtime.

Input shape is (batch, channels, samples); channels and window length are read
from the processed dataset's meta.json when --data_dir is given.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import load_model


def export(model: torch.nn.Module, onnx_path: Path, in_ch: int, n_samples: int, opset: int = 17) -> float:
    """Export and return the max abs difference between PyTorch and ONNX Runtime outputs."""
    model.eval()
    dummy = torch.randn(1, in_ch, n_samples)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, (dummy,), str(onnx_path), dynamo=False, **kwargs)
    except TypeError:  # torch < 2.5 has no dynamo argument
        torch.onnx.export(model, (dummy,), str(onnx_path), **kwargs)

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, in_ch, n_samples)
    ort_out = sess.run(["logits"], {"input": x.numpy()})[0]
    with torch.no_grad():
        pt_out = model(x).numpy()
    return float(np.max(np.abs(ort_out - pt_out)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--onnx_path", type=str, required=True)
    ap.add_argument("--data_dir", type=str, default=None, help="processed dataset dir with meta.json")
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--in_ch", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    meta = json.loads((Path(args.data_dir) / "meta.json").read_text()) if args.data_dir else {}
    num_classes = args.num_classes or len(meta.get("classes", [])) or 5
    in_ch = args.in_ch or len(meta.get("channels", [])) or 1
    n_samples = args.n_samples or meta.get("n_samples") or 3000

    model = load_model(args.model, num_classes, in_ch)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))  # strict: fail on mismatch
    diff = export(model, Path(args.onnx_path), in_ch, n_samples, args.opset)
    print(f"Exported {args.onnx_path} (input 1x{in_ch}x{n_samples}, {num_classes} classes); "
          f"ONNX Runtime vs PyTorch max abs diff {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("ONNX output does not match PyTorch")


if __name__ == "__main__":
    main()
