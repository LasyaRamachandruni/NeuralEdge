#!/usr/bin/env python3
import argparse
import importlib
import torch
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--onnx_path", type=str, required=True)
    args = ap.parse_args()

    if args.model == "resnet1d_tiny":
        mod = importlib.import_module("models.resnet1d_tiny")
    elif args.model == "mobilenet1d":
        mod = importlib.import_module("models.mobilenet1d")
    else:
        raise ValueError(args.model)
    model = mod.create_model(in_ch=1, num_classes=args.num_classes)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy = torch.randn(1, 1, 3000)  # 30 s at 100 Hz
    out_path = Path(args.onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_f=str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch", 2: "time"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX to {out_path}")

    # Round-trip check with ONNX Runtime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"]) 
        ort_out = sess.run(["logits"], {"input": dummy.numpy()})[0]
        with torch.no_grad():
            pt_out = model(dummy).numpy()
        max_abs = float(np.max(np.abs(ort_out - pt_out)))
        print(f"ONNX round-trip max abs diff: {max_abs:.6f}")
    except Exception as e:
        print("ONNXRuntime check skipped:", e)


if __name__ == "__main__":
    main()

