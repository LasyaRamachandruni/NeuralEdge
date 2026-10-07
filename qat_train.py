#!/usr/bin/env python3
"""Quantization-aware fine-tuning to INT8 (PyTorch FX graph mode, x86/fbgemm or qnnpack/ARM).

Starts from the FP32 deploy checkpoint, fine-tunes with fake-quant on the training
subjects of that checkpoint's fold, converts to INT8 and reports macro-F1 of FP32
vs INT8 on the same held-out test subjects. Saves a TorchScript INT8 model.

Note: this is PyTorch-side INT8 (CPU). The TensorRT INT8 engine is built separately
from the FP32 ONNX with a calibration set (build_trt.py).
"""
import argparse
import copy
import json
import warnings
from pathlib import Path

import torch
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

from neuraledge.finetune import dataset_info, deploy_split, finetune, loaders, macro_f1
from train import load_model


def quantize_qat(model, example, train_loader, epochs, lr, backend="x86", max_batches=None):
    torch.backends.quantized.engine = "qnnpack" if backend == "qnnpack" else "x86"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_qat_fx(copy.deepcopy(model).train(), get_default_qat_qconfig_mapping(backend),
                                  example_inputs=(example,))
        finetune(prepared, train_loader, epochs, lr, max_batches)
        return convert_fx(prepared.eval())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet1d_tiny")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--backend", choices=["x86", "qnnpack"], default="x86", help="qnnpack for ARM (Pi)")
    ap.add_argument("--log_dir", type=str, default="results")
    ap.add_argument("--checkpoint", type=str, default=None, help="default: deploy checkpoint from train.py")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--max_batches", type=int, default=None, help="limit batches per epoch (smoke tests)")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root) / args.dataset
    meta, n_cls, in_ch, n_samples = dataset_info(data_root)
    split = deploy_split(args.log_dir, args.dataset, args.model)
    fp32 = load_model(args.model, n_cls, in_ch)
    fp32.load_state_dict(torch.load(args.checkpoint or split["checkpoint"], map_location="cpu"))
    train_loader, test_loader = loaders(data_root, split, args.batch_size)

    int8 = quantize_qat(fp32, torch.randn(2, in_ch, n_samples), train_loader, args.epochs, args.lr,
                        args.backend, args.max_batches)
    res = {"dataset": args.dataset, "model": args.model, "fold": split["fold"],
           "test_subjects": split["test"], "synthetic_data": bool(meta.get("synthetic")),
           "fp32_macro_f1": macro_f1(fp32, test_loader, n_cls),
           "int8_qat_macro_f1": macro_f1(int8, test_loader, n_cls), "backend": args.backend}
    out = Path(args.output or f"artifacts/{args.dataset}_{args.model}_qat_int8.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch.jit is deprecated in newer torch
        torch.jit.save(torch.jit.trace(int8, torch.randn(1, in_ch, n_samples)), str(out))
    res["output"] = str(out)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.log_dir) / f"{args.dataset}_{args.model}_qat.json").write_text(json.dumps(res, indent=2))
    print(f"fold {split['fold']} test macro-F1: FP32 {res['fp32_macro_f1']:.3f} -> "
          f"INT8 QAT {res['int8_qat_macro_f1']:.3f}; saved {out}")
    return res


if __name__ == "__main__":
    main()
