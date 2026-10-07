#!/usr/bin/env python3
"""L1 structured (output-channel) pruning of Conv1d layers + fine-tuning.

Pruned channels are zeroed with torch.nn.utils.prune masks that stay applied during
fine-tuning (so they cannot grow back) and are then made permanent. The tensors keep
their shape: this measures the accuracy cost of removing channels, but it does not
make the ONNX model smaller or faster until the zeroed channels are physically
removed (not implemented yet).
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import prune as tprune

from neuraledge.finetune import dataset_info, deploy_split, finetune, loaders, macro_f1
from train import load_model


def apply_pruning(model: nn.Module, ratio: float) -> list[nn.Conv1d]:
    convs = [m for m in model.modules() if isinstance(m, nn.Conv1d) and m.groups == 1]
    for m in convs:
        tprune.ln_structured(m, name="weight", amount=ratio, n=1, dim=0)
    return convs


def make_permanent(convs: list[nn.Conv1d]) -> None:
    for m in convs:
        tprune.remove(m, "weight")


def zero_channel_fraction(model: nn.Module) -> float:
    tot = zero = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d) and m.groups == 1:
            norms = m.weight.detach().abs().sum(dim=(1, 2))
            tot += norms.numel()
            zero += int((norms == 0).sum())
    return zero / max(tot, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet1d_tiny")
    ap.add_argument("--prune_ratio", type=float, default=0.3)
    ap.add_argument("--finetune_epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--log_dir", type=str, default="results")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--max_batches", type=int, default=None)
    args = ap.parse_args(argv)

    data_root = Path(args.data_root) / args.dataset
    meta, n_cls, in_ch, _ = dataset_info(data_root)
    split = deploy_split(args.log_dir, args.dataset, args.model)
    fp32 = load_model(args.model, n_cls, in_ch)
    fp32.load_state_dict(torch.load(args.checkpoint or split["checkpoint"], map_location="cpu"))
    train_loader, test_loader = loaders(data_root, split, args.batch_size)

    pruned = copy.deepcopy(fp32)
    convs = apply_pruning(pruned, args.prune_ratio)
    finetune(pruned, train_loader, args.finetune_epochs, args.lr, args.max_batches)
    make_permanent(convs)

    res = {"dataset": args.dataset, "model": args.model, "fold": split["fold"], "test_subjects": split["test"],
           "synthetic_data": bool(meta.get("synthetic")), "prune_ratio": args.prune_ratio,
           "zeroed_channel_fraction": zero_channel_fraction(pruned),
           "fp32_macro_f1": macro_f1(fp32, test_loader, n_cls),
           "pruned_macro_f1": macro_f1(pruned, test_loader, n_cls)}
    out = Path(args.output or f"artifacts/{args.dataset}_{args.model}_pruned.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pruned.state_dict(), out)
    res["output"] = str(out)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.log_dir) / f"{args.dataset}_{args.model}_pruned.json").write_text(json.dumps(res, indent=2))
    print(f"fold {split['fold']} test macro-F1: FP32 {res['fp32_macro_f1']:.3f} -> pruned "
          f"{res['pruned_macro_f1']:.3f} ({res['zeroed_channel_fraction']:.0%} conv channels zeroed)")
    return res


if __name__ == "__main__":
    main()
