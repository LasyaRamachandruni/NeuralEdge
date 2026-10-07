#!/usr/bin/env python3
"""Collect a Sleep-EDF run into results/sleepedf_results.json + a paste-ready Markdown table.

Reads the CV summary written by train.py, the dataset meta.json and (optionally) the
ONNX CPU latency JSON written by bench.py. Also saves the pooled confusion matrix as
a PNG if matplotlib is available.
"""
import argparse
import json
import platform
import subprocess
from pathlib import Path

REFERENCE = {
    "name": "DeepSleepNet",
    "citation": ("A. Supratak, H. Dong, C. Wu, Y. Guo, \"DeepSleepNet: a Model for Automatic Sleep Stage "
                 "Scoring based on Raw Single-Channel EEG\", IEEE Trans. Neural Syst. Rehabil. Eng. 25(11), "
                 "2017. arXiv:1703.04046"),
    "url": "https://arxiv.org/abs/1703.04046",
    "setting": "Sleep-EDF (2013 release, 20 subjects), Fpz-Cz, 20-fold subject-wise CV, CNN + BiLSTM sequence model",
    "accuracy": 0.820,
    "macro_f1": 0.769,
    "note": ("Not a like-for-like comparison: DeepSleepNet uses the 2013 Sleep-EDF release and models "
             "sequences of epochs; this repo classifies each 30-s epoch independently with a much smaller CNN."),
}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def gpu_name() -> str | None:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def plot_confusion(cm, classes, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (hypnogram)")
    ax.set_title("Pooled test-fold confusion (row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet1d_tiny")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--data_dir", default="data/processed/sleepedf")
    ap.add_argument("--latency_json", default=None)
    args = ap.parse_args(argv)

    res_dir = Path(args.results_dir)
    cv = json.loads((res_dir / f"sleepedf_{args.model}_cv.json").read_text())
    meta = json.loads((Path(args.data_dir) / "meta.json").read_text())
    lat = json.loads(Path(args.latency_json).read_text()) if args.latency_json else None
    pooled = cv["pooled"]

    out = {
        "synthetic_data": cv["synthetic_data"],
        "dataset": {"name": meta["source"], "subjects": meta["subjects"], "n_subjects": len(meta["subjects"]),
                    "channel": meta["channels"], "classes": meta["classes"], "fs": meta["fs"],
                    "edge_minutes": meta.get("edge_minutes"), "n_recordings": len(meta.get("recordings", []))},
        "model": args.model,
        "cv_scheme": cv["cv"]["scheme"],
        "n_folds_run": cv["cv"]["n_folds_run"],
        "macro_f1_mean": cv["macro_f1_mean"],
        "macro_f1_std": cv["macro_f1_std"],
        "pooled_macro_f1": pooled["macro_f1"],
        "pooled_accuracy": pooled["accuracy"],
        "pooled_cohen_kappa": pooled["cohen_kappa"],
        "per_class_f1": pooled["per_class_f1"],
        "confusion_matrix": {"labels": cv["classes"], "rows_true_cols_pred": pooled["confusion_matrix"]},
        "n_test_epochs": pooled["n"],
        "per_fold_macro_f1": [f["test"]["macro_f1"] for f in cv["folds"]],
        "onnx_cpu_latency": None if lat is None else {k: lat[k] for k in (
            "cpu", "runtime", "threads", "batch", "input_shape", "iters", "p50_ms", "p90_ms", "p99_ms", "model_bytes")},
        "reference": REFERENCE,
        "environment": {**cv["environment"], "gpu": gpu_name(), "git_commit": git_commit(),
                        "python": platform.python_version()},
        "train_config": {k: cv["config"][k] for k in ("epochs", "batch_size", "lr", "early_stop_patience",
                                                     "folds", "val_frac", "class_weights", "seed")},
    }
    (res_dir / "sleepedf_results.json").write_text(json.dumps(out, indent=2))

    pc = out["per_class_f1"]
    lat_txt = f"{lat['p50_ms']:.2f} ms" if lat else "n/a"
    warn = "\n> **SYNTHETIC DATA - not a result.**\n" if out["synthetic_data"] else ""
    subj = meta["subjects"]
    subj_txt = (f"{subj[0]}-{subj[-1]}" if subj == list(range(subj[0], subj[-1] + 1))
                else ", ".join(map(str, subj)))
    folds_txt = out["cv_scheme"] + (f", first {out['n_folds_run']} folds only"
                                    if cv["config"].get("max_folds") else "")
    md = [
        warn,
        f"Sleep-EDF Expanded (sleep-cassette), subjects {subj_txt} ({len(subj)} subjects, "
        f"{pooled['n']} scored 30-s test epochs), Fpz-Cz, {folds_txt} CV.",
        "",
        "| Model | Macro-F1 (mean ± std over folds) | Pooled acc. | Cohen's κ | W | N1 | N2 | N3 | REM | ONNX CPU p50 (batch 1) |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| `{args.model}` (this repo) | {out['macro_f1_mean']:.3f} ± {out['macro_f1_std']:.3f} | "
        f"{out['pooled_accuracy']:.3f} | {out['pooled_cohen_kappa']:.3f} | "
        + " | ".join(f"{pc[c]:.3f}" for c in ["W", "N1", "N2", "N3", "REM"]) + f" | {lat_txt} |",
        f"| DeepSleepNet (Supratak et al., 2017) [ref] | {REFERENCE['macro_f1']:.3f} (reported) | "
        f"{REFERENCE['accuracy']:.3f} | - | - | - | - | - | - | - |",
        "",
        f"Per-class F1 is computed on all test folds pooled. Latency: {lat['runtime']} on {lat['cpu']}, "
        f"{lat['threads']} thread(s) - a cloud CPU, not an edge device." if lat else "Latency not measured.",
        "",
        f"[ref] {REFERENCE['citation']}. {REFERENCE['note']}",
        "",
    ]
    (res_dir / "sleepedf_results.md").write_text("\n".join(md))
    if plot_confusion(pooled["confusion_matrix"], cv["classes"], res_dir / "sleepedf_confusion.png"):
        print("saved", res_dir / "sleepedf_confusion.png")
    print("\n".join(md))
    return out


if __name__ == "__main__":
    main()
