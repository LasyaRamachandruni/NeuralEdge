#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

"""
Prepare WESAD: normalize per subject, window 4–8 s, create binary labels
(stress vs. baseline). This is a scaffold producing synthetic placeholders.
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--calib_out", type=str, required=True)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--fs_out", type=int, default=32)
    p.add_argument("--download", type=str, default="false")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    calib_dir = Path(args.calib_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    num_subjects = 10
    samples_per_window = args.window * args.fs_out
    feature_dim = samples_per_window * 3  # PPG, EDA, Temp
    meta = []
    for subj in range(num_subjects):
        num_windows = 600
        x = rng.standard_normal((num_windows, feature_dim)).astype(np.float32)
        y = rng.integers(0, 2, size=(num_windows,), endpoint=False).astype(np.int64)
        np.save(out_dir / f"subject{subj:02d}_x.npy", x)
        np.save(out_dir / f"subject{subj:02d}_y.npy", y)
        meta.append({"subject": subj, "num_windows": num_windows})

    calib = rng.standard_normal((1024, feature_dim)).astype(np.float32)
    np.save(calib_dir / "wesad_calib_x.npy", calib)
    pd.DataFrame(meta).to_csv(out_dir / "meta.csv", index=False)
    print(f"Wrote synthetic WESAD placeholders into {out_dir}")


if __name__ == "__main__":
    main()

