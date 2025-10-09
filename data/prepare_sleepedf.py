#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
Prepare Sleep-EDF: download (optional), extract single-channel EEG/EOG, resample,
z-score per recording, window into 30 s segments at 100 Hz, and create
leave-one-subject-out splits. Also write a calibration subset for INT8.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--calib_out", type=str, required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--fs_out", type=int, default=100)
    parser.add_argument("--download", type=str, default="false")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    calib_dir = Path(args.calib_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder: in a full implementation, actually parse EDFs.
    # Here we scaffold expected files for the pipeline to run.
    meta = []
    num_subjects = 5
    samples_per_epoch = args.window * args.fs_out
    rng = np.random.default_rng(42)
    for subj in range(num_subjects):
        num_epochs = 200
        x = rng.standard_normal((num_epochs, samples_per_epoch)).astype(np.float32)
        # labels: 0=W,1=NREM,2=REM
        y = rng.integers(0, 3, size=(num_epochs,), endpoint=False)
        np.save(out_dir / f"subject{subj:02d}_x.npy", x)
        np.save(out_dir / f"subject{subj:02d}_y.npy", y)
        meta.append({"subject": subj, "num_epochs": num_epochs})

    # Write a tiny calibration set
    calib = rng.standard_normal((1024, samples_per_epoch)).astype(np.float32)
    np.save(calib_dir / "sleepedf_calib_x.npy", calib)

    pd.DataFrame(meta).to_csv(out_dir / "meta.csv", index=False)
    print(f"Wrote synthetic Sleep-EDF placeholders into {out_dir}")


if __name__ == "__main__":
    main()

