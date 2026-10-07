#!/usr/bin/env python3
"""Prepare WESAD for training.

WESAD is distributed as a zip after accepting the dataset terms (Uni Siegen);
unpack it and pass --raw_dir pointing at the WESAD/ folder (containing S2/S2.pkl ...).

Each window is (3 channels x window*fs_out samples): wrist BVP/EDA/TEMP by default,
or chest ECG/EDA/Temp with --source chest. Use leave-one-subject-out CV
(train.py --folds 0) for this dataset.

Example
  python data/prepare_wesad.py --raw_dir data/raw/WESAD --out data/processed/wesad --calib_out data/calib/wesad
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuraledge import wesad  # noqa: E402
from neuraledge.dataio import write_meta, write_subject  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--calib_out", type=str, default=None)
    p.add_argument("--calib_n", type=int, default=512)
    p.add_argument("--source", choices=sorted(wesad.SIGNALS), default="wrist")
    p.add_argument("--labels", choices=sorted(wesad.LABEL_SETS), default="binary")
    p.add_argument("--window", type=float, default=8.0, help="window length in seconds")
    p.add_argument("--step", type=float, default=4.0, help="hop between windows in seconds")
    p.add_argument("--fs_out", type=float, default=32.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    found = wesad.find_subjects(Path(args.raw_dir))
    if not found:
        raise SystemExit(f"no S<k>.pkl files under {args.raw_dir}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, classes = wesad.LABEL_SETS[args.labels]
    subjects = []
    for s, path in sorted(found.items()):
        x, y = wesad.load_subject(path, args.source, args.fs_out, args.window, args.step, args.labels)
        if len(y) == 0:
            print(f"S{s}: no usable windows, skipping")
            continue
        write_subject(out_dir, s, x, y)
        subjects.append(s)
        print(f"S{s}: {len(y)} windows " + " ".join(f"{c}={n}" for c, n in
                                                     zip(classes, np.bincount(y, minlength=len(classes)))))
    write_meta(out_dir, dataset="wesad", fs=args.fs_out, window_sec=args.window,
               channels=list(wesad.SIGNALS[args.source]), classes=classes, subjects=subjects,
               synthetic=False, source=f"WESAD ({args.source} signals)",
               extra={"step_sec": args.step, "normalization": "z-score per subject per channel"})

    if args.calib_out:
        rng = np.random.default_rng(args.seed)
        xs = [np.load(out_dir / f"subject{s:02d}_x.npy", mmap_mode="r") for s in subjects]
        pool = [(k, i) for k, x in enumerate(xs) for i in range(len(x))]
        pick = sorted(rng.choice(len(pool), size=min(args.calib_n, len(pool)), replace=False))
        calib = np.stack([xs[pool[j][0]][pool[j][1]] for j in pick]).astype(np.float32)
        Path(args.calib_out).mkdir(parents=True, exist_ok=True)
        np.save(Path(args.calib_out) / "wesad_calib_x.npy", calib)
    print(f"Wrote {len(subjects)} subjects to {out_dir}")


if __name__ == "__main__":
    main()
