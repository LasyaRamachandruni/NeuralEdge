#!/usr/bin/env python3
"""Write a tiny SYNTHETIC dataset in the processed format, for tests/CI and smoke runs.

The output is flagged ``"synthetic": true`` in meta.json and train.py warns loudly
when it sees it. Use data/prepare_sleepedf.py or data/prepare_wesad.py for real data.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuraledge.synthetic import make_synthetic  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["sleepedf", "wesad"], default="sleepedf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--windows_per_subject", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = make_synthetic(Path(args.out), args.dataset, args.subjects, args.windows_per_subject, args.seed)
    print(f"Wrote SYNTHETIC {args.dataset} data ({len(meta['subjects'])} subjects) to {args.out}")


if __name__ == "__main__":
    main()
