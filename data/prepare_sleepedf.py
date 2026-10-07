#!/usr/bin/env python3
"""Prepare Sleep-EDF Expanded (sleep-cassette) for training.

Either point --raw_dir at a folder that already holds the PhysioNet EDF files
(SC4xxxE0-PSG.edf + SC4xxx??-Hypnogram.edf), or pass --download to fetch them
through MNE (needs internet access to physionet.org).

Output (see neuraledge/dataio.py): one subjectNN_x.npy / subjectNN_y.npy pair per
subject (both nights concatenated), meta.json, meta.csv, plus an unlabelled INT8
calibration subset in --calib_out.

Examples
  # Sleep-EDF-20 equivalent: SC subjects 0-19, both nights
  python data/prepare_sleepedf.py --download --subjects 0-19 --out data/processed/sleepedf
  # files already on disk
  python data/prepare_sleepedf.py --raw_dir data/raw/sleep-cassette --out data/processed/sleepedf
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuraledge import sleepedf  # noqa: E402
from neuraledge.dataio import write_meta, write_subject  # noqa: E402


def parse_subjects(spec: str | None) -> list[int] | None:
    """'0-19' -> [0..19], '0,3,5-7' -> [0,3,5,6,7], None -> None (all found)."""
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw_dir", type=str, help="folder with Sleep-EDF SC .edf files")
    src.add_argument("--download", action="store_true", help="download via mne.datasets.sleep_physionet")
    p.add_argument("--mne_path", type=str, default=None, help="MNE data folder for --download")
    p.add_argument("--subjects", type=str, default=None, help="e.g. 0-19 (default: all found / 0-19 for --download)")
    p.add_argument("--nights", type=int, nargs="+", default=[1, 2])
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--calib_out", type=str, default=None)
    p.add_argument("--calib_n", type=int, default=512)
    p.add_argument("--channel", type=str, default=sleepedf.DEFAULT_CHANNEL)
    p.add_argument("--fs_out", type=float, default=100.0)
    p.add_argument("--edge_minutes", type=float, default=30.0, help="wake kept before/after sleep")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    subjects = parse_subjects(args.subjects)
    if args.download:
        recs = sleepedf.fetch_with_mne(subjects if subjects is not None else list(range(20)),
                                       args.nights, args.mne_path)
    else:
        recs = sleepedf.find_recordings(Path(args.raw_dir), subjects)
    recs = [r for r in recs if r.night in args.nights]
    if not recs:
        raise SystemExit("no Sleep-EDF recordings found")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subject = defaultdict(lambda: ([], []))
    for r in recs:
        x, y = sleepedf.load_recording(r, args.channel, args.fs_out, args.edge_minutes)
        counts = np.bincount(y, minlength=len(sleepedf.CLASSES))
        print(f"subject {r.subject:02d} night {r.night}: {len(y)} epochs  "
              + " ".join(f"{c}={n}" for c, n in zip(sleepedf.CLASSES, counts)))
        per_subject[r.subject][0].append(x)
        per_subject[r.subject][1].append(y)

    subject_ids = sorted(per_subject)
    n_epochs = {}
    for s in subject_ids:
        xs, ys = per_subject[s]
        write_subject(out_dir, s, np.concatenate(xs), np.concatenate(ys))
        n_epochs[s] = sum(len(y) for y in ys)

    meta = write_meta(
        out_dir, dataset="sleepedf", fs=args.fs_out, window_sec=sleepedf.EPOCH_SEC,
        channels=[args.channel], classes=sleepedf.CLASSES, subjects=subject_ids, synthetic=False,
        source="Sleep-EDF Expanded, sleep-cassette (PhysioNet)",
        extra={"recordings": [{"subject": r.subject, "night": r.night, "psg": r.psg.name,
                               "hypnogram": r.hypnogram.name} for r in recs],
               "edge_minutes": args.edge_minutes,
               "label_map": "W,N1,N2,N3(=S3+S4),REM; movement/unknown dropped",
               "normalization": "z-score per recording"})

    if args.calib_out:
        # Unlabelled inputs for INT8 calibration. Drawn from all subjects; for a strict
        # protocol, regenerate from the training subjects of the fold you deploy.
        rng = np.random.default_rng(args.seed)
        pool = [(s, i) for s in subject_ids for i in range(n_epochs[s])]
        pick = rng.choice(len(pool), size=min(args.calib_n, len(pool)), replace=False)
        x_mm = {s: np.load(out_dir / f"subject{s:02d}_x.npy", mmap_mode="r") for s in subject_ids}
        calib = np.stack([x_mm[pool[k][0]][pool[k][1]] for k in sorted(pick)])
        Path(args.calib_out).mkdir(parents=True, exist_ok=True)
        np.save(Path(args.calib_out) / "sleepedf_calib_x.npy", calib.astype(np.float32))

    total = sum(n_epochs.values())
    print(f"Wrote {len(subject_ids)} subjects / {total} epochs to {out_dir} (classes {meta['classes']})")


if __name__ == "__main__":
    main()
