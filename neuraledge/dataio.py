"""On-disk format shared by every prepare script and every consumer.

A processed dataset directory looks like::

    <root>/meta.json            dataset description (see write_meta)
    <root>/meta.csv             one row per subject, for humans / spreadsheets
    <root>/subjectNN_x.npy      float32, shape (n_windows, n_channels, n_samples)
    <root>/subjectNN_y.npy      int64,   shape (n_windows,)

Arrays are plain .npy so they can be memory-mapped (np.load(..., mmap_mode="r")),
which keeps RAM use low for the full Sleep-EDF set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

META_JSON = "meta.json"
META_CSV = "meta.csv"


def subject_paths(root: Path, subject: int) -> tuple[Path, Path]:
    root = Path(root)
    return root / f"subject{subject:02d}_x.npy", root / f"subject{subject:02d}_y.npy"


def write_subject(root: Path, subject: int, x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must be (n, channels, samples), got {x.shape}")
    if len(x) != len(y):
        raise ValueError(f"x/y length mismatch: {len(x)} vs {len(y)}")
    xp, yp = subject_paths(root, subject)
    np.save(xp, np.ascontiguousarray(x, dtype=np.float32))
    np.save(yp, np.asarray(y, dtype=np.int64))


def write_meta(root: Path, *, dataset: str, fs: float, window_sec: float, channels: list[str],
               classes: list[str], subjects: list[int], synthetic: bool, source: str,
               extra: dict | None = None) -> dict:
    """Write meta.json + meta.csv. Per-subject class counts are read back from disk."""
    root = Path(root)
    rows = []
    for s in subjects:
        y = np.load(subject_paths(root, s)[1])
        counts = np.bincount(y, minlength=len(classes))
        rows.append({"subject": int(s), "n_windows": int(len(y)),
                     **{c: int(n) for c, n in zip(classes, counts)}})
    meta = {
        "dataset": dataset,
        "fs": fs,
        "window_sec": window_sec,
        "n_samples": int(round(fs * window_sec)),
        "channels": list(channels),
        "classes": list(classes),
        "subjects": [int(s) for s in subjects],
        "synthetic": bool(synthetic),
        "source": source,
        **(extra or {}),
    }
    (root / META_JSON).write_text(json.dumps(meta, indent=2))
    header = ["subject", "n_windows", *classes]
    lines = [",".join(header)] + [",".join(str(r[h]) for h in header) for r in rows]
    (root / META_CSV).write_text("\n".join(lines) + "\n")
    return meta


def load_meta(root: Path) -> dict:
    path = Path(root) / META_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run one of the data/prepare_*.py scripts (or "
            f"data/make_synthetic.py for a smoke test) first.")
    return json.loads(path.read_text())


def load_subject(root: Path, subject: int, mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    xp, yp = subject_paths(root, subject)
    x = np.load(xp, mmap_mode="r" if mmap else None)
    y = np.load(yp)
    return x, y


class SubjectWindowDataset(Dataset):
    """Windows from a list of subjects, read lazily from memory-mapped .npy files."""

    def __init__(self, root: Path, subjects: list[int]):
        self.arrays = []
        index = []
        for k, s in enumerate(subjects):
            x, y = load_subject(root, s)
            self.arrays.append((x, y))
            index.append(np.stack([np.full(len(y), k), np.arange(len(y))], axis=1))
        self.index = np.concatenate(index) if index else np.zeros((0, 2), dtype=np.int64)

    @property
    def labels(self) -> np.ndarray:
        return np.concatenate([y for _, y in self.arrays]) if self.arrays else np.zeros(0, np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i):
        k, j = self.index[i]
        x, y = self.arrays[k]
        return torch.from_numpy(np.array(x[j], dtype=np.float32)), torch.tensor(int(y[j]))
