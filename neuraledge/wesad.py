"""WESAD loader (Schmidt et al., ICMI 2018).

Reads the per-subject pickles (``WESAD/S<k>/S<k>.pkl``), keeps three signals as
three input *channels* (not concatenated into one long vector), resamples them to
a common rate, z-scores each channel per subject and cuts fixed windows.

Signal sets
  wrist (Empatica E4): BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz
  chest (RespiBAN):    ECG, EDA, Temp, all 700 Hz

Labels in the pickle are sampled at 700 Hz: 0 transient, 1 baseline, 2 stress,
3 amusement, 4 meditation, 5-7 not to be used. A window is kept only if every
label sample inside it is the same, allowed class.
"""
from __future__ import annotations

import pickle
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

LABEL_FS = 700.0
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]  # S1 and S12 are not in the release
SIGNALS = {
    "wrist": {"BVP": 64.0, "EDA": 4.0, "TEMP": 4.0},
    "chest": {"ECG": 700.0, "EDA": 700.0, "Temp": 700.0},
}
LABEL_SETS = {
    "binary": ({1: 0, 2: 1}, ["baseline", "stress"]),
    "three": ({1: 0, 2: 1, 3: 2}, ["baseline", "stress", "amusement"]),
}


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, dtype=np.float64)
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=np.float64), ratio.numerator, ratio.denominator)


def window_subject(signals: dict[str, tuple[np.ndarray, float]], labels: np.ndarray, *,
                   fs_out: float, window_sec: float, step_sec: float, label_map: dict[int, int],
                   label_fs: float = LABEL_FS, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """signals: name -> (1-D array, sampling rate). Returns x (n, channels, samples), y (n,)."""
    chans = [resample(np.ravel(sig), fs, fs_out) for sig, fs in signals.values()]
    if normalize:
        chans = [(c - c.mean()) / (c.std() + 1e-8) for c in chans]
    duration = min(min(len(c) for c in chans) / fs_out, len(labels) / label_fs)
    win, step = int(round(window_sec * fs_out)), step_sec
    xs, ys = [], []
    t = 0.0
    while t + window_sec <= duration + 1e-9:
        lab = labels[int(round(t * label_fs)): int(round((t + window_sec) * label_fs))]
        first = int(lab[0]) if len(lab) else -1
        if first in label_map and np.all(lab == first):
            i = int(round(t * fs_out))
            xs.append(np.stack([c[i: i + win] for c in chans]))
            ys.append(label_map[first])
        t += step
    if not xs:
        return np.zeros((0, len(chans), win), np.float32), np.zeros(0, np.int64)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")  # written with Python 2


def load_subject(path: Path, source: str = "wrist", fs_out: float = 32.0, window_sec: float = 8.0,
                 step_sec: float = 4.0, labels: str = "binary") -> tuple[np.ndarray, np.ndarray]:
    d = load_pickle(path)
    sig = d["signal"][source]
    signals = {name: (np.asarray(sig[name]).ravel(), fs) for name, fs in SIGNALS[source].items()}
    label_map, _ = LABEL_SETS[labels]
    return window_subject(signals, np.asarray(d["label"]).ravel(), fs_out=fs_out, window_sec=window_sec,
                          step_sec=step_sec, label_map=label_map)


def find_subjects(raw_dir: Path) -> dict[int, Path]:
    found = {}
    for p in sorted(Path(raw_dir).rglob("S*.pkl")):
        stem = p.stem[1:]
        if stem.isdigit():
            found[int(stem)] = p
    return found
