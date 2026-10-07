"""Synthetic stand-in data for tests and CI ONLY.

Nothing here resembles real EEG or wearable physiology. Each class is a noisy
sinusoid at its own frequency so that a model can learn *something* and tests can
check that the training loop actually reduces the loss. Never report metrics
computed on this data as results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from neuraledge.dataio import write_meta, write_subject

SLEEP_CLASSES = ["W", "N1", "N2", "N3", "REM"]
WESAD_CLASSES = ["baseline", "stress"]


def _class_signal(rng, label: int, n_channels: int, n_samples: int, fs: float, base_hz: float):
    t = np.arange(n_samples) / fs
    freq = base_hz * (label + 1)
    phase = rng.uniform(0, 2 * np.pi, size=(n_channels, 1))
    sig = np.sin(2 * np.pi * freq * t[None, :] + phase)
    return (sig + rng.normal(0, 1.0, size=(n_channels, n_samples))).astype(np.float32)


def make_synthetic(out: Path, dataset: str = "sleepedf", n_subjects: int = 4,
                   windows_per_subject: int = 40, seed: int = 0,
                   fs: float | None = None, window_sec: float | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if dataset == "sleepedf":
        classes, channels = SLEEP_CLASSES, ["synthetic_eeg"]
        fs, window_sec, base_hz = fs or 100.0, window_sec or 30.0, 1.5
    elif dataset == "wesad":
        classes, channels = WESAD_CLASSES, ["synthetic_bvp", "synthetic_eda", "synthetic_temp"]
        fs, window_sec, base_hz = fs or 32.0, window_sec or 8.0, 1.0
    else:
        raise ValueError(dataset)
    n_samples = int(round(fs * window_sec))
    subjects = list(range(n_subjects))
    for s in subjects:
        y = np.arange(windows_per_subject) % len(classes)
        rng.shuffle(y)
        x = np.stack([_class_signal(rng, int(c), len(channels), n_samples, fs, base_hz) for c in y])
        write_subject(out, s, x, y)
    return write_meta(out, dataset=dataset, fs=fs, window_sec=window_sec, channels=channels,
                      classes=classes, subjects=subjects, synthetic=True,
                      source="neuraledge.synthetic (random test data, not physiological)")
