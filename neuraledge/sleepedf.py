"""Sleep-EDF Expanded (sleep-cassette, "SC") loader.

Follows the usual single-channel protocol from the sleep-staging literature
(e.g. DeepSleepNet, Supratak et al. 2017):

* EEG Fpz-Cz channel (100 Hz in the SC recordings)
* 30-s epochs scored with the R&K hypnogram, mapped to 5 AASM classes:
  W, N1, N2, N3 (= R&K stages 3 + 4), REM
* "Movement time" and "Sleep stage ?" epochs are dropped
* long wake periods are trimmed to 30 min before the first and after the last
  sleep epoch (recordings are ~20 h, mostly daytime wake)
* each recording is z-scored on its kept epochs

Files are matched by the PhysioNet naming scheme::

    SC4<ss><n>E0-PSG.edf        ss = subject (00-82), n = night (1/2)
    SC4<ss><n>??-Hypnogram.edf  e.g. SC4001EC-Hypnogram.edf

Everything that is not file IO is a plain numpy function so it can be unit tested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPOCH_SEC = 30.0
CLASSES = ["W", "N1", "N2", "N3", "REM"]
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # R&K S3 + S4 -> AASM N3
    "Sleep stage R": 4,
}
UNKNOWN = -1  # "Sleep stage ?", "Movement time", unscored
DEFAULT_CHANNEL = "EEG Fpz-Cz"

_PSG_RE = re.compile(r"^SC4(\d{2})(\d)E0-PSG\.edf$", re.IGNORECASE)


@dataclass
class Recording:
    subject: int
    night: int
    psg: Path
    hypnogram: Path


def hypnogram_to_epoch_labels(onsets, durations, descriptions, n_epochs: int,
                              epoch_sec: float = EPOCH_SEC) -> np.ndarray:
    """Expand hypnogram annotations (seconds from recording start) into one label per epoch.

    Epochs not covered by a known sleep-stage annotation get UNKNOWN.
    """
    labels = np.full(n_epochs, UNKNOWN, dtype=np.int64)
    for onset, dur, desc in zip(onsets, durations, descriptions):
        stage = STAGE_MAP.get(str(desc).strip(), UNKNOWN)
        start = int(round(float(onset) / epoch_sec))
        stop = start + int(round(float(dur) / epoch_sec))
        start, stop = max(start, 0), min(stop, n_epochs)
        if stop > start:
            labels[start:stop] = stage
    return labels


def wake_trim_bounds(labels: np.ndarray, edge_minutes: float = 30.0,
                     epoch_sec: float = EPOCH_SEC) -> tuple[int, int]:
    """[start, stop) epoch range keeping `edge_minutes` of data around the sleep period.

    Returns (0, 0) if the recording contains no sleep epochs.
    """
    sleep = np.flatnonzero((labels >= 1) & (labels <= 4))
    if sleep.size == 0:
        return 0, 0
    edge = int(round(edge_minutes * 60 / epoch_sec))
    return max(0, int(sleep[0]) - edge), min(len(labels), int(sleep[-1]) + 1 + edge)


def epoch_and_label(signal: np.ndarray, fs: float, labels: np.ndarray,
                    edge_minutes: float = 30.0, normalize: bool = True,
                    epoch_sec: float = EPOCH_SEC) -> tuple[np.ndarray, np.ndarray]:
    """Cut a 1-D signal into labelled epochs, trim wake, drop unknown epochs.

    Returns x of shape (n, 1, epoch_sec * fs) float32 and y of shape (n,) int64.
    """
    spe = int(round(epoch_sec * fs))
    n = min(len(labels), len(signal) // spe)
    labels = labels[:n]
    start, stop = wake_trim_bounds(labels, edge_minutes, epoch_sec)
    x = signal[: n * spe].reshape(n, spe)[start:stop]
    y = labels[start:stop]
    keep = y != UNKNOWN
    x, y = x[keep].astype(np.float32), y[keep]
    if normalize and len(x):
        x = (x - x.mean()) / (x.std() + 1e-8)
    return x[:, None, :].astype(np.float32), y.astype(np.int64)


def find_recordings(raw_dir: Path, subjects: list[int] | None = None) -> list[Recording]:
    """Pair PSG and hypnogram files found anywhere under raw_dir."""
    raw_dir = Path(raw_dir)
    hyps = {p.name[:7].upper(): p for p in raw_dir.rglob("*-Hypnogram.edf")}
    recs = []
    for psg in sorted(raw_dir.rglob("*-PSG.edf")):
        m = _PSG_RE.match(psg.name)
        if not m:
            continue
        subj, night = int(m.group(1)), int(m.group(2))
        if subjects is not None and subj not in subjects:
            continue
        hyp = hyps.get(psg.name[:7].upper())
        if hyp is None:
            print(f"warning: no hypnogram for {psg.name}, skipping")
            continue
        recs.append(Recording(subj, night, psg, hyp))
    return recs


def fetch_with_mne(subjects: list[int], recordings=(1, 2), path: str | None = None) -> list[Recording]:
    """Download sleep-cassette recordings through mne.datasets.sleep_physionet.age."""
    from mne.datasets.sleep_physionet.age import fetch_data

    kwargs = {"subjects": list(subjects), "recording": list(recordings), "path": path}
    try:
        files = fetch_data(on_missing="warn", **kwargs)  # some subjects lack a night
    except TypeError:  # older MNE without on_missing
        files = fetch_data(**kwargs)
    recs = []
    for psg, hyp in files:
        m = _PSG_RE.match(Path(psg).name)
        if m:
            recs.append(Recording(int(m.group(1)), int(m.group(2)), Path(psg), Path(hyp)))
    return recs


def load_recording(rec: Recording, channel: str = DEFAULT_CHANNEL, fs_out: float = 100.0,
                   edge_minutes: float = 30.0, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    import mne

    raw = mne.io.read_raw_edf(rec.psg, preload=False, verbose="error")
    if channel not in raw.ch_names:
        raise KeyError(f"{channel!r} not in {rec.psg.name}: {raw.ch_names}")
    raw.pick([channel]).load_data(verbose="error")
    if abs(raw.info["sfreq"] - fs_out) > 1e-6:
        raw.resample(fs_out, verbose="error")
    signal = raw.get_data()[0] * 1e6  # volts -> microvolts

    annot = mne.read_annotations(rec.hypnogram)
    offset = 0.0
    meas_date = raw.info.get("meas_date")
    if annot.orig_time is not None and meas_date is not None:
        offset = (annot.orig_time - meas_date).total_seconds()
    n_epochs = int(len(signal) // (EPOCH_SEC * fs_out))
    labels = hypnogram_to_epoch_labels(annot.onset + offset, annot.duration, annot.description, n_epochs)
    return epoch_and_label(signal, fs_out, labels, edge_minutes, normalize)
