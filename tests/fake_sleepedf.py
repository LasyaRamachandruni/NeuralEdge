"""Write tiny EDF files that mimic the Sleep-EDF sleep-cassette layout (for tests)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FS = 100.0
# (description, number of 30-s epochs)
HYPNOGRAM = [
    ("Sleep stage W", 80),
    ("Sleep stage 1", 2),
    ("Sleep stage 2", 6),
    ("Sleep stage 3", 3),
    ("Sleep stage 4", 2),
    ("Movement time", 1),
    ("Sleep stage R", 4),
    ("Sleep stage 2", 2),
    ("Sleep stage W", 70),
    ("Sleep stage ?", 10),
]
N_EPOCHS = sum(n for _, n in HYPNOGRAM)


def write_fake_recording(folder: Path, subject: int = 0, night: int = 1, seed: int = 0) -> tuple[Path, Path]:
    import mne

    folder.mkdir(parents=True, exist_ok=True)
    meas_date = datetime(1989, 4, 24, 16, 13, tzinfo=timezone.utc)
    rng = np.random.default_rng(seed)
    n = int(N_EPOCHS * 30 * FS)
    data = rng.normal(0, 20e-6, size=(2, n))  # volts
    info = mne.create_info(["EEG Fpz-Cz", "EEG Pz-Oz"], FS, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(meas_date)
    stem = f"SC4{subject:02d}{night}"
    psg = folder / f"{stem}E0-PSG.edf"
    mne.export.export_raw(psg, raw, fmt="edf", overwrite=True, verbose="error")

    onsets, durations, desc, t = [], [], [], 0.0
    for d, k in HYPNOGRAM:
        onsets.append(t)
        durations.append(k * 30.0)
        desc.append(d)
        t += k * 30.0
    hyp_raw = mne.io.RawArray(np.zeros((1, int(t))), mne.create_info(["dummy"], 1.0, "misc"), verbose="error")
    hyp_raw.set_meas_date(meas_date)
    hyp_raw.set_annotations(mne.Annotations(onsets, durations, desc, orig_time=meas_date))
    hyp = folder / f"{stem}EC-Hypnogram.edf"
    mne.export.export_raw(hyp, hyp_raw, fmt="edf", overwrite=True, verbose="error")
    return psg, hyp
