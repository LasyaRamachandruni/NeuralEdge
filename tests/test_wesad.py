"""WESAD windowing: three signals stay three channels, windows never straddle labels."""
import json
import pickle

import numpy as np

from neuraledge import wesad


def fake_subject(seconds=60, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.zeros(int(seconds * 700), dtype=np.int64)
    labels[: 20 * 700] = 1            # baseline 0-20 s
    labels[25 * 700: 45 * 700] = 2    # stress 25-45 s (20-25 s transient)
    labels[45 * 700:] = 3             # amusement
    wrist = {name: rng.normal(size=(int(seconds * fs), 1)) for name, fs in wesad.SIGNALS["wrist"].items()}
    chest = {name: rng.normal(size=(int(seconds * fs), 1)) for name, fs in wesad.SIGNALS["chest"].items()}
    return {"signal": {"wrist": wrist, "chest": chest}, "label": labels, "subject": "S2"}


def test_window_subject_keeps_channels_separate_and_labels_pure():
    d = fake_subject()
    signals = {k: (v.ravel(), wesad.SIGNALS["wrist"][k]) for k, v in d["signal"]["wrist"].items()}
    x, y = wesad.window_subject(signals, d["label"], fs_out=32, window_sec=8, step_sec=4,
                                label_map={1: 0, 2: 1})
    assert x.shape == (7, 3, 256)
    # baseline windows start at 0,4,8,12 s; stress windows at 28,32,36 s
    assert y.tolist() == [0, 0, 0, 0, 1, 1, 1]


def test_three_class_mode_adds_amusement():
    d = fake_subject()
    signals = {k: (v.ravel(), wesad.SIGNALS["chest"][k]) for k, v in d["signal"]["chest"].items()}
    _, y = wesad.window_subject(signals, d["label"], fs_out=32, window_sec=8, step_sec=4,
                                label_map=wesad.LABEL_SETS["three"][0])
    assert set(y.tolist()) == {0, 1, 2}


def test_resample_lengths():
    assert len(wesad.resample(np.zeros(40), 4, 32)) == 320
    assert len(wesad.resample(np.zeros(7000), 700, 32)) == 320


def test_prepare_wesad_from_pickles(tmp_path):
    from data.prepare_wesad import main

    raw = tmp_path / "WESAD"
    for s in (2, 3, 4):
        (raw / f"S{s}").mkdir(parents=True)
        with open(raw / f"S{s}" / f"S{s}.pkl", "wb") as f:
            pickle.dump(fake_subject(seed=s), f)
    out = tmp_path / "wesad"
    main(["--raw_dir", str(raw), "--out", str(out), "--calib_out", str(tmp_path / "calib"), "--calib_n", "5"])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["subjects"] == [2, 3, 4]
    assert meta["channels"] == ["BVP", "EDA", "TEMP"]
    assert np.load(out / "subject02_x.npy").shape == (7, 3, 256)
    assert np.load(tmp_path / "calib" / "wesad_calib_x.npy").shape == (5, 3, 256)
