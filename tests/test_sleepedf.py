"""Sleep-EDF loader: label mapping, wake trimming, and an end-to-end run on a fake EDF pair."""
import json

import numpy as np
import pytest

from neuraledge import sleepedf
from neuraledge.sleepedf import UNKNOWN, epoch_and_label, hypnogram_to_epoch_labels, wake_trim_bounds


def test_stage_mapping_merges_s3_s4_and_drops_movement():
    labels = hypnogram_to_epoch_labels(
        onsets=[0, 30, 60, 90, 120, 150, 180, 210],
        durations=[30] * 8,
        descriptions=["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3",
                      "Sleep stage 4", "Sleep stage R", "Movement time", "Sleep stage ?"],
        n_epochs=9,
    )
    assert labels.tolist() == [0, 1, 2, 3, 3, 4, UNKNOWN, UNKNOWN, UNKNOWN]


def test_multi_epoch_annotation_expands():
    labels = hypnogram_to_epoch_labels([0, 90], [90, 60], ["Sleep stage W", "Sleep stage 2"], 5)
    assert labels.tolist() == [0, 0, 0, 2, 2]


def test_wake_trim_keeps_edge_minutes():
    labels = np.array([0] * 100 + [2] * 10 + [0] * 100)
    start, stop = wake_trim_bounds(labels, edge_minutes=30)  # 60 epochs either side
    assert (start, stop) == (40, 170)
    assert wake_trim_bounds(np.zeros(50, dtype=int)) == (0, 0)


def test_epoch_and_label_shapes_and_dropping():
    fs = 10.0
    labels = np.array([0, 1, UNKNOWN, 2, 4])
    signal = np.arange(len(labels) * 300, dtype=np.float64)
    x, y = epoch_and_label(signal, fs, labels, edge_minutes=30, normalize=False)
    assert x.shape == (4, 1, 300) and x.dtype == np.float32
    assert y.tolist() == [0, 1, 2, 4]
    # the dropped epoch is the third one, so the 3rd kept epoch starts at sample 900
    assert x[2, 0, 0] == 900


edf = pytest.importorskip("mne")
pytest.importorskip("edfio")
from fake_sleepedf import write_fake_recording  # noqa: E402


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sleep-cassette")
    write_fake_recording(d, subject=0, night=1, seed=0)
    write_fake_recording(d, subject=0, night=2, seed=1)
    write_fake_recording(d, subject=1, night=1, seed=2)
    write_fake_recording(d, subject=2, night=1, seed=3)
    return d


def test_load_fake_edf_recording(raw_dir):
    recs = sleepedf.find_recordings(raw_dir)
    assert [(r.subject, r.night) for r in recs] == [(0, 1), (0, 2), (1, 1), (2, 1)]
    x, y = sleepedf.load_recording(recs[0])
    # fake hypnogram: 80 W, sleep block of 20 epochs (1 movement), 70 W, 10 unknown
    # -> keep 60 W before, 19 scored sleep epochs, 60 W after
    assert x.shape == (139, 1, 3000)
    assert np.bincount(y, minlength=5).tolist() == [120, 2, 8, 5, 4]
    assert abs(float(x.mean())) < 1e-3 and abs(float(x.std()) - 1) < 1e-3


def test_prepare_script_writes_subject_wise_files(raw_dir, tmp_path):
    from data.prepare_sleepedf import main

    out = tmp_path / "sleepedf"
    main(["--raw_dir", str(raw_dir), "--out", str(out), "--calib_out", str(tmp_path / "calib"), "--calib_n", "16"])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["subjects"] == [0, 1, 2]
    assert meta["classes"] == ["W", "N1", "N2", "N3", "REM"]
    assert meta["channels"] == ["EEG Fpz-Cz"] and meta["synthetic"] is False
    x0, y0 = np.load(out / "subject00_x.npy"), np.load(out / "subject00_y.npy")
    assert x0.shape == (278, 1, 3000) and len(y0) == 278  # two nights concatenated
    assert np.load(tmp_path / "calib" / "sleepedf_calib_x.npy").shape == (16, 1, 3000)
