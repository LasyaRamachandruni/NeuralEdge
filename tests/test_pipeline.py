"""End-to-end smoke test on SYNTHETIC data: train (CV) -> export ONNX -> CPU latency -> QAT -> prune."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train
from neuraledge.synthetic import make_synthetic

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    make_synthetic(root / "data" / "sleepedf", "sleepedf", n_subjects=4, windows_per_subject=20, seed=0)
    args = train.build_parser().parse_args([
        "--data_root", str(root / "data"), "--epochs", "2", "--batch_size", "16", "--folds", "0",
        "--max_folds", "2", "--artifacts_dir", str(root / "artifacts"), "--log_dir", str(root / "results"),
        "--quiet"])
    res = train.run_cv(args)
    return SimpleNamespace(root=root, res=res, args=args)


def test_folds_are_subject_disjoint():
    folds = train.make_folds(list(range(20)), 5, seed=0)
    assert len(folds) == 5 and sorted(s for f in folds for s in f) == list(range(20))
    tr, va = train.split_train_val([s for s in range(20) if s not in folds[0]], 0.15, 0)
    assert not set(tr) & set(va) and not (set(tr) | set(va)) & set(folds[0])
    assert train.make_folds([3, 5, 9], 0, 0) == [[3], [5], [9]]


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    model = train.load_model("resnet1d_tiny", 5, 1)
    x, y = torch.randn(16, 1, 600), torch.randint(0, 5, (16,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_cv_writes_per_fold_checkpoints_and_summary(trained):
    res, root = trained.res, trained.root
    assert res["synthetic_data"] is True
    assert len(res["folds"]) == 2
    for f in res["folds"]:
        assert (root / "artifacts" / f"sleepedf_resnet1d_tiny_fold{f['fold']}.pt").exists()
        assert not set(f["test_subjects"]) & (set(f["train_subjects"]) | set(f["val_subjects"]))
    f1s = [f["test"]["macro_f1"] for f in res["folds"]]
    assert res["macro_f1_mean"] == pytest.approx(np.mean(f1s))
    assert res["macro_f1_std"] == pytest.approx(np.std(f1s))
    assert set(res["pooled"]["per_class_f1"]) == {"W", "N1", "N2", "N3", "REM"}
    assert np.array(res["pooled"]["confusion_matrix"]).shape == (5, 5)
    saved = json.loads((root / "results" / "sleepedf_resnet1d_tiny_cv.json").read_text())
    assert saved["deploy_checkpoint"]["selected_by"].startswith("best validation")
    assert (root / "artifacts" / "sleepedf_resnet1d_tiny_fp32.pt").exists()


def test_wesad_three_channel_training(tmp_path):
    make_synthetic(tmp_path / "data" / "wesad", "wesad", n_subjects=3, windows_per_subject=12)
    args = train.build_parser().parse_args([
        "--dataset", "wesad", "--model", "mobilenet1d", "--data_root", str(tmp_path / "data"),
        "--epochs", "1", "--batch_size", "8", "--folds", "0", "--max_folds", "1",
        "--artifacts_dir", str(tmp_path / "a"), "--log_dir", str(tmp_path / "r"), "--quiet"])
    res = train.run_cv(args)
    assert res["channels"] == ["synthetic_bvp", "synthetic_eda", "synthetic_temp"]
    state = torch.load(tmp_path / "a" / "wesad_mobilenet1d_fp32.pt")
    assert state["stem.0.weight"].shape[1] == 3  # first conv sees 3 input channels


def test_export_onnx_and_cpu_latency(trained, tmp_path):
    pytest.importorskip("onnxruntime")
    from bench import main as bench_main
    from export_onnx import export

    model = train.load_model("resnet1d_tiny", 5, 1)
    model.load_state_dict(torch.load(trained.root / "artifacts" / "sleepedf_resnet1d_tiny_fp32.pt"))
    onnx_path = tmp_path / "m.onnx"
    assert export(model, onnx_path, 1, 3000) < 1e-4
    out = tmp_path / "lat.json"
    res = bench_main(["--mode", "onnx-cpu", "--onnx", str(onnx_path), "--iters", "5", "--warmup", "1",
                      "--json_out", str(out), "--results", str(tmp_path / "summary.csv")])
    assert res["p50_ms"] > 0 and res["input_shape"] == [1, 1, 3000]
    assert json.loads(out.read_text())["iters"] == 5


def test_edge_modes_are_explicit_todos():
    from bench import main as bench_main

    for mode in ("jetson", "pi"):
        with pytest.raises(SystemExit) as e:
            bench_main(["--mode", mode])
        assert e.value.code == 2


def test_qat_and_pruning_run(trained, tmp_path):
    import prune
    import qat_train

    common = ["--dataset", "sleepedf", "--data_root", str(trained.root / "data"),
              "--log_dir", str(trained.root / "results"), "--batch_size", "16", "--max_batches", "2"]
    q = qat_train.main(common + ["--epochs", "1", "--output", str(tmp_path / "q.pt")])
    assert 0 <= q["int8_qat_macro_f1"] <= 1
    int8 = torch.jit.load(str(tmp_path / "q.pt"))
    assert int8(torch.randn(2, 1, 3000)).shape == (2, 5)
    p = prune.main(common + ["--finetune_epochs", "1", "--prune_ratio", "0.5", "--output", str(tmp_path / "p.pt")])
    assert p["zeroed_channel_fraction"] >= 0.4
