# NeuralEdge: compact 1-D CNNs for sleep staging and stress detection

Small 1-D CNNs for two biosignal tasks, with the tooling to compress and export them for edge devices:

* **Sleep staging** on Sleep-EDF Expanded (sleep-cassette): single-channel EEG Fpz-Cz, 30-s epochs, 5 AASM classes (W, N1, N2, N3, REM).
* **Stress detection** on WESAD: three wearable signals (wrist BVP / EDA / TEMP, or chest ECG / EDA / Temp) as three input channels, baseline vs stress.

Pipeline: real data loaders → subject-wise cross-validated training → QAT (INT8) / structured pruning → ONNX export → latency benchmarking.

## Status

| Part | State |
|---|---|
| Sleep-EDF loader (MNE, Fpz-Cz, R&K→AASM, wake trimming, subject-wise files) | Implemented; tested on generated EDF files in CI. Real-data run pending (see below). |
| WESAD loader (pickles, 3-channel windows, leave-one-subject-out) | Implemented; tested on generated pickles in CI. Not yet run on the real dataset. |
| Training with subject-wise CV, per-fold checkpoints, mean ± std macro-F1 | Implemented and tested |
| QAT to INT8 (PyTorch FX) and L1 structured pruning | Implemented and tested on synthetic data. Pruning zeroes channels but does not remove them yet, so it does not make the model smaller or faster. |
| ONNX export + ONNX Runtime CPU latency | Implemented, tested, and measured (table below) |
| TensorRT engines on Jetson Nano, TFLite on Raspberry Pi 4, Jetson power | **Not yet measured.** Scripts exist but have not been run on hardware; `bench.py --mode jetson/pi` exits with a TODO. See [docs/edge_benchmarking.md](docs/edge_benchmarking.md). |

## Results

### Sleep-EDF accuracy: pending

No accuracy numbers on real data yet. The real run (about 2 GB downloaded from PhysioNet, GPU training) is set up in
[`notebooks/run_sleepedf_colab.ipynb`](notebooks/run_sleepedf_colab.ipynb). That notebook writes `results/sleepedf_results.json` and a
table that goes here. The table compares against DeepSleepNet's published Sleep-EDF result (macro-F1 0.769, accuracy 0.820 [1]).
DeepSleepNet is a larger CNN + BiLSTM that looks at sequences of epochs, so it is a reference point, not a like-for-like baseline.

### ONNX Runtime CPU latency (architecture only)

Measured with `scripts/measure_arch_latency.py` and saved in [`results/cpu_latency_architecture.json`](results/cpu_latency_architecture.json).
The weights are **random**, so these numbers say nothing about accuracy. They only show what each architecture costs at inference time.
Setup: ONNX Runtime 1.29.0 CPUExecutionProvider, batch 1, 1 thread, 2000 timed runs after 200 warm-up runs. The machine is a
shared 2-vCPU cloud VM ("Intel Xeon Processor @ 2.10GHz"), not an edge device, and other workloads on the host add noise to the tail latencies.

| Model | Input | Params | ONNX size | p50 | p90 | p99 |
|---|---|---:|---:|---:|---:|---:|
| `mobilenet1d` | Sleep-EDF, 1×3000 (5 classes) | 12,997 | 54 kB | 0.367 ms | 0.445 ms | 0.664 ms |
| `resnet1d_tiny` | Sleep-EDF, 1×3000 (5 classes) | 61,013 | 246 kB | 0.844 ms | 0.979 ms | 1.804 ms |
| `mobilenet1d` | WESAD, 3×256 (2 classes) | 12,834 | 53 kB | 0.049 ms | 0.065 ms | 0.088 ms |
| `resnet1d_tiny` | WESAD, 3×256 (2 classes) | 61,042 | 246 kB | 0.094 ms | 0.140 ms | 0.418 ms |

### Edge devices

Jetson Nano (TensorRT FP16/INT8), Raspberry Pi 4 (ONNX Runtime / TFLite) and power: **not yet measured.**

## Run the real Sleep-EDF experiment on Colab

1. Push this repo to GitHub, then open `notebooks/run_sleepedf_colab.ipynb` in Colab (File → Open notebook → GitHub).
2. Runtime → Change runtime type → T4 GPU.
3. The settings cell clones `LasyaRamachandruni/NeuralEdge` (branch `main`); change `REPO_URL` / `BRANCH` if needed. Leave `SUBJECTS = "0-19"` and `FOLDS = 20` to match DeepSleepNet's 20-fold protocol, or use `FOLDS = 5` for a faster run.
4. Runtime → Run all. The notebook downloads ~40 recordings from PhysioNet through MNE, trains, evaluates, exports ONNX and measures CPU latency.
5. Download the zip from the last cell and commit `results/sleepedf_results.{json,md}`, `results/sleepedf_confusion.png`, `results/sleepedf_<model>_cv.json` and `results/latency_cpu_colab.json`. Then paste the table from `sleepedf_results.md` into the Results section above.

## Data processing

**Sleep-EDF** (`data/prepare_sleepedf.py`, `neuraledge/sleepedf.py`)
- Pairs `SC4ssNE0-PSG.edf` with its `*-Hypnogram.edf`, from a local folder (`--raw_dir`) or downloaded through `mne.datasets.sleep_physionet` (`--download`).
- EEG Fpz-Cz at 100 Hz, cut into 30-s epochs. Stages map as W→W, S1→N1, S2→N2, S3+S4→N3, R→REM. "Movement time" and "Sleep stage ?" epochs are dropped.
- Wake is trimmed to 30 min before the first and after the last sleep epoch (recordings run about 20 h). Each recording is z-scored.
- Both nights of a subject go into the same file, so a person is never in both train and test.

**WESAD** (`data/prepare_wesad.py`, `neuraledge/wesad.py`)
- Reads `S<k>/S<k>.pkl` from the official release (download it by hand after accepting the dataset terms).
- Uses three signals as three channels (`--source wrist`: BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz; `--source chest`: ECG, EDA, Temp at 700 Hz). All are resampled to `--fs_out` (default 32 Hz) and z-scored per subject and channel.
- Uses 8-s windows with a 4-s hop. A window is kept only if every 700 Hz label sample inside it is the same class: baseline vs stress, or `--labels three` to add amusement.

Both scripts write the same format: `subjectNN_x.npy` with shape (windows, channels, samples), `subjectNN_y.npy`, `meta.json`, `meta.csv`, and an
unlabelled INT8 calibration subset. `data/make_synthetic.py` writes random data in this format **for tests only**. It is flagged
`"synthetic": true`, and training on it prints a warning.

## Evaluation protocol

- Folds are subject-wise: `--folds 5` by default, `--folds 0` for leave-one-subject-out (the WESAD default).
- Inside each fold, about 15% of the non-test subjects are held out for early stopping (`--early_stop_patience`). The test subjects are never used for model selection.
- The loss uses class-balanced cross-entropy (`--class_weights balanced`).
- Reported metrics: macro-F1 mean ± std across folds, plus pooled per-class F1, accuracy, Cohen's κ and the confusion matrix → `results/<dataset>_<model>_cv.json`.
- Each fold's best checkpoint is saved as `artifacts/<dataset>_<model>_fold<k>.pt`. The fold with the best *validation* F1 is copied to `<dataset>_<model>_fp32.pt` for export. QAT and pruning reuse that fold's split, so the compressed models are scored on the same held-out subjects.

## Local quickstart

```bash
make setup                      # venv + requirements.txt
make test                       # pytest (synthetic data + generated EDF files)
make smoke                      # synthetic end-to-end: train -> export -> CPU latency

# real data
make prepare_data_sleep SUBJECTS=0-19     # needs access to physionet.org
make train_fp32 DATASET=sleepedf MODEL=resnet1d_tiny
make export_onnx bench_cpu
make qat prune                            # optional
```

## Layout

```
neuraledge/        dataio (on-disk format), sleepedf, wesad, synthetic (tests only), finetune helpers
data/              prepare_sleepedf.py, prepare_wesad.py, make_synthetic.py
models/            resnet1d_tiny.py, mobilenet1d.py   (global average pooling: any channel count / window length)
train.py           subject-wise CV training
qat_train.py       INT8 quantization-aware training (PyTorch FX)
prune.py           L1 structured channel pruning + fine-tuning
export_onnx.py     ONNX export with ONNX Runtime parity check
bench.py           ONNX Runtime CPU latency (Jetson / Pi modes: TODO)
build_trt.py, power_jetson.sh, scripts/onnx_to_tflite.py   edge paths, not yet run on hardware
scripts/           measure_arch_latency.py, summarize_sleepedf.py, make_report.py
notebooks/         run_sleepedf_colab.ipynb
tests/             pytest suite, run by .github/workflows/ci.yml
```

## Known limitations / next steps

- Run the Colab notebook and fill in the Sleep-EDF results.
- The model classifies each 30-s epoch on its own, with no context from neighbouring epochs. Sequence models such as DeepSleepNet use that context, which helps N1 and REM in particular.
- Pruning zeroes channels but does not remove them. A real size/latency gain needs the channels physically removed before export.
- INT8 calibration samples are drawn from all subjects. For a strict protocol, draw them only from the deployed fold's training subjects.
- Edge benchmarks: `docker/jetson.Dockerfile` uses an L4T r35 (JetPack 5) image, which does not support the Jetson Nano. A Nano needs a JetPack 4.6 / r32.7 image.
- WESAD needs a manual download, so it is not part of the Colab notebook yet.

## References

1. A. Supratak, H. Dong, C. Wu, Y. Guo. "DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG." *IEEE Trans. Neural Systems and Rehabilitation Engineering* 25(11), 2017. [arXiv:1703.04046](https://arxiv.org/abs/1703.04046). Sleep-EDF Fpz-Cz, 20-fold CV: accuracy 82.0%, macro-F1 76.9.
2. B. Kemp et al. "Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG." *IEEE Trans. Biomedical Engineering* 47(9), 2000. Sleep-EDF, distributed via PhysioNet (Goldberger et al., *Circulation* 101(23), 2000).
3. P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, K. Van Laerhoven. "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection." *ICMI* 2018.
