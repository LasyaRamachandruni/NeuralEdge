# NeuralEdge — Sleep & Wearables Edge-AI

Pipelines to train compact biosignal models and deploy optimized inference on Jetson Nano and Raspberry Pi 4. Primary track: Sleep-EDF sleep staging (EEG/EOG, 30 s windows). Alternate: WESAD stress detection (PPG/EDA/Temp).

> **Status:** the training, compression, export, TensorRT and benchmarking stages are implemented. The data-preparation scripts currently generate **synthetic placeholder data** in the expected format so the full pipeline runs end to end; parsing the real Sleep-EDF (EDF) and WESAD recordings is the next step. Accuracy numbers from the current pipeline are therefore not meaningful yet; latency, model size and power measurements are.

## Pipeline

```
 data/prepare_*.py        30 s windows @ 100 Hz, per-subject .npy files, INT8 calibration set
        │
        ▼
 train.py                 FP32 training, leave-subjects-out cross-validation, macro-F1
 qat_train.py / prune.py  quantization-aware training / structured pruning
        │
        ▼
 export_onnx.py           ONNX export (+ onnxsim)
        │
        ├──► build_trt.py            TensorRT FP16 / INT8 engines   (Jetson Nano)
        └──► scripts/onnx_to_tflite  TFLite                         (Raspberry Pi 4)
        │
        ▼
 bench.py                 latency, throughput and power (power_jetson.sh) → results/summary.csv
 scripts/make_report.py   results/report.md        dash/app.py   Streamlit dashboard
```

## Models

Both are 1-D CNNs ending in global average pooling, so they accept any window length and any number of input channels.

| Model | Design | Parameters (3 classes) |
|---|---|---:|
| `resnet1d_tiny` | Stem + 3 residual blocks (kernel 7), widths 16 → 32 → 64 | 60,883 |
| `mobilenet1d` | Stem + 3 depthwise-separable blocks, widths 16 → 32 → 64 → 128 | 12,739 |

## Quickstart

```bash
make setup
make prepare_data_sleep
make train_fp32 MODEL=resnet1d_tiny DATASET=sleepedf NUM_CLASSES=3
make export_onnx
make build_trt_fp16
make build_trt_int8
make bench_host
```

Jetson Nano (on device):
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
python build_trt.py --onnx artifacts/sleepedf_resnet1d_tiny.onnx --engine engines/sleepedf_resnet1d_tiny_int8.engine --precision int8 --calib_dir data/calib/sleepedf
./power_jetson.sh python bench.py --dataset sleepedf --data_root data/processed --trt_engine engines/sleepedf_resnet1d_tiny_int8.engine --mode jetson --results results/summary.csv
```

## Tests

```bash
pip install torch scikit-learn tqdm pytest
pytest
```

The tests check output shapes for single- and multi-channel inputs and several window lengths, gradient flow, model size, and that `train.py` can build each model.

## Deliverables
- Makefile targets: setup, prepare_data_{sleep,wesad}, train_fp32, qat, prune, export_onnx, build_trt_fp16, build_trt_int8, bench_{host,jetson,pi}, report, bench_all
- Dockerfiles: `docker/train.Dockerfile`, `docker/jetson.Dockerfile`, `docker/pi.Dockerfile`
- Scripts: data prep, training, QAT, pruning, ONNX export, TRT build, benchmarking, Streamlit dashboard
- Results (generated): `results/summary.csv`, `results/report.md`

See inline help in each script for arguments.

## Reproducibility
- Subject-wise CV with fixed RNG seeds; `make bench_all` regenerates host results.
- Exact driver/container versions captured in Dockerfiles.
- Jetson power/clock fixed via `nvpmodel` and `jetson_clocks`.
