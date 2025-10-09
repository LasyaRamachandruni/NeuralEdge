# NeuralEdge — Sleep & Wearables Edge-AI

Reproducible pipelines to train compact biosignal models and deploy optimized inference on Jetson Nano and Raspberry Pi 4. Primary track: Sleep-EDF sleep staging (EEG/EOG, 30 s windows). Alternate: WESAD stress detection (PPG/EDA/Temp).

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

## Deliverables
- Makefile targets: setup, prepare_data_{sleep,wesad}, train_fp32, qat, prune, export_onnx, build_trt_fp16, build_trt_int8, bench_{host,jetson,pi}, report, bench_all
- Dockerfiles: `docker/train.Dockerfile`, `docker/jetson.Dockerfile`, `docker/pi.Dockerfile`
- Scripts: data prep, training, QAT, pruning, ONNX export, TRT build, benchmarking, Streamlit dashboard
- Results: `results/summary.csv`, `results/report.md`

See inline help in each script for arguments.

## Reproducibility
- Subject-wise CV with fixed RNG seeds; `make bench_all` regenerates host results.
- Exact driver/container versions captured in Dockerfiles.
- Jetson power/clock fixed via `nvpmodel` and `jetson_clocks`.