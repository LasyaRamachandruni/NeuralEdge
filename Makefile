SHELL := /bin/bash
PY ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYBIN ?= $(VENV)/bin/python

DATA_DIR := data
RAW_DIR := $(DATA_DIR)/raw
PROCESSED_DIR := $(DATA_DIR)/processed
CALIB_DIR := $(DATA_DIR)/calib
RESULTS_DIR := results

# Defaults (override: make train_fp32 MODEL=mobilenet1d DATASET=wesad)
MODEL ?= resnet1d_tiny
DATASET ?= sleepedf
SUBJECTS ?= 0-19
EPOCHS ?= 30
# Sleep-EDF: 5 subject-wise folds; WESAD: leave-one-subject-out
FOLDS ?= $(if $(filter wesad,$(DATASET)),0,5)

.PHONY: help setup test smoke prepare_data_sleep prepare_data_sleep_local prepare_data_wesad \
        train_fp32 qat prune export_onnx bench_cpu bench_jetson bench_pi build_trt_fp16 build_trt_int8 \
        tflite report clean

help:
	@echo "Targets: setup test smoke prepare_data_sleep prepare_data_sleep_local prepare_data_wesad"
	@echo "         train_fp32 qat prune export_onnx bench_cpu report"
	@echo "         (TODO / untested on hardware: build_trt_fp16 build_trt_int8 bench_jetson bench_pi tflite)"

setup:
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

test:
	OMP_NUM_THREADS=1 $(PYBIN) -m pytest -q

# Whole pipeline on tiny SYNTHETIC data (numbers are meaningless; checks the plumbing)
smoke:
	$(PYBIN) data/make_synthetic.py --out /tmp/neuraledge_smoke/data/sleepedf
	$(PYBIN) train.py --data_root /tmp/neuraledge_smoke/data --epochs 2 --batch_size 32 --folds 0 --max_folds 2 \
		--artifacts_dir /tmp/neuraledge_smoke/artifacts --log_dir /tmp/neuraledge_smoke/results
	$(PYBIN) export_onnx.py --model $(MODEL) --checkpoint /tmp/neuraledge_smoke/artifacts/sleepedf_$(MODEL)_fp32.pt \
		--data_dir /tmp/neuraledge_smoke/data/sleepedf --onnx_path /tmp/neuraledge_smoke/artifacts/sleepedf_$(MODEL).onnx
	$(PYBIN) bench.py --mode onnx-cpu --onnx /tmp/neuraledge_smoke/artifacts/sleepedf_$(MODEL).onnx --iters 100

# Downloads Sleep-EDF sleep-cassette subjects through MNE (needs physionet.org access)
prepare_data_sleep:
	$(PYBIN) data/prepare_sleepedf.py --download --subjects $(SUBJECTS) --out $(PROCESSED_DIR)/sleepedf --calib_out $(CALIB_DIR)/sleepedf

# Same, from EDF files already in data/raw/sleep-cassette
prepare_data_sleep_local:
	$(PYBIN) data/prepare_sleepedf.py --raw_dir $(RAW_DIR)/sleep-cassette --subjects $(SUBJECTS) --out $(PROCESSED_DIR)/sleepedf --calib_out $(CALIB_DIR)/sleepedf

# Expects the unzipped WESAD release in data/raw/WESAD
prepare_data_wesad:
	$(PYBIN) data/prepare_wesad.py --raw_dir $(RAW_DIR)/WESAD --out $(PROCESSED_DIR)/wesad --calib_out $(CALIB_DIR)/wesad

train_fp32:
	$(PYBIN) train.py --dataset $(DATASET) --data_root $(PROCESSED_DIR) --model $(MODEL) --folds $(FOLDS) \
		--epochs $(EPOCHS) --batch_size 128 --lr 1e-3 --early_stop_patience 5 --log_dir $(RESULTS_DIR)

qat:
	$(PYBIN) qat_train.py --dataset $(DATASET) --data_root $(PROCESSED_DIR) --model $(MODEL) --epochs 3 --log_dir $(RESULTS_DIR)

prune:
	$(PYBIN) prune.py --dataset $(DATASET) --data_root $(PROCESSED_DIR) --model $(MODEL) --prune_ratio 0.3 --finetune_epochs 5 --log_dir $(RESULTS_DIR)

export_onnx:
	$(PYBIN) export_onnx.py --model $(MODEL) --checkpoint artifacts/$(DATASET)_$(MODEL)_fp32.pt \
		--data_dir $(PROCESSED_DIR)/$(DATASET) --onnx_path artifacts/$(DATASET)_$(MODEL).onnx

bench_cpu:
	$(PYBIN) bench.py --mode onnx-cpu --onnx artifacts/$(DATASET)_$(MODEL).onnx --threads 1 --iters 1000 \
		--json_out $(RESULTS_DIR)/latency_cpu_$(DATASET)_$(MODEL).json --results $(RESULTS_DIR)/summary.csv

# ---- not yet run on hardware; see docs/edge_benchmarking.md ----
build_trt_fp16:
	$(PYBIN) build_trt.py --onnx artifacts/$(DATASET)_$(MODEL).onnx --engine engines/$(DATASET)_$(MODEL)_fp16.engine --precision fp16

build_trt_int8:
	$(PYBIN) build_trt.py --onnx artifacts/$(DATASET)_$(MODEL).onnx --engine engines/$(DATASET)_$(MODEL)_int8.engine --precision int8 --calib_dir $(CALIB_DIR)/$(DATASET)

bench_jetson:
	$(PYBIN) bench.py --mode jetson

bench_pi:
	$(PYBIN) bench.py --mode pi

tflite:
	$(PYBIN) scripts/onnx_to_tflite.py --onnx artifacts/$(DATASET)_$(MODEL).onnx --out artifacts/$(DATASET)_$(MODEL)_fp32.tflite

report:
	$(PYBIN) scripts/make_report.py --results $(RESULTS_DIR)/summary.csv --out $(RESULTS_DIR)/report.md

clean:
	rm -rf __pycache__ */__pycache__ engines/*.engine artifacts/*.onnx artifacts/*.pt /tmp/neuraledge_smoke
