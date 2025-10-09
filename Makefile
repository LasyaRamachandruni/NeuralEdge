SHELL := /bin/bash
PY ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

DATA_DIR := data
PROCESSED_DIR := $(DATA_DIR)/processed
CALIB_DIR := $(DATA_DIR)/calib
RESULTS_DIR := results
MODELS_DIR := models

# Defaults (can be overridden: make train_fp32 MODEL=mobilenet1d DATASET=wesad)
MODEL ?= resnet1d_tiny
DATASET ?= sleepedf
NUM_CLASSES ?= 3

.PHONY: help setup prepare_data_sleep prepare_data_wesad train_fp32 qat prune export_onnx build_trt_fp16 build_trt_int8 bench_host bench_jetson bench_pi report bench_all clean

help:
	@echo "Targets: setup, prepare_data_sleep, prepare_data_wesad, train_fp32, qat, prune, export_onnx, build_trt_fp16, build_trt_int8, bench_host, bench_jetson, bench_pi, report, bench_all, clean"

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

prepare_data_sleep:
	$(PYBIN) data/prepare_sleepedf.py --out $(PROCESSED_DIR)/sleepedf --calib_out $(CALIB_DIR)/sleepedf --window 30 --fs_out 100 --download false

prepare_data_wesad:
	$(PYBIN) data/prepare_wesad.py --out $(PROCESSED_DIR)/wesad --calib_out $(CALIB_DIR)/wesad --window 8 --download false

train_fp32:
	$(PYBIN) train.py --dataset $(DATASET) --data_root $(PROCESSED_DIR) --model $(MODEL) --num_classes $(NUM_CLASSES) --epochs 30 --batch_size 128 --lr 1e-3 --early_stop_patience 5 --log_dir $(RESULTS_DIR)

qat:
	$(PYBIN) qat_train.py --dataset $(DATASET) --data_root $(PROCESSED_DIR) --model $(MODEL) --num_classes $(NUM_CLASSES) --epochs 10 --batch_size 128 --lr 5e-4 --checkpoint artifacts/$(DATASET)_$(MODEL)_fp32.pt --output artifacts/$(DATASET)_$(MODEL)_qat.pt

prune:
	$(PYBIN) prune.py --dataset $(DATASET) --data_root $(PROCESSED_DIR) --model $(MODEL) --num_classes $(NUM_CLASSES) --prune_ratio 0.3 --finetune_epochs 5 --checkpoint artifacts/$(DATASET)_$(MODEL)_fp32.pt --output artifacts/$(DATASET)_$(MODEL)_pruned.pt

export_onnx:
	$(PYBIN) export_onnx.py --model $(MODEL) --num_classes $(NUM_CLASSES) --checkpoint artifacts/$(DATASET)_$(MODEL)_fp32.pt --opset 17 --onnx_path artifacts/$(DATASET)_$(MODEL).onnx

build_trt_fp16:
	$(PYBIN) build_trt.py --onnx artifacts/$(DATASET)_$(MODEL).onnx --engine engines/$(DATASET)_$(MODEL)_fp16.engine --precision fp16

build_trt_int8:
	$(PYBIN) build_trt.py --onnx artifacts/$(DATASET)_$(MODEL).onnx --engine engines/$(DATASET)_$(MODEL)_int8.engine --precision int8 --calib_dir $(CALIB_DIR)/$(DATASET)

bench_host:
	$(PYBIN) bench.py --dataset $(DATASET) --data_root $(PROCESSED_DIR) --model $(MODEL) --num_classes $(NUM_CLASSES) --mode host --results $(RESULTS_DIR)/summary.csv

bench_jetson:
	$(PYBIN) bench.py --dataset $(DATASET) --data_root $(PROCESSED_DIR) --trt_engine engines/$(DATASET)_$(MODEL)_int8.engine --mode jetson --results $(RESULTS_DIR)/summary.csv

bench_pi:
	$(PYBIN) bench.py --dataset $(DATASET) --data_root $(PROCESSED_DIR) --tflite_model artifacts/$(DATASET)_$(MODEL)_int8.tflite --mode pi --results $(RESULTS_DIR)/summary.csv

report:
	$(PYBIN) scripts/make_report.py --results $(RESULTS_DIR)/summary.csv --out results/report.md

bench_all: prepare_data_sleep train_fp32 export_onnx build_trt_fp16 build_trt_int8 bench_host

clean:
	rm -rf $(VENV) __pycache__ **/__pycache__ *.engine engines/*.engine artifacts/*.onnx artifacts/*.pt

.PHONY: tflite
tflite:
	$(PYBIN) scripts/onnx_to_tflite.py --onnx artifacts/$(DATASET)_$(MODEL).onnx --out artifacts/$(DATASET)_$(MODEL)_fp32.tflite
