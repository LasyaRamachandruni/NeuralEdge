# Edge benchmarking: what runs where

| Target | Status | How |
|---|---|---|
| x86 / any CPU, ONNX Runtime | implemented, measured (see README) | `bench.py --mode onnx-cpu` |
| Raspberry Pi 4, ONNX Runtime | implemented, **not yet measured** | same command, run on the Pi |
| Raspberry Pi 4, TFLite int8 | **TODO** (`bench.py --mode pi` exits with a message) | `scripts/onnx_to_tflite.py` exists but is untested |
| Jetson Nano, TensorRT FP16/INT8 | **TODO** (`bench.py --mode jetson` exits with a message) | `build_trt.py` + `trtexec` below, untested on hardware |
| Jetson power | **TODO** | `power_jetson.sh` (untested on hardware) |

Nothing in this repo reports a Jetson or Pi number yet. When one is measured,
commit the raw output next to the result.

## Raspberry Pi 4 (ONNX Runtime CPU)

```bash
# Raspberry Pi OS 64-bit
python3 -m pip install onnxruntime numpy
python3 bench.py --mode onnx-cpu --onnx artifacts/sleepedf_resnet1d_tiny.onnx \
    --threads 4 --iters 1000 --label "Pi 4, 64-bit OS" \
    --json_out results/latency_pi4_onnx.json
```

## Jetson Nano (TensorRT)

The Nano supports JetPack 4.6.x (TensorRT 8.2) at most. `docker/jetson.Dockerfile`
uses an L4T r35 (JetPack 5) image, which targets Xavier/Orin, not the Nano; use the
JetPack 4.6 `l4t-ml:r32.7.1-py3` image on a Nano.

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
# FP16 engine + latency in one step
/usr/src/tensorrt/bin/trtexec --onnx=artifacts/sleepedf_resnet1d_tiny.onnx --fp16 \
    --shapes=input:1x1x3000 --saveEngine=engines/sleepedf_resnet1d_tiny_fp16.engine \
    --iterations=1000 --avgRuns=100 | tee results/trtexec_nano_fp16.txt
# power while running it
./power_jetson.sh /usr/src/tensorrt/bin/trtexec \
    --loadEngine=engines/sleepedf_resnet1d_tiny_fp16.engine --shapes=input:1x1x3000 --duration=30
```

INT8 needs a calibration set: `python build_trt.py --precision int8 --calib_dir data/calib/sleepedf ...`
(written by the prepare scripts).
