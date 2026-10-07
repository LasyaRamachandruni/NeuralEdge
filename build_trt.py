#!/usr/bin/env python3
"""Build a TensorRT engine from the exported ONNX model (run on the Jetson).

Status: written against the TensorRT 8.x Python API but NOT yet run on a Jetson;
no engine or latency from it has been measured. See docs/edge_benchmarking.md for
the trtexec route, which is the simplest way to get the first numbers.
"""
import argparse
from pathlib import Path
import numpy as np

try:
    import tensorrt as trt
except Exception as e:
    trt = None


class DataLoader:
    def __init__(self, calib_dir: Path, max_samples: int = 1024):
        # calibration arrays are (n, channels, samples), written by data/prepare_*.py
        self.samples = [np.load(f) for f in sorted(calib_dir.glob("*.npy"))] if calib_dir.exists() else []
        if not self.samples:
            raise FileNotFoundError(f"no calibration .npy files in {calib_dir}; run data/prepare_*.py --calib_out")
        self.samples = np.ascontiguousarray(np.concatenate(self.samples, axis=0)[:max_samples], dtype=np.float32)
        self.idx = 0

    def get_batch(self, size: int = 8):
        if self.idx + size > len(self.samples):  # only full batches
            return None
        chunk = self.samples[self.idx : self.idx + size]
        self.idx += size
        return chunk


def build_engine(onnx_path: Path, engine_path: Path, precision: str, calib_dir: Path | None,
                 in_ch: int = 1, n_samples: int = 3000):
    if trt is None:
        raise RuntimeError("TensorRT is not available in this environment")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 29
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("Warning: platform lacks fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("Warning: platform lacks fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        assert calib_dir is not None
        dl = DataLoader(calib_dir)

        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self.d = dl
                self.device_input = None

            def get_batch_size(self):
                return 8

            def get_batch(self, names):
                batch = self.d.get_batch(8)
                if batch is None:
                    return None
                import pycuda.driver as cuda
                import pycuda.autoinit  # noqa: F401

                if self.device_input is None:
                    self.device_input = cuda.mem_alloc(batch.nbytes)
                cuda.memcpy_htod(self.device_input, batch)
                return [int(self.device_input)]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, cache):
                pass

        config.int8_calibrator = Calibrator()
    else:
        pass
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, in_ch, n_samples), (1, in_ch, n_samples), (8, in_ch, n_samples))
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"Built TensorRT engine: {engine_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--engine", type=str, required=True)
    ap.add_argument("--precision", type=str, choices=["fp16", "int8"], required=True)
    ap.add_argument("--calib_dir", type=str, default=None)
    ap.add_argument("--in_ch", type=int, default=1, help="3 for WESAD")
    ap.add_argument("--n_samples", type=int, default=3000, help="256 for WESAD (8 s at 32 Hz)")
    args = ap.parse_args()
    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    calib_dir = Path(args.calib_dir) if args.calib_dir else None
    build_engine(onnx_path, engine_path, args.precision, calib_dir, args.in_ch, args.n_samples)


if __name__ == "__main__":
    main()

