#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

try:
    import tensorrt as trt
except Exception as e:
    trt = None


class DataLoader:
    def __init__(self, calib_dir: Path, max_samples: int = 1024, seq_len: int = 3000):
        self.seq_len = seq_len
        self.samples = []
        if calib_dir.exists():
            for f in calib_dir.glob("*.npy"):
                arr = np.load(f)
                self.samples.append(arr)
        if not self.samples:
            self.samples = [np.random.randn(max_samples, seq_len).astype(np.float32)]
        self.samples = np.concatenate(self.samples, axis=0)[:max_samples]
        self.idx = 0

    def get_batch(self, size: int = 8):
        if self.idx >= len(self.samples):
            return None
        chunk = self.samples[self.idx : self.idx + size]
        self.idx += size
        return chunk


def build_engine(onnx_path: Path, engine_path: Path, precision: str, calib_dir: Path | None):
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
    profile.set_shape("input", (1, 1, 3000), (1, 1, 3000), (1, 1, 6000))
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
    args = ap.parse_args()
    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    calib_dir = Path(args.calib_dir) if args.calib_dir else None
    build_engine(onnx_path, engine_path, args.precision, calib_dir)


if __name__ == "__main__":
    main()

