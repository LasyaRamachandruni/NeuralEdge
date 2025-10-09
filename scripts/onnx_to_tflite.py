#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import onnx
import onnxsim
import onnx2tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib_dir", type=str, default=None)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    tmp_dir = onnx_path.parent / "tmp_onnx2tf"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Simplify ONNX
    model = onnx.load(str(onnx_path))
    model_simplified, _ = onnxsim.simplify(model)
    sim_path = tmp_dir / "model_simplified.onnx"
    onnx.save(model_simplified, str(sim_path))

    # Convert via onnx2tf (which yields a SavedModel)
    onnx2tf.convert(input_onnx_file_path=str(sim_path), output_folder_path=str(tmp_dir / "saved_model"))

    # Convert SavedModel to TFLite
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tmp_dir / "saved_model"))
    if args.int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if args.calib_dir and Path(args.calib_dir).exists():
            reps = []
            for f in sorted(Path(args.calib_dir).glob("*.npy")):
                arr = np.load(f).astype(np.float32)
                reps.append(arr)
            if reps:
                reps = np.concatenate(reps, axis=0)
                reps = reps[:1024]
                def rep_ds():
                    for i in range(len(reps)):
                        yield [reps[i][None, None, :]]
                converter.representative_dataset = rep_ds
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
