#!/usr/bin/env python3
"""
Export a YOLOv8 or YOLOv5 ONNX model to INT8 dynamic quantization.

Dynamic quantization converts weight matrices from FP32 to INT8 at export
time, while activations are quantized on-the-fly during inference.  The
resulting model is ~3.5x smaller and faster on modern CPUs that support INT8
acceleration (Intel VNNI, ARM Neon dot-product, Apple Silicon).

No calibration dataset is required (unlike static quantization).

Usage
-----
# Quantize both models (default)
python scripts/export_onnx_quant.py

# Quantize a specific model
python scripts/export_onnx_quant.py --model yolov8
python scripts/export_onnx_quant.py --model yolov5

# Custom paths
python scripts/export_onnx_quant.py \\
    --input  backend/weights/yolov8n.onnx \\
    --output backend/weights/yolov8n_int8.onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure onnxruntime is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


DEFAULT_MODELS = {
    "yolov8": {
        "input":  "backend/weights/yolov8n.onnx",
        "output": "backend/weights/yolov8n_int8.onnx",
    },
    "yolov5": {
        "input":  "backend/weights/yolov5s.onnx",
        "output": "backend/weights/yolov5s_int8.onnx",
    },
}


def quantize(input_path: str, output_path: str) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    inp = Path(input_path)
    out = Path(output_path)

    if not inp.exists():
        print(f"  ERROR: input not found: {inp}")
        print("         Run scripts/export_onnx.py first.")
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Quantizing {inp.name} → {out.name} ...")
    t0 = time.perf_counter()
    quantize_dynamic(str(inp), str(out), weight_type=QuantType.QUInt8)
    elapsed = time.perf_counter() - t0

    in_mb  = inp.stat().st_size / 1e6
    out_mb = out.stat().st_size / 1e6
    print(f"  ✓ Done in {elapsed:.1f}s | {in_mb:.1f} MB → {out_mb:.1f} MB "
          f"({in_mb / out_mb:.1f}× smaller)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantize ONNX detection models to INT8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", nargs="+",
        default=list(DEFAULT_MODELS.keys()),
        choices=list(DEFAULT_MODELS.keys()),
        help="Model(s) to quantize",
    )
    p.add_argument("--input",  default=None, help="Override input ONNX path (single model)")
    p.add_argument("--output", default=None, help="Override output ONNX path (single model)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  ONNX INT8 Dynamic Quantization")
    print("=" * 60)

    if args.input and args.output:
        quantize(args.input, args.output)
    else:
        for model_name in args.model:
            cfg = DEFAULT_MODELS[model_name]
            print(f"\n[{model_name}]")
            quantize(cfg["input"], cfg["output"])

    print("\nDone.  Load in backend with backend_type=onnx_quant.")
    print("Or benchmark with:")
    print("  python scripts/benchmark_models.py --backends onnx onnx_quant")


if __name__ == "__main__":
    main()
