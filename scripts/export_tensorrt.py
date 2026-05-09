#!/usr/bin/env python3
"""
Export YOLOv8, YOLOv5, or RT-DETR to a TensorRT .engine file.

TensorRT engines are NVIDIA-GPU-specific and provide the fastest possible
inference on an NVIDIA GPU by compiling the model for the exact hardware and
CUDA version present at export time.

Requirements
------------
  • NVIDIA GPU with CUDA 11.8+ drivers
  • tensorrt Python package  →  pip install tensorrt
  • Ultralytics >= 8.0        →  pip install ultralytics

The script will fail with a clear message on machines without a CUDA device.

Usage examples
--------------
# Export YOLOv8n (fp16)
python scripts/export_tensorrt.py --model yolov8

# Export YOLOv5s (fp16)
python scripts/export_tensorrt.py --model yolov5

# Export RT-DETR-L (fp16)
python scripts/export_tensorrt.py --model rtdetr

# Export all models with int8 calibration
python scripts/export_tensorrt.py --model yolov8 yolov5 rtdetr --half False --int8

# Custom output directory and image size
python scripts/export_tensorrt.py --model yolov8 --output-dir weights/ --imgsz 1280
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export models to TensorRT .engine files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", nargs="+",
        default=["yolov8"],
        choices=["yolov8", "yolov5", "rtdetr"],
        dest="models",
        help="Model(s) to export",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (square)",
    )
    parser.add_argument(
        "--half", action="store_true", default=True,
        help="Export with FP16 precision (faster, slightly lower accuracy)",
    )
    parser.add_argument(
        "--int8", action="store_true", default=False,
        help="Export with INT8 precision (requires calibration data)",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Batch size for TensorRT engine optimization",
    )
    parser.add_argument(
        "--output-dir", default="weights",
        help="Directory to write .engine files",
    )
    return parser.parse_args()


def check_cuda() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n" + "!" * 65)
            print("  ERROR: No CUDA device detected.")
            print("!" * 65)
            print("  TensorRT engines can only be exported on an NVIDIA GPU.")
            print("  Current device: CPU-only")
            print()
            print("  Options:")
            print("    • Run on a machine with an NVIDIA GPU")
            print("    • Use Google Colab (free GPU runtime)")
            print("    • Use a cloud GPU instance (AWS/GCP/Azure)")
            print("!" * 65 + "\n")
            sys.exit(1)
        device_name = torch.cuda.get_device_name(0)
        print(f"[export_tensorrt] CUDA device detected: {device_name}")
    except ImportError:
        print("[export_tensorrt] ERROR: PyTorch not installed. Run: pip install torch")
        sys.exit(1)


def export_yolov8(imgsz: int, half: bool, int8: bool, batch: int, output_dir: Path) -> None:
    from ultralytics import YOLO
    from app.core.config import get_settings
    settings = get_settings()

    weights = settings.YOLOV8_WEIGHTS
    print(f"[export_tensorrt] Loading YOLOv8 weights: {weights}")
    model = YOLO(weights)

    out_path = output_dir / "yolov8n.engine"
    print(f"[export_tensorrt] Exporting YOLOv8 → {out_path} ...")
    saved = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        int8=int8,
        batch=batch,
        device=0,
    )
    saved_path = Path(str(saved))
    output_dir.mkdir(parents=True, exist_ok=True)
    if saved_path != out_path and saved_path.exists():
        saved_path.rename(out_path)
    print(f"[export_tensorrt] YOLOv8 TensorRT engine saved → {out_path}")


def export_yolov5(imgsz: int, half: bool, int8: bool, batch: int, output_dir: Path) -> None:
    import torch
    import os
    from app.core.config import get_settings
    settings = get_settings()

    backend_root = Path(__file__).resolve().parents[1] / "backend"
    hub_dir = str(backend_root / ".torch_cache")
    os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)

    weights_str = settings.YOLOV5_WEIGHTS
    variant = weights_str if "/" not in weights_str else "yolov5s"
    print(f"[export_tensorrt] Loading YOLOv5 ({variant}) ...")
    model = torch.hub.load("ultralytics/yolov5", variant, pretrained=True, verbose=False)

    out_path = output_dir / "yolov5s.engine"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export via Ultralytics YOLO CLI since torch.hub YOLOv5 lacks direct TRT export
    from ultralytics import YOLO as UltralyticsYOLO
    ult_model = UltralyticsYOLO(f"{variant}.pt")
    print(f"[export_tensorrt] Exporting YOLOv5 → {out_path} ...")
    saved = ult_model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        int8=int8,
        batch=batch,
        device=0,
    )
    saved_path = Path(str(saved))
    if saved_path != out_path and saved_path.exists():
        saved_path.rename(out_path)
    print(f"[export_tensorrt] YOLOv5 TensorRT engine saved → {out_path}")


def export_rtdetr(imgsz: int, half: bool, int8: bool, batch: int, output_dir: Path) -> None:
    from ultralytics import RTDETR
    from app.core.config import get_settings
    settings = get_settings()

    weights = settings.RTDETR_WEIGHTS
    print(f"[export_tensorrt] Loading RT-DETR weights: {weights}")
    model = RTDETR(weights)

    out_path = output_dir / "rtdetr-l.engine"
    print(f"[export_tensorrt] Exporting RT-DETR → {out_path} ...")
    saved = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        int8=int8,
        batch=batch,
        device=0,
    )
    saved_path = Path(str(saved))
    output_dir.mkdir(parents=True, exist_ok=True)
    if saved_path != out_path and saved_path.exists():
        saved_path.rename(out_path)
    print(f"[export_tensorrt] RT-DETR TensorRT engine saved → {out_path}")


def main() -> None:
    args = parse_args()
    check_cuda()

    output_dir = Path(args.output_dir)
    precision = "int8" if args.int8 else ("fp16" if args.half else "fp32")

    print("=" * 65)
    print("  TensorRT Export")
    print("=" * 65)
    print(f"  Models    : {args.models}")
    print(f"  Image size: {args.imgsz}×{args.imgsz}")
    print(f"  Precision : {precision}")
    print(f"  Batch size: {args.batch}")
    print(f"  Output dir: {output_dir}")
    print()

    for model in args.models:
        try:
            if model == "yolov8":
                export_yolov8(args.imgsz, args.half, args.int8, args.batch, output_dir)
            elif model == "yolov5":
                export_yolov5(args.imgsz, args.half, args.int8, args.batch, output_dir)
            elif model == "rtdetr":
                export_rtdetr(args.imgsz, args.half, args.int8, args.batch, output_dir)
        except Exception as exc:
            print(f"\n[export_tensorrt] ERROR exporting {model}: {exc}")
            print(f"  Skipping {model} — continuing with remaining models.\n")

    print("\n[export_tensorrt] Done. Load the .engine files with --backend tensorrt")
    print("  Benchmark: python scripts/benchmark_models.py --backends tensorrt")
    print("  Evaluate : python scripts/evaluate_dataset.py --backend tensorrt \\")
    print("               --annotations data/annotations/instances_manual.json \\")
    print("               --images-dir data/images/val")


if __name__ == "__main__":
    main()
