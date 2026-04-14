#!/usr/bin/env python3
"""
Export a detection model to ONNX format.

Usage:
    python scripts/export_onnx.py --model yolov8 --weights yolov8n.pt --output weights/yolov8n.onnx
    python scripts/export_onnx.py --model yolov5 --output weights/yolov5s.onnx
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export detection model to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["yolov8", "yolov5"], required=True,
        help="Which model to export"
    )
    parser.add_argument(
        "--weights", default=None,
        help="Path to PyTorch weights or model variant string"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for the .onnx file"
    )
    parser.add_argument(
        "--image-size", type=int, default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--opset", type=int, default=12,
        help="ONNX opset version"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model == "yolov8":
        from app.models.yolov8_detector import YOLOv8Detector
        from app.schemas.detection import BackendType
        detector = YOLOv8Detector(
            backend_type=BackendType.pytorch,
            weights_path=args.weights or "yolov8n.pt",
            image_size=args.image_size,
        )
        output = args.output or "weights/yolov8n.onnx"

    elif args.model == "yolov5":
        from app.models.yolov5_detector import YOLOv5Detector
        from app.schemas.detection import BackendType
        detector = YOLOv5Detector(
            backend_type=BackendType.pytorch,
            weights_path=None,
            model_variant=args.weights or "yolov5s",
            image_size=args.image_size,
        )
        output = args.output or "weights/yolov5s.onnx"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"[export] Loading {args.model} weights…")
    detector.load()

    print(f"[export] Exporting to ONNX → {output}")
    saved_path = detector.export_onnx(output)
    print(f"[export] Done: {saved_path}")

    # Optionally verify the exported model
    try:
        import onnx
        model_proto = onnx.load(saved_path)
        onnx.checker.check_model(model_proto)
        print("[export] ONNX model verified ✓")
    except Exception as e:
        print(f"[export] ONNX verification warning: {e}")


if __name__ == "__main__":
    main()
