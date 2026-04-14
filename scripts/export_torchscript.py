#!/usr/bin/env python3
"""
Export a detection model to TorchScript format.

Usage:
    python scripts/export_torchscript.py --model yolov8 --weights yolov8n.pt --output weights/yolov8n.torchscript
    python scripts/export_torchscript.py --model yolov5 --weights yolov5s --output weights/yolov5s.torchscript
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export detection model to TorchScript",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["yolov8", "yolov5"], required=True,
        help="Which model to export"
    )
    parser.add_argument(
        "--weights", default=None,
        help="Path to weights file (PyTorch .pt) or model variant (e.g. yolov5s)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for the .torchscript file"
    )
    parser.add_argument(
        "--image-size", type=int, default=640,
        help="Input image size used during export"
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
        output = args.output or "weights/yolov8n.torchscript"

    elif args.model == "yolov5":
        from app.models.yolov5_detector import YOLOv5Detector
        from app.schemas.detection import BackendType
        detector = YOLOv5Detector(
            backend_type=BackendType.pytorch,
            weights_path=None,
            model_variant=args.weights or "yolov5s",
            image_size=args.image_size,
        )
        output = args.output or "weights/yolov5s.torchscript"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"[export] Loading {args.model} weights…")
    detector.load()

    print(f"[export] Exporting to TorchScript → {output}")
    saved_path = detector.export_torchscript(output)
    print(f"[export] Done: {saved_path}")


if __name__ == "__main__":
    main()
