#!/usr/bin/env python3
"""
Export all detection models to TorchScript and ONNX in one command.

Loads each model's PyTorch weights, exports them to every requested format,
and reports file sizes and export times.  Existing files are skipped unless
--force is passed.

Usage examples
--------------
# Export everything with defaults
python scripts/run_all_exports.py

# Export only ONNX, skip TorchScript
python scripts/run_all_exports.py --formats onnx

# Export a specific model only
python scripts/run_all_exports.py --models yolov8

# Custom output directory and image size
python scripts/run_all_exports.py --weights-dir backend/weights --image-size 416

# Force re-export even if files already exist
python scripts/run_all_exports.py --force
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all detection models to TorchScript and ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["yolov8", "yolov5"],
        choices=["yolov8", "yolov5"],
        help="Models to export",
    )
    parser.add_argument(
        "--formats", nargs="+",
        default=["torchscript", "onnx"],
        choices=["torchscript", "onnx"],
        help="Export formats",
    )
    parser.add_argument(
        "--weights-dir", default="backend/weights",
        help="Directory where exported files are saved (resolved relative to project root)",
    )
    parser.add_argument(
        "--yolov8-weights", default="yolov8n.pt",
        help="YOLOv8 PyTorch weights filename or path",
    )
    parser.add_argument(
        "--yolov5-variant", default="yolov5s",
        help="YOLOv5 model variant (yolov5n / yolov5s / yolov5m / yolov5l / yolov5x)",
    )
    parser.add_argument("--image-size", type=int, default=640, help="Export image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-export even if the output file already exists",
    )
    return parser.parse_args()


# ── Export helpers ────────────────────────────────────────────────────────


def _load_yolov8(weights: str, image_size: int):
    from app.models.yolov8_detector import YOLOv8Detector
    from app.schemas.detection import BackendType

    detector = YOLOv8Detector(
        backend_type=BackendType.pytorch,
        weights_path=weights,
        image_size=image_size,
    )
    detector.load()
    return detector


def _load_yolov5(variant: str, image_size: int):
    from app.models.yolov5_detector import YOLOv5Detector
    from app.schemas.detection import BackendType

    detector = YOLOv5Detector(
        backend_type=BackendType.pytorch,
        weights_path=None,
        model_variant=variant,
        image_size=image_size,
    )
    detector.load()
    return detector


def export_one(
    detector,
    fmt: str,
    output_path: str,
    force: bool,
    opset: int,
) -> dict:
    out = Path(output_path)

    if out.exists() and not force:
        size_mb = out.stat().st_size / 1_048_576
        return {"path": str(out), "size_mb": size_mb, "status": "skipped (already exists)"}

    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    try:
        if fmt == "torchscript":
            saved = detector.export_torchscript(str(out))
        elif fmt == "onnx":
            saved = detector.export_onnx(str(out))
        else:
            raise ValueError(f"Unknown format: {fmt}")

        elapsed = time.perf_counter() - t0
        size_mb = Path(saved).stat().st_size / 1_048_576
        return {
            "path": str(saved),
            "size_mb": size_mb,
            "elapsed_s": elapsed,
            "status": "ok",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "path": str(out),
            "size_mb": 0.0,
            "elapsed_s": elapsed,
            "status": f"error: {exc}",
        }


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    weights_dir = (project_root / args.weights_dir).resolve()

    # Map (model, format) → output path
    export_plan: list[tuple[str, str, str]] = []
    for model in args.models:
        for fmt in args.formats:
            ext = ".torchscript" if fmt == "torchscript" else ".onnx"
            if model == "yolov8":
                stem = Path(args.yolov8_weights).stem
                out = str(weights_dir / f"{stem}{ext}")
            else:
                out = str(weights_dir / f"{args.yolov5_variant}{ext}")
            export_plan.append((model, fmt, out))

    print("=" * 60)
    print("  Object Detection — Export All Models")
    print("=" * 60)
    print(f"  Models:    {args.models}")
    print(f"  Formats:   {args.formats}")
    print(f"  Image size: {args.image_size}")
    print(f"  Output dir: {weights_dir}")
    print(f"  Force:      {args.force}")
    print()

    # Load detectors (once per model, not once per format)
    detectors: dict[str, object] = {}
    for model in args.models:
        print(f"[load] Loading {model} weights …")
        t0 = time.perf_counter()
        try:
            if model == "yolov8":
                detectors[model] = _load_yolov8(args.yolov8_weights, args.image_size)
            else:
                detectors[model] = _load_yolov5(args.yolov5_variant, args.image_size)
            print(f"       Done ({time.perf_counter() - t0:.1f}s)\n")
        except Exception as exc:
            print(f"       ERROR loading {model}: {exc}\n")
            detectors[model] = None

    # Export
    results: list[dict] = []
    for model, fmt, out_path in export_plan:
        detector = detectors.get(model)
        if detector is None:
            print(f"[skip] {model}/{fmt} — model failed to load")
            results.append({"model": model, "format": fmt, "path": out_path,
                             "size_mb": 0, "status": "skipped (load failed)"})
            continue

        print(f"[export] {model} → {fmt.upper()}  →  {out_path}")
        info = export_one(detector, fmt, out_path, args.force, args.opset)
        info["model"] = model
        info["format"] = fmt
        results.append(info)

        if info["status"] == "ok":
            print(f"         ✓ {info['size_mb']:.1f} MB  ({info.get('elapsed_s', 0):.1f}s)")
        elif "skipped" in info["status"]:
            print(f"         → {info['status']}  ({info['size_mb']:.1f} MB)")
        else:
            print(f"         ✗ {info['status']}")

    # Summary
    print()
    print("=" * 60)
    print("  Export Summary")
    print("=" * 60)
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if "skipped" in r["status"]]
    errors = [r for r in results if r["status"] not in ("ok",) and "skipped" not in r["status"]]

    print(f"  Exported:  {len(ok)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Errors:    {len(errors)}")
    print()

    if ok or skipped:
        print("  Exported files:")
        for r in results:
            if r["status"] == "ok" or "skipped" in r["status"]:
                print(f"    {r['model']:>8} / {r['format']:<12}  {r['path']}")
        print()

    if errors:
        print("  Failed exports:")
        for r in errors:
            print(f"    {r['model']} / {r['format']}: {r['status']}")
        print()
        print("  Common fixes:")
        print("    • Make sure the backend venv is active (source backend/venv/bin/activate)")
        print("    • For YOLOv8: ensure yolov8n.pt (or your weights) is accessible")
        print("    • For YOLOv5: requires internet access for the first download")
        print()

    if ok or skipped:
        print("  Add these paths to backend/.env to enable the exported backends:")
        for r in [*ok, *skipped]:
            if "yolov8" in r["model"] and "torchscript" in r["format"]:
                print(f"    YOLOV8_TORCHSCRIPT_PATH={r['path']}")
            elif "yolov8" in r["model"] and "onnx" in r["format"]:
                print(f"    YOLOV8_ONNX_PATH={r['path']}")
            elif "yolov5" in r["model"] and "torchscript" in r["format"]:
                print(f"    YOLOV5_TORCHSCRIPT_PATH={r['path']}")
            elif "yolov5" in r["model"] and "onnx" in r["format"]:
                print(f"    YOLOV5_ONNX_PATH={r['path']}")


if __name__ == "__main__":
    main()
