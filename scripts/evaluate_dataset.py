#!/usr/bin/env python3
"""
Evaluate one or more detection model/backend combinations on a COCO-annotated dataset.

Computes mAP@0.50 and mAP@0.50:0.95 using pycocotools and reports latency / FPS.

Usage examples
--------------
# Single model evaluation
python scripts/evaluate_dataset.py \\
    --model yolov8 --backend pytorch \\
    --annotations data/annotations/instances_custom.json \\
    --images-dir data/images/val

# Compare all backends for one model
python scripts/evaluate_dataset.py \\
    --model yolov8 --compare \\
    --annotations data/annotations/instances_custom.json \\
    --images-dir data/images/val

# Compare ALL model/backend combos and save a report
python scripts/evaluate_dataset.py \\
    --model yolov8 yolov5 --compare \\
    --annotations data/annotations/instances_custom.json \\
    --images-dir data/images/val \\
    --output results/eval_report.csv

# Tune thresholds and save COCO prediction JSON
python scripts/evaluate_dataset.py \\
    --model yolov5 --backend onnx \\
    --annotations data/annotations/instances_custom.json \\
    --images-dir data/images/val \\
    --confidence 0.3 --iou 0.5 \\
    --save-predictions results/yolov5_onnx_preds.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detection models with COCO mAP metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", nargs="+",
        choices=["yolov8", "yolov5"],
        default=["yolov8"],
        dest="models",
        help="Model(s) to evaluate",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "torchscript", "onnx", "onnx_quant", "coreml"],
        default="pytorch",
        help="Backend to use (ignored when --compare is set)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Evaluate all backends for every selected model and print a comparison table",
    )
    parser.add_argument(
        "--annotations", required=True, metavar="JSON",
        help="Path to COCO-format ground-truth annotations JSON",
    )
    parser.add_argument(
        "--images-dir", required=True, metavar="DIR",
        help="Directory containing the evaluation images",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Save summary to .csv or .json (auto-detected by extension)",
    )
    parser.add_argument(
        "--save-predictions", default=None, metavar="PATH",
        help="Save COCO-format prediction JSON for the last run",
    )
    return parser.parse_args()


# ── Formatting ────────────────────────────────────────────────────────────


def print_single(result) -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Evaluation — {result.model_name} / {result.backend_type}")
    print(sep)
    print(f"  Images evaluated :  {result.num_images}")
    print(f"  mAP @ IoU=0.50   :  {result.map_50:.4f}  ({result.map_50*100:.1f}%)")
    print(f"  mAP @ 0.50:0.95  :  {result.map_50_95:.4f}  ({result.map_50_95*100:.1f}%)")
    print(f"  Avg latency      :  {result.average_latency_ms:.2f} ms")
    print(f"  FPS              :  {result.fps:.1f}")
    print(f"{sep}\n")


def print_comparison_table(rows: list[dict]) -> None:
    header = (
        f"  {'Model':<10} {'Backend':<14} "
        f"{'mAP@.50':>9} {'mAP@.5:.95':>11} "
        f"{'Avg ms':>9} {'FPS':>7}  {'Status'}"
    )
    width = len(header)
    print(f"\n{'─'*width}")
    print(f"  Evaluation Comparison")
    print(f"{'─'*width}")
    print(header)
    print(f"{'─'*width}")
    for r in rows:
        if r["status"] == "ok":
            print(
                f"  {r['model_name']:<10} {r['backend_type']:<14} "
                f"{r['map_50']:>9.4f} {r['map_50_95']:>11.4f} "
                f"{r['average_latency_ms']:>9.2f} {r['fps']:>7.1f}  ok"
            )
        else:
            err = (r.get("error") or "failed")[:30]
            print(f"  {r['model_name']:<10} {r['backend_type']:<14}  {'— error —':>33}  {err}")
    print(f"{'─'*width}")
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        best = max(ok_rows, key=lambda r: r["map_50_95"])
        print(f"  ✓ Best mAP: {best['model_name']}/{best['backend_type']} — "
              f"mAP@.5:.95 = {best['map_50_95']:.4f}")
    print()


def save_results(rows: list[dict], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()

    if ext == ".json":
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[evaluate] Summary saved → {out}")
    else:
        if ext not in (".csv",):
            out = out.with_suffix(".csv")
        fieldnames = [
            "model_name", "backend_type",
            "num_images", "map_50", "map_50_95",
            "average_latency_ms", "fps", "status", "error",
        ]
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"[evaluate] Summary saved → {out}")


# ── Core evaluation ───────────────────────────────────────────────────────


def evaluate_one(
    model: str,
    backend: str,
    annotations: str,
    images_dir: str,
    confidence: float,
    iou: float,
) -> dict:
    from app.schemas.detection import BackendType, EvaluationRequest, ModelName
    from app.services.evaluation import evaluate_dataset

    request = EvaluationRequest(
        model_name=ModelName(model),
        backend_type=BackendType(backend),
        annotations_path=annotations,
        images_dir=images_dir,
        confidence_threshold=confidence,
        iou_threshold=iou,
    )

    try:
        result = evaluate_dataset(request)
        return {
            "model_name": result.model_name,
            "backend_type": result.backend_type,
            "num_images": result.num_images,
            "map_50": result.map_50,
            "map_50_95": result.map_50_95,
            "average_latency_ms": result.average_latency_ms,
            "fps": result.fps,
            "per_image_latencies_ms": result.per_image_latencies_ms,
            "status": "ok",
            "error": None,
            "_result_obj": result,
        }
    except Exception as exc:
        return {
            "model_name": model,
            "backend_type": backend,
            "num_images": 0,
            "map_50": 0.0,
            "map_50_95": 0.0,
            "average_latency_ms": 0.0,
            "fps": 0.0,
            "per_image_latencies_ms": [],
            "status": "error",
            "error": str(exc),
            "_result_obj": None,
        }


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    all_backends = ["pytorch", "torchscript", "onnx", "onnx_quant"]
    combos: list[tuple[str, str]] = []

    if args.compare:
        for model in args.models:
            for backend in all_backends:
                combos.append((model, backend))
    else:
        for model in args.models:
            combos.append((model, args.backend))

    print("=" * 60)
    print("  Object Detection — Dataset Evaluation")
    print("=" * 60)
    print(f"  Annotations : {args.annotations}")
    print(f"  Images dir  : {args.images_dir}")
    print(f"  Confidence  : {args.confidence}   IoU: {args.iou}")
    print(f"  Combos      : {len(combos)}")
    print()

    rows: list[dict] = []
    last_result_obj = None

    for i, (model, backend) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] Evaluating {model} / {backend} …")
        row = evaluate_one(
            model=model,
            backend=backend,
            annotations=args.annotations,
            images_dir=args.images_dir,
            confidence=args.confidence,
            iou=args.iou,
        )
        rows.append(row)
        last_result_obj = row.get("_result_obj")

        if row["status"] == "ok":
            print(
                f"       mAP@.50={row['map_50']:.4f}  "
                f"mAP@.5:.95={row['map_50_95']:.4f}  "
                f"FPS={row['fps']:.1f}"
            )
        else:
            print(f"       ERROR: {row['error']}")

    # Output
    if len(combos) == 1 and rows[0]["status"] == "ok":
        print_single(last_result_obj)
    else:
        print_comparison_table(rows)

    # Save predictions for the last successful run
    if args.save_predictions and last_result_obj is not None:
        preds_path = Path(args.save_predictions)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        data = last_result_obj.model_dump()
        with open(preds_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[evaluate] Predictions saved → {preds_path}")

    # Save summary table
    if args.output:
        clean_rows = [{k: v for k, v in r.items() if k != "_result_obj"} for r in rows]
        save_results(clean_rows, args.output)


if __name__ == "__main__":
    main()
