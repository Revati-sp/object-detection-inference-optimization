#!/usr/bin/env python3
"""
Evaluate one or more detection model/backend combinations on a COCO-annotated dataset.

Computes mAP@0.50 and mAP@0.50:0.95 using pycocotools and reports latency / FPS.

Annotation types
----------------
  manual  — human-annotated ground truth  (REQUIRED for final submission)
  pseudo  — model-generated pseudo-labels (sanity check only, not valid for submission)

By default (--annotation-type auto), the script detects the type automatically
and prints a clear warning if pseudo-labels are found.  Pass --annotation-type manual
to enforce that only real annotations are accepted, or --annotation-type pseudo to
explicitly run a sanity check without being blocked.

Usage examples
--------------
# Final submission evaluation (manual annotations required):
python scripts/evaluate_dataset.py \\
    --model yolov8 yolov5 --compare \\
    --annotation-type manual \\
    --annotations data/annotations/instances_manual.json \\
    --images-dir data/images/val \\
    --output results/eval_report_manual.csv

# Sanity check with pseudo-labels:
python scripts/evaluate_dataset.py \\
    --model yolov8 yolov5 --compare \\
    --annotation-type pseudo \\
    --annotations data/annotations/instances_custom.json \\
    --images-dir data/images/val \\
    --output results/eval_report_pseudo.csv

# Single model evaluation (auto-detect annotation type):
python scripts/evaluate_dataset.py \\
    --model yolov8 --backend pytorch \\
    --annotations data/annotations/instances_manual.json \\
    --images-dir data/images/val

# Tune thresholds and save COCO prediction JSON
python scripts/evaluate_dataset.py \\
    --model yolov5 --backend onnx \\
    --annotations data/annotations/instances_manual.json \\
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
        "--annotation-type",
        choices=["auto", "manual", "pseudo"],
        default="auto",
        dest="annotation_type",
        help=(
            "'auto' (default) = detect type from file content; "
            "'manual' = require human annotations, abort if pseudo-labels detected; "
            "'pseudo' = explicitly allow pseudo-labels (sanity-check mode only)"
        ),
    )
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
        f"{'Avg ms':>9} {'FPS':>7}  {'Ann.Type':<10}  {'Valid?':<7}  {'Status'}"
    )
    width = len(header)
    print(f"\n{'─'*width}")
    print(f"  Evaluation Comparison")
    print(f"{'─'*width}")
    print(header)
    print(f"{'─'*width}")
    for r in rows:
        ann_tag = r.get("annotation_type", "unknown")
        valid = "YES" if ann_tag == "manual" else "NO"
        if r["status"] == "ok":
            print(
                f"  {r['model_name']:<10} {r['backend_type']:<14} "
                f"{r['map_50']:>9.4f} {r['map_50_95']:>11.4f} "
                f"{r['average_latency_ms']:>9.2f} {r['fps']:>7.1f}  {ann_tag:<10}  {valid:<7}  ok"
            )
        else:
            err = (r.get("error") or "failed")[:28]
            print(f"  {r['model_name']:<10} {r['backend_type']:<14}  {'— error —':>33}  {ann_tag:<10}  {valid:<7}  {err}")
    print(f"{'─'*width}")
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        best = max(ok_rows, key=lambda r: r["map_50_95"])
        print(f"  ✓ Best mAP: {best['model_name']}/{best['backend_type']} — "
              f"mAP@.5:.95 = {best['map_50_95']:.4f}")

    # Print footer warning if any pseudo-label results are included
    pseudo_rows = [r for r in ok_rows if r.get("annotation_type") == "pseudo"]
    if pseudo_rows:
        print()
        print("  ⚠  PSEUDO-LABEL WARNING — Valid? = NO for all rows above")
        print("     Results computed against YOLOv8-generated annotations (circular).")
        print("     mAP = 1.000 for YOLOv8/PyTorch is expected — this is NOT real accuracy.")
        print("     These numbers are a SANITY CHECK ONLY.")
        print("     DO NOT report these as final accuracy in your submission.")
        print()
        print("     To get submission-valid results:")
        print("       1. Annotate images manually (see docs/annotation_workflow.md)")
        print("       2. Save to data/annotations/instances_manual.json")
        print("       3. Re-run with --annotation-type manual --annotations data/annotations/instances_manual.json")
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
            "average_latency_ms", "fps",
            "annotation_type", "annotations_file", "num_gt_boxes", "submission_valid",
            "status", "error",
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
            "annotation_type": result.annotation_type,
            "annotation_warning": result.annotation_warning,
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
            "annotation_type": "unknown",
            "annotation_warning": None,
            "status": "error",
            "error": str(exc),
            "_result_obj": None,
        }


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── Pre-flight annotation validation ────────────────────────────────────
    # Import the shared detection logic from validate_annotations.py
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_annotations import detect_annotation_type, validate_file

    detected_type, ann_reasons, ann_stats = validate_file(args.annotations)

    # Resolve effective annotation type
    if args.annotation_type == "auto":
        effective_type = detected_type
    elif args.annotation_type == "manual":
        if detected_type in ("pseudo", "template"):
            print("\n" + "!" * 65)
            print("  ANNOTATION TYPE ERROR")
            print("!" * 65)
            print(f"  --annotation-type manual was requested, but the file")
            print(f"  '{args.annotations}' contains:")
            for r in ann_reasons:
                print(f"    • {r}")
            print()
            print("  Final mAP evaluation requires human annotations.")
            print("  Please annotate images manually and save to:")
            print("    data/annotations/instances_manual.json")
            print("  See docs/annotation_workflow.md for instructions.")
            print("!" * 65 + "\n")
            sys.exit(1)
        effective_type = "manual"
    else:
        # --annotation-type pseudo  →  explicitly allow, just print note
        effective_type = detected_type

    all_backends = ["pytorch", "torchscript", "onnx", "onnx_quant"]
    combos: list[tuple[str, str]] = []

    if args.compare:
        for model in args.models:
            for backend in all_backends:
                combos.append((model, backend))
    else:
        for model in args.models:
            combos.append((model, args.backend))

    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  Object Detection — Dataset Evaluation")
    print("=" * 65)
    print(f"  Annotations  : {args.annotations}")
    print(f"  Images dir   : {args.images_dir}")
    print(f"  Confidence   : {args.confidence}   IoU: {args.iou}")
    print(f"  Combos       : {len(combos)}")
    print(f"  Images (GT)  : {ann_stats['num_images']}")
    print(f"  Boxes (GT)   : {ann_stats['num_annotations']}")
    print(f"  Contributor  : {ann_stats['contributor']}")

    if effective_type == "manual":
        print(f"  Annotation   : ✓ MANUAL — valid for final submission")
    elif effective_type == "pseudo":
        print(f"  Annotation   : ⚠ PSEUDO-LABELS — sanity check only, NOT for submission")
        print()
        print("  ┌─ WARNING ──────────────────────────────────────────────┐")
        print("  │ Pseudo-label annotations detected. mAP results reflect │")
        print("  │ model self-consistency, NOT real detection accuracy.    │")
        print("  │ Do NOT report these as final accuracy in your report.  │")
        print("  └────────────────────────────────────────────────────────┘")
    elif effective_type == "template":
        print(f"  Annotation   : ✗ PLACEHOLDER — no annotations present")
        print("\n  ERROR: Annotation file is empty. Add human annotations first.")
        print("  See docs/annotation_workflow.md\n")
        sys.exit(1)
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
        clean_rows = []
        for r in rows:
            row = {k: v for k, v in r.items() if k != "_result_obj"}
            # Enrich each row with annotation metadata for self-documenting CSVs
            row.setdefault("annotations_file", args.annotations)
            row.setdefault("num_gt_boxes", ann_stats.get("num_annotations", 0))
            row.setdefault("submission_valid", row.get("annotation_type") == "manual")
            clean_rows.append(row)
        save_results(clean_rows, args.output)


if __name__ == "__main__":
    main()
