#!/usr/bin/env python3
"""
End-to-end model comparison: benchmark latency/FPS AND (optionally) COCO mAP.

Runs synthetic-image latency benchmarks for all requested model/backend combos
and, if COCO annotations are supplied, runs mAP evaluation too.  Produces a
single Markdown + CSV report that can be included directly in an assignment.

Usage examples
--------------
# Benchmark only (no dataset required)
python scripts/compare_models.py --output results/comparison

# Benchmark + evaluation
python scripts/compare_models.py \\
    --annotations data/annotations/instances_custom.json \\
    --images-dir data/images/val \\
    --output results/comparison

# Limit to specific models/backends
python scripts/compare_models.py \\
    --models yolov8 yolov5 --backends pytorch onnx \\
    --runs 50 --output results/comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare detection models — benchmark + optional COCO evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["yolov8", "yolov5"],
        choices=["yolov8", "yolov5"],
    )
    parser.add_argument(
        "--backends", nargs="+",
        default=["pytorch", "torchscript", "onnx"],
        choices=["pytorch", "torchscript", "onnx"],
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--annotations", default=None, metavar="JSON",
        help="COCO annotations JSON (enables mAP evaluation)",
    )
    parser.add_argument(
        "--images-dir", default=None, metavar="DIR",
        help="Images directory (required with --annotations)",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--output", default="results/comparison", metavar="PREFIX",
        help="Output file prefix; writes <PREFIX>.csv and <PREFIX>.md",
    )
    return parser.parse_args()


# ── Benchmark step ────────────────────────────────────────────────────────


def run_benchmarks(models, backends, runs, warmup, image_size) -> list[dict]:
    from app.schemas.detection import BackendType, BenchmarkRequest, ModelName
    from app.services.benchmark import run_benchmark

    request = BenchmarkRequest(
        model_names=[ModelName(m) for m in models],
        backend_types=[BackendType(b) for b in backends],
        num_runs=runs,
        warmup_runs=warmup,
        image_size=image_size,
    )
    result = run_benchmark(request)

    # Build baseline (pytorch) lookup for speedup computation
    baseline: dict[str, float] = {}
    for e in result.results:
        if e.backend_type == "pytorch" and e.status == "ok":
            baseline[e.model_name] = e.avg_latency_ms

    rows: list[dict] = []
    for e in result.results:
        base = baseline.get(e.model_name)
        speedup = (
            base / e.avg_latency_ms
            if (base and e.status == "ok" and e.avg_latency_ms > 0)
            else None
        )
        rows.append({
            "model_name": e.model_name,
            "backend_type": e.backend_type,
            "bench_avg_ms": e.avg_latency_ms if e.status == "ok" else None,
            "bench_fps": e.fps if e.status == "ok" else None,
            "bench_speedup": speedup,
            "bench_status": e.status,
            "bench_error": e.error,
        })
    return rows


# ── Evaluation step ───────────────────────────────────────────────────────


def run_evaluations(models, backends, annotations, images_dir, confidence, iou) -> dict[tuple, dict]:
    from app.schemas.detection import BackendType, EvaluationRequest, ModelName
    from app.services.evaluation import evaluate_dataset

    results: dict[tuple, dict] = {}
    combos = [(m, b) for m in models for b in backends]
    total = len(combos)

    for i, (model, backend) in enumerate(combos, 1):
        print(f"  [{i}/{total}] Evaluating {model}/{backend} …", end=" ", flush=True)
        try:
            req = EvaluationRequest(
                model_name=ModelName(model),
                backend_type=BackendType(backend),
                annotations_path=annotations,
                images_dir=images_dir,
                confidence_threshold=confidence,
                iou_threshold=iou,
            )
            r = evaluate_dataset(req)
            results[(model, backend)] = {
                "eval_map50": r.map_50,
                "eval_map50_95": r.map_50_95,
                "eval_fps": r.fps,
                "eval_status": "ok",
                "eval_error": None,
            }
            print(f"mAP@.5={r.map_50:.3f}  mAP@.5:.95={r.map_50_95:.3f}")
        except Exception as exc:
            results[(model, backend)] = {
                "eval_map50": None,
                "eval_map50_95": None,
                "eval_fps": None,
                "eval_status": "error",
                "eval_error": str(exc),
            }
            print(f"ERROR: {exc}")

    return results


# ── Report generation ─────────────────────────────────────────────────────


def merge_rows(bench_rows: list[dict], eval_map: dict[tuple, dict]) -> list[dict]:
    merged = []
    for row in bench_rows:
        key = (row["model_name"], row["backend_type"])
        eval_data = eval_map.get(key, {})
        merged.append({**row, **eval_data})
    return merged


def save_csv(rows: list[dict], path: Path, has_eval: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name", "backend_type",
        "bench_avg_ms", "bench_fps", "bench_speedup", "bench_status",
    ]
    if has_eval:
        fields += ["eval_map50", "eval_map50_95", "eval_fps", "eval_status"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare] CSV  → {path}")


def _fmt(val, fmt=".2f") -> str:
    if val is None:
        return "—"
    return format(val, fmt)


def save_markdown(rows: list[dict], path: Path, has_eval: bool, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Model Comparison Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Benchmark:** {args.runs} runs, {args.warmup} warmup, {args.image_size}×{args.image_size}px  ",
    ]
    if has_eval:
        lines += [
            f"**Dataset:** `{args.annotations}`  ",
            f"**Confidence threshold:** {args.confidence}  |  **IoU threshold:** {args.iou}  ",
        ]
    lines += ["", "---", "", "## Results", ""]

    # Header
    if has_eval:
        header = "| Model | Backend | Avg ms | FPS | Speedup | mAP@.50 | mAP@.5:.95 |"
        divider = "|-------|---------|-------:|----:|--------:|--------:|-----------:|"
    else:
        header = "| Model | Backend | Avg ms | FPS | Speedup |"
        divider = "|-------|---------|-------:|----:|--------:|"

    lines += [header, divider]

    for r in rows:
        speedup_str = f"{r['bench_speedup']:.2f}×" if r.get("bench_speedup") else "—"
        bench_status = r.get("bench_status", "")
        if bench_status == "error":
            avg_ms, fps = "error", "—"
        else:
            avg_ms = _fmt(r.get("bench_avg_ms"))
            fps = _fmt(r.get("bench_fps"), ".1f")

        if has_eval:
            map50 = _fmt(r.get("eval_map50"), ".4f")
            map5095 = _fmt(r.get("eval_map50_95"), ".4f")
            lines.append(
                f"| {r['model_name']} | {r['backend_type']} | "
                f"{avg_ms} | {fps} | {speedup_str} | {map50} | {map5095} |"
            )
        else:
            lines.append(
                f"| {r['model_name']} | {r['backend_type']} | "
                f"{avg_ms} | {fps} | {speedup_str} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- **Speedup** is computed relative to the PyTorch baseline for each model.",
        "- `—` indicates a failed or skipped run (see CSV for error details).",
        "- **TorchScript / ONNX** backends require exported model files.",
        "  Run `scripts/run_all_exports.py` to generate them.",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[compare] Markdown → {path}")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    has_eval = bool(args.annotations and args.images_dir)

    print("=" * 60)
    print("  Object Detection — Model Comparison")
    print("=" * 60)
    print(f"  Models:   {args.models}")
    print(f"  Backends: {args.backends}")
    print(f"  Runs:     {args.runs} + {args.warmup} warmup  @ {args.image_size}px")
    print(f"  Eval:     {'yes — ' + str(args.annotations) if has_eval else 'no (add --annotations to enable)'}")
    print()

    # --- Step 1: Benchmark ---
    print("[Step 1/2] Running inference benchmarks …")
    bench_rows = run_benchmarks(
        args.models, args.backends, args.runs, args.warmup, args.image_size
    )
    print(f"  Done — {len(bench_rows)} combinations benchmarked\n")

    # --- Step 2: Evaluation (optional) ---
    eval_map: dict[tuple, dict] = {}
    if has_eval:
        print("[Step 2/2] Running COCO mAP evaluation …")
        eval_map = run_evaluations(
            args.models, args.backends,
            args.annotations, args.images_dir,
            args.confidence, args.iou,
        )
        print()
    else:
        print("[Step 2/2] Skipping evaluation (no --annotations supplied)\n")

    # --- Merge & save ---
    merged = merge_rows(bench_rows, eval_map)

    prefix = Path(args.output)
    save_csv(merged, prefix.with_suffix(".csv"), has_eval)
    save_markdown(merged, prefix.with_suffix(".md"), has_eval, args)

    print("\n[compare] Done ✓")
    print(f"  Open {prefix.with_suffix('.md')} for a human-readable summary.")


if __name__ == "__main__":
    main()
