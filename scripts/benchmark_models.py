#!/usr/bin/env python3
"""
Benchmark inference latency and FPS for model/backend combinations.

Runs N timed inference passes on a synthetic random image and reports
avg/min/max/std latency (ms), FPS, and speedup vs. the PyTorch baseline.

Usage examples
--------------
# All models × all backends, 100 runs at 640×640
python scripts/benchmark_models.py

# Quick single-model comparison
python scripts/benchmark_models.py --models yolov8 --backends pytorch torchscript onnx --runs 50

# Multi-size sweep — tests 320, 640, 1280
python scripts/benchmark_models.py --sizes 320 640 1280 --runs 100

# Save results as CSV and JSON
python scripts/benchmark_models.py --output results/benchmark.csv

# Warmup-heavy run for stable GPU numbers
python scripts/benchmark_models.py --warmup 20 --runs 200
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add backend to sys.path so app.* imports work without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark object detection inference latency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["yolov8", "yolov5"],
        choices=["yolov8", "yolov5"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--backends", nargs="+",
        default=["pytorch", "torchscript", "onnx"],
        choices=["pytorch", "torchscript", "onnx"],
        help="Inference backends to benchmark",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[640],
        metavar="N",
        help="One or more input image sizes (e.g. 320 640 1280)",
    )
    parser.add_argument("--runs", type=int, default=100, help="Timed inference runs")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs (excluded from timing)")
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Save results to .csv or .json (auto-detected by extension)",
    )
    parser.add_argument(
        "--baseline", default="pytorch",
        choices=["pytorch", "torchscript", "onnx"],
        help="Backend used as speedup baseline (1.00×)",
    )
    return parser.parse_args()


# ── Formatting ────────────────────────────────────────────────────────────


def print_table(rows: list[dict], image_size: int, num_runs: int) -> None:
    """Pretty-print a benchmark result table to stdout."""
    print(f"\n{'─'*80}")
    print(f"  Benchmark Results  │  Image: {image_size}×{image_size}  │  Runs: {num_runs}")
    print(f"{'─'*80}")
    header = (
        f"  {'Model':<10} {'Backend':<14} "
        f"{'Avg ms':>9} {'Min ms':>9} {'Max ms':>9} {'Std ms':>9} "
        f"{'FPS':>7} {'Speedup':>8}  {'Status':<6}"
    )
    print(header)
    print(f"{'─'*80}")
    for r in rows:
        if r["status"] == "ok":
            speedup_str = f"{r['speedup']:.2f}×" if r["speedup"] is not None else "  —  "
            print(
                f"  {r['model_name']:<10} {r['backend_type']:<14} "
                f"{r['avg_latency_ms']:>9.2f} {r['min_latency_ms']:>9.2f} "
                f"{r['max_latency_ms']:>9.2f} {r['std_latency_ms']:>9.2f} "
                f"{r['fps']:>7.1f} {speedup_str:>8}  ok"
            )
        else:
            err = (r.get("error") or "failed")[:30]
            print(f"  {r['model_name']:<10} {r['backend_type']:<14}  {'— error —':>43}  {err}")
    print(f"{'─'*80}")
    # Highlight best
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        best = min(ok_rows, key=lambda r: r["avg_latency_ms"])
        print(f"  ✓ Fastest: {best['model_name']}/{best['backend_type']} — "
              f"{best['avg_latency_ms']:.2f} ms  ({best['fps']:.1f} FPS)")
    print()


def save_results(all_rows: list[dict], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()

    if ext == ".json":
        with open(out, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"[benchmark] Results saved → {out}")

    elif ext in (".csv", ""):
        if ext == "":
            out = out.with_suffix(".csv")
        fieldnames = [
            "image_size", "model_name", "backend_type",
            "avg_latency_ms", "min_latency_ms", "max_latency_ms", "std_latency_ms",
            "fps", "speedup", "num_runs", "status", "error",
        ]
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[benchmark] Results saved → {out}")

    else:
        # Unknown extension — write JSON anyway
        with open(out, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"[benchmark] Results saved as JSON → {out}")


# ── Core logic ────────────────────────────────────────────────────────────


def run_benchmark_for_size(
    model_names: list,
    backend_types: list,
    image_size: int,
    num_runs: int,
    warmup_runs: int,
    baseline_backend: str,
) -> list[dict]:
    """Run benchmark for one image size and return flat row dicts."""
    from app.schemas.detection import BackendType, BenchmarkRequest, ModelName
    from app.services.benchmark import run_benchmark

    request = BenchmarkRequest(
        model_names=[ModelName(m) for m in model_names],
        backend_types=[BackendType(b) for b in backend_types],
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        image_size=image_size,
    )

    result = run_benchmark(request)

    # Build a lookup for speedup calculation: model → pytorch baseline latency
    baseline_latency: dict[str, Optional[float]] = {}
    for entry in result.results:
        if entry.backend_type == baseline_backend and entry.status == "ok":
            baseline_latency[entry.model_name] = entry.avg_latency_ms

    rows: list[dict] = []
    for entry in result.results:
        base = baseline_latency.get(entry.model_name)
        speedup = (base / entry.avg_latency_ms) if (base and entry.status == "ok" and entry.avg_latency_ms > 0) else None

        rows.append({
            "image_size": image_size,
            "model_name": entry.model_name,
            "backend_type": entry.backend_type,
            "avg_latency_ms": entry.avg_latency_ms,
            "min_latency_ms": entry.min_latency_ms,
            "max_latency_ms": entry.max_latency_ms,
            "std_latency_ms": entry.std_latency_ms,
            "fps": entry.fps,
            "speedup": speedup,
            "num_runs": num_runs,
            "status": entry.status,
            "error": entry.error,
        })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Object Detection — Inference Benchmark")
    print("=" * 60)
    print(f"  Models:   {args.models}")
    print(f"  Backends: {args.backends}")
    print(f"  Sizes:    {args.sizes}")
    print(f"  Runs:     {args.runs}  (warmup: {args.warmup})")
    print(f"  Baseline: {args.baseline}")
    if args.output:
        print(f"  Output:   {args.output}")
    print()

    all_rows: list[dict] = []

    for size in args.sizes:
        print(f"[benchmark] Running at {size}×{size}…")
        t0 = time.perf_counter()
        rows = run_benchmark_for_size(
            model_names=args.models,
            backend_types=args.backends,
            image_size=size,
            num_runs=args.runs,
            warmup_runs=args.warmup,
            baseline_backend=args.baseline,
        )
        elapsed = time.perf_counter() - t0
        all_rows.extend(rows)
        print_table(rows, size, args.runs)
        print(f"  (completed in {elapsed:.1f}s)\n")

    # Report any errors with actionable tips
    errors = [r for r in all_rows if r["status"] == "error"]
    if errors:
        print("[benchmark] Errors encountered:")
        for e in errors:
            print(f"  ✗ {e['model_name']}/{e['backend_type']}: {e.get('error', 'unknown')}")
        print()
        print("  Tip: TorchScript and ONNX backends require exported model files.")
        print("  Run these first:")
        for m in args.models:
            ext = "torchscript" if "torchscript" in args.backends else ""
            if "torchscript" in args.backends:
                print(f"    python scripts/export_torchscript.py --model {m}")
            if "onnx" in args.backends:
                print(f"    python scripts/export_onnx.py --model {m}")
        print()

    if args.output:
        save_results(all_rows, args.output)


if __name__ == "__main__":
    main()
