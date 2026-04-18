#!/usr/bin/env python3
"""
Run object detection inference on a video file and save:
  • annotated output video
  • per-frame CSV with latency data
  • summary JSON with FPS / totals

Runs all requested model × backend combinations so you get comparative data.

Usage
-----
# Single model / backend
python scripts/run_video_inference.py \\
    --video data/videos/sample.mp4 \\
    --model yolov8 --backend pytorch

# All 6 combinations (2 models × 3 backends)
python scripts/run_video_inference.py \\
    --video data/videos/sample.mp4 \\
    --compare \\
    --max-frames 150

# Quick smoke test on first 30 frames
python scripts/run_video_inference.py \\
    --video data/videos/sample.mp4 \\
    --model yolov8 --backend pytorch \\
    --max-frames 30

Notes
-----
• Annotated videos are written to outputs/
• Summary JSON + CSV written to results/video_benchmark.json and .csv
• If no video is available, pass --generate-test-video to create a synthetic one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Video object detection inference with latency benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", default=None,
                   help="Path to input video file")
    p.add_argument("--generate-test-video", action="store_true",
                   help="Create a synthetic 10-second test video if --video is not given")
    p.add_argument("--model", nargs="+",
                   choices=["yolov8", "yolov5"],
                   default=["yolov8"],
                   help="Model(s) to run")
    p.add_argument("--backend", nargs="+",
                   choices=["pytorch", "torchscript", "onnx"],
                   default=["pytorch"],
                   help="Backend(s) to run")
    p.add_argument("--compare", action="store_true",
                   help="Run ALL model × backend combos (overrides --model / --backend)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit inference to first N frames (None = full video)")
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--output-dir", default="outputs",
                   help="Directory for annotated output videos")
    p.add_argument("--results-dir", default="results",
                   help="Directory for CSV/JSON summary files")
    p.add_argument("--no-save-video", action="store_true",
                   help="Skip writing annotated video (faster, saves disk)")
    return p.parse_args()


# ── Synthetic video generator ──────────────────────────────────────────────

def generate_test_video(output_path: Path, fps: int = 25, duration: int = 10) -> Path:
    """Create a simple synthetic video with moving shapes for smoke testing."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV is required to generate a test video: pip install opencv-python-headless")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = fps * duration
    print(f"  Generating {total_frames}-frame synthetic video ({duration}s @ {fps}fps) …")

    rng = __import__("random")
    rng.seed(42)

    for i in range(total_frames):
        frame = __import__("numpy").zeros((height, width, 3), dtype=__import__("numpy").uint8)
        frame[:] = (20, 20, 30)  # dark background

        # Bounce a few colored rectangles around
        for j in range(4):
            t = i / fps
            x = int((width - 100) * abs(__import__("math").sin(t * (j + 1) * 0.7 + j)))
            y = int((height - 80) * abs(__import__("math").cos(t * (j + 1) * 0.5 + j * 2)))
            color = [(200, 80, 80), (80, 200, 80), (80, 80, 200), (200, 200, 80)][j]
            import cv2 as _cv2
            _cv2.rectangle(frame, (x, y), (x + 100, y + 80), color, -1)

        import cv2 as _cv2
        _cv2.putText(frame, f"Frame {i:04d}", (10, 30),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        writer.write(frame)

    writer.release()
    print(f"  Synthetic test video → {output_path}")
    return output_path


# ── Core inference ─────────────────────────────────────────────────────────

def run_video_inference_one(
    video_path: str,
    model_name: str,
    backend_type: str,
    confidence: float,
    iou: float,
    output_path: Optional[str],
    max_frames: Optional[int],
) -> dict:
    """Run inference for one (model, backend) combo. Returns summary dict."""
    from app.schemas.detection import BackendType, ModelName
    from app.services.inference import get_detector

    print(f"\n  [{model_name}/{backend_type}] Loading detector …")
    try:
        detector = get_detector(
            ModelName(model_name),
            BackendType(backend_type),
            confidence,
            iou,
        )
    except Exception as exc:
        print(f"  ✗ Could not load detector: {exc}")
        return {
            "model_name": model_name,
            "backend_type": backend_type,
            "status": "error",
            "error": str(exc),
            "frame_count": 0,
            "average_fps": 0.0,
            "average_latency_ms": 0.0,
            "total_detections": 0,
            "output_path": None,
        }

    print(f"  Running inference (max_frames={max_frames}) …")
    t_wall_start = time.perf_counter()
    result = detector.predict_video(
        video_path=video_path,
        output_path=output_path,
        max_frames=max_frames,
    )
    wall_s = time.perf_counter() - t_wall_start

    summary = {
        "model_name": model_name,
        "backend_type": backend_type,
        "status": "ok",
        "error": None,
        "frame_count": result["frame_count"],
        "average_fps": result["average_fps"],
        "average_latency_ms": result["average_latency_per_frame_ms"],
        "total_latency_ms": result["total_latency_ms"],
        "total_detections": result["total_detections"],
        "wall_time_s": wall_s,
        "output_path": result.get("output_path"),
        "frames_summary": result.get("frames_summary", []),
    }

    print(
        f"  ✓ Done | frames={summary['frame_count']} | "
        f"avg_latency={summary['average_latency_ms']:.1f}ms | "
        f"avg_fps={summary['average_fps']:.1f} | "
        f"total_dets={summary['total_detections']}"
    )
    if output_path:
        print(f"    Annotated video → {output_path}")

    return summary


# ── Reporting ──────────────────────────────────────────────────────────────

def print_table(rows: list[dict]) -> None:
    print(f"\n{'─'*80}")
    print(f"  Video Inference Results")
    print(f"{'─'*80}")
    print(
        f"  {'Model':<10} {'Backend':<14} {'Frames':>7} {'Avg ms':>9} "
        f"{'Avg FPS':>9} {'Total Det':>10}  Status"
    )
    print(f"{'─'*80}")
    for r in rows:
        if r["status"] == "ok":
            print(
                f"  {r['model_name']:<10} {r['backend_type']:<14} "
                f"{r['frame_count']:>7} {r['average_latency_ms']:>9.2f} "
                f"{r['average_fps']:>9.1f} {r['total_detections']:>10}  ok"
            )
        else:
            err = (r.get("error") or "unknown")[:35]
            print(f"  {r['model_name']:<10} {r['backend_type']:<14}  {'— error —':>37}  {err}")
    print(f"{'─'*80}\n")


def save_results(rows: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON (includes per-frame detail)
    json_path = results_dir / "video_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Results JSON → {json_path}")

    # CSV (flat summary, one row per combo)
    csv_path = results_dir / "video_benchmark.csv"
    fields = [
        "model_name", "backend_type", "frame_count",
        "average_fps", "average_latency_ms", "total_detections",
        "total_latency_ms", "wall_time_s", "status", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results CSV  → {csv_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve video path
    video_path: Optional[str] = args.video
    if video_path is None:
        if args.generate_test_video:
            test_video = Path("data/videos/test_video.mp4")
            generate_test_video(test_video)
            video_path = str(test_video)
        else:
            print("Error: Provide --video <path> or use --generate-test-video")
            print("\nExample:")
            print("  python scripts/run_video_inference.py --generate-test-video --compare")
            sys.exit(1)

    if not Path(video_path).exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    # Build combos
    if args.compare:
        combos = [
            (m, b)
            for m in ["yolov8", "yolov5"]
            for b in ["pytorch", "torchscript", "onnx"]
        ]
    else:
        combos = [(m, b) for m in args.model for b in args.backend]

    out_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem

    print("=" * 60)
    print("  Video Object Detection Inference")
    print("=" * 60)
    print(f"  Video     : {video_path}")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Combos    : {len(combos)}")
    print()

    all_rows: list[dict] = []

    for model_name, backend_type in combos:
        # Decide output video path
        if args.no_save_video:
            out_video: Optional[str] = None
        else:
            out_video = str(
                out_dir / f"annotated_{video_stem}_{model_name}_{backend_type}.mp4"
            )

        row = run_video_inference_one(
            video_path=video_path,
            model_name=model_name,
            backend_type=backend_type,
            confidence=args.confidence,
            iou=args.iou,
            output_path=out_video,
            max_frames=args.max_frames,
        )
        all_rows.append(row)

    print_table(all_rows)
    save_results(all_rows, results_dir)


if __name__ == "__main__":
    import math  # needed for synthetic video generation
    main()
