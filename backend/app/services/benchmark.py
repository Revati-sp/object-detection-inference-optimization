"""
Benchmark service: compare inference latency and FPS across model/backend combos
using synthetic random images to isolate model performance from I/O.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from app.core.logging import logger
from app.schemas.detection import (
    BackendType,
    BenchmarkEntry,
    BenchmarkRequest,
    BenchmarkResult,
    ModelName,
)
from app.services.inference import get_detector


def run_benchmark(request: BenchmarkRequest) -> BenchmarkResult:
    """
    For each (model, backend) pair:
    1. Load the detector (or reuse from registry).
    2. Run *warmup_runs* inference passes to fill GPU pipelines / JIT caches.
    3. Time *num_runs* inference passes on a random image.
    4. Report avg/min/max/std latency and FPS.

    Using a fixed random image ensures all models are compared fairly on
    pure inference cost, independent of image content.
    """
    results: List[BenchmarkEntry] = []
    # Synthetic image: same random seed across models for reproducibility
    rng = np.random.default_rng(seed=42)
    dummy_image = (
        rng.integers(0, 255, (request.image_size, request.image_size, 3), dtype=np.uint8)
    )

    for model_name in request.model_names:
        for backend_type in request.backend_types:
            entry = _benchmark_one(
                model_name=model_name,
                backend_type=backend_type,
                image=dummy_image,
                num_runs=request.num_runs,
                warmup_runs=request.warmup_runs,
                image_size=request.image_size,
            )
            results.append(entry)
            logger.info(
                "Benchmark | %s/%s | avg=%.2fms | fps=%.1f | status=%s",
                model_name, backend_type, entry.avg_latency_ms, entry.fps, entry.status,
            )

    return BenchmarkResult(
        results=results,
        image_size=request.image_size,
        num_runs=request.num_runs,
        warmup_runs=request.warmup_runs,
    )


def _benchmark_one(
    model_name: ModelName,
    backend_type: BackendType,
    image: np.ndarray,
    num_runs: int,
    warmup_runs: int,
    image_size: int,
) -> BenchmarkEntry:
    try:
        detector = get_detector(model_name, backend_type)
    except Exception as exc:
        logger.warning("Cannot load %s/%s for benchmark: %s", model_name, backend_type, exc)
        return BenchmarkEntry(
            model_name=model_name.value,
            backend_type=backend_type.value,
            avg_latency_ms=0.0,
            min_latency_ms=0.0,
            max_latency_ms=0.0,
            std_latency_ms=0.0,
            fps=0.0,
            image_size=image_size,
            num_runs=num_runs,
            status="error",
            error=str(exc),
        )

    # Warmup — not timed; fills GPU pipeline and triggers JIT compilation
    for _ in range(warmup_runs):
        try:
            detector.predict_image(image)
        except Exception:
            pass

    latencies: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        try:
            detector.predict_image(image)
        except Exception as exc:
            return BenchmarkEntry(
                model_name=model_name.value,
                backend_type=backend_type.value,
                avg_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                std_latency_ms=0.0,
                fps=0.0,
                image_size=image_size,
                num_runs=num_runs,
                status="error",
                error=str(exc),
            )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    avg = float(arr.mean())
    return BenchmarkEntry(
        model_name=model_name.value,
        backend_type=backend_type.value,
        avg_latency_ms=avg,
        min_latency_ms=float(arr.min()),
        max_latency_ms=float(arr.max()),
        std_latency_ms=float(arr.std()),
        fps=1000.0 / avg if avg > 0 else 0.0,
        image_size=image_size,
        num_runs=num_runs,
        status="ok",
    )
