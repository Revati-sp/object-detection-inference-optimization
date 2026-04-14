"""
Evaluation and benchmark routes:
  POST /evaluate   — compute mAP on a COCO-annotated dataset
  POST /benchmark  — compare latency/FPS across model/backend combos
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.logging import logger
from app.schemas.detection import (
    BenchmarkRequest,
    BenchmarkResult,
    EvaluationRequest,
    EvaluationResult,
)
from app.services.benchmark import run_benchmark
from app.services.evaluation import evaluate_dataset

router = APIRouter(prefix="/api", tags=["Evaluation"])


@router.post(
    "/evaluate",
    response_model=EvaluationResult,
    summary="Compute mAP on a COCO-annotated dataset",
)
def evaluate(request: EvaluationRequest) -> EvaluationResult:
    """
    Run inference on every image listed in the COCO annotations file, then
    compute mAP@0.5 and mAP@0.5:0.95 using pycocotools.

    **Provide paths relative to the backend working directory or absolute paths.**
    """
    try:
        result = evaluate_dataset(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "Evaluation complete | mAP@0.5=%.4f | mAP@0.5:0.95=%.4f | images=%d",
        result.map_50, result.map_50_95, result.num_images,
    )
    return result


@router.post(
    "/benchmark",
    response_model=BenchmarkResult,
    summary="Benchmark inference latency and FPS across model/backend combinations",
)
def benchmark(request: BenchmarkRequest) -> BenchmarkResult:
    """
    Run a synthetic latency benchmark (random dummy images) for each requested
    (model, backend) pair.  Warmup passes are executed first to stabilize
    measurements and trigger JIT compilation.
    """
    try:
        result = run_benchmark(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result
