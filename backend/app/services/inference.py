"""
Inference service: model registry, lazy loading, and unified dispatch.

All detectors are instantiated once per (model_name, backend_type) pair and
reused across requests. Loading is deferred until first use.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.logging import logger
from app.models.base import BaseDetector
from app.models.yolov8_detector import YOLOv8Detector
from app.models.yolov5_detector import YOLOv5Detector
from app.schemas.detection import (
    BackendType,
    BoundingBox,
    Detection,
    DetectionResponse,
    ModelName,
)
from app.utils.timing import TimingResult, timer

settings = get_settings()

# Registry key: (ModelName, BackendType)
_registry: Dict[Tuple[str, str], BaseDetector] = {}


def _make_detector(model_name: ModelName, backend_type: BackendType) -> BaseDetector:
    """Factory: create and load the appropriate detector."""
    if model_name == ModelName.yolov8:
        weights = None
        if backend_type == BackendType.pytorch:
            weights = settings.YOLOV8_WEIGHTS
        elif backend_type == BackendType.torchscript:
            weights = settings.YOLOV8_TORCHSCRIPT_PATH
        elif backend_type == BackendType.onnx:
            weights = settings.YOLOV8_ONNX_PATH
        return YOLOv8Detector(
            backend_type=backend_type,
            weights_path=weights,
            confidence_threshold=settings.DEFAULT_CONFIDENCE,
            iou_threshold=settings.DEFAULT_IOU,
            image_size=settings.DEFAULT_IMAGE_SIZE,
        )
    elif model_name == ModelName.yolov5:
        weights = None
        if backend_type == BackendType.pytorch:
            # Use local .pt file if YOLOV5_WEIGHTS looks like a path, else hub variant
            weights = settings.YOLOV5_WEIGHTS if "/" in settings.YOLOV5_WEIGHTS else None
        elif backend_type == BackendType.torchscript:
            weights = settings.YOLOV5_TORCHSCRIPT_PATH
        elif backend_type == BackendType.onnx:
            weights = settings.YOLOV5_ONNX_PATH
        # model_variant is only used for torch.hub auto-download (no local .pt)
        hub_variant = settings.YOLOV5_WEIGHTS if "/" not in settings.YOLOV5_WEIGHTS else "yolov5s"
        return YOLOv5Detector(
            backend_type=backend_type,
            weights_path=weights,
            model_variant=hub_variant,
            confidence_threshold=settings.DEFAULT_CONFIDENCE,
            iou_threshold=settings.DEFAULT_IOU,
            image_size=settings.DEFAULT_IMAGE_SIZE,
        )
    raise ValueError(f"Unknown model: {model_name}")


def get_detector(
    model_name: ModelName,
    backend_type: BackendType,
    confidence_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
) -> BaseDetector:
    """
    Retrieve a loaded detector from the registry, instantiating and loading
    it on first access.  Per-request confidence/IoU overrides are applied
    without rebuilding the whole detector.
    """
    key = (model_name.value, backend_type.value)
    if key not in _registry:
        detector = _make_detector(model_name, backend_type)
        detector.load()
        _registry[key] = detector
        logger.info("Detector registered: %s/%s", model_name, backend_type)

    detector = _registry[key]

    # Apply per-request overrides (cheap; no model reload needed)
    if confidence_threshold is not None:
        detector.confidence_threshold = confidence_threshold
        if backend_type == BackendType.pytorch and hasattr(detector.model, "conf"):
            detector.model.conf = confidence_threshold
    if iou_threshold is not None:
        detector.iou_threshold = iou_threshold
        if backend_type == BackendType.pytorch and hasattr(detector.model, "iou"):
            detector.model.iou = iou_threshold

    return detector


def list_models() -> List[dict]:
    """Return metadata for all supported model/backend combinations."""
    combos = []
    for model in ModelName:
        for backend in BackendType:
            key = (model.value, backend.value)
            combos.append({
                "model_name": model.value,
                "backend_type": backend.value,
                "loaded": key in _registry,
            })
    return combos


def run_image_inference(
    image: np.ndarray,
    model_name: ModelName,
    backend_type: BackendType,
    confidence_threshold: float,
    iou_threshold: float,
) -> DetectionResponse:
    """High-level function: run inference and wrap in a DetectionResponse."""
    detector = get_detector(model_name, backend_type, confidence_threshold, iou_threshold)

    t = TimingResult()
    with timer(t, "total"):
        detections, timing = detector.predict_image(image)

    h, w = image.shape[:2]
    return DetectionResponse(
        model_name=model_name.value,
        backend_type=backend_type.value,
        image_width=w,
        image_height=h,
        detections=detections,
        total_detections=len(detections),
        latency_ms=t.get("total"),
        preprocessing_ms=timing.get("preprocessing_ms", 0.0),
        inference_ms=timing.get("inference_ms", 0.0),
        postprocessing_ms=timing.get("postprocessing_ms", 0.0),
    )
