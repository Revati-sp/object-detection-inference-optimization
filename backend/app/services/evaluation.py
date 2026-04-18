"""
Evaluation service: runs inference on a COCO-annotated image dataset,
converts predictions to COCO detection format, and computes mAP metrics
using pycocotools.

Expected COCO annotations file structure:
{
  "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}, ...],
  "annotations": [...],
  "categories": [{"id": 1, "name": "cat"}, ...]
}
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import numpy as np

from app.core.logging import logger
from app.schemas.detection import BackendType, EvaluationRequest, EvaluationResult, ModelName
from app.services.inference import get_detector
from app.utils.image import load_image_from_path

# COCO pretrained models predict 80-class indices (0-indexed).
# The official COCO dataset uses a *non-contiguous* 91-class ID scheme.
# This mapping converts 0-indexed class IDs → official COCO category IDs.
# Source: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
_COCO80_TO_COCO91: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def evaluate_dataset(request: EvaluationRequest) -> EvaluationResult:
    """
    Run inference across an annotated image dataset and compute COCO mAP.

    Steps:
    1. Load COCO annotations JSON.
    2. For each image, run detector.predict_image().
    3. Save predictions in COCO detection result format.
    4. Use pycocotools COCOeval to compute mAP@0.5 and mAP@0.5:0.95.
    5. Return metrics with latency statistics.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise RuntimeError(
            "pycocotools is required for evaluation. "
            "Install with: pip install pycocotools"
        )

    annotations_path = Path(request.annotations_path)
    images_dir = Path(request.images_dir)

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Load ground-truth annotations
    coco_gt = COCO(str(annotations_path))
    image_ids = coco_gt.getImgIds()

    # Build a mapping: COCO category_id → detector class name (if custom weights)
    # For pretrained models, COCO category IDs follow the standard 1-indexed scheme.
    # We store predictions with the COCO category_id, not the 0-indexed class_id.
    cat_info = coco_gt.loadCats(coco_gt.getCatIds())
    # Map class name → coco category_id for the predictions we write out
    name_to_coco_id = {c["name"]: c["id"] for c in cat_info}

    detector = get_detector(
        request.model_name,
        request.backend_type,
        request.confidence_threshold,
        request.iou_threshold,
    )

    coco_predictions: List[dict] = []
    per_image_latencies: List[float] = []

    logger.info(
        "Evaluating %d images | model=%s | backend=%s",
        len(image_ids), request.model_name, request.backend_type,
    )

    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = images_dir / img_info["file_name"]

        if not img_path.exists():
            logger.warning("Image not found, skipping: %s", img_path)
            continue

        image = load_image_from_path(img_path)

        t_start = time.perf_counter()
        detections, _ = detector.predict_image(image)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        per_image_latencies.append(elapsed_ms)

        for det in detections:
            # Convert to COCO [x, y, width, height] format
            x1, y1 = det.bbox.x1, det.bbox.y1
            w, h = det.bbox.width, det.bbox.height

            # Map detector label → COCO category_id (priority order):
            # 1. Direct name lookup in the GT annotation categories.
            # 2. COCO80→COCO91 mapping (handles non-contiguous official IDs).
            # 3. Raw class_id + 1 last resort.
            coco_cat_id = name_to_coco_id.get(det.label)
            if coco_cat_id is None:
                cls = det.class_id
                if 0 <= cls < len(_COCO80_TO_COCO91):
                    coco_cat_id = _COCO80_TO_COCO91[cls]
                else:
                    coco_cat_id = cls + 1

            coco_predictions.append({
                "image_id": img_id,
                "category_id": coco_cat_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(det.confidence),
            })

    if not coco_predictions:
        logger.warning("No predictions generated — check model weights and confidence threshold.")
        avg_lat = float(np.mean(per_image_latencies)) if per_image_latencies else 0.0
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
        return EvaluationResult(
            model_name=request.model_name.value,
            backend_type=request.backend_type.value,
            num_images=len(per_image_latencies),
            map_50=0.0,
            map_50_95=0.0,
            per_image_latencies_ms=per_image_latencies,
            average_latency_ms=avg_lat,
            fps=fps,
        )

    # Load predictions into COCOeval
    coco_dt = coco_gt.loadRes(coco_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats[0] = AP @ IoU=0.50:0.95 (primary metric)
    # stats[1] = AP @ IoU=0.50
    map_50_95 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])

    avg_latency = float(np.mean(per_image_latencies)) if per_image_latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    logger.info("Evaluation done | mAP@0.5=%.4f | mAP@0.5:0.95=%.4f", map_50, map_50_95)

    return EvaluationResult(
        model_name=request.model_name.value,
        backend_type=request.backend_type.value,
        num_images=len(per_image_latencies),
        map_50=map_50,
        map_50_95=map_50_95,
        per_image_latencies_ms=per_image_latencies,
        average_latency_ms=avg_latency,
        fps=fps,
    )


def save_predictions_json(predictions: List[dict], output_path: str) -> str:
    """
    Persist COCO-format predictions to a JSON file for offline evaluation
    or submission to a leaderboard.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(predictions, f, indent=2)
    return str(out)
