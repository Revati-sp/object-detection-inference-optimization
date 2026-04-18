"""
Detection routes:
  POST /detect/image  — run inference on an uploaded image
  POST /detect/video  — run inference on an uploaded video
  GET  /models        — list available model/backend combinations
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.detection import (
    BackendType,
    DetectionResponse,
    ModelName,
    VideoDetectionResponse,
)
from app.services.inference import list_models, run_image_inference
from app.services.video_processing import process_video
from app.utils.image import load_image_from_bytes

router = APIRouter(prefix="/api", tags=["Detection"])
settings = get_settings()

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"}
_ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo",
                         "video/x-matroska", "video/webm"}


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------

@router.get("/models", summary="List available models and backends")
def get_models():
    """Return all supported (model, backend) pairs and their load status."""
    return {"models": list_models()}


# ---------------------------------------------------------------------------
# Image detection
# ---------------------------------------------------------------------------

@router.post(
    "/detect/image",
    response_model=DetectionResponse,
    summary="Run object detection on an uploaded image",
)
async def detect_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, WebP)"),
    model_name: ModelName = Form(ModelName.yolov8, description="Detector model"),
    backend_type: BackendType = Form(BackendType.pytorch, description="Inference backend"),
    confidence_threshold: float = Form(0.25, ge=0.0, le=1.0),
    iou_threshold: float = Form(0.45, ge=0.0, le=1.0),
    save_result: bool = Form(False, description="Save annotated image to outputs/"),
):
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: {_ALLOWED_IMAGE_TYPES}",
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = load_image_from_bytes(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Optionally persist the upload for debugging
    if save_result:
        safe_name = f"{uuid.uuid4().hex}_{Path(file.filename or 'image').name}"
        upload_path = settings.UPLOAD_DIR / safe_name
        with open(upload_path, "wb") as f_out:
            f_out.write(raw)
        logger.info("Saved upload: %s", upload_path)

    try:
        response = run_image_inference(
            image=image,
            model_name=model_name,
            backend_type=backend_type,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (RuntimeError, PermissionError, OSError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during image inference")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    logger.info(
        "Image inference | model=%s | backend=%s | dets=%d | latency=%.1fms",
        model_name, backend_type, response.total_detections, response.latency_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Video detection
# ---------------------------------------------------------------------------

@router.post(
    "/detect/video",
    response_model=VideoDetectionResponse,
    summary="Run object detection on an uploaded video",
)
async def detect_video(
    file: UploadFile = File(..., description="Video file"),
    model_name: ModelName = Form(ModelName.yolov8, description="Detector model"),
    backend_type: BackendType = Form(BackendType.pytorch, description="Inference backend"),
    confidence_threshold: float = Form(0.25, ge=0.0, le=1.0),
    iou_threshold: float = Form(0.45, ge=0.0, le=1.0),
    save_output_video: bool = Form(True, description="Save annotated video to outputs/"),
    max_frames: Optional[int] = Form(None, description="Limit frames processed (None = all)"),
):
    content_type = file.content_type or ""
    # Some clients send application/octet-stream for video; be lenient
    if content_type and content_type not in _ALLOWED_VIDEO_TYPES and \
       content_type != "application/octet-stream":
        logger.warning("Unusual video content-type: %s — proceeding anyway", content_type)

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")

    # Save upload to disk (required for OpenCV VideoCapture)
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename or 'video.mp4').name}"
    upload_path = settings.UPLOAD_DIR / safe_name
    with open(upload_path, "wb") as f_out:
        f_out.write(raw)

    output_path: Optional[str] = None
    if save_output_video:
        out_name = f"annotated_{safe_name}"
        output_path = str(settings.OUTPUT_DIR / out_name)

    try:
        response = process_video(
            video_path=str(upload_path),
            model_name=model_name,
            backend_type=backend_type,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            output_path=output_path,
            max_frames=max_frames,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (RuntimeError, PermissionError, OSError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during video inference")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    logger.info(
        "Video inference | model=%s | backend=%s | frames=%d | avg_fps=%.1f",
        model_name, backend_type, response.frame_count, response.average_fps,
    )
    return response
