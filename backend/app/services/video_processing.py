"""
Video processing service: orchestrates frame-by-frame inference and
annotated video creation via the model registry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.logging import logger
from app.schemas.detection import BackendType, FrameSummary, ModelName, VideoDetectionResponse
from app.services.inference import get_detector


def process_video(
    video_path: str,
    model_name: ModelName,
    backend_type: BackendType,
    confidence_threshold: float,
    iou_threshold: float,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> VideoDetectionResponse:
    """
    Run detection on every frame of a video and return summary metrics.

    Args:
        video_path:   Path to the input video file.
        model_name:   Which detector to use.
        backend_type: Inference backend.
        confidence_threshold / iou_threshold: NMS parameters.
        output_path:  If provided, write the annotated video here.
        max_frames:   Limit processing to first N frames (useful for demos).

    Returns:
        VideoDetectionResponse with per-frame summaries and aggregate metrics.
    """
    detector = get_detector(model_name, backend_type, confidence_threshold, iou_threshold)
    logger.info(
        "Video processing | model=%s | backend=%s | input=%s",
        model_name, backend_type, video_path,
    )

    result = detector.predict_video(
        video_path=video_path,
        output_path=output_path,
        max_frames=max_frames,
    )

    frame_summaries = [
        FrameSummary(
            frame_index=fs["frame_index"],
            detections=fs["detections"],
            latency_ms=fs["latency_ms"],
        )
        for fs in result["frames_summary"]
    ]

    return VideoDetectionResponse(
        model_name=model_name.value,
        backend_type=backend_type.value,
        frame_count=result["frame_count"],
        average_fps=result["average_fps"],
        total_latency_ms=result["total_latency_ms"],
        average_latency_per_frame_ms=result["average_latency_per_frame_ms"],
        total_detections=result["total_detections"],
        output_path=result.get("output_path"),
        frames_summary=frame_summaries,
    )
