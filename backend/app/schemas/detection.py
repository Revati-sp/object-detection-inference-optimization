from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ModelName(str, Enum):
    yolov8 = "yolov8"
    yolov5 = "yolov5"


class BackendType(str, Enum):
    pytorch = "pytorch"
    torchscript = "torchscript"
    onnx = "onnx"


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left pixel coordinate")
    y1: float = Field(..., description="Top pixel coordinate")
    x2: float = Field(..., description="Right pixel coordinate")
    y2: float = Field(..., description="Bottom pixel coordinate")
    width: float = Field(..., description="Box width in pixels")
    height: float = Field(..., description="Box height in pixels")


class Detection(BaseModel):
    bbox: BoundingBox
    label: str
    class_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Image detection response
# ---------------------------------------------------------------------------

class DetectionResponse(BaseModel):
    model_name: str
    backend_type: str
    image_width: int
    image_height: int
    detections: List[Detection]
    total_detections: int
    latency_ms: float = Field(..., description="End-to-end wall-clock time")
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float


# ---------------------------------------------------------------------------
# Video detection response
# ---------------------------------------------------------------------------

class FrameSummary(BaseModel):
    frame_index: int
    detections: int
    latency_ms: float


class VideoDetectionResponse(BaseModel):
    model_name: str
    backend_type: str
    frame_count: int
    average_fps: float
    total_latency_ms: float
    average_latency_per_frame_ms: float
    total_detections: int
    output_path: Optional[str] = None
    frames_summary: List[FrameSummary]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvaluationRequest(BaseModel):
    model_name: ModelName = ModelName.yolov8
    backend_type: BackendType = BackendType.pytorch
    annotations_path: str = Field(
        ...,
        description="Path to COCO-style annotations JSON file",
        example="data/annotations/instances_val.json",
    )
    images_dir: str = Field(
        ...,
        description="Directory containing evaluation images",
        example="data/images/val",
    )
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    model_name: str
    backend_type: str
    num_images: int
    map_50: float = Field(..., description="mAP @ IoU=0.50")
    map_50_95: float = Field(..., description="mAP @ IoU=0.50:0.95")
    per_image_latencies_ms: List[float]
    average_latency_ms: float
    fps: float


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

class BenchmarkRequest(BaseModel):
    model_names: List[ModelName] = Field(
        default=[ModelName.yolov8, ModelName.yolov5]
    )
    backend_types: List[BackendType] = Field(
        default=[BackendType.pytorch, BackendType.torchscript, BackendType.onnx]
    )
    num_runs: int = Field(100, ge=1)
    image_size: int = Field(640, ge=32)
    warmup_runs: int = Field(10, ge=0)


class BenchmarkEntry(BaseModel):
    model_name: str
    backend_type: str
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    fps: float
    image_size: int
    num_runs: int
    status: str = "ok"
    error: Optional[str] = None


class BenchmarkResult(BaseModel):
    results: List[BenchmarkEntry]
    image_size: int
    num_runs: int
    warmup_runs: int
