from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.schemas.detection import BackendType, Detection


class BaseDetector(ABC):
    """
    Common interface that every detector must implement.
    Subclasses override load/predict/export methods while sharing
    the same output schema so services can treat all models uniformly.
    """

    def __init__(
        self,
        model_name: str,
        backend_type: BackendType = BackendType.pytorch,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
    ) -> None:
        self.model_name = model_name
        self.backend_type = backend_type
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size

        self.model = None          # PyTorch / TorchScript model handle
        self.ort_session = None    # onnxruntime.InferenceSession handle
        self.class_names: List[str] = []
        self.device: str = "cpu"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """
        Load weights into memory for the configured backend_type.
        After this call, is_loaded() must return True.
        """

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @abstractmethod
    def predict_image(
        self, image: np.ndarray
    ) -> Tuple[List[Detection], dict]:
        """
        Run inference on a single BGR image (H x W x 3, uint8).

        Returns:
            detections: list of Detection pydantic objects
            timing: dict with keys preprocessing_ms, inference_ms, postprocessing_ms
        """

    @abstractmethod
    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        """
        Run inference on every frame of a video.

        Returns a summary dict with frame_count, average_fps,
        total_latency_ms, frames_summary, output_path.
        """

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @abstractmethod
    def export_torchscript(self, output_path: str) -> str:
        """Export to TorchScript (.torchscript). Returns the saved file path."""

    @abstractmethod
    def export_onnx(self, output_path: str) -> str:
        """Export to ONNX (.onnx). Returns the saved file path."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self.model is not None or self.ort_session is not None

    def _resolve_device(self) -> str:
        """Return 'cuda' if available, else 'cpu'."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
