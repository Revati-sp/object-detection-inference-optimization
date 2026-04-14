from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models.base import BaseDetector
from app.schemas.detection import BackendType, BoundingBox, Detection
from app.utils.image import (
    draw_detections,
    preprocess_for_onnx,
    scale_boxes_back,
    save_image,
)
from app.utils.video import VideoWriter, get_video_properties, iter_frames
from app.core.logging import logger


# COCO 80-class names shipped with YOLOv8
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Pure-NumPy NMS. Used as a fallback when torchvision is unavailable."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        idx = np.where(iou <= iou_threshold)[0]
        order = order[idx + 1]
    return keep


class YOLOv8Detector(BaseDetector):
    """
    YOLOv8 detector supporting three inference backends:

    • pytorch    — Ultralytics YOLO Python API (default; handles pre/post internally)
    • torchscript — torch.jit.load on an exported .torchscript file
    • onnx       — onnxruntime.InferenceSession on an exported .onnx file
                   (set CUDAExecutionProvider in config for GPU acceleration)
    """

    def __init__(
        self,
        backend_type: BackendType = BackendType.pytorch,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
    ) -> None:
        super().__init__(
            model_name="yolov8",
            backend_type=backend_type,
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        self.device = self._resolve_device()
        logger.info("YOLOv8 loading | backend=%s | device=%s", self.backend_type, self.device)

        if self.backend_type == BackendType.pytorch:
            self._load_pytorch()
        elif self.backend_type == BackendType.torchscript:
            self._load_torchscript()
        elif self.backend_type == BackendType.onnx:
            self._load_onnx()
        else:
            raise ValueError(f"Unsupported backend: {self.backend_type}")

        # Prefer model's own class list; fall back to COCO
        self.class_names = self._get_class_names()
        logger.info("YOLOv8 ready | classes=%d", len(self.class_names))

    def _load_pytorch(self) -> None:
        from ultralytics import YOLO
        weights = self.weights_path or "yolov8n.pt"
        self.model = YOLO(weights)
        if self.device == "cuda":
            self.model.to("cuda")

    def _load_torchscript(self) -> None:
        import torch
        path = self.weights_path or "weights/yolov8n.torchscript"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"TorchScript file not found: {path}\n"
                "Run: python scripts/export_torchscript.py --model yolov8"
            )
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        path = self.weights_path or "weights/yolov8n.onnx"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"ONNX file not found: {path}\n"
                "Run: python scripts/export_onnx.py --model yolov8"
            )

        # Prefer CUDA, automatically fall back to CPU if unavailable.
        # To enable GPU: set ONNX_EXECUTION_PROVIDER=CUDAExecutionProvider in .env
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        logger.info("YOLOv8 ONNX providers selected: %s", providers)
        self.ort_session = ort.InferenceSession(path, providers=providers)

    def _get_class_names(self) -> List[str]:
        if self.backend_type == BackendType.pytorch and self.model is not None:
            try:
                names = self.model.names
                if isinstance(names, dict):
                    return [names[i] for i in sorted(names.keys())]
                return list(names)
            except Exception:
                pass
        return COCO_CLASSES

    # ------------------------------------------------------------------
    # Predict image
    # ------------------------------------------------------------------

    def predict_image(
        self, image: np.ndarray
    ) -> Tuple[List[Detection], Dict[str, float]]:
        if self.backend_type == BackendType.pytorch:
            return self._predict_pytorch(image)
        elif self.backend_type == BackendType.torchscript:
            return self._predict_torchscript(image)
        elif self.backend_type == BackendType.onnx:
            return self._predict_onnx(image)
        raise ValueError(f"Unsupported backend: {self.backend_type}")

    # --- PyTorch path ---

    def _predict_pytorch(self, image: np.ndarray):
        t0 = time.perf_counter()
        # Ultralytics handles resizing and normalization internally
        import torch
        t_pre = time.perf_counter()

        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            verbose=False,
            device=self.device,
        )
        t_inf = time.perf_counter()

        detections = self._parse_ultralytics_results(results, image.shape)
        t_post = time.perf_counter()

        timing = {
            "preprocessing_ms": (t_pre - t0) * 1000,
            "inference_ms": (t_inf - t_pre) * 1000,
            "postprocessing_ms": (t_post - t_inf) * 1000,
        }
        return detections, timing

    def _parse_ultralytics_results(self, results, orig_shape) -> List[Detection]:
        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = (
                    self.class_names[cls_id]
                    if cls_id < len(self.class_names)
                    else str(cls_id)
                )
                detections.append(
                    Detection(
                        bbox=BoundingBox(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            width=x2 - x1, height=y2 - y1,
                        ),
                        label=label,
                        class_id=cls_id,
                        confidence=conf,
                    )
                )
        return detections

    # --- TorchScript path ---

    def _predict_torchscript(self, image: np.ndarray):
        import torch

        t0 = time.perf_counter()
        blob, scale, padding = preprocess_for_onnx(image, self.image_size)
        input_tensor = torch.from_numpy(blob).to(self.device)
        t_pre = time.perf_counter()

        with torch.no_grad():
            raw = self.model(input_tensor)
        t_inf = time.perf_counter()

        # raw shape: [1, 84, 8400] for YOLOv8 on 80-class COCO
        output = raw[0].cpu().numpy() if isinstance(raw, (list, tuple)) else raw.cpu().numpy()
        detections = self._postprocess_yolov8_output(output, scale, padding, image.shape)
        t_post = time.perf_counter()

        timing = {
            "preprocessing_ms": (t_pre - t0) * 1000,
            "inference_ms": (t_inf - t_pre) * 1000,
            "postprocessing_ms": (t_post - t_inf) * 1000,
        }
        return detections, timing

    # --- ONNX path ---

    def _predict_onnx(self, image: np.ndarray):
        t0 = time.perf_counter()
        blob, scale, padding = preprocess_for_onnx(image, self.image_size)
        t_pre = time.perf_counter()

        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: blob})
        t_inf = time.perf_counter()

        # YOLOv8 ONNX output: [1, 84, 8400]
        raw = outputs[0]
        detections = self._postprocess_yolov8_output(raw, scale, padding, image.shape)
        t_post = time.perf_counter()

        timing = {
            "preprocessing_ms": (t_pre - t0) * 1000,
            "inference_ms": (t_inf - t_pre) * 1000,
            "postprocessing_ms": (t_post - t_inf) * 1000,
        }
        return detections, timing

    def _postprocess_yolov8_output(
        self,
        raw: np.ndarray,       # [1, 84, 8400]
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple,
    ) -> List[Detection]:
        """
        Decode YOLOv8 raw ONNX output into Detection objects.
        Output layout: [cx, cy, w, h, class_0_score, ..., class_79_score]
        """
        pred = raw[0]          # [84, 8400]
        pred = pred.T          # [8400, 84]

        boxes_cxcywh = pred[:, :4]
        class_scores = pred[:, 4:]

        # Convert centre-format to corner-format
        cx, cy, bw, bh = (
            boxes_cxcywh[:, 0], boxes_cxcywh[:, 1],
            boxes_cxcywh[:, 2], boxes_cxcywh[:, 3],
        )
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Best class per anchor
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        # Confidence filter
        mask = confidences >= self.confidence_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # NMS per class
        keep_indices: List[int] = []
        for cid in np.unique(class_ids):
            cls_mask = class_ids == cid
            cls_boxes = boxes[cls_mask]
            cls_scores = confidences[cls_mask]
            cls_orig_idx = np.where(cls_mask)[0]
            kept = _nms_numpy(cls_boxes, cls_scores, self.iou_threshold)
            keep_indices.extend(cls_orig_idx[kept].tolist())

        boxes = boxes[keep_indices]
        confidences = confidences[keep_indices]
        class_ids = class_ids[keep_indices]

        # Map back to original image coordinates
        boxes_orig = scale_boxes_back(boxes, scale, padding, orig_shape)

        detections: List[Detection] = []
        for box, conf, cls_id in zip(boxes_orig, confidences, class_ids):
            label = (
                self.class_names[cls_id]
                if cls_id < len(self.class_names)
                else str(cls_id)
            )
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            detections.append(
                Detection(
                    bbox=BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        width=x2 - x1, height=y2 - y1,
                    ),
                    label=label,
                    class_id=int(cls_id),
                    confidence=float(conf),
                )
            )
        return detections

    # ------------------------------------------------------------------
    # Predict video
    # ------------------------------------------------------------------

    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        props = get_video_properties(video_path)
        source_fps = props["fps"] or 25.0

        frames_summary = []
        total_latency = 0.0
        total_dets = 0

        writer: Optional[VideoWriter] = None
        if output_path:
            writer = VideoWriter(
                output_path,
                fps=source_fps,
                width=props["width"],
                height=props["height"],
            )

        try:
            for frame_idx, frame in iter_frames(video_path, max_frames=max_frames):
                t_start = time.perf_counter()
                dets, timing = self.predict_image(frame)
                elapsed = (time.perf_counter() - t_start) * 1000.0

                frames_summary.append({
                    "frame_index": frame_idx,
                    "detections": len(dets),
                    "latency_ms": elapsed,
                })
                total_latency += elapsed
                total_dets += len(dets)

                if writer:
                    annotated = draw_detections(frame, dets)
                    writer.write(annotated)
        finally:
            if writer:
                writer.release()

        frame_count = len(frames_summary)
        avg_latency = total_latency / frame_count if frame_count else 0.0
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        return {
            "frame_count": frame_count,
            "average_fps": avg_fps,
            "total_latency_ms": total_latency,
            "average_latency_per_frame_ms": avg_latency,
            "total_detections": total_dets,
            "output_path": output_path,
            "frames_summary": frames_summary,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_torchscript(self, output_path: str) -> str:
        """Export YOLOv8 to TorchScript using Ultralytics API."""
        from ultralytics import YOLO
        weights = self.weights_path or "yolov8n.pt"
        model = YOLO(weights)
        saved = model.export(format="torchscript", imgsz=self.image_size)
        # Move to requested output path if different from Ultralytics default
        saved_path = Path(str(saved))
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if saved_path != out:
            saved_path.rename(out)
        logger.info("YOLOv8 TorchScript exported to %s", out)
        return str(out)

    def export_onnx(self, output_path: str) -> str:
        """Export YOLOv8 to ONNX using Ultralytics API."""
        from ultralytics import YOLO
        weights = self.weights_path or "yolov8n.pt"
        model = YOLO(weights)
        saved = model.export(
            format="onnx",
            imgsz=self.image_size,
            opset=12,
            simplify=False,  # set True only if onnxsim is installed: pip install onnxsim
        )
        saved_path = Path(str(saved))
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if saved_path != out:
            saved_path.rename(out)
        logger.info("YOLOv8 ONNX exported to %s", out)
        return str(out)
