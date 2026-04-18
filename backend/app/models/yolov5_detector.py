from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models.base import BaseDetector
from app.schemas.detection import BackendType, BoundingBox, Detection
from app.utils.image import draw_detections, preprocess_for_onnx, scale_boxes_back
from app.utils.video import VideoWriter, get_video_properties, iter_frames
from app.core.logging import logger


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
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


class YOLOv5Detector(BaseDetector):
    """
    YOLOv5 detector supporting three inference backends:

    • pytorch    — torch.hub.load('ultralytics/yolov5', variant)
    • torchscript — torch.jit.load on an exported .torchscript file
    • onnx       — onnxruntime.InferenceSession on an exported .onnx file
                   Output shape: [1, 25200, 85] (4 box coords + 1 objectness + 80 classes)
    """

    def __init__(
        self,
        backend_type: BackendType = BackendType.pytorch,
        weights_path: Optional[str] = None,
        model_variant: str = "yolov5s",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
    ) -> None:
        super().__init__(
            model_name="yolov5",
            backend_type=backend_type,
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
        )
        self.model_variant = model_variant

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        self.device = self._resolve_device()
        logger.info("YOLOv5 loading | backend=%s | device=%s", self.backend_type, self.device)

        if self.backend_type == BackendType.pytorch:
            self._load_pytorch()
        elif self.backend_type == BackendType.torchscript:
            self._load_torchscript()
        elif self.backend_type == BackendType.onnx:
            self._load_onnx()
        else:
            raise ValueError(f"Unsupported backend: {self.backend_type}")

        self.class_names = self._get_class_names()
        logger.info("YOLOv5 ready | classes=%d", len(self.class_names))

    def _load_pytorch(self) -> None:
        import torch
        if self.weights_path and Path(self.weights_path).exists():
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=self.weights_path,
                verbose=False,
            )
        else:
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                self.model_variant,
                pretrained=True,
                verbose=False,
            )
        self.model.conf = self.confidence_threshold
        self.model.iou = self.iou_threshold
        self.model.to(self.device)
        self.model.eval()

    def _load_torchscript(self) -> None:
        import torch
        path = self.weights_path or "weights/yolov5s.torchscript"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"TorchScript file not found: {path}\n"
                "Run: python scripts/export_torchscript.py --model yolov5"
            )
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        path = self.weights_path or "weights/yolov5s.onnx"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"ONNX file not found: {path}\n"
                "Run: python scripts/export_onnx.py --model yolov5"
            )
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        logger.info("YOLOv5 ONNX providers selected: %s", providers)
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
        # Default COCO 80 classes (same as YOLOv8)
        from app.models.yolov8_detector import COCO_CLASSES
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
        import torch
        t0 = time.perf_counter()
        # torch.hub YOLOv5 accepts numpy BGR directly
        t_pre = time.perf_counter()

        results = self.model(image, size=self.image_size)
        t_inf = time.perf_counter()

        detections = self._parse_hub_results(results)
        t_post = time.perf_counter()

        timing = {
            "preprocessing_ms": (t_pre - t0) * 1000,
            "inference_ms": (t_inf - t_pre) * 1000,
            "postprocessing_ms": (t_post - t_inf) * 1000,
        }
        return detections, timing

    def _parse_hub_results(self, results) -> List[Detection]:
        detections: List[Detection] = []
        # results.xyxy[0]: tensor of [x1, y1, x2, y2, confidence, class_id]
        for *box, conf, cls_id in results.xyxy[0].tolist():
            x1, y1, x2, y2 = box
            cls_id = int(cls_id)
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
                    confidence=float(conf),
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

        # YOLOv5 TorchScript output: list of tensors, first is [1, 25200, 85]
        output = raw[0] if isinstance(raw, (list, tuple)) else raw
        output_np = output.cpu().numpy()
        detections = self._postprocess_yolov5_output(output_np, scale, padding, image.shape)
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

        # YOLOv5 ONNX output[0]: [1, 25200, 85]
        raw = outputs[0]
        detections = self._postprocess_yolov5_output(raw, scale, padding, image.shape)
        t_post = time.perf_counter()

        timing = {
            "preprocessing_ms": (t_pre - t0) * 1000,
            "inference_ms": (t_inf - t_pre) * 1000,
            "postprocessing_ms": (t_post - t_inf) * 1000,
        }
        return detections, timing

    def _postprocess_yolov5_output(
        self,
        raw: np.ndarray,       # [1, 25200, 85]
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple,
    ) -> List[Detection]:
        """
        Decode YOLOv5 raw ONNX / TorchScript output.
        Layout: [cx, cy, w, h, objectness, class_0, ..., class_79]
        """
        pred = raw[0]          # [25200, 85]

        objectness = pred[:, 4]
        class_scores = pred[:, 5:]
        # Combined confidence = objectness * max_class_score
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(class_scores)), class_ids]
        confidences = objectness * class_conf

        # Filter
        mask = confidences >= self.confidence_threshold
        pred_f = pred[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(pred_f) == 0:
            return []

        # Convert cx,cy,w,h → x1,y1,x2,y2
        cx, cy, bw, bh = pred_f[:, 0], pred_f[:, 1], pred_f[:, 2], pred_f[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

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
                dets, _ = self.predict_image(frame)
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
        """Export YOLOv5 PyTorch model to TorchScript via torch.jit.trace.

        Wraps the inner DetectionModel in a thin nn.Module that returns only
        the decoded prediction tensor [1, 25200, 85], matching the expected
        input shape of ``_postprocess_yolov5_output``.
        """
        import torch
        import torch.nn as nn

        if self.model is None:
            self._load_pytorch()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # The hub AutoShape wrapper exposes .model (DetectionModel).
        inner = self.model.model if hasattr(self.model, "model") else self.model
        inner.eval()

        # Thin wrapper: strips the tuple so trace sees a single Tensor output.
        class _SingleOutputWrapper(nn.Module):
            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.m = m

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.m(x)
                return out[0] if isinstance(out, (list, tuple)) else out

        wrapper = _SingleOutputWrapper(inner).eval().to(self.device)
        dummy = torch.zeros(1, 3, self.image_size, self.image_size).to(self.device)

        # Warm-up: force YOLOv5 Detect module to initialise its lazy grid tensors
        # so they have a fixed shape when torch.jit.trace inspects them.
        with torch.no_grad():
            wrapper(dummy)
            # check_trace=False skips the post-trace shape-comparison that breaks
            # on YOLOv5's dynamic grid (torch.Size([]) vs torch.Size([1,3,20,20,2]))
            traced = torch.jit.trace(wrapper, dummy, strict=False, check_trace=False)

        traced.save(str(out))
        logger.info("YOLOv5 TorchScript exported to %s", out)
        return str(out)

    def export_onnx(self, output_path: str) -> str:
        """Export YOLOv5 to ONNX via torch.onnx.export.

        Uses the same single-output wrapper as TorchScript export so the ONNX
        model returns exactly [1, 25200, 85], matching ``_postprocess_yolov5_output``.
        """
        import torch
        import torch.nn as nn

        if self.model is None:
            self._load_pytorch()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        inner = self.model.model if hasattr(self.model, "model") else self.model
        inner.eval()

        class _SingleOutputWrapper(nn.Module):
            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.m = m

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.m(x)
                return out[0] if isinstance(out, (list, tuple)) else out

        wrapper = _SingleOutputWrapper(inner).eval().to(self.device)
        dummy = torch.zeros(1, 3, self.image_size, self.image_size).to(self.device)

        # Warm-up pass so Detect grids are initialised before tracing.
        with torch.no_grad():
            wrapper(dummy)

        torch.onnx.export(
            wrapper,
            dummy,
            str(out),
            opset_version=12,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        )
        logger.info("YOLOv5 ONNX exported to %s", out)
        return str(out)
