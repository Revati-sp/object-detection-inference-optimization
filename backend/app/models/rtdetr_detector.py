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


class RTDETRDetector(BaseDetector):
    """
    RT-DETR (Real-Time DEtection TRansformer) detector.

    Supported backends:
    • pytorch   — Ultralytics RTDETR Python API (.pt weights)
    • onnx      — onnxruntime.InferenceSession on an exported .onnx file
    • tensorrt  — Ultralytics TensorRT engine (.engine) on an NVIDIA GPU
                  Export with: python scripts/export_tensorrt.py --model rtdetr
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
            model_name="rtdetr",
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
        logger.info("RT-DETR loading | backend=%s | device=%s", self.backend_type, self.device)

        if self.backend_type == BackendType.pytorch:
            self._load_pytorch()
        elif self.backend_type == BackendType.onnx:
            self._load_onnx()
        elif self.backend_type == BackendType.tensorrt:
            self._load_tensorrt()
        else:
            raise ValueError(
                f"RT-DETR does not support backend '{self.backend_type}'. "
                "Supported: pytorch, onnx, tensorrt"
            )

        self.class_names = self._get_class_names()
        logger.info("RT-DETR ready | classes=%d", len(self.class_names))

    def _load_pytorch(self) -> None:
        from ultralytics import RTDETR
        weights = self.weights_path or "rtdetr-l.pt"
        self.model = RTDETR(weights)
        if self.device == "cuda":
            self.model.to("cuda")

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        path = self.weights_path or "weights/rtdetr-l.onnx"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"ONNX file not found: {path}\n"
                "Run: python scripts/export_onnx.py --model rtdetr"
            )
        available = ort.get_available_providers()
        cuda_requested = "CUDAExecutionProvider" in available
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda_requested
            else ["CPUExecutionProvider"]
        )
        logger.info("RT-DETR ONNX requested providers: %s", providers)
        self.ort_session = ort.InferenceSession(path, providers=providers)
        self._ort_actual_providers = self.ort_session.get_providers()
        logger.info("RT-DETR ONNX actual providers in use: %s", self._ort_actual_providers)
        if cuda_requested and "CUDAExecutionProvider" not in self._ort_actual_providers:
            logger.warning(
                "⚠ RT-DETR ONNX CUDA FALLBACK — CUDAExecutionProvider was requested but is "
                "NOT active. Running on CPUExecutionProvider instead. "
                "Fix: pip install onnxruntime-gpu and ensure CUDA 11+ drivers are installed."
            )

    def _load_tensorrt(self) -> None:
        """Load a TensorRT .engine file via the Ultralytics RTDETR API."""
        from ultralytics import RTDETR
        path = self.weights_path or "weights/rtdetr-l.engine"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"TensorRT engine not found: {path}\n"
                "Run: python scripts/export_tensorrt.py --model rtdetr"
            )
        if self.device != "cuda":
            raise RuntimeError(
                "TensorRT backend requires a CUDA-capable GPU. "
                "No CUDA device was detected on this machine."
            )
        self.model = RTDETR(path)
        logger.info("RT-DETR TensorRT engine loaded from %s", path)

    def get_provider_info(self) -> dict:
        if self.backend_type == BackendType.onnx:
            actual = self._ort_actual_providers
            first = actual[0] if actual else "CPUExecutionProvider"
            hw_accel = first not in ("CPUExecutionProvider",)
            if "CUDA" in first:
                dev = "cuda"
            elif "TensorRT" in first:
                dev = "tensorrt"
            else:
                dev = "cpu"
            return {"actual_provider": first, "hardware_accelerated": hw_accel, "device_info": dev}
        if self.backend_type == BackendType.tensorrt:
            return {
                "actual_provider": "TensorRT_RTDETR",
                "hardware_accelerated": True,
                "device_info": "cuda (tensorrt)",
            }
        return {
            "actual_provider": f"pytorch_{self.device}",
            "hardware_accelerated": self.device == "cuda",
            "device_info": self.device,
        }

    def _get_class_names(self) -> List[str]:
        if self.backend_type in (BackendType.pytorch, BackendType.tensorrt) and self.model is not None:
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
        if self.backend_type in (BackendType.pytorch, BackendType.tensorrt):
            return self._predict_ultralytics(image)
        elif self.backend_type == BackendType.onnx:
            return self._predict_onnx(image)
        raise ValueError(f"Unsupported backend: {self.backend_type}")

    def _predict_ultralytics(self, image: np.ndarray):
        t0 = time.perf_counter()
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

    def _predict_onnx(self, image: np.ndarray):
        """
        RT-DETR ONNX output: two tensors — boxes [1, N, 4] and scores [1, N, 80].
        Boxes are in cx,cy,w,h format normalised to [0,1].
        """
        t0 = time.perf_counter()
        blob, scale, padding = preprocess_for_onnx(image, self.image_size)
        t_pre = time.perf_counter()

        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: blob})
        t_inf = time.perf_counter()

        detections = self._postprocess_rtdetr_output(outputs, image.shape)
        t_post = time.perf_counter()

        timing = {
            "preprocessing_ms": (t_pre - t0) * 1000,
            "inference_ms": (t_inf - t_pre) * 1000,
            "postprocessing_ms": (t_post - t_inf) * 1000,
        }
        return detections, timing

    def _postprocess_rtdetr_output(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple,
    ) -> List[Detection]:
        """Decode RT-DETR ONNX output (Ultralytics export format)."""
        h, w = orig_shape[:2]
        # Ultralytics RT-DETR ONNX: single output [1, 300, 6] — [cx,cy,w,h,conf,cls]
        if len(outputs) == 1:
            pred = outputs[0][0]  # [300, 6]
            detections: List[Detection] = []
            for row in pred:
                cx, cy, bw, bh, conf, cls_raw = row
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(cls_raw)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(w), x2), min(float(h), y2)
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                detections.append(
                    Detection(
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, width=x2 - x1, height=y2 - y1),
                        label=label,
                        class_id=cls_id,
                        confidence=float(conf),
                    )
                )
            return detections

        # Fallback: two-tensor format [1, N, 4] boxes + [1, N, 80] scores
        boxes_raw, scores_raw = outputs[0][0], outputs[1][0]  # [N, 4], [N, 80]
        class_ids = np.argmax(scores_raw, axis=1)
        confidences = scores_raw[np.arange(len(scores_raw)), class_ids]
        mask = confidences >= self.confidence_threshold
        boxes_raw = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        detections = []
        for box, conf, cls_id in zip(boxes_raw, confidences, class_ids):
            cx, cy, bw, bh = box
            x1 = max(0.0, (cx - bw / 2) * w)
            y1 = max(0.0, (cy - bh / 2) * h)
            x2 = min(float(w), (cx + bw / 2) * w)
            y2 = min(float(h), (cy + bh / 2) * h)
            label = self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else str(cls_id)
            detections.append(
                Detection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, width=x2 - x1, height=y2 - y1),
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
        raise NotImplementedError("RT-DETR TorchScript export is not supported.")

    def export_onnx(self, output_path: str) -> str:
        from ultralytics import RTDETR
        weights = self.weights_path or "rtdetr-l.pt"
        model = RTDETR(weights)
        saved = model.export(format="onnx", imgsz=self.image_size, opset=16)
        saved_path = Path(str(saved))
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if saved_path != out:
            saved_path.rename(out)
        logger.info("RT-DETR ONNX exported to %s", out)
        return str(out)
