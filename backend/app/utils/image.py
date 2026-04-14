from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to an OpenCV BGR image array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from uploaded bytes.")
    return image


def load_image_from_path(path: str | Path) -> np.ndarray:
    """Load an image from disk as BGR numpy array."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Preprocessing for ONNX inference
# ---------------------------------------------------------------------------

def letterbox(
    image: np.ndarray,
    target_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to *target_size* x *target_size* with letterboxing (preserving
    aspect ratio), then pad to a square with *color*.

    Returns:
        resized_image: padded square image (uint8, BGR)
        scale: scale factor applied to original image
        padding: (pad_width, pad_height) added to each side
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_h,
        target_size - new_h - pad_h,
        pad_w,
        target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, scale, (pad_w, pad_h)


def preprocess_for_onnx(
    image: np.ndarray, target_size: int = 640
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Full preprocessing pipeline for ONNX model input:
    1. Letterbox resize
    2. BGR → RGB
    3. HWC → CHW
    4. Normalize [0, 255] → [0.0, 1.0]
    5. Add batch dimension → [1, C, H, W] float32

    Returns:
        blob: numpy array ready for onnxruntime, shape [1, 3, H, W]
        scale: letterbox scale
        padding: (pad_w, pad_h)
    """
    padded, scale, padding = letterbox(image, target_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    blob = np.expand_dims(chw, axis=0)
    return blob, scale, padding


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def scale_boxes_back(
    boxes: np.ndarray,
    scale: float,
    padding: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Convert bounding boxes from the letterboxed coordinate space back to
    the original image coordinate space.

    Args:
        boxes: [N, 4] array of [x1, y1, x2, y2] in letterboxed space
        scale: the scale factor used during letterboxing
        padding: (pad_w, pad_h) padding added to each side
        orig_shape: (height, width) of the original image

    Returns:
        boxes_orig: [N, 4] array clipped to original image bounds
    """
    pad_w, pad_h = padding
    boxes_orig = boxes.copy().astype(np.float32)
    boxes_orig[:, [0, 2]] = (boxes_orig[:, [0, 2]] - pad_w) / scale
    boxes_orig[:, [1, 3]] = (boxes_orig[:, [1, 3]] - pad_h) / scale

    h, w = orig_shape[:2]
    boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, w)
    boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, h)
    return boxes_orig


# ---------------------------------------------------------------------------
# Annotation / visualization
# ---------------------------------------------------------------------------

_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def get_color(class_id: int) -> Tuple[int, int, int]:
    return _COLORS[class_id % len(_COLORS)]


def draw_detections(
    image: np.ndarray,
    detections: list,  # List[Detection]
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and confidence scores onto a copy of *image*.
    Accepts Detection pydantic objects or dicts with keys bbox/label/confidence.
    """
    annotated = image.copy()
    for det in detections:
        if hasattr(det, "bbox"):
            bbox = det.bbox
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            label = det.label
            conf = det.confidence
            class_id = det.class_id
        else:
            x1, y1, x2, y2 = (
                int(det["bbox"]["x1"]),
                int(det["bbox"]["y1"]),
                int(det["bbox"]["x2"]),
                int(det["bbox"]["y2"]),
            )
            label = det["label"]
            conf = det["confidence"]
            class_id = det.get("class_id", 0)

        color = get_color(class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            annotated,
            text,
            (x1 + 1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_image_to_bytes(image: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode OpenCV image array to bytes for HTTP response or disk write."""
    success, buf = cv2.imencode(ext, image)
    if not success:
        raise RuntimeError("Failed to encode image.")
    return buf.tobytes()


def save_image(image: np.ndarray, path: str | Path) -> None:
    cv2.imwrite(str(path), image)


def numpy_image_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(image))
