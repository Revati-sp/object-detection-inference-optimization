# API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## Health

### `GET /health`
Returns service status.

**Response**
```json
{ "status": "ok", "app": "Object Detection API", "version": "1.0.0" }
```

---

## Detection

### `GET /api/models`
List all (model, backend) combinations and their load status.

**Response**
```json
{
  "models": [
    { "model_name": "yolov8", "backend_type": "pytorch", "loaded": true },
    { "model_name": "yolov8", "backend_type": "torchscript", "loaded": false },
    ...
  ]
}
```

---

### `POST /api/detect/image`
Run object detection on an uploaded image.

**Form fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | File | required | Image (JPEG/PNG/BMP/WebP) |
| `model_name` | string | `yolov8` | `yolov8` or `yolov5` |
| `backend_type` | string | `pytorch` | `pytorch`, `torchscript`, or `onnx` |
| `confidence_threshold` | float | `0.25` | Min detection confidence (0–1) |
| `iou_threshold` | float | `0.45` | NMS IoU threshold (0–1) |
| `save_result` | bool | `false` | Persist upload to `uploads/` |

**Response**
```json
{
  "model_name": "yolov8",
  "backend_type": "pytorch",
  "image_width": 1280,
  "image_height": 720,
  "detections": [
    {
      "bbox": { "x1": 100, "y1": 50, "x2": 300, "y2": 250, "width": 200, "height": 200 },
      "label": "person",
      "class_id": 0,
      "confidence": 0.87
    }
  ],
  "total_detections": 1,
  "latency_ms": 18.4,
  "preprocessing_ms": 1.2,
  "inference_ms": 15.6,
  "postprocessing_ms": 1.6
}
```

---

### `POST /api/detect/video`
Run object detection on every frame of an uploaded video.

**Form fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | File | required | Video file |
| `model_name` | string | `yolov8` | `yolov8` or `yolov5` |
| `backend_type` | string | `pytorch` | `pytorch`, `torchscript`, or `onnx` |
| `confidence_threshold` | float | `0.25` | |
| `iou_threshold` | float | `0.45` | |
| `save_output_video` | bool | `true` | Save annotated video to `outputs/` |
| `max_frames` | int | `null` | Max frames to process (null = all) |

**Response**
```json
{
  "model_name": "yolov8",
  "backend_type": "onnx",
  "frame_count": 120,
  "average_fps": 42.3,
  "total_latency_ms": 2836.4,
  "average_latency_per_frame_ms": 23.6,
  "total_detections": 347,
  "output_path": "/path/to/outputs/annotated_video.mp4",
  "frames_summary": [
    { "frame_index": 0, "detections": 3, "latency_ms": 22.1 },
    ...
  ]
}
```

---

## Evaluation

### `POST /api/evaluate`
Compute mAP on a COCO-annotated image dataset.

**Request body (JSON)**
```json
{
  "model_name": "yolov8",
  "backend_type": "pytorch",
  "annotations_path": "data/annotations/instances_val.json",
  "images_dir": "data/images/val",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45
}
```

**Response**
```json
{
  "model_name": "yolov8",
  "backend_type": "pytorch",
  "num_images": 100,
  "map_50": 0.612,
  "map_50_95": 0.421,
  "per_image_latencies_ms": [18.2, 19.4, ...],
  "average_latency_ms": 18.9,
  "fps": 52.9
}
```

---

## Benchmark

### `POST /api/benchmark`
Measure inference latency for all requested (model, backend) pairs.

**Request body (JSON)**
```json
{
  "model_names": ["yolov8", "yolov5"],
  "backend_types": ["pytorch", "torchscript", "onnx"],
  "num_runs": 100,
  "image_size": 640,
  "warmup_runs": 10
}
```

**Response**
```json
{
  "results": [
    {
      "model_name": "yolov8",
      "backend_type": "pytorch",
      "avg_latency_ms": 18.4,
      "min_latency_ms": 16.1,
      "max_latency_ms": 25.3,
      "std_latency_ms": 1.2,
      "fps": 54.3,
      "image_size": 640,
      "num_runs": 100,
      "status": "ok"
    },
    {
      "model_name": "yolov8",
      "backend_type": "torchscript",
      "avg_latency_ms": 14.2,
      ...
    },
    {
      "model_name": "yolov8",
      "backend_type": "onnx",
      "avg_latency_ms": 11.8,
      ...
    }
  ],
  "image_size": 640,
  "num_runs": 100,
  "warmup_runs": 10
}
```

---

## Error Responses

All errors follow FastAPI's standard format:

```json
{ "detail": "Human-readable error message" }
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (empty file, unsupported format) |
| 404 | Model weights file not found |
| 415 | Unsupported media type |
| 500 | Inference or server error |
