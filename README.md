# Object Detection — Inference Optimization

A full-stack academic project demonstrating real-time object detection with multiple models
and inference acceleration strategies.

| | |
|---|---|
| **Models** | YOLOv8 · YOLOv5 |
| **Backends** | PyTorch (baseline) · TorchScript · ONNX Runtime |
| **API** | FastAPI · OpenAPI / Swagger |
| **Frontend** | Next.js 14 · TypeScript · Tailwind CSS |
| **Evaluation** | COCO mAP (pycocotools) · Latency · FPS |
| **Python** | 3.9 + |
| **Node** | 18 + |

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Running the Application](#running-the-application)
4. [Exporting Models](#exporting-models)
5. [Scripts Reference](#scripts-reference)
6. [Benchmarking](#benchmarking)
7. [Evaluation](#evaluation)
8. [Model Comparison Report](#model-comparison-report)
9. [Sample API Requests](#sample-api-requests)
10. [GPU Acceleration](#gpu-acceleration)
11. [Troubleshooting](#troubleshooting)
12. [Assignment Mapping](#assignment-mapping)

---

## Project Structure

```
Object Detection/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app entry point, CORS, lifespan
│   │   ├── api/
│   │   │   ├── routes_detection.py    # POST /detect/image, /detect/video, GET /models
│   │   │   └── routes_eval.py         # POST /evaluate, /benchmark
│   │   ├── core/
│   │   │   ├── config.py              # Pydantic Settings (environment variables)
│   │   │   └── logging.py             # Structured console logger
│   │   ├── models/
│   │   │   ├── base.py                # Abstract BaseDetector interface
│   │   │   ├── yolov8_detector.py     # YOLOv8 — PyTorch / TorchScript / ONNX
│   │   │   └── yolov5_detector.py     # YOLOv5 — PyTorch / TorchScript / ONNX
│   │   ├── services/
│   │   │   ├── inference.py           # Model registry and lazy loading
│   │   │   ├── video_processing.py    # Frame iteration and annotation writer
│   │   │   ├── evaluation.py          # COCO mAP via pycocotools
│   │   │   └── benchmark.py           # Synthetic latency benchmarking
│   │   ├── schemas/
│   │   │   └── detection.py           # All Pydantic request/response models
│   │   └── utils/
│   │       ├── image.py               # Letterbox, preprocess, draw, encode
│   │       ├── video.py               # Frame iterator, VideoWriter context manager
│   │       └── timing.py              # TimingResult, timer context manager
│   ├── uploads/                       # Temporary uploaded media (auto-created)
│   ├── outputs/                       # Annotated output videos/images (auto-created)
│   ├── weights/                       # Exported model files (.torchscript, .onnx)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── layout.tsx                 # Root layout — header + HealthBadge
│   │   ├── page.tsx                   # Main tabbed UI (Detect / Benchmark / Evaluate)
│   │   └── globals.css                # Tailwind base + CSS variables
│   ├── components/
│   │   ├── ui.tsx                     # Shared primitives: Card, Spinner
│   │   ├── HealthBadge.tsx            # Backend live-status indicator
│   │   ├── ImageResultViewer.tsx      # Canvas bbox overlay for detected images
│   │   ├── VideoResultViewer.tsx      # Per-frame sparkline charts
│   │   ├── MetricsPanel.tsx           # Latency breakdown, FPS, detection list
│   │   ├── BenchmarkPanel.tsx         # Run benchmarks, FPS/latency bar charts
│   │   └── EvaluatePanel.tsx          # COCO mAP evaluation UI
│   ├── lib/
│   │   └── api.ts                     # Typed fetch wrappers for every endpoint
│   ├── types/
│   │   └── index.ts                   # TypeScript mirrors of API schemas
│   ├── next.config.js                 # API proxy rewrite rule
│   ├── package.json
│   └── .env.example
├── scripts/
│   ├── export_torchscript.py          # Export one model → TorchScript
│   ├── export_onnx.py                 # Export one model → ONNX
│   ├── run_all_exports.py             # Export ALL models to ALL formats in one go
│   ├── benchmark_models.py            # Latency benchmark with CSV output + speedup
│   ├── evaluate_dataset.py            # COCO mAP evaluation with compare mode
│   └── compare_models.py              # Benchmark + eval combined Markdown/CSV report
├── data/
│   ├── images/val/                    # Place evaluation images here
│   └── annotations/                   # Place COCO-format JSON annotations here
├── results/                           # CSV / Markdown reports written here
└── docs/
    └── api_reference.md
```

---

## Setup

### Prerequisites

- Python **3.9+**  (tested on 3.9.6 on macOS)
- Node.js **18+**
- (Optional) CUDA-capable GPU — see [GPU Acceleration](#gpu-acceleration)

### 1 — Backend

```bash
cd backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install all Python dependencies
pip install -r requirements.txt

# Create your local environment file
cp .env.example .env
# Edit .env to set custom weights paths, thresholds, or GPU settings
```

> **Note:** `onnxsim` is intentionally excluded from `requirements.txt` because it
> requires `cmake` to build.  ONNX export still works without it (`simplify=False`).
> Install it separately if you want graph simplification:
> `pip install onnxsim`

### 2 — Frontend

```bash
cd frontend

npm install

cp .env.example .env.local
# NEXT_PUBLIC_API_URL defaults to http://localhost:8000
```

---

## Running the Application

### Start the backend

```bash
cd backend
source venv/bin/activate

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs (Swagger UI): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

### Start the frontend

```bash
cd frontend
npm run dev
```

Open: http://localhost:3000

> **macOS users — if you see `EMFILE: too many open files`:**
> The default macOS file-descriptor limit (256) is too low for Next.js's file watcher.
> Start the dev server with:
> ```bash
> ulimit -n 65536 && npm run dev
> ```
> This raises the limit for the current shell only and does not require any system changes.

---

## Exporting Models

TorchScript and ONNX backends require exported model files.
Always run export scripts from the **project root** with the backend venv active.

### Option A — Export everything at once (recommended)

```bash
source backend/venv/bin/activate

python scripts/run_all_exports.py
```

This exports YOLOv8n and YOLOv5s to both TorchScript and ONNX formats under
`backend/weights/` and prints the `.env` variable names to paste.

Additional options:

```bash
# Different image size
python scripts/run_all_exports.py --image-size 416

# Only ONNX, only YOLOv8
python scripts/run_all_exports.py --models yolov8 --formats onnx

# Force re-export even if files already exist
python scripts/run_all_exports.py --force
```

### Option B — Export individual models

```bash
# YOLOv8 → TorchScript
python scripts/export_torchscript.py --model yolov8 --weights yolov8n.pt \
    --output backend/weights/yolov8n.torchscript

# YOLOv8 → ONNX
python scripts/export_onnx.py --model yolov8 --weights yolov8n.pt \
    --output backend/weights/yolov8n.onnx

# YOLOv5 → TorchScript
python scripts/export_torchscript.py --model yolov5 --weights yolov5s \
    --output backend/weights/yolov5s.torchscript

# YOLOv5 → ONNX
python scripts/export_onnx.py --model yolov5 --weights yolov5s \
    --output backend/weights/yolov5s.onnx
```

### Point the backend at the exported files

Add to `backend/.env`:

```env
YOLOV8_TORCHSCRIPT_PATH=weights/yolov8n.torchscript
YOLOV8_ONNX_PATH=weights/yolov8n.onnx
YOLOV5_TORCHSCRIPT_PATH=weights/yolov5s.torchscript
YOLOV5_ONNX_PATH=weights/yolov5s.onnx
```

---

## Scripts Reference

All scripts are run from the **project root** with the backend venv active.

| Script | Purpose |
|--------|---------|
| `run_all_exports.py` | Export all models to all formats in one command |
| `export_torchscript.py` | Export a single model to TorchScript |
| `export_onnx.py` | Export a single model to ONNX |
| `benchmark_models.py` | Latency/FPS table with speedup column; CSV/JSON output |
| `evaluate_dataset.py` | COCO mAP evaluation; compare mode; CSV/JSON output |
| `compare_models.py` | Combined benchmark + eval → Markdown + CSV report |

---

## Benchmarking

```bash
# All models × all backends, 100 runs at 640×640 (default)
python scripts/benchmark_models.py

# Quick two-backend comparison, saved as CSV
python scripts/benchmark_models.py \
    --models yolov8 --backends pytorch onnx \
    --runs 50 \
    --output results/bench_yolov8.csv

# Multi-resolution sweep: 320, 640, 1280
python scripts/benchmark_models.py \
    --sizes 320 640 1280 \
    --runs 100 \
    --output results/bench_sweep.csv

# Via the API
curl -X POST http://localhost:8000/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "model_names": ["yolov8", "yolov5"],
    "backend_types": ["pytorch", "torchscript", "onnx"],
    "num_runs": 50,
    "warmup_runs": 10,
    "image_size": 640
  }'
```

Sample console output:

```
────────────────────────────────────────────────────────────────────
  Benchmark Results  │  Image: 640×640  │  Runs: 100
────────────────────────────────────────────────────────────────────
  Model      Backend        Avg ms    Min ms    Max ms    Std ms     FPS  Speedup  Status
────────────────────────────────────────────────────────────────────
  yolov8     pytorch         18.42     16.10     24.31      1.23    54.3    1.00×  ok
  yolov8     torchscript     14.87     13.50     19.44      0.98    67.2    1.24×  ok
  yolov8     onnx            11.65     10.20     14.73      0.77    85.8    1.58×  ok
  yolov5     pytorch         22.10     19.80     28.65      1.44    45.2    1.00×  ok
  yolov5     torchscript     17.93     16.40     22.10      1.01    55.8    1.23×  ok
  yolov5     onnx            13.41     11.90     17.22      0.89    74.6    1.65×  ok
────────────────────────────────────────────────────────────────────
  ✓ Fastest: yolov8/onnx — 11.65 ms  (85.8 FPS)
```

---

## Evaluation

### Dataset layout

```
data/
├── images/
│   └── val/
│       ├── img001.jpg
│       └── ...
└── annotations/
    └── instances_val.json     ← COCO-format ground truth
```

### Run evaluation (CLI)

```bash
# Single model/backend
python scripts/evaluate_dataset.py \
    --model yolov8 --backend pytorch \
    --annotations data/annotations/instances_val.json \
    --images-dir data/images/val

# Compare all three backends for YOLOv8
python scripts/evaluate_dataset.py \
    --model yolov8 --compare \
    --annotations data/annotations/instances_val.json \
    --images-dir data/images/val \
    --output results/eval_yolov8.csv

# Compare ALL model/backend combos
python scripts/evaluate_dataset.py \
    --model yolov8 yolov5 --compare \
    --annotations data/annotations/instances_val.json \
    --images-dir data/images/val \
    --output results/eval_all.csv

# Save COCO prediction JSON for further analysis
python scripts/evaluate_dataset.py \
    --model yolov8 --backend onnx \
    --annotations data/annotations/instances_val.json \
    --images-dir data/images/val \
    --save-predictions results/yolov8_onnx_preds.json
```

### Run evaluation (API)

```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "yolov8",
    "backend_type": "pytorch",
    "annotations_path": "data/annotations/instances_val.json",
    "images_dir": "data/images/val",
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45
  }'
```

Sample response:

```json
{
  "model_name": "yolov8",
  "backend_type": "pytorch",
  "num_images": 100,
  "map_50": 0.612,
  "map_50_95": 0.421,
  "average_latency_ms": 18.4,
  "fps": 54.3,
  "per_image_latencies_ms": [17.1, 19.3, ...]
}
```

---

## Model Comparison Report

Generates a combined benchmark + evaluation report as both **CSV** and **Markdown**:

```bash
# Benchmark only (no dataset needed)
python scripts/compare_models.py --output results/comparison

# Benchmark + COCO mAP
python scripts/compare_models.py \
    --annotations data/annotations/instances_val.json \
    --images-dir data/images/val \
    --output results/comparison

# Specific models and backends, 50 runs
python scripts/compare_models.py \
    --models yolov8 yolov5 --backends pytorch onnx \
    --runs 50 \
    --output results/comparison
```

Output files:
- `results/comparison.csv` — machine-readable table
- `results/comparison.md` — Markdown table ready to paste into a report

---

## Sample API Requests

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok", "timestamp": "..."}
```

### List available models

```bash
curl http://localhost:8000/api/models
```

### Image detection

```bash
curl -X POST http://localhost:8000/api/detect/image \
  -F "file=@/path/to/image.jpg" \
  -F "model_name=yolov8" \
  -F "backend_type=pytorch" \
  -F "confidence_threshold=0.3"
```

Returns JSON with bounding boxes, labels, confidence scores, and latency breakdown.

### Video detection

```bash
curl -X POST http://localhost:8000/api/detect/video \
  -F "file=@/path/to/video.mp4" \
  -F "model_name=yolov8" \
  -F "backend_type=onnx" \
  -F "max_frames=30"
```

---

## GPU Acceleration

### PyTorch (CUDA)

CUDA is detected automatically. For the CUDA-enabled build of PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### ONNX Runtime (CUDA)

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

Then in `backend/.env`:

```env
ONNX_EXECUTION_PROVIDER=CUDAExecutionProvider
```

The code automatically falls back to `CPUExecutionProvider` if CUDA is unavailable.

---

## Troubleshooting

### `EMFILE: too many open files` (macOS)

The default macOS file descriptor limit (256) is too low for Next.js.

```bash
ulimit -n 65536 && npm run dev
```

### `cmake` error when installing requirements

`onnxsim` is commented out of `requirements.txt` because it requires `cmake`.
The project works without it. If you want model simplification:

```bash
brew install cmake
pip install onnxsim
```

### TorchScript / ONNX backend returns an error

These backends require exported model files. Run:

```bash
python scripts/run_all_exports.py
```

Then add the output paths to `backend/.env` as shown in the export section.

### YOLOv5 first run is slow

YOLOv5 downloads model weights from the PyTorch Hub on the first load.
Subsequent runs use the cached version.

### Frontend shows stale results

Clear the Next.js build cache:

```bash
rm -rf frontend/.next && npm run dev
```

---

## Evaluation Results

> Results obtained on a 200-image COCO val2017 subset.
> Run `./scripts/run_complete_pipeline.sh` to reproduce.

### Accuracy (mAP)

| Model | Backend | mAP@0.5 | mAP@0.5:0.95 | Images | Avg Latency (ms) | FPS |
|-------|---------|---------|--------------|--------|-----------------|-----|
| YOLOv8n | PyTorch | — | — | 200 | — | — |
| YOLOv8n | TorchScript | — | — | 200 | — | — |
| YOLOv8n | ONNX Runtime | — | — | 200 | — | — |
| YOLOv5s | PyTorch | — | — | 200 | — | — |
| YOLOv5s | TorchScript | — | — | 200 | — | — |
| YOLOv5s | ONNX Runtime | — | — | 200 | — | — |

*Fill in after running: `python scripts/evaluate_dataset.py --model yolov8 yolov5 --compare --annotations data/annotations/instances_val200.json --images-dir data/images/val --output results/eval_report.csv`*

### Inference Speed (synthetic 640×640 image, 100 runs)

| Model | Backend | Avg (ms) | Min (ms) | Max (ms) | FPS | Speedup vs PyTorch |
|-------|---------|---------|---------|---------|-----|--------------------|
| YOLOv8n | PyTorch | — | — | — | — | 1.00× |
| YOLOv8n | TorchScript | — | — | — | — | —× |
| YOLOv8n | ONNX Runtime | — | — | — | — | —× |
| YOLOv5s | PyTorch | — | — | — | — | 1.00× |
| YOLOv5s | TorchScript | — | — | — | — | —× |
| YOLOv5s | ONNX Runtime | — | — | — | — | —× |

*Fill in after running: `python scripts/benchmark_models.py --models yolov8 yolov5 --backends pytorch torchscript onnx --runs 100 --output results/benchmark.csv`*

### Video Inference

| Model | Backend | Frames | Avg Latency/Frame (ms) | Avg FPS | Total Detections |
|-------|---------|--------|----------------------|---------|-----------------|
| YOLOv8n | PyTorch | — | — | — | — |
| YOLOv8n | TorchScript | — | — | — | — |
| YOLOv8n | ONNX Runtime | — | — | — | — |
| YOLOv5s | PyTorch | — | — | — | — |
| YOLOv5s | TorchScript | — | — | — | — |
| YOLOv5s | ONNX Runtime | — | — | — | — |

*Fill in after running: `python scripts/run_video_inference.py --generate-test-video --compare --results-dir results`*

### How to Reproduce All Results

```bash
# 1. Activate backend virtualenv
cd backend && source venv/bin/activate && cd ..

# 2. Run the full pipeline (steps 1–5 automated)
./scripts/run_complete_pipeline.sh

# --- OR run each step individually ---

# Step 1: Download 200 real COCO val2017 images + annotations
python scripts/prepare_coco_subset.py --num-images 200

# Step 2: Export YOLOv8 + YOLOv5 to TorchScript and ONNX
cd backend
python ../scripts/export_torchscript.py --model yolov8 --output weights/yolov8n.torchscript
python ../scripts/export_torchscript.py --model yolov5 --output weights/yolov5s.torchscript
python ../scripts/export_onnx.py        --model yolov8 --output weights/yolov8n.onnx
python ../scripts/export_onnx.py        --model yolov5 --output weights/yolov5s.onnx
cd ..

# Step 3: Evaluate mAP across all 6 model/backend combos
cd backend
python ../scripts/evaluate_dataset.py \
    --model yolov8 yolov5 --compare \
    --annotations ../data/annotations/instances_val200.json \
    --images-dir   ../data/images/val \
    --output       ../results/eval_report.csv
cd ..

# Step 4: Benchmark latency/FPS
cd backend
python ../scripts/benchmark_models.py \
    --models yolov8 yolov5 \
    --backends pytorch torchscript onnx \
    --runs 100 --warmup 20 \
    --output ../results/benchmark.csv
cd ..

# Step 5: Video inference
cd backend
python ../scripts/run_video_inference.py \
    --generate-test-video --compare \
    --max-frames 150 \
    --results-dir ../results --output-dir ../outputs
cd ..
```

---

## Assignment Mapping

| Requirement | Where it is implemented |
|-------------|------------------------|
| **2 detection models** | `YOLOv8Detector` (`backend/app/models/yolov8_detector.py`) and `YOLOv5Detector` (`backend/app/models/yolov5_detector.py`), both sharing the `BaseDetector` abstract interface |
| **FastAPI backend** | `backend/app/main.py` — CORS, lifespan; routers for detection, evaluation, and benchmarking; full OpenAPI docs at `/docs` |
| **React / Next.js frontend** | `frontend/` — Next.js 14 App Router, TypeScript, Tailwind CSS, drag-and-drop upload, canvas bbox overlay, per-frame sparkline charts, Benchmark tab, Evaluate tab, live backend health badge |
| **Inference acceleration 1: TorchScript** | `export_torchscript()` in both detector classes; TorchScript inference path `_predict_torchscript()`; export via `scripts/export_torchscript.py` and `scripts/run_all_exports.py` |
| **Inference acceleration 2: ONNX Runtime** | `export_onnx()` in both detector classes; ONNX inference via `onnxruntime.InferenceSession` with `CUDAExecutionProvider` / CPU fallback; export via `scripts/export_onnx.py` and `scripts/run_all_exports.py` |
| **Custom dataset + annotations** | `data/images/val/` — 139 custom screenshots; `data/annotations/instances_custom.json` — per-object COCO bounding-box annotations generated by `scripts/create_custom_annotations.py` using YOLOv8n at conf=0.5 (pseudo-labeling) |
| **mAP evaluation on annotated dataset** | `backend/app/services/evaluation.py` — loads COCO-format annotations, runs inference, computes mAP@0.5 and mAP@0.5:0.95 via pycocotools; `POST /api/evaluate`; `scripts/evaluate_dataset.py`; results in `results/eval_report.csv` |
| **Latency / FPS metrics** | Returned in every inference response; per-frame metrics for video; dedicated `/api/benchmark` endpoint; `scripts/benchmark_models.py` for CLI access; results in `results/benchmark.csv` |
| **Video inference** | `backend/app/services/video_processing.py`; `scripts/run_video_inference.py`; annotated videos in `outputs/`; FPS summary in `results/video_benchmark.csv` |
