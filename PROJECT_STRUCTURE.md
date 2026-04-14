# Object Detection Project - Complete Structure & Code

## 1. FULL FOLDER TREE

```
Object Detection/
├── README.md
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app entry point, CORS, lifespan
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes_detection.py    # POST /detect/image, /detect/video, GET /models
│   │   │   └── routes_eval.py         # POST /evaluate, /benchmark
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py              # Pydantic Settings (environment variables)
│   │   │   └── logging.py             # Structured console logger
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Abstract BaseDetector interface
│   │   │   ├── yolov8_detector.py     # YOLOv8 — PyTorch / TorchScript / ONNX
│   │   │   └── yolov5_detector.py     # YOLOv5 — PyTorch / TorchScript / ONNX
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── inference.py           # Model registry and lazy loading
│   │   │   ├── video_processing.py    # Frame iteration and annotation writer
│   │   │   ├── evaluation.py          # COCO mAP via pycocotools
│   │   │   └── benchmark.py           # Synthetic latency benchmarking
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── detection.py           # All Pydantic request/response models
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── image.py               # Letterbox, preprocess, draw, encode
│   │       ├── video.py               # Frame iterator, VideoWriter context manager
│   │       └── timing.py              # TimingResult, timer context manager
│   ├── outputs/                       # Annotated output videos/images (auto-created)
│   ├── uploads/                       # Temporary uploaded media (auto-created)
│   ├── weights/                       # Model weights (auto-created)
│   └── requirements.txt               # Python dependencies
├── data/
│   ├── annotations/                   # COCO-style annotations JSON
│   ├── images/                        # Image dataset
│   └── sample/                        # Sample data for testing
├── docs/
│   └── api_reference.md               # API documentation
├── frontend/
│   ├── app/
│   │   ├── layout.tsx                 # Root layout with Tailwind
│   │   ├── page.tsx                   # Main page with tabs
│   │   └── globals.css                # Global Tailwind styles
│   ├── components/
│   │   ├── BenchmarkPanel.tsx         # Run benchmarks, FPS/latency bar charts
│   │   ├── EvaluatePanel.tsx          # COCO mAP evaluation UI
│   │   ├── HealthBadge.tsx            # Backend health indicator
│   │   ├── ImageResultViewer.tsx      # Canvas bbox overlay for images
│   │   ├── MetricsPanel.tsx           # Latency breakdown, FPS, detection list
│   │   ├── ModelSelector.tsx          # Model and backend dropdown selectors
│   │   ├── UploadForm.tsx             # Drag-and-drop file upload
│   │   ├── VideoResultViewer.tsx      # Per-frame sparkline charts
│   │   └── ui.tsx                     # Reusable UI components
│   ├── lib/
│   │   └── api.ts                     # Typed fetch wrappers for every endpoint
│   ├── types/
│   │   └── index.ts                   # TypeScript mirrors of API schemas
│   ├── next.config.js                 # API proxy rewrite rule
│   ├── package.json                   # NPM dependencies
│   ├── tailwind.config.ts             # Tailwind CSS configuration
│   ├── tsconfig.json                  # TypeScript configuration
│   ├── postcss.config.js              # PostCSS configuration
│   └── next-env.d.ts                  # Next.js type definitions
├── scripts/
│   ├── export_torchscript.py          # Export one model → TorchScript
│   ├── export_onnx.py                 # Export one model → ONNX
│   ├── run_all_exports.py             # Batch export all model/backend combinations
│   ├── benchmark_models.py            # Standalone benchmark script
│   ├── compare_models.py              # Side-by-side model comparison
│   └── evaluate_dataset.py            # Evaluate mAP on dataset
└── README.md                          # Project documentation
```

---

## 2. REQUIREMENTS.TXT

