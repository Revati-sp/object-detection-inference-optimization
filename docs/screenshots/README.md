# Screenshots

Place the following screenshots here to complete the visual documentation:

| Filename | Content |
|----------|---------|
| `frontend-upload.png` | Detection tab — drag-and-drop upload UI with bounding-box overlay |
| `detection-results.png` | Annotated output image with labels and confidence scores |
| `benchmark-results.png` | Benchmark tab — FPS/latency bar charts across model × backend |
| `evaluation-results.png` | Evaluate tab — mAP@0.5 and mAP@0.5:0.95 per model/backend |

## How to capture

```bash
# 1. Start the backend
cd backend && source venv/bin/activate
uvicorn app.main:app --reload --port 8000

# 2. Start the frontend (new terminal)
cd frontend && npm run dev
# Open http://localhost:3000
```

Upload one of the images from `data/images/val/`, run detection, and take screenshots of each tab.
