#!/usr/bin/env bash
# =============================================================================
# run_complete_pipeline.sh
#
# Executes the full assignment pipeline in the correct order:
#   1. Download COCO val2017 subset (200 images + real annotations)
#   2. Export YOLOv8 + YOLOv5 → TorchScript + ONNX  (4 files)
#   3. Run mAP evaluation for all 6 model/backend combinations
#   4. Run latency/FPS benchmark for all 6 combinations
#   5. Run video inference on a test video for all 6 combinations
#
# All results are saved to results/ and committed to git.
#
# Usage (from project root):
#   chmod +x scripts/run_complete_pipeline.sh
#   ./scripts/run_complete_pipeline.sh
#
# Requirements: backend venv must be active.
#   cd backend && source venv/bin/activate && cd ..
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

BACKEND_DIR="$PROJECT_ROOT/backend"
RESULTS_DIR="$PROJECT_ROOT/results"
DATA_DIR="$PROJECT_ROOT/data"
OUTPUTS_DIR="$PROJECT_ROOT/outputs"

mkdir -p "$RESULTS_DIR" "$OUTPUTS_DIR"

echo "============================================================"
echo "  Object Detection — Full Assignment Pipeline"
echo "  Project root: $PROJECT_ROOT"
echo "============================================================"
echo ""

# ─────────────────────────────────────────────────────────────────
# STEP 1: Auto-annotate custom images (or download COCO subset)
# ─────────────────────────────────────────────────────────────────
echo "[STEP 1/5] Preparing dataset annotations ..."
echo "─────────────────────────────────────────────────────────────"

IMG_DIR="$DATA_DIR/images/val"
ANNO_FILE="$DATA_DIR/annotations/instances_custom.json"

if [ -d "$IMG_DIR" ] && [ "$(ls -A "$IMG_DIR" 2>/dev/null)" ]; then
    # User already has images in data/images/val/ — auto-annotate them
    echo "  Found existing images in $IMG_DIR — running auto-annotation ..."
    cd "$BACKEND_DIR"
    python ../scripts/create_custom_annotations.py \
        --images-dir "$IMG_DIR" \
        --output     "$ANNO_FILE" \
        --conf 0.50 \
        --min-anns 1
    cd "$PROJECT_ROOT"
else
    # No images present — download COCO val2017 subset
    echo "  No images found — downloading COCO val2017 subset (200 images) ..."
    python scripts/prepare_coco_subset.py \
        --num-images 200 \
        --min-anns 1 \
        --output-dir data
    ANNO_FILE="$DATA_DIR/annotations/instances_val200.json"
fi

if [ ! -f "$ANNO_FILE" ]; then
    echo "ERROR: Annotation file not found: $ANNO_FILE"
    exit 1
fi

echo ""
echo "✓ Dataset ready"
echo "  Annotations : $ANNO_FILE"
echo "  Images      : $IMG_DIR"
echo ""

# ─────────────────────────────────────────────────────────────────
# STEP 2: Export model weights
# ─────────────────────────────────────────────────────────────────
echo "[STEP 2/5] Exporting models to TorchScript + ONNX ..."
echo "─────────────────────────────────────────────────────────────"

cd "$BACKEND_DIR"

echo "  Exporting YOLOv8 → TorchScript ..."
python ../scripts/export_torchscript.py \
    --model yolov8 \
    --weights yolov8n.pt \
    --output weights/yolov8n.torchscript \
    --image-size 640

echo "  Exporting YOLOv8 → ONNX ..."
python ../scripts/export_onnx.py \
    --model yolov8 \
    --weights yolov8n.pt \
    --output weights/yolov8n.onnx \
    --image-size 640 \
    --opset 12

echo "  Exporting YOLOv5 → TorchScript ..."
python ../scripts/export_torchscript.py \
    --model yolov5 \
    --weights yolov5s \
    --output weights/yolov5s.torchscript \
    --image-size 640

echo "  Exporting YOLOv5 → ONNX ..."
python ../scripts/export_onnx.py \
    --model yolov5 \
    --weights yolov5s \
    --output weights/yolov5s.onnx \
    --image-size 640 \
    --opset 12

cd "$PROJECT_ROOT"

echo ""
echo "✓ Weights exported:"
ls -lh backend/weights/
echo ""

# ─────────────────────────────────────────────────────────────────
# STEP 3: Run mAP evaluation
# ─────────────────────────────────────────────────────────────────
echo "[STEP 3/5] Running mAP evaluation (all 6 combos) ..."
echo "─────────────────────────────────────────────────────────────"

cd "$BACKEND_DIR"

python ../scripts/evaluate_dataset.py \
    --model yolov8 yolov5 \
    --compare \
    --annotations "../$ANNO_FILE" \
    --images-dir  "../$IMG_DIR" \
    --confidence 0.25 \
    --iou 0.45 \
    --output "$RESULTS_DIR/eval_report.csv" \
    --save-predictions "$RESULTS_DIR/predictions_last.json"

cd "$PROJECT_ROOT"

echo ""
echo "✓ Evaluation complete"
echo "  Report : $RESULTS_DIR/eval_report.csv"
echo ""

# ─────────────────────────────────────────────────────────────────
# STEP 4: Run latency/FPS benchmark
# ─────────────────────────────────────────────────────────────────
echo "[STEP 4/5] Running inference benchmark (all 6 combos) ..."
echo "─────────────────────────────────────────────────────────────"

cd "$BACKEND_DIR"

python ../scripts/benchmark_models.py \
    --models yolov8 yolov5 \
    --backends pytorch torchscript onnx \
    --sizes 640 \
    --runs 100 \
    --warmup 20 \
    --baseline pytorch \
    --output "$RESULTS_DIR/benchmark.csv"

# Also save JSON for programmatic use
python ../scripts/benchmark_models.py \
    --models yolov8 yolov5 \
    --backends pytorch torchscript onnx \
    --sizes 640 \
    --runs 100 \
    --warmup 20 \
    --baseline pytorch \
    --output "$RESULTS_DIR/benchmark.json"

cd "$PROJECT_ROOT"

echo ""
echo "✓ Benchmark complete"
echo "  CSV  : $RESULTS_DIR/benchmark.csv"
echo "  JSON : $RESULTS_DIR/benchmark.json"
echo ""

# ─────────────────────────────────────────────────────────────────
# STEP 5: Video inference
# ─────────────────────────────────────────────────────────────────
echo "[STEP 5/5] Running video inference ..."
echo "─────────────────────────────────────────────────────────────"

cd "$BACKEND_DIR"

# Use a real video if present; otherwise generate a synthetic test video
VIDEO_PATH="$DATA_DIR/videos/sample.mp4"
if [ ! -f "$VIDEO_PATH" ]; then
    echo "  No video found at $VIDEO_PATH"
    echo "  Generating a synthetic test video ..."
    python ../scripts/run_video_inference.py \
        --generate-test-video \
        --compare \
        --max-frames 100 \
        --output-dir "$OUTPUTS_DIR" \
        --results-dir "$RESULTS_DIR"
else
    echo "  Using video: $VIDEO_PATH"
    python ../scripts/run_video_inference.py \
        --video "$VIDEO_PATH" \
        --compare \
        --max-frames 150 \
        --output-dir "$OUTPUTS_DIR" \
        --results-dir "$RESULTS_DIR"
fi

cd "$PROJECT_ROOT"

echo ""
echo "✓ Video inference complete"
echo "  CSV  : $RESULTS_DIR/video_benchmark.csv"
echo "  JSON : $RESULTS_DIR/video_benchmark.json"
echo ""

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Pipeline Complete — Files Generated"
echo "============================================================"
echo ""
echo "  results/"
ls -lh "$RESULTS_DIR/" 2>/dev/null || echo "    (empty)"
echo ""
echo "  outputs/"
ls -lh "$OUTPUTS_DIR/" 2>/dev/null | head -20 || echo "    (empty)"
echo ""
echo "  Next: commit results to git"
echo "    git add results/ outputs/"
echo "    git commit -m 'Add evaluation + benchmark results'"
echo ""
