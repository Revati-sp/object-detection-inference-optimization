# Manual Annotation Workflow

> **This step is required before final submission.**
> The assignment requires that mAP accuracy be measured against *your own human annotations*,
> not model-generated pseudo-labels.  This document explains the end-to-end workflow.

---

## Why manual annotations are required

The file `data/annotations/instances_custom.json` was created automatically by
`scripts/create_custom_annotations.py`, which runs YOLOv8n on the evaluation images and
saves its detections as "ground truth" (pseudo-labeling).

Evaluating YOLOv8/PyTorch against annotations that YOLOv8 itself produced gives
**mAP = 1.000** — a circular, trivially-perfect result that proves nothing about
real detection accuracy.  The assignment requirement is:

> *"Accuracy/mAP must be based on the project's own annotations."*

This means **human-labelled bounding boxes** that are independent of any model output.
The pseudo-label file can be kept as a sanity check (do all backends agree?), but it
must not appear as the primary accuracy result in your report.

---

## Quick-start checklist

- [ ] Choose an annotation tool (see options below)
- [ ] Annotate ≥ 50 images in `data/images/val/`
- [ ] Export as COCO JSON
- [ ] Save to `data/annotations/instances_manual.json`
- [ ] Run `python scripts/validate_annotations.py --annotations data/annotations/instances_manual.json`
- [ ] Confirm output: `Status: ✓ MANUAL ANNOTATIONS — valid for final submission`
- [ ] Run final mAP evaluation (see Step 5 below)

---

## Step 1 — Choose an annotation tool

Three free options are listed below.  **LabelImg** is the fastest to get started with
locally; **Roboflow** requires less setup if you are comfortable uploading images to the
cloud.

### Option A — LabelImg (desktop, recommended for quick setup)

```bash
pip install labelImg
labelImg
```

1. Open `data/images/val/` as the image folder.
2. Set the save directory to a temporary folder (e.g. `data/annotations/labelimg_tmp/`).
3. In the menu, select **Format → COCO** (not PascalVOC).
4. Draw rectangles around objects, assign class labels from the COCO vocabulary
   (person, car, bicycle, laptop, chair, …).
5. Press **Save** after each image.

> LabelImg COCO export produces one JSON file per image.  Use the converter in Step 3.

### Option B — CVAT (web-based, free cloud or self-hosted)

**Cloud (no install required):**

1. Go to [app.cvat.ai](https://app.cvat.ai) and create a free account.
2. Create a new project → set label schema to COCO 80-class.
3. Create a task → upload all images from `data/images/val/`.
4. Annotate using the rectangle tool.
5. **Export → COCO 1.0** format.
6. Download the exported ZIP and extract `instances_default.json`.

**Self-hosted (Docker):**

```bash
git clone https://github.com/opencv/cvat && cd cvat
docker compose up -d
# Open http://localhost:8080
```

CVAT exports a single `instances_default.json` directly in COCO format — no
conversion needed.

### Option C — Roboflow (cloud, free tier)

1. Go to [roboflow.com](https://roboflow.com) and create a free account.
2. Create a new project → type **Object Detection**.
3. Upload images from `data/images/val/`.
4. Annotate using the web interface.
5. **Export → COCO JSON** format.
6. Download the dataset ZIP; the annotation file is `_annotations.coco.json`.

---

## Step 2 — Annotate the images

Guidelines for annotation quality:

- **Minimum coverage:** annotate ≥ 50 images for a statistically meaningful mAP score.
  All 139 images is ideal.
- **Box tightness:** draw boxes that closely fit the visible part of each object.
- **Class names:** use the COCO 80-class vocabulary (same classes as
  `data/annotations/instances_custom.json` — person, car, laptop, chair, etc.).
- **Truncated objects:** if an object is partly cut off by the image edge, still annotate
  the visible portion; set `iscrowd: 0`.
- **Minimum 1 annotation per image:** images with zero detections are valid (just
  don't annotate anything for that image).

---

## Step 3 — Convert to COCO format and save

Your exported file must follow this COCO structure:

```json
{
  "images": [
    {"id": 1, "file_name": "Screenshot 2026-04-17 at 7.26.32 PM.png", "width": 714, "height": 1034}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 73,
      "bbox": [120.0, 45.0, 310.0, 210.0],
      "area": 65100.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 73, "name": "laptop", "supercategory": "electronic"}
  ]
}
```

**Important notes:**

- `bbox` format is `[x, y, width, height]` (top-left corner + dimensions), NOT
  `[x1, y1, x2, y2]`.
- `category_id` must use the **official COCO non-contiguous IDs** (1–90, with gaps
  at 12, 26, 29, 30, 45, 66, 68, 69, 71, 83).  The correct mapping is already in
  `data/annotations/instances_manual.json` under `"categories"`.
- `file_name` must match exactly the filenames in `data/images/val/` (including spaces
  and the `.png` extension).

**If using LabelImg** (which exports per-image JSON files), merge them with:

```python
import json, glob, pathlib

images, annotations, ann_id = [], [], 1
for i, path in enumerate(sorted(glob.glob("data/annotations/labelimg_tmp/*.json")), start=1):
    data = json.load(open(path))
    img = data["images"][0]
    img["id"] = i
    images.append(img)
    for ann in data["annotations"]:
        ann["id"] = ann_id
        ann["image_id"] = i
        annotations.append(ann)
        ann_id += 1

merged = {
    "info": {"description": "Manual annotations", "contributor": "human-annotated"},
    "images": images,
    "annotations": annotations,
    "categories": json.load(open("data/annotations/instances_manual.json"))["categories"],
}
json.dump(merged, open("data/annotations/instances_manual.json", "w"), indent=2)
print(f"Merged: {len(images)} images, {len(annotations)} annotations")
```

**If using CVAT or Roboflow**, simply copy the exported file:

```bash
# CVAT
cp path/to/cvat_export/instances_default.json data/annotations/instances_manual.json

# Roboflow
cp path/to/roboflow_export/_annotations.coco.json data/annotations/instances_manual.json
```

> Make sure the `info.contributor` field does **not** contain "auto-annotated".
> The validator checks this field.  Set it to your name or "human-annotated".

---

## Step 4 — Validate the annotation file

```bash
# From the project root with the backend venv active:
python scripts/validate_annotations.py \
    --annotations data/annotations/instances_manual.json
```

Expected output for a valid file:

```
=================================================================
  Annotation File Validation Report
=================================================================
  File         : data/annotations/instances_manual.json
  Images       : 139
  Annotations  : 412
  Categories   : 80
  Contributor  : human-annotated

  Status  : ✓  MANUAL ANNOTATIONS — valid for final submission

  No pseudo-label indicators were detected. These annotations are
  human-created and suitable for reporting final mAP accuracy.
=================================================================
```

If you see `Status: ✗ PSEUDO-LABELS DETECTED`, the file still contains
model-generated annotations or the `info.contributor` field says "auto-annotated".
Fix those issues and re-validate.

---

## Step 5 — Run the final mAP evaluation

```bash
# From the project root, with the backend venv active:
cd backend

python ../scripts/evaluate_dataset.py \
    --model yolov8 yolov5 --compare \
    --annotation-type manual \
    --annotations ../data/annotations/instances_manual.json \
    --images-dir   ../data/images/val \
    --output       ../results/eval_report_manual.csv

cd ..
```

The `--annotation-type manual` flag makes the script abort if it detects pseudo-labels,
ensuring you never accidentally report pseudo-label results as final accuracy.

Results are saved to `results/eval_report_manual.csv` with an `annotation_type` column
that reads `manual` — proving to a grader that the numbers come from real annotations.

---

## Final Evaluation Protocol (all steps)

| # | Command | Purpose |
|---|---------|---------|
| 1 | `python scripts/validate_annotations.py --annotations data/annotations/instances_manual.json` | Confirm human-annotated |
| 2 | `cd backend && python ../scripts/run_all_exports.py` | Build TorchScript + ONNX weight files |
| 3 | `cd backend && python ../scripts/evaluate_dataset.py --model yolov8 yolov5 --compare --annotation-type manual --annotations ../data/annotations/instances_manual.json --images-dir ../data/images/val --output ../results/eval_report_manual.csv` | Final accuracy (mAP) |
| 4 | `cd backend && python ../scripts/benchmark_models.py --models yolov8 yolov5 --backends pytorch torchscript onnx onnx_quant --runs 100 --output ../results/benchmark.csv` | Speed (latency / FPS) |
| 5 | `cd backend && python ../scripts/run_video_inference.py --video ../data/videos/<your_video>.mp4 --model yolov8 yolov5 --backend pytorch --max-frames 150 --results-dir ../results` | Video benchmark (both models) |

Results to include in your report:
- `results/eval_report_manual.csv` — mAP@0.5 and mAP@0.5:0.95 per model/backend
- `results/benchmark.csv`          — latency and FPS
- `results/video_benchmark.csv`    — per-model video inference speed

---

## Notes on pseudo-labels (instances_custom.json)

The existing `data/annotations/instances_custom.json` file is kept for convenience:

- **Sanity check**: compare backends against each other — if TorchScript and ONNX both
  score ~0.97 while PyTorch scores 1.0, the backends are consistent.
- **Cross-model comparison**: YOLOv5's score (~0.71) against YOLOv8-generated labels
  shows real model-to-model difference.

To run a pseudo-label sanity check without being blocked:

```bash
cd backend
python ../scripts/evaluate_dataset.py \
    --model yolov8 yolov5 --compare \
    --annotation-type pseudo \
    --annotations ../data/annotations/instances_custom.json \
    --images-dir   ../data/images/val \
    --output       ../results/eval_report_pseudo.csv
```

The output CSV will have `annotation_type = pseudo` in every row, clearly marking
these results as a sanity check rather than final accuracy.
