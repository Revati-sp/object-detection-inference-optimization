# Data Directory

This folder contains the custom dataset used for object detection evaluation.

## Folder Structure

```
data/
├── images/
│   └── val/                         # 139 committed PNG screenshots
│       ├── Screenshot 2026-04-17 at 7.26.32 PM.png
│       └── ...
├── annotations/
│   ├── instances_custom.json        # COCO-format ground truth (committed)
│   └── instances_sample.json        # Minimal 5-image sample for smoke-testing
└── sample/                          # Legacy 5 synthetic images (committed)
    ├── sample_01.jpg
    └── ...
```

## Dataset Description

| Property | Value |
|----------|-------|
| Images | **139** PNG screenshots in `data/images/val/` |
| Annotations | `data/annotations/instances_custom.json` (COCO format) |
| Total bounding boxes | **374** |
| Object classes | 36 COCO classes (person, laptop, chair, car, dog, cat, …) |
| Annotation method | Pseudo-labeling via YOLOv8n at confidence ≥ 0.5 |
| Image source | Custom screenshots captured during real environment use |

### Object classes present

apple, bed, bench, bicycle, bird, boat, book, bottle, bowl, car, carrot, cat,
chair, clock, couch, cup, dining table, dog, donut, keyboard, laptop,
microwave, mouse, orange, oven, person, potted plant, refrigerator, sandwich,
spoon, sports ball, suitcase, teddy bear, traffic light, tv, vase

## Annotation File: `instances_custom.json`

COCO-format JSON with three required keys:

```json
{
  "images":      [ { "id": 1, "file_name": "Screenshot …png", "height": 900, "width": 1440 } ],
  "annotations": [ { "id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "area": …, "iscrowd": 0 } ],
  "categories":  [ { "id": 1, "name": "person" }, … ]
}
```

Category IDs use the **official 91-class COCO mapping** (non-contiguous 1–90),
matching what `pycocotools` expects.

## Reproducing Annotations

If you need to re-generate `instances_custom.json` (e.g. after adding more images):

```bash
# From the project root, with the backend venv active
source backend/venv/bin/activate

python scripts/create_custom_annotations.py \
    --images-dir data/images/val \
    --output     data/annotations/instances_custom.json \
    --conf       0.50 \
    --min-anns   1
```

## Running Evaluation

```bash
# From the project root, with the backend venv active
source backend/venv/bin/activate
cd backend

python ../scripts/evaluate_dataset.py \
    --model yolov8 yolov5 --compare \
    --annotations ../data/annotations/instances_custom.json \
    --images-dir  ../data/images/val \
    --output      ../results/eval_report.csv
```

## Notes

- `data/images/val/` and `data/annotations/instances_custom.json` are **committed to git**.
- Large files (model weights, generated videos) are **excluded** by `.gitignore`.
- `data/sample/` contains 5 legacy synthetic images kept for smoke-testing the API.
