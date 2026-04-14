# Data Directory

This folder contains datasets and annotations for object detection evaluation and testing.

## Folder Structure

```
data/
├── images/              # Evaluation dataset images
│   └── val/            # Validation images (add your COCO val2017 images here)
├── annotations/        # COCO-format annotation files
│   └── instances_sample.json    # Sample COCO annotation for testing
├── sample/             # Quick test images (committed to git)
    ├── sample_01.jpg
    ├── sample_02.jpg
    ├── sample_03.jpg
    ├── sample_04.jpg
    └── sample_05.jpg
```

## Usage

### 1. Quick Testing (Use Sample Images)
```bash
# These sample images are already available for testing
# Use them in the web UI by uploading via drag-and-drop
# They're in: data/sample/
```

### 2. Running Evaluation
```bash
# Place your dataset here:
# 1. Add images to: data/images/
# 2. Add COCO annotations to: data/annotations/
# 3. Run evaluation via the UI or:

python scripts/evaluate_dataset.py --dataset-dir data/images --annotations data/annotations/instances.json
```

### 3. Downloading Full COCO Dataset
```bash
# To download official COCO 2017 validation set (~1GB)
python scripts/download_sample_data.py

# Then select 'y' when prompted
```

## COCO Annotation Format

Each annotation file should be in COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [10, 20, 100, 150],
      "area": 15000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"}
  ]
}
```

## Sample Data

Sample images and annotations are included:

| File | Purpose | Size |
|------|---------|------|
| `data/sample/*.jpg` | Quick testing images (5 files) | ~52 KB |
| `data/annotations/instances_sample.json` | Sample COCO annotation | ~5 KB |

These are **tracked in git** for immediate testing after cloning.

## Adding Your Own Data

### Option 1: Using COCO Format
1. Place images in: `data/images/`
2. Add `instances.json` to: `data/annotations/`
3. Run evaluation

### Option 2: Using Custom Format
For non-COCO images, you can:
1. Upload via the web UI (drag-and-drop)
2. No need to use this folder
3. Web UI handles individual images

### Option 3: Using Official COCO
```bash
# Download COCO 2017 validation set
cd data
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d images/

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d annotations/
```

## Scripts

### Generate Sample Images
```bash
python scripts/generate_sample_images.py
```
Creates synthetic sample images for quick testing.

### Download Official Data
```bash
python scripts/download_sample_data.py
```
Downloads COCO dataset (optional, ~1GB).

### Evaluate Dataset
```bash
python scripts/evaluate_dataset.py --dataset-dir data/images --annotations data/annotations/instances.json
```

### Compare Models
```bash
python scripts/compare_models.py --dataset-dir data/images --annotations data/annotations/instances.json
```

## Notes

- `.gitignore` excludes large dataset files
- `sample/` folder **IS committed** for immediate use
- `images/` and `annotations/` for **user data** (not committed)
- Maximum image size: ~50 MB per image
- Supported formats: JPG, PNG, BMP, GIF

## Getting Started

**No data setup needed!** You can:

1. **Use sample images immediately** (they're already included)
2. **Upload any image via the web UI** (no folder setup required)
3. **Add COCO datasets later** if you want evaluation metrics

---

**Questions?** See `README.md` for full documentation.
