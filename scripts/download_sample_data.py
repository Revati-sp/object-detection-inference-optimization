#!/usr/bin/env python3
"""
Download sample COCO data for testing and evaluation.

This script downloads:
- A small subset of COCO 2017 validation images
- Corresponding annotations
- Organizes them in the data/ folder

Usage:
    python scripts/download_sample_data.py
"""

import os
import json
import urllib.request
import zipfile
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
SAMPLE_DIR = DATA_DIR / "sample"

# COCO data URLs (small sample)
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Sample image URLs (alternative if COCO download fails)
SAMPLE_IMAGES = [
    "https://images.unsplash.com/photo-1552053831-71594a27c62d?w=640&q=80",  # car
    "https://images.unsplash.com/photo-1494119e36519-410a18fd38a4?w=640&q=80",  # dog
    "https://images.unsplash.com/photo-1520763185298-1b434c919eba?w=640&q=80",  # people
    "https://images.unsplash.com/photo-1543269863-cbf427effbad?w=640&q=80",  # sports
    "https://images.unsplash.com/photo-1489749798305-4fea3ba63d60?w=640&q=80",  # crowd
]


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Directories ready")


def download_file(url, destination, chunk_size=8192):
    """Download file with progress bar."""
    print(f"📥 Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✅ Downloaded to: {destination}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def download_coco_data():
    """Download full COCO dataset (optional, large)."""
    print("\n🔄 COCO Dataset Download (Optional)")
    print("⚠️  This downloads ~1GB of data. Proceed? (y/n): ", end="")
    
    response = input().strip().lower()
    if response != 'y':
        print("⏭️  Skipped COCO download")
        return False
    
    # Download images
    images_zip = DATA_DIR / "val2017.zip"
    if not images_zip.exists():
        if not download_file(COCO_IMAGES_URL, images_zip):
            return False
        
        # Extract
        print("📦 Extracting images...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(IMAGES_DIR)
        os.remove(images_zip)
        print("✅ Images extracted")
    
    # Download annotations
    annotations_zip = DATA_DIR / "annotations_trainval2017.zip"
    if not annotations_zip.exists():
        if not download_file(COCO_ANNOTATIONS_URL, annotations_zip):
            return False
        
        # Extract
        print("📦 Extracting annotations...")
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(ANNOTATIONS_DIR)
        os.remove(annotations_zip)
        print("✅ Annotations extracted")
    
    return True


def download_sample_images():
    """Download sample images from Unsplash."""
    print("\n📸 Downloading sample images...")
    
    for i, url in enumerate(SAMPLE_IMAGES, 1):
        filename = f"sample_{i:02d}.jpg"
        filepath = SAMPLE_DIR / filename
        
        if not filepath.exists():
            try:
                print(f"  📥 Downloading sample {i}/5...")
                urllib.request.urlretrieve(url, filepath)
                print(f"  ✅ {filename}")
            except Exception as e:
                print(f"  ⚠️  Failed to download {filename}: {e}")
        else:
            print(f"  ✓ {filename} (already exists)")
    
    print("✅ Sample images ready")


def create_minimal_coco_annotation():
    """Create a minimal COCO annotation file for testing."""
    print("\n📝 Creating minimal COCO annotation...")
    
    coco_annotation = {
        "info": {
            "description": "Sample COCO dataset for testing Object Detection model",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            }
        ],
        "images": [
            {
                "license": 1,
                "file_name": "sample_01.jpg",
                "coco_url": "http://mscoco.org/images/1",
                "height": 426,
                "width": 640,
                "date_captured": "2024-01-01 00:00:00",
                "flickr_url": "http://www.flickr.com/photos/1/1",
                "id": 1
            },
            {
                "license": 1,
                "file_name": "sample_02.jpg",
                "coco_url": "http://mscoco.org/images/2",
                "height": 426,
                "width": 640,
                "date_captured": "2024-01-01 00:00:00",
                "flickr_url": "http://www.flickr.com/photos/1/2",
                "id": 2
            },
            {
                "license": 1,
                "file_name": "sample_03.jpg",
                "coco_url": "http://mscoco.org/images/3",
                "height": 426,
                "width": 640,
                "date_captured": "2024-01-01 00:00:00",
                "flickr_url": "http://www.flickr.com/photos/1/3",
                "id": 3
            },
            {
                "license": 1,
                "file_name": "sample_04.jpg",
                "coco_url": "http://mscoco.org/images/4",
                "height": 426,
                "width": 640,
                "date_captured": "2024-01-01 00:00:00",
                "flickr_url": "http://www.flickr.com/photos/1/4",
                "id": 4
            },
            {
                "license": 1,
                "file_name": "sample_05.jpg",
                "coco_url": "http://mscoco.org/images/5",
                "height": 426,
                "width": 640,
                "date_captured": "2024-01-01 00:00:00",
                "flickr_url": "http://www.flickr.com/photos/1/5",
                "id": 5
            }
        ],
        "annotations": [
            {
                "segmentation": [[0, 0, 640, 0, 640, 426, 0, 426]],
                "area": 272640,
                "iscrowd": 0,
                "image_id": 1,
                "bbox": [0, 0, 640, 426],
                "category_id": 1,
                "id": 1
            },
            {
                "segmentation": [[0, 0, 640, 0, 640, 426, 0, 426]],
                "area": 272640,
                "iscrowd": 0,
                "image_id": 2,
                "bbox": [0, 0, 640, 426],
                "category_id": 18,
                "id": 2
            },
            {
                "segmentation": [[0, 0, 640, 0, 640, 426, 0, 426]],
                "area": 272640,
                "iscrowd": 0,
                "image_id": 3,
                "bbox": [0, 0, 640, 426],
                "category_id": 1,
                "id": 3
            },
            {
                "segmentation": [[0, 0, 640, 0, 640, 426, 0, 426]],
                "area": 272640,
                "iscrowd": 0,
                "image_id": 4,
                "bbox": [0, 0, 640, 426],
                "category_id": 1,
                "id": 4
            },
            {
                "segmentation": [[0, 0, 640, 0, 640, 426, 0, 426]],
                "area": 272640,
                "iscrowd": 0,
                "image_id": 5,
                "bbox": [0, 0, 640, 426],
                "category_id": 1,
                "id": 5
            }
        ],
        "categories": [
            {"supercategory": "person", "id": 1, "name": "person"},
            {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
            {"supercategory": "vehicle", "id": 3, "name": "car"},
            {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
            {"supercategory": "vehicle", "id": 5, "name": "airplane"},
            {"supercategory": "vehicle", "id": 6, "name": "bus"},
            {"supercategory": "vehicle", "id": 7, "name": "train"},
            {"supercategory": "vehicle", "id": 8, "name": "truck"},
            {"supercategory": "vehicle", "id": 9, "name": "boat"},
            {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
            {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
            {"supercategory": "outdoor", "id": 12, "name": "stop sign"},
            {"supercategory": "outdoor", "id": 13, "name": "parking meter"},
            {"supercategory": "outdoor", "id": 14, "name": "bench"},
            {"supercategory": "animal", "id": 15, "name": "cat"},
            {"supercategory": "animal", "id": 16, "name": "dog"},
            {"supercategory": "animal", "id": 17, "name": "horse"},
            {"supercategory": "animal", "id": 18, "name": "sheep"},
            {"supercategory": "animal", "id": 19, "name": "cow"},
            {"supercategory": "animal", "id": 20, "name": "elephant"}
        ]
    }
    
    annotation_file = ANNOTATIONS_DIR / "instances_sample.json"
    with open(annotation_file, 'w') as f:
        json.dump(coco_annotation, f, indent=2)
    
    print(f"✅ Created: {annotation_file}")


def print_summary():
    """Print summary of downloaded data."""
    print("\n" + "="*60)
    print("📊 DATA SETUP COMPLETE")
    print("="*60)
    
    # Check what's available
    sample_images = list(SAMPLE_DIR.glob("*.jpg"))
    annotation_files = list(ANNOTATIONS_DIR.glob("*.json"))
    
    print(f"\n📁 Sample Images: {len(sample_images)} files")
    for img in sorted(sample_images):
        print(f"   ✓ {img.name}")
    
    print(f"\n📄 Annotations: {len(annotation_files)} files")
    for ann in sorted(annotation_files):
        print(f"   ✓ {ann.name}")
    
    print(f"\n📁 Folder structure:")
    print(f"   data/")
    print(f"   ├── images/        {len(list(IMAGES_DIR.glob('*')))} items")
    print(f"   ├── annotations/   {len(list(ANNOTATIONS_DIR.glob('*')))} items")
    print(f"   └── sample/        {len(list(SAMPLE_DIR.glob('*')))} items")
    
    print(f"\n✅ You can now:")
    print(f"   1. Use 'data/sample/' images in the UI (drag-and-drop)")
    print(f"   2. Run evaluation with 'data/images/' and 'data/annotations/'")
    print(f"   3. Test benchmarking on sample images")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("🚀 Object Detection Sample Data Downloader")
    print("="*60)
    
    try:
        create_directories()
        download_sample_images()
        create_minimal_coco_annotation()
        download_coco_data()  # Optional
        print_summary()
        
        print("\n✅ Setup complete! Ready to use.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
