#!/usr/bin/env python3
"""
Download a 200-image subset of COCO val2017 with proper ground-truth annotations.

Strategy
--------
1. Download the COCO val2017 annotation JSON (~241 MB compressed, ~121 MB JSON).
2. Pick the first N images that have at least MIN_ANNS annotations.
3. Download only those images from images.cocodataset.org (each ~50–300 KB).
4. Write a trimmed annotation file to data/annotations/instances_val200.json.

Usage
-----
    python scripts/prepare_coco_subset.py
    python scripts/prepare_coco_subset.py --num-images 100 --output-dir data
    python scripts/prepare_coco_subset.py --num-images 500 --min-anns 3

The script is idempotent: already-downloaded images are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Set

ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
IMAGES_BASE_URL = "http://images.cocodataset.org/val2017"


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare a COCO val2017 subset for evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-images", type=int, default=200,
                   help="How many images to include")
    p.add_argument("--min-anns", type=int, default=1,
                   help="Skip images with fewer than this many annotations")
    p.add_argument("--output-dir", default="data",
                   help="Root data directory (images → {dir}/images/val, "
                        "annotations → {dir}/annotations/)")
    p.add_argument("--skip-images", action="store_true",
                   help="Only write the annotation JSON; don't download images "
                        "(useful if images already exist)")
    return p.parse_args()


# ── Download helpers ───────────────────────────────────────────────────────

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:5.1f}%  "
              f"({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)", end="", flush=True)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    print(f"  → {dest}")
    urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
    print()


def download_image(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a single image; returns True on success."""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, str(dest))
            return True
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"    ✗ Failed after {retries} attempts: {exc}")
    return False


# ── Main logic ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    root = Path(args.output_dir)
    ann_dir = root / "annotations"
    img_dir = root / "images" / "val"
    cache_dir = root / ".cache"

    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: get annotations JSON ──────────────────────────────────────
    full_ann_path = ann_dir / "instances_val2017.json"
    if not full_ann_path.exists():
        zip_path = cache_dir / "annotations_trainval2017.zip"
        if not zip_path.exists():
            print("[1/3] Downloading COCO annotation archive (~242 MB) …")
            download_file(ANNOTATIONS_URL, zip_path)
        else:
            print("[1/3] Using cached annotation zip.")
        print("  Extracting …")
        with zipfile.ZipFile(zip_path) as zf:
            # Only extract the val2017 instances file
            target = "annotations/instances_val2017.json"
            zf.extract(target, str(root))
        print(f"  Extracted → {full_ann_path}")
    else:
        print(f"[1/3] Found existing annotations: {full_ann_path}")

    # ── Step 2: build subset ──────────────────────────────────────────────
    print(f"\n[2/3] Building {args.num_images}-image subset (min_anns={args.min_anns}) …")
    with open(full_ann_path) as f:
        coco = json.load(f)

    # Count annotations per image
    ann_count: Dict[int, int] = {}
    for ann in coco["annotations"]:
        ann_count[ann["image_id"]] = ann_count.get(ann["image_id"], 0) + 1

    # Filter images with enough annotations
    filtered_images = [
        img for img in coco["images"]
        if ann_count.get(img["id"], 0) >= args.min_anns
    ]
    print(f"  Images with ≥{args.min_anns} annotation(s): {len(filtered_images)}")

    selected = filtered_images[: args.num_images]
    selected_ids: Set[int] = {img["id"] for img in selected}
    print(f"  Selected: {len(selected)} images")

    # Build trimmed annotation file
    subset_anns = [a for a in coco["annotations"] if a["image_id"] in selected_ids]
    used_cat_ids: Set[int] = {a["category_id"] for a in subset_anns}
    subset_cats = [c for c in coco["categories"] if c["id"] in used_cat_ids]

    subset = {
        "info": {
            "description": f"COCO val2017 {len(selected)}-image subset",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "url": "http://cocodataset.org",
        },
        "licenses": coco.get("licenses", []),
        "images": selected,
        "annotations": subset_anns,
        "categories": coco["categories"],   # keep all 80 cats for correct ID mapping
    }

    out_json = ann_dir / f"instances_val{len(selected)}.json"
    with open(out_json, "w") as f:
        json.dump(subset, f)
    print(f"  Wrote annotation subset → {out_json}")
    print(f"  Images: {len(selected)}, Annotations: {len(subset_anns)}, "
          f"Categories used: {len(used_cat_ids)}")

    # ── Step 3: download images ───────────────────────────────────────────
    if args.skip_images:
        print("\n[3/3] --skip-images set; skipping image download.")
    else:
        print(f"\n[3/3] Downloading {len(selected)} images → {img_dir} …")
        ok = skipped = failed = 0
        for i, img_info in enumerate(selected, 1):
            fname = img_info["file_name"]
            dest = img_dir / fname
            if dest.exists():
                skipped += 1
                continue
            url = f"{IMAGES_BASE_URL}/{fname}"
            success = download_image(url, dest)
            if success:
                ok += 1
            else:
                failed += 1
            if i % 20 == 0 or i == len(selected):
                print(f"  Progress: {i}/{len(selected)}  "
                      f"(ok={ok} skipped={skipped} failed={failed})", flush=True)

        print(f"\n  Done — ok={ok}, skipped={skipped}, failed={failed}")

    print("\n" + "=" * 60)
    print("  COCO subset ready!")
    print("=" * 60)
    print(f"  Annotation file : {out_json}")
    print(f"  Images dir      : {img_dir}")
    print()
    print("  Next steps:")
    print(f"  1. Export model weights:")
    print(f"     cd backend")
    print(f"     python ../scripts/export_torchscript.py --model yolov8")
    print(f"     python ../scripts/export_torchscript.py --model yolov5")
    print(f"     python ../scripts/export_onnx.py --model yolov8")
    print(f"     python ../scripts/export_onnx.py --model yolov5")
    print()
    print(f"  2. Run evaluation:")
    print(f"     python scripts/evaluate_dataset.py \\")
    print(f"       --model yolov8 yolov5 --compare \\")
    print(f"       --annotations {out_json} \\")
    print(f"       --images-dir {img_dir} \\")
    print(f"       --output results/eval_report.csv")
    print()
    print(f"  3. Run benchmark:")
    print(f"     python scripts/benchmark_models.py \\")
    print(f"       --models yolov8 yolov5 \\")
    print(f"       --backends pytorch torchscript onnx \\")
    print(f"       --runs 100 --output results/benchmark.csv")


if __name__ == "__main__":
    main()
