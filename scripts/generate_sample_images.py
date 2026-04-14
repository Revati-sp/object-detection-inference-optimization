#!/usr/bin/env python3
"""
Generate sample images for testing without external downloads.
Creates simple synthetic images with various content.
"""

import os
from pathlib import Path

def generate_sample_images():
    """Generate sample images locally."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random
    except ImportError:
        print("❌ PIL not available. Install with: pip install Pillow")
        return False
    
    data_dir = Path(__file__).parent.parent / "data"
    sample_dir = data_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Generating sample images...")
    
    # Define sample images to create
    samples = [
        {
            "name": "sample_01.jpg",
            "title": "Simple Scene",
            "color": (100, 150, 200),
            "objects": ["Rectangle", "Circle"]
        },
        {
            "name": "sample_02.jpg",
            "title": "Colorful Shapes",
            "color": (200, 100, 150),
            "objects": ["Triangle", "Square", "Circle"]
        },
        {
            "name": "sample_03.jpg",
            "title": "Pattern Test",
            "color": (150, 200, 100),
            "objects": ["Grid", "Lines"]
        },
        {
            "name": "sample_04.jpg",
            "title": "Mixed Objects",
            "color": (100, 100, 200),
            "objects": ["Multiple", "Shapes"]
        },
        {
            "name": "sample_05.jpg",
            "title": "Gradient Scene",
            "color": (200, 150, 100),
            "objects": ["Objects", "Gradient"]
        },
    ]
    
    for i, sample in enumerate(samples, 1):
        filepath = sample_dir / sample["name"]
        
        if filepath.exists():
            print(f"  ✓ {sample['name']} (already exists)")
            continue
        
        # Create image
        width, height = 640, 480
        img = Image.new('RGB', (width, height), sample["color"])
        draw = ImageDraw.Draw(img)
        
        # Add title
        title_bbox = draw.textbbox((20, 20), sample["title"])
        draw.text((20, 20), sample["title"], fill=(255, 255, 255))
        
        # Add some random shapes
        for j in range(3):
            x1 = random.randint(50, width - 100)
            y1 = random.randint(100, height - 100)
            x2 = x1 + random.randint(50, 200)
            y2 = y1 + random.randint(50, 150)
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Add some circles
        for j in range(2):
            cx = random.randint(100, width - 100)
            cy = random.randint(150, height - 100)
            r = random.randint(20, 60)
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
        
        # Save
        img.save(filepath, quality=85)
        print(f"  ✅ Created {sample['name']}")
    
    return True


def main():
    data_dir = Path(__file__).parent.parent / "data"
    sample_dir = data_dir / "sample"
    annotations_dir = data_dir / "annotations"
    
    print("="*60)
    print("📊 Sample Data Setup")
    print("="*60)
    
    # Generate images
    success = generate_sample_images()
    
    if not success:
        print("\n⚠️  Could not generate sample images (PIL missing)")
    
    # Check files
    sample_images = sorted(sample_dir.glob("*.jpg"))
    annotation_files = sorted(annotations_dir.glob("*.json"))
    
    print(f"\n✅ Data folder ready:")
    print(f"\n📸 Sample Images: {len(sample_images)}")
    for img in sample_images:
        print(f"   ✓ {img.name}")
    
    print(f"\n📄 Annotations: {len(annotation_files)}")
    for ann in annotation_files:
        print(f"   ✓ {ann.name}")
    
    print(f"\n✨ You can now:")
    print(f"   1. Upload images via the UI (drag-and-drop)")
    print(f"   2. Run detection on sample images")
    print(f"   3. Test evaluation and benchmarking")
    print("="*60)


if __name__ == "__main__":
    main()
