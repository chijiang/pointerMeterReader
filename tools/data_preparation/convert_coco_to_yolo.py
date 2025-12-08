#!/usr/bin/env python3
"""
Convert COCO format to YOLO format for object detection.

Supports Roboflow datasets with pre-split train/valid/test directories.
Each split directory should contain _annotations.coco.json and images.

COCO format (input):
    input_dir/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    └── test/
        ├── _annotations.coco.json
        └── *.jpg

YOLO format (output):
    output_dir/
    ├── images/
    │   ├── train/
    │   │   └── *.jpg
    │   └── val/
    │       └── *.jpg
    ├── labels/
    │   ├── train/
    │   │   └── *.txt
    │   └── val/
    │       └── *.txt
    └── data.yaml

Usage:
    python tools/data_preparation/convert_coco_to_yolo.py \
        --input_dir data/Spill.v2i.coco \
        --output_dir data/spill_yolo
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from tqdm import tqdm


def load_coco_annotations(coco_json_path: Path) -> Tuple[Dict, Dict, List]:
    """
    Load COCO annotations from JSON file.

    Returns:
        images: Dict mapping image_id to image info
        annotations: Dict mapping image_id to list of annotations
        categories: List of category info
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Build image lookup
    images = {}
    for img in coco_data.get('images', []):
        images[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height'],
        }

    # Build annotations lookup (by image_id)
    annotations = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        annotations[ann['image_id']].append(ann)

    categories = coco_data.get('categories', [])

    return images, dict(annotations), categories


def coco_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] (normalized).

    COCO bbox: top-left corner (x, y) and box dimensions (width, height)
    YOLO bbox: center point (x_center, y_center) and dimensions (width, height), all normalized 0-1
    """
    x, y, w, h = bbox

    # Calculate center point
    x_center = x + w / 2
    y_center = y + h / 2

    # Normalize by image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    # Clamp values to [0, 1]
    x_center_norm = max(0, min(1, x_center_norm))
    y_center_norm = max(0, min(1, y_center_norm))
    w_norm = max(0, min(1, w_norm))
    h_norm = max(0, min(1, h_norm))

    return x_center_norm, y_center_norm, w_norm, h_norm


def convert_split(
    source_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    category_id_mapping: Dict[int, int],
) -> Dict[str, int]:
    """
    Convert a single split (train/valid/test) from COCO to YOLO format.

    Returns:
        stats: Dict with conversion statistics
    """
    coco_json = source_dir / "_annotations.coco.json"

    if not coco_json.exists():
        print(f"  Warning: No annotations found at {coco_json}")
        return {'images': 0, 'labels': 0, 'skipped': 0}

    images, annotations, _ = load_coco_annotations(coco_json)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {'images': 0, 'labels': 0, 'skipped': 0, 'boxes': 0}

    for img_id, img_info in tqdm(images.items(), desc=f"Converting {source_dir.name}"):
        src_image = source_dir / img_info['file_name']

        if not src_image.exists():
            stats['skipped'] += 1
            continue

        # Copy image
        dst_image = output_images_dir / img_info['file_name']
        shutil.copy2(src_image, dst_image)
        stats['images'] += 1

        # Create label file
        label_name = Path(img_info['file_name']).stem + '.txt'
        dst_label = output_labels_dir / label_name

        img_annotations = annotations.get(img_id, [])

        with open(dst_label, 'w') as f:
            for ann in img_annotations:
                # Get category ID (mapped to 0-based index)
                cat_id = ann['category_id']
                yolo_class_id = category_id_mapping.get(cat_id, 0)

                # Convert bbox
                bbox = ann.get('bbox', [0, 0, 0, 0])
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue  # Skip invalid boxes

                x_center, y_center, w, h = coco_bbox_to_yolo(
                    bbox,
                    img_info['width'],
                    img_info['height']
                )

                # Write YOLO format: class_id x_center y_center width height
                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                stats['boxes'] += 1

        stats['labels'] += 1

    return stats


def convert_coco_to_yolo(
    input_dir: str,
    output_dir: str,
    clean_output: bool = False,
):
    """
    Main conversion function.

    Args:
        input_dir: Path to COCO dataset (with train/valid/test splits)
        output_dir: Path to output YOLO dataset
        clean_output: Remove existing output directory before conversion
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Clean output if requested
    if clean_output and output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all categories across all splits
    all_categories = {}
    split_mapping = {
        'train': 'train',
        'valid': 'val',
        'test': 'test',
    }

    for src_split in split_mapping.keys():
        coco_json = input_dir / src_split / "_annotations.coco.json"
        if coco_json.exists():
            with open(coco_json, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            for cat in coco_data.get('categories', []):
                all_categories[cat['id']] = cat['name']

    if not all_categories:
        raise ValueError("No categories found in any split")

    # Create category ID mapping (COCO ID -> 0-based index)
    sorted_cat_ids = sorted(all_categories.keys())
    category_id_mapping = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    class_names = [all_categories[cat_id] for cat_id in sorted_cat_ids]

    print(f"\nFound {len(class_names)} classes: {class_names}")
    print(f"Category ID mapping: {category_id_mapping}")

    # Process each split
    all_stats = {}

    for src_split, dst_split in split_mapping.items():
        src_split_dir = input_dir / src_split

        if not src_split_dir.exists():
            print(f"\nSkipping {src_split} (not found)")
            continue

        print(f"\nProcessing {src_split} -> {dst_split}...")

        output_images = output_dir / "images" / dst_split
        output_labels = output_dir / "labels" / dst_split

        stats = convert_split(
            src_split_dir,
            output_images,
            output_labels,
            category_id_mapping,
        )

        all_stats[dst_split] = stats
        print(f"  Images: {stats['images']}, Labels: {stats['labels']}, Boxes: {stats['boxes']}, Skipped: {stats['skipped']}")

    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test' if (output_dir / 'images' / 'test').exists() else None,
        'nc': len(class_names),
        'names': class_names,
    }

    # Remove None values
    data_yaml = {k: v for k, v in data_yaml.items() if v is not None}

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        # Manual YAML writing for simple structure
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        if 'test' in data_yaml:
            f.write(f"test: {data_yaml['test']}\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write("names:\n")
        for name in class_names:
            f.write(f"  - {name}\n")

    print(f"\nSaved data.yaml to: {yaml_path}")

    # Print summary
    print(f"\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/")
    for split, stats in all_stats.items():
        print(f"  │   └── {split}/  ({stats['images']} images)")
    print(f"  ├── labels/")
    for split, stats in all_stats.items():
        print(f"  │   └── {split}/  ({stats['labels']} labels, {stats['boxes']} boxes)")
    print(f"  └── data.yaml")

    total_images = sum(s['images'] for s in all_stats.values())
    total_boxes = sum(s['boxes'] for s in all_stats.values())
    print(f"\nTotal: {total_images} images, {total_boxes} bounding boxes")
    print(f"Classes: {class_names}")

    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO format dataset to YOLO format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to COCO dataset directory (with train/valid/test splits)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output YOLO dataset directory')
    parser.add_argument('--clean', action='store_true',
                        help='Remove existing output directory before conversion')

    args = parser.parse_args()

    convert_coco_to_yolo(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        clean_output=args.clean,
    )


if __name__ == '__main__':
    main()
