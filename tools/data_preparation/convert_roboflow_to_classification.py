#!/usr/bin/env python3
"""
Convert Roboflow COCO format to classification format.

For Roboflow datasets with pre-split train/valid/test directories:
- Images with target annotations -> positive class (1_has_spill)
- Images without target annotations -> negative class (0_no_spill)

Roboflow structure:
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

Output structure:
    output_dir/
    ├── train/
    │   ├── 0_no_spill/
    │   └── 1_has_spill/
    ├── val/
    │   ├── 0_no_spill/
    │   └── 1_has_spill/
    └── class_mapping.json

Usage:
    python tools/data_preparation/convert_roboflow_to_classification.py \
        --input_dir data/Spill.v2i.coco \
        --output_dir data/spill_classification \
        --positive_class has_spill \
        --negative_class no_spill
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from tqdm import tqdm


def load_coco_split(split_dir: Path) -> Tuple[Dict, Set]:
    """
    Load COCO annotations from a split directory.

    Args:
        split_dir: Directory containing _annotations.coco.json and images

    Returns:
        images_info: Dict mapping filename to image info
        images_with_annotations: Set of filenames that have annotations
    """
    coco_json = split_dir / "_annotations.coco.json"

    if not coco_json.exists():
        print(f"  Warning: No annotations found in {split_dir}")
        return {}, set()

    with open(coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Build image info dict (by filename)
    images_info = {}
    id_to_filename = {}
    for img in coco_data.get('images', []):
        filename = img['file_name']
        images_info[filename] = {
            'file_name': filename,
            'width': img.get('width', 0),
            'height': img.get('height', 0),
            'id': img['id'],
        }
        id_to_filename[img['id']] = filename

    # Find images with annotations
    images_with_annotations = set()
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        if image_id in id_to_filename:
            images_with_annotations.add(id_to_filename[image_id])

    # Get category info
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

    print(f"  Loaded {len(images_info)} images, {len(images_with_annotations)} with annotations")
    print(f"  Categories: {list(categories.values())}")

    return images_info, images_with_annotations


def copy_images_to_classification(
    source_dir: Path,
    output_dir: Path,
    images_info: Dict,
    images_with_annotations: Set,
    class_names: Tuple[str, str],
) -> Dict[str, int]:
    """
    Copy images to classification folder structure.

    Returns:
        stats: Dict with copy statistics
    """
    stats = {'positive': 0, 'negative': 0, 'missing': 0}

    neg_dir = output_dir / class_names[0]
    pos_dir = output_dir / class_names[1]
    neg_dir.mkdir(parents=True, exist_ok=True)
    pos_dir.mkdir(parents=True, exist_ok=True)

    for filename, info in tqdm(images_info.items(), desc=f"Processing {source_dir.name}"):
        src = source_dir / filename

        if not src.exists():
            stats['missing'] += 1
            continue

        if filename in images_with_annotations:
            # Has annotation -> positive class
            dst = pos_dir / filename
            stats['positive'] += 1
        else:
            # No annotation -> negative class
            dst = neg_dir / filename
            stats['negative'] += 1

        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"  Error copying {src}: {e}")

    return stats


def convert_roboflow_to_classification(
    input_dir: str,
    output_dir: str,
    positive_class_name: str = 'has_spill',
    negative_class_name: str = 'no_spill',
    use_valid_as_val: bool = True,
    clean_output: bool = False,
):
    """
    Main conversion function for Roboflow COCO datasets.

    Args:
        input_dir: Path to Roboflow dataset (with train/valid/test)
        output_dir: Path to output directory
        positive_class_name: Name for positive class (has annotations)
        negative_class_name: Name for negative class (no annotations)
        use_valid_as_val: Rename 'valid' to 'val' in output
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

    # Define class folder names
    class_names = (f'0_{negative_class_name}', f'1_{positive_class_name}')

    # Process each split
    split_mapping = {
        'train': 'train',
        'valid': 'val' if use_valid_as_val else 'valid',
        'test': 'test',
    }

    all_stats = {}

    for src_split, dst_split in split_mapping.items():
        src_split_dir = input_dir / src_split

        if not src_split_dir.exists():
            print(f"\nSkipping {src_split} (not found)")
            continue

        print(f"\nProcessing {src_split} -> {dst_split}...")

        # Load annotations
        images_info, images_with_annotations = load_coco_split(src_split_dir)

        if not images_info:
            continue

        # Create output directory for this split
        dst_split_dir = output_dir / dst_split

        # Copy images
        stats = copy_images_to_classification(
            src_split_dir,
            dst_split_dir,
            images_info,
            images_with_annotations,
            class_names,
        )

        all_stats[dst_split] = stats
        print(f"  Positive: {stats['positive']}, Negative: {stats['negative']}, Missing: {stats['missing']}")

    # Check if we have any negative samples
    total_negative = sum(s.get('negative', 0) for s in all_stats.values())
    total_positive = sum(s.get('positive', 0) for s in all_stats.values())

    if total_negative == 0:
        print("\n" + "=" * 60)
        print("WARNING: No negative samples found!")
        print("This dataset only contains positive (annotated) images.")
        print("For binary classification, you need to add negative samples.")
        print("=" * 60)
        print("\nOptions:")
        print("1. Add images without spills to the 0_no_spill folders")
        print("2. Use data augmentation to create synthetic negative samples")
        print("3. Collect additional negative samples manually")

    # Save class mapping
    mapping = {
        'classes': {
            '0': negative_class_name,
            '1': positive_class_name,
        },
        'class_to_idx': {
            negative_class_name: 0,
            positive_class_name: 1,
        },
        'num_classes': 2,
        'positive_class': positive_class_name,
        'negative_class': negative_class_name,
        'stats': all_stats,
    }

    mapping_file = output_dir / 'class_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\nSaved class mapping to: {mapping_file}")

    # Print summary
    print(f"\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")

    for split, stats in all_stats.items():
        print(f"  ├── {split}/")
        print(f"  │   ├── {class_names[0]}/  ({stats.get('negative', 0)} images)")
        print(f"  │   └── {class_names[1]}/  ({stats.get('positive', 0)} images)")

    print(f"  └── class_mapping.json")
    print(f"\nTotal: {total_positive} positive, {total_negative} negative")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Roboflow COCO dataset to classification format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to Roboflow dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--positive_class', type=str, default='has_spill',
                        help='Name for positive class (default: has_spill)')
    parser.add_argument('--negative_class', type=str, default='no_spill',
                        help='Name for negative class (default: no_spill)')
    parser.add_argument('--clean', action='store_true',
                        help='Remove existing output directory before conversion')

    args = parser.parse_args()

    convert_roboflow_to_classification(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        positive_class_name=args.positive_class,
        negative_class_name=args.negative_class,
        clean_output=args.clean,
    )


if __name__ == '__main__':
    main()
