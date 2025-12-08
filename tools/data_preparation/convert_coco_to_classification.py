#!/usr/bin/env python3
"""
Convert COCO format annotations to classification format.

For binary classification tasks (e.g., water detection):
- Images with target annotations -> positive class (1_has_water)
- Images without target annotations -> negative class (0_no_water)

Output structure:
    output_dir/
    ├── train/
    │   ├── 0_no_water/
    │   └── 1_has_water/
    ├── val/
    │   ├── 0_no_water/
    │   └── 1_has_water/
    └── class_mapping.json

Usage:
    python tools/data_preparation/convert_coco_to_classification.py \
        --coco_json data/water_coco/annotations.json \
        --images_dir data/water_coco/images \
        --output_dir data/water_classification \
        --val_split 0.2 \
        --seed 42
"""

import os
import sys
import json
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from tqdm import tqdm


def load_coco_annotations(coco_json_path: str) -> Tuple[Dict, Dict, Set]:
    """
    Load COCO annotations and extract image info.

    Returns:
        images_info: Dict mapping image_id to image info
        annotations_by_image: Dict mapping image_id to list of annotations
        images_with_annotations: Set of image_ids that have annotations
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Build image info dict
    images_info = {}
    for img in coco_data.get('images', []):
        images_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img.get('width', 0),
            'height': img.get('height', 0),
        }

    # Build annotations by image
    annotations_by_image = defaultdict(list)
    images_with_annotations = set()

    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        annotations_by_image[image_id].append(ann)
        images_with_annotations.add(image_id)

    # Get category info
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

    print(f"Loaded COCO annotations:")
    print(f"  Total images: {len(images_info)}")
    print(f"  Images with annotations: {len(images_with_annotations)}")
    print(f"  Images without annotations: {len(images_info) - len(images_with_annotations)}")
    print(f"  Categories: {categories}")

    return images_info, annotations_by_image, images_with_annotations


def split_dataset(
    positive_images: List[str],
    negative_images: List[str],
    val_split: float = 0.2,
    seed: int = 42
) -> Dict[str, Dict[str, List[str]]]:
    """
    Split dataset into train and val with stratification.

    Returns:
        {
            'train': {'positive': [...], 'negative': [...]},
            'val': {'positive': [...], 'negative': [...]}
        }
    """
    random.seed(seed)

    # Shuffle
    positive_images = positive_images.copy()
    negative_images = negative_images.copy()
    random.shuffle(positive_images)
    random.shuffle(negative_images)

    # Split
    val_pos_count = max(1, int(len(positive_images) * val_split))
    val_neg_count = max(1, int(len(negative_images) * val_split))

    splits = {
        'train': {
            'positive': positive_images[val_pos_count:],
            'negative': negative_images[val_neg_count:],
        },
        'val': {
            'positive': positive_images[:val_pos_count],
            'negative': negative_images[:val_neg_count],
        }
    }

    print(f"\nDataset split (val_split={val_split}):")
    print(f"  Train: {len(splits['train']['positive'])} positive, {len(splits['train']['negative'])} negative")
    print(f"  Val: {len(splits['val']['positive'])} positive, {len(splits['val']['negative'])} negative")

    return splits


def copy_images(
    splits: Dict[str, Dict[str, List[str]]],
    images_dir: Path,
    output_dir: Path,
    class_names: Tuple[str, str] = ('0_no_water', '1_has_water'),
    use_symlink: bool = False
) -> Dict[str, int]:
    """
    Copy or symlink images to classification folder structure.

    Returns:
        stats: Dict with copy statistics
    """
    stats = {'copied': 0, 'missing': 0, 'errors': 0}

    for split_name, split_data in splits.items():
        # Create directories
        for class_name in class_names:
            class_dir = output_dir / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

        # Copy negative class (no annotations)
        neg_dir = output_dir / split_name / class_names[0]
        for img_name in tqdm(split_data['negative'], desc=f'{split_name}/negative'):
            src = images_dir / img_name
            dst = neg_dir / img_name

            if not src.exists():
                # Try without path components
                src = images_dir / Path(img_name).name

            if src.exists():
                try:
                    if use_symlink:
                        dst.symlink_to(src.resolve())
                    else:
                        shutil.copy2(src, dst)
                    stats['copied'] += 1
                except Exception as e:
                    print(f"Error copying {src}: {e}")
                    stats['errors'] += 1
            else:
                stats['missing'] += 1

        # Copy positive class (has annotations)
        pos_dir = output_dir / split_name / class_names[1]
        for img_name in tqdm(split_data['positive'], desc=f'{split_name}/positive'):
            src = images_dir / img_name
            dst = pos_dir / img_name

            if not src.exists():
                src = images_dir / Path(img_name).name

            if src.exists():
                try:
                    if use_symlink:
                        dst.symlink_to(src.resolve())
                    else:
                        shutil.copy2(src, dst)
                    stats['copied'] += 1
                except Exception as e:
                    print(f"Error copying {src}: {e}")
                    stats['errors'] += 1
            else:
                stats['missing'] += 1

    return stats


def save_class_mapping(
    output_dir: Path,
    class_names: Tuple[str, str] = ('no_water', 'has_water')
):
    """Save class mapping JSON file."""
    mapping = {
        'classes': {
            '0': class_names[0],
            '1': class_names[1],
        },
        'class_to_idx': {
            class_names[0]: 0,
            class_names[1]: 1,
        },
        'num_classes': 2,
        'positive_class': class_names[1],
        'negative_class': class_names[0],
    }

    mapping_file = output_dir / 'class_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\nSaved class mapping to: {mapping_file}")


def convert_coco_to_classification(
    coco_json: str,
    images_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42,
    positive_class_name: str = 'has_water',
    negative_class_name: str = 'no_water',
    use_symlink: bool = False,
    clean_output: bool = False,
):
    """
    Main conversion function.

    Args:
        coco_json: Path to COCO annotations JSON file
        images_dir: Path to images directory
        output_dir: Path to output directory
        val_split: Validation split ratio (0-1)
        seed: Random seed for reproducibility
        positive_class_name: Name for positive class (has annotations)
        negative_class_name: Name for negative class (no annotations)
        use_symlink: Use symlinks instead of copying files
        clean_output: Remove existing output directory before conversion
    """
    coco_json = Path(coco_json)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    # Validate inputs
    if not coco_json.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Clean output if requested
    if clean_output and output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    print(f"\nLoading COCO annotations from: {coco_json}")
    images_info, annotations_by_image, images_with_annotations = load_coco_annotations(str(coco_json))

    # Separate positive and negative images
    positive_images = []  # Has annotations (water detected)
    negative_images = []  # No annotations (no water)

    for image_id, info in images_info.items():
        file_name = info['file_name']
        if image_id in images_with_annotations:
            positive_images.append(file_name)
        else:
            negative_images.append(file_name)

    if len(positive_images) == 0:
        print("\nWarning: No images with annotations found!")
        print("This might indicate a problem with the COCO JSON format.")
        return

    if len(negative_images) == 0:
        print("\nWarning: No images without annotations found!")
        print("All images have annotations - this is unusual for binary classification.")

    # Split dataset
    splits = split_dataset(positive_images, negative_images, val_split, seed)

    # Define class folder names
    class_names = (f'0_{negative_class_name}', f'1_{positive_class_name}')

    # Copy images
    print(f"\nCopying images to: {output_dir}")
    stats = copy_images(splits, images_dir, output_dir, class_names, use_symlink)

    print(f"\nCopy statistics:")
    print(f"  Successfully copied: {stats['copied']}")
    print(f"  Missing files: {stats['missing']}")
    print(f"  Errors: {stats['errors']}")

    # Save class mapping
    save_class_mapping(output_dir, (negative_class_name, positive_class_name))

    # Save split info
    split_info = {
        'val_split': val_split,
        'seed': seed,
        'train': {
            'positive': len(splits['train']['positive']),
            'negative': len(splits['train']['negative']),
            'total': len(splits['train']['positive']) + len(splits['train']['negative']),
        },
        'val': {
            'positive': len(splits['val']['positive']),
            'negative': len(splits['val']['negative']),
            'total': len(splits['val']['positive']) + len(splits['val']['negative']),
        },
        'class_names': class_names,
    }

    split_file = output_dir / 'split_info.json'
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved split info to: {split_file}")
    print(f"\nConversion complete!")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── {class_names[0]}/  ({split_info['train']['negative']} images)")
    print(f"  │   └── {class_names[1]}/  ({split_info['train']['positive']} images)")
    print(f"  ├── val/")
    print(f"  │   ├── {class_names[0]}/  ({split_info['val']['negative']} images)")
    print(f"  │   └── {class_names[1]}/  ({split_info['val']['positive']} images)")
    print(f"  ├── class_mapping.json")
    print(f"  └── split_info.json")


def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO annotations to classification format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--coco_json', type=str, required=True,
                        help='Path to COCO annotations JSON file')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to images directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--positive_class', type=str, default='has_water',
                        help='Name for positive class (default: has_water)')
    parser.add_argument('--negative_class', type=str, default='no_water',
                        help='Name for negative class (default: no_water)')
    parser.add_argument('--symlink', action='store_true',
                        help='Use symlinks instead of copying files')
    parser.add_argument('--clean', action='store_true',
                        help='Remove existing output directory before conversion')

    args = parser.parse_args()

    convert_coco_to_classification(
        coco_json=args.coco_json,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        seed=args.seed,
        positive_class_name=args.positive_class,
        negative_class_name=args.negative_class,
        use_symlink=args.symlink,
        clean_output=args.clean,
    )


if __name__ == '__main__':
    main()
