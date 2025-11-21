"""
Convert COCO format annotations to VOC segmentation format.

This script converts the COCO format data from data/coco_data/ to the 
segmentation format used in data/segmentation/.

File name mapping:
- COCO format uses: <uuid>-<original_filename>
- Where spaces are replaced with underscores and brackets are removed
- This script reverses that mapping to restore original filenames
"""

import json
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import re
from tqdm import tqdm
import argparse


def restore_original_filename(coco_filename):
    """
    Restore original filename from COCO format.
    
    COCO format: <uuid>-<modified_filename>
    Where:
    - <uuid> is a hex string like 'c3d029b5'
    - spaces are replaced with underscores
    - brackets are removed
    
    This function:
    1. Removes the UUID prefix
    2. (Note: We keep underscores and no brackets as-is since we don't know the original)
    
    Args:
        coco_filename: Filename in COCO format
        
    Returns:
        Original filename (without UUID prefix)
    """
    # Remove UUID prefix (pattern: xxxxxxxx- where x is hex digit)
    # UUID can be 8 characters followed by dash
    match = re.match(r'^[0-9a-f]{8}-(.+)$', coco_filename)
    if match:
        return match.group(1)
    return coco_filename


def find_matching_image(modified_filename, images_dir):
    """
    Find the actual image file that matches the modified filename.
    
    Since we don't know the exact reverse mapping (where underscores become spaces/brackets),
    we need to search for the best matching file.
    
    Args:
        modified_filename: Filename after removing UUID prefix
        images_dir: Directory containing the original images
        
    Returns:
        Actual filename if found, else modified_filename
    """
    images_dir = Path(images_dir)
    
    # First try exact match
    if (images_dir / modified_filename).exists():
        return modified_filename
    
    # Get all image files in the directory
    all_images = list(images_dir.glob('*'))
    all_image_names = [img.name for img in all_images]
    
    # Remove extension for comparison
    base_name = os.path.splitext(modified_filename)[0]
    ext = os.path.splitext(modified_filename)[1]
    
    # Try to find a matching file by:
    # 1. Converting underscores back to spaces or brackets
    # Pattern: Look for files that match when we normalize both
    
    def normalize_for_comparison(name):
        """Normalize filename for fuzzy matching."""
        # Remove extension, convert to lowercase, remove special chars
        base = os.path.splitext(name)[0].lower()
        # Remove spaces, underscores, and brackets for comparison
        return re.sub(r'[\s_\(\)]', '', base)
    
    modified_norm = normalize_for_comparison(modified_filename)
    
    for img_name in all_image_names:
        if normalize_for_comparison(img_name) == modified_norm:
            return img_name
    
    # If still not found, return the modified filename
    return modified_filename


def polygon_to_mask(polygon, width, height):
    """
    Convert polygon coordinates to binary mask.
    
    Args:
        polygon: List of [x1, y1, x2, y2, ...] coordinates
        width: Image width
        height: Image height
        
    Returns:
        Binary mask as numpy array
    """
    mask = Image.new('L', (width, height), 0)
    
    # Convert flat list to list of tuples
    xy = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
    
    ImageDraw.Draw(mask).polygon(xy, outline=1, fill=1)
    
    return np.array(mask)


def convert_coco_to_segmentation(
    coco_json_path,
    coco_images_dir,
    output_dir,
    train_ratio=0.8
):
    """
    Convert COCO format data to VOC segmentation format.
    
    Args:
        coco_json_path: Path to result_coco.json
        coco_images_dir: Path to coco images directory
        output_dir: Output directory for segmentation data
        train_ratio: Ratio of training data (rest will be validation)
    """
    # Create output directories
    output_dir = Path(output_dir)
    jpeg_images_dir = output_dir / 'JPEGImages'
    seg_class_dir = output_dir / 'SegmentationClass'
    imagesets_dir = output_dir / 'ImageSets' / 'Segmentation'
    
    jpeg_images_dir.mkdir(parents=True, exist_ok=True)
    seg_class_dir.mkdir(parents=True, exist_ok=True)
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    print(f"Loading COCO annotations from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    print(f"Found {len(images)} images and {len(annotations)} annotations")
    print(f"Categories: {categories}")
    
    # Group annotations by image_id
    image_annotations = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Process each image
    processed_files = []
    skipped_files = []
    
    for img_info in tqdm(images, desc="Converting images"):
        image_id = img_info['id']
        coco_filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Restore original filename (remove UUID prefix)
        modified_filename = restore_original_filename(coco_filename)
        
        # Find the actual matching image file
        original_filename = find_matching_image(modified_filename, coco_images_dir)
        base_name = os.path.splitext(original_filename)[0]
        
        # Source image path
        src_image_path = Path(coco_images_dir) / original_filename
        
        # Check if source image exists
        if not src_image_path.exists():
            print(f"Warning: Source image not found: {src_image_path}")
            print(f"  COCO filename: {coco_filename}")
            print(f"  Modified filename: {modified_filename}")
            print(f"  Looking for: {original_filename}")
            skipped_files.append(original_filename)
            continue
        
        # Copy image to JPEGImages
        dst_image_path = jpeg_images_dir / original_filename
        shutil.copy2(src_image_path, dst_image_path)
        
        # Create segmentation mask
        seg_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get annotations for this image
        if image_id in image_annotations:
            for ann in image_annotations[image_id]:
                category_id = ann['category_id']
                segmentation = ann['segmentation']
                
                # Process each polygon in the segmentation
                for polygon in segmentation:
                    if len(polygon) < 6:  # Need at least 3 points
                        continue
                    
                    # Convert polygon to mask
                    poly_mask = polygon_to_mask(polygon, width, height)
                    
                    # Add to segmentation mask with category id as pixel value
                    seg_mask[poly_mask > 0] = category_id
        
        # Save segmentation mask
        seg_mask_path = seg_class_dir / f"{base_name}.png"
        Image.fromarray(seg_mask).save(seg_mask_path)
        
        processed_files.append(base_name)
    
    print(f"\nProcessed {len(processed_files)} images")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} images (source not found)")
    
    # Create train/val splits
    np.random.seed(42)
    np.random.shuffle(processed_files)
    
    split_idx = int(len(processed_files) * train_ratio)
    train_files = processed_files[:split_idx]
    val_files = processed_files[split_idx:]
    
    # Write ImageSets files
    with open(imagesets_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_files))
    
    with open(imagesets_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(val_files))
    
    with open(imagesets_dir / 'trainval.txt', 'w') as f:
        f.write('\n'.join(processed_files))
    
    # For compatibility, also create test.txt (same as val)
    with open(imagesets_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(val_files))
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val:   {len(val_files)} images")
    print(f"  Total: {len(processed_files)} images")
    
    print(f"\nConversion complete! Output saved to: {output_dir}")
    print(f"\nDirectory structure:")
    print(f"  {jpeg_images_dir}")
    print(f"  {seg_class_dir}")
    print(f"  {imagesets_dir}")
    
    if skipped_files:
        print(f"\nSkipped files:")
        for fname in skipped_files[:10]:  # Show first 10
            print(f"  - {fname}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO format annotations to VOC segmentation format'
    )
    parser.add_argument(
        '--coco-json',
        type=str,
        default='data/coco_data/result_coco.json',
        help='Path to COCO JSON file'
    )
    parser.add_argument(
        '--coco-images',
        type=str,
        default='data/coco_data/images',
        help='Path to COCO images directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/segmentation_new',
        help='Output directory for segmentation data'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of training data (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths if needed
    coco_json = Path(args.coco_json)
    coco_images = Path(args.coco_images)
    output_dir = Path(args.output_dir)
    
    if not coco_json.exists():
        print(f"Error: COCO JSON file not found: {coco_json}")
        return
    
    if not coco_images.exists():
        print(f"Error: COCO images directory not found: {coco_images}")
        return
    
    convert_coco_to_segmentation(
        coco_json_path=coco_json,
        coco_images_dir=coco_images,
        output_dir=output_dir,
        train_ratio=args.train_ratio
    )


if __name__ == '__main__':
    main()

