"""
Merge multiple COCO result_coco.json files from HumanSignal/Label Studio exports.

Usage:
    python tools/data_preparation/merge_coco_exports.py \
        --input-glob "data/export_*/result_coco.json" \
        --output data/coco_data/result_coco_merged.json

Notes:
    - Category ids are normalized to: 1=pointer, 2=scale.
    - Image and annotation ids are re-indexed to keep them unique.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_CATEGORY_ORDER = ["pointer", "scale"]


def merge_exports(files: List[Path]) -> Dict:
    """Merge multiple COCO files into one with normalized category ids."""
    categories = [
        {"id": 1, "name": "pointer"},
        {"id": 2, "name": "scale"},
    ]
    name_to_id = {c["name"]: c["id"] for c in categories}

    images = []
    annotations = []
    next_image_id = 1
    next_ann_id = 1

    merged_info = {"description": "Merged COCO exports"}

    for fpath in files:
        data = json.load(fpath.open("r", encoding="utf-8"))

        # Use the first info block if present
        if "info" in data and merged_info == {"description": "Merged COCO exports"}:
            merged_info = data["info"]

        # Map old image ids to new ones
        image_id_map = {}
        for img in data.get("images", []):
            new_img = dict(img)
            new_img["id"] = next_image_id
            images.append(new_img)
            image_id_map[img["id"]] = next_image_id
            next_image_id += 1

        # Remap annotations
        for ann in data.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_id_map[ann["image_id"]]

            cat_name = None
            for cat in data.get("categories", []):
                if cat["id"] == ann["category_id"]:
                    cat_name = cat["name"]
                    break
            if cat_name is None or cat_name not in name_to_id:
                raise ValueError(f"Unknown category id {ann['category_id']} in {fpath}")
            new_ann["category_id"] = name_to_id[cat_name]

            annotations.append(new_ann)
            next_ann_id += 1

    merged = {
        "info": merged_info,
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="Merge COCO result_coco.json files")
    parser.add_argument(
        "--input-glob",
        default="data/export_*/result_coco.json",
        help="Glob pattern to find input COCO files",
    )
    parser.add_argument(
        "--output",
        default="data/coco_data/result_coco_merged.json",
        help="Path to save merged COCO file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(Path().glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {args.input_glob}")

    merged = merge_exports(files)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(files)} files into: {output_path}")
    print(f"Images: {len(merged['images'])}, Annotations: {len(merged['annotations'])}")


if __name__ == "__main__":
    main()
