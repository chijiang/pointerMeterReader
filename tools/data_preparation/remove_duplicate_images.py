#!/usr/bin/env python3
"""
重复图片清理脚本

该工具会遍历一个或多个目录，找出内容完全相同（字节级别）的图片文件，
并根据用户选择执行删除或仅预览重复项。默认不会直接删除，请加 --delete
参数确认后再执行，以避免误删。

用法示例：
    python tools/data_preparation/remove_duplicate_images.py data/images --delete

作者: AI 助手
日期: 2025-02-21
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".webp",
    ".gif",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Scan directories for duplicate images and optionally delete them."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more directories to scan",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan directories recursively",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=None,
        help=(
            "Whitelist of file extensions (e.g. --ext .jpg .png). "
            "Defaults to common image formats."
        ),
    )
    parser.add_argument(
        "--hash",
        default="sha256",
        choices=sorted(hashlib.algorithms_available),
        help="Hash algorithm used to compare files (default: sha256)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1 << 20,
        help="Chunk size (bytes) when hashing files (default: 1MB)",
    )
    parser.add_argument(
        "--keep",
        choices=("oldest", "newest"),
        default="oldest",
        help="Which duplicate to keep when deleting (default: oldest)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete duplicated files (keep one copy in each group)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force preview mode even if --delete is provided",
    )
    return parser.parse_args()


def iterate_image_files(
    roots: Sequence[str],
    recursive: bool,
    extensions: Iterable[str],
) -> List[Path]:
    """Collect image file paths under the specified roots."""
    files: List[Path] = []
    allowed = {ext.lower() for ext in extensions}

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"⚠️  Path does not exist: {root_path}")
            continue

        if root_path.is_file():
            if root_path.suffix.lower() in allowed:
                files.append(root_path)
            continue

        glob_pattern = "**/*" if recursive else "*"
        for path in root_path.glob(glob_pattern):
            if path.is_file() and path.suffix.lower() in allowed:
                files.append(path)

    return files


def compute_hash(file_path: Path, algorithm: str, chunk_size: int) -> str:
    """Compute the hash of a file."""
    hasher = hashlib.new(algorithm)
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicate_files(
    files: Sequence[Path],
    algorithm: str,
    chunk_size: int,
) -> Dict[str, List[Path]]:
    """
    Find duplicate files grouped by hash value.

    Files are grouped by file size first to avoid unnecessary hashing.
    """
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    candidates: Dict[int, List[Path]] = defaultdict(list)

    for file_path in files:
        try:
            file_size = file_path.stat().st_size
        except OSError as exc:
            print(f"⚠️  Skipping {file_path}: {exc}")
            continue
        candidates[file_size].append(file_path)

    for size, size_group in candidates.items():
        if len(size_group) < 2:
            continue
        for file_path in size_group:
            try:
                hash_value = compute_hash(file_path, algorithm, chunk_size)
            except OSError as exc:
                print(f"⚠️  Failed to hash {file_path}: {exc}")
                continue
            duplicates[hash_value].append(file_path)

    # Remove hash groups that only have one file
    return {h: paths for h, paths in duplicates.items() if len(paths) > 1}


def format_size(num_bytes: int) -> str:
    """Convert byte count to a human-friendly string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


def choose_file_to_keep(files: List[Path], strategy: str) -> Tuple[Path, List[Path]]:
    """Select which file to keep based on the desired strategy."""
    try:
        sorted_files = sorted(files, key=lambda p: p.stat().st_mtime)
    except OSError:
        sorted_files = files[:]

    if strategy == "newest":
        sorted_files.reverse()

    keep = sorted_files[0]
    to_remove = [path for path in files if path != keep]
    return keep, to_remove


def delete_duplicates(
    duplicates: Dict[str, List[Path]],
    strategy: str,
    delete_files: bool,
) -> Tuple[int, int]:
    """Delete duplicates according to the chosen strategy."""
    removed_count = 0
    freed_bytes = 0

    for hash_value, paths in duplicates.items():
        print(f"\n🔁 Hash: {hash_value}")
        keep_path, remove_paths = choose_file_to_keep(paths, strategy)
        print(f"  Keeping : {keep_path}")
        for dup_path in remove_paths:
            try:
                file_size = dup_path.stat().st_size
            except OSError:
                file_size = 0
            if delete_files:
                try:
                    dup_path.unlink()
                    print(f"  Deleted: {dup_path}")
                except OSError as exc:
                    print(f"  ⚠️  Failed to delete {dup_path}: {exc}")
                    continue
            else:
                print(f"  Duplicate: {dup_path}")
            removed_count += 1
            freed_bytes += file_size

    return removed_count, freed_bytes


def main() -> None:
    args = parse_args()
    extensions = args.ext if args.ext else DEFAULT_EXTENSIONS
    files = iterate_image_files(args.paths, args.recursive, extensions)

    if not files:
        print("⚠️  No matching image files found.")
        return

    print(f"🔍 Scanning {len(files)} files using {args.hash.upper()} ...")
    duplicates = find_duplicate_files(files, args.hash, args.chunk_size)

    if not duplicates:
        print("✅ No duplicate images detected.")
        return

    duplicate_files = sum(len(paths) - 1 for paths in duplicates.values())
    print(f"Found {len(duplicates)} duplicate groups covering {duplicate_files} files.")

    delete_files = args.delete and not args.dry_run
    if args.delete and args.dry_run:
        print("ℹ️  Dry-run enabled: duplicates will not be deleted.")
    elif not args.delete:
        print("ℹ️  Preview mode: use --delete to remove duplicates.")

    removed_count, freed_bytes = delete_duplicates(duplicates, args.keep, delete_files)
    print(
        f"\nSummary: {removed_count} duplicate files "
        f"{'would be removed' if not delete_files else 'removed'} "
        f"(~{format_size(freed_bytes)})."
    )


if __name__ == "__main__":
    main()
