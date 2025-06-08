#!/usr/bin/env python3
"""
创建分割数据的训练/验证分割文件

生成的文件将保存在 data/segmentation/ImageSets/ 目录下：
- train.txt: 训练集文件名列表
- val.txt: 验证集文件名列表
"""

import os
import random
from pathlib import Path
import argparse


def create_segmentation_splits(
    segmentation_dir,
    train_ratio=0.8,
    val_ratio=0.2,
    random_seed=42
):
    """
    创建分割数据的训练/验证分割
    
    Args:
        segmentation_dir: 分割数据根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_seed: 随机种子
    """
    
    segmentation_dir = Path(segmentation_dir)
    
    # 检查目录结构
    image_dir = segmentation_dir / "JPEGImages"
    mask_dir = segmentation_dir / "SegmentationClass_unified"
    imageset_dir = segmentation_dir / "ImageSets"
    
    if not image_dir.exists():
        print(f"错误: 图像目录不存在 {image_dir}")
        return
    
    if not mask_dir.exists():
        print(f"错误: mask目录不存在 {mask_dir}")
        return
    
    # 创建ImageSets目录
    imageset_dir.mkdir(exist_ok=True)
    
    # 获取所有有效的图像-mask对
    valid_samples = []
    
    # 遍历图像文件
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    for image_file in image_files:
        name = image_file.stem
        mask_file = mask_dir / f"{name}.png"
        
        if mask_file.exists():
            valid_samples.append(name)
    
    if not valid_samples:
        print("错误: 没有找到有效的图像-mask对")
        return
    
    print(f"找到 {len(valid_samples)} 个有效样本")
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 随机打乱
    random.shuffle(valid_samples)
    
    # 计算分割数量
    total_samples = len(valid_samples)
    train_size = int(total_samples * train_ratio)
    val_size = total_samples - train_size
    
    # 分割数据
    train_samples = valid_samples[:train_size]
    val_samples = valid_samples[train_size:]
    
    # 保存分割文件
    train_file = imageset_dir / "train.txt"
    val_file = imageset_dir / "val.txt"
    
    with open(train_file, 'w') as f:
        for name in train_samples:
            f.write(f"{name}\n")
    
    with open(val_file, 'w') as f:
        for name in val_samples:
            f.write(f"{name}\n")
    
    print(f"\n数据分割完成:")
    print(f"  训练集: {len(train_samples)} 个样本 ({train_ratio*100:.1f}%)")
    print(f"  验证集: {len(val_samples)} 个样本 ({val_ratio*100:.1f}%)")
    print(f"  分割文件保存在: {imageset_dir}")
    
    # 统计不同类型的样本
    new_count = len([name for name in valid_samples if name.startswith('new_')])
    roi_count = len([name for name in valid_samples if name.startswith('roi_image_')])
    other_count = len(valid_samples) - new_count - roi_count
    
    print(f"\n样本类型统计:")
    print(f"  new_* 格式: {new_count} 个")
    print(f"  roi_image_* 格式: {roi_count} 个")
    print(f"  其他格式: {other_count} 个")


def main():
    parser = argparse.ArgumentParser(description="创建分割数据的训练/验证分割")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/segmentation",
        help="分割数据目录路径"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="验证集比例"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 验证比例
    if abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
        print("错误: 训练集和验证集比例之和必须等于1")
        return
    
    create_segmentation_splits(
        segmentation_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main() 