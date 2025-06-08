#!/usr/bin/env python3
"""
分割数据检查脚本

检查分割数据的完整性、格式正确性，并可视化样本
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm


def check_mask_format(mask_path):
    """检查mask格式"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, "无法读取文件"
    
    unique_values = np.unique(mask)
    
    # 检查是否只包含0, 1, 2
    valid_values = set([0, 1, 2])
    if not set(unique_values).issubset(valid_values):
        return False, f"包含无效像素值: {unique_values}"
    
    return True, f"像素值: {unique_values}"


def visualize_sample(image_path, mask_path, output_path, class_colors):
    """可视化单个样本"""
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 读取mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    
    # 创建彩色mask
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        colored_mask[mask == class_id] = color
    
    # 创建叠加图像
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask (Grayscale)')
    axes[1].axis('off')
    
    # 彩色mask
    axes[2].imshow(colored_mask)
    axes[2].set_title('Colored Mask')
    axes[2].axis('off')
    
    # 叠加图像
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def analyze_class_distribution(mask_dir):
    """分析类别分布"""
    mask_files = list(Path(mask_dir).glob("*.png"))
    
    total_pixels = {0: 0, 1: 0, 2: 0}  # 背景、指针、刻度
    file_counts = {0: 0, 1: 0, 2: 0}
    
    print("分析类别分布...")
    
    for mask_file in tqdm(mask_files):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        unique_values, counts = np.unique(mask, return_counts=True)
        
        for val, count in zip(unique_values, counts):
            if val in total_pixels:
                total_pixels[val] += count
                file_counts[val] += 1
    
    total_all_pixels = sum(total_pixels.values())
    
    print(f"\n类别分布分析:")
    print(f"  总像素数: {total_all_pixels:,}")
    
    class_names = ["背景", "指针", "刻度"]
    for class_id in [0, 1, 2]:
        pixels = total_pixels[class_id]
        files = file_counts[class_id]
        percentage = (pixels / total_all_pixels) * 100 if total_all_pixels > 0 else 0
        
        print(f"  {class_names[class_id]} (类别{class_id}):")
        print(f"    像素数: {pixels:,} ({percentage:.2f}%)")
        print(f"    出现在: {files} 个文件中")


def main():
    parser = argparse.ArgumentParser(description="检查分割数据")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/segmentation",
        help="分割数据目录路径"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="SegmentationClass_unified",
        help="mask目录名称"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="是否生成可视化样本"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="可视化样本数量"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/segmentation_data_check",
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 设置路径
    data_dir = Path(args.data_dir)
    image_dir = data_dir / "JPEGImages"
    mask_dir = data_dir / args.mask_dir
    output_dir = Path(args.output_dir)
    
    # 检查目录存在性
    if not data_dir.exists():
        print(f"错误: 数据目录不存在 {data_dir}")
        return
    
    if not image_dir.exists():
        print(f"错误: 图像目录不存在 {image_dir}")
        return
    
    if not mask_dir.exists():
        print(f"错误: mask目录不存在 {mask_dir}")
        return
    
    # 创建输出目录
    if args.visualize:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"检查分割数据: {data_dir}")
    print(f"图像目录: {image_dir}")
    print(f"Mask目录: {mask_dir}")
    
    # 获取所有图像文件
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"\n找到 {len(image_files)} 个图像文件")
    
    # 检查图像-mask对应关系
    valid_pairs = []
    missing_masks = []
    invalid_masks = []
    
    print("检查图像-mask对应关系...")
    for image_file in tqdm(image_files):
        name = image_file.stem
        mask_file = mask_dir / f"{name}.png"
        
        if not mask_file.exists():
            missing_masks.append(name)
            continue
        
        # 检查mask格式
        is_valid, info = check_mask_format(mask_file)
        if not is_valid:
            invalid_masks.append((name, info))
            continue
        
        valid_pairs.append((image_file, mask_file))
    
    # 打印统计信息
    print(f"\n数据检查结果:")
    print(f"  有效对: {len(valid_pairs)}")
    print(f"  缺失mask: {len(missing_masks)}")
    print(f"  无效mask: {len(invalid_masks)}")
    
    if missing_masks:
        print(f"\n缺失mask的图像 (前10个):")
        for name in missing_masks[:10]:
            print(f"  - {name}")
        if len(missing_masks) > 10:
            print(f"  ... 还有 {len(missing_masks) - 10} 个")
    
    if invalid_masks:
        print(f"\n无效mask:")
        for name, info in invalid_masks:
            print(f"  - {name}: {info}")
    
    if len(valid_pairs) == 0:
        print("❌ 没有找到有效的图像-mask对")
        return
    
    # 分析类别分布
    analyze_class_distribution(mask_dir)
    
    # 可视化样本
    if args.visualize and len(valid_pairs) > 0:
        print(f"\n生成可视化样本...")
        
        # 类别颜色
        class_colors = {
            0: [0, 0, 0],        # 背景 - 黑色
            1: [255, 0, 0],      # 指针 - 红色
            2: [0, 255, 0]       # 刻度 - 绿色
        }
        
        # 随机选择样本
        import random
        random.seed(42)
        selected_pairs = random.sample(valid_pairs, min(args.num_samples, len(valid_pairs)))
        
        for i, (image_file, mask_file) in enumerate(tqdm(selected_pairs, desc="生成可视化")):
            output_file = output_dir / f"sample_{i+1}_{image_file.stem}.png"
            
            success = visualize_sample(
                image_file, mask_file, output_file, class_colors
            )
            
            if not success:
                print(f"Warning: 无法可视化样本 {image_file.stem}")
        
        print(f"✅ 可视化样本保存在: {output_dir}")
    
    print(f"\n✅ 数据检查完成!")
    print(f"建议:")
    if len(valid_pairs) > 0:
        print(f"  - 可以开始训练，有 {len(valid_pairs)} 个有效样本")
    if missing_masks:
        print(f"  - 有 {len(missing_masks)} 个图像缺失对应的mask")
    if invalid_masks:
        print(f"  - 有 {len(invalid_masks)} 个mask格式无效，需要修复")


if __name__ == "__main__":
    main() 