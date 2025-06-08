#!/usr/bin/env python3
"""
统一分割数据格式脚本
将不同格式的分割标注统一为索引格式：背景=0，指针=1，刻度=2

输入格式：
1. new_* 文件：已经是索引格式（背景=0，指针=1，刻度=2）
2. roi_image_* 文件：RGB格式（黑色=背景，红色=指针，绿色=刻度）

输出：统一的索引格式
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def rgb_to_index_mask(rgb_mask):
    """
    将RGB格式的mask转换为索引格式
    
    Args:
        rgb_mask: RGB格式的mask (H, W, 3)
        
    Returns:
        index_mask: 索引格式的mask (H, W)
            0: 背景 (黑色)
            1: 指针 (红色)
            2: 刻度 (绿色)
    """
    h, w = rgb_mask.shape[:2]
    index_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 定义颜色阈值
    # 黑色背景：RGB值都很低
    # 红色指针：R高，G和B低
    # 绿色刻度：G高，R和B低
    
    # 红色区域 (指针) - 类别1
    red_mask = (rgb_mask[:, :, 2] > 100) & (rgb_mask[:, :, 1] < 100) & (rgb_mask[:, :, 0] < 100)
    index_mask[red_mask] = 1
    
    # 绿色区域 (刻度) - 类别2
    green_mask = (rgb_mask[:, :, 1] > 100) & (rgb_mask[:, :, 2] < 100) & (rgb_mask[:, :, 0] < 100)
    index_mask[green_mask] = 2
    
    # 其余区域默认为背景 (0)
    
    return index_mask


def is_index_format(mask):
    """
    判断mask是否已经是索引格式
    """
    unique_values = np.unique(mask)
    # 索引格式的值应该只包含0, 1, 2
    return all(val in [0, 1, 2] for val in unique_values) and len(unique_values) <= 3


def process_mask(input_path, output_path):
    """
    处理单个mask文件
    """
    # 读取mask
    mask = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if mask is None:
        print(f"Warning: 无法读取文件 {input_path}")
        return False
    
    # 如果是单通道图像，检查是否已经是索引格式
    if len(mask.shape) == 2:
        if is_index_format(mask):
            # 已经是索引格式，直接复制
            cv2.imwrite(output_path, mask)
            return True
        else:
            print(f"Warning: 单通道图像但不是有效的索引格式: {input_path}")
            return False
    
    # 如果是三通道图像，转换为索引格式
    elif len(mask.shape) == 3:
        index_mask = rgb_to_index_mask(mask)
        cv2.imwrite(output_path, index_mask)
        return True
    
    else:
        print(f"Warning: 不支持的图像格式: {input_path}")
        return False


def validate_unified_data(segmentation_dir):
    """
    验证统一后的数据
    """
    unified_dir = segmentation_dir / "SegmentationClass_unified"
    
    if not unified_dir.exists():
        print("统一后的数据目录不存在")
        return
    
    mask_files = list(unified_dir.glob("*.png"))
    total_files = len(mask_files)
    
    class_counts = {0: 0, 1: 0, 2: 0}  # 背景、指针、刻度
    
    print(f"\n验证统一后的数据...")
    print(f"总文件数: {total_files}")
    
    for mask_path in tqdm(mask_files[:10]):  # 只检查前10个文件作为示例
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_values = np.unique(mask)
            for val in unique_values:
                if val in class_counts:
                    class_counts[val] += 1
    
    print(f"检查的前10个文件中各类别分布:")
    print(f"  背景 (0): {class_counts[0]} 个文件包含")
    print(f"  指针 (1): {class_counts[1]} 个文件包含")
    print(f"  刻度 (2): {class_counts[2]} 个文件包含")


def main():
    parser = argparse.ArgumentParser(description="统一分割数据格式")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/segmentation",
        help="分割数据目录路径"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_unified",
        help="输出目录后缀"
    )
    
    args = parser.parse_args()
    
    # 设置路径
    segmentation_dir = Path(args.data_dir)
    input_mask_dir = segmentation_dir / "SegmentationClass"
    output_mask_dir = segmentation_dir / f"SegmentationClass{args.output_suffix}"
    
    if not input_mask_dir.exists():
        print(f"错误: 输入目录不存在 {input_mask_dir}")
        return
    
    # 创建输出目录
    output_mask_dir.mkdir(exist_ok=True)
    
    # 获取所有mask文件
    mask_files = list(input_mask_dir.glob("*.png"))
    
    if not mask_files:
        print(f"错误: 在 {input_mask_dir} 中没有找到PNG文件")
        return
    
    print(f"找到 {len(mask_files)} 个mask文件")
    print(f"开始统一格式...")
    
    # 统计处理结果
    new_format_count = 0  # new_* 格式文件数
    roi_format_count = 0  # roi_image_* 格式文件数
    success_count = 0
    failed_count = 0
    
    # 处理每个mask文件
    for mask_path in tqdm(mask_files, desc="处理mask文件"):
        filename = mask_path.name
        output_path = output_mask_dir / filename
        
        # 统计文件类型
        if filename.startswith("new_"):
            new_format_count += 1
        elif filename.startswith("roi_image_"):
            roi_format_count += 1
        
        # 处理文件
        if process_mask(str(mask_path), str(output_path)):
            success_count += 1
        else:
            failed_count += 1
    
    # 打印统计信息
    print(f"\n处理完成!")
    print(f"文件类型统计:")
    print(f"  new_* 格式: {new_format_count} 个")
    print(f"  roi_image_* 格式: {roi_format_count} 个")
    print(f"  其他格式: {len(mask_files) - new_format_count - roi_format_count} 个")
    print(f"\n处理结果:")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {failed_count} 个")
    print(f"\n统一后的数据保存在: {output_mask_dir}")
    
    # 验证统一后的数据
    validate_unified_data(segmentation_dir)


if __name__ == "__main__":
    main() 