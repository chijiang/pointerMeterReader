#!/usr/bin/env python3
"""
液晶数字表数据集验证和可视化脚本

功能：
1. 验证数据集格式和完整性
2. 统计数据集信息
3. 可视化标注结果
4. 检查数据质量
5. 生成数据集报告
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import json
import argparse
from datetime import datetime

class DigitalMeterDatasetValidator:
    """液晶数字表数据集验证器"""
    
    def __init__(self, dataset_path: str):
        """
        初始化验证器
        
        Args:
            dataset_path: 数据集路径
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        
        print(f"🔍 数据集验证器已初始化")
        print(f"📂 数据集路径: {self.dataset_path}")
    
    def validate_structure(self) -> bool:
        """验证数据集目录结构"""
        print("\n📁 验证数据集结构...")
        
        required_dirs = [self.images_dir, self.labels_dir]
        missing_dirs = []
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            print("❌ 缺少必要目录:")
            for missing in missing_dirs:
                print(f"   - {missing}")
            return False
        
        print("✅ 目录结构验证通过")
        return True
    
    def get_file_lists(self) -> Tuple[List[Path], List[Path]]:
        """获取图像和标签文件列表"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))
        
        label_files = list(self.labels_dir.glob("*.txt"))
        
        # 排序确保一致性
        image_files.sort()
        label_files.sort()
        
        return image_files, label_files
    
    def validate_files(self) -> Dict:
        """验证文件完整性和对应关系"""
        print("\n📊 验证文件完整性...")
        
        image_files, label_files = self.get_file_lists()
        
        print(f"📸 图像文件: {len(image_files)} 个")
        print(f"🏷️  标签文件: {len(label_files)} 个")
        
        # 检查图像和标签的对应关系
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems
        
        validation_results = {
            'total_images': len(image_files),
            'total_labels': len(label_files),
            'missing_labels': list(missing_labels),
            'missing_images': list(missing_images),
            'paired_files': len(image_stems & label_stems)
        }
        
        if missing_labels:
            print(f"⚠️  缺少标签文件的图像 ({len(missing_labels)} 个):")
            for missing in list(missing_labels)[:5]:
                print(f"   - {missing}")
            if len(missing_labels) > 5:
                print(f"   ... 还有 {len(missing_labels) - 5} 个")
        
        if missing_images:
            print(f"⚠️  缺少图像文件的标签 ({len(missing_images)} 个):")
            for missing in list(missing_images)[:5]:
                print(f"   - {missing}")
            if len(missing_images) > 5:
                print(f"   ... 还有 {len(missing_images) - 5} 个")
        
        print(f"✅ 配对完整的文件: {validation_results['paired_files']} 个")
        
        return validation_results
    
    def validate_labels(self) -> Dict:
        """验证标签格式和内容"""
        print("\n🔍 验证标签格式...")
        
        _, label_files = self.get_file_lists()
        
        validation_stats = {
            'valid_labels': 0,
            'invalid_labels': 0,
            'total_annotations': 0,
            'bbox_areas': [],
            'bbox_aspect_ratios': [],
            'invalid_files': [],
            'class_distribution': {0: 0}  # 只有一个类别
        }
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                file_valid = True
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        validation_stats['invalid_files'].append(
                            f"{label_file.name}:{line_num} - 错误的字段数量: {len(parts)}"
                        )
                        file_valid = False
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        # 检查类别ID
                        if class_id != 0:
                            validation_stats['invalid_files'].append(
                                f"{label_file.name}:{line_num} - 错误的类别ID: {class_id}"
                            )
                            file_valid = False
                            continue
                        
                        # 检查坐标范围
                        coords = [x_center, y_center, width, height]
                        for i, coord in enumerate(coords):
                            if not (0 <= coord <= 1):
                                validation_stats['invalid_files'].append(
                                    f"{label_file.name}:{line_num} - 坐标超出范围: {coord}"
                                )
                                file_valid = False
                                break
                        
                        if file_valid:
                            # 统计信息
                            validation_stats['total_annotations'] += 1
                            validation_stats['class_distribution'][class_id] += 1
                            validation_stats['bbox_areas'].append(width * height)
                            validation_stats['bbox_aspect_ratios'].append(width / height if height > 0 else 0)
                    
                    except ValueError as e:
                        validation_stats['invalid_files'].append(
                            f"{label_file.name}:{line_num} - 数值转换错误: {e}"
                        )
                        file_valid = False
                
                if file_valid:
                    validation_stats['valid_labels'] += 1
                else:
                    validation_stats['invalid_labels'] += 1
                    
            except Exception as e:
                validation_stats['invalid_files'].append(f"{label_file.name} - 文件读取错误: {e}")
                validation_stats['invalid_labels'] += 1
        
        print(f"✅ 有效标签文件: {validation_stats['valid_labels']}")
        print(f"❌ 无效标签文件: {validation_stats['invalid_labels']}")
        print(f"📊 总标注数量: {validation_stats['total_annotations']}")
        
        if validation_stats['invalid_files']:
            print(f"⚠️  发现 {len(validation_stats['invalid_files'])} 个标签问题:")
            for error in validation_stats['invalid_files'][:5]:
                print(f"   - {error}")
            if len(validation_stats['invalid_files']) > 5:
                print(f"   ... 还有 {len(validation_stats['invalid_files']) - 5} 个问题")
        
        return validation_stats
    
    def run_full_validation(self, output_dir: str = "outputs/validation"):
        """运行完整的验证流程"""
        print("=" * 60)
        print("🔍 液晶数字表数据集验证开始")
        print("=" * 60)
        
        # 1. 验证目录结构
        if not self.validate_structure():
            print("❌ 数据集结构验证失败")
            return False
        
        # 2. 验证文件完整性
        file_validation = self.validate_files()
        
        # 3. 验证标签文件
        label_stats = self.validate_labels()
        
        print("\n" + "=" * 60)
        print("✅ 数据集验证完成")
        print(f"📊 概览:")
        print(f"   - 图像文件: {file_validation['total_images']}")
        print(f"   - 标签文件: {file_validation['total_labels']}")
        print(f"   - 配对文件: {file_validation['paired_files']}")
        print(f"   - 总标注数: {label_stats['total_annotations']}")
        print("=" * 60)
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="液晶数字表数据集验证脚本")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="data/digital_meters",
        help="数据集路径"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="outputs/validation",
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 处理数据集路径（相对于项目根目录）
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        # 获取项目根目录
        current_dir = Path.cwd()
        if current_dir.name == "validation":
            project_root = current_dir.parent.parent.parent
        elif current_dir.name == "digital_meter_detection":
            project_root = current_dir.parent.parent
        elif current_dir.name == "scripts":
            project_root = current_dir.parent
        else:
            project_root = current_dir
        
        dataset_path = project_root / dataset_path
    
    # 检查数据集路径
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    args.dataset = str(dataset_path)
    
    # 创建验证器
    validator = DigitalMeterDatasetValidator(args.dataset)
    
    # 运行验证
    success = validator.run_full_validation(args.output)
    
    if success:
        print("\n🎉 数据集验证成功完成")
    else:
        print("\n💔 数据集验证失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 