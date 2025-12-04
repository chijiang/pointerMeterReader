#!/usr/bin/env python3
"""
数据集验证脚本

检查COCO格式数据集的完整性，包括图像文件、标注文件、数据格式等。

作者: chijiang
日期: 2025-06-06
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter, defaultdict

class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, data_root: str):
        """
        初始化验证器
        
        Args:
            data_root: 数据集根目录
        """
        self.data_root = Path(data_root)
        self.issues = []
        
    def validate_detection_dataset(self) -> Dict:
        """
        验证检测数据集
        
        Returns:
            验证结果字典
        """
        print("🔍 Validating detection dataset...")
        
        results = {
            'train': self._validate_split('train'),
            'val': self._validate_split('val'),
            'summary': {}
        }
        
        # 汇总统计
        total_images = results['train']['num_images'] + results['val']['num_images']
        total_annotations = results['train']['num_annotations'] + results['val']['num_annotations']
        
        results['summary'] = {
            'total_images': total_images,
            'total_annotations': total_annotations,
            'avg_annotations_per_image': total_annotations / total_images if total_images > 0 else 0,
            'issues_found': len(self.issues)
        }
        
        # 输出验证结果
        self._print_validation_results(results)
        
        return results
    
    def _validate_split(self, split: str) -> Dict:
        """验证单个数据分割"""
        print(f"📊 Validating {split} split...")
        
        # 文件路径
        if split == 'train':
            image_dir = self.data_root / "train2017"
            ann_file = self.data_root / "annotations" / "meter_coco_train.json"
        else:
            image_dir = self.data_root / "val2017"
            ann_file = self.data_root / "annotations" / "meter_coco_val.json"
        
        # 检查目录和文件存在性
        if not image_dir.exists():
            self.issues.append(f"Image directory not found: {image_dir}")
            return {'error': f"Image directory not found: {image_dir}"}
        
        if not ann_file.exists():
            self.issues.append(f"Annotation file not found: {ann_file}")
            return {'error': f"Annotation file not found: {ann_file}"}
        
        # 加载标注文件
        try:
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            self.issues.append(f"Failed to load annotation file {ann_file}: {str(e)}")
            return {'error': f"Failed to load annotation file: {str(e)}"}
        
        # 验证COCO格式
        validation_result = self._validate_coco_format(coco_data, image_dir, split)
        
        # 统计信息
        validation_result.update(self._compute_statistics(coco_data, image_dir))
        
        return validation_result
    
    def _validate_coco_format(self, coco_data: Dict, image_dir: Path, split: str) -> Dict:
        """验证COCO格式正确性"""
        issues = []
        
        # 检查必要字段
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                issues.append(f"Missing required field '{field}' in {split} annotations")
        
        if issues:
            self.issues.extend(issues)
            return {'format_valid': False, 'issues': issues}
        
        # 验证图像信息
        image_issues = self._validate_images(coco_data['images'], image_dir)
        
        # 验证标注信息
        annotation_issues = self._validate_annotations(coco_data['annotations'], coco_data['images'])
        
        # 验证类别信息
        category_issues = self._validate_categories(coco_data['categories'])
        
        all_issues = image_issues + annotation_issues + category_issues
        self.issues.extend(all_issues)
        
        return {
            'format_valid': len(all_issues) == 0,
            'issues': all_issues
        }
    
    def _validate_images(self, images: List[Dict], image_dir: Path) -> List[str]:
        """验证图像信息"""
        issues = []
        
        # 检查图像文件存在性
        missing_files = []
        corrupted_files = []
        size_mismatches = []
        
        for img_info in tqdm(images, desc="Validating images"):
            file_path = image_dir / img_info['file_name']
            
            # 检查文件存在
            if not file_path.exists():
                missing_files.append(img_info['file_name'])
                continue
            
            # 检查图像是否可读
            try:
                image = cv2.imread(str(file_path))
                if image is None:
                    corrupted_files.append(img_info['file_name'])
                    continue
                
                # 检查尺寸是否匹配
                h, w = image.shape[:2]
                if w != img_info['width'] or h != img_info['height']:
                    size_mismatches.append({
                        'file': img_info['file_name'],
                        'expected': (img_info['width'], img_info['height']),
                        'actual': (w, h)
                    })
            
            except Exception as e:
                corrupted_files.append(f"{img_info['file_name']}: {str(e)}")
        
        # 记录问题
        if missing_files:
            issues.append(f"Missing image files: {len(missing_files)} files")
            if len(missing_files) <= 10:
                issues.extend([f"  - {f}" for f in missing_files])
            else:
                issues.extend([f"  - {f}" for f in missing_files[:10]])
                issues.append(f"  - ... and {len(missing_files) - 10} more")
        
        if corrupted_files:
            issues.append(f"Corrupted image files: {len(corrupted_files)} files")
            if len(corrupted_files) <= 5:
                issues.extend([f"  - {f}" for f in corrupted_files])
        
        if size_mismatches:
            issues.append(f"Image size mismatches: {len(size_mismatches)} files")
            for mismatch in size_mismatches[:5]:
                issues.append(f"  - {mismatch['file']}: expected {mismatch['expected']}, got {mismatch['actual']}")
        
        return issues
    
    def _validate_annotations(self, annotations: List[Dict], images: List[Dict]) -> List[str]:
        """验证标注信息"""
        issues = []
        
        # 创建图像ID到图像信息的映射
        image_id_to_info = {img['id']: img for img in images}
        
        invalid_boxes = []
        orphan_annotations = []
        
        for ann in annotations:
            # 检查标注是否有对应的图像
            if ann['image_id'] not in image_id_to_info:
                orphan_annotations.append(ann['id'])
                continue
            
            # 检查边界框格式
            bbox = ann['bbox']
            if len(bbox) != 4:
                invalid_boxes.append(f"Annotation {ann['id']}: bbox should have 4 values, got {len(bbox)}")
                continue
            
            x, y, w, h = bbox
            
            # 检查边界框值的有效性
            if w <= 0 or h <= 0:
                invalid_boxes.append(f"Annotation {ann['id']}: invalid bbox size (w={w}, h={h})")
                continue
            
            if x < 0 or y < 0:
                invalid_boxes.append(f"Annotation {ann['id']}: negative bbox coordinates (x={x}, y={y})")
                continue
            
            # 检查边界框是否超出图像范围
            img_info = image_id_to_info[ann['image_id']]
            if x + w > img_info['width'] or y + h > img_info['height']:
                invalid_boxes.append(f"Annotation {ann['id']}: bbox exceeds image boundaries")
        
        if invalid_boxes:
            issues.append(f"Invalid bounding boxes: {len(invalid_boxes)}")
            issues.extend(invalid_boxes[:10])  # 只显示前10个
        
        if orphan_annotations:
            issues.append(f"Orphan annotations (no corresponding image): {len(orphan_annotations)}")
        
        return issues
    
    def _validate_categories(self, categories: List[Dict]) -> List[str]:
        """验证类别信息"""
        issues = []
        
        if not categories:
            issues.append("No categories found")
            return issues
        
        # 检查类别字段
        for cat in categories:
            if 'id' not in cat or 'name' not in cat:
                issues.append(f"Category missing required fields: {cat}")
        
        # 检查类别名称
        expected_categories = {'meter'}
        actual_categories = {cat['name'] for cat in categories}
        
        if actual_categories != expected_categories:
            issues.append(f"Unexpected categories. Expected: {expected_categories}, Got: {actual_categories}")
        
        return issues
    
    def _compute_statistics(self, coco_data: Dict, image_dir: Path) -> Dict:
        """计算数据集统计信息"""
        images = coco_data['images']
        annotations = coco_data['annotations']
        
        # 基本统计
        num_images = len(images)
        num_annotations = len(annotations)
        
        # 图像尺寸分布
        widths = [img['width'] for img in images]
        heights = [img['height'] for img in images]
        
        # 每张图像的标注数量
        annotations_per_image = Counter()
        bbox_areas = []
        bbox_aspect_ratios = []
        
        for ann in annotations:
            annotations_per_image[ann['image_id']] += 1
            
            bbox = ann['bbox']
            area = bbox[2] * bbox[3]  # width * height
            aspect_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 0
            
            bbox_areas.append(area)
            bbox_aspect_ratios.append(aspect_ratio)
        
        # 统计结果
        stats = {
            'num_images': num_images,
            'num_annotations': num_annotations,
            'avg_annotations_per_image': num_annotations / num_images if num_images > 0 else 0,
            'image_sizes': {
                'width': {'min': min(widths), 'max': max(widths), 'mean': np.mean(widths)},
                'height': {'min': min(heights), 'max': max(heights), 'mean': np.mean(heights)}
            },
            'bbox_stats': {
                'areas': {
                    'min': min(bbox_areas) if bbox_areas else 0,
                    'max': max(bbox_areas) if bbox_areas else 0,
                    'mean': np.mean(bbox_areas) if bbox_areas else 0
                },
                'aspect_ratios': {
                    'min': min(bbox_aspect_ratios) if bbox_aspect_ratios else 0,
                    'max': max(bbox_aspect_ratios) if bbox_aspect_ratios else 0,
                    'mean': np.mean(bbox_aspect_ratios) if bbox_aspect_ratios else 0
                }
            },
            'annotations_per_image_dist': dict(Counter(annotations_per_image.values()))
        }
        
        return stats
    
    def _print_validation_results(self, results: Dict):
        """打印验证结果"""
        print("\n" + "="*60)
        print("📋 DATASET VALIDATION RESULTS")
        print("="*60)
        
        # 汇总信息
        summary = results['summary']
        print(f"📊 Summary:")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Total annotations: {summary['total_annotations']}")
        print(f"  Avg annotations per image: {summary['avg_annotations_per_image']:.2f}")
        print(f"  Issues found: {summary['issues_found']}")
        
        # 分割统计
        for split in ['train', 'val']:
            if split in results and 'error' not in results[split]:
                split_data = results[split]
                print(f"\n📈 {split.capitalize()} Split:")
                print(f"  Images: {split_data['num_images']}")
                print(f"  Annotations: {split_data['num_annotations']}")
                print(f"  Avg annotations per image: {split_data['avg_annotations_per_image']:.2f}")
                
                # 图像尺寸
                img_sizes = split_data['image_sizes']
                print(f"  Image sizes:")
                print(f"    Width: {img_sizes['width']['min']}-{img_sizes['width']['max']} (avg: {img_sizes['width']['mean']:.1f})")
                print(f"    Height: {img_sizes['height']['min']}-{img_sizes['height']['max']} (avg: {img_sizes['height']['mean']:.1f})")
                
                # 边界框统计
                bbox_stats = split_data['bbox_stats']
                print(f"  Bbox areas: {bbox_stats['areas']['min']:.0f}-{bbox_stats['areas']['max']:.0f} (avg: {bbox_stats['areas']['mean']:.0f})")
                print(f"  Bbox aspect ratios: {bbox_stats['aspect_ratios']['min']:.2f}-{bbox_stats['aspect_ratios']['max']:.2f} (avg: {bbox_stats['aspect_ratios']['mean']:.2f})")
        
        # 问题报告
        if self.issues:
            print(f"\n❌ Issues Found ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ No issues found! Dataset is valid.")
        
        print("="*60)
    
    def visualize_statistics(self, results: Dict, output_dir: str = None):
        """可视化数据集统计信息"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "outputs" / "data_analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Statistics', fontsize=16)
        
        # 收集所有数据
        all_bbox_areas = []
        all_aspect_ratios = []
        split_labels = []
        
        for split in ['train', 'val']:
            if split in results and 'error' not in results[split]:
                # 这里需要重新加载数据来获取详细统计
                pass
        
        # 暂时显示基本信息
        axes[0, 0].text(0.5, 0.5, f"Total Images: {results['summary']['total_images']}\n"
                                   f"Total Annotations: {results['summary']['total_annotations']}\n"
                                   f"Avg Ann/Image: {results['summary']['avg_annotations_per_image']:.2f}",
                        ha='center', va='center', fontsize=14, transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Dataset Overview')
        axes[0, 0].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Statistics visualization saved to: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据集验证脚本')
    parser.add_argument('--data-root', type=str, default='data/detection',
                       help='数据集根目录')
    parser.add_argument('--visualize', action='store_true',
                       help='生成统计可视化')
    parser.add_argument('--output-dir', type=str,
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 初始化验证器
    validator = DatasetValidator(args.data_root)
    
    # 运行验证
    results = validator.validate_detection_dataset()
    
    # 生成可视化
    if args.visualize:
        validator.visualize_statistics(results, args.output_dir)
    
    # 返回适当的退出代码
    if results['summary']['issues_found'] > 0:
        print("\n⚠️  Dataset validation completed with issues.")
        return 1
    else:
        print("\n✅ Dataset validation completed successfully.")
        return 0


if __name__ == "__main__":
    exit(main()) 