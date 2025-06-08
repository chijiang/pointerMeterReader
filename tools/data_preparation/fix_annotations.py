#!/usr/bin/env python3
"""
数据集标注修复脚本

修复COCO格式数据集中的常见问题，如边界框超出图像边界等。

作者: chijiang
日期: 2025-06-06
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

class AnnotationFixer:
    """标注修复器"""
    
    def __init__(self, data_root: str):
        """
        初始化修复器
        
        Args:
            data_root: 数据集根目录
        """
        self.data_root = Path(data_root)
        self.fixes_applied = []
        
    def fix_detection_dataset(self, backup: bool = True) -> Dict:
        """
        修复检测数据集
        
        Args:
            backup: 是否备份原始文件
            
        Returns:
            修复结果字典
        """
        print("🔧 Fixing detection dataset annotations...")
        
        results = {
            'train': self._fix_split('train', backup),
            'val': self._fix_split('val', backup),
            'summary': {}
        }
        
        # 汇总统计
        total_fixes = results['train']['fixes_applied'] + results['val']['fixes_applied']
        total_removed = results['train']['annotations_removed'] + results['val']['annotations_removed']
        
        results['summary'] = {
            'total_fixes_applied': total_fixes,
            'total_annotations_removed': total_removed,
            'backup_created': backup
        }
        
        # 输出修复结果
        self._print_fix_results(results)
        
        return results
    
    def _fix_split(self, split: str, backup: bool = True) -> Dict:
        """修复单个数据分割"""
        print(f"🛠️  Fixing {split} split...")
        
        # 文件路径
        ann_file = self.data_root / "annotations" / f"meter_coco_{split}.json"
        
        if not ann_file.exists():
            return {'error': f"Annotation file not found: {ann_file}"}
        
        # 备份原始文件
        if backup:
            backup_file = ann_file.parent / f"meter_coco_{split}_backup.json"
            if not backup_file.exists():
                shutil.copy2(ann_file, backup_file)
                print(f"📁 Backup created: {backup_file}")
        
        # 加载标注文件
        try:
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            return {'error': f"Failed to load annotation file: {str(e)}"}
        
        # 修复标注
        fix_result = self._fix_annotations(coco_data)
        
        # 保存修复后的文件
        try:
            with open(ann_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            print(f"✅ Fixed annotations saved to: {ann_file}")
        except Exception as e:
            return {'error': f"Failed to save fixed annotations: {str(e)}"}
        
        return fix_result
    
    def _fix_annotations(self, coco_data: Dict) -> Dict:
        """修复标注数据"""
        images = coco_data['images']
        annotations = coco_data['annotations']
        
        # 创建图像ID到图像信息的映射
        image_id_to_info = {img['id']: img for img in images}
        
        fixes_applied = 0
        annotations_removed = []
        fixed_annotations = []
        
        for ann in tqdm(annotations, desc="Fixing annotations"):
            if ann['image_id'] not in image_id_to_info:
                # 孤立标注，移除
                annotations_removed.append(ann['id'])
                continue
            
            img_info = image_id_to_info[ann['image_id']]
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # 检查并修复边界框
            fixed = False
            original_bbox = bbox.copy()
            
            # 修复负坐标
            if x < 0:
                w += x  # 调整宽度
                x = 0
                fixed = True
            
            if y < 0:
                h += y  # 调整高度
                y = 0
                fixed = True
            
            # 修复超出图像边界的情况
            if x + w > img_info['width']:
                w = img_info['width'] - x
                fixed = True
            
            if y + h > img_info['height']:
                h = img_info['height'] - y
                fixed = True
            
            # 检查修复后的边界框是否仍然有效
            if w <= 0 or h <= 0:
                # 边界框无效，移除该标注
                annotations_removed.append(ann['id'])
                self.fixes_applied.append({
                    'type': 'removed',
                    'annotation_id': ann['id'],
                    'image_id': ann['image_id'],
                    'reason': 'Invalid bbox after fixing',
                    'original_bbox': original_bbox
                })
                continue
            
            # 更新边界框
            if fixed:
                ann['bbox'] = [x, y, w, h]
                fixes_applied += 1
                self.fixes_applied.append({
                    'type': 'fixed',
                    'annotation_id': ann['id'],
                    'image_id': ann['image_id'],
                    'original_bbox': original_bbox,
                    'fixed_bbox': [x, y, w, h]
                })
            
            fixed_annotations.append(ann)
        
        # 更新标注列表
        coco_data['annotations'] = fixed_annotations
        
        return {
            'fixes_applied': fixes_applied,
            'annotations_removed': len(annotations_removed),
            'removed_annotation_ids': annotations_removed,
            'total_annotations_before': len(annotations),
            'total_annotations_after': len(fixed_annotations)
        }
    
    def _print_fix_results(self, results: Dict):
        """打印修复结果"""
        print("\n" + "="*60)
        print("🔧 ANNOTATION FIX RESULTS")
        print("="*60)
        
        # 汇总信息
        summary = results['summary']
        print(f"📊 Summary:")
        print(f"  Total fixes applied: {summary['total_fixes_applied']}")
        print(f"  Total annotations removed: {summary['total_annotations_removed']}")
        print(f"  Backup created: {summary['backup_created']}")
        
        # 分割详情
        for split in ['train', 'val']:
            if split in results and 'error' not in results[split]:
                split_data = results[split]
                print(f"\n🛠️  {split.capitalize()} Split:")
                print(f"  Annotations before: {split_data['total_annotations_before']}")
                print(f"  Annotations after: {split_data['total_annotations_after']}")
                print(f"  Fixes applied: {split_data['fixes_applied']}")
                print(f"  Annotations removed: {split_data['annotations_removed']}")
        
        # 详细修复日志
        if self.fixes_applied:
            print(f"\n📝 Detailed Fix Log:")
            for fix in self.fixes_applied[:10]:  # 只显示前10个
                if fix['type'] == 'fixed':
                    print(f"  Fixed annotation {fix['annotation_id']}: {fix['original_bbox']} → {fix['fixed_bbox']}")
                else:
                    print(f"  Removed annotation {fix['annotation_id']}: {fix['reason']}")
            
            if len(self.fixes_applied) > 10:
                print(f"  ... and {len(self.fixes_applied) - 10} more fixes")
        
        print("="*60)
    
    def validate_after_fix(self) -> bool:
        """修复后验证数据集"""
        print("🔍 Validating dataset after fixes...")
        
        # 这里可以调用之前的验证脚本
        # 为了简化，我们只做基本检查
        
        try:
            from check_data import DatasetValidator
            validator = DatasetValidator(str(self.data_root))
            results = validator.validate_detection_dataset()
            
            issues_found = results['summary']['issues_found']
            if issues_found == 0:
                print("✅ Validation passed! No issues found after fixing.")
                return True
            else:
                print(f"⚠️  Still found {issues_found} issues after fixing.")
                return False
                
        except ImportError:
            print("⚠️  Cannot import validator. Please run check_data.py manually to verify fixes.")
            return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据集标注修复脚本')
    parser.add_argument('--data-root', type=str, default='data/detection',
                       help='数据集根目录')
    parser.add_argument('--no-backup', action='store_true',
                       help='不创建备份文件')
    parser.add_argument('--validate', action='store_true',
                       help='修复后验证数据集')
    
    args = parser.parse_args()
    
    # 初始化修复器
    fixer = AnnotationFixer(args.data_root)
    
    # 运行修复
    results = fixer.fix_detection_dataset(backup=not args.no_backup)
    
    # 验证修复结果
    if args.validate:
        validation_passed = fixer.validate_after_fix()
        if not validation_passed:
            print("⚠️  Consider running the fixer again or manually checking the remaining issues.")
    
    # 返回适当的退出代码
    if 'error' in str(results):
        print("\n❌ Fix operation completed with errors.")
        return 1
    else:
        print(f"\n✅ Fix operation completed successfully!")
        print(f"📊 Applied {results['summary']['total_fixes_applied']} fixes, removed {results['summary']['total_annotations_removed']} invalid annotations.")
        return 0


if __name__ == "__main__":
    exit(main()) 