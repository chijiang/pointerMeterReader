#!/usr/bin/env python3
"""
指针仪表批量检测和裁剪脚本
Pointer Meter Detection and Cropping Script

功能：
1. 加载训练好的YOLO检测模型
2. 批量处理指定目录下的图像
3. 检测图像中的指针仪表
4. 裁剪仪表区域并保存到目标目录

使用方法：
python scripts/meter_detection.py \
    --data-dir <输入图像目录> \
    --target-dir <输出目录> \
    --model <模型路径，默认: models/detection/yolo11_meter.pt> \
    --conf 0.5 \
    --padding 20

作者: chijiang
日期: 2025-11-21
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ 错误: 未安装 ultralytics 库")
    print("请运行: pip install ultralytics")
    sys.exit(1)

from scripts.image_utils import unwarp_meter_using_ellipse


class PointerMeterDetector:
    """指针仪表检测器"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, padding: int = 20):
        """
        初始化检测器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
            padding: 裁剪时的边界扩展像素
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.padding = padding
        
        # 检查模型文件是否存在
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载模型
        print(f"🔧 加载模型: {self.model_path}")
        try:
            self.model = YOLO(str(self.model_path))
            print("✅ 模型加载成功")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'successful_detections': 0,
            'total_meters_detected': 0,
            'failed_images': []
        }
    
    def detect_meters(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的仪表
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个元素包含 bbox 和 confidence
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf)
                    })
        
        return detections
    
    def crop_meter(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        裁剪仪表区域
        
        Args:
            image: 原始图像
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            裁剪后的图像
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # 添加padding
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w, x2 + self.padding)
        y2 = min(h, y2 + self.padding)
        
        return image[y1:y2, x1:x2]
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果
            
        Returns:
            标注后的图像
        """
        vis_image = image.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签
            label = f"Meter {i+1}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # 绘制标签背景
            cv2.rectangle(vis_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         (0, 255, 0), -1)
            
            # 绘制标签文字
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return vis_image
    
    def process_single_image(self, image_path: Path, output_dir: Path, 
                            save_visualization: bool = True,
                            apply_unwarp: bool = False) -> Dict:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
            save_visualization: 是否保存可视化结果
            apply_unwarp: 是否对裁剪结果做透视/形变矫正
            
        Returns:
            处理结果字典
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                'image_path': str(image_path),
                'success': False,
                'error': 'Failed to read image'
            }
        
        # 检测仪表
        detections = self.detect_meters(image)
        
        result = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'success': len(detections) > 0,
            'num_detections': len(detections),
            'detections': detections,
            'cropped_files': [],
            'unwarped_files': []
        }
        
        if len(detections) == 0:
            result['error'] = 'No meters detected'
            return result
        
        # 创建输出子目录
        cropped_dir = output_dir / "cropped"
        viz_dir = output_dir / "visualizations"
        unwarp_dir = output_dir / "unwarped"
        cropped_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        if apply_unwarp:
            unwarp_dir.mkdir(parents=True, exist_ok=True)
        
        # 裁剪并保存每个检测到的仪表
        base_name = image_path.stem
        for i, detection in enumerate(detections):
            try:
                # 裁剪仪表
                cropped_meter = self.crop_meter(image, detection['bbox'])
                
                # 生成输出文件名
                if len(detections) == 1:
                    # 如果只有一个仪表，不添加序号
                    output_filename = f"{base_name}.jpg"
                else:
                    # 多个仪表时添加序号
                    output_filename = f"{base_name}_meter_{i+1:02d}.jpg"
                
                output_path = cropped_dir / output_filename
                
                # 保存裁剪图像
                cv2.imwrite(str(output_path), cropped_meter)
                
                result['cropped_files'].append({
                    'filename': output_filename,
                    'path': str(output_path),
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'size': cropped_meter.shape[:2]  # (height, width)
                })

                if apply_unwarp:
                    # 对裁剪结果做透视矫正，输出正方形展开图和调试图
                    unwarp_base = Path(output_filename).stem
                    unwarp_output_path = unwarp_dir / f"{unwarp_base}_unwarp.jpg"
                    unwarp_debug_path = unwarp_dir / f"{unwarp_base}_unwarp_debug.jpg"
                    try:
                        unwarp_meter_using_ellipse(
                            str(output_path),
                            str(unwarp_output_path),
                            str(unwarp_debug_path)
                        )
                        result['unwarped_files'].append({
                            'filename': unwarp_output_path.name,
                            'path': str(unwarp_output_path),
                            'debug_path': str(unwarp_debug_path),
                            'source_cropped': str(output_path)
                        })
                    except Exception as e:
                        print(f"⚠️  矫正仪表时出错 ({output_filename}): {e}")
                
            except Exception as e:
                print(f"⚠️  裁剪仪表时出错 ({image_path.name}, meter {i+1}): {e}")
        
        # 保存可视化结果
        if save_visualization and len(detections) > 0:
            try:
                vis_image = self.visualize_detections(image, detections)
                viz_path = viz_dir / f"{base_name}_detection.jpg"
                cv2.imwrite(str(viz_path), vis_image)
                result['visualization_path'] = str(viz_path)
            except Exception as e:
                print(f"⚠️  保存可视化时出错 ({image_path.name}): {e}")
        
        return result
    
    def process_batch(self, data_dir: Path, target_dir: Path, 
                     save_visualization: bool = True,
                     apply_unwarp: bool = False) -> Dict:
        """
        批量处理图像
        
        Args:
            data_dir: 输入图像目录
            target_dir: 输出目录
            save_visualization: 是否保存可视化结果
            apply_unwarp: 是否对裁剪结果做透视/形变矫正
            
        Returns:
            处理结果统计
        """
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(data_dir.glob(f"*{ext}"))
            image_files.extend(data_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"在 {data_dir} 中未找到图像文件")
        
        print(f"🔍 找到 {len(image_files)} 张图像")
        print(f"📁 输出目录: {target_dir}")
        print(f"🎯 置信度阈值: {self.conf_threshold}")
        print(f"📏 边界扩展: {self.padding} 像素")
        print("")
        
        # 创建输出目录
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每张图像
        all_results = []
        self.stats['total_images'] = len(image_files)
        
        for image_path in tqdm(image_files, desc="处理图像"):
            try:
                result = self.process_single_image(
                    image_path,
                    target_dir,
                    save_visualization,
                    apply_unwarp
                )
                all_results.append(result)
                
                if result['success']:
                    self.stats['successful_detections'] += 1
                    self.stats['total_meters_detected'] += result['num_detections']
                else:
                    self.stats['failed_images'].append({
                        'path': str(image_path),
                        'reason': result.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                print(f"❌ 处理 {image_path.name} 时出错: {e}")
                self.stats['failed_images'].append({
                    'path': str(image_path),
                    'reason': str(e)
                })
        
        # 保存处理结果
        self._save_results(target_dir, all_results)
        
        return {
            'all_results': all_results,
            'statistics': self.stats
        }
    
    def _save_results(self, output_dir: Path, results: List[Dict]):
        """保存处理结果到JSON文件"""
        results_file = output_dir / "detection_results.json"
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'conf_threshold': self.conf_threshold,
            'padding': self.padding,
            'statistics': self.stats,
            'results': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 检测结果已保存: {results_file}")
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("📊 处理统计")
        print("="*60)
        print(f"总图像数: {self.stats['total_images']}")
        print(f"成功检测: {self.stats['successful_detections']}")
        print(f"检测率: {self.stats['successful_detections']/self.stats['total_images']*100:.2f}%")
        print(f"检测到的仪表总数: {self.stats['total_meters_detected']}")
        print(f"平均每张图像的仪表数: {self.stats['total_meters_detected']/max(self.stats['successful_detections'], 1):.2f}")
        
        if self.stats['failed_images']:
            print(f"\n⚠️  未检测到仪表的图像 ({len(self.stats['failed_images'])} 张):")
            for i, failed in enumerate(self.stats['failed_images'][:10]):
                print(f"  {i+1}. {Path(failed['path']).name}: {failed['reason']}")
            if len(self.stats['failed_images']) > 10:
                print(f"  ... 还有 {len(self.stats['failed_images']) - 10} 张")
        
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='指针仪表批量检测和裁剪脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用
  python scripts/meter_detection.py --data-dir data/raw_images --target-dir data/cropped_meters
  
  # 指定模型和置信度
  python scripts/meter_detection.py \\
    --data-dir data/raw_images \\
    --target-dir data/cropped_meters \\
    --model models/detection/best.pt \\
    --conf 0.6
  
  # 不保存可视化结果
  python scripts/meter_detection.py \\
    --data-dir data/raw_images \\
    --target-dir data/cropped_meters \\
    --no-visualization
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='输入图像目录'
    )
    
    parser.add_argument(
        '--target-dir',
        type=str,
        required=True,
        help='输出目录（裁剪后的图像将保存在此）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/detection/detection_model.pt',
        help='YOLO检测模型路径 (默认: models/detection/detection_model.pt)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='检测置信度阈值 (默认: 0.5)'
    )
    
    parser.add_argument(
        '--padding',
        type=int,
        default=20,
        help='裁剪时的边界扩展像素 (默认: 20)'
    )
    
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='不保存可视化结果'
    )

    parser.add_argument(
        '--unwarp',
        action='store_true',
        help='对裁剪后的仪表进行透视/形变矫正并单独输出'
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    data_dir = Path(args.data_dir)
    target_dir = Path(args.target_dir)
    model_path = Path(args.model)
    
    # 检查输入目录
    if not data_dir.exists():
        print(f"❌ 错误: 输入目录不存在: {data_dir}")
        sys.exit(1)
    
    if not data_dir.is_dir():
        print(f"❌ 错误: {data_dir} 不是一个目录")
        sys.exit(1)
    
    # 检查模型文件
    if not model_path.exists():
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        print(f"提示: 请确保模型文件路径正确，或使用 --model 参数指定模型路径")
        sys.exit(1)
    
    print("🚀 指针仪表批量检测和裁剪脚本")
    print("="*60)
    
    try:
        # 创建检测器
        detector = PointerMeterDetector(
            model_path=str(model_path),
            conf_threshold=args.conf,
            padding=args.padding
        )
        
        # 批量处理
        results = detector.process_batch(
            data_dir=data_dir,
            target_dir=target_dir,
            save_visualization=not args.no_visualization,
            apply_unwarp=args.unwarp
        )
        
        # 打印统计信息
        detector.print_statistics()
        
        print("\n✅ 处理完成!")
        print(f"📁 裁剪图像保存在: {target_dir / 'cropped'}")
        if not args.no_visualization:
            print(f"📁 可视化结果保存在: {target_dir / 'visualizations'}")
        if args.unwarp:
            print(f"📁 矫正图像保存在: {target_dir / 'unwarped'}")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
  python scripts/meter_detection.py \
    --data-dir /Users/joe.lu/Desktop/诺华工厂/panel_image \
    --target-dir /Users/joe.lu/Desktop/诺华工厂/output \
    --model models/detection/detection_model.pt \
    --unwarp
"""