#!/usr/bin/env python3
"""
液晶数字表读数提取脚本
Digital Meter Reading Extraction Script

完整流程：检测 -> 裁剪 -> 增强 -> OCR -> 结果输出
Complete pipeline: Detection -> Crop -> Enhancement -> OCR -> Result Output

作者: chijiang
日期: 2025-06-09
"""

import cv2
import numpy as np
import torch
import argparse
import os
import sys
from pathlib import Path
import json
import time
from typing import Tuple, List, Optional, Dict, Any, Union
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# 导入项目模块
try:
    from scripts.digital_meter_detection.enhancement.digital_display_enhancer import DigitalDisplayEnhancer
    from scripts.digital_meter_detection.ocr.digital_ocr_extractor import DigitalOCRExtractor
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 尝试导入YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  警告: ultralytics未安装，无法使用YOLO检测功能")


class DigitalMeterReader:
    """液晶数字表读数提取器"""
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str = None,
                 ocr_engine: str = "easyocr",
                 device: str = "auto",
                 confidence_threshold: float = 0.5,
                 enhancement_enabled: bool = True,
                 debug: bool = False):
        """
        初始化液晶数字表读数提取器
        
        Args:
            model_path: YOLO检测模型路径
            output_dir: 输出目录
            ocr_engine: OCR引擎 ("easyocr", "paddleocr", "tesseract")
            device: 设备选择 ("auto", "cpu", "cuda")
            confidence_threshold: 检测置信度阈值
            enhancement_enabled: 是否启用图像增强
            debug: 调试模式
        """
        self.model_path = Path(model_path)
        self.ocr_engine = ocr_engine
        self.confidence_threshold = confidence_threshold
        self.enhancement_enabled = enhancement_enabled
        self.debug = debug
        
        # 设置输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = project_root / "outputs" / "digital_meter_reading" / f"reading_{timestamp}"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.detection_dir = self.output_dir / "1_detection"
        self.cropped_dir = self.output_dir / "2_cropped"
        self.enhanced_dir = self.output_dir / "3_enhanced"
        self.ocr_dir = self.output_dir / "4_ocr_results"
        self.visualization_dir = self.output_dir / "5_visualization"
        
        for dir_path in [self.detection_dir, self.cropped_dir, self.enhanced_dir, 
                        self.ocr_dir, self.visualization_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 设置设备
        self.device = self._get_device(device)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self._initialize_components()
        
        self.logger.info(f"🔧 液晶数字表读数提取器已初始化")
        self.logger.info(f"📁 输出目录: {self.output_dir}")
        self.logger.info(f"🔧 OCR引擎: {self.ocr_engine}")
        self.logger.info(f"💻 设备: {self.device}")
        
    def _get_device(self, device: str) -> str:
        """智能设备选择"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.output_dir / "digital_meter_reading.log"
        
        # 创建logger
        self.logger = logging.getLogger('DigitalMeterReader')
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def _initialize_components(self):
        """初始化各个组件"""
        # 初始化YOLO模型
        self.model = None
        if YOLO_AVAILABLE and self.model_path.exists():
            try:
                self.model = YOLO(str(self.model_path))
                self.logger.info(f"✅ YOLO模型加载成功: {self.model_path}")
            except Exception as e:
                self.logger.error(f"❌ YOLO模型加载失败: {e}")
                raise
        elif not self.model_path.exists():
            self.logger.error(f"❌ 模型文件不存在: {self.model_path}")
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 初始化图像增强器
        self.enhancer = None
        if self.enhancement_enabled:
            try:
                # 为增强器创建专用输出目录，避免冲突
                enhancer_output_dir = self.enhanced_dir / "enhancer_workspace"
                self.enhancer = DigitalDisplayEnhancer(
                    output_dir=enhancer_output_dir
                )
                self.logger.info("✅ 图像增强器初始化成功")
            except Exception as e:
                self.logger.warning(f"⚠️  图像增强器初始化失败: {e}")
                self.enhancement_enabled = False
        
        # 初始化OCR提取器
        try:
            self.ocr_extractor = DigitalOCRExtractor(
                ocr_engine=self.ocr_engine
            )
            self.logger.info("✅ OCR提取器初始化成功")
        except Exception as e:
            self.logger.error(f"❌ OCR提取器初始化失败: {e}")
            raise
    
    def detect_digital_displays(self, image: np.ndarray, image_name: str) -> List[Dict]:
        """
        检测液晶显示屏区域
        
        Args:
            image: 输入图像
            image_name: 图像名称
            
        Returns:
            检测结果列表，每个元素包含边界框和置信度
        """
        if self.model is None:
            self.logger.error("YOLO模型未加载")
            return []
        
        self.logger.info(f"🔍 检测液晶显示屏: {image_name}")
        
        # YOLO检测
        results = self.model(image, conf=self.confidence_threshold, device=self.device)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # 获取边界框坐标 (xyxy格式)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'detection_id': i
                    }
                    detections.append(detection)
        
        self.logger.info(f"🎯 检测到 {len(detections)} 个液晶显示屏")
        
        # 保存检测可视化
        if detections:
            vis_image = self._visualize_detections(image, detections)
            vis_path = self.detection_dir / f"{Path(image_name).stem}_detection.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            self.logger.info(f"💾 检测可视化保存到: {vis_path}")
        
        return detections
    
    def _visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """可视化检测结果"""
        vis_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制置信度标签
            label = f"Digital Display: {confidence:.3f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return vis_image
    
    def crop_digital_displays(self, image: np.ndarray, detections: List[Dict], 
                            image_name: str) -> List[Dict]:
        """
        裁剪液晶显示屏区域
        
        Args:
            image: 原始图像
            detections: 检测结果
            image_name: 图像名称
            
        Returns:
            裁剪结果列表，包含裁剪图像和相关信息
        """
        self.logger.info(f"✂️  裁剪液晶显示屏区域: {image_name}")
        
        cropped_results = []
        base_name = Path(image_name).stem
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # 添加边界扩展（避免裁剪过紧）
            padding = 10
            h, w = image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 裁剪图像
            cropped_image = image[y1:y2, x1:x2]
            
            if cropped_image.size == 0:
                self.logger.warning(f"⚠️  跳过无效裁剪区域: detection {i}")
                continue
            
            # 保存裁剪图像
            crop_filename = f"{base_name}_crop_{i:02d}.jpg"
            crop_path = self.cropped_dir / crop_filename
            cv2.imwrite(str(crop_path), cropped_image)
            
            cropped_result = {
                'image': cropped_image,
                'bbox': [x1, y1, x2, y2],
                'confidence': detection['confidence'],
                'crop_filename': crop_filename,
                'crop_path': str(crop_path),
                'detection_id': detection['detection_id']
            }
            cropped_results.append(cropped_result)
            
            self.logger.info(f"💾 裁剪图像保存: {crop_path}")
        
        return cropped_results
    
    def enhance_cropped_images(self, cropped_results: List[Dict]) -> List[Dict]:
        """
        增强裁剪后的图像
        
        Args:
            cropped_results: 裁剪结果列表
            
        Returns:
            增强结果列表
        """
        if not self.enhancement_enabled or self.enhancer is None:
            self.logger.info("⏭️  跳过图像增强")
            return cropped_results
        
        self.logger.info(f"🎨 开始图像增强，共 {len(cropped_results)} 张图像")
        
        enhanced_results = []
        
        for cropped_result in cropped_results:
            try:
                # 增强图像
                enhanced_result_dict = self.enhancer.enhance_single_image(cropped_result['image'])
                enhanced_image = enhanced_result_dict['final']
                
                # 保存增强图像
                base_name = Path(cropped_result['crop_filename']).stem
                enhanced_filename = f"{base_name}_enhanced.jpg"
                enhanced_path = self.enhanced_dir / enhanced_filename
                cv2.imwrite(str(enhanced_path), enhanced_image)
                
                # 更新结果
                enhanced_result = cropped_result.copy()
                enhanced_result.update({
                    'enhanced_image': enhanced_image,
                    'enhanced_filename': enhanced_filename,
                    'enhanced_path': str(enhanced_path)
                })
                enhanced_results.append(enhanced_result)
                
                self.logger.info(f"✨ 图像增强完成: {enhanced_filename}")
                
            except Exception as e:
                self.logger.error(f"❌ 图像增强失败: {cropped_result['crop_filename']}, 错误: {e}")
                # 使用原图像
                enhanced_results.append(cropped_result)
        
        return enhanced_results
    
    def extract_readings(self, enhanced_results: List[Dict]) -> List[Dict]:
        """
        从增强图像中提取数字读数
        
        Args:
            enhanced_results: 增强结果列表
            
        Returns:
            OCR结果列表
        """
        self.logger.info(f"🔤 开始OCR提取，共 {len(enhanced_results)} 张图像")
        
        ocr_results = []
        
        for enhanced_result in enhanced_results:
            try:
                # 选择处理图像（增强图像优先，否则使用原图）
                process_image = enhanced_result.get('enhanced_image', enhanced_result['image'])
                image_name = enhanced_result.get('enhanced_filename', enhanced_result['crop_filename'])
                
                # OCR提取
                ocr_result = self.ocr_extractor.extract_from_image(process_image)
                
                # 整合结果
                final_result = enhanced_result.copy()
                final_result.update({
                    'ocr_raw_results': ocr_result['raw_results'],
                    'ocr_validated_results': ocr_result['validated_results'],
                    'extracted_value': ocr_result['best_result']['value'] if ocr_result['best_result'] else None,
                    'confidence': ocr_result['best_result']['confidence'] if ocr_result['best_result'] else 0.0,
                    'ocr_success': ocr_result['best_result'] is not None
                })
                
                ocr_results.append(final_result)
                
                if final_result['ocr_success']:
                    self.logger.info(f"🔢 OCR成功: {image_name} -> {final_result['extracted_value']}")
                else:
                    self.logger.warning(f"⚠️  OCR失败: {image_name}")
                
            except Exception as e:
                self.logger.error(f"❌ OCR提取失败: {enhanced_result.get('crop_filename', 'unknown')}, 错误: {e}")
                # 添加失败结果
                final_result = enhanced_result.copy()
                final_result.update({
                    'extracted_value': None,
                    'confidence': 0.0,
                    'ocr_success': False,
                    'error': str(e)
                })
                ocr_results.append(final_result)
        
        return ocr_results
    
    def process_single_image(self, image_path: str) -> Dict:
        """
        处理单张图像的完整流程
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果字典
        """
        image_path = Path(image_path)
        self.logger.info(f"🚀 开始处理图像: {image_path.name}")
        
        start_time = time.time()
        
        # 加载图像
        image = cv2.imread(str(image_path))
        if image is None:
            error_msg = f"无法加载图像: {image_path}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        try:
            # 1. 检测液晶显示屏
            detections = self.detect_digital_displays(image, image_path.name)
            if not detections:
                self.logger.warning("⚠️  未检测到液晶显示屏")
                return {
                    'success': True,
                    'image_name': image_path.name,
                    'processing_time': time.time() - start_time,
                    'detections_count': 0,
                    'readings': []
                }
            
            # 2. 裁剪显示屏区域
            cropped_results = self.crop_digital_displays(image, detections, image_path.name)
            
            # 3. 图像增强（可选）
            enhanced_results = self.enhance_cropped_images(cropped_results)
            
            # 4. OCR数字提取
            final_results = self.extract_readings(enhanced_results)
            
            # 5. 生成可视化结果
            self.create_final_visualization(image, final_results, image_path.name)
            
            # 统计成功率
            successful_readings = [r for r in final_results if r['ocr_success']]
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ 图像处理完成: {image_path.name}")
            self.logger.info(f"⏱️  处理时间: {processing_time:.2f}秒")
            self.logger.info(f"📊 成功读取: {len(successful_readings)}/{len(final_results)}")
            
            return {
                'success': True,
                'image_name': image_path.name,
                'processing_time': processing_time,
                'detections_count': len(detections),
                'readings': final_results,
                'successful_readings': len(successful_readings),
                'total_detections': len(final_results)
            }
            
        except Exception as e:
            error_msg = f"处理图像时发生错误: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'image_name': image_path.name,
                'error': error_msg,
                'processing_time': time.time() - start_time
            }
    
    def create_final_visualization(self, original_image: np.ndarray, 
                                 results: List[Dict], image_name: str):
        """创建最终可视化结果"""
        vis_image = original_image.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            success = result['ocr_success']
            value = result['extracted_value']
            confidence = result['confidence']
            
            # 选择颜色
            color = (0, 255, 0) if success else (0, 0, 255)  # 绿色=成功，红色=失败
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签
            if success and value is not None:
                label = f"{value:.3f} ({confidence:.3f})"
            else:
                label = "Failed"
            
            # 绘制标签背景和文字
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            text_color = (255, 255, 255) if success else (255, 255, 255)
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # 保存可视化结果
        vis_path = self.visualization_dir / f"{Path(image_name).stem}_final_result.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        self.logger.info(f"🎨 最终可视化保存: {vis_path}")
    
    def process_batch(self, input_dir: str) -> Dict:
        """
        批量处理图像
        
        Args:
            input_dir: 输入目录
            
        Returns:
            批量处理结果
        """
        input_dir = Path(input_dir)
        self.logger.info(f"📦 开始批量处理: {input_dir}")
        
        # 查找图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            error_msg = f"在目录中未找到图像文件: {input_dir}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        self.logger.info(f"📸 找到 {len(image_files)} 张图像")
        
        batch_results = []
        start_time = time.time()
        
        for i, image_file in enumerate(image_files):
            self.logger.info(f"📊 处理进度: {i+1}/{len(image_files)}")
            result = self.process_single_image(str(image_file))
            batch_results.append(result)
        
        # 统计批量结果
        total_time = time.time() - start_time
        successful_images = len([r for r in batch_results if r['success']])
        total_detections = sum(r.get('detections_count', 0) for r in batch_results if r['success'])
        total_readings = sum(r.get('successful_readings', 0) for r in batch_results if r['success'])
        
        batch_summary = {
            'success': True,
            'total_images': len(image_files),
            'successful_images': successful_images,
            'total_detections': total_detections,
            'successful_readings': total_readings,
            'processing_time': total_time,
            'average_time_per_image': total_time / len(image_files),
            'results': batch_results
        }
        
        self.logger.info(f"✅ 批量处理完成!")
        self.logger.info(f"📊 图像: {successful_images}/{len(image_files)}")
        self.logger.info(f"📊 检测: {total_detections}")
        self.logger.info(f"📊 读数: {total_readings}")
        self.logger.info(f"⏱️  总时间: {total_time:.2f}秒")
        
        # 保存批量结果
        self.save_batch_results(batch_summary)
        
        return batch_summary
    
    def save_batch_results(self, batch_summary: Dict):
        """保存批量处理结果"""
        # 保存JSON结果
        json_path = self.output_dir / "batch_results.json"
        
        # 准备可序列化的数据
        serializable_summary = {}
        for key, value in batch_summary.items():
            if key == 'results':
                # 处理结果列表，移除不可序列化的图像数据
                serializable_results = []
                for result in value:
                    clean_result = {}
                    for k, v in result.items():
                        if k not in ['image', 'enhanced_image']:  # 跳过图像数据
                            if isinstance(v, list):
                                # 清理列表中的图像数据
                                clean_list = []
                                for item in v:
                                    if isinstance(item, dict):
                                        clean_item = {ik: iv for ik, iv in item.items() 
                                                    if ik not in ['image', 'enhanced_image']}
                                        clean_list.append(clean_item)
                                    else:
                                        clean_list.append(item)
                                clean_result[k] = clean_list
                            else:
                                clean_result[k] = v
                    serializable_results.append(clean_result)
                serializable_summary[key] = serializable_results
            else:
                serializable_summary[key] = value
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 批量结果保存: {json_path}")
        
        # 生成Markdown报告
        self.generate_markdown_report(batch_summary)
    
    def generate_markdown_report(self, batch_summary: Dict):
        """生成Markdown报告"""
        report_path = self.output_dir / "batch_report.md"
        
        report_content = f"""# 液晶数字表读数提取报告
Digital Meter Reading Extraction Report

## 处理摘要

- **处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总图像数**: {batch_summary['total_images']}
- **成功处理**: {batch_summary['successful_images']}
- **总检测数**: {batch_summary['total_detections']}
- **成功读数**: {batch_summary['successful_readings']}
- **处理时间**: {batch_summary['processing_time']:.2f}秒
- **平均处理时间**: {batch_summary['average_time_per_image']:.2f}秒/图像

## 成功率统计

- **图像处理成功率**: {batch_summary['successful_images']/batch_summary['total_images']*100:.1f}%
- **OCR读数成功率**: {batch_summary['successful_readings']/max(batch_summary['total_detections'], 1)*100:.1f}%

## 详细结果

| 图像名称 | 检测数量 | 成功读数 | 处理时间(s) | 状态 |
|---------|---------|---------|------------|------|
"""
        
        for result in batch_summary['results']:
            if result['success']:
                status = "✅ 成功"
                detections = result.get('detections_count', 0)
                readings = result.get('successful_readings', 0)
                time_taken = result.get('processing_time', 0)
            else:
                status = "❌ 失败"
                detections = 0
                readings = 0
                time_taken = result.get('processing_time', 0)
            
            report_content += f"| {result['image_name']} | {detections} | {readings} | {time_taken:.2f} | {status} |\n"
        
        report_content += f"""
## 输出文件结构

```
{self.output_dir.name}/
├── 1_detection/          # 检测可视化结果
├── 2_cropped/            # 裁剪的显示屏区域
├── 3_enhanced/           # 增强后的图像
├── 4_ocr_results/        # OCR详细结果
├── 5_visualization/      # 最终可视化结果
├── batch_results.json    # 详细结果数据
├── batch_report.md       # 本报告
└── digital_meter_reading.log  # 处理日志
```

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*处理引擎: 液晶数字表读数提取器 v1.0*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"📄 报告生成: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="液晶数字表读数提取")
    parser.add_argument('--input', '-i', required=True, 
                       help='输入图像文件或目录路径')
    parser.add_argument('--model', '-m', 
                       default='models/detection/digital_detection_model.pt',
                       help='YOLO检测模型路径')
    parser.add_argument('--output', '-o', 
                       help='输出目录（默认自动生成）')
    parser.add_argument('--ocr-engine', choices=['easyocr', 'paddleocr', 'tesseract'],
                       default='easyocr', help='OCR引擎选择')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'],
                       default='auto', help='设备选择')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='检测置信度阈值')
    parser.add_argument('--no-enhancement', action='store_true',
                       help='禁用图像增强')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    args = parser.parse_args()
    
    try:
        # 检查模型文件路径，支持相对路径和绝对路径
        model_path = Path(args.model)
        
        # 如果是相对路径，尝试从项目根目录解析
        if not model_path.is_absolute():
            # 尝试相对于项目根目录
            project_relative_path = project_root / model_path
            if project_relative_path.exists():
                model_path = project_relative_path
            elif not model_path.exists():
                print(f"❌ 模型文件不存在: {model_path}")
                print(f"   也尝试了: {project_relative_path}")
                print("请确保模型文件路径正确，或先训练模型")
                print("💡 提示：可以使用 run.py 中的训练功能生成模型")
                return
        
        # 检查输入路径
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ 输入路径不存在: {input_path}")
            print("请检查文件或目录路径是否正确")
            return
        
        # 验证模型文件确实可以加载
        if not YOLO_AVAILABLE:
            print("❌ ultralytics库未安装，无法使用YOLO功能")
            print("请安装: pip install ultralytics")
            return
        
        # 创建读数提取器
        reader = DigitalMeterReader(
            model_path=str(model_path),
            output_dir=args.output,
            ocr_engine=args.ocr_engine,
            device=args.device,
            confidence_threshold=args.confidence,
            enhancement_enabled=not args.no_enhancement,
            debug=args.debug
        )
        
        print(f"🚀 开始液晶数字表读数提取")
        print(f"📁 输入: {input_path}")
        print(f"🤖 模型: {model_path}")
        print(f"📁 输出: {reader.output_dir}")
        
        # 处理输入
        if input_path.is_file():
            # 单文件处理
            result = reader.process_single_image(str(input_path))
            if result['success']:
                print(f"✅ 处理完成!")
                if result['detections_count'] > 0:
                    print(f"📊 检测到 {result['detections_count']} 个显示屏")
                    print(f"📊 成功读取 {result['successful_readings']} 个数值")
                else:
                    print("⚠️  未检测到液晶显示屏")
            else:
                print(f"❌ 处理失败: {result['error']}")
        
        else:
            # 批量处理
            result = reader.process_batch(str(input_path))
            if result['success']:
                print(f"✅ 批量处理完成!")
                print(f"📊 成功处理: {result['successful_images']}/{result['total_images']} 张图像")
                print(f"📊 总检测数: {result['total_detections']}")
                print(f"📊 成功读数: {result['successful_readings']}")
            else:
                print(f"❌ 批量处理失败: {result['error']}")
        
        print(f"📁 详细结果请查看: {reader.output_dir}")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()