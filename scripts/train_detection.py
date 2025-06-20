#!/usr/bin/env python3
"""
YOLOv10 仪表检测模型训练脚本

此脚本用于训练YOLOv10模型来检测工业仪表。
支持从COCO格式数据集进行训练，包含数据预处理、模型训练、验证等功能。

作者: chijiang
日期: 2025-06-06
"""

import os
import sys
import argparse
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class MeterDetectionTrainer:
    """仪表检测训练器"""
    
    def __init__(self, config_path: str):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.project_root = Path(__file__).parent.parent
        self.setup_directories()
        
    def load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _get_device(self) -> str:
        """
        智能检测最佳设备
        
        Returns:
            设备字符串
        """
        config_device = self.config.get('device', 'auto')
        
        if config_device != 'auto':
            return config_device
        
        # 自动检测设备
        if torch.cuda.is_available():
            device = '0'
            print(f"🔥 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("🍎 Using Apple MPS acceleration")
        else:
            device = 'cpu'
            print("💻 Using CPU")
        
        # 输出设备信息
        print(f"🎯 Device selected: {device}")
        if device == 'mps':
            print("ℹ️  Apple Silicon detected - using Metal Performance Shaders for acceleration")
        elif device == 'cpu':
            print("⚠️  No GPU acceleration available - training will be slower")
            
        return device
    
    def setup_directories(self):
        """设置输出目录"""
        self.output_dir = self.project_root / "outputs"
        self.checkpoint_dir = self.output_dir / "checkpoints" / "detection"
        self.log_dir = self.output_dir / "logs" / "detection"
        self.result_dir = self.output_dir / "results" / "detection"
        
        # 创建目录
        for dir_path in [self.checkpoint_dir, self.log_dir, self.result_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def prepare_yolo_dataset(self) -> str:
        """
        将COCO格式数据转换为YOLO格式
        
        Returns:
            YOLO数据集配置文件路径
        """
        print("🔄 Converting COCO dataset to YOLO format...")
        
        # 数据路径
        data_root = self.project_root / "data" / "detection"
        yolo_root = self.project_root / "data" / "detection_yolo"
        
        # 创建YOLO格式目录
        yolo_root.mkdir(exist_ok=True)
        (yolo_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # 转换训练集和验证集
        for split in ['train', 'val']:
            self._convert_coco_to_yolo(
                coco_ann_file=data_root / "annotations" / f"meter_coco_{split}.json",
                image_dir=data_root / f"{split}2017",
                output_image_dir=yolo_root / "images" / split,
                output_label_dir=yolo_root / "labels" / split
            )
        
        # 创建数据集配置文件
        dataset_config = {
            'train': str(yolo_root / "images" / "train"),
            'val': str(yolo_root / "images" / "val"),
            'nc': 1,  # 类别数量
            'names': ['meter']  # 类别名称
        }
        
        config_path = yolo_root / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f)
        
        return str(config_path)
    
    def _convert_coco_to_yolo(self, coco_ann_file: Path, image_dir: Path, 
                             output_image_dir: Path, output_label_dir: Path):
        """转换单个COCO数据集到YOLO格式"""
        
        # 加载COCO标注
        with open(coco_ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # 构建图像ID到文件名的映射
        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
        
        # 按图像ID分组标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # 转换每个图像
        for image_id, filename in tqdm(image_id_to_filename.items(), desc=f"Converting {coco_ann_file.stem}"):
            # 复制图像文件
            src_image_path = image_dir / filename
            dst_image_path = output_image_dir / filename
            
            if src_image_path.exists():
                if not dst_image_path.exists():
                    shutil.copy2(src_image_path, dst_image_path)
                
                # 转换标注
                if image_id in annotations_by_image:
                    width, height = image_id_to_size[image_id]
                    yolo_labels = []
                    
                    for ann in annotations_by_image[image_id]:
                        bbox = ann['bbox']  # [x, y, width, height]
                        
                        # 转换为YOLO格式 [class_id, x_center, y_center, width, height] (相对坐标)
                        x_center = (bbox[0] + bbox[2] / 2) / width
                        y_center = (bbox[1] + bbox[3] / 2) / height
                        norm_width = bbox[2] / width
                        norm_height = bbox[3] / height
                        
                        # COCO类别ID从1开始，YOLO从0开始
                        class_id = ann['category_id'] - 1
                        
                        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                    
                    # 保存YOLO标注文件
                    label_filename = filename.rsplit('.', 1)[0] + '.txt'
                    label_path = output_label_dir / label_filename
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_labels))
                else:
                    # 创建空标注文件（负样本）
                    label_filename = filename.rsplit('.', 1)[0] + '.txt'
                    label_path = output_label_dir / label_filename
                    label_path.touch()
    
    def train(self) -> str:
        """
        训练YOLOv10模型
        
        Returns:
            最佳模型权重路径
        """
        print("🚀 Starting YOLOv10 training...")
        
        # 准备数据集
        # dataset_config_path = self.prepare_yolo_dataset()
        dataset_config_path = self.project_root / "data" / "detection_yolo" / "dataset.yaml"
        
        # 初始化模型
        model_name = self.config.get('model', 'yolov10n.pt')
        model = YOLO(model_name)
        
        # 训练参数
        train_args = {
            'data': dataset_config_path,
            'epochs': self.config.get('epochs', 100),
            'imgsz': self.config.get('image_size', 640),
            'batch': self.config.get('batch_size', 16),
            'lr0': self.config.get('learning_rate', 0.01),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'momentum': self.config.get('momentum', 0.937),
            'patience': self.config.get('patience', 50),
            'save_period': self.config.get('save_period', 10),
            'project': str(self.checkpoint_dir),
            'name': self.config.get('experiment_name', 'meter_detection'),
            'exist_ok': True,
            'pretrained': True,
            'optimize': True,
            'verbose': True,
            'seed': self.config.get('seed', 42),
            'deterministic': True,
            'single_cls': True,  # 单类别检测
            'device': self._get_device(),
            'workers': self.config.get('workers', 8),
            'cos_lr': self.config.get('cos_lr', False),
            'close_mosaic': self.config.get('close_mosaic', 10),
            'amp': self.config.get('amp', True),
        }
        
        # 数据增强参数
        augment_args = self.config.get('augmentation', {})
        train_args.update({
            'hsv_h': augment_args.get('hsv_h', 0.015),
            'hsv_s': augment_args.get('hsv_s', 0.7),
            'hsv_v': augment_args.get('hsv_v', 0.4),
            'degrees': augment_args.get('degrees', 0.0),
            'translate': augment_args.get('translate', 0.1),
            'scale': augment_args.get('scale', 0.5),
            'shear': augment_args.get('shear', 0.0),
            'perspective': augment_args.get('perspective', 0.0),
            'flipud': augment_args.get('flipud', 0.0),
            'fliplr': augment_args.get('fliplr', 0.5),
            'mosaic': augment_args.get('mosaic', 1.0),
            'mixup': augment_args.get('mixup', 0.0),
            'copy_paste': augment_args.get('copy_paste', 0.0),
        })
        
        print(f"📊 Training configuration:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        
        # 开始训练
        try:
            results = model.train(**train_args)
            
            # 获取最佳模型路径
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"
            
            print(f"✅ Training completed!")
            print(f"📁 Best model saved at: {best_model_path}")
            
            return str(best_model_path)
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise
    
    def evaluate(self, model_path: str) -> Dict:
        """
        评估训练好的模型
        
        Args:
            model_path: 模型权重路径
            
        Returns:
            评估结果字典
        """
        print("📊 Evaluating model...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 准备验证数据
        dataset_config_path = self.project_root / "data" / "detection_yolo" / "dataset.yaml"
        
        # 运行验证
        results = model.val(
            data=str(dataset_config_path),
            imgsz=self.config.get('image_size', 640),
            batch=self.config.get('batch_size', 16),
            conf=0.001,
            iou=0.6,
            max_det=300,
            device=self._get_device(),
            save_json=True,
            save_hybrid=True,
            plots=True,
            verbose=True
        )
        
        # 提取评估指标
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / (float(results.box.mp) + float(results.box.mr)) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
        }
        
        print(f"📈 Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 保存评估结果
        results_file = self.result_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def visualize_predictions(self, model_path: str, num_samples: int = 10):
        """
        可视化模型预测结果
        
        Args:
            model_path: 模型权重路径
            num_samples: 可视化样本数量
        """
        print("🎨 Visualizing predictions...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 获取验证图像
        val_image_dir = self.project_root / "data" / "detection_yolo" / "images" / "val"
        image_files = list(val_image_dir.glob("*.jpg"))[:num_samples]
        
        # 创建可视化目录
        viz_dir = self.result_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for i, image_path in enumerate(image_files):
            # 预测
            results = model(str(image_path), conf=0.5)
            
            # 绘制结果
            annotated_img = results[0].plot()
            
            # 保存可视化结果
            output_path = viz_dir / f"prediction_{i+1}_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_img)
        
        print(f"✅ Visualizations saved to: {viz_dir}")
    
    def export_model(self, model_path: str, formats: List[str] = ['onnx']):
        """
        导出模型到不同格式
        
        Args:
            model_path: 模型权重路径
            formats: 导出格式列表
        """
        print("📦 Exporting model...")
        
        model = YOLO(model_path)
        
        for format_name in formats:
            try:
                print(f"  Exporting to {format_name.upper()}...")
                exported_model = model.export(
                    format=format_name,
                    imgsz=self.config.get('image_size', 640),
                    optimize=True,
                    half=False,
                    int8=False,
                    dynamic=False,
                    simplify=True,
                    opset=None,
                    workspace=4,
                    nms=False
                )
                print(f"  ✅ {format_name.upper()} model exported: {exported_model}")
            except Exception as e:
                print(f"  ❌ Failed to export {format_name}: {str(e)}")


def create_default_config() -> Dict:
    """创建默认配置"""
    return {
        'model': 'yolov10n.pt',  # YOLOv10 nano模型
        'epochs': 100,
        'batch_size': 16,
        'image_size': 640,
        'learning_rate': 0.01,
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'patience': 50,
        'save_period': 10,
        'experiment_name': 'meter_detection',
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'seed': 42,
        'cos_lr': False,
        'close_mosaic': 10,
        'amp': True,
        'augmentation': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv10 仪表检测训练脚本')
    parser.add_argument('--config', type=str, default='config/detection_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--create-config', action='store_true',
                       help='创建默认配置文件')
    parser.add_argument('--eval-only', action='store_true',
                       help='只运行评估')
    parser.add_argument('--model-path', type=str,
                       help='预训练模型路径（用于评估）')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化预测结果')
    parser.add_argument('--export', action='store_true',
                       help='导出模型')
    parser.add_argument('--export-formats', nargs='+', default=['onnx'],
                       help='导出格式列表')
    
    args = parser.parse_args()
    
    # 创建默认配置文件
    if args.create_config:
        config_path = Path(args.config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = create_default_config()
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Default configuration created at: {config_path}")
        return
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        print("💡 Use --create-config to create a default configuration file")
        return
    
    # 初始化训练器
    trainer = MeterDetectionTrainer(args.config)
    
    try:
        if args.eval_only:
            # 只运行评估
            if not args.model_path:
                print("❌ Model path required for evaluation")
                return
            
            metrics = trainer.evaluate(args.model_path)
            
            if args.visualize:
                trainer.visualize_predictions(args.model_path)
            
            if args.export:
                trainer.export_model(args.model_path, args.export_formats)
        
        else:
            # 运行完整训练流程
            best_model_path = trainer.train()
            
            # 评估模型
            metrics = trainer.evaluate(best_model_path)
            
            if args.visualize:
                trainer.visualize_predictions(best_model_path)
            
            if args.export:
                trainer.export_model(best_model_path, args.export_formats)
            
            print("🎉 Training pipeline completed successfully!")
            print(f"📁 Best model: {best_model_path}")
            print(f"📊 Final mAP@0.5: {metrics['mAP50']:.4f}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 