#!/usr/bin/env python3
"""
YOLOv10 液晶数字表检测模型训练脚本

此脚本用于训练YOLOv10模型来检测和识别液晶数字表中的数字和小数点。
支持多类别检测：数字0-9和小数点，用于读取液晶数字表的示数。

作者: chijiang
日期: 2025-01-15
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
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class DigitsDetectionTrainer:
    """液晶数字表检测训练器"""
    
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
        self.checkpoint_dir = self.output_dir / "checkpoints" / "digits"
        self.log_dir = self.output_dir / "logs" / "digits"
        self.result_dir = self.output_dir / "results" / "digits"
        
        # 创建目录
        for dir_path in [self.checkpoint_dir, self.log_dir, self.result_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def prepare_yolo_dataset(self) -> str:
        """
        准备YOLO格式数据集，将数据分割为训练集和验证集
        
        Returns:
            YOLO数据集配置文件路径
        """
        print("🔄 Preparing YOLO dataset for digit detection...")
        
        # 数据路径
        data_root = self.project_root / "data" / "digits"
        yolo_root = self.project_root / "data" / "digits_yolo"
        
        # 创建YOLO格式目录
        yolo_root.mkdir(exist_ok=True)
        (yolo_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_dir = data_root / "images"
        label_dir = data_root / "labels"
        
        image_files = list(image_dir.glob("*.jpg"))
        
        # 过滤出有对应标注文件的图像
        valid_images = []
        for img_file in image_files:
            label_file = label_dir / (img_file.stem + ".txt")
            if label_file.exists():
                valid_images.append(img_file)
        
        print(f"📊 Found {len(valid_images)} valid image-label pairs")
        
        # 分割数据集
        train_split = self.config.get('train_split', 0.8)
        train_images, val_images = train_test_split(
            valid_images, 
            train_size=train_split, 
            random_state=self.config.get('seed', 42)
        )
        
        print(f"📝 Training set: {len(train_images)} images")
        print(f"📝 Validation set: {len(val_images)} images")
        
        # 复制训练数据
        self._copy_dataset(train_images, data_root, yolo_root, "train")
        # 复制验证数据
        self._copy_dataset(val_images, data_root, yolo_root, "val")
        
        # 创建数据集配置文件
        # 从classes.txt读取类别信息
        classes_file = data_root / "classes.txt"
        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        
        dataset_config = {
            'train': str(yolo_root / "images" / "train"),
            'val': str(yolo_root / "images" / "val"),
            'nc': len(class_names),  # 类别数量
            'names': class_names  # 类别名称
        }
        
        config_path = yolo_root / "dataset.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Dataset configuration saved to: {config_path}")
        print(f"📋 Classes: {class_names}")
        
        return str(config_path)
    
    def _copy_dataset(self, image_files: List[Path], data_root: Path, 
                     yolo_root: Path, split: str):
        """复制数据集文件"""
        
        for img_file in tqdm(image_files, desc=f"Copying {split} data"):
            # 复制图像文件
            dst_img_path = yolo_root / "images" / split / img_file.name
            if not dst_img_path.exists():
                shutil.copy2(img_file, dst_img_path)
            
            # 复制标注文件
            src_label_path = data_root / "labels" / (img_file.stem + ".txt")
            dst_label_path = yolo_root / "labels" / split / (img_file.stem + ".txt")
            if src_label_path.exists() and not dst_label_path.exists():
                shutil.copy2(src_label_path, dst_label_path)
    
    def train(self) -> str:
        """
        训练YOLOv10模型
        
        Returns:
            最佳模型权重路径
        """
        print("🚀 Starting YOLOv10 digit detection training...")
        
        # 准备数据集
        dataset_config_path = self.prepare_yolo_dataset()
        
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
            'name': self.config.get('experiment_name', 'digit_detection'),
            'exist_ok': True,
            'pretrained': True,
            'optimize': True,
            'verbose': True,
            'seed': self.config.get('seed', 42),
            'deterministic': True,
            'device': self._get_device(),
            'workers': self.config.get('workers', 8),
            'cos_lr': self.config.get('cos_lr', False),
            'close_mosaic': self.config.get('close_mosaic', 10),
            'amp': self.config.get('amp', True),
        }
        
        # 数据增强参数（针对数字识别优化）
        augment_args = self.config.get('augmentation', {})
        train_args.update({
            'hsv_h': augment_args.get('hsv_h', 0.01),  # 较小的色调变化
            'hsv_s': augment_args.get('hsv_s', 0.5),   # 适中的饱和度变化
            'hsv_v': augment_args.get('hsv_v', 0.3),   # 适中的亮度变化
            'degrees': augment_args.get('degrees', 5.0),  # 小幅旋转
            'translate': augment_args.get('translate', 0.05),  # 小幅平移
            'scale': augment_args.get('scale', 0.3),    # 适中的缩放
            'shear': augment_args.get('shear', 2.0),    # 小幅剪切
            'perspective': augment_args.get('perspective', 0.0),  # 不使用透视变换
            'flipud': augment_args.get('flipud', 0.0),  # 不进行上下翻转
            'fliplr': augment_args.get('fliplr', 0.0),  # 不进行左右翻转
            'mosaic': augment_args.get('mosaic', 0.5),  # 减少马赛克增强
            'mixup': augment_args.get('mixup', 0.0),    # 不使用mixup
            'copy_paste': augment_args.get('copy_paste', 0.0),  # 不使用copy_paste
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
        print("📊 Evaluating digit detection model...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 准备验证数据
        dataset_config_path = self.project_root / "data" / "digits_yolo" / "dataset.yaml"
        
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
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def visualize_predictions(self, model_path: str, num_samples: int = 10):
        """
        可视化模型预测结果
        
        Args:
            model_path: 模型权重路径
            num_samples: 可视化样本数量
        """
        print("🎨 Visualizing digit detection predictions...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 获取验证图像
        val_image_dir = self.project_root / "data" / "digits_yolo" / "images" / "val"
        image_files = list(val_image_dir.glob("*.jpg"))[:num_samples]
        
        # 创建可视化目录
        viz_dir = self.result_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for i, image_path in enumerate(image_files):
            # 预测
            results = model(str(image_path), conf=0.3)
            
            # 绘制结果
            annotated_img = results[0].plot()
            
            # 保存可视化结果
            output_path = viz_dir / f"digit_prediction_{i+1}_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_img)
        
        print(f"✅ Visualizations saved to: {viz_dir}")
    
    def predict_reading(self, model_path: str, image_path: str) -> str:
        """
        预测液晶数字表的读数
        
        Args:
            model_path: 模型权重路径
            image_path: 图像路径
            
        Returns:
            识别的数字读数
        """
        print(f"🔍 Predicting reading from: {image_path}")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 预测
        results = model(image_path, conf=0.3)
        
        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # 获取类别名称
                    class_name = result.names[cls]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'center_x': (x1 + x2) / 2
                    })
        
        # 按x坐标排序（从左到右）
        detections.sort(key=lambda x: x['center_x'])
        
        # 构建读数字符串
        reading = ""
        for det in detections:
            if det['class'] == 'point':
                reading += '.'
            else:
                reading += det['class']
        
        print(f"📊 Detected reading: {reading}")
        return reading
    
    def export_model(self, model_path: str, formats: List[str] = ['onnx']):
        """
        导出模型到不同格式
        
        Args:
            model_path: 模型权重路径
            formats: 导出格式列表
        """
        print("📦 Exporting digit detection model...")
        
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
        'epochs': 150,
        'batch_size': 16,
        'image_size': 640,
        'learning_rate': 0.01,
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'patience': 50,
        'save_period': 10,
        'train_split': 0.8,
        'experiment_name': 'digit_detection',
        'device': 'auto',
        'workers': 8,
        'seed': 42,
        'cos_lr': False,
        'close_mosaic': 10,
        'amp': True,
        'augmentation': {
            'hsv_h': 0.01,    # 小幅色调变化
            'hsv_s': 0.5,     # 适中饱和度变化
            'hsv_v': 0.3,     # 适中亮度变化
            'degrees': 5.0,   # 小幅旋转
            'translate': 0.05, # 小幅平移
            'scale': 0.3,     # 适中缩放
            'shear': 2.0,     # 小幅剪切
            'perspective': 0.0, # 不使用透视变换
            'flipud': 0.0,    # 不使用上下翻转
            'fliplr': 0.0,    # 不使用左右翻转
            'mosaic': 0.5,    # 减少马赛克增强
            'mixup': 0.0,     # 不使用mixup
            'copy_paste': 0.0  # 不使用copy_paste
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv10 液晶数字表检测训练脚本')
    parser.add_argument('--config', type=str, default='config/digits_config.yaml',
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
    parser.add_argument('--predict', type=str,
                       help='预测单张图像的读数')
    
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
    trainer = DigitsDetectionTrainer(args.config)
    
    try:
        if args.predict:
            # 预测单张图像
            if not args.model_path:
                print("❌ Model path required for prediction")
                return
            
            reading = trainer.predict_reading(args.model_path, args.predict)
            print(f"🎯 Final reading: {reading}")
            
        elif args.eval_only:
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