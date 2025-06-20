#!/usr/bin/env python3
"""
YOLOv10 LED颜色识别模型训练脚本

此脚本用于训练YOLOv10模型来识别LED灯的颜色（绿色、红色、黄色）。
支持从YOLO格式数据集进行训练，包含数据预处理、模型训练、验证等功能。

作者: chijiang
日期: 2025-06-10
"""

import os
import sys
import argparse
import yaml
import json
import shutil
import random
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

class LEDColorTrainer:
    """LED颜色识别训练器"""
    
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
        self.checkpoint_dir = self.output_dir / "checkpoints" / "led_color"
        self.log_dir = self.output_dir / "logs" / "led_color"
        self.result_dir = self.output_dir / "results" / "led_color"
        
        # 创建目录
        for dir_path in [self.checkpoint_dir, self.log_dir, self.result_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def prepare_yolo_dataset(self) -> str:
        """
        准备YOLO格式数据集，包括数据分割
        
        Returns:
            YOLO数据集配置文件路径
        """
        print("🔄 Preparing LED color YOLO dataset...")
        
        # 数据路径
        data_root = self.project_root / "data" / "led_color"
        yolo_root = self.project_root / "data" / "led_color_yolo"
        
        # 创建YOLO格式目录
        yolo_root.mkdir(exist_ok=True)
        (yolo_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_dir = data_root / "images"
        label_dir = data_root / "labels"
        
        image_files = list(image_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        
        # 过滤出有对应标签文件的图像
        valid_images = []
        for img_file in image_files:
            label_file = label_dir / (img_file.stem + '.txt')
            if label_file.exists():
                valid_images.append(img_file)
        
        print(f"📁 Found {len(valid_images)} valid image-label pairs")
        
        # 随机分割数据集
        random.seed(42)
        random.shuffle(valid_images)
        
        # 80% 训练，20% 验证
        split_idx = int(len(valid_images) * 0.8)
        train_images = valid_images[:split_idx]
        val_images = valid_images[split_idx:]
        
        print(f"📊 Train: {len(train_images)} images, Val: {len(val_images)} images")
        
        # 复制文件到相应目录
        for split, images in [('train', train_images), ('val', val_images)]:
            for img_file in tqdm(images, desc=f"Copying {split} data"):
                # 复制图像
                dst_img = yolo_root / "images" / split / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # 复制标签
                label_file = label_dir / (img_file.stem + '.txt')
                dst_label = yolo_root / "labels" / split / (img_file.stem + '.txt')
                shutil.copy2(label_file, dst_label)
        
        # 创建数据集配置文件
        dataset_config = {
            'train': str(yolo_root / "images" / "train"),
            'val': str(yolo_root / "images" / "val"),
            'nc': 3,  # 类别数量：绿色、红色、黄色
            'names': ['green', 'red', 'yellow']  # 类别名称
        }
        
        config_path = yolo_root / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f)
        
        print(f"✅ Dataset configuration saved to: {config_path}")
        return str(config_path)
    
    def train(self) -> str:
        """
        训练LED颜色识别模型
        
        Returns:
            训练完成的模型路径
        """
        print("🚀 Starting LED color detection training...")
        
        # 准备数据集
        dataset_config = self.prepare_yolo_dataset()
        
        # 获取设备
        device = self._get_device()
        
        # 初始化模型
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'yolo11n.pt')  # 使用较小的模型用于LED颜色识别
        
        print(f"📦 Loading model: {model_name}")
        model = YOLO(model_name)
        
        # 训练参数
        train_config = self.config.get('training', {})
        
        # 设置训练参数
        train_args = {
            'data': dataset_config,
            'epochs': train_config.get('epochs', 100),
            'imgsz': train_config.get('imgsz', 640),
            'batch': train_config.get('batch', 16),
            'lr0': train_config.get('lr0', 0.01),
            'weight_decay': train_config.get('weight_decay', 0.0005),
            'warmup_epochs': train_config.get('warmup_epochs', 3),
            'patience': train_config.get('patience', 50),
            'save': True,
            'save_period': train_config.get('save_period', 10),
            'cache': train_config.get('cache', False),
            'device': device,
            'workers': train_config.get('workers', 8),
            'project': str(self.checkpoint_dir),
            'name': 'led_color_model',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': train_config.get('optimizer', 'auto'),
            'verbose': True,
            'seed': train_config.get('seed', 0),
            'deterministic': train_config.get('deterministic', True),
            'single_cls': False,
            'rect': False,
            'cos_lr': train_config.get('cos_lr', False),
            'close_mosaic': train_config.get('close_mosaic', 10),
            'resume': train_config.get('resume', False),
            'amp': train_config.get('amp', True),
            'fraction': train_config.get('fraction', 1.0),
            'profile': False,
            'overlap_mask': train_config.get('overlap_mask', True),
            'mask_ratio': train_config.get('mask_ratio', 4),
            'dropout': train_config.get('dropout', 0.0),
            'val': True,
        }
        
        print("🔧 Training configuration:")
        for key, value in train_args.items():
            if key != 'data':  # 数据路径太长，不打印
                print(f"   {key}: {value}")
        
        # 开始训练
        print("🏃‍♂️ Training started...")
        results = model.train(**train_args)
        
        # 获取最佳模型路径
        best_model_path = str(self.checkpoint_dir / "led_color_model" / "weights" / "best.pt")
        
        print(f"✅ Training completed! Best model saved to: {best_model_path}")
        
        # 保存训练结果摘要
        self._save_training_summary(results, best_model_path)
        
        return best_model_path
    
    def _save_training_summary(self, results, model_path: str):
        """保存训练结果摘要"""
        summary = {
            'model_path': model_path,
            'training_completed': True,
            'config': self.config,
            'results_summary': {
                'metrics': getattr(results, 'results_dict', {}) if hasattr(results, 'results_dict') else {}
            }
        }
        
        summary_path = self.result_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Training summary saved to: {summary_path}")
    
    def evaluate(self, model_path: str) -> Dict:
        """
        评估LED颜色识别模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            评估结果字典
        """
        print("📊 Evaluating LED color detection model...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 准备数据集配置
        dataset_config = self.prepare_yolo_dataset()
        
        # 运行验证
        device = self._get_device()
        results = model.val(
            data=dataset_config,
            device=device,
            imgsz=self.config.get('training', {}).get('imgsz', 640),
            batch=1,
            conf=0.25,
            iou=0.6,
            max_det=300,
            split='val',
            save_json=True,
            save_hybrid=False,
            plots=True,
            verbose=True
        )
        
        # 保存评估结果
        eval_results = {
            'mAP50': float(results.box.map50) if hasattr(results, 'box') else 0.0,
            'mAP50_95': float(results.box.map) if hasattr(results, 'box') else 0.0,
            'precision': float(results.box.mp) if hasattr(results, 'box') else 0.0,
            'recall': float(results.box.mr) if hasattr(results, 'box') else 0.0,
            'model_path': model_path
        }
        
        # 按类别的结果
        if hasattr(results, 'box') and hasattr(results.box, 'maps'):
            class_names = ['green', 'red', 'yellow']
            for i, class_name in enumerate(class_names):
                if i < len(results.box.maps):
                    eval_results[f'mAP50_{class_name}'] = float(results.box.maps[i])
        
        eval_path = self.result_dir / "evaluation_results.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        print("📈 Evaluation Results:")
        print(f"   mAP@0.5: {eval_results['mAP50']:.4f}")
        print(f"   mAP@0.5:0.95: {eval_results['mAP50_95']:.4f}")
        print(f"   Precision: {eval_results['precision']:.4f}")
        print(f"   Recall: {eval_results['recall']:.4f}")
        
        return eval_results
    
    def visualize_predictions(self, model_path: str, num_samples: int = 10):
        """
        可视化LED颜色识别预测结果
        
        Args:
            model_path: 模型路径
            num_samples: 可视化样本数量
        """
        print(f"🎨 Visualizing {num_samples} LED color predictions...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 获取验证数据
        val_image_dir = self.project_root / "data" / "led_color_yolo" / "images" / "val"
        val_images = list(val_image_dir.glob("*"))[:num_samples]
        
        # 创建可视化目录
        viz_dir = self.result_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        class_names = ['green', 'red', 'yellow']
        class_colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255)]  # 绿、红、黄
        
        for i, img_path in enumerate(val_images):
            # 预测
            results = model.predict(
                source=str(img_path),
                conf=0.25,
                iou=0.6,
                device=self._get_device()
            )
            
            # 读取原始图像
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 绘制预测结果
            annotator = Annotator(img_rgb, line_width=2)
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    # 获取边界框坐标和置信度
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # 绘制边界框和标签
                    label = f"{class_names[cls]} {conf:.2f}"
                    color = class_colors[cls]
                    
                    annotator.box_label(
                        box=[x1, y1, x2, y2],
                        label=label,
                        color=color
                    )
            
            # 保存可视化结果
            output_path = viz_dir / f"prediction_{i+1}_{img_path.stem}.jpg"
            img_bgr = cv2.cvtColor(annotator.result(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), img_bgr)
        
        print(f"🖼️  Visualizations saved to: {viz_dir}")
    
    def export_model(self, model_path: str, formats: List[str] = ['onnx']):
        """
        导出LED颜色识别模型到不同格式
        
        Args:
            model_path: 模型路径
            formats: 导出格式列表
        """
        print(f"📤 Exporting LED color model to formats: {formats}")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 导出模型
        for format_type in formats:
            try:
                export_path = model.export(
                    format=format_type,
                    imgsz=self.config.get('training', {}).get('imgsz', 640),
                    device=self._get_device()
                )
                print(f"✅ Model exported to {format_type}: {export_path}")
            except Exception as e:
                print(f"❌ Failed to export to {format_type}: {e}")

def create_default_config() -> Dict:
    """创建默认配置"""
    return {
        'model': {
            'name': 'yolo11n.pt',  # 使用nano版本，适合LED颜色识别
            'pretrained': True
        },
        'training': {
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'patience': 50,
            'save_period': 10,
            'cache': False,
            'workers': 8,
            'optimizer': 'auto',
            'seed': 0,
            'deterministic': True,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0
        },
        'device': 'auto',
        'data': {
            'augmentation': True,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0
        }
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LED颜色识别模型训练')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'export', 'visualize'],
                       help='运行模式')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径（用于eval/export/visualize模式）')
    parser.add_argument('--export-formats', nargs='+', default=['onnx'],
                       help='导出格式')
    parser.add_argument('--num-viz', type=int, default=10,
                       help='可视化样本数量')
    
    args = parser.parse_args()
    
    # 创建配置文件如果不存在
    if args.config is None:
        config_path = Path(__file__).parent.parent / "config" / "led_color_config.yaml"
        config_path.parent.mkdir(exist_ok=True)
        
        if not config_path.exists():
            default_config = create_default_config()
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
            print(f"📝 Created default config at: {config_path}")
        
        args.config = str(config_path)
    
    # 初始化训练器
    trainer = LEDColorTrainer(args.config)
    
    try:
        if args.mode == 'train':
            print("🚀 Starting LED color detection training...")
            model_path = trainer.train()
            
            # 训练完成后进行评估
            print("\n📊 Evaluating trained model...")
            trainer.evaluate(model_path)
            
            # 生成可视化
            print("\n🎨 Generating visualizations...")
            trainer.visualize_predictions(model_path, num_samples=args.num_viz)
            
        elif args.mode == 'eval':
            if args.model is None:
                print("❌ Please provide --model path for evaluation")
                return
            trainer.evaluate(args.model)
            
        elif args.mode == 'export':
            if args.model is None:
                print("❌ Please provide --model path for export")
                return
            trainer.export_model(args.model, args.export_formats)
            
        elif args.mode == 'visualize':
            if args.model is None:
                print("❌ Please provide --model path for visualization")
                return
            trainer.visualize_predictions(args.model, args.num_viz)
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 