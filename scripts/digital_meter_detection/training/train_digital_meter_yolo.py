#!/usr/bin/env python3
"""
液晶数字表检测 YOLO v10 模型训练脚本

此脚本参考指针表训练脚本架构，提供完整的液晶数字表检测模型训练功能。
包含数据预处理、模型训练、验证、可视化等完整功能。

功能特性：
1. 智能数据集验证和预处理
2. YOLO v10模型训练和优化
3. 完整的评估和可视化
4. 模型导出和部署支持
5. 详细的训练报告生成

作者: chijiang
日期: 2025-06-09
"""

import os
import sys
import yaml
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import shutil
import random
import argparse
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 导入ultralytics库
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    from ultralytics.utils.plotting import Annotator, colors
except ImportError:
    print("❌ 错误：请安装ultralytics库")
    print("运行: pip install ultralytics")
    sys.exit(1)

class DigitalMeterDetectionTrainer:
    """液晶数字表检测训练器"""
    
    def __init__(self, config_path: str = "config/digital_meter_yolo_config.yaml"):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        # 获取项目根目录（智能路径检测）
        self.project_root = self._get_project_root()
        
        # 确保配置文件路径相对于项目根目录
        if not os.path.isabs(config_path):
            self.config_path = self.project_root / config_path
        else:
            self.config_path = Path(config_path)
            
        self.config = self.load_config()
        self.setup_directories()
        
        # 设置随机种子
        self.set_seed(self.config.get('seed', 42))
        
        print(f"🚀 液晶数字表检测训练器已初始化")
        print(f"📂 项目根目录: {self.project_root}")
        print(f"📊 输出目录: {self.output_dir}")
    
    def _get_project_root(self) -> Path:
        """智能获取项目根目录"""
        current_dir = Path.cwd()
        if current_dir.name == "training":
            return current_dir.parent.parent.parent
        elif current_dir.name == "digital_meter_detection":
            return current_dir.parent.parent
        elif current_dir.name == "scripts":
            return current_dir.parent
        else:
            return current_dir
    
    def load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            sys.exit(1)
    
    def setup_directories(self):
        """设置输出目录"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = self.config.get('experiment_name', 'digital_meter_detection')
        
        # 输出目录
        self.output_dir = self.project_root / "outputs" / "digital_meter_detection"
        self.checkpoint_dir = self.output_dir / "checkpoints" / f"{self.experiment_name}_{self.timestamp}"
        self.log_dir = self.output_dir / "logs" / f"{self.experiment_name}_{self.timestamp}"
        self.result_dir = self.output_dir / "results" / f"{self.experiment_name}_{self.timestamp}"
        self.viz_dir = self.result_dir / "visualizations"
        
        # 创建目录
        for dir_path in [self.checkpoint_dir, self.log_dir, self.result_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 目录结构创建完成:")
        print(f"  - 检查点: {self.checkpoint_dir}")
        print(f"  - 日志: {self.log_dir}")
        print(f"  - 结果: {self.result_dir}")
        print(f"  - 可视化: {self.viz_dir}")
    
    def set_seed(self, seed: int):
        """设置随机种子确保可复现性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"🎲 随机种子设置为: {seed}")
    
    def _get_device(self) -> str:
        """智能检测最佳设备"""
        config_device = self.config.get('device', 'auto')
        
        if config_device != 'auto':
            return config_device
        
        # 自动检测设备
        if torch.cuda.is_available():
            device = '0'
            print(f"🔥 使用CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("🍎 使用Apple MPS加速")
        else:
            device = 'cpu'
            print("💻 使用CPU")
        
        print(f"🎯 设备选择: {device}")
        return device
    
    def validate_dataset(self) -> bool:
        """验证数据集格式和完整性"""
        print("\n🔍 验证数据集...")
        
        dataset_path = Path(self.config['dataset']['path'])
        if not dataset_path.is_absolute():
            dataset_path = self.project_root / dataset_path
        
        # 检查基本目录结构
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        
        if not images_dir.exists():
            print(f"❌ 图像目录不存在: {images_dir}")
            return False
            
        if not labels_dir.exists():
            print(f"❌ 标签目录不存在: {labels_dir}")
            return False
        
        # 检查图像和标签文件
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt"))
        
        print(f"📸 找到图像文件: {len(image_files)} 个")
        print(f"🏷️  找到标签文件: {len(label_files)} 个")
        
        if len(image_files) == 0:
            print("❌ 未找到图像文件")
            return False
        
        # 检查图像和标签文件的对应关系
        missing_labels = []
        invalid_labels = []
        total_annotations = 0
        
        for img_file in tqdm(image_files, desc="验证文件对应关系"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                missing_labels.append(img_file.name)
                continue
            
            # 验证标签格式
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        invalid_labels.append(f"{label_file.name}:{line_num}")
                        continue
                    
                    # 检查类别ID和坐标
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        
                        if class_id != 0:  # 液晶表只有一个类别
                            invalid_labels.append(f"{label_file.name}:{line_num} - 错误类别ID: {class_id}")
                            continue
                        
                        for coord in coords:
                            if not (0 <= coord <= 1):
                                invalid_labels.append(f"{label_file.name}:{line_num} - 坐标超出范围")
                                break
                        else:
                            total_annotations += 1
                            
                    except ValueError:
                        invalid_labels.append(f"{label_file.name}:{line_num} - 格式错误")
                        
            except Exception as e:
                invalid_labels.append(f"{label_file.name} - 读取错误: {e}")
        
        # 输出验证结果
        print(f"\n📊 数据集验证结果:")
        print(f"  ✅ 有效图像: {len(image_files) - len(missing_labels)} 个")
        print(f"  ✅ 有效标注: {total_annotations} 个")
        
        if missing_labels:
            print(f"  ⚠️  缺少标签: {len(missing_labels)} 个")
        
        if invalid_labels:
            print(f"  ❌ 无效标签: {len(invalid_labels)} 个")
            for error in invalid_labels[:5]:
                print(f"    - {error}")
            if len(invalid_labels) > 5:
                print(f"    ... 还有 {len(invalid_labels) - 5} 个错误")
        
        # 保存验证报告
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'total_labels': len(label_files),
            'valid_annotations': total_annotations,
            'missing_labels': len(missing_labels),
            'invalid_labels': len(invalid_labels),
            'validation_passed': len(invalid_labels) == 0
        }
        
        report_file = self.result_dir / "dataset_validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 验证报告已保存: {report_file}")
        return len(invalid_labels) == 0
    
    def prepare_yolo_dataset(self) -> str:
        """准备YOLO格式数据集配置"""
        print("\n📦 准备YOLO数据集配置...")
        
        dataset_path = Path(self.config['dataset']['path'])
        if not dataset_path.is_absolute():
            dataset_path = self.project_root / dataset_path
        
        # 创建数据集配置文件
        dataset_config = {
            'path': str(dataset_path),
            'train': 'images',
            'val': 'images',  # 使用相同目录，通过split参数分割
            'nc': 1,
            'names': ['digital_meter']
        }
        
        config_file = dataset_path / "dataset.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ YOLO数据集配置创建: {config_file}")
        return str(config_file)
    
    def visualize_dataset(self, num_samples: int = 16):
        """可视化数据集样本"""
        print(f"\n🎨 可视化数据集样本 ({num_samples} 个)...")
        
        dataset_path = Path(self.config['dataset']['path'])
        if not dataset_path.is_absolute():
            dataset_path = self.project_root / dataset_path
        
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        
        # 随机选择样本
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        selected_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        # 设置绘图
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle('液晶数字表数据集样本', fontsize=16, fontweight='bold')
        
        for idx, img_file in enumerate(selected_files):
            if idx >= 16:
                break
                
            row, col = idx // 4, idx % 4
            ax = axes[row, col]
            
            # 加载图像
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # 加载标签
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # 绘制边界框
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, w, h = map(float, parts)
                        
                        # 转换为像素坐标
                        x1 = int((x_center - w/2) * width)
                        y1 = int((y_center - h/2) * height)
                        x2 = int((x_center + w/2) * width)
                        y2 = int((y_center + h/2) * height)
                        
                        # 绘制边界框
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(image, 'digital_meter', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            ax.imshow(image)
            ax.set_title(f'样本 {idx+1}: {img_file.name}', fontsize=10)
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(len(selected_files), 16):
            row, col = idx // 4, idx % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # 保存可视化结果
        viz_file = self.viz_dir / "dataset_samples.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 数据集可视化保存: {viz_file}")
    
    def train(self) -> str:
        """训练YOLO v10模型"""
        print("\n🚀 开始YOLO v10模型训练...")
        
        # 验证数据集
        if not self.validate_dataset():
            print("❌ 数据集验证失败，停止训练")
            return None
        
        # 可视化数据集
        self.visualize_dataset()
        
        # 准备数据集配置
        dataset_config_path = self.prepare_yolo_dataset()
        
        # 初始化模型
        model_name = self.config.get('model', 'yolov10n.pt')
        print(f"🤖 初始化模型: {model_name}")
        model = YOLO(model_name)
        
        # 训练参数
        train_args = self._get_training_args(dataset_config_path)
        
        print(f"📋 训练配置:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        
        # 开始训练
        try:
            results = model.train(**train_args)
            
            # 获取最佳模型路径
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"
            
            print(f"✅ 训练完成!")
            print(f"📁 最佳模型: {best_model_path}")
            
            # 复制模型到结果目录
            shutil.copy2(best_model_path, self.result_dir / "best_model.pt")
            
            return str(best_model_path)
            
        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")
            raise
    
    def _get_training_args(self, dataset_config_path: str) -> Dict:
        """获取训练参数"""
        return {
            'data': dataset_config_path,
            'epochs': self.config.get('epochs', 200),
            'imgsz': self.config.get('image_size', 640),
            'batch': self.config.get('batch_size', 16),
            'lr0': self.config.get('learning_rate', 0.01),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'momentum': self.config.get('momentum', 0.937),
            'patience': self.config.get('patience', 50),
            'save_period': self.config.get('save_period', 10),
            'project': str(self.checkpoint_dir.parent),
            'name': self.checkpoint_dir.name,
            'exist_ok': True,
            'pretrained': True,
            'optimize': True,
            'verbose': True,
            'seed': self.config.get('seed', 42),
            'deterministic': True,
            'single_cls': True,
            'device': self._get_device(),
            'workers': self.config.get('workers', 8),
            'cos_lr': self.config.get('cos_lr', True),
            'close_mosaic': self.config.get('close_mosaic', 10),
            'amp': self.config.get('amp', True),
            'split': self.config.get('train_split', 0.8),
            **self._get_augmentation_args()
        }
    
    def _get_augmentation_args(self) -> Dict:
        """获取数据增强参数"""
        augment = self.config.get('augmentation', {})
        return {
            'hsv_h': augment.get('hsv_h', 0.01),
            'hsv_s': augment.get('hsv_s', 0.6),
            'hsv_v': augment.get('hsv_v', 0.5),
            'degrees': augment.get('degrees', 10.0),
            'translate': augment.get('translate', 0.2),
            'scale': augment.get('scale', 0.6),
            'shear': augment.get('shear', 2.0),
            'perspective': augment.get('perspective', 0.0002),
            'flipud': augment.get('flipud', 0.0),
            'fliplr': augment.get('fliplr', 0.5),
            'mosaic': augment.get('mosaic', 1.0),
            'mixup': augment.get('mixup', 0.1),
            'copy_paste': augment.get('copy_paste', 0.1),
        }
    
    def evaluate(self, model_path: str) -> Dict:
        """评估训练好的模型"""
        print("\n📊 评估模型性能...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 准备验证数据
        dataset_config_path = self.prepare_yolo_dataset()
        
        # 运行验证
        results = model.val(
            data=dataset_config_path,
            imgsz=self.config.get('image_size', 640),
            batch=self.config.get('batch_size', 16),
            conf=0.001,
            iou=0.6,
            max_det=300,
            device=self._get_device(),
            save_json=True,
            save_hybrid=True,
                plots=True,
            verbose=True,
            split='val'
        )
        
        # 提取评估指标
        metrics = {
            'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            'mAP50-95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
        }
        
        # 计算F1分数
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        print(f"📈 评估结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 保存评估结果
        results_file = self.result_dir / "evaluation_metrics.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        return metrics
    
    def visualize_predictions(self, model_path: str, num_samples: int = 16):
        """可视化模型预测结果"""
        print(f"\n🎨 可视化预测结果 ({num_samples} 个样本)...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 获取测试图像
        dataset_path = Path(self.config['dataset']['path'])
        if not dataset_path.is_absolute():
            dataset_path = self.project_root / dataset_path
        
        images_dir = dataset_path / "images"
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        selected_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        # 设置绘图
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle('液晶数字表检测预测结果', fontsize=16, fontweight='bold')
        
        for idx, img_file in enumerate(selected_files):
            if idx >= 16:
                break
                
            row, col = idx // 4, idx % 4
            ax = axes[row, col]
            
            # 预测
            results = model(str(img_file), conf=0.25, verbose=False)
            
            # 绘制结果
            annotated_img = results[0].plot(conf=True, line_width=2, font_size=12)
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            ax.imshow(annotated_img)
            ax.set_title(f'预测 {idx+1}: {img_file.name}', fontsize=10)
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(len(selected_files), 16):
            row, col = idx // 4, idx % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # 保存可视化结果
        viz_file = self.viz_dir / "prediction_results.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 预测可视化保存: {viz_file}")
    
    def plot_training_curves(self, model_path: str):
        """绘制训练曲线"""
        print("\n📈 绘制训练曲线...")
        
        # 查找训练结果目录
        model_dir = Path(model_path).parent.parent
        results_csv = model_dir / "results.csv"
        
        if not results_csv.exists():
            print(f"⚠️  未找到训练结果文件: {results_csv}")
            return
        
        # 读取训练结果
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # 清理列名
        
        # 绘制训练曲线
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('训练曲线分析', fontsize=16, fontweight='bold')
        
        # 损失曲线
        if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='训练损失', color='blue')
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='验证损失', color='red')
            axes[0, 0].set_title('边界框损失')
            axes[0, 0].set_xlabel('轮次')
            axes[0, 0].set_ylabel('损失值')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 精度曲线
        if 'metrics/precision(B)' in df.columns:
            axes[0, 1].plot(df['epoch'], df['metrics/precision(B)'], label='精度', color='green')
            axes[0, 1].set_title('精度曲线')
            axes[0, 1].set_xlabel('轮次')
            axes[0, 1].set_ylabel('精度')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 召回率曲线
        if 'metrics/recall(B)' in df.columns:
            axes[0, 2].plot(df['epoch'], df['metrics/recall(B)'], label='召回率', color='orange')
            axes[0, 2].set_title('召回率曲线')
            axes[0, 2].set_xlabel('轮次')
            axes[0, 2].set_ylabel('召回率')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # mAP曲线
        if 'metrics/mAP50(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='purple')
            axes[1, 0].set_title('mAP@0.5曲线')
            axes[1, 0].set_xlabel('轮次')
            axes[1, 0].set_ylabel('mAP@0.5')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'metrics/mAP50-95(B)' in df.columns:
            axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='brown')
            axes[1, 1].set_title('mAP@0.5:0.95曲线')
            axes[1, 1].set_xlabel('轮次')
            axes[1, 1].set_ylabel('mAP@0.5:0.95')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 学习率曲线
        if 'lr/pg0' in df.columns:
            axes[1, 2].plot(df['epoch'], df['lr/pg0'], label='学习率', color='red')
            axes[1, 2].set_title('学习率曲线')
            axes[1, 2].set_xlabel('轮次')
            axes[1, 2].set_ylabel('学习率')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存训练曲线
        curves_file = self.viz_dir / "training_curves.png"
        plt.savefig(curves_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练曲线保存: {curves_file}")
    
    def export_model(self, model_path: str, formats: List[str] = ['onnx']):
        """导出模型到不同格式"""
        print(f"\n📦 导出模型...")
        
        model = YOLO(model_path)
        
        export_dir = self.result_dir / "exported_models"
        export_dir.mkdir(exist_ok=True)
        
        for format_name in formats:
            try:
                print(f"  导出到 {format_name.upper()}...")
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
                
                # 复制到结果目录
                if exported_model and Path(exported_model).exists():
                    export_file = export_dir / Path(exported_model).name
                    shutil.copy2(exported_model, export_file)
                    print(f"  ✅ {format_name.upper()} 模型导出: {export_file}")
                
            except Exception as e:
                    print(f"  ❌ {format_name} 导出失败: {str(e)}")
    
    def generate_report(self, model_path: str, metrics: Dict):
        """生成训练报告"""
        print("\n📄 生成训练报告...")
        
        report = {
            'experiment_info': {
                'name': self.experiment_name,
                'timestamp': self.timestamp,
                'model_path': str(model_path),
                'config_path': str(self.config_path)
            },
            'dataset_info': {
                'path': self.config['dataset']['path'],
                'classes': ['digital_meter'],
                'num_classes': 1
            },
            'training_config': {
                'model': self.config.get('model', 'yolov10n.pt'),
                'epochs': self.config.get('epochs', 200),
                'batch_size': self.config.get('batch_size', 16),
                'image_size': self.config.get('image_size', 640),
                'device': self._get_device()
            },
            'performance_metrics': metrics,
            'files': {
                'best_model': str(self.result_dir / "best_model.pt"),
                'training_curves': str(self.viz_dir / "training_curves.png"),
                'predictions': str(self.viz_dir / "prediction_results.png"),
                'dataset_samples': str(self.viz_dir / "dataset_samples.png")
            }
        }
        
        # 保存JSON报告
        report_file = self.result_dir / "training_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
        
        print(f"✅ 训练报告生成完成: {report_file}")
    
    def _generate_markdown_report(self, report: Dict):
        """生成Markdown格式报告"""
        md_content = f"""# 液晶数字表检测训练报告

## 实验信息
- **实验名称**: {report['experiment_info']['name']}
- **时间戳**: {report['experiment_info']['timestamp']}
- **模型路径**: {report['experiment_info']['model_path']}

## 数据集信息
- **数据集路径**: {report['dataset_info']['path']}
- **类别数量**: {report['dataset_info']['num_classes']}
- **类别**: {', '.join(report['dataset_info']['classes'])}

## 训练配置
- **基础模型**: {report['training_config']['model']}
- **训练轮数**: {report['training_config']['epochs']}
- **批次大小**: {report['training_config']['batch_size']}
- **图像尺寸**: {report['training_config']['image_size']}
- **设备**: {report['training_config']['device']}

## 性能指标
| 指标 | 数值 |
|------|------|
| mAP@0.5 | {report['performance_metrics']['mAP50']:.4f} |
| mAP@0.5:0.95 | {report['performance_metrics']['mAP50-95']:.4f} |
| 精度 | {report['performance_metrics']['precision']:.4f} |
| 召回率 | {report['performance_metrics']['recall']:.4f} |
| F1分数 | {report['performance_metrics']['f1_score']:.4f} |

## 生成文件
- 最佳模型: `best_model.pt`
- 训练曲线: `visualizations/training_curves.png`
- 预测结果: `visualizations/prediction_results.png`
- 数据集样本: `visualizations/dataset_samples.png`

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        md_file = self.result_dir / "training_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def run_complete_training(self):
        """运行完整的训练流程"""
        print("🎯 开始完整训练流程...")
        
        try:
            # 1. 训练模型
            best_model_path = self.train()
            if not best_model_path:
                return
            
            # 2. 评估模型
            metrics = self.evaluate(best_model_path)
            
            # 3. 可视化预测结果
            self.visualize_predictions(best_model_path)
            
            # 4. 绘制训练曲线
            self.plot_training_curves(best_model_path)
            
            # 5. 导出模型
            self.export_model(best_model_path, ['onnx', 'torchscript'])
            
            # 6. 生成报告
            self.generate_report(best_model_path, metrics)
            
            print(f"\n🎉 训练流程完成!")
            print(f"📁 结果目录: {self.result_dir}")
            print(f"📊 最终mAP@0.5: {metrics['mAP50']:.4f}")
            
        except Exception as e:
            print(f"❌ 训练流程失败: {str(e)}")
            raise


def create_default_config() -> Dict:
    """创建默认配置"""
    return {
        'experiment_name': 'digital_meter_detection',
        'model': 'yolov10n.pt',
        'dataset': {
            'path': 'data/digital_meters'
        },
        'epochs': 200,
        'batch_size': 16,
        'image_size': 640,
        'learning_rate': 0.01,
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'patience': 50,
        'save_period': 10,
        'device': 'auto',
        'workers': 8,
        'seed': 42,
        'cos_lr': True,
        'close_mosaic': 10,
        'amp': True,
        'train_split': 0.8,
        'augmentation': {
            'hsv_h': 0.01,
            'hsv_s': 0.6,
            'hsv_v': 0.5,
            'degrees': 10.0,
            'translate': 0.2,
            'scale': 0.6,
            'shear': 2.0,
            'perspective': 0.0002,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='液晶数字表检测YOLO v10训练脚本')
    parser.add_argument('--config', type=str, default='config/digital_meter_yolo_config.yaml',
                       help='配置文件路径（相对于项目根目录）')
    parser.add_argument('--create-config', action='store_true',
                       help='创建默认配置文件')
    parser.add_argument('--validate-only', action='store_true',
                       help='仅验证数据集')
    parser.add_argument('--eval-only', action='store_true',
                       help='仅评估模型')
    parser.add_argument('--model-path', type=str,
                       help='模型路径（用于评估）')
    parser.add_argument('--export', action='store_true',
                       help='导出模型')
    parser.add_argument('--formats', nargs='+', default=['onnx'],
                       help='导出格式')
    
    args = parser.parse_args()
    
    # 创建默认配置
    if args.create_config:
        config_path = Path(args.config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = create_default_config()
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 默认配置文件创建: {config_path}")
        return
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"❌ 配置文件不存在: {args.config}")
        print("💡 使用 --create-config 创建默认配置文件")
        return
    
    # 初始化训练器
    trainer = DigitalMeterDetectionTrainer(args.config)
    
    try:
        if args.validate_only:
            # 仅验证数据集
                trainer.validate_dataset()
                trainer.visualize_dataset()
                
        elif args.eval_only:
            # 仅评估模型
            if not args.model_path:
                print("❌ 评估模式需要提供模型路径")
                return
            
            metrics = trainer.evaluate(args.model_path)
            trainer.visualize_predictions(args.model_path)
            trainer.plot_training_curves(args.model_path)
            
            if args.export:
                trainer.export_model(args.model_path, args.formats)
        
            else:
        # 运行完整训练流程
                trainer.run_complete_training()
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 