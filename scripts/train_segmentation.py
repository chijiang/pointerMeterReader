#!/usr/bin/env python3
"""
DeepLabV3+ 分割训练脚本
用于训练指针表盘的语义分割模型：背景、指针、刻度三分类

使用方法:
python scripts/train_segmentation.py --config config/segmentation_config.yaml
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models.segmentation as segmentation

from tqdm import tqdm
import cv2


class SegmentationDataset(Dataset):
    """分割数据集"""
    
    def __init__(self, root_dir, image_dir, mask_dir, split_file, transform=None, mask_transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_dir
        self.mask_dir = self.root_dir / mask_dir
        self.image_transform = transform
        self.mask_transform = mask_transform
        
        # 读取数据分割文件
        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.image_names = [line.strip() for line in f.readlines()]
        else:
            # 如果没有分割文件，使用所有图像
            image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
            self.image_names = [f.stem for f in image_files]
        
        # 验证数据存在性
        valid_names = []
        for name in self.image_names:
            image_path = self._find_image_file(name)
            mask_path = self._find_mask_file(name)
            if image_path and mask_path:
                valid_names.append(name)
        
        self.image_names = valid_names
        print(f"有效样本数: {len(self.image_names)}")
    
    def _find_image_file(self, name):
        """查找图像文件"""
        for ext in ['.jpg', '.jpeg', '.png']:
            path = self.image_dir / f"{name}{ext}"
            if path.exists():
                return path
        return None
    
    def _find_mask_file(self, name):
        """查找mask文件"""
        path = self.mask_dir / f"{name}.png"
        return path if path.exists() else None
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        name = self.image_names[idx]
        
        # 加载图像
        image_path = self._find_image_file(name)
        image = Image.open(image_path).convert('RGB')
        
        # 加载mask
        mask_path = self._find_mask_file(name)
        mask = Image.open(mask_path)
        
        # 转换为numpy数组
        mask_array = np.array(mask)
        
        # 确保mask值在有效范围内
        mask_array = np.clip(mask_array, 0, 2)
        mask = Image.fromarray(mask_array)
        
        # 同步变换图像和mask
        if self.image_transform and self.mask_transform:
            # 获取相同的随机种子用于同步变换
            seed = np.random.randint(2147483647)
            
            # 变换图像
            np.random.seed(seed)
            torch.manual_seed(seed)
            image = self.image_transform(image)
            
            # 用相同的随机种子变换mask
            np.random.seed(seed)
            torch.manual_seed(seed)
            mask_transformed = self.mask_transform(mask)
            mask = torch.from_numpy(np.array(mask_transformed)).long()
            
        elif self.image_transform:
            image = self.image_transform(image)
            mask = torch.from_numpy(np.array(mask)).long()
        else:
            # 不应用变换，直接转换为tensor
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask


class SegmentationTransform:
    """分割数据变换"""
    
    def __init__(self, config):
        self.config = config
        
    def get_train_transform(self):
        """训练数据变换"""
        aug_config = self.config['data']['augmentation']
        
        transform_list = [
            transforms.Resize(aug_config['resize']),
        ]
        
        # 随机裁剪
        if 'random_crop' in aug_config:
            transform_list.append(transforms.RandomCrop(aug_config['random_crop']))
        
        # 随机翻转
        if aug_config.get('horizontal_flip', 0) > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip']))
        
        # 颜色变换
        if 'color_jitter' in aug_config:
            jitter = aug_config['color_jitter']
            transform_list.append(transforms.ColorJitter(
                brightness=jitter.get('brightness', 0),
                contrast=jitter.get('contrast', 0),
                saturation=jitter.get('saturation', 0),
                hue=jitter.get('hue', 0)
            ))
        
        # 转换为张量
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    def get_val_transform(self):
        """验证数据变换"""
        aug_config = self.config['data']['augmentation']
        
        # 验证时使用center crop确保尺寸一致
        target_size = aug_config.get('random_crop', aug_config['resize'])
        
        return transforms.Compose([
            transforms.Resize(aug_config['resize']),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_mask_transform(self, is_training=True):
        """mask变换"""
        aug_config = self.config['data']['augmentation']
        
        transform_list = [
            transforms.Resize(aug_config['resize'], interpolation=Image.NEAREST),
        ]
        
        if 'random_crop' in aug_config:
            if is_training:
                transform_list.append(transforms.RandomCrop(aug_config['random_crop']))
            else:
                # 验证时使用center crop
                transform_list.append(transforms.CenterCrop(aug_config['random_crop']))
        
        return transforms.Compose(transform_list)


class SegmentationMetrics:
    """分割评估指标"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        """更新混淆矩阵"""
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # 只考虑有效像素
        valid_mask = (target >= 0) & (target < self.num_classes)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # 更新混淆矩阵
        for i in range(len(pred)):
            self.confusion_matrix[target[i], pred[i]] += 1
    
    def get_pixel_accuracy(self):
        """像素准确率"""
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc
    
    def get_class_iou(self):
        """各类别IoU"""
        iou_per_class = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            if tp + fp + fn == 0:
                iou = 0.0
            else:
                iou = tp / (tp + fp + fn)
            iou_per_class.append(iou)
        
        return np.array(iou_per_class)
    
    def get_mean_iou(self):
        """平均IoU"""
        return self.get_class_iou().mean()
    
    def get_dice_score(self):
        """Dice分数"""
        dice_per_class = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            if 2 * tp + fp + fn == 0:
                dice = 0.0
            else:
                dice = 2 * tp / (2 * tp + fp + fn)
            dice_per_class.append(dice)
        
        return np.array(dice_per_class)


class SegmentationTrainer:
    """分割训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        
        # 创建输出目录
        self._create_output_dirs()
        
        # 设置日志
        self._setup_logging()
        
        # 初始化模型
        self.model = self._create_model()
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # 初始化优化器和损失函数
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.scheduler = self._create_scheduler()
        
        # 初始化TensorBoard
        if config['logging']['tensorboard']:
            self.writer = SummaryWriter(config['save']['log_dir'])
        else:
            self.writer = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_miou = 0.0
        self.best_loss = float('inf')
        
    def _get_device(self):
        """获取设备"""
        device_config = self.config['device']
        
        if device_config == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif device_config.startswith('cuda') and torch.cuda.is_available():
            return torch.device(device_config)
        else:
            return torch.device('cpu')
    
    def _create_output_dirs(self):
        """创建输出目录"""
        dirs = [
            self.config['save']['checkpoint_dir'],
            self.config['save']['log_dir'],
            self.config['visualization']['prediction_dir']
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.config['logging']
        
        # 配置日志格式
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[]
        )
        
        # 控制台输出
        if log_config['console']:
            logging.getLogger().addHandler(logging.StreamHandler())
        
        # 文件输出
        if log_config['file']:
            log_file = Path(self.config['save']['log_dir']) / 'training.log'
            logging.getLogger().addHandler(logging.FileHandler(log_file))
    
    def _create_model(self):
        """创建模型"""
        model_config = self.config['model']
        num_classes = model_config['num_classes']
        
        # 创建DeepLabV3+模型
        if model_config['architecture'] == 'deeplabv3_resnet50':
            if model_config['pretrained']:
                # 先加载预训练模型（21类）
                model = segmentation.deeplabv3_resnet50(pretrained=True)
                # 替换分类器为我们的类别数
                model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
                model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            else:
                # 直接创建指定类别数的模型
                model = segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
                
        elif model_config['architecture'] == 'deeplabv3_resnet101':
            if model_config['pretrained']:
                model = segmentation.deeplabv3_resnet101(pretrained=True)
                model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
                model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            else:
                model = segmentation.deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
                
        elif model_config['architecture'] == 'deeplabv3_mobilenet_v3_large':
            if model_config['pretrained']:
                model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
                model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
                model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            else:
                model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(f"不支持的模型架构: {model_config['architecture']}")
        
        model = model.to(self.device)
        logging.info(f"创建模型: {model_config['architecture']}, 设备: {self.device}")
        logging.info(f"输出类别数: {num_classes}, 预训练: {model_config['pretrained']}")
        
        return model
    
    def _create_data_loaders(self):
        """创建数据加载器"""
        data_config = self.config['data']
        
        # 创建数据变换
        transform_manager = SegmentationTransform(self.config)
        train_transform = transform_manager.get_train_transform()
        val_transform = transform_manager.get_val_transform()
        train_mask_transform = transform_manager.get_mask_transform(is_training=True)
        val_mask_transform = transform_manager.get_mask_transform(is_training=False)
        
        # 数据分割文件路径
        split_dir = Path(data_config['root_dir']) / data_config['split_dir']
        train_split = split_dir / 'train.txt' if split_dir.exists() else None
        val_split = split_dir / 'val.txt' if split_dir.exists() else None
        
        # 创建数据集
        train_dataset = SegmentationDataset(
            root_dir=data_config['root_dir'],
            image_dir=data_config['image_dir'],
            mask_dir=data_config['mask_dir'],
            split_file=train_split,
            transform=train_transform,
            mask_transform=train_mask_transform
        )
        
        val_dataset = SegmentationDataset(
            root_dir=data_config['root_dir'],
            image_dir=data_config['image_dir'],
            mask_dir=data_config['mask_dir'],
            split_file=val_split,
            transform=val_transform,
            mask_transform=val_mask_transform
        )
        
        # 如果没有验证集，从训练集中分割
        if len(val_dataset) == 0:
            logging.info("没有找到验证集，从训练集中分割20%作为验证集")
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
        
        # 创建数据加载器
        # MPS设备不支持pin_memory
        pin_memory = data_config['pin_memory'] and self.device.type != 'mps'
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=data_config['shuffle'],
            num_workers=data_config['num_workers'],
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=pin_memory
        )
        
        logging.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _create_optimizer(self):
        """创建优化器"""
        train_config = self.config['training']
        
        # 确保数值参数是正确的类型
        learning_rate = float(train_config['learning_rate'])
        weight_decay = float(train_config['weight_decay'])
        momentum = float(train_config.get('momentum', 0.9))
        
        if train_config['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif train_config['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif train_config['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {train_config['optimizer']}")
        
        logging.info(f"创建优化器: {train_config['optimizer']}, lr={learning_rate}, weight_decay={weight_decay}")
        return optimizer
    
    def _create_criterion(self):
        """创建损失函数"""
        loss_config = self.config['training']['loss']
        
        if loss_config['type'] == 'CrossEntropyLoss':
            class_weights = loss_config.get('class_weights', None)
            if class_weights:
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=loss_config.get('ignore_index', -100)
            )
        else:
            raise ValueError(f"不支持的损失函数: {loss_config['type']}")
        
        return criterion
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['type'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(scheduler_config['T_max']),
                eta_min=float(scheduler_config['eta_min'])
            )
        elif scheduler_config['type'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(scheduler_config.get('step_size', 30)),
                gamma=float(scheduler_config.get('gamma', 0.1))
            )
        elif scheduler_config['type'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=float(scheduler_config.get('factor', 0.5)),
                patience=int(scheduler_config.get('patience', 10))
            )
        else:
            return None
        
        logging.info(f"创建学习率调度器: {scheduler_config['type']}")
        return scheduler
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        metrics = SegmentationMetrics(self.config['model']['num_classes'])
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)['out']
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新指标
            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            metrics.update(pred, targets)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(self.train_loader)
        pixel_acc = metrics.get_pixel_accuracy()
        mean_iou = metrics.get_mean_iou()
        
        return avg_loss, pixel_acc, mean_iou
    
    def validate(self):
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        metrics = SegmentationMetrics(self.config['model']['num_classes'])
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(images)['out']
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # 更新指标
                pred = torch.argmax(outputs, dim=1)
                metrics.update(pred, targets)
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(self.val_loader)
        pixel_acc = metrics.get_pixel_accuracy()
        mean_iou = metrics.get_mean_iou()
        class_iou = metrics.get_class_iou()
        dice_scores = metrics.get_dice_score()
        
        return avg_loss, pixel_acc, mean_iou, class_iou, dice_scores
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config
        }
        
        # 保存常规检查点
        checkpoint_path = Path(self.config['save']['checkpoint_dir']) / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = Path(self.config['save']['best_model_path'])
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_path)
            logging.info(f"保存最佳模型: {best_path}")
    
    def visualize_predictions(self, num_samples=None):
        """可视化预测结果"""
        if not self.config['visualization']['save_predictions']:
            return
        
        num_samples = num_samples or self.config['visualization']['num_samples']
        output_dir = Path(self.config['visualization']['prediction_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        class_colors = self.config['visualization']['class_colors']
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                
                images = images.to(self.device)
                outputs = self.model(images)['out']
                predictions = torch.argmax(outputs, dim=1)
                
                # 处理第一个样本
                image = images[0].cpu()
                target = targets[0].cpu().numpy()
                pred = predictions[0].cpu().numpy()
                
                # 反归一化图像
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image.permute(1, 2, 0).numpy()
                image = std * image + mean
                image = np.clip(image, 0, 1)
                
                # 创建可视化
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 原图
                axes[0].imshow(image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # 真实标签
                target_colored = self._colorize_mask(target, class_colors)
                axes[1].imshow(target_colored)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # 预测结果
                pred_colored = self._colorize_mask(pred, class_colors)
                axes[2].imshow(pred_colored)
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'prediction_epoch_{self.current_epoch}_sample_{i}.png')
                plt.close()
    
    def _colorize_mask(self, mask, class_colors):
        """给mask着色"""
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in class_colors.items():
            colored_mask[mask == class_id] = color
        
        return colored_mask
    
    def train(self):
        """完整训练流程"""
        logging.info("开始分割模型训练")
        
        early_stopping_config = self.config['training']['early_stopping']
        patience = early_stopping_config['patience']
        min_delta = early_stopping_config['min_delta']
        no_improve_epochs = 0
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_acc, train_miou = self.train_epoch()
            
            # 验证
            if epoch % self.config['validation']['interval'] == 0:
                val_loss, val_acc, val_miou, class_iou, dice_scores = self.validate()
                
                # 记录日志
                logging.info(f'Epoch {epoch + 1}/{self.config["training"]["epochs"]}')
                logging.info(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, mIoU: {train_miou:.4f}')
                logging.info(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {val_miou:.4f}')
                
                # TensorBoard记录
                if self.writer:
                    self.writer.add_scalar('Loss/Train', train_loss, epoch)
                    self.writer.add_scalar('Loss/Val', val_loss, epoch)
                    self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                    self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
                    self.writer.add_scalar('mIoU/Train', train_miou, epoch)
                    self.writer.add_scalar('mIoU/Val', val_miou, epoch)
                    
                    # 各类别IoU
                    class_names = self.config['classes']['names']
                    for i, (name, iou) in enumerate(zip(class_names, class_iou)):
                        self.writer.add_scalar(f'IoU/{name}', iou, epoch)
                
                # 检查是否是最佳模型
                is_best = val_miou > self.best_miou + min_delta
                if is_best:
                    self.best_miou = val_miou
                    no_improve_epochs = 0
                    self.save_checkpoint(is_best=True)
                else:
                    no_improve_epochs += 1
                
                # 早停检查
                if no_improve_epochs >= patience:
                    logging.info(f"早停触发，连续{patience}个epoch无改善")
                    break
            
            # 保存检查点
            if epoch % self.config['save']['save_interval'] == 0:
                self.save_checkpoint()
            
            # 可视化预测结果
            if epoch % 10 == 0:
                self.visualize_predictions()
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
        
        logging.info("训练完成!")
        
        # 导出模型
        self.export_models()
        
        if self.writer:
            self.writer.close()
    
    def export_models(self):
        """导出训练好的模型"""
        export_config = self.config.get('export', {})
        if not export_config:
            return
        
        output_dir = Path(export_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载最佳模型
        best_model_path = self.config['save']['best_model_path']
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"加载最佳模型用于导出: {best_model_path}")
        
        self.model.eval()
        
        # 创建示例输入
        example_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        formats = export_config.get('formats', [])
        
        # 导出ONNX
        if 'onnx' in formats:
            onnx_path = output_dir / 'segmentation_model.onnx'
            onnx_config = export_config.get('onnx', {})
            
            try:
                torch.onnx.export(
                    self.model,
                    example_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=onnx_config.get('opset_version', 11),
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=onnx_config.get('dynamic_axes', {
                        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                    })
                )
                logging.info(f"ONNX模型导出成功: {onnx_path}")
            except Exception as e:
                logging.error(f"ONNX导出失败: {e}")
        
        # 导出TorchScript
        if 'torchscript' in formats:
            script_path = output_dir / 'segmentation_model.pt'
            try:
                traced_model = torch.jit.trace(self.model, example_input)
                traced_model.save(str(script_path))
                logging.info(f"TorchScript模型导出成功: {script_path}")
            except Exception as e:
                logging.error(f"TorchScript导出失败: {e}")
        
        # 导出纯PyTorch模型（只保存模型结构和权重）
        pytorch_path = output_dir / 'segmentation_model_pytorch.pth'
        try:
            # 保存完整的模型信息
            model_info = {
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'architecture': self.config['model']['architecture'],
                    'num_classes': self.config['model']['num_classes'],
                    'pretrained': self.config['model']['pretrained']
                },
                'input_size': [224, 224],
                'class_names': self.config['classes']['names']
            }
            torch.save(model_info, pytorch_path)
            logging.info(f"PyTorch模型导出成功: {pytorch_path}")
        except Exception as e:
            logging.error(f"PyTorch模型导出失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="DeepLabV3+ 分割训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/segmentation_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="只导出模型，不进行训练"
    )
    parser.add_argument(
        "--export-formats",
        nargs='+',
        default=['onnx', 'torchscript'],
        help="导出格式列表"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建训练器
    trainer = SegmentationTrainer(config)
    
    if args.export_only:
        # 只导出模型
        print("🚀 开始导出模型...")
        
        # 临时修改配置中的导出格式
        if 'export' not in config:
            config['export'] = {}
        config['export']['formats'] = args.export_formats
        trainer.config = config
        
        trainer.export_models()
        print("✅ 模型导出完成!")
    else:
        # 正常训练流程
        trainer.train()


if __name__ == "__main__":
    main()