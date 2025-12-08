"""
Classification Trainer - Image classification training with GPU optimization

Supports binary and multi-class classification using pretrained models:
- EfficientNet-B0 (default)
- ResNet-18/50
- MobileNetV3

Features:
- ImageFolder dataset loading
- Class weight balancing for imbalanced datasets
- Mixed precision training
- Early stopping and checkpoint management
- ONNX export
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .base_trainer import BaseTrainer
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ClassificationDataset(Dataset):
    """
    Classification dataset with custom transforms.
    Wraps ImageFolder with additional functionality.
    """

    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose = None,
        class_names: List[str] = None,
    ):
        """
        Initialize classification dataset.

        Args:
            root_dir: Root directory with class subdirectories
            transform: Image transforms
            class_names: Optional list of class names (auto-detected if None)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Use ImageFolder for directory structure
        self.dataset = ImageFolder(root=str(self.root_dir))

        # Get class names
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = self.dataset.classes

        self.class_to_idx = self.dataset.class_to_idx
        self.samples = self.dataset.samples

        logger.info(f"Loaded {len(self)} samples from {root_dir}")
        logger.info(f"Classes: {self.class_names}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples per class."""
        counts = {}
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            counts[class_name] = sum(1 for _, label in self.samples if label == class_idx)
        return counts


class ClassificationTrainer(BaseTrainer):
    """
    Image Classification Trainer.

    Features:
    - EfficientNet/ResNet/MobileNet support
    - ImageFolder-style dataset loading
    - Mixed precision training
    - Class weight balancing
    - ONNX export
    """

    def __init__(self, config: ConfigManager):
        """
        Initialize classification trainer.

        Args:
            config: ConfigManager instance with classification config
        """
        super().__init__(config)

        # Classification-specific settings
        self.num_classes = self.cfg.num_classes
        self.class_names = self.cfg.class_names
        self.image_size = self.cfg.image_size

        # Class weights for imbalanced data
        self.class_weights = None

    def _create_model(self) -> nn.Module:
        """Create classification model."""
        model_name = self.cfg.model_name.lower()
        pretrained = self.cfg.pretrained
        num_classes = self.num_classes

        logger.info(f"Creating model: {model_name} (pretrained={pretrained})")

        if 'efficientnet' in model_name:
            if model_name == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.efficientnet_b0(weights=weights)
            elif model_name == 'efficientnet_b1':
                weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.efficientnet_b1(weights=weights)
            else:
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.efficientnet_b0(weights=weights)

            # Replace classifier
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=self.cfg.dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )

        elif 'resnet' in model_name:
            if model_name == 'resnet18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.resnet18(weights=weights)
            elif model_name == 'resnet50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.resnet50(weights=weights)
            else:
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.resnet18(weights=weights)

            # Replace fc layer
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        elif 'mobilenet' in model_name:
            if 'small' in model_name:
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.mobilenet_v3_small(weights=weights)
            else:
                weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.mobilenet_v3_large(weights=weights)

            # Replace classifier
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)

        else:
            # Default to EfficientNet-B0
            logger.warning(f"Unknown model {model_name}, using efficientnet_b0")
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=self.cfg.dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        return model

    def _create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create train and val transforms."""
        # Get normalization stats
        mean = self.cfg.normalize_mean
        std = self.cfg.normalize_std

        # Train transforms with augmentation
        train_transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]

        # Add augmentations based on config
        aug_cfg = self.cfg.augmentation

        if aug_cfg.get('horizontal_flip', 0) > 0:
            train_transform_list.append(
                transforms.RandomHorizontalFlip(p=aug_cfg['horizontal_flip'])
            )

        if aug_cfg.get('vertical_flip', 0) > 0:
            train_transform_list.append(
                transforms.RandomVerticalFlip(p=aug_cfg['vertical_flip'])
            )

        if aug_cfg.get('rotation', 0) > 0:
            train_transform_list.append(
                transforms.RandomRotation(degrees=aug_cfg['rotation'])
            )

        if 'color_jitter' in aug_cfg:
            cj = aug_cfg['color_jitter']
            train_transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get('brightness', 0),
                    contrast=cj.get('contrast', 0),
                    saturation=cj.get('saturation', 0),
                    hue=cj.get('hue', 0),
                )
            )

        if 'random_affine' in aug_cfg:
            ra = aug_cfg['random_affine']
            train_transform_list.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=tuple(ra.get('translate', [0, 0])),
                    scale=tuple(ra.get('scale', [1.0, 1.0])),
                )
            )

        if aug_cfg.get('gaussian_blur', 0) > 0:
            train_transform_list.append(
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3)
                ], p=aug_cfg['gaussian_blur'])
            )

        # Convert to tensor and normalize
        train_transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        if aug_cfg.get('random_erasing', 0) > 0:
            train_transform_list.append(
                transforms.RandomErasing(p=aug_cfg['random_erasing'])
            )

        train_transform = transforms.Compose(train_transform_list)

        # Val transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        return train_transform, val_transform

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and val data loaders."""
        data_root = Path(self.cfg.data_root)
        train_dir = data_root / self.cfg.train_dir
        val_dir = data_root / self.cfg.val_dir

        # Check directories exist
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"Val directory not found: {val_dir}")

        # Create transforms
        train_transform, val_transform = self._create_transforms()

        # Create datasets
        train_dataset = ClassificationDataset(
            root_dir=str(train_dir),
            transform=train_transform,
            class_names=self.class_names,
        )

        val_dataset = ClassificationDataset(
            root_dir=str(val_dir),
            transform=val_transform,
            class_names=self.class_names,
        )

        # Calculate class weights for imbalanced data
        class_counts = train_dataset.get_class_counts()
        logger.info(f"Training class distribution: {class_counts}")

        if self.cfg.class_weights == 'balanced':
            total = sum(class_counts.values())
            n_classes = len(class_counts)
            weights = []
            for class_name in self.class_names:
                count = class_counts.get(class_name, 1)
                weight = total / (n_classes * count)
                weights.append(weight)
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
            logger.info(f"Computed class weights: {self.class_weights.tolist()}")
        elif isinstance(self.cfg.class_weights, list):
            self.class_weights = torch.tensor(self.cfg.class_weights, dtype=torch.float32)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and self.device.type == 'cuda',
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and self.device.type == 'cuda',
        )

        return train_loader, val_loader

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        optimizer_name = self.cfg.optimizer.lower()
        lr = self.cfg.learning_rate
        weight_decay = self.cfg.weight_decay

        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            logger.warning(f"Unknown optimizer {optimizer_name}, using AdamW")
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.cfg.scheduler_type.lower()
        epochs = self.cfg.epochs

        if scheduler_type == 'cosineannealinglr':
            T_max = self.cfg.scheduler_T_max or epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=self.cfg.scheduler_eta_min,
            )
        elif scheduler_type == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=epochs // 3,
                gamma=0.1,
            )
        elif scheduler_type == 'reducelronplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6,
            )

        return scheduler

    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        label_smoothing = self.cfg.label_smoothing

        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights.to(self.device),
                label_smoothing=label_smoothing,
            )
        else:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
            )

        return criterion

    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Single training step."""
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'batch_size': labels.size(0),
        }

    def _validate(self) -> Dict[str, float]:
        """Run validation and compute metrics."""
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                total_loss += loss.item()
                num_batches += 1

        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)

        # For binary classification
        if self.num_classes == 2:
            precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        else:
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

        # Log confusion matrix
        logger.info(f"Confusion Matrix:\n{cm}")

        return metrics

    def _get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Get primary metric for early stopping (F1 score for classification)."""
        return metrics.get('f1_score', 0.0)

    def export_model(self):
        """Export trained model to ONNX format."""
        logger.info("Exporting model...")

        # Load best checkpoint
        best_checkpoint = self.checkpoint_manager.load_best()
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            logger.info("Loaded best checkpoint for export")

        self.model.eval()

        # Create export directories
        export_dir = Path(self.cfg.export_dir)
        models_dir = Path(self.cfg.models_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Export to ONNX
        model_name = self.cfg.model_name.replace('/', '_')
        onnx_path = export_dir / f"{model_name}_classifier.onnx"

        # Create dummy input
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=self.cfg.onnx_opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            },
        )

        logger.info(f"Exported ONNX model to: {onnx_path}")

        # Copy to models directory
        final_path = models_dir / f"{model_name}_classifier.onnx"
        import shutil
        shutil.copy2(onnx_path, final_path)
        logger.info(f"Copied to: {final_path}")

        # Also save PyTorch model
        pth_path = export_dir / f"{model_name}_classifier.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'model_name': self.cfg.model_name,
        }, pth_path)
        logger.info(f"Saved PyTorch model to: {pth_path}")

        return str(final_path)
