"""
Segmentation Trainer - SegFormer semantic segmentation with GPU optimization

Trains SegFormer model for meter pointer/scale segmentation.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from tqdm import tqdm

try:
    from transformers import SegformerForSemanticSegmentation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .base_trainer import BaseTrainer
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """Segmentation dataset for meter pointer/scale"""

    def __init__(self, root_dir: str, image_dir: str, mask_dir: str,
                 split_file: str = None, image_size: int = 512,
                 is_training: bool = True, augment: bool = True):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_dir
        self.mask_dir = self.root_dir / mask_dir
        self.image_size = image_size
        self.is_training = is_training
        self.augment = augment and is_training

        # Load image list
        if split_file and Path(split_file).exists():
            with open(split_file, 'r') as f:
                self.image_names = [line.strip() for line in f if line.strip()]
        else:
            image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
            self.image_names = [f.stem for f in image_files]

        # Validate
        valid_names = []
        for name in self.image_names:
            if self._find_image(name) and self._find_mask(name):
                valid_names.append(name)

        self.image_names = valid_names
        logger.info(f"Dataset: {len(self.image_names)} valid samples ({'train' if is_training else 'val'})")

    def _find_image(self, name: str) -> Optional[Path]:
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            path = self.image_dir / f"{name}{ext}"
            if path.exists():
                return path
        return None

    def _find_mask(self, name: str) -> Optional[Path]:
        path = self.mask_dir / f"{name}.png"
        return path if path.exists() else None

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        # Load image and mask
        image = Image.open(self._find_image(name)).convert('RGB')
        mask = np.array(Image.open(self._find_mask(name)))

        # Ensure mask values are valid (0=background, 1=pointer, 2=scale)
        mask = np.clip(mask, 0, 2)

        # Apply transforms
        image, mask = self._transform(image, mask)

        return image, mask

    def _transform(self, image: Image.Image, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms to image and mask"""

        if self.augment:
            # Random scale
            scale = np.random.uniform(0.8, 1.2)
            new_size = int(self.image_size * scale)

            image = TF.resize(image, [new_size, new_size])
            mask_pil = Image.fromarray(mask)
            mask_pil = TF.resize(mask_pil, [new_size, new_size], interpolation=Image.NEAREST)
            mask = np.array(mask_pil)

            # Random crop
            if new_size > self.image_size:
                i = np.random.randint(0, new_size - self.image_size)
                j = np.random.randint(0, new_size - self.image_size)
                image = TF.crop(image, i, j, self.image_size, self.image_size)
                mask = mask[i:i+self.image_size, j:j+self.image_size]
            else:
                # Pad if needed
                image = TF.resize(image, [self.image_size, self.image_size])
                mask_pil = Image.fromarray(mask)
                mask_pil = TF.resize(mask_pil, [self.image_size, self.image_size], interpolation=Image.NEAREST)
                mask = np.array(mask_pil)

            # Random horizontal flip
            if np.random.random() > 0.5:
                image = TF.hflip(image)
                mask = np.fliplr(mask).copy()

            # Random rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask_pil = Image.fromarray(mask)
                mask_pil = TF.rotate(mask_pil, angle, fill=0)
                mask = np.array(mask_pil)

            # Color jitter
            if np.random.random() > 0.5:
                jitter = transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                )
                image = jitter(image)

        else:
            # Validation: just resize
            image = TF.resize(image, [self.image_size, self.image_size])
            mask_pil = Image.fromarray(mask)
            mask_pil = TF.resize(mask_pil, [self.image_size, self.image_size], interpolation=Image.NEAREST)
            mask = np.array(mask_pil)

        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = torch.from_numpy(mask).long()

        return image, mask


class SegmentationMetrics:
    """Compute segmentation metrics"""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        # Valid mask
        valid = (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]

        # Update confusion matrix
        for p, t in zip(pred, target):
            self.confusion_matrix[t, p] += 1

    @property
    def pixel_accuracy(self) -> float:
        return np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)

    @property
    def class_iou(self) -> np.ndarray:
        iou = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            iou.append(tp / (tp + fp + fn + 1e-10))
        return np.array(iou)

    @property
    def mean_iou(self) -> float:
        return self.class_iou.mean()

    @property
    def dice_scores(self) -> np.ndarray:
        dice = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            dice.append(2 * tp / (2 * tp + fp + fn + 1e-10))
        return np.array(dice)


class SegmentationTrainer(BaseTrainer):
    """
    SegFormer Segmentation Trainer with GPU optimization.

    Features:
    - SegFormer-B0 to B5 support
    - Mixed precision training
    - Gradient accumulation
    - Class-weighted loss
    - ONNX export with upsampling
    """

    CLASS_NAMES = ['background', 'pointer', 'scale']
    CLASS_COLORS = {
        0: [0, 0, 0],       # background - black
        1: [255, 0, 0],     # pointer - red
        2: [0, 255, 0],     # scale - green
    }

    def __init__(self, config: ConfigManager):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required. Install: pip install transformers")

        super().__init__(config)

        # Override paths for segmentation
        self.output_dir = Path(self.cfg.save_dir).parent / "segmentation"
        self.export_dir = self.output_dir / "exported"
        self.model_dir = Path(__file__).parent.parent / "models" / "segmentation"

        for d in [self.output_dir, self.export_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Update checkpoint manager
        self.checkpoint_manager.save_dir = self.output_dir / "checkpoints"
        self.checkpoint_manager.save_dir.mkdir(parents=True, exist_ok=True)

    def _create_model(self) -> nn.Module:
        """Create SegFormer model"""
        model_name = self.cfg.model_name
        num_classes = self.cfg.num_classes

        logger.info(f"Loading SegFormer: {model_name}")

        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # Check for encoder freezing
        segformer_config = self.cfg.extra.get('segformer', {})
        if segformer_config.get('freeze_encoder', False):
            freeze_layers = segformer_config.get('freeze_encoder_layers', 0)
            for name, param in model.segformer.encoder.named_parameters():
                layer_num = int(name.split('.')[1]) if name.split('.')[1].isdigit() else -1
                if layer_num < freeze_layers:
                    param.requires_grad = False
                    logger.info(f"Frozen: {name}")

        return model

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders"""
        data_config = self.cfg.extra.get('data', {})

        root_dir = data_config.get('root_dir', 'data/segmentation')
        image_dir = data_config.get('image_dir', 'JPEGImages')
        mask_dir = data_config.get('mask_dir', 'SegmentationClass')
        split_dir = Path(root_dir) / data_config.get('split_dir', 'ImageSets/Segmentation')

        train_split = split_dir / 'train.txt' if split_dir.exists() else None
        val_split = split_dir / 'val.txt' if split_dir.exists() else None

        # Create datasets
        train_dataset = SegmentationDataset(
            root_dir=root_dir,
            image_dir=image_dir,
            mask_dir=mask_dir,
            split_file=str(train_split) if train_split and train_split.exists() else None,
            image_size=self.cfg.image_size,
            is_training=True,
            augment=True,
        )

        val_dataset = SegmentationDataset(
            root_dir=root_dir,
            image_dir=image_dir,
            mask_dir=mask_dir,
            split_file=str(val_split) if val_split and val_split.exists() else None,
            image_size=self.cfg.image_size,
            is_training=False,
            augment=False,
        )

        # If no val split, create from train
        if len(val_dataset) == 0:
            logger.info("No validation split, using 80/20 split")
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )

        # DataLoader kwargs
        loader_kwargs = self.device_manager.get_dataloader_kwargs()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > self.cfg.batch_size,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        return train_loader, val_loader

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        total_steps = len(self.train_loader) * self.cfg.epochs

        scheduler_type = self.cfg.scheduler_type.lower()

        if scheduler_type in ['polynomial', 'poly', 'polynomiallr']:
            return optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=total_steps,
                power=1.0,
            )
        elif scheduler_type in ['cosine', 'cosineannealing']:
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-7,
            )
        else:
            return None

    def _create_criterion(self) -> nn.Module:
        """Create loss function with class weights"""
        loss_config = self.cfg.extra.get('training', {}).get('loss', {})

        class_weights = loss_config.get('class_weights', [0.5, 2.0, 2.0])
        if class_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        return nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=loss_config.get('ignore_index', 255),
        )

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step"""
        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        outputs = self.model(images)
        logits = outputs.logits

        # Upsample to target size
        logits = F.interpolate(
            logits,
            size=targets.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

        # Compute loss
        loss = self.criterion(logits, targets)

        return {
            'loss': loss,
            'batch_size': images.size(0),
        }

    def _validate(self) -> Dict[str, float]:
        """Run validation"""
        metrics = SegmentationMetrics(self.cfg.num_classes)
        total_loss = 0.0
        num_batches = 0

        for images, targets in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            logits = outputs.logits

            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

            loss = self.criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1

            pred = torch.argmax(logits, dim=1)
            metrics.update(pred, targets)

        # Compute metrics
        result = {
            'loss': total_loss / num_batches,
            'pixel_acc': metrics.pixel_accuracy,
            'mean_iou': metrics.mean_iou,
        }

        # Per-class IoU
        class_iou = metrics.class_iou
        for i, name in enumerate(self.CLASS_NAMES):
            result[f'iou_{name}'] = class_iou[i]

        return result

    def _get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Get primary metric (mIoU)"""
        return metrics.get('mean_iou', 0.0)

    def export_model(self):
        """Export model to ONNX"""
        export_config = self.cfg.extra.get('export', {})

        # Load best model
        best_checkpoint = self.checkpoint_manager.load_best()
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            logger.info("Loaded best model for export")

        self.model.eval()

        # Get input size
        h = w = self.cfg.image_size

        # Create wrapper for ONNX export (includes upsampling)
        class SegFormerONNXWrapper(nn.Module):
            def __init__(self, model, output_size):
                super().__init__()
                self.model = model
                self.output_size = output_size

            def forward(self, x):
                outputs = self.model(x)
                logits = outputs.logits
                logits = F.interpolate(
                    logits,
                    size=self.output_size,
                    mode='bilinear',
                    align_corners=False,
                )
                return logits

        wrapper = SegFormerONNXWrapper(self.model, (h, w))
        wrapper.eval()

        # Example input
        example_input = torch.randn(1, 3, h, w).to(self.device)

        # Export ONNX
        onnx_path = self.export_dir / 'segformer_meter.onnx'

        try:
            torch.onnx.export(
                wrapper,
                example_input,
                str(onnx_path),
                export_params=True,
                opset_version=export_config.get('onnx', {}).get('opset_version', 14),
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                },
            )

            # Copy to models directory
            final_onnx = self.model_dir / 'segformer_meter.onnx'
            shutil.copy2(onnx_path, final_onnx)
            logger.info(f"ONNX exported: {final_onnx}")

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

        # Save PyTorch model
        pth_path = self.export_dir / 'segformer_meter.pth'
        model_info = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'name': self.cfg.model_name,
                'num_classes': self.cfg.num_classes,
            },
            'input_size': [h, w],
            'class_names': self.CLASS_NAMES,
        }
        torch.save(model_info, pth_path)

        final_pth = self.model_dir / 'segformer_meter.pth'
        shutil.copy2(pth_path, final_pth)
        logger.info(f"PyTorch model exported: {final_pth}")

    def visualize_predictions(self, num_samples: int = 5):
        """Visualize predictions on validation set"""
        self.model.eval()
        output_dir = self.output_dir / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for i, (images, targets) in enumerate(self.val_loader):
                if i >= num_samples:
                    break

                images = images.to(self.device)
                outputs = self.model(images)
                logits = outputs.logits

                logits = F.interpolate(
                    logits,
                    size=targets.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )

                pred = torch.argmax(logits, dim=1)

                # Visualize first sample in batch
                image = images[0].cpu()
                target = targets[0].numpy()
                prediction = pred[0].cpu().numpy()

                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image.permute(1, 2, 0).numpy()
                image = std * image + mean
                image = np.clip(image, 0, 1)

                # Create figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(image)
                axes[0].set_title('Input')
                axes[0].axis('off')

                axes[1].imshow(self._colorize_mask(target))
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(self._colorize_mask(prediction))
                axes[2].set_title('Prediction')
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(output_dir / f'prediction_epoch_{self.current_epoch}_sample_{i}.png')
                plt.close()

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to RGB visualization"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in self.CLASS_COLORS.items():
            colored[mask == cls_id] = color
        return colored
