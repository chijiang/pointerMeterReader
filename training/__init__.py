"""
Unified Training Pipeline for Meter Reading Models

This module provides a GPU-optimized training infrastructure for:
- YOLOv11 object detection (meter localization)
- SegFormer semantic segmentation (pointer/scale identification)
- EfficientNet image classification (binary/multi-class)

Features:
- Automatic GPU/MPS/CPU device selection
- Mixed precision training (AMP)
- Gradient accumulation for large effective batch sizes
- Memory optimization and gradient checkpointing
- Unified configuration system
- Early stopping and learning rate scheduling
- TensorBoard and checkpoint management

Usage:
    # Train detection model
    python train.py --task detection --config config/detection.yaml

    # Train segmentation model
    python train.py --task segmentation --config config/segmentation.yaml

    # Train classification model
    python train.py --task classification --config config/classification.yaml
"""

from .base_trainer import BaseTrainer
from .device_manager import DeviceManager
from .config_manager import ConfigManager
from .detection_trainer import DetectionTrainer
from .segmentation_trainer import SegmentationTrainer
from .classification_trainer import ClassificationTrainer

__all__ = [
    'BaseTrainer',
    'DeviceManager',
    'ConfigManager',
    'DetectionTrainer',
    'SegmentationTrainer',
    'ClassificationTrainer',
]
