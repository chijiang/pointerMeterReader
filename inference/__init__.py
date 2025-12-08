"""
Unified Inference Pipeline for Meter Reading Models

This module provides GPU-optimized inference for:
- YOLOv11 object detection (meter localization)
- SegFormer semantic segmentation (pointer/scale identification)
- EfficientNet image classification (binary/multi-class)
- Complete meter reading pipeline

Features:
- Automatic GPU/MPS/CPU device selection
- ONNX and PyTorch model support
- Batch inference with optimal throughput
- Memory-efficient processing
- Post-processing and visualization

Usage:
    # Detection inference
    python predict.py --task detection --source image.jpg

    # Segmentation inference
    python predict.py --task segmentation --source image.jpg

    # Classification inference
    python predict.py --task classification --source image.jpg

    # Complete pipeline (detect + segment + read)
    python predict.py --task pipeline --source image.jpg
"""

from .base_predictor import BasePredictor
from .detection_predictor import DetectionPredictor
from .segmentation_predictor import SegmentationPredictor
from .classification_predictor import ClassificationPredictor
from .pipeline_predictor import PipelinePredictor

__all__ = [
    'BasePredictor',
    'DetectionPredictor',
    'SegmentationPredictor',
    'ClassificationPredictor',
    'PipelinePredictor',
]
