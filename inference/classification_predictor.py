"""
Classification Predictor - Image classification inference

Supports binary and multi-class classification with:
- ONNX models (.onnx)
- PyTorch models (.pth, .pt)

Features:
- GPU/MPS/CPU acceleration
- Batch inference
- Probability output
- Visualization with class label overlay
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
import torch

from .base_predictor import BasePredictor, PredictionResult, ImageLoader

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Classification result container"""
    class_id: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]


class ClassificationPredictor(BasePredictor):
    """
    Image Classification Predictor.

    Supports:
    - ONNX models (.onnx)
    - PyTorch models (.pth, .pt)
    - Automatic device selection (CUDA/MPS/CPU)
    - Batch inference
    """

    # Default class names for water detection
    DEFAULT_CLASSES = ['no_water', 'has_water']

    # ImageNet normalization stats
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        class_names: List[str] = None,
        threshold: float = 0.5,
        input_size: int = 224,
        **kwargs
    ):
        """
        Initialize classification predictor.

        Args:
            model_path: Path to model (.onnx or .pth)
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            class_names: List of class names (auto-detect from model if None)
            threshold: Confidence threshold for positive class
            input_size: Model input size (default: 224 for EfficientNet)
            **kwargs: Additional configuration
        """
        self.class_names = class_names
        self.threshold = threshold
        self._input_size = input_size

        # Override input_size in kwargs
        kwargs['input_size'] = (input_size, input_size)

        super().__init__(model_path, device, **kwargs)

        # Set default class names if not provided
        if self.class_names is None:
            self.class_names = self.DEFAULT_CLASSES

        self.num_classes = len(self.class_names)

    def _load_model(self):
        """Load classification model (ONNX or PyTorch)."""
        suffix = self.model_path.suffix.lower()

        if suffix == '.onnx':
            self._load_onnx_model()
        elif suffix in ['.pth', '.pt']:
            self._load_pytorch_model()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def _load_onnx_model(self):
        """Load ONNX model."""
        import onnxruntime as ort

        # Select execution provider
        if self.device.type == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.model = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        self.model_type = 'onnx'

        # Get input/output names
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

        # Get input shape
        input_shape = self.model.get_inputs()[0].shape
        if len(input_shape) == 4 and input_shape[2] is not None:
            self._input_size = input_shape[2]
            self.input_size = (input_shape[2], input_shape[3])

        logger.info(f"Loaded ONNX model: {self.model_path.name}")
        logger.info(f"Input size: {self._input_size}x{self._input_size}")

    def _load_pytorch_model(self):
        """Load PyTorch model."""
        from torchvision import models
        import torch.nn as nn

        # Load checkpoint
        checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)

        # Get model info from checkpoint
        model_name = checkpoint.get('model_name', 'efficientnet_b0')
        num_classes = checkpoint.get('num_classes', 2)
        saved_class_names = checkpoint.get('class_names')

        if saved_class_names and self.class_names == self.DEFAULT_CLASSES:
            self.class_names = saved_class_names
            self.num_classes = len(self.class_names)

        if 'image_size' in checkpoint:
            self._input_size = checkpoint['image_size']
            self.input_size = (self._input_size, self._input_size)

        # Create model architecture
        model_name = model_name.lower()

        if 'efficientnet' in model_name:
            if 'b0' in model_name:
                self.model = models.efficientnet_b0(weights=None)
            elif 'b1' in model_name:
                self.model = models.efficientnet_b1(weights=None)
            else:
                self.model = models.efficientnet_b0(weights=None)

            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes),
            )

        elif 'resnet' in model_name:
            if '50' in model_name:
                self.model = models.resnet50(weights=None)
            else:
                self.model = models.resnet18(weights=None)

            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        elif 'mobilenet' in model_name:
            if 'small' in model_name:
                self.model = models.mobilenet_v3_small(weights=None)
            else:
                self.model = models.mobilenet_v3_large(weights=None)

            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, num_classes)

        else:
            # Default to EfficientNet-B0
            self.model = models.efficientnet_b0(weights=None)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes),
            )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model_type = 'pytorch'
        logger.info(f"Loaded PyTorch model: {self.model_path.name}")
        logger.info(f"Model architecture: {model_name}")
        logger.info(f"Classes: {self.class_names}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: Input image (BGR, HWC)

        Returns:
            Preprocessed tensor (NCHW, float32)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        resized = cv2.resize(rgb, (self._input_size, self._input_size))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        normalized = (normalized - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # Convert to NCHW format
        tensor = normalized.transpose(2, 0, 1)  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension

        return tensor.astype(np.float32)

    def predict(self, image: Union[str, np.ndarray]) -> PredictionResult:
        """
        Run classification on single image.

        Args:
            image: Image path or numpy array (BGR)

        Returns:
            PredictionResult with ClassificationResult
        """
        start_time = time.time()

        try:
            # Load image
            if isinstance(image, (str, Path)):
                image_np, image_path = ImageLoader.load(image)
            else:
                image_np = image
                image_path = None

            # Preprocess
            tensor = self.preprocess(image_np)

            # Inference
            if self.model_type == 'onnx':
                result = self._predict_onnx(tensor)
            else:
                result = self._predict_pytorch(tensor)

            # Create visualization
            visualization = self.visualize(image_np, result)

            inference_time = (time.time() - start_time) * 1000

            return PredictionResult(
                success=True,
                image_path=image_path,
                predictions=result,
                visualization=visualization,
                metadata={
                    'class_id': result.class_id,
                    'class_name': result.class_name,
                    'confidence': result.confidence,
                    'probabilities': result.probabilities,
                },
                inference_time_ms=inference_time,
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return PredictionResult(
                success=False,
                image_path=str(image) if isinstance(image, (str, Path)) else None,
                error=str(e),
                inference_time_ms=(time.time() - start_time) * 1000,
            )

    def _predict_onnx(self, tensor: np.ndarray) -> ClassificationResult:
        """Run ONNX inference."""
        outputs = self.model.run(None, {self.input_name: tensor})
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        probabilities = {
            name: float(p) for name, p in zip(self.class_names, probs)
        }

        return ClassificationResult(
            class_id=class_id,
            class_name=self.class_names[class_id],
            confidence=confidence,
            probabilities=probabilities,
        )

    def _predict_pytorch(self, tensor: np.ndarray) -> ClassificationResult:
        """Run PyTorch inference."""
        with torch.no_grad():
            input_tensor = torch.from_numpy(tensor).to(self.device)
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        probabilities = {
            name: float(p) for name, p in zip(self.class_names, probs)
        }

        return ClassificationResult(
            class_id=class_id,
            class_name=self.class_names[class_id],
            confidence=confidence,
            probabilities=probabilities,
        )

    def predict_proba(self, image: Union[str, np.ndarray]) -> Dict[str, float]:
        """
        Get class probabilities.

        Args:
            image: Image path or numpy array

        Returns:
            Dict mapping class names to probabilities
        """
        result = self.predict(image)
        if result.success:
            return result.predictions.probabilities
        return {}

    def is_positive(self, image: Union[str, np.ndarray]) -> bool:
        """
        Check if image is positive class (has water).

        Args:
            image: Image path or numpy array

        Returns:
            True if positive class with confidence above threshold
        """
        result = self.predict(image)
        if result.success:
            return (result.predictions.class_id == 1 and
                    result.predictions.confidence >= self.threshold)
        return False

    def visualize(self, image: np.ndarray, predictions: ClassificationResult) -> np.ndarray:
        """
        Create visualization with classification result.

        Args:
            image: Original image (BGR)
            predictions: ClassificationResult

        Returns:
            Visualization image (BGR)
        """
        vis = image.copy()
        h, w = vis.shape[:2]

        # Determine colors based on class
        if predictions.class_id == 0:  # no_water
            color = (0, 255, 0)  # Green
            bg_color = (0, 180, 0)
        else:  # has_water
            color = (0, 0, 255)  # Red
            bg_color = (0, 0, 180)

        # Create label
        label = f"{predictions.class_name}: {predictions.confidence:.1%}"

        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) / 500  # Scale based on image size
        font_scale = max(0.5, min(font_scale, 1.5))
        thickness = max(1, int(font_scale * 2))

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw background rectangle
        padding = 10
        rect_x1 = 10
        rect_y1 = 10
        rect_x2 = rect_x1 + text_w + padding * 2
        rect_y2 = rect_y1 + text_h + padding * 2 + baseline

        cv2.rectangle(vis, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

        # Draw text
        text_x = rect_x1 + padding
        text_y = rect_y1 + text_h + padding
        cv2.putText(vis, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Add probability bar (optional)
        bar_height = 5
        bar_y = rect_y2 + 5
        bar_width = int((rect_x2 - rect_x1) * predictions.confidence)
        cv2.rectangle(vis, (rect_x1, bar_y), (rect_x1 + bar_width, bar_y + bar_height), color, -1)
        cv2.rectangle(vis, (rect_x1, bar_y), (rect_x2, bar_y + bar_height), (128, 128, 128), 1)

        return vis

    def warmup(self, iterations: int = 3):
        """Warmup model with dummy inputs."""
        logger.info(f"Warming up model ({iterations} iterations)...")

        dummy_image = np.random.randint(0, 255, (self._input_size, self._input_size, 3), dtype=np.uint8)

        for _ in range(iterations):
            try:
                self.predict(dummy_image)
            except Exception:
                pass

        self.device_manager.clear_cache()
        logger.info("Warmup complete")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'threshold': self.threshold,
        })
        return info
