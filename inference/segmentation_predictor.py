"""
Segmentation Predictor - SegFormer meter segmentation inference
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from .base_predictor import BasePredictor, PredictionResult, ImageLoader

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Segmentation result container"""
    mask: np.ndarray  # (H, W) class indices
    class_areas: Dict[int, int]  # class_id -> pixel count
    class_ratios: Dict[int, float]  # class_id -> ratio


class SegmentationPredictor(BasePredictor):
    """
    SegFormer Segmentation Predictor with GPU optimization.

    Supports:
    - ONNX models (.onnx)
    - PyTorch models (.pth, .pt)
    - Automatic device selection
    - Post-processing with morphological operations
    """

    # Class definitions
    CLASS_NAMES = ['background', 'pointer', 'scale']
    CLASS_COLORS = {
        0: (0, 0, 0),       # background - black
        1: (255, 0, 0),     # pointer - red
        2: (0, 255, 0),     # scale - green
    }
    CLASS_COLORS_BGR = {
        0: (0, 0, 0),
        1: (0, 0, 255),     # BGR: red
        2: (0, 255, 0),     # BGR: green
    }

    def __init__(self, model_path: str, device: str = 'auto', **kwargs):
        """
        Initialize segmentation predictor.

        Args:
            model_path: Path to model (.onnx or .pth)
            device: Device to use
            **kwargs: Additional options
                - input_size: Model input size (default: 480)
                - post_process: Enable post-processing (default: True)
                - alpha: Visualization overlay alpha (default: 0.5)
        """
        self.post_process_enabled = kwargs.get('post_process', True)
        self.alpha = kwargs.get('alpha', 0.5)

        # Post-processing config
        self.post_process_config = kwargs.get('post_process_config', {
            'remove_noise': True,
            'keep_largest_component': False,
            'pointer_erosion': 1,
            'scale_erosion': 2,
        })

        super().__init__(model_path, device, **kwargs)

    def _load_model(self):
        """Load segmentation model"""
        suffix = self.model_path.suffix.lower()

        if suffix == '.onnx':
            self._load_onnx_model()
        elif suffix in ['.pth', '.pt']:
            self._load_pytorch_model()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def _load_onnx_model(self):
        """Load ONNX segmentation model"""
        if not HAS_ONNX:
            raise ImportError("onnxruntime required. Install: pip install onnxruntime")

        logger.info(f"Loading ONNX model: {self.model_path}")

        providers = ['CPUExecutionProvider']
        if self.device_manager.is_cuda:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.model = ort.InferenceSession(str(self.model_path), providers=providers)
        self.model_type = 'onnx'

        # Get input shape
        input_info = self.model.get_inputs()[0]
        self.input_name = input_info.name
        shape = input_info.shape

        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = (shape[2], shape[3])
        else:
            self.input_size = (480, 480)

        logger.info(f"ONNX model loaded, input size: {self.input_size}")

    def _load_pytorch_model(self):
        """Load PyTorch segmentation model"""
        if not HAS_TORCH:
            raise ImportError("torch required. Install: pip install torch")

        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError:
            raise ImportError("transformers required. Install: pip install transformers")

        logger.info(f"Loading PyTorch model: {self.model_path}")

        # Load checkpoint
        device = 'cuda' if self.device_manager.is_cuda else 'cpu'
        checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)

        # Get model config
        model_config = checkpoint.get('model_config', {})
        model_name = model_config.get('name', 'nvidia/segformer-b2-finetuned-ade-512-512')
        num_classes = model_config.get('num_classes', 3)

        # Create model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model_type = 'pytorch'

        # Get input size
        input_size = checkpoint.get('input_size', [480, 480])
        self.input_size = tuple(input_size)

        logger.info(f"PyTorch model loaded, input size: {self.input_size}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for inference.

        Args:
            image: Input image (BGR)

        Returns:
            (preprocessed tensor, original size)
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        h, w = self.input_size
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # To NCHW
        tensor = normalized.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)

        return tensor.astype(np.float32), original_size

    def predict(self, image: Union[str, np.ndarray]) -> PredictionResult:
        """
        Run segmentation on single image.

        Args:
            image: Image path or numpy array (BGR)

        Returns:
            PredictionResult with segmentation mask
        """
        start_time = time.time()

        try:
            # Load image
            if isinstance(image, (str, Path)):
                image_bgr, image_path = ImageLoader.load(image)
            else:
                image_bgr = image
                image_path = None

            # Preprocess
            tensor, original_size = self.preprocess(image_bgr)

            # Run inference
            if self.model_type == 'onnx':
                mask = self._predict_onnx(tensor, original_size)
            else:
                mask = self._predict_pytorch(tensor, original_size)

            # Post-process
            if self.post_process_enabled:
                mask = self._post_process(mask)

            inference_time = (time.time() - start_time) * 1000

            # Calculate statistics
            seg_result = self._calculate_stats(mask)

            # Create visualization
            visualization = self.visualize(image_bgr, mask)

            return PredictionResult(
                success=True,
                image_path=image_path,
                predictions=seg_result,
                visualization=visualization,
                metadata={
                    'class_names': self.CLASS_NAMES,
                    'class_areas': seg_result.class_areas,
                    'class_ratios': seg_result.class_ratios,
                },
                inference_time_ms=inference_time,
            )

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return PredictionResult(
                success=False,
                image_path=str(image) if isinstance(image, (str, Path)) else None,
                error=str(e),
            )

    def _predict_onnx(self, tensor: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Run ONNX inference"""
        outputs = self.model.run(None, {self.input_name: tensor})
        output = outputs[0]

        # Get predictions
        if len(output.shape) == 4:
            mask = np.argmax(output[0], axis=0)
        else:
            mask = output[0]

        # Resize to original size
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

        return mask

    def _predict_pytorch(self, tensor: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Run PyTorch inference"""
        with torch.no_grad():
            input_tensor = torch.from_numpy(tensor).to(self.device)
            outputs = self.model(input_tensor)
            logits = outputs.logits

            # Upsample
            logits = F.interpolate(
                logits,
                size=self.input_size,
                mode='bilinear',
                align_corners=False
            )

            mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

        # Resize to original
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

        return mask

    def _post_process(self, mask: np.ndarray) -> np.ndarray:
        """Apply post-processing to segmentation mask"""
        config = self.post_process_config
        processed = mask.copy()

        for class_id in [1, 2]:  # pointer and scale
            class_mask = (mask == class_id).astype(np.uint8)

            if np.sum(class_mask) == 0:
                continue

            # Remove noise
            if config.get('remove_noise', True):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

            # Keep largest component
            if config.get('keep_largest_component', False):
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask)
                if num_labels > 1:
                    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    class_mask = (labels == largest).astype(np.uint8)

            # Class-specific erosion
            if class_id == 1 and config.get('pointer_erosion', 0) > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                class_mask = cv2.erode(class_mask, kernel, iterations=config['pointer_erosion'])
            elif class_id == 2 and config.get('scale_erosion', 0) > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                class_mask = cv2.erode(class_mask, kernel, iterations=config['scale_erosion'])

            # Update mask
            processed[mask == class_id] = 0
            processed[class_mask == 1] = class_id

        return processed

    def _calculate_stats(self, mask: np.ndarray) -> SegmentationResult:
        """Calculate segmentation statistics"""
        total_pixels = mask.size

        class_areas = {}
        class_ratios = {}

        for class_id in range(len(self.CLASS_NAMES)):
            area = int(np.sum(mask == class_id))
            class_areas[class_id] = area
            class_ratios[class_id] = area / total_pixels

        return SegmentationResult(
            mask=mask,
            class_areas=class_areas,
            class_ratios=class_ratios,
        )

    def visualize(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create visualization with colored overlay.

        Args:
            image: Original image (BGR)
            mask: Segmentation mask

        Returns:
            Visualization image (BGR)
        """
        # Create colored mask
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in self.CLASS_COLORS_BGR.items():
            colored[mask == class_id] = color

        # Blend with original
        overlay = cv2.addWeighted(image, 1 - self.alpha, colored, self.alpha, 0)

        # Add legend
        overlay = self._add_legend(overlay, mask)

        return overlay

    def _add_legend(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add legend to visualization"""
        vis = image.copy()
        h, w = image.shape[:2]

        y = 20
        for class_id, name in enumerate(self.CLASS_NAMES):
            if class_id == 0:
                continue  # Skip background

            area = np.sum(mask == class_id)
            ratio = area / mask.size * 100
            color = self.CLASS_COLORS_BGR[class_id]

            # Draw color box
            cv2.rectangle(vis, (10, y - 12), (25, y + 3), color, -1)
            cv2.rectangle(vis, (10, y - 12), (25, y + 3), (255, 255, 255), 1)

            # Draw text
            text = f"{name}: {ratio:.1f}%"
            cv2.putText(vis, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            y += 20

        return vis

    def get_mask(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Get segmentation mask only (no visualization).

        Args:
            image: Input image

        Returns:
            Segmentation mask or None on error
        """
        result = self.predict(image)
        if result.success and result.predictions:
            return result.predictions.mask
        return None

    def get_pointer_mask(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Get pointer class mask only"""
        mask = self.get_mask(image)
        if mask is not None:
            return (mask == 1).astype(np.uint8) * 255
        return None

    def get_scale_mask(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Get scale class mask only"""
        mask = self.get_mask(image)
        if mask is not None:
            return (mask == 2).astype(np.uint8) * 255
        return None

    def segment_cropped_meter(self, cropped_image: np.ndarray) -> Dict[str, Any]:
        """
        Segment a cropped meter image.

        Args:
            cropped_image: Cropped meter image (BGR)

        Returns:
            Dict with 'mask', 'pointer_mask', 'scale_mask', 'visualization'
        """
        result = self.predict(cropped_image)

        if not result.success:
            return {'error': result.error}

        mask = result.predictions.mask

        return {
            'mask': mask,
            'pointer_mask': (mask == 1).astype(np.uint8) * 255,
            'scale_mask': (mask == 2).astype(np.uint8) * 255,
            'visualization': result.visualization,
            'class_areas': result.predictions.class_areas,
            'class_ratios': result.predictions.class_ratios,
        }
