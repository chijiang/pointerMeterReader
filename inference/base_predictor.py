"""
Base Predictor - Abstract base class for all inference tasks
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass

import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.device_manager import DeviceManager

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results"""
    success: bool
    image_path: Optional[str] = None
    predictions: Any = None
    visualization: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    inference_time_ms: float = 0.0


class ImageLoader:
    """Utility for loading images from various sources"""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    @staticmethod
    def load(source: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Optional[str]]:
        """
        Load image from various sources.

        Args:
            source: File path, directory, URL, or numpy array

        Returns:
            (image_bgr, source_path)
        """
        if isinstance(source, np.ndarray):
            return source, None

        source = Path(source) if isinstance(source, str) else source

        if not source.exists():
            raise FileNotFoundError(f"Image not found: {source}")

        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Failed to load image: {source}")

        return image, str(source)

    @staticmethod
    def find_images(source: Union[str, Path]) -> List[Path]:
        """
        Find all images in source (file or directory).

        Args:
            source: File path or directory

        Returns:
            List of image paths
        """
        source = Path(source)

        if source.is_file():
            if source.suffix.lower() in ImageLoader.SUPPORTED_EXTENSIONS:
                return [source]
            return []

        if source.is_dir():
            images = []
            for ext in ImageLoader.SUPPORTED_EXTENSIONS:
                images.extend(source.glob(f'*{ext}'))
                images.extend(source.glob(f'*{ext.upper()}'))
            return sorted(images)

        return []


class BasePredictor(ABC):
    """
    Abstract base predictor with GPU optimization.

    Provides:
    - Device management (CUDA/MPS/CPU)
    - Model loading (ONNX/PyTorch)
    - Batch processing
    - Visualization utilities
    """

    def __init__(self, model_path: str, device: str = 'auto', **kwargs):
        """
        Initialize predictor.

        Args:
            model_path: Path to model file
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            **kwargs: Additional configuration
        """
        self.model_path = Path(model_path)
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device
        self.kwargs = kwargs

        # Model (to be loaded by subclass)
        self.model = None
        self.model_type = None  # 'onnx', 'pytorch', 'ultralytics'

        # Configuration
        self.input_size = kwargs.get('input_size', (640, 640))
        self.conf_threshold = kwargs.get('conf_threshold', 0.25)

        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load model
        self._load_model()

        logger.info(f"Predictor initialized: {self.__class__.__name__}")
        logger.info(f"Device: {self.device_manager.device_info.device_name}")
        logger.info(f"Model: {self.model_path.name}")

    @abstractmethod
    def _load_model(self):
        """Load model - to be implemented by subclass"""
        pass

    @abstractmethod
    def predict(self, image: Union[str, np.ndarray]) -> PredictionResult:
        """
        Run prediction on single image.

        Args:
            image: Image path or numpy array (BGR)

        Returns:
            PredictionResult
        """
        pass

    def predict_batch(self, images: List[Union[str, np.ndarray]],
                      batch_size: int = None) -> List[PredictionResult]:
        """
        Run prediction on multiple images.

        Args:
            images: List of image paths or arrays
            batch_size: Batch size (default: auto)

        Returns:
            List of PredictionResult
        """
        if batch_size is None:
            batch_size = self.device_manager.device_info.recommended_batch_size

        results = []
        for image in images:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image}: {e}")
                results.append(PredictionResult(
                    success=False,
                    image_path=str(image) if isinstance(image, (str, Path)) else None,
                    error=str(e)
                ))

        return results

    def predict_directory(self, directory: str, output_dir: str = None,
                          save_visualization: bool = True) -> List[PredictionResult]:
        """
        Run prediction on all images in directory.

        Args:
            directory: Input directory
            output_dir: Output directory for results
            save_visualization: Save visualization images

        Returns:
            List of PredictionResult
        """
        images = ImageLoader.find_images(directory)
        if not images:
            logger.warning(f"No images found in {directory}")
            return []

        logger.info(f"Found {len(images)} images in {directory}")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        from tqdm import tqdm

        for image_path in tqdm(images, desc="Processing"):
            result = self.predict(str(image_path))
            results.append(result)

            # Save visualization
            if save_visualization and output_dir and result.visualization is not None:
                vis_path = output_dir / f"{image_path.stem}_result.jpg"
                cv2.imwrite(str(vis_path), result.visualization)

        return results

    @abstractmethod
    def visualize(self, image: np.ndarray, predictions: Any) -> np.ndarray:
        """
        Create visualization of predictions.

        Args:
            image: Original image (BGR)
            predictions: Model predictions

        Returns:
            Visualization image (BGR)
        """
        pass

    def warmup(self, iterations: int = 3):
        """
        Warmup model with dummy inputs.

        Args:
            iterations: Number of warmup iterations
        """
        logger.info(f"Warming up model ({iterations} iterations)...")

        h, w = self.input_size
        dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        for _ in range(iterations):
            try:
                self.predict(dummy_image)
            except Exception:
                pass

        self.device_manager.clear_cache()
        logger.info("Warmup complete")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_path': str(self.model_path),
            'model_type': self.model_type,
            'device': str(self.device),
            'input_size': self.input_size,
            'conf_threshold': self.conf_threshold,
        }
