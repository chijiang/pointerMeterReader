"""
Pipeline Predictor - Complete meter reading pipeline
Combines detection + segmentation + reading extraction
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import cv2

from .base_predictor import PredictionResult, ImageLoader
from .detection_predictor import DetectionPredictor, Detection
from .segmentation_predictor import SegmentationPredictor

logger = logging.getLogger(__name__)


@dataclass
class MeterReading:
    """Single meter reading result"""
    meter_id: int
    reading: Optional[float]
    confidence: float
    bbox: List[int]
    pointer_angle: Optional[float] = None
    scale_range: Tuple[float, float] = (0.0, 1.6)


@dataclass
class PipelineResult:
    """Complete pipeline result"""
    success: bool
    num_meters: int = 0
    readings: List[MeterReading] = field(default_factory=list)
    visualizations: Dict[str, np.ndarray] = field(default_factory=dict)
    error: Optional[str] = None
    inference_time_ms: float = 0.0


class PipelinePredictor:
    """
    Complete Meter Reading Pipeline.

    Pipeline:
    1. Detection: Find meters in image (YOLO)
    2. Cropping: Extract meter regions
    3. Segmentation: Identify pointer and scale (SegFormer)
    4. Reading: Extract numeric value

    Features:
    - GPU-optimized inference
    - Batch processing
    - Configurable scale range
    - Visualization output
    """

    def __init__(self,
                 detection_model: str,
                 segmentation_model: str,
                 device: str = 'auto',
                 **kwargs):
        """
        Initialize pipeline.

        Args:
            detection_model: Path to detection model
            segmentation_model: Path to segmentation model
            device: Device to use
            **kwargs: Additional options
                - scale_min: Minimum scale value (default: 0.0)
                - scale_max: Maximum scale value (default: 1.6)
                - conf_threshold: Detection confidence (default: 0.25)
                - crop_padding: Padding for cropping (default: 20)
        """
        self.scale_min = kwargs.get('scale_min', 0.0)
        self.scale_max = kwargs.get('scale_max', 1.6)
        self.conf_threshold = kwargs.get('conf_threshold', 0.25)
        self.crop_padding = kwargs.get('crop_padding', 20)

        # Initialize predictors
        logger.info("Initializing detection model...")
        self.detector = DetectionPredictor(
            detection_model,
            device=device,
            conf_threshold=self.conf_threshold,
        )

        logger.info("Initializing segmentation model...")
        self.segmentor = SegmentationPredictor(
            segmentation_model,
            device=device,
        )

        # Initialize meter reader
        self.reader = self._init_reader()

        logger.info("Pipeline initialized")

    def _init_reader(self):
        """Initialize meter reading module"""
        try:
            from scripts.meter_reading import MeterReader
            return MeterReader(
                scale_range=(self.scale_min, self.scale_max),
                debug=False
            )
        except ImportError:
            logger.warning("MeterReader not available, reading extraction disabled")
            return None

    def predict(self, image: Union[str, np.ndarray],
                scale_min: float = None,
                scale_max: float = None) -> PipelineResult:
        """
        Run complete pipeline on single image.

        Args:
            image: Image path or numpy array (BGR)
            scale_min: Override minimum scale value
            scale_max: Override maximum scale value

        Returns:
            PipelineResult with readings and visualizations
        """
        start_time = time.time()
        scale_range = (
            scale_min if scale_min is not None else self.scale_min,
            scale_max if scale_max is not None else self.scale_max,
        )

        try:
            # Load image
            if isinstance(image, (str, Path)):
                image_bgr, image_path = ImageLoader.load(image)
            else:
                image_bgr = image
                image_path = None

            # Step 1: Detection
            det_result = self.detector.predict(image_bgr)
            if not det_result.success or not det_result.predictions:
                return PipelineResult(
                    success=False,
                    error="No meters detected in image",
                    visualizations={'detection': det_result.visualization} if det_result.visualization is not None else {},
                )

            detections = det_result.predictions
            logger.info(f"Detected {len(detections)} meter(s)")

            # Step 2-4: Process each meter
            readings = []
            visualizations = {
                'detection': det_result.visualization,
            }

            for i, detection in enumerate(detections):
                meter_result = self._process_meter(
                    image_bgr, detection, i, scale_range
                )

                if meter_result:
                    readings.append(meter_result['reading'])
                    visualizations[f'meter_{i}_crop'] = meter_result['crop']
                    visualizations[f'meter_{i}_segmentation'] = meter_result['segmentation']
                    if meter_result.get('result'):
                        visualizations[f'meter_{i}_result'] = meter_result['result']

            inference_time = (time.time() - start_time) * 1000

            return PipelineResult(
                success=len(readings) > 0,
                num_meters=len(detections),
                readings=readings,
                visualizations=visualizations,
                inference_time_ms=inference_time,
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return PipelineResult(
                success=False,
                error=str(e),
            )

    def _process_meter(self, image: np.ndarray, detection: Detection,
                       meter_id: int, scale_range: Tuple[float, float]) -> Optional[Dict]:
        """Process single detected meter"""
        try:
            # Crop meter region
            x1, y1, x2, y2 = detection.bbox
            h, w = image.shape[:2]
            padding = self.crop_padding

            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(w, x2 + padding)
            crop_y2 = min(h, y2 + padding)

            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

            # Segment
            seg_result = self.segmentor.predict(cropped)
            if not seg_result.success:
                logger.warning(f"Segmentation failed for meter {meter_id}")
                return None

            mask = seg_result.predictions.mask

            # Extract reading
            reading_value = None
            result_vis = None

            if self.reader:
                try:
                    self.reader.scale_beginning = scale_range[0]
                    self.reader.scale_end = scale_range[1]
                    reading_value = self.reader.process_single_meter(cropped, mask)

                    if reading_value is not None:
                        result_vis = self._create_result_visualization(
                            cropped, mask, reading_value
                        )
                except Exception as e:
                    logger.warning(f"Reading extraction failed: {e}")

            return {
                'reading': MeterReading(
                    meter_id=meter_id,
                    reading=reading_value,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    scale_range=scale_range,
                ),
                'crop': cropped,
                'segmentation': seg_result.visualization,
                'result': result_vis,
            }

        except Exception as e:
            logger.warning(f"Error processing meter {meter_id}: {e}")
            return None

    def _create_result_visualization(self, image: np.ndarray,
                                     mask: np.ndarray,
                                     reading: float) -> np.ndarray:
        """Create result visualization with reading overlay"""
        vis = image.copy()

        # Add reading text
        text = f"Reading: {reading:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Get text size
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Draw background
        cv2.rectangle(vis, (5, 5), (text_size[0] + 15, text_size[1] + 15), (0, 0, 0), -1)

        # Draw text
        cv2.putText(vis, text, (10, text_size[1] + 10), font, font_scale, (0, 255, 0), thickness)

        return vis

    def predict_batch(self, images: List[Union[str, np.ndarray]],
                      scale_min: float = None,
                      scale_max: float = None) -> List[PipelineResult]:
        """
        Run pipeline on multiple images.

        Args:
            images: List of images
            scale_min: Scale minimum
            scale_max: Scale maximum

        Returns:
            List of PipelineResult
        """
        results = []
        from tqdm import tqdm

        for image in tqdm(images, desc="Processing"):
            result = self.predict(image, scale_min, scale_max)
            results.append(result)

        return results

    def predict_directory(self, directory: str,
                          output_dir: str = None,
                          scale_min: float = None,
                          scale_max: float = None,
                          save_visualizations: bool = True) -> List[PipelineResult]:
        """
        Run pipeline on directory of images.

        Args:
            directory: Input directory
            output_dir: Output directory
            scale_min: Scale minimum
            scale_max: Scale maximum
            save_visualizations: Save visualization images

        Returns:
            List of PipelineResult
        """
        images = ImageLoader.find_images(directory)
        if not images:
            logger.warning(f"No images found in {directory}")
            return []

        logger.info(f"Found {len(images)} images")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        from tqdm import tqdm

        for image_path in tqdm(images, desc="Processing"):
            result = self.predict(str(image_path), scale_min, scale_max)
            results.append(result)

            # Save visualizations
            if save_visualizations and output_dir and result.visualizations:
                stem = image_path.stem
                for name, vis in result.visualizations.items():
                    if vis is not None:
                        vis_path = output_dir / f"{stem}_{name}.jpg"
                        cv2.imwrite(str(vis_path), vis)

            # Save readings to text
            if output_dir and result.readings:
                txt_path = output_dir / f"{image_path.stem}_readings.txt"
                with open(txt_path, 'w') as f:
                    for reading in result.readings:
                        if reading.reading is not None:
                            f.write(f"Meter {reading.meter_id}: {reading.reading:.4f}\n")
                        else:
                            f.write(f"Meter {reading.meter_id}: N/A\n")

        return results

    def get_readings_only(self, image: Union[str, np.ndarray],
                          scale_min: float = None,
                          scale_max: float = None) -> List[float]:
        """
        Get meter readings without visualizations.

        Args:
            image: Input image
            scale_min: Scale minimum
            scale_max: Scale maximum

        Returns:
            List of reading values
        """
        result = self.predict(image, scale_min, scale_max)

        if not result.success:
            return []

        readings = []
        for r in result.readings:
            if r.reading is not None:
                readings.append(r.reading)

        return readings

    def warmup(self):
        """Warmup both models"""
        logger.info("Warming up models...")
        self.detector.warmup()
        self.segmentor.warmup()
        logger.info("Warmup complete")
