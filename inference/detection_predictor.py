"""
Detection Predictor - YOLOv11 meter detection inference
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from .base_predictor import BasePredictor, PredictionResult, ImageLoader

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result"""
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class DetectionPredictor(BasePredictor):
    """
    YOLOv11 Detection Predictor with GPU optimization.

    Supports:
    - Ultralytics YOLO models (.pt)
    - ONNX models (.onnx)
    - Automatic device selection
    - Batch inference
    """

    DEFAULT_CLASSES = ['meter']

    def __init__(self, model_path: str, device: str = 'auto', **kwargs):
        """
        Initialize detection predictor.

        Args:
            model_path: Path to model (.pt or .onnx)
            device: Device to use
            **kwargs: Additional options
                - conf_threshold: Confidence threshold (default: 0.25)
                - iou_threshold: NMS IoU threshold (default: 0.45)
                - input_size: Input size (default: 640)
                - class_names: List of class names
        """
        self.conf_threshold = kwargs.get('conf_threshold', 0.25)
        self.iou_threshold = kwargs.get('iou_threshold', 0.45)
        self.class_names = kwargs.get('class_names', self.DEFAULT_CLASSES)

        super().__init__(model_path, device, **kwargs)

    def _load_model(self):
        """Load detection model"""
        suffix = self.model_path.suffix.lower()

        if suffix in ['.pt', '.pth']:
            self._load_ultralytics_model()
        elif suffix == '.onnx':
            self._load_onnx_model()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def _load_ultralytics_model(self):
        """Load Ultralytics YOLO model"""
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics required. Install: pip install ultralytics")

        logger.info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.model_type = 'ultralytics'

        # Update class names from model
        if hasattr(self.model, 'names'):
            self.class_names = list(self.model.names.values())

        logger.info(f"YOLO model loaded, classes: {self.class_names}")

    def _load_onnx_model(self):
        """Load ONNX detection model"""
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
        if len(input_info.shape) == 4:
            self.input_size = (input_info.shape[2], input_info.shape[3])

        logger.info(f"ONNX model loaded, input size: {self.input_size}")

    def _get_device_string(self) -> str:
        """Get device string for YOLO"""
        if self.device_manager.is_cuda:
            return '0'
        elif self.device_manager.is_mps:
            return 'mps'
        return 'cpu'

    def predict(self, image: Union[str, np.ndarray]) -> PredictionResult:
        """
        Run detection on single image.

        Args:
            image: Image path or numpy array (BGR)

        Returns:
            PredictionResult with detections
        """
        start_time = time.time()

        try:
            # Load image
            if isinstance(image, (str, Path)):
                image_bgr, image_path = ImageLoader.load(image)
            else:
                image_bgr = image
                image_path = None

            # Run detection
            if self.model_type == 'ultralytics':
                detections = self._predict_ultralytics(image_bgr)
            else:
                detections = self._predict_onnx(image_bgr)

            inference_time = (time.time() - start_time) * 1000

            # Create visualization
            visualization = self.visualize(image_bgr, detections)

            return PredictionResult(
                success=True,
                image_path=image_path,
                predictions=detections,
                visualization=visualization,
                metadata={
                    'num_detections': len(detections),
                    'class_names': self.class_names,
                },
                inference_time_ms=inference_time,
            )

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return PredictionResult(
                success=False,
                image_path=str(image) if isinstance(image, (str, Path)) else None,
                error=str(e),
            )

    def _predict_ultralytics(self, image: np.ndarray) -> List[Detection]:
        """Run Ultralytics YOLO prediction"""
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self._get_device_string(),
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = r.names.get(cls_id, f'class_{cls_id}')

                    detections.append(Detection(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    ))

        return detections

    def _predict_onnx(self, image: np.ndarray) -> List[Detection]:
        """Run ONNX prediction"""
        # Preprocess
        h, w = image.shape[:2]
        input_h, input_w = self.input_size

        # Resize and normalize
        resized = cv2.resize(image, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        tensor = normalized.transpose(2, 0, 1)
        batch = np.expand_dims(tensor, axis=0)

        # Run inference
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: batch})
        output = outputs[0]

        # Parse YOLO output format
        detections = self._parse_yolo_output(output, w, h)

        return detections

    def _parse_yolo_output(self, output: np.ndarray, orig_w: int, orig_h: int) -> List[Detection]:
        """Parse YOLO ONNX output"""
        detections = []

        # Output shape: (1, num_detections, 5+num_classes) or (1, 5+num_classes, num_detections)
        if output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)

        output = output[0]  # Remove batch dimension

        for detection in output:
            # Format: [x, y, w, h, conf, class_scores...]
            x, y, w, h = detection[:4]
            conf = detection[4]

            if conf < self.conf_threshold:
                continue

            # Get class with highest score
            class_scores = detection[5:]
            cls_id = int(np.argmax(class_scores))
            cls_conf = class_scores[cls_id] if len(class_scores) > 0 else 1.0

            # Convert to bbox
            scale_x = orig_w / self.input_size[1]
            scale_y = orig_h / self.input_size[0]

            x1 = int((x - w/2) * scale_x)
            y1 = int((y - h/2) * scale_y)
            x2 = int((x + w/2) * scale_x)
            y2 = int((y + h/2) * scale_y)

            # Clip
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'

            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                confidence=float(conf * cls_conf),
                class_id=cls_id,
                class_name=cls_name,
            ))

        # NMS
        detections = self._nms(detections)

        return detections

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if self._iou(best.bbox, d.bbox) < self.iou_threshold
            ]

        return keep

    def _iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def visualize(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection results on image.

        Args:
            image: Original image (BGR)
            detections: List of Detection objects

        Returns:
            Visualization image (BGR)
        """
        vis = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            conf = det.confidence
            cls_name = det.class_name

            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Label background
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)

            # Label text
            cv2.putText(vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return vis

    def crop_detections(self, image: np.ndarray, detections: List[Detection],
                        padding: int = 20) -> List[np.ndarray]:
        """
        Crop detected regions from image.

        Args:
            image: Original image
            detections: List of detections
            padding: Padding around bounding box

        Returns:
            List of cropped images
        """
        h, w = image.shape[:2]
        crops = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            crop = image[y1:y2, x1:x2]
            crops.append(crop)

        return crops

    def detect_and_crop(self, image: Union[str, np.ndarray],
                        padding: int = 20) -> List[Dict[str, Any]]:
        """
        Detect and crop all meters in image.

        Args:
            image: Input image
            padding: Padding for crops

        Returns:
            List of dicts with 'detection', 'crop', 'bbox'
        """
        # Load image
        if isinstance(image, (str, Path)):
            image_bgr, _ = ImageLoader.load(image)
        else:
            image_bgr = image

        # Detect
        result = self.predict(image_bgr)
        if not result.success or not result.predictions:
            return []

        # Crop
        crops = self.crop_detections(image_bgr, result.predictions, padding)

        # Combine results
        results = []
        for det, crop in zip(result.predictions, crops):
            results.append({
                'detection': det,
                'crop': crop,
                'bbox': det.bbox,
                'confidence': det.confidence,
            })

        return results
