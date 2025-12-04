"""
Batch segmentation using ONNX model.

Supports both DeepLabV3+ and SegFormer ONNX models.

Given an input directory of images, this script will:
1) Load the ONNX segmentation model (pointer/scale).
2) Run segmentation on each image.
3) Save per-pixel masks and optional overlays blended on the original image.

Usage:
    # Using app.py's MeterSegmentor (legacy DeepLabV3+)
    python tools/batch_segment.py \
        --input-dir data/some_images \
        --output-dir outputs/segmentation_masks \
        --onnx models/segmentation/segmentation_model.onnx \
        --device cpu \
        --save-overlay

    # Using standalone ONNX inference (SegFormer or any model)
    python tools/batch_segment.py \
        --input-dir data/some_images \
        --output-dir outputs/segmentation_masks \
        --onnx models/segmentation/segformer_meter.onnx \
        --standalone \
        --device cpu \
        --save-overlay
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class StandaloneONNXSegmentor:
    """Standalone ONNX segmentation without app.py dependency.

    Works with any ONNX segmentation model (DeepLabV3+, SegFormer, etc.)
    """

    # Class colors (BGR)
    CLASS_COLORS = {
        0: (0, 0, 0),       # background - black
        1: (0, 0, 255),     # pointer - red (BGR)
        2: (0, 255, 0),     # scale - green (BGR)
    }

    def __init__(self, model_path: str, device: str = 'cpu', input_size: Tuple[int, int] = None):
        """
        Initialize ONNX segmentor.

        Args:
            model_path: Path to ONNX model
            device: 'cpu' or 'cuda'
            input_size: Optional (height, width) to override model's expected size
        """
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        self.model_path = model_path
        self.device = device
        self.input_size = input_size

        # Load model
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get input info
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name

        # Try to get input size from model
        if input_size is None:
            shape = input_info.shape
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self.input_size = (shape[2], shape[3])
            else:
                # Default size
                self.input_size = (480, 480)

        print(f"Loaded ONNX model: {model_path}")
        print(f"Input size: {self.input_size}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image for inference."""
        original_size = (image.shape[1], image.shape[0])  # (width, height)

        # Resize
        h, w = self.input_size
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize
        rgb = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

        # NHWC to NCHW
        tensor = rgb.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)

        return tensor.astype(np.float32), original_size

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Run segmentation on an image.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            Segmentation mask (H, W) with class indices
        """
        # Preprocess
        tensor, original_size = self.preprocess(image)

        # Run inference
        outputs = self.session.run(None, {self.input_name: tensor})
        output = outputs[0]  # (1, C, H, W) or (1, H, W)

        # Get predictions
        if len(output.shape) == 4:
            # (1, C, H, W) - need argmax
            mask = np.argmax(output[0], axis=0)  # (H, W)
        else:
            # (1, H, W) - already class indices
            mask = output[0]

        # Resize back to original size
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

        return mask

    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to colored visualization."""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in self.CLASS_COLORS.items():
            colored[mask == cls_id] = color
        return colored


def segment_image_standalone(segmentor: StandaloneONNXSegmentor, img_path: Path):
    """Segment image using standalone segmentor."""
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Failed to read image: {img_path}")
    mask = segmentor.segment(image)
    return image, mask


def segment_image_legacy(segmentor, img_path: Path):
    """Segment image using app.py's MeterSegmentor."""
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Failed to read image: {img_path}")
    mask = segmentor.segment_meter(image)
    return image, mask


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map classes to colors: 0=black, 1=red, 2=green."""
    colors = {
        0: (0, 0, 0),       # background
        1: (0, 0, 255),     # pointer (BGR: red)
        2: (0, 255, 0),     # scale (BGR: green)
    }
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, c in colors.items():
        color[mask == cls] = c
    return color


def main():
    parser = argparse.ArgumentParser(description="Batch segmentation with ONNX model")
    parser.add_argument("--input-dir", required=True, help="Directory of input images")
    parser.add_argument("--output-dir", required=True, help="Directory to save masks (png)")
    parser.add_argument("--onnx", default="models/segmentation/segmentation_model.onnx",
                       help="Path to ONNX model")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--save-overlay", action="store_true",
                       help="Also save mask overlaid on the original image")
    parser.add_argument("--standalone", action="store_true",
                       help="Use standalone ONNX inference (no app.py dependency)")
    parser.add_argument("--input-size", type=int, nargs=2, default=None,
                       help="Input size (height width) for standalone mode")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlay"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    # Initialize segmentor
    if args.standalone:
        input_size = tuple(args.input_size) if args.input_size else None
        segmentor = StandaloneONNXSegmentor(args.onnx, device=args.device, input_size=input_size)
        segment_func = segment_image_standalone
    else:
        from app import MeterSegmentor
        segmentor = MeterSegmentor(args.onnx, device=args.device)
        segment_func = segment_image_legacy

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"Found {len(images)} images. Running segmentation...")
    for img_path in images:
        try:
            image, mask = segment_func(segmentor, img_path)
            out_mask = mask_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(out_mask), mask)

            if args.save_overlay:
                color = colorize_mask(mask)
                overlay = cv2.addWeighted(image, 0.6, color, 0.4, 0)
                over_path = overlay_dir / f"{img_path.stem}_overlay.png"
                cv2.imwrite(str(over_path), overlay)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed on {img_path}: {exc}")

    print(f"Done. Masks in: {mask_dir}")
    if args.save_overlay:
        print(f"Overlays in: {overlay_dir}")


if __name__ == "__main__":
    main()
