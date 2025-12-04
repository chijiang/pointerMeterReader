#!/usr/bin/env python3
"""
Unified Prediction Pipeline for Meter Reading Models

This script provides a single entry point for inference:
- Detection: Find meters in images (YOLOv11)
- Segmentation: Identify pointer and scale (SegFormer)
- Pipeline: Complete detection + segmentation + reading extraction

Features:
- Automatic GPU/MPS/CPU device selection
- ONNX and PyTorch model support
- Single image and batch processing
- Visualization output

Usage:
    # Detection only
    python predict.py --task detection --source image.jpg --model models/detection/yolo11_meter.pt

    # Segmentation only
    python predict.py --task segmentation --source image.jpg --model models/segmentation/segformer_meter.onnx

    # Complete pipeline
    python predict.py --task pipeline --source image.jpg \
        --detection-model models/detection/yolo11_meter.pt \
        --segmentation-model models/segmentation/segformer_meter.onnx

    # Batch processing
    python predict.py --task detection --source data/images/ --output outputs/predictions/

    # Custom settings
    python predict.py --task pipeline --source image.jpg --conf 0.5 --scale-min 0 --scale-max 2.0
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(level: str = 'INFO'):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def print_banner():
    """Print prediction banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║           Meter Reading Prediction Pipeline                      ║
║                                                                  ║
║  Detection: YOLOv11 (meter localization)                         ║
║  Segmentation: SegFormer (pointer/scale identification)          ║
║  Pipeline: Complete meter reading extraction                     ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def find_model(task: str, model_type: str = 'pt') -> Optional[str]:
    """Auto-find model file"""
    search_paths = [
        Path(f"models/detection") if task == 'detection' else Path(f"models/segmentation"),
        Path("outputs/detection/checkpoints") if task == 'detection' else Path("outputs/segmentation/checkpoints"),
        Path("runs_gauge"),
        Path("runs"),
    ]

    extensions = ['.pt', '.pth', '.onnx'] if model_type == 'any' else [f'.{model_type}']

    for base_path in search_paths:
        if base_path.exists():
            for ext in extensions:
                for model_file in base_path.rglob(f'*best*{ext}'):
                    return str(model_file)
                for model_file in base_path.rglob(f'*{ext}'):
                    return str(model_file)

    return None


def run_detection(args):
    """Run detection inference"""
    from inference.detection_predictor import DetectionPredictor

    # Find model
    model_path = args.model
    if not model_path:
        model_path = find_model('detection')
        if not model_path:
            print("Error: No detection model found. Specify with --model")
            sys.exit(1)
        print(f"Auto-detected model: {model_path}")

    # Create predictor
    predictor = DetectionPredictor(
        model_path=model_path,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    # Warmup
    if args.warmup:
        predictor.warmup()

    # Process
    source = Path(args.source)
    output_dir = Path(args.output) if args.output else None

    if source.is_dir():
        results = predictor.predict_directory(
            str(source),
            output_dir=str(output_dir) if output_dir else None,
            save_visualization=args.save,
        )
        print(f"\nProcessed {len(results)} images")

        # Summary
        total_detections = sum(
            len(r.predictions) for r in results if r.success and r.predictions
        )
        print(f"Total detections: {total_detections}")

    else:
        result = predictor.predict(str(source))

        if result.success:
            print(f"\nDetections: {len(result.predictions)}")
            for det in result.predictions:
                print(f"  {det.class_name}: {det.confidence:.2f} @ {det.bbox}")

            # Save
            if args.save and output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                import cv2
                cv2.imwrite(str(output_dir / f"{source.stem}_detection.jpg"), result.visualization)
                print(f"Saved to: {output_dir}")
        else:
            print(f"Error: {result.error}")


def run_segmentation(args):
    """Run segmentation inference"""
    from inference.segmentation_predictor import SegmentationPredictor

    # Find model
    model_path = args.model
    if not model_path:
        model_path = find_model('segmentation', 'any')
        if not model_path:
            print("Error: No segmentation model found. Specify with --model")
            sys.exit(1)
        print(f"Auto-detected model: {model_path}")

    # Create predictor
    predictor = SegmentationPredictor(
        model_path=model_path,
        device=args.device,
        post_process=not args.no_postprocess,
        alpha=args.alpha,
    )

    # Warmup
    if args.warmup:
        predictor.warmup()

    # Process
    source = Path(args.source)
    output_dir = Path(args.output) if args.output else None

    if source.is_dir():
        results = predictor.predict_directory(
            str(source),
            output_dir=str(output_dir) if output_dir else None,
            save_visualization=args.save,
        )
        print(f"\nProcessed {len(results)} images")

    else:
        result = predictor.predict(str(source))

        if result.success:
            print(f"\nSegmentation complete")
            print(f"Class areas: {result.metadata['class_areas']}")
            print(f"Class ratios: {result.metadata['class_ratios']}")

            # Save
            if args.save and output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                import cv2
                cv2.imwrite(str(output_dir / f"{source.stem}_segmentation.jpg"), result.visualization)

                # Save mask
                if args.save_mask:
                    cv2.imwrite(str(output_dir / f"{source.stem}_mask.png"), result.predictions.mask)

                print(f"Saved to: {output_dir}")
        else:
            print(f"Error: {result.error}")


def run_pipeline(args):
    """Run complete pipeline"""
    from inference.pipeline_predictor import PipelinePredictor

    # Find models
    det_model = args.detection_model
    if not det_model:
        det_model = find_model('detection')
        if not det_model:
            print("Error: No detection model found. Specify with --detection-model")
            sys.exit(1)
        print(f"Auto-detected detection model: {det_model}")

    seg_model = args.segmentation_model
    if not seg_model:
        seg_model = find_model('segmentation', 'any')
        if not seg_model:
            print("Error: No segmentation model found. Specify with --segmentation-model")
            sys.exit(1)
        print(f"Auto-detected segmentation model: {seg_model}")

    # Create pipeline
    pipeline = PipelinePredictor(
        detection_model=det_model,
        segmentation_model=seg_model,
        device=args.device,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        conf_threshold=args.conf,
        crop_padding=args.padding,
    )

    # Warmup
    if args.warmup:
        pipeline.warmup()

    # Process
    source = Path(args.source)
    output_dir = Path(args.output) if args.output else None

    if source.is_dir():
        results = pipeline.predict_directory(
            str(source),
            output_dir=str(output_dir) if output_dir else None,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            save_visualizations=args.save,
        )

        # Summary
        print(f"\nProcessed {len(results)} images")
        successful = sum(1 for r in results if r.success)
        print(f"Successful: {successful}/{len(results)}")

        # Print all readings
        all_readings = []
        for i, result in enumerate(results):
            for reading in result.readings:
                if reading.reading is not None:
                    all_readings.append(reading.reading)
                    print(f"Image {i}, Meter {reading.meter_id}: {reading.reading:.4f}")

        if all_readings:
            print(f"\nTotal readings: {len(all_readings)}")
            print(f"Average: {sum(all_readings) / len(all_readings):.4f}")

    else:
        result = pipeline.predict(str(source), args.scale_min, args.scale_max)

        if result.success:
            print(f"\nFound {result.num_meters} meter(s)")
            print(f"Inference time: {result.inference_time_ms:.1f} ms")

            for reading in result.readings:
                if reading.reading is not None:
                    print(f"  Meter {reading.meter_id}: {reading.reading:.4f} (conf: {reading.confidence:.2f})")
                else:
                    print(f"  Meter {reading.meter_id}: Unable to extract reading")

            # Save
            if args.save and output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                import cv2

                for name, vis in result.visualizations.items():
                    if vis is not None:
                        cv2.imwrite(str(output_dir / f"{source.stem}_{name}.jpg"), vis)

                # Save readings JSON
                readings_data = []
                for r in result.readings:
                    readings_data.append({
                        'meter_id': r.meter_id,
                        'reading': r.reading,
                        'confidence': r.confidence,
                        'bbox': r.bbox,
                    })

                with open(output_dir / f"{source.stem}_readings.json", 'w') as f:
                    json.dump(readings_data, f, indent=2)

                print(f"Saved to: {output_dir}")
        else:
            print(f"Error: {result.error}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Prediction Pipeline for Meter Reading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Task selection
    parser.add_argument('--task', type=str, required=True,
                        choices=['detection', 'segmentation', 'pipeline'],
                        help='Prediction task')

    # Input/Output
    parser.add_argument('--source', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--save', action='store_true',
                        help='Save visualization results')
    parser.add_argument('--save-mask', action='store_true',
                        help='Save segmentation masks (segmentation task)')

    # Model paths
    parser.add_argument('--model', type=str, default=None,
                        help='Model path (for detection/segmentation)')
    parser.add_argument('--detection-model', type=str, default=None,
                        help='Detection model path (for pipeline)')
    parser.add_argument('--segmentation-model', type=str, default=None,
                        help='Segmentation model path (for pipeline)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use')

    # Detection settings
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')

    # Segmentation settings
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Visualization overlay alpha')
    parser.add_argument('--no-postprocess', action='store_true',
                        help='Disable segmentation post-processing')

    # Pipeline settings
    parser.add_argument('--scale-min', type=float, default=0.0,
                        help='Meter scale minimum value')
    parser.add_argument('--scale-max', type=float, default=1.6,
                        help='Meter scale maximum value')
    parser.add_argument('--padding', type=int, default=20,
                        help='Crop padding around detected meters')

    # Performance
    parser.add_argument('--warmup', action='store_true',
                        help='Warmup model before inference')

    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    print_banner()

    # Validate source
    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source not found: {args.source}")
        sys.exit(1)

    # Run task
    try:
        if args.task == 'detection':
            run_detection(args)
        elif args.task == 'segmentation':
            run_segmentation(args)
        else:
            run_pipeline(args)

        print("\nPrediction complete!")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise


if __name__ == '__main__':
    main()
