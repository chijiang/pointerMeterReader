#!/usr/bin/env python3
"""
Unified Training Pipeline for Meter Reading Models

This script provides a single entry point for training both:
- YOLOv11 detection models (meter localization)
- SegFormer segmentation models (pointer/scale identification)

Features:
- Automatic GPU/MPS/CPU device selection
- Mixed precision training
- Gradient accumulation for large effective batch sizes
- Memory optimization
- Unified configuration system
- ONNX export after training

Usage:
    # Train detection model
    python train.py --task detection --config config/train_yolo11m_detection.yaml

    # Train segmentation model
    python train.py --task segmentation --config config/segformer_config.yaml

    # Train with GPU optimization flags
    python train.py --task detection --config config/train_yolo11m_detection.yaml \
        --batch-size 32 --amp --workers 8

    # Resume training
    python train.py --task detection --config config/train_yolo11m_detection.yaml --resume

    # Export only (no training)
    python train.py --task detection --config config/train_yolo11m_detection.yaml --export-only

    # Evaluate model
    python train.py --task detection --config config/train_yolo11m_detection.yaml --eval-only

Environment Variables:
    TRAIN_BATCH_SIZE: Override batch size
    TRAIN_EPOCHS: Override epochs
    TRAIN_LR: Override learning rate
    TRAIN_DEVICE: Override device (cuda, mps, cpu)
    TRAIN_WORKERS: Override number of workers
    TRAIN_AMP: Enable/disable mixed precision (true/false)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config_manager import ConfigManager
from training.device_manager import DeviceManager


def setup_logging(level: str = 'INFO'):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def print_banner():
    """Print training banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║            Meter Reading Model Training Pipeline                 ║
║                                                                  ║
║  Detection: YOLOv11 (meter localization)                         ║
║  Segmentation: SegFormer (pointer/scale identification)          ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def train_detection(config: ConfigManager, args):
    """Train detection model"""
    from training.detection_trainer import DetectionTrainer

    trainer = DetectionTrainer(config)

    if args.eval_only:
        metrics = trainer.evaluate(args.model)
        return metrics

    if args.export_only:
        exported = trainer.export(args.model)
        return exported

    # Train
    best_model = trainer.train(resume=args.resume)

    # Export after training
    if not args.no_export:
        trainer.export(best_model)

    # Evaluate
    metrics = trainer.evaluate(best_model)

    return best_model


def train_segmentation(config: ConfigManager, args):
    """Train segmentation model"""
    from training.segmentation_trainer import SegmentationTrainer

    trainer = SegmentationTrainer(config)
    trainer.setup()

    if args.eval_only:
        metrics = trainer.evaluate(args.model)
        return metrics

    if args.export_only:
        trainer.export_model()
        return

    # Train
    best_score = trainer.train(resume=args.resume)

    return best_score


def main():
    parser = argparse.ArgumentParser(
        description='Unified Training Pipeline for Meter Reading Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Task selection
    parser.add_argument('--task', type=str, required=True,
                        choices=['detection', 'segmentation'],
                        help='Training task: detection or segmentation')

    # Configuration
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')

    # Training modes
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--export-only', action='store_true',
                        help='Only export model (no training)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate model (no training)')
    parser.add_argument('--no-export', action='store_true',
                        help='Skip export after training')

    # Model path (for export/eval)
    parser.add_argument('--model', type=str, default=None,
                        help='Model path for export/evaluation')

    # GPU optimization overrides
    parser.add_argument('--device', type=str, default=None,
                        help='Device: auto, cuda, cuda:0, mps, cpu')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (0 or -1 for auto)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of data loader workers')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')

    # Training overrides
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')

    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Print banner
    print_banner()

    # Check config file
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load configuration
    logger.info(f"Loading configuration: {args.config}")
    config = ConfigManager(args.config)

    # Apply command-line overrides
    if args.device:
        config.update(device=args.device)
    if args.batch_size is not None:
        config.update(batch_size=args.batch_size)
    if args.workers is not None:
        config.update(num_workers=args.workers)
    if args.epochs is not None:
        config.update(epochs=args.epochs)
    if args.lr is not None:
        config.update(learning_rate=args.lr)
    if args.amp:
        config.update(mixed_precision=True)
    if args.no_amp:
        config.update(mixed_precision=False)

    # Show device info
    device_manager = DeviceManager(config.config.device)
    logger.info(f"Device: {device_manager.device_info.device_name}")
    logger.info(f"Mixed Precision: {config.config.mixed_precision and device_manager.supports_amp}")

    # Run task
    try:
        if args.task == 'detection':
            logger.info("Starting detection training (YOLOv11)")
            result = train_detection(config, args)
        else:
            logger.info("Starting segmentation training (SegFormer)")
            result = train_segmentation(config, args)

        logger.info("Training pipeline completed successfully!")

        if result:
            logger.info(f"Result: {result}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
