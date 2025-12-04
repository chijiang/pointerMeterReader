
  training/
  ├── __init__.py           # Module exports
  ├── device_manager.py     # GPU/MPS/CPU device management
  ├── config_manager.py     # Unified configuration handling
  ├── base_trainer.py       # Abstract base trainer with GPU optimization
  ├── detection_trainer.py  # YOLOv11 detection trainer
  └── segmentation_trainer.py  # SegFormer segmentation trainer

  train.py                  # Unified entry point

  Key Features:

  1. Automatic Device Selection
    - CUDA GPU detection with memory-based batch size recommendations
    - Apple Silicon MPS support
    - CPU fallback
  2. GPU Optimization
    - Mixed precision training (AMP) for CUDA
    - cuDNN benchmark mode
    - TF32 acceleration on Ampere+ GPUs
    - Gradient accumulation for effective larger batches
    - Memory fraction control
  3. Unified Interface
  # Train detection
  python train.py --task detection --config config/train_yolo11m_detection.yaml

  # Train segmentation
  python train.py --task segmentation --config config/segformer_config.yaml

  # With GPU flags
  python train.py --task detection --config ... --batch-size 32 --amp --workers 8
  4. Environment Variable Overrides
    - TRAIN_BATCH_SIZE, TRAIN_EPOCHS, TRAIN_LR
    - TRAIN_DEVICE, TRAIN_WORKERS, TRAIN_AMP

  Running Training:

  # Detection (YOLOv11m)
  python train.py --task detection --config config/train_yolo11m_detection.yaml

  # Segmentation (SegFormer-B2)
  python train.py --task segmentation --config config/segformer_config.yaml
