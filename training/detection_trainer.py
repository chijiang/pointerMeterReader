"""
Detection Trainer - YOLOv11 meter detection training with GPU optimization

Uses ultralytics YOLO API with optimized settings for GPU training.
"""

import os
import sys
import json
import shutil
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

from .device_manager import DeviceManager
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class DetectionTrainer:
    """
    YOLOv11 Detection Trainer with GPU optimization.

    Features:
    - Automatic dataset format conversion (COCO -> YOLO)
    - Roboflow format support
    - GPU optimization (CUDA, MPS)
    - Mixed precision training
    - Automatic batch size adjustment
    - ONNX export
    """

    def __init__(self, config: ConfigManager):
        """
        Initialize detection trainer.

        Args:
            config: ConfigManager instance
        """
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics package required. Install: pip install ultralytics")

        self.config = config
        self.cfg = config.config
        self.extra = config.config.extra

        # Initialize device manager
        self.device_manager = DeviceManager(
            device_config=self.cfg.device,
            memory_fraction=0.9
        )

        # Setup directories
        self._setup_directories()

        # Model will be created during training
        self.model: Optional[YOLO] = None
        self.dataset_yaml: Optional[str] = None

        logger.info(f"Detection Trainer initialized")
        logger.info(f"Device: {self.device_manager.device_info.device_name}")

    def _setup_directories(self):
        """Create output directories"""
        self.project_root = Path(__file__).parent.parent

        # Use save_dir from config if specified
        save_dir = self.extra.get('save_dir', 'outputs/detection')
        self.output_dir = self.project_root / save_dir
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.export_dir = self.output_dir / "exported"
        self.model_dir = self.project_root / "models" / "detection"

        for d in [self.checkpoint_dir, self.export_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_device_string(self) -> str:
        """Get device string for YOLO"""
        device_type = self.device_manager.device_info.device_type

        if device_type == 'cuda':
            return '0'  # GPU index
        elif device_type == 'mps':
            return 'mps'
        else:
            return 'cpu'

    def prepare_dataset(self) -> str:
        """
        Prepare YOLO format dataset.

        Returns:
            Path to dataset.yaml
        """
        logger.info("Preparing dataset...")

        # Check if data_yaml is directly specified (already converted dataset)
        data_yaml = self.extra.get('data_yaml')
        if data_yaml:
            data_yaml_path = Path(data_yaml)
            if data_yaml_path.exists():
                logger.info(f"Using pre-converted dataset: {data_yaml}")
                return str(data_yaml_path.resolve())
            else:
                raise FileNotFoundError(f"Specified data_yaml not found: {data_yaml}")

        use_roboflow = self.extra.get('use_roboflow_format', False)

        if use_roboflow:
            return self._prepare_roboflow_dataset()
        else:
            return self._prepare_coco_dataset()

    def _prepare_roboflow_dataset(self) -> str:
        """Handle Roboflow format (pre-split train/valid/test)"""
        dataset_root = Path(self.extra.get('dataset_root', 'data/gauge.v4-release-640.coco'))
        yolo_output = Path(self.extra.get('yolo_output', 'data/gauge_yolo'))

        dataset_yaml = yolo_output / "dataset.yaml"

        # Check if already converted
        if dataset_yaml.exists():
            train_images = list((yolo_output / "images" / "train").glob("*"))
            if len(train_images) > 0:
                logger.info(f"Using existing dataset: {dataset_yaml}")
                return str(dataset_yaml)

        # Find splits
        splits = {'train': dataset_root / 'train'}

        # Roboflow uses 'valid' not 'val'
        if (dataset_root / 'valid').exists():
            splits['val'] = dataset_root / 'valid'
        elif (dataset_root / 'val').exists():
            splits['val'] = dataset_root / 'val'

        if (dataset_root / 'test').exists():
            splits['test'] = dataset_root / 'test'

        # Collect categories from all splits
        all_categories = {}
        for split_name, split_dir in splits.items():
            coco_json = split_dir / '_annotations.coco.json'
            if coco_json.exists():
                with open(coco_json, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                for cat in coco_data.get('categories', []):
                    all_categories[cat['id']] = cat['name']

        if not all_categories:
            raise ValueError("No categories found in annotations")

        # Create ID mapping
        sorted_ids = sorted(all_categories.keys())
        cat_id_map = {cat_id: idx for idx, cat_id in enumerate(sorted_ids)}
        cat_names = [all_categories[cid] for cid in sorted_ids]

        logger.info(f"Categories: {cat_names}")

        # Create output structure
        for split in ['train', 'val', 'test']:
            (yolo_output / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_output / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Convert each split
        for split_name, split_dir in splits.items():
            coco_json = split_dir / '_annotations.coco.json'
            if not coco_json.exists():
                continue

            with open(coco_json, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)

            images = coco_data.get('images', [])
            anns_dict = {}
            for ann in coco_data.get('annotations', []):
                anns_dict.setdefault(ann['image_id'], []).append(ann)

            # Map split names
            yolo_split = 'val' if split_name == 'valid' else split_name

            self._convert_split(
                images, anns_dict, cat_id_map,
                split_dir,  # Images are in split directory
                yolo_output / "images" / yolo_split,
                yolo_output / "labels" / yolo_split
            )

            logger.info(f"Converted {split_name}: {len(images)} images")

        # Create dataset.yaml
        dataset_config = {
            'path': str(yolo_output.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(cat_names),
            'names': cat_names,
        }

        if 'test' in splits:
            dataset_config['test'] = 'images/test'

        with open(dataset_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, sort_keys=False, allow_unicode=True)

        logger.info(f"Dataset prepared: {dataset_yaml}")
        return str(dataset_yaml)

    def _prepare_coco_dataset(self) -> str:
        """Handle single COCO file with auto-split"""
        dataset_root = Path(self.cfg.data_root)
        coco_json = Path(self.extra.get('coco_json', dataset_root / 'result_coco.json'))
        images_dir = Path(self.extra.get('images_dir', dataset_root / 'images'))
        yolo_output = Path(self.extra.get('yolo_output', dataset_root.parent / 'meter_yolo'))
        val_split = float(self.extra.get('val_split', 0.1))
        seed = int(self.extra.get('seed', 42))

        dataset_yaml = yolo_output / "dataset.yaml"

        # Check if exists
        if dataset_yaml.exists():
            logger.info(f"Using existing dataset: {dataset_yaml}")
            return str(dataset_yaml)

        # Find annotation file
        if not coco_json.exists():
            alt_paths = [
                dataset_root / "result_coco.json",
                dataset_root / "annotations" / "instances_train.json",
            ]
            for alt in alt_paths:
                if alt.exists():
                    coco_json = alt
                    break
            else:
                raise FileNotFoundError(f"COCO annotation not found: {coco_json}")

        logger.info(f"Loading: {coco_json}")

        with open(coco_json, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        images = coco_data.get('images', [])
        categories = sorted(coco_data.get('categories', []), key=lambda x: x['id'])
        cat_id_map = {cat['id']: i for i, cat in enumerate(categories)}
        cat_names = [cat['name'] for cat in categories] or ['meter']

        anns_dict = {}
        for ann in coco_data.get('annotations', []):
            anns_dict.setdefault(ann['image_id'], []).append(ann)

        # Shuffle and split
        rng = random.Random(seed)
        rng.shuffle(images)

        val_count = max(1, int(len(images) * val_split))
        val_images = images[:val_count]
        train_images = images[val_count:]

        logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}")

        # Create directories
        for split in ['train', 'val']:
            (yolo_output / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_output / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Convert
        self._convert_split(train_images, anns_dict, cat_id_map,
                           images_dir, yolo_output / "images" / "train",
                           yolo_output / "labels" / "train")

        self._convert_split(val_images, anns_dict, cat_id_map,
                           images_dir, yolo_output / "images" / "val",
                           yolo_output / "labels" / "val")

        # Create dataset.yaml
        dataset_config = {
            'path': str(yolo_output.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(cat_names),
            'names': cat_names,
        }

        with open(dataset_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, sort_keys=False, allow_unicode=True)

        logger.info(f"Dataset prepared: {dataset_yaml}")
        return str(dataset_yaml)

    def _convert_split(self, images: List[Dict], anns_dict: Dict,
                       cat_id_map: Dict, src_images: Path,
                       dst_images: Path, dst_labels: Path):
        """Convert a split to YOLO format"""
        for img_info in tqdm(images, desc="Converting"):
            filename = img_info['file_name']
            src_path = src_images / filename
            dst_path = dst_images / filename

            # Link or copy image
            if src_path.exists() and not dst_path.exists():
                try:
                    os.symlink(src_path.resolve(), dst_path)
                except (OSError, AttributeError):
                    shutil.copy2(src_path, dst_path)

            # Convert annotations
            w, h = img_info['width'], img_info['height']
            lines = []

            for ann in anns_dict.get(img_info['id'], []):
                if ann['category_id'] not in cat_id_map:
                    continue

                bbox = ann['bbox']  # [x, y, width, height]
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                norm_w = bbox[2] / w
                norm_h = bbox[3] / h

                # Clip values
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                cls_idx = cat_id_map[ann['category_id']]
                lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            # Save label
            label_file = dst_labels / (Path(filename).stem + ".txt")
            with open(label_file, 'w') as f:
                f.write('\n'.join(lines))

    def train(self, resume: bool = False) -> str:
        """
        Run training.

        Args:
            resume: Resume from checkpoint

        Returns:
            Path to best model
        """
        logger.info("Starting YOLOv11 training...")

        # Prepare dataset
        self.dataset_yaml = self.prepare_dataset()

        # Determine model path
        experiment_name = self.extra.get('experiment_name', 'yolo11_meter')
        last_ckpt = self.checkpoint_dir / experiment_name / "weights" / "last.pt"

        if resume and last_ckpt.exists():
            logger.info(f"Resuming from: {last_ckpt}")
            self.model = YOLO(str(last_ckpt))
            resume_flag = True
        else:
            model_name = self.cfg.model_name
            logger.info(f"Loading model: {model_name}")
            self.model = YOLO(model_name)
            resume_flag = False

        # Build training arguments
        train_args = self._build_train_args()

        logger.info("Training configuration:")
        for key, value in train_args.items():
            if key not in ['augmentation']:
                logger.info(f"  {key}: {value}")

        # Train
        try:
            results = self.model.train(**train_args, resume=resume_flag)
            best_model = Path(results.save_dir) / "weights" / "best.pt"

            if best_model.exists():
                # Copy to models directory
                final_path = self.model_dir / "yolo11_meter.pt"
                shutil.copy2(best_model, final_path)
                logger.info(f"Training complete! Best model: {final_path}")
                return str(final_path)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        return str(best_model)

    def _build_train_args(self) -> Dict[str, Any]:
        """Build YOLO training arguments with GPU optimization"""

        # Get recommended batch size if auto
        batch_size = self.cfg.batch_size
        if batch_size == 0 or batch_size == -1:
            batch_size = self.device_manager.device_info.recommended_batch_size

        # Adjust workers based on device
        workers = min(self.cfg.num_workers, self.device_manager.device_info.recommended_workers)

        # Base arguments
        args = {
            'data': self.dataset_yaml,
            'project': str(self.checkpoint_dir),
            'name': self.extra.get('experiment_name', 'yolo11_meter'),
            'epochs': self.cfg.epochs,
            'batch': batch_size,
            'imgsz': self.cfg.image_size,
            'device': self._get_device_string(),
            'workers': workers,
            'seed': self.extra.get('seed', 42),
            'patience': self.cfg.patience,
            'save_period': self.cfg.save_interval,
            'cache': self.cfg.cache_images,
            'amp': self.cfg.mixed_precision and self.device_manager.supports_amp,
            'cos_lr': self.extra.get('cos_lr', True),
            'close_mosaic': self.extra.get('close_mosaic', 15),
            'exist_ok': True,
            'verbose': True,
            'plots': True,
        }

        # Optimizer settings
        optimizer_config = self.extra.get('optimizer', {})
        args['optimizer'] = optimizer_config.get('type', 'AdamW')
        args['lr0'] = float(optimizer_config.get('lr0', self.cfg.learning_rate))
        args['lrf'] = float(optimizer_config.get('lrf', 0.01))
        args['weight_decay'] = float(optimizer_config.get('weight_decay', self.cfg.weight_decay))

        # Augmentation
        aug_config = self.extra.get('augmentation', {})
        for key in ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate',
                    'scale', 'shear', 'perspective', 'flipud', 'fliplr',
                    'mosaic', 'mixup', 'copy_paste']:
            if key in aug_config:
                args[key] = aug_config[key]

        return args

    def export(self, model_path: str = None, formats: List[str] = None) -> Dict[str, str]:
        """
        Export model to various formats.

        Args:
            model_path: Path to model (uses best if not specified)
            formats: Export formats (default: ['onnx'])

        Returns:
            Dict mapping format to exported path
        """
        if model_path is None:
            model_path = self.model_dir / "yolo11_meter.pt"
            if not model_path.exists():
                # Try checkpoint
                experiment_name = self.extra.get('experiment_name', 'yolo11_meter')
                model_path = self.checkpoint_dir / experiment_name / "weights" / "best.pt"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if formats is None:
            formats = self.extra.get('export', {}).get('formats', ['onnx'])

        logger.info(f"Exporting model: {model_path}")

        model = YOLO(str(model_path))
        export_config = self.extra.get('export', {})
        exported = {}

        for fmt in formats:
            try:
                if fmt == 'onnx':
                    path = model.export(
                        format='onnx',
                        dynamic=export_config.get('dynamic', True),
                        simplify=export_config.get('simplify', True),
                        opset=12,
                        half=export_config.get('half', False),
                    )
                elif fmt == 'torchscript':
                    path = model.export(format='torchscript')
                else:
                    path = model.export(format=fmt)

                if path:
                    # Copy to export directory
                    dst = self.export_dir / f"yolo11_meter.{fmt}"
                    shutil.copy2(path, dst)
                    exported[fmt] = str(dst)
                    logger.info(f"Exported {fmt}: {dst}")

            except Exception as e:
                logger.error(f"Failed to export {fmt}: {e}")

        return exported

    def evaluate(self, model_path: str = None) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Returns:
            Evaluation metrics
        """
        if model_path is None:
            model_path = self.model_dir / "yolo11_meter.pt"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if self.dataset_yaml is None:
            self.dataset_yaml = self.prepare_dataset()

        model = YOLO(str(model_path))

        results = model.val(
            data=self.dataset_yaml,
            split='val',
            device=self._get_device_string(),
        )

        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        }

        logger.info("Evaluation Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        return metrics
