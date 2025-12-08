"""
Configuration Manager - Unified configuration handling for all training tasks
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Unified training configuration"""

    # Model settings
    model_type: str = 'detection'  # 'detection', 'segmentation', or 'classification'
    model_name: str = 'yolo11m.pt'
    num_classes: int = 1
    pretrained: bool = True
    dropout: float = 0.2

    # Data settings
    data_root: str = 'data'
    train_dir: str = 'train'
    val_dir: str = 'val'
    train_split: str = 'train'
    val_split: str = 'val'
    image_size: int = 640
    batch_size: int = 16
    num_workers: int = 4

    # Class names (for classification)
    class_names: list = field(default_factory=lambda: ['class_0', 'class_1'])
    class_weights: Any = None  # 'balanced', 'none', or list of weights

    # Normalization (for classification)
    normalize_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Augmentation config
    augmentation: Dict[str, Any] = field(default_factory=dict)

    # Training settings
    epochs: int = 100
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    optimizer: str = 'AdamW'
    label_smoothing: float = 0.0

    # GPU optimization
    device: str = 'auto'
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False

    # Memory optimization
    cache_images: bool = True
    pin_memory: bool = True

    # Scheduler
    scheduler_type: str = 'cosine'
    scheduler_T_max: int = None
    scheduler_eta_min: float = 1e-6
    warmup_epochs: int = 3

    # Early stopping
    patience: int = 30
    min_delta: float = 0.001

    # Saving
    save_dir: str = 'outputs'
    experiment_name: str = 'experiment'
    save_interval: int = 10
    save_top_k: int = 3

    # Export
    export_dir: str = 'outputs/exported'
    models_dir: str = 'models'
    onnx_opset_version: int = 12

    # Logging
    log_level: str = 'INFO'
    tensorboard: bool = True

    # Extra config (model-specific)
    extra: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """
    Manages configuration loading, validation, and merging.

    Supports:
    - YAML configuration files
    - Environment variable overrides
    - Command-line argument merging
    - Configuration validation
    """

    # Default configurations for different model types
    DEFAULT_CONFIGS = {
        'detection': {
            'model_name': 'yolo11m.pt',
            'image_size': 640,
            'batch_size': 16,
            'learning_rate': 0.01,
            'epochs': 200,
            'optimizer': 'AdamW',
            'scheduler_type': 'cosine',
            'warmup_epochs': 3,
        },
        'segmentation': {
            'model_name': 'nvidia/segformer-b2-finetuned-ade-512-512',
            'image_size': 512,
            'batch_size': 8,
            'learning_rate': 6e-5,
            'epochs': 100,
            'optimizer': 'AdamW',
            'scheduler_type': 'polynomial',
            'warmup_epochs': 5,
        },
        'classification': {
            'model_name': 'efficientnet_b0',
            'image_size': 224,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 50,
            'optimizer': 'AdamW',
            'scheduler_type': 'CosineAnnealingLR',
            'warmup_epochs': 3,
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.raw_config: Dict[str, Any] = {}
        self.config: TrainingConfig = TrainingConfig()

        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> 'ConfigManager':
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f) or {}

        self._parse_config()
        self._apply_env_overrides()
        self._validate_config()

        logger.info(f"Configuration loaded from: {config_path}")
        return self

    def _parse_config(self):
        """Parse raw config into TrainingConfig"""
        raw = self.raw_config

        # Detect model type from config structure
        if 'classes' in raw and 'names' in raw.get('classes', {}):
            # Classification-style config
            model_type = 'classification'
            model_name = raw.get('model', {}).get('name', 'efficientnet_b0')
            num_classes = raw.get('model', {}).get('num_classes', 2)
        elif 'model' in raw and isinstance(raw['model'], dict):
            # Check if it's classification or segmentation
            if raw['model'].get('name', '').startswith('efficientnet') or \
               raw['model'].get('name', '').startswith('resnet') or \
               raw['model'].get('name', '').startswith('mobilenet'):
                model_type = 'classification'
                model_name = raw['model'].get('name', 'efficientnet_b0')
                num_classes = raw['model'].get('num_classes', 2)
            else:
                # SegFormer-style config
                model_type = 'segmentation'
                model_name = raw['model'].get('name', 'nvidia/segformer-b2-finetuned-ade-512-512')
                num_classes = raw['model'].get('num_classes', 3)
        elif 'model' in raw and isinstance(raw['model'], str):
            # YOLO-style config
            model_type = 'detection'
            model_name = raw['model']
            num_classes = 1  # Will be determined from dataset
        else:
            model_type = raw.get('model_type', 'detection')
            model_name = raw.get('model_name', self.DEFAULT_CONFIGS[model_type]['model_name'])
            num_classes = raw.get('num_classes', 1)

        # Apply defaults for model type
        defaults = self.DEFAULT_CONFIGS.get(model_type, {})

        # Data config
        data_config = raw.get('data', {})
        training_config = raw.get('training', {})
        save_config = raw.get('save', {})
        logging_config = raw.get('logging', {})
        export_config = raw.get('export', {})
        classes_config = raw.get('classes', {})

        # Get class names and weights for classification
        class_names = classes_config.get('names', ['class_0', 'class_1'])
        class_weights = training_config.get('loss', {}).get('class_weights', None)

        # Get normalization config
        normalize_config = data_config.get('normalize', {})
        normalize_mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
        normalize_std = normalize_config.get('std', [0.229, 0.224, 0.225])

        # Get augmentation config
        augmentation = data_config.get('augmentation', {})

        self.config = TrainingConfig(
            # Model
            model_type=model_type,
            model_name=model_name,
            num_classes=num_classes,
            pretrained=raw.get('model', {}).get('pretrained', True) if isinstance(raw.get('model'), dict) else True,
            dropout=raw.get('model', {}).get('dropout', 0.2) if isinstance(raw.get('model'), dict) else 0.2,

            # Data
            data_root=data_config.get('root_dir', raw.get('dataset_root', 'data')),
            train_dir=data_config.get('train_dir', 'train'),
            val_dir=data_config.get('val_dir', 'val'),
            train_split=data_config.get('train_split', 'train'),
            val_split=data_config.get('val_split', 'val'),
            image_size=data_config.get('image_size', raw.get('image_size', defaults.get('image_size', 640))),
            batch_size=training_config.get('batch_size', data_config.get('batch_size', raw.get('batch_size', defaults.get('batch_size', 16)))),
            num_workers=raw.get('num_workers', data_config.get('num_workers', raw.get('workers', 4))),

            # Class names and weights (for classification)
            class_names=class_names,
            class_weights=class_weights,

            # Normalization
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,

            # Augmentation
            augmentation=augmentation,

            # Training
            epochs=training_config.get('epochs', raw.get('epochs', defaults.get('epochs', 100))),
            learning_rate=float(training_config.get('learning_rate', raw.get('learning_rate', defaults.get('learning_rate', 0.01)))),
            weight_decay=float(training_config.get('weight_decay', raw.get('weight_decay', 0.01))),
            optimizer=training_config.get('optimizer', raw.get('optimizer', {}).get('type', defaults.get('optimizer', 'AdamW'))),
            label_smoothing=float(training_config.get('loss', {}).get('label_smoothing', 0.0)),

            # GPU optimization
            device=raw.get('device', 'auto'),
            mixed_precision=training_config.get('mixed_precision', raw.get('amp', True)),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
            gradient_checkpointing=training_config.get('gradient_checkpointing', False),

            # Memory
            cache_images=raw.get('cache', True),
            pin_memory=raw.get('pin_memory', data_config.get('pin_memory', True)),

            # Scheduler
            scheduler_type=training_config.get('scheduler', {}).get('type', raw.get('lr_scheduler', {}).get('type', defaults.get('scheduler_type', 'cosine'))),
            scheduler_T_max=training_config.get('scheduler', {}).get('T_max', None),
            scheduler_eta_min=float(training_config.get('scheduler', {}).get('eta_min', 1e-6)),
            warmup_epochs=training_config.get('warmup', {}).get('epochs', raw.get('lr_scheduler', {}).get('warmup_epochs', defaults.get('warmup_epochs', 3))),

            # Early stopping
            patience=training_config.get('early_stopping', {}).get('patience', raw.get('patience', 30)),
            min_delta=float(training_config.get('early_stopping', {}).get('min_delta', 0.001)),

            # Saving
            save_dir=save_config.get('checkpoint_dir', raw.get('save_dir', f'outputs/{model_type}')),
            experiment_name=raw.get('experiment_name', 'experiment'),
            save_interval=save_config.get('save_interval', raw.get('save_period', 10)),
            save_top_k=save_config.get('save_top_k', 3),

            # Export
            export_dir=export_config.get('output_dir', f'outputs/{model_type}/exported'),
            models_dir=export_config.get('models_dir', f'models/{model_type}'),
            onnx_opset_version=export_config.get('onnx', {}).get('opset_version', 12),

            # Logging
            log_level=logging_config.get('level', 'INFO'),
            tensorboard=logging_config.get('tensorboard', True),

            # Extra config (preserve all original)
            extra=raw,
        )

    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            'TRAIN_BATCH_SIZE': ('batch_size', int),
            'TRAIN_EPOCHS': ('epochs', int),
            'TRAIN_LR': ('learning_rate', float),
            'TRAIN_DEVICE': ('device', str),
            'TRAIN_WORKERS': ('num_workers', int),
            'TRAIN_AMP': ('mixed_precision', lambda x: x.lower() == 'true'),
        }

        for env_var, (attr, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    setattr(self.config, attr, converter(value))
                    logger.info(f"Override from env: {attr}={value}")
                except ValueError:
                    logger.warning(f"Invalid env value for {env_var}: {value}")

    def _validate_config(self):
        """Validate configuration"""
        errors = []

        # Validate batch size
        if self.config.batch_size < 1:
            errors.append("batch_size must be >= 1")

        # Validate epochs
        if self.config.epochs < 1:
            errors.append("epochs must be >= 1")

        # Validate learning rate
        if self.config.learning_rate <= 0:
            errors.append("learning_rate must be > 0")

        # Validate gradient accumulation
        if self.config.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps must be >= 1")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with fallback to extra config"""
        if hasattr(self.config, key):
            return getattr(self.config, key)
        return self.config.extra.get(key, default)

    def update(self, **kwargs) -> 'ConfigManager':
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.extra[key] = value
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for key in dir(self.config):
            if not key.startswith('_'):
                result[key] = getattr(self.config, key)
        return result

    def save(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def create_default(cls, model_type: str = 'detection') -> 'ConfigManager':
        """Create ConfigManager with default configuration"""
        manager = cls()
        defaults = cls.DEFAULT_CONFIGS.get(model_type, cls.DEFAULT_CONFIGS['detection'])

        manager.config.model_type = model_type
        for key, value in defaults.items():
            if hasattr(manager.config, key):
                setattr(manager.config, key, value)

        return manager

    def __repr__(self) -> str:
        return f"ConfigManager(model_type={self.config.model_type}, epochs={self.config.epochs})"
