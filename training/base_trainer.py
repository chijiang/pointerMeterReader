"""
Base Trainer - Abstract base class for all training tasks with GPU optimization
"""

import os
import sys
import time
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from .device_manager import DeviceManager
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class CheckpointManager:
    """Manages model checkpoints"""

    def __init__(self, save_dir: str, save_top_k: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.checkpoints: List[Tuple[float, Path]] = []

    def save(self, state: Dict[str, Any], score: float, filename: str = None) -> Path:
        """Save checkpoint and manage top-k"""
        if filename is None:
            filename = f"checkpoint_epoch_{state.get('epoch', 0)}.pth"

        path = self.save_dir / filename
        torch.save(state, path)

        self.checkpoints.append((score, path))
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Remove old checkpoints beyond top-k
        while len(self.checkpoints) > self.save_top_k:
            _, old_path = self.checkpoints.pop()
            if old_path.exists() and old_path.name != 'best.pth':
                old_path.unlink()

        return path

    def save_best(self, state: Dict[str, Any]) -> Path:
        """Save as best model"""
        path = self.save_dir / 'best.pth'
        torch.save(state, path)
        return path

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load best checkpoint"""
        path = self.save_dir / 'best.pth'
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=False)
        return None

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint"""
        checkpoints = list(self.save_dir.glob('checkpoint_*.pth'))
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return torch.load(latest, map_location='cpu', weights_only=False)


class BaseTrainer(ABC):
    """
    Abstract base trainer with GPU optimization.

    Provides:
    - Device management (CUDA/MPS/CPU)
    - Mixed precision training
    - Gradient accumulation
    - Early stopping
    - Checkpoint management
    - TensorBoard logging
    - Progress tracking
    """

    def __init__(self, config: ConfigManager):
        """
        Initialize trainer.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.cfg = config.config

        # Setup logging
        self._setup_logging()

        # Initialize device
        self.device_manager = DeviceManager(
            device_config=self.cfg.device,
            memory_fraction=0.9
        )
        self.device = self.device_manager.device

        # Create output directories
        self._create_directories()

        # Initialize components (to be set by subclass)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        # Mixed precision
        self.use_amp = self.cfg.mixed_precision and self.device_manager.supports_amp
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.accumulation_steps = self.cfg.gradient_accumulation_steps

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.cfg.patience,
            min_delta=self.cfg.min_delta,
            mode='max'  # For mIoU, mAP, etc.
        )

        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.cfg.save_dir,
            save_top_k=self.cfg.save_top_k
        )

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if self.cfg.tensorboard:
            log_dir = Path(self.cfg.save_dir) / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(log_dir=str(log_dir))

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = 0.0
        self.training_start_time = None

    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.cfg.log_level.upper(), logging.INFO)

        # Remove existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Setup new handlers
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

        # Add file handler
        log_dir = Path(self.cfg.save_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

    def _create_directories(self):
        """Create output directories"""
        dirs = [
            self.cfg.save_dir,
            Path(self.cfg.save_dir) / 'checkpoints',
            Path(self.cfg.save_dir) / 'logs',
            Path(self.cfg.save_dir) / 'predictions',
            Path(self.cfg.save_dir) / 'exported',
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create model - to be implemented by subclass"""
        pass

    @abstractmethod
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders - to be implemented by subclass"""
        pass

    @abstractmethod
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer - to be implemented by subclass"""
        pass

    @abstractmethod
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create scheduler - to be implemented by subclass"""
        pass

    @abstractmethod
    def _create_criterion(self) -> nn.Module:
        """Create loss function - to be implemented by subclass"""
        pass

    @abstractmethod
    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step - to be implemented by subclass"""
        pass

    @abstractmethod
    def _validate(self) -> Dict[str, float]:
        """Run validation - to be implemented by subclass"""
        pass

    @abstractmethod
    def _get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Get primary metric for early stopping - to be implemented by subclass"""
        pass

    def setup(self):
        """Setup all training components"""
        logger.info("Setting up training components...")

        # Create model
        logger.info("Creating model...")
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # Enable gradient checkpointing if requested
        if self.cfg.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

        # Create data loaders
        logger.info("Creating data loaders...")
        self.train_loader, self.val_loader = self._create_dataloaders()

        # Create optimizer
        logger.info("Creating optimizer...")
        self.optimizer = self._create_optimizer()

        # Create scheduler
        logger.info("Creating scheduler...")
        self.scheduler = self._create_scheduler()

        # Create criterion
        logger.info("Creating criterion...")
        self.criterion = self._create_criterion()

        logger.info("Setup complete!")
        self._log_training_info()

    def _log_training_info(self):
        """Log training configuration"""
        logger.info("=" * 60)
        logger.info("Training Configuration")
        logger.info("=" * 60)
        logger.info(f"Model Type: {self.cfg.model_type}")
        logger.info(f"Model Name: {self.cfg.model_name}")
        logger.info(f"Device: {self.device_manager.device_info.device_name}")
        logger.info(f"Mixed Precision: {self.use_amp}")
        logger.info(f"Batch Size: {self.cfg.batch_size}")
        logger.info(f"Effective Batch Size: {self.cfg.batch_size * self.accumulation_steps}")
        logger.info(f"Learning Rate: {self.cfg.learning_rate}")
        logger.info(f"Epochs: {self.cfg.epochs}")
        logger.info(f"Train Samples: {len(self.train_loader.dataset) if self.train_loader else 'N/A'}")
        logger.info(f"Val Samples: {len(self.val_loader.dataset) if self.val_loader else 'N/A'}")
        logger.info("=" * 60)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        epoch_metrics = {
            'loss': 0.0,
            'samples': 0,
        }

        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.cfg.epochs}',
            leave=True
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    step_metrics = self._train_step(batch)
                    loss = step_metrics['loss'] / self.accumulation_steps

                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.scheduler and self._is_step_scheduler():
                        self.scheduler.step()
            else:
                step_metrics = self._train_step(batch)
                loss = step_metrics['loss'] / self.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler and self._is_step_scheduler():
                        self.scheduler.step()

            # Update metrics
            batch_size = step_metrics.get('batch_size', self.cfg.batch_size)
            epoch_metrics['loss'] += step_metrics['loss'].item() * batch_size
            epoch_metrics['samples'] += batch_size

            # Update other metrics
            for key, value in step_metrics.items():
                if key not in ['loss', 'batch_size']:
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value * batch_size

            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{step_metrics['loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # TensorBoard logging
            if self.writer and self.global_step % 10 == 0:
                self.writer.add_scalar('Train/Loss_Step', step_metrics['loss'].item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)

        # Average metrics
        for key in epoch_metrics:
            if key != 'samples':
                epoch_metrics[key] /= epoch_metrics['samples']

        return epoch_metrics

    def _is_step_scheduler(self) -> bool:
        """Check if scheduler should step per batch"""
        step_schedulers = (
            torch.optim.lr_scheduler.OneCycleLR,
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        )
        return isinstance(self.scheduler, step_schedulers)

    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()

        with torch.no_grad():
            metrics = self._validate()

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint"""
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_score': self.best_score,
            'metrics': metrics,
            'config': self.config.to_dict(),
        }

        score = self._get_primary_metric(metrics)
        self.checkpoint_manager.save(state, score)

        if is_best:
            self.checkpoint_manager.save_best(state)
            logger.info(f"New best model saved! Score: {score:.4f}")

    def load_checkpoint(self, path: str = None):
        """Load training checkpoint"""
        if path:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        else:
            checkpoint = self.checkpoint_manager.load_latest()

        if checkpoint is None:
            logger.warning("No checkpoint found to load")
            return

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_score = checkpoint['best_score']

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self, resume: bool = False):
        """Main training loop"""
        logger.info("Starting training...")
        self.training_start_time = time.time()

        # Resume if requested
        if resume:
            self.load_checkpoint()
            self.current_epoch += 1  # Start from next epoch

        try:
            for epoch in range(self.current_epoch, self.cfg.epochs):
                self.current_epoch = epoch
                epoch_start = time.time()

                # Train epoch
                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Epoch scheduler step
                if self.scheduler and not self._is_step_scheduler():
                    self.scheduler.step()

                # Log metrics
                epoch_time = time.time() - epoch_start
                self._log_epoch(train_metrics, val_metrics, epoch_time)

                # Check if best
                primary_metric = self._get_primary_metric(val_metrics)
                is_best = primary_metric > self.best_score
                if is_best:
                    self.best_score = primary_metric

                # Save checkpoint
                if (epoch + 1) % self.cfg.save_interval == 0 or is_best:
                    self.save_checkpoint(val_metrics, is_best)

                # Early stopping
                if self.early_stopping(primary_metric):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                # Memory cleanup
                self.device_manager.clear_cache()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint({'interrupted': True})

        finally:
            total_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {total_time / 3600:.2f} hours")

            if self.writer:
                self.writer.close()

        # Export best model
        self.export_model()

        return self.best_score

    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """Log epoch results"""
        logger.info(f"Epoch {self.current_epoch + 1}/{self.cfg.epochs} ({epoch_time:.1f}s)")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")

        for key, value in val_metrics.items():
            if isinstance(value, float):
                logger.info(f"  Val {key}: {value:.4f}")

        # TensorBoard
        if self.writer:
            self.writer.add_scalar('Train/Loss_Epoch', train_metrics['loss'], self.current_epoch)

            for key, value in val_metrics.items():
                if isinstance(value, float):
                    self.writer.add_scalar(f'Val/{key}', value, self.current_epoch)

            # Memory stats
            mem_stats = self.device_manager.get_memory_stats()
            if mem_stats:
                self.writer.add_scalar('Memory/Allocated_GB', mem_stats['allocated_gb'], self.current_epoch)
                self.writer.add_scalar('Memory/Max_Allocated_GB', mem_stats['max_allocated_gb'], self.current_epoch)

    @abstractmethod
    def export_model(self):
        """Export trained model - to be implemented by subclass"""
        pass

    def evaluate(self, model_path: str = None):
        """Evaluate model on validation set"""
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        metrics = self.validate()

        logger.info("Evaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")

        return metrics
