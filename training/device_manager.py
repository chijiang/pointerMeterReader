"""
Device Manager - Handles GPU/MPS/CPU selection and optimization
"""

import os
import torch
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Track whether thread settings have been applied (must be done once, before any parallel work)
_thread_settings_applied = False


@dataclass
class DeviceInfo:
    """Device information container"""
    device: torch.device
    device_type: str  # 'cuda', 'mps', 'cpu'
    device_name: str
    memory_total: Optional[int] = None  # bytes
    memory_available: Optional[int] = None
    compute_capability: Optional[tuple] = None
    supports_amp: bool = True
    supports_pin_memory: bool = True
    recommended_batch_size: int = 16
    recommended_workers: int = 4


class DeviceManager:
    """
    Manages device selection and GPU optimization.

    Automatically detects and configures:
    - NVIDIA CUDA GPUs
    - Apple Silicon MPS
    - CPU fallback

    Provides memory optimization and batch size recommendations.
    """

    # Memory thresholds for batch size recommendations (in GB)
    BATCH_SIZE_THRESHOLDS = {
        2: 4,    # < 2GB: batch 4
        4: 8,    # 2-4GB: batch 8
        8: 16,   # 4-8GB: batch 16
        16: 32,  # 8-16GB: batch 32
        32: 64,  # 16-32GB: batch 64
    }

    def __init__(self, device_config: str = 'auto', memory_fraction: float = 0.9):
        """
        Initialize DeviceManager.

        Args:
            device_config: 'auto', 'cuda', 'cuda:0', 'mps', 'cpu', or device index
            memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        """
        self.device_config = device_config
        self.memory_fraction = memory_fraction
        self.device_info = self._detect_device()
        self._apply_optimizations()

    def _detect_device(self) -> DeviceInfo:
        """Detect and configure the best available device"""

        if self.device_config == 'auto':
            return self._auto_detect()
        elif self.device_config.startswith('cuda'):
            return self._setup_cuda(self.device_config)
        elif self.device_config == 'mps':
            return self._setup_mps()
        else:
            return self._setup_cpu()

    def _auto_detect(self) -> DeviceInfo:
        """Automatically detect the best device"""

        # Try CUDA first
        if torch.cuda.is_available():
            return self._setup_cuda('cuda:0')

        # Try MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return self._setup_mps()

        # Fall back to CPU
        return self._setup_cpu()

    def _setup_cuda(self, device_str: str) -> DeviceInfo:
        """Setup CUDA device"""
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return self._setup_cpu()

        # Parse device index
        if ':' in device_str:
            device_idx = int(device_str.split(':')[1])
        else:
            device_idx = 0

        if device_idx >= torch.cuda.device_count():
            logger.warning(f"CUDA device {device_idx} not found, using device 0")
            device_idx = 0

        device = torch.device(f'cuda:{device_idx}')

        # Get device properties
        props = torch.cuda.get_device_properties(device_idx)
        memory_total = props.total_memory
        memory_available = memory_total - torch.cuda.memory_allocated(device_idx)

        # Compute capability
        compute_cap = (props.major, props.minor)

        # Determine recommended batch size based on memory
        memory_gb = memory_total / (1024 ** 3)
        batch_size = self._get_recommended_batch_size(memory_gb)

        # Workers based on system
        workers = min(8, os.cpu_count() or 4)

        info = DeviceInfo(
            device=device,
            device_type='cuda',
            device_name=props.name,
            memory_total=memory_total,
            memory_available=memory_available,
            compute_capability=compute_cap,
            supports_amp=True,
            supports_pin_memory=True,
            recommended_batch_size=batch_size,
            recommended_workers=workers,
        )

        logger.info(f"CUDA Device: {props.name}")
        logger.info(f"  Memory: {memory_gb:.1f} GB")
        logger.info(f"  Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        logger.info(f"  Recommended batch size: {batch_size}")

        return info

    def _setup_mps(self) -> DeviceInfo:
        """Setup Apple MPS device"""
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available, falling back to CPU")
            return self._setup_cpu()

        device = torch.device('mps')

        # MPS-specific settings
        # Note: MPS has limited AMP support in some PyTorch versions
        supports_amp = hasattr(torch.cuda.amp, 'autocast')

        info = DeviceInfo(
            device=device,
            device_type='mps',
            device_name='Apple Silicon MPS',
            memory_total=None,  # MPS doesn't expose memory info
            memory_available=None,
            compute_capability=None,
            supports_amp=False,  # MPS AMP is experimental
            supports_pin_memory=False,  # MPS doesn't support pin_memory
            recommended_batch_size=8,  # Conservative for MPS
            recommended_workers=4,
        )

        logger.info("Apple MPS Device activated")
        logger.info("  Note: Mixed precision disabled (experimental on MPS)")
        logger.info("  Recommended batch size: 8")

        return info

    def _setup_cpu(self) -> DeviceInfo:
        """Setup CPU device"""
        device = torch.device('cpu')

        # Get CPU count
        cpu_count = os.cpu_count() or 4

        info = DeviceInfo(
            device=device,
            device_type='cpu',
            device_name=f'CPU ({cpu_count} cores)',
            memory_total=None,
            memory_available=None,
            compute_capability=None,
            supports_amp=False,
            supports_pin_memory=False,
            recommended_batch_size=4,  # Conservative for CPU
            recommended_workers=min(4, cpu_count),
        )

        logger.info(f"CPU Device: {cpu_count} cores")
        logger.info("  Mixed precision disabled on CPU")
        logger.info("  Recommended batch size: 4")

        return info

    def _get_recommended_batch_size(self, memory_gb: float) -> int:
        """Get recommended batch size based on memory"""
        for threshold, batch_size in sorted(self.BATCH_SIZE_THRESHOLDS.items()):
            if memory_gb < threshold:
                return batch_size
        return 64  # For very large GPUs

    def _apply_optimizations(self):
        """Apply device-specific optimizations"""

        if self.device_info.device_type == 'cuda':
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Set memory fraction
            if self.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.memory_fraction,
                    self.device_info.device
                )

            logger.info("CUDA optimizations enabled: cuDNN benchmark, TF32")

        elif self.device_info.device_type == 'mps':
            # MPS optimizations
            # Currently limited options for MPS
            pass

        elif self.device_info.device_type == 'cpu':
            # CPU optimizations - only set thread counts once
            global _thread_settings_applied
            if not _thread_settings_applied:
                try:
                    torch.set_num_threads(os.cpu_count() or 4)
                    if hasattr(torch, 'set_num_interop_threads'):
                        torch.set_num_interop_threads(2)
                    _thread_settings_applied = True
                except RuntimeError as e:
                    # Thread settings already applied or parallel work started
                    logger.debug(f"Thread settings skipped: {e}")
                    _thread_settings_applied = True

    @property
    def device(self) -> torch.device:
        """Get the selected device"""
        return self.device_info.device

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA"""
        return self.device_info.device_type == 'cuda'

    @property
    def is_mps(self) -> bool:
        """Check if using MPS"""
        return self.device_info.device_type == 'mps'

    @property
    def supports_amp(self) -> bool:
        """Check if device supports mixed precision"""
        return self.device_info.supports_amp

    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get optimized DataLoader kwargs for this device"""
        return {
            'pin_memory': self.device_info.supports_pin_memory,
            'num_workers': self.device_info.recommended_workers,
        }

    def get_memory_stats(self) -> Optional[Dict[str, float]]:
        """Get current memory usage (CUDA only)"""
        if not self.is_cuda:
            return None

        return {
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / (1024**3),
        }

    def clear_cache(self):
        """Clear GPU cache"""
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_mps:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

    def synchronize(self):
        """Synchronize device operations"""
        if self.is_cuda:
            torch.cuda.synchronize()
        elif self.is_mps:
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

    def __repr__(self) -> str:
        return f"DeviceManager({self.device_info.device_name})"
