"""
GPU utilities and optimizations for RL training.
"""

import warnings


def check_gpu_availability():
    """
    Check if GPU is available and print details.

    Returns:
        Dict with GPU information
    """
    try:
        import torch

        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': None,
            'device_name': None,
            'total_memory_gb': None,
            'allocated_memory_gb': None,
            'cached_memory_gb': None,
            'free_memory_gb': None
        }

        if torch.cuda.is_available():
            gpu_info['current_device'] = torch.cuda.current_device()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)

            # Memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            cached_memory = torch.cuda.memory_reserved(0)

            gpu_info['total_memory_gb'] = total_memory / 1e9
            gpu_info['allocated_memory_gb'] = allocated_memory / 1e9
            gpu_info['cached_memory_gb'] = cached_memory / 1e9
            gpu_info['free_memory_gb'] = (total_memory - cached_memory) / 1e9

        return gpu_info

    except ImportError:
        warnings.warn("PyTorch not installed. GPU support unavailable.")
        return {'cuda_available': False}


def print_gpu_info():
    """Print GPU information in a formatted way."""
    info = check_gpu_availability()

    print("=" * 70)
    print("GPU INFORMATION")
    print("=" * 70)

    if info['cuda_available']:
        print(f"✓ CUDA Available: YES")
        print(f"  CUDA Version: {info['cuda_version']}")
        print(f"  Device Count: {info['device_count']}")
        print(f"  Current Device: {info['current_device']}")
        print(f"  Device Name: {info['device_name']}")
        print(f"  Total Memory: {info['total_memory_gb']:.2f} GB")
        print(f"  Allocated Memory: {info['allocated_memory_gb']:.2f} GB")
        print(f"  Cached Memory: {info['cached_memory_gb']:.2f} GB")
        print(f"  Free Memory: {info['free_memory_gb']:.2f} GB")
        print()

        # Recommendations
        if info['total_memory_gb'] >= 16:
            print("💪 GPU Memory: Excellent (16+ GB)")
            print("   Recommended batch_size: 256-512")
            print("   Recommended n_steps: 4096-8192")
        elif info['total_memory_gb'] >= 8:
            print("✓ GPU Memory: Good (8-16 GB)")
            print("   Recommended batch_size: 128-256")
            print("   Recommended n_steps: 2048-4096")
        else:
            print("⚠ GPU Memory: Limited (<8 GB)")
            print("   Recommended batch_size: 64-128")
            print("   Recommended n_steps: 1024-2048")

    else:
        print("✗ CUDA Available: NO")
        print("  Training will use CPU (slower)")
        print("  Consider:")
        print("    1. Install CUDA-enabled PyTorch")
        print("    2. Check NVIDIA drivers")
        print("    3. Verify GPU is recognized by system")

    print("=" * 70)


def get_optimal_batch_size(total_memory_gb: float = None) -> int:
    """
    Get optimal batch size based on GPU memory.

    Args:
        total_memory_gb: Total GPU memory in GB (auto-detected if None)

    Returns:
        Recommended batch size
    """
    if total_memory_gb is None:
        info = check_gpu_availability()
        if info['cuda_available']:
            total_memory_gb = info['total_memory_gb']
        else:
            return 64  # Default for CPU

    # Batch size recommendations based on memory
    if total_memory_gb >= 16:
        return 512
    elif total_memory_gb >= 12:
        return 256
    elif total_memory_gb >= 8:
        return 128
    elif total_memory_gb >= 4:
        return 64
    else:
        return 32


def get_optimal_n_steps(total_memory_gb: float = None) -> int:
    """
    Get optimal n_steps based on GPU memory.

    Args:
        total_memory_gb: Total GPU memory in GB (auto-detected if None)

    Returns:
        Recommended n_steps
    """
    if total_memory_gb is None:
        info = check_gpu_availability()
        if info['cuda_available']:
            total_memory_gb = info['total_memory_gb']
        else:
            return 2048  # Default for CPU

    # n_steps recommendations based on memory
    if total_memory_gb >= 16:
        return 8192
    elif total_memory_gb >= 12:
        return 4096
    elif total_memory_gb >= 8:
        return 2048
    elif total_memory_gb >= 4:
        return 1024
    else:
        return 512


def get_gpu_optimized_config():
    """
    Get GPU-optimized hyperparameters.

    Returns:
        Dict with optimized hyperparameters
    """
    info = check_gpu_availability()

    if not info['cuda_available']:
        # CPU configuration
        return {
            'device': 'cpu',
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'learning_rate': 3e-4,
            'use_mixed_precision': False
        }

    total_memory = info['total_memory_gb']

    # GPU configuration based on memory
    if total_memory >= 16:
        # High-end GPU (like yours!)
        config = {
            'device': 'cuda',
            'n_steps': 8192,         # Large rollout buffer
            'batch_size': 512,        # Large batch for stable gradients
            'n_epochs': 10,
            'learning_rate': 3e-4,
            'use_mixed_precision': True,  # FP16 for even more speed
            'max_grad_norm': 0.5,
            'gae_lambda': 0.95,
            'gamma': 0.99
        }
    elif total_memory >= 12:
        # Mid-high GPU
        config = {
            'device': 'cuda',
            'n_steps': 4096,
            'batch_size': 256,
            'n_epochs': 10,
            'learning_rate': 3e-4,
            'use_mixed_precision': True,
            'max_grad_norm': 0.5
        }
    elif total_memory >= 8:
        # Mid-range GPU
        config = {
            'device': 'cuda',
            'n_steps': 2048,
            'batch_size': 128,
            'n_epochs': 10,
            'learning_rate': 3e-4,
            'use_mixed_precision': False,
            'max_grad_norm': 0.5
        }
    else:
        # Low-end GPU
        config = {
            'device': 'cuda',
            'n_steps': 1024,
            'batch_size': 64,
            'n_epochs': 10,
            'learning_rate': 3e-4,
            'use_mixed_precision': False,
            'max_grad_norm': 0.5
        }

    return config


def enable_gpu_optimizations():
    """Enable PyTorch GPU optimizations."""
    try:
        import torch

        if torch.cuda.is_available():
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True

            # Enable TF32 for Ampere GPUs (30xx, 40xx series)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✓ Enabled TF32 optimizations for Ampere+ GPU")

            # Set memory allocator settings
            torch.cuda.empty_cache()

            print("✓ GPU optimizations enabled")
            return True
        else:
            print("⚠ No GPU available, skipping GPU optimizations")
            return False

    except ImportError:
        warnings.warn("PyTorch not available")
        return False


def monitor_gpu_usage():
    """
    Monitor GPU usage during training.

    Returns:
        Dict with current GPU usage stats
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        usage = {
            'allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
            'reserved_gb': torch.cuda.memory_reserved(0) / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated(0) / 1e9,
            'utilization_percent': None
        }

        # Try to get GPU utilization (requires nvidia-ml-py3)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            usage['utilization_percent'] = util.gpu
            pynvml.nvmlShutdown()
        except:
            pass

        return usage

    except ImportError:
        return None


def clear_gpu_memory():
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ GPU memory cache cleared")
    except ImportError:
        pass


def estimate_training_time(
    total_timesteps: int,
    n_steps: int,
    n_epochs: int,
    batch_size: int,
    steps_per_second: float = None
) -> dict:
    """
    Estimate training time.

    Args:
        total_timesteps: Total training timesteps
        n_steps: Steps per update
        n_epochs: Training epochs per update
        batch_size: Batch size
        steps_per_second: Measured performance (auto-estimated if None)

    Returns:
        Dict with time estimates
    """
    # Number of updates
    n_updates = total_timesteps // n_steps

    # Batches per update
    batches_per_update = (n_steps // batch_size) * n_epochs

    # Total batches
    total_batches = n_updates * batches_per_update

    # Estimate steps per second if not provided
    if steps_per_second is None:
        info = check_gpu_availability()
        if info['cuda_available']:
            # GPU estimates (steps/second)
            if info['total_memory_gb'] >= 16:
                steps_per_second = 2000  # Fast GPU
            elif info['total_memory_gb'] >= 8:
                steps_per_second = 1000  # Mid GPU
            else:
                steps_per_second = 500   # Slow GPU
        else:
            steps_per_second = 100  # CPU

    # Time estimates
    total_seconds = total_timesteps / steps_per_second
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60

    return {
        'total_timesteps': total_timesteps,
        'n_updates': n_updates,
        'total_batches': total_batches,
        'estimated_steps_per_second': steps_per_second,
        'estimated_seconds': total_seconds,
        'estimated_minutes': total_minutes,
        'estimated_hours': total_hours,
        'estimated_time_str': f"{int(total_hours)}h {int(total_minutes % 60)}m"
    }


def print_training_estimate(total_timesteps: int, config: dict):
    """Print training time estimate."""
    estimate = estimate_training_time(
        total_timesteps=total_timesteps,
        n_steps=config.get('n_steps', 2048),
        n_epochs=config.get('n_epochs', 10),
        batch_size=config.get('batch_size', 64)
    )

    print("Training Time Estimate:")
    print(f"  Total Updates: {estimate['n_updates']:,}")
    print(f"  Total Batches: {estimate['total_batches']:,}")
    print(f"  Estimated Speed: {estimate['estimated_steps_per_second']:,.0f} steps/sec")
    print(f"  Estimated Time: {estimate['estimated_time_str']}")
    print()
