"""Device utility functions for model training and inference."""

import torch as th


def get_available_device() -> str:
    """
    Get the best available device for computation.

    Checks in order: CUDA, XPU, CPU

    Returns:
        Device string ("cuda", "xpu", or "cpu")
    """
    if th.cuda.is_available():
        return "cuda"
    if th.xpu.is_available():
        return "xpu"
    return "cpu"


def clear_device_cache() -> None:
    """Clear cache for the available accelerator device."""
    if th.cuda.is_available():
        th.cuda.empty_cache()
    elif th.xpu.is_available():
        th.xpu.empty_cache()
