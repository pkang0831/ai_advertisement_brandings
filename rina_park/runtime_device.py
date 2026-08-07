"""Prefer CUDA (Colab) then MPS (Apple) then CPU."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_torch_device_str() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_accelerator(prefer: tuple[str, ...] = ("cuda", "mps")) -> str:
    """Return device string; raise if none of prefer is available."""
    device = get_torch_device_str()
    if device in prefer:
        return device
    raise RuntimeError(
        f"Need one of {prefer}, got {device!r}. "
        "On Colab: Runtime → Change runtime type → GPU."
    )


def empty_cache() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
