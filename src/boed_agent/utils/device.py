"""Torch device helpers.

Chooses the best available torch device at runtime, preferring Apple Silicon
(`mps`) → NVIDIA (`cuda`) → CPU. `torch` is imported lazily so this module can
live in the base package without making `torch` a hard dependency.
"""

from __future__ import annotations

from typing import Any


def get_torch_device(prefer: str | None = None) -> Any:
    """Return a `torch.device` picked for the current machine.

    Preference order, unless overridden via `prefer`:

    1. Apple Silicon GPU (``mps``) — MacBook Pro (M1/M2/M3/...) with torch>=2.0
    2. NVIDIA GPU (``cuda``)
    3. CPU (always available fallback)

    Parameters
    ----------
    prefer:
        Optional explicit device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        If the preferred device is not available, we silently fall through the
        preference order rather than raising — BOED workloads should always
        remain runnable on CPU.
    """

    import torch  # local import keeps torch an optional dep

    if prefer == "cpu":
        return torch.device("cpu")

    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    cuda_available = torch.cuda.is_available()

    if prefer == "mps" and mps_available:
        return torch.device("mps")
    if prefer == "cuda" and cuda_available:
        return torch.device("cuda")

    if mps_available:
        return torch.device("mps")
    if cuda_available:
        return torch.device("cuda")
    return torch.device("cpu")


def device_summary() -> dict[str, Any]:
    """Small diagnostic dict — handy for logging at startup.

    Returns a plain dict so callers don't need torch imported to use it.
    """

    try:
        import torch
    except ImportError:
        return {"torch": None, "mps": False, "cuda": False, "chosen": "cpu"}

    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    cuda_available = torch.cuda.is_available()
    return {
        "torch": torch.__version__,
        "mps": mps_available,
        "cuda": cuda_available,
        "chosen": str(get_torch_device()),
    }


__all__ = ["get_torch_device", "device_summary"]
