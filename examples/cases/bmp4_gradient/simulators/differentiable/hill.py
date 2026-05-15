from __future__ import annotations

from typing import Any, Mapping

import torch

from .core import hill_function


parameter_names = ["bottom", "top", "ec50", "hill_n"]


def simulate(theta: Any, xi: Any) -> torch.Tensor:
    """Evaluate the Hill candidate at concentration design ``xi``."""

    params = _coerce_theta(theta)
    return hill_function(concentration=xi, **params)


def _coerce_theta(theta: Any) -> dict[str, torch.Tensor]:
    if isinstance(theta, Mapping):
        missing = [name for name in parameter_names if name not in theta]
        if missing:
            raise KeyError(f"Hill theta mapping is missing keys: {missing}")
        return {
            name: torch.as_tensor(theta[name], dtype=torch.float32)
            for name in parameter_names
        }

    values = torch.as_tensor(theta, dtype=torch.float32)
    if values.ndim != 1 or values.numel() != len(parameter_names):
        raise ValueError(
            "Hill theta must be a mapping with keys "
            f"{parameter_names} or a flat length-{len(parameter_names)} vector."
        )
    return dict(zip(parameter_names, values.unbind(), strict=True))


__all__ = ["parameter_names", "simulate"]
