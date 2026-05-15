from __future__ import annotations

from typing import Any, Mapping

import torch

from .core import simple_multireceptor_dose_response


required_theta_keys = [
    "kd",
    "receptor_abundance",
    "receptor_weight",
    "top",
    "bottom",
    "s50",
    "response_hill",
]


def simulate(theta: Any, xi: Any) -> torch.Tensor:
    """Evaluate the multireceptor candidate at concentration design ``xi``."""

    params = _coerce_theta(theta)
    return simple_multireceptor_dose_response(concentration=xi, **params)


def _coerce_theta(theta: Any) -> dict[str, torch.Tensor]:
    if not isinstance(theta, Mapping):
        raise TypeError(
            "Multireceptor theta must be a mapping with keys "
            f"{required_theta_keys}."
        )

    missing = [name for name in required_theta_keys if name not in theta]
    if missing:
        raise KeyError(f"Multireceptor theta mapping is missing keys: {missing}")

    return {
        "kd": torch.as_tensor(theta["kd"], dtype=torch.float32),
        "receptor_abundance": torch.as_tensor(
            theta["receptor_abundance"], dtype=torch.float32
        ),
        "receptor_weight": torch.as_tensor(theta["receptor_weight"], dtype=torch.float32),
        "top": torch.as_tensor(theta["top"], dtype=torch.float32),
        "bottom": torch.as_tensor(theta["bottom"], dtype=torch.float32),
        "s50": torch.as_tensor(theta["s50"], dtype=torch.float32),
        "response_hill": torch.as_tensor(theta["response_hill"], dtype=torch.float32),
    }


__all__ = ["required_theta_keys", "simulate"]
