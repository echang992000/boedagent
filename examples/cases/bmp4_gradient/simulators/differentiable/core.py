from __future__ import annotations

import torch


TensorLike = torch.Tensor | float


def hill_function(
    concentration: TensorLike,
    top: TensorLike,
    bottom: TensorLike,
    ec50: TensorLike,
    hill_n: TensorLike,
) -> torch.Tensor:
    """Standard 4-parameter Hill / dose-response function."""

    c = torch.as_tensor(concentration, dtype=torch.float32)
    top = torch.as_tensor(top, dtype=torch.float32)
    bottom = torch.as_tensor(bottom, dtype=torch.float32)
    ec50 = torch.as_tensor(ec50, dtype=torch.float32)
    hill_n = torch.as_tensor(hill_n, dtype=torch.float32)

    eps = torch.tensor(1e-12, dtype=c.dtype, device=c.device)
    c = torch.clamp(c, min=eps)
    ec50 = torch.clamp(ec50, min=eps)

    return bottom + (top - bottom) * (c**hill_n) / (ec50**hill_n + c**hill_n)


def receptor_occupancy(
    concentration: TensorLike,
    kd: TensorLike,
) -> torch.Tensor:
    """Simple 1:1 fractional occupancy for one or many receptors."""

    c = torch.as_tensor(concentration, dtype=torch.float32)
    kd = torch.as_tensor(kd, dtype=torch.float32)

    eps = torch.tensor(1e-12, dtype=c.dtype, device=c.device)
    c = torch.clamp(c, min=eps)
    kd = torch.clamp(kd, min=eps)

    if c.ndim == 0 and kd.ndim == 0:
        return c / (kd + c)
    if c.ndim == 0 and kd.ndim == 1:
        return c / (kd + c)
    if c.ndim == 1 and kd.ndim == 0:
        return c / (kd + c)
    if c.ndim == 1 and kd.ndim == 1:
        return c[:, None] / (kd[None, :] + c[:, None])

    raise ValueError("concentration must be scalar or 1D, and kd must be scalar or 1D")


def simple_multireceptor_dose_response(
    concentration: TensorLike,
    kd: TensorLike,
    receptor_abundance: TensorLike,
    receptor_weight: TensorLike,
    top: TensorLike,
    bottom: TensorLike,
    s50: TensorLike,
    response_hill: TensorLike = 1.0,
) -> torch.Tensor:
    """Binding-informed dose-response model with any number of receptors."""

    kd = torch.as_tensor(kd, dtype=torch.float32)
    receptor_abundance = torch.as_tensor(receptor_abundance, dtype=torch.float32)
    receptor_weight = torch.as_tensor(receptor_weight, dtype=torch.float32)

    if kd.ndim != 1:
        raise ValueError("kd must be a 1D tensor of shape [n_receptors]")
    if receptor_abundance.shape != kd.shape:
        raise ValueError("receptor_abundance must have same shape as kd")
    if receptor_weight.shape != kd.shape:
        raise ValueError("receptor_weight must have same shape as kd")

    theta = receptor_occupancy(concentration, kd)
    effective_strength = receptor_abundance * receptor_weight

    if theta.ndim == 1:
        s = torch.sum(theta * effective_strength)
    elif theta.ndim == 2:
        s = torch.sum(theta * effective_strength[None, :], dim=1)
    else:
        raise ValueError("Unexpected occupancy tensor shape")

    s = torch.clamp(s, min=1e-12)
    s50 = torch.clamp(torch.as_tensor(s50, dtype=torch.float32), min=1e-12)
    m = torch.as_tensor(response_hill, dtype=torch.float32)

    return hill_function(
        concentration=s,
        top=top,
        bottom=bottom,
        ec50=s50,
        hill_n=m,
    )


__all__ = [
    "TensorLike",
    "hill_function",
    "receptor_occupancy",
    "simple_multireceptor_dose_response",
]
