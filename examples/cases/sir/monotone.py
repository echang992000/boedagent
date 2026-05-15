"""Strictly increasing design-time reparameterisations.

The sequential BOED loops both optimise one fresh measurement time per
round subject to ``xi_t > xi_{t-1}``. We never optimise ``xi_t``
directly; instead we optimise an unconstrained real ``raw`` and map it
through a positive increment to get a strictly monotone sequence.

Two helpers are exported:

* :func:`next_xi_from_raw` — map a scalar ``raw`` onto
  ``(xi_prev, t_max)`` via a softplus increment, clipped at ``t_max``.
  This is the per-round form that the sequential loops use.
* :func:`pack_deltas` / :func:`unpack_deltas` — full-trajectory form
  that maps a length-``T`` vector of unconstrained reals onto a
  strictly increasing vector in ``(0, t_max)``. Useful for whole-plan
  joint optimisation; also exercised by the module smoke test.

All functions are pure numpy. A tiny torch-aware wrapper lives at the
bottom for the Pyro path that imports ``next_xi_from_raw_torch``.
"""

from __future__ import annotations

from typing import Tuple

import math
import numpy as np


# -----------------------------------------------------------------------------
# Per-round helper (primary API for the sequential loops)
# -----------------------------------------------------------------------------


def next_xi_from_raw(
    raw: float,
    *,
    xi_prev: float,
    t_max: float,
    scale: float | None = None,
    eps: float = 1e-4,
) -> float:
    """Map an unconstrained real to a scalar ``xi > xi_prev`` in ``(xi_prev, t_max]``.

    ``scale`` defaults to ``0.5 * (t_max - xi_prev)``, which keeps the
    effective step size proportional to the *remaining* interval so
    later rounds still have room to move. ``eps`` guarantees strict
    monotonicity even when ``softplus(raw)`` underflows to 0.
    """
    remaining = max(t_max - xi_prev, eps)
    if scale is None:
        scale = 0.5 * remaining
    delta = _softplus(raw) * scale
    xi_next = xi_prev + delta + eps
    return float(min(xi_next, t_max))


def next_xi_grad_wrt_raw(
    raw: float,
    *,
    xi_prev: float,
    t_max: float,
    scale: float | None = None,
) -> float:
    """Analytic derivative ``d xi_next / d raw`` (used by the design optimiser)."""

    remaining = max(t_max - xi_prev, 1e-4)
    if scale is None:
        scale = 0.5 * remaining
    # d/d raw [ softplus(raw) ] = sigmoid(raw)
    return float(_sigmoid(raw) * scale)


# -----------------------------------------------------------------------------
# Full-trajectory helpers (used in plots and for validation)
# -----------------------------------------------------------------------------


def unpack_deltas(
    raw: np.ndarray,
    *,
    t_max: float,
) -> np.ndarray:
    """Map ``raw ∈ R^T`` onto a strictly increasing ``xi ∈ (0, t_max]^T``.

    The construction uses cumulative softplus increments squashed through
    a sigmoid-based timeline so the whole vector lives in ``(0, t_max]``
    and is strictly monotone by construction.
    """
    raw = np.asarray(raw, dtype=float)
    deltas = _softplus(raw)  # > 0 elementwise
    cumulative = np.cumsum(deltas)
    # Normalised cumulative scores in (0, 1), strictly increasing.
    # We shift by a small positive constant so element 0 isn't 0.5.
    squashed = _sigmoid(cumulative - cumulative[0] * 0.5)
    # Renormalise so the last entry sits just below 1.
    scaled = squashed * (1.0 - 1e-6)
    return np.asarray(t_max * scaled, dtype=float)


def pack_deltas(
    xi: np.ndarray,
    *,
    t_max: float,
) -> np.ndarray:
    """Approximate inverse of :func:`unpack_deltas` (for tests)."""
    xi = np.asarray(xi, dtype=float)
    scaled = np.clip(xi / t_max, 1e-6, 1.0 - 1e-6)
    cumulative = _logit(scaled) + xi[0] * 0.5 / max(t_max, 1e-6)
    deltas = np.diff(np.concatenate([[0.0], cumulative]))
    deltas = np.maximum(deltas, 1e-6)
    # inverse softplus
    return np.log(np.expm1(deltas))


# -----------------------------------------------------------------------------
# Primitives
# -----------------------------------------------------------------------------


def _softplus(x: np.ndarray | float) -> np.ndarray | float:
    # Numerically stable: log(1 + exp(x)) = max(x, 0) + log1p(exp(-|x|))
    x = np.asarray(x, dtype=float) if not isinstance(x, float) else x
    if isinstance(x, float):
        return math.log1p(math.exp(-abs(x))) + max(x, 0.0)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    if isinstance(x, float):
        return 1.0 / (1.0 + math.exp(-x))
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


# -----------------------------------------------------------------------------
# Torch-aware companion (optional — only used by the Pyro linreg path)
# -----------------------------------------------------------------------------


def next_xi_from_raw_torch(raw, *, xi_prev: float, t_max: float, scale: float | None = None, eps: float = 1e-4):
    """Same as :func:`next_xi_from_raw` but for a ``torch.Tensor`` input.

    Imported lazily so this module stays numpy-only on machines without
    torch installed.
    """
    import torch  # local import

    remaining = max(t_max - xi_prev, eps)
    if scale is None:
        scale = 0.5 * remaining
    delta = torch.nn.functional.softplus(raw) * scale
    xi_next = xi_prev + delta + eps
    return torch.clamp(xi_next, max=t_max)


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------


def _smoke_test() -> None:
    rng = np.random.default_rng(0)
    t_max = 100.0
    # Full-trajectory form: random raw → strictly increasing xi.
    for _ in range(1000):
        raw = rng.standard_normal(8) * 2.0
        xi = unpack_deltas(raw, t_max=t_max)
        assert xi.shape == (8,)
        assert (xi > 0.0).all() and (xi <= t_max).all(), xi
        assert (np.diff(xi) > 0.0).all(), xi

    # Per-round form: monotone by construction for any raw.
    xi_prev = 0.0
    xis: list[float] = []
    for _ in range(16):
        raw = float(rng.standard_normal() * 1.5)
        xi_next = next_xi_from_raw(raw, xi_prev=xi_prev, t_max=t_max)
        assert xi_next > xi_prev, (xi_next, xi_prev)
        assert xi_next <= t_max
        xi_prev = xi_next
        xis.append(xi_next)
    print("monotone self-test ok; sample xi sequence:", [round(x, 2) for x in xis])


if __name__ == "__main__":
    _smoke_test()
