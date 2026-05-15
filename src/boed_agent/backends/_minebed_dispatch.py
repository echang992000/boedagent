"""Lazy dispatcher for the :mod:`minebed` library.

Kept in a separate module so that ``MINEBEDBackend`` can be imported
on a torch-less machine.  All torch / minebed imports happen here.
"""

from __future__ import annotations

from typing import Any

from boed_agent.models import (
    EIGEstimate,
    ExperimentSpec,
    OptimizationResult,
    OptimizationStep,
)
from boed_agent.utils.imports import instantiate_reference, resolve_reference


def _load() -> Any:
    try:
        import minebed  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "MINEBED backend requires the `minebed` package. Install with "
            "`pip install boed-agent[minebed]` or from source at "
            "https://github.com/stevenkleinegesse/minebed."
        ) from exc
    return minebed


def estimate_eig_impl(spec: ExperimentSpec, design: list[float]) -> EIGEstimate:
    try:
        _load()
    except RuntimeError as exc:
        return EIGEstimate(
            backend="minebed",
            estimator=spec.objective.estimator,
            design=list(design),
            value=None,
            status="unavailable",
            warnings=[str(exc)],
        )
    # Real implementations route to ``minebed.estimators`` here; we keep
    # the adapter defensive so test envs without torch still succeed.
    return EIGEstimate(
        backend="minebed",
        estimator=spec.objective.estimator,
        design=list(design),
        value=None,
        status="not_implemented",
        warnings=[
            "MINEBED integration is a stub — wire up the real estimator in "
            "_minebed_dispatch.estimate_eig_impl to run inference."
        ],
    )


def optimize_impl(spec: ExperimentSpec) -> OptimizationResult:
    try:
        _load()
    except RuntimeError as exc:
        return OptimizationResult(
            backend="minebed",
            estimator=spec.objective.estimator,
            status="unavailable",
            design=spec.effective_initial_design(),
            eig=None,
            warnings=[str(exc)],
        )
    return OptimizationResult(
        backend="minebed",
        estimator=spec.objective.estimator,
        status="not_implemented",
        design=spec.effective_initial_design(),
        eig=None,
        history=[
            OptimizationStep(step=0, design=spec.effective_initial_design(), eig=None)
        ],
        warnings=[
            "MINEBED optimization is a stub — populate _minebed_dispatch.optimize_impl "
            "to call minebed.optimization with torch tensors."
        ],
    )
