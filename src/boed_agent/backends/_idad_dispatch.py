"""Lazy dispatcher for the iDAD reference implementation."""

from __future__ import annotations

from typing import Any

from boed_agent.models import (
    EIGEstimate,
    ExperimentSpec,
    OptimizationResult,
    OptimizationStep,
)


def _load() -> Any:
    try:
        import idad  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "iDAD backend requires the upstream `idad` package. Follow the "
            "conda-env workaround documented in the README — some examples "
            "depend on `torchsde`."
        ) from exc
    return idad


def estimate_eig_impl(spec: ExperimentSpec, design: list[float]) -> EIGEstimate:
    try:
        _load()
    except RuntimeError as exc:
        return EIGEstimate(
            backend="idad",
            estimator=spec.objective.estimator,
            design=list(design),
            value=None,
            status="unavailable",
            warnings=[str(exc)],
        )
    return EIGEstimate(
        backend="idad",
        estimator=spec.objective.estimator,
        design=list(design),
        value=None,
        status="not_implemented",
        warnings=[
            "iDAD integration is a stub — wire up the policy-network inference "
            "loop in _idad_dispatch.estimate_eig_impl."
        ],
    )


def optimize_impl(spec: ExperimentSpec) -> OptimizationResult:
    try:
        _load()
    except RuntimeError as exc:
        return OptimizationResult(
            backend="idad",
            estimator=spec.objective.estimator,
            status="unavailable",
            design=spec.effective_initial_design(),
            eig=None,
            warnings=[str(exc)],
        )
    return OptimizationResult(
        backend="idad",
        estimator=spec.objective.estimator,
        status="not_implemented",
        design=spec.effective_initial_design(),
        eig=None,
        history=[
            OptimizationStep(step=0, design=spec.effective_initial_design(), eig=None)
        ],
        warnings=[
            "iDAD optimization is a stub — populate _idad_dispatch.optimize_impl "
            "to call the policy-network training loop."
        ],
    )
