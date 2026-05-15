"""MINEBED backend adapter.

Thin dispatcher to the upstream ``minebed`` library.  The import is
intentionally lazy so that merely importing the package on a
pytorch-less machine does not fail.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from boed_agent.backends.base import BackendAdapter
from boed_agent.models import (
    BackendDescriptor,
    EIGEstimate,
    ExperimentSpec,
    OptimizationResult,
    OptimizationStep,
    ValidationIssue,
    ValidationReport,
    field_is_missing,
)


class MINEBEDBackend(BackendAdapter):
    name = "minebed"
    description = (
        "MI Neural Estimation for BOED on implicit but differentiable "
        "simulators (Kleinegesse & Gutmann, ICML 2020). Requires torch."
    )

    def supports(self, spec: ExperimentSpec) -> bool:
        if spec.backend == self.name:
            return True
        # Heuristic: differentiable=True and a simulator_ref.
        return bool(spec.differentiable) and bool(spec.simulator_ref)

    def required_fields(self, spec: ExperimentSpec | None = None) -> list[str]:
        return [
            "backend",
            "simulator_ref",
            "prior_sampler_ref",
            "design_variables",
            "compute_budget.num_optimization_steps",
            "compute_budget.design_learning_rate",
        ]

    def validate(self, spec: ExperimentSpec) -> ValidationReport:
        errors: list[ValidationIssue] = []
        missing: list[str] = []
        for path in self.required_fields(spec):
            if field_is_missing(spec, path):
                errors.append(
                    ValidationIssue(path=path, message=f"{path} is required for MINEBED.")
                )
                missing.append(path)
        return ValidationReport(
            valid=not errors,
            errors=errors,
            missing_fields=missing,
            backend=self.name,
        )

    def estimate_eig(
        self, spec: ExperimentSpec, design: list[float] | None = None
    ) -> EIGEstimate:
        from boed_agent.backends._minebed_dispatch import estimate_eig_impl

        return estimate_eig_impl(spec, design or spec.effective_initial_design())

    def optimize(self, spec: ExperimentSpec) -> OptimizationResult:
        from boed_agent.backends._minebed_dispatch import optimize_impl

        return optimize_impl(spec)

    def describe(self, spec: ExperimentSpec | None = None) -> BackendDescriptor:
        return BackendDescriptor(
            name=self.name,
            description=self.description,
            capabilities={
                "family": "mine_neural_estimator",
                "simulator": "implicit_differentiable",
                "framework": "pytorch",
            },
            required_fields=self.required_fields(spec),
            status="beta",
        )


__all__ = ["MINEBEDBackend"]
