"""iDAD backend adapter.

Policy-network amortised BOED for differentiable implicit simulators
(Ivanova et al., NeurIPS 2021).  Some upstream examples need
``torchsde``; we note that in the descriptor and defer installation
guidance to the README.
"""

from __future__ import annotations

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


class IDADBackend(BackendAdapter):
    name = "idad"
    description = (
        "Policy-network amortised BOED for differentiable implicit simulators "
        "(Ivanova et al., NeurIPS 2021)."
    )

    def supports(self, spec: ExperimentSpec) -> bool:
        if spec.backend == self.name:
            return True
        # iDAD is preferred when the simulator is differentiable AND the
        # experiment is sequential (``num_optimization_steps`` > 1 and a
        # policy network flag is supplied in backend_options).
        sequential = int(spec.compute_budget.num_optimization_steps or 0) > 1
        policy = bool(spec.backend_options.get("policy_network"))
        return bool(spec.differentiable) and sequential and policy

    def required_fields(self, spec: ExperimentSpec | None = None) -> list[str]:
        return [
            "backend",
            "simulator_ref",
            "prior_sampler_ref",
            "design_variables",
            "backend_options.policy_network",
            "compute_budget.num_optimization_steps",
        ]

    def validate(self, spec: ExperimentSpec) -> ValidationReport:
        errors: list[ValidationIssue] = []
        missing: list[str] = []
        for path in self.required_fields(spec):
            if field_is_missing(spec, path):
                errors.append(
                    ValidationIssue(path=path, message=f"{path} is required for iDAD.")
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
        from boed_agent.backends._idad_dispatch import estimate_eig_impl

        return estimate_eig_impl(spec, design or spec.effective_initial_design())

    def optimize(self, spec: ExperimentSpec) -> OptimizationResult:
        from boed_agent.backends._idad_dispatch import optimize_impl

        return optimize_impl(spec)

    def describe(self, spec: ExperimentSpec | None = None) -> BackendDescriptor:
        return BackendDescriptor(
            name=self.name,
            description=self.description,
            capabilities={
                "family": "policy_amortised",
                "simulator": "implicit_differentiable",
                "framework": "pytorch",
                "notes": "Some examples require `torchsde`; install via conda env.",
            },
            required_fields=self.required_fields(spec),
            status="beta",
        )


__all__ = ["IDADBackend"]
