"""Backend adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from boed_agent.models import (
    BackendDescriptor,
    EIGEstimate,
    ExperimentSpec,
    OptimizationResult,
    ValidationReport,
)


class BackendAdapter(ABC):
    name: str
    description: str

    @abstractmethod
    def supports(self, spec: ExperimentSpec) -> bool:
        """Return whether this backend can handle the spec."""

    @abstractmethod
    def required_fields(self, spec: ExperimentSpec | None = None) -> list[str]:
        """Return required experiment-spec fields."""

    @abstractmethod
    def validate(self, spec: ExperimentSpec) -> ValidationReport:
        """Validate the experiment spec for this backend."""

    @abstractmethod
    def estimate_eig(self, spec: ExperimentSpec, design: list[float] | None = None) -> EIGEstimate:
        """Estimate the expected information gain for a given design."""

    @abstractmethod
    def optimize(self, spec: ExperimentSpec) -> OptimizationResult:
        """Optimize the BOED objective for this backend."""

    @abstractmethod
    def describe(self, spec: ExperimentSpec | None = None) -> BackendDescriptor:
        """Return backend metadata."""
