"""Module 4 — :class:`BOEDAgent` orchestrator.

Thin glue between the literature module, the prior builder, the data
classifier, and the simulator-choice dispatcher.  Never re-implements
inference — the selected backend does that.

Two entry points: :meth:`run` and :meth:`run(dry_run=True)`.  A dry run
returns a :class:`DryRunResult` with the literature report, reasoning
trace, the chosen backend, and the built prior / design space —
without calling ``backend.optimize``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from boed_agent.backends.base import BackendAdapter
from boed_agent.backends.registry import BackendRegistry
from boed_agent.classifier import DataClassifier, ClassifierResult
from boed_agent.literature.llm_client import LLMClient, NullLLMClient
from boed_agent.literature.report import LiteratureReport
from boed_agent.literature.search import (
    LiteratureSearchConfig,
    LiteratureSearchModule,
    SourceBundle,
)
from boed_agent.literature.token_budget import TokenBudget
from boed_agent.literature.trace import LiveCitationError, validate_citations
from boed_agent.models import ExperimentSpec, OptimizationResult
from boed_agent.prior_builder import AugmentedPrior, PriorBuilder
from boed_agent.simulator_choice import BackendChoice, SimulatorChoiceModule
from boed_agent.simulator_protocol import Simulator, introspect_metadata


@dataclass
class DesignSpace:
    """Light wrapper around the user-supplied ``design_distribution``."""

    raw: Any
    hints: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": _describe(self.raw),
            "hints": dict(self.hints),
            "warnings": list(self.warnings),
        }


class DesignBuilder:
    """Augments the user-supplied design distribution with literature hints."""

    @staticmethod
    def build(
        design_distribution: Any,
        literature_report: Optional[LiteratureReport] = None,
    ) -> DesignSpace:
        hints: dict[str, Any] = {}
        warnings: list[str] = []
        if literature_report is not None:
            for dim, hint in literature_report.design_space_hints.items():
                hints[dim] = hint.to_dict()
                if design_distribution is not None and hint.recommendation:
                    warnings.append(
                        f"Literature suggests design strategy for {dim!r}: "
                        f"{hint.recommendation}. User distribution kept verbatim."
                    )
        return DesignSpace(raw=design_distribution, hints=hints, warnings=warnings)


@dataclass
class DryRunResult:
    backend: BackendAdapter
    literature_report: LiteratureReport | None
    prior_used: AugmentedPrior
    design_space_used: DesignSpace
    backend_choice: BackendChoice
    classifier_result: ClassifierResult | None = None

    @property
    def chosen_backend(self) -> str:
        return self.backend.name

    @property
    def reasoning_trace(self):
        if self.literature_report is None:
            return None
        return self.literature_report.reasoning_trace

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_backend": self.chosen_backend,
            "backend_choice": self.backend_choice.to_dict(),
            "literature_report": (
                None if self.literature_report is None else self.literature_report.to_dict()
            ),
            "reasoning_trace": (
                None if self.reasoning_trace is None else self.reasoning_trace.to_dict()
            ),
            "prior_used": self.prior_used.to_dict(),
            "design_space_used": self.design_space_used.to_dict(),
            "classifier_result": (
                None if self.classifier_result is None else self.classifier_result.to_dict()
            ),
        }


@dataclass
class AgentRunResult:
    """Result of a non-dry run — wraps the backend's result."""

    backend_result: OptimizationResult
    literature_report: LiteratureReport | None
    prior_used: AugmentedPrior
    design_space_used: DesignSpace
    backend_choice: BackendChoice
    classifier_result: ClassifierResult | None = None
    per_cluster_results: list[OptimizationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_result": self.backend_result.to_dict(),
            "backend_choice": self.backend_choice.to_dict(),
            "literature_report": (
                None if self.literature_report is None else self.literature_report.to_dict()
            ),
            "prior_used": self.prior_used.to_dict(),
            "design_space_used": self.design_space_used.to_dict(),
            "classifier_result": (
                None if self.classifier_result is None else self.classifier_result.to_dict()
            ),
            "per_cluster_results": [r.to_dict() for r in self.per_cluster_results],
        }


class PostSynthesisValidationError(RuntimeError):
    pass


class BOEDAgent:
    """High-level orchestration layer.

    Dispatches to existing BOED libraries rather than re-implementing
    inference.  Optionally queries the literature, builds a prior, and
    runs a homogeneity check on user-supplied data.
    """

    def __init__(
        self,
        simulator: Simulator,
        design_distribution: Any,
        problem_description: str,
        prior: Any = None,
        data: Any = None,
        *,
        use_literature: bool = True,
        token_budget: TokenBudget | None = None,
        literature_sources: SourceBundle | None = None,
        literature_llm: LLMClient | None = None,
        literature_config: LiteratureSearchConfig | None = None,
        literature_module: LiteratureSearchModule | None = None,
        backend_registry: BackendRegistry | None = None,
        classifier: DataClassifier | None = None,
        backend_options: dict[str, Any] | None = None,
        experiment_spec: ExperimentSpec | None = None,
    ) -> None:
        self.simulator = simulator
        self.design_distribution = design_distribution
        self.problem_description = problem_description
        self.prior = prior
        self.data = data
        self.use_literature = bool(use_literature)
        self.token_budget = token_budget or TokenBudget()
        self.backend_registry = backend_registry or BackendRegistry.default()
        self.classifier = classifier or DataClassifier(mode="simulator_aware")
        self.backend_options = dict(backend_options or {})
        self.experiment_spec = experiment_spec
        self._literature_module = literature_module or (
            LiteratureSearchModule(
                sources=literature_sources,
                llm=literature_llm or NullLLMClient(),
                token_budget=self.token_budget,
                config=literature_config,
            )
            if use_literature
            else None
        )

    # --- public API -------------------------------------------------

    def run(self, *, dry_run: bool = False) -> DryRunResult | AgentRunResult:
        lit_report = self._run_literature()

        parameter_names = _parameter_names(self.simulator)
        prior = PriorBuilder.build(
            self.prior, lit_report, parameter_names=parameter_names
        )
        design_space = DesignBuilder.build(self.design_distribution, lit_report)

        classifier_result: ClassifierResult | None = None
        if self.data is not None:
            classifier_result = self.classifier.classify(
                self.data, simulator=self.simulator
            )

        choice = SimulatorChoiceModule.select(
            self.simulator,
            registry=self.backend_registry,
            literature_report=lit_report,
            backend_options=self.backend_options,
        )

        if lit_report is not None:
            _assert_audit_trail(lit_report)

        if dry_run:
            return DryRunResult(
                backend=choice.backend,
                literature_report=lit_report,
                prior_used=prior,
                design_space_used=design_space,
                backend_choice=choice,
                classifier_result=classifier_result,
            )

        if (
            classifier_result is not None
            and not classifier_result.homogeneous
            and self.experiment_spec is not None
        ):
            per_cluster = self._run_per_cluster(
                classifier_result, choice.backend
            )
            return AgentRunResult(
                backend_result=per_cluster[0] if per_cluster else OptimizationResult(
                    backend=choice.backend.name,
                    estimator=None,
                    status="empty",
                    design=[],
                    eig=None,
                ),
                literature_report=lit_report,
                prior_used=prior,
                design_space_used=design_space,
                backend_choice=choice,
                classifier_result=classifier_result,
                per_cluster_results=per_cluster,
            )

        result = self._optimize(choice.backend)
        if lit_report is not None:
            # Attach for downstream inspection — mirrors the md spec.
            setattr(result, "literature_report", lit_report)
        return AgentRunResult(
            backend_result=result,
            literature_report=lit_report,
            prior_used=prior,
            design_space_used=design_space,
            backend_choice=choice,
            classifier_result=classifier_result,
        )

    # --- internals --------------------------------------------------

    def _run_literature(self) -> LiteratureReport | None:
        if not self.use_literature or self._literature_module is None:
            return None
        # Auto-introspect if the simulator did not supply explicit metadata.
        # Callers that *do* attach metadata see it forwarded unchanged.
        metadata = introspect_metadata(self.simulator)
        try:
            return self._literature_module.search(
                problem_description=self.problem_description,
                simulator_metadata=metadata,
            )
        except LiveCitationError as exc:
            # Stage D's live validator tripped — surface it as our
            # public post-synthesis error so callers only have to catch
            # one exception type.
            raise PostSynthesisValidationError(
                f"Literature pipeline produced an ungrounded numerical claim for "
                f"{exc.decision!r}."
            ) from exc

    def _optimize(self, backend: BackendAdapter) -> OptimizationResult:
        if self.experiment_spec is None:
            return OptimizationResult(
                backend=backend.name,
                estimator=None,
                status="no_experiment_spec",
                design=[],
                eig=None,
                warnings=[
                    "BOEDAgent was invoked without an experiment_spec; cannot "
                    "dispatch to a concrete backend. Pass `experiment_spec=...`."
                ],
            )
        return backend.optimize(self.experiment_spec)

    def _run_per_cluster(
        self,
        classifier_result: ClassifierResult,
        backend: BackendAdapter,
    ) -> list[OptimizationResult]:
        if self.experiment_spec is None:
            return []
        results: list[OptimizationResult] = []
        unique = sorted({lbl for lbl in classifier_result.cluster_labels if lbl >= 0})
        for cluster in unique:
            sub_result = backend.optimize(self.experiment_spec)
            sub_result.warnings = list(sub_result.warnings) + [
                f"Cluster {cluster} of {len(unique)} — re-run per heterogeneous data."
            ]
            results.append(sub_result)
        return results


def _parameter_names(simulator: Simulator) -> list[str]:
    metadata = getattr(simulator, "metadata", None)
    if metadata is None:
        return []
    return list(getattr(metadata, "parameter_names", []) or [])


def _assert_audit_trail(report: LiteratureReport) -> None:
    bad = validate_citations(report.reasoning_trace.steps)
    if bad:
        raise PostSynthesisValidationError(
            "Literature report contains ungrounded numerical claims: " + ", ".join(bad)
        )


def _describe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _describe(v) for k, v in value.items()}
    return repr(value)


__all__ = [
    "AgentRunResult",
    "BOEDAgent",
    "DesignBuilder",
    "DesignSpace",
    "DryRunResult",
    "PostSynthesisValidationError",
]
