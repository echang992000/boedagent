"""The :class:`LiteratureReport` returned by the search module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from boed_agent.literature.trace import ReasoningStep, ReasoningTrace


@dataclass
class PriorSuggestion:
    parameter: str
    distribution: str | None
    params: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    cited_papers: list[str] = field(default_factory=list)
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "distribution": self.distribution,
            "params": dict(self.params),
            "reasoning": self.reasoning,
            "cited_papers": list(self.cited_papers),
            "fallback": self.fallback,
        }


@dataclass
class DesignHint:
    dimension: str
    recommendation: Any
    reasoning: str = ""
    cited_papers: list[str] = field(default_factory=list)
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
            "cited_papers": list(self.cited_papers),
            "fallback": self.fallback,
        }


@dataclass
class BackendPreference:
    ranked: list[str] = field(default_factory=list)
    reasoning: str = ""
    cited_papers: list[str] = field(default_factory=list)
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked": list(self.ranked),
            "reasoning": self.reasoning,
            "cited_papers": list(self.cited_papers),
            "fallback": self.fallback,
        }


@dataclass
class CostReport:
    tokens_by_stage: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0
    api_calls: int = 0
    papers_considered: int = 0
    papers_filtered: int = 0
    papers_processed: int = 0
    cache_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_by_stage": dict(self.tokens_by_stage),
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "papers_considered": self.papers_considered,
            "papers_filtered": self.papers_filtered,
            "papers_processed": self.papers_processed,
            "cache_hits": self.cache_hits,
        }


@dataclass
class LiteratureReport:
    prior_suggestions: dict[str, PriorSuggestion] = field(default_factory=dict)
    design_space_hints: dict[str, DesignHint] = field(default_factory=dict)
    backend_preference: BackendPreference = field(default_factory=BackendPreference)
    benchmarks: list[dict] = field(default_factory=list)
    reasoning_trace: ReasoningTrace = field(default_factory=ReasoningTrace)
    cost_report: CostReport = field(default_factory=CostReport)
    papers: list[dict] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior_suggestions": {
                k: v.to_dict() for k, v in self.prior_suggestions.items()
            },
            "design_space_hints": {
                k: v.to_dict() for k, v in self.design_space_hints.items()
            },
            "backend_preference": self.backend_preference.to_dict(),
            "benchmarks": list(self.benchmarks),
            "reasoning_trace": self.reasoning_trace.to_dict(),
            "cost_report": self.cost_report.to_dict(),
            "papers": list(self.papers),
            "queries": list(self.queries),
            "notes": list(self.notes),
            "diagnostics": dict(self.diagnostics),
        }

    def absorb_steps(self, steps: list[ReasoningStep]) -> None:
        """Populate prior / design / backend slots from Stage D output."""
        self.reasoning_trace.record(steps)
        for step in steps:
            if step.decision.startswith("prior for "):
                parameter = step.decision[len("prior for ") :]
                self.prior_suggestions[parameter] = PriorSuggestion(
                    parameter=parameter,
                    distribution=step.conclusion.get("distribution"),
                    params=step.conclusion.get("params") or {},
                    reasoning=step.reasoning,
                    cited_papers=list(step.cited_papers),
                    fallback=step.is_fallback,
                )
            elif step.decision.startswith("design for "):
                dimension = step.decision[len("design for ") :]
                self.design_space_hints[dimension] = DesignHint(
                    dimension=dimension,
                    recommendation=step.conclusion.get("recommendation"),
                    reasoning=step.reasoning,
                    cited_papers=list(step.cited_papers),
                    fallback=step.is_fallback,
                )
            elif step.decision == "backend preference":
                self.backend_preference = BackendPreference(
                    ranked=list(step.conclusion.get("ranked") or []),
                    reasoning=step.reasoning,
                    cited_papers=list(step.cited_papers),
                    fallback=step.is_fallback,
                )


__all__ = [
    "BackendPreference",
    "CostReport",
    "DesignHint",
    "LiteratureReport",
    "PriorSuggestion",
]
