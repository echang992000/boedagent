"""Stage E — reasoning-trace assembly.

The :class:`ReasoningTrace` is what a human reads to understand *why*
the agent picked a particular prior / design / backend.  It is also
the object that a post-synthesis validator walks to confirm every
numerical recommendation is either grounded in citations or marked
``fallback``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class ReasoningStep:
    decision: str
    evidence_summary: str
    reasoning: str
    conclusion: dict
    cited_papers: list[str] = field(default_factory=list)
    token_cost: int = 0

    @property
    def is_fallback(self) -> bool:
        return bool(self.conclusion.get("fallback"))

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "evidence_summary": self.evidence_summary,
            "reasoning": self.reasoning,
            "conclusion": dict(self.conclusion),
            "cited_papers": list(self.cited_papers),
            "token_cost": int(self.token_cost),
        }


@dataclass
class ReasoningTrace:
    steps: list[ReasoningStep] = field(default_factory=list)
    total_tokens: int = 0
    total_api_calls: int = 0

    def record(self, steps: Iterable[ReasoningStep]) -> None:
        for step in steps:
            self.steps.append(step)
            self.total_tokens += int(step.token_cost)

    def to_markdown(self) -> str:
        """Render a human-readable audit report.

        Every numerical recommendation is accompanied by its
        ``cited_papers`` list, and fallbacks are explicitly flagged.
        """
        lines = ["# Literature Reasoning Trace", ""]
        lines.append(f"Steps: {len(self.steps)}")
        lines.append(f"Total tokens: {self.total_tokens}")
        lines.append(f"Total API calls: {self.total_api_calls}")
        lines.append("")
        for idx, step in enumerate(self.steps, start=1):
            lines.append(f"## {idx}. {step.decision}")
            if step.is_fallback:
                lines.append("_Fallback: insufficient evidence._")
            lines.append("")
            lines.append("### Evidence")
            lines.append("")
            lines.append("```")
            lines.append(step.evidence_summary or "(no evidence)")
            lines.append("```")
            lines.append("")
            lines.append("### Reasoning")
            lines.append("")
            lines.append(step.reasoning or "_No reasoning returned._")
            lines.append("")
            lines.append("### Conclusion")
            lines.append("")
            for key, value in step.conclusion.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
            if step.cited_papers:
                lines.append(
                    "**Cited papers:** " + ", ".join(sorted(step.cited_papers))
                )
            elif step.is_fallback:
                lines.append("**Cited papers:** _none (fallback)_")
            else:
                lines.append("**Cited papers:** _none reported_")
            lines.append(f"**Token cost:** {step.token_cost}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_tokens": self.total_tokens,
            "total_api_calls": self.total_api_calls,
            "steps": [s.to_dict() for s in self.steps],
        }


_DEFAULT_NUMERIC_KEYS: tuple[str, ...] = (
    "distribution",
    "recommendation",
    "ranked",
)


def step_is_grounded(
    step: ReasoningStep,
    required_keys: Sequence[str] = _DEFAULT_NUMERIC_KEYS,
) -> bool:
    """Return True iff *this single step* satisfies the citation invariant.

    A step is considered grounded when it either (a) does not emit any
    of the numerical claim keys, (b) is explicitly marked as a fallback,
    or (c) has at least one citation in ``cited_papers``.
    """
    conclusion = step.conclusion or {}
    if not any(conclusion.get(key) not in (None, "") for key in required_keys):
        return True
    if step.is_fallback:
        return True
    return bool(step.cited_papers)


def validate_citations(
    steps: Sequence[ReasoningStep],
    required_keys: Sequence[str] = _DEFAULT_NUMERIC_KEYS,
) -> list[str]:
    """Return a list of decisions that lack both citations *and* a fallback flag.

    Used by :class:`BOEDAgent` as a post-synthesis assertion — the
    specification says every numerical value must be either cited or
    explicitly marked fallback.
    """
    return [
        step.decision
        for step in steps
        if not step_is_grounded(step, required_keys=required_keys)
    ]


class LiveCitationError(RuntimeError):
    """Raised by Stage D in strict mode when a step is not grounded.

    The orchestrator catches this and re-raises as
    :class:`PostSynthesisValidationError`, but Stage D wants its own
    error type so tests can pinpoint *which* decision tripped the
    invariant without parsing a message.
    """

    def __init__(self, decision: str) -> None:
        super().__init__(
            f"Stage D produced an ungrounded numerical claim for: {decision!r}"
        )
        self.decision = decision


__all__ = [
    "LiveCitationError",
    "ReasoningStep",
    "ReasoningTrace",
    "step_is_grounded",
    "validate_citations",
]
