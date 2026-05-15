"""Token-budget accounting for the literature pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class TokenBudgetExceeded(RuntimeError):
    """Raised when the literature pipeline exceeds its configured budget.

    The pipeline halts gracefully — already-produced evidence is
    preserved and returned in the partial :class:`LiteratureReport`.
    """


@dataclass
class TokenBudget:
    """Track token consumption across the 5-stage pipeline.

    The counters are accumulated in-place by :class:`LLMClient`
    implementations.  :meth:`check` is called at stage boundaries to
    abort early if the per-stage or total cap has been exceeded.
    """

    max_total_tokens: Optional[int] = None
    max_per_stage: Optional[int] = None
    raise_on_exceed: bool = False

    total_tokens: int = 0
    per_stage: dict[str, int] = field(default_factory=dict)
    api_calls: int = 0

    def record(self, stage: str, tokens: int) -> None:
        self.total_tokens += int(tokens)
        self.per_stage[stage] = self.per_stage.get(stage, 0) + int(tokens)
        self.api_calls += 1

    def check(self, stage: str) -> bool:
        """Return True when the budget is exhausted for ``stage``.

        If ``raise_on_exceed`` is set, a :class:`TokenBudgetExceeded`
        is raised instead of returning a sentinel.
        """
        over_total = (
            self.max_total_tokens is not None
            and self.total_tokens >= self.max_total_tokens
        )
        over_stage = (
            self.max_per_stage is not None
            and self.per_stage.get(stage, 0) >= self.max_per_stage
        )
        if over_total or over_stage:
            if self.raise_on_exceed:
                raise TokenBudgetExceeded(
                    f"Token budget exceeded at stage={stage!r}: "
                    f"total={self.total_tokens}, stage={self.per_stage.get(stage, 0)}"
                )
            return True
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "max_total_tokens": self.max_total_tokens,
            "max_per_stage": self.max_per_stage,
            "total_tokens": self.total_tokens,
            "per_stage": dict(self.per_stage),
            "api_calls": self.api_calls,
        }


__all__ = ["TokenBudget", "TokenBudgetExceeded"]
