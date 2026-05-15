"""Structured outputs for the OpenAI Agents SDK path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class StructuredQuestion:
    field: str
    prompt: str
    reason: str


@dataclass
class ClarificationSpecialistOutput:
    blocking_fields: list[str] = field(default_factory=list)
    questions: list[StructuredQuestion] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BackendAdvisorOutput:
    recommended_backend: str
    rationale: str
    required_next_inputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResultInterpreterOutput:
    summary: str
    caveats: list[str] = field(default_factory=list)
    suggested_next_action: str | None = None
    optimized_design_histories: list[list[dict]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ManagerResponse:
    response_kind: Literal["clarification", "backend_advice", "result", "general"] = "general"
    message: str = ""
    blocking_fields: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    suggested_next_action: str | None = None
    optimized_design_histories: list[list[dict]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
