"""Provider-neutral LLM client protocol used by the literature pipeline.

Only the shape of the interface matters here: concrete backends live
next to the existing :mod:`boed_agent.providers` adapters and are
instantiated lazily so that the literature pipeline can run completely
offline when a mock client is injected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Protocol, runtime_checkable

from boed_agent.literature.token_budget import TokenBudget


ModelTier = Literal["cheap", "reasoning"]


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return int(self.input_tokens) + int(self.output_tokens)


@runtime_checkable
class LLMClient(Protocol):
    """Provider-neutral protocol.

    A single entry point ``extract`` — deliberately narrow so tests can
    mock it with three lines of code.  Implementations may route to
    different providers based on ``model_tier``.
    """

    def extract(
        self,
        prompt: str,
        *,
        model_tier: ModelTier = "cheap",
        stage: str = "unknown",
        budget: TokenBudget | None = None,
    ) -> LLMResponse:  # pragma: no cover - protocol
        ...


@dataclass
class NullLLMClient:
    """No-op client used when ``use_literature=False`` or for dry runs."""

    def extract(
        self,
        prompt: str,
        *,
        model_tier: ModelTier = "cheap",
        stage: str = "unknown",
        budget: TokenBudget | None = None,
    ) -> LLMResponse:
        if budget is not None:
            budget.record(stage, 0)
        return LLMResponse(text="", input_tokens=0, output_tokens=0, model="null")


@dataclass
class RecordingLLMClient:
    """Deterministic LLM client used by tests.

    ``responder`` is a callable ``(prompt, tier) -> str`` — usually a
    lookup table.  Token counts are estimated as len(prompt) / 4 and
    len(response) / 4.  The client transparently caches responses by
    ``(tier, sha256(prompt))``.
    """

    responder: Callable[[str, ModelTier], str]
    cache: dict[str, LLMResponse] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def _key(self, prompt: str, tier: ModelTier) -> str:
        return tier + ":" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def extract(
        self,
        prompt: str,
        *,
        model_tier: ModelTier = "cheap",
        stage: str = "unknown",
        budget: TokenBudget | None = None,
    ) -> LLMResponse:
        key = self._key(prompt, model_tier)
        if key in self.cache:
            cached = self.cache[key]
            self.calls.append({"stage": stage, "tier": model_tier, "cached": True})
            # Cached replies cost zero tokens.
            if budget is not None:
                budget.record(stage, 0)
            return LLMResponse(
                text=cached.text,
                input_tokens=0,
                output_tokens=0,
                model=cached.model,
                cached=True,
            )
        text = self.responder(prompt, model_tier)
        in_tok = max(1, len(prompt) // 4)
        out_tok = max(1, len(text) // 4)
        resp = LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=f"recording:{model_tier}",
        )
        self.cache[key] = resp
        self.calls.append(
            {
                "stage": stage,
                "tier": model_tier,
                "cached": False,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
        )
        if budget is not None:
            budget.record(stage, in_tok + out_tok)
        return resp


def parse_json_strict(text: str) -> Optional[Any]:
    """Best-effort JSON parsing used across the pipeline.

    The Stage B prompt asks for JSON only, but LLMs sometimes wrap it
    in prose.  We try a fenced code block first, then the raw text.
    """
    candidates = [text]
    if "```" in text:
        # pull out the first fenced block
        segments = text.split("```")
        if len(segments) >= 2:
            body = segments[1]
            if body.startswith("json"):
                body = body[4:]
            candidates.insert(0, body.strip())
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


__all__ = [
    "LLMClient",
    "LLMResponse",
    "ModelTier",
    "NullLLMClient",
    "RecordingLLMClient",
    "parse_json_strict",
]
