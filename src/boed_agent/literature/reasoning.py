"""Stage D — chain-of-thought reasoning over distilled evidence.

One focused LLM call per decision (per prior parameter, per design
dimension, and one for backend preference).  The reasoning model is
hit with ≤1K tokens because Stage C has already done the distillation.

Every output carries ``reasoning`` and ``cited_papers`` fields, which
are surfaced in the :class:`ReasoningTrace`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from boed_agent.literature.aggregation import (
    AggregationResult,
    DesignAggregate,
    MethodAggregate,
    PriorAggregate,
)
from boed_agent.literature.llm_client import (
    LLMClient,
    ModelTier,
    parse_json_strict,
)
from boed_agent.literature.token_budget import TokenBudget
from boed_agent.literature.trace import (
    LiveCitationError,
    ReasoningStep,
    step_is_grounded,
)


@dataclass
class ReasoningConfig:
    model_tier: ModelTier = "reasoning"
    min_sources_for_llm: int = 3
    uncertainty_inflation: float = 1.5
    default_backend_rank: Sequence[str] = (
        "PyroVI",
        "LFIAX",
    )
    problem_context: str = ""
    # When True, the pipeline raises ``LiveCitationError`` the moment a
    # step produces a numerical claim without a citation *or* a
    # ``fallback=True`` flag.  Default False preserves the historical
    # "collect all, validate at the end" behaviour so existing callers
    # do not regress.
    strict_validation: bool = False


def reason_over_prior(
    aggregate_: PriorAggregate,
    llm: LLMClient,
    *,
    config: ReasoningConfig,
    budget: TokenBudget | None = None,
) -> ReasoningStep:
    """Produce a prior recommendation for ``aggregate_.parameter``.

    Fallback: if evidence is thin (``n_sources`` below threshold) we
    skip the LLM call entirely and emit a weakly-informative prior
    flagged with ``reasoning='insufficient_evidence_fallback'``.
    """
    cited = sorted({r.paper_id for r in aggregate_.records})
    evidence_md = _format_prior_evidence(aggregate_)

    if aggregate_.n_sources < config.min_sources_for_llm:
        distribution, params = _fallback_prior(aggregate_, config)
        conclusion = {
            "distribution": distribution,
            "params": params,
            "uncertainty_inflation": config.uncertainty_inflation,
            "fallback": True,
        }
        return ReasoningStep(
            decision=f"prior for {aggregate_.parameter}",
            evidence_summary=evidence_md,
            reasoning="insufficient_evidence_fallback",
            conclusion=conclusion,
            cited_papers=cited,
            token_cost=0,
        )

    prompt = _build_prior_prompt(aggregate_, config)
    response = llm.extract(
        prompt, model_tier=config.model_tier, stage="stage_d", budget=budget
    )
    parsed = parse_json_strict(response.text)
    conclusion, reasoning = _coerce_prior_output(parsed, aggregate_, config)
    return ReasoningStep(
        decision=f"prior for {aggregate_.parameter}",
        evidence_summary=evidence_md,
        reasoning=reasoning,
        conclusion=conclusion,
        cited_papers=cited,
        token_cost=response.total_tokens,
    )


def reason_over_design(
    aggregate_: DesignAggregate,
    llm: LLMClient,
    *,
    config: ReasoningConfig,
    budget: TokenBudget | None = None,
) -> ReasoningStep:
    cited = sorted({r["paper_id"] for r in aggregate_.records if r.get("paper_id")})
    evidence_md = _format_design_evidence(aggregate_)
    if sum(aggregate_.reported_choices.values()) == 0:
        return ReasoningStep(
            decision=f"design for {aggregate_.dimension}",
            evidence_summary=evidence_md,
            reasoning="insufficient_evidence_fallback",
            conclusion={"recommendation": None, "fallback": True},
            cited_papers=cited,
            token_cost=0,
        )
    prompt = (
        f"Decision: propose a design strategy for dimension `{aggregate_.dimension}`.\n\n"
        f"Evidence table:\n{evidence_md}\n\n"
        "Think step by step, then output JSON:\n"
        "{\"recommendation\": string, \"reasoning\": string, \"cited_papers\": [string]}\n"
    )
    response = llm.extract(
        prompt, model_tier=config.model_tier, stage="stage_d", budget=budget
    )
    parsed = parse_json_strict(response.text) or {}
    reasoning = str(parsed.get("reasoning", "")).strip() or "no_reasoning_returned"
    conclusion = {
        "recommendation": parsed.get("recommendation"),
        "fallback": False,
    }
    return ReasoningStep(
        decision=f"design for {aggregate_.dimension}",
        evidence_summary=evidence_md,
        reasoning=reasoning,
        conclusion=conclusion,
        cited_papers=cited,
        token_cost=response.total_tokens,
    )


def reason_over_backend(
    methods: Sequence[MethodAggregate],
    llm: LLMClient,
    *,
    config: ReasoningConfig,
    budget: TokenBudget | None = None,
    available_backends: Sequence[str] | None = None,
) -> ReasoningStep:
    default = list(available_backends or config.default_backend_rank)
    evidence_md = _format_method_evidence(methods)
    if not methods:
        return ReasoningStep(
            decision="backend preference",
            evidence_summary=evidence_md or "no method evidence",
            reasoning="insufficient_evidence_fallback",
            conclusion={"ranked": list(default), "fallback": True},
            cited_papers=[],
            token_cost=0,
        )
    prompt = (
        "Decision: rank the candidate BOED backends.\n"
        f"Candidates: {', '.join(default)}.\n\n"
        f"Method usage in the literature:\n{evidence_md}\n\n"
        "Think step by step, then output JSON:\n"
        "{\"ranked\": [string], \"reasoning\": string, \"cited_papers\": [string]}\n"
        "Only rank candidates from the provided list.\n"
    )
    response = llm.extract(
        prompt, model_tier=config.model_tier, stage="stage_d", budget=budget
    )
    parsed = parse_json_strict(response.text) or {}
    ranked_raw = parsed.get("ranked") or []
    ranked = [b for b in ranked_raw if isinstance(b, str) and b in default]
    # Fill in missing backends at the end so the output is always complete.
    for backend in default:
        if backend not in ranked:
            ranked.append(backend)
    cited = list(dict.fromkeys(parsed.get("cited_papers") or []))
    reasoning = str(parsed.get("reasoning", "")).strip() or "no_reasoning_returned"
    return ReasoningStep(
        decision="backend preference",
        evidence_summary=evidence_md,
        reasoning=reasoning,
        conclusion={"ranked": ranked, "fallback": False},
        cited_papers=cited,
        token_cost=response.total_tokens,
    )


# --- prompt builders / output coercion ---------------------------------


def _build_prior_prompt(aggregate_: PriorAggregate, config: ReasoningConfig) -> str:
    evidence_md = _format_prior_evidence(aggregate_)
    context = " ".join((config.problem_context or "").split())
    context_block = (
        f"Problem context and parameter-scale rules:\n{context[:1600]}\n\n"
        if context
        else ""
    )
    return (
        f"Decision: propose a prior for parameter `{aggregate_.parameter}`.\n\n"
        f"{context_block}"
        f"Evidence table:\n{evidence_md}\n\n"
        "Think step by step, then output JSON.\n"
        "Reasoning requirements:\n"
        "  1. Identify the dominant distributional family and why.\n"
        "  2. Note any outlier studies and whether to down-weight them.\n"
        "  3. Choose hyperparameters that cover the IQR with ~80% mass and the\n"
        "     full range with ~99% mass. State the math.\n"
        f"  4. Apply an uncertainty-inflation factor (default {config.uncertainty_inflation}×)\n"
        "     and explain.\n"
        "Output schema: {\"distribution\": string, \"params\": object,\n"
        "                \"reasoning\": string, \"cited_papers\": [string]}.\n"
    )


def _format_prior_evidence(aggregate_: PriorAggregate) -> str:
    lines: list[str] = []
    for record in aggregate_.records:
        parts: list[str] = []
        if record.distribution:
            parts.append(record.distribution)
            if record.params:
                params_txt = ", ".join(
                    f"{k}={v}" for k, v in sorted(record.params.items())
                )
                parts.append(f"({params_txt})")
        if record.low is not None or record.high is not None:
            parts.append(f"[{record.low}, {record.high}]")
        lines.append(
            f"  - {' '.join(parts) or 'unparsed'} — {record.paper_id}"
        )
    lines.append(f"  Range reported: {list(aggregate_.range_union)} across {aggregate_.n_sources} studies")
    lines.append(f"  IQR across studies: {list(aggregate_.range_iqr)}")
    return "\n".join(lines)


def _format_design_evidence(aggregate_: DesignAggregate) -> str:
    lines = [
        f"  - {choice}: {count} mentions"
        for choice, count in aggregate_.reported_choices.most_common()
    ]
    if not lines:
        lines.append("  - no explicit design choices reported")
    for record in aggregate_.records[:10]:
        lines.append(f"    {record['paper_id']}: {record['raw_text'][:120]}")
    return "\n".join(lines)


def _format_method_evidence(methods: Sequence[MethodAggregate]) -> str:
    if not methods:
        return ""
    return "\n".join(
        f"  - {m.method}: {m.count} papers ({', '.join(m.papers[:5])})"
        for m in methods
    )


def _coerce_prior_output(
    parsed: object,
    aggregate_: PriorAggregate,
    config: ReasoningConfig,
) -> tuple[dict, str]:
    if not isinstance(parsed, dict):
        dist, params = _fallback_prior(aggregate_, config)
        return (
            {
                "distribution": dist,
                "params": params,
                "uncertainty_inflation": config.uncertainty_inflation,
                "fallback": True,
            },
            "llm_output_unparsed_fallback",
        )
    distribution = parsed.get("distribution")
    params = parsed.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    reasoning = str(parsed.get("reasoning", "")).strip() or "no_reasoning_returned"
    conclusion = {
        "distribution": distribution,
        "params": {k: _maybe_float(v) or v for k, v in params.items()},
        "uncertainty_inflation": config.uncertainty_inflation,
        "fallback": False,
    }
    return conclusion, reasoning


def _fallback_prior(
    aggregate_: PriorAggregate, config: ReasoningConfig
) -> tuple[str, dict]:
    """Weakly-informative prior when evidence is thin."""
    low, high = aggregate_.range_union
    if low is None or high is None:
        return "Normal", {"loc": 0.0, "scale": 1.0}
    span = max(float(high) - float(low), 1e-6)
    centre = (float(low) + float(high)) / 2.0
    scale = span * config.uncertainty_inflation / 4.0  # 2 sigma ≈ half-span
    return "Normal", {"loc": centre, "scale": scale}


def _maybe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _emit(step: ReasoningStep, config: ReasoningConfig) -> ReasoningStep:
    """Live-validate a step before handing it to the caller.

    In strict mode, any numerical claim that lacks both a citation and a
    ``fallback=True`` marker raises ``LiveCitationError`` immediately —
    halting the pipeline before the downstream trace is assembled.
    """

    if config.strict_validation and not step_is_grounded(step):
        raise LiveCitationError(step.decision)
    return step


def run_stage_d(
    aggregation: AggregationResult,
    llm: LLMClient,
    *,
    config: ReasoningConfig | None = None,
    budget: TokenBudget | None = None,
    available_backends: Sequence[str] | None = None,
    include_design_reasoning: bool = True,
    include_backend_reasoning: bool = True,
) -> list[ReasoningStep]:
    """Run Stage D over every decision, returning all reasoning steps."""
    config = config or ReasoningConfig()
    steps: list[ReasoningStep] = []
    for parameter in sorted(aggregation.priors):
        if budget is not None and budget.check("stage_d"):
            break
        steps.append(
            _emit(
                reason_over_prior(
                    aggregation.priors[parameter], llm, config=config, budget=budget
                ),
                config,
            )
        )
    if include_design_reasoning:
        design_dimensions = sorted(aggregation.designs)
    else:
        design_dimensions = []
    for dimension in design_dimensions:
        if budget is not None and budget.check("stage_d"):
            break
        steps.append(
            _emit(
                reason_over_design(
                    aggregation.designs[dimension], llm, config=config, budget=budget
                ),
                config,
            )
        )
    if include_backend_reasoning and not (
        budget is not None and budget.check("stage_d")
    ):
        steps.append(
            _emit(
                reason_over_backend(
                    aggregation.methods,
                    llm,
                    config=config,
                    budget=budget,
                    available_backends=available_backends,
                ),
                config,
            )
        )
    return steps


__all__ = [
    "ReasoningConfig",
    "reason_over_backend",
    "reason_over_design",
    "reason_over_prior",
    "run_stage_d",
]
