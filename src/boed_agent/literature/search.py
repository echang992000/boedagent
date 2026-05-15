"""The :class:`LiteratureSearchModule` orchestrator.

Runs the 5-stage pipeline end-to-end.  The module is careful to never
fail hard on network errors — a partial report is strictly better
than a missing one, because :class:`BOEDAgent` uses the report for
*suggestions* only (the user's own inputs are never overridden).
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from boed_agent.literature.aggregation import aggregate
from boed_agent.literature.clients.arxiv import ArxivClient
from boed_agent.literature.clients.base import Paper
from boed_agent.literature.clients.openalex import OpenAlexClient
from boed_agent.literature.clients.pubmed import PubMedClient
from boed_agent.literature.clients.semantic_scholar import SemanticScholarClient
from boed_agent.literature.extraction import (
    ExtractionConfig,
    ExtractionStats,
    run_extraction,
)
from boed_agent.literature.filters import PRIOR_EVIDENCE_TERMS, FilterResult, prefilter
from boed_agent.literature.llm_client import LLMClient, NullLLMClient
from boed_agent.literature.reasoning import ReasoningConfig, run_stage_d
from boed_agent.literature.report import CostReport, LiteratureReport
from boed_agent.literature.token_budget import TokenBudget
from boed_agent.literature.trace import ReasoningTrace
from boed_agent.simulator_protocol import SimulatorMetadata


@dataclass
class SourceBundle:
    semantic_scholar: SemanticScholarClient | None = None
    arxiv: ArxivClient | None = None
    openalex: OpenAlexClient | None = None
    pubmed: PubMedClient | None = None
    # Arbitrary extra sources (e.g., ``LocalCorpusClient`` or a
    # user-defined HTTP client).  Each entry is ``(name, client)`` —
    # ``client`` only needs a ``.search(query, limit) -> list[Paper]``
    # method; the name is used for provenance and error reporting.
    extra: list[tuple[str, Any]] = field(default_factory=list)

    def all(self) -> list[tuple[str, Any]]:
        named: list[tuple[str, Any]] = [
            (name, client)
            for name, client in [
                ("semantic_scholar", self.semantic_scholar),
                ("arxiv", self.arxiv),
                ("openalex", self.openalex),
                ("pubmed", self.pubmed),
            ]
            if client is not None
        ]
        named.extend((n, c) for n, c in self.extra if c is not None)
        return named


@dataclass
class RankingWeights:
    source_score: float = 1.0
    log_citation: float = 0.5
    recency: float = 0.8
    keyword: float = 1.0


@dataclass
class LiteratureSearchConfig:
    max_papers: int = 20
    min_filter_signals: int = 1
    current_year: int = 2025
    recency_tau: float = 10.0
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    weights: RankingWeights = field(default_factory=RankingWeights)
    available_backends: tuple[str, ...] = ("PyroVI", "LFIAX")
    verbose: bool = False
    prior_only: bool = False
    include_design_reasoning: bool = True
    include_backend_reasoning: bool = True
    # Templates for Stage 1 query construction.  Overridden when the
    # user supplies ``queries`` directly to ``.search(...)``.
    query_templates: tuple[str, ...] = (
        "{desc}",
        "{desc} Bayesian experimental design",
        "{desc} expected information gain",
        "{desc} prior distribution parameters",
        "{desc} simulator likelihood-free inference",
    )
    prior_query_templates: tuple[str, ...] = (
        "{desc} prior distribution parameters",
        "{desc} parameter prior measurements",
        "{desc} binding affinity Kd dissociation constant receptor",
        "{desc} dose response EC50 half maximal Hill coefficient",
        "{desc} response baseline maximum observation noise",
    )


class LiteratureSearchModule:
    """End-to-end orchestrator for the 5-stage pipeline."""

    def __init__(
        self,
        sources: SourceBundle | None = None,
        llm: LLMClient | None = None,
        *,
        token_budget: TokenBudget | None = None,
        config: LiteratureSearchConfig | None = None,
    ) -> None:
        self.sources = sources or SourceBundle()
        self.llm = llm or NullLLMClient()
        self.token_budget = token_budget or TokenBudget()
        self.config = config or LiteratureSearchConfig()

    # --- public API -------------------------------------------------

    def search(
        self,
        *,
        problem_description: str,
        simulator_metadata: SimulatorMetadata | Mapping[str, Any] | None = None,
        queries: Sequence[str] | None = None,
        papers: Sequence[Paper] | None = None,
    ) -> LiteratureReport:
        """Run the five-stage pipeline and return a :class:`LiteratureReport`."""
        metadata = _coerce_metadata(simulator_metadata)
        report = LiteratureReport()
        self._log("Starting literature search")
        if metadata is not None:
            self._log(
                "Target parameters: "
                + (", ".join(metadata.parameter_names) if metadata.parameter_names else "<none>")
            )
        self._log("Available backend candidates: " + ", ".join(self.config.available_backends))

        # Stage 1 — query construction (rule-based).
        effective_queries = list(queries) if queries else self._build_queries(
            problem_description
        )
        report.queries = list(effective_queries)
        self._log(f"Stage 1 queries: {len(effective_queries)}")
        for index, query in enumerate(effective_queries, start=1):
            self._log(f"  query[{index}]: {query}")

        # Stage 2 — multi-source fanout.
        candidates: list[Paper] = []
        if papers is not None:
            candidates = list(papers)
            self._log(f"Stage 2 using supplied papers: {len(candidates)}")
        else:
            sources = self.sources.all()
            self._log(
                "Stage 2 source fanout: "
                + (", ".join(name for name, _client in sources) if sources else "<none>")
            )
            for query in effective_queries:
                for name, client in sources:
                    try:
                        fetched = client.search(query, limit=self.config.max_papers)
                    except Exception as exc:  # pragma: no cover - network guard
                        report.notes.append(f"source error: {exc}")
                        self._log(f"  source error from {name}: {exc}")
                        continue
                    self._log(f"  {name}: fetched {len(fetched)} papers")
                    candidates.extend(fetched)

        dedup = _dedup_papers(candidates)
        ranked = self._rank(dedup, metadata)[: self.config.max_papers]
        report.papers = [p.to_dict() for p in ranked]
        self._log(
            f"Stage 2 candidates: raw={len(candidates)}, dedup={len(dedup)}, ranked={len(ranked)}"
        )
        for index, paper in enumerate(ranked[:5], start=1):
            self._log(
                f"  ranked[{index}]: {paper.title or '<untitled>'} "
                f"(source={paper.source}, id={paper.paper_id})"
            )

        # Stage A — prefilter.
        filtered: list[FilterResult] = prefilter(
            ranked,
            simulator_metadata=metadata,
            min_signals=self.config.min_filter_signals,
            include_method_terms=not self.config.prior_only,
            extra_terms=PRIOR_EVIDENCE_TERMS if self.config.prior_only else (),
        )
        kept = [r for r in filtered if r.keep]
        self._log(f"Stage A prefilter: kept={len(kept)}/{len(filtered)}")
        for item in filtered[:8]:
            terms = ", ".join(item.matched_terms[:5])
            params = ", ".join(item.matched_parameters[:5])
            self._log(
                f"  filter {item.paper.title or '<untitled>'}: keep={item.keep}, "
                f"reason={item.reason}, terms=[{terms}], params=[{params}], "
                f"ranges={item.matched_ranges}"
            )

        parameter_names = metadata.parameter_names if metadata else []

        # Stage B — extraction.
        records, extraction_stats = run_extraction(
            filtered,
            self.llm,
            budget=self.token_budget,
            config=self.config.extraction,
            parameter_names=parameter_names,
            problem_description=problem_description,
        )
        record_counts = Counter(record.type for record in records)
        self._log(
            "Stage B extraction: "
            f"papers_processed={extraction_stats.papers_processed}, "
            f"sentences={extraction_stats.sentences_in}, "
            f"batches={extraction_stats.batches}, "
            f"records={extraction_stats.records_out}, "
            f"record_types={dict(record_counts)}"
        )
        for record in records[:10]:
            self._log(
                f"  record {record.type}: value={record.value} "
                f"paper={record.paper_id} sentence={record.sentence_id}"
        )

        # Stage C — aggregation.
        aggregation = aggregate(
            records,
            parameter_names=parameter_names or None,
        )
        report.benchmarks = list(aggregation.benchmarks)
        self._log(
            "Stage C aggregation: "
            f"priors={list(aggregation.priors)}, "
            f"designs={list(aggregation.designs)}, "
            f"methods={[method.method for method in aggregation.methods]}, "
            f"benchmarks={len(aggregation.benchmarks)}"
        )
        for name, aggregate_ in aggregation.priors.items():
            self._log(
                f"  prior aggregate {name}: sources={aggregate_.n_sources}, "
                f"records={len(aggregate_.records)}, "
                f"range={aggregate_.range_union}, "
                f"distributions={dict(aggregate_.reported_distributions)}"
            )

        # Stage D — reasoning.
        reasoning_config = self.config.reasoning
        if problem_description and not reasoning_config.problem_context:
            reasoning_config = replace(
                reasoning_config,
                problem_context=problem_description,
            )
        steps = run_stage_d(
            aggregation,
            self.llm,
            config=reasoning_config,
            budget=self.token_budget,
            available_backends=self.config.available_backends,
            include_design_reasoning=self.config.include_design_reasoning,
            include_backend_reasoning=self.config.include_backend_reasoning,
        )
        self._log(f"Stage D reasoning steps: {len(steps)}")
        for step in steps:
            self._log(
                f"  {step.decision}: fallback={step.is_fallback}, "
                f"tokens={step.token_cost}, conclusion={step.conclusion}"
            )

        # Stage E — trace assembly.
        report.reasoning_trace = ReasoningTrace()
        report.reasoning_trace.total_api_calls = self.token_budget.api_calls
        report.absorb_steps(steps)
        report.diagnostics.update(
            _build_prior_diagnostics(
                parameter_names=parameter_names,
                record_counts=record_counts,
                aggregation_priors=aggregation.priors,
                prior_suggestions=report.prior_suggestions,
                reasoning_steps=steps,
            )
        )
        if report.diagnostics.get("zero_prior_reasoning_failure"):
            message = str(report.diagnostics["zero_prior_reasoning_reason"])
            report.notes.append(f"zero_prior_reasoning_steps: {message}")
            self._log(f"WARNING: zero prior reasoning steps: {message}")
        self._log(
            "Stage E report: "
            f"prior_suggestions={list(report.prior_suggestions)}, "
            f"design_hints={list(report.design_space_hints)}, "
            f"backend_rank={report.backend_preference.ranked}"
        )

        # Cost report.
        report.cost_report = CostReport(
            tokens_by_stage=dict(self.token_budget.per_stage),
            total_tokens=self.token_budget.total_tokens,
            api_calls=self.token_budget.api_calls,
            papers_considered=len(ranked),
            papers_filtered=len(ranked) - len(kept),
            papers_processed=extraction_stats.papers_processed,
        )
        self._log(
            "Cost report: "
            f"tokens={report.cost_report.total_tokens}, "
            f"api_calls={report.cost_report.api_calls}, "
            f"tokens_by_stage={report.cost_report.tokens_by_stage}"
        )
        return report

    # --- helpers ----------------------------------------------------

    def _build_queries(self, problem_description: str) -> list[str]:
        desc = (problem_description or "").strip()
        if not desc:
            return []
        seen: set[str] = set()
        result: list[str] = []
        templates = (
            self.config.prior_query_templates
            if self.config.prior_only
            else self.config.query_templates
        )
        for template in templates:
            q = template.format(desc=desc).strip()
            if q and q not in seen:
                seen.add(q)
                result.append(q)
            if len(result) >= 5:
                break
        return result

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[literature] {message}", flush=True)

    def _rank(
        self,
        papers: Sequence[Paper],
        metadata: SimulatorMetadata | None,
    ) -> list[Paper]:
        weights = self.config.weights
        current = self.config.current_year

        def score(paper: Paper) -> float:
            base = weights.source_score * paper.source_score
            cites = math.log1p(max(paper.citation_count, 0))
            base += weights.log_citation * cites
            if paper.year:
                age = max(current - int(paper.year), 0)
                base += weights.recency * math.exp(-age / max(self.config.recency_tau, 1e-6))
            base += weights.keyword * _keyword_score(paper, metadata)
            return base

        return sorted(papers, key=score, reverse=True)


def _dedup_papers(papers: Sequence[Paper]) -> list[Paper]:
    seen: dict[str, Paper] = {}
    for paper in papers:
        key = paper.paper_id
        if key not in seen:
            seen[key] = paper
            continue
        # Prefer the record with a longer abstract.
        if len(paper.abstract or "") > len(seen[key].abstract or ""):
            seen[key] = paper
    return list(seen.values())


_KEYWORDS = {"boed", "eig", "mine", "design", "prior", "posterior"}


def _keyword_score(paper: Paper, metadata: SimulatorMetadata | None) -> float:
    haystack = f"{paper.title} {paper.abstract}".lower()
    hits = sum(1 for kw in _KEYWORDS if kw in haystack)
    if metadata:
        for name in metadata.parameter_names:
            if name and name.lower() in haystack:
                hits += 1
    return float(hits)


def _coerce_metadata(
    value: SimulatorMetadata | Mapping[str, Any] | None,
) -> SimulatorMetadata | None:
    if value is None:
        return None
    if isinstance(value, SimulatorMetadata):
        return value
    return SimulatorMetadata.from_dict(value)


def _build_prior_diagnostics(
    *,
    parameter_names: Sequence[str],
    record_counts: Counter,
    aggregation_priors: Mapping[str, Any],
    prior_suggestions: Mapping[str, Any],
    reasoning_steps: Sequence[Any],
) -> dict[str, Any]:
    prior_step_count = sum(
        1
        for step in reasoning_steps
        if getattr(step, "decision", "").startswith("prior for ")
    )
    target_parameters = list(parameter_names)
    prior_record_count = int(
        record_counts.get("prior_range", 0)
        + record_counts.get("prior_distribution", 0)
    )
    aggregated_prior_parameters = list(aggregation_priors)
    suggested_prior_parameters = list(prior_suggestions)
    missing_prior_parameters = [
        name for name in target_parameters if name not in prior_suggestions
    ]

    zero_prior_failure = bool(target_parameters and prior_step_count == 0)
    reason = ""
    if zero_prior_failure:
        if prior_record_count == 0:
            reason = (
                "Stage B extracted no prior_range or prior_distribution records "
                "for the target parameters, so Stage C produced no prior aggregates "
                "and Stage D had no prior decisions to reason over."
            )
        elif not aggregated_prior_parameters:
            reason = (
                "Stage B extracted prior-like records, but Stage C could not map "
                "them onto the target parameter names, so Stage D had no prior "
                "decisions to reason over."
            )
        else:
            reason = (
                "Stage C produced prior aggregates, but Stage D emitted no "
                "prior decisions. This indicates a reasoning-stage failure or "
                "unexpected aggregate filtering."
            )

    return {
        "target_prior_parameters": target_parameters,
        "extracted_record_counts": dict(record_counts),
        "extracted_prior_record_count": prior_record_count,
        "aggregated_prior_parameters": aggregated_prior_parameters,
        "suggested_prior_parameters": suggested_prior_parameters,
        "missing_prior_parameters": missing_prior_parameters,
        "prior_reasoning_step_count": prior_step_count,
        "zero_prior_reasoning_failure": zero_prior_failure,
        "zero_prior_reasoning_reason": reason,
    }


__all__ = [
    "LiteratureSearchConfig",
    "LiteratureSearchModule",
    "RankingWeights",
    "SourceBundle",
]
