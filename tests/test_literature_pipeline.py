"""Tests for the 5-stage literature pipeline."""

from __future__ import annotations

import json

import pytest

from boed_agent.literature.aggregation import aggregate
from boed_agent.literature.clients.base import Paper
from boed_agent.literature.extraction import (
    ExtractionConfig,
    run_extraction,
    split_sentences,
)
from boed_agent.literature.filters import FilterResult, prefilter
from boed_agent.literature.llm_client import RecordingLLMClient
from boed_agent.literature.reasoning import (
    ReasoningConfig,
    reason_over_prior,
    run_stage_d,
)
from boed_agent.literature.search import LiteratureSearchModule
from boed_agent.literature.search import LiteratureSearchConfig
from boed_agent.literature.token_budget import TokenBudget
from boed_agent.literature.trace import ReasoningTrace, validate_citations
from boed_agent.simulator_protocol import ParameterInfo, SimulatorMetadata


_METADATA = SimulatorMetadata(
    parameters=[ParameterInfo(name="k_a", units="1/hr")],
    domain_tags=["pk"],
)


@pytest.fixture
def fixture_papers():
    return [
        Paper(
            title="PK study of drug X",
            abstract=(
                "We placed a LogNormal prior on k_a with mu=-1.2, sigma=0.4. "
                "BOED with EIG converged in 200 steps. "
                "k_a range 0.05 to 3.1."
            ),
            doi="10.0/a",
            year=2021,
            citation_count=10,
            source="semantic_scholar",
        ),
        Paper(
            title="Unrelated botany paper",
            abstract="Trees grow tall in the forest.",
            doi="10.0/b",
            year=2019,
            source="arxiv",
        ),
    ]


def test_stage_a_filters_drop_unrelated(fixture_papers):
    results = prefilter(fixture_papers, simulator_metadata=_METADATA, min_signals=1)
    keep = [r.keep for r in results]
    assert keep[0] is True
    assert keep[1] is False


def test_sentence_splitter():
    s = split_sentences("Hello world. This is a test. Great!")
    assert s == ["Hello world.", "This is a test.", "Great!"]


def test_stage_b_extracts_structured_records(fixture_papers):
    def responder(prompt, tier):
        return json.dumps(
            [
                {
                    "id": 0,
                    "type": "prior_distribution",
                    "value": {
                        "parameter": "k_a",
                        "distribution": "LogNormal",
                        "params": {"mu": -1.2, "sigma": 0.4},
                    },
                },
                {
                    "id": 1,
                    "type": "method_used",
                    "value": "BOED",
                },
                {
                    "id": 2,
                    "type": "prior_range",
                    "value": {"parameter": "k_a", "low": 0.05, "high": 3.1},
                },
            ]
        )

    llm = RecordingLLMClient(responder=responder)
    budget = TokenBudget()
    filtered = prefilter(fixture_papers, simulator_metadata=_METADATA, min_signals=1)
    records, stats = run_extraction(filtered, llm, budget=budget)
    assert stats.papers_processed == 1
    assert any(r.type == "prior_distribution" for r in records)
    assert any(r.type == "prior_range" for r in records)
    assert budget.total_tokens > 0


def test_stage_b_uses_body_sections_for_local_corpus_content():
    def responder(prompt, tier):
        if "Beta(2,5)" not in prompt:
            return json.dumps([])
        return json.dumps(
            [
                {
                    "id": 0,
                    "type": "prior_distribution",
                    "value": {
                        "parameter": "k_a",
                        "distribution": "Beta",
                        "params": {"a": 2, "b": 5},
                    },
                }
            ]
        )

    paper = Paper(
        title="Local corpus note",
        abstract="",
        doi="local-1",
        source="local_corpus",
        sections={
            "body": "We estimate k_a with a Beta(2,5) prior from the local corpus."
        },
    )
    llm = RecordingLLMClient(responder=responder)
    filtered = [
        FilterResult(
            paper=paper,
            keep=True,
            matched_terms=[],
            matched_parameters=["k_a"],
            matched_distributions=["Beta"],
            matched_ranges=0,
            reason="body-only regression fixture",
        )
    ]

    records, stats = run_extraction(filtered, llm, budget=TokenBudget())

    assert stats.papers_processed == 1
    assert any(record.type == "prior_distribution" for record in records)
    assert any("local corpus" in record.raw_text for record in records)


def test_stage_c_aggregation():
    # Hand-craft records to exercise the deterministic aggregator.
    from boed_agent.literature.extraction import EvidenceRecord

    records = [
        EvidenceRecord(
            paper_id="p1",
            sentence_id=0,
            raw_text="...",
            type="prior_distribution",
            value={"parameter": "k_a", "distribution": "LogNormal", "params": {"mu": -1.2, "sigma": 0.4}},
        ),
        EvidenceRecord(
            paper_id="p2",
            sentence_id=0,
            raw_text="...",
            type="prior_distribution",
            value={"parameter": "k_a", "distribution": "LogNormal", "params": {"mu": -0.9, "sigma": 0.5}},
        ),
        EvidenceRecord(
            paper_id="p3",
            sentence_id=0,
            raw_text="...",
            type="prior_distribution",
            value={"parameter": "k_a", "distribution": "Gamma", "params": {"alpha": 2, "beta": 3}},
        ),
        EvidenceRecord(
            paper_id="p1",
            sentence_id=1,
            raw_text="...",
            type="prior_range",
            value={"parameter": "k_a", "low": 0.05, "high": 3.1},
        ),
        EvidenceRecord(
            paper_id="p4",
            sentence_id=0,
            raw_text="...",
            type="method_used",
            value="MINEBED",
        ),
    ]
    aggregation = aggregate(records, parameter_names=["k_a"])
    assert "k_a" in aggregation.priors
    assert aggregation.priors["k_a"].n_sources == 3
    assert aggregation.priors["k_a"].reported_distributions["LogNormal"] == 2
    assert aggregation.methods[0].method == "MINEBED"


def test_stage_d_fallback_when_evidence_thin():
    from boed_agent.literature.aggregation import PriorAggregate, PriorEvidence

    aggregate_ = PriorAggregate(
        parameter="k_a",
        records=[
            PriorEvidence(
                paper_id="p1",
                sentence_id=0,
                raw_text="...",
                distribution="LogNormal",
                params={"mu": -1.0, "sigma": 0.5},
                low=0.05,
                high=3.0,
            )
        ],
        n_sources=1,
        range_union=(0.05, 3.0),
    )
    config = ReasoningConfig(min_sources_for_llm=3)
    called = {"hits": 0}

    def responder(prompt, tier):
        called["hits"] += 1
        return "{}"

    step = reason_over_prior(
        aggregate_, RecordingLLMClient(responder=responder), config=config
    )
    assert step.is_fallback is True
    assert step.reasoning == "insufficient_evidence_fallback"
    assert called["hits"] == 0  # LLM never invoked


def test_token_budget_halts_pipeline():
    """Pipeline stops cleanly when the budget is exhausted."""
    papers = [
        Paper(
            title=f"Paper {i}",
            abstract=(
                "We used LogNormal(mu=-1, sigma=0.5) prior on k_a. "
                "BOED with EIG ran for 100 steps. k_a in 0.05 to 3.0."
            ),
            doi=f"p{i}",
            year=2020 + i,
            source="demo",
        )
        for i in range(5)
    ]

    def responder(prompt, tier):
        return json.dumps([])  # no records

    llm = RecordingLLMClient(responder=responder)
    budget = TokenBudget(max_total_tokens=50)
    filtered = prefilter(papers, simulator_metadata=_METADATA)
    _, stats = run_extraction(filtered, llm, budget=budget)
    # Pipeline halted early once the budget was exceeded — not every paper
    # was processed (there are 5 in the fixture).
    assert stats.papers_processed < len(papers)


def test_literature_search_module_with_fixture_papers():
    def responder(prompt, tier):
        if "For each sentence below" in prompt:
            return json.dumps(
                [
                    {
                        "id": 0,
                        "type": "prior_distribution",
                        "value": {
                            "parameter": "k_a",
                            "distribution": "LogNormal",
                            "params": {"mu": -1.2, "sigma": 0.4},
                        },
                    },
                    {
                        "id": 1,
                        "type": "method_used",
                        "value": "MINEBED",
                    },
                ]
            )
        if "propose a prior" in prompt:
            return json.dumps(
                {
                    "distribution": "LogNormal",
                    "params": {"mu": -1.1, "sigma": 0.5},
                    "reasoning": "lit supports LogNormal",
                    "cited_papers": ["10.0/a"],
                }
            )
        if "rank the candidate" in prompt:
            return json.dumps(
                {
                    "ranked": ["MINEBED", "PyroVI"],
                    "reasoning": "mine dominates",
                    "cited_papers": ["10.0/a"],
                }
            )
        return "{}"

    llm = RecordingLLMClient(responder=responder)
    from boed_agent.literature.reasoning import ReasoningConfig
    from boed_agent.literature.search import LiteratureSearchConfig

    module = LiteratureSearchModule(
        llm=llm,
        config=LiteratureSearchConfig(
            reasoning=ReasoningConfig(min_sources_for_llm=1),
        ),
    )
    papers = [
        Paper(
            title="PK paper",
            abstract=(
                "We placed a LogNormal prior on k_a with mu=-1.2, sigma=0.4. "
                "BOED ran 200 steps."
            ),
            doi="10.0/a",
            year=2021,
            source="semantic_scholar",
        )
    ]
    report = module.search(
        problem_description="pharmacokinetic absorption-rate inference",
        simulator_metadata=_METADATA,
        papers=papers,
    )
    assert "k_a" in report.prior_suggestions
    assert report.prior_suggestions["k_a"].distribution == "LogNormal"
    assert report.backend_preference.ranked == ["PyroVI", "LFIAX"]
    md = report.reasoning_trace.to_markdown()
    # Every numerical recommendation has a citation in the rendered markdown.
    assert "10.0/a" in md


def test_literature_search_reports_zero_prior_reasoning_explicitly():
    def responder(prompt, tier):
        if "For each sentence below" in prompt:
            return json.dumps(
                [
                    {
                        "id": 0,
                        "type": "method_used",
                        "value": "BOED",
                    },
                ]
            )
        if "rank the candidate" in prompt:
            return json.dumps(
                {
                    "ranked": ["PyroVI", "LFIAX"],
                    "reasoning": "available backend ranking",
                    "cited_papers": ["10.0/no-prior"],
                }
            )
        return "{}"

    llm = RecordingLLMClient(responder=responder)
    module = LiteratureSearchModule(llm=llm)
    papers = [
        Paper(
            title="PK BOED paper without explicit priors",
            abstract="We estimate k_a with BOED and report posterior performance.",
            doi="10.0/no-prior",
            year=2022,
            source="semantic_scholar",
        )
    ]
    report = module.search(
        problem_description="pharmacokinetic absorption-rate inference",
        simulator_metadata=_METADATA,
        papers=papers,
    )

    assert report.prior_suggestions == {}
    assert report.diagnostics["prior_reasoning_step_count"] == 0
    assert report.diagnostics["zero_prior_reasoning_failure"] is True
    assert report.diagnostics["extracted_record_counts"] == {"method_used": 1}
    assert report.diagnostics["missing_prior_parameters"] == ["k_a"]
    assert any("zero_prior_reasoning_steps" in note for note in report.notes)


def test_prior_only_search_extracts_measurements_and_skips_boed_reasoning():
    metadata = SimulatorMetadata(
        parameters=[ParameterInfo(name="kd")],
        domain_tags=["bmp4_gradient"],
    )

    def responder(prompt, tier):
        if "For each sentence below" in prompt:
            assert "This run is synthesizing parameter priors" in prompt
            assert "Target prior parameters: kd" in prompt
            return json.dumps(
                [
                    {
                        "id": 0,
                        "type": "prior_range",
                        "value": {
                            "parameter": "kd",
                            "low": 20.0,
                            "high": 20.0,
                            "params": {"raw_kd_nm": 0.5},
                        },
                    }
                ]
            )
        if "propose a prior" in prompt:
            assert "K_eqtk = 10 / K_d_nM" in prompt
            return json.dumps(
                {
                    "distribution": "LogNormal",
                    "params": {"loc": 3.0, "scale": 0.5},
                    "reasoning": "Converted the reported 0.5 nM receptor Kd into the model affinity scale.",
                    "cited_papers": ["10.0/kd"],
                }
            )
        if "rank the candidate BOED backends" in prompt:
            raise AssertionError("prior-only search should not reason about BOED backends")
        return "{}"

    module = LiteratureSearchModule(
        llm=RecordingLLMClient(responder=responder),
        config=LiteratureSearchConfig(
            prior_only=True,
            include_design_reasoning=False,
            include_backend_reasoning=False,
            extraction=ExtractionConfig(prior_synthesis_mode=True),
            reasoning=ReasoningConfig(min_sources_for_llm=1),
        ),
    )
    papers = [
        Paper(
            title="BMP4 receptor affinity",
            abstract="BMP4 binds BMPR2 with a Kd of 0.5 nM in SPR measurements. BOED was not studied.",
            doi="10.0/kd",
            year=2021,
            source="bmp4_local",
        )
    ]

    report = module.search(
        problem_description="BMP4 receptor affinity prior. Convert SPR dissociation constants as K_eqtk = 10 / K_d_nM.",
        simulator_metadata=metadata,
        papers=papers,
    )

    assert report.prior_suggestions["kd"].distribution == "LogNormal"
    assert report.backend_preference.ranked == []
    assert [step.decision for step in report.reasoning_trace.steps] == ["prior for kd"]


def test_validate_citations_flags_ungrounded():
    trace = ReasoningTrace()
    from boed_agent.literature.trace import ReasoningStep

    trace.record(
        [
            ReasoningStep(
                decision="prior for x",
                evidence_summary="",
                reasoning="",
                conclusion={"distribution": "Normal", "fallback": False},
                cited_papers=[],
            )
        ]
    )
    bad = validate_citations(trace.steps)
    assert bad == ["prior for x"]
