"""Tests for the BOEDAgent orchestrator."""

from __future__ import annotations

import json

from boed_agent.agent import BOEDAgent
from boed_agent.literature.clients.base import Paper
from boed_agent.literature.llm_client import RecordingLLMClient
from boed_agent.literature.search import LiteratureSearchModule
from boed_agent.literature.token_budget import TokenBudget
from boed_agent.simulator_protocol import (
    ParameterInfo,
    SimpleSimulator,
    SimulatorMetadata,
)


def _sim(is_explicit: bool, is_diff: bool) -> SimpleSimulator:
    return SimpleSimulator(
        fn=lambda theta, xi: 0,
        metadata=SimulatorMetadata(
            parameters=[ParameterInfo(name="alpha")],
        ),
        is_explicit=is_explicit,
        is_differentiable=is_diff,
    )


def test_dry_run_without_literature_returns_backend():
    agent = BOEDAgent(
        simulator=_sim(True, True),
        design_distribution=None,
        problem_description="",
        use_literature=False,
    )
    result = agent.run(dry_run=True)
    assert result.chosen_backend == "pyro"
    assert result.literature_report is None


def test_dry_run_with_literature_attaches_report():
    def responder(prompt, tier):
        if "For each sentence" in prompt:
            return json.dumps(
                [
                    {
                        "id": 0,
                        "type": "prior_distribution",
                        "value": {
                            "parameter": "alpha",
                            "distribution": "Beta",
                            "params": {"a": 2, "b": 5},
                        },
                    }
                ]
            )
        if "propose a prior" in prompt:
            return json.dumps(
                {
                    "distribution": "Beta",
                    "params": {"a": 2, "b": 5},
                    "reasoning": "lit",
                    "cited_papers": ["demo"],
                }
            )
        if "rank the candidate" in prompt:
            return json.dumps(
                {
                    "ranked": ["PyroVI"],
                    "reasoning": "lit",
                    "cited_papers": ["demo"],
                }
            )
        return "{}"

    llm = RecordingLLMClient(responder=responder)
    module = LiteratureSearchModule(llm=llm, token_budget=TokenBudget())
    papers = [
        Paper(
            title="Demo",
            abstract=(
                "Prior Beta(2,5) for alpha. Variational BOED with EIG."
            ),
            doi="demo",
            year=2021,
            source="demo",
        )
    ]

    agent = BOEDAgent(
        simulator=_sim(True, True),
        design_distribution=None,
        problem_description="demo",
        use_literature=True,
        literature_module=module,
    )
    original_search = module.search
    agent._literature_module.search = lambda **kw: original_search(papers=papers, **kw)  # type: ignore[assignment]
    result = agent.run(dry_run=True)
    assert result.literature_report is not None
    assert "alpha" in result.prior_used.distributions
    assert result.prior_used.distributions["alpha"].source == "literature"


def test_user_prior_wins_even_with_literature(monkeypatch):
    def responder(prompt, tier):
        if "For each sentence" in prompt:
            return json.dumps(
                [
                    {
                        "id": 0,
                        "type": "prior_distribution",
                        "value": {
                            "parameter": "alpha",
                            "distribution": "Gamma",
                            "params": {"alpha": 2, "beta": 3},
                        },
                    }
                ]
            )
        if "propose a prior" in prompt:
            return json.dumps(
                {"distribution": "Gamma", "params": {"alpha": 2, "beta": 3},
                 "reasoning": "", "cited_papers": ["demo"]}
            )
        if "rank" in prompt:
            return json.dumps({"ranked": ["PyroVI"], "reasoning": "", "cited_papers": ["demo"]})
        return "{}"

    llm = RecordingLLMClient(responder=responder)
    module = LiteratureSearchModule(llm=llm)
    papers = [Paper(title="x", abstract="alpha prior Gamma(2,3). BOED EIG.", doi="demo", source="demo")]

    user_prior = {"alpha": {"distribution": "Uniform", "params": {"low": 0, "high": 1}}}
    agent = BOEDAgent(
        simulator=_sim(True, True),
        design_distribution=None,
        problem_description="demo",
        prior=user_prior,
        use_literature=True,
        literature_module=module,
    )
    original_search = module.search
    agent._literature_module.search = lambda **kw: original_search(papers=papers, **kw)  # type: ignore[assignment]
    result = agent.run(dry_run=True)
    assert result.prior_used.distributions["alpha"].name == "Uniform"
    assert any("Literature suggests" in w for w in result.prior_used.warnings)


def test_token_budget_is_respected():
    # Budget of zero → every LLM call counts as "over budget" right away.
    def responder(prompt, tier):
        return json.dumps([])

    llm = RecordingLLMClient(responder=responder)
    budget = TokenBudget(max_total_tokens=1)
    module = LiteratureSearchModule(llm=llm, token_budget=budget)
    papers = [
        Paper(
            title="demo",
            abstract="alpha LogNormal(-1, 0.5). BOED.",
            doi="demo",
            source="demo",
        )
    ]
    agent = BOEDAgent(
        simulator=_sim(True, True),
        design_distribution=None,
        problem_description="demo",
        use_literature=True,
        literature_module=module,
        token_budget=budget,
    )
    original_search = module.search
    agent._literature_module.search = lambda **kw: original_search(papers=papers, **kw)  # type: ignore[assignment]
    result = agent.run(dry_run=True)
    # Pipeline still completes — just with no reasoning steps beyond fallbacks.
    assert result.literature_report is not None


def test_classifier_attached_when_data_supplied():
    agent = BOEDAgent(
        simulator=_sim(True, True),
        design_distribution=None,
        problem_description="",
        use_literature=False,
        data=[1.0, 1.1, 0.9, 1.05],
    )
    result = agent.run(dry_run=True)
    assert result.classifier_result is not None
    assert result.classifier_result.homogeneous is True
