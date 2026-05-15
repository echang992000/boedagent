"""Example: lit-only dry-run → returns a LiteratureReport + ReasoningTrace.

No inference is performed.  The script is useful as a smoke test that
the literature pipeline runs end-to-end with a local corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

from boed_agent import (
    BOEDAgent,
    ParameterInfo,
    SimpleSimulator,
    SimulatorMetadata,
    TokenBudget,
)
from boed_agent.literature.clients import LocalCorpusClient
from boed_agent.literature.llm_client import RecordingLLMClient
from boed_agent.literature.search import LiteratureSearchModule, SourceBundle


def _responder(prompt: str, tier: str) -> str:
    if "For each sentence below" in prompt:
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
                "reasoning": "Single source. Using as-is with 1.5x inflation.",
                "cited_papers": ["demo-1"],
            }
        )
    if "rank the candidate" in prompt:
        return json.dumps(
            {
                "ranked": ["PyroVI", "LFIAX"],
                "reasoning": "Only one method mentioned in evidence.",
                "cited_papers": ["demo-1"],
            }
        )
    return "{}"


def main() -> None:
    corpus_dir = Path(__file__).with_name("local_corpus") / "lit_only"
    metadata = SimulatorMetadata(
        parameters=[ParameterInfo(name="alpha")],
        domain_tags=["demo"],
    )
    simulator = SimpleSimulator(
        fn=lambda theta, xi: theta[0] + xi[0],
        metadata=metadata,
        is_explicit=True,
        is_differentiable=True,
    )
    llm = RecordingLLMClient(responder=_responder)
    local_client = LocalCorpusClient(corpus_dir=corpus_dir, source_name="local_demo")
    lit_module = LiteratureSearchModule(
        sources=SourceBundle(extra=[("local_demo", local_client)]),
        llm=llm,
        token_budget=TokenBudget(),
    )

    agent = BOEDAgent(
        simulator=simulator,
        design_distribution=None,
        problem_description="Toy literature-only smoke test",
        use_literature=True,
        literature_module=lit_module,
    )

    dry = agent.run(dry_run=True)
    print("Backend:", dry.chosen_backend)
    print("Cost report:", dry.literature_report.cost_report.to_dict())
    print()
    print(dry.reasoning_trace.to_markdown())


if __name__ == "__main__":
    main()
