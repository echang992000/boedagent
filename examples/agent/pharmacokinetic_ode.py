"""Example: pharmacokinetic ODE → iDAD / MINEBED with literature priors.

The simulator is differentiable but implicit, so the waterfall
dispatcher selects MINEBED by default.  Setting
``backend_options['policy_network']=True`` switches to iDAD.

A recording LLM client plus a local corpus keep the run fully
deterministic and offline.
"""

from __future__ import annotations

import json
import math
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
    # Very small rule-based responder: emits plausible JSON for both
    # stages so the pipeline exercises its full code path.
    if "For each sentence below" in prompt:
        # Stage B — emit structured evidence.
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
                    "type": "prior_range",
                    "value": {"parameter": "k_a", "low": 0.05, "high": 3.1},
                },
                {
                    "id": 2,
                    "type": "method_used",
                    "value": "BOED",
                },
            ]
        )
    if "propose a prior" in prompt:
        return json.dumps(
            {
                "distribution": "LogNormal",
                "params": {"mu": -1.05, "sigma": 0.5},
                "reasoning": (
                    "Dominant family is LogNormal; Gamma is an outlier "
                    "down-weighted. Hyperparameters cover the IQR with "
                    "~80% mass after 1.5x inflation."
                ),
                "cited_papers": ["10.0/smith2019", "10.0/chen2021"],
            }
        )
    if "rank the candidate BOED backends" in prompt:
        return json.dumps(
            {
                "ranked": ["iDAD", "MINEBED", "PyroVI", "LFIAX"],
                "reasoning": "Policy-network literature points to iDAD for this differentiable implicit simulator.",
                "cited_papers": ["10.0/smith2019", "10.0/chen2021"],
            }
        )
    return "{}"


def _one_compartment(theta, xi):
    k_a, k_e, V = theta
    t = xi[0]
    if abs(k_a - k_e) < 1e-9:
        k_e = k_a + 1e-6
    return (k_a / (V * (k_a - k_e))) * (math.exp(-k_e * t) - math.exp(-k_a * t))


def main() -> None:
    corpus_dir = Path(__file__).with_name("local_corpus") / "pharmacokinetic_ode"
    metadata = SimulatorMetadata(
        parameters=[
            ParameterInfo(name="k_a", units="1/hr", description="absorption rate"),
            ParameterInfo(name="k_e", units="1/hr", description="elimination rate"),
            ParameterInfo(name="V", units="L", description="distribution volume"),
        ],
        observation_labels=["concentration"],
        domain_tags=["pharmacokinetics", "ode"],
    )
    simulator = SimpleSimulator(
        fn=_one_compartment,
        metadata=metadata,
        is_explicit=False,
        is_differentiable=True,
        name="pk_ode",
    )

    llm = RecordingLLMClient(responder=_responder)
    local_client = LocalCorpusClient(corpus_dir=corpus_dir, source_name="local_pk")
    lit_module = LiteratureSearchModule(
        sources=SourceBundle(extra=[("local_pk", local_client)]),
        llm=llm,
        token_budget=TokenBudget(),
    )

    agent = BOEDAgent(
        simulator=simulator,
        design_distribution={"dose_time": {"lower": 0.0, "upper": 24.0}},
        problem_description="Pharmacokinetic one-compartment ODE with three parameters",
        prior=None,
        use_literature=True,
        literature_module=lit_module,
        backend_options={"policy_network": True},  # → iDAD
    )

    result = agent.run(dry_run=True)
    print("Chosen backend:", result.chosen_backend)
    print()
    print(result.reasoning_trace.to_markdown())


if __name__ == "__main__":
    main()
