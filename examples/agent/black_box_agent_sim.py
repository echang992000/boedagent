"""Example: non-differentiable agent-based simulator → LFIAX."""

from __future__ import annotations

import random

from boed_agent import (
    BOEDAgent,
    ParameterInfo,
    SimpleSimulator,
    SimulatorMetadata,
)


def _agent_sim(theta, xi):
    random.seed(int(theta[0] * 1000) ^ int(xi[0] * 1000))
    return [sum(random.random() for _ in range(int(theta[0] * 10) + 1))]


def main() -> None:
    metadata = SimulatorMetadata(
        parameters=[
            ParameterInfo(name="social_temperature", units="unitless"),
        ],
        observation_labels=["attendance"],
        domain_tags=["abm", "non_differentiable"],
    )
    simulator = SimpleSimulator(
        fn=_agent_sim,
        metadata=metadata,
        is_explicit=False,
        is_differentiable=False,
        name="schelling_light",
    )
    agent = BOEDAgent(
        simulator=simulator,
        design_distribution={"announcement_lead_time": {"lower": 0.0, "upper": 7.0}},
        problem_description="Agent-based attendance simulator with discrete dynamics",
        prior=None,
        use_literature=False,
    )
    result = agent.run(dry_run=True)
    assert result.chosen_backend == "lfiax"
    print(result.backend_choice.to_dict())


if __name__ == "__main__":
    main()
