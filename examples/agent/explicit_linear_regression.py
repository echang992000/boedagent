"""Example: explicit simulator → Pyro VI.

A tiny synthetic linear-regression workflow exercising the full agent
orchestration.  The simulator carries ``is_explicit=True`` which makes
the :class:`SimulatorChoiceModule` dispatch to Pyro.

Run::

    python examples/agent/explicit_linear_regression.py
"""

from __future__ import annotations

import json
from pathlib import Path

from boed_agent import (
    BOEDAgent,
    ParameterInfo,
    SimpleSimulator,
    SimulatorMetadata,
)


def main() -> None:
    metadata = SimulatorMetadata(
        parameters=[
            ParameterInfo(name="slope", units="unitless", description="linear slope"),
            ParameterInfo(name="intercept", units="unitless"),
        ],
        observation_labels=["y"],
        domain_tags=["toy", "linear_regression"],
    )
    simulator = SimpleSimulator(
        fn=lambda theta, xi: theta[0] * xi + theta[1],
        metadata=metadata,
        is_explicit=True,
        is_differentiable=True,
        name="linear_regression",
    )

    agent = BOEDAgent(
        simulator=simulator,
        design_distribution={"xi": {"lower": -1.0, "upper": 1.0}},
        problem_description=(
            "Bayesian optimal experimental design for a 1-D linear regression"
        ),
        prior=None,
        use_literature=False,
    )
    result = agent.run(dry_run=True)
    print("Chosen backend:", result.chosen_backend)
    print(json.dumps(result.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
