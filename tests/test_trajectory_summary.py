from __future__ import annotations

from pathlib import Path

from boed_agent.backends.registry import BackendRegistry
from boed_agent.clarification.planner import ClarificationPlanner
from boed_agent.models import DesignVariable, OptimizationStep
from boed_agent.tools.registry import build_default_tool_registry, summarize_result_payload
from boed_agent.utils.trajectory import summarize_optimized_design_histories


def test_summarize_result_includes_requested_design_histories() -> None:
    backend_registry = BackendRegistry.default()
    planner = ClarificationPlanner(backend_registry)
    tools = build_default_tool_registry(backend_registry, planner)
    result = {
        "backend": "pyro",
        "status": "completed",
        "eig": 1.23,
        "artifacts": {
            "trajectory_recreated": True,
            "optimized_design_histories": [
                [
                    {"step": 0, "design": [0.1], "eig": 0.5},
                    {"step": 1, "design": [0.3], "eig": 1.23},
                ]
            ],
        },
    }

    summary = summarize_result_payload(result)
    payload = tools.execute("summarize_result", {"result": result})

    assert "Trajectory recreation was requested" in summary
    assert len(payload["optimized_design_histories"]) == 1


def test_compressed_trajectory_summary_contains_dimension_ranges() -> None:
    summaries = summarize_optimized_design_histories(
        [
            [
                OptimizationStep(step=0, design=[0.1, 0.0], eig=0.3),
                OptimizationStep(step=1, design=[0.4, 0.2], eig=0.5),
                OptimizationStep(step=2, design=[0.6, 0.25], eig=0.7),
            ]
        ],
        [
            DesignVariable(name="temperature", lower=0.0, upper=1.0),
            DesignVariable(name="dose", lower=0.0, upper=1.0),
        ],
    )

    assert summaries[0]["num_steps"] == 3
    assert summaries[0]["design_dimensions"][0]["name"] == "temperature"
    assert summaries[0]["design_dimensions"][0]["delta"] == 0.5
    assert summaries[0]["best_eig"] == 0.7
