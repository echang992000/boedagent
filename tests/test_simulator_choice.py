"""Unit tests for the SimulatorChoiceModule dispatcher."""

from __future__ import annotations

import pytest

from boed_agent.backends.registry import BackendRegistry
from boed_agent.literature.report import BackendPreference, LiteratureReport
from boed_agent.simulator_choice import SimulatorChoiceModule
from boed_agent.simulator_protocol import SimpleSimulator, SimulatorMetadata


def _make_sim(is_explicit: bool, is_differentiable: bool) -> SimpleSimulator:
    return SimpleSimulator(
        fn=lambda theta, xi: theta,
        metadata=SimulatorMetadata(),
        is_explicit=is_explicit,
        is_differentiable=is_differentiable,
    )


@pytest.mark.parametrize(
    "is_explicit,is_differentiable,expected",
    [
        (True, False, "pyro"),
        (True, True, "pyro"),
        (False, True, "minebed"),
        (False, False, "lfiax"),
    ],
)
def test_waterfall(is_explicit, is_differentiable, expected):
    registry = BackendRegistry.default()
    sim = _make_sim(is_explicit=is_explicit, is_differentiable=is_differentiable)
    choice = SimulatorChoiceModule.select(sim, registry=registry)
    assert choice.backend.name == expected
    assert choice.literature_override is False


def test_policy_network_selects_idad():
    registry = BackendRegistry.default()
    sim = _make_sim(is_explicit=False, is_differentiable=True)
    choice = SimulatorChoiceModule.select(
        sim, registry=registry, backend_options={"policy_network": True}
    )
    assert choice.backend.name == "idad"


def test_literature_override_fires_when_compatible():
    registry = BackendRegistry.default()
    sim = _make_sim(is_explicit=False, is_differentiable=True)
    report = LiteratureReport(
        backend_preference=BackendPreference(
            ranked=["iDAD", "MINEBED", "PyroVI", "LFIAX"],
            reasoning="lit_says_idad",
            cited_papers=["p1"],
        )
    )
    choice = SimulatorChoiceModule.select(sim, registry=registry, literature_report=report)
    assert choice.backend.name == "idad"
    assert choice.literature_override is True
    assert choice.cited_papers == ["p1"]


def test_literature_override_rejected_when_incompatible():
    """Literature suggests Pyro but simulator is not explicit — stay."""
    registry = BackendRegistry.default()
    sim = _make_sim(is_explicit=False, is_differentiable=True)
    report = LiteratureReport(
        backend_preference=BackendPreference(
            ranked=["PyroVI", "LFIAX"],
            reasoning="lit_says_pyro_but_incompatible",
        )
    )
    choice = SimulatorChoiceModule.select(sim, registry=registry, literature_report=report)
    assert choice.backend.name == "minebed"
    assert choice.literature_override is False
