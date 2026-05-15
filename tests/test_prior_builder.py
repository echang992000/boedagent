"""Tests for the PriorBuilder."""

from __future__ import annotations

from boed_agent.literature.report import LiteratureReport, PriorSuggestion
from boed_agent.prior_builder import PriorBuilder


def _report() -> LiteratureReport:
    report = LiteratureReport()
    report.prior_suggestions["alpha"] = PriorSuggestion(
        parameter="alpha",
        distribution="Beta",
        params={"a": 2, "b": 5},
        reasoning="lit supports Beta",
        cited_papers=["p1"],
    )
    report.prior_suggestions["beta"] = PriorSuggestion(
        parameter="beta",
        distribution="Normal",
        params={"loc": 0.0, "scale": 1.0},
        reasoning="lit supports Normal",
        cited_papers=["p2"],
        fallback=True,
    )
    return report


def test_user_prior_is_never_overridden():
    user_prior = {
        "alpha": {"distribution": "Uniform", "params": {"low": 0, "high": 1}},
    }
    augmented = PriorBuilder.build(user_prior, _report())
    assert augmented.distributions["alpha"].name == "Uniform"
    assert augmented.distributions["alpha"].source == "user"
    # Warning is attached because literature disagreed.
    assert any("Literature suggests" in w for w in augmented.warnings)


def test_none_prior_is_filled_from_literature():
    augmented = PriorBuilder.build(None, _report())
    assert augmented.distributions["alpha"].name == "Beta"
    assert augmented.distributions["alpha"].source == "literature"
    assert augmented.distributions["beta"].fallback is True


def test_empty_literature_report_adds_zero_prior_note():
    report = LiteratureReport()
    report.diagnostics["zero_prior_reasoning_reason"] = (
        "Stage B extracted no prior_range or prior_distribution records."
    )
    augmented = PriorBuilder.build(None, report, parameter_names=["alpha"])

    assert augmented.distributions == {}
    assert any("zero prior suggestions" in note for note in augmented.notes)
    assert any("No literature evidence for parameter 'alpha'" in note for note in augmented.notes)


def test_opaque_user_prior_kept_verbatim():
    sentinel = object()
    augmented = PriorBuilder.build(sentinel, None)
    assert augmented.raw_user_prior is sentinel
    assert augmented.distributions == {}
