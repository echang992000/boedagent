"""Module 2 — :class:`PriorBuilder`.

Given the user's (possibly ``None``) prior and a
:class:`LiteratureReport`, build a prior object that:

* keeps the user's own prior verbatim when supplied — literature
  suggestions surface as *warnings*, never silent overrides;
* fills in a literature-derived prior (or a fallback) per parameter
  when the user did not supply anything;
* attaches ``.source``, ``.reasoning`` and ``.cited_papers`` metadata
  to every distribution for auditability.

The builder is intentionally a pure-Python object graph — no
dependency on torch / jax / pyro.  Backends are free to translate
:class:`AugmentedPrior` into their preferred representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from boed_agent.literature.report import LiteratureReport, PriorSuggestion


@dataclass
class DistributionSpec:
    """Light-weight distribution record with provenance."""

    name: str | None
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "user"
    reasoning: str = ""
    cited_papers: list[str] = field(default_factory=list)
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "source": self.source,
            "reasoning": self.reasoning,
            "cited_papers": list(self.cited_papers),
            "fallback": self.fallback,
        }


@dataclass
class AugmentedPrior:
    """User-visible result of :meth:`PriorBuilder.build`."""

    distributions: dict[str, DistributionSpec] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw_user_prior: Any = None
    literature_prior: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "distributions": {
                k: v.to_dict() for k, v in self.distributions.items()
            },
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


class PriorBuilder:
    """Augments a user-supplied prior with literature recommendations."""

    @staticmethod
    def build(
        user_prior: Any,
        literature_report: LiteratureReport | None = None,
        *,
        parameter_names: Sequence[str] | None = None,
    ) -> AugmentedPrior:
        augmented = AugmentedPrior(
            raw_user_prior=user_prior,
            literature_prior=_lit_prior(literature_report),
        )
        if user_prior is None:
            if literature_report is not None:
                _fill_from_literature(augmented, literature_report, parameter_names)
            else:
                augmented.notes.append(
                    "No user prior and no literature report; caller should "
                    "specify parameters explicitly."
                )
            return augmented

        # User prior supplied — keep it verbatim, surface lit as warnings.
        _keep_user_prior(augmented, user_prior)
        if literature_report is not None:
            _append_literature_warnings(augmented, literature_report)
        return augmented


def _fill_from_literature(
    augmented: AugmentedPrior,
    literature_report: LiteratureReport,
    parameter_names: Sequence[str] | None,
) -> None:
    names: Iterable[str]
    if parameter_names:
        names = parameter_names
        if not literature_report.prior_suggestions:
            reason = literature_report.diagnostics.get("zero_prior_reasoning_reason")
            if reason:
                augmented.notes.append(
                    "Literature report produced zero prior suggestions: " + str(reason)
                )
            else:
                augmented.notes.append(
                    "Literature report produced zero prior suggestions; no "
                    "distributions were created."
                )
    else:
        names = list(literature_report.prior_suggestions)
    for name in names:
        suggestion = literature_report.prior_suggestions.get(name)
        if suggestion is None:
            augmented.notes.append(
                f"No literature evidence for parameter {name!r}; caller should supply."
            )
            continue
        augmented.distributions[name] = _from_suggestion(suggestion)


def _keep_user_prior(augmented: AugmentedPrior, user_prior: Any) -> None:
    # Accept three shapes:
    #   - mapping {param: distribution_spec_like}
    #   - object with ``.distributions`` attribute (existing AugmentedPrior)
    #   - opaque callable / distribution (kept as-is, no override)
    if isinstance(user_prior, Mapping):
        for name, spec in user_prior.items():
            augmented.distributions[str(name)] = _coerce_user_spec(name, spec)
        return
    existing = getattr(user_prior, "distributions", None)
    if isinstance(existing, Mapping):
        for name, spec in existing.items():
            augmented.distributions[str(name)] = _coerce_user_spec(name, spec)
        return
    augmented.notes.append(
        "User prior kept verbatim as an opaque object; no per-parameter overrides."
    )


def _coerce_user_spec(name: Any, value: Any) -> DistributionSpec:
    if isinstance(value, DistributionSpec):
        value.source = value.source or "user"
        return value
    if isinstance(value, Mapping):
        return DistributionSpec(
            name=value.get("name") or value.get("distribution"),
            params=dict(value.get("params") or {}),
            source="user",
            reasoning=str(value.get("reasoning", "user-supplied")),
            cited_papers=list(value.get("cited_papers") or []),
        )
    return DistributionSpec(
        name=None,
        params={"value": value},
        source="user",
        reasoning="user-supplied opaque distribution",
    )


def _from_suggestion(suggestion: PriorSuggestion) -> DistributionSpec:
    return DistributionSpec(
        name=suggestion.distribution,
        params=dict(suggestion.params),
        source="literature",
        reasoning=suggestion.reasoning,
        cited_papers=list(suggestion.cited_papers),
        fallback=bool(suggestion.fallback),
    )


def _append_literature_warnings(
    augmented: AugmentedPrior, literature_report: LiteratureReport
) -> None:
    for name, suggestion in literature_report.prior_suggestions.items():
        if name in augmented.distributions:
            user_spec = augmented.distributions[name]
            if user_spec.name and suggestion.distribution and user_spec.name.lower() != (suggestion.distribution or "").lower():
                augmented.warnings.append(
                    f"Literature suggests {suggestion.distribution} for {name!r} "
                    f"but user supplied {user_spec.name}. Keeping user prior."
                )
            elif suggestion.fallback:
                augmented.warnings.append(
                    f"Literature evidence for {name!r} was insufficient "
                    "(fallback recommendation). User prior kept."
                )
            else:
                augmented.warnings.append(
                    f"Literature concurs on distribution family for {name!r}; "
                    "user prior kept verbatim."
                )
        else:
            augmented.warnings.append(
                f"Literature has a prior suggestion for {name!r} but user did "
                "not parametrise it — consider adding."
            )


def _lit_prior(report: LiteratureReport | None) -> Any:
    if report is None:
        return None
    return {k: v.to_dict() for k, v in report.prior_suggestions.items()}


__all__ = ["AugmentedPrior", "DistributionSpec", "PriorBuilder"]
