"""Module 1 — :class:`SimulatorChoiceModule`.

Waterfall dispatcher from :class:`Simulator` → :class:`BackendAdapter`.

Dispatch logic:

* ``simulator.is_explicit`` → Pyro (VI)
* ``simulator.is_differentiable`` → iDAD preferred if
  ``backend_options.policy_network`` is set, otherwise MINEBED.
  The literature report can override this via ``backend_preference``.
* otherwise → LFIAX (black-box)

Every literature override is returned alongside the chosen backend so
that the calling code can attach the :class:`ReasoningStep` to the
run log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from boed_agent.backends.base import BackendAdapter
from boed_agent.backends.registry import BackendRegistry
from boed_agent.literature.report import LiteratureReport
from boed_agent.simulator_protocol import Simulator


_LIT_NAME_TO_BACKEND = {
    "pyrovi": "pyro",
    "pyro": "pyro",
    "minebed": "minebed",
    "idad": "idad",
    "lfiax": "lfiax",
}


@dataclass
class BackendChoice:
    backend: BackendAdapter
    reason: str
    literature_override: bool = False
    cited_papers: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.name,
            "reason": self.reason,
            "literature_override": self.literature_override,
            "cited_papers": list(self.cited_papers),
            "alternatives": list(self.alternatives),
            "notes": list(self.notes),
        }


class SimulatorChoiceModule:
    """Pick the right backend given a simulator and (optional) literature."""

    @staticmethod
    def select(
        simulator: Simulator,
        registry: BackendRegistry | None = None,
        literature_report: Optional[LiteratureReport] = None,
        *,
        backend_options: dict[str, Any] | None = None,
    ) -> BackendChoice:
        registry = registry or BackendRegistry.default()

        literature_candidate = _literature_candidate(literature_report)
        base_choice = _waterfall(simulator, backend_options or {})
        notes: list[str] = []

        if literature_candidate and literature_candidate != base_choice:
            compatible = _compatible(literature_candidate, simulator)
            if compatible:
                try:
                    adapter = registry.get(literature_candidate)
                except KeyError:
                    notes.append(
                        f"Literature preferred {literature_candidate} but no adapter "
                        "is registered; staying with the waterfall pick."
                    )
                    adapter = registry.get(base_choice)
                    return BackendChoice(
                        backend=adapter,
                        reason=_waterfall_reason(simulator, base_choice),
                        literature_override=False,
                        cited_papers=_lit_cites(literature_report),
                        alternatives=_alternatives(base_choice),
                        notes=notes,
                    )
                return BackendChoice(
                    backend=adapter,
                    reason=(
                        f"Literature prefers {literature_candidate} "
                        f"(over waterfall pick {base_choice})."
                    ),
                    literature_override=True,
                    cited_papers=_lit_cites(literature_report),
                    alternatives=_alternatives(literature_candidate),
                    notes=notes,
                )
            notes.append(
                f"Literature suggested {literature_candidate} but it is "
                f"incompatible with this simulator; keeping {base_choice}."
            )
        adapter = registry.get(base_choice)
        return BackendChoice(
            backend=adapter,
            reason=_waterfall_reason(simulator, base_choice),
            literature_override=False,
            cited_papers=_lit_cites(literature_report),
            alternatives=_alternatives(base_choice),
            notes=notes,
        )


def _waterfall(simulator: Simulator, backend_options: dict[str, Any]) -> str:
    if getattr(simulator, "is_explicit", False):
        return "pyro"
    if getattr(simulator, "is_differentiable", False):
        if backend_options.get("policy_network"):
            return "idad"
        return "minebed"
    return "lfiax"


def _waterfall_reason(simulator: Simulator, backend: str) -> str:
    if backend == "pyro":
        return "Simulator has an explicit likelihood; Pyro VI is the natural fit."
    if backend == "minebed":
        return "Simulator is implicit but differentiable; MINEBED is the default."
    if backend == "idad":
        return "Simulator is differentiable and a policy network is configured; using iDAD."
    return "Simulator is black-box; falling back to LFIAX."


def _literature_candidate(report: LiteratureReport | None) -> Optional[str]:
    if report is None:
        return None
    ranked = report.backend_preference.ranked
    for candidate in ranked:
        normalised = _LIT_NAME_TO_BACKEND.get(candidate.lower())
        if normalised:
            return normalised
    return None


def _compatible(backend: str, simulator: Simulator) -> bool:
    if backend == "pyro":
        return bool(getattr(simulator, "is_explicit", False))
    if backend in ("minebed", "idad"):
        return bool(getattr(simulator, "is_differentiable", False))
    return True  # LFIAX is always compatible


def _alternatives(primary: str) -> list[str]:
    order = ["pyro", "minebed", "idad", "lfiax"]
    return [name for name in order if name != primary]


def _lit_cites(report: LiteratureReport | None) -> list[str]:
    if report is None:
        return []
    return list(report.backend_preference.cited_papers)


__all__ = ["BackendChoice", "SimulatorChoiceModule"]
