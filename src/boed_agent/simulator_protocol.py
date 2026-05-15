"""Simulator protocol consumed by :class:`BOEDAgent`.

The protocol is a thin structural interface: any object that exposes the
attributes / methods listed here can be passed to the agent.  The agent
dispatches on two booleans:

* ``is_explicit`` — the simulator has a closed-form / explicit likelihood
  usable by Pyro VI.
* ``is_differentiable`` — the simulator is implicit but still
  differentiable (eligible for MINEBED / iDAD). Ignored when
  ``is_explicit`` is true.

``metadata`` is free-form structured information about parameters and
observations.  It is surfaced to the literature search module when
building queries and to the prior builder when reasoning about prior
ranges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass
class ParameterInfo:
    """Description of a single latent parameter."""

    name: str
    units: str | None = None
    domain_tag: str | None = None
    rough_magnitude: float | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "units": self.units,
            "domain_tag": self.domain_tag,
            "rough_magnitude": self.rough_magnitude,
            "description": self.description,
        }


@dataclass
class SimulatorMetadata:
    """Structured metadata about a simulator.

    The fields are intentionally loose — the literature search module
    only needs enough signal to build search queries; the prior builder
    only needs parameter names to align with extracted evidence.
    """

    parameters: list[ParameterInfo] = field(default_factory=list)
    observation_labels: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    notes: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SimulatorMetadata":
        data = data or {}
        params = [
            ParameterInfo(**p) if isinstance(p, dict) else p
            for p in data.get("parameters", [])
        ]
        return cls(
            parameters=params,
            observation_labels=list(data.get("observation_labels", [])),
            domain_tags=list(data.get("domain_tags", [])),
            notes=data.get("notes"),
            extras=dict(data.get("extras", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": [p.to_dict() for p in self.parameters],
            "observation_labels": list(self.observation_labels),
            "domain_tags": list(self.domain_tags),
            "notes": self.notes,
            "extras": dict(self.extras),
        }

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameters]


@runtime_checkable
class Simulator(Protocol):
    """Structural protocol the agent dispatches on.

    Concrete simulators do not need to inherit from this class — any
    object with the expected attributes is accepted at runtime.
    """

    metadata: SimulatorMetadata
    is_explicit: bool
    is_differentiable: bool

    def __call__(self, theta: Any, xi: Any) -> Any:  # pragma: no cover - protocol
        """Run the simulator at latent ``theta`` and design ``xi``."""
        ...


@dataclass
class SimpleSimulator:
    """Minimal concrete implementation of :class:`Simulator`.

    Used in tests / examples where the user has a plain callable and
    just wants to attach metadata + dispatch flags.
    """

    fn: Callable[[Any, Any], Any]
    metadata: SimulatorMetadata = field(default_factory=SimulatorMetadata)
    is_explicit: bool = False
    is_differentiable: bool = False
    name: str | None = None

    def __call__(self, theta: Any, xi: Any) -> Any:
        return self.fn(theta, xi)


def introspect_metadata(
    obj: Any,
    *,
    override: SimulatorMetadata | Mapping[str, Any] | None = None,
    parameter_names: list[str] | None = None,
) -> SimulatorMetadata:
    """Best-effort extraction of :class:`SimulatorMetadata` from a user object.

    Precedence:

    1. ``override`` wins outright (either a :class:`SimulatorMetadata`
       or a dict that :meth:`SimulatorMetadata.from_dict` can parse).
    2. If ``obj.metadata`` already exists as a :class:`SimulatorMetadata`,
       return it unchanged.
    3. If ``obj.metadata`` is a ``dict``, coerce through ``from_dict``.
    4. Otherwise, introspect the simulator's call signature — build
       :class:`ParameterInfo` entries for each parameter of the first
       argument (``theta``), skipping ``self`` and ``xi`` by convention.
    5. As a last resort, fall back to ``parameter_names`` supplied by
       the caller, or an empty :class:`SimulatorMetadata` — *always*
       returning something callers can treat as non-None.

    This never raises for missing attributes: the literature pipeline
    runs in a best-effort mode and a simulator with no metadata simply
    gets fewer literature hints.
    """

    if override is not None:
        if isinstance(override, SimulatorMetadata):
            return override
        return SimulatorMetadata.from_dict(override)

    existing = getattr(obj, "metadata", None)
    if isinstance(existing, SimulatorMetadata):
        return existing
    if isinstance(existing, Mapping):
        return SimulatorMetadata.from_dict(existing)

    # Signature-based introspection.  We look at the call signature of
    # whichever is most informative: the simulator callable itself, or
    # the object's ``__call__``.
    names: list[str] = list(parameter_names or [])
    if not names:
        names = _guess_parameter_names(obj)

    return SimulatorMetadata(
        parameters=[ParameterInfo(name=n) for n in names],
    )


def _guess_parameter_names(obj: Any) -> list[str]:
    """Inspect ``obj`` for a ``theta`` argument and pull parameter names off it.

    Handles three common shapes:

    * plain callable ``fn(theta, xi)`` with no deeper structure — returns ``["theta"]``
    * a simulator exposing ``parameter_names`` or ``param_names`` directly
    * an object whose call signature includes named keyword parameters
      other than ``theta`` / ``xi`` / ``self`` / ``cls`` / ``design``
    """

    import inspect

    for attr in ("parameter_names", "param_names", "theta_names"):
        value = getattr(obj, attr, None)
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            return [str(v) for v in value]

    target = obj
    if not callable(obj) and hasattr(obj, "__call__"):
        target = obj.__call__

    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return []

    skip = {"self", "cls", "xi", "design", "designs"}
    names: list[str] = []
    for param in sig.parameters.values():
        if param.name in skip:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        names.append(param.name)
    return names


__all__ = [
    "ParameterInfo",
    "SimulatorMetadata",
    "Simulator",
    "SimpleSimulator",
    "introspect_metadata",
]
