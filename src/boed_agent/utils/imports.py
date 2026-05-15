"""Import helpers and a small runtime registry for callables."""

from __future__ import annotations

import importlib
from typing import Any


_REGISTRY: dict[str, Any] = {}


def register_callable(name: str, obj: Any) -> Any:
    _REGISTRY[name] = obj
    return obj


def registered_callables() -> dict[str, Any]:
    return dict(_REGISTRY)


def resolve_reference(reference: str | None) -> Any:
    if reference is None:
        return None
    if reference in _REGISTRY:
        return _REGISTRY[reference]
    module_name, attr_name = _split_reference(reference)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def instantiate_reference(reference: str | None, spec: Any | None = None) -> Any:
    resolved = resolve_reference(reference)
    if resolved is None:
        return None
    if not callable(resolved):
        return resolved
    for args in ((spec,), ()):
        try:
            return resolved(*args)
        except TypeError:
            continue
    return resolved


def _split_reference(reference: str) -> tuple[str, str]:
    if ":" in reference:
        module_name, attr_name = reference.split(":", 1)
        return module_name, attr_name
    module_name, _, attr_name = reference.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(
            f"Unable to resolve reference '{reference}'. Use 'module:attribute' or 'module.attribute'."
        )
    return module_name, attr_name
