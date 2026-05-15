"""Backend registry and selection helpers."""

from __future__ import annotations

from typing import Iterable

from boed_agent.backends.base import BackendAdapter
from boed_agent.backends.idad_backend import IDADBackend
from boed_agent.backends.lfiax_backend import LFIAXBackend
from boed_agent.backends.minebed_backend import MINEBEDBackend
from boed_agent.backends.pyro_backend import PyroBackend
from boed_agent.models import BackendDescriptor, ExperimentSpec


class BackendRegistry:
    def __init__(self, adapters: Iterable[BackendAdapter] | None = None) -> None:
        self._adapters: dict[str, BackendAdapter] = {}
        for adapter in adapters or []:
            self.register(adapter)

    @classmethod
    def default(cls) -> "BackendRegistry":
        return cls(
            adapters=[
                PyroBackend(),
                LFIAXBackend(),
                MINEBEDBackend(),
                IDADBackend(),
            ]
        )

    def register(self, adapter: BackendAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def list_backends(self) -> list[BackendDescriptor]:
        return [adapter.describe() for adapter in self._adapters.values()]

    def get(self, name: str) -> BackendAdapter:
        if name not in self._adapters:
            available = ", ".join(sorted(self._adapters))
            raise KeyError(f"Unknown backend '{name}'. Available backends: {available}.")
        return self._adapters[name]

    def select_backend(self, spec: ExperimentSpec) -> BackendAdapter:
        if spec.backend:
            return self.get(spec.backend)
        for adapter in self._adapters.values():
            if adapter.supports(spec):
                return adapter
        raise KeyError("Unable to infer a backend. Specify `backend` explicitly.")
