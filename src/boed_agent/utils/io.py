"""Spec and JSON IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from boed_agent.models import ExperimentSpec, to_jsonable


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(file_path.read_text())
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML support requires PyYAML. Install it separately or use JSON specs."
            ) from exc
        data = yaml.safe_load(file_path.read_text())
    else:
        raise ValueError(f"Unsupported spec extension '{suffix}'. Use .json, .yaml, or .yml.")
    if not isinstance(data, dict):
        raise ValueError("Experiment spec must deserialize to a JSON object.")
    return ExperimentSpec.from_dict(data)


def write_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
