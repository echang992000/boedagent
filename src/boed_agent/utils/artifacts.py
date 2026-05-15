"""Artifact writing helpers for normalized BOED runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from boed_agent.utils.io import write_json


@dataclass
class ArtifactWriter:
    root_dir: str
    run_name: str = "run"
    run_dir: Path | None = None
    files: dict[str, str] = field(default_factory=dict)

    def start(self) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(self.root_dir) / f"{self.run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def write(self, file_name: str, payload: Any) -> str:
        if self.run_dir is None:
            self.start()
        assert self.run_dir is not None
        path = self.run_dir / file_name
        write_json(path, payload)
        self.files[file_name] = str(path)
        return str(path)
