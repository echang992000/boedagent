"""Codex CLI adapter for the literature LLM protocol."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boed_agent.literature.llm_client import LLMResponse


SYSTEM_PROMPT = (
    "You help extract grounded BOED literature evidence. "
    "Follow the requested output format exactly and do not add markdown fences."
)


@dataclass
class CodexCLILLMClient:
    """Adapter that routes literature prompts through ``codex exec``.

    This is intended for local developer workflows where Codex CLI is
    authenticated through a ChatGPT account rather than an API key.
    """

    model: str | None = None
    codex_bin: str = "codex"
    cwd: str | os.PathLike[str] | None = None
    sandbox: str = "read-only"

    def extract(
        self,
        prompt: str,
        *,
        model_tier: str = "cheap",
        stage: str = "unknown",
        budget: Any = None,
    ) -> LLMResponse:
        _ = model_tier, stage
        wrapped_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Return only the requested output. Do not add markdown fences or commentary.\n\n"
            f"{prompt}"
        )

        with tempfile.TemporaryDirectory(prefix="codex_lit_") as temp_dir:
            output_path = Path(temp_dir) / "last_message.txt"
            command = [
                self.codex_bin,
                "exec",
                "--sandbox",
                self.sandbox,
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
                "-o",
                str(output_path),
                "-",
            ]
            if self.cwd is not None:
                command[2:2] = ["-C", str(self.cwd)]
            if self.model:
                command[2:2] = ["-m", self.model]

            completed = subprocess.run(
                command,
                input=wrapped_prompt,
                text=True,
                capture_output=True,
                check=False,
                cwd=str(self.cwd) if self.cwd is not None else None,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                detail = stderr or stdout or "no stderr/stdout captured"
                raise RuntimeError(f"Codex CLI literature call failed: {detail}")
            if not output_path.exists():
                raise RuntimeError(
                    "Codex CLI literature call completed without writing the last message output file."
                )
            text = output_path.read_text(encoding="utf-8")

        if budget is not None:
            budget.record(stage, 0)
        return LLMResponse(
            text=text,
            input_tokens=0,
            output_tokens=0,
            model=self.model or "codex-cli",
        )


__all__ = ["CodexCLILLMClient"]
