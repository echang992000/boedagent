from __future__ import annotations

import json
import subprocess
from pathlib import Path

from boed_agent.backends.registry import BackendRegistry
from boed_agent.clarification.planner import ClarificationPlanner
from boed_agent.literature.advisory import (
    build_literature_llm_client,
    build_source_bundle_for_spec,
    run_literature_dry_run,
    validate_literature_spec,
)
from boed_agent.literature.codex_cli_client import CodexCLILLMClient
from boed_agent.literature.llm_client import NullLLMClient
from boed_agent.literature.llm_client import RecordingLLMClient
from boed_agent.models import ExperimentSpec
from boed_agent.tools import registry as tool_registry_module
from boed_agent.tools.registry import build_default_tool_registry


def _write_local_corpus(root: Path) -> Path:
    root.mkdir()
    (root / "paper.md").write_text(
        "# Demo prior note\n\n"
        "We estimate alpha with a Beta(2,5) prior. "
        "Variational Bayesian experimental design performs well here.\n"
    )
    return root


def _base_spec(corpus_dir: Path | None = None, *, mode: str = "local") -> ExperimentSpec:
    payload = {
        "backend": "pyro",
        "model_ref": "demo.module:model",
        "problem_summary": "Toy BOED literature prior for alpha",
        "use_literature": True,
        "literature_source_mode": mode,
        "target_latent_labels": ["alpha"],
        "observation_labels": ["y"],
        "metadata": {"domain_tags": ["demo"]},
        "design_variables": [{"name": "dose", "lower": 0.0, "upper": 1.0}],
    }
    if corpus_dir is not None:
        payload["literature_corpus_dir"] = str(corpus_dir)
    return ExperimentSpec.from_dict(payload)


def _responder(prompt: str, tier: str) -> str:
    if "For each sentence below" in prompt:
        return json.dumps(
            [
                {
                    "id": 0,
                    "type": "prior_distribution",
                    "value": {
                        "parameter": "alpha",
                        "distribution": "Beta",
                        "params": {"a": 2, "b": 5},
                    },
                }
            ]
        )
    if "propose a prior" in prompt:
        return json.dumps(
            {
                "distribution": "Beta",
                "params": {"a": 2, "b": 5},
                "reasoning": "local corpus supports Beta",
                "cited_papers": ["title:15b3fd123f4f8fbf"],
            }
        )
    if "rank the candidate" in prompt:
        return json.dumps(
            {
                "ranked": ["PyroVI", "MINEBED"],
                "reasoning": "explicit-model literature points to PyroVI",
                "cited_papers": ["title:15b3fd123f4f8fbf"],
            }
        )
    return "{}"


def test_build_source_bundle_for_online_mode() -> None:
    bundle = build_source_bundle_for_spec(_base_spec(mode="online"))

    assert bundle.semantic_scholar is not None
    assert bundle.arxiv is not None
    assert bundle.openalex is not None
    assert bundle.pubmed is not None
    assert bundle.extra == []


def test_build_source_bundle_for_both_mode(tmp_path: Path) -> None:
    corpus_dir = _write_local_corpus(tmp_path / "corpus")
    bundle = build_source_bundle_for_spec(_base_spec(corpus_dir, mode="both"))

    assert bundle.semantic_scholar is not None
    assert bundle.extra
    assert bundle.extra[0][0] == "local_corpus"


def test_validate_literature_spec_requires_corpus_dir_for_local_mode() -> None:
    report = validate_literature_spec(_base_spec(mode="local"))

    assert report.valid is False
    assert "literature_corpus_dir" in report.missing_fields
    assert any(issue.path == "literature_corpus_dir" for issue in report.errors)


def test_run_literature_dry_run_returns_expected_payload_for_local_corpus(tmp_path: Path) -> None:
    corpus_dir = _write_local_corpus(tmp_path / "corpus")
    result, warnings = run_literature_dry_run(
        _base_spec(corpus_dir),
        llm=RecordingLLMClient(responder=_responder),
    )
    payload = result.to_dict()

    assert warnings == []
    assert payload["literature_report"] is not None
    assert payload["reasoning_trace"] is not None
    assert payload["backend_choice"]["backend"] == "pyro"
    assert "alpha" in payload["prior_used"]["distributions"]


def test_literature_tool_returns_advisory_only_payload(tmp_path: Path, monkeypatch) -> None:
    corpus_dir = _write_local_corpus(tmp_path / "corpus")

    monkeypatch.setattr(
        tool_registry_module,
        "run_literature_dry_run",
        lambda spec, **kwargs: run_literature_dry_run(
            spec,
            llm=RecordingLLMClient(responder=_responder),
        ),
    )

    backend_registry = BackendRegistry.default()
    planner = ClarificationPlanner(backend_registry)
    tools = build_default_tool_registry(backend_registry, planner)
    payload = tools.execute(
        "run_literature_dry_run",
        {"spec": _base_spec(corpus_dir).to_dict()},
    )

    assert payload["advisory_only"] is True
    assert payload["literature_report"] is not None
    assert payload["reasoning_trace"] is not None
    assert payload["backend_choice"]["backend"] == "pyro"
    assert "alpha" in payload["prior_used"]["distributions"]


def test_literature_tool_validates_missing_local_corpus_dir() -> None:
    backend_registry = BackendRegistry.default()
    planner = ClarificationPlanner(backend_registry)
    tools = build_default_tool_registry(backend_registry, planner)
    payload = tools.execute(
        "run_literature_dry_run",
        {"spec": _base_spec(mode="local").to_dict()},
    )

    assert payload["status"] == "needs_clarification"
    assert any(
        issue["path"] == "literature_corpus_dir"
        for issue in payload["validation"]["errors"]
    )


def test_build_literature_llm_client_supports_codex_without_model(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "boed_agent.literature.advisory.shutil.which",
        lambda name: "/opt/homebrew/bin/codex" if name == "codex" else None,
    )

    client, warnings = build_literature_llm_client("codex-cli", None)

    assert warnings == []
    assert isinstance(client, CodexCLILLMClient)
    assert client.model is None


def test_build_literature_llm_client_returns_null_when_codex_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "boed_agent.literature.advisory.shutil.which",
        lambda name: None,
    )

    client, warnings = build_literature_llm_client("codex-cli", None)

    assert isinstance(client, NullLLMClient)
    assert any("Codex CLI is not installed" in warning for warning in warnings)


def test_codex_cli_client_extract_uses_output_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        check: bool,
        cwd: str | None,
    ) -> subprocess.CompletedProcess[str]:
        _ = text, capture_output, check
        captured["command"] = command
        captured["input"] = input
        captured["cwd"] = cwd
        output_index = command.index("-o") + 1
        Path(command[output_index]).write_text('{"ok": true}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("boed_agent.literature.codex_cli_client.subprocess.run", fake_run)

    client = CodexCLILLMClient(cwd=tmp_path, model=None)
    response = client.extract("Return JSON only.")

    assert response.text == '{"ok": true}'
    assert response.model == "codex-cli"
    assert "-m" not in captured["command"]
    assert captured["cwd"] == str(tmp_path)
    assert "Return JSON only." in str(captured["input"])
