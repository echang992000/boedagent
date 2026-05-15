"""Helpers for literature-only advisory runs from :class:`ExperimentSpec`.

These utilities bridge the spec/CLI/chat world into the BOEDAgent
literature orchestrator without changing backend execution semantics.
The resulting advisory payload is a dry-run: it returns prior
recommendations, backend hints, and reasoning trace, but does not
inject literature-derived priors back into backend execution.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boed_agent.agent import BOEDAgent, DryRunResult
from boed_agent.literature.clients import (
    ArxivClient,
    LocalCorpusClient,
    OpenAlexClient,
    PubMedClient,
    SemanticScholarClient,
)
from boed_agent.literature.clients.base import ClientConfig
from boed_agent.literature.codex_cli_client import CodexCLILLMClient
from boed_agent.literature.llm_client import LLMClient, LLMResponse, NullLLMClient
from boed_agent.literature.search import LiteratureSearchConfig, SourceBundle
from boed_agent.models import ExperimentSpec, Message, ValidationIssue, ValidationReport
from boed_agent.providers import ClaudeProvider, OpenAIProvider
from boed_agent.providers.base import LLMProvider
from boed_agent.simulator_protocol import ParameterInfo, SimpleSimulator, SimulatorMetadata


SYSTEM_PROMPT = (
    "You help extract grounded BOED literature evidence. "
    "Follow the requested output format exactly and do not add markdown fences."
)

ALLOWED_SOURCE_MODES = {"online", "local", "both"}


@dataclass
class ProviderLLMClient:
    """Adapter from the repo's provider classes to the literature LLM protocol."""

    cheap_provider: LLMProvider
    reasoning_provider: LLMProvider | None = None

    def extract(
        self,
        prompt: str,
        *,
        model_tier: str = "cheap",
        stage: str = "unknown",
        budget: Any = None,
    ) -> LLMResponse:
        provider = (
            self.reasoning_provider
            if model_tier == "reasoning" and self.reasoning_provider is not None
            else self.cheap_provider
        )
        request = provider.build_request(
            [Message(role="user", content=prompt)],
            [],
            SYSTEM_PROMPT,
            state=None,
        )
        response = provider.generate(request)
        parsed = provider.parse_response(response)
        input_tokens, output_tokens = _usage_counts(response, prompt, parsed.text)
        total_tokens = input_tokens + output_tokens
        if budget is not None:
            budget.record(stage, total_tokens)
        model_name = getattr(provider, "model", None)
        return LLMResponse(
            text=parsed.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_name,
        )


def prepare_literature_spec(spec: ExperimentSpec) -> ExperimentSpec:
    prepared = ExperimentSpec.from_dict(spec.to_dict())
    prepared.use_literature = True
    return prepared


def validate_literature_spec(spec: ExperimentSpec) -> ValidationReport:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    missing_fields: list[str] = []

    mode = (spec.literature_source_mode or "").strip().lower()
    if not mode:
        missing_fields.append("literature_source_mode")
        errors.append(
            ValidationIssue(
                path="literature_source_mode",
                message="Missing required field `literature_source_mode`.",
            )
        )
    elif mode not in ALLOWED_SOURCE_MODES:
        errors.append(
            ValidationIssue(
                path="literature_source_mode",
                message=(
                    "`literature_source_mode` must be one of "
                    f"{sorted(ALLOWED_SOURCE_MODES)}."
                ),
            )
        )

    if mode in {"local", "both"}:
        corpus_dir = (spec.literature_corpus_dir or "").strip()
        if not corpus_dir:
            missing_fields.append("literature_corpus_dir")
            errors.append(
                ValidationIssue(
                    path="literature_corpus_dir",
                    message="Missing required field `literature_corpus_dir`.",
                )
            )
        else:
            corpus_path = Path(corpus_dir).expanduser()
            if not corpus_path.exists():
                errors.append(
                    ValidationIssue(
                        path="literature_corpus_dir",
                        message=f"Local corpus directory does not exist: {corpus_path}",
                    )
                )
            elif not corpus_path.is_dir():
                errors.append(
                    ValidationIssue(
                        path="literature_corpus_dir",
                        message=f"Local corpus path is not a directory: {corpus_path}",
                    )
                )

    return ValidationReport(
        valid=len(errors) == 0,
        backend=spec.backend,
        errors=errors,
        warnings=warnings,
        missing_fields=missing_fields,
    )


def build_literature_llm_client(
    provider_name: str | None = None,
    model: str | None = None,
) -> tuple[LLMClient, list[str]]:
    warnings: list[str] = []
    provider_name = (provider_name or "").strip().lower() or None
    if provider_name is None:
        warnings.append(
            "No literature LLM provider/model configured; using NullLLMClient. "
            "Stage B/D outputs may be sparse or fallback-only."
        )
        return NullLLMClient(), warnings

    if provider_name in {"codex", "codex-cli"}:
        if shutil.which("codex") is None:
            warnings.append(
                "Codex CLI is not installed or not on PATH; using NullLLMClient for literature advisory."
            )
            return NullLLMClient(), warnings
        return CodexCLILLMClient(model=model, cwd=os.getcwd()), warnings

    if provider_name == "openai":
        if model is None:
            warnings.append(
                "No literature LLM provider/model configured; using NullLLMClient. "
                "Stage B/D outputs may be sparse or fallback-only."
            )
            return NullLLMClient(), warnings
        if not os.environ.get("OPENAI_API_KEY"):
            warnings.append(
                "OPENAI_API_KEY is not set; using NullLLMClient for literature advisory."
            )
            return NullLLMClient(), warnings
        return ProviderLLMClient(
            cheap_provider=OpenAIProvider(
                model=model,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        ), warnings
    if provider_name == "claude":
        if model is None:
            warnings.append(
                "No literature LLM provider/model configured; using NullLLMClient. "
                "Stage B/D outputs may be sparse or fallback-only."
            )
            return NullLLMClient(), warnings
        if not os.environ.get("ANTHROPIC_API_KEY"):
            warnings.append(
                "ANTHROPIC_API_KEY is not set; using NullLLMClient for literature advisory."
            )
            return NullLLMClient(), warnings
        return ProviderLLMClient(
            cheap_provider=ClaudeProvider(
                model=model,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        ), warnings

    warnings.append(
        f"Unsupported literature provider '{provider_name}'; using NullLLMClient."
    )
    return NullLLMClient(), warnings


def build_source_bundle_for_spec(
    spec: ExperimentSpec,
    *,
    client_config: ClientConfig | None = None,
) -> SourceBundle:
    cfg = client_config or ClientConfig(timeout_seconds=20.0)
    mode = (spec.literature_source_mode or "").strip().lower()

    semantic_scholar = None
    arxiv = None
    openalex = None
    pubmed = None
    extra: list[tuple[str, Any]] = []

    if mode in {"online", "both"}:
        semantic_scholar = SemanticScholarClient(
            config=cfg,
            api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        )
        arxiv = ArxivClient(config=cfg)
        openalex = OpenAlexClient(config=cfg)
        pubmed = PubMedClient(
            config=cfg,
            api_key=os.environ.get("PUBMED_API_KEY"),
            email=os.environ.get("PUBMED_EMAIL") or os.environ.get("NCBI_EMAIL"),
        )

    if mode in {"local", "both"}:
        local = LocalCorpusClient(corpus_dir=str(spec.literature_corpus_dir))
        extra.append(("local_corpus", local))

    return SourceBundle(
        semantic_scholar=semantic_scholar,
        arxiv=arxiv,
        openalex=openalex,
        pubmed=pubmed,
        extra=extra,
    )


def build_spec_bridge_agent(
    spec: ExperimentSpec,
    *,
    llm: LLMClient | None = None,
    client_config: ClientConfig | None = None,
    literature_config: LiteratureSearchConfig | None = None,
) -> tuple[BOEDAgent, list[str]]:
    spec = prepare_literature_spec(spec)
    warnings: list[str] = []

    metadata = _metadata_from_spec(spec)
    simulator = SimpleSimulator(
        fn=lambda theta, xi: theta[0] if theta else 0,
        metadata=metadata,
        is_explicit=_is_explicit(spec),
        is_differentiable=_is_differentiable(spec),
        name=spec.simulator_ref or spec.model_ref or (spec.backend or "spec_bridge"),
    )
    design_distribution = _design_distribution_from_spec(spec)
    sources = build_source_bundle_for_spec(spec, client_config=client_config)

    agent = BOEDAgent(
        simulator=simulator,
        design_distribution=design_distribution,
        problem_description=spec.problem_summary or "",
        prior=None,
        use_literature=True,
        literature_sources=sources,
        literature_llm=llm,
        literature_config=literature_config,
        backend_options=dict(spec.backend_options),
    )
    if isinstance(llm, NullLLMClient):
        warnings.append(
            "Literature advisory is running without a live LLM; prior synthesis may be limited."
        )
    return agent, warnings


def run_literature_dry_run(
    spec: ExperimentSpec,
    *,
    llm: LLMClient | None = None,
    provider_name: str | None = None,
    model: str | None = None,
    client_config: ClientConfig | None = None,
    literature_config: LiteratureSearchConfig | None = None,
) -> tuple[DryRunResult, list[str]]:
    warnings: list[str] = []
    llm_client = llm
    if llm_client is None:
        llm_client, llm_warnings = build_literature_llm_client(provider_name, model)
        warnings.extend(llm_warnings)
    if isinstance(llm_client, LLMProvider):
        llm_client = ProviderLLMClient(cheap_provider=llm_client)

    agent, agent_warnings = build_spec_bridge_agent(
        spec,
        llm=llm_client,
        client_config=client_config,
        literature_config=literature_config,
    )
    warnings.extend(agent_warnings)
    return agent.run(dry_run=True), warnings


def _metadata_from_spec(spec: ExperimentSpec) -> SimulatorMetadata:
    raw_tags = (spec.metadata or {}).get("domain_tags") or []
    domain_tags = [str(tag) for tag in raw_tags if str(tag).strip()]
    raw_params = list(spec.target_latent_labels)
    if not raw_params:
        raw_params = [
            str(name)
            for name in ((spec.metadata or {}).get("parameter_names") or [])
            if str(name).strip()
        ]
    parameters = [ParameterInfo(name=name) for name in raw_params]
    return SimulatorMetadata(
        parameters=parameters,
        observation_labels=list(spec.observation_labels),
        domain_tags=domain_tags,
    )


def _design_distribution_from_spec(spec: ExperimentSpec) -> dict[str, Any] | None:
    if not spec.design_variables:
        return None
    return {
        variable.name: {
            "lower": variable.lower,
            "upper": variable.upper,
            "initial": variable.initial,
            "dtype": variable.dtype,
            "description": variable.description,
            "shape": variable.shape,
        }
        for variable in spec.design_variables
    }


def _is_explicit(spec: ExperimentSpec) -> bool:
    if spec.backend == "pyro" or spec.model_ref is not None:
        return True
    if spec.backend == "lfiax" or spec.simulator_ref is not None:
        return False
    return False


def _is_differentiable(spec: ExperimentSpec) -> bool:
    if spec.differentiable is not None:
        return bool(spec.differentiable)
    return _is_explicit(spec)


def _usage_counts(response: Any, prompt: str, text: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None and isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None and isinstance(usage, dict):
            output_tokens = usage.get("output_tokens")
        if input_tokens is not None or output_tokens is not None:
            return int(input_tokens or 0), int(output_tokens or 0)
    return max(1, len(prompt) // 4), max(1, len(text or "") // 4)
