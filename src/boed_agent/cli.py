"""CLI entrypoint for the BOED agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any

from boed_agent.backends.registry import BackendRegistry
from boed_agent.clarification.interactive import collect_interactive_spec
from boed_agent.clarification.planner import ClarificationPlanner
from boed_agent.core.engine import ManualAgentEngine, OpenAIAgentsSdkEngine
from boed_agent.models import ExperimentSpec, Message, SessionConfig, to_jsonable
from boed_agent.providers import ClaudeProvider, OpenAIProvider
from boed_agent.tools.registry import build_default_tool_registry, summarize_result_payload
from boed_agent.utils.artifacts import ArtifactWriter
from boed_agent.utils.io import load_experiment_spec
from boed_agent.utils.trajectory import (
    compact_optimized_design_history_summaries,
    save_design_trajectory_plot,
    save_optimization_history_npz,
    summarize_optimized_design_histories,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    backend_registry = BackendRegistry.default()
    planner = ClarificationPlanner(backend_registry)
    tools = build_default_tool_registry(
        backend_registry,
        planner,
        literature_provider_name=getattr(args, "provider", None),
        literature_model=getattr(args, "model", None),
    )

    if args.command == "list-backends":
        payload = {"backends": [descriptor.to_dict() for descriptor in backend_registry.list_backends()]}
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "validate":
        spec = load_experiment_spec(args.spec)
        result = tools.execute("validate_experiment_spec", {"spec": spec.to_dict()})
        print(json.dumps(to_jsonable(result), indent=2))
        validation = result["validation"]
        return 0 if validation["valid"] else 1

    if args.command == "run":
        spec = load_experiment_spec(args.spec)
        return run_spec(spec, backend_registry, planner, interactive=args.interactive)

    if args.command == "literature-dry-run":
        spec = load_experiment_spec(args.spec)
        result = tools.execute("run_literature_dry_run", {"spec": spec.to_dict()})
        print(json.dumps(to_jsonable(result), indent=2))
        return 0 if result.get("status") != "needs_clarification" else 1

    if args.command == "chat":
        return chat(args, tools)

    parser.error(f"Unknown command {args.command!r}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boed-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-backends")

    validate = subparsers.add_parser("validate")
    validate.add_argument("spec")

    run = subparsers.add_parser("run")
    run.add_argument("spec")
    run.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for missing BOED fields in the terminal before executing the backend.",
    )

    literature = subparsers.add_parser("literature-dry-run")
    literature.add_argument("spec")
    literature.add_argument("--provider", choices=["openai", "claude"])
    literature.add_argument("--model")

    chat = subparsers.add_parser("chat")
    chat.add_argument("--provider", choices=["openai", "claude"], required=True)
    chat.add_argument("--model", required=True)
    chat.add_argument("--runtime-mode", choices=["manual", "agents-sdk"])
    chat.add_argument("--spec", help="Optional experiment spec to preload into the conversation.")
    chat.add_argument("--session-id")
    chat.add_argument("--session-db", default="artifacts/agent_sessions.sqlite")
    chat.add_argument("--disable-tracing", action="store_true")

    return parser


def run_spec(
    spec: ExperimentSpec,
    backend_registry: BackendRegistry,
    planner: ClarificationPlanner,
    interactive: bool = False,
) -> int:
    caveat = _execution_literature_caveat(spec)
    interactive_transcript: list[dict[str, Any]] = []
    if interactive:
        spec, interactive_transcript = collect_interactive_spec(spec, planner)
        caveat = _execution_literature_caveat(spec)

    try:
        backend = backend_registry.select_backend(spec)
    except KeyError:
        payload = {
            "status": "needs_clarification",
            "clarification_questions": [
                question.to_dict() for question in planner.plan(spec)
            ],
        }
        if caveat:
            payload["warnings"] = [caveat]
        print(json.dumps(payload, indent=2))
        return 1

    validation = backend.validate(spec)
    writer = ArtifactWriter(spec.artifacts.output_dir, run_name=backend.name)
    writer.start()
    if spec.artifacts.save_normalized_spec:
        writer.write("normalized_spec.json", spec.to_dict())

    questions = [question.to_dict() for question in planner.plan(spec)]
    if spec.artifacts.save_transcript:
        writer.write(
            "clarification_log.json",
            {
                "questions": questions,
                "interactive_answers": interactive_transcript,
            },
        )

    if not validation.valid:
        payload = {
            "status": "needs_clarification",
            "validation": validation.to_dict(),
            "clarification_questions": questions,
            "artifacts": writer.files,
        }
        if caveat:
            payload["warnings"] = [caveat]
        print(json.dumps(payload, indent=2))
        return 1

    result = backend.optimize(spec)
    payload = result.to_dict()
    payload.pop("history", None)
    payload.setdefault("artifacts", {})
    payload["artifacts"].pop("sigma_history", None)
    payload["artifacts"].pop("history_summary", None)
    if result.history:
        trajectory_summaries = summarize_optimized_design_histories(
            [result.history],
            spec.design_variables,
        )
        compact_summaries = compact_optimized_design_history_summaries(trajectory_summaries)
        payload["history_summary"] = compact_summaries[0]
        sigma_history = result.artifacts.get("sigma_history")
        if "optimized_design_histories" in payload["artifacts"]:
            payload["artifacts"].pop("optimized_design_histories")
        try:
            assert writer.run_dir is not None
            npz_path = save_optimization_history_npz(
                result.history,
                writer.run_dir / "optimization_history.npz",
                sigma_history=sigma_history if isinstance(sigma_history, list) else None,
            )
            payload["artifacts"]["optimization_history_npz"] = npz_path
            writer.files["optimization_history.npz"] = npz_path
        except RuntimeError as exc:
            payload.setdefault("warnings", [])
            payload["warnings"].append(str(exc))
    if spec.wants_recreated_trajectory() and result.history:
        payload.setdefault("artifacts", {})
        payload["artifacts"]["optimized_design_history_summaries"] = compact_summaries
        writer.write(spec.artifacts.trajectory_summary_filename, compact_summaries)

        if spec.artifacts.save_trajectory_plot:
            try:
                assert writer.run_dir is not None
                plot_path = save_design_trajectory_plot(
                    [result.history],
                    spec.design_variables,
                    writer.run_dir / spec.artifacts.trajectory_plot_filename,
                )
                payload["artifacts"]["design_trajectory_plot"] = plot_path
                writer.files[spec.artifacts.trajectory_plot_filename] = plot_path
            except RuntimeError as exc:
                payload.setdefault("warnings", [])
                payload["warnings"].append(str(exc))
    payload["summary"] = summarize_result_payload(payload)
    payload["artifacts"] = {**payload.get("artifacts", {}), **writer.files}
    if caveat:
        payload.setdefault("warnings", [])
        if caveat not in payload["warnings"]:
            payload["warnings"].append(caveat)

    if spec.artifacts.save_backend_summary:
        writer.write("backend_summary.json", backend.describe(spec).to_dict())
    if spec.artifacts.save_result_payload:
        writer.write("result.json", payload)

    print(json.dumps(to_jsonable(payload), indent=2))
    return 0 if result.status != "failed" else 1


def chat(args: argparse.Namespace, tools: Any) -> int:
    runtime_mode = resolve_runtime_mode(args.provider, args.runtime_mode)
    runtime = build_chat_engine(args, tools, runtime_mode)
    history: list[Message] = []
    session_config = getattr(runtime, "session_config", None)
    if session_config is not None:
        print(
            f"OpenAI Agents SDK session: {session_config.session_id} "
            f"(db: {session_config.db_path}, tracing: {'on' if session_config.tracing_enabled else 'off'})"
        )

    spec_prefix = ""
    if args.spec:
        spec = load_experiment_spec(args.spec)
        spec_prefix = f"Current normalized experiment spec:\n{json.dumps(spec.to_dict(), indent=2)}\n\n"
        if runtime_mode == "manual":
            history.append(Message(role="user", content=spec_prefix.strip()))

    try:
        spec_seed_pending = bool(args.spec and runtime_mode == "agents-sdk")
        while True:
            prompt = input("> ").strip()
            if prompt.lower() in {"exit", "quit"}:
                return 0
            effective_prompt = prompt
            if spec_seed_pending:
                effective_prompt = f"{spec_prefix}{prompt}"
                spec_seed_pending = False

            if runtime_mode == "manual":
                result = runtime.run_turn(effective_prompt, history_or_session=history)
                history = result.history
            else:
                result = runtime.run_turn(
                    effective_prompt,
                    history_or_session=None,
                    context={"session_id": session_config.session_id if session_config else None},
                )
            if result.text:
                print(result.text)
    except KeyboardInterrupt:
        print()
        return 0


def build_provider(provider_name: str, model: str) -> Any:
    if provider_name == "openai":
        return OpenAIProvider(model=model, api_key=os.environ.get("OPENAI_API_KEY"))
    if provider_name == "claude":
        return ClaudeProvider(model=model, api_key=os.environ.get("ANTHROPIC_API_KEY"))
    raise ValueError(f"Unsupported provider '{provider_name}'.")


def resolve_runtime_mode(provider_name: str, requested_mode: str | None) -> str:
    if requested_mode:
        return requested_mode
    if provider_name == "openai":
        return "agents-sdk"
    return "manual"


def build_chat_engine(args: argparse.Namespace, tools: Any, runtime_mode: str) -> Any:
    if runtime_mode == "manual":
        provider = build_provider(args.provider, args.model)
        return ManualAgentEngine(provider=provider, tools=tools)

    if args.provider != "openai":
        raise ValueError("The `agents-sdk` runtime is only supported for the OpenAI provider.")

    session_config = SessionConfig(
        session_id=args.session_id or uuid.uuid4().hex,
        db_path=args.session_db,
        tracing_enabled=not args.disable_tracing,
        runtime_mode="agents-sdk",
        resumed=bool(args.session_id),
    )
    return OpenAIAgentsSdkEngine(
        model=args.model,
        tools=tools,
        session_config=session_config,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def _execution_literature_caveat(spec: ExperimentSpec) -> str | None:
    if not spec.wants_literature():
        return None
    return (
        "Literature settings are advisory only in v1. "
        "This execution path does not auto-apply literature-derived priors or design hints; "
        "run `boed-agent literature-dry-run <spec>` first if you want literature recommendations."
    )


if __name__ == "__main__":
    sys.exit(main())
