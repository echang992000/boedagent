"""Lazy imports for the optional OpenAI Agents SDK."""

from __future__ import annotations


def import_agents_sdk():
    try:
        import agents as sdk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "OpenAI Agents SDK support requires the optional `openai-agents` dependency. "
            "Install with `pip install -e \".[agents]\"`."
        ) from exc
    return sdk
