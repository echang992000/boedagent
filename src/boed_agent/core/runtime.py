"""Compatibility wrapper for the legacy manual runtime."""

from __future__ import annotations

from boed_agent.core.engine import DEFAULT_SYSTEM_PROMPT, ManualAgentEngine


class AgentRuntime(ManualAgentEngine):
    """Backwards-compatible wrapper that preserves the old return shape."""

    def run_turn(self, prompt, history=None, max_loops=8):
        result = super().run_turn(
            prompt=prompt,
            history_or_session=history,
            context=None,
            max_loops=max_loops,
        )
        return result.text, result.history
