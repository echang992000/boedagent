"""Literature-informed reasoning pipeline for :class:`BOEDAgent`.

The subpackage is organised as a five-stage pipeline plus a handful of
supporting clients:

* ``clients/`` — HTTP clients for the external APIs.  Each file is
  optional — the pipeline degrades gracefully when one source is
  unreachable.
* ``filters.py`` — Stage A regex / rule-based pre-filter (no LLM).
* ``extraction.py`` — Stage B sentence-level evidence mining
  (cheap-tier LLM, batched).
* ``aggregation.py`` — Stage C deterministic aggregation of extracted
  records (pure Python).
* ``reasoning.py`` — Stage D per-decision chain-of-thought reasoning
  (reasoning-tier LLM, small distilled inputs).
* ``trace.py`` — Stage E reasoning-trace assembly and markdown
  rendering.
* ``token_budget.py`` — :class:`TokenBudget` helper.
* ``llm_client.py`` — provider-neutral :class:`LLMClient` protocol.
* ``report.py`` — :class:`LiteratureReport` dataclass.
* ``search.py`` — :class:`LiteratureSearchModule` orchestrator.
"""

from boed_agent.literature.llm_client import LLMClient, NullLLMClient, RecordingLLMClient
from boed_agent.literature.codex_cli_client import CodexCLILLMClient
from boed_agent.literature.report import LiteratureReport
from boed_agent.literature.search import LiteratureSearchModule
from boed_agent.literature.token_budget import TokenBudget, TokenBudgetExceeded
from boed_agent.literature.trace import ReasoningStep, ReasoningTrace

__all__ = [
    "LLMClient",
    "CodexCLILLMClient",
    "NullLLMClient",
    "RecordingLLMClient",
    "LiteratureReport",
    "LiteratureSearchModule",
    "ReasoningStep",
    "ReasoningTrace",
    "TokenBudget",
    "TokenBudgetExceeded",
]
