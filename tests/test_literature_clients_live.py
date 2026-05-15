"""Optional live HTTP smoke tests for the literature API clients.

These tests are guarded by the ``live`` pytest marker — they are
skipped by default (see ``addopts`` in ``pyproject.toml``).  Run them
explicitly with:

    pytest -m live tests/test_literature_clients_live.py

They are intentionally minimal: each one asks for a single, famous
search term that should reliably return at least one result.  We do
*not* pin result counts because third-party rankings change over time.
"""

from __future__ import annotations

import socket

import pytest

from boed_agent.literature.clients import (
    ArxivClient,
    OpenAlexClient,
    SemanticScholarClient,
)
from boed_agent.literature.clients.base import ClientConfig


pytestmark = pytest.mark.live


def _has_network(host: str = "api.semanticscholar.org", port: int = 443) -> bool:
    """Quick DNS+TCP probe so the test itself reports the root cause
    when the CI runner is offline, instead of burying it in a urllib
    stacktrace."""
    try:
        socket.create_connection((host, port), timeout=2.0).close()
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def _online() -> None:
    if not _has_network():
        pytest.skip("no network available")


def test_semantic_scholar_returns_results(_online: None) -> None:
    client = SemanticScholarClient(config=ClientConfig(timeout_seconds=15.0))
    results = client.search("Bayesian optimal experimental design", limit=3)
    assert isinstance(results, list)
    assert any(p.title for p in results), "expected at least one titled result"


def test_arxiv_returns_results(_online: None) -> None:
    # arXiv enforces ≥3s between requests; the default ClientConfig
    # honours that via RateLimiter so a single test call is fine.
    client = ArxivClient(config=ClientConfig(timeout_seconds=15.0))
    results = client.search("variational Bayesian experimental design", limit=3)
    assert isinstance(results, list)
    assert any(p.title for p in results)


def test_openalex_returns_results(_online: None) -> None:
    client = OpenAlexClient(config=ClientConfig(timeout_seconds=15.0))
    results = client.search("expected information gain", limit=3)
    assert isinstance(results, list)
    assert any(p.title for p in results)
