"""Semantic Scholar Graph API client.

Uses the public ``graph/v1/paper/search`` endpoint by default.  No API
key is required for the rate-limited public tier; authentication is
supported via the ``x-api-key`` header when configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from boed_agent.literature.clients.base import (
    ClientConfig,
    Paper,
    RateLimiter,
    with_retries,
)


_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "paperId,title,abstract,authors,year,citationCount,externalIds,url"


@dataclass
class SemanticScholarClient:
    config: ClientConfig = field(default_factory=ClientConfig)
    api_key: str | None = None

    def __post_init__(self) -> None:
        # Semantic Scholar is 100 req / 5 min unauthenticated → 0.3 / s is safe
        interval = self.config.rate_limit_interval or 3.1
        self._limiter = RateLimiter(min_interval_seconds=interval)
        # Wrap the raw-IO call in the retry decorator once per client
        # instance so the wrapping cost is amortised across calls.
        self._retrying_fetch = with_retries(self.config)(self._raw_fetch)

    def search(self, query: str, *, limit: int = 20) -> list[Paper]:
        params = {"query": query, "limit": int(limit), "fields": _FIELDS}
        payload = self._fetch(params)
        results: list[Paper] = []
        for idx, entry in enumerate(payload.get("data", []) or []):
            results.append(self._normalize(entry, idx))
        return results

    # --- helpers ---------------------------------------------------

    def _fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        self._limiter.wait()
        raw = self._retrying_fetch(params)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"data": []}

    def _raw_fetch(self, params: dict[str, Any]) -> str:
        url = f"{_ENDPOINT}?{urlencode(params)}"
        if self.config.fetcher is not None:
            return self.config.fetcher(
                url, {"x-api-key": self.api_key} if self.api_key else None
            )
        return _http_get(url, self.api_key, timeout=self.config.timeout_seconds)

    @staticmethod
    def _normalize(entry: dict[str, Any], rank: int) -> Paper:
        external = entry.get("externalIds") or {}
        authors = [a.get("name", "") for a in entry.get("authors") or []]
        return Paper(
            title=str(entry.get("title", "") or ""),
            abstract=str(entry.get("abstract", "") or ""),
            authors=[a for a in authors if a],
            year=_maybe_int(entry.get("year")),
            doi=_lc(external.get("DOI")),
            arxiv_id=_lc(external.get("ArXiv")),
            pubmed_id=_lc(external.get("PubMed")),
            url=entry.get("url"),
            citation_count=int(entry.get("citationCount") or 0),
            source="semantic_scholar",
            source_score=max(0.0, 1.0 - rank * 0.02),
            raw=entry,
        )


def _http_get(url: str, api_key: str | None, timeout: float) -> str:
    """Tiny urllib wrapper — keeps the dependency surface flat.

    The literature pipeline injects a custom ``fetcher`` in tests so
    this function only runs in live mode.
    """
    import urllib.request

    req = urllib.request.Request(url)
    if api_key:
        req.add_header("x-api-key", api_key)
    req.add_header("User-Agent", "boed-agent/0.1")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # pragma: no cover - net
        return resp.read().decode("utf-8", errors="replace")


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _lc(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).lower()


__all__ = ["SemanticScholarClient"]
