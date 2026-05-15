"""OpenAlex Works API client."""

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


_ENDPOINT = "https://api.openalex.org/works"


@dataclass
class OpenAlexClient:
    config: ClientConfig = field(default_factory=ClientConfig)
    mailto: str | None = None

    def __post_init__(self) -> None:
        interval = self.config.rate_limit_interval or 1.0
        self._limiter = RateLimiter(min_interval_seconds=interval)
        self._retrying_fetch = with_retries(self.config)(self._raw_fetch)

    def search(self, query: str, *, limit: int = 20) -> list[Paper]:
        params: dict[str, Any] = {
            "search": query,
            "per-page": int(limit),
        }
        if self.mailto:
            params["mailto"] = self.mailto
        payload = self._fetch(params)
        results: list[Paper] = []
        for rank, entry in enumerate(payload.get("results") or []):
            results.append(self._normalize(entry, rank))
        return results

    def _fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        self._limiter.wait()
        raw = self._retrying_fetch(params)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _raw_fetch(self, params: dict[str, Any]) -> str:
        url = f"{_ENDPOINT}?{urlencode(params)}"
        if self.config.fetcher is not None:
            return self.config.fetcher(url, None)
        return _http_get(url, timeout=self.config.timeout_seconds)

    @staticmethod
    def _normalize(entry: dict[str, Any], rank: int) -> Paper:
        ids = entry.get("ids") or {}
        authors = [
            (auth.get("author") or {}).get("display_name", "")
            for auth in entry.get("authorships") or []
        ]
        # OpenAlex stores the abstract as an inverted index for copyright reasons.
        abstract = _invert_abstract(entry.get("abstract_inverted_index"))
        return Paper(
            title=str(entry.get("title", "") or entry.get("display_name", "") or ""),
            abstract=abstract,
            authors=[a for a in authors if a],
            year=_maybe_int(entry.get("publication_year")),
            doi=(_strip_doi_prefix(ids.get("doi")) if ids.get("doi") else None),
            openalex_id=str(entry.get("id") or "").lower() or None,
            pubmed_id=str(ids.get("pmid") or "").rsplit("/", 1)[-1] or None,
            url=ids.get("doi") or entry.get("id"),
            citation_count=int(entry.get("cited_by_count") or 0),
            source="openalex",
            source_score=max(0.0, 1.0 - rank * 0.02),
            raw=entry,
        )


def _invert_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, locs in index.items():
        for loc in locs:
            positions.append((loc, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def _strip_doi_prefix(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _http_get(url: str, timeout: float) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "boed-agent/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # pragma: no cover - net
        return resp.read().decode("utf-8", errors="replace")


__all__ = ["OpenAlexClient"]
