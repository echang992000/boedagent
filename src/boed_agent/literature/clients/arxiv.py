"""arXiv API client.

The arXiv API returns Atom XML.  We parse it with the stdlib
``xml.etree.ElementTree`` to avoid pulling in an extra dependency.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from boed_agent.literature.clients.base import (
    ClientConfig,
    Paper,
    RateLimiter,
    with_retries,
)


_ENDPOINT = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass
class ArxivClient:
    config: ClientConfig = field(default_factory=ClientConfig)

    def __post_init__(self) -> None:
        # arXiv policy: ≥3 s between requests
        interval = self.config.rate_limit_interval or 3.0
        self._limiter = RateLimiter(min_interval_seconds=interval)
        self._retrying_fetch = with_retries(self.config)(self._raw_fetch)

    def search(self, query: str, *, limit: int = 20) -> list[Paper]:
        params = {
            "search_query": f"all:{query}",
            "max_results": int(limit),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        xml_text = self._fetch(params)
        return self._parse(xml_text)

    # --- helpers ---------------------------------------------------

    def _fetch(self, params: dict[str, Any]) -> str:
        self._limiter.wait()
        return self._retrying_fetch(params)

    def _raw_fetch(self, params: dict[str, Any]) -> str:
        url = f"{_ENDPOINT}?{urlencode(params)}"
        if self.config.fetcher is not None:
            return self.config.fetcher(url, None)
        return _http_get(url, timeout=self.config.timeout_seconds)

    def _parse(self, xml_text: str) -> list[Paper]:
        if not xml_text:
            return []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        results: list[Paper] = []
        for rank, entry in enumerate(root.findall("atom:entry", _NS)):
            title = (entry.findtext("atom:title", default="", namespaces=_NS) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=_NS) or "").strip()
            published = entry.findtext("atom:published", default="", namespaces=_NS) or ""
            year = None
            if published[:4].isdigit():
                year = int(published[:4])
            authors = [
                (a.findtext("atom:name", default="", namespaces=_NS) or "").strip()
                for a in entry.findall("atom:author", _NS)
            ]
            arxiv_id = None
            for id_elem in entry.findall("atom:id", _NS):
                text = (id_elem.text or "").strip()
                if "arxiv.org/abs/" in text:
                    arxiv_id = text.rsplit("/", 1)[-1].lower()
                    break
            doi_elem = entry.find("arxiv:doi", _NS)
            doi = (doi_elem.text or "").strip().lower() if doi_elem is not None else None
            url = None
            for link in entry.findall("atom:link", _NS):
                if link.attrib.get("type") == "text/html":
                    url = link.attrib.get("href")
                    break
            results.append(
                Paper(
                    title=title,
                    abstract=summary,
                    authors=[a for a in authors if a],
                    year=year,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    url=url,
                    source="arxiv",
                    source_score=max(0.0, 1.0 - rank * 0.02),
                    raw=ET.tostring(entry, encoding="unicode"),
                )
            )
        return results


def _http_get(url: str, timeout: float) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "boed-agent/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # pragma: no cover - net
        return resp.read().decode("utf-8", errors="replace")


__all__ = ["ArxivClient"]
