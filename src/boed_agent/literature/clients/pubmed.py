"""PubMed E-utilities client (biomedical only)."""

from __future__ import annotations

import json
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


_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class PubMedClient:
    config: ClientConfig = field(default_factory=ClientConfig)
    api_key: str | None = None
    email: str | None = None

    def __post_init__(self) -> None:
        interval = self.config.rate_limit_interval or (
            0.35 if self.api_key else 1.0
        )  # 10 req/s with key, else 3/s
        self._limiter = RateLimiter(min_interval_seconds=interval)
        self._retrying_fetch = with_retries(self.config)(self._raw_fetch)

    def search(self, query: str, *, limit: int = 20) -> list[Paper]:
        ids = self._esearch(query, limit)
        if not ids:
            return []
        records = self._efetch(ids)
        results: list[Paper] = []
        for rank, entry in enumerate(records):
            results.append(self._normalize(entry, rank))
        return results

    # --- helpers ---------------------------------------------------

    def _esearch(self, query: str, limit: int) -> list[str]:
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": int(limit),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        raw = self._fetch(_ESEARCH, params)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return list((data.get("esearchresult") or {}).get("idlist") or [])

    def _efetch(self, ids: list[str]) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        xml_text = self._fetch(_EFETCH, params)
        return _parse_pubmed_xml(xml_text)

    def _fetch(self, endpoint: str, params: dict[str, Any]) -> str:
        self._limiter.wait()
        return self._retrying_fetch(endpoint, params)

    def _raw_fetch(self, endpoint: str, params: dict[str, Any]) -> str:
        url = f"{endpoint}?{urlencode(params)}"
        if self.config.fetcher is not None:
            return self.config.fetcher(url, None)
        return _http_get(url, timeout=self.config.timeout_seconds)

    @staticmethod
    def _normalize(entry: dict[str, Any], rank: int) -> Paper:
        return Paper(
            title=str(entry.get("title", "")),
            abstract=str(entry.get("abstract", "")),
            authors=list(entry.get("authors") or []),
            year=_maybe_int(entry.get("year")),
            pubmed_id=str(entry.get("pmid") or "") or None,
            doi=_lc(entry.get("doi")),
            url=(
                f"https://pubmed.ncbi.nlm.nih.gov/{entry['pmid']}/"
                if entry.get("pmid")
                else None
            ),
            source="pubmed",
            source_score=max(0.0, 1.0 - rank * 0.02),
            raw=entry,
        )


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    records: list[dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle", default="") or ""
        abstract = " ".join(
            (el.text or "") for el in article.findall(".//Abstract/AbstractText")
        ).strip()
        year = article.findtext(".//PubDate/Year")
        authors = [
            " ".join(
                filter(
                    None,
                    [
                        a.findtext("ForeName"),
                        a.findtext("LastName"),
                    ],
                )
            )
            for a in article.findall(".//Author")
        ]
        doi = None
        for eid in article.findall(".//ArticleId"):
            if (eid.attrib.get("IdType") or "").lower() == "doi":
                doi = (eid.text or "").lower()
                break
        records.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "authors": [a for a in authors if a],
                "doi": doi,
            }
        )
    return records


def _http_get(url: str, timeout: float) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "boed-agent/0.1"})
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


__all__ = ["PubMedClient"]
