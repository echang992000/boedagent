"""Unpaywall API client.

Unpaywall is used to resolve open-access PDFs for papers that pass the
ranking stage.  We do not require email auth to be present — callers
who do not set it simply receive ``None``.

On top of returning the PDF URL, the client can also download the PDF
and extract text via ``pypdf`` (see :meth:`fetch_oa_text`).  ``pypdf``
is an *optional* dependency from the ``[literature]`` extra — the code
degrades gracefully when it is not installed.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from boed_agent.literature.clients.base import ClientConfig, RateLimiter, with_retries


_ENDPOINT = "https://api.unpaywall.org/v2"
_LOG = logging.getLogger(__name__)


@dataclass
class UnpaywallClient:
    config: ClientConfig = field(default_factory=ClientConfig)
    email: str | None = None
    # When set, used instead of urllib to fetch raw PDF bytes.  Tests
    # inject an in-memory byte source this way.
    pdf_fetcher: Callable[[str], bytes] | None = None

    def __post_init__(self) -> None:
        interval = self.config.rate_limit_interval or 0.2
        self._limiter = RateLimiter(min_interval_seconds=interval)
        retry_wrapper = with_retries(self.config)
        self._retrying_meta = retry_wrapper(self._raw_meta_fetch)
        self._retrying_pdf = retry_wrapper(self._raw_pdf_fetch)

    def oa_pdf_url(self, doi: str) -> str | None:
        if not doi or not self.email:
            return None
        self._limiter.wait()
        try:
            raw = self._retrying_meta(doi)
        except Exception as exc:  # pragma: no cover - network guard
            _LOG.debug("unpaywall metadata fetch failed for %s: %s", doi, exc)
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        best = payload.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")
        return pdf_url

    def _raw_meta_fetch(self, doi: str) -> str:
        url = f"{_ENDPOINT}/{doi}?email={self.email}"
        if self.config.fetcher is not None:
            return self.config.fetcher(url, None)
        return _http_get(url, timeout=self.config.timeout_seconds)

    def fetch_oa_text(self, doi: str, *, max_pages: int | None = None) -> str | None:
        """Download the OA PDF for ``doi`` and return extracted text.

        Returns ``None`` when: no OA URL is resolvable, ``pypdf`` is not
        installed, the download fails, or the PDF is unparseable.  The
        method never raises for missing dependencies or network errors —
        the literature pipeline treats text as best-effort enrichment.

        Parameters
        ----------
        doi:
            The canonical DOI (lower-cased is fine).
        max_pages:
            Optional upper bound on pages to extract.  Useful when a
            paper is long and only the abstract / methods are needed —
            set ``max_pages=6`` for a typical quick scan.
        """

        pdf_url = self.oa_pdf_url(doi)
        if not pdf_url:
            return None
        raw = self._download_pdf(pdf_url)
        if not raw:
            return None
        return _extract_pdf_text(raw, max_pages=max_pages)

    def _download_pdf(self, url: str) -> bytes | None:
        self._limiter.wait()
        try:
            return self._retrying_pdf(url)
        except Exception as exc:  # pragma: no cover - network guard
            _LOG.debug("unpaywall PDF download failed: %s", exc)
            return None

    def _raw_pdf_fetch(self, url: str) -> bytes:
        if self.pdf_fetcher is not None:
            return self.pdf_fetcher(url)
        return _http_get_bytes(url, timeout=self.config.timeout_seconds)


def _extract_pdf_text(raw: bytes, *, max_pages: int | None = None) -> str | None:
    """Parse ``raw`` PDF bytes with ``pypdf``; return ``None`` on any failure.

    Kept separate from the class so tests can exercise the parser with
    synthetic byte streams without instantiating an ``UnpaywallClient``.
    """

    try:
        import pypdf  # optional dep — degrades to None if missing
    except ImportError:  # pragma: no cover - optional dep
        _LOG.debug("pypdf not installed; cannot extract PDF text")
        return None

    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
    except Exception as exc:  # pragma: no cover - parse guard
        _LOG.debug("pypdf failed to open bytes: %s", exc)
        return None

    pieces: list[str] = []
    pages = reader.pages
    limit = min(len(pages), max_pages) if max_pages else len(pages)
    for page in pages[:limit]:
        try:
            pieces.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - parse guard
            _LOG.debug("pypdf page extraction failed: %s", exc)
            pieces.append("")
    text = "\n".join(p for p in pieces if p).strip()
    return text or None


def _http_get(url: str, timeout: float) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "boed-agent/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # pragma: no cover - net
        return resp.read().decode("utf-8", errors="replace")


def _http_get_bytes(url: str, timeout: float) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "boed-agent/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # pragma: no cover - net
        return resp.read()


__all__ = ["UnpaywallClient"]
