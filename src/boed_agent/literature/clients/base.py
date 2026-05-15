"""Shared types and helpers for literature API clients."""

from __future__ import annotations

import functools
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")


@dataclass
class Paper:
    """Normalised cross-source paper record.

    Source-specific scores are kept in ``source_score`` so the ranking
    stage can re-weight them uniformly.
    """

    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    pubmed_id: str | None = None
    url: str | None = None
    citation_count: int = 0
    source: str = ""
    source_score: float = 0.0
    sections: dict[str, str] = field(default_factory=dict)
    raw: Any = None

    @property
    def paper_id(self) -> str:
        """Stable identifier for dedup / provenance.

        Preference order: DOI → arXiv ID → OpenAlex ID → PubMed ID →
        SHA-256 of title.  Using a deterministic hash means the same
        paper fetched from two sources collapses to one record.
        """
        for candidate in (self.doi, self.arxiv_id, self.openalex_id, self.pubmed_id):
            if candidate:
                return str(candidate).lower()
        return "title:" + hashlib.sha256(self.title.lower().strip().encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "authors": list(self.authors),
            "year": self.year,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "openalex_id": self.openalex_id,
            "pubmed_id": self.pubmed_id,
            "url": self.url,
            "citation_count": self.citation_count,
            "source": self.source,
            "source_score": self.source_score,
            "paper_id": self.paper_id,
        }


class RateLimiter:
    """Minimal token-bucket rate limiter for the literature clients.

    The clients are best-effort — if a caller hammers them the limiter
    just sleeps instead of raising, which is exactly what we want for
    a long-running literature search.
    """

    def __init__(self, min_interval_seconds: float = 0.0) -> None:
        self._interval = float(min_interval_seconds)
        self._last = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        remaining = self._interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()


@dataclass
class ClientConfig:
    """Common knobs for every literature client."""

    timeout_seconds: float = 20.0
    max_retries: int = 3
    rate_limit_interval: float = 0.0
    fetcher: Optional[Callable[[str, dict[str, Any] | None], str]] = None
    """When set, bypass real network I/O — used by tests and dry-runs."""

    # Exponential-backoff seeds for transient errors.  Tuned for the
    # rate-limit windows of the five supported sources without being so
    # aggressive that a flaky network hangs the pipeline.
    backoff_initial: float = 0.5
    backoff_max: float = 8.0


def with_retries(config: ClientConfig) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorate a callable with exponential-backoff retries on transient errors.

    Prefers ``tenacity`` when installed (matches the spec's
    ``"Use tenacity for backoff"`` requirement); falls back to a local
    loop otherwise so offline test environments still work.

    Only network-shaped exceptions — ``OSError`` (covers urllib
    transport errors), ``TimeoutError``, ``ConnectionError`` — are
    retried.  ``ValueError`` / ``json.JSONDecodeError`` are *not*
    retried because they indicate a contract mismatch, which retrying
    will never fix.
    """

    retryable: tuple[type[BaseException], ...] = (
        TimeoutError,
        ConnectionError,
        OSError,
    )

    try:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )
    except ImportError:
        def decorator(fn: Callable[..., _T]) -> Callable[..., _T]:
            @functools.wraps(fn)
            def wrapped(*args: Any, **kwargs: Any) -> _T:
                attempts = max(int(config.max_retries), 1)
                delay = float(config.backoff_initial)
                last_exc: BaseException | None = None
                for attempt in range(attempts):
                    try:
                        return fn(*args, **kwargs)
                    except retryable as exc:
                        last_exc = exc
                        if attempt == attempts - 1:
                            raise
                        _LOG.debug(
                            "retrying %s after %s (attempt %s/%s)",
                            fn.__qualname__, exc, attempt + 1, attempts,
                        )
                        time.sleep(min(delay, float(config.backoff_max)))
                        delay *= 2
                # unreachable; the loop either returns or re-raises
                raise last_exc  # type: ignore[misc]

            return wrapped

        return decorator

    # tenacity path — preferred.
    def decorator(fn: Callable[..., _T]) -> Callable[..., _T]:
        wrapped = retry(
            reraise=True,
            stop=stop_after_attempt(max(int(config.max_retries), 1)),
            wait=wait_exponential(
                multiplier=float(config.backoff_initial),
                max=float(config.backoff_max),
            ),
            retry=retry_if_exception_type(retryable),
        )(fn)
        return wrapped

    return decorator


__all__ = ["Paper", "RateLimiter", "ClientConfig", "with_retries"]
