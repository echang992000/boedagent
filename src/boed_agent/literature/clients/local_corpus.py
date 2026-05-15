"""A :class:`LocalCorpusClient` — a 'source' with no API, no network.

Most literature sources in this package hit a public API (Semantic
Scholar, arXiv, OpenAlex, PubMed).  Some deployments don't have network
access at all — air-gapped labs, regulated environments, or
sandboxes whose HTTP proxies don't allow the metadata hosts.  For those
cases we want the agent's Module 0 to still produce a grounded
literature report, just sourced from whatever the user already has on
disk.

This module scans a directory of user-provided papers and turns them
into :class:`~boed_agent.literature.clients.base.Paper` records that
look identical to ones returned by the HTTP clients.  The rest of the
pipeline (Stages A–E) runs unchanged.

Supported inputs
----------------

* ``*.pdf``  — text extracted via :mod:`pypdf` (same path as
  ``UnpaywallClient.fetch_oa_text``).  If pypdf is not installed, PDFs
  are skipped and a warning is logged.
* ``*.txt`` / ``*.md`` — read as UTF-8; first non-empty line is the
  title, the rest becomes the abstract/body.
* ``*.bib`` / ``*.json`` — optional metadata sidecars.  Any entry whose
  ``file`` or ``basename`` field matches a paper file contributes DOI,
  authors, year, and venue without overriding extracted text.

Matching
--------

The client uses a small TF-IDF-lite scorer so users can test without
pulling a real retrieval dependency.  The scorer is deterministic,
stable under query-case changes, and tokenises on word boundaries.
For richer retrieval, swap in an external index (Whoosh, tantivy,
pyserini, FAISS, or a hosted search endpoint) by subclassing and
overriding :meth:`_score`.

Example
-------

    from boed_agent.literature.clients import LocalCorpusClient
    from boed_agent.literature.search import SourceBundle

    source = LocalCorpusClient(
        corpus_dir="~/papers/pk",
        source_name="local_pk_library",
    )
    bundle = SourceBundle()
    bundle.semantic_scholar = None          # or keep online sources too
    bundle.arxiv = None
    # Plus any subclass of a Source: see LiteratureSearchModule._scan_sources

Or inject directly when you already have ``Paper`` records in memory::

    report = lit_module.search(
        problem_description=desc,
        papers=my_paper_list,                # bypasses source fanout
    )
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from boed_agent.literature.clients.base import Paper

_LOG = logging.getLogger(__name__)

# Very small English stopword list — enough to keep the scorer from
# being drowned out by "the", "of", "a", "and" in queries like "the
# Bayesian optimal experimental design of experiments".
_STOPWORDS = frozenset(
    """
    a an and or of to the is are was were be been being in on at by for with
    from as into than then this that these those it its if so not no yes do
    does did done have has had how what which when where why who whom whose
    we you they them i me he she him her our your their
    """.split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def _tokenise(text: str) -> list[str]:
    return [
        tok.lower()
        for tok in _WORD.findall(text or "")
        if tok.lower() not in _STOPWORDS and len(tok) > 1
    ]


def _extract_pdf_text(path: Path, max_pages: int | None = None) -> str | None:
    """Best-effort PDF extraction — returns None on any failure.

    Mirrors the import strategy in ``UnpaywallClient._extract_pdf_text``
    so that installing ``[literature]`` is the only prerequisite.
    """
    try:
        import pypdf
    except ImportError:
        _LOG.debug("pypdf not installed; skipping PDF %s", path)
        return None

    try:
        reader = pypdf.PdfReader(str(path))
        pages = reader.pages[:max_pages] if max_pages else reader.pages
        parts = []
        for page in pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001
                _LOG.debug("pypdf page extract error on %s: %s", path, exc)
        text = "\n".join(parts).strip()
        return text or None
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("pypdf load error on %s: %s", path, exc)
        return None


def _read_plaintext(path: Path) -> tuple[str, str]:
    """Split a .txt/.md file into (title, body).

    Title = first non-empty line, stripped of Markdown header markers.
    Body = everything after the first blank line, or the remainder.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _LOG.debug("read error on %s: %s", path, exc)
        return path.stem, ""
    lines = raw.splitlines()
    title = path.stem
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            title = stripped
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]).strip()
    return title, body


@dataclass
class CorpusEntry:
    """One file's normalised fields before turning it into a :class:`Paper`."""

    path: Path
    title: str
    body: str
    tokens: list[str]
    sidecar: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalCorpusClient:
    """Offline source: scans a directory for papers and scores them.

    Parameters
    ----------
    corpus_dir:
        Path to a directory of papers.  Recursively scanned; missing
        directories raise ``FileNotFoundError`` at construction.
    source_name:
        Goes into :attr:`Paper.source` — used downstream for provenance.
    max_pdf_pages:
        Safety cap for PDFs; long PDFs are truncated.  ``None`` disables.
    extensions:
        File suffixes to pick up (case-insensitive).  The default covers
        the realistic cases; override if your corpus uses something exotic.
    source_score:
        Baseline weight fed into :class:`RankingWeights.source_score`.
        Defaults to a middle-ground value so a local hit doesn't auto-beat
        a Semantic Scholar hit but isn't trivially dominated by recency.
    preload:
        If True (default), files are read once at construction so each
        ``search()`` call only does query-vs-tokens matching.  Set False
        for very large corpora where you'd rather re-scan on each query.
    """

    corpus_dir: os.PathLike[str] | str
    source_name: str = "local_corpus"
    max_pdf_pages: int | None = 40
    extensions: tuple[str, ...] = (".pdf", ".txt", ".md")
    source_score: float = 0.4
    preload: bool = True
    _entries: list[CorpusEntry] = field(default_factory=list, init=False, repr=False)
    _sidecars: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.corpus_dir = Path(self.corpus_dir).expanduser()
        if not self.corpus_dir.exists():
            raise FileNotFoundError(
                f"LocalCorpusClient: corpus_dir does not exist: {self.corpus_dir}"
            )
        if not self.corpus_dir.is_dir():
            raise NotADirectoryError(
                f"LocalCorpusClient: corpus_dir is not a directory: {self.corpus_dir}"
            )
        self._sidecars = self._load_sidecars()
        if self.preload:
            self._entries = list(self._iter_entries())

    # --- Source protocol -------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        """Return up to ``limit`` local papers ranked by TF-IDF-ish overlap.

        An empty query or empty corpus returns ``[]`` rather than raising,
        matching the HTTP-client contract so the pipeline's
        ``source error: ...`` path never fires for us.
        """
        q_tokens = _tokenise(query)
        entries = self._entries if self.preload else list(self._iter_entries())
        if not q_tokens or not entries:
            return []

        idf = self._idf(entries)
        scored: list[tuple[float, CorpusEntry]] = []
        for entry in entries:
            s = self._score(q_tokens, entry, idf)
            if s > 0.0:
                scored.append((s, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        papers: list[Paper] = []
        for rank, (score, entry) in enumerate(scored[:limit]):
            papers.append(self._entry_to_paper(entry, score, rank))
        return papers

    # --- internals -------------------------------------------------

    def _iter_entries(self) -> Iterable[CorpusEntry]:
        for root, _dirs, files in os.walk(self.corpus_dir):
            for fname in sorted(files):
                path = Path(root) / fname
                ext = path.suffix.lower()
                if ext not in self.extensions:
                    continue
                if ext == ".pdf":
                    body = _extract_pdf_text(path, max_pages=self.max_pdf_pages)
                    if body is None:
                        continue
                    title = path.stem.replace("_", " ").replace("-", " ")
                else:
                    title, body = _read_plaintext(path)
                tokens = _tokenise(f"{title} {body}")
                if not tokens:
                    continue
                sidecar = self._sidecars.get(path.stem, {})
                yield CorpusEntry(
                    path=path, title=title, body=body, tokens=tokens, sidecar=sidecar
                )

    def _load_sidecars(self) -> dict[str, dict[str, Any]]:
        """Load optional ``papers.json`` / ``*.json`` metadata files.

        Schema per entry (all keys optional)::

            {"basename": "smith-2019-pk", "doi": "10.1/abc", "year": 2019,
             "authors": ["Smith", "Doe"], "venue": "JPB", "citation_count": 42}
        """
        sidecars: dict[str, dict[str, Any]] = {}
        for path in sorted(Path(self.corpus_dir).rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _LOG.debug("sidecar %s unreadable: %s", path, exc)
                continue
            entries = data if isinstance(data, list) else data.get("papers") or []
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                key = entry.get("basename") or entry.get("file") or ""
                key = str(key).rsplit(".", 1)[0]
                if key:
                    sidecars[key] = entry
        return sidecars

    # --- scoring ---------------------------------------------------

    def _idf(self, entries: Sequence[CorpusEntry]) -> dict[str, float]:
        """Inverse document frequency — classic log((N+1)/(df+1)) + 1."""
        N = len(entries)
        df: dict[str, int] = {}
        for e in entries:
            for tok in set(e.tokens):
                df[tok] = df.get(tok, 0) + 1
        return {tok: math.log((N + 1) / (count + 1)) + 1.0 for tok, count in df.items()}

    def _score(
        self,
        query_tokens: Sequence[str],
        entry: CorpusEntry,
        idf: dict[str, float],
    ) -> float:
        """TF (sub-linear) × IDF, summed over the query.

        Returns 0 when there's no overlap; the search loop filters
        those out so the result list never contains zero-score entries.
        """
        if not entry.tokens:
            return 0.0
        tf: dict[str, int] = {}
        for tok in entry.tokens:
            tf[tok] = tf.get(tok, 0) + 1
        score = 0.0
        for tok in query_tokens:
            if tok in tf:
                score += (1.0 + math.log(tf[tok])) * idf.get(tok, 1.0)
        # Title-hit bonus so a query that matches the title doesn't get
        # outranked by a body-only match with higher TF.
        title_tokens = set(_tokenise(entry.title))
        score += 0.5 * sum(1 for tok in query_tokens if tok in title_tokens)
        return score

    # --- marshalling -----------------------------------------------

    def _entry_to_paper(self, entry: CorpusEntry, score: float, rank: int) -> Paper:
        sidecar = entry.sidecar
        abstract = (entry.body or "")[:2000]
        return Paper(
            title=entry.title or entry.path.stem,
            abstract=abstract,
            authors=list(sidecar.get("authors", []) or []),
            year=_as_int(sidecar.get("year")),
            doi=sidecar.get("doi"),
            arxiv_id=sidecar.get("arxiv_id"),
            openalex_id=sidecar.get("openalex_id"),
            pubmed_id=sidecar.get("pubmed_id"),
            url=sidecar.get("url") or entry.path.as_uri(),
            citation_count=_as_int(sidecar.get("citation_count"), default=0) or 0,
            source=self.source_name,
            # Normalise the TF-IDF score into [0,1]-ish via a soft squash
            # so later ranking weights behave similarly to online clients.
            source_score=self.source_score * (1.0 - math.exp(-score / 5.0)),
            sections={"body": entry.body},
            raw={"path": str(entry.path), "rank": rank, "score": score},
        )


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
