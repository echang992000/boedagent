#!/usr/bin/env python3
"""Ad-hoc live literature search across the three free metadata APIs.

Runs the exact same clients the BOED agent uses, but outside pytest, so
you can sanity-check that your network, rate-limiter, and retry config
are all working before kicking off a full agent run.

Usage
-----

    # Install once (picks up requests / tenacity / pypdf):
    pip install -e ".[literature]"

    # Default query
    python scripts/live_literature_search.py

    # Custom query, more results per source
    python scripts/live_literature_search.py \\
        --query "expected information gain" --limit 5

Semantic Scholar sometimes rate-limits unauthenticated traffic; if you
have an API key, set SEMANTIC_SCHOLAR_API_KEY in the environment and
this script will pick it up via ClientConfig.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Iterable

from boed_agent.literature.clients import (
    ArxivClient,
    OpenAlexClient,
    SemanticScholarClient,
)
from boed_agent.literature.clients.base import ClientConfig


def _format_record(record: object, index: int) -> str:
    title = getattr(record, "title", None) or "(untitled)"
    authors = getattr(record, "authors", None) or []
    year = getattr(record, "year", None) or "n/d"
    doi = getattr(record, "doi", None) or ""
    url = getattr(record, "url", None) or ""
    venue = getattr(record, "venue", None) or ""

    author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    header = f"  [{index}] {title}".strip()
    meta_parts = [str(year)]
    if venue:
        meta_parts.append(venue)
    if author_str:
        meta_parts.append(author_str)
    meta = " · ".join(meta_parts)
    link = doi and f"https://doi.org/{doi}" or url or ""

    wrapped = textwrap.fill(header, width=100, subsequent_indent="      ")
    out = [wrapped, f"      {meta}"]
    if link:
        out.append(f"      {link}")
    return "\n".join(out)


def _print_section(name: str, results: Iterable[object]) -> None:
    records = list(results)
    print(f"\n{name}  ({len(records)} result{'s' if len(records) != 1 else ''})")
    print("-" * len(name))
    if not records:
        print("  (no results)")
        return
    for i, record in enumerate(records, start=1):
        print(_format_record(record, i))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--query",
        default="Bayesian optimal experimental design",
        help="search string (default: %(default)r)",
    )
    parser.add_argument("--limit", type=int, default=3, help="results per source (default: 3)")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds")
    args = parser.parse_args()

    config = ClientConfig(timeout_seconds=args.timeout)

    print(f"Query : {args.query!r}")
    print(f"Limit : {args.limit} per source")
    print(f"Timeout: {args.timeout}s")

    any_error = False

    # Semantic Scholar -------------------------------------------------
    ss_kwargs = {"config": config}
    if api_key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        ss_kwargs["api_key"] = api_key
    try:
        ss = SemanticScholarClient(**ss_kwargs)
        _print_section("Semantic Scholar", ss.search(args.query, limit=args.limit))
    except Exception as exc:  # noqa: BLE001
        any_error = True
        print(f"\nSemantic Scholar — ERROR: {type(exc).__name__}: {exc}")

    # arXiv ------------------------------------------------------------
    try:
        arxiv = ArxivClient(config=config)
        _print_section("arXiv", arxiv.search(args.query, limit=args.limit))
    except Exception as exc:  # noqa: BLE001
        any_error = True
        print(f"\narXiv — ERROR: {type(exc).__name__}: {exc}")

    # OpenAlex ---------------------------------------------------------
    try:
        openalex = OpenAlexClient(config=config)
        _print_section("OpenAlex", openalex.search(args.query, limit=args.limit))
    except Exception as exc:  # noqa: BLE001
        any_error = True
        print(f"\nOpenAlex — ERROR: {type(exc).__name__}: {exc}")

    print()
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
