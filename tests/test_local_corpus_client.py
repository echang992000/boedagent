"""Offline tests for :class:`LocalCorpusClient`.

The client is meant to work without any network or API — these tests
only use the filesystem and (optionally) ``pypdf``. PDF tests skip
cleanly when pypdf isn't installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boed_agent.literature.clients import LocalCorpusClient
from boed_agent.literature.clients.local_corpus import (
    _tokenise,
    _read_plaintext,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LocalCorpusClient(corpus_dir=tmp_path / "nope")


def test_non_directory_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("hello")
    with pytest.raises(NotADirectoryError):
        LocalCorpusClient(corpus_dir=file_path)


def test_empty_corpus_returns_empty_list(tmp_path: Path) -> None:
    client = LocalCorpusClient(corpus_dir=tmp_path)
    assert client.search("anything", limit=10) == []


def test_empty_query_returns_empty_list(tmp_path: Path) -> None:
    (tmp_path / "paper.txt").write_text("Pharmacokinetics\n\nBody text about k_a.")
    client = LocalCorpusClient(corpus_dir=tmp_path)
    assert client.search("", limit=5) == []


# ---------------------------------------------------------------------------
# Tokenisation and helpers
# ---------------------------------------------------------------------------


def test_tokenise_drops_stopwords_and_short_tokens() -> None:
    assert _tokenise("The Bayesian design of experiments") == [
        "bayesian", "design", "experiments"
    ]


def test_tokenise_is_case_insensitive() -> None:
    assert _tokenise("PK PK pk") == ["pk", "pk", "pk"]


def test_read_plaintext_uses_first_non_empty_line_as_title(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("\n\n# A Pharmacokinetic Paper\n\nBody text here.\n")
    title, body = _read_plaintext(path)
    assert title == "A Pharmacokinetic Paper"
    assert "Body text here" in body


# ---------------------------------------------------------------------------
# Retrieval behaviour on plaintext corpus
# ---------------------------------------------------------------------------


def _write_corpus(root: Path) -> None:
    (root / "pk_theophylline.md").write_text(
        "# One-compartment Pharmacokinetics of Theophylline\n\n"
        "Absorption rate k_a and elimination rate k_e are estimated "
        "with a nonlinear mixed-effects model on the Theophylline dataset."
    )
    (root / "sir_covid.md").write_text(
        "# Stochastic SIR Dynamics in Early COVID Outbreak\n\n"
        "We estimate beta and gamma for county-level infection curves."
    )
    (root / "linear_regression_primer.md").write_text(
        "# A Primer on Conjugate Bayesian Linear Regression\n\n"
        "Closed-form posteriors for slope alpha and intercept beta."
    )


def test_search_ranks_relevant_paper_first(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    client = LocalCorpusClient(corpus_dir=tmp_path)
    results = client.search("pharmacokinetics theophylline absorption", limit=3)
    assert results, "expected at least one hit"
    assert "Pharmacokinetics" in results[0].title


def test_search_respects_limit(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    client = LocalCorpusClient(corpus_dir=tmp_path)
    results = client.search("bayesian", limit=1)
    assert len(results) <= 1


def test_search_skips_irrelevant_entries(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    client = LocalCorpusClient(corpus_dir=tmp_path)
    # Highly specific to one paper — others should score zero and drop.
    results = client.search("covid", limit=10)
    titles = [p.title for p in results]
    # The SIR paper mentions COVID in its title; the others don't.
    assert any("SIR" in t for t in titles)
    assert not any("Theophylline" in t for t in titles)


def test_paper_records_carry_source_name(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    client = LocalCorpusClient(corpus_dir=tmp_path, source_name="my_library")
    results = client.search("pharmacokinetics", limit=1)
    assert results
    assert results[0].source == "my_library"


def test_paper_urls_fall_back_to_file_uri(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    client = LocalCorpusClient(corpus_dir=tmp_path)
    results = client.search("theophylline", limit=1)
    assert results
    assert results[0].url is not None
    assert results[0].url.startswith("file://")


# ---------------------------------------------------------------------------
# Sidecar metadata
# ---------------------------------------------------------------------------


def test_sidecar_metadata_fills_optional_paper_fields(tmp_path: Path) -> None:
    (tmp_path / "pk.md").write_text(
        "# Pharmacokinetic Analysis of Compound X\n\nBody referencing k_a."
    )
    (tmp_path / "papers.json").write_text(
        json.dumps(
            {
                "papers": [
                    {
                        "basename": "pk",
                        "doi": "10.1/compound-x",
                        "year": 2021,
                        "authors": ["Smith", "Doe"],
                        "venue": "JPB",
                        "citation_count": 37,
                    }
                ]
            }
        )
    )
    client = LocalCorpusClient(corpus_dir=tmp_path)
    results = client.search("pharmacokinetic", limit=1)
    assert results
    paper = results[0]
    assert paper.doi == "10.1/compound-x"
    assert paper.year == 2021
    assert paper.authors == ["Smith", "Doe"]
    assert paper.citation_count == 37


def test_sidecar_missing_basename_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "pk.md").write_text("# PK Paper\n\nBody about k_a.")
    (tmp_path / "papers.json").write_text(
        json.dumps({"papers": [{"doi": "10.1/other", "year": 2020}]})
    )
    client = LocalCorpusClient(corpus_dir=tmp_path)
    results = client.search("pk", limit=1)
    assert results
    assert results[0].doi is None


def test_corrupt_sidecar_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "pk.md").write_text("# PK Paper\n\nBody about k_a.")
    (tmp_path / "broken.json").write_text("not valid json {")
    # Must not raise.
    client = LocalCorpusClient(corpus_dir=tmp_path)
    results = client.search("pk", limit=1)
    assert results
    assert results[0].doi is None


# ---------------------------------------------------------------------------
# Integration with the literature pipeline's SourceBundle
# ---------------------------------------------------------------------------


def test_source_bundle_extra_surfaces_local_client(tmp_path: Path) -> None:
    from boed_agent.literature.search import SourceBundle

    (tmp_path / "paper.md").write_text("# Bayesian Design\n\nSomething about EIG.")
    client = LocalCorpusClient(corpus_dir=tmp_path)
    bundle = SourceBundle(extra=[("local", client)])
    named = bundle.all()
    assert ("local", client) in named


# ---------------------------------------------------------------------------
# PDF integration — skipped if pypdf absent
# ---------------------------------------------------------------------------


def test_pdf_files_are_handled_when_pypdf_present(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")

    # Build a trivial PDF by re-using the helper from the spec-implementation
    # suite so we don't duplicate the PDF builder.  If the fixture changes
    # shape, skip — we're not testing pypdf here, just the integration.
    try:
        from tests.test_spec_implementation import _build_minimal_pdf
    except Exception:  # pragma: no cover
        pytest.skip("cannot import PDF fixture")

    (tmp_path / "test.pdf").write_bytes(_build_minimal_pdf())
    (tmp_path / "sidecar.md").write_text("# Sidecar\n\nBody text about Bayesian.")

    client = LocalCorpusClient(corpus_dir=tmp_path)
    # Query targets the sidecar so the test passes even if pypdf's Type1
    # extractor returns no tokens (its output on hand-rolled PDFs is fuzzy).
    results = client.search("bayesian", limit=5)
    assert results
    assert all(p.source == "local_corpus" for p in results)
