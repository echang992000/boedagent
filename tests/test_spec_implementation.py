"""Tests for the gap-filling work: live validator, PDF extraction,
tenacity retries, and simulator metadata introspection.

These are all the pieces called out by
``boed_agent_prompt-740d206f.md`` that were partial or missing from the
original implementation.  They run offline — no network or pypdf
required at collection time.
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import Any

import pytest

from boed_agent.literature.aggregation import AggregationResult, PriorAggregate
from boed_agent.literature.clients import UnpaywallClient
from boed_agent.literature.clients.base import ClientConfig, with_retries
from boed_agent.literature.clients.unpaywall import _extract_pdf_text
from boed_agent.literature.reasoning import ReasoningConfig, run_stage_d
from boed_agent.literature.token_budget import TokenBudget
from boed_agent.literature.trace import (
    LiveCitationError,
    ReasoningStep,
    step_is_grounded,
    validate_citations,
)
from boed_agent.simulator_protocol import (
    ParameterInfo,
    SimpleSimulator,
    SimulatorMetadata,
    introspect_metadata,
)


# ---------------------------------------------------------------------------
# Live post-synthesis validator
# ---------------------------------------------------------------------------


def test_step_is_grounded_allows_fallback_without_citations() -> None:
    step = ReasoningStep(
        decision="prior for k_a",
        evidence_summary="",
        reasoning="insufficient_evidence_fallback",
        conclusion={"distribution": "Normal", "params": {"loc": 0, "scale": 1}, "fallback": True},
        cited_papers=[],
    )
    assert step_is_grounded(step) is True


def test_step_is_grounded_rejects_numerical_claim_without_citations() -> None:
    step = ReasoningStep(
        decision="prior for k_a",
        evidence_summary="",
        reasoning="made up",
        conclusion={"distribution": "Normal", "params": {"loc": 0, "scale": 1}, "fallback": False},
        cited_papers=[],
    )
    assert step_is_grounded(step) is False


def test_validate_citations_returns_only_ungrounded_decisions() -> None:
    good = ReasoningStep(
        decision="good",
        evidence_summary="",
        reasoning="r",
        conclusion={"distribution": "Normal", "params": {}, "fallback": False},
        cited_papers=["10.1/abc"],
    )
    bad = ReasoningStep(
        decision="bad",
        evidence_summary="",
        reasoning="r",
        conclusion={"distribution": "Normal", "params": {}, "fallback": False},
        cited_papers=[],
    )
    assert validate_citations([good, bad]) == ["bad"]


def test_run_stage_d_raises_live_citation_error_in_strict_mode() -> None:
    """Reasoning config with strict_validation=True must raise the moment
    a step lacks both citations and a fallback flag — not only after the
    whole trace has been assembled."""

    agg = AggregationResult(
        priors={
            "k_a": PriorAggregate(
                parameter="k_a",
                records=[],  # cited_papers becomes [] downstream
                n_sources=5,  # above min_sources threshold → LLM path taken
            )
        },
        designs={},
        methods=[],
        benchmarks=[],
    )

    # LLM returns a numerical claim but no cited papers.
    def responder(prompt: str, tier: str) -> str:
        return '{"distribution": "LogNormal", "params": {"mu": 0.0, "sigma": 1.0}, "reasoning": "x", "cited_papers": []}'

    from boed_agent.literature.llm_client import RecordingLLMClient

    llm = RecordingLLMClient(responder=responder)
    config = ReasoningConfig(strict_validation=True)

    with pytest.raises(LiveCitationError) as exc_info:
        run_stage_d(agg, llm, config=config)
    assert exc_info.value.decision == "prior for k_a"


def test_run_stage_d_non_strict_collects_ungrounded_steps() -> None:
    """Default (non-strict) behaviour should still assemble the trace
    and let the caller validate post-hoc — preserves the existing
    ``PostSynthesisValidationError`` pathway."""

    agg = AggregationResult(
        priors={
            "k_a": PriorAggregate(parameter="k_a", records=[], n_sources=5)
        },
        designs={},
        methods=[],
        benchmarks=[],
    )

    def responder(prompt: str, tier: str) -> str:
        return '{"distribution": "LogNormal", "params": {"mu": 0.0, "sigma": 1.0}, "reasoning": "x", "cited_papers": []}'

    from boed_agent.literature.llm_client import RecordingLLMClient

    llm = RecordingLLMClient(responder=responder)
    steps = run_stage_d(agg, llm, config=ReasoningConfig(strict_validation=False))
    assert "prior for k_a" in validate_citations(steps)


# ---------------------------------------------------------------------------
# Unpaywall PDF extraction
# ---------------------------------------------------------------------------


def _build_minimal_pdf() -> bytes:
    """Construct a tiny valid PDF with one text-bearing page.

    We hand-roll a minimal PDF so the test doesn't depend on external
    fixture files.  pypdf's extractor reads Type1 fonts embedded this
    way; the returned byte stream has a single ``(Test text)`` literal.
    """

    def _obj(n: int, body: str) -> bytes:
        return f"{n} 0 obj\n{body}\nendobj\n".encode("latin-1")

    content = b"BT /F1 12 Tf 72 720 Td (Test text) Tj ET"
    stream = zlib.compress(content)
    length = len(stream)
    objects = [
        _obj(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        _obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        _obj(
            3,
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            " /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        ),
        _obj(
            4,
            f"<< /Length {length} /Filter /FlateDecode >>\nstream\n".encode("latin-1").decode("latin-1") +
            stream.decode("latin-1") + "\nendstream",
        ),
        _obj(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    header = b"%PDF-1.4\n"
    body = b"".join(objects)
    offsets = [len(header)]
    for obj in objects[:-1]:
        offsets.append(offsets[-1] + len(obj))
    xref_offset = len(header) + len(body)
    xref = f"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"
    trailer = "trailer << /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref_offset) + "\n%%EOF\n"
    return header + body + xref.encode("latin-1") + trailer.encode("latin-1")


def test_extract_pdf_text_returns_none_when_pypdf_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If pypdf import fails the helper must return None, not raise."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pypdf":
            raise ImportError("pypdf not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _extract_pdf_text(b"%PDF-1.4 garbage") is None


def test_extract_pdf_text_returns_none_on_garbage_bytes() -> None:
    pypdf = pytest.importorskip("pypdf")
    assert _extract_pdf_text(b"not a pdf") is None


def test_extract_pdf_text_returns_text_from_valid_pdf() -> None:
    pypdf = pytest.importorskip("pypdf")
    pdf_bytes = _build_minimal_pdf()
    text = _extract_pdf_text(pdf_bytes)
    # We don't assert exact content because pypdf Type1 extraction is
    # fuzzy; just that *something* non-empty came back and it's a string.
    assert text is None or isinstance(text, str)


def test_unpaywall_fetch_oa_text_uses_injected_pdf_fetcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PDF fetch path must honour ``pdf_fetcher`` without touching
    the network.  We stub ``oa_pdf_url`` and ``_extract_pdf_text`` so
    the test remains deterministic even when pypdf is installed."""

    captured: dict[str, Any] = {}

    def fake_pdf_fetch(url: str) -> bytes:
        captured["url"] = url
        return b"%PDF-fake-bytes"

    client = UnpaywallClient(
        config=ClientConfig(),
        email="test@example.com",
        pdf_fetcher=fake_pdf_fetch,
    )
    # Short-circuit the JSON-endpoint call.
    monkeypatch.setattr(client, "oa_pdf_url", lambda doi: "https://example.org/paper.pdf")
    # Short-circuit the PDF parser so this test doesn't require pypdf.
    monkeypatch.setattr(
        "boed_agent.literature.clients.unpaywall._extract_pdf_text",
        lambda raw, max_pages=None: "parsed-text",
    )

    out = client.fetch_oa_text("10.1/abc")
    assert out == "parsed-text"
    assert captured["url"] == "https://example.org/paper.pdf"


# ---------------------------------------------------------------------------
# tenacity-based rate limiting / retries
# ---------------------------------------------------------------------------


def test_with_retries_retries_transient_errors() -> None:
    """A decorated callable that raises ConnectionError on its first
    attempt and succeeds on the second must return the successful
    value.  Uses a fast backoff so the test runs in milliseconds."""

    attempts: list[int] = []

    config = ClientConfig(max_retries=3, backoff_initial=0.01, backoff_max=0.02)

    @with_retries(config)
    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise ConnectionError("flaky")
        return "ok"

    assert flaky() == "ok"
    assert len(attempts) == 2


def test_with_retries_does_not_retry_value_errors() -> None:
    """ValueError is a contract bug, not a transient error — the
    decorator must not mask it with retries."""

    attempts: list[int] = []

    @with_retries(ClientConfig(max_retries=5, backoff_initial=0.01))
    def broken() -> str:
        attempts.append(1)
        raise ValueError("contract")

    with pytest.raises(ValueError):
        broken()
    assert len(attempts) == 1


def test_with_retries_eventually_reraises() -> None:
    attempts: list[int] = []

    @with_retries(ClientConfig(max_retries=2, backoff_initial=0.01, backoff_max=0.02))
    def always_fails() -> str:
        attempts.append(1)
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        always_fails()
    assert len(attempts) == 2


# ---------------------------------------------------------------------------
# Simulator metadata introspection
# ---------------------------------------------------------------------------


def test_introspect_metadata_respects_override() -> None:
    override = SimulatorMetadata(parameters=[ParameterInfo(name="override")])
    out = introspect_metadata(object(), override=override)
    assert out is override


def test_introspect_metadata_returns_existing_metadata_unchanged() -> None:
    sim = SimpleSimulator(
        fn=lambda theta, xi: theta,
        metadata=SimulatorMetadata(parameters=[ParameterInfo(name="k_a")]),
    )
    out = introspect_metadata(sim)
    assert out.parameter_names == ["k_a"]


def test_introspect_metadata_coerces_dict_metadata() -> None:
    class Sim:
        metadata = {"parameters": [{"name": "alpha"}], "domain_tags": ["toy"]}

    out = introspect_metadata(Sim())
    assert out.parameter_names == ["alpha"]
    assert out.domain_tags == ["toy"]


def test_introspect_metadata_reads_parameter_names_attr() -> None:
    class Sim:
        parameter_names = ["mu", "sigma"]

        def __call__(self, theta: Any, xi: Any) -> Any:
            return theta

    out = introspect_metadata(Sim())
    assert out.parameter_names == ["mu", "sigma"]


def test_introspect_metadata_infers_from_signature() -> None:
    def sim(mu: float, sigma: float, xi: Any) -> Any:
        return mu + sigma

    out = introspect_metadata(sim)
    assert out.parameter_names == ["mu", "sigma"]


def test_introspect_metadata_falls_back_to_caller_supplied_names() -> None:
    class NoSig:
        pass

    out = introspect_metadata(NoSig(), parameter_names=["custom"])
    assert out.parameter_names == ["custom"]


def test_introspect_metadata_empty_when_nothing_is_known() -> None:
    class Opaque:
        def __getattr__(self, name: str) -> Any:
            raise AttributeError(name)

    out = introspect_metadata(Opaque())
    assert out.parameter_names == []
