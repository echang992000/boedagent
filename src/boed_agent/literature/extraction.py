"""Stage B — sentence-level evidence mining via a cheap LLM tier.

Each call handles up to ``batch_size`` sentences in one round-trip.
Results are strictly typed and parsed; malformed records are silently
dropped (no recovery attempts) to keep the pipeline auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from boed_agent.literature.clients.base import Paper
from boed_agent.literature.filters import FilterResult
from boed_agent.literature.llm_client import (
    LLMClient,
    ModelTier,
    parse_json_strict,
)
from boed_agent.literature.token_budget import TokenBudget


EvidenceType = str  # one of the ALLOWED_TYPES below
ALLOWED_TYPES = {
    "prior_range",
    "prior_distribution",
    "design_choice",
    "method_used",
    "benchmark_number",
}


@dataclass
class EvidenceRecord:
    """Full provenance for every extracted claim."""

    paper_id: str
    sentence_id: int
    raw_text: str
    type: EvidenceType
    value: dict | str
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "sentence_id": self.sentence_id,
            "raw_text": self.raw_text,
            "type": self.type,
            "value": self.value,
            "source": self.source,
        }


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=\S)")


def split_sentences(text: str) -> list[str]:
    """Very-small sentence splitter.

    Deliberately simple — we avoid pulling in NLTK for one regex.
    """
    text = (text or "").strip()
    if not text:
        return []
    # First, collapse whitespace inside sentences.
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


_EXTRACTION_INSTRUCTIONS = (
    "You are extracting evidence for Bayesian experimental design.\n"
    "For each sentence below, output JSON: {id, type, value}.\n"
    "type must be one of {prior_range, prior_distribution, design_choice, "
    "method_used, benchmark_number, none}.\n"
    "If type=none, omit the sentence.\n"
    "Be literal — copy numbers exactly. Do not reason or infer.\n"
    "Only extract what is explicitly stated.\n"
    "Output a JSON array. No prose.\n"
)


_PRIOR_SYNTHESIS_INSTRUCTIONS = (
    "This run is synthesizing parameter priors, not selecting a BOED method.\n"
    "Treat empirical numeric measurements as prior evidence when they inform a "
    "target parameter, even if the sentence does not literally call them priors.\n"
    "For example, receptor binding affinity, dissociation constant, Kd/K_D, EC50, "
    "half-maximal dose, Hill coefficient, response baseline/maximum, and noise "
    "measurements may be prior_range or prior_distribution evidence.\n"
    "For a target parameter named kd, extract receptor affinity or dissociation "
    "constant measurements as prior_range evidence for kd. Preserve the exact "
    "source sentence; if the problem context gives a unit conversion, use it when "
    "you can do so directly from stated units, and include raw numeric fields in "
    "params when useful.\n"
    "Do not emit method_used or design_choice records unless the sentence itself "
    "contains prior-relevant numerical evidence.\n"
)


def build_extraction_prompt(
    sentences: list[tuple[int, str]],
    *,
    parameter_names: Iterable[str] | None = None,
    problem_description: str = "",
    prior_synthesis_mode: bool = False,
) -> str:
    lines = [_EXTRACTION_INSTRUCTIONS, ""]
    if prior_synthesis_mode:
        parameters = [name for name in (parameter_names or []) if name]
        lines.extend(
            [
                _PRIOR_SYNTHESIS_INSTRUCTIONS,
                "Target prior parameters: "
                + (", ".join(parameters) if parameters else "<unspecified>"),
            ]
        )
        context = " ".join((problem_description or "").split())
        if context:
            lines.append(f"Problem context: {context[:1600]}")
        lines.append("")
    for idx, sentence in sentences:
        lines.append(f"[{idx}] {sentence}")
    return "\n".join(lines)


@dataclass
class ExtractionConfig:
    batch_size: int = 30
    model_tier: ModelTier = "cheap"
    section_keys: tuple[str, ...] = ("methods", "results", "body")
    include_abstract: bool = True
    prior_synthesis_mode: bool = False


@dataclass
class ExtractionStats:
    papers_processed: int = 0
    sentences_in: int = 0
    records_out: int = 0
    batches: int = 0
    tokens: int = 0


def run_extraction(
    filtered: Iterable[FilterResult],
    llm: LLMClient,
    *,
    budget: TokenBudget | None = None,
    config: ExtractionConfig | None = None,
    parameter_names: Iterable[str] | None = None,
    problem_description: str = "",
) -> tuple[list[EvidenceRecord], ExtractionStats]:
    """Run Stage B across every surviving paper.

    The function is deliberately pure wrt its inputs: the same papers
    and the same LLM client produce the same records, making
    determinism easy in tests.
    """
    config = config or ExtractionConfig()
    stats = ExtractionStats()
    records: list[EvidenceRecord] = []

    for item in filtered:
        if not item.keep:
            continue
        paper = item.paper
        sentences = _collect_sentences(paper, config)
        if not sentences:
            continue
        stats.papers_processed += 1
        stats.sentences_in += len(sentences)
        id_to_text = {idx: text for idx, text in sentences}

        for batch in _batched(sentences, config.batch_size):
            if budget is not None and budget.check("stage_b"):
                return records, stats
            prompt = build_extraction_prompt(
                batch,
                parameter_names=parameter_names,
                problem_description=problem_description,
                prior_synthesis_mode=config.prior_synthesis_mode,
            )
            response = llm.extract(
                prompt,
                model_tier=config.model_tier,
                stage="stage_b",
                budget=budget,
            )
            stats.batches += 1
            stats.tokens += response.total_tokens
            parsed = parse_json_strict(response.text)
            if not isinstance(parsed, list):
                continue
            for entry in parsed:
                record = _coerce_record(entry, paper, id_to_text)
                if record is not None:
                    records.append(record)

    stats.records_out = len(records)
    return records, stats


def _collect_sentences(paper: Paper, config: ExtractionConfig) -> list[tuple[int, str]]:
    parts: list[str] = []
    if config.include_abstract and paper.abstract:
        parts.append(paper.abstract)
    for key in config.section_keys:
        text = paper.sections.get(key)
        if text:
            parts.append(text)
    sentences = split_sentences(" ".join(parts))
    return list(enumerate(sentences))


def _batched(items: list[tuple[int, str]], size: int) -> list[list[tuple[int, str]]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _coerce_record(
    entry: object, paper: Paper, id_to_text: dict[int, str]
) -> EvidenceRecord | None:
    if not isinstance(entry, dict):
        return None
    try:
        sentence_id = int(entry.get("id"))
    except (TypeError, ValueError):
        return None
    evidence_type = entry.get("type")
    if evidence_type == "none" or evidence_type not in ALLOWED_TYPES:
        return None
    value = entry.get("value")
    if value is None:
        return None
    if not (isinstance(value, dict) or isinstance(value, str)):
        return None
    raw_text = id_to_text.get(sentence_id, "")
    if not raw_text:
        return None
    return EvidenceRecord(
        paper_id=paper.paper_id,
        sentence_id=sentence_id,
        raw_text=raw_text,
        type=str(evidence_type),
        value=value,
        source=paper.source,
    )


__all__ = [
    "ALLOWED_TYPES",
    "EvidenceRecord",
    "ExtractionConfig",
    "ExtractionStats",
    "build_extraction_prompt",
    "run_extraction",
    "split_sentences",
]
