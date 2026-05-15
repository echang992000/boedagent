"""Stage C — deterministic aggregation of evidence records.

Pure Python, no LLM.  The goal is to compress every extracted record
into a compact, per-parameter / per-dimension / per-method table that
Stage D can reason over without re-reading the source papers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Sequence

from boed_agent.literature.extraction import EvidenceRecord


@dataclass
class PriorEvidence:
    paper_id: str
    sentence_id: int
    raw_text: str
    distribution: str | None
    params: dict[str, float] = field(default_factory=dict)
    low: float | None = None
    high: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "sentence_id": self.sentence_id,
            "raw_text": self.raw_text,
            "distribution": self.distribution,
            "params": dict(self.params),
            "low": self.low,
            "high": self.high,
        }


@dataclass
class PriorAggregate:
    parameter: str
    records: list[PriorEvidence] = field(default_factory=list)
    reported_distributions: Counter = field(default_factory=Counter)
    range_union: tuple[float | None, float | None] = (None, None)
    range_iqr: tuple[float | None, float | None] = (None, None)
    n_sources: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "n_sources": self.n_sources,
            "reported_distributions": dict(self.reported_distributions),
            "range_union": list(self.range_union),
            "range_iqr": list(self.range_iqr),
            "records": [r.to_dict() for r in self.records],
        }


@dataclass
class DesignAggregate:
    dimension: str
    records: list[dict] = field(default_factory=list)
    reported_choices: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "reported_choices": dict(self.reported_choices),
            "records": list(self.records),
        }


@dataclass
class MethodAggregate:
    method: str
    count: int
    papers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "count": self.count,
            "papers": list(self.papers),
        }


@dataclass
class AggregationResult:
    priors: dict[str, PriorAggregate]
    designs: dict[str, DesignAggregate]
    methods: list[MethodAggregate]
    benchmarks: list[dict]

    def to_dict(self) -> dict[str, object]:
        return {
            "priors": {k: v.to_dict() for k, v in self.priors.items()},
            "designs": {k: v.to_dict() for k, v in self.designs.items()},
            "methods": [m.to_dict() for m in self.methods],
            "benchmarks": list(self.benchmarks),
        }


def aggregate(
    records: Iterable[EvidenceRecord],
    *,
    parameter_names: Sequence[str] | None = None,
    design_dimensions: Sequence[str] | None = None,
) -> AggregationResult:
    """Collapse raw records into structured per-decision tables."""
    priors: dict[str, PriorAggregate] = {}
    designs: dict[str, DesignAggregate] = {}
    method_counts: Counter = Counter()
    method_papers: dict[str, list[str]] = {}
    benchmarks: list[dict] = []

    for record in records:
        if record.type in {"prior_range", "prior_distribution"}:
            _ingest_prior(record, priors, parameter_names)
        elif record.type == "design_choice":
            _ingest_design(record, designs, design_dimensions)
        elif record.type == "method_used":
            name = _method_label(record.value)
            if name:
                method_counts[name] += 1
                method_papers.setdefault(name, []).append(record.paper_id)
        elif record.type == "benchmark_number":
            entry = {
                "paper_id": record.paper_id,
                "raw_text": record.raw_text,
                "value": record.value,
            }
            benchmarks.append(entry)

    for aggregate_ in priors.values():
        _finalize_prior(aggregate_)

    methods = [
        MethodAggregate(method=name, count=count, papers=method_papers[name])
        for name, count in method_counts.most_common()
    ]
    return AggregationResult(
        priors=priors, designs=designs, methods=methods, benchmarks=benchmarks
    )


def _ingest_prior(
    record: EvidenceRecord,
    priors: dict[str, PriorAggregate],
    parameter_names: Sequence[str] | None,
) -> None:
    value = record.value if isinstance(record.value, dict) else {}
    parameter = _coerce_string(value.get("parameter"))
    if not parameter and parameter_names:
        lower_text = record.raw_text.lower()
        for name in parameter_names:
            if name and name.lower() in lower_text:
                parameter = name
                break
    if not parameter:
        return
    distribution = _coerce_string(value.get("distribution")) if record.type == "prior_distribution" else None
    params_raw = value.get("params") if isinstance(value.get("params"), dict) else {}
    params: dict[str, float] = {}
    for key, v in (params_raw or {}).items():
        try:
            params[str(key)] = float(v)
        except (TypeError, ValueError):
            continue
    low = _maybe_float(value.get("low"))
    high = _maybe_float(value.get("high"))
    if record.type == "prior_range" and low is None and high is None:
        low = _maybe_float(value.get("min"))
        high = _maybe_float(value.get("max"))

    aggregate_ = priors.setdefault(parameter, PriorAggregate(parameter=parameter))
    aggregate_.records.append(
        PriorEvidence(
            paper_id=record.paper_id,
            sentence_id=record.sentence_id,
            raw_text=record.raw_text,
            distribution=distribution,
            params=params,
            low=low,
            high=high,
        )
    )


def _ingest_design(
    record: EvidenceRecord,
    designs: dict[str, DesignAggregate],
    design_dimensions: Sequence[str] | None,
) -> None:
    value = record.value if isinstance(record.value, dict) else {}
    dim = _coerce_string(value.get("dimension")) or _coerce_string(value.get("name"))
    if not dim and design_dimensions:
        lower_text = record.raw_text.lower()
        for name in design_dimensions:
            if name.lower() in lower_text:
                dim = name
                break
    if not dim:
        dim = "unspecified"
    aggregate_ = designs.setdefault(dim, DesignAggregate(dimension=dim))
    choice = _coerce_string(value.get("choice") or value.get("value"))
    if choice:
        aggregate_.reported_choices[choice] += 1
    aggregate_.records.append(
        {
            "paper_id": record.paper_id,
            "sentence_id": record.sentence_id,
            "raw_text": record.raw_text,
            "value": value,
        }
    )


def _finalize_prior(aggregate_: PriorAggregate) -> None:
    lows = [r.low for r in aggregate_.records if r.low is not None]
    highs = [r.high for r in aggregate_.records if r.high is not None]
    dists = [r.distribution for r in aggregate_.records if r.distribution]
    aggregate_.reported_distributions = Counter(dists)
    if lows or highs:
        aggregate_.range_union = (
            min(lows) if lows else None,
            max(highs) if highs else None,
        )
    if lows and highs:
        paired = sorted(zip(lows, highs), key=lambda pair: pair[0])
        aggregate_.range_iqr = (_quartile(paired, 0.25, 0), _quartile(paired, 0.75, 1))
    aggregate_.n_sources = len({r.paper_id for r in aggregate_.records})


def _quartile(
    sorted_pairs: list[tuple[float, float]], q: float, index: int
) -> float | None:
    if not sorted_pairs:
        return None
    values = [pair[index] for pair in sorted_pairs]
    if len(values) == 1:
        return float(values[0])
    # Simple quartile via the median-of-halves rule.
    mid = len(values) // 2
    if q < 0.5:
        lower = values[:mid]
        return float(median(lower)) if lower else float(values[0])
    upper = values[mid + 1 :] if len(values) % 2 else values[mid:]
    return float(median(upper)) if upper else float(values[-1])


def _method_label(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("name", "method", "value"):
            if key in value and isinstance(value[key], str):
                return value[key].strip() or None
    return None


def _coerce_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "AggregationResult",
    "DesignAggregate",
    "MethodAggregate",
    "PriorAggregate",
    "PriorEvidence",
    "aggregate",
]
