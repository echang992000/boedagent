"""Stage A — cheap local pre-filter (no LLM).

The goal is to drop 40–60 % of papers before any token is spent, using
rule-based pattern matching over abstracts.  The tests cover the
individual detectors so that we can guarantee stable behaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from boed_agent.literature.clients.base import Paper
from boed_agent.simulator_protocol import SimulatorMetadata


# Distribution names that commonly appear in prior-specifying sentences.
_DIST_NAMES = [
    "LogNormal",
    "Log-Normal",
    "Gaussian",
    "Normal",
    "Beta",
    "Gamma",
    "Uniform",
    "Exponential",
    "InverseGamma",
    "HalfNormal",
    "Student",
    "Cauchy",
    "Dirichlet",
    "Poisson",
    "Bernoulli",
]

# Method / acronym terms that mark a paper as methodologically relevant.
_METHOD_TERMS = [
    "BOED",
    "EIG",
    "MINE",
    "MINEBED",
    "iDAD",
    "LFIAX",
    "variational",
    "Bayesian experimental design",
    "optimal design",
    "posterior",
    "mutual information",
]

PRIOR_EVIDENCE_TERMS = [
    "prior",
    "Kd",
    "K_D",
    "dissociation constant",
    "binding affinity",
    "affinity",
    "EC50",
    "half-maximal",
    "Hill coefficient",
    "dose response",
    "pSMAD",
    "BMP4",
    "BMPR",
    "ACVR",
]

# Range patterns like "0.5 ± 0.1", "0.2 +/- 0.05", "1 to 3".
_RANGE_RE = re.compile(
    r"(?P<lo>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*(?:±|\+/-|\+-|--|–|-|to)\s*"
    r"(?P<hi>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


@dataclass
class FilterResult:
    paper: Paper
    keep: bool
    matched_terms: list[str]
    matched_parameters: list[str]
    matched_distributions: list[str]
    matched_ranges: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "paper_id": self.paper.paper_id,
            "keep": self.keep,
            "matched_terms": list(self.matched_terms),
            "matched_parameters": list(self.matched_parameters),
            "matched_distributions": list(self.matched_distributions),
            "matched_ranges": self.matched_ranges,
            "reason": self.reason,
        }


def prefilter(
    papers: Iterable[Paper],
    simulator_metadata: SimulatorMetadata | None = None,
    *,
    min_signals: int = 1,
    include_method_terms: bool = True,
    extra_terms: Iterable[str] = (),
) -> list[FilterResult]:
    """Stage A — return a :class:`FilterResult` per paper.

    Each paper needs at least ``min_signals`` positive signals to be
    kept.  A positive signal is a match on a parameter name, a unit
    string, a distribution name, a method term, or a numeric range
    pattern.  The default ``min_signals=1`` keeps the gate loose —
    tighten it to aggressively prune.
    """
    param_names: list[str] = []
    units: list[str] = []
    if simulator_metadata is not None:
        param_names = [p.name for p in simulator_metadata.parameters if p.name]
        units = [p.units for p in simulator_metadata.parameters if p.units]

    results: list[FilterResult] = []
    for paper in papers:
        haystack = f"{paper.title} \n {paper.abstract}"
        lower = haystack.lower()

        matched_terms = [t for t in extra_terms if t and t.lower() in lower]
        if include_method_terms:
            matched_terms.extend(t for t in _METHOD_TERMS if t.lower() in lower)
        matched_parameters = [n for n in param_names if _contains_token(haystack, n)]
        matched_units = [u for u in units if u and u.lower() in lower]
        matched_distributions = [d for d in _DIST_NAMES if d.lower() in lower]
        matched_ranges = len(list(_RANGE_RE.finditer(haystack)))

        score = (
            len(matched_terms)
            + len(matched_parameters)
            + len(matched_units)
            + len(matched_distributions)
            + (1 if matched_ranges else 0)
        )
        keep = score >= int(min_signals)
        reason = _describe_reason(
            matched_terms,
            matched_parameters,
            matched_units,
            matched_distributions,
            matched_ranges,
        )
        results.append(
            FilterResult(
                paper=paper,
                keep=keep,
                matched_terms=matched_terms,
                matched_parameters=matched_parameters,
                matched_distributions=matched_distributions,
                matched_ranges=matched_ranges,
                reason=reason,
            )
        )
    return results


def _contains_token(text: str, token: str) -> bool:
    if not token:
        return False
    # Tokens like "k_a" should match "k_a", "$k_a$", "k_a," etc. — the safe
    # check is substring-lower, but we guard against accidental matches
    # inside longer identifiers by requiring a non-word boundary on one side.
    needle = token.lower()
    lower = text.lower()
    idx = 0
    while True:
        idx = lower.find(needle, idx)
        if idx < 0:
            return False
        left_ok = idx == 0 or not lower[idx - 1].isalnum()
        right = idx + len(needle)
        right_ok = right == len(lower) or not lower[right].isalnum()
        if left_ok and right_ok:
            return True
        idx += 1


def _describe_reason(
    terms: list[str],
    params: list[str],
    units: list[str],
    dists: list[str],
    ranges: int,
) -> str:
    parts: list[str] = []
    if terms:
        parts.append(f"methods={terms}")
    if params:
        parts.append(f"params={params}")
    if units:
        parts.append(f"units={units}")
    if dists:
        parts.append(f"dists={dists}")
    if ranges:
        parts.append(f"ranges={ranges}")
    return "; ".join(parts) if parts else "no-signal"


__all__ = ["FilterResult", "PRIOR_EVIDENCE_TERMS", "prefilter"]
