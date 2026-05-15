"""End-to-end test of the ``boed_agent`` literature pipeline with live APIs.

This script exercises the full 5-stage literature pipeline against real
Semantic Scholar, arXiv, and PubMed endpoints, with Claude as the LLM
backend for Stages B and D.  The BOED agent itself runs in ``dry_run``
mode — so you don't need Pyro, MINEBED, iDAD, or LFIAX installed, and
no actual design optimization is performed.  The script is a test of
the **literature search and reasoning pipeline**, not of the optimizer.

What you see when it runs:

  1. DATA SUMMARY        — Theophylline dataset stats (pulled from GitHub)
  2. LITERATURE REPORT   — full reasoning trace (markdown) showing why the
                           agent picked each prior distribution, plus citations
  3. PRIOR CONSTRUCTED   — per-parameter prior with provenance
  4. BACKEND CHOICE      — which BOED backend the dispatcher would pick,
                           and any literature-driven overrides
  5. COST REPORT         — tokens per stage, API calls, paper funnel

Usage
-----

Live mode (requires ANTHROPIC_API_KEY):

    $ python test_live_literature_search.py

Offline mode (no network, no API key needed — uses NullLLMClient):

    $ python test_live_literature_search.py --offline

Knob down token spend while iterating:

    $ python test_live_literature_search.py --max-papers 5 --max-tokens 15000

Skip the data-loading step entirely (pure literature test):

    $ python test_live_literature_search.py --no-data

Pick a specific problem domain instead of PK:

    $ python test_live_literature_search.py --task sir
    $ python test_live_literature_search.py --task linear
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# --- boed_agent imports ---------------------------------------------------

from boed_agent import (
    BOEDAgent,
    ParameterInfo,
    PostSynthesisValidationError,
    SimpleSimulator,
    SimulatorMetadata,
    TokenBudget,
)
from boed_agent.literature.clients.arxiv import ArxivClient
from boed_agent.literature.clients.base import ClientConfig
from boed_agent.literature.clients.pubmed import PubMedClient
from boed_agent.literature.clients.semantic_scholar import SemanticScholarClient
from boed_agent.literature.llm_client import (
    LLMClient,
    LLMResponse,
    ModelTier,
    NullLLMClient,
)
from boed_agent.literature.search import (
    LiteratureSearchConfig,
    LiteratureSearchModule,
    SourceBundle,
)


# =============================================================================
# Task definitions
# =============================================================================

@dataclass
class Task:
    name: str
    description: str                      # feeds the literature search
    simulator_fn: Callable                # plain callable for SimpleSimulator
    metadata: SimulatorMetadata
    design_distribution: dict
    is_explicit: bool
    is_differentiable: bool


def _pk_simulator(theta, xi):
    """One-compartment oral PK — used for the Theophylline-style task."""
    import math

    k_a, k_e, V = theta
    t = xi[0] if hasattr(xi, "__iter__") else xi
    if abs(k_a - k_e) < 1e-9:
        k_e = k_a + 1e-6
    return (k_a / (V * (k_a - k_e))) * (math.exp(-k_e * t) - math.exp(-k_a * t))


def _sir_simulator(theta, xi):
    """Placeholder — never actually called in dry-run mode."""
    beta, gamma = theta
    t = xi[0] if hasattr(xi, "__iter__") else xi
    return float(beta * t - gamma * t)  # noqa: E501 — purely a structural stub


def _linear_simulator(theta, xi):
    alpha, beta = theta
    x = xi[0] if hasattr(xi, "__iter__") else xi
    return float(alpha * x + beta)


TASKS: dict[str, Task] = {
    "pk": Task(
        name="pk",
        description=(
            "One-compartment oral pharmacokinetic model. Parameters: absorption "
            "rate k_a (1/hr), elimination rate k_e (1/hr), apparent volume of "
            "distribution V (L/kg). Design variable: blood sample time in [0, 24] "
            "hours after dose. Goal: select sampling times that maximize expected "
            "information gain on (k_a, k_e, V) for a theophylline-like drug."
        ),
        simulator_fn=_pk_simulator,
        metadata=SimulatorMetadata(
            parameters=[
                ParameterInfo(name="k_a", units="1/hr", description="absorption rate"),
                ParameterInfo(name="k_e", units="1/hr", description="elimination rate"),
                ParameterInfo(name="V", units="L/kg", description="apparent volume of distribution"),
            ],
            observation_labels=["concentration"],
            domain_tags=["pharmacokinetics", "one_compartment", "oral_absorption"],
        ),
        design_distribution={"t": {"lower": 0.0, "upper": 24.0}},
        is_explicit=True,
        is_differentiable=True,
    ),
    "sir": Task(
        name="sir",
        description=(
            "Stochastic SIR epidemic model with infection rate beta and recovery "
            "rate gamma. Population N=1000, initial infected I0=10. Design "
            "variable: observation day in [1, 30] at which cumulative confirmed "
            "infections are measured. Goal: select observation days that maximize "
            "expected information gain on (beta, gamma) — equivalently, on R_0 = "
            "beta / gamma. Relevant to COVID-style early outbreak surveillance."
        ),
        simulator_fn=_sir_simulator,
        metadata=SimulatorMetadata(
            parameters=[
                ParameterInfo(name="beta", units="1/day", description="infection rate"),
                ParameterInfo(name="gamma", units="1/day", description="recovery rate"),
            ],
            observation_labels=["cumulative_infected"],
            domain_tags=["epidemiology", "sir", "compartmental", "covid"],
        ),
        design_distribution={"t": {"lower": 1.0, "upper": 30.0}},
        is_explicit=False,
        is_differentiable=True,
    ),
    "linear": Task(
        name="linear",
        description=(
            "Univariate Bayesian linear regression y = alpha * xi + beta + noise "
            "with Gaussian noise. Parameters: slope alpha, intercept beta. "
            "Design: xi in [-3, 3]. Goal: pick xi to maximize expected information "
            "gain on (alpha, beta)."
        ),
        simulator_fn=_linear_simulator,
        metadata=SimulatorMetadata(
            parameters=[
                ParameterInfo(name="alpha", description="slope"),
                ParameterInfo(name="beta", description="intercept"),
            ],
            observation_labels=["y"],
            domain_tags=["linear_regression", "conjugate_bayesian"],
        ),
        design_distribution={"xi": {"lower": -3.0, "upper": 3.0}},
        is_explicit=True,
        is_differentiable=True,
    ),
}


# =============================================================================
# Anthropic LLM client — implements the boed_agent LLMClient protocol
# =============================================================================

@dataclass
class AnthropicLLMClient:
    """LLMClient adapter that routes to Anthropic's Messages API.

    Implements the :class:`boed_agent.literature.llm_client.LLMClient`
    protocol: a single ``extract(prompt, *, model_tier, stage, budget)``
    method returning an :class:`LLMResponse`.

    Failures are non-fatal — on rate limit / API error / parse failure
    the client returns an empty ``LLMResponse(text="{}")`` and logs the
    reason, letting the pipeline fall back to weakly-informative priors
    rather than crashing. This mirrors the pipeline's own design
    principle that a partial report is strictly better than a missing
    one.
    """

    cheap_model: str = "claude-haiku-4-5"
    reasoning_model: str = "claude-sonnet-4-6"
    max_tokens_cheap: int = 2048
    max_tokens_reasoning: int = 4096
    api_key: str | None = None
    cache: dict[str, LLMResponse] = field(default_factory=dict)
    verbose: bool = False

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export the variable or pass "
                "--offline to use the NullLLMClient instead."
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "`anthropic` package not found. `pip install anthropic`."
            ) from exc
        self._client = Anthropic(api_key=self.api_key)

    def _key(self, prompt: str, tier: ModelTier) -> str:
        return tier + ":" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _model_for(self, tier: ModelTier) -> tuple[str, int]:
        if tier == "reasoning":
            return self.reasoning_model, self.max_tokens_reasoning
        return self.cheap_model, self.max_tokens_cheap

    def extract(
        self,
        prompt: str,
        *,
        model_tier: ModelTier = "cheap",
        stage: str = "unknown",
        budget: TokenBudget | None = None,
    ) -> LLMResponse:
        key = self._key(prompt, model_tier)
        if key in self.cache:
            cached = self.cache[key]
            if budget is not None:
                budget.record(stage, 0)
            if self.verbose:
                print(f"  [cache hit] stage={stage} tier={model_tier}")
            return LLMResponse(
                text=cached.text,
                input_tokens=0,
                output_tokens=0,
                model=cached.model,
                cached=True,
            )

        model, max_tokens = self._model_for(model_tier)
        t0 = time.monotonic()
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            # Graceful degradation — don't crash the pipeline.
            if self.verbose:
                print(
                    f"  [api error] stage={stage} tier={model_tier}: "
                    f"{type(exc).__name__}: {exc}"
                )
            return LLMResponse(text="{}", input_tokens=0, output_tokens=0, model=model)

        text_parts = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        text = "".join(text_parts).strip()

        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", None) or max(1, len(prompt) // 4)
        out_tok = getattr(usage, "output_tokens", None) or max(1, len(text) // 4)

        resp = LLMResponse(
            text=text,
            input_tokens=int(in_tok),
            output_tokens=int(out_tok),
            model=model,
        )
        self.cache[key] = resp
        if budget is not None:
            budget.record(stage, int(in_tok) + int(out_tok))
        if self.verbose:
            elapsed = time.monotonic() - t0
            print(
                f"  [api call] stage={stage} tier={model_tier} model={model} "
                f"in={in_tok} out={out_tok} elapsed={elapsed:.1f}s"
            )
        return resp


# =============================================================================
# Optional: Theophylline data loader
# =============================================================================

THEOPH_URLS = (
    # Primary mirror — raw GitHub.
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
    "master/csv/datasets/Theoph.csv",
    # Secondary mirror — GitHub Pages (different host, tends to work
    # when raw.githubusercontent.com returns 403 from locked-down nets).
    "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Theoph.csv",
)


def _load_theoph() -> dict | None:
    """Fetch the Theophylline dataset; cache under ~/.cache/boed_agent.

    Tries each URL in :data:`THEOPH_URLS` in order.  Returns None on
    total failure — the rest of the script still runs.
    """
    cache_dir = Path.home() / ".cache" / "boed_agent"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "theoph.csv"

    if not cache_path.exists():
        last_err: Exception | None = None
        for url in THEOPH_URLS:
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (boed-agent-test/0.1; "
                            "+https://github.com/anthropics/boed-agent)"
                        ),
                        "Accept": "text/csv,text/plain,*/*",
                    },
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    cache_path.write_bytes(resp.read())
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                last_err = exc
                continue
        else:
            print(f"  (could not fetch Theophylline data from any mirror: {last_err})")
            return None

    try:
        lines = cache_path.read_text().strip().splitlines()
        header = lines[0].split(",")
        rows = [line.split(",") for line in lines[1:] if line.strip()]
    except Exception as exc:
        print(f"  (could not parse Theophylline data: {exc})")
        return None

    # Column layout: ,Subject,Wt,Dose,Time,conc
    try:
        subj_idx = header.index('"Subject"')
        time_idx = header.index('"Time"')
        conc_idx = header.index('"conc"')
    except ValueError:
        # Header might be unquoted depending on the CSV dialect
        subj_idx = next(i for i, h in enumerate(header) if "Subject" in h)
        time_idx = next(i for i, h in enumerate(header) if "Time" in h)
        conc_idx = next(i for i, h in enumerate(header) if "conc" in h)

    def _num(s):
        try:
            return float(s.strip('"'))
        except ValueError:
            return None

    times, concs, subjects = [], [], []
    for r in rows:
        t = _num(r[time_idx])
        c = _num(r[conc_idx])
        s = r[subj_idx].strip('"')
        if t is None or c is None:
            continue
        times.append(t)
        concs.append(c)
        subjects.append(s)

    return {
        "n_observations": len(times),
        "n_subjects": len(set(subjects)),
        "time_range": (min(times), max(times)),
        "conc_range": (min(concs), max(concs)),
        "source_url": THEOPH_URLS[0],
        "cache_path": str(cache_path),
    }


# =============================================================================
# Pretty printing helpers
# =============================================================================

def _section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def _subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def _indent(s: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in s.splitlines())


# =============================================================================
# Main
# =============================================================================

def build_agent(args: argparse.Namespace, task: Task) -> tuple[BOEDAgent, TokenBudget]:
    simulator = SimpleSimulator(
        fn=task.simulator_fn,
        metadata=task.metadata,
        is_explicit=task.is_explicit,
        is_differentiable=task.is_differentiable,
        name=f"{task.name}_sim",
    )

    budget = TokenBudget(max_total_tokens=args.max_tokens, raise_on_exceed=False)

    # Build the LLM client.  Skip the Anthropic adapter entirely when
    # literature is disabled or we're in offline mode — no point
    # requiring an API key for a run that won't make any LLM calls.
    if args.offline or args.no_literature:
        llm: LLMClient = NullLLMClient()
    else:
        llm = AnthropicLLMClient(
            cheap_model=args.cheap_model,
            reasoning_model=args.reasoning_model,
            verbose=args.verbose,
        )

    # Build the source bundle
    if args.offline or args.no_network:
        sources = SourceBundle()  # empty → no network calls
    else:
        cfg = ClientConfig(timeout_seconds=20.0)
        sources = SourceBundle(
            semantic_scholar=SemanticScholarClient(config=cfg),
            arxiv=ArxivClient(config=cfg),
            pubmed=PubMedClient(config=cfg) if task.name == "pk" else None,
        )

    lit_config = LiteratureSearchConfig(
        max_papers=args.max_papers, current_year=2026
    )
    lit_module = LiteratureSearchModule(
        sources=sources, llm=llm, token_budget=budget, config=lit_config
    )

    agent = BOEDAgent(
        simulator=simulator,
        design_distribution=task.design_distribution,
        problem_description=task.description,
        prior=None,  # let the literature search construct the prior
        use_literature=not args.no_literature,
        token_budget=budget,
        literature_module=lit_module,
        backend_options={},
    )
    return agent, budget


def run(args: argparse.Namespace) -> int:
    task = TASKS[args.task]

    _section(f"boed_agent live literature-search test — task: {task.name}")
    mode = "OFFLINE" if args.offline else "LIVE"
    print(f"  mode         : {mode}")
    print(f"  task         : {task.name}")
    print(f"  max papers   : {args.max_papers}")
    print(f"  max tokens   : {args.max_tokens}")
    print(f"  cheap model  : {args.cheap_model}")
    print(f"  reasoning    : {args.reasoning_model}")
    print(f"  use literature: {not args.no_literature}")

    # ------------------------------------------------------------------
    # 1. Optional data summary
    # ------------------------------------------------------------------
    if task.name == "pk" and not args.no_data:
        _section("1. DATA SUMMARY — Theophylline")
        summary = _load_theoph()
        if summary:
            print(f"  subjects      : {summary['n_subjects']}")
            print(f"  observations  : {summary['n_observations']}")
            print(f"  time range    : {summary['time_range']} hr")
            print(f"  conc range    : {summary['conc_range']} mg/L")
            print(f"  source        : {summary['source_url']}")
        else:
            print("  (data load skipped)")

    # ------------------------------------------------------------------
    # 2. Build and run the agent (dry-run)
    # ------------------------------------------------------------------
    _section("2. LITERATURE SEARCH + REASONING")
    agent, budget = build_agent(args, task)

    t0 = time.monotonic()
    try:
        result = agent.run(dry_run=True)
    except PostSynthesisValidationError as exc:
        print(
            "\n  !! PostSynthesisValidationError — literature report contains "
            "ungrounded numerical claims. This typically means the LLM "
            "returned prior values without cited_papers. Partial report:\n"
        )
        print(f"  {exc}")
        print(
            "\n  Re-run with --verbose to see raw LLM outputs, or with "
            "--offline for a deterministic fallback run."
        )
        return 2
    except Exception as exc:
        print(f"\n  !! Unexpected error: {type(exc).__name__}: {exc}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 3

    elapsed = time.monotonic() - t0
    print(f"  agent.run(dry_run=True) completed in {elapsed:.1f}s")

    if result.literature_report is None:
        print("\n  literature module was disabled (use_literature=False)")
        _section("3. BACKEND CHOICE")
        print(f"  chosen backend : {result.chosen_backend}")
        print(f"  reason         : {result.backend_choice.reason}")
        return 0

    # ------------------------------------------------------------------
    # 3. Reasoning trace
    # ------------------------------------------------------------------
    _section("3. REASONING TRACE (markdown)")
    md = result.reasoning_trace.to_markdown()
    print(_indent(md or "(empty trace)", 2))

    # ------------------------------------------------------------------
    # 4. Prior built
    # ------------------------------------------------------------------
    _section("4. PRIOR CONSTRUCTED")
    prior_dict = result.prior_used.to_dict()
    for name, spec in prior_dict.get("distributions", {}).items():
        print(f"  {name}:")
        print(f"    distribution : {spec.get('name')}")
        print(f"    params       : {spec.get('params')}")
        print(f"    source       : {spec.get('source')}")
        print(f"    fallback     : {spec.get('fallback')}")
        citations = spec.get("cited_papers", [])
        if citations:
            print(f"    cited papers : {citations[:3]}"
                  + ("..." if len(citations) > 3 else ""))
        reasoning = (spec.get("reasoning") or "").strip()
        if reasoning:
            wrapped = textwrap.fill(
                reasoning, width=72, initial_indent="    ",
                subsequent_indent="    ",
            )
            print(f"    reasoning    :\n{wrapped}")
        print()
    if prior_dict.get("warnings"):
        print("  Warnings:")
        for w in prior_dict["warnings"]:
            print(f"    - {w}")

    # ------------------------------------------------------------------
    # 5. Backend choice
    # ------------------------------------------------------------------
    _section("5. BACKEND CHOICE")
    bc = result.backend_choice
    print(f"  chosen backend     : {bc.backend.name}")
    print(f"  reason             : {bc.reason}")
    print(f"  literature override: {bc.literature_override}")
    if bc.alternatives:
        print(f"  alternatives       : {bc.alternatives}")
    if bc.cited_papers:
        print(f"  cited papers       : {bc.cited_papers[:3]}"
              + ("..." if len(bc.cited_papers) > 3 else ""))
    if bc.notes:
        for n in bc.notes:
            print(f"  note               : {n}")

    lit_rank = result.literature_report.backend_preference.ranked
    if lit_rank:
        print(f"  lit ranking        : {lit_rank}")
        br = result.literature_report.backend_preference
        if br.cited_papers:
            print(f"  rank citations     : {br.cited_papers[:3]}")

    # ------------------------------------------------------------------
    # 6. Cost report
    # ------------------------------------------------------------------
    _section("6. COST REPORT")
    cost = result.literature_report.cost_report.to_dict()
    print(f"  total tokens    : {cost['total_tokens']}")
    print(f"  api calls       : {cost['api_calls']}")
    print(f"  papers considered: {cost['papers_considered']}")
    print(f"  papers filtered  : {cost['papers_filtered']}")
    print(f"  papers processed : {cost['papers_processed']}")
    print(f"  tokens by stage :")
    for stage, n in sorted(cost["tokens_by_stage"].items()):
        print(f"    {stage:>12s}: {n}")

    # ------------------------------------------------------------------
    # 7. Save artifacts
    # ------------------------------------------------------------------
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        dry_path = out_dir / f"dry_run_{task.name}_{ts}.json"
        md_path = out_dir / f"reasoning_trace_{task.name}_{ts}.md"
        dry_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        md_path.write_text(md or "")
        print(f"\n  artifacts saved to: {out_dir}")
        print(f"    - {dry_path.name}")
        print(f"    - {md_path.name}")

    print("\n  done.")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Test boed_agent with a live literature search. Exercises the full "
            "5-stage pipeline (filter → extract → aggregate → reason → trace) "
            "and prints the resulting prior, backend choice, and cost report."
        ),
    )
    p.add_argument(
        "--task", choices=sorted(TASKS.keys()), default="pk",
        help="which problem description to search the literature for",
    )
    p.add_argument(
        "--offline", action="store_true",
        help="use NullLLMClient and no network — deterministic smoke test",
    )
    p.add_argument(
        "--no-network", action="store_true",
        help="disable literature source clients but keep Claude LLM calls "
        "(useful for testing the LLM adapter in isolation)",
    )
    p.add_argument(
        "--no-literature", action="store_true",
        help="bypass the literature pipeline entirely (use_literature=False)",
    )
    p.add_argument(
        "--no-data", action="store_true",
        help="skip the Theophylline data fetch",
    )
    p.add_argument(
        "--max-papers", type=int, default=10,
        help="cap papers retained after ranking (default: 10)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=60_000,
        help="total LLM token budget (default: 60K)",
    )
    p.add_argument(
        "--cheap-model", default="claude-haiku-4-5",
        help="Anthropic model for Stage B (extraction). Default: claude-haiku-4-5",
    )
    p.add_argument(
        "--reasoning-model", default="claude-sonnet-4-6",
        help="Anthropic model for Stage D (reasoning). Default: claude-sonnet-4-6",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="print each LLM call as it happens",
    )
    p.add_argument(
        "--output-dir", default="artifacts/lit_search_tests",
        help="directory to save reasoning trace and dry-run JSON",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(_parse_args()))
