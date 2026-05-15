# Literature Search Test Plan

Target: `boed_agent` literature pipeline (Module 0, Stages A–E), as exercised by
`examples/agent/test_live_literature_search.py` and `examples/agent/lit_only_dry_run.py`.

## 1. Problem statement

Two failure modes are known to degrade or silence real paper retrieval:

- **F1. Sandboxed network** — the runtime blocks `api.semanticscholar.org`,
  `export.arxiv.org`, `eutils.ncbi.nlm.nih.gov`, `api.biorxiv.org`, etc. Result:
  the pipeline produces a "partial report" with an empty paper list, the
  reasoning trace flags every parameter as `fallback=True`, and the prior
  collapses to weakly-informative defaults. The system does *not* crash, so
  the fault is easy to miss in CI.
- **F2. No LLM API key** — `ANTHROPIC_API_KEY` (or OpenAI equivalent) is not
  set. Today, `AnthropicLLMClient` raises `RuntimeError` at construction, and
  the only workaround is `--offline` (which also disables the paper sources).
  This means we can't currently exercise arXiv/bioRxiv retrieval in isolation
  without paying for an LLM — a gap worth closing.

These two modes are distinct and should be tested separately, because the
mitigations are different (local corpus / alternate free APIs for F1;
null-LLM-with-live-sources for F2).

## 2. Scope and non-goals

In scope: `LiteratureSearchModule.search(...)` and its callers, the five
stages (A filter, B extract, C aggregate, D reason, E trace), the
`Source` clients (`SemanticScholarClient`, `ArxivClient`, `PubMedClient`,
`LocalCorpusClient`, plus any new `CrossrefClient` / `EuropePMCClient` /
`BiorxivClient` added as part of F1 mitigation), and the cost/provenance
reporting.

Out of scope: the BOED inference backends (Pyro VI, MINEBED, iDAD, LFIAX).
All runs use `agent.run(dry_run=True)`.

## 3. Test matrix

Rows are environment configurations; columns are the questions each row
must answer. "Pass" criteria are defined in §5.

| #  | Mode label            | LLM key | Network to paper APIs | Local corpus | Tests                                                             |
|----|-----------------------|:-------:|:---------------------:|:------------:|-------------------------------------------------------------------|
| M1 | Full live (baseline)  | yes     | yes                   | no           | End-to-end smoke; real DOIs; provenance carries through           |
| M2 | Sandbox — LLM only    | yes     | no (APIs blocked)     | no           | Pipeline reports `fallback=True`, does not crash, cost==0 Stage B |
| M3 | Sandbox + local corpus| yes     | no                    | yes          | Papers come from local corpus; citations tagged `local_*`         |
| M4 | No LLM, live sources  | no      | yes                   | no           | arXiv/bioRxiv still fetch; Stage B/D skipped cleanly; trace notes why |
| M5 | No LLM, local corpus  | no      | no                    | yes          | Fully offline smoke; deterministic; CI-friendly                   |
| M6 | No LLM, no sources    | no      | no                    | no           | Graceful degradation to weakly-informative prior                  |
| M7 | Hallucination guard   | yes     | no                    | yes, poisoned| LLM-proposed DOIs not in corpus flagged `verified=False`          |
| M8 | Rate-limit / 429      | yes     | yes (throttled)       | no           | Retry/backoff works; pipeline completes without tokens wasted     |

M4 and M5 are the cases the current codebase does not cleanly support; they
will require a small change to allow `NullLLMClient` while retaining live
source clients. That change is part of this plan (§6.2).

## 4. Test assets to build

### 4.1 Seed local corpora (for M3, M5, M7)

Three task-specific directories under `examples/agent/local_corpus/`. Two
already exist (`lit_only/`, `pharmacokinetic_ode/`, `bmp4_gradient/`). For
the SIR task — which is what this repo is staged around — add:

```
examples/agent/local_corpus/sir/
├── britton_2010_sir_priors.md          # hand-authored summary of a real paper
├── kiss_miller_simon_2017.md           # ditto
├── keeling_rohani_2008_ch3.md          # ditto
└── poisoned_for_M7.md                  # contains a fake "DOI: 10.9999/xyz"
```

Each `.md` file's first line is the title; the body contains 3–10 sentences
with explicit prior-relevant numeric claims ("β ≈ 0.3 day⁻¹ for influenza-
like illness", etc.) so Stage B has something to extract. The `.pdf` variant
should also be tested for at least one file to exercise the pypdf path.

### 4.2 Mocked LLM responder (for deterministic M3/M5)

A `RecordingLLMClient` responder keyed on prompt substring — already
demonstrated in `lit_only_dry_run.py`. Extend it to cover the SIR
parameters (β, γ) and save responder fixtures under
`tests/fixtures/llm_responses/sir_*.json` so the same canned answers are
reused across CI runs.

### 4.3 Network-denial fixture (for M2, M3, M5, M6)

A pytest fixture that monkeypatches `urllib.request.urlopen` (and, if the
clients use `requests`, `requests.adapters.HTTPAdapter.send`) to raise
`urllib.error.URLError("blocked")`. This simulates the sandbox without
depending on real firewall state. Place in `tests/conftest.py` as
`network_blocked`.

### 4.4 Rate-limit fixture (for M8)

A fixture that returns HTTP 429 on the first call per host and 200 on
subsequent calls, verifying the client's `tenacity` retry/backoff wrapper.

## 5. Per-mode pass criteria

Every mode must satisfy three universal invariants, then adds its own:

**Universal invariants** (assert in every test)

- `agent.run(dry_run=True)` returns a `DryRunResult` without raising.
- `result.literature_report.cost_report.to_dict()["total_tokens"]` is a
  non-negative int matching the mode (0 for null-LLM modes).
- `result.reasoning_trace.to_markdown()` is non-empty and parses as
  markdown (no raw Python reprs leak through).
- `result.prior_used.to_dict()` has exactly one entry per simulator
  parameter with `distribution`, `params`, `source`, `cited_papers`,
  `fallback` fields populated.

**Per-mode**

- **M1**: ≥1 paper per parameter in `evidence`, each with a resolvable
  DOI or arXiv ID. At least one parameter's prior is *not* flagged
  `fallback`. Total wall time < 180 s. Token spend < 60 000.
- **M2**: Every parameter flagged `fallback=True`. `cost_report` shows
  Stage A > 0 (regex filter ran on the empty set), Stages B/D == 0
  tokens. No unhandled exceptions in the log. Trace contains the
  string "no candidate papers" or equivalent user-facing explanation.
- **M3**: At least one paper per parameter has `source == "local_sir"`
  (or whichever `source_name` was passed). No `fallback=True` on the
  parameters that had ≥3 matching corpus sentences. No network calls
  (verified via the `network_blocked` fixture).
- **M4**: Real papers present in `evidence` (source == "arxiv" /
  "semantic_scholar"), but Stages B and D report 0 LLM calls. The
  trace says something like "LLM unavailable — papers listed but
  not summarised" and the prior falls back. This is the happy path
  for users who want citations without paying for an LLM.
- **M5**: Fully deterministic. Two back-to-back runs produce
  byte-identical `reasoning_trace_*.md` (modulo timestamps, which
  should be injected via a frozen clock fixture). Runtime < 2 s.
- **M6**: `evidence` is empty for every parameter. `fallback=True`
  everywhere. Prior matches the documented default family per
  parameter type. No crash.
- **M7**: The poisoned `local_corpus/sir/poisoned_for_M7.md` triggers
  a `verified=False` marker on the citation in the trace, *and* the
  prior for that parameter is not grounded on the unverified DOI
  alone. (Requires implementing the post-retrieval verifier
  sketched in `docs/literature_without_apis.md` §Option 4.)
- **M8**: Client eventually succeeds. Retry count observable via the
  cost report or a new `api_retries` counter. No double-counted
  tokens.

## 6. Implementation plan

### 6.1 Wire existing tests into `pytest`

Today the two scripts (`test_live_literature_search.py`,
`lit_only_dry_run.py`) are executable examples, not pytest targets. Add:

- `tests/literature/test_modes.py` — one `test_m*` function per mode,
  parametrised on task ∈ {`sir`, `pk`, `linear`}.
- Mark M1 as `@pytest.mark.live` (skipped unless
  `RUN_LIVE_TESTS=1` and `ANTHROPIC_API_KEY` is set).
- Mark M4 as `@pytest.mark.network` (requires outbound HTTP but no key).
- Everything else runs by default.

### 6.2 Allow `NullLLMClient` + live sources (fixes M4)

Currently `build_agent` in `test_live_literature_search.py` couples
`--offline` to both the LLM *and* the source bundle. Split the flags so
the user can pick `--no-llm` independently of `--no-network`. Change in
the test script only; the package already supports this combination —
`LiteratureSearchModule` accepts `llm=NullLLMClient()` with a populated
`SourceBundle`.

Verify that Stage B (extraction) exits early when `llm` is a
`NullLLMClient` instead of sending an empty prompt; if it doesn't, patch
`LiteratureSearchModule._extract_per_paper` to short-circuit.

### 6.3 Add free-tier source clients (raises M2→M4 coverage)

Per `docs/literature_without_apis.md` §Option 3, two new clients unblock
sandboxed environments whose proxies allow *some* hosts:

- `CrossrefClient` — `https://api.crossref.org/works?query=...&mailto=...`
- `EuropePMCClient` — `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=...`
- `BiorxivClient` — `https://api.biorxiv.org/details/biorxiv/<DOI>`
  (or the collection endpoint for keyword search).

Copy-paste from `OpenAlexClient`; add one unit test per client that hits a
recorded VCR cassette, and one integration test behind
`@pytest.mark.network`. The user's stated case "sources such as
arxiv/biorxiv need to be accessed" is covered once `BiorxivClient` lands.

### 6.4 Citation verifier (closes M7)

A lightweight resolver that accepts a `cited_papers` list from Stage D and
cross-checks each entry against (a) the list of `Paper` records that went
into Stage B (must be a subset) and (b) an optional DOI syntactic check.
Entries failing (a) get `verified=False` and the prior's `fallback` flag
is set if *all* its citations are unverified. Lives in
`boed_agent/literature/verification.py`, ~80 LoC.

### 6.5 Network-probe preflight (UX improvement)

At `LiteratureSearchModule.__init__`, optionally probe each source client
with a 2-second HEAD request; if *all* probes fail, log a single warning
once ("sandboxed environment detected; consider `LocalCorpusClient`") and
set an internal flag so subsequent per-query calls short-circuit. Keeps
the trace readable.

## 7. Run commands (human-friendly)

From `examples/agent/`:

```bash
# M1 — full live (needs ANTHROPIC_API_KEY, needs network)
python test_live_literature_search.py --task sir --verbose --max-papers 5

# M2 — sandboxed with LLM (key set, network off)
# Use the network_blocked fixture in pytest, or in practice:
python test_live_literature_search.py --task sir --no-network --verbose

# M3 — sandboxed with LLM + local corpus
python test_live_literature_search.py --task sir --no-network \
    --local-corpus ./local_corpus/sir --verbose
# (needs --local-corpus flag added to the script; see 6.2)

# M4 — no LLM, live sources
python test_live_literature_search.py --task sir --no-llm --verbose
# (needs --no-llm flag; see 6.2)

# M5 — fully offline smoke
python lit_only_dry_run.py

# Under pytest
pytest tests/literature/                   # M2, M3, M5, M6, M7 by default
RUN_LIVE_TESTS=1 pytest tests/literature/  # adds M1
pytest -m network tests/literature/        # adds M4, M8
```

## 8. Acceptance gates for this plan

The plan is "done" when:

1. All eight modes have a corresponding pytest test that passes locally.
2. CI runs M2/M3/M5/M6/M7 on every PR; M1/M4/M8 run nightly.
3. `docs/literature_without_apis.md` is updated with the three new
   free-tier clients (§6.3) and the verifier (§6.4).
4. A short entry is added to `README.md` under "Running tests" pointing
   to `LITERATURE_TEST_PLAN.md`.
5. For a new task (e.g., `sir`), the repo contains a seed
   `local_corpus/sir/` directory that makes M3 green without any
   network access.

## 9. Open questions

- Should the verifier (§6.4) be a hard fail or a soft flag by default?
  Hard fail catches more bugs in CI but punishes the "LLM cited a DOI
  we retrieved but the verifier's cache is stale" case.
- For M4, what should the reasoning trace say when it has papers but
  no LLM summaries? Options: (a) list titles only, (b) run a
  regex-based sentence extractor as a no-LLM Stage B, (c) emit a
  machine-readable "needs_llm" stub so a downstream tool can finish
  the job later. Recommend (b) as the shipping default, (c) as a
  flag for audit-heavy deployments.
- bioRxiv's search API returns only DOIs/metadata, not abstracts. Do
  we want a separate `BiorxivFulltextClient` that goes through the
  OA link, or fold it into `UnpaywallClient`? Probably the latter —
  one full-text path, many metadata paths.
