# Live Literature-Search Test for `boed_agent`

A standalone script that exercises the full 5-stage literature pipeline against
real APIs (Semantic Scholar + arXiv + PubMed), with Claude as the LLM backend.
The agent runs in `dry_run=True` mode — the *literature search and reasoning*
is what's being tested, not the design optimizer. You do not need Pyro,
MINEBED, iDAD, or LFIAX installed.

## Setup

From the `boed_agent-main` repo root:

```bash
pip install -e ".[literature]"
pip install anthropic
cp test_live_literature_search.py examples/agent/        # or just run in place
```

For live mode:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Running

**Smoke test — no network, no API key, no Pyro.** Deterministic; good for CI.

```bash
python test_live_literature_search.py --offline --task linear
```

Exercises: the dispatcher, the 5-stage pipeline (with empty results because
there's no LLM), the prior builder, the reasoning-trace renderer, and
artifact writing. Should finish in under 2 seconds and produce a
`reasoning_trace_linear_<timestamp>.md` file.

**Live test — Claude + Semantic Scholar + arXiv + PubMed, PK task.**

```bash
python test_live_literature_search.py --task pk --verbose --max-papers 8
```

This:

1. Fetches the real Theophylline dataset from GitHub (cached under
   `~/.cache/boed_agent/theoph.csv`).
2. Issues ~5 queries to Semantic Scholar, arXiv, and PubMed.
3. Runs Stage A (regex filter), Stage B (Claude Haiku on each paper's
   abstract), Stage C (deterministic aggregation), Stage D (Claude Sonnet
   reasoning per parameter + backend choice), Stage E (trace assembly).
4. Prints the full markdown reasoning trace to stdout.
5. Saves `dry_run_pk_<ts>.json` and `reasoning_trace_pk_<ts>.md` to
   `artifacts/lit_search_tests/`.

Expected wall clock: 60–120 seconds. Expected token spend: 15K–50K depending
on how chatty the abstracts are.

**Three problem domains are defined:**

- `--task pk` — one-compartment oral pharmacokinetics (dispatches to Pyro VI
  because the model has a closed-form likelihood)
- `--task sir` — stochastic SIR epidemic (dispatches to MINEBED because it's
  differentiable-implicit)
- `--task linear` — univariate linear regression (dispatches to Pyro VI)

## What to look for

**In the reasoning trace (section 3):** one `## N. prior for <param>`
section per parameter, with an evidence table pulled from Stage C, a
reasoning paragraph from Stage D, a conclusion with a distribution +
params, and a list of cited DOIs / arXiv IDs. If a parameter has thin
evidence (`n_sources < 3`), the step is flagged `_Fallback: insufficient
evidence._` and `fallback=True` — this is *not* an error; it's the
pipeline falling back to a weakly-informative prior as designed.

**In the prior (section 4):** one entry per simulator parameter, with a
distribution family, params, cited papers, and a human-readable reasoning
paragraph. Provenance (user-supplied vs. literature-derived vs. fallback)
is explicit on every entry.

**In the backend choice (section 5):** the waterfall pick plus the
literature ranking. For the PK task, both should agree on PyroVI; for SIR,
the waterfall picks MINEBED and the literature ranking may or may not
concur depending on which papers were retrieved. A `literature_override:
True` would indicate the literature ranking beat the waterfall — rare but
informative when it happens.

**In the cost report (section 6):** tokens per stage. Stage A and C should
be 0 (no LLM). Stage B (extraction) typically dominates. Stage D
(reasoning) is smaller but uses the expensive model. If Stage B tokens
exceed 70% of the budget, consider raising `--max-tokens` or lowering
`--max-papers`.

## Flags worth knowing

| Flag | Purpose |
|---|---|
| `--offline` | Null LLM + no network. Full pipeline, zero API cost. |
| `--no-network` | Claude on, literature sources off. Tests the LLM adapter in isolation. |
| `--no-literature` | Skips Module 0 entirely. Tests only the dispatcher + prior builder + agent plumbing. |
| `--no-data` | Skips the Theophylline fetch. Useful in sandboxed CI. |
| `--verbose` | Prints every LLM call with latency and token counts. |
| `--max-papers N` | Caps the ranked-papers list. Set to 3–5 while iterating. |
| `--max-tokens N` | Hard cap on total LLM spend. Pipeline halts gracefully at the limit. |
| `--cheap-model` / `--reasoning-model` | Override Haiku / Sonnet defaults. |
| `--output-dir` | Where to save artifacts. Default: `artifacts/lit_search_tests/`. |

## Failure modes

- **`ANTHROPIC_API_KEY` missing** — the script refuses to start in live mode.
  Pass `--offline` or `--no-literature` to bypass.
- **`PostSynthesisValidationError`** — the LLM returned a prior value without
  `cited_papers`, tripping the audit-trail validator. The partial report is
  printed; rerun with `--verbose` to see what the model actually returned, or
  switch to `--offline` for a deterministic baseline.
- **HTTP 403 on Theophylline fetch** — some sandboxed environments block the
  GitHub raw-content host. The script tries a second mirror and then falls
  through with a warning; the rest of the pipeline is unaffected.
- **Rate limits** — Semantic Scholar unauthenticated is 100 req / 5 min.
  The client throttles internally at ≥3 s between calls. If you hit a 429,
  wait a few minutes.

## Integration into the package

The script imports the public `boed_agent` API only. To include it in the
repo, drop it into `examples/agent/test_live_literature_search.py` and add a
line to `README.md` under "Examples". No changes to the package source
required.

The `AnthropicLLMClient` defined inside the script is deliberately kept
local (not added to `boed_agent.providers/`) so the example works without
modifying package internals. If it graduates into the package, it belongs
alongside `claude_provider.py` but adapted to implement `LLMClient` instead
of the `LLMProvider` protocol.
