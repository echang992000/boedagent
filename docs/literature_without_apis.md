# Live literature search without metadata APIs

Module 0 of `boed_agent` normally calls out to Semantic Scholar, arXiv,
OpenAlex, and PubMed for candidate papers. Several deployments can't
do that:

* air-gapped or regulated environments (hospitals, defense, some labs),
* sandboxed runners whose HTTP proxies allow github.com but not
  `api.semanticscholar.org` / `api.openalex.org` / `export.arxiv.org`
  (this is what happened during development in the Cowork sandbox),
* personal laptops without a network connection,
* runs that have already been audited and mustn't make outbound
  requests mid-execution.

The pipeline was built to degrade to a partial report rather than
crash, so "no network" is already a supported mode (`use_literature=
False`, or passing a `SourceBundle()` with no clients). The question
is: if we want Module 0 to still produce *grounded* priors and a real
reasoning trace without hitting any paper metadata API, what are the
realistic options?

This doc enumerates them, ranked by "how much new code" vs. "how good
the output is". Option 1 already ships in this commit. The others are
documented so future contributors have a map.

## Option 1 — `LocalCorpusClient`  (ships in this commit)

A `Source` that scans a user-supplied directory of papers and returns
`Paper` records identical to the HTTP clients' output. Runs entirely
offline, no API key, no network.

**Supported inputs**

- `*.pdf` — text extracted with `pypdf` (same code path as
  `UnpaywallClient.fetch_oa_text`). Page count is capped per file.
- `*.txt` / `*.md` — first non-empty line is the title, the rest is
  the body. Good for hand-curated notes or LLM-summarised abstracts.
- `*.json` sidecars (optional) — DOI, authors, year, venue, citation
  count. Matched by basename.

**Retrieval**

A deterministic TF-IDF-lite scorer (sub-linear TF × log IDF with a
title-hit bonus) ranks corpus entries per query. This is dependency-
free; swap in Whoosh / tantivy / FAISS / pyserini by subclassing and
overriding `_score`.

**Cost model**

Zero tokens in Module 0 before Stage B (the whole point of Stage A is
to avoid paying the LLM for obviously-off-topic papers). With a
local corpus you're trading TF-IDF hits for SS's citation-graph
ranking — coarser, but it's *your* corpus, which is typically already
curated.

**Provenance**

Every `Paper.source` is set to the `source_name` you pass in
(default: `"local_corpus"`). The reasoning trace carries that through
to the citation list, so a reader can tell at a glance that a prior
was grounded in local evidence rather than a public DOI.

**Usage**

```python
from boed_agent.literature.clients import LocalCorpusClient
from boed_agent.literature.search import (
    LiteratureSearchConfig, LiteratureSearchModule, SourceBundle,
)

local = LocalCorpusClient(corpus_dir="~/papers/pharmacokinetics")
bundle = SourceBundle(extra=[("local_pk", local)])

module = LiteratureSearchModule(
    sources=bundle,
    llm=my_llm_client,           # still needs an LLM for Stages B + D
    config=LiteratureSearchConfig(max_papers=10),
)
report = module.search(
    problem_description="One-compartment oral PK with absorption k_a...",
    simulator_metadata=my_sim.metadata,
)
```

For spec-driven BOED chat / CLI flows, the recommended end-to-end path is now to set:

- `use_literature: true`
- `literature_source_mode: "local"` (or `"both"`)
- `literature_corpus_dir: "/path/to/papers"`

and run:

```bash
boed-agent literature-dry-run spec.json
```

That keeps local-corpus retrieval explicit and reproducible while still using the same Stage A-E literature pipeline.

**Limitations**

- Quality is bounded by what's in the directory. A 20-paper local
  corpus won't have the breadth of Semantic Scholar.
- No citation-graph signal, so popular papers don't auto-rank higher.
- PDF text extraction is lossy; equations and tables usually drop.
- Still needs an LLM for Stages B (per-paper extraction) and D
  (per-parameter reasoning). If the LLM itself is gated too, see
  Option 4.

## Option 2 — user-supplied `Paper` list  (already supported)

`LiteratureSearchModule.search(papers=[...])` bypasses the source
fanout entirely. Pass a list of `Paper` objects you constructed
however you like — BibTeX, Zotero export, a DOI list you ran through
your institution's own resolver, a colleague's notes.

**When to use**: highest control, lowest effort. Pair with a
one-time export script so the agent gets the same corpus every run.
This remains useful for tests and custom integrations, but the repo's
default examples now use `LocalCorpusClient` so local files participate
in the same source-bundle path as online retrieval.

## Option 3 — Crossref / Europe PMC  (no key, still needs network)

Neither is in `boed_agent` yet, but both have free, no-key HTTP
endpoints:

- **Crossref** — `https://api.crossref.org/works?query=...` — covers
  almost every journal-published paper, including the ones PubMed
  misses. The `mailto=` parameter gets you into the "polite pool"
  with higher rate limits; treat that like the `email=` we already
  pass to Unpaywall.
- **Europe PMC** — `https://www.ebi.ac.uk/europepmc/webservices/rest/
  search?query=...` — biomedical-leaning but covers preprints too.
  Has an OA full-text endpoint when available, which would feed
  UnpaywallClient's existing PDF path for free.

Implementing either is a copy-paste of `OpenAlexClient` with a
different URL and a thinner JSON parser. Half a day of work, including
tests. Worth doing if the *only* blocker for a given user is "I don't
have a Semantic Scholar key" — Crossref has no such gate.

## Option 4 — LLM web search as the source  (retrieval inside the LLM call)

If the deployment has an LLM with a web-search tool (Anthropic's
`web_search`, OpenAI's browsing, Gemini, Tavily/Perplexity), the LLM
itself can do retrieval. Sketch:

- Wrap the provider's web-search tool in a class that implements the
  existing `Source` contract. Its `search(query, limit)` method sends
  the LLM a prompt like *"return N papers relevant to QUERY as JSON
  with fields title, authors, year, doi, abstract"* and parses the
  response into `Paper` records.
- Token budget is non-trivial and Stage A's prefilter no longer
  protects you — every candidate already cost tokens. Either cap
  `max_papers` much lower than you would with SS, or run the LLM
  retriever first, then have Stage B re-examine only the top K.

**Risks**: hallucinated DOIs. The existing `step_is_grounded`
validator catches ungrounded numerical *claims* but won't notice a
plausible-looking but nonexistent citation. If we do this, we should
add a post-retrieval DOI verification step that either (a) checks
syntactic validity + a known-good resolver cache the user ships with
the deployment, or (b) flags every LLM-retrieved citation with a
`verified=False` marker that propagates into the trace.

## Option 5 — pre-built offline index  (enterprise-scale)

For users who ship their own mirror of a slice of the literature
(e.g., a pharma company's internal PK corpus), we can consume a
pre-indexed directory:

- Whoosh (pure Python, embedded) — one-file installers, no server.
- tantivy-py (Rust backend) — faster, bigger install.
- SQLite FTS5 — zero dependencies, fine for < 50k papers.

All three are drop-in replacements for the `_score` method on
`LocalCorpusClient`. The indexer script ("take a directory of PDFs,
produce a `.whoosh` or `.db` file") is not in scope for this repo,
but the *reader* can be — a `WhooshCorpusClient` would be ~60 lines.

## Decision matrix

|                             | No network | No LLM | Citation quality | New code         |
|-----------------------------|:----------:|:------:|:----------------:|------------------|
| 1. `LocalCorpusClient`      | yes        | needs LLM | as good as corpus | ~280 LoC (this commit) |
| 2. User-supplied `Paper`s   | yes        | needs LLM | as good as input  | 0 (already supported) |
| 3. Crossref / Europe PMC    | no         | needs LLM | excellent         | ~150 LoC per client |
| 4. LLM web search           | no         | needs LLM | medium + hallucination risk | ~200 LoC + verifier |
| 5. Whoosh / tantivy index   | yes        | needs LLM | excellent at scale | ~60 LoC reader + user's indexer |

"Needs LLM" is not a blocker for no-API-key deployments if the LLM
runs locally (Ollama, llama.cpp, vLLM). The `LLMClient` protocol is
provider-neutral — pointing `AnthropicLLMClient` at a local OpenAI-
compatible endpoint is a one-liner in practice.

## Recommendation

For the "some users have no paper API" case, ship Options 1 + 2 in
the default install (both are free, no-network, no-extra-deps beyond
`pypdf`), document Option 3 as the next tier (free but networked),
and treat Options 4 and 5 as specialised add-ons gated behind
explicit user opt-in. This matches the pipeline's existing philosophy
— a partial report is better than no report, and provenance is always
explicit.
