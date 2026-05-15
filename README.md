# BOED Agent

`boed-agent` is a Python-first backbone for building an AI agent that helps users set up and run Bayesian optimal experimental design workflows across multiple backends.

The implementation now includes two layers:

1. A **literature-informed orchestrator** (`BOEDAgent`) that, given a `(prior, simulator, design_distribution, data, problem_description)` tuple, queries the scientific literature, synthesises priors and design hints via a token-efficient chain-of-thought pipeline, and dispatches to the appropriate backend.
2. The original **provider-neutral agent core + CLI** that drives BOED runs from JSON specs, including the OpenAI Responses API, Claude Messages API, and OpenAI Agents SDK manager-and-specialist runtimes.

Capabilities:

- `BOEDAgent` orchestrator with literature-informed reasoning
- `LiteratureSearchModule` with the 5-stage extraction pipeline (filter → extract → aggregate → reason → trace)
- API clients for Semantic Scholar, arXiv, OpenAlex, PubMed, and Unpaywall
- `PriorBuilder` that augments user priors without ever silently overriding them
- `DataClassifier` with raw and simulator-aware homogeneity checks
- `SimulatorChoiceModule` waterfall dispatcher (explicit → Pyro, differentiable → MINEBED / iDAD, else → LFIAX)
- live Pyro backend for variational OED estimators
- live `lfiax` simulator backend bridged through `cli-anything-lfiax`
- MINEBED and iDAD backend stubs with lazy imports
- a clarification planner that asks for missing BOED inputs before execution
- a thin CLI for validation, execution, backend inspection, and interactive chat

## High-level agent API

```python
from boed_agent import BOEDAgent, SimpleSimulator, SimulatorMetadata, ParameterInfo

simulator = SimpleSimulator(
    fn=lambda theta, xi: theta[0] * xi + theta[1],
    metadata=SimulatorMetadata(
        parameters=[ParameterInfo(name="slope"), ParameterInfo(name="intercept")],
        domain_tags=["toy"],
    ),
    is_explicit=True,
    is_differentiable=True,
)

agent = BOEDAgent(
    simulator=simulator,
    design_distribution={"xi": {"lower": -1.0, "upper": 1.0}},
    problem_description="1-D linear regression",
    prior=None,
    use_literature=False,  # or True with a real LLMClient
)
result = agent.run(dry_run=True)
print(result.chosen_backend)
```

Four end-to-end example scripts live under `examples/agent/`:

- `explicit_linear_regression.py` — explicit simulator → Pyro VI
- `pharmacokinetic_ode.py` — differentiable implicit simulator → iDAD / MINEBED with literature-derived priors on `(k_a, k_e, V)` from a local corpus directory
- `black_box_agent_sim.py` — non-differentiable agent-based simulator → LFIAX
- `lit_only_dry_run.py` — returns `LiteratureReport` + `ReasoningTrace` + chosen backend, no inference, using `LocalCorpusClient`

## Simulator Protocol

`boed_agent.simulator_protocol` exposes a structural `Simulator` protocol. Any object with the following attributes can be passed to the agent:

- `is_explicit: bool` — closed-form likelihood available (→ Pyro VI)
- `is_differentiable: bool` — implicit but differentiable (→ MINEBED / iDAD)
- `metadata: SimulatorMetadata` — parameter names, units, domain tags

`SimpleSimulator` is the minimal reference implementation for tests and examples.

## Backend dispatch

`SimulatorChoiceModule.select(simulator, registry, literature_report)` runs the waterfall:

1. `simulator.is_explicit` → `pyro`
2. `simulator.is_differentiable` and `backend_options['policy_network']` → `idad`; otherwise `minebed`
3. else → `lfiax`

When a `LiteratureReport` is supplied, its `backend_preference.ranked` list can override the waterfall pick — but only when the literature candidate is still compatible with the simulator flags. Every override is recorded on `BackendChoice.literature_override` and the associated citations are preserved.

## Literature pipeline

`LiteratureSearchModule` runs five stages, each independently mockable for tests:

| Stage | What it does | LLM |
|-------|--------------|-----|
| A — filters | regex / keyword prefilter against abstracts | no |
| B — extraction | sentence-level structured evidence mining, batched | cheap |
| C — aggregation | deterministic per-parameter / per-dimension tables | no |
| D — reasoning | one focused chain-of-thought call per decision | reasoning |
| E — trace | assembles `ReasoningTrace` with citations and token cost | no |

The recommended token budget for a typical 20-paper, 5-parameter run is ≈70K tokens (vs. ≈500K if every paper were dumped into a single LLM call). Pass `TokenBudget(max_total_tokens=...)` to stop early; the pipeline halts gracefully and returns a partial report.

`LLMClient` is a protocol — swap in `OpenAILLMClient`, `ClaudeLLMClient`, or a `RecordingLLMClient` (used by the test suite) without touching the pipeline. Every prompt is cached by SHA-256 hash so re-runs with the same evidence table cost zero tokens.

Every numerical recommendation in the final `LiteratureReport` must be either cited (`cited_papers`) or explicitly marked `fallback=True`. The orchestrator enforces this with a post-synthesis validator — a raised `PostSynthesisValidationError` means the LLM produced a claim with neither a citation nor a fallback flag.

## Data classifier

`DataClassifier(mode="simulator_aware" | "raw")` runs HDBSCAN or GMM-BIC to detect whether a dataset is homogeneous. **The simulator-aware caveat**: homogeneity depends on the model, not the data alone. In `"raw"` mode a warning is always attached. In `"simulator_aware"` mode the classifier either calls `simulator.summary(point)` if defined or uses the simulator's forward evaluation at a default design. When the result is not homogeneous, the agent runs a per-cluster optimization loop.

## Caching and offline use

- `use_literature=False` disables the literature pipeline entirely.
- `NullLLMClient` is a drop-in that returns empty strings — useful for dry runs.
- `RecordingLLMClient(responder=...)` is used by the test suite; `cache` is a plain dict keyed by `(tier, sha256(prompt))`.
- API clients accept an optional `fetcher` callable that bypasses network I/O.

## Literature Advisory From Specs

`ExperimentSpec` now supports literature preferences as first-class fields:

- `use_literature`
- `literature_source_mode` — one of `online`, `local`, or `both`
- `literature_corpus_dir` — required for `local` and `both`

Example:

```json
{
  "backend": "pyro",
  "model_ref": "demo.module:model",
  "problem_summary": "One-compartment PK model with uncertain absorption rate",
  "use_literature": true,
  "literature_source_mode": "local",
  "literature_corpus_dir": "examples/agent/local_corpus/pharmacokinetic_ode",
  "target_latent_labels": ["k_a", "k_e", "V"],
  "observation_labels": ["concentration"],
  "design_variables": [{"name": "dose_time", "lower": 0.0, "upper": 24.0}]
}
```

Run the advisory-only literature path from the CLI:

```bash
boed-agent literature-dry-run spec.json
```

Or with a live provider for Stages B/D:

```bash
boed-agent literature-dry-run spec.json --provider openai --model gpt-4.1-mini
```

The `literature-dry-run` command returns the same core fields as `DryRunResult.to_dict()` plus `advisory_only: true`. In v1 this path is advisory only: it returns literature-derived priors, backend hints, and reasoning trace, but it does not auto-apply those priors to `boed-agent run`. Normal backend execution remains unchanged.

## Resilience

- Every literature API client (Semantic Scholar, arXiv, OpenAlex, PubMed, Unpaywall) wraps its raw network call with `tenacity`-style exponential backoff via `with_retries(ClientConfig)`. The decorator retries only transport-shaped exceptions (`OSError`, `TimeoutError`, `ConnectionError`); contract-shaped exceptions like `ValueError` are raised immediately. When `tenacity` is missing, a pure-stdlib fallback with equivalent semantics is used automatically.
- Rate limits follow each source's published quota: Semantic Scholar throttles at roughly 0.3 req/sec unauthenticated, arXiv enforces 3 sec between requests, OpenAlex is 1 sec, PubMed is 0.35 sec with an API key (else 1 sec), Unpaywall 0.2 sec.
- `UnpaywallClient.fetch_oa_text(doi, max_pages=...)` downloads the OA PDF and extracts text with `pypdf` (from the `[literature]` extra). Returns `None` when the DOI has no OA copy, `pypdf` is not installed, or parsing fails — callers treat the text as best-effort enrichment.
- `ReasoningConfig(strict_validation=True)` opts into **live** Stage D validation: the first numerical claim without a citation or `fallback=True` flag raises `LiveCitationError` immediately, instead of being collected for a post-hoc `PostSynthesisValidationError`. The default remains post-hoc validation so existing callers do not regress.
- `introspect_metadata(simulator)` returns a `SimulatorMetadata` even when the simulator object has no explicit metadata attached — it inspects `parameter_names` / `param_names` / `theta_names` attributes, then the call signature (skipping `self`, `xi`, `design`), and finally falls back to an empty metadata object. `BOEDAgent` calls this automatically before dispatching the literature search, so unadorned callables still produce useful queries.

## Running tests

```bash
pytest                 # default — unit tests only
pytest -m live         # optional — real HTTP calls to arXiv, Semantic Scholar, OpenAlex
```

Live tests are guarded by the `live` pytest marker and are skipped by the default `addopts` in `pyproject.toml`. Expect occasional flakiness — they depend on third-party availability and rate limits.

In particular, the Semantic Scholar client uses the public tier unless you provide an API key, so the live test may fail with `HTTP 429` even when networking is working correctly. That response means the request reached Semantic Scholar and was throttled; it is not a local DNS/connectivity failure. For reliable repeated runs of the literature-search tests or manual online BOED dry-runs, use a Semantic Scholar API key.

## Installation

Base install:

```bash
pip install -e .
```

### macOS (MacBook Pro) notes

The package runs on both Apple Silicon (M1/M2/M3/...) and Intel Macs with Python 3.11+. A few caveats worth knowing before you install the heavier extras:

- **PyTorch / MPS acceleration.** On Apple Silicon, install a recent PyTorch (`torch>=2.1`) to get the `mps` GPU backend. The helper `boed_agent.utils.device.get_torch_device()` picks `mps → cuda → cpu` automatically. `python -c "from boed_agent.utils.device import device_summary; print(device_summary())"` prints what was detected.
- **`hdbscan` (literature extra).** Wheels for Apple Silicon have historically been spotty. If `pip install -e ".[literature]"` fails while building hdbscan, fall back to conda/mamba: `conda install -c conda-forge hdbscan` then re-run the pip install with `--no-deps` on hdbscan.
- **`torchsde` (iDAD extra).** Needs a working C/C++ toolchain. Run `xcode-select --install` once on a fresh Mac, or install via `conda install -c conda-forge torchsde`.
- **`cli-anything-lfiax` (LFIAX backend).** Not a PyPI wheel — install the external harness from source: `pip install -e /path/to/lfiax/agent-harness` and confirm `which cli-anything-lfiax` resolves. The backend raises a clear error if the CLI is missing from `PATH`.
- **Git-backed dependencies.** `minebed`, `idad`, and `lfiax` are pulled via `pip install -e ".[all]"` from git. Make sure `git` is available (`brew install git` or Xcode Command Line Tools) and your SSH/HTTPS credentials are set up.

A known-good path on a fresh MacBook Pro:

```bash
xcode-select --install                                    # once
conda create -n boed-agent python=3.11 && conda activate boed-agent
conda install -c conda-forge hdbscan                      # optional, for [literature]
pip install -e ".[agents,pyro,dev]"
python scripts/smoketest.py                               # verify the install
```

### Post-install smoke test

After any `pip install -e .` (or an optional extra), run:

```bash
python scripts/smoketest.py                 # fast: imports, CLI, dispatcher, device
python scripts/smoketest.py --with-example  # also runs the offline dry-run example
python scripts/smoketest.py --with-tests    # also runs the pytest suite
```

The script reports which extras are installed (`[PASS]`), which are optional but absent (`[SKIP]`/`[WARN]`), and which hard-fail the install. It exits 0 on success, 1 on any required failure — suitable for CI.

### Conda Or Mamba

Create and activate an environment:

```bash
conda create -n boed-agent python=3.11
conda activate boed-agent
```

With `mamba`:

```bash
mamba create -n boed-agent python=3.11
conda activate boed-agent
```

Install the repo plus common agent and Pyro dependencies:

```bash
pip install -e ".[agents,pyro,dev]"
```

This now includes `openai-agents`, so the OpenAI `agents-sdk` chat runtime is available after install.

Add the simulator-oriented `lfiax` stack when needed:

```bash
pip install -e ".[lfiax]"
```

To run the LFIAX backend end to end, also install the external CLI harness:

```bash
pip install -e /path/to/lfiax/agent-harness
which cli-anything-lfiax
```

Add the literature extras:

```bash
pip install -e ".[literature]"
```

Add the MINEBED / iDAD extras (upstream repos, require PyTorch):

```bash
pip install -e ".[minebed]"
pip install -e ".[idad]"
```

The iDAD upstream examples use `torchsde`; the extra pulls it in, but some conda-env workarounds may be needed depending on CUDA.

If you want the full local environment in one shot:

```bash
pip install -e ".[all]"
```

### uv

Create a virtual environment and install the package with `uv`:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[agents,pyro,dev]"
```

Add optional extras as needed:

```bash
uv pip install -e ".[lfiax]"
uv pip install -e ".[yaml]"
```

Or install everything:

```bash
uv pip install -e ".[all]"
```

With optional integrations:

```bash
pip install -e ".[agents,pyro,dev]"
```

To add the simulator-oriented `lfiax` adapter dependency:

```bash
pip install -e ".[lfiax]"
```

To enable YAML specs:

```bash
pip install -e ".[yaml]"
```

To enable trajectory plot generation:

```bash
pip install -e ".[plot]"
```

For a full local development environment:

```bash
pip install -e ".[all]"
```

## CLI

### Runtime Modes

The `chat` command supports two runtime modes:

- `agents-sdk`: preferred for `--provider openai`; uses the OpenAI Agents SDK with SQLite sessions, guardrails, manager/specialist orchestration, and tracing.
- `manual`: legacy provider-neutral request/tool loop; still used for Claude and available for OpenAI as a fallback.

If you use `--provider openai` and do not specify `--runtime-mode`, the CLI defaults to `agents-sdk`.

List supported backends:

```bash
boed-agent list-backends
```

Validate a spec:

```bash
boed-agent validate examples/specs/pyro_linear_regression.json
```

Run BOED optimization:

```bash
boed-agent run examples/specs/pyro_linear_regression.json
```

Run the empty interactive linear-regression example. The CLI will ask for the missing fields and print numbered options at each step:

```bash
boed-agent run --interactive examples/specs/pyro_linear_regression_interactive.json
```

Run the Foster et al. paper-aligned defaults directly from JSON:

```bash
boed-agent run examples/specs/foster_ab_test_linear.json
boed-agent run examples/specs/foster_revealed_preference.json
```

Run the new LFIAX linear examples:

```bash
boed-agent validate examples/specs/lfiax_linear_point.json
boed-agent run examples/specs/lfiax_linear_point.json

boed-agent validate examples/specs/lfiax_linear_distribution.json
boed-agent run examples/specs/lfiax_linear_distribution.json
```

The point example uses the differentiable path. The distribution example uses the black-box path with annealed narrowing around `xi_mu`. Both require `cli-anything-lfiax` plus the JAX/Haiku/Optax stack in the active environment.

Run the interactive Foster preset. The CLI will ask which paper benchmark and which estimator to use, then it will fill the remaining BOED settings from the paper defaults:

```bash
boed-agent run --interactive examples/specs/foster_variational_interactive.json
```

For selection prompts, enter the numbered option shown in the terminal. If you want to type a custom value, first choose the custom/manual option number and then enter the value on the follow-up prompt.

If the user wants to recreate the optimization trajectory, set `"recreate_trajectory": true` in the experiment spec. The backend result and `summarize_result` tool will then include `artifacts.optimized_design_histories`.

For a compressed representation, the run output also includes `artifacts.optimized_design_history_summaries`.

If you also want a saved plot in the project artifacts directory, set:

```json
{
  "recreate_trajectory": true,
  "artifacts": {
    "save_trajectory_plot": true
  }
}
```

When plotting is enabled and `matplotlib` is installed, the run saves a PNG under the run artifact directory, for example `artifacts/pyro_<timestamp>/design_trajectory.png`.

Interactive chat:

```bash
boed-agent chat --provider openai --model gpt-4.1 --runtime-mode agents-sdk --spec examples/specs/pyro_linear_regression.json
```

Resume an OpenAI Agents SDK chat session:

```bash
boed-agent chat --provider openai --model gpt-4.1 --session-id my-session --session-db artifacts/agent_sessions.sqlite
```

Disable tracing for an OpenAI Agents SDK chat:

```bash
boed-agent chat --provider openai --model gpt-4.1 --disable-tracing
```

## Example Specs

- [Pyro variational example](examples/specs/pyro_linear_regression.json)
- [Pyro interactive example](examples/specs/pyro_linear_regression_interactive.json)
- [Foster A/B test default example](examples/specs/foster_ab_test_linear.json)
- [Foster revealed-preference default example](examples/specs/foster_revealed_preference.json)
- [Foster interactive preset](examples/specs/foster_variational_interactive.json)
- [LFIAX differentiable point example](examples/specs/lfiax_linear_point.json)
- [LFIAX black-box distribution example](examples/specs/lfiax_linear_distribution.json)
- [Minimal LFIAX spec example](examples/specs/lfiax_stub.json)

## Design Principles

- Keep one internal tool contract regardless of LLM provider.
- Treat BOED engines as pluggable adapters, not hard-coded dependencies.
- Refuse execution when the experiment spec is underspecified.
- Save normalized artifacts so humans and agents can inspect runs consistently.
