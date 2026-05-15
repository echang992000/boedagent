# Architecture

## High-level modules

- `boed_agent.simulator_protocol`: `Simulator` structural protocol plus `SimpleSimulator` and `SimulatorMetadata`.
- `boed_agent.literature`: 5-stage literature pipeline (Stage A filters → Stage B extraction → Stage C aggregation → Stage D reasoning → Stage E trace assembly) + API clients for Semantic Scholar, arXiv, OpenAlex, PubMed, Unpaywall.
- `boed_agent.classifier`: `DataClassifier` (homogeneity triage) with raw and simulator-aware modes.
- `boed_agent.prior_builder`: `PriorBuilder` — augments user priors with literature report; never silently overrides.
- `boed_agent.simulator_choice`: `SimulatorChoiceModule` waterfall dispatcher (explicit → Pyro, differentiable → MINEBED / iDAD, else → LFIAX). Honours literature overrides when compatible.
- `boed_agent.agent`: `BOEDAgent` orchestrator + `DryRunResult` / `AgentRunResult`.

## Core layers

- `boed_agent.core`: runtime abstractions and engine selection.
- `boed_agent.agents`: OpenAI Agents SDK-native manager, specialists, and guardrails.
- `boed_agent.providers`: transport adapters for OpenAI and Claude.
- `boed_agent.tools`: stable tool registry exposed to the agent.
- `boed_agent.backends`: BOED execution adapters (Pyro, LFIAX, MINEBED, iDAD).
- `boed_agent.clarification`: missing-field detection and question generation.

## Data flow

1. A user prompt or CLI run loads an `ExperimentSpec`.
2. Validation and clarification determine whether execution is safe.
3. The selected engine runs:
   - manual mode uses the provider adapter and local tool loop
   - OpenAI `agents-sdk` mode uses a manager agent, specialist agents-as-tools, SQLite sessions, guardrails, and tracing
4. Tool handlers route into backend adapters.
5. Results are normalized into shared result models and written as artifacts.

## Chat engines

### Manual

The manual engine keeps the original provider-neutral loop and is used for Claude plus OpenAI fallback mode.

### OpenAI Agents SDK

The preferred OpenAI path uses:

- a BOED manager agent as the user-facing orchestrator
- specialist agents exposed as tools
- local BOED function tools
- input, output, and tool guardrails
- SQLite-backed sessions
- built-in tracing

## Backends

### Pyro

The Pyro backend targets `pyro.contrib.oed` estimators:

- `vi_eig`
- `posterior_eig`
- `marginal_eig`
- `vnmc_eig`

The adapter accepts callables by registry name or import reference and normalizes outputs into `EIGEstimate` and `OptimizationResult`.

### LFIAX

The `lfiax` adapter is a CLI bridge to `cli-anything-lfiax`. It validates simulator-oriented specs locally, shells out for backend metadata and optimization, and normalizes the returned payload into shared BOED result types.

The harness supports two execution paths selected by `differentiable`:

- `true`: differentiable joint optimization through the simulator
- `false` or missing: black-box joint optimization using the learned likelihood surrogate

For `backend_options.design_mode="distribution"`, the harness keeps an annealed design distribution that narrows over time around per-slot `xi_mu` values.

## Clarification policy

The clarification planner owns:

- backend selection prompts
- required-field prompts by backend
- ordering of questions
- safe defaults vs must-ask fields

It is used by both the CLI and the tool registry so the agent cannot silently guess high-impact BOED settings.
