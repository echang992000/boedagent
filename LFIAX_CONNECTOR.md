# LFIAX Connector

How `boed_agent` talks to the external `cli-anything-lfiax` harness.

## Summary

`boed_agent` does not implement the LFIAX optimizer itself. The adapter in
[`src/boed_agent/backends/lfiax_backend.py`](src/boed_agent/backends/lfiax_backend.py)
serializes an `ExperimentSpec`, calls the external CLI, and maps the JSON
response back into `ValidationReport`, `EIGEstimate`, and `OptimizationResult`.

The harness is the `agent-harness` package inside the upstream
[`lfiax` repository](https://github.com/vz415/lfiax). Cloning and
`pip install -e /path/to/lfiax/agent-harness` installs the
`cli-anything-lfiax` executable onto your `PATH`.

## CLI Surface

The bridge uses these commands:

```bash
cli-anything-lfiax --json oed describe
cli-anything-lfiax --json oed validate <spec.json>
cli-anything-lfiax --json oed optimize <spec.json>
```

`estimate_eig()` is implemented as a thin single-step `oed optimize` call with
a fixed point design and artifact writing disabled.

## Spec Contract

The LFIAX path uses the same `ExperimentSpec` JSON shape as the rest of
`boed_agent`.

Required fields at the `boed_agent` layer:

- `backend`
- `simulator_ref`
- `design_variables`
- `backend_options.design_mode`
- `differentiable`
- `objective.estimator`
- one of `prior_sampler_ref` or `latent_sampler_ref`

Important LFIAX-specific fields:

```json
{
  "backend": "lfiax",
  "simulator_ref": "my_pkg.problem:simulator",
  "prior_sampler_ref": "my_pkg.problem:prior",
  "differentiable": false,
  "design_variables": [
    {"name": "xi_1", "lower": -2.0, "upper": 2.0},
    {"name": "xi_2", "lower": -1.0, "upper": 1.0}
  ],
  "compute_budget": {
    "num_outer_samples": 256,
    "num_inner_samples": 16,
    "num_optimization_steps": 500,
    "design_learning_rate": 0.05,
    "flow_learning_rate": 0.001
  },
  "objective": {
    "estimator": "lf_pce_eig_scan",
    "estimator_kwargs": {"lam": 0.5}
  },
  "backend_options": {
    "design_mode": "distribution",
    "xi_mu_init": [0.0, 0.0],
    "xi_stddev_init": [1.0, 1.0],
    "xi_stddev_min": 0.01,
    "end_sigma": 0.05,
    "decay_rate": 10.0
  },
  "surrogate": {
    "checkpoint_filename": "likelihood_checkpoint.pkl"
  }
}
```

For multi-slot designs:

- `len(xi_mu_init)` must match `len(design_variables)`
- `len(xi_stddev_init)` must match `len(design_variables)` in distribution mode

## Execution Paths

Path selection is driven by `differentiable`:

- `true`: differentiable joint optimization
- `false`: black-box joint optimization
- missing in the harness: defaults to black-box
- missing in `boed_agent`: treated as a clarification error so the user must choose

Both paths jointly update:

- the conditional likelihood surrogate `q(y | theta, xi)`
- the design parameters

They do that in the same loop on the same simulated batches.

## Design Modes

`backend_options.design_mode` controls the design parameterization:

- `point`
  - optimizes `xi_mu`
  - final `design` is the optimized vector
- `distribution`
  - optimizes `xi_mu` and `xi_stddev`
  - applies exponential annealing driven by `end_sigma` and `decay_rate`
  - narrows the effective design distribution over time around the learned mean

For vector-valued designs, the optimizer works on one joint design vector, not
on separate per-slot optimization jobs.

## Returned Artifacts

The harness returns a normalized payload that includes:

- `execution_path`
- `design`
- `eig`
- `history`
- `xi_mu`
- `xi_stddev` in distribution mode
- `artifacts.run_dir`
- `artifacts.likelihood_checkpoint`
- `artifacts.likelihood_metadata`
- `artifacts.sigma_history`

`boed_agent` copies those fields into the final `OptimizationResult`, and when
`"recreate_trajectory": true`, it also materializes
`artifacts.optimized_design_histories` and
`artifacts.optimized_design_history_summaries`.

## Example Specs

The repo now ships two native LFIAX examples:

- [`examples/specs/lfiax_linear_point.json`](examples/specs/lfiax_linear_point.json)
- [`examples/specs/lfiax_linear_distribution.json`](examples/specs/lfiax_linear_distribution.json)

Both call into:

- [`src/boed_agent/demo/lfiax_linear.py`](src/boed_agent/demo/lfiax_linear.py)

That module provides:

- a prior sampler
- a linear Gaussian simulator
- checkpoint loading helpers
- a helper to rebuild the saved likelihood log-probability function

## Installation

`boed_agent` itself only provides the bridge. To execute LFIAX runs you also
need the harness and its numerical stack available in the active Python
environment:

```bash
# Clone the upstream repo somewhere convenient, then:
cd /path/to/lfiax/agent-harness
pip install -e .
which cli-anything-lfiax
cli-anything-lfiax --json oed describe
```

On macOS, `which` resolves the entry point installed by pip into your current
Python environment (`conda`, `venv`, or `uv`). If it does not resolve,
re-activate the environment that owns the harness install.

If `jax`, `haiku`, or `optax` are missing, LFIAX validation can still work, but
real optimize runs will fail before training starts.
