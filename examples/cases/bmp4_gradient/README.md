# BMP4 Gradient Bundle

This directory holds the problem bundle for the BMP4 gradient example.

Current status:

- shared problem structure is recorded in `problem.json`
- observed-data and literature source locations are linked there
- two differentiable simulator candidates live under `simulators/differentiable/`
- family-specific Pyro fitting wrappers live under `pyro/`
- reproducible literature prompt templates live under `prompts/`
- example-local literature, prior translation, fitting, posterior predictive, and BOED orchestration live in `examples/agent/bmp4_gradient_agent.py`
- the multireceptor family uses per-cell-line qPCR `Rs` values as wide truncated-lognormal abundance priors on `[0, 5]`

The intended split is:

1. Keep shared problem context, data references, and candidate names in `problem.json`.
2. Keep BMP4-specific execution code local to this bundle and the BMP4 example runner.
3. Treat `problem.json` as the stable shared bundle, while fitting and BOED logic stay family-specific in nearby Python modules.

Main local modules:

- `data.py` loads the NPZ into one dataset object per cell line.
- `priors.py` translates literature output and qPCR values into concrete Pyro priors.
- `registry.py` maps conceptual model families to fitting and BOED helpers.
- `inference.py` runs SVI, posterior summarization, and empirical EIG optimization.
- `plotting.py` writes the BMP4-specific posterior predictive and EIG trajectory plots.
- `prompts/` holds versioned prompt templates for reproducible literature-prior synthesis.
