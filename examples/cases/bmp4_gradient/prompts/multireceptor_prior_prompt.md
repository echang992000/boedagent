Read these local files and directories:

- `data/bmp4_data/BMP4_GRADIENT_README.md`
- `examples/cases/bmp4_gradient/problem.json`
- `examples/cases/bmp4_gradient/prompts/PRIOR_OUTPUT_SCHEMA.md`
- `examples/agent/local_corpus/bmp4_gradient`

Task:

Synthesize literature-informed priors for the BMP4 gradient multireceptor model
used in the BMP4 example bundle.

Model-family context:

- Family name: `multireceptor`
- Interpretation: BMP4 dose-response model with receptor-specific binding and
  signaling weights, plus a downstream Hill-type response nonlinearity and
  observation noise
- Shared receptor set for literature reasoning:
  - `BMPR1A`
  - `BMPR1B`
  - `BMPR2`
  - `ACVR1`
  - `ACVR2A`
- Parameters to synthesize:
  - `kd`
  - `weight`
  - `bottom`
  - `top`
  - `s50`
  - `response_hill`
  - `sigma_y`

Requirements:

- Use only the local BMP4 corpus and the BMP4 readme/problem bundle listed
  above.
- Search for priors relevant to the shared NMuMG receptor set, not per-cell-line
  priors.
- Keep parameter names exactly as listed.
- Prefer one of these distribution families:
  - `Normal`
  - `LogNormal`
  - `Gamma`
  - `Uniform`
  - `Beta`
- If evidence is weak, conflicting, or absent, choose a weak prior and set
  `fallback` to `true`.
- Keep `cited_papers` grounded in the local files only.
- Return JSON only, following the shared schema in
  `examples/cases/bmp4_gradient/prompts/PRIOR_OUTPUT_SCHEMA.md`.

Important exclusions:

- Do not create literature priors for receptor-abundance parameters.
- Do not add `abundance`, `abundance_*`, `Rs`, or qPCR measurement terms to the
  output.
- In this BMP4 example, receptor abundances are handled downstream as
  cell-line-specific wide truncated-lognormal priors on `[0, 5]` centered on
  the per-cell-line `Rs` values.

Additional guidance:

- `kd` should represent a shared receptor-binding scale prior family for the
  receptor set.
- `weight` should represent receptor signaling contribution strength.
- `bottom` and `top` should capture baseline and maximal response behavior.
- `s50` should represent the downstream signaling half-max scale.
- `response_hill` should represent downstream response steepness.
- `sigma_y` should represent observation-noise scale.

Output contract:

- `family` must be `"multireceptor"`
- `corpus_scope` must be `"local_bmp4_corpus_only"`
- `parameter_order` must be `["kd", "weight", "bottom", "top", "s50", "response_hill", "sigma_y"]`
- `priors` must contain exactly those seven parameters and nothing else
