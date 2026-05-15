# Sequential BOED Plans — SIR (LFIAX) and Linear Regression (Pyro)

Two automated, sequential Bayesian-optimal-experimental-design (BOED) pipelines
layered on top of the existing `boed_agent` repo. Both follow the same
skeleton as `examples/cases/bmp4_gradient/` — a problem bundle plus nearby
Python modules — but they run **multi-round** loops and persist
per-round posterior + EIG artifacts so we can visualise shrinkage.

No literature search, no prior elicitation. The point is to show that the
BOED machinery is automated end-to-end for two known baselines.

---

## 0. Context and what we are matching

The reference plot shows an SIR outbreak on `t ∈ [0, 100]`:

- orange = one true SIR trajectory `I_true(t)` (the "real" curve we want to recover),
- blue = prior predictive mean of `I(t)` with a grey 90 % band,
- green = a fixed design distribution over measurement time `xi ∈ [0, 100]`.

At every experimental step we pick a measurement time `xi`, observe (noisy)
`I(xi)`, and want the posterior over the SIR parameters to contract toward
the parameter values that generated the orange curve. Because clock time is
monotone, the plan enforces `xi_1 < xi_2 < … < xi_T` throughout.

---

## 1. Repo layout to add

Working on top of the uploaded `boed_agent-main` snapshot — which only
contains `examples/`, `docs/`, `data/`, `README.md`, `LFIAX_CONNECTOR.md`
(no `src/boed_agent/`). All new files sit under `examples/` plus one
top-level `PLAN.md` (this document):

```
PLAN.md                                          # this file
examples/
  cases/
    sir/
      README.md
      __init__.py
      problem.json                               # v1 bundle (LFIAX candidate)
      simulator.py                               # stochastic SIR + summariser
      prior.py                                   # prior sampler, posterior reweighting helpers
      monotone.py                                # strict-increasing design reparam
      sequential_lfiax.py                        # per-round LFIAX call + posterior update
      plotting.py                                # posterior-shrink + EIG plots + I(t) curve
    linreg_sequential/
      README.md
      __init__.py
      problem.json                               # v1 bundle (Pyro candidate)
      model.py                                   # pyro model/guide for y = a*x + b + noise
      sequential_pyro.py                         # VI update per round + EIG optimise
      monotone.py                                # same reparam, shared with SIR via import
      plotting.py
  agent/
    sir_sequential_agent.py                      # thin driver: repeats BOEDAgent rounds
    linreg_sequential_agent.py                   # thin driver: same, explicit Pyro path
artifacts/                                       # created at runtime (per-round outputs)
```

`problem.json` in both bundles matches the v1 schema described in
`examples/cases/PROBLEM_BUNDLE_FIELDS.md`: a single candidate (simulator
for SIR, explicit for linreg), shared latent/observation spec, one
`design_variable`.

---

## 2. Plan A — Stochastic SIR via LFIAX (implicit backend)

### 2.1 Simulator

`examples/cases/sir/simulator.py` defines a **continuous-time Markov chain**
SIR on a closed population `N` with two latent parameters:

- `beta`  — contact rate,
- `gamma` — recovery rate.

Optionally add `I0` (seed infections) as a third latent. A Gillespie
implementation gives non-differentiable, likelihood-free dynamics — exactly
what LFIAX is built for, and it matches the noisy orange curve in the
uploaded plot. Public API:

```python
def simulate_sir_trajectory(theta, *, N=1000, t_max=100.0, seed=None)
    # → dict with "times" (jump times) and "I" (infected counts at those times)

def simulate_sir_at_times(theta, xi, *, N=1000, I0=5, seed=None)
    # xi: length-T vector of strictly increasing measurement times
    # → length-T vector of I(xi_k) (integer counts), via step-function lookup
```

The boed_agent-facing **simulator callable** `simulate` takes `(theta, xi)`
and returns the length-`T` observation vector. `theta` is a flat length-2
vector `[beta, gamma]`. `is_differentiable=False`, `is_explicit=False`, so
`SimulatorChoiceModule` routes to LFIAX per the README waterfall (explicit
→ Pyro; differentiable with `backend_options['policy_network']=True` → iDAD,
else MINEBED; otherwise LFIAX).

### 2.2 Prior over `(beta, gamma)`

`examples/cases/sir/prior.py`:

- Round 0: broad independent log-normal priors,
  `log beta ~ N(log 0.3, 0.5^2)`, `log gamma ~ N(log 0.1, 0.5^2)`,
  truncated so `R0 = beta / gamma ∈ [0.5, 10]`.
- Exposes `sample_prior(rng, num_samples) -> np.ndarray` (LFIAX
  `prior_sampler_ref` contract).
- Also exposes `weighted_resample(samples, log_weights, num_out)` for the
  posterior update between rounds (importance-resample using the LFIAX
  likelihood surrogate — see §2.5).

### 2.3 Monotone-time design reparameterisation

`examples/cases/sir/monotone.py` (shared with the linreg bundle):

```python
def pack_deltas(xi_abs, t_max):
    # absolute times → unconstrained reals via inverse softplus on deltas
    # xi_abs = [xi_1, xi_2, ..., xi_T], 0 < xi_1 < ... < xi_T <= t_max

def unpack_deltas(raw, t_max):
    # raw ∈ R^T → xi_abs ∈ (0, t_max]^T, strictly increasing
    deltas = softplus(raw)                        # > 0
    cumulative = cumsum(deltas)
    xi_abs = t_max * sigmoid(cumulative)          # keeps everything in (0, t_max)
    return xi_abs
```

The LFIAX harness optimises in the unconstrained `raw` space; the simulator
wrapper maps back before calling `simulate_sir_at_times`. That gives
`xi_t > xi_{t-1}` by construction and removes the need for any custom
inequality constraints in LFIAX. For the **sequential** loop we pin the
already-chosen times and only optimise the remaining slot, so at round `t`
the packed variable is a scalar `raw_t` and the design becomes
`xi_t = previous_xi_{t-1} + softplus(raw_t) · scale` clipped to
`(xi_{t-1}, t_max)`.

### 2.4 `problem.json` for SIR

```jsonc
{
  "schema_version": "v1",
  "problem_summary": "Stochastic (Gillespie) SIR with latent (beta, gamma); observe noisy I(xi_k) at strictly increasing measurement times.",
  "shared": {
    "latent_spec": {"shape": [2], "labels": ["beta", "gamma"]},
    "observation_spec": {"shape": [1], "labels": ["I_t"]},
    "design_variables": [
      {"name": "xi", "lower": 0.0, "upper": 100.0,
       "description": "Next measurement time; optimised subject to xi > xi_prev."}
    ]
  },
  "candidates": [
    {
      "name": "sir_gillespie_lfiax",
      "kind": "simulator",
      "description": "Stochastic SIR with likelihood-free inference via LFIAX.",
      "simulator_ref": "examples.cases.sir.simulator:simulate",
      "prior_sampler_ref": "examples.cases.sir.prior:sample_prior",
      "differentiable": false,
      "backend": "lfiax",
      "backend_options": {
        "design_mode": "point",
        "xi_mu_init": [50.0],
        "xi_stddev_init": [10.0],
        "xi_stddev_min": 0.01
      }
    }
  ]
}
```

The `backend` field matches what the adapter expects (see
`LFIAX_CONNECTOR.md` §Spec Contract). `design_mode="point"` is appropriate
because each round picks **one** new time.

### 2.5 Sequential loop (per round)

`examples/cases/sir/sequential_lfiax.py` ties it together. One round is:

1. **Construct a per-round prior sampler** from the current posterior
   particles (for round 0 this is the wide log-normal prior). Write it out
   as a `.npy` file that the prior sampler module loads lazily — LFIAX
   expects a sampler *callable* referenced by `prior_sampler_ref`.
2. **Call LFIAX via `BOEDAgent`** to jointly train the amortised likelihood
   surrogate `q(y | theta, xi)` and optimise `xi_t` on the admissible interval
   `(xi_{t-1}, t_max)`. The `ExperimentSpec` for the round is compiled from
   `problem.json` with a round-specific design bound and
   `artifacts.save_trajectory_plot=true`, `recreate_trajectory=true` so the
   harness saves the optimisation history (EIG trajectory, xi trajectory,
   sigma history).
3. **Record** `xi_t*`, the per-step EIG curve, the final EIG, and the saved
   likelihood surrogate checkpoint (`artifacts.likelihood_checkpoint`).
4. **Run the ground-truth simulator** with `theta_true = (beta_true,
   gamma_true)` at the vector `xi_{1:t}*`, but only *append* the new
   observation `y_t = I(xi_t*) + noise`. Earlier observations are cached
   from previous rounds, so we never re-simulate them. (Gillespie is
   non-reversible, but sampling `I_true(·)` once at round 0 and reusing it
   is equivalent to observing the same trajectory at progressively denser
   times, which is exactly the "find the real curve" framing.)
5. **Posterior update** by self-normalised importance sampling: draw `M`
   fresh theta samples from the wide prior (or from the current posterior
   particles), score them under the accumulated log-likelihood
   `sum_k log q(y_k | theta, xi_k*)` using the saved LFIAX surrogate,
   normalise weights, and resample `M` particles. This gives the posterior
   particles that seed round `t+1`.
6. **Save per-round artifacts**:
   - `artifacts/sir/round_{t:02d}/posterior_samples.npy`
   - `artifacts/sir/round_{t:02d}/eig_history.json`
   - `artifacts/sir/round_{t:02d}/xi_history.json`
   - `artifacts/sir/round_{t:02d}/observation.json`  (xi_t*, y_t)

Expose a single entry point:

```python
def run_sequential_sir_lfiax(
    *, theta_true, num_rounds=8, t_max=100.0, N=1000, sigma_obs=5.0,
    num_posterior_particles=2000, compute_budget=None, artifacts_dir,
    seed=0,
) -> SequentialResult
```

`SequentialResult` holds per-round `posterior_particles`, `xi_star`,
`eig_final`, `eig_history`, and a `true_trajectory` for plotting.

### 2.6 Why the BOEDAgent still makes sense here

Round-by-round we build a fresh `ExperimentSpec` and hand it to
`BOEDAgent.run(...)` — the waterfall in `SimulatorChoiceModule` picks LFIAX
because `is_explicit=False` and `is_differentiable=False`, matching
`examples/agent/black_box_agent_sim.py`. No literature is invoked
(`use_literature=False`). The agent produces an `OptimizationResult` whose
`artifacts.run_dir` we harvest for the surrogate checkpoint.

### 2.7 GPU / JAX note

LFIAX is JAX/Haiku/Optax and optimises **both** the normalising-flow
likelihood surrogate *and* the design. On CPU, one round of a few hundred
optimisation steps on `num_outer_samples=256`, `num_inner_samples=16` can
take several minutes; across 8 rounds this becomes a wall-clock problem.
Recommended environment before running the LFIAX plan:

```bash
pip install --upgrade "jax[cuda12]" haiku-nnx optax  # or jax-metal on Apple Silicon
python -c "import jax; print(jax.devices())"         # confirm a GPU is detected
```

Fallback: drop `num_optimization_steps` to ~200 and `num_outer_samples` to
128 for a quick smoke test on CPU.

---

## 3. Plan B — Linear regression via Pyro (explicit backend)

Same sequential skeleton, but the forward model is the stock
`y = a·x + b + Normal(0, sigma)` that already ships as
`examples/agent/explicit_linear_regression.py`. Because the simulator is
explicit we land on `backend="pyro"` via `SimulatorChoiceModule`, and we
get exact variational posterior updates.

### 3.1 Model + guide

`examples/cases/linreg_sequential/model.py`:

```python
def model(xi, y_obs=None, *, prior_means, prior_stds, sigma_obs=0.1):
    a = pyro.sample("a", dist.Normal(prior_means[0], prior_stds[0]))
    b = pyro.sample("b", dist.Normal(prior_means[1], prior_stds[1]))
    with pyro.plate("obs", xi.shape[0]):
        pyro.sample("y", dist.Normal(a * xi + b, sigma_obs), obs=y_obs)
```

The **prior is deliberately vague** — `Normal(0, 3)` on both slope and
intercept — and gets tightened each round by reading the previous round's
`AutoDiagonalNormal` posterior means/stds back into `prior_means` /
`prior_stds`. That is the "automation" part: no manual prior elicitation.

### 3.2 EIG and design optimisation

Per round, use `pyro.contrib.oed.eig.marginal_eig` (or `posterior_eig` for
tighter estimates) with a design `xi_t` parameterised through the same
`monotone.py` helper:

```python
xi_t = xi_prev + softplus(raw) * scale         # scale = 0.5 * (x_max - xi_prev)
raw = torch.tensor(0.0, requires_grad=True)
optim = torch.optim.Adam([raw], lr=0.05)
for step in range(num_eig_steps):
    optim.zero_grad()
    xi_candidate = torch.clamp(xi_t_from_raw(raw), min=xi_prev + eps, max=x_max)
    eig = marginal_eig(model_fn, xi_candidate, "y", ["a", "b"],
                       num_samples=256, num_steps=50, final_num_samples=512)
    loss = -eig
    loss.backward()
    optim.step()
    history.append({"step": step, "xi": float(xi_candidate), "eig": float(eig)})
```

This mirrors `examples/cases/bmp4_gradient/inference.py:optimize_empirical_eig`
but uses the official Pyro EIG estimators instead of the hand-rolled
empirical one.

### 3.3 Sequential loop (per round)

`sequential_pyro.py`:

1. Compile `ExperimentSpec` for this round with the current `prior_means`,
   `prior_stds`, and `xi_prev`.
2. Run `BOEDAgent.run(...)` which dispatches to the Pyro backend and
   returns `xi_t*` plus EIG history.
3. Evaluate the **ground-truth** generator — a fixed `theta_true = (a_true,
   b_true)` plus Gaussian noise — at `xi_t*` to get `y_t`.
4. Append `(xi_t*, y_t)` to the running dataset.
5. Run SVI with `AutoDiagonalNormal` on the full accumulated dataset
   (equivalent to conjugate update for this model, but cheaper to keep
   code uniform with multi-parameter extensions). Extract posterior mean
   and std per parameter and per round.
6. Update `prior_means`, `prior_stds` for the next round (= posterior
   from this round). Under a Gaussian linear model this is exactly the
   conjugate sequential posterior.
7. Save per-round artifacts in `artifacts/linreg/round_{t:02d}/`:
   - `posterior_samples.pt` (5k draws from the AutoGuide)
   - `eig_history.json`
   - `design_history.json`
   - `observation.json`

Entry point:

```python
def run_sequential_linreg_pyro(
    *, theta_true=(1.5, -0.2), num_rounds=8, x_max=1.0, sigma_obs=0.1,
    num_eig_steps=200, artifacts_dir, seed=0,
) -> SequentialResult
```

---

## 4. Monotone-design constraint — shared rules

Both plans enforce `xi_t > xi_{t-1}` with the **same** primitive
(`examples/cases/sir/monotone.py`, reused in linreg):

- Absolute times are never optimised directly; only `raw_t ∈ R` is.
- `xi_t = xi_{t-1} + softplus(raw_t) · scale_t`, then clipped to
  `(xi_{t-1}, t_max)` to keep the last round feasible.
- `scale_t = 0.5 · (t_max - xi_{t-1})` keeps a reasonable dynamic range
  as rounds progress and the remaining interval shrinks.
- The xi history we plot on `t` is therefore always strictly monotone —
  matching the user's intuition that later rounds sit further right.

The *design distribution* path (LFIAX `design_mode="distribution"`)
becomes a truncated Gaussian on `(xi_{t-1}, t_max)`; the mean is
parameterised as above and the std still anneals via LFIAX's built-in
`end_sigma` / `decay_rate`.

---

## 5. Plotting specification (identical for both bundles)

Each bundle ships a `plotting.py` with three figures produced at the end
of the sequential loop. `matplotlib` only — no seaborn.

### 5.1 Posterior-shrinkage panels

`posterior_shrinkage.png` — for each latent parameter, one subplot with one
KDE (or histogram) per round overlaid in an ordered colormap (`viridis`
round 0 → round T). Vertical dashed line at `theta_true`. Expected
behaviour: densities tighten and centre on the dashed line as rounds
increase.

### 5.2 EIG-per-round

`eig_per_round.png` — two subplots:

- top: final per-round EIG as a function of round index (a scatter/line plot),
- bottom: per-round **inner** EIG optimisation trajectories (one faint line
  per round, the winner annotated with the chosen `xi_t*`).

### 5.3 Design trajectory on the SIR curve (SIR bundle only)

`design_on_curve.png` — reproduces the reference plot layout:

- orange: the one true infected trajectory `I_true(t)` sampled once from
  Gillespie with `theta_true`;
- blue + grey: prior predictive mean and 90 % band from 2k prior draws;
- green (optional): kernel density of the round-T posterior predictive
  measurement-time distribution, for visual parity with the uploaded
  figure;
- vertical rules at `xi_1, xi_2, ..., xi_T`, labelled with round index,
  demonstrating strict monotonicity.

### 5.4 Plot for linreg

`linreg_fit_overlay.png` — `y` vs `xi`, true line `y = a_true·x + b_true`
in orange, per-round posterior predictive band, and the chosen designs
`xi_t*` as vertical ticks along the x-axis in the same viridis colouring
as §5.1.

All plot functions mirror the signature used in
`examples/cases/bmp4_gradient/plotting.py:save_posterior_predictive_plot`
(accept numpy, return the written path).

---

## 6. Entry-point drivers

Two thin scripts mirror `examples/agent/black_box_agent_sim.py` /
`explicit_linear_regression.py`. They only parse CLI args and call the
loop functions:

```bash
python examples/agent/sir_sequential_agent.py --rounds 8 --seed 0 \
       --artifacts artifacts/sir_seq
python examples/agent/linreg_sequential_agent.py --rounds 8 --seed 0 \
       --artifacts artifacts/linreg_seq
```

Each script at the end writes a `summary.json` with per-round posterior
means/stds and EIG values and invokes the three plots from §5.

---

## 7. Validation plan (what we check before declaring "automated")

Cheap smoke tests that live next to the bundles (no heavy tests, just
asserts inside a `__main__` guard of each file) so a reviewer can run:

- `python -m examples.cases.sir.simulator`  → prints mean peak-I and mean
  peak-time over 100 Gillespie runs with default `theta_true` and asserts
  both are within broad sanity bounds (peak-I ∈ [100, 600], peak-time ∈ [5, 40]).
- `python -m examples.cases.sir.monotone`  → checks `unpack_deltas` is
  strictly increasing on 1k random `raw` vectors.
- `python -m examples.cases.linreg_sequential.model`  → a 2-round dry run
  where the posterior means move toward `theta_true` and posterior stds
  shrink.
- Success criteria for the full runs:
  1. Posterior std for every latent at round `T` is ≤ 25 % of round-0 std.
  2. `xi_t` is strictly monotone across rounds.
  3. Per-round EIG is non-negative and decreases (non-strictly) on
     average — shrinking posteriors yield less information per new design.
  4. Final posterior mean is within 2·(round-T std) of `theta_true`.

---

## 8. Run commands (after the above is implemented)

```bash
# one-time env setup
pip install -e .                                 # the boed_agent package (from parent repo)
pip install -e ".[agents,pyro,dev]"              # Pyro path
pip install -e ".[lfiax]"                        # LFIAX backend adapter
pip install -e ".[plot]"                         # matplotlib for saved plots
pip install -e /path/to/lfiax/agent-harness      # provides cli-anything-lfiax
pip install --upgrade "jax[cuda12]"              # GPU for LFIAX, recommended

# Plan A — SIR / LFIAX (GPU strongly recommended)
python examples/agent/sir_sequential_agent.py \
    --rounds 8 --seed 0 --artifacts artifacts/sir_seq

# Plan B — linear regression / Pyro (CPU is fine)
python examples/agent/linreg_sequential_agent.py \
    --rounds 8 --seed 0 --artifacts artifacts/linreg_seq
```

Outputs (per bundle) end up at:

```
artifacts/sir_seq/
  round_00/   posterior_samples.npy  eig_history.json  xi_history.json  observation.json
  round_07/   ...
  summary.json
  posterior_shrinkage.png
  eig_per_round.png
  design_on_curve.png

artifacts/linreg_seq/
  round_00/   posterior_samples.pt   eig_history.json  design_history.json  observation.json
  round_07/   ...
  summary.json
  posterior_shrinkage.png
  eig_per_round.png
  linreg_fit_overlay.png
```

---

## 9. Open decisions (flag before implementation)

1. **SIR parameterisation** — I'm planning `(beta, gamma)`; we could
   swap to `(R0, gamma)` if the posterior plots are easier to read.
2. **Observation noise model** — currently additive Gaussian on integer
   `I(xi)` counts; a more principled choice is `Binomial(N, I(xi)/N)` or
   Poisson. Sticking with Gaussian keeps the LFIAX surrogate simple.
3. **Importance-resampling vs SNPE-style refit** — round-to-round we
   importance-resample under the LFIAX likelihood. If particle degeneracy
   hits, swap to refitting a Gaussian-mixture or flow prior from the
   current weighted particles.
4. **Number of rounds** — 8 is a visually clean default; can be 4 or 16
   with no code change.
5. **Design-mode per round** — starting with `point` for simplicity;
   switch to `distribution` if the user wants to see the green design
   density from the reference plot reproduced as an annealed posterior
   predictive design distribution.
