# Sequential SIR Bundle (LFIAX-style)

Stochastic (Gillespie) SIR outbreak with two latent parameters
``(beta, gamma)``. Each BOED round picks one new measurement time
``xi_t > xi_{t-1}`` and observes noisy ``I(xi_t)`` on a single ground-
truth trajectory.

Files:

- ``problem.json`` — v1 problem bundle (`simulator_ref`,
  `prior_sampler_ref`, LFIAX candidate configuration).
- ``simulator.py`` — Gillespie trajectory generator plus the LFIAX-
  compatible ``simulate(theta, xi)`` callable.
- ``prior.py`` — truncated log-normal prior over ``(beta, gamma)`` and a
  particle-cloud resampler.
- ``monotone.py`` — ``xi_t > xi_{t-1}`` reparameterisation, shared with
  the linreg bundle.
- ``sequential_lfiax.py`` — per-round joint surrogate + design
  optimisation, LF-PCE EIG estimator, importance-reweight posterior
  update.
- ``plotting.py`` — posterior-shrinkage panels, per-round EIG, and the
  "designs on the curve" figure that reproduces the reference layout.

The in-repo path uses a small conditional-Gaussian likelihood
surrogate (``μ_φ(θ,ξ)`` and ``σ_φ(θ,ξ)`` from a 2-layer MLP) so the
whole demo runs in pure numpy. Swapping in the real LFIAX harness
(``cli-anything-lfiax``) is a surrogate-class-level change: replace
``_init_params`` / ``_forward`` / ``_neg_log_prob_grads`` with the
normalising-flow model and feed the same ``(θ, ξ, y)`` triplets.

Run the full loop via the driver:

```bash
python examples/agent/sir_sequential_agent.py --rounds 6 --seed 0 \
    --artifacts artifacts/sir_seq
```

Artifacts land under ``artifacts/sir_seq/``:

```
round_{t:02d}/   posterior_samples.npy  xi_history.json  eig_history.json  observation.json
summary.json
posterior_shrinkage.png
eig_per_round.png
design_on_curve.png
```
