# Sequential Linear-Regression Bundle (Pyro path)

Classic ``y = a * xi + b + Normal(0, sigma_obs)`` linear regression
used as a clean reference case for sequential BOED. The model is
**explicit** (closed-form likelihood), so on a machine with Pyro
installed the waterfall in ``SimulatorChoiceModule`` routes this to
the Pyro backend and the design optimisation goes through
``pyro.contrib.oed.eig.posterior_eig`` with an amortised full-rank
bivariate-normal posterior guide over ``(a, b) | y``. This is the
exact variational family for a Gaussian linear model, so the
posterior-entropy (APE) EIG estimator becomes tight in the
asymptotic limit — usually lower variance than the marginal form.

Files:

- ``problem.json`` — v1 problem bundle, explicit candidate, Pyro
  backend configured with ``posterior_eig`` + amortised bivariate
  normal guide (with ``marginal_eig`` + ``AutoDiagonalNormal`` kept
  as a legacy comparison).
- ``model.py`` — two equivalent implementations:
    * a **Pyro reference** (``model`` / ``fit_svi`` /
      ``optimize_design_pyro`` (posterior_eig) /
      ``optimize_design_pyro_marginal`` (marginal_eig, comparison))
      that targets ``pyro-ppl`` installed in the user's environment,
    * a **numpy conjugate fallback** (``GaussianLinearModel``). The
      Gaussian-Gaussian linear model has closed-form posteriors and
      an analytic EIG, and this is what the in-repo driver actually
      runs — the math is identical to what Pyro estimates in the
      infinite-sample limit.
- ``sequential_pyro.py`` — per-round loop: design optimisation,
  ground-truth observation, posterior update.
- ``plotting.py`` — posterior shrinkage, per-round EIG, and the
  predictive-overlay figure.
- ``monotone.py`` — re-export of the shared monotone reparam.
  **Not used for linreg** (xi is a coordinate, not a time), but kept
  importable for consistency with the SIR bundle.

Run via the driver:

```bash
python examples/agent/linreg_sequential_agent.py --rounds 6 --seed 0 \
    --artifacts artifacts/linreg_seq
```

Artifacts:

```
round_{t:02d}/   posterior_samples.npy  eig_history.json  design_history.json  observation.json
summary.json
posterior_shrinkage.png
eig_per_round.png
linreg_fit_overlay.png
```

On the user's machine, switch from the numpy conjugate path to the
Pyro path by calling ``model.fit_svi`` and ``model.optimize_design_pyro``
from a thin wrapper — same inputs and outputs, just routed through
``pyro.contrib.oed`` and ``SVI``.
