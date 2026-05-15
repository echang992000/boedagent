"""Linear regression model for the Pyro sequential BOED path.

Two equivalent implementations are provided:

* :func:`model` — Pyro version with vague Gaussian priors on slope and
  intercept. This is what the user will import when running on a
  machine that has ``pyro-ppl`` installed. Fits via ``SVI`` /
  ``AutoDiagonalNormal`` and optimises the per-round design via
  ``pyro.contrib.oed.eig.posterior_eig`` (the posterior / APE
  estimator ``EIG = H[p(θ)] − APE(ξ)`` from Foster et al. 2019). The
  amortised posterior guide is a conditional bivariate normal over
  ``(a, b) | y``, which is the correct variational family for this
  linear-Gaussian model — so the posterior EIG estimator is exact in
  the asymptotic limit. A ``marginal_eig`` reference version is also
  provided in :func:`optimize_design_pyro_marginal` for comparison.
* :class:`GaussianLinearModel` — numpy closed-form conjugate update.
  For Gaussian priors + Gaussian observation noise, the posterior is
  also Gaussian and the expected information gain at a new design
  ``xi`` reduces to a closed-form scalar. This is what the in-repo
  demo runs end-to-end without torch.

Both paths encode the same forward model::

    y = a * xi + b + Normal(0, sigma_obs)
    a ~ Normal(prior_mean_a, prior_std_a)
    b ~ Normal(prior_mean_b, prior_std_b)

with default vague priors ``Normal(0, 3)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import math
import numpy as np


# -----------------------------------------------------------------------------
# Pyro implementation (not executed in-sandbox; kept for the user's env)
# -----------------------------------------------------------------------------


def model(xi, y_obs=None, *, prior_means=(0.0, 0.0), prior_stds=(3.0, 3.0), sigma_obs=0.1):
    """Pyro model for ``y = a * xi + b + Normal(0, sigma_obs)``.

    Called during SVI (with ``y_obs`` observed) and during design
    optimisation (with ``y_obs=None`` so Pyro samples it).
    """
    import pyro
    import pyro.distributions as dist
    import torch

    xi = xi if isinstance(xi, torch.Tensor) else torch.as_tensor(xi, dtype=torch.float32)
    a = pyro.sample("a", dist.Normal(float(prior_means[0]), float(prior_stds[0])))
    b = pyro.sample("b", dist.Normal(float(prior_means[1]), float(prior_stds[1])))
    with pyro.plate("obs", xi.shape[0]):
        mean = a * xi + b
        return pyro.sample("y", dist.Normal(mean, float(sigma_obs)), obs=y_obs)


def fit_svi(
    *,
    xi: "torch.Tensor",
    y: "torch.Tensor",
    prior_means,
    prior_stds,
    sigma_obs: float,
    num_steps: int = 500,
    lr: float = 1e-2,
    num_posterior_samples: int = 5000,
):
    """Reference SVI fit — returns posterior samples and a summary.

    Requires ``pyro-ppl``; raises ``ImportError`` cleanly otherwise.
    """
    import pyro
    from pyro.infer import SVI, Predictive, Trace_ELBO
    from pyro.infer.autoguide import AutoDiagonalNormal

    pyro.clear_param_store()

    def _model(xi_, y_obs_=None):
        return model(
            xi_,
            y_obs=y_obs_,
            prior_means=prior_means,
            prior_stds=prior_stds,
            sigma_obs=sigma_obs,
        )

    guide = AutoDiagonalNormal(_model)
    svi = SVI(_model, guide, pyro.optim.Adam({"lr": lr}), loss=Trace_ELBO())
    losses = []
    for _ in range(num_steps):
        loss = svi.step(xi, y)
        losses.append(float(loss))

    predictive = Predictive(_model, guide=guide, num_samples=num_posterior_samples)
    draws = predictive(xi, None)
    return {
        "losses": losses,
        "a_samples": draws["a"].detach().cpu().numpy(),
        "b_samples": draws["b"].detach().cpu().numpy(),
    }


def _posterior_guide_factory(prior_means, prior_stds):
    """Build an amortised posterior guide ``q(a, b | y, ξ)`` for posterior_eig.

    Returns a callable with the Pyro OED signature
    ``guide(y_dict, design, observation_labels, target_labels)``.
    The guide is a full-rank bivariate normal over ``(a, b)`` whose
    mean is an affine function of the observation ``y``. This is the
    exact family for the linear-Gaussian model, so the posterior EIG
    estimator becomes the true EIG in the asymptotic limit.

    Initial scale_tril is set to the prior std so the guide doesn't
    start with a degenerate / over-tight posterior and the optimiser
    has a well-conditioned loss surface.
    """

    import torch
    import pyro
    import pyro.distributions as dist

    prior_std_a = float(prior_stds[0])
    prior_std_b = float(prior_stds[1])
    prior_mean_a = float(prior_means[0])
    prior_mean_b = float(prior_means[1])

    def guide(y_dict, design, observation_labels, target_labels):
        # Flatten y to a scalar-per-sample view. The sPCE / posterior
        # estimator calls this with ``y`` shaped like the expanded
        # design, i.e. (num_particles, 1).
        y = y_dict["y"]
        y_flat = y.reshape(*y.shape[:-1]) if y.dim() >= 1 else y

        # Amortised mean: (a, b) = affine(y). Initialise at the prior
        # mean so the first-step gradients are well-behaved.
        bias = pyro.param(
            "guide_bias_ab",
            torch.tensor([prior_mean_a, prior_mean_b], dtype=torch.float32),
        )
        slope = pyro.param(
            "guide_slope_ab",
            torch.tensor([0.0, 0.0], dtype=torch.float32),
        )
        scale_tril = pyro.param(
            "guide_scale_tril_ab",
            torch.tensor(
                [[prior_std_a, 0.0], [0.0, prior_std_b]], dtype=torch.float32
            ),
            constraint=dist.constraints.lower_cholesky,
        )

        mean_ab = bias[None, :] + slope[None, :] * y_flat[..., None]  # (..., 2)

        # Sample a and b jointly through a single MVN, then split via
        # pyro.sample on each name so the conditioning on "a", "b" in
        # the posterior_eig inner loop lines up.
        ab = pyro.sample(
            "_ab_joint",
            dist.MultivariateNormal(mean_ab, scale_tril=scale_tril),
            infer={"is_auxiliary": True},
        )
        pyro.sample("a", dist.Delta(ab[..., 0]))
        pyro.sample("b", dist.Delta(ab[..., 1]))

    return guide


def optimize_design_pyro(
    *,
    prior_means,
    prior_stds,
    sigma_obs: float,
    xi_prev: float,
    x_max: float,
    num_eig_steps: int = 200,
    num_posterior_samples: int = 256,
    num_guide_steps: int = 200,
    lr: float = 5e-2,
    guide_lr: float = 5e-2,
):
    """Reference per-round design optimisation using ``pyro.contrib.oed.eig.posterior_eig``.

    Uses the amortised posterior guide ``q(a, b | y, ξ)`` (full-rank
    bivariate normal with an affine mean in ``y``) — exact for this
    linear-Gaussian model in the asymptotic limit. The outer loop
    optimises the monotone raw ∈ ℝ → ξ reparam via Adam on ``-EIG``.

    Lives here as a reference; not exercised inside the sandbox because
    pyro isn't installable in the test environment.
    """
    import torch
    import pyro
    import pyro.optim as pyro_optim
    from pyro.contrib.oed.eig import posterior_eig

    from .monotone import next_xi_from_raw_torch

    raw = torch.tensor(0.0, requires_grad=True)
    optim = torch.optim.Adam([raw], lr=lr)
    history = []

    def _model_fn(design):
        return model(
            design,
            y_obs=None,
            prior_means=prior_means,
            prior_stds=prior_stds,
            sigma_obs=sigma_obs,
        )

    guide_fn = _posterior_guide_factory(prior_means, prior_stds)

    for step in range(num_eig_steps):
        optim.zero_grad()
        # Fresh guide params per outer step so the inner optimisation
        # learns the correct q(θ|y, ξ) for THIS ξ rather than reusing
        # stale params from a previous design.
        pyro.clear_param_store()
        xi_t = next_xi_from_raw_torch(raw, xi_prev=xi_prev, t_max=x_max).reshape(1)
        eig = posterior_eig(
            _model_fn,
            xi_t,
            "y",
            ["a", "b"],
            num_samples=num_posterior_samples,
            num_steps=num_guide_steps,
            guide=guide_fn,
            optim=pyro_optim.Adam({"lr": guide_lr}),
            final_num_samples=num_posterior_samples * 2,
        )
        loss = -eig
        loss.backward()
        optim.step()
        history.append({"step": step, "xi": float(xi_t.detach()), "eig": float(eig.detach())})

    best = max(history, key=lambda h: h["eig"])
    return best, history


def optimize_design_pyro_marginal(
    *,
    prior_means,
    prior_stds,
    sigma_obs: float,
    xi_prev: float,
    x_max: float,
    num_eig_steps: int = 200,
    num_marginal_samples: int = 256,
    lr: float = 5e-2,
):
    """Legacy ``marginal_eig`` version — kept for comparison with ``posterior_eig``.

    Uses ``pyro.contrib.oed.eig.marginal_eig`` with an ``AutoDiagonalNormal``
    guide on the marginal ``q(y|ξ)``. For the linear-Gaussian model
    both estimators converge to the same EIG, but the posterior form
    typically gives lower variance because the amortised q(θ|y, ξ)
    guide is closer to the true posterior than the amortised q(y|ξ)
    is to the true marginal.
    """
    import torch
    from pyro.contrib.oed.eig import marginal_eig

    from .monotone import next_xi_from_raw_torch

    raw = torch.tensor(0.0, requires_grad=True)
    optim = torch.optim.Adam([raw], lr=lr)
    history = []

    def _model_fn(design):
        return model(
            design,
            y_obs=None,
            prior_means=prior_means,
            prior_stds=prior_stds,
            sigma_obs=sigma_obs,
        )

    for step in range(num_eig_steps):
        optim.zero_grad()
        xi_t = next_xi_from_raw_torch(raw, xi_prev=xi_prev, t_max=x_max).reshape(1)
        eig = marginal_eig(
            _model_fn,
            xi_t,
            "y",
            ["a", "b"],
            num_samples=num_marginal_samples,
            num_steps=50,
            final_num_samples=num_marginal_samples * 2,
        )
        loss = -eig
        loss.backward()
        optim.step()
        history.append({"step": step, "xi": float(xi_t.detach()), "eig": float(eig.detach())})

    best = max(history, key=lambda h: h["eig"])
    return best, history


# -----------------------------------------------------------------------------
# Numpy conjugate path (this is what runs in-sandbox and in-CI)
# -----------------------------------------------------------------------------


@dataclass
class GaussianLinearModel:
    """Closed-form Gaussian linear regression.

    Maintains the Gaussian posterior over (a, b) as a mean vector and a
    2x2 covariance. All BOED quantities have analytic forms.
    """

    prior_mean: np.ndarray  # shape (2,)
    prior_cov: np.ndarray  # shape (2, 2)
    sigma_obs: float

    def posterior_given(self, xs: np.ndarray, ys: np.ndarray) -> "GaussianLinearModel":
        """Conjugate posterior over (a, b) given observations (xs, ys)."""
        xs = np.asarray(xs, dtype=float).reshape(-1)
        ys = np.asarray(ys, dtype=float).reshape(-1)
        if xs.size == 0:
            return self
        # Design matrix Φ = [x, 1] per row.
        Phi = np.stack([xs, np.ones_like(xs)], axis=1)
        prior_prec = np.linalg.inv(self.prior_cov)
        post_prec = prior_prec + (Phi.T @ Phi) / (self.sigma_obs ** 2)
        post_cov = np.linalg.inv(post_prec)
        post_mean = post_cov @ (prior_prec @ self.prior_mean + Phi.T @ ys / (self.sigma_obs ** 2))
        return GaussianLinearModel(prior_mean=post_mean, prior_cov=post_cov, sigma_obs=self.sigma_obs)

    def eig_at(self, xi: float) -> float:
        """Expected information gain at a single new design xi.

        For a Gaussian linear model the EIG reduces to a log-ratio of
        predictive variances:

            EIG(xi) = 0.5 * log( 1 + φ^T Σ φ / σ_obs^2 )

        with ``φ = [xi, 1]`` and ``Σ`` the current posterior cov on
        (a, b). This is the exact sPCE limit as L → ∞ and also what
        `pyro.contrib.oed.eig.marginal_eig` estimates stochastically.
        """
        phi = np.array([float(xi), 1.0])
        var = phi @ self.prior_cov @ phi
        return 0.5 * math.log(1.0 + var / (self.sigma_obs ** 2))

    def predictive(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xi = np.asarray(xi, dtype=float).reshape(-1)
        Phi = np.stack([xi, np.ones_like(xi)], axis=1)
        mean = Phi @ self.prior_mean
        var = np.einsum("ij,jk,ik->i", Phi, self.prior_cov, Phi) + self.sigma_obs ** 2
        return mean, np.sqrt(var)

    def sample_posterior(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        L = np.linalg.cholesky(self.prior_cov)
        z = rng.standard_normal((num_samples, 2))
        return self.prior_mean + z @ L.T


def default_prior_model(sigma_obs: float = 0.1) -> GaussianLinearModel:
    return GaussianLinearModel(
        prior_mean=np.zeros(2),
        prior_cov=np.diag([3.0 ** 2, 3.0 ** 2]),
        sigma_obs=sigma_obs,
    )


if __name__ == "__main__":
    m = default_prior_model()
    rng = np.random.default_rng(0)
    theta_true = np.array([1.5, -0.2])  # a_true, b_true
    xs = rng.uniform(-1.0, 1.0, size=12)
    ys = theta_true[0] * xs + theta_true[1] + 0.1 * rng.standard_normal(12)
    post = m.posterior_given(xs, ys)
    print("posterior mean:", np.round(post.prior_mean, 3), "std:", np.round(np.sqrt(np.diag(post.prior_cov)), 3))
    print("EIG at xi=0.5:", round(post.eig_at(0.5), 4))
