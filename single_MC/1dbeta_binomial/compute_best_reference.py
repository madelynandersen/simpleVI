"""
Adds Stan's saved ADVI random restarts as candidates for beta_config.json's
"best variational approximation" (best_mean/best_std/best_method/best_elbo_np),
alongside the existing numpyro/pymc/tfp-derived candidate already recorded
there. Picks whichever single restart (from any method) scores highest,
matching the convention single_MC/logistic_regression/compute_best_reference.py
already uses -- and, like that script, routes the actual candidate scoring
through modulars.utils.find_best_params rather than a separate bespoke loop,
so this gets the same score_seed_by_restart=False fix (see modulars/utils.py):
every candidate is scored with the same seed, so the winner is decided by
genuine ELBO differences rather than by which candidate happened to get the
more favorable Monte Carlo draw.

Stan's random-restart pipeline (modulars/stan_rr_test.py) only saves the
constrained theta-space mean/std of each restart's posterior draws, not the
unconstrained (mu, sigma) of its internal mean-field Normal(mu, sigma) ->
sigmoid -> theta guide. Since beta_config.json's best_mean/best_std ARE that
unconstrained (mu, sigma) (see random_restarts/elbo_space_stan.ipynb), the
score_fn below inverts each Stan candidate's theta mean/std back to (mu,
sigma) by moment matching a LogitNormal via the same Gauss-Hermite quadrature
elbo_space_stan.ipynb uses, then scores with the same NumPyro
TraceMeanField_ELBO fixed-guide scorer so the comparison is apples to apples.
"""
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import TraceMeanField_ELBO
from scipy.optimize import fsolve
from scipy.special import expit, logit


sys.path.append(str(Path(__file__).resolve().parents[2]))

from modulars.distributions import binomial
from modulars.stan_rr_test import stan_result_tuple
from modulars.utils import find_best_params, load_config, save_best_run_to_config, save_to_csv


STAN_RESULT_PKL = "random_restarts/stan_cmdstan_output/stan_result.pkl"
STAN_PROCESSED_RESTARTS = "random_restarts/stan_processed_restarts.csv"

_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(100)


def _logitnormal_moments(mu, sigma):
    """Mean/std of theta = sigmoid(z), z ~ Normal(mu, sigma), via Gauss-Hermite quadrature."""
    z = mu + np.sqrt(2.0) * sigma * _GH_NODES
    theta = expit(z)
    mean = (_GH_WEIGHTS * theta).sum() / np.sqrt(np.pi)
    second_moment = (_GH_WEIGHTS * theta**2).sum() / np.sqrt(np.pi)
    std = np.sqrt(max(second_moment - mean**2, 0.0))
    return mean, std


def invert_logitnormal_moments(theta_mean, theta_std):
    """
    Find (mu, sigma) in unconstrained z-space whose LogitNormal mean/std match
    the given constrained theta_mean/theta_std, starting from the delta-method
    approximation and refining to an (almost) exact match via fsolve.
    """
    theta_mean = float(np.clip(theta_mean, 1e-6, 1 - 1e-6))
    theta_std = float(max(theta_std, 1e-6))

    mu0 = logit(theta_mean)
    sigma0 = max(theta_std / (theta_mean * (1 - theta_mean)), 1e-3)

    def residuals(params):
        mu, log_sigma = params
        mean, std = _logitnormal_moments(mu, np.exp(log_sigma))
        return [mean - theta_mean, std - theta_std]

    mu, log_sigma = fsolve(residuals, x0=[mu0, np.log(sigma0)])
    return float(mu), float(np.exp(log_sigma))


def make_beta_model(alpha_prior, beta_prior, n_samples):
    def model(y):
        theta = numpyro.sample("theta", dist.Beta(alpha_prior, beta_prior))
        numpyro.sample("obs", dist.Binomial(n_samples, theta), obs=y)

    return model


def make_fixed_guide(mu, sigma):
    mu = float(mu)
    sigma = max(float(sigma), 1e-8)

    def guide(y):
        base = dist.Normal(mu, sigma)
        numpyro.sample("theta", dist.TransformedDistribution(base, dist.transforms.SigmoidTransform()))

    return guide


def score_elbo(model, mu, sigma, y_data, grad_samps, seed):
    guide = make_fixed_guide(mu, sigma)
    elbo = TraceMeanField_ELBO(num_particles=int(grad_samps))
    loss = elbo.loss(jax.random.PRNGKey(int(seed)), {}, model, guide, y_data)
    return float(-loss)


def make_stan_score_fn(model, y_data):
    """
    find_best_params hands this the *raw theta-space* mean/std it loaded from
    the processed-restart file (Stan never reports its unconstrained (mu,
    sigma) -- see module docstring), so we invert before scoring.
    """

    def score_fn(mean, std, grad_samps, seed, **_kwargs):
        theta_mean = float(np.asarray(mean).reshape(-1)[0])
        theta_std = float(np.asarray(std).reshape(-1)[0])
        mu, sigma = invert_logitnormal_moments(theta_mean, theta_std)
        return score_elbo(model, mu, sigma, y_data, grad_samps, seed)

    return score_fn


def _prepare_stan_processed_restarts(base_dir):
    pkl_path = base_dir / STAN_RESULT_PKL
    if not pkl_path.exists():
        print(f"skipping missing stan result file: {pkl_path}")
        return None

    with open(pkl_path, "rb") as f:
        stan_result = pickle.load(f)

    # find_best_params' generic processed-restart-payload reader expects
    # (restarts, iters, params); beta is a 1-param model, and
    # stan_result_tuple squeezes away that trailing size-1 param axis, so we
    # add it back here.
    arrays = tuple(np.asarray(arr)[..., None] for arr in stan_result_tuple(stan_result))

    processed_path = base_dir / STAN_PROCESSED_RESTARTS
    save_to_csv(processed_path, [arrays])
    return processed_path


def main(grad_samps=20_000, seed=20240512):
    base_dir = Path(__file__).resolve().parent
    config_file = base_dir / "beta_config.json"
    config = load_config(config_file)

    alpha_prior = float(config["alpha_prior"])
    beta_prior = float(config["beta_prior"])
    n_samples = int(config["n_samples"])
    theta_like = float(config["theta_like"])

    y = int(binomial(n_samples, theta_like))
    y_data = jnp.array(y)
    model = make_beta_model(alpha_prior, beta_prior, n_samples)

    candidates = []

    current_mean = float(config["best_mean"])
    current_std = float(config["best_std"])
    current_elbo = score_elbo(model, current_mean, current_std, y_data, grad_samps, seed)
    candidates.append(
        {
            "method": config.get("best_method", "numpyro"),
            "source": "beta_config.json (existing best)",
            "restart_idx": config.get("best_restart_idx"),
            "mean": current_mean,
            "std": current_std,
            "elbo": current_elbo,
        }
    )

    stan_processed_path = _prepare_stan_processed_restarts(base_dir)
    if stan_processed_path is not None:
        stan_best = find_best_params(
            file_name_list=[str(stan_processed_path)],
            alpha_prior=None,
            obs_counts=y_data,
            grad_samps=grad_samps,
            seed=seed,
            score_fn=make_stan_score_fn(model, y_data),
            result_specs=[("stan", str(stan_processed_path))],
            score_seed_by_restart=False,
        )
        for run in sorted(stan_best["all_scored_runs"], key=lambda r: -r["best_elbo_np"])[:5]:
            print(
                f"  stan/{run.get('mc_setting')}/restart{run['restart_idx']:<2d} "
                f"elbo={run['best_elbo_np']:+.5f}  theta_mean={float(run['mean'][0]):.4f}  "
                f"theta_std={float(run['std'][0]):.4f}"
            )
        mu, sigma = invert_logitnormal_moments(
            float(stan_best["best_mean"][0]), float(stan_best["best_std"][0])
        )
        candidates.append(
            {
                "method": "stan",
                "source": f"{STAN_RESULT_PKL} ({stan_best['mc_setting']})",
                "restart_idx": stan_best["restart_idx"],
                "mean": mu,
                "std": sigma,
                "elbo": stan_best["best_elbo_np"],
            }
        )

    for c in sorted(candidates, key=lambda c: -c["elbo"]):
        print(f"{c['method']:>8s}  elbo={c['elbo']:+.5f}  mean={c['mean']:+.4f}  std={c['std']:.4f}  ({c['source']})")

    best = max(candidates, key=lambda c: c["elbo"])
    print(f"\nbest candidate: {best['method']} (elbo={best['elbo']:+.5f})")

    best_info = {
        "best_mean": np.asarray([best["mean"]]),
        "best_cov": np.asarray([[best["std"] ** 2]]),
        "best_std": np.asarray([best["std"]]),
        "best_elbo_np": best["elbo"],
        "source_file": best["source"],
        "restart_idx": best["restart_idx"],
        "final_elbo": None,
    }
    save_best_run_to_config(
        config_file,
        {
            **best_info,
            "best_mean": best["mean"],
            "best_cov": best["std"] ** 2,
            "best_std": best["std"],
        },
        extra_updates={"best_method": best["method"]},
    )
    print(f"updated {config_file}")


if __name__ == "__main__":
    main()
