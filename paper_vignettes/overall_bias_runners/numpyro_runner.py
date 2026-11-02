"""NumPyro random restarts for the overall bias comparison (run in the num_pyro env).

Each model's setup below is copied from its single_MC random-restart notebook
(named in the function's docstring). The only intended changes are that
n_iters comes from the command line and the run_restart_* helpers are given
`track_start` so only the last `window` iterations are recorded (every
iteration). The two NumPyro setups are the two runs every run_restart_* call
makes: 1 MC sample (PRNGKey(seed)) and 100 MC samples (PRNGKey(seed + 1000)),
both with the helpers' Adam(step_size=5e-4) default.

--full-default 1 runs only the 1 MC sample fit, for 30,000 iterations unless --max-iters is given, saved as setup
"full_default" to overall_bias_results_full_default/ with each seed's fit time (overall_default_bias_comparisons.ipynb).

usage: python numpyro_runner.py --model gamma_poisson [--max-iters 1000000 --n-seeds 20 --window 2500 --overwrite 0 --full-default 0]
"""

import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bias_io  # noqa: E402

sys.path.insert(0, str(bias_io.REPO_ROOT))
SINGLE_MC = bias_io.REPO_ROOT / "single_MC"
PACKAGE = "numpyro"
args = bias_io.parse_runner_args(PACKAGE)

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

from modulars.utils import (  # noqa: E402
    load_config, load_best_values, exp_lognormal_moments, logistic_moments,
    apply_traj_transform, apply_traj_transform_multid,
)
from modulars.numpyro_rr_test import run_restart_1d, run_restart_multid, run_restart_multid_autonormal  # noqa: E402

max_iters = args.max_iters
TRACK_START = bias_io.track_start(args.max_iters, args.window)
ITERATIONS = bias_io.window_iterations(args.max_iters, args.window)
SEEDS = list(range(args.n_seeds))
# --full-default 1: only the 1 MC sample fit (Adam(step_size=5e-4), PRNGKey(seed)), saved as setup "full_default"
FULL_DEFAULT = args.full_default
RUN_MULTI = not FULL_DEFAULT


def run_all_seeds(run_one_seed, seeds=SEEDS):
    """run_one_seed(seed) -> (single_loc, single_scale, multi_loc, multi_scale), cached per seed."""
    def one_seed(seed):
        result = run_one_seed(seed)
        gc.collect()
        return [np.asarray(r) for r in result]
    results = bias_io.run_seeds_cached(args, PACKAGE, seeds, one_seed)
    if FULL_DEFAULT:   # only the 1 MC fit ran: repeat it as the 100 MC slot so the processing below is unchanged
        results = [list(r) + list(r) for r in results]
    return results


def setups_from_processed(processed):
    single_means, single_stds, multi_means, multi_stds = processed
    if FULL_DEFAULT:
        return {bias_io.FULL_DEFAULT_SETUP: (single_means, single_stds)}
    return {"default": (single_means, single_stds), "100mc": (multi_means, multi_stds)}


"""
Model setups, copied from the single_MC random-restart notebooks
"""
def model_1d_gaussian():
    """single_MC/1dgaussian_knownvar/random_restarts/rr_numpyro.ipynb"""
    from modulars import gaussian_1d_posterior, gaussian_1d
    os.chdir(SINGLE_MC / "1dgaussian_knownvar" / "random_restarts")

    mu_prior, sigma_prior = 0, 10
    sigma_like = 1.0

    # no observations, as in 3_1_numpyro.ipynb and rr_stan.ipynb, so the posterior is the prior
    DATA = False
    if DATA:
        n_samples = 100
        mu_like = 3
        data = gaussian_1d(n_samples, mu_like, sigma_like, seed=0)
        true_mu_post, true_sigma_post = gaussian_1d_posterior(data, mu_prior=mu_prior, sigma_prior=sigma_prior, sigma_like=sigma_like)
    else:
        data = jnp.array([])
        true_mu_post, true_sigma_post = mu_prior, sigma_prior

    def model(y):
        mu = numpyro.sample("mu", dist.Normal(mu_prior, sigma_prior))
        numpyro.sample("obs", dist.Normal(mu, sigma_like), obs=y)

    results = run_all_seeds(lambda seed: run_restart_1d(
        seed, model, data, 'mu', max_iters, n_particles=100, track_start=TRACK_START, run_multi=RUN_MULTI))
    setups = setups_from_processed(apply_traj_transform(results, n_samples=10_000, seed=1, NOTEBOOK=False))
    return {"1d_gaussian": dict(setups=setups, coordinates=["mu"], best_mean=true_mu_post, best_std=true_sigma_post,
                                reference_source="prior (no observations, as 3_1_numpyro.ipynb / rr_stan.ipynb)")}


def model_beta_binomial():
    """single_MC/1dbeta_binomial/random_restarts/rr_numpyro_beta.ipynb"""
    from modulars.distributions import binomial
    os.chdir(SINGLE_MC / "1dbeta_binomial" / "random_restarts")
    trace_fn = logistic_moments
    transform_fn = logistic_moments
    data_gen = binomial

    config_file = "../beta_config.json"
    config = load_config(config_file)

    theta_like = config['theta_like']
    alpha_prior = config['alpha_prior']
    beta_prior = config['beta_prior']
    n_samples = config['n_samples']

    data = data_gen(n_samples) if n_samples > 0 else 0

    def model(y):
        theta = numpyro.sample("theta", dist.Beta(alpha_prior, beta_prior))
        numpyro.sample("obs", dist.Binomial(n_samples, theta), obs=y)

    results = run_all_seeds(lambda seed: run_restart_1d(
        seed, model, data, 'theta', max_iters, n_particles=100, track_start=TRACK_START, run_multi=RUN_MULTI))
    setups = setups_from_processed(apply_traj_transform(results, transform_fn, n_samples=100_000, seed=1, NOTEBOOK=False))
    # If we have "best" from config, these should be on theta-scale (0,1)
    best_mu, best_std = load_best_values(
        config=config, transform=trace_fn,
        n_samples=100_000, seed=1)
    return {"beta_binomial": dict(setups=setups, coordinates=["theta"], best_mean=best_mu, best_std=best_std,
                                  reference_source="beta_config.json via load_best_values(logistic_moments, n_samples=100_000, seed=1)")}


def model_gamma_exponential():
    """single_MC/gamma_exponential/random_restarts/rr_numpyro_gam_exp.ipynb"""
    from modulars import exponential
    os.chdir(SINGLE_MC / "gamma_exponential" / "random_restarts")
    transform_fn = exp_lognormal_moments

    config_file = '../exp_config.json'
    config = load_config(config_file)

    lambda_like = config['lambda_like']
    alpha_prior = config['alpha_prior']
    beta_prior = config['beta_prior']
    n_samples = config['n_samples']
    data = exponential(n_samples, lambda_like)

    def model(y):
        lambd = numpyro.sample('lambd', dist.Gamma(alpha_prior, beta_prior))
        numpyro.sample('y', dist.Exponential(lambd), obs=y)

    results = run_all_seeds(lambda seed: run_restart_1d(
        seed, model, data, 'lambd', max_iters, n_particles=100, track_start=TRACK_START, run_multi=RUN_MULTI))
    setups = setups_from_processed(apply_traj_transform(results, transform_fn, n_samples=10_000, seed=1, NOTEBOOK=False))
    # exact best mean-field (log-normal) approximation of the conjugate Gamma posterior
    from modulars.distributions import gamma_exponential_posterior, gamma_lognormal_optimum
    m_star, s_star = gamma_lognormal_optimum(*gamma_exponential_posterior(data, alpha_prior, beta_prior))
    best_mu, best_std = exp_lognormal_moments(m_star, s_star)
    return {"gamma_exponential": dict(setups=setups, coordinates=["lambda"], best_mean=best_mu, best_std=best_std,
                                      reference_source="closed form: gamma_lognormal_optimum(gamma_exponential_posterior(data))")}


def model_gamma_poisson():
    """single_MC/gamma_poisson/random_restarts/rr_numpyro_gam_pois.ipynb"""
    from modulars import poisson
    os.chdir(SINGLE_MC / "gamma_poisson" / "random_restarts")
    transform_fn = exp_lognormal_moments

    config_file = '../poisson_config.json'
    config = load_config(config_file)

    lambda_like = config['lambda_like']
    alpha_prior = config['alpha_prior']
    beta_prior = config['beta_prior']
    n_samples = config['n_samples']
    data = poisson(n_samples, lambda_like)

    def model(y):
        lambd = numpyro.sample('lambd', dist.Gamma(alpha_prior, beta_prior))
        numpyro.sample('y', dist.Poisson(lambd), obs=y)

    results = run_all_seeds(lambda seed: run_restart_1d(
        seed, model, data, 'lambd', max_iters, n_particles=100, track_start=TRACK_START, run_multi=RUN_MULTI))
    setups = setups_from_processed(apply_traj_transform(results, transform_fn, n_samples=10_000, seed=1, NOTEBOOK=False))
    # exact best mean-field (log-normal) approximation of the conjugate Gamma posterior
    from modulars.distributions import gamma_poisson_posterior, gamma_lognormal_optimum
    m_star, s_star = gamma_lognormal_optimum(*gamma_poisson_posterior(data, alpha_prior, beta_prior))
    best_mu, best_std = exp_lognormal_moments(m_star, s_star)
    return {"gamma_poisson": dict(setups=setups, coordinates=["lambda"], best_mean=best_mu, best_std=best_std,
                                  reference_source="closed form: gamma_lognormal_optimum(gamma_poisson_posterior(data))")}


def model_inverse_gamma():
    """single_MC/invgamma_gaussianknownmean/random_restarts/rr_numpyro_invgam.ipynb"""
    from modulars.distributions import gaussian_1d
    os.chdir(SINGLE_MC / "invgamma_gaussianknownmean" / "random_restarts")

    config_file = '../invgamma_config.json'
    config = load_config(config_file)

    mu_like = config['mu_like']
    sigma_like = config['sigma_like']
    alpha_prior = config['alpha_prior']
    beta_prior = config['beta_prior']
    n_samples = config['n_samples']

    DATA = True
    if DATA:
        data = gaussian_1d(n_samples, mu_like, sigma_like=sigma_like).astype(np.float32)
    else:
        data = []

    def model(y):
        sigmasq = numpyro.sample("sigma_sq", dist.InverseGamma(alpha_prior, beta_prior))
        numpyro.sample('y', dist.Normal(mu_like, jnp.sqrt(sigmasq)), obs=y)

    results = run_all_seeds(lambda seed: run_restart_1d(
        seed, model, data, 'sigma_sq', max_iters, n_particles=100, track_start=TRACK_START, run_multi=RUN_MULTI))
    setups = setups_from_processed(apply_traj_transform(results, transform_fn=exp_lognormal_moments,
                                                        n_samples=10_000, seed=1, NOTEBOOK=False))
    best_mu, best_std = load_best_values(
        config=config, transform=exp_lognormal_moments,
        n_samples=50_000, seed=1)
    return {"inverse_gamma": dict(setups=setups, coordinates=["sigmasq"], best_mean=best_mu, best_std=best_std,
                                  reference_source="invgamma_config.json via load_best_values(exp_lognormal_moments, n_samples=50_000, seed=1)")}


def model_40d_gaussian():
    """single_MC/multidim_gaussian_knowncovar/random_restarts/rr_numpyro_multid_gaussian.ipynb"""
    from modulars.distributions import gaussian_multid, gaussian_multid_posterior, ensure_arr
    from modulars.utils import check_all_int_or_float_or_1darrays
    os.chdir(SINGLE_MC / "multidim_gaussian_knowncovar" / "random_restarts")

    # this is 40 dimensions
    mu_prior = np.zeros(40)
    sigma_prior = np.ones(40) * 10  # diag std of prior
    mu_like = 2.0
    sigma_like = 1.0  # std of likelihood
    n_obs = 3
    dim = 40
    np.random.seed(42)

    DIAG = True

    # Generate observations for each dimension
    observed_data = gaussian_multid(
        seed=42, n_samples=n_obs, mu_like=mu_like,
        scale_like=sigma_like, dim=dim)
    # compute known posterior,this is the posterior
    # diagonal standard deviation vector as true_sigma_post
    true_mu_post, true_sigma_post = gaussian_multid_posterior(
        observed_data, n_samples=n_obs,
        mu_prior=mu_prior, scale_prior=sigma_prior,
        scale_like=1.0, std_devs=True,
        diag=DIAG, dim=40)

    # this runs the easiest case where all dimensions are i.i.d.
    assert DIAG, "this code only works for diagonal covariance matrices"
    assert check_all_int_or_float_or_1darrays(mu_prior, sigma_prior, mu_like, sigma_like), "all parameters must be floats or 1d arrays"

    def model(y=None):
        mu = numpyro.sample('mu',
            dist.MultivariateNormal(
                loc=ensure_arr(mu_prior, dim),
                covariance_matrix=jnp.diag(ensure_arr(sigma_prior**2, dim)))
        )
        numpyro.sample('z',
            dist.MultivariateNormal(
                loc=mu,
                covariance_matrix=jnp.diag(ensure_arr(sigma_like**2, dim))),
            obs=y
        )

    results = run_all_seeds(lambda seed: run_restart_multid(
        seed, model, observed_data, 'mu', max_iters,
        dim=40,
        track_start=TRACK_START, run_multi=RUN_MULTI,
    ))
    setups = setups_from_processed(apply_traj_transform_multid(results, NOTEBOOK=False))
    return {"40d_gaussian": dict(setups=setups, coordinates=[f"mu[{i}]" for i in range(dim)],
                                 best_mean=true_mu_post, best_std=true_sigma_post,
                                 reference_source="gaussian_multid_posterior (analytic posterior std)")}


def model_logistic_regression():
    """single_MC/logistic_regression/rr_numpyro_logistic_regression.ipynb"""
    import pandas as pd
    from modulars.logistic_regression import build_numpyro_logistic_regression_model, load_logistic_regression_data
    os.chdir(SINGLE_MC / "logistic_regression")

    data = load_logistic_regression_data()
    labels = data["labels"]
    dim = len(labels)
    n_particles = 100
    track_every = 1

    model = build_numpyro_logistic_regression_model()
    model_args = (data["x_train"], data["y_train"])
    site_names = ("b",)

    results = run_all_seeds(lambda seed: run_restart_multid_autonormal(
        seed=seed,
        model=model,
        data=None,
        param_name=None,
        n_iters=max_iters,
        dim=dim,
        n_particles=n_particles,
        model_args=model_args,
        site_names=site_names,
        track_every=track_every,
        track_start=TRACK_START, run_multi=RUN_MULTI,
    ))
    setups = setups_from_processed(apply_traj_transform_multid(results, NOTEBOOK=False))

    reference_file = Path("best_variational_reference.csv")
    if not reference_file.exists():
        raise FileNotFoundError(
            "Run compute_best_reference.py after generating processed restart files."
        )
    best_reference = pd.read_csv(reference_file)
    assert list(best_reference["parameter"]) == list(labels), "reference rows must be in label order"
    best_mean = best_reference["mean"].to_numpy()
    best_std = best_reference["sd"].to_numpy()
    return {"logistic_regression": dict(setups=setups, coordinates=list(labels), best_mean=best_mean, best_std=best_std,
                                        reference_source="single_MC/logistic_regression/best_variational_reference.csv")}


def model_dirichlet_multinomial():
    """single_MC/multinomial_dirichlet/rr_dirich_numpyro.ipynb"""
    from modulars.distributions import gen_multinomial_data_shared
    from modulars.utils import load_best_multid_reference, stickbreaking_simplex_forward, apply_traj_transform_simplex
    os.chdir(SINGLE_MC / "multinomial_dirichlet")

    config_file = "dirichlet_config.json"
    config = load_config(config_file)

    theta_like = np.array(config["theta_like"], dtype=np.float32)
    N = config["N"]
    alpha_prior = np.array(config["alpha_prior"], dtype=np.float32)
    n_cats = config["n_cats"]
    total_count = config["total_count"]
    seed = config.get("seed", 15)

    n_vec = np.full(N, total_count, int)
    obs_counts = gen_multinomial_data_shared(theta_like, N, total_count, SEED=seed)
    best_mean, best_cov, best_std = load_best_multid_reference(config)

    def model(obs_counts, n_vec, alpha_prior):
        """NumPyro model for Dirichlet-Multinomial"""
        # Sample theta from Dirichlet prior
        theta = numpyro.sample(
            'theta',
            dist.Dirichlet(concentration=jnp.array(alpha_prior))
        )

        # Observe multinomial data
        with numpyro.plate('observations', len(obs_counts)):
            numpyro.sample('y',
                dist.Multinomial(
                    total_count=n_vec,
                    probs=theta
                ),
                obs=obs_counts
            )

    results = run_all_seeds(lambda rr_seed: run_restart_multid(
        rr_seed,
        model,
        data=None,
        param_name="theta",
        n_iters=max_iters,
        dim=n_cats - 1,
        n_particles=100,
        model_args=(obs_counts, n_vec, alpha_prior),
        track_start=TRACK_START, run_multi=RUN_MULTI,
    ))
    single_means, single_stds, multi_means, multi_stds = apply_traj_transform_simplex(
        results,
        n_cats=n_cats,
        forward_fn=stickbreaking_simplex_forward,
        n_samples=50_000,
        seed=1,
        NOTEBOOK=False,
    )
    setups = setups_from_processed((single_means, single_stds, multi_means, multi_stds))
    return {"dirichlet_multinomial": dict(setups=setups, coordinates=[f"theta[{i}]" for i in range(n_cats)],
                                          best_mean=best_mean, best_std=best_std,
                                          reference_source="dirichlet_config.json via load_best_multid_reference")}


def model_radon():
    """single_MC/radon_example/pymc_tutorial/rr_numpyro_radon.ipynb"""
    from modulars.radon import (
        build_numpyro_radon_model, load_best_references, load_radon_data, make_radon_unconstrained_param_names,
    )
    os.chdir(SINGLE_MC / "radon_example" / "pymc_tutorial")

    radon_data = load_radon_data()
    county_names = radon_data['county_names']
    param_names = make_radon_unconstrained_param_names(county_names)
    dim = len(param_names)
    site_names = ('mu_a', 'sigma_a', 'alpha', 'beta', 'sd_y')
    n_particles = 100
    track_every = 1

    reference_summary, best_reference, reference_means, reference_stds = load_best_references()

    model = build_numpyro_radon_model()
    model_args = (
        radon_data['log_radon'],
        radon_data['floor'],
        radon_data['log_uranium'],
        radon_data['county'],
        radon_data['floor_by_county'],
    )

    results = run_all_seeds(lambda seed: run_restart_multid_autonormal(
        seed=seed,
        model=model,
        data=None,
        param_name='radon_latent',
        n_iters=max_iters,
        dim=dim,
        n_particles=n_particles,
        model_args=model_args,
        site_names=site_names,
        track_every=track_every,
        track_start=TRACK_START, run_multi=RUN_MULTI,
    ))
    setups = setups_from_processed(apply_traj_transform_multid(results, NOTEBOOK=False))

    best_by_parameter = best_reference.set_index('parameter')
    best_mean = best_by_parameter.loc[param_names, 'mean'].to_numpy()
    best_std = best_by_parameter.loc[param_names, 'sd'].to_numpy()
    return bias_io.radon_outputs(setups, param_names, best_mean, best_std, exp_lognormal_moments)


MODEL_FUNCTIONS = {
    "1d_gaussian": model_1d_gaussian,
    "40d_gaussian": model_40d_gaussian,
    "beta_binomial": model_beta_binomial,
    "gamma_exponential": model_gamma_exponential,
    "gamma_poisson": model_gamma_poisson,
    "inverse_gamma": model_inverse_gamma,
    "dirichlet_multinomial": model_dirichlet_multinomial,
    "logistic_regression": model_logistic_regression,
    "radon": model_radon,
}


if __name__ == "__main__":
    if not bias_io.already_saved(args, PACKAGE):
        start = time.time()
        outputs = MODEL_FUNCTIONS[args.model]()
        info = bias_io.base_run_info(args, PACKAGE, SEEDS, {
            "setups": {
                bias_io.FULL_DEFAULT_SETUP: "1 MC sample (Trace_ELBO()), Adam(step_size=5e-4), PRNGKey(seed), "
                                            f"{max_iters:,} iterations; seed_seconds = wall-clock time of each fit",
            } if FULL_DEFAULT else {
                "default": "1 MC sample (Trace_ELBO()), Adam(step_size=5e-4), PRNGKey(seed)",
                "100mc": "100 MC samples (Trace_ELBO(num_particles=100)), Adam(step_size=5e-4), PRNGKey(seed + 1000)",
            },
            "runtime_seconds": time.time() - start,
            "numpyro_version": numpyro.__version__,
            "jax_version": jax.__version__,
        })
        for model, out in outputs.items():
            bias_io.save_model_outputs(args, PACKAGE, model, out["setups"], ITERATIONS, out["coordinates"],
                                       out["best_mean"], out["best_std"], out["reference_source"], info)
