"""PyMC random restarts for the overall bias comparison (run in the pymc_env_pm env).

Each model's setup below is copied from its single_MC random-restart notebook
(named in the function's docstring). The only intended changes are that
max_iters comes from the command line and run_pymc_VI is given
`track_start` so only the last `window` iterations are recorded (every
iteration). The four PyMC setups are exactly those of
run_single_seed_pymc_VI: default (1 MC), 100 MC, Adam (1 MC), Adam + 100 MC.

--full-default 1 runs only the default fit (1 MC, PyMC's default optimizer and, on the Dirichlet, PyMC's own simplex
transform), for pm.fit's default 10,000 iterations unless --max-iters is given, saved as setup "full_default" to
overall_bias_results_full_default/ with each seed's fit time (overall_default_bias_comparisons.ipynb).

usage: python pymc_runner.py --model gamma_poisson [--max-iters 1000000 --n-seeds 20 --window 2500 --overwrite 0 --full-default 0]
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
PACKAGE = "pymc"
args = bias_io.parse_runner_args(PACKAGE)

if args.model in ("logistic_regression", "radon"):
    # as in rr_pymc_logistic_regression.ipynb / rr_pymc_radon.ipynb (must be set before importing pytensor)
    os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/tmp/pytensor")

import numpy as np  # noqa: E402
import pymc as pm  # noqa: E402
import pytensor  # noqa: E402
import pytensor.tensor as pt  # noqa: E402
if os.environ.get("SIMPLEVI_PYTENSOR_CXX"):
    pytensor.config.cxx = os.environ["SIMPLEVI_PYTENSOR_CXX"]

from modulars import load_config, load_best_values  # noqa: E402
from modulars.pymc_rr_test import run_pymc_VI, run_single_seed_pymc_VI  # noqa: E402
from modulars.utils import apply_traj_transform, apply_traj_transform_multid, exp_lognormal_moments  # noqa: E402

max_iters = args.max_iters
TRACK_START = bias_io.track_start(args.max_iters, args.window)
ITERATIONS = bias_io.window_iterations(args.max_iters, args.window)
SEEDS = list(range(args.n_seeds))
# --full-default 1: only run_single_seed_pymc_VI's first fit (1 MC sample, PyMC's default optimizer, seed), with
# PyMC's defaults throughout (on the Dirichlet, PyMC's own simplex transform); saved as setup "full_default"
FULL_DEFAULT = args.full_default


def setups_from_processed(default_processed, adam_processed):
    single_means, single_stds, multi_means, multi_stds = default_processed
    if FULL_DEFAULT:
        return {bias_io.FULL_DEFAULT_SETUP: (single_means, single_stds)}
    adam_single_means, adam_single_stds, adam_multi_means, adam_multi_stds = adam_processed
    return {
        "default": (single_means, single_stds),
        "100mc": (multi_means, multi_stds),
        "adam": (adam_single_means, adam_single_stds),
        "adam_100mc": (adam_multi_means, adam_multi_stds),
    }


def run_all_seeds_single_seed_vi(run_model):
    """Loop seeds through run_single_seed_pymc_VI (cached per seed) -> results, adam_results.

    With --full-default only its first fit runs; it fills every slot so the processing below is unchanged
    (setups_from_processed then keeps just that fit).
    """
    if FULL_DEFAULT:
        def one_default_fit(seed):
            np.random.seed(seed)
            means, stds = run_model(n_mc_samples=1, seed=seed, optimizer='default', track_start=TRACK_START)
            gc.collect()
            return [means, stds]

        raw = bias_io.run_seeds_cached(args, PACKAGE, SEEDS, one_default_fit)
        results = [list(r) + list(r) for r in raw]
        return results, results

    def one_seed(seed):
        all_results = run_single_seed_pymc_VI(seed, run_model, track_start=TRACK_START)
        gc.collect()
        return list(all_results[0]) + list(all_results[1])

    raw = bias_io.run_seeds_cached(args, PACKAGE, SEEDS, one_seed)
    results = [r[:4] for r in raw]
    adam_results = [r[4:] for r in raw]
    return results, adam_results


"""
Model setups, copied from the single_MC random-restart notebooks
"""
def model_1d_gaussian():
    """single_MC/1dgaussian_knownvar/random_restarts/rr_pymc.ipynb"""
    from modulars import gaussian_1d_posterior, gaussian_1d
    os.chdir(SINGLE_MC / "1dgaussian_knownvar" / "random_restarts")

    mu_prior, sigma_prior = 0, 10
    sigma_like = 1.0

    # 100 observations, as rr_pymc.ipynb (False: no observations, as 3_1_pymc.ipynb, so the posterior is the prior)
    DATA = True
    if DATA:
        data = gaussian_1d(100, 3, sigma_like, seed=0)
        true_mu_post, true_sigma_post = gaussian_1d_posterior(data, mu_prior=mu_prior, sigma_prior=sigma_prior, sigma_like=sigma_like)
    else:
        data = np.array([])
        true_mu_post, true_sigma_post = mu_prior, sigma_prior

    def run_model(n_mc_samples=1, optimizer='default', seed=1, track_start=None):
        with pm.Model() as model:
            # Define priors and likelihood
            mu = pm.Normal("mu", mu_prior, sigma_prior)
            likelihood = pm.Normal('y', mu=mu, sigma=sigma_like, observed=np.array(data))

            # run my pymc wrapper that allows for random restarts and
            # tracking values across MC samples and optimizers
            tracked_vals = run_pymc_VI(model, n_mc_samples, max_iters, optimizer, seed, track_start=track_start)
        return tracked_vals

    results, adam_results = run_all_seeds_single_seed_vi(run_model)
    setups = setups_from_processed(
        apply_traj_transform(results, NOTEBOOK=False),
        apply_traj_transform(adam_results, NOTEBOOK=False),
    )
    return {"1d_gaussian": dict(setups=setups, coordinates=["mu"],
                                best_mean=true_mu_post, best_std=true_sigma_post,
                                reference_source=("gaussian_1d_posterior (analytic, as rr_pymc.ipynb)" if DATA
                                                  else "prior (no observations, as 3_1_pymc.ipynb / rr_stan.ipynb)"))}


def model_beta_binomial():
    """single_MC/1dbeta_binomial/random_restarts/rr_pymc_beta.ipynb"""
    from modulars import binomial  # data generating function
    from modulars.utils import logistic_moments  # same function, seed and draw count as the NumPyro / TFP runners
    os.chdir(SINGLE_MC / "1dbeta_binomial" / "random_restarts")
    trace_fn = logistic_moments
    pushforward_fn = logistic_moments
    data_gen = binomial

    config_file = "../beta_config.json"
    config = load_config(config_file)

    theta_like = config['theta_like']
    alpha_prior = config['alpha_prior']
    beta_prior = config['beta_prior']
    n_samples = config['n_samples']

    data = data_gen(n_samples) if n_samples > 0 else 0

    def run_model(n_mc_samples=1, optimizer='default', seed=1, track_start=None):
        with pm.Model() as model:
            # Define the prior
            theta = pm.Beta('theta', alpha=alpha_prior, beta=beta_prior)

            # Define the likelihood
            y = pm.Binomial('y', n=n_samples, p=theta, observed=data)

            # run my pymc wrapper that allows for random restarts
            # and tracking values across MC samples and optimizers
            tracked_vals = run_pymc_VI(model, n_mc_samples, max_iters, optimizer, seed, track_start=track_start)
        return tracked_vals

    results, results_adam = run_all_seeds_single_seed_vi(run_model)
    setups = setups_from_processed(
        apply_traj_transform(results, pushforward_fn, n_samples=100_000, seed=1, NOTEBOOK=False),
        apply_traj_transform(results_adam, pushforward_fn, n_samples=100_000, seed=1, NOTEBOOK=False),
    )
    # If we have "best" from config, these should be on theta-scale (0,1)
    best_mu, best_std = load_best_values(
        config=config, transform=trace_fn,
        n_samples=100_000, seed=1)
    return {"beta_binomial": dict(setups=setups, coordinates=["theta"], best_mean=best_mu, best_std=best_std,
                                  reference_source="beta_config.json via load_best_values(logistic_moments, n_samples=100_000, seed=1)")}


def model_gamma_exponential():
    """single_MC/gamma_exponential/random_restarts/rr_pymc_gam_exp.ipynb"""
    from modulars import exponential
    os.chdir(SINGLE_MC / "gamma_exponential" / "random_restarts")
    transform_fn = exp_lognormal_moments

    config_file = '../exp_config.json'
    config = load_config(config_file)

    lambda_like = config['lambda_like']
    alpha_prior = config['alpha_prior']
    beta_prior = config['beta_prior']
    n_samples = config['n_samples']
    data = exponential(n_samples)

    def run_model(n_mc_samples=1, optimizer='default', seed=0, track_start=None):
        with pm.Model() as model:
            lambd = pm.Gamma('lambda', alpha=alpha_prior, beta=beta_prior)

            # Likelihood
            likelihood = pm.Exponential('likelihood', lam=lambd, observed=data)

            # run my pymc wrapper that allows for random restarts
            # and tracking values across MC samples and optimizers
            tracked_vals = run_pymc_VI(model, n_mc_samples, max_iters, optimizer, seed, track_start=track_start)
        return tracked_vals

    results, results_adam = run_all_seeds_single_seed_vi(run_model)
    setups = setups_from_processed(
        apply_traj_transform(results, transform_fn, n_samples=10_000, seed=1, NOTEBOOK=False),
        apply_traj_transform(results_adam, transform_fn, n_samples=10_000, seed=1, NOTEBOOK=False),
    )
    # exact best mean-field (log-normal) approximation of the conjugate Gamma posterior
    from modulars.distributions import gamma_exponential_posterior, gamma_lognormal_optimum
    m_star, s_star = gamma_lognormal_optimum(*gamma_exponential_posterior(data, alpha_prior, beta_prior))
    best_mu, best_std = exp_lognormal_moments(m_star, s_star)
    return {"gamma_exponential": dict(setups=setups, coordinates=["lambda"], best_mean=best_mu, best_std=best_std,
                                      reference_source="closed form: gamma_lognormal_optimum(gamma_exponential_posterior(data))")}


def model_gamma_poisson():
    """single_MC/gamma_poisson/random_restarts/rr_pymc_gam_pois.ipynb"""
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

    def run_model(n_mc_samples=1, optimizer='default', seed=0, track_start=None):
        with pm.Model() as model:
            lambd = pm.Gamma('lambda', alpha=alpha_prior, beta=beta_prior)

            # Likelihood
            likelihood = pm.Poisson('likelihood', mu=lambd, observed=data)

            # run my pymc wrapper that allows for random restarts
            # and tracking values across MC samples and optimizers
            tracked_vals = run_pymc_VI(model, n_mc_samples, max_iters, optimizer, seed, track_start=track_start)
        return tracked_vals

    results, results_adam = run_all_seeds_single_seed_vi(run_model)
    setups = setups_from_processed(
        apply_traj_transform(results, transform_fn, n_samples=10_000, seed=1, NOTEBOOK=False),
        apply_traj_transform(results_adam, transform_fn, n_samples=10_000, seed=1, NOTEBOOK=False),
    )
    # exact best mean-field (log-normal) approximation of the conjugate Gamma posterior
    from modulars.distributions import gamma_poisson_posterior, gamma_lognormal_optimum
    m_star, s_star = gamma_lognormal_optimum(*gamma_poisson_posterior(data, alpha_prior, beta_prior))
    best_mu, best_std = exp_lognormal_moments(m_star, s_star)
    return {"gamma_poisson": dict(setups=setups, coordinates=["lambda"], best_mean=best_mu, best_std=best_std,
                                  reference_source="closed form: gamma_lognormal_optimum(gamma_poisson_posterior(data))")}


def model_inverse_gamma():
    """single_MC/invgamma_gaussianknownmean/random_restarts/rr_pymc_invgam.ipynb"""
    from modulars import gaussian_1d
    os.chdir(SINGLE_MC / "invgamma_gaussianknownmean" / "random_restarts")
    transform_fn = exp_lognormal_moments

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

    def run_model(n_mc_samples=1, optimizer='default', seed=0, track_start=None):
        with pm.Model() as model:
            # Prior
            sigmasq = pm.InverseGamma('sigmasq', alpha=alpha_prior, beta=beta_prior)

            # Likelihood
            like = pm.Normal('like', mu_like, pt.sqrt(sigmasq), observed=data)

            # run my pymc wrapper that allows for random restarts
            # and tracking values across MC samples and optimizers
            tracked_vals = run_pymc_VI(model, n_mc_samples, max_iters, optimizer, seed, track_start=track_start)
        return tracked_vals

    results, results_adam = run_all_seeds_single_seed_vi(run_model)
    setups = setups_from_processed(
        apply_traj_transform(results, transform_fn, n_samples=10_000, seed=1, NOTEBOOK=False),
        apply_traj_transform(results_adam, transform_fn, n_samples=10_000, seed=1, NOTEBOOK=False),
    )
    best_mu, best_std = load_best_values(
        config=config, transform=transform_fn,
        n_samples=50_000, seed=1)
    return {"inverse_gamma": dict(setups=setups, coordinates=["sigmasq"], best_mean=best_mu, best_std=best_std,
                                  reference_source="invgamma_config.json via load_best_values(exp_lognormal_moments, n_samples=50_000, seed=1)")}


def model_40d_gaussian():
    """single_MC/multidim_gaussian_knowncovar/random_restarts/rr_pymc_multid_gaussian.ipynb

    The notebook's apply_traj_transform_multid(..., PYMC=True) keeps the
    diagonal of the tracked covariance (variances) and compares it to the
    analytic posterior variance; here we take square roots of both so the
    saved statistic is the standard deviation like every other model.
    """
    from modulars.distributions import gaussian_multid, gaussian_multid_posterior
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
    # compute known posterior, this is the posterior diagonal standard deviation vector as true_sigma_post
    true_mu_post, true_sigma_post = gaussian_multid_posterior(
        observed_data, n_samples=n_obs,
        mu_prior=mu_prior, scale_prior=sigma_prior**2,
        scale_like=1.0, std_devs=False,
        diag=DIAG, dim=40)

    def run_model(n_mc_samples=1, optimizer='default', seed=1, track_start=None):
        with pm.Model() as model:
            mu = pm.Normal('mu', mu_prior, sigma_prior, shape=dim)
            likelihood = pm.Normal('y', mu, sigma=np.ones(dim), observed=observed_data)
            tracked_vals = run_pymc_VI(model, n_mc_samples, max_iters, optimizer, seed, SINGLE_DIM=False,
                                       track_start=track_start)
        return tracked_vals

    results, adam_results = run_all_seeds_single_seed_vi(run_model)
    single_means, single_covs, multi_means, multi_covs = \
        apply_traj_transform_multid(results, PYMC=True, NOTEBOOK=False)
    adam_single_means, adam_single_covs, adam_multi_means, adam_multi_covs = \
        apply_traj_transform_multid(adam_results, PYMC=True, NOTEBOOK=False)
    setups = setups_from_processed(
        (single_means, np.sqrt(single_covs), multi_means, np.sqrt(multi_covs)),
        (adam_single_means, np.sqrt(adam_single_covs), adam_multi_means, np.sqrt(adam_multi_covs)),
    )
    return {"40d_gaussian": dict(setups=setups, coordinates=[f"mu[{i}]" for i in range(dim)],
                                 best_mean=true_mu_post, best_std=np.sqrt(true_sigma_post),
                                 reference_source="gaussian_multid_posterior (analytic posterior variance, square-rooted)")}


def model_logistic_regression():
    """single_MC/logistic_regression/rr_pymc_logistic_regression.ipynb"""
    import pandas as pd
    from modulars.logistic_regression import build_pymc_logistic_regression_model, load_logistic_regression_data
    os.chdir(SINGLE_MC / "logistic_regression")

    data = load_logistic_regression_data()
    labels = data["labels"]
    dim = len(labels)
    track_every = 1

    def run_model(n_mc_samples=1, optimizer="default", seed=1, track_start=None):
        model = build_pymc_logistic_regression_model(data)
        return run_pymc_VI(
            model,
            n_mc_samples=n_mc_samples,
            n_iters=max_iters,
            optimizer=optimizer,
            seed=seed,
            SINGLE_DIM=False,
            track_every=track_every,
            track_cov=False,
            track_start=track_start,
        )

    # the notebook's two seed loops (default then adam, seeds seed / seed + 1000)
    # are exactly run_single_seed_pymc_VI's runs
    results, results_adam = run_all_seeds_single_seed_vi(run_model)
    setups = setups_from_processed(
        apply_traj_transform_multid(results, NOTEBOOK=False),
        apply_traj_transform_multid(results_adam, NOTEBOOK=False),
    )

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
    """single_MC/multinomial_dirichlet/rr_dirich_pymc.ipynb

    Uses the notebook's own build_pymc_dirichlet_model / run_single_seed_pymc_scenarios
    (seed = rr_seed for every scenario) with its four stick-breaking scenarios:
    stickbreaking_only, stickbreaking_mc100, stickbreaking_adam,
    stickbreaking_mc100_adam. These swap PyMC's default simplex transform for the
    stick-breaking transform (StickBreakingSimplexTransform) so PyMC uses the same
    kind of simplex transform as the other packages, and are processed with the
    notebook's stickbreaking_simplex_forward.
    """
    from modulars.distributions import gen_multinomial_data_shared
    from modulars.utils import (load_best_multid_reference, stickbreaking_simplex_forward, simplex_moments_from_cov,
                                anchored_softmax_simplex_forward)
    from modulars.pymc_rr_test import StickBreakingSimplexTransform
    os.chdir(SINGLE_MC / "multinomial_dirichlet")

    config_file = "dirichlet_config.json"
    config = load_config(config_file)

    theta_like = np.array(config["theta_like"], dtype=np.float32)
    alpha_prior = np.array(config["alpha_prior"], dtype=np.float32)
    n_cats = config["n_cats"]
    N = config["N"]
    total_count = config["total_count"]
    seed = config.get("seed", 15)

    n_vec = np.full(N, total_count, dtype=int)
    obs_counts = gen_multinomial_data_shared(theta_like, N, total_count, SEED=seed)
    best_mean, best_cov, best_std = load_best_multid_reference(config)

    def build_pymc_dirichlet_model(
            alpha_prior,
            n_vec,
            obs_counts,
            n_cats,
            simplex_transform="default"):
        """
        we build the standard PyMC Dirichlet-multinomial model.
        For the stick-breaking case, we replace PyMC's default simplex transform
        with the custom stick-breaking transform.
        """
        with pm.Model() as model:
            if simplex_transform == "default":
                theta = pm.Dirichlet("theta", a=alpha_prior)
            elif simplex_transform == "stickbreaking":
                theta = pm.Dirichlet(
                    "theta",
                    a=alpha_prior,
                    default_transform=StickBreakingSimplexTransform(),
                )
            else:
                raise ValueError(
                    "we expected simplex_transform to be 'default' or 'stickbreaking'"
                )
            pm.Multinomial("y", n=n_vec, p=theta, observed=np.array(obs_counts))
        return model

    def run_single_seed_pymc_scenarios(
            seed,
            alpha_prior,
            n_vec,
            obs_counts,
            n_cats,
            n_iters,
            scenario_name,
            track_start=None):
        scenario_kwargs = {
            "default": {
                "n_mc_samples": 1,
                "optimizer": "default",
                "simplex_transform": "default",
            },
            "stickbreaking_only": {
                "n_mc_samples": 1,
                "optimizer": "default",
                "simplex_transform": "stickbreaking",
            },
            "stickbreaking_mc100": {
                "n_mc_samples": 100,
                "optimizer": "default",
                "simplex_transform": "stickbreaking",
            },
            "stickbreaking_adam": {
                "n_mc_samples": 1,
                "optimizer": "adam",
                "simplex_transform": "stickbreaking",
            },
            "stickbreaking_mc100_adam": {
                "n_mc_samples": 100,
                "optimizer": "adam",
                "simplex_transform": "stickbreaking",
            }
        }
        dirichlet_kwargs = {
            'alpha_prior': alpha_prior,
            'n_vec': n_vec,
            'obs_counts': obs_counts,
            'n_cats': n_cats,
        }
        pymc_VI_kwargs = {
            "n_iters": n_iters,
            "seed": seed,
            "SINGLE_DIM": False,
            "track_start": track_start,
        }

        def build_model(which_scenario):
            return build_pymc_dirichlet_model(
                simplex_transform=scenario_kwargs[which_scenario]['simplex_transform'],
                **dirichlet_kwargs
            )

        def run_pymc_VI_scenario(which_scenario):
            model = build_model(which_scenario)
            return run_pymc_VI(
                model, **pymc_VI_kwargs,
                **scenario_kwargs[which_scenario]
            )

        assert scenario_name in scenario_kwargs, f"scenario_name must be one of {list(scenario_kwargs.keys())}"
        return {scenario_name: run_pymc_VI_scenario(scenario_name)}

    scenario_to_setup = {
        "stickbreaking_only": "default",
        "stickbreaking_mc100": "100mc",
        "stickbreaking_adam": "adam",
        "stickbreaking_mc100_adam": "adam_100mc",
    }
    # the latent-to-simplex map of each scenario's transform, as the notebook's scenario_meta
    forward_fns = {scenario: stickbreaking_simplex_forward for scenario in scenario_to_setup}
    if FULL_DEFAULT:   # the notebook's "default" scenario: PyMC's own simplex transform
        scenario_to_setup = {"default": bias_io.FULL_DEFAULT_SETUP}
        forward_fns = {"default": anchored_softmax_simplex_forward}

    def one_seed(rr_seed):
        arrays = []
        for scenario in scenario_to_setup:
            out = run_single_seed_pymc_scenarios(
                seed=rr_seed, alpha_prior=alpha_prior, n_vec=n_vec, obs_counts=obs_counts,
                n_cats=n_cats, n_iters=max_iters, scenario_name=scenario, track_start=TRACK_START,
            )
            loc_traj, cov_traj = out[scenario]
            arrays.extend([loc_traj, cov_traj])
            gc.collect()
        return arrays

    raw = bias_io.run_seeds_cached(args, PACKAGE, SEEDS, one_seed)

    def process_single_pymc_scenario_results(
            raw_results,
            n_cats,
            forward_fn,
            n_samples=50_000,
            seed=10):
        """
        we process one scenario where each restart stores only one latent mean/cov trajectory.
        """
        means = []
        stds = []
        for loc_traj, cov_traj in raw_results:
            mean_traj, std_traj = simplex_moments_from_cov(
                loc_traj,
                cov_traj,
                n_cats=n_cats,
                forward_fn=forward_fn,
                n_samples=n_samples,
                seed=seed,
            )
            means.append(mean_traj)
            stds.append(std_traj)
        return np.asarray(means), np.asarray(stds)

    setups = {}
    for k, (scenario, setup) in enumerate(scenario_to_setup.items()):
        raw_results = [(r[2 * k], r[2 * k + 1]) for r in raw]
        setups[setup] = process_single_pymc_scenario_results(
            raw_results,
            n_cats=n_cats,
            forward_fn=forward_fns[scenario],
            n_samples=50_000,  # same draw count and seed as the NumPyro / TFP runners
            seed=1,
        )
    return {"dirichlet_multinomial": dict(setups=setups, coordinates=[f"theta[{i}]" for i in range(n_cats)],
                                          best_mean=best_mean, best_std=best_std,
                                          reference_source="dirichlet_config.json via load_best_multid_reference")}


def model_radon():
    """single_MC/radon_example/pymc_tutorial/rr_pymc_radon.ipynb"""
    from modulars.radon import build_pymc_radon_model, load_best_references, load_radon_data
    os.chdir(SINGLE_MC / "radon_example" / "pymc_tutorial")

    radon_data = load_radon_data()
    county_names = radon_data['county_names']
    track_every = 1

    reference_summary, best_reference, reference_means, reference_stds = load_best_references()

    # PyMC's free variable order is the model construction order, with positive
    # variables represented on the unconstrained log scale.
    pymc_param_names = (
        ['mu_a', 'sigma_a_log__']
        + [f'alpha[{county}]' for county in county_names]
        + ['beta', 'sd_y_log__']
    )

    def run_model(n_mc_samples=1, optimizer='default', seed=1, track_start=None):
        model = build_pymc_radon_model(radon_data)
        try:
            return run_pymc_VI(
                model,
                n_mc_samples=n_mc_samples,
                n_iters=max_iters,
                optimizer=optimizer,
                seed=seed,
                SINGLE_DIM=False,
                track_every=track_every,
                track_cov=False,
                track_start=track_start,
            )
        finally:
            del model
            gc.collect()

    # the notebook's two seed loops (default then adam, seeds seed / seed + 1000)
    # are exactly run_single_seed_pymc_VI's runs
    results, results_adam = run_all_seeds_single_seed_vi(run_model)
    setups = setups_from_processed(
        apply_traj_transform_multid(results, NOTEBOOK=False),
        apply_traj_transform_multid(results_adam, NOTEBOOK=False),
    )

    best_by_parameter = best_reference.set_index('parameter')
    best_mean = best_by_parameter.loc[pymc_param_names, 'mean'].to_numpy()
    best_std = best_by_parameter.loc[pymc_param_names, 'sd'].to_numpy()
    return bias_io.radon_outputs(setups, pymc_param_names, best_mean, best_std, exp_lognormal_moments)


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
                bias_io.FULL_DEFAULT_SETUP: f"1 MC sample, PyMC default optimizer (seed), {max_iters:,} iterations"
                                            + ("; PyMC's own simplex transform (rr_dirich_pymc.ipynb 'default' scenario)"
                                               if args.model == "dirichlet_multinomial" else "")
                                            + "; seed_seconds = wall-clock time of each fit",
            } if FULL_DEFAULT else {
                "default": "1 MC sample, PyMC default optimizer (seed)",
                "100mc": "100 MC samples, PyMC default optimizer (seed + 1000)",
                "adam": "1 MC sample, pm.adam() (seed)",
                "adam_100mc": "100 MC samples, pm.adam() (seed + 1000)",
            } if args.model != "dirichlet_multinomial" else
            "rr_dirich_pymc.ipynb stick-breaking scenarios stickbreaking_only / stickbreaking_mc100 / "
            "stickbreaking_adam / stickbreaking_mc100_adam (PyMC's simplex transform swapped for stick-breaking "
            "to match the other packages), all with seed = rr_seed",
            "runtime_seconds": time.time() - start,
            "pymc_version": pm.__version__,
        })
        for model, out in outputs.items():
            bias_io.save_model_outputs(args, PACKAGE, model, out["setups"], ITERATIONS, out["coordinates"],
                                       out["best_mean"], out["best_std"], out["reference_source"], info)
