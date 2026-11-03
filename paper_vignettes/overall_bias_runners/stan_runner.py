"""Stan (CmdStan ADVI) random restarts for the overall bias comparison (run in the stan3 env).

Each model's Stan file, data and parameter columns below are copied from its
single_MC Stan random-restart notebook (named in the function's docstring), and
the runs go through modulars.stan_rr_test.run_stan_random_restarts exactly as
there: Stan only writes the final fit, so iteration k of the window is a
separate rerun of the same seed with iter=k (here every iteration from
max_iters - window + 1 to max_iters, via run_stan_random_restarts' min_iter).

The two Stan setups are
    default: every CmdStan default except iter, tol_rel_obj=1e-6, output_samples=10000
    100mc:   grad_samples=100, iter, tol_rel_obj=1e-6, output_samples=10000, rest default
and the mean / std at each iteration are the moments of the 10k output draws.

Seeds run in chunks of max_workers seeds (each seed's two scenarios are separate
parallel tasks; the 1 MC tasks finish quickly, so up to max_workers long 100 MC
reruns run at once) and each finished chunk is cached per seed, so an
interrupted run only repeats the unfinished chunk.

--full-default 1 runs one scenario with every CmdStan ADVI default (tol_rel_obj=0.01, output_samples=1000, ...;
on the Dirichlet Stan's own simplex transform), for CmdStan's default iter=10,000 unless --max-iters is given, saved
as setup "full_default" to overall_bias_results_full_default/ with each seed's fit time (its iter=max_iters run).
Runs that meet tol_rel_obj stop early, so the reruns of the window are then all the same fit.
--single-setup NAME runs only the "default" scenario (1 MC, tol_rel_obj=1e-6, 10,000 output draws) and saves it as
setup NAME to overall_bias_results_1mc/, e.g. --single-setup default_50k --max-iters 50000.

usage: python stan_runner.py --model gamma_poisson [--max-iters 1000000 --n-seeds 20 --window 2500 --overwrite 0 --max-workers 13]
"""

import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bias_io  # noqa: E402

sys.path.insert(0, str(bias_io.REPO_ROOT))
SINGLE_MC = bias_io.REPO_ROOT / "single_MC"
PACKAGE = "stan"
args = bias_io.parse_runner_args(PACKAGE)

# CmdStan: SIMPLEVI_CMDSTAN_PATH, else the CmdStan bundled in the active conda env, else cmdstanpy's default
from cmdstanpy import cmdstan_path, set_cmdstan_path  # noqa: E402
_env_cmdstan = Path(sys.prefix).resolve() / "bin" / "cmdstan"
STAN3_CMDSTAN = Path(os.environ.get("SIMPLEVI_CMDSTAN_PATH")
                     or (_env_cmdstan if _env_cmdstan.exists() else cmdstan_path()))
os.environ["CMDSTAN"] = str(STAN3_CMDSTAN)
set_cmdstan_path(str(STAN3_CMDSTAN))


import numpy as np  # noqa: E402

from modulars.stan_rr_test import run_stan_random_restarts  # noqa: E402
from modulars.utils import load_config, load_best_values, exp_lognormal_moments, logistic_moments  # noqa: E402

TOL_REL_OBJ = 1e-6
OUTPUT_SAMPLES = 10_000
STAN_SCENARIOS = (
    {
        "name": "default",
        "mc_label": "1",
        "cmd_args": (f"output_samples={OUTPUT_SAMPLES}",),
        "tol_rel_obj": TOL_REL_OBJ,
    },
    {
        "name": "100mc",
        "mc_label": "100",
        "cmd_args": ("grad_samples=100", f"output_samples={OUTPUT_SAMPLES}"),
        "tol_rel_obj": TOL_REL_OBJ,
    },
)

# --full-default 1: one scenario with every CmdStan ADVI default (iter, tol_rel_obj=0.01, output_samples=1000, ...)
# and, on the Dirichlet, Stan's own simplex transform; saved as setup "full_default"
FULL_DEFAULT = args.full_default
if FULL_DEFAULT:
    STAN_SCENARIOS = (
        {
            "name": bias_io.FULL_DEFAULT_SETUP,
            "mc_label": "1",
            "cmd_args": (),
        },
    )

# --single-setup NAME: only the 1 MC "default" scenario (tol_rel_obj=1e-6, 10,000 output draws), saved as setup NAME
if args.single_setup:
    STAN_SCENARIOS = (dict(STAN_SCENARIOS[0], name=args.single_setup),)

max_iters = args.max_iters
TRACK_START = bias_io.track_start(args.max_iters, args.window)
ITERATIONS = bias_io.window_iterations(args.max_iters, args.window)
SEEDS = list(range(args.n_seeds))
MAX_WORKERS = args.max_workers if args.max_workers else max(1, (os.cpu_count() or 2) - 1)
RUN_DIR = args.results_dir / "_stan_runs" / args.model


def run_stan_seeds(stan_file, stan_data, param_columns, log_columns, seed_offset=0):
    """Run every seed through run_stan_random_restarts in parallel chunks.

    Returns {setup: (means, stds)} with arrays (n_seeds, window, n_params) and the failures.
    """
    # the 1 MC scenario's reruns are much faster than the 100 MC ones, so a chunk
    # holds max_workers seeds to keep every worker on a long 100 MC chain
    chunk_size = max(1, MAX_WORKERS)
    per_seed, failures = {}, []
    for seed in SEEDS:
        cached = bias_io.load_cached_seed(args, PACKAGE, seed)
        if cached is not None:
            per_seed[seed] = cached
    todo = [s for s in SEEDS if s not in per_seed]
    if len(todo) < len(SEEDS):
        print(f"[stan/{args.model}] loaded {len(SEEDS) - len(todo)} seeds from cache", flush=True)
    # consecutive runs of seeds so each chunk is one (seed_offset, n_restarts) call
    chunks, current = [], []
    for seed in todo:
        if current and (seed != current[-1] + 1 or len(current) == chunk_size):
            chunks.append(current)
            current = []
        current.append(seed)
    if current:
        chunks.append(current)

    for chunk in chunks:
        start = time.time()
        stan_result = run_stan_random_restarts(
            stan_file=stan_file,
            data=stan_data,
            param_columns=param_columns,
            log_columns=log_columns,
            output_dir=RUN_DIR / f"seeds_{chunk[0]:03d}_{chunk[-1]:03d}" / "cmdstan_runs",
            max_iters=max_iters,
            n_restarts=len(chunk),
            track_every=1,
            min_iter=TRACK_START,
            scenarios=STAN_SCENARIOS,
            seed_offset=seed_offset + chunk[0],
            parallel=True,
            max_workers=MAX_WORKERS,
            keep_outputs=False,
            refresh=0,
            dtype=np.float64,
            show_progress=True,
        )
        with open(RUN_DIR / f"seeds_{chunk[0]:03d}_{chunk[-1]:03d}" / "stan_result.pkl", "wb") as f:
            pickle.dump(stan_result, f)
        for scenario in STAN_SCENARIOS:
            got = np.asarray(stan_result["iterations_by_scenario"][scenario["name"]])
            assert np.array_equal(got, ITERATIONS), f"unexpected tracked iterations for {scenario['name']}"
        for i, seed in enumerate(chunk):
            arrays = []
            for scenario in STAN_SCENARIOS:
                arrays.append(stan_result["means_by_scenario"][scenario["name"]][i])
                arrays.append(stan_result["stds_by_scenario"][scenario["name"]][i])
            # a full-default seed's fit time is its CmdStan run with iter=max_iters (the last rerun)
            seconds = stan_result["seconds_by_scenario"][STAN_SCENARIOS[0]["name"]][i][-1] if FULL_DEFAULT else None
            bias_io.save_cached_seed(args, PACKAGE, seed, arrays, seconds=seconds)
            per_seed[seed] = arrays
        failures.extend(stan_result["failures"])
        print(f"[stan/{args.model}] seeds {chunk[0]}-{chunk[-1]} done in {time.time() - start:.1f}s", flush=True)

    setups = {}
    for k, scenario in enumerate(STAN_SCENARIOS):
        means = np.stack([per_seed[s][2 * k] for s in SEEDS])
        stds = np.stack([per_seed[s][2 * k + 1] for s in SEEDS])
        setups[scenario["name"]] = (means, stds)
    return setups, failures


"""
Model setups, copied from the single_MC Stan random-restart notebooks
"""
def model_1d_gaussian():
    """single_MC/1dgaussian_knownvar/random_restarts/rr_stan.ipynb

    n_samples = 100 observations; every package's runner uses the same setup
    (n_samples = 0 gives no observations, so the posterior is the prior, as in the 3_1 notebooks).
    """
    from modulars import gaussian_1d, gaussian_1d_posterior
    nb_dir = SINGLE_MC / "1dgaussian_knownvar" / "random_restarts"
    os.chdir(nb_dir)

    mu_prior, sigma_prior = 0.0, 10.0
    sigma_like = 1.0
    n_samples = 100
    mu_like = 3.0

    data = gaussian_1d(n_samples, mu_like, sigma_like, seed=0).astype(np.float32)
    true_mu_post, true_sigma_post = gaussian_1d_posterior(
        data,
        n_samples=n_samples,
        mu_prior=mu_prior,
        sigma_prior=sigma_prior,
        sigma_like=sigma_like,
    )

    stan_data = {
        "N": len(data),
        "y": data,
        "sigma_like": sigma_like,
        "mu_prior": mu_prior,
        "sigma_prior": sigma_prior,
    }
    PARAM_COLUMNS = ["mu"]
    LOG_COLUMNS = []
    # the notebook's main restart sweep uses seed_offset=100
    setups, failures = run_stan_seeds(nb_dir / "stan_1dgaussian_knownvar.stan", stan_data, PARAM_COLUMNS, LOG_COLUMNS,
                                      seed_offset=100)
    return {"1d_gaussian": dict(setups=setups, coordinates=["mu"], best_mean=true_mu_post, best_std=true_sigma_post,
                                reference_source=f"gaussian_1d_posterior (analytic, n_samples={n_samples})")}, failures, 100


def model_beta_binomial():
    """single_MC/1dbeta_binomial/random_restarts/rr_stan_beta.ipynb"""
    from modulars import binomial
    nb_dir = SINGLE_MC / "1dbeta_binomial" / "random_restarts"
    os.chdir(nb_dir)

    config_file = nb_dir.parent / "beta_config.json"
    config = load_config(config_file)

    theta_like = config["theta_like"]
    alpha_prior = config["alpha_prior"]
    beta_prior = config["beta_prior"]
    n_samples = int(config["n_samples"])

    data = int(binomial(n_samples, theta_like)) if n_samples > 0 else 0
    stan_data = {
        "n_trials": n_samples,
        "y": data,
        "alpha_prior": alpha_prior,
        "beta_prior": beta_prior,
    }
    PARAM_COLUMNS = ["theta"]
    LOG_COLUMNS = []
    best_mu, best_std = load_best_values(config=config, transform=logistic_moments, n_samples=100_000, seed=1)
    setups, failures = run_stan_seeds(nb_dir / "stan_beta_binomial.stan", stan_data, PARAM_COLUMNS, LOG_COLUMNS)
    return {"beta_binomial": dict(setups=setups, coordinates=["theta"], best_mean=best_mu, best_std=best_std,
                                  reference_source="beta_config.json via load_best_values(logistic_moments, n_samples=100_000, seed=1)")}, failures, 0


def model_gamma_exponential():
    """single_MC/gamma_exponential/random_restarts/rr_stan_gam_exp.ipynb"""
    from modulars import exponential
    nb_dir = SINGLE_MC / "gamma_exponential" / "random_restarts"
    os.chdir(nb_dir)

    config_file = nb_dir.parent / "exp_config.json"
    config = load_config(config_file)

    lambda_like = config["lambda_like"]
    alpha_prior = config["alpha_prior"]
    beta_prior = config["beta_prior"]
    n_samples = int(config["n_samples"])

    data = exponential(n_samples, lambda_like).astype(float)
    stan_data = {
        "N": len(data),
        "y": data,
        "alpha_prior": alpha_prior,
        "beta_prior": beta_prior,
    }
    PARAM_COLUMNS = ["lambd"]
    LOG_COLUMNS = []
    # exact best mean-field (log-normal) approximation of the conjugate Gamma posterior
    from modulars.distributions import gamma_exponential_posterior, gamma_lognormal_optimum
    m_star, s_star = gamma_lognormal_optimum(*gamma_exponential_posterior(data, alpha_prior, beta_prior))
    best_mu, best_std = exp_lognormal_moments(m_star, s_star)
    setups, failures = run_stan_seeds(nb_dir / "stan_gamma_exponential.stan", stan_data, PARAM_COLUMNS, LOG_COLUMNS)
    return {"gamma_exponential": dict(setups=setups, coordinates=["lambda"], best_mean=best_mu, best_std=best_std,
                                      reference_source="closed form: gamma_lognormal_optimum(gamma_exponential_posterior(data))")}, failures, 0


def model_gamma_poisson():
    """single_MC/gamma_poisson/random_restarts/rr_stan_gam_pois.ipynb"""
    from modulars import poisson
    nb_dir = SINGLE_MC / "gamma_poisson" / "random_restarts"
    os.chdir(nb_dir)

    config_file = nb_dir.parent / "poisson_config.json"
    config = load_config(config_file)

    lambda_like = config["lambda_like"]
    alpha_prior = config["alpha_prior"]
    beta_prior = config["beta_prior"]
    n_samples = int(config["n_samples"])

    data = poisson(n_samples, lambda_like).astype(int)
    stan_data = {
        "N": len(data),
        "y": data.astype(int),
        "alpha_prior": alpha_prior,
        "beta_prior": beta_prior,
    }
    PARAM_COLUMNS = ["lambd"]
    LOG_COLUMNS = []
    # exact best mean-field (log-normal) approximation of the conjugate Gamma posterior
    from modulars.distributions import gamma_poisson_posterior, gamma_lognormal_optimum
    m_star, s_star = gamma_lognormal_optimum(*gamma_poisson_posterior(data, alpha_prior, beta_prior))
    best_mu, best_std = exp_lognormal_moments(m_star, s_star)
    setups, failures = run_stan_seeds(nb_dir / "stan_gamma_poisson.stan", stan_data, PARAM_COLUMNS, LOG_COLUMNS)
    return {"gamma_poisson": dict(setups=setups, coordinates=["lambda"], best_mean=best_mu, best_std=best_std,
                                  reference_source="closed form: gamma_lognormal_optimum(gamma_poisson_posterior(data))")}, failures, 0


def model_inverse_gamma():
    """single_MC/invgamma_gaussianknownmean/random_restarts/rr_stan_invgam.ipynb"""
    from modulars import gaussian_1d
    nb_dir = SINGLE_MC / "invgamma_gaussianknownmean" / "random_restarts"
    os.chdir(nb_dir)

    config_file = nb_dir.parent / "invgamma_config.json"
    config = load_config(config_file)

    mu_like = config["mu_like"]
    sigma_like = config["sigma_like"]
    alpha_prior = config["alpha_prior"]
    beta_prior = config["beta_prior"]
    n_samples = int(config["n_samples"])

    data = gaussian_1d(n_samples, mu_like, sigma_like=sigma_like).astype(float)
    stan_data = {
        "N": len(data),
        "y": data,
        "mu_like": mu_like,
        "alpha_prior": alpha_prior,
        "beta_prior": beta_prior,
    }
    PARAM_COLUMNS = ["sigma_sq"]
    LOG_COLUMNS = []
    best_mu, best_std = load_best_values(config=config, transform=exp_lognormal_moments, n_samples=50_000, seed=1)
    setups, failures = run_stan_seeds(nb_dir / "stan_invgamma_gaussian_knownmean.stan", stan_data, PARAM_COLUMNS, LOG_COLUMNS)
    return {"inverse_gamma": dict(setups=setups, coordinates=["sigmasq"], best_mean=best_mu, best_std=best_std,
                                  reference_source="invgamma_config.json via load_best_values(exp_lognormal_moments, n_samples=50_000, seed=1)")}, failures, 0


def model_40d_gaussian():
    """single_MC/multidim_gaussian_knowncovar/random_restarts/rr_stan_multid_gaussian.ipynb"""
    from modulars.distributions import gaussian_multid, gaussian_multid_posterior
    nb_dir = SINGLE_MC / "multidim_gaussian_knowncovar" / "random_restarts"
    os.chdir(nb_dir)

    mu_prior = np.zeros(40)
    sigma_prior = np.ones(40) * 10.0
    mu_like = 2.0
    sigma_like = 1.0
    n_obs = 3
    dim = 40

    observed_data = gaussian_multid(
        seed=42,
        n_samples=n_obs,
        mu_like=mu_like,
        scale_like=sigma_like,
        dim=dim,
    )
    true_mu_post, true_sigma_post = gaussian_multid_posterior(
        observed_data,
        n_samples=n_obs,
        mu_prior=mu_prior,
        scale_prior=sigma_prior,
        scale_like=sigma_like,
        std_devs=True,
        diag=True,
        dim=dim,
    )

    stan_data = {
        "D": dim,
        "N": n_obs,
        "y": observed_data,
        "mu_prior": mu_prior,
        "sigma_prior": sigma_prior,
        "sigma_like": sigma_like,
    }
    PARAM_COLUMNS = [f"mu[{i}]" for i in range(1, dim + 1)]
    LOG_COLUMNS = []
    setups, failures = run_stan_seeds(nb_dir / "stan_multidim_gaussian_knowncovar.stan", stan_data, PARAM_COLUMNS, LOG_COLUMNS)
    return {"40d_gaussian": dict(setups=setups, coordinates=[f"mu[{i}]" for i in range(dim)],
                                 best_mean=true_mu_post, best_std=true_sigma_post,
                                 reference_source="gaussian_multid_posterior (analytic posterior std)")}, failures, 0


def model_logistic_regression():
    """single_MC/logistic_regression/rr_stan_logistic_regression.ipynb"""
    import pandas as pd
    from modulars.logistic_regression import load_logistic_regression_data
    nb_dir = SINGLE_MC / "logistic_regression"
    os.chdir(nb_dir)

    data = load_logistic_regression_data()
    labels = data["labels"]
    dim = len(labels)

    stan_data = {
        "N": int(data["x_train"].shape[0]),
        "K": int(data["x_train"].shape[1]),
        "X": data["x_train"].astype(float),
        "y": data["y_train"].astype(int),
    }
    PARAM_COLUMNS = [f"b[{i}]" for i in range(1, dim + 1)]
    LOG_COLUMNS = []

    best_reference = pd.read_csv(nb_dir / "best_variational_reference.csv")
    assert list(best_reference["parameter"]) == list(labels), "reference rows must be in label order"
    best_mean = best_reference["mean"].to_numpy()
    best_std = best_reference["sd"].to_numpy()
    setups, failures = run_stan_seeds(nb_dir / "stan_logistic_regression.stan", stan_data, PARAM_COLUMNS, LOG_COLUMNS)
    return {"logistic_regression": dict(setups=setups, coordinates=list(labels), best_mean=best_mean, best_std=best_std,
                                        reference_source="single_MC/logistic_regression/best_variational_reference.csv")}, failures, 0


def model_dirichlet_multinomial():
    """single_MC/multinomial_dirichlet/rr_dirich_stan.ipynb, but with stan_dirichlet_multinomial_stickbreaking.stan

    Stan's built-in simplex transform (softmax-ILR) gives a different mean-field family from the
    stick-breaking transform the other packages use, so, like PyMC's Dirichlet runs, Stan uses a
    user-defined stick-breaking transform here and all packages share one reference.
    """
    from modulars.distributions import gen_multinomial_data_shared, lda_posterior_shared_theta
    from modulars.utils import load_best_multid_reference
    nb_dir = SINGLE_MC / "multinomial_dirichlet"
    os.chdir(nb_dir)

    config_file = Path("dirichlet_config.json")
    config = load_config(config_file)

    # float32, as in the pymc / tfp / numpyro Dirichlet notebooks: with float64
    # theta_like, np.random.multinomial draws different counts from the same seed
    theta_like = np.array(config["theta_like"], dtype=np.float32)
    alpha_prior = np.array(config["alpha_prior"], dtype=float)
    n_cats = int(config["n_cats"])
    N = int(config["N"])
    total_count = int(config["total_count"])
    seed = int(config.get("seed", 15))
    shared = bool(config.get("shared", True))
    if not shared:
        raise NotImplementedError("This Stan notebook matches the shared-theta setup used by the other methods.")

    n_vec = np.full(N, total_count, dtype=int)
    obs_counts = gen_multinomial_data_shared(theta_like, N, total_count, SEED=seed).astype(int)
    true_post = lda_posterior_shared_theta(observations=obs_counts, alpha=alpha_prior)

    best_mean, best_cov, best_std = load_best_multid_reference(
        config,
        fallback_mean=true_post["mean_post"],
        fallback_cov=np.diag(np.square(true_post["std_post"])),
    )

    stan_data = {
        "N": N,
        "K": n_cats,
        "total_count": n_vec,
        "y": obs_counts,
        "alpha_prior": alpha_prior,
    }
    PARAM_COLUMNS = [f"theta[{i}]" for i in range(1, n_cats + 1)]
    LOG_COLUMNS = []
    # stick-breaking instead of Stan's default softmax-ILR simplex transform, so Stan's variational
    # family (and reference) matches NumPyro / TFP / PyMC; see the .stan file's header
    # (full defaults: Stan's own simplex transform, stan_dirichlet_multinomial.stan as rr_dirich_stan.ipynb)
    stan_file = "stan_dirichlet_multinomial.stan" if FULL_DEFAULT else "stan_dirichlet_multinomial_stickbreaking.stan"
    setups, failures = run_stan_seeds(nb_dir / stan_file, stan_data, PARAM_COLUMNS, LOG_COLUMNS)
    return {"dirichlet_multinomial": dict(setups=setups, coordinates=[f"theta[{i}]" for i in range(n_cats)],
                                          best_mean=best_mean, best_std=best_std,
                                          reference_source="dirichlet_config.json via load_best_multid_reference")}, failures, 0


def model_radon():
    """single_MC/radon_example/pymc_tutorial/rr_stan_radon.ipynb

    As in the notebook, sigma_a / sd_y moments come from the log of the draws
    (LOG_COLUMNS) so they line up with best_reference_values.csv's
    sigma_a_log__ / sd_y_log__ rows; the constrained-scale copy then pushes them
    through exp_lognormal_moments like the other packages.
    """
    from modulars.radon import load_best_references, load_radon_data, make_radon_unconstrained_param_names
    nb_dir = SINGLE_MC / "radon_example" / "pymc_tutorial"
    os.chdir(nb_dir)

    radon_data = load_radon_data()
    county_names = radon_data["county_names"]
    param_names = make_radon_unconstrained_param_names(county_names)

    stan_data = {
        "N": int(len(radon_data["log_radon"])),
        "C": int(len(county_names)),
        "log_radon": np.asarray(radon_data["log_radon"], dtype=float),
        "floor": np.asarray(radon_data["floor"], dtype=float),
        "county": np.asarray(radon_data["county"], dtype=int) + 1,
    }
    PARAM_COLUMNS = (
        ["mu_a", "sigma_a"]
        + [f"alpha[{i}]" for i in range(1, len(county_names) + 1)]
        + ["beta", "sd_y"]
    )
    LOG_COLUMNS = ["sigma_a", "sd_y"]

    reference_summary, best_reference, reference_means, reference_stds = load_best_references()
    best_by_parameter = best_reference.set_index("parameter")
    best_mean = best_by_parameter.loc[param_names, "mean"].to_numpy()
    best_std = best_by_parameter.loc[param_names, "sd"].to_numpy()

    setups, failures = run_stan_seeds(nb_dir / "stan_radon_varying_intercept.stan", stan_data, PARAM_COLUMNS, LOG_COLUMNS)
    return bias_io.radon_outputs(setups, param_names, best_mean, best_std, exp_lognormal_moments), failures, 0


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
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        outputs, failures, seed_offset = MODEL_FUNCTIONS[args.model]()
        import cmdstanpy
        info = bias_io.base_run_info(args, PACKAGE, SEEDS, {
            "setups": {scenario["name"]: scenario for scenario in STAN_SCENARIOS},
            **({"full_default_note": "every CmdStan ADVI default (tol_rel_obj=0.01, output_samples=1000, eta adaptation, "
                                     "...) except iter; seed_seconds = wall-clock time of each seed's iter=max_iters run"
                                     + ("; Stan's own simplex transform (stan_dirichlet_multinomial.stan)"
                                        if args.model == "dirichlet_multinomial" else "")} if FULL_DEFAULT else {}),
            "stan_seeds": [seed_offset + s for s in SEEDS],
            "n_failed_checkpoints": len(failures),
            "failures": failures[:200],
            "runtime_seconds": time.time() - start,
            "max_workers": MAX_WORKERS,
            "cmdstan_path": str(STAN3_CMDSTAN),
            "cmdstanpy_version": cmdstanpy.__version__,
        })
        if failures:
            print(f"[stan/{args.model}] WARNING: {len(failures)} failed checkpoints (saved as NaN); see run_info.json")
        for model, out in outputs.items():
            bias_io.save_model_outputs(args, PACKAGE, model, out["setups"], ITERATIONS, out["coordinates"],
                                       out["best_mean"], out["best_std"], out["reference_source"], info)
