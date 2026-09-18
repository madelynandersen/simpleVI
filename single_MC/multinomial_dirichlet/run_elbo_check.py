"""Numpyro-style ELBO check for the Stan grad100_tol_rel_obj1e-6 scenario.

Scores grad100_tol_rel_obj1e-6's final variational approximation with the
same ELBO metric (modulars.elbo_computations.compute_elbo, model="multidirich",
guide="stickbreak_mvn") used by rr_dirich_find_best.ipynb to pick the
cross-framework "best variational approximation" recorded in
dirichlet_config.json, so the two numbers are directly comparable.

This needs jax/numpyro, which are not installed in the "stan3" conda
environment this notebook otherwise runs in -- rr_dirich_stan.ipynb invokes
this script as a subprocess under a separate interpreter that has them
(see the ELBO-check cell), the same way it shells out to the `stan`
CmdStan binary rather than importing it.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/Users/madelynandersen/Documents/VB Bakeoff/simpleVI")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NB_DIR = REPO_ROOT / "single_MC/multinomial_dirichlet"
os.chdir(NB_DIR)

import cmdstanpy
STAN3_CMDSTAN = Path("/Users/madelynandersen/miniconda3/envs/stan3/bin/cmdstan")
os.environ["CMDSTAN"] = str(STAN3_CMDSTAN)
os.environ["SIMPLEVI_CMDSTAN_PATH"] = str(STAN3_CMDSTAN)
cmdstanpy.set_cmdstan_path(str(STAN3_CMDSTAN))

from modulars.distributions import gen_multinomial_data_shared, lda_posterior_shared_theta
from modulars.utils import load_config
from modulars.stan_rr_test import run_stan_random_restarts, read_variational_draws
from modulars.elbo_computations import compute_elbo

config_file = Path("dirichlet_config.json")
config = load_config(config_file)

theta_like = np.array(config["theta_like"], dtype=float)
alpha_prior = np.array(config["alpha_prior"], dtype=float)
n_cats = int(config["n_cats"])
N = int(config["N"])
total_count = int(config["total_count"])
seed = int(config.get("seed", 15))

n_vec = np.full(N, total_count, dtype=int)
obs_counts = gen_multinomial_data_shared(theta_like, N, total_count, SEED=seed).astype(int)

STAN_FILE = NB_DIR / "stan_dirichlet_multinomial.stan"
PARAM_COLUMNS = [f"theta[{i}]" for i in range(1, n_cats + 1)]

stan_data = {
    "N": N,
    "K": n_cats,
    "total_count": n_vec,
    "y": obs_counts,
    "alpha_prior": alpha_prior,
}

CHECK_ITERS = 150_000  # matches the completed grad100_tol_rel_obj1e-6 scenario's final horizon
N_RESTARTS_CHECK = 3
GRAD_SAMPS = 10_000  # matches the grad_samps used to compute best_elbo_np in rr_dirich_find_best.ipynb

output_dir = NB_DIR / "results" / "elbo_check_runs"
output_dir.mkdir(parents=True, exist_ok=True)

scenario = {
    "name": "elbo_check",
    "mc_label": "elbo_check",
    "cmd_args": ("grad_samples=100",),
    "tol_rel_obj": 1e-6,
}

result = run_stan_random_restarts(
    stan_file=STAN_FILE,
    data=stan_data,
    param_columns=PARAM_COLUMNS,
    log_columns=[],
    output_dir=output_dir,
    max_iters=CHECK_ITERS,
    n_restarts=N_RESTARTS_CHECK,
    track_every=CHECK_ITERS,
    seed_offset=0,
    parallel=False,
    keep_outputs=True,
    refresh=0,
    scenarios=[scenario],
    show_progress=False,
)

if result["failures"]:
    print("FAILURES:", result["failures"])

print(f"===== rr_dirich_stan: numpyro-style ELBO check for grad100_tol_rel_obj1e-6 ({CHECK_ITERS} iters) =====")
print(f"reference best_elbo_np (from config, best across numpyro/pymc/tfp): {config['best_elbo_np']:.4f}")
print(f"  (source: {config['best_source_file']}, restart {config['best_restart_idx']})")
print(f"reference best_mean: {config['best_mean']}")
print()

check_rows = []
for restart_idx in range(N_RESTARTS_CHECK):
    seed_used = restart_idx
    run_dir = output_dir / "elbo_check" / f"restart_{restart_idx:03d}"
    csv_path = run_dir / f"iter_{CHECK_ITERS:06d}.csv"

    draws = read_variational_draws(csv_path, param_columns=PARAM_COLUMNS, log_columns=[])
    mean_K = draws.mean(axis=0)
    cov_K = np.cov(draws, rowvar=False, ddof=1)

    elbo_full_cov = compute_elbo(
        model_str="multidirich",
        guide_str="stickbreak_mvn",
        param_name="theta",
        model_vals=[alpha_prior, np.sum(obs_counts)],
        guide_vals=[mean_K, cov_K],
        data=obs_counts,
        with_data=True,
        grad_samps=GRAD_SAMPS,
        seed=seed,
    )

    diag_cov = np.diag(np.var(draws, axis=0, ddof=1))
    elbo_diag_cov = compute_elbo(
        model_str="multidirich",
        guide_str="stickbreak_mvn",
        param_name="theta",
        model_vals=[alpha_prior, np.sum(obs_counts)],
        guide_vals=[mean_K, diag_cov],
        data=obs_counts,
        with_data=True,
        grad_samps=GRAD_SAMPS,
        seed=seed,
    )

    print(f"restart {restart_idx} (stan seed={seed_used}, n_draws={draws.shape[0]}):")
    print(f"  mean: {mean_K}")
    print(f"  numpyro-style ELBO (full covariance)   : {elbo_full_cov:9.4f}   (vs best {config['best_elbo_np']:.4f}, diff={elbo_full_cov - config['best_elbo_np']:+.4f})")
    print(f"  numpyro-style ELBO (diagonal covariance): {elbo_diag_cov:9.4f}   (vs best {config['best_elbo_np']:.4f}, diff={elbo_diag_cov - config['best_elbo_np']:+.4f})")
    print()

    check_rows.append({
        "restart_idx": restart_idx,
        "mean": mean_K.tolist(),
        "elbo_full_cov": float(elbo_full_cov),
        "elbo_diag_cov": float(elbo_diag_cov),
    })

with open(output_dir / "elbo_check_summary.json", "w") as f:
    json.dump({
        "check_iters": CHECK_ITERS,
        "n_restarts_check": N_RESTARTS_CHECK,
        "grad_samps": GRAD_SAMPS,
        "reference_best_elbo_np": config["best_elbo_np"],
        "reference_best_source_file": config["best_source_file"],
        "rows": check_rows,
    }, f, indent=2)
