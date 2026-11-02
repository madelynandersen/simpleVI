"""Recompute the Dirichlet-multinomial best variational reference for the current data
(run in the tfp_2 env, which has both TFP and NumPyro, after the numpyro / tfp / pymc Dirichlet runners).

This follows single_MC/multinomial_dirichlet/rr_dirich_find_best.ipynb, applied to
the overall-bias runs (which use the data generated from the config's seed):

  1. per-restart final simplex summaries, built exactly as rr_dirich_numpyro.ipynb
     (build_summary_runs_from_scale_results, stick-breaking, multi = 100 MC),
     rr_dirich_tfp.ipynb (build_summary_runs_from_scale_results, iterated sigmoid
     centered, multi) and rr_dirich_pymc.ipynb (build_summary_runs_from_single_cov_results,
     one file per PyMC scenario, here the four stick-breaking scenarios);
  2. score every candidate restart and keep the best (see --scorer);
  3. save_best_run_to_config writes the best run back into dirichlet_config.json
     (a copy of the config from before this recomputation is kept next to it);
  4. every package's saved dirichlet_multinomial reference csv is rewritten from
     the updated config with load_best_multid_reference, as the runners do.

--scorer latent_elbo (default): each candidate is scored by its own ELBO in the
latent space it was fitted in: z ~ N(loc, diag(scale^2)) with the fitted final
loc / scale, theta = T(z) with that package's simplex transform (stick-breaking
for NumPyro and the stick-breaking PyMC runs, TFP's IteratedSigmoidCentered for
TFP), and ELBO = E_q[log Dirichlet(theta) + log Multinomial(y | theta) + log|det dT|]
+ H(q), estimated with the same standard-normal draws for every candidate.
--scorer find_best_params: rr_dirich_find_best.ipynb's find_best_params
(grad_samps=10000), which rebuilds a stick-breaking MVN guide from each
candidate's simplex mean / covariance.

usage: python recompute_dirichlet_reference.py [--scorer latent_elbo|find_best_params] [--results-dir ...]
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bias_io  # noqa: E402

sys.path.insert(0, str(bias_io.REPO_ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit, gammaln, log_expit  # noqa: E402

from modulars.distributions import gen_multinomial_data_shared  # noqa: E402
from modulars.utils import (  # noqa: E402
    load_config, load_best_multid_reference, save_to_csv, find_best_params, save_best_run_to_config,
    build_summary_runs_from_scale_results, stickbreaking_simplex_forward, iterated_sigmoid_centered_forward,
)

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default=str(bias_io.DEFAULT_RESULTS_DIR))
parser.add_argument("--scorer", choices=("latent_elbo", "find_best_params"), default="latent_elbo")
parser.add_argument("--n-elbo-samples", type=int, default=200_000, help="(latent_elbo) Monte Carlo draws per candidate")
args = parser.parse_args()
RESULTS_DIR = Path(args.results_dir).resolve()
MODEL = "dirichlet_multinomial"
CACHE = RESULTS_DIR / "_seed_cache" / MODEL
OUT = RESULTS_DIR / MODEL / "reference_build"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = bias_io.REPO_ROOT / "single_MC" / "multinomial_dirichlet" / "dirichlet_config.json"
config = load_config(CONFIG_FILE)
theta_like = np.array(config["theta_like"], dtype=np.float32)
alpha_prior = np.array(config["alpha_prior"], dtype=np.float32)
n_cats = config["n_cats"]
N = config["N"]
total_count = config["total_count"]
seed = config.get("seed", 15)
obs_counts = gen_multinomial_data_shared(theta_like, N, total_count, SEED=seed)
print(f"data seed {seed}, counts per category {obs_counts.sum(0)}")


def load_seed_cache(package):
    files = sorted((CACHE / package).glob("seed_*.npz"))
    if not files:
        raise FileNotFoundError(f"no {package} seed cache in {CACHE / package}; run its Dirichlet runner first")
    out = []
    for f in files:
        with np.load(f) as z:
            out.append([z[f"arr_{i}"] for i in range(int(z["n_arrays"]))])
    return out


def build_summary_runs_from_single_cov_results(
        raw_results,
        n_cats,
        forward_fn,
        n_samples=50_000,
        seed=0):
    """
    we build final simplex summaries for one scenario where each restart stores
    only one latent mean/cov trajectory.
    (copied from rr_dirich_pymc.ipynb)
    """
    summary_runs = []
    for restart_idx, (loc_traj, cov_traj) in enumerate(raw_results):
        final_loc = np.asarray(loc_traj[-1], dtype=float)
        final_cov = np.asarray(cov_traj[-1], dtype=float)
        rng = np.random.default_rng(seed + restart_idx)
        z = rng.multivariate_normal(
            mean=final_loc,
            cov=final_cov,
            size=int(n_samples),
        )
        theta = forward_fn(z, n_cats=n_cats)
        summary_runs.append(
            {
                "restart_idx": restart_idx,
                "mean": theta.mean(axis=0),
                "cov": np.cov(theta, rowvar=False),
            }
        )
    return summary_runs


"""
Exact simplex transforms with their log |det Jacobian| (float64), for the latent ELBO
"""
def stickbreaking_with_logdet(z):
    """modulars.utils.stickbreaking_simplex_forward plus its triangular-Jacobian log determinant."""
    n, d = z.shape
    remaining = np.ones(n)
    theta = np.zeros((n, d + 1))
    logdet = np.zeros(n)
    for k in range(d):
        u = z[:, k] - np.log(d + 1 - (k + 1))
        frac = expit(u)
        theta[:, k] = remaining * frac
        # d theta_k / d z_k = remaining * frac * (1 - frac); theta_k does not depend on z_j, j > k
        logdet += np.log(remaining) + log_expit(u) + log_expit(-u)
        remaining = remaining * (1.0 - frac)
    theta[:, -1] = remaining
    return theta, logdet


def iterated_sigmoid_centered_with_logdet(z):
    """TFP's IteratedSigmoidCentered (as modulars.utils.iterated_sigmoid_centered_forward), in float64."""
    import tensorflow as tf
    import tensorflow_probability as tfp
    bij = tfp.bijectors.IteratedSigmoidCentered()
    z = tf.constant(z, dtype=tf.float64)
    theta = bij.forward(z).numpy()
    # TFP's fldj is w.r.t. the first K-1 simplex coordinates, the chart Dirichlet.log_prob uses
    logdet = bij.forward_log_det_jacobian(z, event_ndims=1).numpy()
    return theta, logdet


TRANSFORMS = {"stickbreaking": stickbreaking_with_logdet, "iterated_sigmoid_centered": iterated_sigmoid_centered_with_logdet}

counts = obs_counts.sum(axis=0).astype(float)
log_lik_const = float(np.sum(gammaln(obs_counts.sum(axis=1) + 1.0)) - np.sum(gammaln(obs_counts + 1.0)))
log_dirichlet_const = float(gammaln(alpha_prior.sum()) - np.sum(gammaln(alpha_prior)))


def log_joint(theta):
    """log Dirichlet(theta | alpha) + sum_n log Multinomial(y_n | theta), on the first K-1 coordinates."""
    log_theta = np.log(np.clip(theta, 1e-300, None))
    return log_dirichlet_const + log_theta @ (alpha_prior - 1.0) + log_lik_const + log_theta @ counts


EPS = np.random.default_rng(seed).standard_normal((int(args.n_elbo_samples), n_cats - 1))


def latent_elbo(loc, scale, transform):
    """ELBO of q = N(loc, diag(scale^2)) in the latent space it was fitted in, common random numbers."""
    z = loc[None, :] + scale[None, :] * EPS
    theta, logdet = TRANSFORMS[transform](z)
    entropy = float(np.sum(0.5 * np.log(2.0 * np.pi * np.e * scale**2)))
    values = log_joint(theta) + logdet
    return float(values.mean() + entropy), float(values.std() / np.sqrt(len(values)))


# 0. check the stick-breaking log determinant against finite differences and the modulars forward
z_check = np.random.default_rng(0).normal(size=(5, n_cats - 1))
theta_check, logdet_check = stickbreaking_with_logdet(z_check)
assert np.allclose(theta_check, stickbreaking_simplex_forward(z_check, n_cats=n_cats))
h = 1e-6
for i in range(len(z_check)):
    jac = np.column_stack([
        (stickbreaking_with_logdet(z_check[i:i + 1] + h * e)[0][0, :-1] - stickbreaking_with_logdet(z_check[i:i + 1] - h * e)[0][0, :-1]) / (2 * h)
        for e in np.eye(n_cats - 1)
    ])
    assert np.isclose(np.log(abs(np.linalg.det(jac))), logdet_check[i], atol=1e-6)
theta_check_tfp, _ = iterated_sigmoid_centered_with_logdet(z_check)
assert np.allclose(theta_check_tfp, iterated_sigmoid_centered_forward(z_check, n_cats=n_cats), atol=1e-5)
print("transform / log-determinant checks passed")


# 1. per-restart final summaries, as in each package's notebook, plus the latent candidates
summary_files = []
candidates = []   # (source_file, restart_idx, transform, final loc, final scale)

numpyro_results = [tuple(r) for r in load_seed_cache("numpyro")]
numpyro_summary = build_summary_runs_from_scale_results(
    numpyro_results, n_cats=n_cats, forward_fn=stickbreaking_simplex_forward,
    which="multi", n_samples=10_000, seed=seed,
)
save_to_csv(OUT / "numpyro_summary_restarts.csv", numpyro_summary)
summary_files.append(OUT / "numpyro_summary_restarts.csv")
for i, r in enumerate(numpyro_results):
    candidates.append((OUT / "numpyro_summary_restarts.csv", i, "stickbreaking", r[2][-1], r[3][-1]))

tfp_results = [tuple(r) for r in load_seed_cache("tfp")]
tfp_summary = build_summary_runs_from_scale_results(
    tfp_results, n_cats=n_cats, forward_fn=iterated_sigmoid_centered_forward,
    which="multi", n_samples=50_000, seed=seed,
)
save_to_csv(OUT / "tfp_summary_restarts.csv", tfp_summary)
summary_files.append(OUT / "tfp_summary_restarts.csv")
for i, r in enumerate(tfp_results):
    candidates.append((OUT / "tfp_summary_restarts.csv", i, "iterated_sigmoid_centered", r[2][-1], r[3][-1]))

pymc_raw = load_seed_cache("pymc")
pymc_scenarios = ("stickbreaking_only", "stickbreaking_mc100", "stickbreaking_adam", "stickbreaking_mc100_adam")
for k, scenario in enumerate(pymc_scenarios):
    raw_results = [(r[2 * k], r[2 * k + 1]) for r in pymc_raw]
    summary = build_summary_runs_from_single_cov_results(
        raw_results, n_cats=n_cats, forward_fn=stickbreaking_simplex_forward, n_samples=20_000, seed=seed,
    )
    path = OUT / f"pymc_summary_{scenario}.csv"
    save_to_csv(path, summary)
    summary_files.append(path)
    for i, (loc_traj, cov_traj) in enumerate(raw_results):
        candidates.append((path, i, "stickbreaking", loc_traj[-1], np.sqrt(np.diag(cov_traj[-1]))))

# 2. score every candidate restart
start = time.time()
if args.scorer == "find_best_params":
    best_info = find_best_params(
        file_name_list=[str(p) for p in summary_files],
        alpha_prior=alpha_prior,
        obs_counts=obs_counts,
        grad_samps=10000,
        seed=seed,
    )
    scores = pd.DataFrame([
        {"source_file": r["source_file"], "restart_idx": r["restart_idx"], "elbo": r["best_elbo_np"]}
        for r in best_info["all_scored_runs"]
    ])
else:
    from modulars.utils import load_from_csv
    summaries = {path: load_from_csv(path) for path in summary_files}
    rows = []
    for path, restart_idx, transform, loc, scale in candidates:
        elbo, elbo_se = latent_elbo(np.asarray(loc, dtype=float), np.asarray(scale, dtype=float), transform)
        summary = summaries[path][restart_idx]
        rows.append({
            "source_file": str(path.relative_to(bias_io.REPO_ROOT)),
            "restart_idx": restart_idx, "transform": transform, "elbo": elbo, "elbo_se": elbo_se,
            "latent_loc": np.round(np.asarray(loc, dtype=float), 6).tolist(),
            "latent_scale": np.round(np.asarray(scale, dtype=float), 6).tolist(),
            "simplex_mean": np.asarray(summary["mean"], dtype=float),
            "simplex_cov": np.asarray(summary["cov"], dtype=float),
        })
    scores = pd.DataFrame(rows)
    best = scores.loc[scores["elbo"].idxmax()]
    best_cov = np.asarray(best["simplex_cov"], dtype=float)
    best_info = {
        "best_mean": np.asarray(best["simplex_mean"], dtype=float),
        "best_cov": best_cov,
        "best_std": np.sqrt(np.clip(np.diag(best_cov), 0.0, None)),
        "best_elbo_np": float(best["elbo"]),
        "source_file": best["source_file"],
        "restart_idx": int(best["restart_idx"]),
    }
    scores = scores.drop(columns=["simplex_mean", "simplex_cov"])
scores.sort_values("elbo", ascending=False).to_csv(OUT / f"candidate_scores_{args.scorer}.csv", index=False)
print(f"scored {len(scores)} restarts with {args.scorer} in {time.time() - start:.0f}s "
      f"(all scores in {OUT / f'candidate_scores_{args.scorer}.csv'})")
print("best source file:", best_info["source_file"], "| restart:", best_info["restart_idx"], "| elbo:", round(best_info["best_elbo_np"], 4))

# 3. write it back to the config (keeping the version from before any recomputation)
backup = CONFIG_FILE.with_name(f"dirichlet_config.before_seed{seed}_reference.json")
if not backup.exists():
    shutil.copy2(CONFIG_FILE, backup)
save_best_run_to_config(CONFIG_FILE, best_info, extra_updates={"best_scorer": args.scorer})
config = load_config(CONFIG_FILE)
best_mean, best_cov, best_std = load_best_multid_reference(config)
np.set_printoptions(precision=4, suppress=True)
print("new best mean:", best_mean, " best std:", best_std)

# 4. rewrite each package's saved reference for this model
coordinates = [f"theta[{i}]" for i in range(n_cats)]
source = f"dirichlet_config.json via load_best_multid_reference (recomputed for the seed-{seed} data, {args.scorer})"
for package in bias_io.PACKAGE_SETUPS:
    path = bias_io.reference_path(RESULTS_DIR, MODEL, package)
    if path.exists():
        bias_io.save_reference(path, coordinates, best_mean, best_std, source)
        print("updated", path)
