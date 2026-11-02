"""Check that choosing the best variational approximation by sampled KL (modulars/sampled_kl.py)
picks the same restart as the NumPyro ELBO scoring that chose each saved reference.

Within one variational family KL(q || p) = log p(y) - ELBO(q), so the two criteria rank the
same candidates in the same order and can only disagree through estimation error. This script
re-scores the saved candidate pools of the models whose references were chosen by an ELBO,
where every candidate is a mean-field Gaussian in the same coordinates:

  dirichlet_multinomial  60 stick-breaking restarts (NumPyro, TFP, PyMC stick-breaking) from
                         overall_bias_results/dirichlet_multinomial/reference_build. Two saved
                         scores: the latent stick-breaking ELBO of recompute_dirichlet_reference.py
                         (which chose the current reference), and modulars.utils.find_best_params
                         (NumPyro, used by rr_dirich_find_best.ipynb), recomputed here.
                         Exact check: 2-D Gauss-Hermite quadrature of the KL to the conjugate posterior.
  logistic_regression    the 100 restarts in single_MC/logistic_regression/
                         best_variational_reference_scored_restarts.csv (NumPyro TraceMeanField_ELBO).
                         Exact check: the ELBO with every expectation done by 1-D Gauss-Hermite
                         quadrature (each likelihood term depends on b only through x_i . b).
  radon                  the 30 restarts in single_MC/radon_example/pymc_tutorial/
                         best_reference_restarts.csv (NumPyro TraceMeanField_ELBO).
  journe                 the 100 PyMC and Stan restarts in Perennial_reproduction/best_vi/
                         restart_elbos.csv (NumPyro TraceMeanField_ELBO, best_vi.py score).

For each model the output has one row per candidate (saved ELBO, sampled KL with its standard
error, the paired standard error of its difference from the KL-best candidate, and the exact
value where available) and one summary row: whether the two criteria choose the same
candidate, their rank correlation, and, if they disagree, whether the two choices are
distinguishable and how far apart their fitted parameters are.

Run in an environment with numpyro (tfp_2 or num_pyro):
    python check_kl_selection.py [--models dirichlet_multinomial logistic_regression radon journe]
                                 [--n-draws 200000] [--out-dir ../overall_bias_results/kl_selection_check]
"""
import argparse
import ast
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bias_io  # noqa: E402

REPO = bias_io.REPO_ROOT
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import log_expit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from modulars.sampled_kl import (  # noqa: E402
    ElementwiseTransform, GaussianApproximation, compare_candidates, dirichlet_log_density, numpyro_log_joint,
    quadrature_kl, simplex_transform,
)
from modulars.utils import load_from_csv  # noqa: E402

MODELS = ("dirichlet_multinomial", "logistic_regression", "radon", "journe")


def score_pool(name, pool, log_target, n_draws, seed=0):
    """pool: DataFrame with candidate, saved_elbo, loc, scale and a shared transform in pool.attrs.

    Adds kl, kl_se, diff_from_kl_best, diff_se, kl_rank and elbo_rank columns.
    """
    transform = pool.attrs["transform"]
    candidates = {c: (GaussianApproximation(loc, scale), transform)
                  for c, loc, scale in zip(pool["candidate"], pool["loc"], pool["scale"])}
    start = time.time()
    table = compare_candidates(candidates, log_target, n_draws=n_draws, seed=seed)
    print(f"[{name}] sampled KL of {len(pool)} candidates, {n_draws:,} draws each, in {time.time() - start:.0f}s")
    out = pool.merge(table.rename(columns={"se": "kl_se", "rank": "kl_rank", "diff_from_best": "diff_from_kl_best"}),
                     on="candidate")
    out["elbo_rank"] = out["saved_elbo"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("kl_rank").reset_index(drop=True)


def summarize(name, scored, elbo_column="saved_elbo", exact_column=None):
    """One summary row comparing the ELBO choice with the KL choice (and the exact choice if known)."""
    elbo_best = scored.loc[scored[elbo_column].idxmax()]
    kl_best = scored.loc[scored["kl"].idxmin()]
    rho = spearmanr(-scored[elbo_column], scored["kl"]).correlation
    row = {
        "model": name, "elbo_score": elbo_column, "n_candidates": len(scored),
        "elbo_choice": elbo_best["candidate"], "kl_choice": kl_best["candidate"],
        "same_choice": elbo_best["candidate"] == kl_best["candidate"],
        "spearman_rank_correlation": rho,
        # how much worse the ELBO choice is by KL, and whether that is more than Monte Carlo error
        "kl_gap_of_elbo_choice": elbo_best["diff_from_kl_best"],
        "kl_gap_se": elbo_best["diff_se"],
        "elbo_choice_distinguishable": bool(elbo_best["distinguishable"]),
        "elbo_gap_of_kl_choice": elbo_best[elbo_column] - kl_best[elbo_column],
        # how different the two chosen fits are, in units of the KL choice's scale
        "max_loc_difference_in_scales": float(np.max(np.abs(np.asarray(elbo_best["loc"]) - np.asarray(kl_best["loc"]))
                                                     / np.asarray(kl_best["scale"]))),
        "max_scale_ratio_minus_1": float(np.max(np.abs(np.asarray(elbo_best["scale"]) / np.asarray(kl_best["scale"]) - 1))),
    }
    if exact_column is not None:
        exact_best = scored.loc[scored[exact_column].idxmin()]
        row.update({"exact_choice": exact_best["candidate"],
                    "kl_choice_is_exact_choice": kl_best["candidate"] == exact_best["candidate"],
                    "elbo_choice_is_exact_choice": elbo_best["candidate"] == exact_best["candidate"],
                    "exact_gap_of_elbo_choice": elbo_best[exact_column] - exact_best[exact_column],
                    "exact_gap_of_kl_choice": kl_best[exact_column] - exact_best[exact_column]})
    return row


"""
Candidate pools
"""
def dirichlet_multinomial(n_draws):
    from modulars.distributions import gen_multinomial_data_shared
    from modulars.utils import find_best_params, load_config

    config = load_config(REPO / "single_MC" / "multinomial_dirichlet" / "dirichlet_config.json")
    alpha_prior = np.array(config["alpha_prior"], dtype=float)
    obs_counts = gen_multinomial_data_shared(np.array(config["theta_like"], dtype=np.float32), config["N"],
                                             config["total_count"], SEED=config.get("seed", 15))
    n_cats = int(config["n_cats"])
    posterior = dirichlet_log_density(alpha_prior + obs_counts.sum(axis=0))
    transform = simplex_transform("stickbreaking", n_cats)   # TFP's IteratedSigmoidCentered is the same map

    build = REPO / "paper_vignettes" / "overall_bias_results" / "dirichlet_multinomial" / "reference_build"
    saved = pd.read_csv(build / "candidate_scores_latent_elbo.csv")
    pool = pd.DataFrame({
        "candidate": [f"{Path(f).stem.replace('_summary', '')}#{i}" for f, i in zip(saved["source_file"], saved["restart_idx"])],
        "source_file": saved["source_file"], "restart_idx": saved["restart_idx"],
        "saved_elbo": saved["elbo"],
        "loc": [np.array(ast.literal_eval(v)) for v in saved["latent_loc"]],
        "scale": [np.array(ast.literal_eval(v)) for v in saved["latent_scale"]],
    })
    pool.attrs["transform"] = transform
    scored = score_pool("dirichlet_multinomial", pool, posterior, n_draws)
    scored["exact_kl"] = [quadrature_kl(GaussianApproximation(loc, scale), transform, posterior, n_nodes=80)
                          for loc, scale in zip(scored["loc"], scored["scale"])]
    # ELBO + KL = log p(y) for every candidate, so this column should be constant up to the ELBO's error
    scored["saved_elbo_plus_exact_kl"] = scored["saved_elbo"] + scored["exact_kl"]

    # find_best_params (NumPyro compute_elbo with the "stickbreak_mvn" guide), scored on the same summaries
    files = sorted({str(REPO / f) for f in saved["source_file"]})
    info = find_best_params(file_name_list=files, alpha_prior=alpha_prior, obs_counts=obs_counts,
                            grad_samps=10_000, seed=int(config.get("seed", 15)))
    fbp = pd.DataFrame([{"source_file": r["source_file"], "restart_idx": int(r["restart_idx"]),
                         "find_best_params_elbo": float(r["best_elbo_np"])} for r in info["all_scored_runs"]])
    scored = scored.merge(fbp, on=["source_file", "restart_idx"], how="left")
    return scored, [summarize("dirichlet_multinomial", scored, "saved_elbo", "exact_kl"),
                    summarize("dirichlet_multinomial", scored, "find_best_params_elbo", "exact_kl")]


def logistic_regression(n_draws):
    from modulars.logistic_regression import load_logistic_regression_data

    base = REPO / "single_MC" / "logistic_regression"
    data = load_logistic_regression_data()
    x, y = np.asarray(data["x_train"], dtype=float), np.asarray(data["y_train"], dtype=float)
    sign = 2.0 * y - 1.0

    def log_joint(b):
        eta = b @ x.T
        return -0.5 * np.sum(b**2, axis=1) - 0.5 * b.shape[1] * np.log(2 * np.pi) + log_expit(sign * eta).sum(axis=1)

    nodes, weights = np.polynomial.hermite_e.hermegauss(100)
    weights = weights / np.sqrt(2 * np.pi)

    def exact_neg_elbo(loc, scale):
        """-ELBO: analytic KL to the N(0, I) prior minus 1-D quadrature of each likelihood term."""
        kl_prior = np.sum(-np.log(scale) + 0.5 * (scale**2 + loc**2) - 0.5)
        mu, tau = x @ loc, np.sqrt((x**2) @ scale**2)
        expected_loglik = (log_expit(sign[:, None] * (mu[:, None] + tau[:, None] * nodes)) @ weights).sum()
        return kl_prior - expected_loglik

    saved = pd.read_csv(base / "best_variational_reference_scored_restarts.csv")
    cache, rows = {}, []
    for _, r in saved.iterrows():
        if r["source_file"] not in cache:
            cache[r["source_file"]] = load_from_csv(REPO / r["source_file"])[0]
        single_m, single_s, multi_m, multi_s = cache[r["source_file"]]
        m, s = (single_m, single_s) if r["mc_setting"] == "1_mc" else (multi_m, multi_s)
        rows.append({"candidate": f"{r['method']}/{r['mc_setting']}#{int(r['restart_idx'])}",
                     "saved_elbo": r["best_elbo_np"],
                     "loc": np.asarray(m[int(r["restart_idx"]), -1], dtype=float),
                     "scale": np.asarray(s[int(r["restart_idx"]), -1], dtype=float)})
    pool = pd.DataFrame(rows)
    pool.attrs["transform"] = ElementwiseTransform("identity", x.shape[1])
    scored = score_pool("logistic_regression", pool, log_joint, n_draws)
    scored["exact_kl"] = [exact_neg_elbo(loc, scale) for loc, scale in zip(scored["loc"], scored["scale"])]
    return scored, [summarize("logistic_regression", scored, "saved_elbo", "exact_kl")]


def radon(n_draws):
    from modulars.radon import build_numpyro_radon_model, load_radon_data

    base = REPO / "single_MC" / "radon_example" / "pymc_tutorial"
    data = load_radon_data()
    n_counties = len(data["county_names"])
    model_args = tuple(np.asarray(data[k]) for k in ("log_radon", "floor", "log_uranium", "county", "floor_by_county"))
    log_joint = numpyro_log_joint(build_numpyro_radon_model(),
                                  [("mu_a", ()), ("sigma_a", ()), ("alpha", (n_counties,)), ("beta", ()), ("sd_y", ())],
                                  model_args=model_args)
    transform = ElementwiseTransform(["identity", "exp"] + ["identity"] * n_counties + ["identity", "exp"])

    saved = pd.read_csv(base / "best_reference_restarts.csv")
    cache, rows = {}, []
    for _, r in saved.iterrows():
        if r["input_file"] not in cache:
            cache[r["input_file"]] = load_from_csv(REPO / r["input_file"])[0]
        means, stds = cache[r["input_file"]]
        rows.append({"candidate": f"{r['package']}/{r['setting']}#{int(r['restart_idx'])}", "saved_elbo": r["numpyro_elbo"],
                     "loc": np.asarray(means[int(r["restart_idx"])], dtype=float),
                     "scale": np.asarray(stds[int(r["restart_idx"])], dtype=float)})
    pool = pd.DataFrame(rows)
    pool.attrs["transform"] = transform
    scored = score_pool("radon", pool, log_joint, n_draws)
    return scored, [summarize("radon", scored)]


def journe(n_draws):
    perennial = REPO / "Perennial_reproduction"
    sys.path.insert(0, str(perennial))
    cwd = os.getcwd()
    os.chdir(perennial)          # best_vi / pymc_reproducing use paths relative to their folder
    try:
        import best_vi
        d = best_vi.P.load_data()
        G = d["n_genus"]
        sites = best_vi.score_sites(G)
        model, _ = best_vi.score_model_and_guide(d)
        log_joint = numpyro_log_joint(model, [(name, shape) for name, shape, _ in sites])
        kinds = [kind for _, shape, kind in sites for _ in range(int(np.prod(shape)))]
        rows = []
        for package, main_file in [("pymc", best_vi.PYMC_FILE), ("stan", best_vi.STAN_FILE)]:
            seeds, means, sds = best_vi.read_moments(best_vi.moments_files(main_file), G)
            rows += [{"candidate": f"{package}#{int(s)}", "package": package, "seed": int(s), "loc": m, "scale": sd}
                     for s, m, sd in zip(seeds, means, sds)]
        saved = pd.read_csv(best_vi.ELBO_FILE)
    finally:
        os.chdir(cwd)
    pool = pd.DataFrame(rows).merge(saved.rename(columns={"elbo": "saved_elbo"}), on=["package", "seed"])
    pool = pool[np.isfinite(pool["saved_elbo"])].reset_index(drop=True)
    pool.attrs["transform"] = ElementwiseTransform(kinds)
    scored = score_pool("journe", pool, log_joint, n_draws)
    return scored, [summarize("journe", scored)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--n-draws", type=int, default=200_000, help="sampled-KL draws per candidate (shared by all)")
    ap.add_argument("--out-dir", default=str(bias_io.DEFAULT_RESULTS_DIR / "kl_selection_check"))
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)   # NumPyro log densities in float64, as best_vi.py scores

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    functions = {"dirichlet_multinomial": dirichlet_multinomial, "logistic_regression": logistic_regression,
                 "radon": radon, "journe": journe}
    summaries = []
    for name in args.models:
        scored, rows = functions[name](args.n_draws)
        keep = [c for c in scored.columns if c not in ("loc", "scale")]
        scored[keep].to_csv(out_dir / f"{name}_candidates.csv", index=False)
        summaries += rows
        pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    summary = pd.DataFrame(summaries)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(summary.T.to_string())
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
