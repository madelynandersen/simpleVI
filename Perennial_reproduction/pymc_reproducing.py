#!/usr/bin/env python3
"""PyMC re-implementation of the brms model behind Table S1 of Journe et al. 2025
(Nat Commun 16:9226), run with four inference methods:

  mcmc        pm.sample (NUTS), same chains/iterations/warmup as the brms run
  vi_default  pm.fit(method="advi") with PyMC defaults
  vi_mc100    as vi_default but obj_n_mc=100 (100 Monte-Carlo samples per gradient step)
  vi_adam     as vi_default but obj_optimizer=pm.adam() (default Adam settings)

MODEL (identical data and structure to run_repro.R; see brms_model_spec.txt, written by run_repro.R):
  Y_i ~ Dirichlet(mu_i * phi), weighted: log-likelihood_i is multiplied by weight_i = 1 / RMSE_i
  mu_i = softmax(0, eta_Pr,i, eta_De,i)               (temperature TeB is the reference category)
  eta_k,i = b0_k + b_MAT,k * MAT_i + b_P1,k * P1_i + b_P2,k * P2_i + r_k[genus_i]
  MAT, MAP standardised (sd with n-1, like base::scale); P1, P2 = poly(MAP_scaled, 2) as in R
  priors (brms defaults, ASSUMED - check them against brms_model_spec.txt):
    intercepts ~ student_t(3, 0, 2.5);  slopes ~ flat;  genus sd ~ half student_t(3, 0, 2.5)
    phi ~ gamma(0.01, 0.01);  random effects non-centred (r = sd * z, z ~ N(0, 1)), as brms does

Outputs (same columns as the R tables: term, Estimate, Est.Error, Q2.5, Q97.5, plus seed, method):
  TableS1_py_mcmc.csv, TableS1_py_vi_default.csv, TableS1_py_vi_mc100.csv, TableS1_py_vi_adam.csv
  and, if TableS1_reproduced.csv (R NUTS) exists, TableS1_py_<method>_vs_NUTS.csv for each.

Usage (run from the folder that holds species.ecoregion.data.csv):
  python pymc_reproducing.py
  python pymc_reproducing.py --only mcmc,vi_adam
Iterations: --iters N sets the number of ADVI iterations (default 10000 = the PyMC default). With any other N the
output files get a suffix (e.g. TableS1_py_vi_default_75k.csv), so the 10k tables used by compare_plots.R are not
overwritten.

Trajectory plots: the ADVI fits record the mean and SD of the 8 Table S1 coefficients at every iteration; after the
runs, extra_figs/pymc_trajectories_means[_75k].png and extra_figs/pymc_trajectories_sd[_75k].png (plus
extra_figs/pymc_trajectory_<method>[_75k].csv) are written. Each panel has the R NUTS value as a dashed black line
and the best VB approximation (TableS1_best_vi.csv) as a solid black line; runs longer than 10000 iterations get a
dashed vertical line at 10000 marked "default number of iterations". Needs matplotlib.
Random restarts (--restarts): seeds 0..9 (--n-seeds) of default, 100 MC, Adam and Adam + 100 MC, 75,000 iterations
(--iters), keeping the last 500 iterations (--window) of the 8 coefficients' mean and SD for
paper_vignettes/overall_bias_comparisons.ipynb (see journe_restarts.py). Needs simpleVI's modulars.
With --restarts --full_defaults: only the default fit, for PyMC's default 10,000 iterations, saved to
overall_bias_results_full_default/ with each fit's time (paper_vignettes/overall_default_bias_comparisons.ipynb).
--timing_analysis: instead of the normal run, time every ADVI setting (default, obj_n_mc=100, Adam, Adam + obj_n_mc=100 and the full defaults;
--only restricts them) over seeds 11-20 (--timing_seeds, e.g. "11-20" or "11,13"): one attempt per seed, no retries, the total wall
time of fit + drawing 10,000 samples + summarising, with no trajectory recording. Writes timing_pymc.csv (one row per seed) and
timing_pymc_summary.csv; no Table S1 files or figures are written. Settings: full_defaults (pm.fit with no arguments, 10,000
iterations) and vi_default, vi_mc100, vi_adam and vi_adam_mc100 at --iters iterations (default 150,000 here, not 10,000).
Requires: numpy, pandas, pymc (>= 5), matplotlib (for the trajectory plots).
"""
import argparse
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=/usr/bin/clang++")

import numpy as np
import pandas as pd


DATA_FILE = "species.ecoregion.data.csv"
NUTS_FILE = "TableS1_reproduced.csv"

# Row order of brms' fixef() output for this model.
TERMS = ["muPrB_Intercept", "muDeB_Intercept",
         "muPrB_MAT", "muPrB_polyMAP21", "muPrB_polyMAP22",
         "muDeB_MAT", "muDeB_polyMAP21", "muDeB_polyMAP22"]

METHODS = {  # tag -> label
    "mcmc": "PyMC NUTS",
    "vi_default": "PyMC ADVI default",
    "vi_mc100": "PyMC ADVI obj_n_mc=100",
    "vi_adam": "PyMC ADVI adam",
}


# ----------------------------------------------------------------------------- data
def r_scale(x):
    """base::scale(): centre and divide by the sample sd (n - 1)."""
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=1)


def r_poly2(x):
    """Equivalent of R's poly(x, 2): orthonormal polynomial columns (unit length, orthogonal to the
    constant), with the sign convention R uses (positive coefficient on x^j)."""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    V = np.column_stack([np.ones_like(xc), xc, xc ** 2])
    Q = np.zeros_like(V)
    for j in range(V.shape[1]):
        v = V[:, j].copy()
        for k in range(j):
            v -= (Q[:, k] @ v) * Q[:, k]          # modified Gram-Schmidt
        Q[:, j] = v / np.linalg.norm(v)
    return Q[:, 1:]


def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    mat = r_scale(df["MAT"])
    map_z = r_scale(df["MAP"])
    poly = r_poly2(map_z)
    X = np.column_stack([mat, poly[:, 0], poly[:, 1]])           # MAT, poly(MAP,2)1, poly(MAP,2)2
    Y = df[["TeB", "PrB", "DeB"]].to_numpy(dtype=float)
    w = 1.0 / df["RMSE"].to_numpy(dtype=float)
    g, genus_levels = pd.factorize(df["Genus"])                  # order of appearance, like as_factor
    return dict(X=X, Y=Y, w=w, g=g.astype(int), n_genus=len(genus_levels), n=len(df))


def dirichlet_loglik_numpy(Y, mu, phi):
    """log Dirichlet(Y | mu * phi) per row (reference implementation used for testing)."""
    from math import lgamma
    alpha = mu * phi
    out = np.empty(len(Y))
    for i in range(len(Y)):
        out[i] = lgamma(alpha[i].sum()) - sum(lgamma(a) for a in alpha[i]) + ((alpha[i] - 1) * np.log(Y[i])).sum()
    return out


# ----------------------------------------------------------------------------- model
def build_model(d):
    import pymc as pm
    import pytensor.tensor as pt
    
    import os
    import pytensor
    if os.environ.get("SIMPLEVI_PYTENSOR_CXX"):
        pytensor.config.cxx = os.environ["SIMPLEVI_PYTENSOR_CXX"]

    X, Y, w, g = d["X"], d["Y"], d["w"], d["g"]
    n, k = X.shape
    logY = np.log(Y)
    with pm.Model() as model:
        b_Intercept = pm.StudentT("b_Intercept", nu=3, mu=0.0, sigma=2.5, shape=2)   # [Pr, De]
        b = pm.Flat("b", shape=(k, 2))                                               # rows: MAT, P1, P2
        sd = pm.HalfStudentT("sd", nu=3, sigma=2.5, shape=2)
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=(d["n_genus"], 2))
        phi = pm.Gamma("phi", alpha=0.01, beta=0.01)

        eta = b_Intercept + pt.dot(X, b) + (z * sd)[g]                               # (n, 2)
        eta_full = pt.concatenate([pt.zeros((n, 1)), eta], axis=1)                   # TeB reference
        mu = pt.special.softmax(eta_full, axis=1)
        alpha = mu * phi
        ll = pt.gammaln(phi) - pt.gammaln(alpha).sum(axis=1) + ((alpha - 1.0) * logY).sum(axis=1)
        pm.Potential("loglik", (w * ll).sum())                                       # brms weights()
    return model


# ----------------------------------------------------------------------------- summaries
def _flat(post, name):
    a = np.asarray(post[name])
    return a.reshape(-1, *a.shape[2:])                                               # merge chain, draw


def term_draws(post):
    bI, b = _flat(post, "b_Intercept"), _flat(post, "b")
    return {"muPrB_Intercept": bI[:, 0], "muDeB_Intercept": bI[:, 1],
            "muPrB_MAT": b[:, 0, 0], "muPrB_polyMAP21": b[:, 1, 0], "muPrB_polyMAP22": b[:, 2, 0],
            "muDeB_MAT": b[:, 0, 1], "muDeB_polyMAP21": b[:, 1, 1], "muDeB_polyMAP22": b[:, 2, 1]}


def summary_table(draws, seed, method, final_loss=np.nan):
    """Same columns as brms::fixef(): Estimate = mean, Est.Error = sd, Q2.5 / Q97.5 quantiles."""
    rows = []
    for t in TERMS:
        x = np.asarray(draws[t], dtype=float)
        rows.append(dict(term=t, Estimate=x.mean(), **{"Est.Error": x.std(ddof=1)},
                         **{"Q2.5": np.quantile(x, 0.025), "Q97.5": np.quantile(x, 0.975)},
                         seed=seed, method=method, final_loss=final_loss))
    return pd.DataFrame(rows)


def compare_to_nuts(tab, nuts, label):
    """Same quantities as run_repro_vi.R's comparison tables (py_* columns = the run being compared)."""
    nuts = nuts.copy()
    if "seed" not in nuts.columns:
        nuts["seed"] = np.nan
    n = nuts[["term", "seed", "Estimate", "Est.Error", "Q2.5", "Q97.5"]].rename(columns={
        "seed": "nuts_seed", "Estimate": "nuts_mean", "Est.Error": "nuts_sd",
        "Q2.5": "nuts_q2.5", "Q97.5": "nuts_q97.5"})
    p = tab[["term", "seed", "Estimate", "Est.Error", "Q2.5", "Q97.5"]].rename(columns={
        "seed": "py_seed", "Estimate": "py_mean", "Est.Error": "py_sd",
        "Q2.5": "py_q2.5", "Q97.5": "py_q97.5"})
    if set(n["term"]) != set(p["term"]):
        raise ValueError(f"Parameter names differ between NUTS and {label}: "
                         f"{sorted(set(n['term']) ^ set(p['term']))}")
    m = n.merge(p, on="term")
    m.insert(0, "comparison", f"{label} vs NUTS")
    m["offset_in_nuts_sd"] = (m["py_mean"] - m["nuts_mean"]) / m["nuts_sd"]
    m["sd_rel_error_pct"] = 100 * (m["py_sd"] - m["nuts_sd"]) / m["nuts_sd"]
    m["interval_excludes_zero_nuts"] = np.sign(m["nuts_q2.5"]) == np.sign(m["nuts_q97.5"])
    m["interval_excludes_zero_py"] = np.sign(m["py_q2.5"]) == np.sign(m["py_q97.5"])
    m["same_significance"] = m["interval_excludes_zero_nuts"] == m["interval_excludes_zero_py"]
    return m


# ----------------------------------------------------------------------------- inference
TRAJ = {}      # tag -> dict(loss=..., iters=..., means={term: trajectory}) filled by run_vi for the trajectory plots


def _term_trajectories(mean_mat, ordering):
    """Tracked flat mean vectors (iterations x parameters) -> {Table S1 term: mean trajectory}."""
    def piece(name):
        o = ordering[name]
        slc, shp = getattr(o, "slc", None) or o[1], getattr(o, "shp", None) or o[2]
        return mean_mat[:, slc].reshape(len(mean_mat), *shp)
    bI, b = piece("b_Intercept"), piece("b")
    return {"muPrB_Intercept": bI[:, 0], "muDeB_Intercept": bI[:, 1],
            "muPrB_MAT": b[:, 0, 0], "muPrB_polyMAP21": b[:, 1, 0], "muPrB_polyMAP22": b[:, 2, 0],
            "muDeB_MAT": b[:, 0, 1], "muDeB_polyMAP21": b[:, 1, 1], "muDeB_polyMAP22": b[:, 2, 1]}

def run_mcmc(model, seed, draws=9000, tune=1000, chains=4, cores=4):
    """brms setup: 4 chains, iter = 10000 with warmup = 1000  ->  9000 kept draws + 1000 tuning each."""
    import pymc as pm
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          random_seed=seed, progressbar=False)
    try:
        import arviz as az
        rhat = float(az.rhat(idata, var_names=["b_Intercept", "b", "sd", "phi"]).to_array().max())
        ndiv = int(idata.sample_stats["diverging"].sum())
        print(f"  NUTS diagnostics: max R-hat = {rhat:.3f}, divergences = {ndiv}")
    except Exception as e:                                    # diagnostics are informative only
        print("  (could not compute NUTS diagnostics:", e, ")")
    return idata.posterior, np.nan


class _NoTracking(Exception):
    pass


def run_vi(model, tag, seeds=range(11, 21), n_draws=10000, n_iter=10000, track=True):
    """PyMC ADVI variants. Try seeds 11..20 and report the first that does not fail. Seeds start at 11 to keep the ADVI seeds separate from the NUTS seed (1)."""

    import pymc as pm
    kwargs = {"vi_default": {}, "vi_mc100": {"obj_n_mc": 100},
              "vi_adam": {"obj_optimizer": pm.adam()},
              "vi_adam_mc100": {"obj_n_mc": 100, "obj_optimizer": pm.adam()}}[tag]       # vi_adam_mc100: --timing_analysis only
    last_err = None
    for s in seeds:
        print(f"  attempt with seed {s}")
        try:
            with model:
                try:      # same fit as pm.fit(method="advi", ...), but recording the mean of every parameter each iteration
                    if not track:                             # --timing_analysis: plain pm.fit, no recording
                        raise _NoTracking()
                    advi = pm.ADVI(random_seed=s)
                    tracker = pm.callbacks.Tracker(mean=advi.approx.mean.eval, std=advi.approx.std.eval)
                    approx = advi.fit(n=n_iter, progressbar=False, callbacks=[tracker], **kwargs)
                    traj_means = _term_trajectories(np.asarray(tracker["mean"]), advi.approx.ordering)
                    traj_sds = _term_trajectories(np.asarray(tracker["std"]), advi.approx.ordering)
                except Exception as e:                        # noqa: BLE001  (trajectories are optional)
                    if track:
                        print(f"    (could not record parameter trajectories: {e}; fitting without them)")
                    approx = pm.fit(n=n_iter, method="advi", random_seed=s, progressbar=False, **kwargs)
                    traj_means = traj_sds = None
            hist = np.asarray(approx.hist)
            if not np.all(np.isfinite(hist)):
                raise FloatingPointError("non-finite ELBO during optimisation")
            post = approx.sample(draws=n_draws, random_seed=s).posterior
            if track:
                TRAJ[tag] = dict(seed=s, loss=hist, means=traj_means, sds=traj_sds)
            return post, s, float(hist[-1000:].mean())
        except Exception as e:                                # noqa: BLE001
            last_err = e
            print(f"    failed: {e}")
    raise RuntimeError(f"ADVI ({tag}) failed for all seeds; last error: {last_err}")


RESTART_SETUPS = {  # overall_bias_comparisons setup -> (MC samples per step, optimizer), as modulars.run_single_seed_pymc_VI
    "default": (1, "default"), "100mc": (100, "default"), "adam": (1, "adam"), "adam_100mc": (100, "adam")}


def run_restarts(args, n_iter):
    """--restarts: every RESTART_SETUPS fit for seeds 0..n_seeds-1, keeping the last --window iterations of the
    8 Table S1 coefficients (simpleVI's modulars.run_pymc_VI with track_start, as overall_bias_runners/pymc_runner.py)."""
    import pymc as pm
    import journe_restarts as J
    sys.path.insert(0, str(J.bias_io.REPO_ROOT))
    from modulars.pymc_rr_test import run_pymc_VI

    rargs = J.restart_args(args, n_iter)
    track_start = J.bias_io.track_start(rargs.max_iters, rargs.window)
    # --full_defaults: only the default fit (1 MC sample, adagrad_window), as overall_bias_runners/pymc_runner.py
    setups = {J.bias_io.FULL_DEFAULT_SETUP: (1, "default")} if rargs.full_default else RESTART_SETUPS
    model = build_model(load_data())
    ordering = pm.ADVI(model=model).approx.ordering          # flat parameter layout, the same for every fit
    failures = []

    def one_seed(seed):
        arrays = []
        for setup, (n_mc, optimizer) in setups.items():
            try:
                mean, std = run_pymc_VI(model, n_mc, rargs.max_iters, optimizer, seed, track_start=track_start)
                arrays += [J.term_matrix(_term_trajectories(np.asarray(mean), ordering)),
                           J.term_matrix(_term_trajectories(np.asarray(std), ordering))]
            except Exception as e:                            # noqa: BLE001  (a failed fit is saved as NaN)
                print(f"  {setup} seed {seed} failed: {e}")
                failures.append(dict(setup=setup, seed=seed, error=str(e)))
                arrays += [J.nan_window(rargs), J.nan_window(rargs)]
        return arrays

    print(f"PyMC {pm.__version__} | restarts: {rargs.n_seeds} seeds x {list(setups)}, "
          f"{rargs.max_iters:,} iterations, last {rargs.window} tracked")
    J.run_and_save(rargs, "pymc", list(setups), one_seed, {
        "setups": {k: dict(obj_n_mc=n, optimizer=o) for k, (n, o) in setups.items()},
        "failures": failures, "pymc_version": pm.__version__})


# ----------------------------------------------------------------------------- trajectory plots
TRAJ_COLOURS = {"vi_default": "#E41A1C", "vi_mc100": "#0072B2", "vi_adam": "#CC79A7"}
TERM_ORDER = ["muPrB_Intercept", "muPrB_MAT", "muPrB_polyMAP21", "muPrB_polyMAP22",
              "muDeB_Intercept", "muDeB_MAT", "muDeB_polyMAP21", "muDeB_polyMAP22"]
TERM_TITLES = {"muPrB_Intercept": "Precipitation: Intercept", "muPrB_MAT": "Precipitation: MAT",
               "muPrB_polyMAP21": "Precipitation: MAP", "muPrB_polyMAP22": "Precipitation: MAP\u00b2",
               "muDeB_Intercept": "Moisture deficit: Intercept", "muDeB_MAT": "Moisture deficit: MAT",
               "muDeB_polyMAP21": "Moisture deficit: MAP", "muDeB_polyMAP22": "Moisture deficit: MAP\u00b2"}
BEST_FILE = "TableS1_best_vi.csv"
DEFAULT_ITERS = 10000
RESTART_ITERS = 75000      # --restarts (overall_bias_comparisons.ipynb)


def _grid_plot(key, ref_col, ylabel, title, fname, nuts, best, n_iter, out):
    """2 x 4 grid of trajectories (key = 'means' or 'sds'); reference lines from the Estimate / Est.Error column."""
    import matplotlib.pyplot as plt
    runs = {tag: t for tag, t in TRAJ.items() if t.get(key) is not None}
    if not runs:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax, term in zip(axes.ravel(), TERM_ORDER):
        for tag, t in runs.items():
            y = np.asarray(t[key][term])
            ax.plot(np.arange(1, len(y) + 1), y, color=TRAJ_COLOURS.get(tag, "grey"), lw=1.5,
                    label=f"{METHODS[tag]} (seed {t['seed']})")
        if nuts is not None and term in nuts.index:
            ax.axhline(nuts.loc[term, ref_col], color="black", ls=(0, (3, 3)), lw=2, label="R NUTS (reference)")
        if best is not None and term in best.index:
            ax.axhline(best.loc[term, ref_col], color="black", ls="-", lw=2, label="best VB approx")
        if n_iter > DEFAULT_ITERS:
            ax.axvline(DEFAULT_ITERS, color="grey", ls="--", lw=1.5, label="default number of iterations")
            ax.text(DEFAULT_ITERS, 0.97, " default number\n of iterations", transform=ax.get_xaxis_transform(),
                    va="top", ha="left", fontsize=10, color="dimgrey")
        ax.set_title(TERM_TITLES[term], fontsize=13, fontweight="bold")
        ax.set_xlabel("ADVI iteration")
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    keep = [i for i, name in enumerate(l) if name != "default number of iterations"]
    fig.legend([h[i] for i in keep], [l[i] for i in keep], loc="lower center", ncol=len(keep), frameon=False, fontsize=12)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out / fname, dpi=300)
    plt.close(fig)
    return fname


def plot_trajectories(nuts, n_iter=DEFAULT_ITERS, suffix="", outdir="extra_figs"):
    """Coefficient-mean and coefficient-SD trajectories of the PyMC ADVI fits that ran in this call."""
    if not TRAJ:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping the trajectory plots (pip install matplotlib)")
        return
    out = Path(outdir)
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 13})
    nuts_i = nuts.set_index("term") if nuts is not None else None
    best_i = pd.read_csv(BEST_FILE).set_index("term") if Path(BEST_FILE).exists() else None
    if best_i is None:
        print(f"Note: {BEST_FILE} not found - no best-VB reference line in the trajectory plots")

    done = [_grid_plot("means", "Estimate", "Mean of the coefficient (logit scale)",
                       f"PyMC ADVI trajectories of the Table S1 coefficient means ({n_iter:,} iterations)",
                       f"pymc_trajectories_means{suffix}.png", nuts_i, best_i, n_iter, out),
            _grid_plot("sds", "Est.Error", "Posterior SD of the coefficient (logit scale)",
                       f"PyMC ADVI trajectories of the Table S1 coefficient SDs ({n_iter:,} iterations)",
                       f"pymc_trajectories_sd{suffix}.png", nuts_i, best_i, n_iter, out)]
    for tag, t in TRAJ.items():                       # numbers behind the plots
        if t.get("means") is None:
            continue
        df = pd.DataFrame({"iteration": np.arange(1, len(t["loss"]) + 1), "loss": t["loss"]})
        for term in TERM_ORDER:
            df[f"{term}_mean"] = np.asarray(t["means"][term])[:len(df)]
            df[f"{term}_sd"] = np.asarray(t["sds"][term])[:len(df)]
        df.to_csv(out / f"pymc_trajectory_{tag}{suffix}.csv", index=False)
    print("Wrote", ", ".join(f for f in done if f), "to", out.resolve())


# ----------------------------------------------------------------------------- main
def run_full_defaults(model, seed=1, n_draws=10000):
    """pm.fit with NO arguments except the seed: method="advi", n=10000, obj_n_mc=1 and the default optimiser
    (adagrad_window). The 10,000 draws are only for summarising the fitted approximation, as in every other table."""
    import pymc as pm
    with model:
        approx = pm.fit(random_seed=seed)
        hist = np.asarray(approx.hist)
        post = approx.sample(draws=n_draws, random_seed=seed).posterior
    return post, float(hist[-1000:].mean())


# ----------------------------------------------------------------------------- --timing_analysis
def parse_seeds(text):
    """'11-20' -> [11, ..., 20];  '11,13,15' -> [11, 13, 15];  '1-3,7' -> [1, 2, 3, 7]."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def time_seeds(package, setting, label, seeds, fit_fn, iterations, mc_samples, rows, save=None):
    """Time fit_fn(seed) for every seed: ONE attempt per seed, no retries, failures recorded. fit_fn must do everything
    (fit, draw from the approximation, summarise), so the time is the total wall time of the seed. One dict per seed is
    appended to rows (and save(rows) is called after every seed, so a killed run keeps the finished seeds); run_order counts the fits of this invocation, so compile / JIT overhead in the first ones is visible."""
    for s in seeds:
        print(f"  [{label}] seed {s} ...", flush=True)
        t0 = time.perf_counter()
        status, err = "ok", ""
        ran = None
        try:
            ran = fit_fn(s)                      # may return the number of iterations actually run (else = iterations)
        except Exception as e:                                # noqa: BLE001
            status, err = "failed", " ".join(str(e).split())[:300]
        secs = time.perf_counter() - t0
        print(f"    {status} in {secs:.1f} s" + (f" ({err})" if err else ""), flush=True)
        rows.append(dict(package=package, setting=setting, label=label, seed=s, iterations=iterations,
                         mc_samples=mc_samples, seconds=round(secs, 3), status=status, error=err,
                         iterations_run=(float("nan") if status != "ok" else iterations if ran is None else ran), run_order=len(rows) + 1, timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))
        if save is not None:
            save(rows)
    return rows


def write_timing(package, rows, quiet=False):
    """timing_<package>.csv (one row per seed) and timing_<package>_summary.csv (one row per setting + iterations).
    Rows already in the file for the same setting and iterations are replaced; other settings are kept."""
    file, sfile = f"timing_{package}.csv", f"timing_{package}_summary.csv"
    df = pd.DataFrame(rows)
    if Path(file).exists():
        old = pd.read_csv(file, dtype={"error": str}).fillna({"error": ""})
        norm = lambda v: -1.0 if pd.isna(v) else float(v)         # the compile row has no iterations (NaN)
        keys = {(a, norm(b)) for a, b in zip(df["setting"], df["iterations"])}
        old = old[[(a, norm(b)) not in keys for a, b in zip(old["setting"], old["iterations"])]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(file, index=False)
    by = ["setting", "iterations"]
    runs = df[df["setting"] != "stan_compile"]            # (the one-off compile row is not a seed)
    ok = runs[runs["status"] == "ok"]
    summ = (ok.groupby(by)["seconds"].agg(n_ok="count", mean_s="mean", sd_s="std", median_s="median",
                                          min_s="min", max_s="max", total_s="sum")
            .join(runs[runs["status"] != "ok"].groupby(by).size().rename("n_failed"), how="outer"))
    summ["n_ok"] = summ["n_ok"].fillna(0).astype(int)
    summ["n_failed"] = summ["n_failed"].fillna(0).astype(int)
    summ = summ.reset_index().merge(runs.drop_duplicates(by)[by + ["package", "label", "mc_samples"]], on=by)
    summ = summ[["package", "label", "setting", "iterations", "mc_samples", "n_ok", "n_failed",
                 "mean_s", "sd_s", "median_s", "min_s", "max_s", "total_s"]]
    summ.to_csv(sfile, index=False)
    if quiet:
        return
    print(f"\nTiming (seconds per seed, total wall time) - wrote {file} and {sfile}:")
    print(summ.drop(columns=["package"]).round(1).to_string(index=False))


# --timing_analysis settings, in run order: the full defaults (pm.fit with no arguments, 10,000 iterations) and every ADVI setup
# at TIMING_ITERS iterations (150,000: long enough to converge, as in the 150k figures); vi_adam_mc100 is used only here
TIMING_ITERS = 150000
TIMING_LABELS = {"full_defaults": "PyMC full defaults", "vi_default": METHODS["vi_default"], "vi_mc100": METHODS["vi_mc100"],
                 "vi_adam": METHODS["vi_adam"], "vi_adam_mc100": "PyMC ADVI adam + obj_n_mc=100"}


def run_timing_analysis(args):
    """--timing_analysis: time every PyMC ADVI setting over the same seeds (one attempt per seed, total wall time)."""
    only = [m.strip() for m in args.only.split(",") if m.strip()]
    settings = list(TIMING_LABELS) if args.only == ",".join(METHODS) else [m for m in only if m != "mcmc"]
    bad = [m for m in settings if m not in TIMING_LABELS]
    if bad:
        sys.exit(f"Unknown setting(s): {bad}. Choose from {list(TIMING_LABELS)}")
    if not Path(DATA_FILE).exists():
        sys.exit(f"{DATA_FILE} not found in {Path.cwd()} - run this from the Perennial_reproduction folder")
    seeds, iters = parse_seeds(args.timing_seeds), args.iters or TIMING_ITERS
    import pymc as pm
    print(f"PyMC {pm.__version__} | timing analysis: settings {settings} | seeds {seeds} | ADVI iterations {iters} "
          "(full defaults: PyMC's own 10,000)")
    model = build_model(load_data())
    rows = []
    for setting in settings:
        if setting == "full_defaults":
            label, n_it, mc = "PyMC full defaults", 10000, 1

            def fit(seed, label=label):
                post, loss = run_full_defaults(model, seed=seed)
                summary_table(term_draws(post), seed=seed, method=label, final_loss=loss)
        else:
            label, n_it, mc = TIMING_LABELS[setting], iters, (100 if "mc100" in setting else 1)

            def fit(seed, tag=setting, label=label):
                post, s, loss = run_vi(model, tag, seeds=[seed], n_iter=iters, track=False)
                summary_table(term_draws(post), seed=seed, method=label, final_loss=loss)
        print(f"\n=== {label}: {len(seeds)} seeds, {n_it} iterations ===")
        time_seeds("pymc", setting, label, seeds, fit, n_it, mc, rows, save=lambda r: write_timing("pymc", r, quiet=True))
        write_timing("pymc", rows)           # written after every setting, so an interrupted run keeps what finished


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(METHODS),
                    help="comma-separated subset of: " + ", ".join(METHODS))
    ap.add_argument("--iters", type=int, default=None,
                    help="number of ADVI iterations (default 10000 = PyMC default, 75000 with --restarts); other values "
                         "add a suffix to the output files")
    ap.add_argument("--mcmc-seed", type=int, default=1, help="seed for NUTS (R run used seed 1)")
    ap.add_argument("--full_defaults", action="store_true",
                    help="run ONLY pm.fit(random_seed=1) with every other argument at the PyMC default (10,000 iterations, "
                         "1 MC sample, adagrad_window); writes TableS1_py_vi_full_defaults.csv")
    ap.add_argument("--timing_analysis", action="store_true",
                    help="time every ADVI setting over seeds 11-20 (see the module docstring); writes timing_pymc.csv and "
                         "timing_pymc_summary.csv only")
    ap.add_argument("--timing_seeds", default="11-20", help='seeds for --timing_analysis, e.g. "11-20" (default) or "11,13,15"')
    import journe_restarts
    journe_restarts.add_restart_args(ap, default_seeds=10)
    args = ap.parse_args(argv)
    if args.timing_analysis:
        return run_timing_analysis(args)
    if args.restarts:
        return run_restarts(args, args.iters or (DEFAULT_ITERS if args.full_defaults else RESTART_ITERS))
    args.iters = args.iters or DEFAULT_ITERS
    chosen = [m.strip() for m in args.only.split(",") if m.strip()]
    bad = [] if args.full_defaults else [m for m in chosen if m not in METHODS]
    if bad:
        sys.exit(f"Unknown method(s): {bad}. Choose from {list(METHODS)}")
    if not Path(DATA_FILE).exists():
        sys.exit(f"{DATA_FILE} not found in {Path.cwd()} - run this from the Perennial_reproduction folder")

    suffix = "" if args.iters == DEFAULT_ITERS else f"_{args.iters // 1000}k"
    import pymc as pm
    print("PyMC", pm.__version__, "| ADVI iterations:", args.iters)
    d = load_data()
    model = build_model(d)
    nuts = pd.read_csv(NUTS_FILE) if Path(NUTS_FILE).exists() else None
    if nuts is None:
        print(f"Note: {NUTS_FILE} not found - skipping the *_vs_NUTS comparison tables")

    if args.full_defaults:
        label = "PyMC full defaults"
        print(f"\n=== {label}: pm.fit(random_seed=1), nothing else ===")
        post, loss = run_full_defaults(model)
        tab = summary_table(term_draws(post), seed=1, method=label, final_loss=loss)
        tab.to_csv("TableS1_py_vi_full_defaults.csv", index=False)
        print(tab.drop(columns=["method"]).round(4).to_string(index=False))
        if nuts is not None:
            compare_to_nuts(tab, nuts, label).to_csv("TableS1_py_vi_full_defaults_vs_NUTS.csv", index=False)
        return

    for tag in chosen:
        label = METHODS[tag]
        print(f"\n=== {label} ===")
        if tag == "mcmc":
            post, loss = run_mcmc(model, seed=args.mcmc_seed)
            seed = args.mcmc_seed
        else:
            post, seed, loss = run_vi(model, tag, n_iter=args.iters)
        tab = summary_table(term_draws(post), seed=seed, method=label, final_loss=loss)
        tab.to_csv(f"TableS1_py_{tag}{suffix if tag != 'mcmc' else ''}.csv", index=False)
        print(tab.drop(columns=["method"]).round(4).to_string(index=False))
        if nuts is not None:
            compare_to_nuts(tab, nuts, label).to_csv(f"TableS1_py_{tag}{suffix if tag != 'mcmc' else ''}_vs_NUTS.csv", index=False)
    plot_trajectories(nuts, n_iter=args.iters, suffix=suffix)
    print("\nDone. Wrote TableS1_py_<method>.csv (and *_vs_NUTS.csv) for:", ", ".join(chosen))


if __name__ == "__main__":
    main()
