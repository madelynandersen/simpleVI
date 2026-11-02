"""
Three-level meta-analysis from Bamberg & Moreau (2025, Psychological Bulletin,
doi:10.1037/bul0000492): effect sizes nested in studies, fit with brms as

    es | se(es_var) ~ 1 + <moderators> + (1 | study_id/es_id)

The PyMC model below mirrors brms's generated Stan code (non-centered group
effects, centered population-level predictors, half-Cauchy(0, 0.5) on both
group-level SDs) and is built from the standata exported by
single_MC/fasting_meta/fit_brms.R, so both use identical data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "single_MC" / "fasting_meta" / "results"

SD_NAMES = {"sd_1": "sd_study_id__Intercept", "sd_2": "sd_study_id:es_id__Intercept"}


def load_fasting_standata(name, results_dir=RESULTS_DIR):
    with open(Path(results_dir) / "brms" / f"standata_{name}.json") as f:
        sd = json.load(f)
    X = np.atleast_2d(np.asarray(sd["X"], dtype=float))
    if X.shape[0] != len(sd["Y"]):
        X = X.T
    b_names = sd["b_names"] if isinstance(sd["b_names"], list) else [sd["b_names"]]
    return {
        "Y": np.asarray(sd["Y"], dtype=float),
        "se": np.asarray(sd["se"], dtype=float),
        "Xc": X[:, 1:] - X[:, 1:].mean(axis=0),
        "means_X": X[:, 1:].mean(axis=0),
        "b_names": [f"b_{n}" for n in b_names[1:]],
        "b_prior": [float(v) for v in sd["b_prior"]],
        "J_1": np.asarray(sd["J_1"], dtype=int) - 1,
        "J_2": np.asarray(sd["J_2"], dtype=int) - 1,
        "N_1": int(sd["N_1"]),
        "N_2": int(sd["N_2"]),
    }


def load_fasting_brms_draws(name, results_dir=RESULTS_DIR, subdir="brms"):
    return pd.read_csv(Path(results_dir) / subdir / f"draws_{name}.csv")


def build_pymc_fasting_model(data):
    import pymc as pm
    import pytensor.tensor as pt

    Kc = data["Xc"].shape[1]
    with pm.Model() as model:
        intercept = pm.Normal("Intercept", 0.0, 1.0)
        sd_1 = pm.HalfCauchy("sd_1", 0.5)
        sd_2 = pm.HalfCauchy("sd_2", 0.5)
        z_1 = pm.Normal("z_1", 0.0, 1.0, shape=data["N_1"])
        z_2 = pm.Normal("z_2", 0.0, 1.0, shape=data["N_2"])
        mu = intercept + sd_1 * z_1[data["J_1"]] + sd_2 * z_2[data["J_2"]]
        if Kc > 0:
            b_mu, b_sd = data["b_prior"]
            b = pm.Normal("b", b_mu, b_sd, shape=Kc)
            mu = mu + pt.dot(data["Xc"], b)
            pm.Deterministic("b_Intercept", intercept - pt.dot(data["means_X"], b))
        else:
            pm.Deterministic("b_Intercept", intercept)
        pm.Normal("Y", mu, data["se"], observed=data["Y"])
    return model


# reference level of each factor moderator (brms treatment coding)
FACTOR_REFERENCE_LEVELS = {
    "cognitive_domain": "attention",
    "has_food_stims": "no_food",
    "study_type": "cross_over",
    "risk": "low",
}


def add_fasting_level_draws(draws):
    """Add per-level subgroup means (Intercept + dummy), as reported in the paper's tables."""
    draws = dict(draws)
    for factor, ref in FACTOR_REFERENCE_LEVELS.items():
        dummies = [k for k in draws if k.startswith(f"b_{factor}")]
        if not dummies:
            continue
        draws[f"level_{factor}={ref}"] = np.asarray(draws["b_Intercept"])
        for k in dummies:
            draws[f"level_{factor}={k[len(factor) + 2:]}"] = np.asarray(draws["b_Intercept"]) + np.asarray(draws[k])
    return draws


def hdi(x, prob=0.95):
    x = np.sort(np.asarray(x))
    n_in = int(np.floor(prob * len(x)))
    widths = x[n_in:] - x[:len(x) - n_in]
    i = int(np.argmin(widths))
    return x[i], x[i + n_in]


def summarize_fasting_draws(draws):
    """Summaries in brms's naming for a dict/DataFrame of named draw arrays."""
    rows = []
    for name, x in add_fasting_level_draws(draws).items():
        x = np.asarray(x)
        lo, hi = hdi(x)
        rows.append({
            "variable": name,
            "mean": x.mean(),
            "median": np.median(x),
            "sd": x.std(ddof=1),
            "q2.5": np.quantile(x, 0.025),
            "q97.5": np.quantile(x, 0.975),
            "hdi_lo": lo,
            "hdi_hi": hi,
            "p_gt_0": (x > 0).mean(),
        })
    return pd.DataFrame(rows)


def pymc_approx_named_draws(approx, data, n_draws=8000, seed=0):
    """Draw from a PyMC approximation and return brms-named population-level and SD draws."""
    idata = approx.sample(n_draws, random_seed=seed)
    post = idata.posterior
    out = {"b_Intercept": post["b_Intercept"].values.ravel()}
    if data["b_names"]:
        b = post["b"].values.reshape(-1, len(data["b_names"]))
        for j, name in enumerate(data["b_names"]):
            out[name] = b[:, j]
    for k, v in SD_NAMES.items():
        out[v] = post[k].values.ravel()
    return out


def meanfield_elbo(model, approx, n_draws=2000, seed=0):
    """Monte Carlo ELBO of a mean-field approximation, using a fixed number of draws."""
    group = approx.groups[0]
    mean = group.params_dict["mu"].eval()
    std = np.log1p(np.exp(group.params_dict["rho"].eval()))
    logp = model.compile_logp(jacobian=True)
    rng = np.random.default_rng(seed)
    z = mean + std * rng.standard_normal((n_draws, mean.size))
    lps = np.array([
        logp({name: zi[slc].reshape(shape) for name, (_, slc, shape, _) in group.ordering.items()})
        for zi in z
    ])
    entropy = np.sum(np.log(std)) + 0.5 * mean.size * (1.0 + np.log(2.0 * np.pi))
    return float(lps.mean() + entropy), float(lps.std(ddof=1) / np.sqrt(n_draws))
