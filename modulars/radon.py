import os
from pathlib import Path

import numpy as np
import pandas as pd


PYMC_REPORTED_SUMMARIES = {
    # From the PyMC multilevel-modeling radon tutorial's varying-intercept model.
    # The page reports the exact beta summary in text; the other parameters are
    # visualized rather than printed as a compact table in the page text.
    "beta": {"mean": -0.664, "sd": 0.069},
}

PYMC_TUTORIAL_REFERENCE_MEANS = {
    "beta": [(-0.664, "PyMC tutorial beta")],
}

PYMC_TUTORIAL_REFERENCE_STDS = {
    "beta": [(0.069, "PyMC tutorial beta")],
}

PRICE_1996_REPORTED_SUMMARIES = {
    # Price, Nero, and Gelman (1996), page 926 of the Health Physics article.
    "mu": {"mean": 4.95},
    "kappa_sq": {"mean": 0.570},
    "sigma_sq": {"mean": 0.097},
}

PRICE_1996_REFERENCE_MEANS_UNCONSTRAINED = {
    "mu": [(4.95, "Price et al. 1996 mu")],
    "kappa_sq_log__": [(np.log(0.570), "Price et al. 1996 log(kappa^2)")],
}

PRICE_1996_MODEL_NOTES = {
    "model": (
        "log(adjusted radon Bq/m^3)_ij = theta_i + error_ij; "
        "theta_i has mean mu and variance sigma^2; "
        "error_ij has mean 0 and variance kappa^2."
    ),
    "comparison_parameters": "eta = (mu, kappa^2, sigma^2)",
    "reported_values": (
        "Price, Nero, and Gelman report mu=4.95, kappa^2=0.570, "
        "and sigma^2=0.097 for the county geometric-mean model."
    ),
    "adjustment": (
        "Measured pCi/L radon is adjusted as "
        "r / 2 + sqrt(r^2 / 4 + 0.25^2) before converting to Bq/m^3."
    ),
}

RADON_REFERENCE_NOTES = {
    "pymc_tutorial": (
        "The notebooks implement the PyMC tutorial's centered varying-intercept "
        "model: county intercepts, a shared floor effect, and Exponential(1) "
        "priors on the positive scale parameters."
    ),
    "best_advi_reference": (
        "Parameters not reported directly by the PyMC tutorial are compared "
        "against the shared saved-output VI reference in best_reference_values.csv. "
        "The reference script selects the saved 100-MC variational restart with "
        "the largest NumPyro-computed fixed-guide ELBO."
    ),
    "price_nero_gelman_1996": (
        "The peer-reviewed Price/Nero/Gelman Health Physics article uses a "
        "different county-geometric-mean model on adjusted Bq/m^3 radon values; "
        "those comparisons live in single_MC/radon_example/price_nero_gelman_1996."
    ),
}


def make_radon_param_names(county_names):
    """Return the ordered latent-vector labels used by the PyMC-style notebooks."""
    return (
        ["mu_a", "sigma_a"]
        + [f"alpha[{county}]" for county in county_names]
        + ["beta", "sd_y"]
    )


def make_radon_unconstrained_param_names(county_names):
    """Return labels for the unconstrained latent vector used by TFP."""
    return (
        ["mu_a", "sigma_a_log__"]
        + [f"alpha[{county}]" for county in county_names]
        + ["beta", "sd_y_log__"]
    )


def make_price_radon_param_names(county_names, unconstrained=False):
    """Return ordered latent labels for the Price/Nero/Gelman county-GM model."""
    if unconstrained:
        scale_names = ["kappa_sq_log__", "sigma_sq_log__"]
    else:
        scale_names = ["kappa_sq", "sigma_sq"]
    return (
        ["mu"]
        + scale_names
        + [f"theta[{county}]" for county in county_names]
    )


def _first_existing_path(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def _read_radon_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.map(str.strip)
    if "log_radon" not in df.columns:
        if "activity" in df.columns:
            df["log_radon"] = np.log(df["activity"] + 0.1)
        elif "log_radon_activity" in df.columns:
            df["log_radon"] = df["log_radon_activity"]

    if "county" not in df.columns and "county_name" in df.columns:
        df["county"] = df["county_name"]

    required = {"log_radon", "floor", "county"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Radon data are missing required columns: {sorted(missing)}")
    return df


def load_radon_data(data_dir=None):
    """
    Load and preprocess the Minnesota radon data used by the radon notebooks.

    The returned arrays are shared by the PyMC tutorial and Price/Nero/Gelman
    notebooks. The uranium/contextual columns are kept because some radon
    examples use them.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    search_dirs = [
        Path(data_dir) if data_dir is not None else None,
        repo_root / "single_MC" / "radon_example",
        repo_root / "single_MC" / "radon_example" / "data",
        repo_root / "data",
        repo_root,
    ]
    search_dirs = [p for p in search_dirs if p is not None]

    radon_path = _first_existing_path(
        [d / name for d in search_dirs for name in ("srrs2.dat", "radon.csv", "mn_radon.csv")]
    )
    cty_path = _first_existing_path(
        [d / name for d in search_dirs for name in ("cty.dat", "county_uranium.csv")]
    )

    if radon_path is None:
        try:
            import pymc as pm

            radon_path = pm.get_data("srrs2.dat")
            cty_path = pm.get_data("cty.dat")
        except Exception as err:
            raise FileNotFoundError(
                "Could not find radon data locally and could not fetch PyMC's "
                "srrs2.dat/cty.dat. Put those files or a radon.csv into "
                "single_MC/radon_example/."
            ) from err

    radon = _read_radon_csv(radon_path)
    radon = radon.copy()
    radon["county"] = radon["county"].astype(str).str.strip().str.upper()

    if "state" in radon.columns:
        radon = radon[radon["state"].astype(str).str.upper().eq("MN")].copy()
    if "state2" in radon.columns:
        radon = radon[radon["state2"].astype(str).str.upper().eq("MN")].copy()

    if cty_path is not None:
        cty = pd.read_csv(cty_path)
        cty.columns = cty.columns.map(str.strip)
        if "st" in cty.columns:
            cty = cty[cty["st"].astype(str).str.upper().eq("MN")].copy()
        if "Uppm" in cty.columns:
            cty["log_uranium"] = np.log(cty["Uppm"])

        if {"stfips", "cntyfips"}.issubset(radon.columns) and {"stfips", "ctfips"}.issubset(cty.columns):
            radon["_fips"] = 1000 * radon["stfips"] + radon["cntyfips"]
            cty["_fips"] = 1000 * cty["stfips"] + cty["ctfips"]
            radon = radon.merge(cty[["_fips", "log_uranium"]], on="_fips", how="left")
            radon = radon.drop(columns=["_fips"])
            if "idnum" in radon.columns:
                radon = radon.drop_duplicates(subset="idnum")
        else:
            if "cty" in cty.columns and "county" not in cty.columns:
                cty = cty.rename(columns={"cty": "county"})
            cty["county"] = cty["county"].astype(str).str.strip().str.upper()
            cty_cols = ["county", "log_uranium"]
            radon = radon.merge(cty[cty_cols], on="county", how="left")

    if "log_uranium" not in radon.columns:
        if "Uppm" in radon.columns:
            radon["log_uranium"] = np.log(radon["Uppm"])
        else:
            radon["log_uranium"] = 0.0
    radon["log_uranium"] = radon["log_uranium"].fillna(0.0)

    radon = radon.dropna(subset=["log_radon", "floor", "county"]).copy()

    if "activity" in radon.columns:
        activity_pci = radon["activity"].astype(float).to_numpy()
        adjusted_pci = 0.5 * activity_pci + np.sqrt(0.25 * activity_pci**2 + 0.25**2)
        radon["radon_bq_adjusted"] = adjusted_pci * 37.0
        radon["log_radon_bq_adjusted"] = np.log(radon["radon_bq_adjusted"])
    else:
        radon["radon_bq_adjusted"] = np.exp(radon["log_radon"]) * 37.0
        radon["log_radon_bq_adjusted"] = np.log(radon["radon_bq_adjusted"])

    county_names, county = np.unique(radon["county"].to_numpy(), return_inverse=True)
    floor = radon["floor"].astype(float).to_numpy()
    floor_by_county = (
        pd.DataFrame({"county": county, "floor": floor})
        .groupby("county")["floor"]
        .mean()
        .reindex(np.arange(len(county_names)))
        .to_numpy()
    )

    return {
        "frame": radon,
        "log_radon": radon["log_radon"].astype(float).to_numpy(),
        "log_radon_bq_adjusted": radon["log_radon_bq_adjusted"].astype(float).to_numpy(),
        "radon_bq_adjusted": radon["radon_bq_adjusted"].astype(float).to_numpy(),
        "floor": floor,
        "log_uranium": radon["log_uranium"].astype(float).to_numpy(),
        "county": county.astype(np.int32),
        "county_names": county_names,
        "floor_by_county": floor_by_county.astype(float),
        "param_names": make_radon_param_names(county_names),
    }


def default_radon_plot_dims(param_names, county_names):
    """Pick a compact, interpretable set of latent dimensions for first-pass plots."""
    beta_dim = 2 + len(county_names)
    dims = [0, beta_dim]
    for target in ("HENNEPIN", "ST LOUIS", "ST. LOUIS"):
        matches = np.flatnonzero(np.asarray(county_names) == target)
        if len(matches):
            dims.append(2 + int(matches[0]))
    dims = list(dict.fromkeys(dims))
    labels = [param_names[d] for d in dims]
    return dims, labels


def build_numpyro_radon_model():
    """Build the PyMC tutorial varying-intercept radon model for NumPyro SVI."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(log_radon, floor, log_uranium, county, floor_by_county):
        num_counties = floor_by_county.shape[0]
        del log_uranium, floor_by_county

        mu_a = numpyro.sample("mu_a", dist.Normal(0.0, 10.0))
        sigma_a = numpyro.sample("sigma_a", dist.Exponential(1.0))
        alpha = numpyro.sample(
            "alpha", dist.Normal(jnp.full(num_counties, mu_a), sigma_a).to_event(1)
        )
        beta = numpyro.sample("beta", dist.Normal(0.0, 10.0))
        sd_y = numpyro.sample("sd_y", dist.Exponential(1.0))

        mean = alpha[county] + beta * floor
        numpyro.sample("log_radon", dist.Normal(mean, sd_y), obs=log_radon)

    return model


def make_tfp_radon_conditioned_log_prob(data, dtype=None):
    """Build the PyMC tutorial varying-intercept radon log-prob function for TFP."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    if dtype is None:
        dtype = tf.float32

    log_radon = tf.convert_to_tensor(data["log_radon"], dtype=dtype)
    floor = tf.convert_to_tensor(data["floor"], dtype=dtype)
    county = tf.convert_to_tensor(data["county"], dtype=tf.int32)
    num_counties = int(len(data["county_names"]))

    @tf.function(jit_compile=False)
    def log_prob_fn(theta):
        theta = tf.convert_to_tensor(theta, dtype=dtype)
        mu_a = theta[..., 0]
        sigma_a_log = theta[..., 1]
        alpha = theta[..., 2 : 2 + num_counties]
        beta = theta[..., 2 + num_counties]
        sd_y_log = theta[..., 3 + num_counties]

        sigma_a = tf.exp(sigma_a_log)
        sd_y = tf.exp(sd_y_log)

        alpha_obs = tf.gather(alpha, county, axis=-1)
        mean = alpha_obs + beta[..., tf.newaxis] * floor

        lp = tfd.Normal(0.0, 10.0).log_prob(mu_a)
        lp += tfd.Exponential(1.0).log_prob(sigma_a) + sigma_a_log
        lp += tf.reduce_sum(tfd.Normal(mu_a[..., tf.newaxis], sigma_a[..., tf.newaxis]).log_prob(alpha), axis=-1)
        lp += tfd.Normal(0.0, 10.0).log_prob(beta)
        lp += tfd.Exponential(1.0).log_prob(sd_y) + sd_y_log
        lp += tf.reduce_sum(tfd.Normal(mean, sd_y[..., tf.newaxis]).log_prob(log_radon), axis=-1)
        return lp

    return log_prob_fn


def build_pymc_radon_model(data):
    """
    Build the PyMC tutorial's varying-intercept radon model.
    """
    import pymc as pm

    num_counties = len(data["county_names"])
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=0.0, sigma=10.0)
        sigma_a = pm.Exponential("sigma_a", 1.0)
        alpha = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, shape=num_counties)
        beta = pm.Normal("beta", mu=0.0, sigma=10.0)
        sd_y = pm.Exponential("sd_y", 1.0)

        mean = alpha[data["county"]] + beta * data["floor"]
        pm.Normal("log_radon", mean, sd_y, observed=data["log_radon"])
    return model


def build_numpyro_price_radon_model():
    """Build the Price/Nero/Gelman 1996 county-GM radon model for NumPyro SVI."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(log_radon, floor, log_uranium, county, floor_by_county):
        num_counties = floor_by_county.shape[0]
        del floor, log_uranium, floor_by_county

        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        kappa_sq = numpyro.sample("kappa_sq", dist.Exponential(1.0))
        sigma_sq = numpyro.sample("sigma_sq", dist.Exponential(1.0))
        theta = numpyro.sample(
            "theta", dist.Normal(jnp.full(num_counties, mu), jnp.sqrt(sigma_sq)).to_event(1)
        )

        numpyro.sample("log_radon", dist.Normal(theta[county], jnp.sqrt(kappa_sq)), obs=log_radon)

    return model


def make_tfp_price_radon_conditioned_log_prob(data, dtype=None):
    """Build the Price/Nero/Gelman county-GM log-prob function for TFP."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    if dtype is None:
        dtype = tf.float32

    log_radon = tf.convert_to_tensor(data["log_radon_bq_adjusted"], dtype=dtype)
    county = tf.convert_to_tensor(data["county"], dtype=tf.int32)
    num_counties = int(len(data["county_names"]))

    @tf.function(jit_compile=False)
    def log_prob_fn(theta_vec):
        theta_vec = tf.convert_to_tensor(theta_vec, dtype=dtype)
        mu = theta_vec[..., 0]
        kappa_sq_log = theta_vec[..., 1]
        sigma_sq_log = theta_vec[..., 2]
        theta = theta_vec[..., 3 : 3 + num_counties]

        kappa_sq = tf.exp(kappa_sq_log)
        sigma_sq = tf.exp(sigma_sq_log)
        kappa = tf.sqrt(kappa_sq)
        sigma = tf.sqrt(sigma_sq)

        theta_obs = tf.gather(theta, county, axis=-1)

        lp = tfd.Normal(0.0, 10.0).log_prob(mu)
        lp += tfd.Exponential(1.0).log_prob(kappa_sq) + kappa_sq_log
        lp += tfd.Exponential(1.0).log_prob(sigma_sq) + sigma_sq_log
        lp += tf.reduce_sum(tfd.Normal(mu[..., tf.newaxis], sigma[..., tf.newaxis]).log_prob(theta), axis=-1)
        lp += tf.reduce_sum(tfd.Normal(theta_obs, kappa[..., tf.newaxis]).log_prob(log_radon), axis=-1)
        return lp

    return log_prob_fn


def build_pymc_price_radon_model(data):
    """Build the Price/Nero/Gelman 1996 county-GM radon model for PyMC ADVI."""
    import pymc as pm
    import pytensor.tensor as pt

    num_counties = len(data["county_names"])
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        kappa_sq = pm.Exponential("kappa_sq", 1.0)
        sigma_sq = pm.Exponential("sigma_sq", 1.0)
        theta = pm.Normal("theta", mu=mu, sigma=pt.sqrt(sigma_sq), shape=num_counties)

        pm.Normal(
            "log_radon",
            theta[data["county"]],
            pt.sqrt(kappa_sq),
            observed=data["log_radon_bq_adjusted"],
        )
    return model


def price_theta_summary_from_unconstrained(loc, scale):
    """
    Compute eta=(mu,kappa^2,sigma^2) moments from diagonal q.

    loc and scale are expected in the Price model unconstrained order:
    mu, log(kappa^2), log(sigma^2), theta...
    """
    loc = np.asarray(loc, dtype=float)
    scale = np.asarray(scale, dtype=float)
    mean = np.array(
        [
            loc[0],
            np.exp(loc[1] + 0.5 * scale[1] ** 2),
            np.exp(loc[2] + 0.5 * scale[2] ** 2),
        ]
    )
    sd = np.array(
        [
            scale[0],
            np.sqrt((np.exp(scale[1] ** 2) - 1.0) * np.exp(2.0 * loc[1] + scale[1] ** 2)),
            np.sqrt((np.exp(scale[2] ** 2) - 1.0) * np.exp(2.0 * loc[2] + scale[2] ** 2)),
        ]
    )
    return mean, sd


def save_reference_summary(path):
    """Write the currently digitized external reference values to CSV."""
    rows = [
        {
            "source": "PyMC tutorial",
            "parameter": name,
            "summary": summary,
            "value": value,
            "notes": "Exact reported summary from the PyMC varying-intercept radon model.",
        }
        for name, summaries in PYMC_REPORTED_SUMMARIES.items()
        for summary, value in summaries.items()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def print_reference_summary():
    print('PyMC tutorial exact reported summaries:')
    print(PYMC_REPORTED_SUMMARIES)
    print('\nReference notes:')
    for key, note in RADON_REFERENCE_NOTES.items():
        print(f"- {key}: {note}")


def load_best_references(path='best_reference_values.csv'):
    reference_summary = pd.read_csv('radon_reference_summary.csv')
    best_reference_path = Path(path)
    if not best_reference_path.exists():
        raise FileNotFoundError(f"Best reference values not found at {best_reference_path}, please run compute_best_reference.py to create that file or update the path to the proper location.")
    best_reference = pd.read_csv(best_reference_path)
    best_reference_means = {
        row['parameter']: [(float(row['mean']), row['source'])]        for _, row in best_reference.iterrows()
    }
    best_reference_stds = {
        row['parameter']: [(float(row['sd']), row['source'])]        for _, row in best_reference.iterrows()
    }
    return reference_summary, best_reference, best_reference_means, best_reference_stds


def plot_radon_selected_dims(
    single_means,
    single_stds,
    multi_means,
    multi_stds,
    dims,
    labels,
    title_prefix="",
    reference_means=None,
    reference_stds=None,
    iteration_stride=1,
    reference_window=False,
    mean_window_sd=3.0,
    std_window=(0.6, 1.6),
    legend_below=False,
    legend_ncol=2,
    legend_y=-0.28,
    subplot_hspace=None,
):
    """Plot radon VI mean/std trajectories for a small set of labeled dimensions."""
    import matplotlib.pyplot as plt

    def _as_ref_list(refs, default_label):
        if isinstance(refs, (int, float, np.floating)):
            return [(float(refs), default_label)]
        return refs

    n_rows = len(dims)
    row_height = 4.8 if legend_below else 4.0
    fig, axs = plt.subplots(n_rows, 2, figsize=(16, row_height * n_rows), squeeze=False)
    iteration_stride = max(1, int(iteration_stride))
    if iteration_stride == 1:
        x = np.arange(single_means.shape[1])
    else:
        x = (np.arange(single_means.shape[1]) + 1) * iteration_stride

    for row, (dim, label) in enumerate(zip(dims, labels)):
        mean_refs = []
        std_refs = []
        for values, color, run_label in (
            (single_means[:, :, dim], "blue", "1 MC sample"),
            (multi_means[:, :, dim], "green", "100 MC samples"),
        ):
            mean = values.mean(axis=0)
            sd = values.std(axis=0)
            axs[row, 0].plot(x, mean, color=color, label=run_label)
            axs[row, 0].fill_between(x, mean - sd, mean + sd, color=color, alpha=0.2)

        for values, color, run_label in (
            (single_stds[:, :, dim], "blue", "1 MC sample"),
            (multi_stds[:, :, dim], "green", "100 MC samples"),
        ):
            mean = values.mean(axis=0)
            sd = values.std(axis=0)
            axs[row, 1].plot(x, mean, color=color, label=run_label)
            axs[row, 1].fill_between(x, mean - sd, mean + sd, color=color, alpha=0.2)

        if reference_means and label in reference_means:
            refs = _as_ref_list(reference_means[label], "external reported mean")
            for ref_idx, (value, ref_label) in enumerate(refs):
                mean_refs.append(float(value))
                if "Saved VI" in ref_label:
                    ref_label = "Best VI reference"
                axs[row, 0].axhline(
                    value,
                    color="red",
                    linestyle="--" if ref_idx == 0 else ":",
                    label=ref_label,
                )

        if reference_stds and label in reference_stds:
            refs = _as_ref_list(reference_stds[label], "external reported std")
            for ref_idx, (value, ref_label) in enumerate(refs):
                std_refs.append(float(value))
                if "Saved VI" in ref_label:
                    ref_label = "Best VI reference"
                axs[row, 1].axhline(
                    value,
                    color="red",
                    linestyle="--" if ref_idx == 0 else ":",
                    label=ref_label,
                )

        if reference_window and mean_refs and std_refs:
            ref_sd = max(max(std_refs), 1e-12)
            axs[row, 0].set_ylim(
                min(mean_refs) - mean_window_sd * ref_sd,
                max(mean_refs) + mean_window_sd * ref_sd,
            )
        if reference_window and std_refs:
            std_low, std_high = std_window
            axs[row, 1].set_ylim(
                max(min(std_refs) * std_low, 0.0),
                max(std_refs) * std_high,
            )

        axs[row, 0].set_title(f"{title_prefix}{label}: variational mean")
        axs[row, 1].set_title(f"{title_prefix}{label}: variational std")
        axs[row, 0].set_xlabel("Iteration")
        axs[row, 1].set_xlabel("Iteration")
        axs[row, 0].grid()
        axs[row, 1].grid()
        if legend_below:
            legend_kwargs = {
                "loc": "upper center",
                "bbox_to_anchor": (0.5, legend_y),
                "ncol": legend_ncol,
                "frameon": True,
            }
            axs[row, 0].legend(**legend_kwargs)
            axs[row, 1].legend(**legend_kwargs)
        else:
            axs[row, 0].legend()
            axs[row, 1].legend()

    plt.tight_layout()
    if legend_below:
        if subplot_hspace is None:
            subplot_hspace = 0.9 if n_rows > 1 else 0.45
        fig.subplots_adjust(
            hspace=subplot_hspace,
            bottom=0.18 if n_rows == 1 else 0.08,
        )
    plt.show()


def plot_radon_selected_dims_zoomed(
    single_means,
    single_stds,
    multi_means,
    multi_stds,
    dims,
    labels,
    title_prefix="",
    reference_means=None,
    reference_stds=None,
    iteration_stride=1,
    mean_window_sd=3.0,
    std_window=(0.6, 1.6),
    legend_ncol=2,
):
    """
    Plot selected radon dimensions with the same reference-window zoom used by
    plot_some_dims_multid: means are shown around reference mean +/- 3 stds,
    and posterior stds are shown around the reference std scale.
    """
    return plot_radon_selected_dims(
        single_means,
        single_stds,
        multi_means,
        multi_stds,
        dims,
        labels,
        title_prefix=title_prefix,
        reference_means=reference_means,
        reference_stds=reference_stds,
        iteration_stride=iteration_stride,
        reference_window=True,
        mean_window_sd=mean_window_sd,
        std_window=std_window,
        legend_below=True,
        legend_ncol=legend_ncol,
    )
