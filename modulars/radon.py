from pathlib import Path

import numpy as np
import pandas as pd


PYMC_REPORTED_SUMMARIES = {
    # beta summary reported in the PyMC multilevel-modeling radon tutorial
    "beta": {"mean": -0.664, "sd": 0.069},
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


def make_tfp_radon_conditioned_log_prob(data, dtype=None, transform_sigmas=True):
    """Build the PyMC tutorial varying-intercept radon log-prob function for TFP.

    transform_sigmas=True (the radon notebooks) takes sigma_a and sd_y on the log scale (with the log-Jacobian), so the
    surrogate is unconstrained; False takes them as they are, TFP's untransformed default (a draw of a
    sigma <= 0 has no density).
    """
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
        alpha = theta[..., 2 : 2 + num_counties]
        beta = theta[..., 2 + num_counties]
        if transform_sigmas:
            sigma_a_log = theta[..., 1]
            sd_y_log = theta[..., 3 + num_counties]
            sigma_a = tf.exp(sigma_a_log)
            sd_y = tf.exp(sd_y_log)
            log_jacobian = sigma_a_log + sd_y_log
        else:
            sigma_a = theta[..., 1]
            sd_y = theta[..., 3 + num_counties]
            log_jacobian = 0.0

        alpha_obs = tf.gather(alpha, county, axis=-1)
        mean = alpha_obs + beta[..., tf.newaxis] * floor

        lp = tfd.Normal(0.0, 10.0).log_prob(mu_a)
        lp += tfd.Exponential(1.0).log_prob(sigma_a)
        lp += tf.reduce_sum(tfd.Normal(mu_a[..., tf.newaxis], sigma_a[..., tf.newaxis]).log_prob(alpha), axis=-1)
        lp += tfd.Normal(0.0, 10.0).log_prob(beta)
        lp += tfd.Exponential(1.0).log_prob(sd_y) + log_jacobian
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
    x=None,
    backend="matplotlib",
    single_label="1 MC sample",
    multi_label="100 MC samples",
):
    """Plot radon VI mean/std trajectories for a small set of labeled dimensions.

    single_label/multi_label: legend text for the two series -- override
    these to name the actual scenarios being compared (e.g. "default" vs
    "grad100_adapt_off") rather than the generic MC-sample-count wording,
    since radon compares several distinct grad100_* scenarios, not just a
    single MC-sample-count ablation.

    backend: "matplotlib" (default, static, savable via plt.savefig) or
    "plotly" (interactive, shown inline via fig.show(); the figure is also
    returned).
    """
    from modulars.plot_rr import plot_mean_band_grid_1d

    return plot_mean_band_grid_1d(
        single_means, single_stds, multi_means, multi_stds,
        dims=dims, labels=labels, title_prefix=title_prefix,
        reference_means=reference_means, reference_stds=reference_stds,
        iteration_stride=iteration_stride, reference_window=reference_window,
        mean_window_sd=mean_window_sd, std_window=std_window,
        legend_below=legend_below, legend_ncol=legend_ncol, legend_y=legend_y,
        subplot_hspace=subplot_hspace, x=x, backend=backend,
        single_label=single_label, multi_label=multi_label,
    )


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
    x=None,
    backend="matplotlib",
    single_label="1 MC sample",
    multi_label="100 MC samples",
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
        x=x,
        reference_window=True,
        mean_window_sd=mean_window_sd,
        std_window=std_window,
        legend_below=True,
        legend_ncol=legend_ncol,
        backend=backend,
        single_label=single_label,
        multi_label=multi_label,
    )


def plot_radon_mean_band_grid_single(
        means, stds, x, title,
        which_dims, which_labels,
        reference_means, reference_stds):
    """Single-scenario summary: mean +/- 1 SD (restart-to-restart spread)
    shaded band for each representative dimension, one red dashed reference
    line per panel; the single-scenario counterpart to
    plot_radon_selected_dims. Plotly only."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from modulars.plot_rr import _grid_vertical_spacing

    n_rows = len(which_dims)
    subplot_titles = []
    for lab in which_labels:
        subplot_titles.append(f"{lab}: posterior mean")
        subplot_titles.append(f"{lab}: posterior std")

    fig = make_subplots(rows=n_rows, cols=2, vertical_spacing=_grid_vertical_spacing(n_rows),
                         horizontal_spacing=0.1, subplot_titles=subplot_titles)

    for row_idx, dim in enumerate(which_dims):
        r = row_idx + 1
        lab = which_labels[row_idx]
        show_legend = row_idx == 0

        m = np.nanmean(means[:, :, dim], axis=0)
        m_spread = np.nanstd(means[:, :, dim], axis=0)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]), y=np.concatenate([m + m_spread, (m - m_spread)[::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.25)", line=dict(width=0), hoverinfo="skip",
            name="±1 SD across restarts", legendgroup="band", showlegend=show_legend,
        ), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=m, mode="lines", line=dict(color="rgb(31,119,180)", width=2),
            name="Mean across restarts", legendgroup="mean", showlegend=show_legend,
        ), row=r, col=1)
        if lab in reference_means:
            ref_mean = reference_means[lab][0][0]
            fig.add_hline(y=ref_mean, line=dict(color="red", dash="dash"), row=r, col=1)
            if lab in reference_stds:
                ref_std_for_window = reference_stds[lab][0][0]
                fig.update_yaxes(range=[ref_mean - 3.0 * ref_std_for_window, ref_mean + 3.0 * ref_std_for_window], row=r, col=1)

        s = np.nanmean(stds[:, :, dim], axis=0)
        s_spread = np.nanstd(stds[:, :, dim], axis=0)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]), y=np.concatenate([s + s_spread, (s - s_spread)[::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.25)", line=dict(width=0), hoverinfo="skip",
            name="±1 SD across restarts", legendgroup="band", showlegend=False,
        ), row=r, col=2)
        fig.add_trace(go.Scatter(
            x=x, y=s, mode="lines", line=dict(color="rgb(31,119,180)", width=2),
            name="Mean across restarts", legendgroup="mean", showlegend=False,
        ), row=r, col=2)
        if lab in reference_stds:
            ref_std = reference_stds[lab][0][0]
            fig.add_hline(y=ref_std, line=dict(color="red", dash="dash"), row=r, col=2)
            fig.update_yaxes(range=[max(ref_std * 0.6, 0.0), ref_std * 1.6], row=r, col=2)

        fig.update_xaxes(title_text="Iteration", row=r, col=1)
        fig.update_xaxes(title_text="Iteration", row=r, col=2)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="red", dash="dash"),
                              name="best variational approximation"), row=1, col=1)
    fig.update_layout(height=300 * n_rows, width=1400, title_text=title)
    fig.show()
