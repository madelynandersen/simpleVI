from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CHEMICAL = "Carbaryl"
DEFAULT_CAS = "63252"
DEFAULT_N_COMPONENTS = 20
DEFAULT_GAMMA = 0.4
DEFAULT_SIGMA_LOW = 0.1
DEFAULT_SIGMA_HIGH = 1.5


BNP_SPECIES_NOTES = {
    "data": (
        "Carbaryl species sensitivity data from data/etc_1891_sm_suppldata1.xls. "
        "Rows are filtered with the Hickey-style acute EC50/LC50 criteria used by "
        "Alamichel et al.; left/right-censored and approximate rows are dropped, "
        "interval-censored rows are replaced by their central value, and repeated "
        "species rows are combined by geometric mean before log10 centering/scaling."
    ),
    "model": (
        "Finite first-pass approximation to the normalized stable BNP mixture: "
        "x_i ~ sum_k w_k Normal(mu_k, sigma_k), v_k ~ Beta(1-gamma, k gamma), "
        "gamma=0.4, mu_k ~ Normal(0,1), sigma_k ~ Uniform(0.1,1.5)."
    ),
}


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _repo_relative_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(_repo_root().resolve()))
    except (OSError, ValueError):
        return str(path)


def _default_source_xls():
    return _repo_root() / "data" / "etc_1891_sm_suppldata1.xls"


def _default_processed_csv():
    return _repo_root() / "single_MC" / "bnp_species" / "carbaryl_hickey_species.csv"


def _as_clean_cas(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace("-", "", regex=False)
    )


def _apply_hickey_acute_filter(df):
    """
    Mirror the Hickey-style acute filter used in the BNP-SSD paper code.

    The original R filter keeps acute EC50/LC50 mortality/immobility rows with
    duration 2 days for crustaceans/insects and 4 days otherwise, removes
    genus-only "sp" entries and mixed communities, and drops approximate rows.
    """
    endpoint = df["endpoint"].astype(str).str.upper()
    effect = df["effect"].astype(str).str.upper()
    major = df["major"].astype(str).str.upper()
    species = df["species"].astype(str)
    conc_ind = df["conc.ind"].astype(str).str.upper()

    acute_effect = endpoint.isin(["LC50", "EC50"]) & effect.isin(["MOR", "IMM"])
    crustacean_or_insect_2d = (df["dur.low"] == 2) & major.isin(["CR", "IN"])
    other_4d = (df["dur.low"] == 4) & ~major.isin(["CR", "IN"])
    not_genus_only = ~species.str.contains(r" sp$| sp\.", regex=True, na=False)
    latin_binomial_like = species.str.contains(" ", na=False)
    not_mixed = major.ne("MI")
    not_approx = conc_ind.ne("A")

    return df[
        acute_effect
        & (crustacean_or_insect_2d | other_4d)
        & not_genus_only
        & latin_binomial_like
        & not_mixed
        & not_approx
    ].copy()


def _central_concentration(df):
    conc_low = pd.to_numeric(df["conc.low"], errors="coerce")
    conc_upp = pd.to_numeric(df["conc.upp"], errors="coerce")
    conc_ind = df["conc.ind"].astype(str).str.upper()

    central = conc_low.copy()
    interval = conc_ind.eq("I")
    central.loc[interval] = 0.5 * (conc_low.loc[interval] + conc_upp.loc[interval])
    return central


def prepare_bnp_species_processed_csv(
    source_xls=None,
    output_csv=None,
    chemical=DEFAULT_CHEMICAL,
    cas=DEFAULT_CAS,
):
    """
    Build the small processed Carbaryl species-level CSV from the source .xls.

    This needs pandas' legacy .xls reader (`xlrd`) at generation time only. The
    notebooks read the generated CSV so the NumPyro/TFP/PyMC environments do not
    need to carry an Excel reader.
    """
    source_xls = Path(source_xls) if source_xls is not None else _default_source_xls()
    output_csv = Path(output_csv) if output_csv is not None else _default_processed_csv()

    df = pd.read_excel(source_xls, sheet_name="database", engine="xlrd")
    clean_cas = _as_clean_cas(df["CAS"])
    chemical_match = (
        df["chem.name2"].astype(str).str.casefold().eq(str(chemical).casefold())
        | df["chem.name"].astype(str).str.contains(str(chemical), case=False, na=False)
        | clean_cas.eq(str(cas).replace("-", ""))
    )
    carbaryl = df[chemical_match].copy()
    filtered = _apply_hickey_acute_filter(carbaryl)

    conc_ind = filtered["conc.ind"].astype(str).str.upper()
    usable = filtered[conc_ind.isin(["P", "I"])].copy()
    usable["central_concentration"] = _central_concentration(usable)
    usable = usable[np.isfinite(usable["central_concentration"]) & (usable["central_concentration"] > 0)].copy()
    usable["log10_central"] = np.log10(usable["central_concentration"].to_numpy(dtype=float))

    species = (
        usable.groupby("species", as_index=False)
        .agg(
            central_concentration=("central_concentration", lambda x: float(np.exp(np.mean(np.log(x))))),
            log10_concentration=("log10_central", "mean"),
            n_rows=("central_concentration", "size"),
            major=("major", "first"),
        )
        .sort_values("log10_concentration")
        .reset_index(drop=True)
    )

    centre = float(species["log10_concentration"].mean())
    scale = float(species["log10_concentration"].std(ddof=1))
    species["standardized_log10_concentration"] = (
        species["log10_concentration"] - centre
    ) / scale
    species["chemical"] = chemical
    species["cas"] = str(cas)
    species["source_xls"] = _repo_relative_path(source_xls)
    species["preprocessing"] = "hickey_acute_drop_left_right_midpoint_interval_species_geomean"
    species["log10_centre"] = centre
    species["log10_scale"] = scale

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    species.to_csv(output_csv, index=False)
    return species


def load_bnp_species_data(processed_csv=None, source_xls=None, rebuild=False):
    """
    Load the processed species-level Carbaryl data used by the notebooks.
    """
    processed_csv = Path(processed_csv) if processed_csv is not None else _default_processed_csv()
    if rebuild or not processed_csv.exists():
        frame = prepare_bnp_species_processed_csv(
            source_xls=source_xls,
            output_csv=processed_csv,
        )
    else:
        frame = pd.read_csv(processed_csv)

    y = frame["standardized_log10_concentration"].to_numpy(dtype=float)
    return {
        "frame": frame,
        "y": y,
        "species": frame["species"].astype(str).to_numpy(),
        "central_concentration": frame["central_concentration"].to_numpy(dtype=float),
        "log10_concentration": frame["log10_concentration"].to_numpy(dtype=float),
        "log10_centre": float(frame["log10_centre"].iloc[0]),
        "log10_scale": float(frame["log10_scale"].iloc[0]),
        "source_xls": str(frame["source_xls"].iloc[0]),
        "notes": dict(BNP_SPECIES_NOTES),
    }


def stable_stick_weights_np(v):
    v = np.asarray(v, dtype=float)
    if v.shape[-1] == 0:
        return np.ones(v.shape[:-1] + (1,), dtype=float)

    prefix = np.concatenate(
        [
            np.ones(v.shape[:-1] + (1,), dtype=float),
            np.cumprod(1.0 - v[..., :-1], axis=-1),
        ],
        axis=-1,
    )
    weights = v * prefix
    residual = np.prod(1.0 - v, axis=-1, keepdims=True)
    return np.concatenate([weights, residual], axis=-1)


def _sigmoid_np(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _logit_np(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-8, 1.0 - 1e-8)
    return np.log(x) - np.log1p(-x)


def latent_dim(n_components=DEFAULT_N_COMPONENTS):
    return 3 * int(n_components) - 1


def summary_dim(n_components=DEFAULT_N_COMPONENTS):
    return 3 * int(n_components)


def split_raw_latent_np(raw, n_components=DEFAULT_N_COMPONENTS):
    raw = np.asarray(raw, dtype=float)
    k = int(n_components)
    eta_raw = raw[..., : k - 1]
    mu = raw[..., k - 1 : 2 * k - 1]
    sigma_raw = raw[..., 2 * k - 1 : 3 * k - 1]
    return eta_raw, mu, sigma_raw


def raw_to_model_params_np(
    raw,
    n_components=DEFAULT_N_COMPONENTS,
    sigma_low=DEFAULT_SIGMA_LOW,
    sigma_high=DEFAULT_SIGMA_HIGH,
    sort_components=True,
):
    raw = np.asarray(raw, dtype=float)
    eta_raw, mu, sigma_raw = split_raw_latent_np(raw, n_components=n_components)
    v = _sigmoid_np(eta_raw)
    weights = stable_stick_weights_np(v)
    sigma = sigma_low + (sigma_high - sigma_low) * _sigmoid_np(sigma_raw)

    if sort_components:
        order = np.argsort(mu, axis=-1)
        mu = np.take_along_axis(mu, order, axis=-1)
        sigma = np.take_along_axis(sigma, order, axis=-1)
        weights = np.take_along_axis(weights, order, axis=-1)

    return np.concatenate([weights, mu, sigma], axis=-1)


def make_bnp_species_param_names(n_components=DEFAULT_N_COMPONENTS, sorted_components=True):
    suffix = "_sorted" if sorted_components else ""
    k = int(n_components)
    return (
        [f"w{suffix}[{i + 1}]" for i in range(k)]
        + [f"mu{suffix}[{i + 1}]" for i in range(k)]
        + [f"sigma{suffix}[{i + 1}]" for i in range(k)]
    )


def make_bnp_species_initial_latent(
    data=None,
    n_components=DEFAULT_N_COMPONENTS,
    sigma_low=DEFAULT_SIGMA_LOW,
    sigma_high=DEFAULT_SIGMA_HIGH,
    sigma_init=0.8,
    initial_scale=0.1,
    seed=None,
    jitter_scale=0.0,
):
    """
    Construct a stable, data-informed mean-field initialization.

    BNP mixture ADVI is sensitive to the initial component ordering.  Starting
    with two active data-anchored sticks and a small geometric tail keeps
    NumPyro, TFP, and PyMC in the same basin while still letting random seeds
    affect the stochastic VI gradients.
    """
    k = int(n_components)
    if data is None:
        y = np.linspace(-2.0, 2.0, max(k, 2), dtype=float)
    else:
        y = np.asarray(data["y"], dtype=float)

    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValueError("BNP species initialization requires at least one finite observation.")

    y_mean = float(np.mean(y))
    q10, q25, q60, q75 = np.quantile(y, [0.10, 0.25, 0.60, 0.75])

    # Start near the simple two-active-component shape that the fits settle on.
    # The remaining components get a small geometric tail.
    primary_weight = 0.72
    secondary_weight = 0.22
    remainder_weight = max(1.0 - primary_weight - secondary_weight, 1e-6)
    if k > 2:
        tail_profile = np.geomspace(1.0, 0.25, k - 2)
        tail_weights = remainder_weight * tail_profile / np.sum(tail_profile)
        weights = np.concatenate([[primary_weight, secondary_weight], tail_weights])
    else:
        weights = np.array([primary_weight, 1.0 - primary_weight], dtype=float)
    weights = weights / np.sum(weights)

    mu = np.empty(k, dtype=float)
    mu[0] = float(q60)
    if k > 1:
        mu[1] = float(q10)
    if k > 2:
        inactive_grid = np.linspace(float(q25), float(q75), k - 2)
        mu[2:] = y_mean + 0.15 * (inactive_grid - y_mean)

    remaining = np.cumprod(np.concatenate([[1.0], 1.0 - weights[:-1]]))
    v = weights[:-1] / np.clip(remaining[:-1], 1e-8, None)
    v = np.clip(v, 1e-5, 1.0 - 1e-5)
    eta_raw = _logit_np(v)

    sigma = np.full(k, float(sigma_init), dtype=float)
    sigma[: min(2, k)] = 0.45
    sigma = np.clip(sigma, sigma_low + 1e-6, sigma_high - 1e-6)
    sigma_unit = (sigma - sigma_low) / (sigma_high - sigma_low)
    sigma_raw = _logit_np(sigma_unit)

    loc = np.concatenate([eta_raw, mu, sigma_raw]).astype(float)
    if jitter_scale:
        rng = np.random.default_rng(seed)
        loc = loc + float(jitter_scale) * rng.normal(size=loc.shape)

    scale = np.full(latent_dim(k), float(initial_scale), dtype=float)
    constrained = {
        "eta": _sigmoid_np(eta_raw),
        "mu": mu.astype(float),
        "sigma": sigma.astype(float),
    }
    pymc_start_sigma = {
        "eta_logodds__": scale[: k - 1],
        "mu": scale[k - 1 : 2 * k - 1],
        "sigma_interval__": scale[2 * k - 1 : 3 * k - 1],
    }
    return {
        "loc": loc,
        "scale": scale,
        "constrained": constrained,
        "pymc_start_sigma": pymc_start_sigma,
    }


def default_bnp_species_plot_dims(n_components=DEFAULT_N_COMPONENTS):
    k = int(n_components)
    component_idxs = sorted(set([0, min(1, k - 1), min(2, k - 1), k // 2, k - 1]))
    dims = []
    for offset in (0, k, 2 * k):
        dims.extend([offset + idx for idx in component_idxs])
    labels = [make_bnp_species_param_names(k)[dim] for dim in dims]
    return dims, labels


def bnp_species_moments_from_raw(
    loc,
    scale,
    n_components=DEFAULT_N_COMPONENTS,
    n_samples=512,
    seed=0,
    batch_size=128,
    sort_components=True,
):
    """
    Estimate marginal means/stds of sorted model-space mixture parameters.

    loc/scale are diagonal Gaussian parameters in the shared unconstrained
    vector order: eta logits, component means, sigma interval logits.
    """
    loc = np.asarray(loc, dtype=float)
    scale = np.asarray(scale, dtype=float)

    squeeze = False
    if loc.ndim == 1:
        loc = loc[None, :]
        scale = scale[None, :]
        squeeze = True

    if loc.shape != scale.shape:
        raise ValueError(f"loc/scale shape mismatch: {loc.shape} vs {scale.shape}")

    n_track, dim = loc.shape
    expected_dim = latent_dim(n_components)
    if dim != expected_dim:
        raise ValueError(f"expected latent dim {expected_dim}, got {dim}")

    rng = np.random.default_rng(seed)
    out_dim = summary_dim(n_components)
    means = np.empty((n_track, out_dim), dtype=float)
    stds = np.empty((n_track, out_dim), dtype=float)

    n_samples = int(n_samples)
    batch_size = max(1, int(batch_size))
    for start in range(0, n_track, batch_size):
        stop = min(start + batch_size, n_track)
        batch_loc = loc[start:stop]
        batch_scale = scale[start:stop]
        eps = rng.normal(size=(n_samples, stop - start, dim))
        draws = batch_loc[None, :, :] + batch_scale[None, :, :] * eps
        params = raw_to_model_params_np(
            draws,
            n_components=n_components,
            sort_components=sort_components,
        )
        means[start:stop] = params.mean(axis=0)
        stds[start:stop] = params.std(axis=0)

    if squeeze:
        return means[0], stds[0]
    return means, stds


def apply_bnp_species_traj_transform(
    results,
    n_components=DEFAULT_N_COMPONENTS,
    n_samples=512,
    seed=0,
    batch_size=128,
    NOTEBOOK=True,
    sort_components=True,
):
    from tqdm import tqdm as standard_tqdm

    try:
        from tqdm.notebook import tqdm as notebook_tqdm
    except Exception:
        notebook_tqdm = standard_tqdm

    wrapper = notebook_tqdm if NOTEBOOK else standard_tqdm
    single_means, single_stds = [], []
    multi_means, multi_stds = [], []

    for restart_idx, (single_loc, single_scale, multi_loc, multi_scale) in enumerate(wrapper(results)):
        local_seed = seed + 10_000 * restart_idx
        s_mean, s_std = bnp_species_moments_from_raw(
            single_loc,
            single_scale,
            n_components=n_components,
            n_samples=n_samples,
            seed=local_seed,
            batch_size=batch_size,
            sort_components=sort_components,
        )
        m_mean, m_std = bnp_species_moments_from_raw(
            multi_loc,
            multi_scale,
            n_components=n_components,
            n_samples=n_samples,
            seed=local_seed + 1,
            batch_size=batch_size,
            sort_components=sort_components,
        )
        single_means.append(s_mean)
        single_stds.append(s_std)
        multi_means.append(m_mean)
        multi_stds.append(m_std)

    return (
        np.stack(single_means, axis=0),
        np.stack(single_stds, axis=0),
        np.stack(multi_means, axis=0),
        np.stack(multi_stds, axis=0),
    )


def build_bnp_species_final_latent_runs(results):
    summary_runs = []
    for restart_idx, run in enumerate(results):
        single_loc, single_scale, multi_loc, multi_scale = run
        summary_runs.append(
            {
                "restart_idx": restart_idx,
                "mc_setting": "1_mc",
                "mean": np.asarray(single_loc[-1], dtype=float),
                "std": np.asarray(single_scale[-1], dtype=float),
            }
        )
        summary_runs.append(
            {
                "restart_idx": restart_idx,
                "mc_setting": "100_mc",
                "mean": np.asarray(multi_loc[-1], dtype=float),
                "std": np.asarray(multi_scale[-1], dtype=float),
            }
        )
    return summary_runs


def mixture_density_from_summary(x, summary_mean, n_components=DEFAULT_N_COMPONENTS):
    x = np.asarray(x, dtype=float)
    params = np.asarray(summary_mean, dtype=float)
    k = int(n_components)
    weights = params[:k]
    mu = params[k : 2 * k]
    sigma = np.clip(params[2 * k : 3 * k], 1e-8, None)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    z = (x[:, None] - mu[None, :]) / sigma[None, :]
    dens = np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * sigma[None, :])
    return dens @ weights


def mixture_cdf_from_summary(x, summary_mean, n_components=DEFAULT_N_COMPONENTS):
    from math import erf

    x = np.asarray(x, dtype=float)
    params = np.asarray(summary_mean, dtype=float)
    k = int(n_components)
    weights = params[:k]
    mu = params[k : 2 * k]
    sigma = np.clip(params[2 * k : 3 * k], 1e-8, None)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    z = (x[:, None] - mu[None, :]) / (np.sqrt(2.0) * sigma[None, :])
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z))
    return cdf @ weights


def plot_bnp_species_selected_params(
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
    std_window=(0.5, 1.8),
    legend_below=False,
    legend_ncol=2,
):
    import matplotlib.pyplot as plt

    def _as_ref_list(refs, default_label):
        if isinstance(refs, (int, float, np.floating)):
            return [(float(refs), default_label)]
        return refs

    n_rows = len(dims)
    row_height = 4.8 if legend_below else 4.0
    fig, axs = plt.subplots(n_rows, 2, figsize=(16, row_height * n_rows), squeeze=False)

    iteration_stride = max(1, int(iteration_stride))
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
            for ref_idx, (value, ref_label) in enumerate(_as_ref_list(reference_means[label], "reference mean")):
                mean_refs.append(float(value))
                axs[row, 0].axhline(
                    value,
                    color="red",
                    linestyle="--" if ref_idx == 0 else ":",
                    label=ref_label,
                )

        if reference_stds and label in reference_stds:
            for ref_idx, (value, ref_label) in enumerate(_as_ref_list(reference_stds[label], "reference std")):
                std_refs.append(float(value))
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
            low, high = std_window
            axs[row, 1].set_ylim(max(min(std_refs) * low, 0.0), max(std_refs) * high)

        axs[row, 0].set_title(f"{title_prefix}{label}: variational mean")
        axs[row, 1].set_title(f"{title_prefix}{label}: variational std")
        axs[row, 0].set_xlabel("Iteration")
        axs[row, 1].set_xlabel("Iteration")
        axs[row, 0].grid()
        axs[row, 1].grid()
        if legend_below:
            legend_kwargs = {
                "loc": "upper center",
                "bbox_to_anchor": (0.5, -0.24),
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
        fig.subplots_adjust(hspace=0.9, bottom=0.08)
    plt.show()


def plot_bnp_species_selected_params_zoomed(*args, **kwargs):
    kwargs.setdefault("reference_window", True)
    kwargs.setdefault("legend_below", True)
    return plot_bnp_species_selected_params(*args, **kwargs)


def plot_bnp_species_fit_summary(
    data,
    reference_mean,
    n_components=DEFAULT_N_COMPONENTS,
    title="Carbaryl finite BNP mixture fit",
):
    import matplotlib.pyplot as plt

    y = np.asarray(data["y"], dtype=float)
    x_grid = np.linspace(y.min() - 0.75, y.max() + 0.75, 300)
    density = mixture_density_from_summary(x_grid, reference_mean, n_components=n_components)
    cdf = mixture_cdf_from_summary(x_grid, reference_mean, n_components=n_components)
    ecdf_x = np.sort(y)
    ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].hist(y, bins="auto", density=True, alpha=0.35, color="gray", label="species data")
    axs[0].plot(x_grid, density, color="red", linestyle="--", linewidth=2, label="best VI mixture")
    axs[0].set_title(f"{title}: density")
    axs[0].set_xlabel("standardized log10 concentration")
    axs[0].set_ylabel("density")
    axs[0].grid()
    axs[0].legend()

    axs[1].step(ecdf_x, ecdf_y, where="post", color="gray", label="empirical CDF")
    axs[1].plot(x_grid, cdf, color="red", linestyle="--", linewidth=2, label="best VI mixture")
    axs[1].set_title(f"{title}: SSD CDF")
    axs[1].set_xlabel("standardized log10 concentration")
    axs[1].set_ylabel("fraction affected")
    axs[1].grid()
    axs[1].legend()
    plt.tight_layout()
    plt.show()


def build_numpyro_bnp_species_model(
    n_components=DEFAULT_N_COMPONENTS,
    gamma=DEFAULT_GAMMA,
    sigma_low=DEFAULT_SIGMA_LOW,
    sigma_high=DEFAULT_SIGMA_HIGH,
):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    k = int(n_components)

    def _weights(v):
        prefix = jnp.concatenate(
            [jnp.ones((1,)), jnp.cumprod(1.0 - v[:-1])],
            axis=0,
        )
        return jnp.concatenate([v * prefix, jnp.prod(1.0 - v, keepdims=True)])

    def model(y):
        eta = numpyro.sample(
            "eta",
            dist.Beta(
                jnp.full((k - 1,), 1.0 - gamma),
                gamma * jnp.arange(1, k, dtype=jnp.float32),
            ).to_event(1),
        )
        mu = numpyro.sample("mu", dist.Normal(jnp.zeros(k), 1.0).to_event(1))
        sigma = numpyro.sample(
            "sigma",
            dist.Uniform(jnp.full((k,), sigma_low), jnp.full((k,), sigma_high)).to_event(1),
        )
        weights = _weights(eta)
        comp_lp = dist.Normal(mu, sigma).log_prob(y[..., None])
        obs_lp = jax.nn.logsumexp(jnp.log(weights) + comp_lp, axis=-1)
        numpyro.factor("obs", jnp.sum(obs_lp))

    return model


def make_tfp_bnp_species_conditioned_log_prob(
    data,
    n_components=DEFAULT_N_COMPONENTS,
    gamma=DEFAULT_GAMMA,
    sigma_low=DEFAULT_SIGMA_LOW,
    sigma_high=DEFAULT_SIGMA_HIGH,
    dtype=None,
):
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    if dtype is None:
        dtype = tf.float32

    k = int(n_components)
    y = tf.convert_to_tensor(data["y"], dtype=dtype)
    sigma_width = tf.convert_to_tensor(sigma_high - sigma_low, dtype=dtype)
    sigma_low_t = tf.convert_to_tensor(sigma_low, dtype=dtype)
    sigma_high_t = tf.convert_to_tensor(sigma_high, dtype=dtype)
    beta_a = tf.fill((k - 1,), tf.cast(1.0 - gamma, dtype))
    beta_b = tf.cast(gamma, dtype) * tf.cast(tf.range(1, k), dtype)

    def _weights(v):
        prefix = tf.concat(
            [
                tf.ones(tf.concat([tf.shape(v)[:-1], [1]], axis=0), dtype=dtype),
                tf.math.cumprod(1.0 - v[..., :-1], axis=-1),
            ],
            axis=-1,
        )
        residual = tf.reduce_prod(1.0 - v, axis=-1, keepdims=True)
        return tf.concat([v * prefix, residual], axis=-1)

    @tf.function(jit_compile=False)
    def log_prob_fn(raw):
        raw = tf.convert_to_tensor(raw, dtype=dtype)
        eta_raw = raw[..., : k - 1]
        mu = raw[..., k - 1 : 2 * k - 1]
        sigma_raw = raw[..., 2 * k - 1 : 3 * k - 1]

        eta = tf.sigmoid(eta_raw)
        sigma_unit = tf.sigmoid(sigma_raw)
        sigma = sigma_low_t + sigma_width * sigma_unit
        weights = _weights(eta)

        lp = tf.reduce_sum(tfd.Beta(beta_a, beta_b).log_prob(eta), axis=-1)
        lp += tf.reduce_sum(tf.math.log_sigmoid(eta_raw) + tf.math.log_sigmoid(-eta_raw), axis=-1)
        lp += tf.reduce_sum(tfd.Normal(0.0, 1.0).log_prob(mu), axis=-1)
        lp += tf.reduce_sum(tfd.Uniform(sigma_low_t, sigma_high_t).log_prob(sigma), axis=-1)
        lp += tf.reduce_sum(
            tf.math.log(sigma_width)
            + tf.math.log_sigmoid(sigma_raw)
            + tf.math.log_sigmoid(-sigma_raw),
            axis=-1,
        )

        comp_lp = tfd.Normal(
            loc=mu[..., tf.newaxis, :],
            scale=sigma[..., tf.newaxis, :],
        ).log_prob(y[..., tf.newaxis])
        obs_lp = tf.reduce_logsumexp(tf.math.log(weights[..., tf.newaxis, :]) + comp_lp, axis=-1)
        lp += tf.reduce_sum(obs_lp, axis=-1)
        return lp

    return log_prob_fn


def build_pymc_bnp_species_model(
    data,
    n_components=DEFAULT_N_COMPONENTS,
    gamma=DEFAULT_GAMMA,
    sigma_low=DEFAULT_SIGMA_LOW,
    sigma_high=DEFAULT_SIGMA_HIGH,
):
    import pymc as pm
    import pytensor.tensor as pt

    k = int(n_components)
    y = np.asarray(data["y"], dtype=float)

    def _weights(v):
        prefix = pt.concatenate(
            [
                pt.ones_like(v[:1]),
                pt.cumprod(1.0 - v[:-1]),
            ],
            axis=0,
        )
        residual = pt.prod(1.0 - v, keepdims=True)
        return pt.concatenate([v * prefix, residual], axis=0)

    with pm.Model() as model:
        eta = pm.Beta(
            "eta",
            alpha=np.full(k - 1, 1.0 - gamma),
            beta=gamma * np.arange(1, k, dtype=float),
            shape=k - 1,
        )
        mu = pm.Normal("mu", mu=0.0, sigma=1.0, shape=k)
        sigma = pm.Uniform("sigma", lower=sigma_low, upper=sigma_high, shape=k)
        weights = pm.Deterministic("weights", _weights(eta))
        components = pm.Normal.dist(mu=mu, sigma=sigma, shape=k)
        pm.Mixture("obs", w=weights, comp_dists=components, observed=y)

    return model
