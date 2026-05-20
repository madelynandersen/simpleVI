from pathlib import Path

import numpy as np
import pandas as pd


PULMONARY_DATA_URL = (
    "https://gist.githubusercontent.com/ucals/"
    "2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/"
    "43034c39052dcf97d4b894d2ec1bc3f90f3623d9/"
    "osic_pulmonary_fibrosis.csv"
)

PULMONARY_TUTORIAL_REPORTED_SUMMARIES = {
    "mu_alpha_global": {"mean": 1660.172, "sd": 309.657, "hdi_3%": 1118.038, "hdi_97%": 2274.933},
    "mu_beta_global": {"mean": -1.252, "sd": 2.062, "hdi_3%": -5.014, "hdi_97%": 2.678},
    "mu_alpha_smoking_status[0]": {"mean": 2970.486, "sd": 227.761, "hdi_3%": 2572.943, "hdi_97%": 3429.343},
    "mu_alpha_smoking_status[1]": {"mean": 2907.950, "sd": 68.011, "hdi_3%": 2782.993, "hdi_97%": 3035.172},
    "mu_alpha_smoking_status[2]": {"mean": 2475.281, "sd": 102.948, "hdi_3%": 2286.072, "hdi_97%": 2671.298},
    "mu_beta_smoking_status[0]": {"mean": 2.061, "sd": 1.713, "hdi_3%": -1.278, "hdi_97%": 5.072},
    "mu_beta_smoking_status[1]": {"mean": -4.625, "sd": 0.498, "hdi_3%": -5.566, "hdi_97%": -3.721},
    "mu_beta_smoking_status[2]": {"mean": -4.513, "sd": 0.789, "hdi_3%": -6.011, "hdi_97%": -3.056},
}

PULMONARY_REFERENCE_NOTES = {
    "numpyro_tutorial": (
        "Reference values are from the NumPyro 0.15.2 Bayesian hierarchical "
        "linear regression tutorial's smoking-status hierarchy summary table."
    ),
    "smoking_status_codes": (
        "The tutorial uses sklearn LabelEncoder order: 0=Currently smokes, "
        "1=Ex-smoker, 2=Never smoked."
    ),
    "vi_scale": (
        "The reported comparison parameters are Normal-location parameters, "
        "so the same values are used for the unconstrained VI coordinates."
    ),
}


def make_pulmonary_param_names(n_patients, n_smoking_statuses=3):
    """Return the ordered latent-vector labels used by the pulmonary notebooks."""
    return (
        [
            "mu_alpha_global",
            "sigma_alpha_global_log__",
            "mu_beta_global",
            "sigma_beta_global_log__",
        ]
        + [f"mu_alpha_smoking_status[{i}]" for i in range(n_smoking_statuses)]
        + [f"mu_beta_smoking_status[{i}]" for i in range(n_smoking_statuses)]
        + [f"alpha[{i}]" for i in range(n_patients)]
        + [f"beta[{i}]" for i in range(n_patients)]
        + ["sigma_log__"]
    )


def make_pulmonary_noncentered_param_names(n_patients, n_smoking_statuses=3):
    """Return the ordered non-centered latent labels used for VI fitting."""
    return (
        [
            "mu_alpha_global",
            "sigma_alpha_global_log__",
            "mu_beta_global",
            "sigma_beta_global_log__",
        ]
        + [f"z_mu_alpha_smoking_status[{i}]" for i in range(n_smoking_statuses)]
        + [f"z_mu_beta_smoking_status[{i}]" for i in range(n_smoking_statuses)]
        + [f"z_alpha[{i}]" for i in range(n_patients)]
        + [f"z_beta[{i}]" for i in range(n_patients)]
        + ["sigma_log__"]
    )


def _first_existing_path(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def load_pulmonary_fibrosis_data(data_path=None):
    """
    Load and preprocess the OSIC pulmonary fibrosis data from the NumPyro tutorial.
    """
    if data_path is None:
        here = Path(__file__).resolve().parent
        repo_root = here.parent
        data_path = _first_existing_path(
            [
                repo_root / "single_MC" / "pulmonary_lin_reg" / "osic_pulmonary_fibrosis.csv",
                repo_root / "single_MC" / "pulmonary_lin_reg" / "data" / "osic_pulmonary_fibrosis.csv",
                repo_root / "data" / "osic_pulmonary_fibrosis.csv",
            ]
        )

    if data_path is None:
        train = pd.read_csv(PULMONARY_DATA_URL)
    else:
        train = pd.read_csv(data_path)

    required = {"Patient", "Weeks", "FVC", "SmokingStatus"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Pulmonary data are missing required columns: {sorted(missing)}")

    train = train.copy()

    patient_categories = pd.Categorical(train["Patient"])
    smoking_categories = pd.Categorical(
        train["SmokingStatus"],
        categories=["Currently smokes", "Ex-smoker", "Never smoked"],
        ordered=True,
    )

    if np.any(smoking_categories.codes < 0):
        unknown = sorted(set(train.loc[smoking_categories.codes < 0, "SmokingStatus"]))
        raise ValueError(f"Unexpected SmokingStatus values: {unknown}")

    train["patient_code"] = patient_categories.codes.astype(np.int32)
    train["smoking_status_code"] = smoking_categories.codes.astype(np.int32)

    map_patient_to_smoking_status = (
        train[["patient_code", "smoking_status_code"]]
        .drop_duplicates()
        .set_index("patient_code", verify_integrity=True)
        .sort_index()["smoking_status_code"]
        .to_numpy(dtype=np.int32)
    )

    patient_names = np.asarray(patient_categories.categories)
    smoking_status_names = np.asarray(smoking_categories.categories)

    return {
        "frame": train,
        "FVC": train["FVC"].astype(float).to_numpy(),
        "Weeks": train["Weeks"].astype(float).to_numpy(),
        "patient_code": train["patient_code"].to_numpy(dtype=np.int32),
        "smoking_status_code": train["smoking_status_code"].to_numpy(dtype=np.int32),
        "map_patient_to_smoking_status": map_patient_to_smoking_status,
        "patient_names": patient_names,
        "smoking_status_names": smoking_status_names,
        "param_names": make_pulmonary_param_names(len(patient_names), len(smoking_status_names)),
        "noncentered_param_names": make_pulmonary_noncentered_param_names(
            len(patient_names), len(smoking_status_names)
        ),
    }


def default_pulmonary_plot_dims(param_names):
    """Pick tutorial-reported smoking-status dimensions for first-pass plots."""
    target_names = [
        "mu_alpha_global",
        "mu_beta_global",
        "mu_alpha_smoking_status[0]",
        "mu_alpha_smoking_status[1]",
        "mu_alpha_smoking_status[2]",
        "mu_beta_smoking_status[0]",
        "mu_beta_smoking_status[1]",
        "mu_beta_smoking_status[2]",
    ]
    dims = [param_names.index(name) for name in target_names if name in param_names]
    labels = [param_names[d] for d in dims]
    return dims, labels


def pulmonary_reference_means_and_stds():
    """Return reference dictionaries in the format expected by plotting helpers."""
    means = {
        name: [(summary["mean"], "NumPyro tutorial mean")]
        for name, summary in PULMONARY_TUTORIAL_REPORTED_SUMMARIES.items()
    }
    stds = {
        name: [(summary["sd"], "NumPyro tutorial sd")]
        for name, summary in PULMONARY_TUTORIAL_REPORTED_SUMMARIES.items()
    }
    return means, stds


def build_numpyro_pulmonary_smoking_model():
    """Build the NumPyro tutorial smoking-status hierarchy for NumPyro SVI."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(patient_code, Weeks, map_patient_to_smoking_status, FVC_obs=None):
        n_patients = map_patient_to_smoking_status.shape[0]
        n_smoking_statuses = 3

        mu_alpha_global = numpyro.sample("mu_alpha_global", dist.Normal(0.0, 500.0))
        sigma_alpha_global = numpyro.sample("sigma_alpha_global", dist.HalfNormal(100.0))
        mu_beta_global = numpyro.sample("mu_beta_global", dist.Normal(0.0, 3.0))
        sigma_beta_global = numpyro.sample("sigma_beta_global", dist.HalfNormal(3.0))

        mu_alpha_smoking_status = numpyro.sample(
            "mu_alpha_smoking_status",
            dist.Normal(
                jnp.full((n_smoking_statuses,), mu_alpha_global),
                sigma_alpha_global,
            ).to_event(1),
        )
        mu_beta_smoking_status = numpyro.sample(
            "mu_beta_smoking_status",
            dist.Normal(
                jnp.full((n_smoking_statuses,), mu_beta_global),
                sigma_beta_global,
            ).to_event(1),
        )

        alpha = numpyro.sample(
            "alpha",
            dist.Normal(
                mu_alpha_smoking_status[map_patient_to_smoking_status],
                sigma_alpha_global,
            ).to_event(1),
        )
        beta = numpyro.sample(
            "beta",
            dist.Normal(
                mu_beta_smoking_status[map_patient_to_smoking_status],
                sigma_beta_global,
            ).to_event(1),
        )

        sigma = numpyro.sample("sigma", dist.HalfNormal(100.0))
        FVC_est = alpha[patient_code] + beta[patient_code] * Weeks
        numpyro.sample("obs", dist.Normal(FVC_est, sigma), obs=FVC_obs)

    return model


def pulmonary_ols_initial_values(data):
    """
    Build a data-informed starting point for the tutorial non-centered model.

    The returned values are not used as a reference target; they only put VI near
    the FVC scale so it does not spend the run climbing out from zero.
    """
    n_patients = len(data["patient_names"])
    n_smoking_statuses = len(data["smoking_status_names"])
    patient_code = data["patient_code"]
    weeks = data["Weeks"]
    fvc = data["FVC"]
    map_patient_to_smoking_status = data["map_patient_to_smoking_status"]

    alpha = np.zeros(n_patients, dtype=float)
    beta = np.zeros(n_patients, dtype=float)
    for patient_idx in range(n_patients):
        mask = patient_code == patient_idx
        design = np.column_stack([np.ones(np.sum(mask)), weeks[mask]])
        alpha[patient_idx], beta[patient_idx] = np.linalg.lstsq(
            design, fvc[mask], rcond=None
        )[0]

    mu_alpha_smoking_status = np.array(
        [
            np.mean(alpha[map_patient_to_smoking_status == status_idx])
            for status_idx in range(n_smoking_statuses)
        ],
        dtype=float,
    )
    mu_beta_smoking_status = np.array(
        [
            np.mean(beta[map_patient_to_smoking_status == status_idx])
            for status_idx in range(n_smoking_statuses)
        ],
        dtype=float,
    )

    sigma_alpha_global = float(
        max(np.std(alpha - mu_alpha_smoking_status[map_patient_to_smoking_status]), 10.0)
    )
    sigma_beta_global = float(
        max(np.std(beta - mu_beta_smoking_status[map_patient_to_smoking_status]), 0.5)
    )
    mu_alpha_global = _normal_prior_shrunken_mean(
        mu_alpha_smoking_status, sigma_alpha_global, prior_sd=500.0
    )
    mu_beta_global = _normal_prior_shrunken_mean(
        mu_beta_smoking_status, sigma_beta_global, prior_sd=3.0
    )
    residual = fvc - (alpha[patient_code] + beta[patient_code] * weeks)
    sigma = float(max(np.std(residual), 10.0))

    z_mu_alpha_smoking_status = (
        mu_alpha_smoking_status - mu_alpha_global
    ) / sigma_alpha_global
    z_mu_beta_smoking_status = (
        mu_beta_smoking_status - mu_beta_global
    ) / sigma_beta_global
    z_alpha = (
        alpha - mu_alpha_smoking_status[map_patient_to_smoking_status]
    ) / sigma_alpha_global
    z_beta = (
        beta - mu_beta_smoking_status[map_patient_to_smoking_status]
    ) / sigma_beta_global

    return {
        "mu_alpha_global": mu_alpha_global,
        "sigma_alpha_global": sigma_alpha_global,
        "sigma_alpha_global_log__": np.log(sigma_alpha_global),
        "mu_beta_global": mu_beta_global,
        "sigma_beta_global": sigma_beta_global,
        "sigma_beta_global_log__": np.log(sigma_beta_global),
        "mu_alpha_smoking_status": mu_alpha_smoking_status,
        "mu_beta_smoking_status": mu_beta_smoking_status,
        "alpha": alpha,
        "beta": beta,
        "z_mu_alpha_smoking_status": z_mu_alpha_smoking_status,
        "z_mu_beta_smoking_status": z_mu_beta_smoking_status,
        "z_alpha": z_alpha,
        "z_beta": z_beta,
        "sigma": sigma,
        "sigma_log__": np.log(sigma),
    }


def _normal_prior_shrunken_mean(values, likelihood_sd, prior_sd):
    values = np.asarray(values, dtype=float)
    likelihood_precision = values.size / likelihood_sd**2
    prior_precision = 1.0 / prior_sd**2
    return float(np.sum(values) / likelihood_sd**2 / (likelihood_precision + prior_precision))


def pulmonary_noncentered_initial_loc(data):
    """Return the initial VI location vector in non-centered unconstrained order."""
    init = pulmonary_ols_initial_values(data)
    return np.concatenate(
        [
            np.array(
                [
                    init["mu_alpha_global"],
                    init["sigma_alpha_global_log__"],
                    init["mu_beta_global"],
                    init["sigma_beta_global_log__"],
                ],
                dtype=float,
            ),
            init["z_mu_alpha_smoking_status"],
            init["z_mu_beta_smoking_status"],
            init["z_alpha"],
            init["z_beta"],
            np.array([init["sigma_log__"]], dtype=float),
        ]
    )


def pulmonary_numpyro_noncentered_init_values(data):
    """Return constrained initial values keyed by NumPyro non-centered site name."""
    import jax.numpy as jnp

    init = pulmonary_ols_initial_values(data)
    return {
        "mu_alpha_global": jnp.asarray(init["mu_alpha_global"]),
        "sigma_alpha_global": jnp.asarray(init["sigma_alpha_global"]),
        "mu_beta_global": jnp.asarray(init["mu_beta_global"]),
        "sigma_beta_global": jnp.asarray(init["sigma_beta_global"]),
        "z_mu_alpha_smoking_status": jnp.asarray(init["z_mu_alpha_smoking_status"]),
        "z_mu_beta_smoking_status": jnp.asarray(init["z_mu_beta_smoking_status"]),
        "z_alpha": jnp.asarray(init["z_alpha"]),
        "z_beta": jnp.asarray(init["z_beta"]),
        "sigma": jnp.asarray(init["sigma"]),
    }


def _unpack_noncentered_arrays(values, n_patients, n_smoking_statuses):
    idx = 0
    out = {}
    out["mu_alpha_global"] = values[..., idx]
    idx += 1
    out["sigma_alpha_global_log"] = values[..., idx]
    idx += 1
    out["mu_beta_global"] = values[..., idx]
    idx += 1
    out["sigma_beta_global_log"] = values[..., idx]
    idx += 1
    out["z_mu_alpha_smoking_status"] = values[..., idx : idx + n_smoking_statuses]
    idx += n_smoking_statuses
    out["z_mu_beta_smoking_status"] = values[..., idx : idx + n_smoking_statuses]
    idx += n_smoking_statuses
    out["z_alpha"] = values[..., idx : idx + n_patients]
    idx += n_patients
    out["z_beta"] = values[..., idx : idx + n_patients]
    idx += n_patients
    out["sigma_log"] = values[..., idx]
    return out


def pulmonary_noncentered_to_centered_values(theta, data):
    """Map non-centered unconstrained samples to original tutorial coordinates."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[None, :]
    n_patients = len(data["patient_names"])
    n_smoking_statuses = len(data["smoking_status_names"])
    map_patient_to_smoking_status = data["map_patient_to_smoking_status"]
    parts = _unpack_noncentered_arrays(theta, n_patients, n_smoking_statuses)

    sigma_alpha_global = np.exp(parts["sigma_alpha_global_log"])
    sigma_beta_global = np.exp(parts["sigma_beta_global_log"])
    mu_alpha_smoking_status = (
        parts["mu_alpha_global"][..., None]
        + sigma_alpha_global[..., None] * parts["z_mu_alpha_smoking_status"]
    )
    mu_beta_smoking_status = (
        parts["mu_beta_global"][..., None]
        + sigma_beta_global[..., None] * parts["z_mu_beta_smoking_status"]
    )
    alpha = (
        mu_alpha_smoking_status[..., map_patient_to_smoking_status]
        + sigma_alpha_global[..., None] * parts["z_alpha"]
    )
    beta = (
        mu_beta_smoking_status[..., map_patient_to_smoking_status]
        + sigma_beta_global[..., None] * parts["z_beta"]
    )

    return np.concatenate(
        [
            parts["mu_alpha_global"][..., None],
            parts["sigma_alpha_global_log"][..., None],
            parts["mu_beta_global"][..., None],
            parts["sigma_beta_global_log"][..., None],
            mu_alpha_smoking_status,
            mu_beta_smoking_status,
            alpha,
            beta,
            parts["sigma_log"][..., None],
        ],
        axis=-1,
    )


def _lognormal_mean_second(log_loc, log_var):
    log_loc = np.asarray(log_loc, dtype=float)
    log_var = np.asarray(log_var, dtype=float)
    mean = np.exp(log_loc + 0.5 * log_var)
    second = np.exp(2.0 * log_loc + 2.0 * log_var)
    return mean, second


def _affine_lognormal_normal_moments(base_loc, base_var, log_loc, log_var, z_loc, z_var):
    scale_mean, scale_second = _lognormal_mean_second(log_loc, log_var)
    mean = base_loc + scale_mean * z_loc
    var = base_var + scale_second * (z_var + z_loc**2) - (scale_mean * z_loc) ** 2
    return mean, np.sqrt(np.clip(var, 0.0, None))


def pulmonary_noncentered_to_centered_moments(loc_trace, scale_trace, data, **kwargs):
    """
    Convert diagonal Normal moments in non-centered coordinates to marginal
    moments of the original tutorial parameters.
    """
    del kwargs
    loc = np.asarray(loc_trace, dtype=float)
    scale = np.asarray(scale_trace, dtype=float)
    if loc.ndim == 1:
        loc = loc[None, :]
    if scale.ndim == 1:
        scale = scale[None, :]
    var = np.square(scale)

    n_patients = len(data["patient_names"])
    n_smoking_statuses = len(data["smoking_status_names"])
    map_patient_to_smoking_status = data["map_patient_to_smoking_status"]
    loc_parts = _unpack_noncentered_arrays(loc, n_patients, n_smoking_statuses)
    var_parts = _unpack_noncentered_arrays(var, n_patients, n_smoking_statuses)

    out_mean = np.zeros(loc.shape, dtype=float)
    out_sd = np.zeros(loc.shape, dtype=float)

    out_mean[..., 0] = loc_parts["mu_alpha_global"]
    out_sd[..., 0] = scale[..., 0]
    out_mean[..., 1] = loc_parts["sigma_alpha_global_log"]
    out_sd[..., 1] = scale[..., 1]
    out_mean[..., 2] = loc_parts["mu_beta_global"]
    out_sd[..., 2] = scale[..., 2]
    out_mean[..., 3] = loc_parts["sigma_beta_global_log"]
    out_sd[..., 3] = scale[..., 3]

    alpha_status_mean, alpha_status_sd = _affine_lognormal_normal_moments(
        loc_parts["mu_alpha_global"][..., None],
        var_parts["mu_alpha_global"][..., None],
        loc_parts["sigma_alpha_global_log"][..., None],
        var_parts["sigma_alpha_global_log"][..., None],
        loc_parts["z_mu_alpha_smoking_status"],
        var_parts["z_mu_alpha_smoking_status"],
    )
    beta_status_mean, beta_status_sd = _affine_lognormal_normal_moments(
        loc_parts["mu_beta_global"][..., None],
        var_parts["mu_beta_global"][..., None],
        loc_parts["sigma_beta_global_log"][..., None],
        var_parts["sigma_beta_global_log"][..., None],
        loc_parts["z_mu_beta_smoking_status"],
        var_parts["z_mu_beta_smoking_status"],
    )

    alpha_z_loc = (
        loc_parts["z_mu_alpha_smoking_status"][..., map_patient_to_smoking_status]
        + loc_parts["z_alpha"]
    )
    alpha_z_var = (
        var_parts["z_mu_alpha_smoking_status"][..., map_patient_to_smoking_status]
        + var_parts["z_alpha"]
    )
    alpha_mean, alpha_sd = _affine_lognormal_normal_moments(
        loc_parts["mu_alpha_global"][..., None],
        var_parts["mu_alpha_global"][..., None],
        loc_parts["sigma_alpha_global_log"][..., None],
        var_parts["sigma_alpha_global_log"][..., None],
        alpha_z_loc,
        alpha_z_var,
    )

    beta_z_loc = (
        loc_parts["z_mu_beta_smoking_status"][..., map_patient_to_smoking_status]
        + loc_parts["z_beta"]
    )
    beta_z_var = (
        var_parts["z_mu_beta_smoking_status"][..., map_patient_to_smoking_status]
        + var_parts["z_beta"]
    )
    beta_mean, beta_sd = _affine_lognormal_normal_moments(
        loc_parts["mu_beta_global"][..., None],
        var_parts["mu_beta_global"][..., None],
        loc_parts["sigma_beta_global_log"][..., None],
        var_parts["sigma_beta_global_log"][..., None],
        beta_z_loc,
        beta_z_var,
    )

    idx = 4
    out_mean[..., idx : idx + n_smoking_statuses] = alpha_status_mean
    out_sd[..., idx : idx + n_smoking_statuses] = alpha_status_sd
    idx += n_smoking_statuses
    out_mean[..., idx : idx + n_smoking_statuses] = beta_status_mean
    out_sd[..., idx : idx + n_smoking_statuses] = beta_status_sd
    idx += n_smoking_statuses
    out_mean[..., idx : idx + n_patients] = alpha_mean
    out_sd[..., idx : idx + n_patients] = alpha_sd
    idx += n_patients
    out_mean[..., idx : idx + n_patients] = beta_mean
    out_sd[..., idx : idx + n_patients] = beta_sd
    idx += n_patients
    out_mean[..., idx] = loc_parts["sigma_log"]
    out_sd[..., idx] = scale[..., idx]

    return out_mean, out_sd


def compact_pulmonary_restart_result(result, data, plot_dims, restart_idx):
    """
    Keep only the pulmonary restart values needed for plotting and reference scoring.

    We keep selected dimensions for trajectories, all dimensions for final summaries,
    and only final non-centered VI parameters for reference scoring.
    """
    single_locs, single_scales, multi_locs, multi_scales = [
        np.asarray(value) for value in result
    ]
    plot_dims = np.asarray(plot_dims, dtype=int)

    single_means, single_stds = pulmonary_noncentered_to_centered_moments(
        single_locs, single_scales, data
    )
    multi_means, multi_stds = pulmonary_noncentered_to_centered_moments(
        multi_locs, multi_scales, data
    )

    return {
        "single_means": single_means[:, plot_dims],
        "single_stds": single_stds[:, plot_dims],
        "multi_means": multi_means[:, plot_dims],
        "multi_stds": multi_stds[:, plot_dims],
        "single_final_means": single_means[-1],
        "single_final_stds": single_stds[-1],
        "multi_final_means": multi_means[-1],
        "multi_final_stds": multi_stds[-1],
        "reference_candidates": [
            {
                "restart_idx": int(restart_idx),
                "mc_setting": "1_mc",
                "loc": single_locs[-1],
                "scale": single_scales[-1],
                "n_records": int(single_locs.shape[0]),
            },
            {
                "restart_idx": int(restart_idx),
                "mc_setting": "100_mc",
                "loc": multi_locs[-1],
                "scale": multi_scales[-1],
                "n_records": int(multi_locs.shape[0]),
            },
        ],
    }


def stack_pulmonary_compact_results(compact_results):
    """Stack compact pulmonary restart dictionaries into one saveable payload."""
    keys = [
        "single_means",
        "single_stds",
        "multi_means",
        "multi_stds",
        "single_final_means",
        "single_final_stds",
        "multi_final_means",
        "multi_final_stds",
    ]
    return {
        key: np.stack([result[key] for result in compact_results], axis=0)
        for key in keys
    }


def build_numpyro_pulmonary_smoking_model_noncentered():
    """Build the tutorial smoking-status hierarchy in non-centered form."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(patient_code, Weeks, map_patient_to_smoking_status, FVC_obs=None):
        n_patients = map_patient_to_smoking_status.shape[0]
        n_smoking_statuses = 3

        mu_alpha_global = numpyro.sample("mu_alpha_global", dist.Normal(0.0, 500.0))
        sigma_alpha_global = numpyro.sample("sigma_alpha_global", dist.HalfNormal(100.0))
        mu_beta_global = numpyro.sample("mu_beta_global", dist.Normal(0.0, 3.0))
        sigma_beta_global = numpyro.sample("sigma_beta_global", dist.HalfNormal(3.0))

        z_mu_alpha_smoking_status = numpyro.sample(
            "z_mu_alpha_smoking_status",
            dist.Normal(jnp.zeros(n_smoking_statuses), 1.0).to_event(1),
        )
        z_mu_beta_smoking_status = numpyro.sample(
            "z_mu_beta_smoking_status",
            dist.Normal(jnp.zeros(n_smoking_statuses), 1.0).to_event(1),
        )
        mu_alpha_smoking_status = numpyro.deterministic(
            "mu_alpha_smoking_status",
            mu_alpha_global + sigma_alpha_global * z_mu_alpha_smoking_status,
        )
        mu_beta_smoking_status = numpyro.deterministic(
            "mu_beta_smoking_status",
            mu_beta_global + sigma_beta_global * z_mu_beta_smoking_status,
        )

        z_alpha = numpyro.sample(
            "z_alpha", dist.Normal(jnp.zeros(n_patients), 1.0).to_event(1)
        )
        z_beta = numpyro.sample(
            "z_beta", dist.Normal(jnp.zeros(n_patients), 1.0).to_event(1)
        )
        alpha = numpyro.deterministic(
            "alpha",
            mu_alpha_smoking_status[map_patient_to_smoking_status]
            + sigma_alpha_global * z_alpha,
        )
        beta = numpyro.deterministic(
            "beta",
            mu_beta_smoking_status[map_patient_to_smoking_status]
            + sigma_beta_global * z_beta,
        )

        sigma = numpyro.sample("sigma", dist.HalfNormal(100.0))
        FVC_est = alpha[patient_code] + beta[patient_code] * Weeks
        numpyro.sample("obs", dist.Normal(FVC_est, sigma), obs=FVC_obs)

    return model


def make_tfp_pulmonary_smoking_conditioned_log_prob(data, dtype=None):
    """Build the pulmonary smoking-status hierarchy log-prob function for TFP."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    if dtype is None:
        dtype = tf.float32

    FVC = tf.convert_to_tensor(data["FVC"], dtype=dtype)
    Weeks = tf.convert_to_tensor(data["Weeks"], dtype=dtype)
    patient_code = tf.convert_to_tensor(data["patient_code"], dtype=tf.int32)
    map_patient_to_smoking_status = tf.convert_to_tensor(
        data["map_patient_to_smoking_status"], dtype=tf.int32
    )
    n_patients = int(len(data["patient_names"]))
    n_smoking_statuses = int(len(data["smoking_status_names"]))

    @tf.function(jit_compile=False)
    def log_prob_fn(theta):
        theta = tf.convert_to_tensor(theta, dtype=dtype)
        idx = 0
        mu_alpha_global = theta[..., idx]
        idx += 1
        sigma_alpha_global_log = theta[..., idx]
        idx += 1
        mu_beta_global = theta[..., idx]
        idx += 1
        sigma_beta_global_log = theta[..., idx]
        idx += 1
        mu_alpha_smoking_status = theta[..., idx : idx + n_smoking_statuses]
        idx += n_smoking_statuses
        mu_beta_smoking_status = theta[..., idx : idx + n_smoking_statuses]
        idx += n_smoking_statuses
        alpha = theta[..., idx : idx + n_patients]
        idx += n_patients
        beta = theta[..., idx : idx + n_patients]
        idx += n_patients
        sigma_log = theta[..., idx]

        sigma_alpha_global = tf.exp(sigma_alpha_global_log)
        sigma_beta_global = tf.exp(sigma_beta_global_log)
        sigma = tf.exp(sigma_log)

        alpha_status_mean = tf.gather(mu_alpha_smoking_status, map_patient_to_smoking_status, axis=-1)
        beta_status_mean = tf.gather(mu_beta_smoking_status, map_patient_to_smoking_status, axis=-1)
        alpha_obs = tf.gather(alpha, patient_code, axis=-1)
        beta_obs = tf.gather(beta, patient_code, axis=-1)
        fvc_est = alpha_obs + beta_obs * Weeks

        lp = tfd.Normal(0.0, 500.0).log_prob(mu_alpha_global)
        lp += tfd.HalfNormal(100.0).log_prob(sigma_alpha_global) + sigma_alpha_global_log
        lp += tfd.Normal(0.0, 3.0).log_prob(mu_beta_global)
        lp += tfd.HalfNormal(3.0).log_prob(sigma_beta_global) + sigma_beta_global_log
        lp += tf.reduce_sum(
            tfd.Normal(mu_alpha_global[..., tf.newaxis], sigma_alpha_global[..., tf.newaxis]).log_prob(
                mu_alpha_smoking_status
            ),
            axis=-1,
        )
        lp += tf.reduce_sum(
            tfd.Normal(mu_beta_global[..., tf.newaxis], sigma_beta_global[..., tf.newaxis]).log_prob(
                mu_beta_smoking_status
            ),
            axis=-1,
        )
        lp += tf.reduce_sum(
            tfd.Normal(alpha_status_mean, sigma_alpha_global[..., tf.newaxis]).log_prob(alpha),
            axis=-1,
        )
        lp += tf.reduce_sum(
            tfd.Normal(beta_status_mean, sigma_beta_global[..., tf.newaxis]).log_prob(beta),
            axis=-1,
        )
        lp += tfd.HalfNormal(100.0).log_prob(sigma) + sigma_log
        lp += tf.reduce_sum(tfd.Normal(fvc_est, sigma[..., tf.newaxis]).log_prob(FVC), axis=-1)
        return lp

    return log_prob_fn


def make_tfp_pulmonary_smoking_noncentered_conditioned_log_prob(data, dtype=None):
    """Build the tutorial smoking-status hierarchy log-prob in non-centered form."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    if dtype is None:
        dtype = tf.float32

    FVC = tf.convert_to_tensor(data["FVC"], dtype=dtype)
    Weeks = tf.convert_to_tensor(data["Weeks"], dtype=dtype)
    patient_code = tf.convert_to_tensor(data["patient_code"], dtype=tf.int32)
    map_patient_to_smoking_status = tf.convert_to_tensor(
        data["map_patient_to_smoking_status"], dtype=tf.int32
    )
    n_patients = int(len(data["patient_names"]))
    n_smoking_statuses = int(len(data["smoking_status_names"]))

    @tf.function(jit_compile=False)
    def log_prob_fn(theta):
        theta = tf.convert_to_tensor(theta, dtype=dtype)
        idx = 0
        mu_alpha_global = theta[..., idx]
        idx += 1
        sigma_alpha_global_log = theta[..., idx]
        idx += 1
        mu_beta_global = theta[..., idx]
        idx += 1
        sigma_beta_global_log = theta[..., idx]
        idx += 1
        z_mu_alpha_smoking_status = theta[..., idx : idx + n_smoking_statuses]
        idx += n_smoking_statuses
        z_mu_beta_smoking_status = theta[..., idx : idx + n_smoking_statuses]
        idx += n_smoking_statuses
        z_alpha = theta[..., idx : idx + n_patients]
        idx += n_patients
        z_beta = theta[..., idx : idx + n_patients]
        idx += n_patients
        sigma_log = theta[..., idx]

        sigma_alpha_global = tf.exp(sigma_alpha_global_log)
        sigma_beta_global = tf.exp(sigma_beta_global_log)
        sigma = tf.exp(sigma_log)

        mu_alpha_smoking_status = (
            mu_alpha_global[..., tf.newaxis]
            + sigma_alpha_global[..., tf.newaxis] * z_mu_alpha_smoking_status
        )
        mu_beta_smoking_status = (
            mu_beta_global[..., tf.newaxis]
            + sigma_beta_global[..., tf.newaxis] * z_mu_beta_smoking_status
        )
        alpha_status_mean = tf.gather(
            mu_alpha_smoking_status, map_patient_to_smoking_status, axis=-1
        )
        beta_status_mean = tf.gather(
            mu_beta_smoking_status, map_patient_to_smoking_status, axis=-1
        )
        alpha = alpha_status_mean + sigma_alpha_global[..., tf.newaxis] * z_alpha
        beta = beta_status_mean + sigma_beta_global[..., tf.newaxis] * z_beta
        alpha_obs = tf.gather(alpha, patient_code, axis=-1)
        beta_obs = tf.gather(beta, patient_code, axis=-1)
        fvc_est = alpha_obs + beta_obs * Weeks

        lp = tfd.Normal(0.0, 500.0).log_prob(mu_alpha_global)
        lp += tfd.HalfNormal(100.0).log_prob(sigma_alpha_global) + sigma_alpha_global_log
        lp += tfd.Normal(0.0, 3.0).log_prob(mu_beta_global)
        lp += tfd.HalfNormal(3.0).log_prob(sigma_beta_global) + sigma_beta_global_log
        lp += tf.reduce_sum(tfd.Normal(0.0, 1.0).log_prob(z_mu_alpha_smoking_status), axis=-1)
        lp += tf.reduce_sum(tfd.Normal(0.0, 1.0).log_prob(z_mu_beta_smoking_status), axis=-1)
        lp += tf.reduce_sum(tfd.Normal(0.0, 1.0).log_prob(z_alpha), axis=-1)
        lp += tf.reduce_sum(tfd.Normal(0.0, 1.0).log_prob(z_beta), axis=-1)
        lp += tfd.HalfNormal(100.0).log_prob(sigma) + sigma_log
        lp += tf.reduce_sum(tfd.Normal(fvc_est, sigma[..., tf.newaxis]).log_prob(FVC), axis=-1)
        return lp

    return log_prob_fn


def pulmonary_noncentered_log_joint_unconstrained(theta, data):
    """Evaluate the non-centered pulmonary log joint for unconstrained samples."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[None, :]
    n_patients = len(data["patient_names"])
    n_smoking_statuses = len(data["smoking_status_names"])
    patient_code = data["patient_code"]
    weeks = data["Weeks"]
    fvc = data["FVC"]

    parts = _unpack_noncentered_arrays(theta, n_patients, n_smoking_statuses)
    sigma_alpha_global = np.exp(parts["sigma_alpha_global_log"])
    sigma_beta_global = np.exp(parts["sigma_beta_global_log"])
    sigma = np.exp(parts["sigma_log"])
    centered = pulmonary_noncentered_to_centered_values(theta, data)
    centered_parts = _unpack_centered_arrays(centered, n_patients, n_smoking_statuses)

    lp = _normal_logpdf_np(parts["mu_alpha_global"], 0.0, 500.0)
    lp += _halfnormal_logpdf_np(sigma_alpha_global, 100.0) + parts["sigma_alpha_global_log"]
    lp += _normal_logpdf_np(parts["mu_beta_global"], 0.0, 3.0)
    lp += _halfnormal_logpdf_np(sigma_beta_global, 3.0) + parts["sigma_beta_global_log"]
    lp += np.sum(_normal_logpdf_np(parts["z_mu_alpha_smoking_status"], 0.0, 1.0), axis=1)
    lp += np.sum(_normal_logpdf_np(parts["z_mu_beta_smoking_status"], 0.0, 1.0), axis=1)
    lp += np.sum(_normal_logpdf_np(parts["z_alpha"], 0.0, 1.0), axis=1)
    lp += np.sum(_normal_logpdf_np(parts["z_beta"], 0.0, 1.0), axis=1)
    lp += _halfnormal_logpdf_np(sigma, 100.0) + parts["sigma_log"]

    fvc_est = (
        centered_parts["alpha"][:, patient_code]
        + centered_parts["beta"][:, patient_code] * weeks[None, :]
    )
    lp += np.sum(_normal_logpdf_np(fvc[None, :], fvc_est, sigma[:, None]), axis=1)
    return lp


def _normal_logpdf_np(x, loc, scale):
    scale = np.clip(scale, 1e-12, None)
    return -0.5 * np.log(2.0 * np.pi) - np.log(scale) - 0.5 * ((x - loc) / scale) ** 2


def _halfnormal_logpdf_np(x, scale):
    out = np.log(2.0) + _normal_logpdf_np(x, 0.0, scale)
    return np.where(x >= 0.0, out, -np.inf)


def _unpack_centered_arrays(values, n_patients, n_smoking_statuses):
    idx = 0
    out = {}
    out["mu_alpha_global"] = values[..., idx]
    idx += 1
    out["sigma_alpha_global_log"] = values[..., idx]
    idx += 1
    out["mu_beta_global"] = values[..., idx]
    idx += 1
    out["sigma_beta_global_log"] = values[..., idx]
    idx += 1
    out["mu_alpha_smoking_status"] = values[..., idx : idx + n_smoking_statuses]
    idx += n_smoking_statuses
    out["mu_beta_smoking_status"] = values[..., idx : idx + n_smoking_statuses]
    idx += n_smoking_statuses
    out["alpha"] = values[..., idx : idx + n_patients]
    idx += n_patients
    out["beta"] = values[..., idx : idx + n_patients]
    idx += n_patients
    out["sigma_log"] = values[..., idx]
    return out


def build_pymc_pulmonary_smoking_model(data):
    """Build the PyMC version of the pulmonary smoking-status hierarchy."""
    import pymc as pm

    n_patients = len(data["patient_names"])
    n_smoking_statuses = len(data["smoking_status_names"])
    map_patient_to_smoking_status = data["map_patient_to_smoking_status"]

    with pm.Model() as model:
        mu_alpha_global = pm.Normal("mu_alpha_global", mu=0.0, sigma=500.0)
        sigma_alpha_global = pm.HalfNormal("sigma_alpha_global", sigma=100.0)
        mu_beta_global = pm.Normal("mu_beta_global", mu=0.0, sigma=3.0)
        sigma_beta_global = pm.HalfNormal("sigma_beta_global", sigma=3.0)

        mu_alpha_smoking_status = pm.Normal(
            "mu_alpha_smoking_status",
            mu=mu_alpha_global,
            sigma=sigma_alpha_global,
            shape=n_smoking_statuses,
        )
        mu_beta_smoking_status = pm.Normal(
            "mu_beta_smoking_status",
            mu=mu_beta_global,
            sigma=sigma_beta_global,
            shape=n_smoking_statuses,
        )

        alpha = pm.Normal(
            "alpha",
            mu=mu_alpha_smoking_status[map_patient_to_smoking_status],
            sigma=sigma_alpha_global,
            shape=n_patients,
        )
        beta = pm.Normal(
            "beta",
            mu=mu_beta_smoking_status[map_patient_to_smoking_status],
            sigma=sigma_beta_global,
            shape=n_patients,
        )
        sigma = pm.HalfNormal("sigma", sigma=100.0)

        fvc_est = alpha[data["patient_code"]] + beta[data["patient_code"]] * data["Weeks"]
        pm.Normal("obs", mu=fvc_est, sigma=sigma, observed=data["FVC"])

    return model


def build_pymc_pulmonary_smoking_model_noncentered(data, init_values=None):
    """Build the PyMC version of the tutorial non-centered hierarchy."""
    import pymc as pm

    if init_values is None:
        init_values = pulmonary_ols_initial_values(data)

    n_patients = len(data["patient_names"])
    n_smoking_statuses = len(data["smoking_status_names"])
    map_patient_to_smoking_status = data["map_patient_to_smoking_status"]

    with pm.Model() as model:
        mu_alpha_global = pm.Normal(
            "mu_alpha_global", mu=0.0, sigma=500.0, initval=init_values["mu_alpha_global"]
        )
        sigma_alpha_global = pm.HalfNormal(
            "sigma_alpha_global", sigma=100.0, initval=init_values["sigma_alpha_global"]
        )
        mu_beta_global = pm.Normal(
            "mu_beta_global", mu=0.0, sigma=3.0, initval=init_values["mu_beta_global"]
        )
        sigma_beta_global = pm.HalfNormal(
            "sigma_beta_global", sigma=3.0, initval=init_values["sigma_beta_global"]
        )

        z_mu_alpha_smoking_status = pm.Normal(
            "z_mu_alpha_smoking_status",
            mu=0.0,
            sigma=1.0,
            shape=n_smoking_statuses,
            initval=init_values["z_mu_alpha_smoking_status"],
        )
        z_mu_beta_smoking_status = pm.Normal(
            "z_mu_beta_smoking_status",
            mu=0.0,
            sigma=1.0,
            shape=n_smoking_statuses,
            initval=init_values["z_mu_beta_smoking_status"],
        )
        mu_alpha_smoking_status = pm.Deterministic(
            "mu_alpha_smoking_status",
            mu_alpha_global + sigma_alpha_global * z_mu_alpha_smoking_status,
        )
        mu_beta_smoking_status = pm.Deterministic(
            "mu_beta_smoking_status",
            mu_beta_global + sigma_beta_global * z_mu_beta_smoking_status,
        )

        z_alpha = pm.Normal(
            "z_alpha",
            mu=0.0,
            sigma=1.0,
            shape=n_patients,
            initval=init_values["z_alpha"],
        )
        z_beta = pm.Normal(
            "z_beta",
            mu=0.0,
            sigma=1.0,
            shape=n_patients,
            initval=init_values["z_beta"],
        )
        alpha = pm.Deterministic(
            "alpha",
            mu_alpha_smoking_status[map_patient_to_smoking_status]
            + sigma_alpha_global * z_alpha,
        )
        beta = pm.Deterministic(
            "beta",
            mu_beta_smoking_status[map_patient_to_smoking_status]
            + sigma_beta_global * z_beta,
        )
        sigma = pm.HalfNormal("sigma", sigma=100.0, initval=init_values["sigma"])

        fvc_est = alpha[data["patient_code"]] + beta[data["patient_code"]] * data["Weeks"]
        pm.Normal("obs", mu=fvc_est, sigma=sigma, observed=data["FVC"])

    return model


def save_pulmonary_reference_summary(path):
    """Write the tutorial-reported pulmonary reference values to CSV."""
    rows = [
        {
            "source": "NumPyro tutorial",
            "parameter": name,
            "summary": summary,
            "value": value,
            "notes": "Smoking-status hierarchy summary from the NumPyro 0.15.2 tutorial.",
        }
        for name, summaries in PULMONARY_TUTORIAL_REPORTED_SUMMARIES.items()
        for summary, value in summaries.items()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plot_pulmonary_selected_dims(
    single_means,
    single_stds,
    multi_means,
    multi_stds,
    dims,
    labels,
    title_prefix="",
    reference_means=None,
    reference_stds=None,
    iteration_stride=1
):
    """Plot pulmonary VI mean/std trajectories for tutorial-reported dimensions."""
    import matplotlib.pyplot as plt

    def _as_ref_list(refs, default_label):
        if isinstance(refs, (int, float, np.floating)):
            return [(float(refs), default_label)]
        return refs

    n_rows = len(dims)
    fig, axs = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows), squeeze=False)
    iteration_stride = max(1, iteration_stride)
    if iteration_stride == 1:
        x = np.arange(single_means.shape[1])
    else:
        x = (np.arange(single_means.shape[1]) + 1) * iteration_stride

    for row, (dim, label) in enumerate(zip(dims, labels)):
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
            refs = _as_ref_list(reference_means[label], "tutorial reported mean")
            for ref_idx, (value, ref_label) in enumerate(refs):
                axs[row, 0].axhline(
                    value,
                    color="red",
                    linestyle="--" if ref_idx == 0 else ":",
                    label=ref_label,
                )

        if reference_stds and label in reference_stds:
            refs = _as_ref_list(reference_stds[label], "tutorial reported sd")
            for ref_idx, (value, ref_label) in enumerate(refs):
                axs[row, 1].axhline(
                    value,
                    color="red",
                    linestyle="--" if ref_idx == 0 else ":",
                    label=ref_label,
                )

        axs[row, 0].set_title(f"{title_prefix}{label}: variational mean")
        axs[row, 1].set_title(f"{title_prefix}{label}: variational std")
        axs[row, 0].set_xlabel("Iteration")
        axs[row, 1].set_xlabel("Iteration")
        axs[row, 0].grid()
        axs[row, 1].grid()
        axs[row, 0].legend()
        axs[row, 1].legend()

    plt.tight_layout()
    plt.show()
