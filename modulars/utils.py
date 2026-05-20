# random functions that we don't want to keep copy and pasting
import json
import os
import numpy as np
from tqdm.notebook import tqdm
from tqdm import tqdm as standard_tqdm
from scipy.special import logit, expit


"""
Helper functions for config files
"""
def load_config(config_file="config.json"):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def load_best_values(config=None, transform=None, n_samples=50_000, seed=1):
    if config is None: return None, None
    else:
        best_mu = float(config["best_mean"]) if "best_mean" in config else None
        best_std = float(config["best_std"]) if "best_std" in config else None
        if transform is not None:
            best_mu, best_std = transform(best_mu, best_std, n_samples=n_samples, seed=seed)
        return best_mu, best_std


def check_int_or_float(x):
    if isinstance(x, int) or isinstance(x, float):
        return True
    else:
        return False
    
def check_all_int_or_float_or_1darrays(*args):
    for arg in args:
        if not (check_int_or_float(arg) or (isinstance(arg, np.ndarray) and arg.ndim == 1)):
            return False
    return True

def check_all_covar_shaped_arrays(dim, *args):
    for arg in args:
        if not (isinstance(arg, np.ndarray) and arg.shape == (dim, dim)):
            return False
    return True


"""
Helper functions for push-forward transforms
"""
def exp_lognormal_moments(mu, sigma, **_kwargs):
    """
    Compute E[exp(X)] and Std[exp(X)] for X ~ Normal(mu, sigma).

    Accepts either:
      - scalar-like mu/sigma -> returns (float, float)
      - trajectory-like mu/sigma arrays -> returns (np.ndarray, np.ndarray)
    """
    mu_arr = np.asarray(mu, dtype=float).squeeze()
    sigma_arr = np.asarray(sigma, dtype=float).squeeze()

    if np.any(sigma_arr < 0):
        raise ValueError("sigma must be nonnegative")

    # Scalar mode
    if mu_arr.ndim == 0 and sigma_arr.ndim == 0:
        mean = np.exp(mu_arr + 0.5 * sigma_arr**2)
        var = (np.exp(sigma_arr**2) - 1.0) * np.exp(2.0 * mu_arr + sigma_arr**2)
        return float(mean), float(np.sqrt(var))

    # Trajectory mode
    mu_vec = np.ravel(mu_arr)
    sigma_vec = np.ravel(sigma_arr)
    if mu_vec.shape[0] != sigma_vec.shape[0]:
        raise ValueError("mu and sigma must have the same length in trajectory mode")

    mean = np.exp(mu_vec + 0.5 * sigma_vec**2)
    var = (np.exp(sigma_vec**2) - 1.0) * np.exp(2.0 * mu_vec + sigma_vec**2)
    return mean, np.sqrt(var)


# Backward-compatible wrappers
def pushforward_exp_mean_std(mu, sigma):
    return exp_lognormal_moments(mu, sigma)

def traces_to_lambda_moments(mu_trace, sigma_trace):
    return exp_lognormal_moments(mu_trace, sigma_trace)


def logistic_moments(mu, sigma, n_samples=10_000, seed=0, batch_size=512, ddof=0, **_kwargs):
    """
    Compute moments of theta = sigmoid(Z), Z ~ Normal(mu, sigma).

    Accepts either:
      - scalar-like mu/sigma -> returns (float, float)
      - trajectory-like mu/sigma arrays -> returns (np.ndarray, np.ndarray)
    """
    mu_arr = np.asarray(mu, dtype=float).squeeze()
    sigma_arr = np.asarray(sigma, dtype=float).squeeze()

    if np.any(sigma_arr < 0):
        raise ValueError("sigma must be nonnegative")

    # scalar mode
    if mu_arr.ndim == 0 and sigma_arr.ndim == 0:
        rng = np.random.default_rng(seed)
        z = rng.normal(loc=float(mu_arr), scale=float(sigma_arr), size=int(n_samples))
        th = expit(z)
        return float(th.mean()), float(th.std(ddof=ddof))

    # trajectory mode
    mu_vec = np.ravel(mu_arr)
    sigma_vec = np.ravel(sigma_arr)
    if mu_vec.shape[0] != sigma_vec.shape[0]:
        raise ValueError("mu and sigma must have the same length in trajectory mode")

    T = mu_vec.shape[0]
    means = np.empty(T, dtype=float)
    stds = np.empty(T, dtype=float)

    rng = np.random.default_rng(seed)
    n_samples = int(n_samples)
    batch_size = max(1, int(batch_size))

    for i in range(0, T, batch_size):
        j = min(i + batch_size, T)
        b = j - i
        eps = rng.normal(size=(n_samples, b))
        z = mu_vec[i:j][None, :] + sigma_vec[i:j][None, :] * eps
        th = expit(z)
        means[i:j] = th.mean(axis=0)
        stds[i:j] = th.std(axis=0, ddof=ddof)

    return means, stds

# Optional: keep old names working
def _logistic_normal_mean_std_mc(mu, sigma, n_samples=10_000, seed=0):
    return logistic_moments(mu, sigma, n_samples=n_samples, seed=seed, ddof=1)

def _trace_locscale_to_theta_moments(loc_trace, scale_trace, n_samples=10_000, seed=0):
    return logistic_moments(loc_trace, scale_trace, n_samples=n_samples, seed=seed, ddof=1)


# a function to apply the given push-forward transformation
# then stack the trajectories for easier plotting
def apply_traj_transform(results, transform_fn=None, n_samples=50_000, seed=0, NOTEBOOK=True):
    if NOTEBOOK:
        wrapper = tqdm
    else:
        wrapper = standard_tqdm
    single_means, single_stds = [], []
    multi_means,  multi_stds  = [], []

    if transform_fn is None:
        transform_fn = lambda loc, scale, **kwargs: (loc, scale)
    for single_loc_trace, single_scale_trace, multi_loc_trace, multi_scale_trace in wrapper(results):
        
        # Use vectorized transformation for speed
        single_mu, single_sigma = transform_fn(
            single_loc_trace, single_scale_trace, n_samples=n_samples, seed=seed
        )
        multi_mu, multi_sigma = transform_fn(
            multi_loc_trace, multi_scale_trace, n_samples=n_samples, seed=seed
        )
        
        single_means.append(single_mu)
        single_stds.append(single_sigma)
        multi_means.append(multi_mu)
        multi_stds.append(multi_sigma)
    return stack_trajectories([single_means, single_stds, multi_means, multi_stds])

def apply_traj_transform_multid(results, transform_fn=None, n_samples=50_000, seed=0, NOTEBOOK=True, TFP=False, PYMC=False):
    # if using TFP, we need to square the scale traces to get the variances, since we're using the lower triangular parameterization of the covariance matrix
    # if using PyMC, we need to take the diagonal of the covariance matrix to get the variances, since PyMC's ADVI returns a full covariance matrix even for the diagonal guide
    if NOTEBOOK:
        wrapper = tqdm
    else:
        wrapper = standard_tqdm
    single_means, single_stds = [], []
    multi_means,  multi_stds  = [], []

    if transform_fn is None:
        transform_fn = lambda param1, param2, **kwargs: (param1, param2)
    for single_loc_trace, single_scale_trace, multi_loc_trace, multi_scale_trace in wrapper(results):
        if TFP:
            # bc we're using lower triangular matrices in TFP
            single_scale_trace = np.square(single_scale_trace)
            multi_scale_trace = np.square(multi_scale_trace)
        if PYMC:
            # bc we're using the diagonal of the covariance matrix in PyMC
            single_scale_trace = np.diagonal(single_scale_trace, axis1=-2, axis2=-1)
            multi_scale_trace = np.diagonal(multi_scale_trace, axis1=-2, axis2=-1)
        # Use vectorized transformation for speed
        single_mu, single_sigma = transform_fn(
            single_loc_trace, single_scale_trace, n_samples=n_samples, seed=seed
        )
        multi_mu, multi_sigma = transform_fn(
            multi_loc_trace, multi_scale_trace, n_samples=n_samples, seed=seed
        )
        
        single_means.append(single_mu)
        single_stds.append(single_sigma)
        multi_means.append(multi_mu)
        multi_stds.append(multi_sigma)
    return stack_trajectories_multid([single_means, single_stds, multi_means, multi_stds])


"""
Helper functions to clean up arrays of trajectories
for plotting to be easier
"""
def stack_trajectories(trajectories):
    out_trajectories = []
    for traj in trajectories:
        out_trajectories.append(
            np.stack(
                traj, axis=0
            )
        )
    if len(out_trajectories[0].shape) == 3:
        N, T = out_trajectories[0].shape[0], out_trajectories[0].shape[1]
        for i in range(len(out_trajectories)):
            out_trajectories[i] = out_trajectories[i].reshape(N, T)
    return out_trajectories


def stack_trajectories_multid(trajectories):
    # assumes trajectories is a list of lists of arrays, where the inner arrays are shape (T, D) and the outer list is over runs
    out_trajectories = []
    for traj in trajectories:
        out_trajectories.append(
            np.stack(
                traj, axis=0
            )
        )
    if len(out_trajectories[0].shape) == 4:
        N, T, D = out_trajectories[0].shape[0], out_trajectories[0].shape[1], out_trajectories[0].shape[2]
        for i in range(len(out_trajectories)):
            out_trajectories[i] = out_trajectories[i].reshape(N, T, D)
    return out_trajectories


"""
Helper functions to print out prior, likelihood, and posterior
information for debugging
"""
def print_model_info(
        prior_name, like_name, posterior_name,
        prior_params, like_params,
        data=None, posterior_func=None, n_samples=None):
    print(f"Prior: {prior_name}({prior_params})")
    print(f"Likelihood: {like_name}({like_params})")
    if data is not None:
        print(f"Data: {data}")
        if n_samples is None:
            n_samples = len(data)
        if posterior_func is not None:
            posterior_info = posterior_func(data, n_samples, *prior_params, *like_params)
            print(f"Posterior: {posterior_name}({posterior_info})")
    else:
        print("Data: None, posterior is just the prior")

# less tested stuff below

def load_best_multid_reference(config=None, fallback_mean=None, fallback_cov=None):
    if config is None:
        best_mean = fallback_mean
        best_cov = fallback_cov
    else:
        best_mean = config.get("best_mean", fallback_mean)
        best_cov = config.get("best_cov", fallback_cov)

    if best_mean is None or best_cov is None:
        raise ValueError("we need either best_mean/best_cov in config or fallback values")

    best_mean = np.asarray(best_mean, dtype=float)
    best_cov = np.asarray(best_cov, dtype=float)
    best_std = np.sqrt(np.clip(np.diag(best_cov), 0.0, None))
    return best_mean, best_cov, best_std

"""
Saving and Loading helper functions
"""
def _encode_for_csv(obj):
    """
    we recursively encode python / numpy objects into a json-safe form
    so we can store each random restart exactly and reconstruct it later.
    """
    if hasattr(obj, "numpy"):
        obj = obj.numpy()

    if hasattr(obj, "__array__") and not isinstance(obj, np.ndarray):
        obj = np.asarray(obj)

    if isinstance(obj, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(obj.dtype),
            "value": obj.tolist(),
        }

    if isinstance(obj, tuple):
        return {
            "__type__": "tuple",
            "value": [_encode_for_csv(x) for x in obj],
        }

    if isinstance(obj, list):
        return {
            "__type__": "list",
            "value": [_encode_for_csv(x) for x in obj],
        }

    if isinstance(obj, dict):
        return {
            "__type__": "dict",
            "value": {str(k): _encode_for_csv(v) for k, v in obj.items()},
        }

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return {"__type__": "nan"}
        if np.isposinf(obj):
            return {"__type__": "posinf"}
        if np.isneginf(obj):
            return {"__type__": "neginf"}
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    raise TypeError(f"we do not know how to encode objects of type {type(obj)}")


def _decode_from_csv(obj):
    """
    we invert _encode_for_csv so we get back the same nested structure
    we saved in the csv.
    """
    if isinstance(obj, dict):
        obj_type = obj.get("__type__", None)

        if obj_type == "ndarray":
            return np.array(obj["value"], dtype=np.dtype(obj["dtype"]))

        if obj_type == "tuple":
            return tuple(_decode_from_csv(x) for x in obj["value"])

        if obj_type == "list":
            return [_decode_from_csv(x) for x in obj["value"]]

        if obj_type == "dict":
            return {k: _decode_from_csv(v) for k, v in obj["value"].items()}

        if obj_type == "nan":
            return float("nan")

        if obj_type == "posinf":
            return float("inf")

        if obj_type == "neginf":
            return float("-inf")

    return obj


def save_to_csv(file_name, results):
    """
    we save one random restart per csv row.

    each row contains:
    - restart_idx
    - payload_json

    this keeps the save/load format generic while still using pandas + csv.
    """
    import pandas as pd

    rows = []
    for restart_idx, result in enumerate(results):
        rows.append(
            {
                "restart_idx": restart_idx,
                "payload_json": json.dumps(_encode_for_csv(result)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(file_name, index=False)


def load_from_csv(file_name):
    """
    we load the csv written by save_to_csv and reconstruct the original
    python / numpy structure in the same restart order.
    """
    import pandas as pd

    df = pd.read_csv(file_name)
    df = df.sort_values("restart_idx")

    results = []
    for _, row in df.iterrows():
        payload = json.loads(row["payload_json"])
        results.append(_decode_from_csv(payload))

    return results


def _extract_mean_cov_result(run):
    """
    we normalize a loaded run into a dict with mean / cov so that
    find_best_params can work across a few related save formats.
    """
    if isinstance(run, dict):
        if "mean" in run and "cov" in run:
            return {
                "mean": np.asarray(run["mean"], dtype=float),
                "cov": np.asarray(run["cov"], dtype=float),
                "final_elbo": run.get("final_elbo", run.get("elbo", None)),
                "raw": run,
            }

    if isinstance(run, (tuple, list)) and len(run) == 2:
        first, second = run
        if isinstance(second, dict) and "mean" in second and "cov" in second:
            return {
                "mean": np.asarray(second["mean"], dtype=float),
                "cov": np.asarray(second["cov"], dtype=float),
                "final_elbo": first,
                "raw": run,
            }

    raise ValueError(
        "we expected each saved run to be either "
        "a dict with mean/cov or a (final_elbo, tracker_dict) pair"
    )


def find_best_params(
        file_name_list,
        alpha_prior,
        obs_counts,
        grad_samps,
        seed,
        true_concentration_scale=None,
        model_str="multidirich",
        guide_str="stickbreak_mvn",
        param_name="theta",
        score_fn=None,
        result_specs=None,
        processed_mc_settings=None,
        score_seed_by_restart=True):
    """
    we load every saved run from every file, score each one with compute_elbo
    or a caller-provided score_fn, and return the best mean / cov along with
    metadata. With a custom score_fn, this also understands the compact
    final-summary payload shape (final_means, final_stds) and the processed
    restart payload shape (single_means, single_stds, multi_means, multi_stds).
    """
    from modulars.elbo_computations import compute_elbo

    if alpha_prior is not None:
        alpha_prior = np.asarray(alpha_prior, dtype=float)
    if score_fn is None:
        obs_counts = np.asarray(obs_counts)

    if true_concentration_scale is None and alpha_prior is not None:
        true_concentration_scale = float(np.sum(alpha_prior) + np.sum(obs_counts))

    all_scored_runs = []

    if result_specs is None:
        result_specs = [(None, file_name) for file_name in file_name_list]
    if processed_mc_settings is not None:
        processed_mc_settings = set(processed_mc_settings)

    for method, file_name in result_specs:
        loaded_runs = load_from_csv(file_name)

        final_summary_payload = (
            score_fn is not None
            and len(loaded_runs) == 1
            and isinstance(loaded_runs[0], (tuple, list))
            and len(loaded_runs[0]) == 2
            and not (
                isinstance(loaded_runs[0][1], dict)
                and "mean" in loaded_runs[0][1]
                and "cov" in loaded_runs[0][1]
            )
        )
        processed_restart_payload = (
            score_fn is not None
            and len(loaded_runs) == 1
            and isinstance(loaded_runs[0], (tuple, list))
            and len(loaded_runs[0]) == 4
        )

        if final_summary_payload:
            final_means, final_stds = loaded_runs[0]
            final_means = np.asarray(final_means, dtype=float)
            final_stds = np.asarray(final_stds, dtype=float)
            if final_means.ndim == 1:
                final_means = final_means[None, :]
                final_stds = final_stds[None, :]
            if final_means.shape != final_stds.shape:
                raise ValueError(
                    f"final mean/std shape mismatch in {file_name}: "
                    f"{final_means.shape} vs {final_stds.shape}"
                )

            run_iter = []
            for local_restart_idx in range(final_means.shape[0]):
                mean = np.asarray(final_means[local_restart_idx, :], dtype=float)
                std = np.asarray(final_stds[local_restart_idx, :], dtype=float)
                run_iter.append(
                    {
                        "restart_idx": local_restart_idx,
                        "mc_setting": None,
                        "mean": mean,
                        "cov": np.diag(np.square(std)),
                        "std": std,
                        "final_elbo": None,
                        "n_records": 1,
                        "raw": {
                            "mean": mean,
                            "std": std,
                            "source_kind": "final_summary",
                        },
                    }
                )
        elif processed_restart_payload:
            single_means, single_stds, multi_means, multi_stds = loaded_runs[0]
            run_iter = []
            for mc_setting, means, stds in (
                ("1_mc", single_means, single_stds),
                ("100_mc", multi_means, multi_stds),
            ):
                if processed_mc_settings is not None and mc_setting not in processed_mc_settings:
                    continue
                means = np.asarray(means, dtype=float)
                stds = np.asarray(stds, dtype=float)
                for local_restart_idx in range(means.shape[0]):
                    mean = np.asarray(means[local_restart_idx, -1, :], dtype=float)
                    std = np.asarray(stds[local_restart_idx, -1, :], dtype=float)
                    run_iter.append(
                        {
                            "restart_idx": local_restart_idx,
                            "mc_setting": mc_setting,
                            "mean": mean,
                            "cov": np.diag(np.square(std)),
                            "std": std,
                            "final_elbo": None,
                            "n_records": int(means.shape[1]),
                            "raw": {
                                "mean": mean,
                                "std": std,
                                "mc_setting": mc_setting,
                                "source_kind": "processed_restart_trace",
                            },
                        }
                    )
        else:
            run_iter = []
            for local_restart_idx, run in enumerate(loaded_runs):
                extracted = _extract_mean_cov_result(run)
                cov = np.asarray(extracted["cov"], dtype=float)
                run_iter.append(
                    {
                        "restart_idx": local_restart_idx,
                        "mc_setting": None,
                        "mean": np.asarray(extracted["mean"], dtype=float),
                        "cov": cov,
                        "std": np.sqrt(np.clip(np.diag(cov), 0.0, None)),
                        "final_elbo": extracted["final_elbo"],
                        "n_records": None,
                        "raw": extracted["raw"],
                    }
                )

        for candidate in run_iter:
            local_restart_idx = candidate["restart_idx"]

            mean = np.asarray(candidate["mean"], dtype=float)
            cov = np.asarray(candidate["cov"], dtype=float)
            std = np.asarray(candidate["std"], dtype=float)
            estimated_alpha = (
                mean * true_concentration_scale
                if true_concentration_scale is not None
                else None
            )

            if score_fn is None:
                numpyro_elbo = compute_elbo(
                    model_str=model_str,
                    guide_str=guide_str,
                    param_name=param_name,
                    model_vals=[alpha_prior, np.sum(obs_counts)],
                    guide_vals=[mean, cov],
                    data=obs_counts,
                    with_data=True,
                    grad_samps=grad_samps,
                    seed=seed,
                )
            else:
                score_seed = seed + local_restart_idx if score_seed_by_restart else seed
                numpyro_elbo = score_fn(
                    mean=mean,
                    cov=cov,
                    std=std,
                    data=obs_counts,
                    grad_samps=grad_samps,
                    seed=score_seed,
                    source_file=file_name,
                    restart_idx=local_restart_idx,
                    method=method,
                    mc_setting=candidate["mc_setting"],
                    n_records=candidate["n_records"],
                )

            scored_run = {
                "source_file": file_name,
                "restart_idx": local_restart_idx,
                "mean": mean,
                "cov": cov,
                "std": std,
                "estimated_alpha": estimated_alpha,
                "best_elbo_np": float(numpyro_elbo),
                "final_elbo": candidate["final_elbo"],
                "n_records": candidate["n_records"],
                "raw": candidate["raw"],
            }
            if method is not None:
                scored_run["method"] = method
            if candidate["mc_setting"] is not None:
                scored_run["mc_setting"] = candidate["mc_setting"]
            all_scored_runs.append(scored_run)

    if len(all_scored_runs) == 0:
        raise ValueError("we did not find any runs in the provided csv files")

    best_candidate_idx = max(
        range(len(all_scored_runs)),
        key=lambda idx: all_scored_runs[idx]["best_elbo_np"],
    )
    best_run = all_scored_runs[best_candidate_idx]

    return {
        "best_candidate_idx": best_candidate_idx,
        "best_mean": best_run["mean"],
        "best_cov": best_run["cov"],
        "best_std": best_run["std"],
        "estimated_alpha": best_run["estimated_alpha"],
        "best_elbo_np": best_run["best_elbo_np"],
        "final_elbo": best_run["final_elbo"],
        "source_file": best_run["source_file"],
        "restart_idx": best_run["restart_idx"],
        "method": best_run.get("method", None),
        "mc_setting": best_run.get("mc_setting", None),
        "raw_best_run": best_run["raw"],
        "all_scored_runs": all_scored_runs,
    }

def save_best_run_to_config(config_file, best_info, extra_updates=None):
    """
    we save the cross-framework best variational approximation back to the config file
    so all later plots can use the same shared reference.
    """
    config_file = str(config_file)
    config = load_config(config_file)

    config["best_mean"] = np.asarray(best_info["best_mean"], dtype=float).tolist()
    config["best_cov"] = np.asarray(best_info["best_cov"], dtype=float).tolist()
    config["best_std"] = np.asarray(best_info["best_std"], dtype=float).tolist()
    config["best_elbo_np"] = float(best_info["best_elbo_np"])
    config["best_source_file"] = str(best_info["source_file"])
    config["best_restart_idx"] = int(best_info["restart_idx"])

    if best_info.get("final_elbo", None) is not None:
        config["best_final_elbo"] = float(best_info["final_elbo"])

    if extra_updates is not None:
        for key, value in extra_updates.items():
            config[key] = value

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

# less tested stuff below
def anchored_softmax_simplex_forward(z, n_cats=None):
    """
    we map latent coordinates in R^(K-1) to the K-simplex using
    logits = [z, -sum(z)] followed by softmax.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[None, :]

    logits = np.concatenate(
        [z, -np.sum(z, axis=-1, keepdims=True)],
        axis=-1,
    )
    if n_cats is not None and logits.shape[-1] != n_cats:
        raise ValueError(
            f"we expected {n_cats} categories after the anchored softmax transform"
        )
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.sum(weights, axis=-1, keepdims=True)



def stickbreaking_simplex_forward(z, n_cats=None):
    """
    we map latent coordinates in R^(K-1) to the K-simplex using a
    stick-breaking construction with the usual centering offsets.
    """
    z = np.asarray(z, dtype=float)

    if z.ndim == 1:
        z = z[None, :]

    n_samples, latent_dim = z.shape

    if n_cats is None:
        n_cats = latent_dim + 1

    if latent_dim != n_cats - 1:
        raise ValueError(
            f"we expected latent dimension {n_cats - 1}, but got {latent_dim}"
        )

    remaining = np.ones(n_samples, dtype=float)
    theta = np.zeros((n_samples, n_cats), dtype=float)

    for k in range(latent_dim):
        offset = np.log(n_cats - (k + 1))
        frac = expit(z[:, k] - offset)
        theta[:, k] = remaining * frac
        remaining = remaining * (1.0 - frac)

    theta[:, -1] = remaining
    return theta

def iterated_sigmoid_centered_forward(z, n_cats=None):
    """
    we map latent coordinates to the simplex using TFP's
    IteratedSigmoidCentered bijector.
    """
    import tensorflow_probability as tfp

    z = np.asarray(z, dtype=np.float32)
    if z.ndim == 1:
        z = z[None, :]

    bij = tfp.bijectors.IteratedSigmoidCentered()
    theta = np.asarray(bij.forward(z), dtype=float)

    if n_cats is not None and theta.shape[-1] != n_cats:
        raise ValueError(
            f"we expected {n_cats} categories after IteratedSigmoidCentered, got {theta.shape[-1]}"
        )

    return theta


def simplex_moments(
        loc,
        scale,
        n_cats,
        forward_fn=stickbreaking_simplex_forward,
        n_samples=10_000,
        seed=0,
        ddof=0,
        **_kwargs):
    """
    we compute marginal means and stds on the simplex for each iteration
    using the chosen forward transform.
    """
    loc = np.asarray(loc, dtype=float)
    scale = np.asarray(scale, dtype=float)

    squeeze = False
    if loc.ndim == 1:
        loc = loc[None, :]
        squeeze = True

    rng = np.random.default_rng(seed)
    T, latent_dim = loc.shape

    means = np.empty((T, n_cats), dtype=float)
    stds = np.empty((T, n_cats), dtype=float)

    for t in range(T):
        if scale.ndim == 2 and scale.shape == loc.shape:
            z = rng.normal(loc=loc[t], scale=scale[t], size=(int(n_samples), latent_dim))
        elif scale.ndim == 2 and scale.shape == (latent_dim, latent_dim):
            z = rng.multivariate_normal(mean=loc[t], cov=scale, size=int(n_samples))
        elif scale.ndim == 3 and scale.shape == (T, latent_dim, latent_dim):
            z = rng.multivariate_normal(mean=loc[t], cov=scale[t], size=int(n_samples))
        else:
            raise ValueError(
                "we expected scale to have shape (T, D), (D, D), or (T, D, D)"
            )

        theta = forward_fn(z, n_cats=n_cats)
        means[t] = theta.mean(axis=0)
        stds[t] = theta.std(axis=0, ddof=ddof)

    if squeeze:
        return means[0], stds[0]
    return means, stds


def apply_traj_transform_simplex(
        results,
        n_cats,
        forward_fn=stickbreaking_simplex_forward,
        n_samples=10_000,
        seed=0,
        NOTEBOOK=True):
    wrapper = tqdm if NOTEBOOK else standard_tqdm
    single_means, single_stds = [], []
    multi_means, multi_stds = [], []

    for single_loc, single_scale, multi_loc, multi_scale in wrapper(results):
        s_mean, s_std = simplex_moments(
            single_loc,
            single_scale,
            n_cats=n_cats,
            forward_fn=forward_fn,
            n_samples=n_samples,
            seed=seed,
        )
        m_mean, m_std = simplex_moments(
            multi_loc,
            multi_scale,
            n_cats=n_cats,
            forward_fn=forward_fn,
            n_samples=n_samples,
            seed=seed,
        )

        single_means.append(s_mean)
        single_stds.append(s_std)
        multi_means.append(m_mean)
        multi_stds.append(m_std)

    return stack_trajectories_multid([single_means, single_stds, multi_means, multi_stds])

def simplex_moments_from_cov(
        loc,
        cov,
        n_cats=None,
        forward_fn=stickbreaking_simplex_forward,
        n_samples=10_000,
        seed=0,
        ddof=0):
    """
    we compute simplex marginal means and stds from latent gaussian
    mean/cov trajectories using the chosen forward transform.
    """
    loc = np.asarray(loc, dtype=float)
    cov = np.asarray(cov, dtype=float)

    squeeze = False
    if loc.ndim == 1:
        loc = loc[None, :]
        squeeze = True

    if loc.ndim != 2:
        raise ValueError("we expected loc to have shape (T, D) or (D,)")

    T, D = loc.shape

    if n_cats is None:
        n_cats = D + 1

    rng = np.random.default_rng(seed)
    means = np.empty((T, n_cats), dtype=float)
    stds = np.empty((T, n_cats), dtype=float)

    for t in range(T):
        if cov.ndim == 3 and cov.shape == (T, D, D):
            z = rng.multivariate_normal(mean=loc[t], cov=cov[t], size=int(n_samples))
        elif cov.ndim == 2 and cov.shape == (D, D):
            z = rng.multivariate_normal(mean=loc[t], cov=cov, size=int(n_samples))
        else:
            raise ValueError("we expected cov to have shape (T, D, D) or (D, D)")

        theta = forward_fn(z, n_cats=n_cats)
        means[t] = theta.mean(axis=0)
        stds[t] = theta.std(axis=0, ddof=ddof)

    if squeeze:
        return means[0], stds[0]
    return means, stds


def apply_traj_transform_simplex_from_cov(
        results,
        n_cats=None,
        forward_fn=anchored_softmax_simplex_forward,
        n_samples=10_000,
        seed=0,
        NOTEBOOK=True):
    wrapper = tqdm if NOTEBOOK else standard_tqdm
    single_means, single_stds = [], []
    multi_means, multi_stds = [], []

    for single_loc, single_cov, multi_loc, multi_cov in wrapper(results):
        s_mean, s_std = simplex_moments_from_cov(
            single_loc,
            single_cov,
            n_cats=n_cats,
            forward_fn=forward_fn,
            n_samples=n_samples,
            seed=seed,
        )
        m_mean, m_std = simplex_moments_from_cov(
            multi_loc,
            multi_cov,
            n_cats=n_cats,
            forward_fn=forward_fn,
            n_samples=n_samples,
            seed=seed,
        )

        single_means.append(s_mean)
        single_stds.append(s_std)
        multi_means.append(m_mean)
        multi_stds.append(m_std)

    return stack_trajectories_multid([single_means, single_stds, multi_means, multi_stds])

# summary builder for saving
def build_summary_runs_from_scale_results(
        results,
        n_cats,
        forward_fn=stickbreaking_simplex_forward,
        which="multi",
        n_samples=10_000,
        seed=0):
    summary_runs = []

    for restart_idx, run in enumerate(results):
        single_loc, single_scale, multi_loc, multi_scale = run

        if which == "single":
            final_loc = np.asarray(single_loc[-1], dtype=float)
            final_scale = np.asarray(single_scale[-1], dtype=float)
        else:
            final_loc = np.asarray(multi_loc[-1], dtype=float)
            final_scale = np.asarray(multi_scale[-1], dtype=float)

        rng = np.random.default_rng(seed + restart_idx)
        z = rng.normal(
            loc=final_loc,
            scale=final_scale,
            size=(int(n_samples), final_loc.shape[0]),
        )
        theta = forward_fn(z, n_cats=n_cats)

        summary_runs.append(
            {
                "restart_idx": restart_idx,
                "which": which,
                "mean": theta.mean(axis=0),
                "cov": np.cov(theta, rowvar=False),
            }
        )

    return summary_runs


def build_summary_runs_from_cov_results(
        results,
        n_cats,
        forward_fn=anchored_softmax_simplex_forward,
        which="multi",
        n_samples=10_000,
        seed=0):
    summary_runs = []

    for restart_idx, run in enumerate(results):
        single_loc, single_cov, multi_loc, multi_cov = run

        if which == "single":
            final_loc = np.asarray(single_loc[-1], dtype=float)
            final_cov = np.asarray(single_cov[-1], dtype=float)
        else:
            final_loc = np.asarray(multi_loc[-1], dtype=float)
            final_cov = np.asarray(multi_cov[-1], dtype=float)

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
                "which": which,
                "mean": theta.mean(axis=0),
                "cov": np.cov(theta, rowvar=False),
            }
        )

    return summary_runs


def anchored_simplex_inverse(theta):
    """
    we map simplex points back to PyMC's anchored coordinates.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[None, :]

    log_theta = np.log(theta)
    shift = log_theta.mean(axis=-1, keepdims=True)
    return log_theta[..., :-1] - shift

def stickbreaking_simplex_inverse(theta):
    """
    we map simplex points back to centered stick-breaking coordinates.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[None, :]

    y_main = theta[..., :-1]
    prev_sum = np.concatenate(
        [
            np.zeros_like(y_main[..., :1]),
            np.cumsum(y_main, axis=-1)[..., :-1],
        ],
        axis=-1,
    )

    z = y_main / (1.0 - prev_sum)
    z = np.clip(z, 1e-12, 1.0 - 1e-12)

    latent_dim = theta.shape[-1] - 1
    offset = np.log(np.arange(latent_dim, 0, -1, dtype=float))

    return np.log(z) - np.log1p(-z) + offset


def offdiag_ratio(corr):
    corr = np.asarray(corr, dtype=float)
    offdiag = corr - np.eye(corr.shape[0])
    return np.linalg.norm(offdiag, ord="fro") / np.sqrt(corr.shape[0] * (corr.shape[0] - 1))


def latent_diag_diagnostic(theta_draws, title_prefix):
    z_anchored = anchored_simplex_inverse(theta_draws)
    z_stick = stickbreaking_simplex_inverse(theta_draws)

    corr_anchored = np.corrcoef(z_anchored, rowvar=False)
    corr_stick = np.corrcoef(z_stick, rowvar=False)

    anchored_recon = anchored_softmax_simplex_forward(z_anchored, n_cats=n_cats)
    stick_recon = stickbreaking_simplex_forward(z_stick, n_cats=n_cats)

    ref_std = theta_draws.std(axis=0)
    anchored_std = anchored_recon.std(axis=0)
    stick_std = stick_recon.std(axis=0)

    print(title_prefix)
    print("anchored latent offdiag ratio:", offdiag_ratio(corr_anchored))
    print("stick latent offdiag ratio:", offdiag_ratio(corr_stick))
    print("anchored total reconstruction error:", np.linalg.norm(anchored_recon.mean(axis=0) - theta_draws.mean(axis=0)) + np.linalg.norm(anchored_std - ref_std))
    print("stick total reconstruction error:", np.linalg.norm(stick_recon.mean(axis=0) - theta_draws.mean(axis=0)) + np.linalg.norm(stick_std - ref_std))
    print("anchored std error:", np.linalg.norm(anchored_std - ref_std))
    print("stick std error:", np.linalg.norm(stick_std - ref_std))
    print()
    print("anchored recon std:", anchored_std)
    print("stick recon std:", stick_std)
    print("reference std:", ref_std)
    print()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    im0 = axes[0].imshow(corr_anchored, vmin=-1, vmax=1, cmap="coolwarm")
    axes[0].set_title(title_prefix + " Anchored")
    axes[0].set_xticks(range(corr_anchored.shape[0]))
    axes[0].set_yticks(range(corr_anchored.shape[0]))

    im1 = axes[1].imshow(corr_stick, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_title(title_prefix + " Stick-Breaking")
    axes[1].set_xticks(range(corr_stick.shape[0]))
    axes[1].set_yticks(range(corr_stick.shape[0]))

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)
    plt.show()
