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


# def _logistic_normal_mean_std_mc(mu, sigma, n_samples=10_000, seed=0):
#     mu = float(np.squeeze(mu))
#     sigma = float(np.squeeze(sigma))
#     rng = np.random.default_rng(seed)
#     z = rng.normal(loc=mu, scale=sigma, size=n_samples)
#     theta = expit(z)
#     return float(theta.mean()), float(theta.std(ddof=1))

# def _trace_locscale_to_theta_moments(loc_trace, scale_trace, n_samples=10_000, seed=0):
#     loc_trace = np.asarray(loc_trace, dtype=float)
#     scale_trace = np.asarray(scale_trace, dtype=float)
#     out_mean = np.empty(loc_trace.shape[0], dtype=float)
#     out_std  = np.empty(loc_trace.shape[0], dtype=float)
#     for i in range(loc_trace.shape[0]):
#         out_mean[i], out_std[i] = _logistic_normal_mean_std_mc(
#             loc_trace[i], scale_trace[i], n_samples=n_samples, seed=seed
#         )
#     return out_mean, out_std

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



