import gc
import os

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor
pytensor.config.cxx = '/usr/bin/clang++'


from pymc.logprob.transforms import Transform


class _ThinnedTracker:
    """Record PyMC ADVI stats every `track_every` iterations."""

    def __init__(self, track_every=1, n_iters=None, dtype=np.float32, **kwargs):
        self.track_every = max(1, int(track_every))
        self.n_iters = None if n_iters is None else int(n_iters)
        self.n_seen = 0
        self.dtype = dtype
        self.whatchdict = kwargs
        self.hist = {key: [] for key in kwargs}

    def __call__(self, approx, hist, i):
        self.n_seen += 1
        if self.n_seen % self.track_every != 0 and self.n_seen != self.n_iters:
            return
        for key, fn in self.whatchdict.items():
            try:
                res = fn()
            except Exception:
                res = fn(approx, hist, i)
            self.hist[key].append(np.asarray(res, dtype=self.dtype).copy())

    def __getitem__(self, item):
        return self.hist[item]

    def array(self, item):
        return np.asarray(self.hist[item], dtype=self.dtype)



"""
Random restart code for running pymc VI
-- note that this does require the model to 
be created externally and passed in as an argument,
so that we can run the same model with different random restarts
"""

def run_pymc_VI(
        model,
        n_mc_samples=1,
        n_iters=100_000,
        optimizer='default',
        seed=0,
        SINGLE_DIM=True,
        track_every=1,
        track_cov=True,
        tracker_dtype=np.float32,
        **kwargs):
    """
    Run pymc VI with the given model and return the mean and std trajectories

    optimizer can be 'default' (pymc's default) or 'adam' (pymc's adam)
    """
    start = kwargs.pop("start", None)
    start_sigma = kwargs.pop("start_sigma", None)
    np.random.seed(seed)
    advi = pm.ADVI(model=model, random_seed=seed, start=start, start_sigma=start_sigma)

    if SINGLE_DIM:
        tracker = _ThinnedTracker(
            track_every=track_every,
            n_iters=n_iters,
            dtype=tracker_dtype,
            mean=advi.approx.mean.eval,
            std=advi.approx.std.eval,
        )
    else:
        scale_name = "cov" if track_cov else "std"
        scale_fn = advi.approx.cov.eval if track_cov else advi.approx.std.eval
        tracker = _ThinnedTracker(
            track_every=track_every,
            n_iters=n_iters,
            dtype=tracker_dtype,
            mean=advi.approx.mean.eval,
            **{scale_name: scale_fn},
        )

    if optimizer == 'default':
            approx = advi.fit(
                n_iters, callbacks=[tracker],
                progressbar=False,
                obj_n_mc=n_mc_samples
            )
    else:
        approx = advi.fit(
            n_iters, callbacks=[tracker],
            progressbar=False,
            obj_n_mc=n_mc_samples,
            obj_optimizer=pm.adam()
        )
    mean_traj = tracker.array('mean')
    scale_traj = tracker.array('std' if SINGLE_DIM else scale_name)
    del approx, advi, tracker
    gc.collect()
    return mean_traj, scale_traj


def run_pymc_fullrank_VI(
        model,
        n_mc_samples=1,
        n_iters=100_000,
        optimizer='default',
        seed=0,
        SINGLE_DIM=False,
        track_every=1,
        tracker_dtype=np.float32,
        **kwargs):
    """
    Run PyMC full-rank ADVI and return the mean and covariance trajectories.
    """
    del SINGLE_DIM, kwargs
    np.random.seed(seed)
    advi = pm.FullRankADVI(model=model, random_seed=seed)

    tracker = _ThinnedTracker(
        track_every=track_every,
        n_iters=n_iters,
        dtype=tracker_dtype,
        mean=advi.approx.mean.eval,
        cov=advi.approx.cov.eval,
    )

    if optimizer == 'default':
        approx = advi.fit(
            n_iters, callbacks=[tracker],
            progressbar=False,
            obj_n_mc=n_mc_samples
        )
    else:
        approx = advi.fit(
            n_iters, callbacks=[tracker],
            progressbar=False,
            obj_n_mc=n_mc_samples,
            obj_optimizer=pm.adam()
        )

    mean_traj = tracker.array('mean')
    cov_traj = tracker.array('cov')
    del approx, advi, tracker
    gc.collect()
    return mean_traj, cov_traj


def fit_pymc_fullrank_covariance(model, n_mc_samples=100, n_iters=100_000, optimizer='default', seed=0):
    """
    Fit a single full-rank ADVI run and return the final mean and covariance.
    """
    mean_traj, cov_traj = run_pymc_fullrank_VI(
        model,
        n_mc_samples=n_mc_samples,
        n_iters=n_iters,
        optimizer=optimizer,
        seed=seed,
    )
    return np.asarray(mean_traj)[-1], np.asarray(cov_traj)[-1]

# runs regular and ADAM VI for a given model and given seed and returns trajectory
def run_single_seed_pymc_VI(seed, run_model_fn, **run_model_kwargs):
    np.random.seed(seed)
    single_means, single_stds = run_model_fn(
        n_mc_samples=1, seed=seed, optimizer='default', **run_model_kwargs
    )
    multi_means, multi_stds = run_model_fn(
        n_mc_samples=100, seed=seed + 1000, optimizer='default', **run_model_kwargs
    )
    adam_single_means, adam_single_stds = run_model_fn(
        n_mc_samples=1, seed=seed, optimizer='adam', **run_model_kwargs
    )
    adam_multi_means, adam_multi_stds = run_model_fn(
        n_mc_samples=100, seed=seed + 1000, optimizer='adam', **run_model_kwargs
    )
    return (
        [single_means, single_stds, multi_means, multi_stds],
        [adam_single_means, adam_single_stds, adam_multi_means, adam_multi_stds],
    )


def run_single_seed_pymc_fullrank_VI(seed, run_model_fn, **run_model_kwargs):
    np.random.seed(seed)
    single_means, single_covs = run_model_fn(
        n_mc_samples=1, seed=seed, optimizer='default', **run_model_kwargs
    )
    multi_means, multi_covs = run_model_fn(
        n_mc_samples=100, seed=seed + 1000, optimizer='default', **run_model_kwargs
    )
    adam_single_means, adam_single_covs = run_model_fn(
        n_mc_samples=1, seed=seed, optimizer='adam', **run_model_kwargs
    )
    adam_multi_means, adam_multi_covs = run_model_fn(
        n_mc_samples=100, seed=seed + 1000, optimizer='adam', **run_model_kwargs
    )
    return (
        [single_means, single_covs, multi_means, multi_covs],
        [adam_single_means, adam_single_covs, adam_multi_means, adam_multi_covs],
    )



"""
The following helpers are for changing the simplex transform of pymc
to be the same stick-breaking transform used by TFP and NumPyro, so we can
understand whether differences in the transforms are driving differences in the results.
"""
# change_value_transforms allows us to change the transformation being used by pymc


class StickBreakingSimplexTransform(Transform):
    """
    we replace PyMC's default simplex value transform with the centered
    stick-breaking transform used by NumPyro / TFP.
    """

    name = "stickbreaking_simplex"

    def forward(self, value, *inputs):
        """
        we map simplex coordinates to unconstrained stick-breaking coordinates.
        """
        y = value
        y_main = y[..., :-1]

        prev_sum = pt.concatenate(
            [
                pt.zeros_like(y_main[..., :1]),
                pt.cumsum(y_main, axis=-1)[..., :-1],
            ],
            axis=-1,
        )

        z = pt.clip(
            y_main / (1.0 - prev_sum),
            1e-12,
            1.0 - 1e-12,
        )

        latent_dim = y.shape[-1] - 1
        offset = pt.log(pt.arange(latent_dim, 0, -1)).astype(y.dtype)

        return pt.log(z) - pt.log1p(-z) + offset

    def backward(self, value, *inputs):
        """
        we map unconstrained stick-breaking coordinates back to the simplex.
        """
        x = value

        latent_dim = x.shape[-1]
        offset = pt.log(pt.arange(latent_dim, 0, -1)).astype(x.dtype)
        x_shifted = x - offset

        z = pt.sigmoid(x_shifted)

        stick_segments = pt.concatenate(
            [
                pt.ones_like(z[..., :1]),
                pt.extra_ops.cumprod(1.0 - z, axis=-1),
            ],
            axis=-1,
        )

        z_padded = pt.concatenate(
            [
                z,
                pt.ones_like(z[..., :1]),
            ],
            axis=-1,
        )

        return z_padded * stick_segments

    def log_jac_det(self, value, *inputs):
        """
        we compute the unconstrained-to-simplex log absolute Jacobian determinant.
        """
        x = value
        y = self.backward(x, *inputs)

        latent_dim = x.shape[-1]
        offset = pt.log(pt.arange(latent_dim, 0, -1)).astype(x.dtype)
        x_shifted = x - offset

        return pt.sum(
            pt.log(y[..., :-1]) + pt.log(pt.sigmoid(x_shifted)) - x_shifted,
            axis=-1,
        )

"""
Specific helper functions for 1d_gaussian_no_obs_advi_optimizer_space notebook, which is where we are doing the most detailed comparisons between pymc and our own implementations, so we want to make sure we have the exact same posteriors and gradients being computed for the comparisons to be as clear as possible.
"""

from modulars.distributions import rho_from_std, exact_kl

def run_actual_pymc_fit(make_model, n_iters, n_mc, optimizer_name, seed=42, mu_p=0.0, sigma_p=1.0):
    with make_model() as model:
        advi = pm.ADVI(model=model, random_seed=seed)
        tracker = pm.callbacks.Tracker(mean=advi.approx.mean.eval, std=advi.approx.std.eval)
        fit_kwargs = {
            'n': n_iters,
            'callbacks': [tracker],
            'progressbar': False,
            'obj_n_mc': n_mc
        }
        if optimizer_name == 'adam':
            fit_kwargs['obj_optimizer'] = pm.adam()
        advi.fit(**fit_kwargs)
    mean = np.asarray(tracker['mean'], dtype=float).reshape(-1)
    std = np.asarray(tracker['std'], dtype=float).reshape(-1)
    rho = rho_from_std(std)
    kl = exact_kl(mean, rho, mu_p, sigma_p) # kl in ADVI's rho parameterization
    return {'mean': mean, 'std': std, 'rho': rho, 'kl': kl}

def run_actual_suite(make_model, n_iters, RUN_SPECS, seed=42, mu_p=0.0, sigma_p=1.0):
    return {
        spec['name']: run_actual_pymc_fit(make_model, n_iters, spec['n_mc'], spec['optimizer_name'], seed=seed, mu_p=mu_p, sigma_p=sigma_p)
        for spec in RUN_SPECS
    }

def print_actual_summary(actual_runs, RUN_SPECS):
    for spec in RUN_SPECS:
        name = spec['name']
        run = actual_runs[name]
        print(name, '| final mean =', run['mean'][-1], '| final std =', run['std'][-1], '| final KL =', run['kl'][-1])

"""
Helper functions for 1d_gaussian_no_obs_advi_optimizer_space notebook, which is where we are doing the most detailed comparisons between pymc and our own implementations, so we want to make sure we have the exact same posteriors and gradients being computed for the comparisons to be as clear as possible.
"""


def compile_pymc_ADVI_update(make_model, n_mc, optimizer_name, seed=42):
    """A PyMC ADVI update function that exposes optimizer internals.
    Created by copying pymc's source code and modifying to return
    internal parameters.
    This function returns the current parameters, gradients, and optimizer-state values before the next update.
    """
    from collections import OrderedDict

    with make_model() as model:
        advi = pm.ADVI(model=model, random_seed=seed)
        obj = advi.objective(n_mc)
        params = advi.approx.params
        grads = pm.updates.get_or_compute_grads(obj, params)
        updates = OrderedDict()
        outputs = []

        if optimizer_name == 'adagrad_window':
            for param, grad in zip(params, grads):
                i = pytensor.shared(pm.floatX(0))
                i_int = i.astype('int32')
                value = param.get_value(borrow=True)
                accu = pytensor.shared(np.zeros((*value.shape, 10), dtype=value.dtype))
                accu_new = pt.set_subtensor(accu[..., i_int], grad**2)
                i_new = pt.switch((i + 1) < 10, i + 1, 0)
                window_sum = accu_new.sum(axis=-1)
                denom = pt.sqrt(window_sum + 0.1)
                step = 0.001 * grad / denom
                outputs.extend([param, grad, window_sum, denom, step])
                updates[accu] = accu_new
                updates[i] = i_new
                updates[param] = param - step
        elif optimizer_name == 'adam':
            t_prev = pytensor.shared(pm.pytensorf.floatX(0.0))
            one = pt.constant(1)
            t = t_prev + 1
            a_t = 0.001 * pt.sqrt(one - 0.999**t) / (one - 0.9**t)
            for param, g_t in zip(params, grads):
                value = param.get_value(borrow=True)
                m_prev = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape)
                v_prev = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape)
                m_t = 0.9 * m_prev + 0.1 * g_t
                v_t = 0.999 * v_prev + 0.001 * g_t**2
                denom = pt.sqrt(v_t) + 1e-8
                step = a_t * m_t / denom
                outputs.extend([param, g_t, m_t, v_t, denom, a_t, step])
                updates[m_prev] = m_t
                updates[v_prev] = v_t
                updates[param] = param - step
            updates[t_prev] = t
        else:
            raise ValueError(optimizer_name)

        compile_seed = advi.approx.rng.randint(2**30, dtype=np.int64)
        step_fn = pm.pytensorf.compile([], outputs, updates=updates, random_seed=compile_seed)
    return advi, step_fn

def run_full_pymc_ADVI(n_iters, n_mc, optimizer_name, seed=42, mu_p=0.0, sigma_p=1.0):
    """
    Iterates over the single-step update function above.
    """
    advi, step_fn = compile_pymc_ADVI_update(n_mc=n_mc, optimizer_name=optimizer_name, seed=seed)
    rows, mean_trace, std_trace = [], [], []
    for _ in range(n_iters):
        rows.append([float(np.array(v).ravel()[0]) for v in step_fn()])
        mean_trace.append(float(np.array(advi.approx.mean.eval()).ravel()[0]))
        std_trace.append(float(np.array(advi.approx.std.eval()).ravel()[0]))

    rows = np.asarray(rows, dtype=float)
    mean_trace = np.asarray(mean_trace, dtype=float)
    std_trace = np.asarray(std_trace, dtype=float)
    rho_trace = rho_from_std(std_trace)
    result = {
        'mean': mean_trace,
        'sigma': std_trace,
        'rho': rho_trace,
        'kl': exact_kl(mean_trace, rho_trace, mu_p=mu_p, sigma_p=sigma_p),
        'raw': rows,
    }
    # results have to return the respective parameters of the optimizers
    if optimizer_name == 'adagrad_window':
        result.update({
            'mu_before': rows[:, 0], 'grad_mu': rows[:, 1], 'window_sum_mu': rows[:, 2], 'denom_mu': rows[:, 3], 'step_mu': rows[:, 4],
            'rho_before': rows[:, 5], 'grad_rho': rows[:, 6], 'window_sum_rho': rows[:, 7], 'denom_rho': rows[:, 8], 'step_rho': rows[:, 9],
        })
    else:
        result.update({
            'mu_before': rows[:, 0], 'grad_mu': rows[:, 1], 'm_mu': rows[:, 2], 'v_mu': rows[:, 3], 'denom_mu': rows[:, 4], 'a_t': rows[:, 5], 'step_mu': rows[:, 6],
            'rho_before': rows[:, 7], 'grad_rho': rows[:, 8], 'm_rho': rows[:, 9], 'v_rho': rows[:, 10], 'denom_rho': rows[:, 11], 'step_rho': rows[:, 13],
        })
    return result


def validate_pymc_copy(make_model, seed=42,  RUN_SPECS=None, VALIDATION_ITERS=1000, mu_p=0.0, sigma_p=1.0):
    """
    Compare the copied/accessible ADVI fit against the real PyMC fit to
    sanity check that it is correct.
    """
    assert RUN_SPECS is not None, "RUN_SPECS must be provided for validation"
    actual = {
        spec['name']: run_actual_pymc_fit(make_model, VALIDATION_ITERS, spec['n_mc'], spec['optimizer_name'], seed=seed, mu_p=mu_p, sigma_p=sigma_p)
        for spec in RUN_SPECS
    }
    our_copy = {
        spec['name']: run_full_pymc_ADVI(VALIDATION_ITERS, spec['n_mc'], spec['optimizer_name'], seed=seed, mu_p=mu_p, sigma_p=sigma_p)
        for spec in RUN_SPECS
    }
    for spec in RUN_SPECS:
        name = spec['name']
        dm = np.max(np.abs(actual[name]['mean'] - our_copy[name]['mean']))
        ds = np.max(np.abs(actual[name]['std'] - our_copy[name]['std']))
        print(f'{name:24s} | max |mean diff| = {dm:.3e} | max |std diff| = {ds:.3e}')

from modulars.distributions import exact_grad
def attach_exact_gradients(long_runs, mu_p=0.0, sigma_p=1.0):
    """
    Add exact objective gradients to each our_copy trace which already
    stores the stochastic gradients from PyMC. We add the truth gradients
    so we can compare signal vs MC noise.
    """
    for run in long_runs.values():
        grads = np.array([exact_grad(m, r, mu_p=mu_p, sigma_p=sigma_p) for m, r in zip(run['mu_before'], run['rho_before'])])
        run['exact_grad_mu_before'] = grads[:, 0]
        run['exact_grad_rho_before'] = grads[:, 1]

from modulars.utils import softplus
from scipy.special import expit
def infer_noise_moments(mu, rho, g_mu, g_rho, sigma_p=1.0):
    """Recover the sampled noise moments implied by one Monte Carlo gradient.
    In our simple Gaussian setup, the stochastic gradients can be
    written in terms of the mean and second moment of the sampled
    reparameterization noise.
    """
    sigma = float(softplus(rho))
    ds = float(expit(rho))
    z_mean = sigma_p**2 * g_mu
    z_eps_mean = sigma_p**2 * (g_rho / ds + 1.0 / sigma)
    noise_mean = (z_mean - mu) / sigma
    noise_second_moment = (z_eps_mean - mu * noise_mean) / sigma
    return {
        'sigma': sigma,
        'sigmoid_rho': ds,
        'z_mean': z_mean,
        'z_eps_mean': z_eps_mean,
        'noise_mean': noise_mean,
        'noise_second_moment': noise_second_moment,
    }



import pandas as pd

def build_inspection_dataframe(long_runs, inspect_idx, RUN_SPECS, mu_p=0.0, sigma_p=1.0):
    """Turn everything into a df"""
    records = []
    for spec in RUN_SPECS:
        name = spec['name']
        run = long_runs[name]
        mu = float(run['mu_before'][inspect_idx])
        rho = float(run['rho_before'][inspect_idx])
        g_mu = float(run['grad_mu'][inspect_idx])
        g_rho = float(run['grad_rho'][inspect_idx])
        moments = infer_noise_moments(mu, rho, g_mu, g_rho, sigma_p=sigma_p)
        row = {
            'run': name,
            'short': spec['short'],
            'iter': inspect_idx,
            'mu': mu,
            'rho': rho,
            'sigma': moments['sigma'],
            'sigmoid_rho': moments['sigmoid_rho'],
            'exact_g_mu': float(run['exact_grad_mu_before'][inspect_idx]),
            'mc_g_mu': g_mu,
            'exact_g_rho': float(run['exact_grad_rho_before'][inspect_idx]),
            'mc_g_rho': g_rho,
            'noise_mean': moments['noise_mean'],
            'noise_second_moment': moments['noise_second_moment'],
            'z_mean': moments['z_mean'],
            'z_eps_mean': moments['z_eps_mean'],
            'delta_mu': -float(run['step_mu'][inspect_idx]),
            'delta_rho': -float(run['step_rho'][inspect_idx]),
            'raw_step_mu': float(run['step_mu'][inspect_idx]),
            'raw_step_rho': float(run['step_rho'][inspect_idx]),
        }
        state_fields = (
            {'window_sum_rho': float(run['window_sum_rho'][inspect_idx]), 'denom_rho': float(run['denom_rho'][inspect_idx])}
            if spec['optimizer_name'] == 'adagrad_window'
            else {'m_rho': float(run['m_rho'][inspect_idx]), 'v_rho': float(run['v_rho'][inspect_idx]), 'a_t': float(run['a_t'][inspect_idx]), 'denom_rho': float(run['denom_rho'][inspect_idx])}
        )
        row.update(state_fields)
        records.append(row)
    return pd.DataFrame(records)




def print_step_breakdown(inspect_df):
    """Print the breakdown of each step taken at the inspected iteration for each run.

    It shows the inspected approximation point, the inferred noise moments, the resulting stochastic gradient, and the optimizer-specific formula for the final optimizer update step.
    """
    for _, row in inspect_df.iterrows():
        sigma = row['sigma']
        ds = row['sigmoid_rho']
        print()
        print('=' * 90)
        print(row['run'])
        print(f"Var. Approx. values: mu={row['mu']:.6f}, rho={row['rho']:.6f}, sigma={sigma:.6f}, sigmoid(rho)={ds:.6f}")
        print(f"Actual step vector at this iteration: delta_mu={row['delta_mu']:.6f}, delta_rho={row['delta_rho']:.6f}")
        print('Current Monte Carlo inputs:')
        print('  noise_mean is the average of the sampled reparameterization noise draws epsilon_i used at this iteration.')
        print('  noise_second_moment is the average of epsilon_i^2 over the same draws.')
        print(f"  noise_mean={row['noise_mean']:.6f}, noise_second_moment={row['noise_second_moment']:.6f}")
        inner = (row['mu'] * row['noise_mean'] + sigma * row['noise_second_moment']) / 100.0 - 1.0 / sigma
        print('Current stochastic rho gradient built from those values:')
        print('  g_rho^MC = sigmoid(rho) * ((mu * noise_mean + sigma * noise_second_moment)/100 - 1/sigma)')
        print(f"           = {ds:.6f} * ((({row['mu']:.6f} * {row['noise_mean']:.6f}) + ({sigma:.6f} * {row['noise_second_moment']:.6f})) / 100 - {1.0/sigma:.6f})")
        print(f"           = {ds:.6f} * ({inner:.6f}) = {row['mc_g_rho']:.6f}")
        if 'AdagradWindow' in row['run']:
            raw_step = 0.001 * row['mc_g_rho'] / row['denom_rho']
            print('AdagradWindow rho step:')
            print('  window_sum_rho comes from the last 10 squared rho-gradients stored by the optimizer.')
            print(f"  window_sum_rho={row['window_sum_rho']:.6f}")
            print('  denom_rho = sqrt(window_sum_rho + 0.1)')
            print(f"            = sqrt({row['window_sum_rho']:.6f} + 0.1) = {row['denom_rho']:.6f}")
            print('  raw step_rho = 0.001 * g_rho^MC / denom_rho')
            print(f"              = 0.001 * {row['mc_g_rho']:.6f} / {row['denom_rho']:.6f} = {raw_step:.6f}")
            print('  actual delta_rho = -step_rho')
            print(f"                   = {-raw_step:.6f}")
        else:
            raw_step = row['a_t'] * row['m_rho'] / row['denom_rho']
            print('Adam rho step:')
            print("  m_rho comes from Adam's running first-moment state after mixing past gradients with the current g_rho^MC.")
            print("  v_rho comes from Adam's running second-moment state after mixing past squared gradients with the current (g_rho^MC)^2.")
            print("  a_t comes from the iteration index through Adam's bias correction.")
            print(f"  m_rho={row['m_rho']:.6f}, v_rho={row['v_rho']:.6f}, a_t={row['a_t']:.6f}")
            print('  denom_rho = sqrt(v_rho) + 1e-8')
            print(f"            = sqrt({row['v_rho']:.6f}) + 1e-8 = {row['denom_rho']:.6f}")
            print('  raw step_rho = a_t * m_rho / denom_rho')
            print(f"              = {row['a_t']:.6f} * {row['m_rho']:.6f} / {row['denom_rho']:.6f} = {raw_step:.6f}")
            print('  actual delta_rho = -step_rho')
            print(f"                   = {-raw_step:.6f}")


def late_noise_summary(long_runs, tail, RUN_SPECS, true_sigma):
    """Summarize late-phase gradient noise and endpoint error for each run."""
    rows = []
    for spec in RUN_SPECS:
        name = spec['name']
        run = long_runs[name]
        rows.append({
            'run': name,
            'sd_mc_g_rho': run['grad_rho'][-tail:].std(ddof=1),
            'sd_delta_rho': (-run['step_rho'][-tail:]).std(ddof=1),
            'mean_abs_sigma_error': np.mean(np.abs(run['std'][-tail:] - true_sigma)),
            'mean_abs_exact_g_rho': np.mean(np.abs(run['exact_grad_rho_before'][-tail:])),
        })
    return pd.DataFrame(rows)





def aw1_history_sum_before_current_step(run, tail):
    """Return the late AdagradWindow history state before the current gradient is added.

    PyMC stores `window_sum_rho` after inserting the current squared rho-gradient.
    Subtracting that current contribution gives the nine-step history that is fixed
    before the next Monte Carlo draw arrives.
    """
    return np.maximum(run['window_sum_rho'][-tail:] - run['grad_rho'][-tail:]**2, 0.0)


from modulars.utils import inv_softplus
def aw1_expected_delta_rho_given_sigma(sigma, mu, history_sum, quadrature_nodes=60, sigma_p=1.0):
    """Compute the exact one-step expected AdagradWindow rho update for 1 MC.

    The only randomness here is the fresh standard-normal reparameterization draw.
    We integrate it out with Gauss-Hermite quadrature while keeping the late replay
    state `(mu_t, history_sum_t)` fixed.
    """
    from numpy.polynomial.hermite import hermgauss
    sigma = float(sigma)
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    history_sum = np.atleast_1d(np.asarray(history_sum, dtype=float))
    nodes, weights = hermgauss(quadrature_nodes)
    eps = np.sqrt(2.0) * nodes[:, None]
    weights = (weights / np.sqrt(np.pi))[:, None]
    ds_drho = expit(inv_softplus(sigma))
    g_rho = ds_drho * ((mu[None, :] * eps + sigma * eps**2) / sigma_p**2 - 1.0 / sigma)
    delta_rho = -0.001 * g_rho / np.sqrt(history_sum[None, :] + g_rho**2 + 0.1)
    return (weights * delta_rho).sum(axis=0)



def summarize_aw1_empirical_steady_state(run, tail, true_sigma, mu_p = 0.0, sigma_p = 1.0, sigma_grid=None, root_bracket=(10.0, 11.5)):
    """Estimate the late-state AdagradWindow 1 MC drift curve and its zero crossing.

    This treats the last `tail` replay states as an empirical sample from the late
    Adagrad state distribution. For each trial sigma we compute the expected next
    rho-step at every late state, average those expectations, and then solve for the
    sigma where that state-averaged drift is zero.
    """
    from modulars.distributions import exact_g_rho_from_sigma
    mu = run['mu_before'][-tail:]
    history_sum = aw1_history_sum_before_current_step(run, tail=tail)
    observed_mean_sigma = float(np.mean(run['std'][-tail:]))
    late_mean_exact_g_rho = float(np.mean(run['exact_grad_rho_before'][-tail:]))
    local_slope_at_optimum = 0.02
    linearized_sigma_prediction = true_sigma + late_mean_exact_g_rho / local_slope_at_optimum
    if sigma_grid is None:
        sigma_grid = [10.0, 10.25, 10.5, observed_mean_sigma, 10.75]

    rows = []
    for sigma in sigma_grid:
        expected_delta = aw1_expected_delta_rho_given_sigma(sigma, mu, history_sum)
        rows.append({
            'sigma': float(sigma),
            'exact_g_rho': exact_g_rho_from_sigma(sigma, mu_p=mu_p, sigma_p=sigma_p),
            'mean_expected_delta_rho': float(np.mean(expected_delta)),
            'median_expected_delta_rho': float(np.median(expected_delta)),
            'frac_states_with_upward_expected_step': float(np.mean(expected_delta > 0.0)),
        })

    mean_drift = lambda sigma: float(np.mean(aw1_expected_delta_rho_given_sigma(sigma, mu, history_sum)))
    sigma_star = float(brentq(mean_drift, *root_bracket))
    return {
        'summary_df': pd.DataFrame(rows),
        'observed_mean_sigma': observed_mean_sigma,
        'late_mean_exact_g_rho': late_mean_exact_g_rho,
        'linearized_sigma_prediction': linearized_sigma_prediction,
        'state_averaged_sigma_star': sigma_star,
        'exact_g_rho_at_sigma_star': exact_g_rho_from_sigma(sigma_star, mu_p=mu_p, sigma_p=sigma_p),
    }