import gc

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
    np.random.seed(seed)
    advi = pm.ADVI(model=model, random_seed=seed)

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
