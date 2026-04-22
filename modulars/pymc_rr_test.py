import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor
pytensor.config.cxx = '/usr/bin/clang++'


from pymc.logprob.transforms import Transform



"""
Random restart code for running pymc VI
-- note that this does require the model to 
be created externally and passed in as an argument,
so that we can run the same model with different random restarts
"""

def run_pymc_VI(model, n_mc_samples=1, n_iters=100_000, optimizer='default', seed=0, SINGLE_DIM=True, **kwargs):
    """
    Run pymc VI with the given model and return the mean and std trajectories

    optimizer can be 'default' (pymc's default) or 'adam' (pymc's adam)
    """
    np.random.seed(seed)
    advi = pm.ADVI(model=model, random_seed=seed)

    if SINGLE_DIM:
        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,
            std=advi.approx.std.eval,
        )
    else:
        tracker = pm.callbacks.Tracker(
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
    if SINGLE_DIM:
        return tracker['mean'], tracker['std']
    else:
        return tracker['mean'], tracker['cov']

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

