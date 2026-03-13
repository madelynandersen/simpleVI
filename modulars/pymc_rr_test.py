import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from tqdm import tqdm as standard_tqdm

import pymc as pm
import pytensor.tensor as pt
import pytensor
pytensor.config.cxx = '/usr/bin/clang++'


"""
Random restart code for running pymc VI
-- note that this does require the model to 
be created externally and passed in as an argument,
so that we can run the same model with different random restarts
"""

def run_pymc_VI(model, n_mc_samples=1, n_iters=100_000, optimizer='default', seed=0, **kwargs):
    """
    Run pymc VI with the given model and return the mean and std trajectories

    optimizer can be 'default' (pymc's default) or 'adam' (pymc's adam)
    """
    np.random.seed(seed)
    advi = pm.ADVI(random_seed=seed)

    tracker = pm.callbacks.Tracker(
        mean=advi.approx.mean.eval,
        std=advi.approx.std.eval,
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
    return tracker['mean'], tracker['std']

# runs regular and ADAM VI for a given model and given seed and returns trajectory
def run_single_seed_pymc_VI(seed, run_model_fn):
     np.random.seed(seed)
     single_means, single_stds = run_model_fn(n_mc_samples=1, seed=seed, optimizer='default')
     multi_means, multi_stds = run_model_fn(n_mc_samples=100, seed=seed+1000, optimizer='default')
     adam_single_means, adam_single_stds = run_model_fn(n_mc_samples=1, seed=seed, optimizer='adam')
     adam_multi_means, adam_multi_stds = run_model_fn(n_mc_samples=100, seed=seed+1000, optimizer='adam')
     return [single_means, single_stds, multi_means, multi_stds], [adam_single_means, adam_single_stds, adam_multi_means, adam_multi_stds]

     