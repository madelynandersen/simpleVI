import json
import os
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
from numpyro.optim import Adam
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def run_restart_multid(seed, model, data, param_name, n_iters, dim, n_particles = 100):
    # up the n_particles to change the comparison point
    y_data = jnp.array(data)
    optimizer = Adam(step_size=5e-4) # this is the suggested default value by the vignettes

    single_mc_elbo = Trace_ELBO()
    multi_mc_elbo = Trace_ELBO(num_particles=n_particles)

    single_mc_guide = AutoDiagonalNormal(model)
    multi_mc_guide = AutoDiagonalNormal(model)

    single_svi = SVI(model, single_mc_guide, optimizer, loss=single_mc_elbo)
    multi_svi = SVI(model, multi_mc_guide, optimizer, loss=multi_mc_elbo)

    def run_svi(rng_key, y_data, svi_model):
        @jax.jit
        def run_single_svi_loop(rng_key, y_data):
            svi_state = svi_model.init(rng_key, y_data)
            def body_fn(i, val):
                svi_state, tracker = val
                svi_state, loss = svi_model.update(svi_state, y_data)
                params = svi_model.get_params(svi_state)
                tracker = {
                    'mu_loc': tracker['mu_loc'].at[i].set(params['auto_loc']),
                    'std_loc': tracker['std_loc'].at[i].set(params['auto_scale'])
                }
                return svi_state, tracker
            tracker = {
                'mu_loc': jnp.zeros(n_iters * dim).reshape((n_iters, dim)),
                'std_loc': jnp.zeros(n_iters * dim).reshape((n_iters, dim))
                # this is in fact returning the standard deviation of each dimension, not the variance, 
                # since numpyro's AutoDiagonalNormal returns the scale (std dev) of each dimension
            }
            final_state, final_tracker = jax.lax.fori_loop(0, n_iters, body_fn, (svi_state, tracker))
            final_elbo = svi_model.evaluate(final_state, y_data)
            return final_elbo, final_tracker
        return run_single_svi_loop(rng_key, y_data)

    single_elbo, single_tracker = run_svi(jax.random.PRNGKey(seed), y_data, single_svi)
    multi_elbo, multi_tracker = run_svi(jax.random.PRNGKey(seed+1000), y_data, multi_svi)
    return single_tracker['mu_loc'], single_tracker['std_loc'], multi_tracker['mu_loc'], multi_tracker['std_loc']



def run_restart_1d(seed, model, data, param_name, n_iters, n_particles = 100):
    # up the n_particles to change the comparison point
    y_data = jnp.array(data)
    optimizer = Adam(step_size=5e-4) # this is the suggested default value by the vignettes

    single_mc_elbo = Trace_ELBO()
    multi_mc_elbo = Trace_ELBO(num_particles=n_particles)

    single_mc_guide = AutoNormal(model)
    multi_mc_guide = AutoNormal(model)

    single_svi = SVI(model, single_mc_guide, optimizer, loss=single_mc_elbo)
    multi_svi = SVI(model, multi_mc_guide, optimizer, loss=multi_mc_elbo)

    def run_svi(rng_key, y_data, svi_model, param_name, n_iters):
        @jax.jit
        def run_single_svi_loop(rng_key, y_data):
            svi_state = svi_model.init(rng_key, y_data)
            def body_fn(i, val):
                svi_state, tracker = val
                svi_state, loss = svi_model.update(svi_state, y_data)
                params = svi_model.get_params(svi_state)
                tracker = {
                    'mu_loc': tracker['mu_loc'].at[i].set(params[param_name + '_auto_loc']),
                    'std_loc': tracker['std_loc'].at[i].set(params[param_name + '_auto_scale'])
                }
                return svi_state, tracker
            tracker = {
                'mu_loc': jnp.zeros(n_iters),
                'std_loc': jnp.zeros(n_iters)
            }
            final_state, final_tracker = jax.lax.fori_loop(0, n_iters, body_fn, (svi_state, tracker))
            final_elbo = svi_model.evaluate(final_state, y_data)
            return final_elbo, final_tracker
        return run_single_svi_loop(rng_key, y_data)
    
    single_elbo, single_tracker = run_svi(jax.random.PRNGKey(seed), y_data, single_svi, param_name, n_iters)
    multi_elbo, multi_tracker = run_svi(jax.random.PRNGKey(seed+1000), y_data, multi_svi, param_name, n_iters)
    return single_tracker['mu_loc'], single_tracker['std_loc'], multi_tracker['mu_loc'], multi_tracker['std_loc']
