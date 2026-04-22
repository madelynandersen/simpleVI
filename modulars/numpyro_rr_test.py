import jax
import jax.numpy as jnp
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
from numpyro.optim import Adam
import matplotlib.pyplot as plt

def _prepare_model_args(data=None, model_args=None):
    if model_args is not None:
        return tuple(jnp.asarray(arg) for arg in model_args)
    if data is None:
        return tuple()
    return (jnp.asarray(data),)


def _run_restart_numpyro(
        seed,
        model,
        guide,
        param_getter,
        n_iters,
        tracker_shape,
        tracker_params = ('mu_loc', 'std_loc'),
        n_particles=100,
        step_size=5e-4,
        data=None,
        model_args=None):
    optimizer = Adam(step_size=step_size)
    args = _prepare_model_args(data=data, model_args=model_args)

    single_svi = SVI(model, guide(model), optimizer, loss=Trace_ELBO())
    multi_svi = SVI(model, guide(model), optimizer, loss=Trace_ELBO(num_particles=n_particles))

    def run_svi(rng_key, svi_model):
        @jax.jit
        def run_loop(rng_key, *args):
            svi_state = svi_model.init(rng_key, *args)

            tracker = {
                param: jnp.zeros((n_iters,) + tracker_shape)
                for param in tracker_params
            }

            def body_fn(i, val):
                svi_state, tracker = val
                svi_state, _ = svi_model.update(svi_state, *args)
                params = svi_model.get_params(svi_state)
                loc, scale = param_getter(params)

                tracker = {
                    param: tracker[param].at[i].set(value)
                    for param, value in zip(tracker_params, (loc, scale))
                }
                return svi_state, tracker

            _, tracker = jax.lax.fori_loop(0, n_iters, body_fn, (svi_state, tracker))
            return tracker

        return run_loop(rng_key, *args)

    single_tracker = run_svi(jax.random.PRNGKey(seed), single_svi)
    multi_tracker = run_svi(jax.random.PRNGKey(seed + 1000), multi_svi)

    return (
        single_tracker[tracker_params[0]], single_tracker[tracker_params[1]],
        multi_tracker[tracker_params[0]], multi_tracker[tracker_params[1]]
    )

def run_restart_multid(
        seed,
        model,
        data,
        param_name,
        n_iters,
        dim,
        n_particles=100, # since this is the default setting
        model_args=None,
        step_size=5e-4):
    return _run_restart_numpyro(
        seed=seed,
        model=model,
        guide=AutoDiagonalNormal,
        param_getter=lambda params: (params["auto_loc"], params["auto_scale"]),
        n_iters=n_iters,
        tracker_shape=(dim,),
        tracker_params=('mu_loc', 'std_loc'),
        n_particles=n_particles,
        step_size=step_size,
        data=data,
        model_args=model_args,
    )

def run_restart_1d(
        seed,
        model,
        data,
        param_name,
        n_iters,
        n_particles=100,
        model_args=None,
        step_size=5e-4):
    return _run_restart_numpyro(
        seed=seed,
        model=model,
        guide=AutoNormal,
        param_getter=lambda params: (
            params[param_name + "_auto_loc"],
            params[param_name + "_auto_scale"],
        ),
        n_iters=n_iters,
        tracker_shape=(),
        tracker_params=('mu_loc', 'std_loc'),
        n_particles=n_particles,
        step_size=step_size,
        data=data,
        model_args=model_args,
    )