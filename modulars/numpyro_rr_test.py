import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal, AutoDiagonalNormal, AutoMultivariateNormal
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
        tracker_shapes=None,
        tracker_params = ('mu_loc', 'std_loc'),
        n_particles=100,
        step_size=5e-4,
        data=None,
        model_args=None,
        track_every=1,
        guide_kwargs=None):
    optimizer = Adam(step_size=step_size)
    args = _prepare_model_args(data=data, model_args=model_args)
    track_every = max(1, int(track_every))
    n_track = int(np.ceil(n_iters / track_every))
    if guide_kwargs is None:
        guide_kwargs = {}

    single_svi = SVI(model, guide(model, **guide_kwargs), optimizer, loss=Trace_ELBO())
    multi_svi = SVI(
        model,
        guide(model, **guide_kwargs),
        optimizer,
        loss=Trace_ELBO(num_particles=n_particles),
    )

    def run_svi(rng_key, svi_model):
        @jax.jit
        def run_loop(rng_key, *args):
            svi_state = svi_model.init(rng_key, *args)

            if tracker_shapes is None:
                local_tracker_shapes = (tracker_shape,) * len(tracker_params)
            else:
                local_tracker_shapes = tracker_shapes

            tracker = {
                param: jnp.zeros((n_track,) + shape)
                for param, shape in zip(tracker_params, local_tracker_shapes)
            }

            def body_fn(i, val):
                svi_state, tracker, n_done = val

                def update_once(_, state):
                    state, _ = svi_model.update(state, *args)
                    return state

                n_to_run = jnp.minimum(track_every, n_iters - n_done)
                svi_state = jax.lax.fori_loop(0, n_to_run, update_once, svi_state)
                params = svi_model.get_params(svi_state)
                loc, scale = param_getter(params)

                tracker = {
                    param: tracker[param].at[i].set(value)
                    for param, value in zip(tracker_params, (loc, scale))
                }
                return svi_state, tracker, n_done + n_to_run

            _, tracker, _ = jax.lax.fori_loop(0, n_track, body_fn, (svi_state, tracker, 0))
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
        step_size=5e-4,
        track_every=1,
        guide_kwargs=None):
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
        track_every=track_every,
        guide_kwargs=guide_kwargs,
    )


def _flatten_autonormal_params(params, site_names):
    locs, scales = [], []
    for site_name in site_names:
        locs.append(jnp.ravel(params[f"{site_name}_auto_loc"]))
        scales.append(jnp.ravel(params[f"{site_name}_auto_scale"]))
    return jnp.concatenate(locs), jnp.concatenate(scales)


def run_restart_multid_autonormal(
        seed,
        model,
        data,
        param_name,
        n_iters,
        dim,
        site_names,
        n_particles=100,
        model_args=None,
        step_size=5e-4,
        track_every=1,
        guide_kwargs=None):
    del param_name
    return _run_restart_numpyro(
        seed=seed,
        model=model,
        guide=AutoNormal,
        param_getter=lambda params: _flatten_autonormal_params(params, site_names),
        n_iters=n_iters,
        tracker_shape=(dim,),
        tracker_params=('mu_loc', 'std_loc'),
        n_particles=n_particles,
        step_size=step_size,
        data=data,
        model_args=model_args,
        track_every=track_every,
        guide_kwargs=guide_kwargs,
    )


def run_restart_multid_fullrank(
        seed,
        model,
        data,
        param_name,
        n_iters,
        dim,
        n_particles=100,
        model_args=None,
        step_size=5e-4,
        track_every=1,
        guide_kwargs=None):
    del param_name
    return _run_restart_numpyro(
        seed=seed,
        model=model,
        guide=AutoMultivariateNormal,
        param_getter=lambda params: (
            params["auto_loc"],
            jnp.sqrt(jnp.sum(jnp.square(params["auto_scale_tril"]), axis=-1)),
        ),
        n_iters=n_iters,
        tracker_shape=(dim,),
        tracker_params=("mu_loc", "std_loc"),
        n_particles=n_particles,
        step_size=step_size,
        data=data,
        model_args=model_args,
        track_every=track_every,
        guide_kwargs=guide_kwargs,
    )


def fit_multid_fullrank_covariance(
        seed,
        model,
        data,
        n_iters,
        dim,
        n_particles=100,
        model_args=None,
        step_size=5e-4,
        track_every=1,
        guide_kwargs=None):
    del dim
    optimizer = Adam(step_size=step_size)
    args = _prepare_model_args(data=data, model_args=model_args)

    guide = AutoMultivariateNormal(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=n_particles))
    svi_state = svi.init(jax.random.PRNGKey(seed), *args)

    @jax.jit
    def run_loop(svi_state, *args):
        def body_fn(_, state):
            state, _ = svi.update(state, *args)
            return state

        return jax.lax.fori_loop(0, n_iters, body_fn, svi_state)

    svi_state = run_loop(svi_state, *args)
    params = svi.get_params(svi_state)
    loc = params["auto_loc"]
    scale_tril = params["auto_scale_tril"]
    covariance = scale_tril @ scale_tril.T
    return np.asarray(loc), np.asarray(covariance)

def run_restart_1d(
        seed,
        model,
        data,
        param_name,
        n_iters,
        n_particles=100,
        model_args=None,
        step_size=5e-4,
        track_every=1,
        guide_kwargs=None):
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
        track_every=track_every,
        guide_kwargs=guide_kwargs,
    )
