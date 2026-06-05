import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

def _get_base_distribution(distribution):
    while hasattr(distribution, "distribution"):
        distribution = distribution.distribution
    return distribution


def _thin_tfp_trace(trace, track_every):
    track_every = max(1, int(track_every))
    if track_every == 1:
        return trace
    return trace[track_every - 1::track_every]


def _tfp_fit_surrogate_posterior_thinned(
        conditioned_log_prob_fn,
        surrogate_posterior,
        n_iters,
        seed,
        sample_size=1,
        track_every=1):
    track_every = max(1, int(track_every))
    n_track = int((int(n_iters) + track_every - 1) // track_every)
    even_chunks = int(n_iters) % track_every == 0
    optimizer = tf.optimizers.Adam()
    trainable_variables = surrogate_posterior.trainable_variables
    if hasattr(optimizer, "build"):
        optimizer.build(trainable_variables)

    @tf.function(jit_compile=False)
    def run_loop():
        loc_ta = tf.TensorArray(tf.float32, size=n_track)
        scale_ta = tf.TensorArray(tf.float32, size=n_track)
        step = tf.constant(0, dtype=tf.int32)

        for track_idx in tf.range(n_track):
            if even_chunks:
                step_iter = range(track_every)
            else:
                n_to_run = tf.minimum(
                    tf.constant(track_every, dtype=tf.int32),
                    tf.constant(n_iters, dtype=tf.int32) - step,
                )
                step_iter = tf.range(n_to_run)

            for _ in step_iter:
                with tf.GradientTape() as tape:
                    samples = surrogate_posterior.sample(
                        sample_size,
                        seed=tf.stack([tf.cast(seed, tf.int32), step]),
                    )
                    loss = -tf.reduce_mean(
                        conditioned_log_prob_fn(samples)
                        - surrogate_posterior.log_prob(samples)
                    )
                grads = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(grads, trainable_variables))
                step += 1

            base_dist = _get_base_distribution(surrogate_posterior.distribution)
            loc_ta = loc_ta.write(track_idx, tf.cast(base_dist.loc, tf.float32))
            scale_ta = scale_ta.write(track_idx, tf.cast(base_dist.scale, tf.float32))

        return loc_ta.stack(), scale_ta.stack()

    return run_loop()


def _tfp_run_restart(single_q_z, multi_q_z, conditioned_log_prob_fn, n_iters, seed, n_particles=100, track_every=1):
    if int(track_every) > 1:
        single_loc, single_scale = _tfp_fit_surrogate_posterior_thinned(
            conditioned_log_prob_fn,
            single_q_z,
            n_iters,
            seed,
            sample_size=1,
            track_every=track_every,
        )
        multi_loc, multi_scale = _tfp_fit_surrogate_posterior_thinned(
            conditioned_log_prob_fn,
            multi_q_z,
            n_iters,
            seed + 1000,
            sample_size=n_particles,
            track_every=track_every,
        )
        return single_loc, single_scale, multi_loc, multi_scale

    def single_trace_fn(traceable_quantities):
        base_dist = _get_base_distribution(single_q_z.distribution)
        return base_dist.loc, base_dist.scale

    def multi_trace_fn(traceable_quantities):
        base_dist = _get_base_distribution(multi_q_z.distribution)
        return base_dist.loc, base_dist.scale

            
    single_losses = tfp.vi.fit_surrogate_posterior(
        conditioned_log_prob_fn,
        surrogate_posterior=single_q_z,
        optimizer=tf.optimizers.Adam(),
        trace_fn=single_trace_fn,
        num_steps=n_iters,
        jit_compile=True,
        seed=seed
    )

    multi_losses = tfp.vi.fit_surrogate_posterior(
        conditioned_log_prob_fn,
        surrogate_posterior=multi_q_z,
        optimizer=tf.optimizers.Adam(),
        trace_fn=multi_trace_fn,
        num_steps=n_iters,
        jit_compile=True,
        sample_size=n_particles,
        seed=seed+1000
    )

    return tuple(_thin_tfp_trace(trace, track_every) for trace in (*single_losses, *multi_losses))


def _iter_bijector_children(bijector):
    if hasattr(bijector, "bijectors"):
        children = getattr(bijector, "bijectors")
        if isinstance(children, (list, tuple)):
            for child in children:
                yield child
        elif children is not None:
            yield children
    if hasattr(bijector, "bijector"):
        child = getattr(bijector, "bijector")
        if child is not None:
            yield child


def _find_shift_bijector(bijector):
    if hasattr(bijector, "shift"):
        return bijector
    for child in _iter_bijector_children(bijector):
        found = _find_shift_bijector(child)
        if found is not None:
            return found
    return None


def _find_scale_bijector(bijector):
    if hasattr(bijector, "scale") and hasattr(bijector.scale, "to_dense"):
        return bijector
    for child in _iter_bijector_children(bijector):
        found = _find_scale_bijector(child)
        if found is not None:
            return found
    return None


def _get_fullrank_loc_and_scale(surrogate_posterior):
    shift_bijector = _find_shift_bijector(surrogate_posterior.bijector)
    scale_bijector = _find_scale_bijector(surrogate_posterior.bijector)
    if shift_bijector is None or scale_bijector is None:
        raise ValueError("Could not recover full-rank surrogate parameters from the TFP bijector chain.")

    loc = tf.cast(tf.convert_to_tensor(shift_bijector.shift), tf.float32)
    scale_tril = tf.cast(scale_bijector.scale.to_dense(), tf.float32)
    marginal_scale = tf.sqrt(tf.reduce_sum(tf.square(scale_tril), axis=-1))
    return loc, marginal_scale


def _get_fullrank_loc_and_covariance(surrogate_posterior):
    shift_bijector = _find_shift_bijector(surrogate_posterior.bijector)
    scale_bijector = _find_scale_bijector(surrogate_posterior.bijector)
    if shift_bijector is None or scale_bijector is None:
        raise ValueError("Could not recover full-rank surrogate parameters from the TFP bijector chain.")

    loc = tf.cast(tf.convert_to_tensor(shift_bijector.shift), tf.float32)
    scale_tril = tf.cast(scale_bijector.scale.to_dense(), tf.float32)
    covariance = tf.linalg.matmul(scale_tril, scale_tril, transpose_b=True)
    return loc, covariance

def tfp_run_restart_1d(seed,
                   conditioned_log_prob_fn,
                   bijector=tfb.Identity(),
                   n_iters=100_000, n_particles=100, track_every=1):
    single_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
        event_shape=(),
        bijector=bijector
    )
    multi_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
        event_shape=(),
        bijector=bijector
    )

    return _tfp_run_restart(single_q_z, multi_q_z, conditioned_log_prob_fn, n_iters, seed, n_particles, track_every)

def tfp_run_restart_multid(
        seed, dim,
        conditioned_log_prob_fn,
        bijector=tfb.Identity(),
        n_iters=100_000, n_particles=100,
        track_every=1,
        initial_loc=None,
        initial_scale=1e-2):
    initial_parameters = {"scale": initial_scale}
    if initial_loc is not None:
        initial_parameters["loc"] = tf.convert_to_tensor(initial_loc, dtype=tf.float32)

    single_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=[dim],
            dtype=tf.float32,
            bijector=bijector,
            initial_parameters=initial_parameters,
        )
    multi_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=[dim],
            dtype=tf.float32,
            bijector=bijector,
            initial_parameters=initial_parameters,
        )
    return _tfp_run_restart(single_q_z, multi_q_z, conditioned_log_prob_fn, n_iters, seed, n_particles, track_every)


def tfp_run_restart_multid_fullrank(
        seed, dim,
        conditioned_log_prob_fn,
        bijector=tfb.Identity(),
        n_iters=100_000, n_particles=100):

    del bijector

    single_q_z = tfp.experimental.vi.build_affine_surrogate_posterior(
        event_shape=[dim],
        operators="tril",
        dtype=tf.float32,
    )
    multi_q_z = tfp.experimental.vi.build_affine_surrogate_posterior(
        event_shape=[dim],
        operators="tril",
        dtype=tf.float32,
    )

    def single_trace_fn(traceable_quantities):
        del traceable_quantities
        return _get_fullrank_loc_and_scale(single_q_z)

    def multi_trace_fn(traceable_quantities):
        del traceable_quantities
        return _get_fullrank_loc_and_scale(multi_q_z)

    single_losses = tfp.vi.fit_surrogate_posterior(
        conditioned_log_prob_fn,
        surrogate_posterior=single_q_z,
        optimizer=tf.optimizers.Adam(),
        trace_fn=single_trace_fn,
        num_steps=n_iters,
        jit_compile=True,
        seed=seed
    )

    multi_losses = tfp.vi.fit_surrogate_posterior(
        conditioned_log_prob_fn,
        surrogate_posterior=multi_q_z,
        optimizer=tf.optimizers.Adam(),
        trace_fn=multi_trace_fn,
        num_steps=n_iters,
        jit_compile=True,
        sample_size=n_particles,
        seed=seed+1000
    )

    return *single_losses, *multi_losses


def tfp_fit_multid_fullrank_covariance(
        seed, dim,
        conditioned_log_prob_fn,
        bijector=tfb.Identity(),
        n_iters=100_000, n_particles=100):

    del bijector

    q_z = tfp.experimental.vi.build_affine_surrogate_posterior(
        event_shape=[dim],
        operators="tril",
        dtype=tf.float32,
    )

    optimizer = tf.optimizers.Adam()

    tfp.vi.fit_surrogate_posterior(
        conditioned_log_prob_fn,
        surrogate_posterior=q_z,
        optimizer=optimizer,
        num_steps=n_iters,
        jit_compile=True,
        sample_size=n_particles,
        seed=seed
    )

    loc, covariance = _get_fullrank_loc_and_covariance(q_z)
    return loc.numpy(), covariance.numpy()

def tfp_run_restart_simplex(
        seed, n_cats, conditioned_log_prob_fn,
        bijector=None, n_iters=100_000, n_particles=100):
    if bijector is None:
        bijector = tfb.IteratedSigmoidCentered()

    single_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
        event_shape=[n_cats],
        bijector=bijector,
        dtype=tf.float32,
    )
    multi_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
        event_shape=[n_cats],
        bijector=bijector,
        dtype=tf.float32,
    )

    return _tfp_run_restart(single_q_z, multi_q_z, conditioned_log_prob_fn, n_iters, seed, n_particles)


def make_conditioned_lp(prior_dist, likelihood_dist, x, multivar=False):
    if multivar:
        if type(x) is not int and (x is None or len(x) == 0):
            return lambda z: prior_dist.log_prob(z)
        else:
            def log_prob_fn(z):
                # z: (sample_size, n_dims)
                # x: (n_data, n_dims)
                z_exp = tf.expand_dims(z, 1)        # (sample_size, 1, n_dims)
                x_exp = tf.expand_dims(x, 0)        # (1, n_data, n_dims)
                lp = prior_dist.log_prob(z)         # (sample_size,)
                # Sum log-likelihood over all data points for each z
                lp += tf.reduce_sum(likelihood_dist(z_exp).log_prob(x_exp), axis=-1)  # (sample_size,)
                return lp
            return log_prob_fn
    else:
        if type(x) is not int and (x is None or len(x) == 0):
            return lambda z: prior_dist.log_prob(z)
        else:
            def log_prob_fn(z):
                # z: (sample_size,)
                # x: (n_data,)
                z_exp = tf.expand_dims(z, -1)        # (sample_size, 1)
                x_exp = tf.expand_dims(x, 0)         # (1, n_data)
                lp = prior_dist.log_prob(z)          # (sample_size,)
                lp += tf.reduce_sum(likelihood_dist(z_exp).log_prob(x_exp), axis=-1)  # (sample_size,)
                return lp
            return log_prob_fn


def make_tf_variable(name, initial_value, dtype=tf.float32):
    return tf.Variable(initial_value, dtype=dtype, name=name)


def make_trace_fn(surrogate):
    def trace_fn(traceable_quantities):
        return surrogate.mean(), surrogate.stddev(), traceable_quantities.loss
    return trace_fn
    