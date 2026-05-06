import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

def _get_base_distribution(distribution):
    while hasattr(distribution, "distribution"):
        distribution = distribution.distribution
    return distribution


def _tfp_run_restart(single_q_z, multi_q_z, conditioned_log_prob_fn, n_iters, seed, n_particles=100):
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

    return *single_losses, *multi_losses


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
                   n_iters=100_000, n_particles=100):
    single_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
        event_shape=(),
        bijector=bijector
    )
    multi_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
        event_shape=(),
        bijector=bijector
    )

    return _tfp_run_restart(single_q_z, multi_q_z, conditioned_log_prob_fn, n_iters, seed, n_particles)

def tfp_run_restart_multid(
        seed, dim,
        conditioned_log_prob_fn,
        bijector=tfb.Identity(),
        n_iters=100_000, n_particles=100):

    single_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=[dim],
            dtype=tf.float32,
            bijector=bijector
        )
    multi_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=[dim],
            dtype=tf.float32,
            bijector=bijector
        )
    return _tfp_run_restart(single_q_z, multi_q_z, conditioned_log_prob_fn, n_iters, seed, n_particles)


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
