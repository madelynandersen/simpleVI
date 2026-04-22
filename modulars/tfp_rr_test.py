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