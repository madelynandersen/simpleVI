import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm.notebook import tqdm

tfd = tfp.distributions
tfb = tfp.bijectors

def tfp_run_restart_1d(seed,
                   conditioned_log_prob_fn,
                   bijector=tfb.Identity(),
                   n_iters=100_000):
    single_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
        event_shape=(),
        bijector=bijector
    )
    multi_q_z = tfp.experimental.vi.build_factored_surrogate_posterior(
        event_shape=(),
        bijector=bijector
    )

    def single_trace_fn(traceable_quantities):
        return single_q_z.distribution.loc, single_q_z.distribution.scale
    def multi_trace_fn(traceable_quantities):
        return multi_q_z.distribution.loc, multi_q_z.distribution.scale
    
    single_tracker = tfp.vi.fit_surrogate_posterior(
        conditioned_log_prob_fn,
        seed=seed,
        surrogate_posterior=single_q_z,
        trainable_variables=single_q_z.trainable_variables,
        optimizer=tf.optimizers.Adam(),
        trace_fn=single_trace_fn,
        num_steps=n_iters,
        jit_compile=True
    )

    multi_tracker = tfp.vi.fit_surrogate_posterior(
        conditioned_log_prob_fn,
        seed=seed+1000,
        surrogate_posterior=multi_q_z,
        trainable_variables=multi_q_z.trainable_variables,
        optimizer=tf.optimizers.Adam(),
        trace_fn=multi_trace_fn,
        num_steps=n_iters,
        sample_size=100,
        jit_compile=True
    )
    return *single_tracker, *multi_tracker