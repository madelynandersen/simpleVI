# this tests the optional track_start argument of the pymc / tfp / numpyro
# random-restart runners: a run that only records its tail window must match
# the tail of the same run recorded in full, exactly (same seed, same updates).
# each test is skipped when its package isn't installed in the current env,
# so run this file once in each package's conda environment.

import importlib.util
import unittest

import numpy as np


def has_module(name):
    return importlib.util.find_spec(name) is not None


N_ITERS = 200
TRACK_START = 151  # 1-indexed: keep iterations 151..200
N_WINDOW = N_ITERS - TRACK_START + 1


def assert_tail_equal(test_case, full, window, label):
    full = np.asarray(full)
    window = np.asarray(window)
    test_case.assertEqual(full.shape[0], N_ITERS, msg=f"{label}: full trace length")
    test_case.assertEqual(window.shape[0], N_WINDOW, msg=f"{label}: window length")
    test_case.assertEqual(full.shape[1:], window.shape[1:], msg=f"{label}: event shape")
    test_case.assertTrue(
        np.array_equal(full[-N_WINDOW:], window),
        msg=f"{label}: we expected the window to equal the tail of the full trace\n"
            f"max abs diff = {np.max(np.abs(full[-N_WINDOW:] - window))}",
    )


@unittest.skipUnless(has_module("pymc"), "pymc not installed in this environment")
class TestPyMCTrackStart(unittest.TestCase):
    def _model_1d(self):
        import pymc as pm
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 10.0)
            pm.Normal("y", mu=mu, sigma=1.0, observed=np.array([2.5, 3.1, 2.9]))
        return model

    def _model_multid(self):
        import pymc as pm
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 10.0, shape=3)
            sigma = pm.Exponential("sigma", 1.0)
            pm.Normal("y", mu=mu, sigma=sigma, observed=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 2.0]]))
        return model

    def test_single_dim(self):
        from modulars.pymc_rr_test import run_pymc_VI
        for optimizer in ("default", "adam"):
            for n_mc in (1, 100):
                full = run_pymc_VI(self._model_1d(), n_mc, N_ITERS, optimizer, seed=3)
                window = run_pymc_VI(self._model_1d(), n_mc, N_ITERS, optimizer, seed=3, track_start=TRACK_START)
                for f, w, name in zip(full, window, ("mean", "std")):
                    assert_tail_equal(self, f, w, f"pymc 1d {optimizer} {n_mc}mc {name}")

    def test_multi_dim_cov_and_std(self):
        from modulars.pymc_rr_test import run_pymc_VI
        for track_cov in (True, False):
            full = run_pymc_VI(self._model_multid(), 1, N_ITERS, "default", seed=4, SINGLE_DIM=False, track_cov=track_cov)
            window = run_pymc_VI(self._model_multid(), 1, N_ITERS, "default", seed=4, SINGLE_DIM=False,
                                 track_cov=track_cov, track_start=TRACK_START)
            for f, w, name in zip(full, window, ("mean", "scale")):
                assert_tail_equal(self, f, w, f"pymc multid track_cov={track_cov} {name}")

    def test_run_single_seed_passthrough(self):
        from modulars.pymc_rr_test import run_pymc_VI, run_single_seed_pymc_VI

        def run_model(n_mc_samples=1, optimizer="default", seed=1, track_start=None):
            return run_pymc_VI(self._model_1d(), n_mc_samples, N_ITERS, optimizer, seed, track_start=track_start)

        full = run_single_seed_pymc_VI(2, run_model)
        window = run_single_seed_pymc_VI(2, run_model, track_start=TRACK_START)
        for scenario_full, scenario_window in zip(full, window):
            for f, w in zip(scenario_full, scenario_window):
                assert_tail_equal(self, f, w, "pymc run_single_seed_pymc_VI")

    def test_track_start_bounds(self):
        from modulars.pymc_rr_test import run_pymc_VI
        full = run_pymc_VI(self._model_1d(), 1, N_ITERS, "default", seed=5)
        same = run_pymc_VI(self._model_1d(), 1, N_ITERS, "default", seed=5, track_start=1)
        last = run_pymc_VI(self._model_1d(), 1, N_ITERS, "default", seed=5, track_start=N_ITERS)
        self.assertTrue(np.array_equal(full[0], same[0]))
        self.assertTrue(np.array_equal(full[0][-1:], last[0]))


@unittest.skipUnless(has_module("numpyro"), "numpyro not installed in this environment")
class TestNumPyroTrackStart(unittest.TestCase):
    def test_restart_1d(self):
        import numpyro
        import numpyro.distributions as dist
        from modulars.numpyro_rr_test import run_restart_1d

        def model(y):
            lambd = numpyro.sample("lambd", dist.Gamma(1.5, 1.0))
            numpyro.sample("y", dist.Poisson(lambd), obs=y)

        data = np.array([1, 3, 2, 4])
        full = run_restart_1d(7, model, data, "lambd", N_ITERS, n_particles=100)
        window = run_restart_1d(7, model, data, "lambd", N_ITERS, n_particles=100, track_start=TRACK_START)
        for f, w, name in zip(full, window, ("single loc", "single scale", "multi loc", "multi scale")):
            assert_tail_equal(self, f, w, f"numpyro 1d {name}")

    def test_restart_multid_and_autonormal(self):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from modulars.numpyro_rr_test import run_restart_multid, run_restart_multid_autonormal

        def model(y=None):
            mu = numpyro.sample("mu", dist.MultivariateNormal(jnp.zeros(3), jnp.eye(3) * 100.0))
            numpyro.sample("z", dist.MultivariateNormal(mu, jnp.eye(3)), obs=y)

        data = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 2.0]])
        full = run_restart_multid(1, model, data, "mu", N_ITERS, dim=3)
        window = run_restart_multid(1, model, data, "mu", N_ITERS, dim=3, track_start=TRACK_START)
        for f, w in zip(full, window):
            assert_tail_equal(self, f, w, "numpyro multid")

        full = run_restart_multid_autonormal(1, model, data, "mu", N_ITERS, dim=3, site_names=("mu",))
        window = run_restart_multid_autonormal(1, model, data, "mu", N_ITERS, dim=3, site_names=("mu",),
                                               track_start=TRACK_START)
        for f, w in zip(full, window):
            assert_tail_equal(self, f, w, "numpyro multid autonormal")


@unittest.skipUnless(has_module("tensorflow_probability"), "tensorflow_probability not installed in this environment")
class TestTFPTrackStart(unittest.TestCase):
    """We record the full trace and the windowed buffers in the same fit and check
    the buffers hold exactly the tail of that fit's trace. (These tests build their
    surrogates without a seed, so a second run would differ; the tfp_run_restart_*
    helpers use stateless seeds and are reproducible, see test_reproducibility.py.)"""

    def _check_same_fit(self, surrogate, lp, sample_size, label):
        import tensorflow as tf
        import tensorflow_probability as tfp
        from modulars.tfp_rr_test import _get_base_distribution, _windowed_trace_fn

        window_trace_fn, loc_buffer, scale_buffer = _windowed_trace_fn(surrogate, N_ITERS, TRACK_START)

        def both_trace_fn(traceable_quantities):
            # the full trace is recorded exactly as _tfp_run_restart's own trace_fns do
            base_dist = _get_base_distribution(surrogate.distribution)
            return window_trace_fn(traceable_quantities), base_dist.loc, base_dist.scale

        _, full_loc, full_scale = tfp.vi.fit_surrogate_posterior(
            lp,
            surrogate_posterior=surrogate,
            optimizer=tf.optimizers.Adam(),
            trace_fn=both_trace_fn,
            num_steps=N_ITERS,
            jit_compile=True,
            sample_size=sample_size,
            seed=0,
        )
        assert_tail_equal(self, full_loc, loc_buffer.read_value(), f"{label} loc")
        assert_tail_equal(self, full_scale, scale_buffer.read_value(), f"{label} scale")
        # the parameters must actually move during the window (a frozen reference would not)
        for name, window in (("loc", loc_buffer.read_value()), ("scale", scale_buffer.read_value())):
            window = np.asarray(window)
            self.assertGreater(len(np.unique(window.reshape(window.shape[0], -1), axis=0)), 1,
                               msg=f"{label} {name}: window is constant")

    def test_window_1d_with_bijector(self):
        import tensorflow_probability as tfp
        from modulars.tfp_rr_test import make_conditioned_lp
        tfd, tfb = tfp.distributions, tfp.bijectors
        lp = make_conditioned_lp(tfd.Gamma(1.5, 1.0), lambda z: tfd.Poisson(z), np.array([1.0, 3.0, 2.0], np.float32))
        for sample_size in (1, 100):
            surrogate = tfp.experimental.vi.build_factored_surrogate_posterior(event_shape=(), bijector=tfb.Exp())
            self._check_same_fit(surrogate, lp, sample_size, f"tfp 1d {sample_size}mc")

    def test_window_multid(self):
        import tensorflow as tf
        import tensorflow_probability as tfp
        from modulars.tfp_rr_test import make_conditioned_lp
        tfd = tfp.distributions
        lp = make_conditioned_lp(
            prior_dist=tfd.MultivariateNormalTriL(loc=tf.zeros(3), scale_tril=tf.eye(3) * 10.0),
            likelihood_dist=lambda z: tfd.MultivariateNormalTriL(loc=z, scale_tril=tf.eye(3)),
            x=tf.constant([[1.0, 2.0, 3.0], [1.5, 2.5, 2.0]]),
            multivar=True,
        )
        surrogate = tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=[3], dtype=tf.float32, bijector=tfp.bijectors.Identity(),
            initial_parameters={"scale": 1e-2},
        )
        self._check_same_fit(surrogate, lp, 100, "tfp multid")

    def test_window_simplex(self):
        import tensorflow as tf
        import tensorflow_probability as tfp
        tfd = tfp.distributions
        counts = tf.constant([20.0, 15.0, 15.0])

        def lp(theta):
            return tfd.Dirichlet(tf.ones(3)).log_prob(theta) + tf.reduce_sum(counts * tf.math.log(theta), axis=-1)

        surrogate = tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=[3], bijector=tfp.bijectors.IteratedSigmoidCentered(), dtype=tf.float32,
        )
        self._check_same_fit(surrogate, lp, 1, "tfp simplex")

    def test_public_wrappers_return_window_shapes(self):
        import tensorflow as tf
        import tensorflow_probability as tfp
        from modulars.tfp_rr_test import make_conditioned_lp, tfp_run_restart_1d, tfp_run_restart_multid, tfp_run_restart_simplex
        tfd, tfb = tfp.distributions, tfp.bijectors
        lp_1d = make_conditioned_lp(tfd.Gamma(1.5, 1.0), lambda z: tfd.Poisson(z), np.array([1.0, 3.0], np.float32))
        out = tfp_run_restart_1d(1, lp_1d, tfb.Exp(), n_iters=N_ITERS, track_start=TRACK_START)
        self.assertEqual([np.asarray(o).shape for o in out], [(N_WINDOW,)] * 4)
        for o in out:
            self.assertGreater(len(np.unique(np.asarray(o))), 1, msg="tfp_run_restart_1d window is constant")
        lp_md = make_conditioned_lp(
            prior_dist=tfd.MultivariateNormalTriL(loc=tf.zeros(3), scale_tril=tf.eye(3) * 10.0),
            likelihood_dist=lambda z: tfd.MultivariateNormalTriL(loc=z, scale_tril=tf.eye(3)),
            x=tf.constant([[1.0, 2.0, 3.0]]), multivar=True,
        )
        out = tfp_run_restart_multid(1, 3, lp_md, n_iters=N_ITERS, track_start=TRACK_START)
        self.assertEqual([np.asarray(o).shape for o in out], [(N_WINDOW, 3)] * 4)
        counts = tf.constant([20.0, 15.0, 15.0])
        lp_sx = lambda theta: tfd.Dirichlet(tf.ones(3)).log_prob(theta) + tf.reduce_sum(counts * tf.math.log(theta), axis=-1)
        out = tfp_run_restart_simplex(1, 3, lp_sx, n_iters=N_ITERS, track_start=TRACK_START)
        # 3 simplex categories -> 2 unconstrained IteratedSigmoidCentered coordinates
        self.assertEqual([np.asarray(o).shape for o in out], [(N_WINDOW, 2)] * 4)

    def test_rejects_thinned_tracking(self):
        import tensorflow_probability as tfp
        from modulars.tfp_rr_test import make_conditioned_lp, tfp_run_restart_1d
        tfd = tfp.distributions
        lp = make_conditioned_lp(tfd.Normal(0.0, 10.0), lambda z: tfd.Normal(z, 1.0), np.array([1.0], np.float32))
        with self.assertRaises(NotImplementedError):
            tfp_run_restart_1d(0, lp, n_iters=N_ITERS, track_every=10, track_start=TRACK_START)


if __name__ == "__main__":
    unittest.main()
