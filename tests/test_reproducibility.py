# Two runs with the same seed must give identical trajectories in every package, so the saved
# results can be regenerated exactly. (TFP needs stateless seeds; see modulars.tfp_rr_test._stateless_seed.)

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._support import SINGLE_MC, requires, requires_cmdstan, set_cmdstan_path

DATA = np.array([2.5, 3.1, 2.9, 3.4])


def assert_identical(test, first, second):
    test.assertEqual(len(first), len(second))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    test.assertGreater(len(np.unique(np.asarray(first[0]))), 1, "trajectory should move")


@requires("pymc")
class TestPyMC(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        import pymc as pm
        from modulars.pymc_rr_test import run_pymc_VI
        def run():
            with pm.Model() as model:
                mu = pm.Normal("mu", 0.0, 10.0)
                pm.Normal("y", mu=mu, sigma=1.0, observed=DATA)
            return run_pymc_VI(model, 1, 200, "default", seed=5)
        assert_identical(self, run(), run())

    def test_run_full_pymc_advi_runs(self):  # regression test for C-1
        import pymc as pm
        from modulars.pymc_rr_test import run_full_pymc_ADVI
        def make_model():
            with pm.Model() as model:
                pm.Normal("mu", 0.0, 1.0)
            return model
        result = run_full_pymc_ADVI(make_model, 20, 1, "adam")
        self.assertEqual(len(result["sigma"]), 20)


@requires("numpyro")
class TestNumPyro(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        import numpyro
        import numpyro.distributions as dist
        from modulars.numpyro_rr_test import run_restart_1d
        def model(y):
            mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
            numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)
        assert_identical(self, run_restart_1d(5, model, DATA, "mu", 200), run_restart_1d(5, model, DATA, "mu", 200))


@requires("tensorflow_probability")
class TestTFP(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        import tensorflow_probability as tfp
        from modulars.tfp_rr_test import make_conditioned_lp, tfp_run_restart_1d
        tfd = tfp.distributions
        lp = make_conditioned_lp(tfd.Normal(0.0, 10.0), lambda z: tfd.Normal(z, 1.0), DATA.astype(np.float32))
        run = lambda: tfp_run_restart_1d(5, lp, n_iters=200, track_start=151)
        assert_identical(self, run(), run())

    def test_different_seeds_differ(self):
        import tensorflow_probability as tfp
        from modulars.tfp_rr_test import make_conditioned_lp, tfp_run_restart_1d
        tfd = tfp.distributions
        lp = make_conditioned_lp(tfd.Normal(0.0, 10.0), lambda z: tfd.Normal(z, 1.0), DATA.astype(np.float32))
        a = tfp_run_restart_1d(5, lp, n_iters=200, track_start=151)
        b = tfp_run_restart_1d(6, lp, n_iters=200, track_start=151)
        self.assertFalse(np.array_equal(np.asarray(a[0]), np.asarray(b[0])))


@requires_cmdstan()
class TestStan(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        from modulars.stan_rr_test import run_stan_random_restarts
        set_cmdstan_path()
        with tempfile.TemporaryDirectory() as tmp:
            stan_file = Path(tmp) / "model.stan"  # compile in a temporary folder
            shutil.copy(SINGLE_MC / "1dgaussian_knownvar" / "random_restarts" / "stan_1dgaussian_knownvar.stan", stan_file)
            data = {"N": len(DATA), "y": DATA, "sigma_like": 1.0, "mu_prior": 0.0, "sigma_prior": 10.0}
            runs = []
            for k in range(2):
                result = run_stan_random_restarts(stan_file, data, ["mu"], Path(tmp) / f"run{k}", max_iters=300,
                                                  n_restarts=1, track_every=100, show_progress=False,
                                                  scenarios=({"name": "default", "cmd_args": ()},
                                                             {"name": "100mc", "cmd_args": ("grad_samples=100",)}))
                self.assertEqual(result["failures"], [])
                runs.append([result["means_by_scenario"]["default"], result["stds_by_scenario"]["default"]])
        assert_identical(self, runs[0], runs[1])


if __name__ == "__main__":
    unittest.main()
