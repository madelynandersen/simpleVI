# Pin the library defaults that the manuscript evaluates (Tables 2-3, Sections 3.2 and 3.5).
#
# If one of these fails after an upgrade, a default the paper describes may have changed:
# re-check the manuscript before updating the expected value here.

import inspect
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._support import requires, requires_cmdstan, set_cmdstan_path

# versions reported in the manuscript's Table 2
PAPER_VERSIONS = {"pymc": "5.25.1", "numpyro": "0.18.0", "tensorflow_probability": "0.25.0",
                  "cmdstanpy": "1.3.0", "cmdstan": (2, 39)}


def defaults(fn):
    return {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not inspect.Parameter.empty}


def required(fn):
    return {k for k, v in inspect.signature(fn).parameters.items()
            if v.default is inspect.Parameter.empty and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)}


@requires("pymc")
class TestPyMCDefaults(unittest.TestCase):
    def test_version(self):
        import pymc as pm
        self.assertEqual(pm.__version__, PAPER_VERSIONS["pymc"])

    def test_one_mc_sample_and_adagrad_window(self):
        import pymc as pm
        from pymc.variational.opvi import ObjectiveFunction
        d = defaults(ObjectiveFunction.step_function)
        self.assertIsNone(d["obj_n_mc"])          # None is treated as one sample (checked below)
        self.assertIs(d["obj_optimizer"], pm.adagrad_window)
        self.assertEqual(defaults(pm.adagrad_window), {"loss_or_grads": None, "params": None,
                                                       "learning_rate": 0.001, "epsilon": 0.1, "n_win": 10})
        self.assertEqual(defaults(pm.variational.inference.Inference.fit)["n"], 10_000)

    def test_adam_settings(self):
        import pymc as pm
        d = defaults(pm.adam)
        self.assertEqual((d["learning_rate"], d["beta1"], d["beta2"], d["epsilon"]), (0.001, 0.9, 0.999, 1e-8))

    def test_default_mc_samples_behave_like_one(self):
        import pymc as pm
        def run(**kw):
            with pm.Model():
                pm.Normal("mu", 0.0, 10.0)
                advi = pm.ADVI(random_seed=3)
                tracker = pm.callbacks.Tracker(std=advi.approx.std.eval)
                advi.fit(50, callbacks=[tracker], progressbar=False, **kw)
            return np.asarray(tracker["std"])
        np.testing.assert_array_equal(run(), run(obj_n_mc=1))

    def test_initial_scale_is_softplus_of_zero(self):
        import pymc as pm
        with pm.Model():
            pm.Normal("mu", 0.0, 10.0)
            self.assertAlmostEqual(float(pm.ADVI().approx.std.eval()[0]), math.log(2.0))

    def test_dirichlet_default_transform(self):
        import pymc as pm
        with pm.Model() as model:
            pm.Dirichlet("theta", a=np.ones(3))
        self.assertEqual(type(model.rvs_to_transforms[model["theta"]]).__name__, "SimplexTransform")


@requires("numpyro")
class TestNumPyroDefaults(unittest.TestCase):
    def test_version(self):
        import numpyro
        self.assertEqual(numpyro.__version__, PAPER_VERSIONS["numpyro"])

    def test_one_mc_sample(self):
        from numpyro.infer import Trace_ELBO
        self.assertEqual(Trace_ELBO().num_particles, 1)

    def test_optimizer_has_no_default(self):
        from numpyro.infer import SVI
        self.assertIn("optim", required(SVI.__init__))

    def test_guide_initialization(self):
        from numpyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
        from numpyro.infer.initialization import init_to_uniform
        for guide in (AutoNormal, AutoDiagonalNormal):
            d = defaults(guide.__init__)
            self.assertEqual(d["init_scale"], 0.1, guide.__name__)
            self.assertIs(d["init_loc_fn"], init_to_uniform, guide.__name__)
        self.assertEqual(defaults(init_to_uniform)["radius"], 2)


@requires("tensorflow_probability")
class TestTFPDefaults(unittest.TestCase):
    def test_version(self):
        import tensorflow_probability as tfp
        self.assertEqual(tfp.__version__, PAPER_VERSIONS["tensorflow_probability"])

    def test_one_mc_sample_and_required_settings(self):
        import tensorflow_probability as tfp
        self.assertEqual(defaults(tfp.vi.fit_surrogate_posterior)["sample_size"], 1)
        self.assertTrue({"optimizer", "num_steps"} <= required(tfp.vi.fit_surrogate_posterior))

    def test_surrogate_initialization_and_no_default_transform(self):
        import tensorflow_probability as tfp
        q = tfp.experimental.vi.build_factored_surrogate_posterior(event_shape=[3], seed=(1, 2))
        # without a bijector the surrogate is a plain Normal on the original space: no transform is applied
        self.assertEqual(type(q.distribution).__name__, "Normal")
        np.testing.assert_allclose(q.distribution.scale.numpy(), 0.01, rtol=1e-6)  # float32


@requires_cmdstan()
class TestStanDefaults(unittest.TestCase):
    """Run CmdStan ADVI with every setting left at its default and read the settings it reports."""

    def test_version_and_variational_defaults(self):
        import cmdstanpy
        from cmdstanpy import CmdStanModel
        set_cmdstan_path()
        self.assertEqual(cmdstanpy.__version__, PAPER_VERSIONS["cmdstanpy"])
        self.assertEqual(cmdstanpy.cmdstan_version(), PAPER_VERSIONS["cmdstan"])
        with tempfile.TemporaryDirectory() as tmp:
            stan_file = Path(tmp) / "normal.stan"
            stan_file.write_text("parameters { real mu; } model { mu ~ normal(0, 10); }\n")
            fit = CmdStanModel(stan_file=str(stan_file)).variational(seed=1, require_converged=False,
                                                                     show_console=False, output_dir=tmp)
            header = Path(fit.runset.csv_files[0]).read_text().split("\nlp__")[0]
        for setting in ["algorithm = meanfield (Default)", "iter = 10000 (Default)", "grad_samples = 1 (Default)",
                        "elbo_samples = 100 (Default)", "eta = 1 (Default)", "engaged = true (Default)",
                        "tol_rel_obj = 0.01 (Default)", "eval_elbo = 100 (Default)", "output_samples = 1000 (Default)",
                        "init = 2 (Default)"]:
            self.assertIn(setting, header)


if __name__ == "__main__":
    unittest.main()
