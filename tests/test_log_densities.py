# Each package's model must have the same log density as an independent scipy implementation.
#
# Packages drop different normalizing constants, so we compare differences of the log density
# between two parameter values, which the constants cancel out of. Covered: the model code shared
# in modulars (logistic regression, radon) in PyMC / NumPyro / TFP, and every Stan model file used
# for Fig. 3. (The conjugate models for PyMC / NumPyro / TFP are written inline in the runners.)

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import log_expit

from tests._support import SINGLE_MC, requires, requires_cmdstan, set_cmdstan_path
from modulars.logistic_regression import load_logistic_regression_data
from modulars.radon import load_radon_data

LOGISTIC = load_logistic_regression_data()
RADON = load_radon_data()
C = len(RADON["county_names"])


# ---------------------------------------------------------------- scipy references
def logistic_logp(b):
    logits = LOGISTIC["x_train"] @ b
    y = LOGISTIC["y_train"]
    return stats.norm.logpdf(b).sum() + np.sum(y * log_expit(logits) + (1 - y) * log_expit(-logits))


def radon_logp(p):
    mean = p["alpha"][RADON["county"]] + p["beta"] * RADON["floor"]
    return (stats.norm.logpdf(p["mu_a"], 0, 10) + stats.expon.logpdf(p["sigma_a"])
            + stats.norm.logpdf(p["alpha"], p["mu_a"], p["sigma_a"]).sum() + stats.norm.logpdf(p["beta"], 0, 10)
            + stats.expon.logpdf(p["sd_y"]) + stats.norm.logpdf(RADON["log_radon"], mean, p["sd_y"]).sum())


rng = np.random.default_rng(0)
LOGISTIC_POINTS = [np.array([-0.4, 0.8, -0.6, 0.3]), np.array([0.1, 0.5, -0.2, 0.0])]
RADON_POINTS = [dict(mu_a=1.4, sigma_a=0.3, alpha=1.4 + 0.3 * rng.normal(size=C), beta=-0.6, sd_y=0.75),
                dict(mu_a=1.2, sigma_a=0.5, alpha=1.2 + 0.5 * rng.normal(size=C), beta=-0.3, sd_y=0.9)]


def difference(logp, points):
    return logp(points[0]) - logp(points[1])


class Base(unittest.TestCase):
    def assert_same_difference(self, package_logp, reference_logp, points, atol=1e-6):
        self.assertAlmostEqual(difference(package_logp, points), difference(reference_logp, points), delta=atol)


# ---------------------------------------------------------------- PyMC / NumPyro / TFP
@requires("pymc")
class TestPyMC(Base):
    def test_logistic(self):
        from modulars.logistic_regression import build_pymc_logistic_regression_model
        logp = build_pymc_logistic_regression_model(LOGISTIC).compile_logp(jacobian=False)
        self.assert_same_difference(lambda b: float(logp({"b": b})), logistic_logp, LOGISTIC_POINTS)

    def test_radon(self):
        from modulars.radon import build_pymc_radon_model
        logp = build_pymc_radon_model(RADON).compile_logp(jacobian=False)
        to_values = lambda p: {"mu_a": p["mu_a"], "sigma_a_log__": np.log(p["sigma_a"]), "alpha": p["alpha"],
                               "beta": p["beta"], "sd_y_log__": np.log(p["sd_y"])}
        self.assert_same_difference(lambda p: float(logp(to_values(p))), radon_logp, RADON_POINTS)


@requires("numpyro")
class TestNumPyro(Base):
    def setUp(self):
        import jax
        jax.config.update("jax_enable_x64", True)
        self.addCleanup(jax.config.update, "jax_enable_x64", False)

    def test_logistic(self):
        import jax.numpy as jnp
        from numpyro.infer.util import log_density
        from modulars.logistic_regression import build_numpyro_logistic_regression_model
        model = build_numpyro_logistic_regression_model()
        args = (jnp.asarray(LOGISTIC["x_train"]), jnp.asarray(LOGISTIC["y_train"]))
        self.assert_same_difference(lambda b: float(log_density(model, args, {}, {"b": jnp.asarray(b)})[0]),
                                    logistic_logp, LOGISTIC_POINTS)

    def test_radon(self):
        import jax.numpy as jnp
        from numpyro.infer.util import log_density
        from modulars.radon import build_numpyro_radon_model
        model = build_numpyro_radon_model()
        args = tuple(jnp.asarray(RADON[k]) for k in ("log_radon", "floor", "log_uranium", "county", "floor_by_county"))
        logp = lambda p: float(log_density(model, args, {}, {k: jnp.asarray(v) for k, v in p.items()})[0])
        self.assert_same_difference(logp, radon_logp, RADON_POINTS)


@requires("tensorflow_probability")
class TestTFP(Base):
    """TFP's log-probabilities are float32, so the tolerance is looser."""

    def test_logistic(self):
        from modulars.logistic_regression import make_tfp_logistic_regression_conditioned_log_prob
        logp = make_tfp_logistic_regression_conditioned_log_prob(LOGISTIC)
        self.assert_same_difference(lambda b: float(logp(b.astype(np.float32))), logistic_logp, LOGISTIC_POINTS, atol=1e-3)

    def test_radon_on_the_unconstrained_scale(self):
        from modulars.radon import make_tfp_radon_conditioned_log_prob
        logp = make_tfp_radon_conditioned_log_prob(RADON)
        to_theta = lambda p: np.concatenate([[p["mu_a"], np.log(p["sigma_a"])], p["alpha"], [p["beta"], np.log(p["sd_y"])]])
        # TFP's function is on the log scale for sigma_a and sd_y, so it includes their log-Jacobians
        reference = lambda p: radon_logp(p) + np.log(p["sigma_a"]) + np.log(p["sd_y"])
        self.assert_same_difference(lambda p: float(logp(to_theta(p).astype(np.float32))), reference, RADON_POINTS, atol=1e-2)


# ---------------------------------------------------------------- Stan model files
@requires_cmdstan()
class TestStanFiles(Base):
    @classmethod
    def setUpClass(cls):
        set_cmdstan_path()
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stan_logp(self, relative_path, data):
        """log density (jacobian=False) of a repo Stan file, compiled in a temporary folder."""
        from cmdstanpy import CmdStanModel
        target = Path(self.tmp.name) / Path(relative_path).name
        shutil.copy(SINGLE_MC / relative_path, target)
        model = CmdStanModel(stan_file=str(target))
        return lambda params: float(model.log_prob(params, data, jacobian=False)["lp__"].iloc[0])

    def test_logistic(self):
        data = {"N": len(LOGISTIC["y_train"]), "K": 4, "X": LOGISTIC["x_train"], "y": LOGISTIC["y_train"].astype(int)}
        logp = self.stan_logp("logistic_regression/stan_logistic_regression.stan", data)
        self.assert_same_difference(lambda b: logp({"b": b.tolist()}), logistic_logp, LOGISTIC_POINTS, atol=1e-4)

    def test_radon(self):
        data = {"N": len(RADON["log_radon"]), "C": C, "log_radon": RADON["log_radon"], "floor": RADON["floor"],
                "county": (RADON["county"] + 1).astype(int)}
        logp = self.stan_logp("radon_example/pymc_tutorial/stan_radon_varying_intercept.stan", data)
        to_params = lambda p: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
        self.assert_same_difference(lambda p: logp(to_params(p)), radon_logp, RADON_POINTS, atol=1e-3)

    def test_conjugate_models(self):
        y = np.array([2.5, 3.1, 2.9, 3.4])
        y_pos = np.array([0.4, 1.1, 0.2, 0.7])
        counts = np.array([3, 1, 0, 2])
        Y40 = np.random.default_rng(1).normal(2.0, 1.0, size=(3, 40))
        multi = np.array([[20, 15, 15], [18, 17, 15]])
        a0 = np.array([1.0, 2.0, 1.5])
        cases = [
            ("1dgaussian_knownvar/random_restarts/stan_1dgaussian_knownvar.stan",
             {"N": 4, "y": y, "sigma_like": 1.0, "mu_prior": 0.0, "sigma_prior": 10.0},
             [{"mu": 2.0}, {"mu": 3.5}],
             lambda p: stats.norm.logpdf(p["mu"], 0, 10) + stats.norm.logpdf(y, p["mu"], 1).sum()),
            ("multidim_gaussian_knowncovar/random_restarts/stan_multidim_gaussian_knowncovar.stan",
             {"D": 40, "N": 3, "y": Y40, "mu_prior": np.zeros(40), "sigma_prior": np.full(40, 10.0), "sigma_like": 1.0},
             [{"mu": np.full(40, 2.0)}, {"mu": np.linspace(1, 3, 40)}],
             lambda p: stats.norm.logpdf(p["mu"], 0, 10).sum() + stats.norm.logpdf(Y40, p["mu"], 1).sum()),
            ("1dbeta_binomial/random_restarts/stan_beta_binomial.stan",
             {"n_trials": 10, "y": 7, "alpha_prior": 0.5, "beta_prior": 0.6},
             [{"theta": 0.7}, {"theta": 0.4}],
             lambda p: stats.beta.logpdf(p["theta"], 0.5, 0.6) + stats.binom.logpmf(7, 10, p["theta"])),
            ("gamma_exponential/random_restarts/stan_gamma_exponential.stan",
             {"N": 4, "y": y_pos, "alpha_prior": 1.5, "beta_prior": 1.0},
             [{"lambd": 1.8}, {"lambd": 0.6}],
             lambda p: stats.gamma.logpdf(p["lambd"], 1.5, scale=1.0) + stats.expon.logpdf(y_pos, scale=1 / p["lambd"]).sum()),
            ("gamma_poisson/random_restarts/stan_gamma_poisson.stan",
             {"N": 4, "y": counts, "alpha_prior": 1.5, "beta_prior": 1.0},
             [{"lambd": 1.8}, {"lambd": 0.6}],
             lambda p: stats.gamma.logpdf(p["lambd"], 1.5, scale=1.0) + stats.poisson.logpmf(counts, p["lambd"]).sum()),
            ("invgamma_gaussianknownmean/random_restarts/stan_invgamma_gaussian_knownmean.stan",
             {"N": 4, "y": y, "mu_like": 3.0, "alpha_prior": 4.0, "beta_prior": 3.0},
             [{"sigma_sq": 0.5}, {"sigma_sq": 1.3}],
             lambda p: stats.invgamma.logpdf(p["sigma_sq"], 4.0, scale=3.0) + stats.norm.logpdf(y, 3.0, np.sqrt(p["sigma_sq"])).sum()),
            ("multinomial_dirichlet/stan_dirichlet_multinomial.stan",
             {"N": 2, "K": 3, "total_count": [50, 50], "y": multi, "alpha_prior": a0},
             [{"theta": [0.4, 0.3, 0.3]}, {"theta": [0.2, 0.5, 0.3]}],
             lambda p: stats.dirichlet.logpdf(p["theta"], a0) + sum(stats.multinomial.logpmf(r, 50, p["theta"]) for r in multi)),
        ]
        for path, data, points, reference in cases:
            with self.subTest(model=path):
                logp = self.stan_logp(path, data)
                to_params = lambda p: {k: (np.asarray(v).tolist()) for k, v in p.items()}
                self.assert_same_difference(lambda p: logp(to_params(p)), reference, points, atol=1e-4)

    def test_stickbreaking_dirichlet_model(self):
        """With jacobian=False the stick-breaking model's density at theta_raw is the Dirichlet-multinomial at theta(theta_raw)."""
        from modulars.utils import stickbreaking_simplex_forward
        multi, a0 = np.array([[20, 15, 15], [18, 17, 15]]), np.array([1.0, 2.0, 1.5])
        data = {"N": 2, "K": 3, "total_count": [50, 50], "y": multi, "alpha_prior": a0}
        logp = self.stan_logp("multinomial_dirichlet/stan_dirichlet_multinomial_stickbreaking.stan", data)
        def reference(z):
            theta = stickbreaking_simplex_forward(np.asarray(z)[None], n_cats=3)[0]
            return stats.dirichlet.logpdf(theta, a0) + sum(stats.multinomial.logpmf(r, 50, theta) for r in multi)
        points = [np.array([0.3, -0.1]), np.array([-0.5, 0.4])]
        self.assert_same_difference(lambda z: logp({"theta_raw": z.tolist()}), reference, points, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
