# Closed-form posteriors, moment transforms and the saved "best variational approximation" references,
# checked against independent calculations. numpy / scipy only, so these run in every environment.

import json
import unittest

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from tests._support import SINGLE_MC
from modulars.distributions import (
    binomial, exponential, poisson, gaussian_1d, gaussian_1d_posterior, gaussian_multid, gaussian_multid_posterior,
    gamma_exponential_posterior, gamma_poisson_posterior, gamma_lognormal_optimum, invgamma_posterior,
    lda_posterior_shared_theta, gen_multinomial_data_shared, exact_g_rho_from_sigma,
)
from modulars.utils import exp_lognormal_moments, logistic_moments, load_best_values, stickbreaking_simplex_forward


def load_config(*parts):
    with open(SINGLE_MC.joinpath(*parts)) as f:
        return json.load(f)


def gauss_hermite(n=200):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


class TestConjugatePosteriors(unittest.TestCase):
    def test_gaussian_1d_posterior(self):
        data = gaussian_1d(100, 3.0, 1.0, seed=0)
        mu, sd = gaussian_1d_posterior(data, n_samples=100, mu_prior=0.0, sigma_prior=10.0, sigma_like=1.0)
        precision = 1 / 10.0**2 + 100 / 1.0**2
        self.assertAlmostEqual(sd, np.sqrt(1 / precision))
        self.assertAlmostEqual(mu, (np.sum(data) / 1.0**2) / precision)

    def test_gaussian_multid_posterior_sd_and_variance_conventions_agree(self):
        data = gaussian_multid(n_samples=3, mu_like=2.0, scale_like=1.0, dim=40, seed=42)
        prior_sd = np.full(40, 10.0)
        mu_sd, sd = gaussian_multid_posterior(data, n_samples=3, mu_prior=np.zeros(40), scale_prior=prior_sd,
                                              scale_like=1.0, std_devs=True, diag=True, dim=40)
        mu_var, var = gaussian_multid_posterior(data, n_samples=3, mu_prior=np.zeros(40), scale_prior=prior_sd**2,
                                                scale_like=1.0, std_devs=False, diag=True, dim=40)
        np.testing.assert_allclose(mu_sd, mu_var)
        np.testing.assert_allclose(sd, np.sqrt(var))
        for j in range(3):  # each coordinate is an independent 1D conjugate update
            m1, s1 = gaussian_1d_posterior(data[:, j], n_samples=3, mu_prior=0.0, sigma_prior=10.0, sigma_like=1.0)
            self.assertAlmostEqual(mu_sd[j], m1)
            self.assertAlmostEqual(sd[j], s1)

    def test_invgamma_posterior_moments(self):
        data = np.array([2.5, 3.5, 3.0, 2.0])
        post = invgamma_posterior(data, 3.0, 4.0, 3.0)
        ref = stats.invgamma(post["alpha_post"], scale=post["beta_post"])
        self.assertAlmostEqual(post["mean_post"], ref.mean())
        self.assertAlmostEqual(post["std_post"], ref.std())

    def test_dirichlet_posterior_moments(self):
        counts = np.array([[20, 15, 15], [18, 17, 15]])
        post = lda_posterior_shared_theta(counts, np.ones(3))
        ref = stats.dirichlet(np.ones(3) + counts.sum(axis=0))
        np.testing.assert_allclose(post["mean_post"], ref.mean())
        np.testing.assert_allclose(post["std_post"], np.sqrt(ref.var()))


class TestMomentTransforms(unittest.TestCase):
    def test_exp_lognormal_moments(self):
        m, s = exp_lognormal_moments(0.3, 0.7)
        ref = stats.lognorm(s=0.7, scale=np.exp(0.3))
        self.assertAlmostEqual(m, ref.mean())
        self.assertAlmostEqual(s, ref.std())

    def test_logistic_moments_match_quadrature(self):
        x, w = gauss_hermite()
        for mu, sigma in [(0.8, 0.67), (-1.5, 0.2), (0.0, 1.5)]:
            theta = expit(mu + sigma * x)
            exact_mean = np.sum(w * theta)
            exact_sd = np.sqrt(np.sum(w * theta**2) - exact_mean**2)
            mc_mean, mc_sd = logistic_moments(mu, sigma, n_samples=100_000, seed=1)
            # Monte Carlo error with 100k draws: ~0.3% of the SD
            self.assertAlmostEqual(mc_mean, exact_mean, delta=4 * exact_sd / np.sqrt(100_000))
            self.assertAlmostEqual(mc_sd, exact_sd, delta=0.01 * exact_sd)


class TestGammaReferences(unittest.TestCase):
    """The gamma references are the closed-form best log-normal approximation (manuscript Fig. 3)."""

    def test_closed_form_minimizes_the_kl(self):
        alpha, beta = 11.5, 6.4
        def kl_up_to_const(p):  # KL(LogNormal(m, s^2) || Gamma(alpha, beta)) without terms free of (m, s)
            m, s = p[0], np.exp(p[1])
            return -alpha * m - np.log(s) + beta * np.exp(m + s**2 / 2)
        res = minimize(kl_up_to_const, [0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=10_000))
        m, s = gamma_lognormal_optimum(alpha, beta)
        self.assertAlmostEqual(res.x[0], m, places=6)
        self.assertAlmostEqual(np.exp(res.x[1]), s, places=6)

    def test_configs_hold_the_closed_form_reference_for_their_data(self):
        for folder, file, gen, post in [("gamma_exponential", "exp_config.json", exponential, gamma_exponential_posterior),
                                        ("gamma_poisson", "poisson_config.json", poisson, gamma_poisson_posterior)]:
            with self.subTest(model=folder):
                c = load_config(folder, file)
                self.assertGreater(c["n_samples"], 0, "the gamma models should be fit to data, not the prior")
                m, s = gamma_lognormal_optimum(*post(gen(c["n_samples"], c["lambda_like"]), c["alpha_prior"], c["beta_prior"]))
                self.assertAlmostEqual(c["best_mean"], m, places=10)
                self.assertAlmostEqual(c["best_std"], s, places=10)


class TestBetaReference(unittest.TestCase):
    def test_config_reference_is_the_best_logit_normal_approximation(self):
        c = load_config("1dbeta_binomial", "beta_config.json")
        k = binomial(c["n_samples"], c["theta_like"])
        a, b = c["alpha_prior"] + k, c["beta_prior"] + c["n_samples"] - k
        x, w = gauss_hermite()
        def neg_elbo(p):  # Beta(a, b) target, logit-normal q, entropy of the latent normal = log s + const
            z = p[0] + np.exp(p[1]) * x
            return -(np.sum(w * (a * log_expit(z) + b * log_expit(-z))) + p[1])
        res = minimize(neg_elbo, [0.5, -0.5], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=20_000))
        self.assertAlmostEqual(c["best_mean"], res.x[0], delta=5e-3)
        self.assertAlmostEqual(c["best_std"], np.exp(res.x[1]), delta=5e-3)
        # the saved config parameters are within 1e-4 nats of the optimum
        self.assertLess(neg_elbo([c["best_mean"], np.log(c["best_std"])]) - res.fun, 1e-4)

    def test_reference_moments_match_quadrature(self):
        """load_best_values with 100k draws (as every runner now uses) is close to the exact moments."""
        c = load_config("1dbeta_binomial", "beta_config.json")
        mean, sd = load_best_values(config=c, transform=logistic_moments, n_samples=100_000, seed=1)
        x, w = gauss_hermite()
        theta = expit(c["best_mean"] + c["best_std"] * x)
        exact_mean = np.sum(w * theta)
        self.assertAlmostEqual(mean, exact_mean, delta=0.002)
        self.assertAlmostEqual(sd, np.sqrt(np.sum(w * theta**2) - exact_mean**2), delta=0.002)


class TestSavedReferences(unittest.TestCase):
    """The reference files the heatmaps read (paper_vignettes/overall_bias_results, not in git).
    Every package must be scored against the same reference for the same model."""

    def setUp(self):
        from tests._support import REPO_ROOT
        self.results = REPO_ROOT / "paper_vignettes" / "overall_bias_results"
        if not self.results.is_dir():
            self.skipTest("no saved results in this checkout")

    def references(self, model):
        import pandas as pd
        files = {p: self.results / model / f"{p}_reference.csv" for p in ("pymc", "numpyro", "tfp", "stan")}
        return {p: pd.read_csv(f) for p, f in files.items() if f.exists()}

    def test_same_reference_in_every_package(self):
        for model in ("1d_gaussian", "40d_gaussian", "beta_binomial", "gamma_exponential", "gamma_poisson",
                      "inverse_gamma", "dirichlet_multinomial", "logistic_regression", "radon_constrained", "radon_log"):
            refs = self.references(model)
            with self.subTest(model=model):
                self.assertEqual(len(refs), 4, f"missing reference files for {model}")
                first = next(iter(refs.values()))
                for package, ref in refs.items():
                    np.testing.assert_allclose(ref["best_mean"], first["best_mean"], rtol=1e-6, err_msg=package)
                    np.testing.assert_allclose(ref["best_std"], first["best_std"], rtol=1e-6, err_msg=package)

    def test_gamma_files_hold_the_closed_form_reference(self):
        for model, folder, file, gen, post in [
                ("gamma_exponential", "gamma_exponential", "exp_config.json", exponential, gamma_exponential_posterior),
                ("gamma_poisson", "gamma_poisson", "poisson_config.json", poisson, gamma_poisson_posterior)]:
            c = load_config(folder, file)
            mean, sd = exp_lognormal_moments(*gamma_lognormal_optimum(
                *post(gen(c["n_samples"], c["lambda_like"]), c["alpha_prior"], c["beta_prior"])))
            for package, ref in self.references(model).items():
                with self.subTest(model=model, package=package):
                    # TFP computes it from float32 data, which moves it by ~1e-8
                    np.testing.assert_allclose([ref["best_mean"].iloc[0], ref["best_std"].iloc[0]], [mean, sd], rtol=1e-6)

    def test_dirichlet_files_hold_the_config_reference(self):
        c = load_config("multinomial_dirichlet", "dirichlet_config.json")
        for package, ref in self.references("dirichlet_multinomial").items():
            with self.subTest(package=package):
                np.testing.assert_allclose(ref["best_mean"], c["best_mean"], rtol=1e-12)
                np.testing.assert_allclose(ref["best_std"], c["best_std"], rtol=1e-12)


class TestDirichletReference(unittest.TestCase):
    def test_config_is_internally_consistent(self):
        c = load_config("multinomial_dirichlet", "dirichlet_config.json")
        mean, cov, sd = np.array(c["best_mean"]), np.array(c["best_cov"]), np.array(c["best_std"])
        self.assertAlmostEqual(mean.sum(), 1.0, places=6)
        np.testing.assert_allclose(sd, np.sqrt(np.diag(cov)))
        np.testing.assert_allclose(cov, cov.T)

    def test_saved_moments_match_the_saved_latent_fit(self):
        c = load_config("multinomial_dirichlet", "dirichlet_config.json")
        if "best_latent_loc" not in c:
            self.skipTest("config has no latent parameters")
        z = np.array(c["best_latent_loc"]) + np.array(c["best_latent_scale"]) * np.random.default_rng(7).standard_normal((400_000, c["n_cats"] - 1))
        theta = stickbreaking_simplex_forward(z, n_cats=c["n_cats"])
        np.testing.assert_allclose(theta.mean(axis=0), c["best_mean"], atol=2e-4)
        np.testing.assert_allclose(theta.std(axis=0), c["best_std"], rtol=5e-3)

    def test_data_generation_is_unchanged(self):
        """The reference belongs to these exact counts; regenerating different data would invalidate it."""
        c = load_config("multinomial_dirichlet", "dirichlet_config.json")
        counts = gen_multinomial_data_shared(np.array(c["theta_like"], dtype=np.float32), c["N"], c["total_count"], SEED=c["seed"])
        self.assertEqual(counts.shape, (c["N"], c["n_cats"]))
        np.testing.assert_array_equal(counts.sum(axis=1), c["total_count"])
        np.testing.assert_array_equal(counts.sum(axis=0), [203, 147, 150])


class TestRegressions(unittest.TestCase):
    def test_multid_posterior_rejects_a_wrong_n_samples(self):  # C-9: used to silently use len(data)
        data = gaussian_multid(n_samples=3, mu_like=2.0, scale_like=1.0, dim=4, seed=0)
        with self.assertRaises(ValueError):
            gaussian_multid_posterior(data, n_samples=5, mu_prior=np.zeros(4), scale_prior=1.0, scale_like=1.0, dim=4)

    def test_numpy_scalars_are_accepted(self):  # C-10: np.float64 / np.int64 used to be rejected
        from modulars.distributions import ensure_arr, ensure_cov_arr
        from modulars.utils import check_int_or_float, check_all_int_or_float_or_1darrays
        np.testing.assert_array_equal(ensure_arr(np.float64(2.0), 3), [2.0, 2.0, 2.0])
        np.testing.assert_array_equal(ensure_cov_arr(np.int64(4), (2, 2)), 4 * np.eye(2))
        self.assertTrue(check_int_or_float(np.float32(1.5)) and check_int_or_float(3))
        self.assertFalse(check_int_or_float(True) or check_int_or_float(np.ones(2)) or check_int_or_float("1"))
        self.assertTrue(check_all_int_or_float_or_1darrays(np.float64(1.0), np.zeros(3), 2))
        with self.assertRaises(ValueError):
            ensure_arr(np.zeros(2), 3)

    def test_exact_g_rho_from_sigma_uses_the_prior(self):  # C-1
        self.assertEqual(exact_g_rho_from_sigma(1.0), 0.0)
        self.assertAlmostEqual(exact_g_rho_from_sigma(1.0, sigma_p=10.0), (-1 + 1 / 100) * expit(np.log(np.e - 1)))


if __name__ == "__main__":
    unittest.main()
