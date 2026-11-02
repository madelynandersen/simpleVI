# modulars/sampled_kl.py: the transforms and their log determinants (against finite differences
# and against each package's own implementation), and the KL estimators (against closed forms).

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from tests._support import requires, requires_cmdstan, set_cmdstan_path
from modulars.sampled_kl import (
    ElementwiseTransform, GaussianApproximation, SIMPLEX_TRANSFORMS, StickBreakingTransform,
    beta_log_density, compare_candidates, dirichlet_log_density, gamma_log_density, inverse_gamma_log_density,
    latent_gaussian_from_draws, normal_log_density, quadrature_kl, sampled_kl, simplex_transform,
    stan_simplex_basis,
)
from modulars.utils import (
    anchored_simplex_inverse, anchored_softmax_simplex_forward, stickbreaking_simplex_forward,
)

K = 4
Z = np.random.default_rng(0).normal(size=(6, K - 1))


def finite_difference_log_det(transform, z, n_free, h=1e-6):
    """log |det| of the Jacobian of the first n_free outputs, by central differences, row by row."""
    out = []
    for row in np.atleast_2d(z):
        jac = np.column_stack([
            (transform.forward(row + h * e)[0, :n_free] - transform.forward(row - h * e)[0, :n_free]) / (2 * h)
            for e in np.eye(len(row))
        ])
        out.append(np.linalg.slogdet(jac)[1])
    return np.array(out)


def lognormal_gamma_kl(m, s, shape, rate):
    """KL(LogNormal(m, s) || Gamma(shape, rate)) in closed form."""
    neg_entropy = -(m + 0.5 * np.log(2 * np.pi * np.e * s**2))
    cross = shape * np.log(rate) - gammaln(shape) + (shape - 1) * m - rate * np.exp(m + s**2 / 2)
    return neg_entropy - cross


class TestSimplexTransforms(unittest.TestCase):
    def test_maps_to_the_simplex_and_inverts(self):
        for name in SIMPLEX_TRANSFORMS:
            T = simplex_transform(name, K)
            theta = T.forward(Z)
            self.assertTrue(np.all(theta > 0), name)
            np.testing.assert_allclose(theta.sum(axis=1), 1.0, err_msg=name)
            np.testing.assert_allclose(T.inverse(theta), Z, atol=1e-10, err_msg=name)

    def test_log_det_matches_finite_differences_in_the_first_k_minus_1_coordinates(self):
        for name in SIMPLEX_TRANSFORMS:
            T = simplex_transform(name, K)
            _, log_det = T.forward_and_log_det(Z)
            np.testing.assert_allclose(log_det, finite_difference_log_det(T, Z, K - 1), atol=1e-6, err_msg=name)

    def test_stickbreaking_is_the_modulars_utils_map(self):
        np.testing.assert_allclose(simplex_transform("stickbreaking", K).forward(Z),
                                   stickbreaking_simplex_forward(Z, n_cats=K), atol=1e-14)

    def test_pymc_default_is_the_anchored_softmax_map(self):
        T = simplex_transform("pymc_default", K)
        theta = T.forward(Z)
        np.testing.assert_allclose(theta, anchored_softmax_simplex_forward(Z, n_cats=K), atol=1e-14)
        np.testing.assert_allclose(T.inverse(theta), anchored_simplex_inverse(theta), atol=1e-12)

    def test_stan_basis_is_orthonormal_and_matches_stans_jacobian_term(self):
        B = stan_simplex_basis(K)
        np.testing.assert_allclose(B.T @ B, np.eye(K - 1), atol=1e-12)
        np.testing.assert_allclose(B.sum(axis=0), 0.0, atol=1e-12)
        # simplex_constrain adds sum(log theta) + 0.5 log K to the log density
        theta, log_det = simplex_transform("stan_default", K).forward_and_log_det(Z)
        np.testing.assert_allclose(log_det, np.log(theta).sum(axis=1) + 0.5 * np.log(K), atol=1e-12)

    def test_extreme_latent_values_stay_finite(self):
        z = np.array([[40.0, -40.0, 30.0]])
        for name in SIMPLEX_TRANSFORMS:
            _, log_det = simplex_transform(name, K).forward_and_log_det(z)
            self.assertTrue(np.isfinite(log_det).all(), name)


class TestElementwiseTransform(unittest.TestCase):
    def test_mixed_kinds_invert_and_have_the_right_log_det(self):
        T = ElementwiseTransform(["identity", "exp", "sigmoid"])
        theta, log_det = T.forward_and_log_det(Z)
        np.testing.assert_allclose(T.inverse(theta), Z, atol=1e-10)
        np.testing.assert_allclose(log_det, finite_difference_log_det(T, Z, 3), atol=1e-6)

    def test_rejects_unknown_kinds(self):
        with self.assertRaises(ValueError):
            ElementwiseTransform(["softplus"])


class TestEstimatorsAgainstClosedForms(unittest.TestCase):
    def test_gaussian_kl(self):
        m_q, s_q, m_p, s_p = np.array([0.3, -1.0]), np.array([0.5, 2.0]), np.array([0.0, 0.5]), np.array([1.0, 1.5])
        exact = np.sum(np.log(s_p / s_q) + (s_q**2 + (m_q - m_p) ** 2) / (2 * s_p**2) - 0.5)
        q, T, p = GaussianApproximation(m_q, s_q), ElementwiseTransform("identity", 2), normal_log_density(m_p, s_p)
        self.assertAlmostEqual(quadrature_kl(q, T, p), exact, places=10)
        est = sampled_kl(q, T, p, n_draws=200_000)
        self.assertLess(abs(est.kl - exact), 4 * est.se)

    def test_lognormal_against_gamma(self):
        q, T, p = GaussianApproximation([0.8], [0.4]), ElementwiseTransform("exp", 1), gamma_log_density(5.0, 2.0)
        exact = lognormal_gamma_kl(0.8, 0.4, 5.0, 2.0)
        self.assertAlmostEqual(quadrature_kl(q, T, p), exact, places=10)
        est = sampled_kl(q, T, p, n_draws=200_000)
        self.assertLess(abs(est.kl - exact), 4 * est.se)

    def test_kl_is_zero_when_q_is_the_posterior(self):
        q = GaussianApproximation([1.0, 2.0], [0.3, 0.7])
        est = sampled_kl(q, ElementwiseTransform("identity", 2), normal_log_density([1.0, 2.0], [0.3, 0.7]), n_draws=1000)
        self.assertAlmostEqual(est.kl, 0.0, places=12)

    def test_closed_form_optimum_of_the_gamma_models_is_the_kl_minimum(self):
        # the gamma-Poisson / gamma-exponential references are the closed-form LogNormal optimum
        from modulars.distributions import gamma_lognormal_optimum
        shape, rate = 12.0, 3.5
        m, s = gamma_lognormal_optimum(shape, rate)
        T, p = ElementwiseTransform("exp", 1), gamma_log_density(shape, rate)
        best = quadrature_kl(GaussianApproximation([m], [s]), T, p)
        for dm, ds in [(1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)]:
            self.assertGreater(quadrature_kl(GaussianApproximation([m + dm], [s + ds]), T, p), best)

    def test_beta_and_inverse_gamma_targets_are_normalized(self):
        # E_q[p_theta(T(z)) |det dT(z)| / q(z)] = integral of p over its support = 1
        for T, p, loc, scale in [(ElementwiseTransform("sigmoid", 1), beta_log_density(3.0, 7.0), [-0.8], [0.6]),
                                 (ElementwiseTransform("exp", 1), inverse_gamma_log_density(4.0, 3.0), [0.0], [0.5])]:
            q = GaussianApproximation(loc, scale)
            nodes, weights = np.polynomial.hermite_e.hermegauss(200)
            eps = nodes[:, None]
            theta, log_det = T.forward_and_log_det(q.draws(eps))
            ratio = np.exp(p(theta) + log_det - q.log_prob_from_eps(eps))
            self.assertAlmostEqual(float(np.sum(weights / np.sqrt(2 * np.pi) * ratio)), 1.0, places=6)


class TestDirichlet(unittest.TestCase):
    counts = np.array([[20, 15, 15], [18, 17, 15], [22, 12, 16]])
    alpha_prior = np.ones(3)

    def test_every_transform_gives_a_positive_kl_that_quadrature_and_sampling_agree_on(self):
        p = dirichlet_log_density(self.alpha_prior + self.counts.sum(axis=0))
        for name in SIMPLEX_TRANSFORMS:
            T = simplex_transform(name, 3)
            q = latent_gaussian_from_draws(np.random.default_rng(1).dirichlet(self.alpha_prior + self.counts.sum(0), 50_000), T)
            exact = quadrature_kl(q, T, p, n_nodes=80)
            est = sampled_kl(q, T, p, n_draws=200_000)
            self.assertGreater(exact, 0.0, name)
            self.assertLess(abs(est.kl - exact), 4 * est.se, name)

    def test_unnormalized_log_joint_differs_from_the_kl_by_the_log_evidence(self):
        alpha_post = self.alpha_prior + self.counts.sum(axis=0)
        log_prior = dirichlet_log_density(self.alpha_prior)
        totals = self.counts.sum(axis=1)
        log_coef = float(np.sum(gammaln(totals + 1)) - np.sum(gammaln(self.counts + 1)))

        def log_joint(theta):
            return log_prior(theta) + log_coef + np.log(theta) @ self.counts.sum(axis=0)

        log_evidence = (log_coef + gammaln(self.alpha_prior.sum()) - gammaln(self.alpha_prior).sum()
                        + gammaln(alpha_post).sum() - gammaln(alpha_post.sum()))
        T = simplex_transform("stickbreaking", 3)
        q = GaussianApproximation([0.2, -0.1], [0.1, 0.12])
        eps = np.random.default_rng(0).standard_normal((1000, 2))
        kl = sampled_kl(q, T, dirichlet_log_density(alpha_post), eps=eps).kl
        neg_elbo = sampled_kl(q, T, log_joint, eps=eps).kl
        self.assertAlmostEqual(kl - neg_elbo, log_evidence, places=8)


class TestCompareCandidates(unittest.TestCase):
    def test_ranks_by_kl_with_small_paired_errors(self):
        p, T = gamma_log_density(5.0, 2.0), ElementwiseTransform("exp", 1)
        m, s = 0.8, 0.45
        candidates = {"near": (GaussianApproximation([m], [s]), T),
                      "off": (GaussianApproximation([m + 0.05], [s]), T),
                      "far": (GaussianApproximation([m + 0.5], [s * 2]), T)}
        table = compare_candidates(candidates, p, n_draws=50_000)
        exact = {name: quadrature_kl(q, t, p) for name, (q, t) in candidates.items()}
        self.assertEqual(list(table["candidate"]), sorted(exact, key=exact.get))
        self.assertTrue(np.all(table["diff_se"].iloc[1:] < table["se"].iloc[1:]))
        self.assertTrue(table["distinguishable"].iloc[1:].all())

    def test_mass_outside_the_support_gives_an_infinite_kl(self):
        # a Normal surrogate placed directly on a positive parameter (no bijector)
        table = compare_candidates({"no bijector": (GaussianApproximation([0.5], [0.5]), ElementwiseTransform("identity", 1)),
                                    "exp": (GaussianApproximation([0.5], [0.5]), ElementwiseTransform("exp", 1))},
                                   gamma_log_density(5.0, 2.0), n_draws=10_000)
        self.assertEqual(table["candidate"].iloc[0], "exp")
        self.assertTrue(np.isinf(table["kl"].iloc[1]))
        self.assertGreater(table["outside_support"].iloc[1], 0.1)

    def test_latent_gaussian_from_draws_recovers_loc_and_scale(self):
        T = simplex_transform("stan_default", 3)
        loc, scale = np.array([0.3, -0.2]), np.array([0.1, 0.25])
        theta = T.forward(GaussianApproximation(loc, scale).draws(np.random.default_rng(3).standard_normal((100_000, 2))))
        q = latent_gaussian_from_draws(theta, T)
        np.testing.assert_allclose(q.loc, loc, atol=3e-3)
        np.testing.assert_allclose(q.scale, scale, atol=3e-3)


@requires("pymc")
class TestPyMCSimplex(unittest.TestCase):
    def test_pymc_default_transform_and_log_jacobian(self):
        import pytensor
        import pytensor.tensor as pt
        from pymc.logprob.transforms import SimplexTransform
        x = pt.dmatrix("x")
        backward = pytensor.function([x], SimplexTransform().backward(x))
        log_jac_det = pytensor.function([x], SimplexTransform().log_jac_det(x))
        theta, log_det = simplex_transform("pymc_default", K).forward_and_log_det(Z)
        np.testing.assert_allclose(backward(Z), theta, atol=1e-12)
        np.testing.assert_allclose(log_jac_det(Z), log_det, atol=1e-10)


@requires("numpyro")
class TestNumPyro(unittest.TestCase):
    def test_stickbreaking_log_det_matches_numpyro(self):
        import jax
        with jax.experimental.enable_x64():
            import jax.numpy as jnp
            from numpyro.distributions.transforms import StickBreakingTransform as NumPyroStickBreaking
            T = NumPyroStickBreaking()
            theta = T(jnp.asarray(Z))
            ours, log_det = StickBreakingTransform(K).forward_and_log_det(Z)
            np.testing.assert_allclose(np.asarray(theta), ours, atol=1e-12)
            np.testing.assert_allclose(np.asarray(T.log_abs_det_jacobian(jnp.asarray(Z), theta)), log_det, atol=1e-10)

    def test_numpyro_log_joint_and_sampled_kl_agree_with_trace_mean_field_elbo(self):
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import TraceMeanField_ELBO
        from modulars.sampled_kl import numpyro_log_joint

        y = jnp.asarray(np.random.default_rng(2).gamma(3.0, 1.0, size=20))

        def model(y):
            rate = numpyro.sample("rate", dist.Gamma(2.0, 1.0))
            numpyro.sample("obs", dist.Exponential(rate), obs=y)

        log_joint = numpyro_log_joint(model, [("rate", ())], model_args=(y,))
        rate = np.array([[0.4], [0.9]])
        np.testing.assert_allclose(
            log_joint(rate),
            [float(dist.Gamma(2.0, 1.0).log_prob(r[0]) + dist.Exponential(r[0]).log_prob(y).sum()) for r in rate], rtol=1e-6)

        loc, scale = -1.0, 0.25

        def guide(y):
            numpyro.sample("rate", dist.TransformedDistribution(dist.Normal(loc, scale), dist.transforms.ExpTransform()))

        with jax.experimental.enable_x64():
            numpyro_elbo = -float(TraceMeanField_ELBO(num_particles=200_000).loss(jax.random.PRNGKey(0), {}, model, guide, y))
        est = sampled_kl(GaussianApproximation([loc], [scale]), ElementwiseTransform("exp", 1), log_joint, n_draws=200_000)
        self.assertLess(abs(est.elbo - numpyro_elbo), 5 * est.se)


@requires("tensorflow_probability")
class TestTFPSimplex(unittest.TestCase):
    def test_iterated_sigmoid_centered_log_det_matches(self):
        import tensorflow as tf
        import tensorflow_probability as tfp
        bij = tfp.bijectors.IteratedSigmoidCentered()
        z = tf.constant(Z, dtype=tf.float64)
        _, log_det = StickBreakingTransform(K).forward_and_log_det(Z)
        np.testing.assert_allclose(bij.forward_log_det_jacobian(z, event_ndims=1).numpy(), log_det, atol=1e-10)


@requires_cmdstan()
class TestStanSimplex(unittest.TestCase):
    def test_stan_log_prob_is_the_dirichlet_density_plus_our_log_det(self):
        """log_prob with jacobian=True is log Dirichlet(theta) + log |det| of Stan's own simplex transform."""
        from cmdstanpy import CmdStanModel
        set_cmdstan_path()
        code = ("data { int<lower=2> K; vector<lower=0>[K] alpha; } parameters { simplex[K] theta; } "
                "model { target += dirichlet_lpdf(theta | alpha); }")
        alpha = np.array([2.0, 3.5, 1.5, 4.0])
        T = simplex_transform("stan_default", K)
        with tempfile.TemporaryDirectory() as tmp:
            stan_file = Path(tmp) / "simplex.stan"
            stan_file.write_text(code)
            model = CmdStanModel(stan_file=str(stan_file))
            for z in Z[:3]:
                theta, log_det = T.forward_and_log_det(z[None])
                out = model.log_prob(params={"theta": theta[0].tolist()}, data={"K": K, "alpha": alpha.tolist()},
                                     jacobian=True)
                expected = dirichlet_log_density(alpha)(theta)[0] + log_det[0]
                self.assertAlmostEqual(float(out["lp__"].iloc[0]), expected, places=5)


if __name__ == "__main__":
    unittest.main()
