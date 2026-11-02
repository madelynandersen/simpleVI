# The simplex transforms. All four packages are compared on the stick-breaking family for the
# Multinomial-Dirichlet model, so each package's stick-breaking map must be the same function.

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._support import SINGLE_MC, requires, requires_cmdstan, set_cmdstan_path
from modulars.utils import stickbreaking_simplex_forward, stickbreaking_simplex_inverse

K = 4
Z = np.random.default_rng(0).normal(size=(6, K - 1))


def finite_difference_log_det(f, z, h=1e-6):
    """log |det d f(z)[:K-1] / dz| by central differences, one row at a time."""
    out = []
    for row in z:
        jac = np.column_stack([(f(row + h * e)[:-1] - f(row - h * e)[:-1]) / (2 * h) for e in np.eye(len(row))])
        out.append(np.log(abs(np.linalg.det(jac))))
    return np.array(out)


class TestNumpyStickBreaking(unittest.TestCase):
    def test_maps_to_the_simplex_and_inverts(self):
        theta = stickbreaking_simplex_forward(Z, n_cats=K)
        self.assertTrue(np.all(theta > 0))
        np.testing.assert_allclose(theta.sum(axis=1), 1.0)
        np.testing.assert_allclose(stickbreaking_simplex_inverse(theta), Z, atol=1e-8)

    def test_zero_maps_to_the_uniform_point(self):
        np.testing.assert_allclose(stickbreaking_simplex_forward(np.zeros((1, K - 1)), n_cats=K), np.full((1, K), 1 / K))


@requires("pymc")
class TestPyMCStickBreaking(unittest.TestCase):
    def test_forward_backward_and_log_jacobian(self):
        import pytensor
        import pytensor.tensor as pt
        from modulars.pymc_rr_test import StickBreakingSimplexTransform
        T = StickBreakingSimplexTransform()
        x, th = pt.dmatrix("x"), pt.dmatrix("th")
        backward = pytensor.function([x], T.backward(x))
        forward = pytensor.function([th], T.forward(th))
        log_jac_det = pytensor.function([x], T.log_jac_det(x))
        theta = backward(Z)
        np.testing.assert_allclose(theta, stickbreaking_simplex_forward(Z, n_cats=K), atol=1e-12)
        np.testing.assert_allclose(forward(theta), Z, atol=1e-8)
        np.testing.assert_allclose(log_jac_det(Z), finite_difference_log_det(lambda r: backward(r[None])[0], Z), atol=1e-6)


@requires("numpyro")
class TestNumPyroStickBreaking(unittest.TestCase):
    def test_numpyro_default_simplex_transform_is_the_same_map(self):
        import jax
        jax.config.update("jax_enable_x64", True)
        try:
            import jax.numpy as jnp
            import numpyro.distributions as dist
            from numpyro.distributions.transforms import biject_to
            T = biject_to(dist.Dirichlet(jnp.ones(K)).support)
            self.assertEqual(type(T).__name__, "StickBreakingTransform")
            theta = np.asarray(T(jnp.asarray(Z)))
            np.testing.assert_allclose(theta, stickbreaking_simplex_forward(Z, n_cats=K), atol=1e-12)
            np.testing.assert_allclose(np.asarray(T.log_abs_det_jacobian(jnp.asarray(Z), jnp.asarray(theta))),
                                       finite_difference_log_det(lambda r: np.asarray(T(jnp.asarray(r))), Z), atol=1e-6)
        finally:
            jax.config.update("jax_enable_x64", False)


@requires("tensorflow_probability")
class TestTFPStickBreaking(unittest.TestCase):
    def test_iterated_sigmoid_centered_is_the_same_map(self):
        import tensorflow as tf
        import tensorflow_probability as tfp
        theta = tfp.bijectors.IteratedSigmoidCentered().forward(tf.constant(Z, dtype=tf.float64)).numpy()
        np.testing.assert_allclose(theta, stickbreaking_simplex_forward(Z, n_cats=K), atol=1e-12)


@requires_cmdstan()
class TestStanStickBreaking(unittest.TestCase):
    def test_user_defined_stan_transform_is_the_same_map(self):
        """stan_dirichlet_multinomial_stickbreaking.stan: every theta draw equals stick-breaking of its theta_raw."""
        from cmdstanpy import CmdStanModel
        set_cmdstan_path()
        with tempfile.TemporaryDirectory() as tmp:
            stan_file = Path(tmp) / "model.stan"   # compile in a temporary folder, not next to the repo's model file
            shutil.copy(SINGLE_MC / "multinomial_dirichlet" / "stan_dirichlet_multinomial_stickbreaking.stan", stan_file)
            model = CmdStanModel(stan_file=str(stan_file))
            counts = np.array([[20, 15, 15], [18, 17, 15]])
            data = {"N": 2, "K": 3, "total_count": [50, 50], "y": counts.tolist(), "alpha_prior": [1.0, 1.0, 1.0]}
            fit = model.variational(data=data, seed=1, iter=2000, draws=500, require_converged=False,
                                    show_console=False, output_dir=tmp)
            draws = fit.variational_sample_pd
        theta = draws[[f"theta[{k}]" for k in (1, 2, 3)]].to_numpy()
        theta_raw = draws[["theta_raw[1]", "theta_raw[2]"]].to_numpy()
        np.testing.assert_allclose(theta, stickbreaking_simplex_forward(theta_raw, n_cats=3), atol=5e-6)  # CSV has 6 significant figures


if __name__ == "__main__":
    unittest.main()
