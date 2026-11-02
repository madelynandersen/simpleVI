# Tests for the saved-output format and the heatmap metrics (paper_vignettes/overall_bias_runners/bias_io.py).
# numpy / pandas only, so these run in every environment.

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import _support  # noqa: F401  (puts the runners folder on sys.path)
import bias_io


class TestWindow(unittest.TestCase):
    def test_window_iterations_are_the_last_window_iterations(self):
        np.testing.assert_array_equal(bias_io.window_iterations(10, 3), [8, 9, 10])
        self.assertEqual(bias_io.track_start(50_000, 500), 49_501)

    def test_window_must_fit_in_the_run(self):
        for window in (0, 11):
            with self.assertRaises(ValueError):
                bias_io.window_iterations(10, window)


class TestTrackingRoundTrip(unittest.TestCase):
    def test_save_and_load_keep_values_shapes_and_coordinate_order(self):
        rng = np.random.default_rng(0)
        iterations = bias_io.window_iterations(100, 5)
        coordinates = ["theta[2]", "theta[0]", "theta[1]"]  # deliberately not sorted
        setups = {
            "default": (rng.normal(size=(4, 5, 3)), rng.uniform(size=(4, 5, 3))),
            "100mc": (rng.normal(size=(4, 5, 3)), rng.uniform(size=(4, 5, 3))),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m" / "pkg_tracking.csv.gz"
            bias_io.save_tracking(path, setups, iterations, coordinates, float_format="%.17g")
            loaded, loaded_iterations, loaded_coordinates = bias_io.load_tracking(path)
        self.assertEqual(loaded_coordinates, coordinates)
        np.testing.assert_array_equal(loaded_iterations, iterations)
        for setup, (means, stds) in setups.items():  # 17 significant digits: exact up to CSV parsing (pandas parses to within a few ulp)
            np.testing.assert_allclose(loaded[setup]["mean"], means, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(loaded[setup]["std"], stds, rtol=1e-12, atol=1e-15)

    def test_one_dimensional_trajectories_get_a_coordinate_axis(self):
        iterations = bias_io.window_iterations(10, 4)
        means, stds = np.arange(8.0).reshape(2, 4), np.ones((2, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv.gz"
            bias_io.save_tracking(path, {"default": (means, stds)}, iterations, ["mu"])
            loaded, _, _ = bias_io.load_tracking(path)
        self.assertEqual(loaded["default"]["mean"].shape, (2, 4, 1))
        np.testing.assert_array_equal(loaded["default"]["mean"][..., 0], means)

    def test_wrong_number_of_iterations_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            bias_io.save_tracking(Path(tmp) / "t.csv.gz", {"default": (np.ones((2, 3)), np.ones((2, 3)))},
                                  bias_io.window_iterations(10, 4), ["mu"])

    def test_reference_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.csv"
            bias_io.save_reference(path, ["a", "b"], [1.5, -2.0], [0.1, 0.2], "test")
            ref = bias_io.load_reference(path)
        self.assertEqual(list(ref["coordinate"]), ["a", "b"])
        np.testing.assert_array_equal(ref["best_mean"], [1.5, -2.0])
        np.testing.assert_array_equal(ref["best_std"], [0.1, 0.2])


class TestRadonScales(unittest.TestCase):
    def test_only_log_scale_coordinates_are_transformed(self):
        from modulars.utils import exp_lognormal_moments
        names = ["mu_a", "sigma_a_log__", "beta", "sd_y_log__"]
        means = np.array([[1.0, 0.1, 2.0, -0.5]])
        stds = np.array([[0.3, 0.2, 0.4, 0.1]])
        c_means, c_stds, c_names = bias_io.radon_log_to_constrained(
            means, stds, names, ["sigma_a_log__", "sd_y_log__"], exp_lognormal_moments)
        self.assertEqual(c_names, ["mu_a", "sigma_a", "beta", "sd_y"])
        np.testing.assert_array_equal(c_means[:, [0, 2]], means[:, [0, 2]])
        np.testing.assert_array_equal(c_stds[:, [0, 2]], stds[:, [0, 2]])
        for j in (1, 3):
            m, s = exp_lognormal_moments(means[0, j], stds[0, j])
            self.assertAlmostEqual(c_means[0, j], m)
            self.assertAlmostEqual(c_stds[0, j], s)
        self.assertEqual(means[0, 1], 0.1, "inputs must not be modified")


class TestHeatmapMetrics(unittest.TestCase):
    """per_coordinate_metrics on synthetic tracks whose metrics are known exactly."""

    def setUp(self):
        self.mu_star, self.sigma_star = 2.0, 0.5
        # per-seed errors of both signs, so taking absolute values matters
        self.delta = np.array([0.1, -0.2, 0.3, 0.05])       # per-seed constant offset of the mean
        self.eps = np.array([0.04, -0.02, 0.03, -0.01])     # per-seed relative bias of the SD
        self.eta = 0.01                                      # within-window SD wiggle
        n_iters = 10
        alternate = np.where(np.arange(n_iters) % 2 == 0, 1.0, -1.0)
        mean = self.mu_star + self.delta[:, None] + 0 * alternate[None, :]
        std = self.sigma_star * (1 + self.eps[:, None] + self.eta * alternate[None, :])
        self.track = {"mean": mean[..., None], "std": std[..., None]}
        self.n_iters = n_iters

    def metrics(self, track=None, best_mean=None):
        best_mean = np.array([self.mu_star]) if best_mean is None else best_mean
        out, _, _ = bias_io.per_coordinate_metrics(track or self.track, best_mean, np.array([self.sigma_star]),
                                                    window=self.n_iters)
        return {k: float(v[0]) for k, v in out.items()}

    def test_known_values(self):
        m = self.metrics()
        self.assertAlmostEqual(m["mean_bias"], np.mean(np.abs(self.delta)) / self.mu_star)
        self.assertAlmostEqual(m["std_bias"], np.mean(np.abs(self.eps)))
        self.assertAlmostEqual(m["mean_std_units"], np.mean(np.abs(self.delta)) / self.sigma_star)
        self.assertAlmostEqual(m["mean_between"], np.std(self.delta, ddof=1) / self.mu_star)
        self.assertAlmostEqual(m["std_between"], np.std(self.eps, ddof=1))
        self.assertAlmostEqual(m["mean_within"], 0.0)
        # alternating +-eta over an even window: variance (ddof=1) = eta^2 * n / (n - 1)
        self.assertAlmostEqual(m["std_within"], self.eta * np.sqrt(self.n_iters / (self.n_iters - 1)))

    def test_only_the_window_is_used(self):
        track = {k: np.concatenate([np.full_like(v, 1e6), v], axis=1) for k, v in self.track.items()}
        self.assertEqual(self.metrics(track), self.metrics())

    def test_zero_reference_mean_is_undefined_not_infinite(self):
        m = self.metrics(best_mean=np.array([0.0]))
        self.assertTrue(np.isnan(m["mean_bias"]))
        self.assertTrue(np.isfinite(m["mean_std_units"]))

    def test_a_failed_seed_is_left_out(self):
        """A seed that is NaN throughout (e.g. failed Stan checkpoints) is dropped from every metric.
        The notebook marks such cells with the failed-seed footnote; this pins the dropping itself."""
        track = {k: np.concatenate([v, np.full((1,) + v.shape[1:], np.nan)]) for k, v in self.track.items()}
        for key, value in self.metrics(track).items():
            self.assertAlmostEqual(value, self.metrics()[key], msg=key)


if __name__ == "__main__":
    unittest.main()
