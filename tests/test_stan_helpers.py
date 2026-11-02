# Tests for the CmdStan output helpers in modulars/stan_rr_test.py (no CmdStan needed).

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import _support  # noqa: F401
from modulars.stan_rr_test import tracked_iterations, read_variational_draws, read_variational_moments, _column_lookup


class TestTrackedIterations(unittest.TestCase):
    def test_grid(self):
        np.testing.assert_array_equal(tracked_iterations(50, track_every=10), [1, 10, 20, 30, 40, 50])
        np.testing.assert_array_equal(tracked_iterations(45, track_every=10), [1, 10, 20, 30, 40, 45])
        np.testing.assert_array_equal(tracked_iterations(5, track_every=10), [1, 5])

    def test_tail_window_every_iteration(self):
        # the overall-bias Stan runner tracks iterations max_iters - window + 1 .. max_iters
        np.testing.assert_array_equal(tracked_iterations(25_000, track_every=1, min_iter=24_998), [24_998, 24_999, 25_000])


CSV = """# stan_version_major = 2
# method = variational
lp__,log_p__,log_g__,theta[1],theta[2],sd_y
0,0,0,0.5,0.5,10.0
0,-1.2,-0.4,0.4,0.6,1.0
0,-1.1,-0.3,0.6,0.4,100.0
0,-1.3,-0.2,0.5,0.5,10.0
"""


class TestReadVariationalDraws(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "fit.csv"
        self.path.write_text(CSV)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mean_row_is_dropped(self):
        """CmdStan's first output row is the approximation's mean, not a draw; keeping it would bias the SD."""
        draws = read_variational_draws(self.path, ["theta[1]", "theta[2]"])
        np.testing.assert_array_equal(draws, [[0.4, 0.6], [0.6, 0.4], [0.5, 0.5]])

    def test_log_columns_are_logged(self):
        draws = read_variational_draws(self.path, ["sd_y"], log_columns=["sd_y"])
        np.testing.assert_allclose(draws[:, 0], np.log([1.0, 100.0, 10.0]))

    def test_moments_use_sample_sd(self):
        mean, sd = read_variational_moments(self.path, ["theta[1]"])
        self.assertAlmostEqual(mean[0], 0.5)
        self.assertAlmostEqual(sd[0], np.std([0.4, 0.6, 0.5], ddof=1))

    def test_column_name_styles(self):
        header = ["lp__", "theta.1", "theta.2"]
        self.assertEqual(_column_lookup(header, "theta[2]"), 2)
        self.assertEqual(_column_lookup(["theta[1]"], "theta.1"), 0)
        with self.assertRaises(ValueError):
            _column_lookup(header, "phi[1]")


if __name__ == "__main__":
    unittest.main()
