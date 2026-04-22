# this tests our saving and loading csv functionality

import math
import os
import tempfile
import unittest

import numpy as np

from modulars.utils import save_to_csv, load_from_csv


def assert_nested_equal(test_case, left, right):
    if isinstance(left, np.ndarray):
        test_case.assertIsInstance(right, np.ndarray)
        test_case.assertEqual(left.dtype, right.dtype)
        test_case.assertEqual(left.shape, right.shape)

        if np.issubdtype(left.dtype, np.floating):
            test_case.assertTrue(
                np.allclose(left, right, equal_nan=True),
                msg=f"we expected arrays to match:\nleft={left}\nright={right}",
            )
        else:
            test_case.assertTrue(
                np.array_equal(left, right),
                msg=f"we expected arrays to match:\nleft={left}\nright={right}",
            )
        return

    if isinstance(left, tuple):
        test_case.assertIsInstance(right, tuple)
        test_case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_equal(test_case, a, b)
        return

    if isinstance(left, list):
        test_case.assertIsInstance(right, list)
        test_case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_equal(test_case, a, b)
        return

    if isinstance(left, dict):
        test_case.assertIsInstance(right, dict)
        test_case.assertEqual(set(left.keys()), set(right.keys()))
        for key in left:
            assert_nested_equal(test_case, left[key], right[key])
        return

    if isinstance(left, float) and math.isnan(left):
        test_case.assertTrue(math.isnan(right))
        return

    test_case.assertEqual(left, right)


class TestCsvRoundTrip(unittest.TestCase):
    def test_save_and_load_round_trip_for_restart_trajectories(self):
        results = [
            (
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
                np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
                np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64),
            ),
            (
                np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float64),
                np.array([[0.9, 1.0], [1.1, 1.2]], dtype=np.float64),
                np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float64),
                np.array([[1.3, 1.4], [1.5, 1.6]], dtype=np.float64),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, "restart_results.csv")

            save_to_csv(file_name, results)
            loaded_results = load_from_csv(file_name)

        self.assertEqual(len(results), len(loaded_results))
        for original, loaded in zip(results, loaded_results):
            assert_nested_equal(self, original, loaded)

    def test_save_and_load_round_trip_for_summary_runs(self):
        results = [
            (
                -123.45,
                {
                    "mean": np.array([0.2, 0.3, 0.5], dtype=np.float64),
                    "std": np.array([0.05, 0.04, 0.06], dtype=np.float64),
                    "cov": np.array(
                        [
                            [0.0025, 0.0002, 0.0001],
                            [0.0002, 0.0016, 0.0003],
                            [0.0001, 0.0003, 0.0036],
                        ],
                        dtype=np.float64,
                    ),
                },
            ),
            (
                -120.00,
                {
                    "mean": np.array([0.25, 0.25, 0.5], dtype=np.float64),
                    "std": np.array([0.03, 0.03, 0.04], dtype=np.float64),
                    "cov": np.array(
                        [
                            [0.0009, 0.0001, 0.0002],
                            [0.0001, 0.0009, 0.0002],
                            [0.0002, 0.0002, 0.0016],
                        ],
                        dtype=np.float64,
                    ),
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, "summary_results.csv")

            save_to_csv(file_name, results)
            loaded_results = load_from_csv(file_name)

        self.assertEqual(len(results), len(loaded_results))
        for original, loaded in zip(results, loaded_results):
            assert_nested_equal(self, original, loaded)

    def test_save_and_load_preserves_restart_order(self):
        results = [
            {"restart_name": "first", "mean": np.array([1.0, 2.0])},
            {"restart_name": "second", "mean": np.array([3.0, 4.0])},
            {"restart_name": "third", "mean": np.array([5.0, 6.0])},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, "ordered_results.csv")

            save_to_csv(file_name, results)
            loaded_results = load_from_csv(file_name)

        loaded_names = [result["restart_name"] for result in loaded_results]
        self.assertEqual(loaded_names, ["first", "second", "third"])
