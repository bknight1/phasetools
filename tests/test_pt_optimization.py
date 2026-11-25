"""Mock-based tests for generic P-T optimisation and its calculator callers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from phasetools import PhasePTEstimator, optimize_pt


class TestOptimizePT(unittest.TestCase):
    def test_quadratic_objective_finds_minimum_and_records_path(self):
        result = optimize_pt(
            lambda pt: float((pt[0] - 2.0) ** 2 + (pt[1] - 650.0) ** 2),
            [(1.0, 3.0), (600.0, 700.0)],
            method="L-BFGS-B",
            x0=[1.5, 625.0],
            return_path=True,
        )

        np.testing.assert_allclose(result.x, [2.0, 650.0], atol=1e-4)
        self.assertLess(result.fun, 1e-8)
        self.assertEqual(result.path.ndim, 2)
        self.assertEqual(result.path.shape[1], 2)

    def test_requires_two_bounds(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            optimize_pt(lambda _pt: 0.0, [(0.0, 1.0)])


class TestPhasePTEstimatorDelegation(unittest.TestCase):
    def test_solve_uses_shared_optimizer_and_still_attaches_composition(self):
        estimator = object.__new__(PhasePTEstimator)
        estimator.calculate_misfit = lambda pt, *_args: float(
            (pt[0] - 2.0) ** 2 + (pt[1] - 650.0) ** 2
        )
        estimator._get_phase_composition = lambda *_args: np.asarray([0.4, 0.6])
        estimator.data = object()
        estimator.X = []
        estimator.Xoxides = []
        estimator.sys_in = "wt"
        estimator.rm_list = []
        fake_magemin = SimpleNamespace(
            single_point_minimization=lambda *_args, **_kwargs: object()
        )

        with patch(
            "phasetools.calculators.pt_estimation.phasetools.MAGEMin_C", fake_magemin
        ):
            result = estimator.solve(
                np.asarray([0.4, 0.6]),
                "g",
                ["py", "alm"],
                "endmember",
                [(1.0, 3.0), (600.0, 700.0)],
                method="L-BFGS-B",
                x0=[1.5, 625.0],
            )

        np.testing.assert_allclose(result.x, [2.0, 650.0], atol=1e-4)
        np.testing.assert_allclose(result.modelled_composition, [0.4, 0.6])


if __name__ == "__main__":
    unittest.main()
