"""Mock-based tests for full-fraction (Rayleigh / Perple_X) garnet fractionation.

``gt_along_path(fractionate=True)`` removes the **entire** garnet fraction
present on the reactive bulk at each P-T point — instantaneous segregation of
refractory garnet — except for the retained seed baseline when
``normalise_start=True``.  These tests pin:

* full-fraction removal (current minus retained seed, NOT a step-to-step
  difference — the old scheme removed only the record-breaking increment);
* the seed rules of ``normalise_start`` (retained vs stripped at step 0);
* the returned ``gt_*_frac`` arrays as the **cumulative segregated pile**
  per basis (fraction of the original system), including mol/wt/vol basis
  conversion of each removal;
* ``fractionate=False`` still returns the equilibrium modes unchanged and
  never calls ``fractionate_phase``.

No live Julia runtime is needed -- all MAGEMin calls are mocked.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from phasetools.calculators.garnet import MAGEMinGarnetCalculator


def _make_calc(sys_in="mol"):
    calc = MAGEMinGarnetCalculator.__new__(MAGEMinGarnetCalculator)
    calc.db = "ig"
    calc.dataset = 636
    calc.verbose = False
    calc.sys_in = sys_in
    calc.X = np.array([50.0, 50.0])
    calc.Xoxides = MagicMock()
    calc._Xoxides_py = ["SiO2", "Al2O3"]
    calc.rm_list = None
    calc.data = MagicMock()
    return calc


def _point(gt_mol, gt_wt, gt_vol):
    """One mocked _gt_single_point_from_jl return value."""
    return (gt_mol, gt_wt, gt_vol, 0.30, 0.05, 0.40, 0.25, MagicMock())


def _run(calc, points, fractionate=True, normalise_start=True):
    calc._gt_single_point_from_jl = MagicMock(side_effect=list(points))
    with (
        patch("phasetools.calculators.phase_search.PhaseFunctions") as pf_cls,
        patch("phasetools.calculators.garnet.calculator.jlconvert") as mock_jlconvert,
    ):
        pf_cls.fractionate_phase = MagicMock(return_value=np.array([50.0, 50.0]))
        mock_jlconvert.side_effect = lambda t, v: np.array(v, dtype=float)
        P = np.arange(len(points), dtype=float) + 10.0
        T = np.arange(len(points), dtype=float) + 800.0
        result = calc.gt_along_path(P, T, fractionate=fractionate,
                                    normalise_start=normalise_start)
    calls = [c.kwargs["frac_amount"] for c in pf_cls.fractionate_phase.call_args_list]
    return result, calls, pf_cls


class TestFullFractionRemoval(unittest.TestCase):
    """No seed (path starts garnet-free): every step removes its full mode."""

    def test_removes_entire_fraction_each_step(self):
        calc = _make_calc("mol")
        points = [
            _point(0.00, 0.00, 0.00),   # outside the garnet field
            _point(0.12, 0.10, 0.08),   # full mode segregated
            _point(0.15, 0.12, 0.10),
        ]
        (_, _, _, *_), calls, _ = _run(calc, points)
        self.assertEqual(len(calls), 2, "one removal per garnet-bearing step")
        self.assertAlmostEqual(calls[0], 0.12, places=12,
                               msg="step 1 must remove the ENTIRE current fraction")
        self.assertAlmostEqual(calls[1], 0.15, places=12,
                               msg="step 2 removes its full mode, not mode minus previous "
                                   "(the old running-max scheme would give 0.03)")

    def test_returns_cumulative_pile_per_basis(self):
        calc = _make_calc("mol")
        points = [
            _point(0.00, 0.00, 0.00),
            _point(0.12, 0.10, 0.08),
            _point(0.15, 0.12, 0.10),
        ]
        (mol, wt, vol, *_), _, _ = _run(calc, points)

        # pile_i = pile_{i-1} + remaining_{i-1} * fraction_of_current_system
        self.assertAlmostEqual(mol[0], 0.0, places=12)
        self.assertAlmostEqual(mol[1], 0.12, places=12)
        self.assertAlmostEqual(mol[2], 0.12 + 0.88 * 0.15, places=12)
        self.assertAlmostEqual(wt[2], 0.10 + 0.90 * 0.12, places=12)
        self.assertAlmostEqual(vol[2], 0.08 + 0.92 * 0.10, places=12)

        # the pile is monotone — the growth record must never resorb
        for arr in (mol, wt, vol):
            self.assertTrue(np.all(np.diff(arr) >= -1e-15), "pile must be monotone")

    def test_x_along_path_rows_normalised(self):
        calc = _make_calc("mol")
        points = [_point(0.0, 0.0, 0.0), _point(0.12, 0.10, 0.08)]
        result, _, _ = _run(calc, points)
        X_along_path = result[-1]
        for i in range(len(points)):
            self.assertAlmostEqual(float(np.sum(X_along_path[i])), 1.0, places=10)


class TestRetainedSeedBaseline(unittest.TestCase):
    """normalise_start=True: the seed stays reactive and is never re-removed."""

    def test_removals_are_current_minus_seed_not_step_difference(self):
        calc = _make_calc("mol")
        points = [
            _point(0.05, 0.04, 0.03),   # seed present at step 0: retained
            _point(0.12, 0.10, 0.08),
            _point(0.15, 0.12, 0.10),
        ]
        (mol, *_), calls, _ = _run(calc, points, normalise_start=True)
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(calls[0], 0.12 - 0.05, places=12,
                               msg="step 1: current minus retained seed")
        self.assertAlmostEqual(calls[1], 0.15 - 0.05, places=12,
                               msg="step 2: current minus seed baseline — the old "
                                   "running-max scheme would give 0.15-0.12=0.03")
        # seed itself is not part of the segregated pile
        self.assertAlmostEqual(mol[0], 0.0, places=12)
        self.assertAlmostEqual(mol[1], 0.07, places=12)
        self.assertAlmostEqual(mol[2], 0.07 + 0.93 * 0.10, places=12)

    def test_seed_stripped_at_step_zero_when_normalise_start_false(self):
        calc = _make_calc("mol")
        points = [
            _point(0.05, 0.04, 0.03),   # overstepped seed: stripped immediately
            _point(0.12, 0.10, 0.08),
        ]
        (mol, *_), calls, _ = _run(calc, points, normalise_start=False)
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(calls[0], 0.05, places=12,
                               msg="seed removed at step 0")
        self.assertAlmostEqual(calls[1], 0.12, places=12,
                               msg="baseline is zero once the seed is gone: full fraction")
        self.assertAlmostEqual(mol[0], 0.05, places=12)
        self.assertAlmostEqual(mol[1], 0.05 + 0.95 * 0.12, places=12)


class TestBasisConversion(unittest.TestCase):
    """wt-basis fractionation converts each removal into all three piles."""

    def test_wt_removal_converted_to_mol_and_vol_piles(self):
        calc = _make_calc("wt")
        points = [
            _point(0.00, 0.00, 0.00),
            _point(0.12, 0.07, 0.10),   # mol=0.12, wt=0.07, vol=0.10
        ]
        (mol, wt, vol, *_), calls, _ = _run(calc, points, normalise_start=True)
        self.assertAlmostEqual(calls[0], 0.07, places=12,
                               msg="wt basis: full current wt fraction")
        self.assertAlmostEqual(wt[1], 0.07, places=12)
        # pile increments scale by this step's equilibrium mode ratios
        self.assertAlmostEqual(mol[1], 0.07 * (0.12 / 0.07), places=12)
        self.assertAlmostEqual(vol[1], 0.07 * (0.10 / 0.07), places=12)


class TestNoFractionationUnchanged(unittest.TestCase):
    """fractionate=False: equilibrium modes returned, no fractionation call."""

    def test_equilibrium_modes_and_no_fractionation_helper(self):
        calc = _make_calc("mol")
        points = [
            _point(0.05, 0.04, 0.03),
            _point(0.12, 0.10, 0.08),
        ]
        calc._gt_single_point_from_jl = MagicMock(side_effect=list(points))
        with (
            patch("phasetools.calculators.phase_search.PhaseFunctions") as pf_cls,
            patch("phasetools.calculators.garnet.calculator.jlconvert") as mock_jlconvert,
        ):
            mock_jlconvert.side_effect = lambda t, v: np.array(v, dtype=float)
            P = np.array([10.0, 11.0])
            T = np.array([800.0, 810.0])
            mol, wt, vol, *_ = calc.gt_along_path(P, T, fractionate=False)
            pf_cls.assert_not_called()
            pf_cls.fractionate_phase.assert_not_called()

        np.testing.assert_allclose(mol, [0.05, 0.12])
        np.testing.assert_allclose(wt, [0.04, 0.10])
        np.testing.assert_allclose(vol, [0.03, 0.08])


if __name__ == "__main__":
    unittest.main()
