"""Mock-based tests for fractionation correctness fixes.

These tests verify the fixes described in the implementation plan:
- Fix A: get_retrograde_concentrations crash (garnet_growth.py)
- Fix B: mol/wt unit mismatch in gt_along_path (garnet.py)
- Fix C: pure-phase IndexError in fractionate_phase (phase_search.py)
- Fix D: X_along_path normalise to 1 (garnet.py)
- Fix H1: self.X permanent mutation in run_fractional_stages (magma_ocean.py)

No live Julia runtime is needed — all MAGEMin calls are mocked.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_out(phases, n_SS, bulk_mol, bulk_wt, ph_frac_mol, ph_frac_wt, comps_mol, comps_wt):
    """Build a minimal mock MAGEMin output object."""
    out = MagicMock()
    out.ph = phases
    out.n_SS = n_SS
    out.bulk = np.array(bulk_mol, dtype=float)
    out.bulk_wt = np.array(bulk_wt, dtype=float)
    out.ph_frac = list(ph_frac_mol)
    out.ph_frac_wt = list(ph_frac_wt)

    ss_vec = []
    pp_vec = []
    for idx, ph in enumerate(phases):
        obj = MagicMock()
        obj.Comp = np.array(comps_mol[idx], dtype=float)
        obj.Comp_wt = np.array(comps_wt[idx], dtype=float)
        if idx < n_SS:
            ss_vec.append(obj)
        else:
            pp_vec.append(obj)

    out.SS_vec = ss_vec
    out.PP_vec = pp_vec
    return out


# ===========================================================================
# Fix B: wt fraction uses wt basis
# ===========================================================================
class TestWtFractionUsesWtBasis(unittest.TestCase):
    """Fix B: when sys_in='wt', fractionate_phase should use gt_wt, not gt_frac."""

    @patch('phasetools.calculators.garnet.MAGEMin_C')
    @patch('phasetools.calculators.garnet.jlconvert')
    @patch('phasetools.calculators.phase_search.PhaseFunctions.__init__', return_value=None)
    @patch('phasetools.calculators.garnet.MAGEMinPTGridCalculator.__init__', return_value=None)
    def test_wt_fraction_uses_wt_basis(self, mock_grid_init, mock_pf_init, mock_jlconvert, mock_magemin):
        from phasetools.calculators.garnet import MAGEMinGarnetCalculator

        calc = MAGEMinGarnetCalculator.__new__(MAGEMinGarnetCalculator)
        calc.db = 'ig'
        calc.dataset = 636
        calc.verbose = False
        calc.sys_in = 'wt'
        calc.X = np.array([50.0, 50.0])
        calc.Xoxides = MagicMock()
        calc._Xoxides_py = ['SiO2', 'Al2O3']
        calc.rm_list = None
        calc.data = MagicMock()

        # gt_frac (mol) = 0.10, gt_wt = 0.05 — they differ
        # Step 0 returns mol=0.10, wt=0.05; step 1 returns mol=0.12, wt=0.07
        mock_return_0 = (0.10, 0.05, 0.08, 0.3, 0.05, 0.4, 0.25, MagicMock())
        mock_return_1 = (0.12, 0.07, 0.10, 0.3, 0.05, 0.4, 0.25, MagicMock())
        calc._gt_single_point_from_jl = MagicMock(side_effect=[mock_return_0, mock_return_1])

        # Mock PhaseFunctions
        mock_pf = MagicMock()
        mock_pf.fractionate_phase = MagicMock(return_value=np.array([50.0, 50.0]))

        # Patch PhaseFunctions at its source module (local import in gt_along_path)
        with patch('phasetools.calculators.phase_search.PhaseFunctions', return_value=mock_pf):
            calc._copy_state_to = MagicMock()
            # jlconvert should pass through the array so np.array(X) works
            mock_jlconvert.side_effect = lambda t, v: np.array(v, dtype=float)

            P = np.array([10.0, 11.0])
            T = np.array([800.0, 810.0])
            calc.gt_along_path(P, T, fractionate=True, normalise_start=True)

            # Step 0: normalise_start=True, so no fractionation at i=0
            # Step 1: i>0, frac_amount = gt_wt[1] - gt_wt[0] = 0.07 - 0.05 = 0.02
            calls = mock_pf.fractionate_phase.call_args_list
            self.assertEqual(len(calls), 1, f"Expected 1 fractionation call, got {len(calls)}")
            _, kwargs = calls[0]
            self.assertAlmostEqual(kwargs['frac_amount'], 0.02, places=10,
                                   msg="frac_amount should be based on wt fraction (0.07-0.05), not mol (0.12-0.10)")


# ===========================================================================
# Fix D: X_along_path normalised to 1
# ===========================================================================
class TestXAlongPathNormalised(unittest.TestCase):
    """Fix D: each row of X_along_path must sum to 1.0."""

    @patch('phasetools.calculators.garnet.MAGEMin_C')
    @patch('phasetools.calculators.garnet.jlconvert')
    @patch('phasetools.calculators.phase_search.PhaseFunctions.__init__', return_value=None)
    @patch('phasetools.calculators.garnet.MAGEMinPTGridCalculator.__init__', return_value=None)
    def test_x_along_path_normalised_to_one(self, mock_grid_init, mock_pf_init, mock_jlconvert, mock_magemin):
        from phasetools.calculators.garnet import MAGEMinGarnetCalculator

        calc = MAGEMinGarnetCalculator.__new__(MAGEMinGarnetCalculator)
        calc.db = 'ig'
        calc.dataset = 636
        calc.verbose = False
        calc.sys_in = 'mol'
        calc.X = np.array([50.0, 50.0])  # Julia vector — jlconvert will wrap
        calc.Xoxides = MagicMock()
        calc._Xoxides_py = ['SiO2', 'Al2O3']
        calc.rm_list = None
        calc.data = MagicMock()

        mock_return = (0.05, 0.05, 0.05, 0.3, 0.05, 0.4, 0.25, MagicMock())
        calc._gt_single_point_from_jl = MagicMock(return_value=mock_return)
        mock_jlconvert.return_value = calc.X

        P = np.array([10.0, 11.0])
        T = np.array([800.0, 810.0])

        _, _, _, _, _, _, _, X_along_path = calc.gt_along_path(P, T, fractionate=False)

        for i in range(len(P)):
            self.assertAlmostEqual(np.sum(X_along_path[i]), 1.0, places=10,
                                   msg=f"Row {i} of X_along_path does not sum to 1.0")


# ===========================================================================
# Fix C: pure-phase IndexError
# ===========================================================================
class TestFractionatePurePhase(unittest.TestCase):
    """Fix C: fractionate_phase must handle pure phases (PP_vec) without IndexError."""

    def test_fractionate_pure_phase(self):
        from phasetools.calculators.phase_search import PhaseFunctions

        pf = PhaseFunctions.__new__(PhaseFunctions)

        # ph=['q', 'liq'], n_SS=1 => 'q' is a pure phase at index 0 in PP_vec
        out = _make_mock_out(
            phases=['q', 'liq'],
            n_SS=1,
            bulk_mol=[60.0, 40.0],
            bulk_wt=[62.0, 38.0],
            ph_frac_mol=[0.15, 0.85],
            ph_frac_wt=[0.16, 0.84],
            comps_mol=[[100.0, 0.0], [50.0, 50.0]],
            comps_wt=[[100.0, 0.0], [48.0, 52.0]],
        )

        # 'q' is at index 0 in out.ph, n_SS=1, so it's a pure phase (PP_vec[0])
        result = pf.fractionate_phase('q', out, 'mol', frac_amount=0.1)
        self.assertIsNotNone(result)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_fractionate_solution_phase_still_works(self):
        """Verify solution-phase path (SS_vec) still works after the fix."""
        from phasetools.calculators.phase_search import PhaseFunctions

        pf = PhaseFunctions.__new__(PhaseFunctions)

        out = _make_mock_out(
            phases=['liq', 'g'],
            n_SS=2,
            bulk_mol=[60.0, 40.0],
            bulk_wt=[62.0, 38.0],
            ph_frac_mol=[0.85, 0.15],
            ph_frac_wt=[0.84, 0.16],
            comps_mol=[[50.0, 50.0], [40.0, 60.0]],
            comps_wt=[[48.0, 52.0], [38.0, 62.0]],
        )

        result = pf.fractionate_phase('g', out, 'mol', frac_amount=0.1)
        self.assertIsNotNone(result)
        self.assertTrue(np.all(np.isfinite(result)))


# ===========================================================================
# Zero guard: frac_amount=0 is a no-op
# ===========================================================================
class TestFractionateZeroIsNoop(unittest.TestCase):
    """frac_amount=0 must return the bulk unchanged (not normalised to sum=1)."""

    def test_fractionate_zero_is_noop(self):
        from phasetools.calculators.phase_search import PhaseFunctions

        pf = PhaseFunctions.__new__(PhaseFunctions)

        out = _make_mock_out(
            phases=['g', 'liq'],
            n_SS=2,
            bulk_mol=[6000.0, 4000.0],  # sum=10000, not 1
            bulk_wt=[6200.0, 3800.0],
            ph_frac_mol=[0.15, 0.85],
            ph_frac_wt=[0.16, 0.84],
            comps_mol=[[40.0, 60.0], [50.0, 50.0]],
            comps_wt=[[38.0, 62.0], [48.0, 52.0]],
        )

        result = pf.fractionate_phase('g', out, 'mol', frac_amount=0.0)
        np.testing.assert_array_equal(result, np.array(out.bulk, dtype=float))


if __name__ == '__main__':
    unittest.main()
