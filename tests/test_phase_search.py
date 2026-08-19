"""Mock-based tests for ``phasetools.calculators.phase_search``.

These tests verify the ``PhaseFunctions.fractionate_phase`` fixes:
- pure-phase IndexError (PP_vec handling);
- solution-phase path still works (SS_vec handling);
- frac_amount=0 returns the bulk unchanged.

No live Julia runtime is needed — all MAGEMin calls are mocked.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

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
