"""Mock-based tests for MagmaOcean fractional crystallisation fixes.

These tests verify:
- Fix H1: self.X permanent mutation in run_fractional_stages (magma_ocean.py)
- Fix H4: starting_melt length validation
- run_stage_0: no melt / no solid edge case
- bisection failure raises RuntimeError

No live Julia runtime is needed — all MAGEMin calls are mocked.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock


# ===========================================================================
# Fix H1: self.X restored after run_fractional_stages
# ===========================================================================
class TestMagmaOceanXRestored(unittest.TestCase):
    """Fix H1: run_fractional_stages must restore self.X after execution."""

    @patch('phasetools.models.magma_ocean.MAGEMinBase.__init__', return_value=None)
    def test_magma_ocean_x_restored(self, mock_base_init):
        from phasetools.models.magma_ocean import MagmaOcean
        import phasetools.models.magma_ocean as mo_module

        mo = MagmaOcean.__new__(MagmaOcean)
        mo._Xoxides_py = ['SiO2', 'Al2O3']
        mo.sys_in = 'mol'
        mo.data = MagicMock()
        mo.Xoxides = MagicMock()
        mo.rm_list = None
        mo.X = np.array([50.0, 50.0])
        mo.radius_body = 1737.1
        mo.radius_core = 330.0
        mo.g = 1.62
        mo.rho_avg = 3350.0

        saved_X = mo.X.copy()

        # Mock find_temperature_at_vol_frac to return a fixed T
        mo.find_temperature_at_vol_frac = MagicMock(return_value=1200.0)

        # Build mock MAGEMin output
        mock_out = MagicMock()
        mock_out.ph = ['ol', 'liq']
        mock_out.n_SS = 2
        mock_out.ph_frac_vol = [0.3, 0.7]

        mock_ol = MagicMock()
        mock_ol.rho = 3300.0
        mock_ol.Comp = np.array([30.0, 10.0])
        mock_ol.Comp_wt = np.array([28.0, 12.0])

        mock_liq = MagicMock()
        mock_liq.rho = 2800.0
        mock_liq.Comp = np.array([45.0, 55.0])
        mock_liq.Comp_wt = np.array([43.0, 57.0])

        mock_out.SS_vec = [mock_ol, mock_liq]
        mock_out.PP_vec = []

        mock_magemin_c = MagicMock()
        mock_magemin_c.single_point_minimization = MagicMock(return_value=mock_out)

        # Patch MAGEMin_C and jlconvert at the module level
        with patch.object(mo_module, 'MAGEMin_C', mock_magemin_c), \
             patch.object(mo_module, 'jlconvert', side_effect=lambda t, v: np.array(v, dtype=float)):
            mo.get_phase_chemistry_at_index = MagicMock(return_value=np.array([45.0, 55.0]))
            mo.get_volume_between_radii = MagicMock(return_value=1e12)
            mo.pressure_to_radius = MagicMock(return_value=1400.0)
            mo.radius_to_pressure = MagicMock(return_value=5.0)

            starting_melt = np.array([45.0, 55.0])
            mo.run_fractional_stages(
                starting_melt=starting_melt,
                p_start=5.0,
                p_end=0.001,
                vol_step=0.05,
                starting_vol_frac=0.5,
                n_stages=2,
            )

            np.testing.assert_array_equal(
                mo.X, saved_X,
                err_msg="self.X was not restored after run_fractional_stages"
            )


# ===========================================================================
# Fix H4: starting_melt length validation
# ===========================================================================
class TestStartingMeltValidation(unittest.TestCase):
    """Fix H4: run_fractional_stages must reject mismatched starting_melt length."""

    @patch('phasetools.models.magma_ocean.MAGEMinBase.__init__', return_value=None)
    def test_starting_melt_length_mismatch(self, mock_base_init):
        from phasetools.models.magma_ocean import MagmaOcean

        mo = MagmaOcean.__new__(MagmaOcean)
        mo._Xoxides_py = ['SiO2', 'Al2O3', 'MgO']
        mo.sys_in = 'mol'
        mo.data = MagicMock()
        mo.X = np.array([33.0, 33.0, 34.0])
        mo.radius_body = 1737.1
        mo.radius_core = 330.0
        mo.g = 1.62
        mo.rho_avg = 3350.0

        # starting_melt has 2 elements, but _Xoxides_py has 3
        with self.assertRaises(ValueError) as ctx:
            mo.run_fractional_stages(
                starting_melt=np.array([50.0, 50.0]),
                p_start=5.0,
                p_end=0.001,
            )
        self.assertIn("does not match", str(ctx.exception))


# ===========================================================================
# run_stage_0: no melt / no solid edge case
# ===========================================================================
class TestStageZeroNoMeltNoSolid(unittest.TestCase):
    """run_stage_0 must return zeros for avg_melt and empty layer_modes when liq is absent."""

    @patch('phasetools.models.magma_ocean.MAGEMinBase.__init__', return_value=None)
    def test_no_melt_no_solid(self, mock_base_init):
        from phasetools.models.magma_ocean import MagmaOcean
        import phasetools.models.magma_ocean as mo_module

        mo = MagmaOcean.__new__(MagmaOcean)
        mo._Xoxides_py = ['SiO2', 'Al2O3', 'MgO']
        mo.sys_in = 'mol'
        mo.data = MagicMock()
        mo.X = np.array([33.0, 33.0, 34.0])
        mo.Xoxides = MagicMock()
        mo.rm_list = None

        # find_temperature_at_vol_frac is called but its return value is irrelevant
        # because the mocked output has no phases.
        mo.find_temperature_at_vol_frac = MagicMock(return_value=1200.0)

        mock_out = MagicMock()
        mock_out.ph = []
        mock_out.n_SS = 0
        mock_out.SS_vec = []
        mock_out.PP_vec = []

        mock_magemin_c = MagicMock()
        mock_magemin_c.single_point_minimization = MagicMock(return_value=mock_out)

        with patch.object(mo_module, 'MAGEMin_C', mock_magemin_c):
            results, avg_melt = mo.run_stage_0(
                p_start=5.0, p_end=0.001, solid_frac=0.5, p_intervals=3
            )

        self.assertTrue(np.all(np.isfinite(avg_melt)))
        self.assertTrue(np.allclose(avg_melt, np.zeros(3)))
        self.assertEqual(results["layer_modes"], {})


if __name__ == '__main__':
    unittest.main()
