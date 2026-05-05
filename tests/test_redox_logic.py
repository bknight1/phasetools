import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock MAGEMin_C before importing PhaseTools components that depend on it
import sys
mock_magemin = MagicMock()
sys.modules['phasetools.MAGEMin_C'] = mock_magemin

from phasetools.core.base import MAGEMinBase

class TestRedoxLogic(unittest.TestCase):
    def setUp(self):
        # Prevent Initialize_MAGEMin from calling Julia
        with patch('phasetools.MAGEMin_C.Initialize_MAGEMin', return_value=MagicMock()):
            self.base = MAGEMinBase()

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_garnet_redox_split(self, mock_get_apfu):
        """Test Garnet with Fe3+ in a redox model (O convention)."""
        # Scenario: Fe_total (from FeO) = 1.715, O = 12.023
        mock_get_apfu.return_value = {
            'FeO': 1.715,
            'O': 12.023,
            'Fe2O3': 0.0
        }
        
        # We don't need a real 'out' object because get_oxide_apfu is mocked
        result = self.base._extract_fe_split_from_apfu(None, 'g')
        
        # Expected:
        # total_fe = 1.715
        # excess_o = 12.023 - 12.0 = 0.023
        # fe3 = 2 * 0.023 = 0.046
        # fe2 = 1.715 - 0.046 = 1.669
        self.assertAlmostEqual(result['fe3'], 0.046, places=3)
        self.assertAlmostEqual(result['fe2'], 1.669, places=3)

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_pyroxene_redox_split(self, mock_get_apfu):
        """Test Pyroxene with significant Acmite component."""
        # Scenario: Fe_total = 1.0, O = 6.05
        mock_get_apfu.return_value = {
            'FeO': 1.0,
            'O': 6.05,
            'Fe2O3': 0.0
        }
        
        result = self.base._extract_fe_split_from_apfu(None, 'dio')
        
        # Expected:
        # excess_o = 6.05 - 6.0 = 0.05
        # fe3 = 0.1
        # fe2 = 0.9
        self.assertAlmostEqual(result['fe3'], 0.1, places=2)
        self.assertAlmostEqual(result['fe2'], 0.9, places=2)

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_spinel_standard_split(self, mock_get_apfu):
        """Test Spinel using the traditional Fe2O3 convention."""
        # Scenario: FeO = 0.8, Fe2O3 = 0.1, O = 0.0 (baseline 4.0)
        mock_get_apfu.return_value = {
            'FeO': 0.8,
            'Fe2O3': 0.1,
            'O': 0.0 # Standard model might return 0 or absent
        }
        
        result = self.base._extract_fe_split_from_apfu(None, 'sp')
        
        # Expected:
        # total_fe = 0.8 + 2*0.1 = 1.0
        # fe3 = 2 * 0.1 = 0.2
        # fe2 = 0.8
        self.assertAlmostEqual(result['fe3'], 0.2, places=2)
        self.assertAlmostEqual(result['fe2'], 0.8, places=2)

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_negative_clamp(self, mock_get_apfu):
        """Test that negative Fe2+ (due to rounding/noise) is clamped to 0."""
        # Scenario: O has high noise, suggesting more Fe3+ than total Fe
        mock_get_apfu.return_value = {
            'FeO': 0.01,
            'O': 6.05, # Suggests Fe3 = 0.1
            'Fe2O3': 0.0
        }
        
        result = self.base._extract_fe_split_from_apfu(None, 'dio')
        
        self.assertEqual(result['fe2'], 0.0)
        self.assertAlmostEqual(result['fe3'], 0.1, places=2)

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_sb24_iron_handling(self, mock_get_apfu):
        """Test iron handling for the sb24 database (uses 'Fe' and 'O')."""
        # Scenario: Fe = 1.0, O = 1.05 (excess O = 0.05)
        mock_get_apfu.return_value = {
            'Fe': 1.0,
            'O': 1.05,
            'FeO': 0.0,
            'Fe2O3': 0.0
        }
        
        result = self.base._extract_fe_split_from_apfu(None, 'metal')
        
        # total_fe should be 1.0 (from 'Fe')
        # fe3 should be 0.1 (from 2 * 0.05)
        self.assertAlmostEqual(result['fe2'], 0.9, places=2)
        self.assertAlmostEqual(result['fe3'], 0.1, places=2)

    def test_get_phase_mg_number_robustness(self):
        """Test that get_phase_mg_number handles Fe and Fe2O3 components."""
        from phasetools.core.phase_properties import get_phase_mg_number
        
        # Case 1: 'Fe' component (sb24)
        mock_out_fe = MagicMock()
        mock_out_fe.ph = ['ol']
        mock_out_fe.oxides = ['MgO', 'Fe']
        mock_ol_fe = MagicMock()
        mock_ol_fe.Comp_apfu = [1.0, 1.0] # MgO=1, Fe=1
        mock_out_fe.SS_vec = [mock_ol_fe]
        self.assertAlmostEqual(get_phase_mg_number(mock_out_fe, 'ol'), 0.5, places=2)

        # Case 2: 'Fe2O3' component (traditional)
        mock_out_fe2o3 = MagicMock()
        mock_out_fe2o3.ph = ['hem']
        mock_out_fe2o3.oxides = ['MgO', 'Fe2O3']
        mock_hem = MagicMock()
        mock_hem.Comp_apfu = [1.0, 0.5] # MgO=1, Fe2O3=0.5 => Fe=1.0
        mock_out_fe2o3.SS_vec = [mock_hem]
        self.assertAlmostEqual(get_phase_mg_number(mock_out_fe2o3, 'hem'), 0.5, places=2)

    def test_get_phase_mg_number_divalent(self):
        """Test that get_phase_mg_number uses FeO directly (MAGEMin style)."""
        from phasetools.core.phase_properties import get_phase_mg_number
        
        # Mock 'out' object
        mock_out = MagicMock()
        mock_out.ph = ['dio']
        mock_out.oxides = ['MgO', 'FeO']
        
        mock_phase = MagicMock()
        mock_phase.Comp_apfu = [1.0, 1.0] # MgO=1.0, FeO=1.0
        mock_out.SS_vec = [mock_phase]
        
        # MAGEMin logic: Mg# = 1.0 / (1.0 + 1.0) = 0.5
        mg_num = get_phase_mg_number(mock_out, 'dio')
        self.assertAlmostEqual(mg_num, 0.5, places=1)

    def test_calculate_kd_fe_mg(self):
        """Test calculation of Fe-Mg Kd between two phases."""
        from phasetools.core.phase_properties import calculate_kd_fe_mg
        
        # Mock 'out' object
        mock_out = MagicMock()
        mock_out.ph = ['g', 'cpx']
        mock_out.oxides = ['MgO', 'FeO']
        
        # Garnet: MgO=1.0, FeO=0.5 => Mg# = 1/1.5 = 2/3, Fe/Mg = 0.5
        mock_g = MagicMock()
        mock_g.Comp_apfu = [1.0, 0.5]
        
        # Cpx: MgO=1.0, FeO=0.25 => Mg# = 1/1.25 = 0.8, Fe/Mg = 0.25
        mock_cpx = MagicMock()
        mock_cpx.Comp_apfu = [1.0, 0.25]
        
        mock_out.SS_vec = [mock_g, mock_cpx]
        
        # Kd = (Fe/Mg)_g / (Fe/Mg)_cpx = 0.5 / 0.25 = 2.0
        kd = calculate_kd_fe_mg(mock_out, 'g', 'cpx')
        self.assertAlmostEqual(kd, 2.0, places=1)

if __name__ == '__main__':
    unittest.main()
