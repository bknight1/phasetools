# Mock MAGEMin_C before importing PhaseTools components that depend on it
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

mock_magemin = MagicMock()
sys.modules['phasetools.MAGEMin_C'] = mock_magemin

from phasetools.core.base import MAGEMinBase
from phasetools.core.phase_properties import get_oxide_apfu, get_phase_fe_split


class TestRedoxLogic(unittest.TestCase):
    def setUp(self):
        # Prevent Initialize_MAGEMin from calling Julia
        with patch('phasetools.core.base.MAGEMin_C') as mock_mc:
            mock_mc.Initialize_MAGEMin.return_value = MagicMock()
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
        self.assertAlmostEqual(result['Fe3'], 0.046, places=3)
        self.assertAlmostEqual(result['Fe2'], 1.669, places=3)

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
        self.assertAlmostEqual(result['Fe3'], 0.1, places=2)
        self.assertAlmostEqual(result['Fe2'], 0.9, places=2)

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
        self.assertAlmostEqual(result['Fe3'], 0.2, places=2)
        self.assertAlmostEqual(result['Fe2'], 0.8, places=2)

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
        
        self.assertEqual(result['Fe2'], 0.0)
        self.assertAlmostEqual(result['Fe3'], 0.1, places=2)

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
        self.assertAlmostEqual(result['Fe2'], 0.9, places=2)
        self.assertAlmostEqual(result['Fe3'], 0.1, places=2)

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_bankers_rounding_half_integer(self, mock_get_apfu):
        """O APFU at x.5 boundary — banker's rounding gives correct result."""
        mock_get_apfu.return_value = {'FeO': 1.0, 'O': 12.5, 'Fe2O3': 0.0}
        result = self.base._extract_fe_split_from_apfu(None, 'g')
        # int(12.5) = 12, excess_o = 0.5, fe3 = 1.0
        self.assertAlmostEqual(result['Fe3'], 1.0, places=4)
        self.assertAlmostEqual(result['Fe2'], 0.0, places=4)

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_bankers_rounding_above_half(self, mock_get_apfu):
        """O APFU above x.5 — int() truncates, giving correct excess."""
        mock_get_apfu.return_value = {'FeO': 1.0, 'O': 13.5, 'Fe2O3': 0.0}
        result = self.base._extract_fe_split_from_apfu(None, 'g')
        # int(13.5) = 13, excess_o = 0.5, fe3 = 1.0
        self.assertAlmostEqual(result['Fe3'], 1.0, places=4)
        self.assertAlmostEqual(result['Fe2'], 0.0, places=4)

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_mixed_fe2o3_and_o_basis(self, mock_get_apfu):
        """Both Fe2O3 and O present — no double-counting."""
        mock_get_apfu.return_value = {'FeO': 0.5, 'O': 6.1, 'Fe2O3': 0.1}
        result = self.base._extract_fe_split_from_apfu(None, 'dio')
        # excess_o = 0.1, fe3 = 2*0.1 + 2*0.1 = 0.4, total_fe = 0.5, fe2 = 0.1
        self.assertAlmostEqual(result['Fe3'], 0.4, places=4)
        self.assertAlmostEqual(result['Fe2'], 0.1, places=4)

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_no_iron_phase(self, mock_get_apfu):
        """Phase with no iron — returns zero split."""
        mock_get_apfu.return_value = {'FeO': 0.0, 'O': 12.0, 'Fe2O3': 0.0}
        result = self.base._extract_fe_split_from_apfu(None, 'q')
        self.assertEqual(result['Fe2'], 0.0)
        self.assertEqual(result['Fe3'], 0.0)

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_fe3_exceeds_total_fe_clamp(self, mock_get_apfu):
        """excess_o suggests more Fe3+ than total Fe — Fe2+ clamped to 0."""
        mock_get_apfu.return_value = {'FeO': 0.1, 'O': 6.3, 'Fe2O3': 0.0}
        result = self.base._extract_fe_split_from_apfu(None, 'dio')
        # excess_o = 0.3, fe3 = 0.6, total_fe = 0.1, fe2 = max(0.1-0.6, 0) = 0
        self.assertAlmostEqual(result['Fe3'], 0.6, places=4)
        self.assertEqual(result['Fe2'], 0.0)

    @patch('phasetools.core.phase_properties.get_oxide_apfu')
    def test_zero_o_basis(self, mock_get_apfu):
        """Traditional FeO/Fe2O3 basis when O = 0."""
        mock_get_apfu.return_value = {'FeO': 0.8, 'O': 0.0, 'Fe2O3': 0.1}
        result = self.base._extract_fe_split_from_apfu(None, 'sp')
        # total_fe = 0.8 + 2*0.1 = 1.0, fe3 = 2*0.1 = 0.2, fe2 = 0.8
        self.assertAlmostEqual(result['Fe3'], 0.2, places=4)
        self.assertAlmostEqual(result['Fe2'], 0.8, places=4)

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

    def test_get_phase_mg2_number_element_wise(self):
        """instance='all' must compute Mg# element-wise, not collapse on one zero denominator."""
        from phasetools.core.phase_properties import get_phase_mg2_number

        # Two instances of 'ol': one Mg-free, one Fe-free.
        mock_out = MagicMock()
        mock_out.ph = ['ol', 'ol']
        mock_out.oxides = ['MgO', 'FeO', 'O']

        mock_ol_0 = MagicMock()
        mock_ol_0.Comp_apfu = [0.0, 1.0, 1.0]   # Mg=0, Fe2+=1 after split

        mock_ol_1 = MagicMock()
        mock_ol_1.Comp_apfu = [1.0, 0.0, 1.0]   # Mg=1, Fe2+=0 after split

        mock_out.SS_vec = [mock_ol_0, mock_ol_1]

        vals = get_phase_mg2_number(mock_out, 'ol', instance='all')
        self.assertEqual(vals.shape, (2,))
        self.assertAlmostEqual(vals[0], 0.0)
        self.assertAlmostEqual(vals[1], 1.0)

class TestOxideApfuVectorDispatch(unittest.TestCase):
    """get_oxide_apfu follows MAGEMin's layout: SS_vec for solution phases,
    PP_vec for pure phases (out.ph lists solution phases first)."""

    @staticmethod
    def _make_output():
        out = SimpleNamespace()
        out.ph = ["g", "q", "ilm"]
        out.oxides = ["SiO2", "FeO", "O"]
        out.n_SS = 1
        out.SS_vec = [SimpleNamespace(Comp_apfu=[3.0, 1.5719, 12.0293])]
        out.PP_vec = [
            SimpleNamespace(Comp_apfu=[1.0, 0.0, 2.0]),    # q: SiO2, no Fe
            SimpleNamespace(Comp_apfu=[0.0, 1.5, 3.01]),   # ilm: Fe-bearing pure phase
        ]
        return out

    def test_solution_phase_reads_ss_vec(self):
        out = self._make_output()
        self.assertEqual(
            get_oxide_apfu(out, "g", ["SiO2", "FeO", "O"]),
            {"SiO2": 3.0, "FeO": 1.5719, "O": 12.0293},
        )

    def test_pure_phase_reads_pp_vec(self):
        """Pure phases previously returned zeros (SS_vec-only read)."""
        out = self._make_output()
        self.assertEqual(
            get_oxide_apfu(out, "q", ["SiO2", "O"]),
            {"SiO2": 1.0, "O": 2.0},
        )

    def test_fe_split_uses_pure_phase_apfu(self):
        """get_phase_fe_split inherits the PP_vec fix: ilm excess O -> Fe3+."""
        out = self._make_output()
        split = get_phase_fe_split(out, "ilm")
        self.assertAlmostEqual(split["Fe3"], 0.02, places=10)   # 2 * (3.01 - 3)
        self.assertAlmostEqual(split["Fe2"], 1.48, places=10)   # 1.5 - 0.02


class TestFeSplitOComponentRoute(unittest.TestCase):
    """get_phase_fe_split prefers MAGEMin's O component over the floor rule.

    ``Comp['O']`` (molar O-component) scaled to the APFU norm is the exact
    excess; the integer-floor fallback only applies when Comp is missing.
    """

    @staticmethod
    def _output(comp=None, apfu=(2.0, 2.0, 7.0), oxides=("SiO2", "FeO", "O")):
        out = SimpleNamespace()
        out.ph = ["dio"]
        out.oxides = list(oxides)
        out.n_SS = 1
        ss = SimpleNamespace(Comp_apfu=list(apfu))
        if comp is not None:
            ss.Comp = list(comp)
        out.SS_vec = [ss]
        return out

    def test_o_component_beats_floor_at_integer_excess(self):
        # structural 6 + excess 1.0 -> APFU O = 7.0 exactly: floor reads 0.
        # molecules per formula: SiO2 2, FeO 2, O 1 -> sum 5 (k = 5).
        out = self._output(comp=[0.4, 0.4, 0.2], apfu=[2.0, 2.0, 7.0])
        split = get_phase_fe_split(out, "dio")
        self.assertAlmostEqual(split["Fe3"], 2.0, places=10)  # floor would give 0
        self.assertAlmostEqual(split["Fe2"], 0.0, places=10)   # FeO 2 - Fe3 2

    def test_zero_o_component_yields_no_ferric(self):
        # APFU O is fractional (floor would invent Fe3 = 0.08) but the O
        # component is zero: MAGEMin assigned no redox oxygen -> pure Fe2+.
        out = self._output(comp=[0.5, 0.5, 0.0], apfu=[3.0, 2.0, 6.04])
        split = get_phase_fe_split(out, "dio")
        self.assertAlmostEqual(split["Fe3"], 0.0, places=12)
        self.assertAlmostEqual(split["Fe2"], 2.0, places=10)

    def test_falls_back_to_floor_without_comp(self):
        # Mock/fixture outputs without Comp keep the historical floor rule.
        out = self._output(comp=None, apfu=[3.0, 2.0, 6.04])
        split = get_phase_fe_split(out, "dio")
        self.assertAlmostEqual(split["Fe3"], 0.08, places=10)  # 2 * (6.04 - 6)
        self.assertAlmostEqual(split["Fe2"], 1.92, places=10)


if __name__ == '__main__':
    unittest.main()
