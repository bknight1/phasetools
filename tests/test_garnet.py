"""Mock-based tests for ``phasetools.calculators.garnet``.

Covers:

* the garnet ``fe_basis`` option (``'FeOt'`` vs ``'Fe2+'``) for X-site
  extraction;
* ``gt_along_path`` behaviour (wt-basis fractionation and row
  normalisation of ``X_along_path``).

No live Julia runtime is needed -- all MAGEMin calls are mocked.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from phasetools.calculators.garnet import MAGEMinGarnetCalculator


class TestGarnetFeBasis(unittest.TestCase):
    """Fe-basis flag on the garnet X-site extraction."""

    def _make_calc(self, fe_basis):
        calc = MAGEMinGarnetCalculator.__new__(MAGEMinGarnetCalculator)
        calc.fe_basis = fe_basis  # canonical lowercase form set by __init__
        return calc

    def test_feot_uses_total_iron(self):
        """FeOt puts FeO + 2*Fe2O3 on the X-site and normalises to 1."""
        calc = self._make_calc('feot')
        out = MagicMock()
        out.ph = ['g', 'q']
        apfu = {'MgO': 0.6, 'MnO': 0.1, 'CaO': 0.8, 'FeO': 1.2, 'Fe2O3': 0.15}
        with patch('phasetools.calculators.garnet.get_oxide_apfu', return_value=apfu):
            Mg, Mn, Fe, Ca = calc._extract_garnet_elements_from_oxides(out, 'mol')
        fe_total = 1.2 + 2.0 * 0.15
        total = 0.6 + 0.1 + 0.8 + fe_total
        self.assertAlmostEqual(Fe, fe_total / total, places=6)
        self.assertAlmostEqual(Mg, 0.6 / total, places=6)
        self.assertAlmostEqual(Mn, 0.1 / total, places=6)
        self.assertAlmostEqual(Ca, 0.8 / total, places=6)
        self.assertAlmostEqual(Mg + Mn + Fe + Ca, 1.0, places=6)

    def test_fe2_uses_split(self):
        """Fe2+ uses only the ferrous split, excluding Fe3+."""
        calc = self._make_calc('fe2+')
        out = MagicMock()
        out.ph = ['g', 'q']
        apfu = {'MgO': 0.6, 'MnO': 0.1, 'CaO': 0.8}
        split = {'Fe2': 1.35, 'Fe3': 0.15}
        with patch('phasetools.calculators.garnet.get_oxide_apfu', return_value=apfu), \
             patch.object(calc, '_extract_fe_split_from_apfu', return_value=split):
            Mg, Mn, Fe, Ca = calc._extract_garnet_elements_from_oxides(out, 'mol')
        total = 0.6 + 0.1 + 0.8 + 1.35
        self.assertAlmostEqual(Fe, 1.35 / total, places=6)
        self.assertAlmostEqual(Mg + Mn + Fe + Ca, 1.0, places=6)
        self.assertAlmostEqual(Mn, 0.1 / total, places=6)

    def test_absent_garnet_returns_zeros(self):
        """No garnet in the assemblage -> all-zero X-site fractions."""
        calc = self._make_calc('feot')
        out = MagicMock()
        out.ph = ['q', 'dio']
        Mg, Mn, Fe, Ca = calc._extract_garnet_elements_from_oxides(out, 'mol')
        self.assertEqual((Mg, Mn, Fe, Ca), (0.0, 0.0, 0.0, 0.0))

    def test_invalid_basis_raises(self):
        """Unsupported fe_basis values must raise ValueError at construction."""
        with (
            patch('phasetools.calculators.garnet.MAGEMinPTGridCalculator.__init__',
                  return_value=None),
            self.assertRaises(ValueError),
        ):
            MAGEMinGarnetCalculator(db='ig', fe_basis='Fe3')

    def test_default_is_feot(self):
        """The default fe_basis is 'FeOt' (community convention)."""
        with patch('phasetools.calculators.garnet.MAGEMinPTGridCalculator.__init__',
                   return_value=None):
            calc = MAGEMinGarnetCalculator(db='ig')
            self.assertEqual(calc.fe_basis, 'feot')
            calc = MAGEMinGarnetCalculator(db='ig', fe_basis='Fe2+')
            self.assertEqual(calc.fe_basis, 'fe2+')
            calc = MAGEMinGarnetCalculator(db='ig', fe_basis='feot')
            self.assertEqual(calc.fe_basis, 'feot')


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


if __name__ == '__main__':
    unittest.main()
