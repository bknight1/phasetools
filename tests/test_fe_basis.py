"""Mock-based tests for the garnet ``fe_basis`` option.

Verifies that ``MAGEMinGarnetCalculator._extract_garnet_elements_from_oxides``
honours the ``fe_basis`` flag:

* ``'FeOt'`` (default) -- total Fe (FeO + 2*Fe2O3 APFU) placed on the
  divalent X-site.  This is the standard community convention for garnet
  end-members / X-site fractions.
* ``'Fe2+'`` -- only the stoichiometrically estimated ferrous iron is
  placed on the divalent site, excluding Fe3+.

No live Julia runtime is needed -- ``get_oxide_apfu`` and the Fe2+/Fe3+
split are mocked.
"""

import unittest
from unittest.mock import MagicMock, patch

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
        with patch('phasetools.calculators.garnet.MAGEMinPTGridCalculator.__init__',
                   return_value=None):
            with self.assertRaises(ValueError):
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


if __name__ == '__main__':
    unittest.main()
