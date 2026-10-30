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
from phasetools.calculators.garnet.composition import (
    garnet_cations_from_apfu,
    normalise_fe_basis,
    normalise_garnet_cations,
)


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
        with patch(
            'phasetools.calculators.garnet.composition.get_oxide_apfu',
            return_value=apfu,
        ):
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
        with (
            patch(
                'phasetools.calculators.garnet.composition.get_oxide_apfu',
                return_value=apfu,
            ),
            patch(
                'phasetools.calculators.garnet.composition.get_phase_fe_split',
                return_value=split,
            ),
        ):
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
            patch('phasetools.calculators.garnet.calculator.MAGEMinPTGridCalculator.__init__',
                  return_value=None),
            self.assertRaises(ValueError),
        ):
            MAGEMinGarnetCalculator(db='ig', fe_basis='Fe3')

    def test_default_is_feot(self):
        """The default fe_basis is 'FeOt' (community convention)."""
        with patch('phasetools.calculators.garnet.calculator.MAGEMinPTGridCalculator.__init__',
                   return_value=None):
            calc = MAGEMinGarnetCalculator(db='ig')
            self.assertEqual(calc.fe_basis, 'feot')
            calc = MAGEMinGarnetCalculator(db='ig', fe_basis='Fe2+')
            self.assertEqual(calc.fe_basis, 'fe2+')
            calc = MAGEMinGarnetCalculator(db='ig', fe_basis='feot')
            self.assertEqual(calc.fe_basis, 'feot')

    def test_wt_output_conversion_preserves_public_tuple_order(self):
        calc = self._make_calc('feot')
        out = MagicMock()
        out.ph = ['g']
        atomic = {'Mg': 0.1, 'Fe': 0.2, 'Mn': 0.3, 'Ca': 0.4}
        weight = {'Mg': 0.4, 'Fe': 0.3, 'Mn': 0.2, 'Ca': 0.1}
        with (
            patch(
                'phasetools.calculators.garnet.calculator.garnet_cations_from_apfu',
                return_value=atomic,
            ),
            patch(
                'phasetools.calculators.garnet.calculator.atomic_frac_to_wt_frac',
                return_value=weight,
            ) as convert,
        ):
            values = calc._extract_garnet_elements_from_oxides(out, 'wt')
        convert.assert_called_once()
        self.assertEqual(convert.call_args.args[0], atomic)
        self.assertEqual(values, (0.4, 0.2, 0.3, 0.1))


class TestSharedGarnetComposition(unittest.TestCase):
    def test_normaliser_returns_fixed_atomic_cation_order_and_closure(self):
        result = normalise_garnet_cations({'Mg': 3.0, 'Fe': 2.0, 'Ca': 1.0})
        self.assertEqual(tuple(result), ('Mg', 'Fe', 'Mn', 'Ca'))
        self.assertEqual(
            result,
            {'Mg': 0.5, 'Fe': 1.0 / 3.0, 'Mn': 0.0, 'Ca': 1.0 / 6.0},
        )
        self.assertAlmostEqual(sum(result.values()), 1.0)

    def test_normaliser_handles_absent_cations_and_rejects_invalid_amounts(self):
        self.assertEqual(
            normalise_garnet_cations({}, absent='zero'),
            {'Mg': 0.0, 'Fe': 0.0, 'Mn': 0.0, 'Ca': 0.0},
        )
        with self.assertRaisesRegex(ValueError, 'no positive'):
            normalise_garnet_cations({})
        with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
            normalise_garnet_cations({'Mg': np.nan})

    def test_apfu_projection_uses_both_supported_iron_bases(self):
        out = MagicMock()
        out.ph = ['g']
        apfu = {'MgO': 0.6, 'FeO': 0.8, 'Fe2O3': 0.2, 'MnO': 0.1, 'CaO': 0.3}
        split = {'Fe2': 0.9, 'Fe3': 0.3}
        with (
            patch(
                'phasetools.calculators.garnet.composition.get_oxide_apfu',
                return_value=apfu,
            ),
            patch(
                'phasetools.calculators.garnet.composition.get_phase_fe_split',
                return_value=split,
            ),
        ):
            feot = garnet_cations_from_apfu(out)
            fe2 = garnet_cations_from_apfu(out, 'fE2')
        self.assertAlmostEqual(feot['Fe'], 1.2 / 2.2)
        self.assertAlmostEqual(fe2['Fe'], 0.9 / 1.9)
        self.assertAlmostEqual(sum(feot.values()), 1.0)
        self.assertAlmostEqual(sum(fe2.values()), 1.0)

    def test_apfu_projection_absence_and_fe_basis_validation(self):
        out = MagicMock()
        out.ph = ['q']
        self.assertEqual(
            garnet_cations_from_apfu(out, absent='zero'),
            {'Mg': 0.0, 'Fe': 0.0, 'Mn': 0.0, 'Ca': 0.0},
        )
        with self.assertRaisesRegex(ValueError, "garnet \\('g'\\) is absent"):
            garnet_cations_from_apfu(out)
        self.assertEqual(normalise_fe_basis('fe2'), 'Fe2+')
        with self.assertRaisesRegex(ValueError, 'fe_basis must be'):
            normalise_fe_basis('Fe3+')


# ===========================================================================
# Fix B: wt fraction uses wt basis
# ===========================================================================
class TestWtFractionUsesWtBasis(unittest.TestCase):
    """Fix B: when sys_in='wt', fractionate_phase should use gt_wt, not gt_frac."""

    @patch('phasetools.calculators.garnet.calculator.MAGEMin_C')
    @patch('phasetools.calculators.garnet.calculator.jlconvert')
    @patch('phasetools.calculators.garnet.calculator.MAGEMinPTGridCalculator.__init__', return_value=None)
    def test_wt_fraction_uses_wt_basis(self, mock_grid_init, mock_jlconvert, mock_magemin):
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

        # Patch PhaseFunctions at its source module (local import in gt_along_path);
        # fractionate_phase is a staticmethod called on the class itself
        with patch('phasetools.calculators.phase_search.PhaseFunctions') as pf_cls:
            pf_cls.fractionate_phase = MagicMock(return_value=np.array([50.0, 50.0]))
            # jlconvert should pass through the array so np.array(X) works
            mock_jlconvert.side_effect = lambda t, v: np.array(v, dtype=float)

            P = np.array([10.0, 11.0])
            T = np.array([800.0, 810.0])
            calc.gt_along_path(P, T, fractionate=True, normalise_start=True)

            # Step 0: normalise_start=True, so the seed is retained (no call)
            # Step 1: full-fraction minus retained seed baseline:
            #         frac_amount = gt_wt[1] - seed = 0.07 - 0.05 = 0.02
            #         (full fraction minus seed, NOT a step-to-step difference)
            calls = pf_cls.fractionate_phase.call_args_list
            self.assertEqual(len(calls), 1, f"Expected 1 fractionation call, got {len(calls)}")
            _, kwargs = calls[0]
            self.assertAlmostEqual(kwargs['frac_amount'], 0.02, places=10,
                                   msg="frac_amount should be wt-basis current fraction minus retained seed (0.07-0.05), not mol (0.12-0.10)")


# ===========================================================================
# Fix D: X_along_path normalised to 1
# ===========================================================================
class TestXAlongPathNormalised(unittest.TestCase):
    """Fix D: each row of X_along_path must sum to 1.0."""

    @patch('phasetools.calculators.garnet.calculator.MAGEMin_C')
    @patch('phasetools.calculators.garnet.calculator.jlconvert')
    @patch('phasetools.calculators.garnet.calculator.MAGEMinPTGridCalculator.__init__', return_value=None)
    def test_x_along_path_normalised_to_one(self, mock_grid_init, mock_jlconvert, mock_magemin):
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
