"""Mock-based tests for solvus (multi-instance phase) handling.

MAGEMin reports coexisting solvus limbs as repeated entries in ``out.ph``
(e.g. two clinopyroxenes ``dio``, or two amphiboles ``amp``).  The
composition helpers now take ``instance``:

* integer index (default ``0``) -- that instance, with a warning listing
  the number of other instances;
* ``'all'`` -- one value per instance (numpy array / per-instance keys).

No live Julia runtime is needed -- ``out`` objects are mocked.
"""

import unittest
import warnings
import numpy as np
from unittest.mock import MagicMock

from phasetools.calculators.pt_grid import MAGEMinPTGridCalculator
from phasetools.core.phase_properties import (
    get_oxide_apfu, get_phase_chemistry, extract_end_member,
    get_phase_mg_number, get_phase_mg2_number, get_phase_fe_split,
    _phase_indices,
)

# mpe-style oxide ordering
OXIDES = ['H2O', 'SiO2', 'Al2O3', 'CaO', 'MgO', 'FeO', 'K2O', 'Na2O',
          'TiO2', 'MnO', 'O']


def _make_ss(apfu, em_names, em_frac):
    """Build a mocked SS_vec entry (one phase instance)."""
    p = MagicMock()
    p.Comp_apfu = apfu
    # molar fractions (0-1) aligned with OXIDES
    p.Comp = [0.0, 0.6, 0.03, 0.15, 0.13, 0.02, 0.0, 0.06, 0.0, 0.01, 0.0]
    p.Comp_wt = [0.0, 0.55, 0.05, 0.12, 0.10, 0.02, 0.0, 0.05, 0.0, 0.01, 0.0]
    p.emNames = em_names
    p.emFrac = em_frac
    p.emFrac_wt = [f * 0.9 for f in em_frac]
    return p


def _dio_0():
    # diopside-rich limb: low Na
    apfu = [0.0, 1.98, 0.05, 0.74, 0.64, 0.12, 0.0, 0.26, 0.0, 0.01, 6.0]
    return _make_ss(apfu, ['di', 'hed', 'om', 'jac'], [0.60, 0.10, 0.25, 0.05])


def _dio_1():
    # omphacite-rich limb: high Na
    apfu = [0.0, 1.96, 0.09, 0.59, 0.51, 0.18, 0.0, 0.41, 0.0, 0.01, 6.0]
    return _make_ss(apfu, ['di', 'hed', 'om', 'jac'], [0.40, 0.10, 0.40, 0.10])


def _garnet():
    apfu = [0.0, 3.02, 2.0, 0.55, 0.70, 1.70, 0.0, 0.0, 0.0, 0.03, 12.0]
    return _make_ss(apfu, ['py', 'alm', 'gr', 'spss'], [0.20, 0.60, 0.16, 0.04])


def _make_out(ph_list, ss_vec, ph_frac=None):
    out = MagicMock()
    out.ph = ph_list
    out.oxides = OXIDES
    n = len(ph_list)
    out.ph_frac = ph_frac if ph_frac is not None else [0.1] * n
    out.ph_frac_wt = [0.1] * n
    out.ph_frac_vol = [0.1] * n
    out.SS_vec = ss_vec
    return out


class TestPhaseIndices(unittest.TestCase):
    """_phase_indices resolution."""

    def setUp(self):
        self.out = _make_out(['dio', 'q', 'dio', 'g'], [_dio_0(), None, _dio_1(), _garnet()])

    def test_all_returns_every_occurrence(self):
        self.assertEqual(_phase_indices(self.out, 'dio', 'all'), [0, 2])

    def test_default_zero_warns_with_others(self):
        with self.assertWarnsRegex(UserWarning, r"2 instances.*1.*others"):
            self.assertEqual(_phase_indices(self.out, 'dio', 0), [0])

    def test_single_instance_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(_phase_indices(self.out, 'g', 0), [3])

    def test_out_of_range_warns_and_empty(self):
        with self.assertWarnsRegex(UserWarning, r"instance 2 does not exist"):
            self.assertEqual(_phase_indices(self.out, 'dio', 2), [])

    def test_absent_phase_empty(self):
        self.assertEqual(_phase_indices(self.out, 'ep', 0), [])

    def test_invalid_instance_type_raises(self):
        with self.assertRaises(ValueError):
            _phase_indices(self.out, 'dio', 'first')


class TestSolvusHelpers(unittest.TestCase):
    """Per-instance behaviour of the composition helpers."""

    def setUp(self):
        self.out = _make_out(['dio', 'q', 'dio', 'g'], [_dio_0(), None, _dio_1(), _garnet()])

    def test_oxide_apfu_default_is_first(self):
        apfu = get_oxide_apfu(self.out, 'dio', ['Na2O', 'MgO'])
        self.assertAlmostEqual(apfu['Na2O'], 0.26)
        self.assertAlmostEqual(apfu['MgO'], 0.64)

    def test_oxide_apfu_second_instance(self):
        apfu = get_oxide_apfu(self.out, 'dio', ['Na2O', 'MgO'], instance=1)
        self.assertAlmostEqual(apfu['Na2O'], 0.41)
        self.assertAlmostEqual(apfu['MgO'], 0.51)

    def test_oxide_apfu_all_returns_arrays(self):
        apfu = get_oxide_apfu(self.out, 'dio', ['Na2O', 'MgO'], instance='all')
        self.assertTrue(np.allclose(apfu['Na2O'], [0.26, 0.41]))
        self.assertTrue(np.allclose(apfu['MgO'], [0.64, 0.51]))

    def test_phase_chemistry_second_instance(self):
        chem = get_phase_chemistry(self.out, 'dio', ['Na2O'], 'mol', instance=1)
        self.assertAlmostEqual(chem['Na2O'], 0.06 * 100.0)  # Comp[7] * 100

    def test_extract_end_member_per_instance(self):
        self.assertAlmostEqual(extract_end_member('dio', self.out, 'di', 'mol'), 0.60)
        self.assertAlmostEqual(extract_end_member('dio', self.out, 'di', 'mol', instance=1), 0.40)
        vals = extract_end_member('dio', self.out, 'di', 'mol', instance='all')
        self.assertTrue(np.allclose(vals, [0.60, 0.40]))

    def test_mg_number_per_instance(self):
        # Mg# = Mg/(Mg+Fe); first limb Mg-rich, second more Fe-rich
        self.assertGreater(get_phase_mg_number(self.out, 'dio'),
                           get_phase_mg_number(self.out, 'dio', instance=1))
        vals = get_phase_mg_number(self.out, 'dio', instance='all')
        self.assertEqual(vals.shape, (2,))
        self.assertAlmostEqual(vals[0], get_phase_mg_number(self.out, 'dio'))

    def test_mg2_number_per_instance(self):
        m0 = get_phase_mg2_number(self.out, 'dio')
        m1 = get_phase_mg2_number(self.out, 'dio', instance=1)
        self.assertGreater(m0, m1)
        vals = get_phase_mg2_number(self.out, 'dio', instance='all')
        self.assertEqual(vals.shape, (2,))

    def test_fe_split_per_instance(self):
        # O-basis: excess O = 0, so Fe2 = total Fe, Fe3 = 0 for both limbs
        s0 = get_phase_fe_split(self.out, 'dio')
        s1 = get_phase_fe_split(self.out, 'dio', instance=1)
        self.assertAlmostEqual(s0['Fe2'], 0.12)
        self.assertAlmostEqual(s1['Fe2'], 0.18)
        all_s = get_phase_fe_split(self.out, 'dio', instance='all')
        self.assertTrue(np.allclose(all_s['Fe2'], [0.12, 0.18]))

    def test_out_of_range_returns_zeros(self):
        apfu = get_oxide_apfu(self.out, 'dio', ['Na2O'], instance=5)
        self.assertEqual(apfu['Na2O'], 0.0)


class TestExtractFromGridSolvus(unittest.TestCase):
    """Grid-level extraction with instance='all' produces _0/_1 columns."""

    def _make_calc(self, grid_out):
        calc = MAGEMinPTGridCalculator.__new__(MAGEMinPTGridCalculator)
        calc.sys_in = 'mol'
        calc.last_grid_out = grid_out
        calc.rm_list = None
        return calc

    def test_all_returns_suffixed_columns(self):
        # point 0: dio solvus (2 limbs); point 1: single dio
        g0 = _make_out(['dio', 'q', 'dio', 'g'],
                       [_dio_0(), None, _dio_1(), _garnet()],
                       ph_frac=[0.3, 0.2, 0.1, 0.4])
        g1 = _make_out(['q', 'g', 'dio'],
                       [None, _garnet(), _dio_0()],
                       ph_frac=[0.5, 0.2, 0.3])
        calc = self._make_calc([g0, g1])

        res = calc.extract_from_grid('dio', oxides=['Na2O', 'MgO'],
                                     mg_number=True, fe_split=True,
                                     instance='all')

        self.assertIn('ox_apfu_Na2O_0', res)
        self.assertIn('ox_apfu_Na2O_1', res)
        self.assertIn('Mg_number_0', res)
        self.assertIn('Mg_number_1', res)
        self.assertIn('Fe2_0', res)
        self.assertIn('Fe2_1', res)
        # point 0 first limb
        self.assertAlmostEqual(res['ox_apfu_Na2O_0'][0], 0.26)
        # point 0 second limb
        self.assertAlmostEqual(res['ox_apfu_Na2O_1'][0], 0.41)
        # point 1 has a single dio -> second column NaN
        self.assertAlmostEqual(res['ox_apfu_Na2O_0'][1], 0.26)
        self.assertTrue(np.isnan(res['ox_apfu_Na2O_1'][1]))

    def test_all_emits_per_instance_fractions(self):
        """instance='all' gives mol/wt/vol_frac_0/_1 plus summed totals."""
        g0 = _make_out(['dio', 'q', 'dio', 'g'],
                       [_dio_0(), None, _dio_1(), _garnet()],
                       ph_frac=[0.3, 0.2, 0.1, 0.4])
        g1 = _make_out(['q', 'g', 'dio'],
                       [None, _garnet(), _dio_0()],
                       ph_frac=[0.5, 0.2, 0.3])
        calc = self._make_calc([g0, g1])

        res = calc.extract_from_grid('dio', oxides=['Na2O'], instance='all')

        # summed totals unchanged (limbs 0.3 + 0.1 at point 0)
        self.assertAlmostEqual(res['mol_frac'][0], 0.4)
        self.assertAlmostEqual(res['mol_frac'][1], 0.3)
        # per-instance columns
        self.assertIn('mol_frac_0', res)
        self.assertIn('mol_frac_1', res)
        self.assertIn('wt_frac_0', res)
        self.assertIn('wt_frac_1', res)
        self.assertIn('vol_frac_0', res)
        self.assertIn('vol_frac_1', res)
        self.assertAlmostEqual(res['mol_frac_0'][0], 0.3)
        self.assertAlmostEqual(res['mol_frac_1'][0], 0.1)
        self.assertAlmostEqual(res['mol_frac_0'][1], 0.3)
        self.assertTrue(np.isnan(res['mol_frac_1'][1]))  # single limb at point 1

    def test_default_has_no_fraction_suffixes(self):
        """Integer instance keeps single (summed) fraction keys."""
        g0 = _make_out(['dio', 'q', 'dio', 'g'],
                       [_dio_0(), None, _dio_1(), _garnet()],
                       ph_frac=[0.3, 0.2, 0.1, 0.4])
        calc = self._make_calc([g0])
        res = calc.extract_from_grid('dio', oxides=['Na2O'], instance=0)
        self.assertIn('mol_frac', res)
        self.assertNotIn('mol_frac_0', res)
        self.assertNotIn('wt_frac_0', res)

    def test_default_is_first_instance_no_suffix(self):
        g0 = _make_out(['dio', 'q', 'dio', 'g'],
                       [_dio_0(), None, _dio_1(), _garnet()],
                       ph_frac=[0.3, 0.2, 0.1, 0.4])
        calc = self._make_calc([g0])
        res = calc.extract_from_grid('dio', oxides=['Na2O'], instance=0)
        self.assertIn('ox_apfu_Na2O', res)
        self.assertNotIn('ox_apfu_Na2O_0', res)
        self.assertAlmostEqual(res['ox_apfu_Na2O'][0], 0.26)

    def test_single_instance_phase_unaffected(self):
        g0 = _make_out(['dio', 'q', 'dio', 'g'],
                       [_dio_0(), None, _dio_1(), _garnet()],
                       ph_frac=[0.3, 0.2, 0.1, 0.4])
        calc = self._make_calc([g0])
        res = calc.extract_from_grid('g', oxides=['MgO'], instance='all')
        # garnet has a single instance -> single _0 column
        self.assertIn('ox_apfu_MgO_0', res)
        self.assertNotIn('ox_apfu_MgO_1', res)
        self.assertAlmostEqual(res['ox_apfu_MgO_0'][0], 0.70)

    def test_cations_per_instance(self):
        g0 = _make_out(['dio', 'q', 'dio', 'g'],
                       [_dio_0(), None, _dio_1(), _garnet()],
                       ph_frac=[0.3, 0.2, 0.1, 0.4])
        calc = self._make_calc([g0])
        res = calc.extract_from_grid('dio', cations=['Mg', 'Fe'], instance='all')
        self.assertIn('cat_Mg_0', res)
        self.assertIn('cat_Mg_1', res)
        # first limb is Mg-richer
        self.assertGreater(res['cat_Mg_0'][0], res['cat_Mg_1'][0])


if __name__ == '__main__':
    unittest.main()
