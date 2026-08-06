"""Mock-based tests for solvus (multi-instance phase) handling.

MAGEMin reports coexisting solvus limbs as repeated entries in ``out.ph``
(e.g. two clinopyroxenes ``dio``, or two amphiboles ``amp``).

Two layers are tested:

* the low-level composition helpers (``get_oxide_apfu``,
  ``extract_end_member``, ...) take ``instance`` (integer index or
  ``'all'`` for per-instance numpy arrays);
* the grid-level ``extract_from_grid`` / ``single_point_calc`` return a
  list of per-instance bundles indexed ``res[0]``, ``res[1]`` ...  Bundle
  keys are unsuffixed (``mol_frac``, ``ox_apfu_Na2O``, ``em_py``, ...),
  and each value is an array over the grid points (NaN where the phase
  or that limb is absent).  The shape is the same for every phase; a
  single-instance phase just has a one-element list.

No live Julia runtime is needed -- ``out`` objects are mocked.
"""

import unittest
import warnings
import numpy as np
from unittest.mock import MagicMock, patch

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
    if ph_frac is None:
        ph_frac = [0.1] * n
    out.ph_frac = ph_frac
    out.ph_frac_wt = list(ph_frac)
    out.ph_frac_vol = list(ph_frac)
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
    """Grid extraction: list of per-instance bundles (res[0], res[1], ...)."""

    def _make_calc(self, grid_out):
        calc = MAGEMinPTGridCalculator.__new__(MAGEMinPTGridCalculator)
        calc.sys_in = 'mol'
        calc.last_grid_out = grid_out
        calc.rm_list = None
        return calc

    def _dio_grid(self):
        # point 0: dio solvus (2 limbs); point 1: single dio
        g0 = _make_out(['dio', 'q', 'dio', 'g'],
                       [_dio_0(), None, _dio_1(), _garnet()],
                       ph_frac=[0.3, 0.2, 0.1, 0.4])
        g1 = _make_out(['q', 'g', 'dio'],
                       [None, _garnet(), _dio_0()],
                       ph_frac=[0.5, 0.2, 0.3])
        return [g0, g1]

    def test_uniform_shape_single_and_solvus(self):
        """A single-instance phase and a solvus return the same bundle keys."""
        calc = self._make_calc(self._dio_grid())
        solvus = calc.extract_from_grid('dio', oxides=['Na2O', 'MgO'],
                                        mg_number=True, fe_split=True)
        single = calc.extract_from_grid('g', oxides=['MgO'],
                                        mg_number=True, fe_split=True)

        # dio: two bundles; garnet: one bundle (always at res[0])
        self.assertEqual(len(solvus), 2)
        self.assertEqual(len(single), 1)

        for bundle in solvus + single:
            for key in ('mol_frac', 'wt_frac', 'vol_frac',
                        'ox_apfu_Na2O' if 'ox_apfu_Na2O' in bundle else 'ox_apfu_MgO',
                        'Mg_number', 'Fe2', 'Fe3'):
                self.assertIn(key, bundle)
        # unsuffixed keys inside each bundle (no _0/_1, no total_*)
        self.assertIn('mol_frac', solvus[0])
        self.assertNotIn('mol_frac_0', solvus[0])
        self.assertNotIn('total_mol_frac', solvus[0])

    def test_limb_bundles(self):
        calc = self._make_calc(self._dio_grid())
        res = calc.extract_from_grid('dio', oxides=['Na2O'])
        c0, c1 = res
        # limb fractions (0.3/0.1 at point 0; point 1 has only limb 0)
        self.assertTrue(np.allclose(c0['mol_frac'], [0.3, 0.3]))
        self.assertAlmostEqual(c0['wt_frac'][0], 0.3)
        self.assertAlmostEqual(c0['vol_frac'][0], 0.3)
        self.assertAlmostEqual(c1['mol_frac'][0], 0.1)
        self.assertAlmostEqual(c0['ox_apfu_Na2O'][0], 0.26)
        self.assertAlmostEqual(c1['ox_apfu_Na2O'][0], 0.41)
        # user can sum the limbs themselves
        total = c0['mol_frac'] + c1['mol_frac']
        self.assertAlmostEqual(total[0], 0.4)

    def test_nan_when_limb_absent(self):
        calc = self._make_calc(self._dio_grid())
        res = calc.extract_from_grid('dio', oxides=['Na2O'], mg_number=True)
        c0, c1 = res
        # point 1 has a single dio -> limb-1 values NaN
        self.assertTrue(np.isnan(c1['mol_frac'][1]))
        self.assertTrue(np.isnan(c1['wt_frac'][1]))
        self.assertTrue(np.isnan(c1['ox_apfu_Na2O'][1]))
        self.assertTrue(np.isnan(c1['Mg_number'][1]))
        # limb 0 is filled
        self.assertAlmostEqual(c0['mol_frac'][1], 0.3)
        self.assertAlmostEqual(c0['ox_apfu_Na2O'][1], 0.26)

    def test_absent_phase_bundles_nan(self):
        g0 = _make_out(['dio', 'q', 'dio', 'g'],
                       [_dio_0(), None, _dio_1(), _garnet()],
                       ph_frac=[0.3, 0.2, 0.1, 0.4])
        g2 = _make_out(['q', 'g'], [None, _garnet()], ph_frac=[0.6, 0.4])
        calc = self._make_calc([g0, g2])
        res = calc.extract_from_grid('dio', oxides=['Na2O'])
        c0, c1 = res
        # point 1: dio absent -> NaN in every bundle
        self.assertTrue(np.isnan(c0['mol_frac'][1]))
        self.assertTrue(np.isnan(c1['mol_frac'][1]))
        self.assertTrue(np.isnan(c0['ox_apfu_Na2O'][1]))

    def test_absent_everywhere_empty(self):
        """Phase never stable -> empty list."""
        g0 = _make_out(['q', 'g'], [None, _garnet()], ph_frac=[0.6, 0.4])
        g1 = _make_out(['q', 'g'], [None, _garnet()], ph_frac=[0.5, 0.5])
        calc = self._make_calc([g0, g1])
        res = calc.extract_from_grid('dio', oxides=['Na2O'], mg_number=True)
        self.assertEqual(res, [])

    def test_single_instance_phase_filled(self):
        calc = self._make_calc(self._dio_grid())
        res = calc.extract_from_grid('g', oxides=['MgO'])
        bundle = res[0]
        self.assertAlmostEqual(bundle['mol_frac'][0], 0.4)
        self.assertAlmostEqual(bundle['ox_apfu_MgO'][0], 0.70)

    def test_cations_per_instance(self):
        calc = self._make_calc(self._dio_grid())
        res = calc.extract_from_grid('dio', cations=['Mg', 'Fe'])
        c0, c1 = res
        self.assertIn('cat_Mg', c0)
        self.assertIn('cat_Mg', c1)
        # first limb is Mg-richer
        self.assertGreater(c0['cat_Mg'][0], c1['cat_Mg'][0])

    def test_end_members_per_instance(self):
        calc = self._make_calc(self._dio_grid())
        res = calc.extract_from_grid('dio', end_members=['di', 'om'])
        c0, c1 = res
        self.assertAlmostEqual(c0['em_di'][0], 0.60)
        self.assertAlmostEqual(c1['em_di'][0], 0.40)
        self.assertAlmostEqual(c0['em_om'][0], 0.25)
        self.assertAlmostEqual(c1['em_om'][0], 0.40)


class TestSinglePointCalcSolvus(unittest.TestCase):
    """single_point_calc returns the same nested-bundle schema."""

    def _make_calc(self):
        calc = MAGEMinPTGridCalculator.__new__(MAGEMinPTGridCalculator)
        calc.sys_in = 'mol'
        calc.data = None
        calc.X = None
        calc.Xoxides = None
        calc.rm_list = None
        return calc

    def test_solvus_returns_limb_bundles(self):
        g0 = _make_out(['dio', 'q', 'dio', 'g'],
                       [_dio_0(), None, _dio_1(), _garnet()],
                       ph_frac=[0.3, 0.2, 0.1, 0.4])
        calc = self._make_calc()
        with patch('phasetools.calculators.pt_grid.MAGEMin_C') as m:
            m.single_point_minimization.return_value = g0
            res, out = calc.single_point_calc(10.0, 600.0, 'dio',
                                              oxides=['Na2O', 'MgO'])
        self.assertEqual(len(res), 2)
        c0, c1 = res
        self.assertAlmostEqual(c0['mol_frac'], 0.3)
        self.assertAlmostEqual(c1['mol_frac'], 0.1)
        self.assertAlmostEqual(c0['wt_frac'], 0.3)
        self.assertAlmostEqual(c1['wt_frac'], 0.1)
        self.assertAlmostEqual(c0['ox_apfu_Na2O'], 0.26)
        self.assertAlmostEqual(c1['ox_apfu_Na2O'], 0.41)
        self.assertIs(out, g0)

    def test_single_instance_has_one_bundle(self):
        g = _make_out(['q', 'g'], [None, _garnet()], ph_frac=[0.6, 0.4])
        calc = self._make_calc()
        with patch('phasetools.calculators.pt_grid.MAGEMin_C') as m:
            m.single_point_minimization.return_value = g
            res, _ = calc.single_point_calc(10.0, 600.0, 'g', oxides=['MgO'])
        self.assertEqual(len(res), 1)
        bundle = res[0]
        self.assertAlmostEqual(bundle['mol_frac'], 0.4)
        self.assertAlmostEqual(bundle['ox_apfu_MgO'], 0.70)

    def test_absent_phase_no_bundles(self):
        g = _make_out(['q', 'g'], [None, _garnet()], ph_frac=[0.6, 0.4])
        calc = self._make_calc()
        with patch('phasetools.calculators.pt_grid.MAGEMin_C') as m:
            m.single_point_minimization.return_value = g
            res, _ = calc.single_point_calc(10.0, 600.0, 'dio')
        self.assertEqual(res, [])


class TestGarnetEndmembersSuffix(unittest.TestCase):
    """generate_2D_grid_gt_endmembers returns the single-instance bundle."""

    def test_returns_bundle_with_historical_keys(self):
        from phasetools.calculators.garnet import MAGEMinGarnetCalculator
        calc = MAGEMinGarnetCalculator.__new__(MAGEMinGarnetCalculator)
        bundle = {
            'mol_frac': np.array([0.10, 0.15]),
            'wt_frac': np.array([0.11, 0.16]),
            'vol_frac': np.array([0.12, 0.17]),
            'em_py': np.array([0.20, 0.30]),
            'em_alm': np.array([0.60, 0.50]),
            'em_gr': np.array([0.16, 0.17]),
            'em_spss': np.array([0.04, 0.03]),
        }
        with patch.object(calc, 'calculate_grid', return_value=None), \
             patch.object(calc, 'extract_from_grid', return_value=[bundle]):
            res = calc.generate_2D_grid_gt_endmembers([10.0], [600.0])
        self.assertIn('em_py', res)
        self.assertIn('em_alm', res)
        self.assertIn('mol_frac', res)
        self.assertTrue(np.allclose(res['em_py'], [0.20, 0.30]))


if __name__ == '__main__':
    unittest.main()
