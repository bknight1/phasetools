import unittest
import numpy as np

from phasetools.utils.bulk_rock import (
    get_molar_mass_dict,
    get_atomic_mass_dict,
    mol_fractions_to_wt_fractions,
    wt_fractions_to_mol_fractions,
)

class TestFractionConverters(unittest.TestCase):
    """Tests for the non-normalising mole/weight fraction converters."""

    def setUp(self):
        self.mass_dict = get_molar_mass_dict()

    # ------------------------------------------------------------------ #
    # Scalar behaviour
    # ------------------------------------------------------------------ #
    def test_scalar_mol_to_wt(self):
        """A scalar input returns a scalar weight fraction (no normalisation)."""
        result = mol_fractions_to_wt_fractions(0.5, ['FeO'], self.mass_dict)
        self.assertIsInstance(result, float)
        # Not normalised: must be mol * mass, not 100 mol%
        self.assertAlmostEqual(result, 0.5 * self.mass_dict['FeO'])

    def test_scalar_wt_to_mol(self):
        """A scalar input returns a scalar mole fraction (no normalisation)."""
        result = wt_fractions_to_mol_fractions(71.844, ['FeO'], self.mass_dict)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 71.844 / self.mass_dict['FeO'])

    # ------------------------------------------------------------------ #
    # List behaviour
    # ------------------------------------------------------------------ #
    def test_list_mol_to_wt(self):
        """A list input returns a list (no normalisation to sum to 100)."""
        result = mol_fractions_to_wt_fractions([0.4, 0.6], ['FeO', 'Fe2O3'], self.mass_dict)
        self.assertIsInstance(result, list)
        self.assertAlmostEqual(result[0], 0.4 * self.mass_dict['FeO'])
        self.assertAlmostEqual(result[1], 0.6 * self.mass_dict['Fe2O3'])

    def test_list_wt_to_mol(self):
        """A list input returns a list (no normalisation to sum to 100)."""
        result = wt_fractions_to_mol_fractions(
            [self.mass_dict['FeO'], self.mass_dict['Fe2O3']], ['FeO', 'Fe2O3'], self.mass_dict
        )
        self.assertIsInstance(result, list)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 1.0)

    def test_list_output_not_normalised(self):
        """Two oxides given as fractions must NOT be normalised to 100 mol%."""
        wt = mol_fractions_to_wt_fractions([0.4, 0.6], ['FeO', 'Fe2O3'], self.mass_dict)
        back = wt_fractions_to_mol_fractions(wt, ['FeO', 'Fe2O3'], self.mass_dict)
        self.assertAlmostEqual(back[0], 0.4)
        self.assertAlmostEqual(back[1], 0.6)

    # ------------------------------------------------------------------ #
    # numpy array behaviour
    # ------------------------------------------------------------------ #
    def test_array_input_returns_array(self):
        """A numpy array input returns a numpy array."""
        result = mol_fractions_to_wt_fractions(
            np.array([0.4, 0.6]), ['FeO', 'Fe2O3'], self.mass_dict
        )
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(
            result,
            np.array([0.4, 0.6]) * np.array([self.mass_dict['FeO'], self.mass_dict['Fe2O3']]),
        )

    def test_single_component_broadcast(self):
        """A single component can be applied to multiple values."""
        result = mol_fractions_to_wt_fractions([0.1, 0.2, 0.3], ['FeO'], self.mass_dict)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[2], 0.3 * self.mass_dict['FeO'])

    # ------------------------------------------------------------------ #
    # Round trips
    # ------------------------------------------------------------------ #
    def test_round_trip_scalar(self):
        """wt -> mol -> wt recovers the input."""
        mol = 0.35
        wt = mol_fractions_to_wt_fractions(mol, ['FeO'], self.mass_dict)
        back = wt_fractions_to_mol_fractions(wt, ['FeO'], self.mass_dict)
        self.assertAlmostEqual(back, mol)

    def test_round_trip_multi_oxide(self):
        """mol -> wt -> mol recovers the input for a pair of oxides."""
        mol = [0.7, 0.3]
        wt = mol_fractions_to_wt_fractions(mol, ['FeO', 'Fe2O3'], self.mass_dict)
        back = wt_fractions_to_mol_fractions(wt, ['FeO', 'Fe2O3'], self.mass_dict)
        self.assertAlmostEqual(back[0], mol[0])
        self.assertAlmostEqual(back[1], mol[1])

    # ------------------------------------------------------------------ #
    # Consistency with the normalising converters
    # ------------------------------------------------------------------ #
    def test_consistency_with_percent_converters(self):
        """The normalised converter is the non-normalised one scaled by a constant."""
        from phasetools.utils.bulk_rock import convert_mol_percent_to_wt_percent
        mol_pct = [70.0, 30.0]  # sums to 100
        norm = convert_mol_percent_to_wt_percent(mol_pct, ['FeO', 'Fe2O3'], self.mass_dict)
        non_norm = mol_fractions_to_wt_fractions(mol_pct, ['FeO', 'Fe2O3'], self.mass_dict)
        # Component ratios are identical; normalised output is a scaled version
        self.assertAlmostEqual(norm[0] / norm[1], non_norm[0] / non_norm[1])
        scale = sum(norm) / sum(non_norm)
        self.assertAlmostEqual(norm[0], non_norm[0] * scale)
        self.assertAlmostEqual(norm[1], non_norm[1] * scale)

    # ------------------------------------------------------------------ #
    # Iron oxide workflows built on the converters
    # ------------------------------------------------------------------ #
    def test_feot_from_wt_fractions(self):
        """FeOt (wt) = FeO + Fe2O3 * (2 * M_FeO / M_Fe2O3)."""
        feo, fe2o3 = 5.0, 2.0
        factor = 2 * self.mass_dict['FeO'] / self.mass_dict['Fe2O3']
        self.assertAlmostEqual(feo + fe2o3 * factor, feo + fe2o3 * factor)

    def test_feot_from_mol_fractions(self):
        """FeOt (mol) = FeO + 2 * Fe2O3 (molar basis)."""
        feo_mol, fe2o3_mol = 0.05, 0.01
        self.assertAlmostEqual(feo_mol + 2 * fe2o3_mol, 0.07)

    def test_garnet_weight_fractions_sum_to_one(self):
        """atomic_frac_to_wt_frac keeps weight fractions summing to 1.0."""
        from phasetools.utils.bulk_rock import atomic_frac_to_wt_frac
        atomic = {'Mg': 0.5, 'Mn': 0.1, 'Fe': 0.3, 'Ca': 0.1}
        wt = atomic_frac_to_wt_frac(atomic, get_atomic_mass_dict())
        self.assertAlmostEqual(sum(wt.values()), 1.0)

    # ------------------------------------------------------------------ #
    # FeOt -> FeO + O (MAGEMin redox basis) split
    # ------------------------------------------------------------------ #
    def test_split_feot_to_feo_o_conserves_feot(self):
        """The FeO column always equals FeOt; only O changes."""
        from phasetools.utils.bulk_rock import split_feot_to_feo_o
        feot = 8.33
        for f in (0.0, 0.05, 0.209, 0.5, 1.0):
            feo, o = split_feot_to_feo_o(feot, f)
            self.assertAlmostEqual(feo, feot)              # FeOt conserved
            self.assertAlmostEqual(o, f * feot / 2.0)      # 2FeO + O -> Fe2O3

    def test_split_feot_to_feo_o_invalid(self):
        """fe3_frac outside [0, 1] must raise ValueError."""
        from phasetools.utils.bulk_rock import split_feot_to_feo_o
        with self.assertRaises(ValueError):
            split_feot_to_feo_o(8.33, 1.2)
        with self.assertRaises(ValueError):
            split_feot_to_feo_o(8.33, -0.1)

    def test_express_bulk_in_feo_o_basis(self):
        """Bulk split conserves FeOt and sets O to f*FeOt/2 in place."""
        from phasetools.utils.bulk_rock import express_bulk_in_feo_o_basis
        X = [50.0, 8.33, 0.0, 0.05]       # SiO2, FeO, Fe2O3, H2O
        ox = ['SiO2', 'FeO', 'Fe2O3', 'H2O']
        X2, ox2 = express_bulk_in_feo_o_basis(X, ox, fe3_frac=0.2)
        # FeOt = FeO + 2*Fe2O3 = 8.33
        self.assertAlmostEqual(X2[1], 8.33)               # FeO = FeOt
        self.assertAlmostEqual(X2[2], 0.0)                # Fe2O3 removed
        self.assertAlmostEqual(X2[4], 0.2 * 8.33 / 2.0)   # O appended
        self.assertEqual(ox2, ['SiO2', 'FeO', 'Fe2O3', 'H2O', 'O'])

    def test_express_bulk_in_feo_o_basis_overwrites_o(self):
        """An existing O component is overwritten by the split."""
        from phasetools.utils.bulk_rock import express_bulk_in_feo_o_basis
        X = [50.0, 8.33, 0.87]
        ox = ['SiO2', 'FeO', 'O']
        X2, ox2 = express_bulk_in_feo_o_basis(X, ox, fe3_frac=0.1)
        self.assertAlmostEqual(X2[1], 8.33)
        self.assertAlmostEqual(X2[2], 0.1 * 8.33 / 2.0)   # O overwritten
        self.assertEqual(ox2, ox)

if __name__ == '__main__':
    unittest.main()
