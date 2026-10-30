"""Live MAGEMin regressions for garnet MDF/PTM cation isopleths."""

import unittest

import numpy as np

from phasetools import MAGEMin_C
from phasetools.calculators.garnet.mdf import (
    GarnetMDFCalculator,
    kho_aware_exchange_potentials,
)
from phasetools.core.engine import evaluate_solution_phase_gibbs

_LIVE_AVAILABLE = MAGEMin_C is not None and all(
    hasattr(MAGEMin_C, name) for name in ("p2x_convert", "pc_convert")
)


@unittest.skipUnless(
    _LIVE_AVAILABLE, "Julia + MAGEMin_C conversion bridge not available"
)
class TestGarnetMDFLive(unittest.TestCase):
    def assert_cation_composition(self, composition):
        self.assertEqual(set(composition), {"Mg", "Fe", "Mn", "Ca"})
        values = np.asarray(tuple(composition.values()), dtype=float)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0.0))
        self.assertAlmostEqual(float(values.sum()), 1.0, places=12)

    def test_mb_kho_aware_three_exchange_mdf_and_fe_total_output(self):
        calc = GarnetMDFCalculator(db="mb", dataset=636)
        oxides = ("H2O", "SiO2", "Al2O3", "CaO", "MgO", "FeO", "Na2O", "TiO2", "O")
        bulk = (1.0, 50.0, 15.0, 12.0, 10.0, 8.0, 2.0, 1.0, 0.5)
        calc.setup_bulk_composition(oxides, bulk, sys_in="mol", rm_list=[])
        result = calc.calculate_compositions(15.0, 600.0, tolerance_kj=0.03)
        self.assertEqual(calc.garnet_endmember_names, ("py", "alm", "gr", "kho"))
        self.assertTrue(result.converged, result.message)
        self.assertLessEqual(result.residual_norm_kj_per_formula, 0.03)
        self.assertEqual(len(result.residual_kj_per_formula), 3)
        self.assert_cation_composition(result.mdf)
        self.assert_cation_composition(result.equilibrium)
        self.assertEqual(result.equilibrium["Mn"], 0.0)
        self.assertIn("excluded", result.cation_closure)

    def test_default_mp_database_has_kho_for_o_bearing_bulk(self):
        calc = GarnetMDFCalculator(dataset=636)
        oxides = (
            "H2O",
            "SiO2",
            "Al2O3",
            "CaO",
            "MgO",
            "FeO",
            "K2O",
            "Na2O",
            "TiO2",
            "MnO",
            "O",
        )
        bulk = (0.92, 54.57, 8.79, 11.20, 8.45, 12.89, 0.24, 2.24, 1.12, 0.22, 0.64)
        calc.setup_bulk_composition(oxides, bulk, sys_in="mol", rm_list=[])
        output = calc.calculate_equilibrium(30.0, 700.0)
        self.assertIn("g", tuple(map(str, output.ph)))
        self.assertIn("kho", calc.garnet_endmember_names)

    def test_mpe_kho_aware_four_exchange_mdf_and_all_components(self):
        calc = GarnetMDFCalculator(db="mpe", dataset=636)
        oxides = (
            "H2O",
            "SiO2",
            "Al2O3",
            "CaO",
            "MgO",
            "FeO",
            "K2O",
            "Na2O",
            "TiO2",
            "MnO",
            "O",
        )
        bulk = (0.92, 54.57, 8.79, 11.20, 8.45, 12.89, 0.24, 2.24, 1.12, 0.22, 0.64)
        calc.setup_bulk_composition(oxides, bulk, sys_in="mol", rm_list=[])
        result = calc.calculate_compositions(30.0, 700.0, tolerance_kj=0.03)
        self.assertIn("kho", calc.garnet_endmember_names)
        self.assertIn("spss", calc.garnet_endmember_names)
        self.assertTrue(result.converged, result.message)
        self.assertLessEqual(result.residual_norm_kj_per_formula, 0.03)
        self.assertEqual(len(result.residual_kj_per_formula), 4)
        self.assert_cation_composition(result.mdf)
        self.assert_cation_composition(result.equilibrium)
        self.assertGreater(result._mdf_endmember_fractions["kho"], 0.0)
        self.assertIn("kho", result._equilibrium_endmember_fractions)
        self.assertGreater(result.equilibrium["Fe"], 0.0)

    def test_kho_mu_normalisation_and_gamma_projection_use_same_formula_basis(self):
        calc = GarnetMDFCalculator(db="mpe", dataset=636)
        oxides = (
            "H2O",
            "SiO2",
            "Al2O3",
            "CaO",
            "MgO",
            "FeO",
            "K2O",
            "Na2O",
            "TiO2",
            "MnO",
            "O",
        )
        bulk = (0.92, 54.57, 8.79, 11.20, 8.45, 12.89, 0.24, 2.24, 1.12, 0.22, 0.64)
        calc.setup_bulk_composition(oxides, bulk, sys_in="mol", rm_list=[])
        output = calc.calculate_equilibrium(30.0, 700.0)
        garnet_index = tuple(map(str, output.ph)).index("g")
        garnet = output.SS_vec[garnet_index]
        names = tuple(map(str, garnet.emNames))
        fractions = dict(zip(names, map(float, garnet.emFrac), strict=True))
        if fractions.get("kho", 0.0) <= 0.0:
            self.skipTest("dataset-636 state has no nonzero kho endmember")

        evaluated = evaluate_solution_phase_gibbs(
            calc, "g", fractions, 30.0, 700.0, endmember_names=names
        )
        self.assertTrue(evaluated.valid)
        self.assertGreater(float(garnet.f), 0.0)
        raw_mu = np.asarray(garnet.emChemPot, dtype=float) / float(garnet.f)
        np.testing.assert_allclose(raw_mu, evaluated.mu, rtol=1e-12, atol=1e-9)

        gamma = dict(
            zip(map(str, output.oxides), map(float, output.Gamma), strict=True)
        )
        exchanges = kho_aware_exchange_potentials(gamma)
        mu = dict(zip(names, evaluated.mu, strict=True))
        for component in ("alm", "spss", "gr", "kho"):
            with self.subTest(component=component):
                self.assertAlmostEqual(
                    mu[component] - mu["py"], exchanges[component], delta=0.03
                )

    def test_fe2o3_only_bulk_is_standardised_to_positive_molar_oxygen(self):
        calc = GarnetMDFCalculator(db="mpe", dataset=636)
        oxides = (
            "H2O",
            "SiO2",
            "Al2O3",
            "CaO",
            "MgO",
            "Fe2O3",
            "K2O",
            "Na2O",
            "TiO2",
            "MnO",
        )
        values = (0.92, 54.57, 8.79, 11.20, 8.45, 6.445, 0.24, 2.24, 1.12, 0.22)
        calc.setup_bulk_composition(oxides, values, sys_in="mol", rm_list=[])
        standardised = dict(
            zip(
                calc._standardized_bulk_molar_oxides,
                calc._standardized_bulk_molar_values,
                strict=True,
            )
        )
        self.assertNotIn("FeO", oxides)
        self.assertNotIn("O", oxides)
        self.assertIn("FeO", standardised)
        self.assertIn("O", standardised)
        self.assertGreater(standardised["O"], 0.0)
        self.assertEqual(calc._standardized_bulk_oxygen(), standardised["O"])


if __name__ == "__main__":
    unittest.main()
