import unittest
from phasetools.models.magma_ocean import MagmaOcean

class TestLMOFix(unittest.TestCase):
    """
    Verification of the Lunar Magma Ocean (LMO) crystallisation logic.
    Replicates the whole-ocean equilibrium behavior and geometric 
    evolution from Johnson et al. (2021).
    """

    def setUp(self):
        # Oxides and Taylor Whole Moon (TWM) composition from Johnson et al. (2021)
        self.oxides = ["SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O"]
        self.twm_wt = [44.40, 0.31, 6.14, 0.40, 10.90, 32.70, 4.73, 0.04, 0.01]
        
        self.mo = MagmaOcean(radius_km=1737.1, core_radius_km=330.0, gravity=1.62, avg_density=3350.0, verbose=False)
        self.mo.setup_bulk_composition(self.oxides, self.twm_wt, sys_in="wt")

    def test_full_crystallisation_path(self):
        """Test the 10-stage fractional crystallisation path."""
        # Stage 0: Equilibrium (0-50 vol.%)
        # Using p_start=45.0 (core) to p_end=17.3 (approx. 50% volume radius)
        results_0, melt_0 = self.mo.run_stage_0(p_start=45.0, p_end=17.3, solid_frac=0.5, p_intervals=5)

        # Fractional Stages (Stages 1-10, 5% total LMO each)
        results_frac = self.mo.run_fractional_stages(melt_0, p_start=17.3, p_end=0.01, vol_step=0.05, n_stages=10)

        # Verification 1: Did we reach 10 stages?
        self.assertEqual(len(results_frac), 10, f"Expected 10 stages, got {len(results_frac)}")

        # Verification 2: Check Stage 1 pressure (~17.3 kbar)
        p_base_1 = results_frac[0]['p_base']
        self.assertAlmostEqual(p_base_1, 17.3, delta=0.3)

        # Verification 3: Check Stage 7 pressure (~6.2 kbar for TWM)
        p_base_7 = results_frac[6]['p_base']
        self.assertAlmostEqual(p_base_7, 6.2, delta=0.5)

        # Verification 4: Does Spinel appear in the intermediate stages?
        has_spl = False
        for res in results_frac[2:8]:
            modes = {k.lower(): v for k, v in res['layer_modes'].items()}
            if 'spl' in modes:
                has_spl = True
                break
        self.assertTrue(has_spl, "Spinel should be present in intermediate stages.")

        # Verification 5: Does Ilmenite appear in the final stages?
        has_ilm = False
        for res in results_frac[-2:]:
            modes = {k.lower(): v for k, v in res['layer_modes'].items()}
            if 'ilm' in modes:
                has_ilm = True
                break
        self.assertTrue(has_ilm, "Ilmenite should be present in final stages.")

    def test_invalid_starting_frac(self):
        """Ensure invalid starting_vol_frac raises ValueError."""
        with self.assertRaises(ValueError):
            self.mo.run_fractional_stages([0]*9, 17.3, 0.01, starting_vol_frac=0.0)
        with self.assertRaises(ValueError):
            self.mo.run_fractional_stages([0]*9, 17.3, 0.01, starting_vol_frac=1.5)

if __name__ == "__main__":
    unittest.main()
