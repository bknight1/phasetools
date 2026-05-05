import unittest
import numpy as np
from phasetools.calculators.pt_grid import MAGEMinPTGridCalculator
from phasetools.core.phase_properties import get_phase_mg_number, get_phase_mg2_number

class TestSiteOccupancy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = 'mpe'
        # Eclogite S10 bulk in mol%
        cls.Xoxides = ['H2O', 'SiO2', 'Al2O3', 'CaO', 'MgO', 'FeO', 'K2O', 'Na2O', 'TiO2', 'MnO', 'O']
        cls.X = [0.92, 54.57, 8.79, 11.20, 8.45, 12.89, 0.24, 2.24, 1.12, 0.22, 0.64]
        
        cls.calc = MAGEMinPTGridCalculator(db=cls.db)
        cls.calc.setup_bulk_composition(cls.Xoxides, cls.X, sys_in='mol')
        
        # 30 kbar (3 GPa), 700 C - where g and dio are stable in mpe
        cls.out = cls.calc.calculate_grid(30.0, 700.0)[0]

    def test_garnet_comparison(self):
        """
        Compare Garnet Fe2+ and Mg# (X-site) from heuristic vs site occupancy.
        
        In the Garnet model (White et al. 2014), divalent cations (Mg, Fe2+, Ca, Mn) 
        are restricted to the 8-fold X-site. However, the Bulk Mg# (Total Iron) 
        and Site Mg# (Divalent) will still differ if Fe3+ is present.
        
        This is because Fe3+ sits on the octahedral (Y) site (e.g., in the 
        khoharite end-member, Mg3Fe2Si3O12). 
        
        - Bulk Mg# (Total): Mg_total / (Mg_total + Fe2_total + Fe3_total)
        - Site Mg# (X-site): Mg_X / (Mg_X + Fe2_X)
        
        Because khoharite adds Magnesium to the X-site but Ferric iron to the 
        Y-site, the Bulk Mg# (Total Iron) will be lower than the Site Mg#.
        """
        print("\n>>> Testing Garnet Site Occupancy Logic")
        print(">>> Compares heuristic Fe2+/Fe3+ split against end-member site totals.")
        if 'g' not in self.out.ph:
            self.skipTest("Garnet not stable")
            
        # 1. Built-in heuristic (uses excess O to split Fe)
        split = self.calc._extract_fe_split_from_apfu(self.out, 'g')
        fe2_h = split['fe2']
        fe3_h = split['fe3']
        mg_num_h = get_phase_mg_number(self.out, 'g')
        mg2_num_h = get_phase_mg2_number(self.out, 'g')
        
        # 2. Site occupancy (based on user tables)
        ph_idx = self.out.ph.index('g')
        em = {str(n): float(f) for n, f in zip(self.out.SS_vec[ph_idx].emNames, self.out.SS_vec[ph_idx].emFrac)}
        
        # Fe2+ total = 3 * alm
        fe2_s = 3.0 * em.get('alm', 0.0)
        # Fe3+ total = 2 * kho
        fe3_s = 2.0 * em.get('kho', 0.0)
        
        # Mg# (X-site) = Mg / (Mg + Fe2+)
        # Mg_X = 3*py + 3*kho
        # Fe2+_X = 3*alm
        mg_x = 3.0 * em.get('py', 0.0) + 3.0 * em.get('kho', 0.0)
        fe2_x = 3.0 * em.get('alm', 0.0)
        mg_num_s = mg_x / (mg_x + fe2_x) if (mg_x + fe2_x) > 0 else 0.0
        
        print(f"\nGarnet (g):")
        print(f"  Heuristic Fe2+: {fe2_h:.4f}, Fe3+: {fe3_h:.4f}, Mg#: {mg_num_h:.4f}")
        print(f"  Site-occ  Fe2+: {fe2_s:.4f}, Fe3+: {fe3_s:.4f}, Mg# (X): {mg_num_s:.4f}")
        
        self.assertAlmostEqual(fe2_h, fe2_s, places=3)
        self.assertAlmostEqual(fe3_h, fe3_s, places=3)
        # Verify get_phase_mg2_number (divalent-only) matches site calculation
        self.assertAlmostEqual(mg2_num_h, mg_num_s, places=3)

    def test_clinopyroxene_comparison(self):
        """
        Compare Clinopyroxene Fe2+ and Mg# (M1m-site) from heuristic vs site occupancy.
        
        In the Clinopyroxene model (Green et al. 2016), Mg and Fe2+ partition 
        between the M1m and M1a sites. The Bulk Mg# and Site Mg# (M1m) will 
        DIFFER because of ordered intermediate end-members:
        
        1. 'om' (Omphacite): Partition Mg into M1m and Al into M1a.
        2. 'cfm': Partitions Fe into M1m and Mg into M1a.
        
        Because 'cfm' puts Mg on the M1a site, the M1m-site Mg# will be lower 
        than the Bulk Mg#. For thermometry (Kd calculations), the BULK Mg# 
        should always be used to remain consistent with empirical calibrations 
        (e.g., Ellis & Green) and to avoid bias from cation ordering.
        """
        print("\n>>> Testing Clinopyroxene Site Occupancy Logic")
        print(">>> Compares heuristic Fe2+/Fe3+ split against ordered site totals (M1m).")
        ph_name = 'dio' if 'dio' in self.out.ph else ('omph' if 'omph' in self.out.ph else None)
        if ph_name is None:
            self.skipTest("Clinopyroxene not stable")
            
        # 1. Built-in heuristic
        split = self.calc._extract_fe_split_from_apfu(self.out, ph_name)
        fe2_h = split['fe2']
        fe3_h = split['fe3']
        mg_num_h = get_phase_mg_number(self.out, ph_name)
        mg2_num_h = get_phase_mg2_number(self.out, ph_name)
        
        # 2. Site occupancy (based on user tables)
        ph_idx = self.out.ph.index(ph_name)
        em = {str(n): float(f) for n, f in zip(self.out.SS_vec[ph_idx].emNames, self.out.SS_vec[ph_idx].emFrac)}
        
        # Fe2+ total = 0.5*hed + 0.5*cfm (M1m + M1a)
        # Note: hed formula CaFeSi2O6 has 1.0 Fe total (0.5 in M1m, 0.5 in M1a)
        # cfm formula CaMg.5Fe.5SiO6 has 0.5 Fe total (0.5 in M1m, 0.0 in M1a)
        # Total atoms:
        fe2_s = 1.0 * em.get('hed', 0.0) + 0.5 * em.get('cfm', 0.0)
        # Fe3+ total = 1.0*acmm + 0.5*jac
        fe3_s = 1.0 * em.get('acmm', 0.0) + 0.5 * em.get('jac', 0.0)
        
        # Mg# (M1m-site) = Mg / (Mg + Fe2+)
        # Mg_M1m = 0.5*di + 0.5*om
        # Fe2+_M1m = 0.5*hed + 0.5*cfm
        mg_m1m = 0.5 * em.get('di', 0.0) + 0.5 * em.get('om', 0.0)
        fe2_m1m = 0.5 * em.get('hed', 0.0) + 0.5 * em.get('cfm', 0.0)
        mg_num_s = mg_m1m / (mg_m1m + fe2_m1m) if (mg_m1m + fe2_m1m) > 0 else 0.0
        
        print(f"\nClinopyroxene ({ph_name}):")
        print(f"  Heuristic Fe2+: {fe2_h:.4f}, Fe3+: {fe3_h:.4f}, Mg#: {mg_num_h:.4f}")
        print(f"  Site-occ  Fe2+: {fe2_s:.4f}, Fe3+: {fe3_s:.4f}, Mg# (M1m): {mg_num_s:.4f}")
        
        self.assertAlmostEqual(fe2_h, fe2_s, places=3)
        self.assertAlmostEqual(fe3_h, fe3_s, places=3)
        
        # Verify get_phase_mg2_number (divalent-only) matches bulk fe2-basis Mg#
        # For CPX, we compare to the bulk fe2 result, not site-M1m result.
        mgo_idx = [str(o) for o in self.out.oxides].index('MgO')
        mg_bulk = float(self.out.SS_vec[ph_idx].Comp_apfu[mgo_idx])
        mg_bulk_fe2 = mg_bulk / (mg_bulk + fe2_h)
        self.assertAlmostEqual(mg2_num_h, mg_bulk_fe2, places=3)

if __name__ == '__main__':
    unittest.main()
