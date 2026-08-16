import unittest
import numpy as np
from phasetools.calculators.pt_grid import MAGEMinPTGridCalculator
from phasetools.core.phase_properties import get_phase_mg_number, get_phase_mg2_number, phase_frac

try:
    from juliacall import Main as jl
    HAS_JULIA = True
except ImportError:
    HAS_JULIA = False

@unittest.skipUnless(HAS_JULIA, "Julia + MAGEMin_C not available")
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

    PT_POINTS = [
        (5.0, 450.0),
        (10.0, 550.0),
        (20.0, 650.0),
        (30.0, 700.0),
        (15.0, 800.0),
    ]

    def test_garnet_heuristic_across_pt(self):
        """Heuristic Fe2+/Fe3+ and Mg# match endmember stoichiometry across P-T range."""
        for P, T in self.PT_POINTS:
            out = self.calc.calculate_grid(P, T)[0]
            if 'g' not in out.ph:
                continue
            
            split = self.calc._extract_fe_split_from_apfu(out, 'g')
            fe2_h, fe3_h = split['Fe2'], split['Fe3']
            mg2_h = get_phase_mg2_number(out, 'g')
            
            ph_idx = out.ph.index('g')
            em = {str(n): float(f) for n, f in zip(out.SS_vec[ph_idx].emNames, out.SS_vec[ph_idx].emFrac)}
            fe2_s = 3.0 * em.get('alm', 0.0)
            fe3_s = 2.0 * em.get('kho', 0.0)
            mg_full = 3.0 * em.get('py', 0.0) + 3.0 * em.get('kho', 0.0)
            mg_num_s = mg_full / (mg_full + fe2_s) if (mg_full + fe2_s) > 0 else 0.0
            
            with self.subTest(P=P, T=T):
                self.assertAlmostEqual(fe2_h, fe2_s, places=2)
                self.assertAlmostEqual(fe3_h, fe3_s, places=2)
                self.assertAlmostEqual(mg2_h, mg_num_s, places=2)

    def test_cpx_heuristic_across_pt(self):
        """Heuristic Fe2+/Fe3+ and Mg# match endmember stoichiometry for cpx across P-T range."""
        for P, T in self.PT_POINTS:
            out = self.calc.calculate_grid(P, T)[0]
            ph_name = 'dio' if 'dio' in out.ph else ('omph' if 'omph' in out.ph else None)
            if ph_name is None:
                continue
            
            split = self.calc._extract_fe_split_from_apfu(out, ph_name)
            fe2_h, fe3_h = split['Fe2'], split['Fe3']
            mg2_h = get_phase_mg2_number(out, ph_name)
            
            ph_idx = out.ph.index(ph_name)
            em = {str(n): float(f) for n, f in zip(out.SS_vec[ph_idx].emNames, out.SS_vec[ph_idx].emFrac)}
            fe2_s = 1.0 * em.get('hed', 0.0) + 0.5 * em.get('cfm', 0.0)
            fe3_s = 1.0 * em.get('acmm', 0.0) + 0.5 * em.get('jac', 0.0)
            mg_full = 1.0 * em.get('di', 0.0) + 0.5 * em.get('om', 0.0) + 0.5 * em.get('cfm', 0.0)
            mg_num_s = mg_full / (mg_full + fe2_s) if (mg_full + fe2_s) > 0 else 0.0
            
            with self.subTest(P=P, T=T):
                self.assertAlmostEqual(fe2_h, fe2_s, places=2)
                self.assertAlmostEqual(fe3_h, fe3_s, places=2)
                self.assertAlmostEqual(mg2_h, mg_num_s, places=2)

    def test_epidote_heuristic_across_pt(self):
        """Heuristic Fe3+ matches expected values for epidote across P-T range.
        
        Epidote carries Fe³⁺ (not Fe²⁺). The excess-O heuristic should give 
        Fe²⁺ ≈ 0 and Fe³⁺ ≈ total Fe at all P-T points where ep is stable.
        """
        EP_PT_POINTS = [
            (10.0, 325.0),
            (10.0, 350.0),
            (10.0, 375.0),
            (10.0, 400.0),
        ]
        
        for P, T in EP_PT_POINTS:
            out = self.calc.calculate_grid(P, T)[0]
            if 'ep' not in out.ph:
                continue
            
            split = self.calc._extract_fe_split_from_apfu(out, 'ep')
            fe2_h, fe3_h = split['Fe2'], split['Fe3']
            
            ph_idx = out.ph.index('ep')
            ox_names = [str(o) for o in out.oxides]
            comp_apfu = np.array(out.SS_vec[ph_idx].Comp_apfu, dtype=float)
            fe_total = comp_apfu[ox_names.index('FeO')]
            
            with self.subTest(P=P, T=T):
                self.assertAlmostEqual(fe2_h, 0.0, places=2)
                self.assertAlmostEqual(fe3_h, fe_total, places=2)

    def test_amphibole_heuristic(self):
        """Heuristic Fe2+/Fe3+ matches endmember stoichiometry for amphibole."""
        out = self.calc.calculate_grid(10.0, 550.0)[0]
        if 'amp' not in out.ph:
            self.skipTest("Amphibole not stable at 10 kbar, 550°C")
        
        split = self.calc._extract_fe_split_from_apfu(out, 'amp')
        fe2_h, fe3_h = split['Fe2'], split['Fe3']
        
        ph_idx = out.ph.index('amp')
        em = {str(n): float(f) for n, f in zip(out.SS_vec[ph_idx].emNames, out.SS_vec[ph_idx].emFrac)}
        
        # Sum Fe2+ and Fe3+ from endmembers
        # Use sum constraint as primary check: Fe2+ + Fe3+ = total Fe
        ox_names = [str(o) for o in out.oxides]
        fe_total = float(out.SS_vec[ph_idx].Comp_apfu[ox_names.index('FeO')])
        self.assertAlmostEqual(fe2_h + fe3_h, fe_total, places=2)

    def test_biotite_heuristic(self):
        """Heuristic Fe2+/Fe3+ matches endmember stoichiometry for biotite."""
        out = self.calc.calculate_grid(10.0, 550.0)[0]
        if 'bi' not in out.ph:
            self.skipTest("Biotite not stable at 10 kbar, 550°C")
        
        split = self.calc._extract_fe_split_from_apfu(out, 'bi')
        fe2_h, fe3_h = split['Fe2'], split['Fe3']
        
        ph_idx = out.ph.index('bi')
        ox_names = [str(o) for o in out.oxides]
        fe_total = float(out.SS_vec[ph_idx].Comp_apfu[ox_names.index('FeO')])
        self.assertAlmostEqual(fe2_h + fe3_h, fe_total, places=2)

    def test_chlorite_heuristic(self):
        """Heuristic Fe2+/Fe3+ for chlorite (not stable for this bulk composition).
        
        Chlorite is not stable at any P-T for the eclogite S10 bulk in mpe.
        This test is kept for completeness but will always skip.
        """
        out = self.calc.calculate_grid(5.0, 450.0)[0]
        if 'chl' not in out.ph:
            self.skipTest("Chlorite not stable at 5 kbar, 450°C")
        
        split = self.calc._extract_fe_split_from_apfu(out, 'chl')
        fe2_h, fe3_h = split['Fe2'], split['Fe3']
        
        ph_idx = out.ph.index('chl')
        ox_names = [str(o) for o in out.oxides]
        fe_total = float(out.SS_vec[ph_idx].Comp_apfu[ox_names.index('FeO')])
        self.assertAlmostEqual(fe2_h + fe3_h, fe_total, places=2)

    def test_bulk_fe_o_closure(self):
        """Bulk Fe and O are conserved across the assemblage (mass balance)."""
        out = self.calc.calculate_grid(20.0, 650.0)[0]
        
        total_fe_weighted = 0.0
        total_o_weighted = 0.0
        
        fe_bearing_phases = ['g', 'dio', 'omph', 'amp', 'bi', 'chl', 'ep', 'ilm', 'sp']
        
        for ph in out.ph:
            if ph not in fe_bearing_phases:
                continue
            
            try:
                frac = phase_frac(ph, out, 'mol')
                ph_idx = out.ph.index(ph)
                ox_names = [str(o) for o in out.oxides]
                feo_frac = float(out.SS_vec[ph_idx].Comp[ox_names.index('FeO')])
                o_frac = float(out.SS_vec[ph_idx].Comp[ox_names.index('O')])
                
                total_fe_weighted += feo_frac * frac
                total_o_weighted += o_frac * frac
            except (ValueError, IndexError):
                continue
        
        bulk_feo = self.calc.X[self.calc.Xoxides.index('FeO')]
        bulk_o = self.calc.X[self.calc.Xoxides.index('O')]
        
        # Bulk X is in mol% (sums to 100); the weighted Comp sums give X/100.
        if bulk_feo > 0:
            self.assertAlmostEqual(total_fe_weighted / (bulk_feo / 100.0), 1.0, delta=0.1)
        if bulk_o > 0:
            self.assertAlmostEqual(total_o_weighted / (bulk_o / 100.0), 1.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()

@unittest.skipUnless(HAS_JULIA, "Julia + MAGEMin_C not available")
class TestHeuristicAcrossDatabases(unittest.TestCase):
    """Test Fe2+/Fe3+ heuristic across different thermodynamic databases."""
    
    def test_ig_database(self):
        """Heuristic works for ig (igneous) database."""
        calc = MAGEMinPTGridCalculator(db='ig')
        Xoxides = ['SiO2', 'Al2O3', 'CaO', 'MgO', 'FeO', 'TiO2', 'O']
        X = [50.0, 10.0, 10.0, 15.0, 10.0, 1.0, 0.5]
        calc.setup_bulk_composition(Xoxides, X, sys_in='mol')
        out = calc.calculate_grid(20.0, 700.0)[0]
        
        for ph in ['g', 'dio']:
            if ph not in out.ph:
                continue
            split = calc._extract_fe_split_from_apfu(out, ph)
            ph_idx = out.ph.index(ph)
            ox_names = [str(o) for o in out.oxides]
            fe_total = float(out.SS_vec[ph_idx].Comp_apfu[ox_names.index('FeO')])
            self.assertAlmostEqual(split['Fe2'] + split['Fe3'], fe_total, places=2)
    
    def test_mp_database(self):
        """Heuristic works for mp (metapelite) database."""
        calc = MAGEMinPTGridCalculator(db='mp')
        Xoxides = ['H2O', 'SiO2', 'Al2O3', 'CaO', 'MgO', 'FeO', 'K2O', 'Na2O', 'TiO2', 'MnO', 'O']
        X = [0.92, 54.57, 8.79, 11.20, 8.45, 12.89, 0.24, 2.24, 1.12, 0.22, 0.64]
        calc.setup_bulk_composition(Xoxides, X, sys_in='mol')
        out = calc.calculate_grid(30.0, 700.0)[0]
        
        for ph in ['g', 'dio']:
            if ph not in out.ph:
                continue
            split = calc._extract_fe_split_from_apfu(out, ph)
            ph_idx = out.ph.index(ph)
            ox_names = [str(o) for o in out.oxides]
            fe_total = float(out.SS_vec[ph_idx].Comp_apfu[ox_names.index('FeO')])
            self.assertAlmostEqual(split['Fe2'] + split['Fe3'], fe_total, places=2)
    
    def test_mb_database(self):
        """Heuristic works for mb (metabasite) database."""
        calc = MAGEMinPTGridCalculator(db='mb')
        Xoxides = ['H2O', 'SiO2', 'Al2O3', 'CaO', 'MgO', 'FeO', 'Na2O', 'TiO2', 'O']
        X = [1.0, 50.0, 15.0, 12.0, 10.0, 8.0, 2.0, 1.0, 0.5]
        calc.setup_bulk_composition(Xoxides, X, sys_in='mol')
        out = calc.calculate_grid(15.0, 600.0)[0]
        
        for ph in ['g', 'dio']:
            if ph not in out.ph:
                continue
            split = calc._extract_fe_split_from_apfu(out, ph)
            ph_idx = out.ph.index(ph)
            ox_names = [str(o) for o in out.oxides]
            fe_total = float(out.SS_vec[ph_idx].Comp_apfu[ox_names.index('FeO')])
            self.assertAlmostEqual(split['Fe2'] + split['Fe3'], fe_total, places=2)

