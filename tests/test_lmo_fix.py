import numpy as np
from phasetools.models.magma_ocean import MagmaOcean

def test_lmo_crystallisation():
    # Oxides and Taylor Whole Moon (TWM) composition from Johnson et al. (2021)
    oxides = ["SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O"]
    twm_wt = [44.40, 0.31, 6.14, 0.40, 10.90, 32.70, 4.73, 0.04, 0.01]

    mo = MagmaOcean(radius_km=1737.1, core_radius_km=330.0, gravity=1.62, avg_density=3350.0, verbose=False)
    mo.setup_bulk_composition(oxides, twm_wt, sys_in="wt")

    print("Running Stage 0 (Equilibrium, 50 vol.%)...")
    # Using p_start=45.0 (core) to p_end=17.3 (approx. 50% volume radius)
    results_0, melt_0 = mo.run_stage_0(p_start=45.0, p_end=17.3, solid_frac=0.5, p_intervals=5)

    print("\nRunning Fractional Stages (Stages 1-10, 5% total LMO each)...")
    # New API: No p_intervals needed as it's a whole-ocean equilibrium calculation
    results_frac = mo.run_fractional_stages(melt_0, p_start=17.3, p_end=0.01, vol_step=0.05, n_stages=10)

    for res in results_frac:
        print(f"Stage {res['stage']}: P_base {res['p_base']:.2f} kbar, P_top {res['p_top']:.2f} kbar.")
        print(f"  Modes: {res['layer_modes']}")

    print(f"\nNumber of fractional stages completed: {len(results_frac)}")
    
    # Verification 1: Did we reach 10 stages?
    assert len(results_frac) == 10, f"Expected 10 stages, got {len(results_frac)}"

    # Verification 2: Check Stage 1 pressure (~17.3 kbar)
    p_base_1 = results_frac[0]['p_base']
    assert 17.0 < p_base_1 < 17.6, f"Expected Stage 1 P_base around 17.3, got {p_base_1:.2f}"

    # Verification 3: Check Stage 7 pressure (~6.2 kbar for TWM)
    p_base_7 = results_frac[6]['p_base']
    assert 6.0 < p_base_7 < 7.0, f"Expected Stage 7 P_base around 6.2, got {p_base_7:.2f}"

    # Verification 3: Does Spinel appear in the intermediate stages?
    # Johnson et al. (2021) show aluminous spinel ('spl') in the upper mantle cumulates.
    has_spl = False
    for res in results_frac[2:8]: # Check intermediate stages
        if 'spl' in res['layer_modes'] or 'Spl' in res['layer_modes']:
            has_spl = True
            print(f"Confirmed: Spinel found in Stage {res['stage']} ({res['layer_modes'].get('spl', res['layer_modes'].get('Spl'))*100:.2f}%)")
            break
    assert has_spl, "Spinel should be present in intermediate stages."

    # Verification 4: Does Ilmenite appear in the final stages?
    has_ilm = False
    for res in results_frac[-2:]: 
        if 'ilm' in res['layer_modes'] or 'Ilm' in res['layer_modes']:
            has_ilm = True
            print(f"Confirmed: Ilmenite found in Stage {res['stage']} ({res['layer_modes'].get('ilm', res['layer_modes'].get('Ilm'))*100:.1f}%)")
            break
    assert has_ilm, "Ilmenite should be present in final stages."
    
    # NOTE: Depending on the database version, ilmenite might be 'ilm' or 'Ilm'.
    # If it doesn't show up with 5 intervals, we might need more intervals or higher resolution,
    # but the logic fix should be evident in the pressure and stage count.
    
    print("\nLMO Fix Verification Successful!")

if __name__ == "__main__":
    test_lmo_crystallisation()
