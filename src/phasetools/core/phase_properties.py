import numpy as np

def get_oxide_apfu(out, ph, oxides):
    """Extract oxide amounts from APFU output for a specific phase."""
    try:
        ph_index = out.ph.index(ph)
        phase_obj = out.SS_vec[ph_index]

        oxide_values = np.array(phase_obj.Comp_apfu, dtype=float)
        oxide_names = [str(ox) for ox in out.oxides]
        oxide_moles_dict = {ox: value for ox, value in zip(oxide_names, oxide_values)}

        results = {}
        for oxide in oxides:
            results[oxide] = oxide_moles_dict.get(oxide, 0.0)
    except (ValueError, IndexError):
        results = {oxide: 0.0 for oxide in oxides}

    return results

def get_phase_chemistry(out, ph, oxides, sys_in):
    """
    Extract oxide concentrations (wt% or mol%) for a specific phase.
    
    Parameters
    ----------
    out : object
        MAGEMin output object.
    ph : str
        Phase name.
    oxides : list[str]
        List of oxides to extract.
    sys_in : str
        Unit system ('wt' or 'mol').
        
    Returns
    -------
    dict
        Oxide concentrations.
    """
    try:
        ph_index = out.ph.index(ph)
        phase_obj = out.SS_vec[ph_index]

        if sys_in.casefold() == 'wt':
            # Comp_wt is weight fraction (0-1) for oxides in the phase
            values = np.array(phase_obj.Comp_wt, dtype=float) * 100.0
        else:
            # Comp is molar fraction (0-1) for oxides in the phase
            values = np.array(phase_obj.Comp, dtype=float) * 100.0
            
        oxide_names = [str(ox) for ox in out.oxides]
        oxide_dict = {ox: value for ox, value in zip(oxide_names, values)}

        results = {}
        for oxide in oxides:
            results[oxide] = oxide_dict.get(oxide, 0.0)
    except (ValueError, IndexError):
        results = {oxide: 0.0 for oxide in oxides}

    return results

def extract_end_member(phase, MAGEMinOutput, end_member, sys_in):
    """Extract specific end-member fraction from MAGEMin output."""
    try:
        phase_ind = MAGEMinOutput.ph.index(phase)
        em_index = MAGEMinOutput.SS_vec[phase_ind].emNames.index(end_member)
        if sys_in.casefold() == 'wt':
            data = MAGEMinOutput.SS_vec[phase_ind].emFrac_wt[em_index]
        else:
            data = MAGEMinOutput.SS_vec[phase_ind].emFrac[em_index]
    except (ValueError, IndexError):
        data = 0.
    return data

def phase_frac(phase, MAGEMinOutput, sys_in):
    """
    Extract phase fraction (mol, wt, or vol) from MAGEMin output.
    If multiple phases with the same name exist (e.g. solvus), returns the sum.
    """
    try:
        total = 0.0
        found = False
        for i, ph_name in enumerate(MAGEMinOutput.ph):
            if str(ph_name) == phase:
                found = True
                if sys_in.casefold() == 'wt':
                    total += MAGEMinOutput.ph_frac_wt[i]
                elif sys_in.casefold() == 'vol':
                    total += MAGEMinOutput.ph_frac_vol[i]
                else:
                    total += MAGEMinOutput.ph_frac[i]
        
        if not found:
            return 0.0
        return float(total)
    except:
        return 0.0

def get_phase_mg_number(out, ph):
    """
    Calculate Mg# (molar Mg / (Mg + Fe_total)) for a specific phase.
    
    NOTE: This uses total iron (FeOt). For the divalent-only Mg#, 
    use `get_phase_mg2_number`.
    
    Matches the logic used by MAGEMin's 'ss_MgNum' mode by pulling 
    MgO and FeO directly from the phase's Comp_apfu array.
    Supports 'FeO', 'Fe' (sb24), and 'Fe2O3' fallback.
    """
    try:
        ph_index = out.ph.index(ph)
        phase_obj = out.SS_vec[ph_index]
        
        oxide_names = [str(ox) for ox in out.oxides]
        
        # Pull Mg
        try:
            mg_idx = oxide_names.index('MgO')
            mg = float(phase_obj.Comp_apfu[mg_idx])
        except ValueError:
            mg = 0.0
            
        # Pull Fe (total iron atoms)
        fe = 0.0
        if 'FeO' in oxide_names:
            fe_idx = oxide_names.index('FeO')
            fe += float(phase_obj.Comp_apfu[fe_idx])
            # If both are present, we sum them (though unlikely in standard MAGEMin output)
            if 'Fe2O3' in oxide_names:
                fe2o3_idx = oxide_names.index('Fe2O3')
                fe += 2.0 * float(phase_obj.Comp_apfu[fe2o3_idx])
        elif 'Fe' in oxide_names:
            fe_idx = oxide_names.index('Fe')
            fe += float(phase_obj.Comp_apfu[fe_idx])
        elif 'Fe2O3' in oxide_names:
            fe2o3_idx = oxide_names.index('Fe2O3')
            fe += 2.0 * float(phase_obj.Comp_apfu[fe2o3_idx])
        else:
            # No iron components found
            if mg == 0: return 0.0
            return 1.0 # Pure Mg phase
            
        denominator = mg + fe
        if denominator == 0:
            return 0.0

        return mg / denominator
    except (ValueError, IndexError, AttributeError):
        return 0.0

def get_phase_fe_split(out, ph):
    """
    Calculate Fe2+ and Fe3+ amounts for a phase using an excess oxygen heuristic.
    
    Works for both traditional FeO-Fe2O3 bases and MAGEMin's O-basis (ig, mp).
    """
    try:
        ox_to_query = ['FeO', 'Fe2O3', 'Fe', 'O']
        apfu = get_oxide_apfu(out, ph, ox_to_query)
        
        feo_val = apfu.get("FeO", 0.0)
        fe2o3_val = apfu.get("Fe2O3", 0.0)
        fe_metal_val = apfu.get("Fe", 0.0)
        atomic_o = apfu.get("O", 0.0)

        # 1. Calculate Total Fe atoms (Atoms per formula unit)
        if atomic_o > 0:
            # MAGEMin O-basis (ig, mp) or sb24 basis
            # If Fe component is present (sb24), use it; otherwise FeO is total iron.
            if fe_metal_val > 0:
                total_fe = fe_metal_val
            else:
                total_fe = feo_val
        else:
            # Traditional FeO/Fe2O3 basis
            total_fe = feo_val + 2.0 * fe2o3_val

        # 2. Calculate Fe3+ atoms using excess oxygen heuristic
        # excess_o identifies oxygen atoms added beyond the stoichiometric baseline.
        # Works for both 'O as total oxygen' and 'O as excess oxygen' components.
        excess_o = max(atomic_o - np.round(atomic_o), 0.0)
        fe3 = 2.0 * fe2o3_val + 2.0 * excess_o
        
        # 3. Divalent iron is the remainder
        fe2 = max(total_fe - fe3, 0.0)

        return {
            "fe2": fe2, 
            "fe3": fe3, 
        }
    except:
        return {"fe2": 0.0, "fe3": 0.0}

def get_phase_mg2_number(out, ph):
    """
    Calculate Mg# (molar Mg / (Mg + Fe2+)) for a specific phase.
    
    Uses an excess oxygen heuristic to split total iron into Fe2+ and Fe3+.
    """
    try:
        apfu = get_oxide_apfu(out, ph, ['MgO'])
        mg = apfu.get('MgO', 0.0)
        
        split = get_phase_fe_split(out, ph)
        fe2 = split['fe2']
        
        denominator = mg + fe2
        if denominator == 0:
            return 0.0
            
        return mg / denominator
    except:
        return 0.0

def calculate_kd_fe_mg(out, phase1, phase2, use_fe2_only=False):
    """
    Calculate the Fe-Mg distribution coefficient (Kd) between two phases.
    Kd = (Fe/Mg)_phase1 / (Fe/Mg)_phase2
    
    Parameters
    ----------
    out : object
        MAGEMin output object.
    phase1 : str
        Name of the first phase (e.g., 'g' for garnet).
    phase2 : str
        Name of the second phase (e.g., 'cpx' for clinopyroxene).
    use_fe2_only : bool, default False
        If True, uses divalent iron (Fe2+) for the calculation. 
        If False, uses total iron (FeOt).
        
    Returns
    -------
    float
        The calculated Kd value. Returns NaN if phase2 Mg# is 0 or 1.
    """
    if use_fe2_only:
        mg1 = get_phase_mg2_number(out, phase1)
        mg2 = get_phase_mg2_number(out, phase2)
    else:
        mg1 = get_phase_mg_number(out, phase1)
        mg2 = get_phase_mg_number(out, phase2)

    if mg1 <= 0 or mg1 >= 1 or mg2 <= 0 or mg2 >= 1:
        return np.nan

    # Kd = (Fe1/Mg1) / (Fe2/Mg2)
    # Since Mg# = Mg / (Mg + Fe), then Fe/Mg = (1 - Mg#) / Mg#
    return ((1.0 - mg1) / mg1) / ((1.0 - mg2) / mg2)



