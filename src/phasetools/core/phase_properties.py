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
    """Extract phase fraction (mol, wt, or vol) from MAGEMin output."""
    try:
        phase_ind = MAGEMinOutput.ph.index(phase)
        if sys_in.casefold() == 'wt':
            data = MAGEMinOutput.ph_frac_wt[phase_ind]
        elif sys_in.casefold() == 'vol':
            data = MAGEMinOutput.ph_frac_vol[phase_ind]
        else:
            data = MAGEMinOutput.ph_frac[phase_ind]
    except (ValueError, IndexError):
        data = 0.
    return data
