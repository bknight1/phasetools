import numpy as np

def get_oxide_apfu(out, ph, oxides):
    """Extract oxide amounts from APFU output for a specific phase."""
    ph_index = out.ph.index(ph)
    phase_obj = out.SS_vec[ph_index]

    oxide_values = np.array(phase_obj.Comp_apfu, dtype=float)
    oxide_names = out.oxides
    oxide_moles_dict = {ox: value for ox, value in zip(oxide_names, oxide_values)}

    results = {}
    for oxide in oxides:
        results[oxide] = oxide_moles_dict.get(oxide, 0.0)

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
