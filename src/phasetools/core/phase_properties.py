import warnings

import numpy as np


def _phase_indices(out: object, phase: str, instance: int | str = 0) -> list[int]:
    """Return the index/indices of ``phase`` in ``out.ph``.

    Parameters
    ----------
    out : object
        MAGEMin output object with a ``ph`` attribute.
    phase : str
        Phase name.
    instance : int or {'all'}, default=0
        Which instance to resolve.  An integer index selects that instance
        (negative indices count from the end).  ``'all'`` returns every
        occurrence.  When the phase appears more than once (a solvus -- e.g.
        two coexisting clinopyroxenes or amphiboles, reported by MAGEMin as
        repeated entries in ``out.ph``), a warning notes how many other
        instances exist.

    Returns
    -------
    list[int]
        Indices of ``phase`` in ``out.ph`` (empty list if the phase is
        absent or the requested instance does not exist).
    """
    idx = [i for i, p in enumerate(out.ph) if str(p) == phase]
    if instance == 'all':
        return idx
    if not isinstance(instance, (int, np.integer)):
        raise TypeError(f"instance must be an integer index or 'all', got {instance!r}")
    if not idx:
        return []
    if instance < 0:
        instance = len(idx) + instance
    if instance < 0 or instance >= len(idx):
        warnings.warn(
            f"phase '{phase}' has {len(idx)} instance(s); requested instance "
            f"{instance} does not exist -- treating as absent.",
            UserWarning, stacklevel=3)
        return []
    if len(idx) > 1:
        warnings.warn(
            f"phase '{phase}' has {len(idx)} instances (solvus) in this "
            f"output; using instance {instance} (there are {len(idx) - 1} "
            f"others). Use instance='all' for per-instance arrays.",
            UserWarning, stacklevel=3)
    return [idx[instance]]

def get_oxide_apfu(out, ph, oxides, instance=0):
    """Extract oxide amounts from APFU output for a specific phase.

    Parameters
    ----------
    out : object
        MAGEMin output object.
    ph : str
        Phase name.
    oxides : list[str]
        Oxides to extract.
    instance : int or {'all'}, default=0
        For a phase that appears multiple times (a solvus), an integer
        index selects that instance (warning if more than one exists) and
        ``'all'`` returns one value per instance as numpy arrays.

    Returns
    -------
    dict
        ``{oxide: value}`` for an integer index, or ``{oxide: ndarray}``
        (one entry per phase instance) for ``'all'``.
    """
    idx = _phase_indices(out, ph, instance)
    if not idx:
        return {oxide: 0.0 for oxide in oxides} if instance != 'all' \
            else {oxide: np.zeros(0) for oxide in oxides}
    oxide_names = [str(ox) for ox in out.oxides]
    per = []
    for i in idx:
        try:
            values = np.array(out.SS_vec[i].Comp_apfu, dtype=float)
            per.append({ox: float(values[oxide_names.index(ox)])
                        if ox in oxide_names else 0.0 for ox in oxides})
        except (ValueError, IndexError, AttributeError):
            per.append({ox: 0.0 for ox in oxides})
    if instance == 'all':
        return {ox: np.array([d[ox] for d in per]) for ox in oxides}
    return per[0]

def get_phase_chemistry(out, ph, oxides, sys_in, instance=0):
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
    instance : int or {'all'}, default=0
        For a phase that appears multiple times (a solvus), an integer
        index selects that instance (warning if more than one exists) and
        ``'all'`` returns one value per instance as numpy arrays.
        
    Returns
    -------
    dict
        Oxide concentrations for an integer index, or per-instance arrays
        for ``'all'``.
    """
    idx = _phase_indices(out, ph, instance)
    if not idx:
        return {oxide: 0.0 for oxide in oxides} if instance != 'all' \
            else {oxide: np.zeros(0) for oxide in oxides}
    oxide_names = [str(ox) for ox in out.oxides]
    per = []
    for i in idx:
        try:
            phase_obj = out.SS_vec[i]
            if sys_in.casefold() == 'wt':
                # Comp_wt is weight fraction (0-1) for oxides in the phase
                values = np.array(phase_obj.Comp_wt, dtype=float) * 100.0
            else:
                # Comp is molar fraction (0-1) for oxides in the phase
                values = np.array(phase_obj.Comp, dtype=float) * 100.0
            per.append({ox: float(values[oxide_names.index(ox)])
                        if ox in oxide_names else 0.0 for ox in oxides})
        except (ValueError, IndexError, AttributeError):
            per.append({ox: 0.0 for ox in oxides})
    if instance == 'all':
        return {ox: np.array([d[ox] for d in per]) for ox in oxides}
    return per[0]

def extract_end_member(phase, MAGEMinOutput, end_member, sys_in, instance=0):
    """Extract specific end-member fraction from MAGEMin output.

    For a phase that appears multiple times (a solvus), ``instance=0``
    returns the first instance (warning if more than one exists) and
    ``instance='all'`` returns one value per instance as a numpy array.
    """
    idx = _phase_indices(MAGEMinOutput, phase, instance)
    if not idx:
        return 0.0 if instance != 'all' else np.zeros(0)
    vals = []
    for i in idx:
        try:
            em_index = MAGEMinOutput.SS_vec[i].emNames.index(end_member)
            if sys_in.casefold() == 'wt':
                vals.append(float(MAGEMinOutput.SS_vec[i].emFrac_wt[em_index]))
            else:
                vals.append(float(MAGEMinOutput.SS_vec[i].emFrac[em_index]))
        except (ValueError, IndexError, AttributeError):
            vals.append(0.0)
    return float(vals[0]) if instance != 'all' else np.array(vals)

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
    except (ValueError, IndexError, AttributeError, TypeError):
        return 0.0

def get_phase_mg_number(out, ph, instance=0):
    """
    Calculate Mg# (molar Mg / (Mg + Fe_total)) for a specific phase.
    
    NOTE: This uses total iron (FeOt). For the divalent-only Mg#, 
    use `get_phase_mg2_number`.
    
    Matches the logic used by MAGEMin's 'ss_MgNum' mode by pulling 
    MgO and FeO directly from the phase's Comp_apfu array.
    Supports 'FeO', 'Fe' (sb24), and 'Fe2O3' fallback.

    For a phase that appears multiple times (a solvus), ``instance=0``
    returns the first instance (warning if more than one exists) and
    ``instance='all'`` returns one value per instance as a numpy array.
    """
    idx = _phase_indices(out, ph, instance)
    if not idx:
        return 0.0 if instance != 'all' else np.zeros(0)
    oxide_names = [str(ox) for ox in out.oxides]

    def _mg(i):
        try:
            phase_obj = out.SS_vec[i]
            mg = float(phase_obj.Comp_apfu[oxide_names.index('MgO')]) if 'MgO' in oxide_names else 0.0
            fe = 0.0
            if 'FeO' in oxide_names:
                fe += float(phase_obj.Comp_apfu[oxide_names.index('FeO')])
                if 'Fe2O3' in oxide_names:
                    fe += 2.0 * float(phase_obj.Comp_apfu[oxide_names.index('Fe2O3')])
            elif 'Fe' in oxide_names:
                fe += float(phase_obj.Comp_apfu[oxide_names.index('Fe')])
            elif 'Fe2O3' in oxide_names:
                fe += 2.0 * float(phase_obj.Comp_apfu[oxide_names.index('Fe2O3')])
            else:
                return 1.0 if mg > 0 else 0.0
            denom = mg + fe
            return mg / denom if denom else 0.0
        except (ValueError, IndexError, AttributeError):
            return 0.0

    vals = [_mg(i) for i in idx]
    return float(vals[0]) if instance != 'all' else np.array(vals)

def get_phase_fe_split(out, ph, instance=0):
    """
    Calculate Fe2+ and Fe3+ amounts for a phase using an excess oxygen heuristic.
    
    Works for both traditional FeO-Fe2O3 bases and MAGEMin's O-basis (ig, mp).

    For a phase that appears multiple times (a solvus), ``instance=0``
    returns the first instance (warning if more than one exists) and
    ``instance='all'`` returns one value per instance as numpy arrays.
    """
    apfu = get_oxide_apfu(out, ph, ['FeO', 'Fe2O3', 'Fe', 'O'], instance=instance)

    feo = np.asarray(apfu.get("FeO", 0.0), dtype=float)
    fe2o3 = np.asarray(apfu.get("Fe2O3", 0.0), dtype=float)
    fem = np.asarray(apfu.get("Fe", 0.0), dtype=float)
    ato = np.asarray(apfu.get("O", 0.0), dtype=float)

    if ato.size == 0:
        empty = {"Fe2": np.zeros(0), "Fe3": np.zeros(0)}
        return {"Fe2": 0.0, "Fe3": 0.0} if instance != 'all' else empty

    # 1. Calculate Total Fe atoms (Atoms per formula unit)
    # Use np.where so each instance follows its own basis (O-bearing vs traditional).
    total_fe = np.where(
        ato > 0,
        np.where(fem > 0, fem, feo),
        feo + 2.0 * fe2o3
    )

    # 2. Calculate Fe3+ atoms using excess oxygen heuristic
    excess_o = np.maximum(ato - np.floor(ato), 0.0)
    fe3 = 2.0 * fe2o3 + 2.0 * excess_o

    # 3. Divalent iron is the remainder
    fe2 = np.maximum(np.asarray(total_fe) - fe3, 0.0)

    if instance != 'all':
        return {"Fe2": float(fe2.item()), "Fe3": float(fe3.item())}
    return {"Fe2": fe2, "Fe3": fe3}

def get_phase_mg2_number(out: object, ph: str, instance: int | str = 0) -> float | np.ndarray:
    """
    Calculate Mg# (molar Mg / (Mg + Fe2+)) for a specific phase.
    
    Uses an excess oxygen heuristic to split total iron into Fe2+ and Fe3+.
    """
    try:
        apfu = get_oxide_apfu(out, ph, ['MgO'], instance=instance)
        mg = apfu.get('MgO', 0.0)

        split = get_phase_fe_split(out, ph, instance=instance)
        fe2 = split['Fe2']

        denominator = np.asarray(mg + fe2, dtype=float)
        if instance != 'all':
            return float(mg / denominator) if denominator > 0 else 0.0
        return np.where(denominator > 0, mg / denominator, 0.0)
    except (ValueError, IndexError, AttributeError, KeyError, TypeError):
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



