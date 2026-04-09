from juliacall import Main as jl, convert as jlconvert
from phasetools import MAGEMin_C

def single_point_minimization_with_conversion(P, T, data, X, Xoxides, sys_in="wt", rm_list=None):
    """
    Perform single-point minimisation with automatic Python-to-Julia conversion, including optional removal list.

    Parameters
    ----------
    P : float
        Pressure in kbar.
    T : float
        Temperature in Celsius.
    data : object
        Initialised MAGEMin data object.
    X : list[float]
        Bulk composition values.
    Xoxides : list[str]
        Bulk composition oxide names.
    sys_in : str, optional
        Input system basis ("wt", "mol", etc.), by default "wt".
    rm_list : list[str], optional
        List of phases to remove, by default None.

    Returns
    -------
    object
        MAGEMin output object for the given P-T point.
    """
    # Convert Python inputs to Julia-compatible types
    P_jl = jlconvert(jl.Float64, P)
    T_jl = jlconvert(jl.Float64, T)
    X_jl = jlconvert(jl.Vector[jl.Float64], X)
    Xox_jl = jlconvert(jl.Vector[jl.String], Xoxides)

    # Convert rm_list if provided
    if rm_list is not None:
        rm_list_conv = jlconvert(jl.Vector[jl.String], rm_list)
        rm_list_jl = MAGEMin_C.remove_phases(rm_list_conv, data.db)
    else:
        rm_list_jl = None

    # Call MAGEMin single_point_minimization
    result = MAGEMin_C.single_point_minimization(P_jl, T_jl, data, X=X_jl, Xoxides=Xox_jl, sys_in=sys_in, rm_list=rm_list_jl)

    return result
