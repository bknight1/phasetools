import numpy as np
from scipy.interpolate import PchipInterpolator


def create_PTt_path(P, T, t, nsteps):
    """
    Create a PTt path array by interpolating the given PTt data while
    ensuring that the original P, T, and t values are embedded in the output.
    
    The final output returns all the original data points as well as 
    interpolated values so that none of the original (embedded) values are lost.
    
    Parameters:
        P (ndarray): Pressure data array.
        T (ndarray): Temperature data array.
        t (ndarray): Time data array.
        nsteps (int): The number of interpolation points desired.
    
    Returns:
        ndarray: A 2D array with three columns (pressure, temperature, time).
                 It contains both the original and the interpolated points.
    """
    n_rows = len(P)
    # Create PCHIP interpolators for each quantity
    pressure_interp = PchipInterpolator(np.arange(n_rows), P)
    temp_interp     = PchipInterpolator(np.arange(n_rows), T)
    time_interp     = PchipInterpolator(np.arange(n_rows), t)
    
    # Obtain interpolation s-values over a uniform grid
    s_interp = np.linspace(0, n_rows - 1, nsteps)
    # Merge with the original indices to embed original values
    s_all = np.unique(np.concatenate([np.arange(n_rows), s_interp]))
    
    # Evaluate the interpolators on the combined s values
    P_all = pressure_interp(s_all)
    T_all = temp_interp(s_all)
    t_all = time_interp(s_all)
    
    ### Optional: Check for nearly identical points along the path and adjust.
    epsP = 0.1 
    epsT = 0.01
    # Compute differences between consecutive points
    dP = P_all[1:] - P_all[:-1]
    dT = T_all[1:] - T_all[:-1]
    # Where differences are below eps thresholds, add a small offset to uniquely separate the points.
    mask = (np.abs(dP) < epsP) & (np.abs(dT) < epsT)
    if np.any(mask):
        # Note: The adjustment below affects only points after the first occurrence.
        P_all[1:][mask] += 0.1 * epsP
        T_all[1:][mask] += 0.1 * epsT

    # Create the combined PTt path and return.
    PTt_path = np.column_stack((P_all, T_all, t_all))
    return PTt_path