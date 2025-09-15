import numpy as np
import sys
import warnings
import juliacall
from juliacall import Main as jl, convert as jlconvert
from .bulk_rock_functions import *

from scipy.optimize import root_scalar

from pyMAGEMin import MAGEMin_C


def extract_end_member(phase, MAGEMinOutput, end_member, sys_in):
    try:
        phase_ind = MAGEMinOutput.ph.index(phase)
        em_index = MAGEMinOutput.SS_vec[phase_ind].emNames.index(end_member)
        if sys_in.casefold() == 'wt':
            data = MAGEMinOutput.SS_vec[phase_ind].emFrac_wt[em_index]
        else:
            data = MAGEMinOutput.SS_vec[phase_ind].emFrac[em_index]
    except:
        data = 0.
    return data

def phase_frac(phase, MAGEMinOutput, sys_in):
    try:
        phase_ind = MAGEMinOutput.ph.index(phase)
        if sys_in.casefold() == 'wt':
            data = MAGEMinOutput.ph_frac_wt[phase_ind]
        elif sys_in.casefold() == 'vol':
            data = MAGEMinOutput.ph_frac_vol[phase_ind]
            # ph_names = MAGEMinOutput.ph
            # n_ph = len(ph_names)
            # V = np.zeros((n_ph, 1))
            # for i, ph in enumerate(ph_names):
            #     ids = [j for j, p in enumerate(MAGEMinOutput.ph) if p == ph]
            #     if ids:
            #         rho = sum(MAGEMinOutput.SS_vec[j].rho if j < MAGEMinOutput.n_SS 
            #                 else MAGEMinOutput.PP_vec[j - MAGEMinOutput.n_SS].rho for j in ids) / len(ids)
            #         V[i, 0] = sum(MAGEMinOutput.ph_frac_wt[j] for j in ids) / rho
            # V /= np.sum(V)
            # data = V[phase_ind]
        else:
            data = MAGEMinOutput.ph_frac[phase_ind]
    except:
        data = 0.
    return data

class MAGEMinGarnetCalculator:
    def __init__(self):
        pass

    def generate_2D_grid_gt_endmembers(self, P, T, data, X, Xoxides, sys_in, rm_list=None):
        """
        Generates a 2D grid of garnet endmember mole fractions by calling MAGEMin.
        Returns arrays for garnet mole, weight, volume fractions and endmember fractions.
        """
        Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        X = jlconvert(jl.Vector[jl.Float64], X)
        out = MAGEMin_C.multi_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in, rm_list=rm_list)
        sys.stdout.flush()

        gt_mol_frac = np.zeros_like(P)
        gt_wt_frac = np.zeros_like(P)
        gt_vol_frac = np.zeros_like(P)
        py_arr  = np.zeros_like(P)
        alm_arr  = np.zeros_like(P)
        spss_arr  = np.zeros_like(P)
        gr_arr  = np.zeros_like(P)
        kho_arr  = np.zeros_like(P)

        for i in range(len(T)):
            gt_mol_frac[i] = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='mol')
            gt_wt_frac[i]  = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='wt')
            gt_vol_frac[i] = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='vol')

            py_arr[i]   = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="py", sys_in=sys_in)
            alm_arr[i]  = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="alm", sys_in=sys_in)
            spss_arr[i] = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="spss", sys_in=sys_in)
            gr_arr[i]   = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="gr", sys_in=sys_in)
            kho_arr[i]  = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="kho", sys_in=sys_in)

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, py_arr, alm_arr, spss_arr, gr_arr, kho_arr

    def generate_2D_grid_gt_elements(self, P, T, data, X, Xoxides, sys_in, rm_list=None):
        """
        Generates 2D grids for garnet elements based on endmember fractions.
        Returns garnet fractions plus element molar fractions.
        """
        gt_mol_frac, gt_wt_frac, gt_vol_frac, py_arr, alm_arr, spss_arr, gr_arr, kho_arr = \
            self.generate_2D_grid_gt_endmembers(P, T, data, X, Xoxides, sys_in, rm_list)
            
        Mgi = np.zeros_like(P)
        Mni = np.zeros_like(P)
        Fei = np.zeros_like(P)
        Cai = np.zeros_like(P)
        for i in range(len(py_arr)):

            garnet_fractions = {"py": py_arr[i], "alm": alm_arr[i], "gr": gr_arr[i], "spss": spss_arr[i], "kho": kho_arr[i]}
            
            # Calculate the molar fractions of the elements in garnet
            mole_fractions = calculate_molar_fractions(garnet_fractions)

            if sys_in == 'wt':
                mole_fractions_list = [mole_fractions["Mg"], mole_fractions["Mn"], mole_fractions["Fe"], mole_fractions["Ca"]]
                wt_percent_list = convert_mol_percent_to_wt_percent(mole_fractions_list, ["Mg", "Mn", "Fe", "Ca"], atomic_mass_dict)

                Mgi[i] = wt_percent_list[0]
                Mni[i] = wt_percent_list[1]
                Fei[i] = wt_percent_list[2]
                Cai[i] = wt_percent_list[3]

            else:
                Mgi[i] = mole_fractions["Mg"]
                Mni[i] = mole_fractions["Mn"]
                Fei[i] = mole_fractions["Fe"]
                Cai[i] = mole_fractions["Ca"]

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, Mgi, Mni, Fei, Cai

    def gt_single_point_calc_endmembers(self, P, T, data, X, Xoxides, sys_in, rm_list=None):
        """
        Calculate garnet endmember fractions for a single P-T point.
        Returns fractions and the full MAGEMin output.
        """
        Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        X = jlconvert(jl.Vector[jl.Float64], X)


        out = MAGEMin_C.single_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in, rm_list=rm_list)


        sys.stdout.flush()
        gt_frac = gt_wt = gt_vol = py = alm = spss = gr = kho = 0.
        if 'g' in out.ph:
            gt_frac  = phase_frac(phase="g", MAGEMinOutput=out, sys_in='mol')
            gt_wt    = phase_frac(phase="g", MAGEMinOutput=out, sys_in='wt')
            gt_vol   = phase_frac(phase="g", MAGEMinOutput=out, sys_in='vol')

            py  = extract_end_member(phase="g", MAGEMinOutput=out, end_member="py", sys_in=sys_in)
            alm = extract_end_member(phase="g", MAGEMinOutput=out, end_member="alm", sys_in=sys_in)
            spss= extract_end_member(phase="g", MAGEMinOutput=out, end_member="spss", sys_in=sys_in)
            gr  = extract_end_member(phase="g", MAGEMinOutput=out, end_member="gr", sys_in=sys_in)
            kho = extract_end_member(phase="g", MAGEMinOutput=out, end_member="kho", sys_in=sys_in)
        
        return gt_frac, gt_wt, gt_vol, py, alm, spss, gr, kho, out

    def gt_single_point_calc_elements(self, P, T, data, X, Xoxides, sys_in, rm_list=None):
        """
        Calculate garnet element fractions for a single P-T point.
        Returns garnet fractions and molar fractions of elements.
        """
        gt_frac, gt_wt, gt_vol, py, alm, spss, gr, kho, out = \
            self.gt_single_point_calc_endmembers(P, T, data, X, Xoxides, sys_in, rm_list)
        
        garnet_fractions = {"py": py, "alm": alm, "gr": gr, "spss": spss, "kho": kho}
        
        # Calculate the molar fractions of the elements in garnet
        mole_fractions = calculate_molar_fractions(garnet_fractions)

        if sys_in == 'wt':
            mole_fractions_list = [mole_fractions["Mg"], mole_fractions["Mn"], mole_fractions["Fe"], mole_fractions["Ca"]]
            wt_percent_list = convert_mol_percent_to_wt_percent(mole_fractions_list, ["Mg", "Mn", "Fe", "Ca"], atomic_mass_dict)

            Mg = wt_percent_list[0]
            Mn = wt_percent_list[1]
            Fe = wt_percent_list[2]
            Ca = wt_percent_list[3]

        else:
            Mg = mole_fractions["Mg"]
            Mn = mole_fractions["Mn"]
            Fe = mole_fractions["Fe"]
            Ca = mole_fractions["Ca"]
        
        return gt_frac, gt_wt, gt_vol, Mg, Mn, Fe, Ca, out

    def gt_along_path(self, P, T, data, X, Xoxides, sys_in, fractionate=False, rm_list=None):
        """
        Calculates garnet composition and fractionation along a P-T path.
        Returns arrays of garnet fractions and element mole fractions along the path.
        """
        Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        X = jlconvert(jl.Vector[jl.Float64], X)
        n_points = len(P)
        gt_wt_frac = np.zeros(n_points)
        gt_mol_frac = np.zeros(n_points)
        gt_vol_frac = np.zeros(n_points)
        d_gt_mol_frac = np.zeros(n_points)
        d_gt_wt_frac = np.zeros(n_points)
        Mgi = np.zeros(n_points)
        Mni = np.zeros(n_points)
        Fei = np.zeros(n_points)
        Cai = np.zeros(n_points)


        
        ### loop over P-T points to get the garnet data
        for i in range(n_points):
            gt_frac, gt_wt, gt_vol, py, alm, spss, gr, kho, out = \
                self.gt_single_point_calc_endmembers(P[i], T[i], data, X, Xoxides, sys_in, rm_list)

            ### MAGEMin converts to a compatible format, so get those here
            Xoxides = out.oxides
            if sys_in == 'wt':
                X = out.bulk_wt
            else:
                X = out.bulk
            

            gt_mol_frac[i] = gt_frac
            gt_wt_frac[i] = gt_wt
            gt_vol_frac[i] = gt_vol


            garnet_fractions = {"py": py, "alm": alm, "gr": gr, "spss": spss, "kho": kho}
            
            # Calculate the molar fractions of the elements in garnet
            mole_fractions = calculate_molar_fractions(garnet_fractions)

            if sys_in == 'wt' and 'g' in out.ph:
                mole_fractions_list = [mole_fractions["Mg"], mole_fractions["Mn"], mole_fractions["Fe"], mole_fractions["Ca"]]
                wt_percent_list = convert_mol_percent_to_wt_percent(mole_fractions_list, ["Mg", "Mn", "Fe", "Ca"], atomic_mass_dict)

                Mgi[i] = wt_percent_list[0]
                Mni[i] = wt_percent_list[1]
                Fei[i] = wt_percent_list[2]
                Cai[i] = wt_percent_list[3]

            else: ### these return 0 anyway so can be used for both if garnet is not present
                Mgi[i] = mole_fractions["Mg"]
                Mni[i] = mole_fractions["Mn"]
                Fei[i] = mole_fractions["Fe"]
                Cai[i] = mole_fractions["Ca"]

            if fractionate:
                from .MAGEMin_functions import PhaseFunctions
                phase_functions = PhaseFunctions()
                # Initialize changes in phase fractions
                d_gt_mol_frac[i] = gt_mol_frac[i] - gt_mol_frac[i - 1] if i > 0 else gt_mol_frac[0]
                d_gt_wt_frac[i] = gt_wt_frac[i] - gt_wt_frac[i - 1] if i > 0 else gt_wt_frac[0]
                ### fractionate the g(arnet) phase
                X = phase_functions.fractionate_phase("g", d_gt_mol_frac[i], d_gt_wt_frac[i], out, X, sys_in)

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, d_gt_mol_frac, d_gt_wt_frac, Mgi, Mni, Fei, Cai


class PhaseFunctions:
    def __init__(self):
        pass

    def find_phase_in(self, P, bracket, data, phase, sys_in='mol', tol=1e-2, verbose=False):
        """
        Finds the solidus temperature (where the phase fraction becomes greater than zero)
        using a robust root-finding algorithm.

        Parameters:
            P: Pressure (kbar)
            bracket: (T_low, T_high) tuple bracketing the solidus
            data: MAGEMin data object
            phase: str, phase name (e.g. 'liq')
            sys_in: str, system units
            tol: float, tolerance for convergence (in degrees)
            verbose: bool, print progress

        Returns:
            solidus_T: float, solidus temperature (°C)
        """
        def solidus_func(T):
            out = MAGEMin_C.single_point_minimization(P, T, data)
            phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_in=sys_in)
            if verbose:
                print(f"phase frac: {phasefrac:.4f}, T: {T:.2f}")
            return phasefrac - 1e-5  # Slightly above zero for numerical stability

        T_low, T_high = bracket
        result = root_scalar(solidus_func, bracket=[T_low, T_high], method='bisect', xtol=tol)
        
        if not result.converged:
            raise RuntimeError("Solidus search did not converge. Adjust bracket or check function.")
        
        return result.root

    def find_phase_saturation(self, P, bracket, data, phase, sys_in='mol', tol=1e-2, verbose=False):
        """
        Finds the liquidus temperature (where the phase fraction reaches unity)
        using a robust root-finding algorithm.

        Parameters:
            P: Pressure (kbar)
            bracket: (T_low, T_high) tuple bracketing the liquidus
            data: MAGEMin data object
            phase: str, phase name (e.g. 'liq')
            sys_in: str, system units
            tol: float, tolerance for convergence (in degrees)
            verbose: bool, print progress

        Returns:
            liquidus_T: float, liquidus temperature (°C)
        """
        def liquidus_func(T):
            out = MAGEMin_C.single_point_minimization(P, T, data)
            phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_in=sys_in)
            if verbose:
                print(f"phase frac: {phasefrac:.4f}, T: {T:.2f}")
            return phasefrac - (1.0 - 1e-5)  # Slightly below one for numerical stability

        T_low, T_high = bracket
        result = root_scalar(liquidus_func, bracket=[T_low, T_high], method='bisect', xtol=tol)
        if not result.converged:
            raise RuntimeError("Liquidus search did not converge. Adjust bracket or check function.")
        return result.root

    def fractionate_phase(self, phase, d_mol_frac, d_wt_frac, out, X, sys_in):
        """
        Handles the fractionation logic for any phase.

        Parameters:
            i (int): Current index in the P-T path.
            phase (str): Phase name (e.g., "g" for garnet).
            mol_frac (array): Molar fractions of the phase along the path.
            wt_frac (array): Weight fractions of the phase along the path.
            out (object): MAGEMin output object for the current step.
            X (array): Bulk composition array.
            sys_in (str): Input system type ('wt' or 'mol').

        Returns:
            tuple: Updated bulk composition (X), change in molar fraction (d_mol_frac),
                and change in weight fraction (d_wt_frac).
        """

        # Only adjust bulk composition if the phase fraction is positive
        if phase in out.ph:
            phase_ind = out.ph.index(phase)
            if sys_in == "wt" and d_wt_frac > 0:
                X = out.bulk_wt - (np.array(out.bulk_wt) * np.array(out.SS_vec[phase_ind].Comp_wt) * d_wt_frac)
            elif sys_in == "mol" and d_mol_frac > 0:
                X = out.bulk - (np.array(out.bulk) * np.array(out.SS_vec[phase_ind].Comp) * d_mol_frac)

        return X