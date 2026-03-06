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
        else:
            data = MAGEMinOutput.ph_frac[phase_ind]
    except:
        data = 0.
    return data

class MAGEMinGarnetCalculator:
    """High-level wrappers for garnet-focused MAGEMin calculations.

    This class provides convenience methods for:
    - single-point garnet chemistry,
    - 2D P-T grids,
    - and full path calculations with optional fractionation.

    Notes
    -----
    Most diffusion/inversion workflows should use the *element* methods
    (Fe, Mg, Mn, Ca). Endmember methods are still available for petrological
    diagnostics and phase-composition inspection.
    """
    def __init__(self):
        pass

    def _extract_garnet_elements_from_oxides(self, out, oxide_names, sys_in):
        """Extract garnet Fe-Mg-Mn-Ca from garnet oxide chemistry in MAGEMin output.

        Parameters
        ----------
        out : object
            MAGEMin output object for a single P-T point.
        oxide_names : list[str]
            Oxide name order corresponding to `Comp` / `Comp_wt` arrays.
        sys_in : str
            Output basis. If ``'wt'``, returns element wt fractions. Otherwise,
            returns cation mol fractions.

        Returns
        -------
        tuple[float, float, float, float]
            ``(Mg, Mn, Fe, Ca)`` in the requested basis.
        """
        if 'g' not in out.ph:
            return 0.0, 0.0, 0.0, 0.0

        ph_index = out.ph.index('g')

        if sys_in == 'wt':
            oxide_values = np.array(out.SS_vec[ph_index].Comp_wt, dtype=float)
            oxide_moles = {
                ox: (oxide_values[i] / molar_mass_dict[ox]) if ox in molar_mass_dict else 0.0
                for i, ox in enumerate(oxide_names)
            }
        else:
            oxide_values = np.array(out.SS_vec[ph_index].Comp, dtype=float)
            oxide_moles = {ox: oxide_values[i] for i, ox in enumerate(oxide_names)}

        mg_moles = oxide_moles.get("MgO", 0.0)
        mn_moles = oxide_moles.get("MnO", 0.0)
        ca_moles = oxide_moles.get("CaO", 0.0)
        fe_moles = oxide_moles.get("FeO", 0.0) + 2.0 * oxide_moles.get("Fe2O3", 0.0)

        total_cation_moles = mg_moles + mn_moles + fe_moles + ca_moles
        if total_cation_moles <= 0:
            return 0.0, 0.0, 0.0, 0.0

        Mg = mg_moles / total_cation_moles
        Mn = mn_moles / total_cation_moles
        Fe = fe_moles / total_cation_moles
        Ca = ca_moles / total_cation_moles

        if sys_in == 'wt':
            wt_percent_list = convert_mol_percent_to_wt_percent(
                [Mg, Mn, Fe, Ca],
                ["Mg", "Mn", "Fe", "Ca"],
                atomic_mass_dict,
            )
            Mg, Mn, Fe, Ca = [val / 100.0 for val in wt_percent_list]

        return Mg, Mn, Fe, Ca

    def generate_2D_grid_gt_endmembers(self, P, T, data, X, Xoxides, sys_in, rm_list=None):
        """Compute garnet endmember fractions over a P-T grid.

        Parameters
        ----------
        P, T : array-like
            Flattened pressure/temperature arrays (same shape).
        data : object
            MAGEMin data object from ``Initialize_MAGEMin``.
        X, Xoxides : array-like
            Bulk composition and oxide names.
        sys_in : str
            Composition basis used for extracted endmembers (``'mol'`` or ``'wt'``).
        rm_list : optional
            MAGEMin phase-removal list.

        Returns
        -------
        tuple of np.ndarray
            ``(gt_mol_frac, gt_wt_frac, gt_vol_frac, py, alm, spss, gr, kho)``.

        Notes
        -----
        Use this method when you specifically need garnet endmember proportions.
        For Fe-Mg-Mn-Ca workflows, prefer ``generate_2D_grid_gt_elements``.
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
        """Compute garnet element fractions (Mg, Mn, Fe, Ca) over a P-T grid.

        Element fractions are derived from the oxide chemistry of each
        equilibrium solution via ``_extract_garnet_elements_from_oxides``.
        The units of the returned element fractions are consistent with the
        ``sys_in`` basis used in the minimization.

        Returns
        -------
        tuple of np.ndarray
            ``(gt_mol_frac, gt_wt_frac, gt_vol_frac, Mg, Mn, Fe, Ca)``.
        """
        oxide_names = list(Xoxides)
        Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        X = jlconvert(jl.Vector[jl.Float64], X)
        out = MAGEMin_C.multi_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in, rm_list=rm_list)
        sys.stdout.flush()

        gt_mol_frac = np.zeros_like(P)
        gt_wt_frac = np.zeros_like(P)
        gt_vol_frac = np.zeros_like(P)
        Mgi = np.zeros_like(P)
        Mni = np.zeros_like(P)
        Fei = np.zeros_like(P)
        Cai = np.zeros_like(P)
        
        for i in range(len(T)):
            gt_mol_frac[i] = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='mol')
            gt_wt_frac[i]  = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='wt')
            gt_vol_frac[i] = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='vol')

            Mgi[i], Mni[i], Fei[i], Cai[i] = self._extract_garnet_elements_from_oxides(out[i], oxide_names, sys_in)

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, Mgi, Mni, Fei, Cai

    def gt_single_point_calc_endmembers(self, P, T, data, X, Xoxides, sys_in, rm_list=None):
        """Calculate single-point garnet endmember fractions.

        Returns
        -------
        tuple
            ``(gt_mol_frac, gt_wt_frac, gt_vol_frac, emDict_mol, emDict_wt, out)``
            where ``out`` is the raw MAGEMin output object.

        Notes
        -----
        This is the low-level chemistry view. For direct Fe-Mg-Mn-Ca values,
        use ``gt_single_point_calc_elements``.
        """
        Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        X = jlconvert(jl.Vector[jl.Float64], X)


        out = MAGEMin_C.single_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in, rm_list=rm_list)


        sys.stdout.flush()
        gt_frac = gt_wt = gt_vol = 0.
        if 'g' in out.ph:
            gt_frac  = phase_frac(phase="g", MAGEMinOutput=out, sys_in='mol')
            gt_wt    = phase_frac(phase="g", MAGEMinOutput=out, sys_in='wt')
            gt_vol   = phase_frac(phase="g", MAGEMinOutput=out, sys_in='vol')

            ph_index = out.ph.index('g')

            emNames = out.SS_vec[ph_index].emNames
            emFrac = out.SS_vec[ph_index].emFrac
            emFrac_wt = out.SS_vec[ph_index].emFrac_wt


            emDict_mol = {name: frac for name, frac in zip(emNames, emFrac)}
            emDict_wt  = {name: frac for name, frac in zip(emNames, emFrac_wt)}
        else:
            emDict_mol = {"py": 0., "alm": 0., "spss": 0., "gr": 0., "kho": 0.}
            emDict_wt  = {"py": 0., "alm": 0., "spss": 0., "gr": 0., "kho": 0.}

        return gt_frac, gt_wt, gt_vol, emDict_mol, emDict_wt, out

    def gt_single_point_calc_elements(self, P, T, data, X, Xoxides, sys_in, rm_list=None):
        """Calculate single-point garnet element fractions (Mg, Mn, Fe, Ca).

        Returns
        -------
        tuple
            ``(gt_mol_frac, gt_wt_frac, gt_vol_frac, Mg, Mn, Fe, Ca, out)``.

        Notes
        -----
        Element fractions are derived directly from the oxide chemistry of the
        minimized assemblage (via :meth:`_extract_garnet_elements_from_oxides`),
        using the specified ``sys_in`` basis.
        """
        oxide_names = list(Xoxides)
        Xoxides_jl = jlconvert(jl.Vector[jl.String], Xoxides)
        X_jl = jlconvert(jl.Vector[jl.Float64], X)
        return self._gt_single_point_from_jl(P, T, data, X_jl, Xoxides_jl, oxide_names, sys_in, rm_list)

    def _gt_single_point_from_jl(self, P, T, data, X_jl, Xoxides_jl, oxide_names, sys_in, rm_list=None):
        """Internal helper: single-point garnet elements from pre-converted Julia vectors."""
        out = MAGEMin_C.single_point_minimization(P, T, data, X=X_jl, Xoxides=Xoxides_jl, sys_in=sys_in, rm_list=rm_list)
        sys.stdout.flush()

        gt_frac = phase_frac(phase="g", MAGEMinOutput=out, sys_in='mol')
        gt_wt = phase_frac(phase="g", MAGEMinOutput=out, sys_in='wt')
        gt_vol = phase_frac(phase="g", MAGEMinOutput=out, sys_in='vol')

        Mg, Mn, Fe, Ca = self._extract_garnet_elements_from_oxides(out, oxide_names, sys_in)

        return gt_frac, gt_wt, gt_vol, Mg, Mn, Fe, Ca, out

    def gt_along_path(self, P, T, data, X, Xoxides, sys_in, fractionate=False, rm_list=None):
        """Calculate garnet fractions and element chemistry along a P-T path.

        Parameters
        ----------
        fractionate : bool, default=False
            If ``True``, removes newly formed garnet incrementally from bulk
            composition between steps.

        Returns
        -------
        tuple of np.ndarray
            ``(gt_mol_frac, gt_wt_frac, gt_vol_frac, Mg, Mn, Fe, Ca, X_along_path)``.

        Notes
        -----
        When ``fractionate=True``, bulk-rock updates are applied only for *new net*
        garnet growth, i.e. when ``gt_frac`` exceeds its historical maximum along
        the path. This avoids no-op/oscillatory fractionation updates from local
        fluctuations and keeps ``X_along_path`` stable after peak growth.
        """
        oxide_names = list(Xoxides)
        Xoxides_jl = jlconvert(jl.Vector[jl.String], Xoxides)
        n_points = len(P)

        gt_wt_frac = np.zeros(n_points)
        gt_mol_frac = np.zeros(n_points)
        gt_vol_frac = np.zeros(n_points)

        Mgi = np.zeros(n_points)
        Mni = np.zeros(n_points)
        Fei = np.zeros(n_points)
        Cai = np.zeros(n_points)

        X_along_path = np.zeros(shape=(n_points, len(X)) )

        gt_frac_max_previous = 0.
        phase_functions = PhaseFunctions() if fractionate else None
        for i, (P_step, T_step) in enumerate(zip(P, T)):
            X_jl = jlconvert(jl.Vector[jl.Float64], X)
            gt_frac, gt_wt, gt_vol, Mg, Mn, Fe, Ca, out = self._gt_single_point_from_jl(
                P_step, T_step, data, X_jl, Xoxides_jl, oxide_names, sys_in, rm_list
            )

            gt_mol_frac[i] = gt_frac
            gt_wt_frac[i] = gt_wt
            gt_vol_frac[i] = gt_vol

            if phase_functions is not None and i > 0:
                frac_amount = max(gt_frac - gt_frac_max_previous, 0.0)
                if frac_amount > 0:
                    X = phase_functions.fractionate_phase('g', out, sys_in, frac_amount=frac_amount)

            gt_frac_max_previous = max(gt_frac_max_previous, gt_frac)
            
            X_along_path[i] = X
            Mgi[i] = Mg
            Mni[i] = Mn
            Fei[i] = Fe
            Cai[i] = Ca

            


        return gt_mol_frac, gt_wt_frac, gt_vol_frac, Mgi, Mni, Fei, Cai, X_along_path


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
    
    def fractionate_phase(self, phase, out, sys_in, frac_amount=None):
        """
        Perform batch fractionation of a phase from the bulk rock composition.

        The bulk composition is adjusted by removing a specified fraction of the
        given phase composition, then renormalized. If ``frac_amount`` is not
        provided, the full phase fraction present in ``out`` is removed.

        Parameters
        ----------
        phase : str
            Name of the phase to fractionate (must be present in ``out.ph``).
        out :
            MAGEMin output object containing bulk and phase properties
            (e.g., ``bulk``, ``bulk_wt``, ``ph``, ``SS_vec``, ``ph_frac``, ``ph_frac_wt``).
        sys_in : str
            Composition basis: use ``"wt"`` for weight fraction (``bulk_wt``,
            ``Comp_wt``, ``ph_frac_wt``), anything else uses molar basis
            (``bulk``, ``Comp``, ``ph_frac``).
        frac_amount : float, optional
            Fraction of the phase to remove from the bulk. If None, the
            phase fraction from ``out`` is used (i.e., complete removal of
            that phase).

        Returns
        -------
        numpy.ndarray
            Updated, renormalized bulk composition array after phase
            fractionation, in the same basis specified by ``sys_in``.
        """

        if sys_in == "wt":
            current_X = out.bulk_wt
        else:
            current_X = out.bulk
        
        if phase in out.ph:
            phase_ind = out.ph.index(phase)
        
            if sys_in == "wt":
                ph_comp = np.array(out.SS_vec[phase_ind].Comp_wt)
                ph_frac = out.ph_frac_wt[phase_ind]
            else:
                ph_comp = np.array(out.SS_vec[phase_ind].Comp)
                ph_frac = out.ph_frac[phase_ind]

            if frac_amount is None:
                frac_amount = ph_frac
            
            numerator = current_X - (frac_amount * ph_comp)
            denominator = 1.0 - frac_amount
            
            # Update the effective bulk composition for the next step
            current_X = numerator / denominator
            
            current_X = current_X / np.sum(current_X)

        return current_X

