import warnings

import numpy as np
from scipy import optimize

from phasetools import MAGEMin_C

from ..core.base import MAGEMinBase
from ..core.phase_properties import phase_frac


class PhaseFunctions(MAGEMinBase):
    """Utility functions for phase stability and fractionation."""
    def __init__(self, db="ig", dataset=636, verbose=False):
        super().__init__(db, dataset, verbose)

    def find_phase_in(self, P, bracket, phase, tol=1e-2, verbose=False):
        """Finds the P-T condition where a phase first becomes stable."""
        def solidus_func(T):
            out = MAGEMin_C.single_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
            phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_in=self.sys_in)
            if verbose:
                print(f"phase frac: {phasefrac:.4f}, T: {T:.2f}")
            return phasefrac - 1e-5

        T_low, T_high = bracket
        result = optimize.root_scalar(solidus_func, bracket=[T_low, T_high], method='bisect', xtol=tol)
        if not result.converged:
            raise RuntimeError("Phase search did not converge.")
        return result.root

    def find_phase_saturation(self, P, bracket, phase, tol=1e-2, verbose=False):
        """Finds the P-T condition where a phase reaches unity fraction."""
        def liquidus_func(T):
            out = MAGEMin_C.single_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
            phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_in=self.sys_in)
            if verbose:
                print(f"phase frac: {phasefrac:.4f}, T: {T:.2f}")
            return phasefrac - (1.0 - 1e-5)

        T_low, T_high = bracket
        result = optimize.root_scalar(liquidus_func, bracket=[T_low, T_high], method='bisect', xtol=tol)
        if not result.converged:
            raise RuntimeError("Phase search did not converge.")
        return result.root
    
    def fractionate_phase(self, phase, out, sys_in, frac_amount=None):
        """Perform batch fractionation of a phase from the bulk rock composition.

        Removes a specified fraction of a phase from the current bulk composition
        and renormalises the remainder. The output is always normalised to sum to 1
        (molar or weight fractions, depending on ``sys_in``).

        MAGEMin orders ``out.ph`` with solution phases first (``SS_vec``), then
        pure phases (``PP_vec``).  This method handles both indexing ranges
        transparently.

        .. note::

            When a phase appears multiple times in ``out.ph`` (e.g. two
            coexisting pyroxenes on a solvus), only the **first** instance is
            removed.  Use ``phase_frac()`` to obtain the summed fraction before
            calling this method if the total solvus fraction is needed.

        Parameters
        ----------
        phase : str
            Phase name as it appears in ``out.ph`` (e.g. ``'g'``, ``'liq'``).
        out : MAGEMinOutput
            Raw output from a MAGEMin single-point minimisation.
        sys_in : str
            ``'mol'`` or ``'wt'`` — determines which bulk composition and
            phase composition vectors are used.
        frac_amount : float or None, optional
            Fraction of the phase to remove.  If ``None``, the full phase
            fraction from the minimisation result is used.  Values of 0 or
            less are treated as a no-op and return the current bulk unchanged.

        Returns
        -------
        numpy.ndarray
            Renormalised bulk composition after fractionation.
        """
        if sys_in.casefold() == "wt":
            current_X = out.bulk_wt
        else:
            current_X = out.bulk

        if phase in out.ph:
            phase_ind = out.ph.index(phase)
            # MAGEMin orders: solution phases (SS_vec) first, then pure phases (PP_vec)
            if phase_ind < out.n_SS:
                if sys_in.casefold() == "wt":
                    ph_comp = np.array(out.SS_vec[phase_ind].Comp_wt)
                    ph_frac = out.ph_frac_wt[phase_ind]
                else:
                    ph_comp = np.array(out.SS_vec[phase_ind].Comp)
                    ph_frac = out.ph_frac[phase_ind]
            else:
                pp_ind = phase_ind - out.n_SS
                if sys_in.casefold() == "wt":
                    ph_comp = np.array(out.PP_vec[pp_ind].Comp_wt)
                    ph_frac = out.ph_frac_wt[phase_ind]
                else:
                    ph_comp = np.array(out.PP_vec[pp_ind].Comp)
                    ph_frac = out.ph_frac[phase_ind]

            if frac_amount is None:
                frac_amount = ph_frac

            if frac_amount <= 0:
                return np.array(current_X, dtype=float)

            if frac_amount >= 1.0:
                warnings.warn(f"fractionate_phase: requested frac_amount={frac_amount} >= 1.0; skipping.")
                return np.array(current_X, dtype=float)

            numerator = current_X - (frac_amount * ph_comp)
            denominator = 1.0 - frac_amount
            current_X = numerator / denominator
            current_X = current_X / np.sum(current_X)

        return current_X
