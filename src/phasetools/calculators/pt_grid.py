import numpy as np
import sys
from ..core.base import MAGEMinBase
from ..core.phase_properties import phase_frac, extract_end_member, get_oxide_apfu, get_phase_chemistry
from ..utils.bulk_rock import atomic_mass_dict, convert_mol_percent_to_wt_percent
from phasetools import MAGEMin_C

class MAGEMinPTGridCalculator(MAGEMinBase):
    """
    Calculator for performing P-T grid minimisations and extracting phase properties.
    """
    def __init__(self, db="ig", dataset=636, verbose=False):
        super().__init__(db, dataset, verbose)
        self.last_grid_out = None
        self.last_P = None
        self.last_T = None

    def calculate_grid(self, P, T):
        """Perform multi-point minimisation over a P-T grid and store the results."""
        P = np.atleast_1d(P)
        T = np.atleast_1d(T)
        
        if P.shape != T.shape:
            if P.ndim == 1 and T.ndim == 1:
                P_orig, T_orig = P, T
                P, T = np.meshgrid(P_orig, T_orig)
                P = P.flatten()
                T = T.flatten()
            else:
                raise ValueError(f"P and T must have the same shape or both be 1D. Got {P.shape} and {T.shape}")

        out = MAGEMin_C.multi_point_minimization(
            P, T, self.data, X=self.X, Xoxides=self.Xoxides, 
            sys_in=self.sys_in, rm_list=self.rm_list
        )
        sys.stdout.flush()
        
        self.last_grid_out = out
        self.last_P = P
        self.last_T = T
        return out

    def get_stable_phases(self, grid_out=None):
        """
        Return the list of stable phases at each P-T point in the grid.
        
        Returns:
            list[list[str]]: A list of phase name lists for each point.
        """
        out = grid_out if grid_out is not None else self.last_grid_out
        if out is None:
            raise ValueError("No grid results found. Run calculate_grid first.")
        return [list(o.ph) for o in out]

    def get_all_unique_phases(self, grid_out=None):
        """
        Return a unique set of all phases present anywhere in the grid.
        
        Returns:
            list[str]: Sorted list of unique phase names.
        """
        out = grid_out if grid_out is not None else self.last_grid_out
        if out is None:
            raise ValueError("No grid results found. Run calculate_grid first.")
        
        unique_phases = set()
        for o in out:
            unique_phases.update(o.ph)
        return sorted(list(unique_phases))

    def _extract_cations_from_apfu(self, out, phase, cations, sys_in):
        """Internal: Extract cation ratios (e.g., XMg, XFe) for a specific phase."""
        ox_to_query = ['MgO', 'MnO', 'CaO', 'FeO', 'Fe2O3']
        apfu = get_oxide_apfu(out, phase, ox_to_query)
        
        mg = apfu.get("MgO", 0.0)
        mn = apfu.get("MnO", 0.0)
        ca = apfu.get("CaO", 0.0)
        fe = apfu.get("FeO", 0.0) + 2*apfu.get("Fe2O3", 0.0)

        total = mg + mn + fe + ca
        if total <= 0:
            return {c: 0.0 for c in cations}

        vals = {"Mg": mg/total, "Mn": mn/total, "Fe": fe/total, "Ca": ca/total}
        
        if sys_in.casefold() == 'wt':
            keys = list(vals.keys())
            raw_vals = [vals[k] for k in keys]
            wt_percents = convert_mol_percent_to_wt_percent(raw_vals, keys, atomic_mass_dict)
            vals = {k: v/100.0 for k, v in zip(keys, wt_percents)}

        return {f"cat_{c}": vals.get(c, 0.0) for c in cations}

    def extract_from_grid(self, phase, end_members=None, oxides=None, chemistry=None, cations=None, grid_out=None):
        """Extract phase properties from a previously calculated grid."""
        out = grid_out if grid_out is not None else self.last_grid_out
        if out is None:
            raise ValueError("No grid results found. Run calculate_grid first or provide grid_out.")

        P_len = len(out)
        results = {
            "mol_frac": np.zeros(P_len), "wt_frac": np.zeros(P_len), "vol_frac": np.zeros(P_len),
        }
        
        if end_members:
            for em in end_members: results[f"em_{em}"] = np.zeros(P_len)
        if oxides:
            for ox in oxides: results[f"ox_apfu_{ox}"] = np.zeros(P_len)
        if chemistry:
            for ox in chemistry: results[f"chem_{ox}"] = np.zeros(P_len)
        if cations:
            for c in cations: results[f"cat_{c}"] = np.zeros(P_len)

        for i in range(P_len):
            if phase in out[i].ph:
                results["mol_frac"][i] = phase_frac(phase, out[i], 'mol')
                results["wt_frac"][i]  = phase_frac(phase, out[i], 'wt')
                results["vol_frac"][i] = phase_frac(phase, out[i], 'vol')

                if end_members:
                    for em in end_members:
                        results[f"em_{em}"][i] = extract_end_member(phase, out[i], em, self.sys_in)
                if oxides:
                    apfu = get_oxide_apfu(out[i], phase, oxides)
                    for ox in oxides: results[f"ox_apfu_{ox}"][i] = apfu.get(ox, 0.0)
                if chemistry:
                    chem = get_phase_chemistry(out[i], phase, chemistry, self.sys_in)
                    for ox in chemistry: results[f"chem_{ox}"][i] = chem.get(ox, 0.0)
                if cations:
                    cat_vals = self._extract_cations_from_apfu(out[i], phase, cations, self.sys_in)
                    for c in cations: results[f"cat_{c}"][i] = cat_vals[f"cat_{c}"]

        return results

    def generate_2D_grid(self, P, T, phase, end_members=None, oxides=None, chemistry=None, cations=None):
        """Convenience wrapper."""
        self.calculate_grid(P, T)
        return self.extract_from_grid(phase, end_members, oxides, chemistry, cations)

    def single_point_calc(self, P, T, phase, end_members=None, oxides=None, chemistry=None, cations=None):
        """Single-point calculation."""
        out = MAGEMin_C.single_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
        sys.stdout.flush()
        results = {"mol_frac": 0.0, "wt_frac": 0.0, "vol_frac": 0.0, "present": False}

        if phase in out.ph:
            results["present"] = True
            results["mol_frac"] = phase_frac(phase, out, 'mol')
            results["wt_frac"]  = phase_frac(phase, out, 'wt')
            results["vol_frac"] = phase_frac(phase, out, 'vol')

            if end_members:
                for em in end_members: results[f"em_{em}"] = extract_end_member(phase, out, em, self.sys_in)
            if oxides:
                apfu = get_oxide_apfu(out, phase, oxides)
                for ox in oxides: results[f"ox_apfu_{ox}"] = apfu.get(ox, 0.0)
            if chemistry:
                chem = get_phase_chemistry(out, phase, chemistry, self.sys_in)
                for ox in chemistry: results[f"chem_{ox}"] = chem.get(ox, 0.0)
            if cations:
                cat_vals = self._extract_cations_from_apfu(out, phase, cations, self.sys_in)
                for c in cations: results[f"cat_{c}"] = cat_vals[f"cat_{c}"]
        
        return results, out
