import numpy as np
import sys
from juliacall import Main as jl, convert as jlconvert
from ..core.base import MAGEMinBase
from ..core.phase_properties import phase_frac, extract_end_member, get_oxide_apfu, get_phase_chemistry, get_phase_mg_number, _phase_indices
from ..utils.bulk_rock import atomic_mass_dict, atomic_frac_to_wt_frac
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
        
        if P.ndim == 1 and T.ndim == 1 and P.shape[0] != T.shape[0]:
            P_orig, T_orig = P, T
            P, T = np.meshgrid(P_orig, T_orig)
            P = P.flatten()
            T = T.flatten()
        elif P.shape != T.shape:
            raise ValueError(f"P and T must have the same shape or both be 1D. Got {P.shape} and {T.shape}")

        P_jl = jlconvert(jl.Vector[jl.Float64], P.astype(float))
        T_jl = jlconvert(jl.Vector[jl.Float64], T.astype(float))

        out = MAGEMin_C.multi_point_minimization(
            P_jl, T_jl, self.data, X=self.X, Xoxides=self.Xoxides, 
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

    def get_phase_endmembers(self, phase, grid_out=None):
        """
        Discover the end-member names for a specific phase from the results.
        Scans the grid until the phase is found stable.
        """
        out = grid_out if grid_out is not None else self.last_grid_out
        if out is None:
            return []
            
        # If it's a single output object (from single_point_calc) — not a list/array/Vector
        if not isinstance(out, (list, np.ndarray)) and hasattr(out, 'ph'):
            if phase in out.ph:
                ph_index = out.ph.index(phase)
                if ph_index < len(out.SS_vec):
                    return [str(n) for n in out.SS_vec[ph_index].emNames]
            return []

        # If it's a grid (list, numpy array, or Julia Vector)
        for o in out:
            if phase in o.ph:
                ph_index = o.ph.index(phase)
                if ph_index < len(o.SS_vec):
                    return [str(n) for n in o.SS_vec[ph_index].emNames]
                return []
        return []

    def _extract_cations_from_apfu(self, out, phase, cations, sys_in, instance=0):
        """Internal: Extract cation ratios (e.g., XMg, XFe) for a specific phase."""
        ox_to_query = ['MgO', 'MnO', 'CaO', 'FeO', 'Fe2O3', 'Fe', 'O']
        apfu = get_oxide_apfu(out, phase, ox_to_query, instance=instance)

        mg = np.asarray(apfu.get("MgO", 0.0), dtype=float)
        mn = np.asarray(apfu.get("MnO", 0.0), dtype=float)
        ca = np.asarray(apfu.get("CaO", 0.0), dtype=float)

        feo = np.asarray(apfu.get("FeO", 0.0), dtype=float)
        fe2o3 = np.asarray(apfu.get("Fe2O3", 0.0), dtype=float)
        fe_metal = np.asarray(apfu.get("Fe", 0.0), dtype=float)
        atomic_o = np.asarray(apfu.get("O", 0.0), dtype=float)

        # Calculate total Fe as FeO equivalent (molar atoms)
        if np.any(atomic_o > 0):
            # MAGEMin O-basis (ig, mp) or sb24 basis
            if np.any(fe_metal > 0):
                fe = fe_metal
            else:
                fe = feo
        else:
            # Traditional FeO/Fe2O3 basis
            fe = feo + 2.0 * fe2o3

        total = mg + mn + fe + ca
        if instance != 'all':
            if total.ndim == 0 and total <= 0:
                return {f"cat_{c}": 0.0 for c in cations}
            total = np.maximum(total, 1e-12)
            vals = {"Mg": mg/total, "Mn": mn/total, "Fe": fe/total, "Ca": ca/total}
            if sys_in.casefold() == 'wt':
                vals = atomic_frac_to_wt_frac(vals, atomic_mass_dict)
            return {f"cat_{c}": float(vals.get(c, 0.0)) for c in cations}

        n = len(total)
        total_safe = np.where(total > 0, total, np.nan)
        vals = {"Mg": mg/total_safe, "Mn": mn/total_safe,
                "Fe": fe/total_safe, "Ca": ca/total_safe}
        if sys_in.casefold() == 'wt':
            vals = atomic_frac_to_wt_frac(vals, atomic_mass_dict)
        return {f"cat_{c}": np.asarray(vals.get(c, 0.0), dtype=float) for c in cations}

    def extract_from_grid(self, phase, end_members=None, oxides=None, chemistry=None, cations=None, mg_number=False, fe_split=False, grid_out=None, instance=0):
        """Extract phase properties from a previously calculated grid.

        Parameters
        ----------
        instance : int or {'all'}, default=0
            For a phase that appears multiple times (a solvus), an integer
            index selects that instance and ``'all'`` returns one column
            per instance, keyed with a ``_0``, ``_1`` ... suffix (missing
            instances at a given point are NaN).  ``mol_frac``/``wt_frac``/
            ``vol_frac`` are always the summed totals; with ``'all'``
            additional per-instance columns ``mol_frac_0``, ``mol_frac_1``
            ... are emitted.
        """
        out = grid_out if grid_out is not None else self.last_grid_out
        if out is None:
            raise ValueError("No grid results found. Run calculate_grid first or provide grid_out.")

        # Discovery of end-members if requested
        if end_members == 'auto':
            end_members = self.get_phase_endmembers(phase, out)

        P_len = len(out)
        results = {
            "mol_frac": np.zeros(P_len), "wt_frac": np.zeros(P_len), "vol_frac": np.zeros(P_len),
        }

        all_inst = instance == 'all'
        if all_inst:
            n_inst = 0
            for o in out:
                n_inst = max(n_inst, len(_phase_indices(o, phase, 'all')))
            # per-instance fraction columns alongside the summed totals
            for k in range(n_inst):
                results[f"mol_frac_{k}"] = np.zeros(P_len)
                results[f"wt_frac_{k}"] = np.zeros(P_len)
                results[f"vol_frac_{k}"] = np.zeros(P_len)
        else:
            n_inst = 1

        def _make_keys(prefix, names):
            if all_inst:
                return [f"{prefix}{n}_{k}" for n in names for k in range(n_inst)]
            return [f"{prefix}{n}" for n in names]

        if end_members:
            for key in _make_keys("em_", end_members): results[key] = np.zeros(P_len)
        if oxides:
            for key in _make_keys("ox_apfu_", oxides): results[key] = np.zeros(P_len)
        if chemistry:
            for key in _make_keys("chem_", chemistry): results[key] = np.zeros(P_len)
        if cations:
            for key in _make_keys("cat_", cations): results[key] = np.zeros(P_len)
        if mg_number:
            for key in _make_keys("Mg_number", [""]): results[key] = np.zeros(P_len)
        if fe_split:
            for key in _make_keys("Fe2", [""]): results[key] = np.zeros(P_len)
            for key in _make_keys("Fe3", [""]): results[key] = np.zeros(P_len)

        for i in range(P_len):
            if phase not in out[i].ph:
                continue
            results["mol_frac"][i] = phase_frac(phase, out[i], 'mol')
            results["wt_frac"][i]  = phase_frac(phase, out[i], 'wt')
            results["vol_frac"][i] = phase_frac(phase, out[i], 'vol')

            if all_inst:
                n_idx = _phase_indices(out[i], phase, 'all')
                fm = [float(out[i].ph_frac[j]) for j in n_idx]
                fw = [float(out[i].ph_frac_wt[j]) for j in n_idx]
                fv = [float(out[i].ph_frac_vol[j]) for j in n_idx]
                for k in range(n_inst):
                    results[f"mol_frac_{k}"][i] = fm[k] if k < len(fm) else np.nan
                    results[f"wt_frac_{k}"][i] = fw[k] if k < len(fw) else np.nan
                    results[f"vol_frac_{k}"][i] = fv[k] if k < len(fv) else np.nan

            def _store(key, value):
                if all_inst:
                    vals = np.asarray(value, dtype=float)
                    n = len(vals)
                    for k in range(n_inst):
                        results[f"{key}_{k}"][i] = vals[k] if k < n else np.nan
                else:
                    results[key][i] = value

            if end_members:
                for em in end_members:
                    _store(f"em_{em}", extract_end_member(phase, out[i], em, self.sys_in, instance=instance))
            if oxides:
                apfu = get_oxide_apfu(out[i], phase, oxides, instance=instance)
                for ox in oxides: _store(f"ox_apfu_{ox}", apfu.get(ox, 0.0))
            if chemistry:
                chem = get_phase_chemistry(out[i], phase, chemistry, self.sys_in, instance=instance)
                for ox in chemistry: _store(f"chem_{ox}", chem.get(ox, 0.0))
            if cations:
                cat_vals = self._extract_cations_from_apfu(out[i], phase, cations, self.sys_in, instance=instance)
                for c in cations: _store(f"cat_{c}", cat_vals[f"cat_{c}"])
            if mg_number:
                _store("Mg_number", get_phase_mg_number(out[i], phase, instance=instance))
            if fe_split:
                split = self._extract_fe_split_from_apfu(out[i], phase, instance=instance)
                _store("Fe2", split["Fe2"])
                _store("Fe3", split["Fe3"])

        return results

    def generate_2D_grid(self, P, T, phase, end_members=None, oxides=None, chemistry=None, cations=None, mg_number=False, fe_split=False, instance=0):
        """Convenience wrapper."""
        self.calculate_grid(P, T)
        return self.extract_from_grid(phase, end_members, oxides, chemistry, cations, mg_number, fe_split, instance=instance)

    def single_point_calc(self, P, T, phase, end_members=None, oxides=None, chemistry=None, cations=None, mg_number=False, fe_split=False, instance=0):
        """Single-point calculation."""
        out = MAGEMin_C.single_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
        sys.stdout.flush()
        results = {"mol_frac": 0.0, "wt_frac": 0.0, "vol_frac": 0.0, "present": False}

        if phase in out.ph:
            results["present"] = True
            results["mol_frac"] = phase_frac(phase, out, 'mol')
            results["wt_frac"]  = phase_frac(phase, out, 'wt')
            results["vol_frac"] = phase_frac(phase, out, 'vol')

            all_inst = instance == 'all'
            n_inst = len(_phase_indices(out, phase, 'all')) if all_inst else 1

            if all_inst:
                for k, j in enumerate(_phase_indices(out, phase, 'all')):
                    results[f"mol_frac_{k}"] = float(out.ph_frac[j])
                    results[f"wt_frac_{k}"] = float(out.ph_frac_wt[j])
                    results[f"vol_frac_{k}"] = float(out.ph_frac_vol[j])

            def _store(key, value):
                if all_inst:
                    vals = np.asarray(value, dtype=float)
                    n = len(vals)
                    for k in range(n_inst):
                        results[f"{key}_{k}"] = vals[k] if k < n else np.nan
                else:
                    results[key] = value

            if end_members:
                for em in end_members:
                    _store(f"em_{em}", extract_end_member(phase, out, em, self.sys_in, instance=instance))
            if oxides:
                apfu = get_oxide_apfu(out, phase, oxides, instance=instance)
                for ox in oxides: _store(f"ox_apfu_{ox}", apfu.get(ox, 0.0))
            if chemistry:
                chem = get_phase_chemistry(out, phase, chemistry, self.sys_in, instance=instance)
                for ox in chemistry: _store(f"chem_{ox}", chem.get(ox, 0.0))
            if cations:
                cat_vals = self._extract_cations_from_apfu(out, phase, cations, self.sys_in, instance=instance)
                for c in cations: _store(f"cat_{c}", cat_vals[f"cat_{c}"])
            if fe_split:
                split = self._extract_fe_split_from_apfu(out, phase, instance=instance)
                _store("Fe2", split["Fe2"])
                _store("Fe3", split["Fe3"])
        
        return results, out
