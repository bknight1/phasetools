import numpy as np
import sys
from juliacall import Main as jl, convert as jlconvert
from ..core.base import MAGEMinBase
from ..core.phase_properties import extract_end_member, get_oxide_apfu, get_phase_chemistry, get_phase_mg_number, _phase_indices
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

    def _extract_cations_from_apfu(self, out, phase, cations, sys_in):
        """Internal: per-instance cation ratios (e.g., XMg, XFe).

        Returns ``cat_<c>`` arrays, one entry per instance of the phase.
        """
        ox_to_query = ['MgO', 'MnO', 'CaO', 'FeO', 'Fe2O3', 'Fe', 'O']
        apfu = get_oxide_apfu(out, phase, ox_to_query, instance='all')

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
        total_safe = np.where(total > 0, total, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = {"Mg": mg/total_safe, "Mn": mn/total_safe,
                    "Fe": fe/total_safe, "Ca": ca/total_safe}

        if sys_in.casefold() == 'wt':
            vals = atomic_frac_to_wt_frac(vals, atomic_mass_dict)

        return {f"cat_{c}": np.asarray(vals[c], dtype=float) for c in cations}

    def extract_from_grid(self, phase, end_members=None, oxides=None, chemistry=None, cations=None, mg_number=False, fe_split=False, grid_out=None):
        """Extract phase properties from a previously calculated grid.

        MAGEMin reports coexisting solvus limbs as repeated entries in
        ``out.ph`` (e.g. two clinopyroxenes ``dio``).  Returns a list
        with one bundle per instance, indexed ``res[0]``, ``res[1]``
        ...:

        ``res[0]`` -- first (or only) instance; every key is an array
        over the grid points (scalars for ``single_point_calc``).

        The key shape is identical for every phase -- a phase with a
        single instance simply has a one-element list, always at
        ``res[0]``.  Sum the per-instance ``mol_frac`` arrays yourself
        if you want the total.

        Each bundle key is NaN where the phase (or that instance) is
        absent at a grid point.  If the phase never appears in the grid,
        the returned list is empty.

        Notes
        -----
        - Bundle ``res[k]`` is the k-th occurrence of the phase in that
          point's ``out.ph``; solvus branch ordering may swap between
          grid points, so track both limbs when plotting isopleths.
        - ``end_members='auto'`` discovers end-members from the first
          occurrence of the phase; solvus limbs normally share the same
          solution model and endmember set.
        """
        out = grid_out if grid_out is not None else self.last_grid_out
        if out is None:
            raise ValueError("No grid results found. Run calculate_grid first or provide grid_out.")

        # Discovery of end-members if requested
        if end_members == 'auto':
            end_members = self.get_phase_endmembers(phase, out)

        P_len = len(out)

        # n_inst = max occurrences of the phase across the whole grid
        n_inst = 0
        for o in out:
            n_inst = max(n_inst, len(_phase_indices(o, phase, 'all')))

        instances = [{} for _ in range(n_inst)]

        def _precreate(prefix, names):
            for k in range(n_inst):
                for n in names:
                    instances[k][f"{prefix}{n}"] = np.full(P_len, np.nan)

        for k in range(n_inst):
            instances[k]["mol_frac"] = np.full(P_len, np.nan)
            instances[k]["wt_frac"] = np.full(P_len, np.nan)
            instances[k]["vol_frac"] = np.full(P_len, np.nan)
        if end_members:
            _precreate("em_", end_members)
        if oxides:
            _precreate("ox_apfu_", oxides)
        if chemistry:
            _precreate("chem_", chemistry)
        if cations:
            _precreate("cat_", cations)
        if mg_number:
            for k in range(n_inst):
                instances[k]["Mg_number"] = np.full(P_len, np.nan)
        if fe_split:
            for k in range(n_inst):
                instances[k]["Fe2"] = np.full(P_len, np.nan)
                instances[k]["Fe3"] = np.full(P_len, np.nan)

        for i in range(P_len):
            if phase not in out[i].ph:
                continue
            n_idx = _phase_indices(out[i], phase, 'all')
            for k, j in enumerate(n_idx):
                instances[k]["mol_frac"][i] = float(out[i].ph_frac[j])
                instances[k]["wt_frac"][i] = float(out[i].ph_frac_wt[j])
                instances[k]["vol_frac"][i] = float(out[i].ph_frac_vol[j])
            # instances k >= len(n_idx) stay NaN

            def _fill(key, value):
                vals = np.atleast_1d(np.asarray(value, dtype=float))
                n = len(vals)
                for k in range(n_inst):
                    if k < n:
                        instances[k][key][i] = vals[k]
                    # else stays NaN

            if end_members:
                for em in end_members:
                    _fill(f"em_{em}", extract_end_member(phase, out[i], em, self.sys_in, instance='all'))
            if oxides:
                apfu = get_oxide_apfu(out[i], phase, oxides, instance='all')
                for ox in oxides: _fill(f"ox_apfu_{ox}", apfu.get(ox, np.zeros(0)))
            if chemistry:
                chem = get_phase_chemistry(out[i], phase, chemistry, self.sys_in, instance='all')
                for ox in chemistry: _fill(f"chem_{ox}", chem.get(ox, np.zeros(0)))
            if cations:
                cat_vals = self._extract_cations_from_apfu(out[i], phase, cations, self.sys_in)
                for c in cations: _fill(f"cat_{c}", cat_vals[f"cat_{c}"])
            if mg_number:
                _fill("Mg_number", get_phase_mg_number(out[i], phase, instance='all'))
            if fe_split:
                split = self._extract_fe_split_from_apfu(out[i], phase, instance='all')
                _fill("Fe2", split["Fe2"])
                _fill("Fe3", split["Fe3"])

        return instances

    def generate_2D_grid(self, P, T, phase, end_members=None, oxides=None, chemistry=None, cations=None, mg_number=False, fe_split=False):
        """Convenience wrapper."""
        self.calculate_grid(P, T)
        return self.extract_from_grid(phase, end_members, oxides, chemistry, cations, mg_number, fe_split)

    def single_point_calc(self, P, T, phase, end_members=None, oxides=None, chemistry=None, cations=None, mg_number=False, fe_split=False):
        """Single-point calculation.

        Returns ``(bundles, out)`` where ``bundles`` is a list with one
        bundle per instance of the phase (``bundles[0]``, ``bundles[1]``,
        ...), keyed like the grid-level ``extract_from_grid`` with scalar
        values.  The list is empty when the phase is not present at this
        P-T.
        """
        out = MAGEMin_C.single_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
        sys.stdout.flush()
        instances = []

        if phase in out.ph:
            n_idx = _phase_indices(out, phase, 'all')
            instances = [{"mol_frac": 0.0, "wt_frac": 0.0, "vol_frac": 0.0}
                         for _ in n_idx]
            for k, j in enumerate(n_idx):
                instances[k]["mol_frac"] = float(out.ph_frac[j])
                instances[k]["wt_frac"] = float(out.ph_frac_wt[j])
                instances[k]["vol_frac"] = float(out.ph_frac_vol[j])

            def _store(key, value):
                vals = np.atleast_1d(np.asarray(value, dtype=float))
                for k in range(len(vals)):
                    instances[k][key] = vals[k]

            if end_members:
                for em in end_members:
                    _store(f"em_{em}", extract_end_member(phase, out, em, self.sys_in, instance='all'))
            if oxides:
                apfu = get_oxide_apfu(out, phase, oxides, instance='all')
                for ox in oxides: _store(f"ox_apfu_{ox}", apfu.get(ox, np.zeros(0)))
            if chemistry:
                chem = get_phase_chemistry(out, phase, chemistry, self.sys_in, instance='all')
                for ox in chemistry: _store(f"chem_{ox}", chem.get(ox, np.zeros(0)))
            if cations:
                cat_vals = self._extract_cations_from_apfu(out, phase, cations, self.sys_in)
                for c in cations: _store(f"cat_{c}", cat_vals[f"cat_{c}"])
            if fe_split:
                split = self._extract_fe_split_from_apfu(out, phase, instance='all')
                _store("Fe2", split["Fe2"])
                _store("Fe3", split["Fe3"])

        return instances, out
