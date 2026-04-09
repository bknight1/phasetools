import numpy as np
import sys
from ..core.base import MAGEMinBase
from ..core.phase_properties import phase_frac, extract_end_member, get_oxide_apfu
from ..utils.bulk_rock import atomic_mass_dict, convert_mol_percent_to_wt_percent
from phasetools import MAGEMin_C
from juliacall import Main as jl, convert as jlconvert

class MAGEMinGarnetCalculator(MAGEMinBase):
    """High-level wrappers for garnet-focused MAGEMin calculations."""
    def __init__(self, db="ig", dataset=636, verbose=False):
        super().__init__(db, dataset, verbose)

    def _extract_garnet_elements_from_oxides(self, out, sys_in):
        """Extract garnet Mg-Mn-Fe-Ca cation fractions from MAGEMin garnet output."""
        if 'g' not in out.ph:
            return 0.0, 0.0, 0.0, 0.0

        elements = get_oxide_apfu(out, 'g', ['MgO', 'MnO', 'CaO', 'FeO', 'Fe2O3'])
        mg_moles = elements.get("MgO", 0.0)
        mn_moles = elements.get("MnO", 0.0)
        ca_moles = elements.get("CaO", 0.0)
        fe_moles = elements.get("FeO", 0.0) + 2*elements.get("Fe2O3", 0.0)

        total_cation_moles = mg_moles + mn_moles + fe_moles + ca_moles
        if total_cation_moles <= 0:
            return 0.0, 0.0, 0.0, 0.0

        Mg = mg_moles / total_cation_moles
        Mn = mn_moles / total_cation_moles
        Fe = fe_moles / total_cation_moles
        Ca = ca_moles / total_cation_moles

        if sys_in.casefold() == 'wt':
            wt_percent_list = convert_mol_percent_to_wt_percent(
                [Mg, Mn, Fe, Ca],
                ["Mg", "Mn", "Fe", "Ca"],
                atomic_mass_dict,
            )
            Mg, Mn, Fe, Ca = [val / 100.0 for val in wt_percent_list]

        return Mg, Mn, Fe, Ca

    def generate_2D_grid_gt_endmembers(self, P, T):
        """Compute garnet endmember fractions over a P-T grid."""
        out = MAGEMin_C.multi_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
        sys.stdout.flush()

        gt_mol_frac = np.zeros_like(P)
        gt_wt_frac = np.zeros_like(P)
        gt_vol_frac = np.zeros_like(P)
        py_arr = np.zeros_like(P)
        alm_arr = np.zeros_like(P)
        spss_arr = np.zeros_like(P)
        gr_arr = np.zeros_like(P)
        kho_arr = np.zeros_like(P)

        for i in range(len(T)):
            gt_mol_frac[i] = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='mol')
            gt_wt_frac[i]  = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='wt')
            gt_vol_frac[i] = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='vol')

            py_arr[i]   = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="py", sys_in=self.sys_in)
            alm_arr[i]  = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="alm", sys_in=self.sys_in)
            spss_arr[i] = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="spss", sys_in=self.sys_in)
            gr_arr[i]   = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="gr", sys_in=self.sys_in)
            kho_arr[i]  = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="kho", sys_in=self.sys_in)

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, py_arr, alm_arr, spss_arr, gr_arr, kho_arr

    def generate_2D_grid_gt_elements(self, P, T):
        """Compute garnet element fractions (Mg, Mn, Fe, Ca) over a P-T grid."""
        out = MAGEMin_C.multi_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
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

            Mgi[i], Mni[i], Fei[i], Cai[i] = self._extract_garnet_elements_from_oxides(out[i], self.sys_in)

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, Mgi, Mni, Fei, Cai

    def gt_single_point_calc_endmembers(self, P, T):
        """Calculate single-point garnet endmember fractions."""
        out = MAGEMin_C.single_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
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

    def gt_single_point_calc_elements(self, P, T):
        """Calculate single-point garnet element fractions (Mg, Mn, Fe, Ca)."""
        return self._gt_single_point_from_jl(P, T, self.X, self.Xoxides, self.sys_in, self.rm_list)

    def _gt_single_point_from_jl(self, P, T, X_jl, Xoxides_jl, sys_in, rm_list=None):
        """Internal helper: single-point garnet elements from pre-converted Julia vectors."""
        out = MAGEMin_C.single_point_minimization(P, T, self.data, X=X_jl, Xoxides=Xoxides_jl, sys_in=sys_in, rm_list=rm_list)
        sys.stdout.flush() 

        gt_frac = phase_frac(phase="g", MAGEMinOutput=out, sys_in='mol')
        gt_wt = phase_frac(phase="g", MAGEMinOutput=out, sys_in='wt')
        gt_vol = phase_frac(phase="g", MAGEMinOutput=out, sys_in='vol')
        
        Mg, Mn, Fe, Ca = self._extract_garnet_elements_from_oxides(out, sys_in)

        return gt_frac, gt_wt, gt_vol, Mg, Mn, Fe, Ca, out

    def gt_along_path(self, P, T, fractionate=False, normalise_start=True):
        """Calculate garnet fractions and element chemistry along a P-T path."""
        from .phase_search import PhaseFunctions

        X = self.X
        Xoxides = self.Xoxides
        
        n_points = len(P)
        gt_wt_frac = np.zeros(n_points)
        gt_mol_frac = np.zeros(n_points)
        gt_vol_frac = np.zeros(n_points)
        Mgi = np.zeros(n_points)
        Mni = np.zeros(n_points)
        Fei = np.zeros(n_points)
        Cai = np.zeros(n_points)
        X_along_path = np.zeros(shape=(n_points, len(self._Xoxides_py)) )

        gt_frac_max_previous = 0.
        phase_functions = PhaseFunctions(db=self.db, dataset=self.dataset, verbose=self.verbose) if fractionate else None
        if phase_functions:
            # Sync standardised state to the helper instance
            self._copy_state_to(phase_functions)

        for i, (P_step, T_step) in enumerate(zip(P, T)):
            gt_frac, gt_wt, gt_vol, Mg, Mn, Fe, Ca, out = self._gt_single_point_from_jl(
                P_step, T_step, X, Xoxides, self.sys_in, self.rm_list
            )

            gt_mol_frac[i] = gt_frac
            gt_wt_frac[i] = gt_wt
            gt_vol_frac[i] = gt_vol

            if phase_functions is not None:
                if i == 0 and not normalise_start:
                    if gt_frac > 0:
                        X_py = phase_functions.fractionate_phase('g', out, self.sys_in, frac_amount=gt_frac)
                        X = jlconvert(jl.Vector[jl.Float64], X_py)
                elif i > 0:
                    frac_amount = max(gt_frac - gt_frac_max_previous, 0.0)
                    if frac_amount > 0:
                        X_py = phase_functions.fractionate_phase('g', out, self.sys_in, frac_amount=frac_amount)
                        X = jlconvert(jl.Vector[jl.Float64], X_py)
            
            X_along_path[i] = np.array(X)
            gt_frac_max_previous = max(gt_frac_max_previous, gt_frac)
            
            Mgi[i] = Mg
            Mni[i] = Mn
            Fei[i] = Fe
            Cai[i] = Ca

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, Mgi, Mni, Fei, Cai, X_along_path
