import numpy as np
import sys
from .pt_grid import MAGEMinPTGridCalculator
from ..core.phase_properties import phase_frac, extract_end_member, get_oxide_apfu
from ..utils.bulk_rock import atomic_mass_dict, atomic_frac_to_wt_frac
from phasetools import MAGEMin_C
from juliacall import Main as jl, convert as jlconvert

class MAGEMinGarnetCalculator(MAGEMinPTGridCalculator):
    """High-level wrappers for garnet-focused MAGEMin calculations."""

    def __init__(self, db="ig", dataset=636, verbose=False, fe_basis="FeOt"):
        """
        Parameters
        ----------
        db : str, default="ig"
            Thermodynamic database label.
        dataset : int, default=636
            Thermodynamic dataset version.
        verbose : bool, default=False
            Whether to print progress information.
        fe_basis : str, default="FeOt"
            Iron basis used for garnet X-site fractions:

            * ``'FeOt'`` -- all iron treated as Fe2+ (total iron,
              ``FeO + 2*Fe2O3`` APFU) placed on the divalent site.  This is
              the standard community convention for garnet end-members and
              X-site fractions (e.g. Williams & Grambling 1990; Krogh Ravna
              2000) and is consistent with the pyralspite garnet model used
              by MAGEMin and Holland-Powell-type databases.
            * ``'Fe2+'`` -- only the stoichiometrically estimated ferrous
              iron (``_extract_fe_split_from_apfu``) is placed on the
              divalent site, excluding Fe3+.  Use only when garnet Fe3+ is
              known to be significant (oxidised eclogites, skarns) or
              measured directly (XANES, Mössbauer).

            Case-insensitive.
        """
        super().__init__(db, dataset, verbose)
        basis = str(fe_basis).casefold()
        if basis not in ("feot", "fe2+", "fe2"):
            raise ValueError(f"fe_basis must be 'FeOt' or 'Fe2+', got {fe_basis!r}")
        self.fe_basis = "fe2+" if basis.startswith("fe2") else "feot"

    def _extract_garnet_elements_from_oxides(self, out, sys_in):
        """
        Extract garnet Mg-Mn-Fe-Ca cation fractions for the divalent (X) site.

        The Fe basis is controlled by ``self.fe_basis``:

        * ``'feot'`` (default) -- all iron treated as Fe2+ (total iron,
          ``FeO + 2*Fe2O3`` APFU) placed on the divalent site.  The standard
          community convention for garnet X-site fractions.
        * ``'fe2+'`` -- only the stoichiometrically estimated ferrous iron
          is placed on the divalent site, excluding Fe3+.
        """
        if 'g' not in out.ph:
            return 0.0, 0.0, 0.0, 0.0

        if self.fe_basis == 'feot':
            # Total iron on the divalent site (all Fe as Fe2+)
            elements = get_oxide_apfu(out, 'g', ['MgO', 'MnO', 'CaO', 'FeO', 'Fe2O3'])
            fe_moles = elements.get("FeO", 0.0) + 2.0 * elements.get("Fe2O3", 0.0)
        else:
            # Ferrous iron only, from the excess-oxygen Fe2+/Fe3+ split
            elements = get_oxide_apfu(out, 'g', ['MgO', 'MnO', 'CaO'])
            fe_moles = self._extract_fe_split_from_apfu(out, 'g')['Fe2']

        mg_moles = elements.get("MgO", 0.0)
        mn_moles = elements.get("MnO", 0.0)
        ca_moles = elements.get("CaO", 0.0)

        # Total atoms in the X-site (should be approx 3.0)
        total_x_site_moles = mg_moles + mn_moles + fe_moles + ca_moles
        if total_x_site_moles <= 0:
            return 0.0, 0.0, 0.0, 0.0

        # Normalise to 1.0 (mole fractions of the divalent site)
        Mg = mg_moles / total_x_site_moles
        Mn = mn_moles / total_x_site_moles
        Fe = fe_moles / total_x_site_moles
        Ca = ca_moles / total_x_site_moles

        if sys_in.casefold() == 'wt':
            wt_frac = atomic_frac_to_wt_frac(
                {"Mg": Mg, "Mn": Mn, "Fe": Fe, "Ca": Ca},
                atomic_mass_dict,
            )
            Mg, Mn, Fe, Ca = wt_frac["Mg"], wt_frac["Mn"], wt_frac["Fe"], wt_frac["Ca"]

        return Mg, Mn, Fe, Ca

    def generate_2D_grid_gt_endmembers(self, P, T):
        """Compute garnet endmember fractions over a P-T grid.

        Garnet is a single-instance phase, so its single per-instance
        bundle is returned directly, preserving the historical key names
        (``em_py``, ``em_alm``, ``mol_frac``, ...).  If garnet ever
        appeared as a solvus, the list of bundles would be returned
        instead.
        """
        self.calculate_grid(P, T)
        # Automatic discovery of end-members
        res = self.extract_from_grid("g", end_members='auto')
        return res[0] if len(res) == 1 else res

    def generate_2D_grid_gt_elements(self, P, T):
        """Compute garnet element fractions (Mg, Mn, Fe, Ca) over a P-T grid."""
        out = self.calculate_grid(P, T)

        gt_mol_frac = np.zeros_like(P)
        gt_wt_frac = np.zeros_like(P)
        gt_vol_frac = np.zeros_like(P)
        Mgi = np.zeros_like(P)
        Mni = np.zeros_like(P)
        Fei = np.zeros_like(P)
        Cai = np.zeros_like(P)
        
        for i in range(len(out)):
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
        emDict_mol = {}
        emDict_wt = {}
        
        if 'g' in out.ph:
            gt_frac  = phase_frac(phase="g", MAGEMinOutput=out, sys_in='mol')
            gt_wt    = phase_frac(phase="g", MAGEMinOutput=out, sys_in='wt')
            gt_vol   = phase_frac(phase="g", MAGEMinOutput=out, sys_in='vol')

            ph_index = out.ph.index('g')
            emNames = [str(n) for n in out.SS_vec[ph_index].emNames]
            emFrac = out.SS_vec[ph_index].emFrac
            emFrac_wt = out.SS_vec[ph_index].emFrac_wt

            emDict_mol = {name: float(frac) for name, frac in zip(emNames, emFrac)}
            emDict_wt  = {name: float(frac) for name, frac in zip(emNames, emFrac_wt)}

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
        """Calculate garnet fractions and element chemistry along a P-T path.

        Parameters
        ----------
        P : array-like
            Pressure values along the path (kbar).
        T : array-like
            Temperature values along the path (°C).
        fractionate : bool, default=False
            If True, fractionate garnet from the bulk composition as it grows.
        normalise_start : bool, default=True
            Controls how the first P-T point is treated:

            * ``True`` — The first P-T point is treated as a nucleation
              barrier: garnet volume starts at zero and only new growth is
              modelled.  The initial garnet fraction (if any) is used as a
              baseline for measuring incremental growth but is **not** removed
              from the reactive bulk.  Use when the path starts outside the
              garnet stability field or at the nucleation threshold.

            * ``False`` — The first P-T point has an initial garnet volume
              (overstepped nucleation).  That fraction is removed from the
              bulk at step 0, and subsequent growth is measured relative to
              it.  Use when the path starts well inside the garnet stability
              field.

        Returns
        -------
        gt_mol_frac : numpy.ndarray
            Garnet molar fraction at each P-T point.
        gt_wt_frac : numpy.ndarray
            Garnet weight fraction at each P-T point.
        gt_vol_frac : numpy.ndarray
            Garnet volume fraction at each P-T point.
        Mgi : numpy.ndarray
            Garnet X-site Mg fraction at each P-T point.
        Mni : numpy.ndarray
            Garnet X-site Mn fraction at each P-T point.
        Fei : numpy.ndarray
            Garnet X-site Fe fraction at each P-T point.  The Fe basis
            (total Fe as Fe2+ by default, or ferrous-only) is set by the
            ``fe_basis`` argument passed to :meth:`__init__`.
        Cai : numpy.ndarray
            Garnet X-site Ca fraction at each P-T point.
        X_along_path : numpy.ndarray
            Bulk composition after each step's fractionation.  Each row is
            normalised to sum to 1.  Row ``i`` is the bulk **after** step
            ``i``'s fractionation has been applied.

        Notes
        -----
        Fractionation uses the **current-step** garnet composition (not the
        growth-increment composition) — a first-order approximation valid for
        small P-T steps.
        """
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
        gt_wt_max_previous = 0.
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

            # Select the fraction basis consistent with sys_in
            if self.sys_in.casefold() == 'wt':
                gt_frac_for_fractionation = gt_wt
                gt_frac_max_prev_for_fractionation = gt_wt_max_previous
            else:
                gt_frac_for_fractionation = gt_frac
                gt_frac_max_prev_for_fractionation = gt_frac_max_previous

            if phase_functions is not None:
                if i == 0 and not normalise_start:
                    if gt_frac_for_fractionation > 0:
                        X_py = phase_functions.fractionate_phase('g', out, self.sys_in, frac_amount=gt_frac_for_fractionation)
                        X = jlconvert(jl.Vector[jl.Float64], X_py)
                elif i > 0:
                    frac_amount = max(gt_frac_for_fractionation - gt_frac_max_prev_for_fractionation, 0.0)
                    if frac_amount > 0:
                        X_py = phase_functions.fractionate_phase('g', out, self.sys_in, frac_amount=frac_amount)
                        X = jlconvert(jl.Vector[jl.Float64], X_py)
            
            X_along_path[i] = np.array(X) / np.sum(X)
            gt_frac_max_previous = max(gt_frac_max_previous, gt_frac)
            gt_wt_max_previous = max(gt_wt_max_previous, gt_wt)
            
            Mgi[i] = Mg
            Mni[i] = Mn
            Fei[i] = Fe
            Cai[i] = Ca

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, Mgi, Mni, Fei, Cai, X_along_path
