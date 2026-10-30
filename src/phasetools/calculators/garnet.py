import sys

import numpy as np
from juliacall import Main as jl
from juliacall import convert as jlconvert

from phasetools import MAGEMin_C

from ..core.phase_properties import phase_frac
from ..utils.bulk_rock import atomic_frac_to_wt_frac, atomic_mass_dict
from .garnet_composition import garnet_cations_from_apfu, normalise_fe_basis
from .pt_grid import MAGEMinPTGridCalculator


class MAGEMinGarnetCalculator(MAGEMinPTGridCalculator):
    """High-level wrappers for garnet-focused MAGEMin calculations."""

    def __init__(
        self, db: str = "ig", dataset: int = 636, verbose: bool = False, fe_basis: str = "FeOt"
    ) -> None:
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
            Iron basis used for closed Mg-Mn-Fe-Ca garnet cation proportions:

            * ``'FeOt'`` -- all iron treated as Fe2+ (total iron,
              ``FeO + 2*Fe2O3`` APFU) included in the closed cation total.
              This is the standard community convention for garnet
              end-members (e.g. Williams & Grambling 1990; Krogh Ravna 2000).
              When ferric kho is present, its two Fe atoms are included even
              though they occupy the Y site, so the result is not strictly an
              X-site normalisation.
            * ``'Fe2+'`` -- only the stoichiometrically estimated ferrous
              iron (``_extract_fe_split_from_apfu``) is placed on the
              divalent site, excluding Fe3+.  Use only when garnet Fe3+ is
              known to be significant (oxidised eclogites, skarns) or
              measured directly (XANES, Mössbauer).
            * ``'Fe2'`` or ``'fe2'`` -- alias for ``'Fe2+'``.

            Case-insensitive.
        """
        super().__init__(db, dataset, verbose)
        self.fe_basis = normalise_fe_basis(fe_basis).casefold()

    def _extract_garnet_elements_from_oxides(self, out, sys_in):
        """
        Extract closed garnet Mg-Mn-Fe-Ca cation proportions.

        The Fe basis is controlled by ``self.fe_basis``:

        * ``'feot'`` (default) -- all iron treated as Fe2+ (total iron,
          ``FeO + 2*Fe2O3`` APFU) in the closed cation total. This follows
          the community convention for garnet end-members. If ferric kho is
          present, its two Y-site Fe atoms are included, so the result is not
          strictly an X-site normalisation.
        * ``'fe2+'`` -- only the stoichiometrically estimated ferrous iron
          is placed on the divalent site, excluding Fe3+.
        """
        proportions = garnet_cations_from_apfu(
            out, fe_basis=self.fe_basis, absent="zero"
        )
        Mg, Fe, Mn, Ca = (proportions[key] for key in ("Mg", "Fe", "Mn", "Ca"))

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

    def gt_along_path(self, P, T, fractionate: bool = False,
                      normalise_start: bool = True) -> tuple[np.ndarray, ...]:
        """Calculate garnet fractions and element chemistry along a P-T path.

        Parameters
        ----------
        P : array-like
            Pressure values along the path (kbar).
        T : array-like
            Temperature values along the path (°C).
        fractionate : bool, default False
            If True, the garnet present at each P-T point is removed from the
            reactive bulk **in full** before stepping on: instantaneous
            segregation of refractory garnet (Rayleigh fractionation in the
            limit of small P-T steps).  Only the amount of garnet *formed since the
            last removal* exists on the depleted bulk at the next point, so
            the entire current fraction is new growth and is segregated.
        normalise_start : bool, default True
            Controls how the first P-T point is treated:

            * ``True`` — The first P-T point is treated as a nucleation
              barrier: the initial garnet fraction (if any) stays in the
              reactive bulk as a retained seed baseline and is never removed;
              each later step removes only the garnet formed beyond it.  Use
              when the path starts outside the garnet stability field or at
              the nucleation threshold (the usual case — the baseline is then
              zero and every step removes the entire current fraction).

            * ``False`` — The first P-T point has an initial garnet volume
              (overstepped nucleation).  That fraction is removed from the
              bulk at step 0, and every subsequent step removes its full
              current garnet fraction.  Use when the path starts well inside
              the garnet stability field.

        Returns
        -------
        gt_mol_frac : numpy.ndarray
            Garnet molar fraction at each P-T point.  With
            ``fractionate=False`` this is the equilibrium mode; with
            ``fractionate=True`` it is the **cumulative segregated pile** as
            a fraction of the original system (monotone — the garnet
            physically present in the rock as isolated grains).
        gt_wt_frac : numpy.ndarray
            As ``gt_mol_frac``, on a weight-fraction basis.
        gt_vol_frac : numpy.ndarray
            As ``gt_mol_frac``, on a volume-fraction basis.  This is the
            curve the growth/cohort bookkeeping consumes, so under
            ``fractionate=True`` it grows monotonically to the final pile.
        Mgi : numpy.ndarray
            Closed Mg-Fe-Mn-Ca cation proportion for Mg of the garnet
            segregating at each P-T point (equilibrium composition of that
            step's new growth).
        Mni : numpy.ndarray
            Closed Mg-Fe-Mn-Ca cation proportion for Mn, as ``Mni``.
        Fei : numpy.ndarray
            Closed Mg-Fe-Mn-Ca cation proportion for Fe, as ``Fei``.
            The basis is set by ``fe_basis``; ``FeOt`` includes ferric kho Fe
            on the Y site, so these proportions are not necessarily X-site
            occupancies.
        Cai : numpy.ndarray
            Closed Mg-Fe-Mn-Ca cation proportion for Ca, as ``Cai``.
        X_along_path : numpy.ndarray
            Bulk composition after each step's fractionation.  Each row is
            normalised to sum to 1.  Row ``i`` is the bulk **after** step
            ``i``'s fractionation has been applied.

        Notes
        -----
        Each removal uses the **current-step** garnet composition — a
        first-order approximation valid for small P-T steps.  The retained
        seed baseline (``normalise_start=True`` with garnet at the first
        point) is a mode of the original bulk reapplied at every step, also
        first-order for a small seed.  Basis conversion of each removal into
        the mol/wt/vol pile curves uses that step's equilibrium mode ratios.
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

        phase_functions = PhaseFunctions(db=self.db, dataset=self.dataset, verbose=self.verbose) if fractionate else None
        if phase_functions:
            # Sync standardised state to the helper instance
            self._copy_state_to(phase_functions)
            # Seed baseline in the fractionation basis: with normalise_start the
            # garnet at the first point stays in the reactive bulk and is never
            # re-removed; otherwise it is stripped at step 0.  For a path that
            # starts outside the garnet field (the usual case) the baseline is
            # zero and every step removes the entire current fraction.
            retained_seed = 0.0
            # Cumulative segregated pile per basis as a fraction of the
            # ORIGINAL system, with the matching remaining fractions.
            pile = {"mol": 0.0, "wt": 0.0, "vol": 0.0}
            remaining = {"mol": 1.0, "wt": 1.0, "vol": 1.0}

        for i, (P_step, T_step) in enumerate(zip(P, T)):
            gt_frac, gt_wt, gt_vol, Mg, Mn, Fe, Ca, out = self._gt_single_point_from_jl(
                P_step, T_step, X, Xoxides, self.sys_in, self.rm_list
            )

            # Select the fraction basis consistent with sys_in
            if self.sys_in.casefold() == 'wt':
                gt_basis = gt_wt
            else:
                gt_basis = gt_frac

            if phase_functions is not None:
                # Full-fraction (Rayleigh) removal: everything the
                # current reactive bulk holds is segregated at this step,
                # except the retained seed baseline when normalise_start.
                if i == 0:
                    if normalise_start:
                        retained_seed = gt_basis
                        frac_amount = 0.0
                    else:
                        frac_amount = gt_basis
                else:
                    frac_amount = max(gt_basis - retained_seed, 0.0)

                if frac_amount > 0.0:
                    X_py = phase_functions.fractionate_phase('g', out, self.sys_in, frac_amount=frac_amount)
                    X = jlconvert(jl.Vector[jl.Float64], X_py)

                    # Accumulate the segregated pile in every basis; the
                    # removal is a fraction of the CURRENT system, so each
                    # basis chain carries its own remaining fraction.  Basis
                    # conversion uses this step's equilibrium mode ratios
                    # (modes of the same system, exact within the step).
                    for key, gt_val in (("mol", gt_frac), ("wt", gt_wt), ("vol", gt_vol)):
                        u = frac_amount * (gt_val / gt_basis) if gt_basis > 0 else 0.0
                        u = min(max(u, 0.0), 1.0)
                        pile[key] += remaining[key] * u
                        remaining[key] *= 1.0 - u

                # Report the segregated pile: the garnet actually present in
                # the rock as isolated grains (the growth record).
                gt_mol_frac[i] = pile["mol"]
                gt_wt_frac[i] = pile["wt"]
                gt_vol_frac[i] = pile["vol"]
            else:
                # Closed system: the equilibrium mode IS the garnet present.
                gt_mol_frac[i] = gt_frac
                gt_wt_frac[i] = gt_wt
                gt_vol_frac[i] = gt_vol

            X_along_path[i] = np.array(X) / np.sum(X)

            Mgi[i] = Mg
            Mni[i] = Mn
            Fei[i] = Fe
            Cai[i] = Ca

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, Mgi, Mni, Fei, Cai, X_along_path
