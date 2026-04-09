import numpy as np
from juliacall import Main as jl, convert as jlconvert
from molmass import Formula
from phasetools import MAGEMin_C

from ..utils.bulk_rock import convert_mol_percent_to_wt_percent, get_molar_mass_dict

class MAGEMinBase:
    def __init__(self, db="ig", dataset=636, verbose=False):
        self.db = db
        self.dataset = dataset
        self.verbose = verbose
        self.data = MAGEMin_C.Initialize_MAGEMin(db, dataset=dataset, verbose=verbose)
        self.X = None
        self.Xoxides = None
        self.sys_in = None
        self.rm_list = None

    def _copy_state_to(self, other):
        """Syncs standardised state to another MAGEMinBase-derived instance."""
        other.data = self.data
        other.X = self.X
        other.Xoxides = self.Xoxides
        other.sys_in = self.sys_in
        other.rm_list = self.rm_list
        other._Xoxides_py = self._Xoxides_py
        other._stoich_map = self._stoich_map

    def setup_bulk_composition(self, Xoxides, X, sys_in, rm_list=None):
        from ..utils.bulk_rock import convert_mol_percent_to_wt_percent, get_molar_mass_dict

        # 1. Standardise using MAGEMin conversion (adds 'O', normalises, etc.)
        # MAGEMin returns standardised composition in molar fractions
        X_jl_raw = jlconvert(jl.Vector[jl.Float64], X)
        Xox_jl_raw = jlconvert(jl.Vector[jl.String], Xoxides)

        X_std_jl, Xox_std_jl = MAGEMin_C.convertBulk4MAGEMin(X_jl_raw, Xox_jl_raw, sys_in, self.db)

        # Convert back to Python lists for processing
        self._Xoxides_py = [str(ox) for ox in Xox_std_jl]
        X_mol = [float(val) for val in X_std_jl]

        # 2. If sys_in is wt, convert the standardised molar fractions back to wt%
        if sys_in.casefold() == 'wt':
            mass_dict = get_molar_mass_dict()
            X_wt = convert_mol_percent_to_wt_percent(X_mol, self._Xoxides_py, mass_dict)
            self.X = jlconvert(jl.Vector[jl.Float64], X_wt)
            self.sys_in = 'wt'
        else:
            self.X = X_std_jl
            self.sys_in = 'mol'

        self.Xoxides = Xox_std_jl

        if rm_list is not None:
            rm_j = jlconvert(jl.Vector[jl.String], rm_list)
            self.rm_list = MAGEMin_C.remove_phases(rm_j, self.db)

        # 3. Pre-calculate stoichiometry using molmass for the standardised oxides
        self._stoich_map = {}
        for ox in self._Xoxides_py:
            try:
                f = Formula(ox)
                df = f.composition().dataframe()
                for el in df.index:
                    if el != 'O':
                        if ox not in self._stoich_map:
                            self._stoich_map[ox] = {}
                        self._stoich_map[ox][el] = float(df.loc[el, 'Count'])
            except:
                pass
