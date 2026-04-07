import numpy as np
from scipy import optimize
import phasetools
import juliacall
from juliacall import Main as jl, convert as jlconvert
from molmass import Formula
from phasetools.functions.MAGEMin_functions import phase_frac, extract_end_member, get_oxide_apfu

class PhasePTEstimator:
    """
    Generalized utility for estimating P-T conditions by minimizing the misfit 
    between measured phase chemistry and MAGEMin equilibrium predictions.
    
    Supports any mineral phase and any combination of elements or end-members.
    """
    def __init__(self, db="mpe", dataset=636, verbose=False):
        self.db = db
        self.MAGEMin_data = phasetools.MAGEMin_C.Initialize_MAGEMin(db, dataset=dataset, verbose=verbose)
        self.X_bulk = None
        self.Xoxides = None
        self.rm_list = None
        self._stoich_map = {} # Cache for element multipliers

    def setup_bulk_composition(self, oxides, values, removed_phases=[], sys_in='mol'):
        """
        Sets up the bulk composition and phase suppression list.
        """
        X_conv, Xox_conv = phasetools.MAGEMin_C.convertBulk4MAGEMin(
            jlconvert(jl.Vector[jl.Float64], values),
            jlconvert(jl.Vector[jl.String], oxides),
            sys_in, self.db
        )
        self.X_bulk = np.array(X_conv) / 100.
        self.Xoxides = list(Xox_conv)
        
        # Pre-calculate stoichiometry using molmass
        self._stoich_map = {}
        for ox in self.Xoxides:
            f = Formula(ox)
            # Use dataframe to extract element counts
            df = f.composition().dataframe()
            # The index of the dataframe is the element symbol
            for el in df.index:
                if el != 'O':
                    if ox not in self._stoich_map:
                        self._stoich_map[ox] = {}
                    self._stoich_map[ox][el] = float(df.loc[el, 'Count'])

        rm_j = jlconvert(jl.Vector[jl.String], removed_phases)
        self.rm_list = phasetools.MAGEMin_C.remove_phases(rm_j, self.db)

    def _get_phase_composition(self, out, phase, components, comp_type='element'):
        """
        Extracts specific chemical components from a MAGEMin output object.
        """
        if phase not in out.ph:
            return np.zeros(len(components))

        if comp_type == 'endmember':
            return np.array([extract_end_member(phase, out, component, 'mol') for component in components])
        
        # Use the oxides defined in the bulk composition
        oxide_moles = get_oxide_apfu(out, phase, self.Xoxides)

        # Build element map dynamically using pre-calculated stoichiometry
        element_map = {}
        for ox, stoichiometry in self._stoich_map.items():
            for el, mult in stoichiometry.items():
                element_map[el] = element_map.get(el, 0.0) + oxide_moles.get(ox, 0.0) * mult
        
        vals = np.array([element_map.get(c, 0.0) for c in components])
        total = np.sum(vals)
        return vals / total if total > 0 else vals

    def calculate_misfit(self, pt_guess, target_chemistry, phase, components, comp_type='element'):
        """
        Objective function: Normalized Root Mean Square Error.
        """
        P, T = pt_guess
        try:
            X_jl = jlconvert(jl.Vector[jl.Float64], self.X_bulk)
            Xox_jl = jlconvert(jl.Vector[jl.String], self.Xoxides)
            
            out = phasetools.MAGEMin_C.single_point_minimization(
                P, T, self.MAGEMin_data, X=X_jl, Xoxides=Xox_jl, sys_in='mol', rm_list=self.rm_list
            )
            
            predicted = self._get_phase_composition(out, phase, components, comp_type)
            return np.sqrt(np.sum(((predicted - target_chemistry) / target_chemistry)**2))
        except Exception:
            return 1e6 

    def solve(self, target_chemistry, 
              phase, components, comp_type, bounds, 
              x0=None, method='differential_evolution', quick=False, **kwargs):
        """
        Finds the optimal P-T using global or local optimization.
        
        Parameters
        ----------
        target_chemistry : array-like
            The measured composition to match.
        phase : str, default='g'
            Phase name in MAGEMin.
        components : list, default=['Mg', 'Mn', 'Fe', 'Ca']
            Components to minimize.
        comp_type : str, default='element'
            'element' or 'endmember'.
        bounds : list of tuples, default=[(5, 28), (400, 750)]
            P and T search ranges.
        x0 : array-like, optional
            Initial guess [P, T] for local methods.
        method : str, default='differential_evolution'
            Optimization strategy: 'differential_evolution', 'dual_annealing', 'shgo',
            or any method accepted by scipy.optimize.minimize.
        quick : bool, default=False
            If True, uses faster settings for global solvers.
        """
        
        args = (target_chemistry, phase, components, comp_type)

        # 1. Global Solvers
        if method == 'differential_evolution':
            de_params = {'strategy': 'best1bin', 'popsize': 10, 'tol': 0.01, 'polish': True}
            if quick: de_params.update({'popsize': 5, 'tol': 0.1, 'polish': False})
            de_params.update(kwargs)
            return optimize.differential_evolution(self.calculate_misfit, bounds=bounds, args=args, **de_params)

        elif method == 'dual_annealing':
            da_params = {'maxiter': 1000}
            if quick: da_params.update({'maxiter': 50})
            da_params.update(kwargs)
            return optimize.dual_annealing(self.calculate_misfit, bounds=bounds, args=args, **da_params)

        elif method == 'shgo':
            shgo_params = {}
            if quick: shgo_params.update({'n': 32, 'iters': 1})
            shgo_params.update(kwargs)
            return optimize.shgo(self.calculate_misfit, bounds=bounds, args=args, **shgo_params)

        # 2. Local Solvers (scipy.optimize.minimize)
        else:
            if x0 is None:
                x0 = [np.mean(b) for b in bounds]
            
            min_params = {'method': method, 'bounds': bounds}
            min_params.update(kwargs)
            return optimize.minimize(self.calculate_misfit, x0=x0, args=args, **min_params)
