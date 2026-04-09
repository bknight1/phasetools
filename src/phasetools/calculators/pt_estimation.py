import numpy as np
import phasetools
from scipy import optimize
from ..core.base import MAGEMinBase
from ..core.phase_properties import extract_end_member, get_oxide_apfu

class PhasePTEstimator(MAGEMinBase):
    """
    Generalized utility for estimating P-T conditions by minimizing the misfit 
    between measured phase chemistry and MAGEMin equilibrium predictions.
    """
    def __init__(self, db="ig", dataset=636, verbose=False):
        super().__init__(db, dataset, verbose)
    
    def _get_phase_composition(self, out, phase, components, comp_type='element'):
        """Extracts specific chemical components from a MAGEMin output object."""
        if phase not in out.ph:
            return np.zeros(len(components))

        if comp_type == 'endmember':
            return np.array([extract_end_member(phase, out, component, self.sys_in) for component in components])
        
        oxide_moles = get_oxide_apfu(out, phase, self._Xoxides_py)

        element_map = {}
        for ox, stoichiometry in self._stoich_map.items():
            for el, mult in stoichiometry.items():
                element_map[el] = element_map.get(el, 0.0) + oxide_moles.get(ox, 0.0) * mult
        
        vals = np.array([element_map.get(c, 0.0) for c in components])
        total = np.sum(vals)
        return vals / total if total > 0 else vals

    def calculate_misfit(self, pt_guess, target_chemistry, phase, components, comp_type='element'):
        """Objective function: Normalised Root Mean Square Error."""
        P, T = pt_guess
        try:
            out = phasetools.MAGEMin_C.single_point_minimization(
                P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list
            )
            predicted = self._get_phase_composition(out, phase, components, comp_type)
            return np.sqrt(np.sum(((predicted - target_chemistry) / target_chemistry)**2))
        except Exception:
            return 1e6 

    def solve(self, target_chemistry, phase, components, comp_type, bounds, 
              x0=None, method='differential_evolution', quick=False, **kwargs):
        """Finds the optimal P-T using global or local optimisation."""
        args = (target_chemistry, phase, components, comp_type)

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
        else:
            if x0 is None:
                x0 = [np.mean(b) for b in bounds]
            min_params = {'method': method, 'bounds': bounds}
            min_params.update(kwargs)
            return optimize.minimize(self.calculate_misfit, x0=x0, args=args, **min_params)
