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

    def calculate_misfit(self, pt_guess, target_chemistry, phase, components, comp_type='element', uncertainty=None):
        """Objective function: Root Mean Square Error."""
        P, T = pt_guess
        try:
            out = phasetools.MAGEMin_C.single_point_minimization(
                P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list
            )
            predicted = self._get_phase_composition(out, phase, components, comp_type)
            
            diff = predicted - target_chemistry
            
            if uncertainty is not None:
                sigma = np.array(uncertainty)
            else:
                sigma = np.abs(target_chemistry)
                
            # Avoid division by zero: if sigma is 0, use 1.0 (absolute error)
            sigma = np.where(sigma < 1e-12, 1.0, sigma)
            
            return np.sqrt(np.mean((diff / sigma)**2))
        except Exception:
            return 1e6 

    def solve(self, target_chemistry, phase, components, comp_type, bounds, uncertainty=None,
              x0=None, method='differential_evolution', quick=False, return_path=False, **kwargs):
        """
        Finds the optimal P-T using global or local optimisation.

        If return_path=True, the OptimizeResult object will include a .path attribute 
        containing the history of the best solutions found during the search.
        """
        args = (target_chemistry, phase, components, comp_type, uncertainty)
        path = []

        def callback(x, *args):
            path.append(np.array(x).copy())

        # 1. Global Solvers
        if method == 'differential_evolution':
            de_params = {'strategy': 'best1bin', 'popsize': 10, 'tol': 0.01, 'polish': True}
            if quick: de_params.update({'popsize': 5, 'tol': 0.1, 'polish': False})
            if return_path: de_params['callback'] = callback
            de_params.update(kwargs)
            res = optimize.differential_evolution(self.calculate_misfit, bounds=bounds, args=args, **de_params)

        elif method == 'dual_annealing':
            da_params = {'maxiter': 1000}
            if quick: da_params.update({'maxiter': 50})
            if return_path: da_params['callback'] = lambda x, f, context: path.append(np.array(x).copy())
            da_params.update(kwargs)
            res = optimize.dual_annealing(self.calculate_misfit, bounds=bounds, args=args, **da_params)

        elif method == 'shgo':
            shgo_params = {}
            if quick: shgo_params.update({'n': 32, 'iters': 1})
            if return_path: shgo_params['callback'] = callback
            shgo_params.update(kwargs)
            res = optimize.shgo(self.calculate_misfit, bounds=bounds, args=args, **shgo_params)

        # 2. Local Solvers (scipy.optimize.minimize)
        else:
            if x0 is None:
                x0 = [np.mean(b) for b in bounds]

            min_params = {'method': method, 'bounds': bounds}
            if return_path: min_params['callback'] = callback
            min_params.update(kwargs)
            res = optimize.minimize(self.calculate_misfit, x0=x0, args=args, **min_params)

        if return_path:
            res.path = np.array(path)

        # 3. Final Modelled Composition
        # Calculate the actual composition at the best P-T point found.
        try:
            out = phasetools.MAGEMin_C.single_point_minimization(
                res.x[0], res.x[1], self.data, X=self.X, Xoxides=self.Xoxides, 
                sys_in=self.sys_in, rm_list=self.rm_list
            )
            res.modelled_composition = self._get_phase_composition(out, phase, components, comp_type)
        except Exception:
            res.modelled_composition = None

        return res

