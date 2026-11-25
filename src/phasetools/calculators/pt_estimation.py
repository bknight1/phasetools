from collections.abc import Callable, Sequence

import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeResult

import phasetools

from ..core.base import MAGEMinBase
from ..core.phase_properties import extract_end_member, get_oxide_apfu


def optimize_pt(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    method: str = "differential_evolution",
    x0: list[float] | None = None,
    uncertainty: np.ndarray | None = None,
    return_path: bool = False,
    quick: bool = False,
    **kwargs: object,
) -> OptimizeResult:
    """Minimise an objective over pressure and temperature.

    Parameters
    ----------
    objective : callable
        Function accepting a two-element ``[P, T]`` array and returning a
        scalar misfit. Any uncertainty weighting belongs in this callable;
        ``uncertainty`` is accepted for API compatibility with P-T callers.
    bounds : list of tuple of float
        Pressure and temperature limits, in kbar and °C respectively.
    method : str, default='differential_evolution'
        A SciPy global method (``differential_evolution``, ``dual_annealing``
        or ``shgo``), or a method supported by ``scipy.optimize.minimize``.
    x0 : list of float, optional
        Initial P-T estimate for local methods. Defaults to the bounds midpoint.
    uncertainty : ndarray, optional
        Reserved for call-site compatibility; apply uncertainty weights inside
        ``objective`` so it remains a one-argument callable.
    return_path : bool, default=False
        Attach accepted optimizer callback points as ``result.path``.
    quick : bool, default=False
        Use reduced global-search effort.
    **kwargs
        Additional keyword arguments passed to the selected SciPy optimiser.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimisation result with ``x`` ordered as ``[P, T]`` and ``fun`` as
        the objective misfit. If requested, ``path`` contains callback points.

    Raises
    ------
    ValueError
        If the bounds do not describe a pressure-temperature pair.
    """
    del uncertainty  # Weighting is objective-specific and belongs in its closure.
    if len(bounds) != 2:
        raise ValueError("bounds must contain exactly (P_bounds, T_bounds)")

    path: list[np.ndarray] = []

    def callback(x: Sequence[float], *args: object) -> None:
        del args
        path.append(np.asarray(x, dtype=float).copy())

    if method == "differential_evolution":
        params: dict[str, object] = {
            "strategy": "best1bin", "popsize": 10, "tol": 0.01, "polish": True,
        }
        if quick:
            params.update({"popsize": 5, "tol": 0.1, "polish": False})
        if return_path:
            params["callback"] = callback
        params.update(kwargs)
        result = optimize.differential_evolution(objective, bounds=bounds, **params)
    elif method == "dual_annealing":
        params = {"maxiter": 50 if quick else 1000}
        if return_path:
            params["callback"] = lambda x, _f, _context: path.append(
                np.asarray(x, dtype=float).copy()
            )
        params.update(kwargs)
        result = optimize.dual_annealing(objective, bounds=bounds, **params)
    elif method == "shgo":
        params = {"n": 32, "iters": 1} if quick else {}
        if return_path:
            params["callback"] = callback
        params.update(kwargs)
        result = optimize.shgo(objective, bounds=bounds, **params)
    else:
        initial = x0 if x0 is not None else [float(np.mean(bound)) for bound in bounds]
        params = {"method": method, "bounds": bounds}
        if return_path:
            params["callback"] = callback
        params.update(kwargs)
        result = optimize.minimize(objective, x0=initial, **params)

    if return_path:
        result.path = np.asarray(path, dtype=float).reshape((-1, 2))
    return result


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
        # Handle special components Fe2 and Fe3 first
        if 'Fe2' in components or 'Fe3' in components:
            split = self._extract_fe_split_from_apfu(out, phase)
            element_map['Fe2'] = split['Fe2']
            element_map['Fe3'] = split['Fe3']

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
        except Exception:  # noqa: BLE001 — optimiser penalty: any MAGEMin failure means this P-T is infeasible
            return 1e6 

    def solve(self, target_chemistry, phase, components, comp_type, bounds, uncertainty=None,
              x0=None, method='differential_evolution', quick=False, return_path=False, **kwargs):
        """
        Finds the optimal P-T using global or local optimisation.

        If return_path=True, the OptimizeResult object will include a .path attribute 
        containing the history of the best solutions found during the search.
        """
        def objective(pt: np.ndarray) -> float:
            return self.calculate_misfit(
                pt, target_chemistry, phase, components, comp_type, uncertainty
            )

        res = optimize_pt(
            objective,
            bounds,
            method=method,
            x0=x0,
            uncertainty=uncertainty,
            return_path=return_path,
            quick=quick,
            **kwargs,
        )

        # 3. Final Modelled Composition
        # Calculate the actual composition at the best P-T point found.
        try:
            out = phasetools.MAGEMin_C.single_point_minimization(
                res.x[0], res.x[1], self.data, X=self.X, Xoxides=self.Xoxides, 
                sys_in=self.sys_in, rm_list=self.rm_list
            )
            res.modelled_composition = self._get_phase_composition(out, phase, components, comp_type)
        except Exception:  # noqa: BLE001 — any calc failure → no composition to compare
            res.modelled_composition = None

        return res
