"""Garnet maximum-driving-force (MDF) and parallel-tangent model (PTM).

This module implements the garnet MDF/PTM workflow with explicit active-model
validation, optional kho exchange, and closed Mg-Fe-Mn-Ca cation projections.

References
----------
- https://doi.org/10.1007/s00410-014-1059-6
- https://doi.org/10.1016/j.chemgeo.2018.05.019
- https://doi.org/10.1016/j.chemgeo.2019.119323
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from juliacall import Main as jl
from juliacall import convert as jlconvert
from scipy.optimize import minimize

from phasetools import MAGEMin_C

from ..core.base import MAGEMinBase
from ..core.engine import evaluate_solution_phase_gibbs
from ..core.thermodynamics import extract_gamma
from .garnet_composition import normalise_fe_basis, normalise_garnet_cations

_DEFAULT_COMPONENTS = {
    "pyrope": "py",
    "almandine": "alm",
    "spessartine": "spss",
    "grossular": "gr",
}
_CATION_KEYS = ("Mg", "Fe", "Mn", "Ca")
_FRACTION_FLOOR = 1.0e-12
_PENALTY = 1.0e12


@dataclass(frozen=True)
class GarnetMDFResult:
    """Closed cation compositions and MDF diagnostics at one P-T point.

    ``mdf`` and ``equilibrium`` always have exactly the keys Mg, Fe, Mn and Ca.
    Both are normalised to the sum of these four cations using ``fe_basis``.
    ``residual_*`` is in kJ/mol garnet formula and measures the MDF
    parallel-tangent equations. Almandine always contributes three Fe; kho
    contributes three Mg and two Fe only on the ``FeOt`` basis. These are not
    APFU or X-site proportions; Al, Si, O and other cations are excluded.
    """

    mdf: Mapping[str, float]
    equilibrium: Mapping[str, float]
    residual_kj_per_formula: tuple[float, ...]
    residual_norm_kj_per_formula: float
    converged: bool
    status: str
    matrix_phases: tuple[str, ...]
    equilibrium_phases: tuple[str, ...]
    active_endmember_names: tuple[str, ...] = ()
    _mdf_endmember_fractions: Mapping[str, float] = field(
        default_factory=dict, repr=False
    )
    _equilibrium_endmember_fractions: Mapping[str, float] = field(
        default_factory=dict, repr=False
    )
    tolerance_kj: float = 0.03
    message: str = ""
    fe_basis: str = "FeOt"
    solver_mode: str = "full"
    cation_closure: str = (
        "Closed Mg-Fe-Mn-Ca cation proportions; the Fe basis is recorded in "
        "fe_basis. Almandine contributes 3 Fe on either basis; kho contributes "
        "2 Fe only for FeOt. These are not APFU or X-site proportions; Al, Si, "
        "O and other cations are excluded."
    )


@dataclass(frozen=True)
class GarnetMDFGridResult:
    """Closed cation compositions and diagnostics over a P-T grid."""

    P_grid: np.ndarray
    T_grid: np.ndarray
    mdf: Mapping[str, np.ndarray]
    equilibrium: Mapping[str, np.ndarray]
    residual_norm_kj_per_formula: np.ndarray
    converged: np.ndarray
    status: np.ndarray
    active_endmember_names: np.ndarray
    matrix_assemblages: np.ndarray
    fe_basis: str = "FeOt"
    solver_mode: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))


def kho_aware_exchange_potentials(
    gamma: Mapping[str, float],
    component_names: Mapping[str, str] | None = None,
    *,
    kho_name: str = "kho",
) -> dict[str, float]:
    """Project oxide Gamma onto exchange potentials relative to pyrope.

    The divalent exchanges use ``3*(Gamma_MO-Gamma_MgO)``. For
    ``Mg3Fe3+2Si3O12`` kho relative to pyrope is ``2 FeO + O - Al2O3`` on the
    TC FeO + excess-O basis. The named oxide mapping makes the calculation
    independent of Gamma vector ordering.

    Parameters
    ----------
    gamma : mapping
        MAGEMin oxide chemical potentials in kJ/mol oxide.
    component_names : mapping, optional
        Semantic roles to active garnet names. Defaults to py, alm, spss, gr.
    kho_name : str, default='kho'
        Active model name for the ferric endmember.

    Returns
    -------
    dict[str, float]
        Target exchange potential for every supplied non-reference
        ``component_names`` role and kho, keyed by active endmember name.
    """
    roles = _validate_component_mapping(component_names)
    values = {str(name): float(value) for name, value in gamma.items()}
    if any(not np.isfinite(value) for value in values.values()):
        raise ValueError("gamma values must be finite")
    required = {"MgO", "FeO", "Al2O3", "O"}
    if "spessartine" in roles:
        required.add("MnO")
    if "grossular" in roles:
        required.add("CaO")
    missing = required.difference(values)
    if missing:
        raise ValueError(
            f"kho-aware matrix Gamma is missing required oxide(s): {sorted(missing)}"
        )
    result = {
        roles["almandine"]: 3.0 * (values["FeO"] - values["MgO"]),
        kho_name: 2.0 * values["FeO"] + values["O"] - values["Al2O3"],
    }
    if "spessartine" in roles:
        result[roles["spessartine"]] = 3.0 * (values["MnO"] - values["MgO"])
    if "grossular" in roles:
        result[roles["grossular"]] = 3.0 * (values["CaO"] - values["MgO"])
    return result


def garnet_fractions_to_mg_fe_mn_ca(
    fractions: Mapping[str, float],
    *,
    component_names: Mapping[str, str] | None = None,
    kho_name: str = "kho",
    fe_basis: str = "FeOt",
) -> dict[str, float]:
    """Convert supported garnet endmember fractions to closed cation ratios.

    Each py/alm/spss/gr formula contributes three corresponding X-site
    cations. Kho contributes three Mg and, on the ``FeOt`` basis, two total Fe
    cations; it contributes no Fe on the ``Fe2+`` basis. Almandine contributes
    three Fe on either basis. All four outputs are normalised to their joint
    represented cation total.

    Parameters
    ----------
    fractions : mapping
        Garnet endmember names and their molar fractions.
    component_names : mapping, optional
        Semantic roles to active garnet names. Defaults to py, alm, spss, gr.
    kho_name : str, default="kho"
        Active model name for the ferric endmember.
    fe_basis : {"FeOt", "Fe2+"}, default="FeOt"
        Whether kho Fe is included in the closed cation projection.

    Raises
    ------
    ValueError
        If the basis or any supplied fraction is unsupported or invalid.
    """
    basis = normalise_fe_basis(fe_basis)
    roles = _validate_component_mapping(component_names)
    amounts = {key: 0.0 for key in _CATION_KEYS}
    mapping = {
        roles["pyrope"]: ("Mg", 3.0),
        roles["almandine"]: ("Fe", 3.0),
        roles["grossular"]: ("Ca", 3.0),
    }
    if "spessartine" in roles:
        mapping[roles["spessartine"]] = ("Mn", 3.0)
    mapping[kho_name] = ("Mg", 3.0)
    for name, value in fractions.items():
        amount = float(value)
        if not np.isfinite(amount) or amount < 0.0:
            raise ValueError("endmember fractions must be finite and non-negative")
        if name not in mapping:
            raise ValueError(
                f"Unsupported garnet endmember {name!r}; cation projection "
                "cannot validate an unrecognised component"
            )
        cation, coefficient = mapping[name]
        amounts[cation] += coefficient * amount
        if name == kho_name and basis == "FeOt":
            amounts["Fe"] += 2.0 * amount
    return normalise_garnet_cations(amounts)


def _validate_component_mapping(
    component_names: Mapping[str, str] | None,
) -> dict[str, str]:
    """Validate role names while allowing Mn to be absent from an active model."""
    if component_names is None:
        roles = dict(_DEFAULT_COMPONENTS)
    else:
        if not isinstance(component_names, Mapping):
            raise TypeError("component_names must be a semantic-role mapping")
        unknown = set(component_names).difference(_DEFAULT_COMPONENTS)
        if unknown:
            raise ValueError(f"Unsupported garnet component role(s): {sorted(unknown)}")
        roles = {
            role: str(component_names[role])
            for role in _DEFAULT_COMPONENTS
            if role in component_names
        }
    required = {"pyrope", "almandine", "grossular"}
    if not required.issubset(roles):
        raise ValueError("pyrope, almandine and grossular mappings are required")
    if any(not name for name in roles.values()) or len(set(roles.values())) != len(
        roles
    ):
        raise ValueError("component_names values must be non-empty and unique")
    return roles


class GarnetMDFCalculator(MAGEMinBase):
    """Solve garnet maximum-driving-force (MDF) equations by a PTM workflow.

    The maximum-driving-force (MDF) and parallel-tangent method/model (PTM)
    describe this garnet calculation and its exchange-equation workflow.

    References
    ----------
    - https://doi.org/10.1007/s00410-014-1059-6
    - https://doi.org/10.1016/j.chemgeo.2018.05.019
    - https://doi.org/10.1016/j.chemgeo.2019.119323

    The active solution model is discovered from garnet-bearing equilibrium.
    Standardised molar bulk O determines the MDF path: absent or zero O uses
    the py-alm-optional-spss-gr restricted simplex with kho exactly zero;
    positive O uses the full kho-aware simplex and requires an active kho
    component. Any unsupported active component is rejected. Ordinary
    equilibrium is retained independently and is never constrained by this
    MDF dispatch. Output mappings are closed Mg/Fe/Mn/Ca cation proportions,
    using ``FeOt`` by default or the ferrous-only ``Fe2+`` projection when
    requested. These are not APFU or X-site proportions.
    """

    def __init__(
        self,
        db: str = "mp",
        dataset: int = 636,
        verbose: bool = False,
        *,
        component_names: Mapping[str, str] | None = None,
        kho_name: str = "kho",
    ) -> None:
        super().__init__(db=db, dataset=dataset, verbose=verbose)
        self.cation_component_names = _validate_component_mapping(component_names)
        self.kho_name = str(kho_name)
        if not self.kho_name:
            raise ValueError("kho_name must be a non-empty name")
        if self.kho_name in self.cation_component_names.values():
            raise ValueError("kho_name must be distinct from divalent component names")
        self._active_names: tuple[str, ...] = ()
        self._standardized_bulk_molar_oxides: tuple[str, ...] = ()
        self._standardized_bulk_molar_values: tuple[float, ...] = ()
        self._standardized_bulk_oxygen_mol = 0.0

    def setup_bulk_composition(
        self,
        Xoxides: Sequence[str],
        X: Sequence[float],
        sys_in: str,
        rm_list: Sequence[str] | None = None,
    ) -> None:
        """Configure a bulk composition and retain the input unit basis."""
        super().setup_bulk_composition(Xoxides, X, sys_in, rm_list=rm_list)
        self._standardized_bulk_molar_oxides = tuple(self._Xoxides_py)
        self._standardized_bulk_molar_values = tuple(self._X_mol_standardized)
        try:
            oxygen_index = self._standardized_bulk_molar_oxides.index("O")
        except ValueError:
            self._standardized_bulk_oxygen_mol = 0.0
        else:
            self._standardized_bulk_oxygen_mol = float(
                self._standardized_bulk_molar_values[oxygen_index]
            )
        if not getattr(self, "_preserve_configured_bulk", False):
            self._configured_bulk_basis = str(sys_in)
            self._configured_bulk_oxides = tuple(str(oxide) for oxide in Xoxides)
            self._configured_bulk_values = tuple(float(value) for value in X)

    def _setup_transient_bulk(
        self,
        oxides: Sequence[str],
        values: Sequence[float],
        *,
        rm_list: Sequence[str] | None,
    ) -> None:
        """Set a calculation's bulk without changing its established basis."""
        basis = getattr(self, "_configured_bulk_basis", None) or self.sys_in or "wt"
        previous = getattr(self, "_preserve_configured_bulk", False)
        self._preserve_configured_bulk = True
        try:
            self.setup_bulk_composition(oxides, values, sys_in=basis, rm_list=rm_list)
        finally:
            self._preserve_configured_bulk = previous

    @staticmethod
    def _check_output_status(output: object, calculation: str) -> None:
        """Raise a useful error when MAGEMin reports a failed minimisation."""
        status = int(getattr(output, "status", 0))
        if status != 0:
            raise RuntimeError(
                f"MAGEMin {calculation} minimisation failed with status={status}"
            )

    def _calculate_single_point(
        self, pressure_kbar: float, temperature_c: float
    ) -> object:
        """Run MAGEMin once using the currently configured bulk and removals."""
        pressure = jlconvert(jl.Float64, pressure_kbar)
        temperature = jlconvert(jl.Float64, temperature_c)
        return MAGEMin_C.single_point_minimization(
            pressure,
            temperature,
            self.data,
            X=self.X,
            Xoxides=self.Xoxides,
            sys_in=self.sys_in,
            rm_list=self.rm_list,
        )

    def calculate_equilibrium(
        self, pressure_kbar: float, temperature_c: float
    ) -> object:
        """Run unrestricted equilibrium on the configured bulk and discover garnet."""
        self._require_configured_bulk()
        previous_rm_list = self.rm_list
        try:
            self._setup_transient_bulk(
                self._configured_bulk_oxides,
                self._configured_bulk_values,
                rm_list=[],
            )
            output = self._calculate_single_point(pressure_kbar, temperature_c)
        finally:
            self.rm_list = previous_rm_list
        self._check_output_status(output, "equilibrium")
        if "g" not in tuple(str(phase) for phase in output.ph):
            self._active_names = ()
            self.garnet_endmember_names = ()
            raise RuntimeError("Garnet is not stable in the ordinary equilibrium")
        self.discover_garnet_endmembers(output)
        return output

    def calculate_matrix_state(
        self,
        pressure_kbar: float,
        temperature_c: float,
    ) -> tuple[object, dict[str, float], tuple[str, ...]]:
        """Return garnet-suppressed output, oxide Gamma and stable phases.

        Unrestricted equilibrium is rerun first to validate garnet stability
        and refresh active garnet names at this P–T before suppressing garnet.
        """
        self._require_configured_bulk()
        oxides = self._configured_bulk_oxides
        values = self._configured_bulk_values
        self._setup_transient_bulk(oxides, values, rm_list=[])
        equilibrium = self._calculate_single_point(pressure_kbar, temperature_c)
        self._check_output_status(equilibrium, "equilibrium")
        if "g" not in tuple(str(phase) for phase in equilibrium.ph):
            self._active_names = ()
            self.garnet_endmember_names = ()
            raise RuntimeError(
                "Garnet is not stable in the unrestricted equilibrium; "
                "a garnet-suppressed matrix state is not applicable"
            )
        self.discover_garnet_endmembers(equilibrium)

        previous_rm_list = self.rm_list
        try:
            self._setup_transient_bulk(oxides, values, rm_list=["g"])
            output = self._calculate_single_point(pressure_kbar, temperature_c)
            self._check_output_status(output, "matrix")
            gamma = extract_gamma(output)
            output_oxides = tuple(str(name) for name in output.oxides)
            if gamma is None or len(gamma) != len(output_oxides):
                raise RuntimeError(
                    "MAGEMin output has no compatible Gamma/oxide vector"
                )
            phases = tuple(str(phase) for phase in output.ph)
            self.last_matrix_phases = phases
        finally:
            self.rm_list = previous_rm_list
        return output, dict(zip(output_oxides, gamma, strict=True)), phases

    def _require_configured_bulk(self) -> None:
        """Raise a clear error unless setup_bulk_composition has been called."""
        if not getattr(self, "_configured_bulk_oxides", ()):
            raise ValueError("Configure a bulk composition before calculating")

    def discover_garnet_endmembers(
        self, grid_out: object | None = None
    ) -> tuple[str, ...]:
        """Discover and validate active garnet components from MAGEMin output."""
        self._active_names = ()
        self.garnet_endmember_names = ()
        output_phases = tuple(str(phase) for phase in getattr(grid_out, "ph", ()))
        if grid_out is not None and "g" not in output_phases:
            raise ValueError(
                "Cannot discover garnet endmembers: garnet ('g') is absent from "
                "the supplied equilibrium output."
            )
        garnet_indices = (
            [index for index, phase in enumerate(grid_out.ph) if str(phase) == "g"]
            if grid_out is not None
            else []
        )
        names = ()
        if garnet_indices and garnet_indices[0] < len(grid_out.SS_vec):
            names = tuple(
                str(name) for name in grid_out.SS_vec[garnet_indices[0]].emNames
            )
        if not names:
            raise ValueError(
                "Cannot discover garnet endmembers: supply a garnet-bearing "
                "MAGEMin output with phase 'g'."
            )
        if (
            "spessartine" in self.cation_component_names
            and self.cation_component_names["spessartine"] not in names
        ):
            self.cation_component_names = {
                role: name
                for role, name in self.cation_component_names.items()
                if role != "spessartine"
            }
        expected = set(self.cation_component_names.values()) | {self.kho_name}
        unsupported = set(names).difference(expected)
        if unsupported:
            raise ValueError(
                "Unsupported active garnet endmember(s) for kho-aware cation MDF: "
                f"{sorted(unsupported)}; supported names are {sorted(expected)}"
            )
        required = {
            self.cation_component_names[role]
            for role in ("pyrope", "almandine", "grossular")
        }
        missing = required.difference(names)
        if missing:
            raise ValueError(
                "Active garnet model lacks required restricted MDF component(s): "
                f"{sorted(missing)}; discovered {names}"
            )
        self.garnet_endmember_names = names
        self._active_names = names
        return names

    def _validate_active_model(self, names: Sequence[str]) -> tuple[str, ...]:
        ordered = tuple(str(name) for name in names)
        supported = set(self.cation_component_names.values()) | {self.kho_name}
        unsupported = set(ordered).difference(supported)
        if unsupported:
            raise ValueError(
                "Unsupported active garnet endmember(s) for kho-aware cation MDF: "
                f"{sorted(unsupported)}"
            )
        required = {
            self.cation_component_names[role]
            for role in ("pyrope", "almandine", "grossular")
        } | {self.kho_name}
        missing = required.difference(ordered)
        if missing:
            raise ValueError(
                f"Active garnet model is missing required component(s): {sorted(missing)}"
            )
        if len(set(ordered)) != len(ordered):
            raise ValueError("active garnet endmember names must be unique")
        return ordered

    def _validate_restricted_model(self, names: Sequence[str]) -> tuple[str, ...]:
        """Validate the divalent garnet simplex used when standardised O is zero."""
        ordered = tuple(str(name) for name in names)
        roles = self.cation_component_names
        supported = {
            roles[role]
            for role in ("pyrope", "almandine", "grossular", "spessartine")
            if role in roles
        } | {self.kho_name}
        unsupported = set(ordered).difference(supported)
        if unsupported:
            raise ValueError(
                "Unsupported active garnet endmember(s) for restricted MDF: "
                f"{sorted(unsupported)}"
            )
        required = {roles[role] for role in ("pyrope", "almandine", "grossular")}
        missing = required.difference(ordered)
        if missing:
            raise ValueError(
                "Active garnet model lacks required restricted MDF component(s): "
                f"{sorted(missing)}"
            )
        if len(set(ordered)) != len(ordered):
            raise ValueError("active garnet endmember names must be unique")
        return ordered

    def _standardized_bulk_oxygen(self) -> float:
        """Return standardised molar O received by MAGEMin, or zero if absent."""
        if hasattr(self, "_standardized_bulk_oxygen_mol"):
            oxygen = float(self._standardized_bulk_oxygen_mol)
            if not np.isfinite(oxygen) or oxygen < 0.0:
                raise ValueError("Standardised molar O must be finite and non-negative")
            return oxygen
        oxides = getattr(self, "_standardized_bulk_molar_oxides", ())
        values = getattr(self, "_standardized_bulk_molar_values", ())
        if len(oxides) != len(values):
            raise RuntimeError("Standardised molar bulk metadata is inconsistent")
        try:
            index = oxides.index("O")
        except ValueError:
            return 0.0
        oxygen = float(values[index])
        if not np.isfinite(oxygen) or oxygen < 0.0:
            raise ValueError("Standardised molar O must be finite and non-negative")
        return oxygen

    def _evaluate_mdf_residual(
        self,
        fractions: Mapping[str, float],
        targets: Mapping[str, float],
        pressure_kbar: float,
        temperature_c: float,
        *,
        names: Sequence[str],
        free_names: Sequence[str],
        reference: str,
        fixed_fractions: Mapping[str, float],
    ) -> np.ndarray:
        """Evaluate named pyrope-referenced exchange residuals on a simplex."""
        ordered = tuple(names)
        if set(fractions) != set(ordered):
            raise ValueError("MDF trial fractions must include every active component")
        values = np.asarray([fractions[name] for name in ordered], dtype=float)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or not np.isclose(values.sum(), 1.0)
        ):
            raise ValueError(
                "MDF fractions must be finite, non-negative and sum to one"
            )
        for name, value in fixed_fractions.items():
            if fractions[name] != value:
                raise ValueError(f"Fixed MDF component {name!r} changed during trial")
        state = evaluate_solution_phase_gibbs(
            self,
            "g",
            dict(fractions),
            pressure_kbar,
            temperature_c,
            endmember_names=ordered,
        )
        if not state.valid or len(state.mu) != len(ordered):
            raise RuntimeError(
                "MAGEMin rejected the garnet composition or returned an "
                "incompatible chemical-potential vector"
            )
        mu = dict(zip(ordered, state.mu, strict=True))
        residual_names = tuple(name for name in free_names if name != reference)
        missing = set(residual_names).difference(targets)
        if missing:
            raise ValueError(f"MDF reaction targets are missing {sorted(missing)}")
        return np.asarray(
            [
                mu[name] - mu[reference] - float(targets[name])
                for name in residual_names
            ],
            dtype=float,
        )

    def _solve_mdf_on_simplex(
        self,
        gamma: Mapping[str, float],
        pressure_kbar: float,
        temperature_c: float,
        *,
        names: Sequence[str],
        free_names: Sequence[str],
        fixed_fractions: Mapping[str, float],
        targets: Mapping[str, float],
        reference: str,
        tolerance_kj: float,
        max_iterations: int,
    ) -> tuple[dict[str, float], tuple[float, ...], float, bool, str]:
        """Shared multi-start BFGS solve over free simplex components."""
        if not np.isfinite(tolerance_kj) or tolerance_kj <= 0.0:
            raise ValueError("tolerance_kj must be finite and greater than zero")
        if (
            not isinstance(max_iterations, (int, np.integer))
            or isinstance(max_iterations, bool)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        ordered = tuple(names)
        free = tuple(free_names)
        fixed = {str(name): float(value) for name, value in fixed_fractions.items()}
        if set(free).intersection(fixed) or set(free) | set(fixed) != set(ordered):
            raise ValueError(
                "Free and fixed MDF components must partition active names"
            )
        if any(not np.isfinite(value) or value < 0.0 for value in fixed.values()):
            raise ValueError("Fixed MDF fractions must be finite and non-negative")
        remaining = 1.0 - sum(fixed.values())
        if remaining <= 0.0 or len(free) < 2:
            raise ValueError("MDF simplex needs at least two free components")
        for name, value in fixed.items():
            if value != fixed_fractions[name]:
                raise ValueError("Fixed MDF fractions must be preserved exactly")

        def simplex(logits: Sequence[float]) -> dict[str, float]:
            vector = np.asarray(logits, dtype=float)
            extended = np.concatenate((vector, [0.0]))
            extended -= np.max(extended)
            values = np.exp(extended)
            values = np.maximum(values, _FRACTION_FLOOR)
            values = remaining * values / np.sum(values)
            result = {name: float(value) for name, value in zip(free, values)}
            result.update(fixed)
            return {name: result[name] for name in ordered}

        starts = [np.zeros(len(free) - 1)]
        ascending = np.arange(1.0, len(free) + 1.0)
        ascending /= ascending.sum()
        for values in (ascending, ascending[::-1]):
            logits = np.log(values[:-1]) - np.log(values[-1])
            if not any(np.allclose(logits, start) for start in starts):
                starts.append(logits)
        failures = 0

        def residual_for(logits: Sequence[float]) -> np.ndarray:
            return self._evaluate_mdf_residual(
                simplex(logits),
                targets,
                pressure_kbar,
                temperature_c,
                names=ordered,
                free_names=free,
                reference=reference,
                fixed_fractions=fixed,
            )

        def objective(logits: np.ndarray) -> float:
            nonlocal failures
            try:
                residual = residual_for(logits)
                if not np.all(np.isfinite(residual)):
                    raise RuntimeError("MDF residual contains non-finite values")
                return float(residual @ residual)
            except Exception:  # noqa: BLE001 — invalid trials receive finite cost.
                failures += 1
                return _PENALTY

        results = [
            minimize(
                objective,
                start,
                method="BFGS",
                options={"gtol": 1.0e-8, "maxiter": int(max_iterations)},
            )
            for start in starts
        ]
        best = min(results, key=lambda result: objective(result.x))
        solution = simplex(best.x)
        try:
            residual = residual_for(best.x)
            message = str(best.message)
        except Exception as exc:  # noqa: BLE001 — preserve final diagnostics.
            failures += 1
            residual = np.full(len(free) - 1, np.sqrt(_PENALTY))
            message = (
                f"Final thermodynamic evaluation failed: {type(exc).__name__}: {exc}"
            )
        norm = float(np.linalg.norm(residual))
        converged = bool(np.isfinite(norm) and norm <= tolerance_kj)
        if failures and not message:
            message = f"{failures} invalid optimiser evaluation(s)"
        return solution, tuple(map(float, residual)), norm, converged, message

    def _restricted_mdf_residual(
        self,
        fractions: Sequence[float],
        gamma: Mapping[str, float],
        pressure_kbar: float,
        temperature_c: float,
    ) -> np.ndarray:
        """Compatibility wrapper for the shared restricted residual evaluator."""
        names = self._validate_restricted_model(self.garnet_endmember_names)
        free = tuple(name for name in names if name != self.kho_name)
        vector = np.asarray(fractions, dtype=float)
        if vector.shape != (len(free),):
            raise ValueError("Restricted MDF fractions must match the divalent simplex")
        model_fractions = {name: float(value) for name, value in zip(free, vector)}
        if self.kho_name in names:
            model_fractions[self.kho_name] = 0.0
        targets = self._restricted_exchange(gamma)
        return self._evaluate_mdf_residual(
            model_fractions,
            targets,
            pressure_kbar,
            temperature_c,
            names=names,
            free_names=free,
            reference=self.cation_component_names["pyrope"],
            fixed_fractions={self.kho_name: 0.0} if self.kho_name in names else {},
        )

    def _restricted_exchange(self, gamma: Mapping[str, float]) -> dict[str, float]:
        """Project divalent oxide Gamma onto restricted garnet exchanges."""
        required = {"MgO", "FeO", "CaO"}
        roles = self.cation_component_names
        if "spessartine" in roles:
            required.add("MnO")
        missing = required.difference(gamma)
        if missing:
            raise ValueError(
                f"Restricted MDF matrix Gamma is missing required oxide(s): {sorted(missing)}"
            )
        result = {
            roles["almandine"]: 3.0 * (float(gamma["FeO"]) - float(gamma["MgO"])),
            roles["grossular"]: 3.0 * (float(gamma["CaO"]) - float(gamma["MgO"])),
        }
        if "spessartine" in roles:
            result[roles["spessartine"]] = 3.0 * (
                float(gamma["MnO"]) - float(gamma["MgO"])
            )
        return result

    def solve_restricted_mdf(
        self,
        gamma: Mapping[str, float],
        pressure_kbar: float,
        temperature_c: float,
        *,
        tolerance_kj: float = 0.03,
        max_iterations: int = 300,
    ) -> tuple[dict[str, float], tuple[float, ...], float, bool, str]:
        """Solve divalent MDF on the garnet simplex with kho fixed exactly zero."""
        names = self._validate_restricted_model(self.garnet_endmember_names)
        free = tuple(name for name in names if name != self.kho_name)
        return self._solve_mdf_on_simplex(
            gamma,
            pressure_kbar,
            temperature_c,
            names=names,
            free_names=free,
            fixed_fractions={self.kho_name: 0.0} if self.kho_name in names else {},
            targets=self._restricted_exchange(gamma),
            reference=self.cation_component_names["pyrope"],
            tolerance_kj=tolerance_kj,
            max_iterations=max_iterations,
        )

    def _active_exchange(self, gamma: Mapping[str, float]) -> np.ndarray:
        names = self._validate_active_model(self.garnet_endmember_names)
        potentials = kho_aware_exchange_potentials(
            gamma,
            self.cation_component_names,
            kho_name=self.kho_name,
        )
        reference = self.cation_component_names["pyrope"]
        return np.asarray(
            [potentials[name] for name in names if name != reference], dtype=float
        )

    def _active_exchange_map(self, gamma: Mapping[str, float]) -> dict[str, float]:
        """Return full named exchange potentials, including the kho equation."""
        return kho_aware_exchange_potentials(
            gamma, self.cation_component_names, kho_name=self.kho_name
        )

    def _simplex(self, logits: Sequence[float], n_components: int) -> np.ndarray:
        vector = np.asarray(logits, dtype=float)
        if vector.shape != (n_components - 1,):
            raise ValueError("simplex logits have an invalid size")
        extended = np.concatenate((vector, [0.0]))
        extended -= np.max(extended)
        values = np.exp(extended)
        values = np.maximum(values, _FRACTION_FLOOR)
        return values / np.sum(values)

    def _mdf_residual(
        self,
        fractions: Sequence[float],
        reaction: np.ndarray,
        pressure_kbar: float,
        temperature_c: float,
    ) -> np.ndarray:
        names = self._validate_active_model(self.garnet_endmember_names)
        vector = np.asarray(fractions, dtype=float)
        if vector.shape != (len(names),):
            raise ValueError("MDF fractions must match the active garnet simplex")
        target_values = np.asarray(reaction, dtype=float)
        if target_values.shape != (len(names) - 1,):
            raise RuntimeError("Garnet and matrix exchange vectors are misaligned")
        reference = self.cation_component_names["pyrope"]
        targets = dict(
            zip(
                (name for name in names if name != reference),
                target_values,
                strict=True,
            )
        )
        return self._evaluate_mdf_residual(
            dict(zip(names, map(float, vector), strict=True)),
            targets,
            pressure_kbar,
            temperature_c,
            names=names,
            free_names=names,
            reference=reference,
            fixed_fractions={},
        )

    def solve_kho_aware_mdf(
        self,
        gamma: Mapping[str, float],
        pressure_kbar: float,
        temperature_c: float,
        *,
        tolerance_kj: float = 0.03,
        max_iterations: int = 300,
    ) -> tuple[dict[str, float], tuple[float, ...], float, bool, str]:
        """Solve all discovered component exchanges on the full supported simplex.

        Convergence is determined by the Euclidean norm of all N-1 exchange
        residuals. The default 0.03 kJ/mol formula tolerance is deliberately
        explicit: live dataset-636 equilibrium comparisons show solver noise
        up to roughly 0.02 kJ/mol formula.
        """
        names = self._validate_active_model(self.garnet_endmember_names)
        reference = self.cation_component_names["pyrope"]
        return self._solve_mdf_on_simplex(
            gamma,
            pressure_kbar,
            temperature_c,
            names=names,
            free_names=names,
            fixed_fractions={},
            targets=self._active_exchange_map(gamma),
            reference=reference,
            tolerance_kj=tolerance_kj,
            max_iterations=max_iterations,
        )

    @staticmethod
    def _read_equilibrium_endmember_fractions(output: object) -> dict[str, float]:
        """Read the active garnet phase's molar endmember fractions."""
        indices = [i for i, phase in enumerate(output.ph) if str(phase) == "g"]
        if not indices:
            raise RuntimeError("Garnet is not stable in the ordinary equilibrium")
        solution = output.SS_vec[indices[0]]
        names = tuple(str(name) for name in solution.emNames)
        values = np.asarray(solution.emFrac, dtype=float)
        if len(names) != values.size or len(set(names)) != len(names):
            raise RuntimeError("MAGEMin garnet emNames/emFrac are inconsistent")
        if not np.all(np.isfinite(values)) or np.any(values < -1.0e-12):
            raise RuntimeError("MAGEMin garnet fractions are invalid")
        if float(np.maximum(values, 0.0).sum()) <= 0.0:
            raise RuntimeError("MAGEMin garnet fractions have no positive total")
        # Preserve MAGEMin's values exactly for audit. The public cation
        # projection closes independently over the represented cations.
        return dict(zip(names, map(float, values)))

    def calculate_compositions(
        self,
        pressure_kbar: float,
        temperature_c: float,
        *,
        fe_basis: str = "FeOt",
        tolerance_kj: float = 0.03,
        max_iterations: int = 300,
    ) -> GarnetMDFResult:
        """Calculate paired MDF and equilibrium closed cation proportions.

        Parameters
        ----------
        pressure_kbar, temperature_c : float
            Conditions in kbar and °C.
        fe_basis : {'FeOt', 'Fe2+'}, default='FeOt'
            Cation-projection basis. Standardised molar O controls solver
            dispatch: zero/absent O uses restricted MDF; positive O requires
            the active model's kho component and uses kho-aware MDF.
        tolerance_kj : float, default=0.03
            Euclidean MDF exchange-residual convergence threshold in kJ/mol
            garnet formula.
        max_iterations : int, default=300
            Maximum BFGS iterations per start.

        Returns
        -------
        GarnetMDFResult
            Four-cation MDF and ordinary-equilibrium compositions plus MDF
            residual and matrix/equilibrium phase diagnostics. ``solver_mode``
            is exactly ``"restricted"`` or ``"full"``.
        """
        basis = normalise_fe_basis(fe_basis)
        self._require_configured_bulk()
        standardized_oxygen = self._standardized_bulk_oxygen()
        solver_mode = "restricted" if standardized_oxygen == 0.0 else "full"
        equilibrium_output = self.calculate_equilibrium(pressure_kbar, temperature_c)
        if "g" not in tuple(str(phase) for phase in equilibrium_output.ph):
            raise ValueError(
                "Garnet ('g') is absent from the unrestricted equilibrium at "
                f"P={pressure_kbar:g} kbar, T={temperature_c:g} °C; garnet "
                "stability is required before validating the active model."
            )
        if solver_mode == "restricted":
            names = self._validate_restricted_model(self.garnet_endmember_names)
        else:
            if self.kho_name not in self.garnet_endmember_names:
                raise ValueError(
                    "Positive standardised bulk O requires an active kho garnet "
                    f"component; discovered {self.garnet_endmember_names}"
                )
            names = self._validate_active_model(self.garnet_endmember_names)
        required_bulk_bases = {"MgO", "FeO"}
        active_roles = {
            name: role for role, name in self.cation_component_names.items()
        }
        if any(active_roles.get(name) == "grossular" for name in names):
            required_bulk_bases.add("CaO")
        if any(active_roles.get(name) == "spessartine" for name in names):
            required_bulk_bases.add("MnO")
        if solver_mode == "full":
            required_bulk_bases.update(("O", "Al2O3"))
        missing_bulk_bases = required_bulk_bases.difference(
            self._standardized_bulk_molar_oxides
        )
        if missing_bulk_bases:
            raise ValueError(
                f"Standardised bulk is missing required {solver_mode} MDF oxide "
                f"basis component(s): {sorted(missing_bulk_bases)}"
            )
        equilibrium_fractions = self._read_equilibrium_endmember_fractions(
            equilibrium_output
        )
        if set(equilibrium_fractions) != set(names):
            raise RuntimeError("Equilibrium garnet endmembers changed after discovery")
        projection_fractions = {
            name: max(float(value), 0.0)
            for name, value in equilibrium_fractions.items()
        }
        equilibrium = garnet_fractions_to_mg_fe_mn_ca(
            projection_fractions,
            component_names=self.cation_component_names,
            kho_name=self.kho_name,
            fe_basis=basis,
        )
        _matrix_output, gamma, matrix_phases = self.calculate_matrix_state(
            pressure_kbar, temperature_c
        )
        if solver_mode == "full":
            missing_kho_oxides = {"FeO", "O", "Al2O3"}.difference(gamma)
            if missing_kho_oxides:
                raise ValueError(
                    "Positive standardised bulk O requires matrix Gamma for FeO, "
                    "O and Al2O3; missing "
                    f"{sorted(missing_kho_oxides)}"
                )
            fractions, residual, norm, converged, message = self.solve_kho_aware_mdf(
                gamma,
                pressure_kbar,
                temperature_c,
                tolerance_kj=tolerance_kj,
                max_iterations=max_iterations,
            )
        else:
            fractions, residual, norm, converged, message = self.solve_restricted_mdf(
                gamma,
                pressure_kbar,
                temperature_c,
                tolerance_kj=tolerance_kj,
                max_iterations=max_iterations,
            )
        mdf = garnet_fractions_to_mg_fe_mn_ca(
            fractions,
            component_names=self.cation_component_names,
            kho_name=self.kho_name,
            fe_basis=basis,
        )
        return GarnetMDFResult(
            mdf=mdf,
            equilibrium=equilibrium,
            residual_kj_per_formula=residual,
            residual_norm_kj_per_formula=norm,
            converged=converged,
            status="success" if converged else "no_convergence",
            matrix_phases=tuple(matrix_phases),
            equilibrium_phases=tuple(str(phase) for phase in equilibrium_output.ph),
            active_endmember_names=names,
            _mdf_endmember_fractions=fractions,
            _equilibrium_endmember_fractions=equilibrium_fractions,
            tolerance_kj=float(tolerance_kj),
            message=message,
            fe_basis=basis,
            solver_mode=solver_mode,
        )

    def scan_isopleths(
        self,
        P_range: tuple[float, float],
        T_range: tuple[float, float],
        *,
        n_P: int = 20,
        n_T: int = 20,
        fe_basis: str = "FeOt",
        tolerance_kj: float = 0.03,
        max_iterations: int = 300,
    ) -> GarnetMDFGridResult:
        """Scan paired MDF and equilibrium closed cation compositions.

        Failed cells contain NaNs in both composition mappings and the
        exception type/message in ``status``; successful but non-converged MDF
        cells retain their finite composition and report ``no_convergence``.
        The selected ``fe_basis`` is applied to both projections and recorded
        in the result, even when individual grid cells fail.
        """
        basis = normalise_fe_basis(fe_basis)
        self._require_configured_bulk()
        if not isinstance(n_P, (int, np.integer)) or isinstance(n_P, bool) or n_P < 1:
            raise ValueError("n_P must be a positive integer")
        if not isinstance(n_T, (int, np.integer)) or isinstance(n_T, bool) or n_T < 1:
            raise ValueError("n_T must be a positive integer")
        p_limits = tuple(float(value) for value in P_range)
        t_limits = tuple(float(value) for value in T_range)
        if (
            len(p_limits) != 2
            or len(t_limits) != 2
            or not np.all(np.isfinite((*p_limits, *t_limits)))
            or p_limits[0] > p_limits[1]
            or t_limits[0] > t_limits[1]
        ):
            raise ValueError("P_range and T_range must be finite ascending pairs")
        pressure, temperature = np.meshgrid(
            np.linspace(*p_limits, int(n_P)),
            np.linspace(*t_limits, int(n_T)),
            indexing="xy",
        )
        shape = pressure.shape
        mdf = {key: np.full(shape, np.nan) for key in _CATION_KEYS}
        equilibrium = {key: np.full(shape, np.nan) for key in _CATION_KEYS}
        norms = np.full(shape, np.nan)
        converged = np.zeros(shape, dtype=bool)
        status = np.full(shape, "not_run", dtype=object)
        active_endmember_names = np.empty(shape, dtype=object)
        active_endmember_names.fill(())
        matrix_assemblages = np.empty(shape, dtype=object)
        matrix_assemblages.fill(())
        solver_mode = np.full(shape, "not_run", dtype=object)
        for index in np.ndindex(shape):
            try:
                result = self.calculate_compositions(
                    float(pressure[index]),
                    float(temperature[index]),
                    fe_basis=basis,
                    tolerance_kj=tolerance_kj,
                    max_iterations=max_iterations,
                )
            except Exception as exc:  # noqa: BLE001 — one failed point must not abort a scan.
                status[index] = f"{type(exc).__name__}: {exc}"
                continue
            for key in _CATION_KEYS:
                mdf[key][index] = result.mdf[key]
                equilibrium[key][index] = result.equilibrium[key]
            norms[index] = result.residual_norm_kj_per_formula
            converged[index] = result.converged
            status[index] = result.status
            solver_mode[index] = result.solver_mode
            active_endmember_names[index] = result.active_endmember_names
            matrix_assemblages[index] = result.matrix_phases
        return GarnetMDFGridResult(
            pressure,
            temperature,
            mdf,
            equilibrium,
            norms,
            converged,
            status,
            active_endmember_names,
            matrix_assemblages,
            basis,
            solver_mode,
        )


__all__ = [
    "GarnetMDFCalculator",
    "GarnetMDFGridResult",
    "GarnetMDFResult",
    "garnet_fractions_to_mg_fe_mn_ca",
    "kho_aware_exchange_potentials",
]
