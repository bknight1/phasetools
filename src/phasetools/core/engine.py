"""Low-level Python-to-Julia bridge helpers for MAGEMin."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from juliacall import Main as jl
from juliacall import convert as jlconvert

from phasetools import MAGEMin_C


@dataclass(frozen=True)
class SolutionPhaseGibbsEvaluation:
    """Gibbs energy and composition derivatives for a solution-phase state.

    Attributes
    ----------
    phase : str
        MAGEMin solution-phase key.
    fractions : Mapping[str, float]
        Normalised endmember fractions passed to MAGEMin.
    df : float
        Gibbs-energy value returned by MAGEMin [kJ/mol].
    mu : tuple of float
        Endmember chemical-potential values exposed by MAGEMin.
    dfx : tuple of float
        Derivatives exposed for the phase composition variables.
    endmember_names : tuple of str
        Optional endmember names corresponding to the order of ``mu`` and
        ``dfx``. These are supplied by the caller from MAGEMin output because
        the generic Julia bridge does not expose database model labels itself.
    valid : bool
        Whether MAGEMin reports ``sf_ok == 1`` for this composition.
    """

    phase: str
    fractions: Mapping[str, float]
    df: float
    mu: tuple[float, ...]
    dfx: tuple[float, ...]
    valid: bool
    endmember_names: tuple[str, ...] = ()


def _julia_field(value: Any, name: str) -> Any:
    """Read a named field from a Julia struct through juliacall explicitly."""
    return jl.getproperty(value, jl.Symbol(name))


def _julia_float_vector(pointer: Any, length: int) -> tuple[float, ...]:
    """Copy a Julia pointer-backed float vector into owned Python values."""
    if length <= 0:
        return ()
    # MAGEMin's `mu` and `dfx` fields are Ptr{Float64} values (rather than
    # Julia Vector fields); unsafe_wrap is therefore the matching non-owning
    # view, and tuple conversion below copies the data before it escapes.
    values = jl.unsafe_wrap(jl.Vector[jl.Float64], pointer, length, own=False)
    return tuple(float(value) for value in values)


def _validate_solution_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    """Validate a named, normalised set of solution endmember fractions."""
    if not isinstance(fractions, Mapping) or not fractions:
        raise ValueError("fractions must be a non-empty mapping of endmember names to values")
    ordered = {}
    for name, value in fractions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("endmember names must be non-empty strings")
        ordered[name] = float(value)
    if any(not np.isfinite(value) or value < 0.0 for value in ordered.values()):
        raise ValueError("Endmember fractions must be finite and non-negative")
    if not np.isclose(sum(ordered.values()), 1.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("Endmember fractions must sum to 1")
    return ordered


def evaluate_solution_phase_gibbs(
    calculator: Any,
    phase: str,
    fractions: Mapping[str, float],
    pressure_kbar: float,
    temperature_c: float,
    *,
    endmember_names: Sequence[str] | None = None,
) -> SolutionPhaseGibbsEvaluation:
    """Evaluate a supplied solution-phase composition using MAGEMin.

    The calculator must have a bulk composition configured with
    ``setup_bulk_composition``. MAGEMin's reference-energy initialisation and
    ``p2x_convert`` / ``pc_convert`` routines are called through phasetools'
    existing Julia bridge. This is a low-level thermodynamic evaluation; it
    does not perform an equilibrium minimisation or an inversion.

    Parameters
    ----------
    calculator : MAGEMinBase
        Initialised calculator with the desired database and bulk composition.
    phase : str
        MAGEMin solution-phase key, such as ``"g"`` for garnet.
    fractions : mapping of str to float
        Named endmember fractions. Values must be finite, non-negative, and
        sum to one.
    pressure_kbar, temperature_c : float
        Evaluation conditions in kbar and degrees Celsius.
    endmember_names : sequence of str, optional
        Names in MAGEMin's output order for this solution model. If provided,
        the count must match MAGEMin's returned endmember chemical potentials.

    Returns
    -------
    SolutionPhaseGibbsEvaluation
        Python values returned by MAGEMin for the requested phase state.
    """
    if MAGEMin_C is None or jl is None or jlconvert is None:
        raise RuntimeError("Julia/MAGEMin_C is unavailable in this environment")
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be a non-empty MAGEMin phase key")
    if calculator.X is None or calculator.original_X is None:
        raise ValueError("Call setup_bulk_composition before evaluating a solution phase")
    ordered = _validate_solution_fractions(fractions)
    data = calculator.data
    gv = _julia_field(data, "gv")[0]
    z_b = _julia_field(data, "z_b")[0]
    database = _julia_field(data, "DB")[0]
    simplex_data = _julia_field(data, "splx_data")[0]

    bulk = jlconvert(jl.Vector[jl.Float64], calculator.original_X)
    oxides = jlconvert(jl.Vector[jl.String], calculator.original_Xoxides)
    gv = MAGEMin_C.define_bulk_rock(
        gv, bulk, oxides, calculator.original_sys_in, calculator.db
    )
    gv, z_b, database, simplex_data = MAGEMin_C.pwm_init(
        float(pressure_kbar), float(temperature_c), gv, z_b, database, simplex_data
    )
    fraction_dict = jlconvert(jl.Dict[jl.String, jl.Float64], ordered)
    phase_state = MAGEMin_C.p2x_convert(gv, database, phase, fraction_dict)
    phase_state = MAGEMin_C.pc_convert(gv, z_b, database, phase, phase_state)

    df = float(_julia_field(phase_state, "df"))
    mu = _julia_float_vector(
        _julia_field(phase_state, "mu"), int(_julia_field(phase_state, "n_em"))
    )
    dfx = _julia_float_vector(
        _julia_field(phase_state, "dfx"), int(_julia_field(phase_state, "n_xeos"))
    )
    valid = int(_julia_field(phase_state, "sf_ok")) == 1
    names = tuple(str(name) for name in endmember_names) if endmember_names is not None else ()
    if endmember_names is not None and (
        len(names) != len(mu) or len(set(names)) != len(names)
    ):
        raise ValueError(
            "endmember_names must be unique and match the returned chemical-potential count"
        )
    return SolutionPhaseGibbsEvaluation(phase, ordered, df, mu, dfx, valid, names)


def single_point_minimization_with_conversion(P, T, data, X, Xoxides, sys_in="wt", rm_list=None):
    """
    Perform single-point minimisation with automatic Python-to-Julia conversion, including optional removal list.

    Parameters
    ----------
    P : float
        Pressure in kbar.
    T : float
        Temperature in Celsius.
    data : object
        Initialised MAGEMin data object.
    X : list[float]
        Bulk composition values.
    Xoxides : list[str]
        Bulk composition oxide names.
    sys_in : str, optional
        Input system basis ("wt", "mol", etc.), by default "wt".
    rm_list : list[str], optional
        List of phases to remove, by default None.

    Returns
    -------
    object
        MAGEMin output object for the given P-T point.
    """
    # Convert Python inputs to Julia-compatible types
    P_jl = jlconvert(jl.Float64, P)
    T_jl = jlconvert(jl.Float64, T)
    X_jl = jlconvert(jl.Vector[jl.Float64], X)
    Xox_jl = jlconvert(jl.Vector[jl.String], Xoxides)

    # Convert rm_list if provided
    if rm_list is not None:
        rm_list_conv = jlconvert(jl.Vector[jl.String], rm_list)
        rm_list_jl = MAGEMin_C.remove_phases(rm_list_conv, data.db)
    else:
        rm_list_jl = None

    # Call MAGEMin single_point_minimization
    result = MAGEMin_C.single_point_minimization(P_jl, T_jl, data, X=X_jl, Xoxides=Xox_jl, sys_in=sys_in, rm_list=rm_list_jl)

    return result
