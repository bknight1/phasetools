"""Generic extraction of thermodynamic diagnostics from MAGEMin outputs."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from phasetools import MAGEMin_C


@dataclass(frozen=True)
class StablePhaseGibbs:
    """Gibbs energy reported for one stable phase entry."""

    phase: str
    gibbs_energy: float


@dataclass(frozen=True)
class CandidatePhaseGibbs:
    """Gibbs data for one candidate or metastable phase entry."""

    phase: str
    gibbs_energy: float | None = None
    delta_g: float | None = None
    phase_id: int | None = None
    endmember_id: int | None = None


@dataclass(frozen=True)
class ThermodynamicSnapshot:
    """Plain-Python thermodynamic data from one MAGEMin calculation.

    ``gamma`` is ordered as returned by MAGEMin. ``gamma_oxides`` records
    corresponding oxide names if provided; the vector is not transformed.
    Fields absent from an output structure become ``None`` or empty tuples.
    """

    gamma: tuple[float, ...] | None = None
    gamma_oxides: tuple[str, ...] = ()
    system_gibbs_energy: float | None = None
    stable_phase_gibbs: tuple[StablePhaseGibbs, ...] = ()
    candidate_phase_gibbs: tuple[CandidatePhaseGibbs, ...] = ()


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from a mapping or Julia/Python object."""
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    try:
        return getattr(value, name)
    except (AttributeError, KeyError, TypeError):
        return default


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _phase_name(entry: Any, fallback: str = "") -> str:
    name = _field(entry, "ph_name")
    return str(name) if name is not None else fallback


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def extract_gamma(output: Any) -> tuple[float, ...] | None:
    """Extract the oxide-potential (``Gamma``) vector as Python floats.

    Falls back to ``MAGEMin_C.equilibrium_state(output).gam`` when the direct
    output does not include ``Gamma``.
    """
    gamma = _field(output, "Gamma")
    if gamma is None and MAGEMin_C is not None:
        try:
            gamma = _field(MAGEMin_C.equilibrium_state(output), "gam")
        except Exception:  # JuliaCall may wrap bridge exceptions as JuliaError
            gamma = None
    if gamma is None:
        return None
    try:
        return tuple(float(value) for value in gamma)
    except (TypeError, ValueError):
        return None


def extract_system_gibbs(output: Any) -> float | None:
    """Extract system Gibbs energy (``G_system``), if reported."""
    return _float_or_none(_field(output, "G_system"))


def extract_stable_phase_gibbs(output: Any) -> tuple[StablePhaseGibbs, ...]:
    """Extract Gibbs energies from stable solution and pure-phase entries."""
    records: list[StablePhaseGibbs] = []
    stable_names = _as_list(_field(output, "ph", ()))
    solution_entries = _as_list(_field(output, "SS_vec", ()))
    solution_count = len(solution_entries)
    for vector_name, offset in (("SS_vec", 0), ("PP_vec", solution_count)):
        for index, entry in enumerate(_as_list(_field(output, vector_name, ()))):
            energy = _float_or_none(_field(entry, "G"))
            if energy is None:
                continue
            name_index = index + offset
            fallback = str(stable_names[name_index]) if name_index < len(stable_names) else ""
            records.append(StablePhaseGibbs(_phase_name(entry, fallback), energy))
    return tuple(records)


def extract_candidate_phase_gibbs(output: Any) -> tuple[CandidatePhaseGibbs, ...]:
    """Extract candidate/metastable phase Gibbs and ``deltaG`` values."""
    records = []
    for entry in _as_list(_field(output, "mSS_vec", ())):
        energy = _float_or_none(_field(entry, "G"))
        delta_g = _float_or_none(_field(entry, "deltaG"))
        if energy is None and delta_g is None:
            continue
        records.append(CandidatePhaseGibbs(
            phase=_phase_name(entry),
            gibbs_energy=energy,
            delta_g=delta_g,
            phase_id=_int_or_none(_field(entry, "ph_id")),
            endmember_id=_int_or_none(_field(entry, "em_id")),
        ))
    return tuple(records)


def extract_thermodynamic_snapshot(
    output: Any,
    oxide_names: Sequence[str] = (),
) -> ThermodynamicSnapshot:
    """Extract available thermodynamic fields from a raw MAGEMin output."""
    return ThermodynamicSnapshot(
        gamma=extract_gamma(output),
        gamma_oxides=tuple(str(name) for name in oxide_names),
        system_gibbs_energy=extract_system_gibbs(output),
        stable_phase_gibbs=extract_stable_phase_gibbs(output),
        candidate_phase_gibbs=extract_candidate_phase_gibbs(output),
    )
