"""Garnet composition maths: cation-basis conversions, closed Mg-Fe-Mn-Ca
projections, and MDF endmember/oxide mixing for growth workflows."""

import re
from collections.abc import Mapping, Sequence

import numpy as np

from ...core.phase_properties import get_oxide_apfu, get_phase_fe_split
from ...utils.bulk_rock import atomic_mass_dict

_CATION_KEYS = ("Mg", "Fe", "Mn", "Ca")


def normalise_fe_basis(fe_basis: str) -> str:
    """Validate and canonicalise the supported garnet iron basis.

    Parameters
    ----------
    fe_basis : str
        Total iron (``'FeOt'``) or ferrous iron only (``'Fe2+'``). The
        case-insensitive alias ``'Fe2'`` is also accepted.

    Returns
    -------
    str
        Canonical basis name: ``'FeOt'`` or ``'Fe2+'``.

    Raises
    ------
    ValueError
        If the basis is unsupported.
    """
    basis = str(fe_basis).casefold()
    if basis == "feot":
        return "FeOt"
    if basis in ("fe2+", "fe2"):
        return "Fe2+"
    raise ValueError(f"fe_basis must be 'FeOt' or 'Fe2+', got {fe_basis!r}")


def normalise_garnet_cations(
    amounts: Mapping[str, float], *, absent: str = "raise"
) -> dict[str, float]:
    """Close Mg-Fe-Mn-Ca amounts to atomic proportions in a fixed key order.

    This is a closed four-cation projection, not an APFU or site-occupancy
    normalisation. Amounts for unlisted cations are ignored.

    Parameters
    ----------
    amounts : mapping
        Atomic amounts keyed by ``Mg``, ``Fe``, ``Mn`` and ``Ca``.
    absent : {'raise', 'zero'}, default='raise'
        Behaviour when the represented cation total is zero. ``'zero'``
        returns four zero values; ``'raise'`` raises :class:`ValueError`.

    Returns
    -------
    dict[str, float]
        Closed proportions ordered as Mg, Fe, Mn, Ca.

    Raises
    ------
    ValueError
        If ``absent`` is unsupported, an amount is negative or non-finite,
        or the total is zero with ``absent='raise'``.
    TypeError
        If ``amounts`` is not a mapping.
    """
    if not isinstance(amounts, Mapping):
        raise TypeError("amounts must be a mapping of cation names to amounts")
    if absent not in ("raise", "zero"):
        raise ValueError("absent must be 'raise' or 'zero'")

    values = {key: float(amounts.get(key, 0.0)) for key in _CATION_KEYS}
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("garnet cation amounts must be finite and non-negative")
    total = float(sum(values.values()))
    if not np.isfinite(total):
        raise ValueError("garnet cation total must be finite")
    if total <= 0.0:
        if absent == "zero":
            return {key: 0.0 for key in _CATION_KEYS}
        raise ValueError("garnet has no positive Mg-Fe-Mn-Ca cation total")
    return {key: values[key] / total for key in _CATION_KEYS}


def garnet_cations_from_apfu(
    output: object,
    fe_basis: str = "FeOt",
    *,
    absent: str = "raise",
) -> dict[str, float]:
    """Project garnet oxide APFU onto closed Mg-Fe-Mn-Ca atomic proportions.

    For ``FeOt``, APFU Fe is ``FeO + 2*Fe2O3``. For ``Fe2+``, Fe is taken
    from :func:`get_phase_fe_split`. The result closes only these four cations
    and is not APFU or an X-site normalisation.

    Parameters
    ----------
    output : object
        MAGEMin output containing phase ``'g'``.
    fe_basis : str, default='FeOt'
        Total-iron or ferrous-only basis; case-insensitive ``'Fe2'`` alias
        accepted.
    absent : {'raise', 'zero'}, default='raise'
        Behaviour when garnet is absent or its represented cation total is
        zero.

    Returns
    -------
    dict[str, float]
        Closed proportions ordered as Mg, Fe, Mn, Ca.
    """
    basis = normalise_fe_basis(fe_basis)
    if absent not in ("raise", "zero"):
        raise ValueError("absent must be 'raise' or 'zero'")
    if "g" not in tuple(str(phase) for phase in getattr(output, "ph", ())):
        if absent == "zero":
            return {key: 0.0 for key in _CATION_KEYS}
        raise ValueError("garnet ('g') is absent from the MAGEMin output")

    apfu = get_oxide_apfu(output, "g", ["MgO", "FeO", "Fe2O3", "MnO", "CaO"])
    fe = (
        float(apfu.get("FeO", 0.0)) + 2.0 * float(apfu.get("Fe2O3", 0.0))
        if basis == "FeOt"
        else float(get_phase_fe_split(output, "g")["Fe2"])
    )
    amounts = {
        "Mg": float(apfu.get("MgO", 0.0)),
        "Fe": fe,
        "Mn": float(apfu.get("MnO", 0.0)),
        "Ca": float(apfu.get("CaO", 0.0)),
    }
    return normalise_garnet_cations(amounts, absent=absent)



_DEFAULT_COMPONENTS = {
    "pyrope": "py",
    "almandine": "alm",
    "spessartine": "spss",
    "grossular": "gr",
}

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


def garnet_endmember_fractions_to_oxide_wt(
    fractions: Mapping[str, float],
    endmember_names: Sequence[str],
    em_comp_wt: np.ndarray,
    em_comp_apfu: np.ndarray,
    oxide_names: Sequence[str],
) -> np.ndarray:
    """Mix garnet endmember fractions into a full oxide weight-fraction vector.

    Reconstructs the complete garnet oxide chemistry — every oxide the
    solution model carries (SiO₂, Al₂O₃, MgO, FeO, MnO, CaO, and any
    component O from ferric kho) — by mass-weighted mixing of MAGEMin's
    per-endmember compositions.  The recipe is verified against
    MAGEMin's own ``Comp_wt`` for equilibrium fractions to machine
    precision: molar fractions are converted to mass fractions via
    endmember molar masses (from ``em_comp_apfu`` atom counts × atomic
    masses) *before* mixing, because normalised endmember rows carry
    different numbers of oxide molecules per formula unit and do not
    mix linearly on a molar basis.

    Parameters
    ----------
    fractions : mapping
        Endmember name → molar fraction (e.g. MDF solution or
        equilibrium ``emFrac``).
    endmember_names : sequence of str
        Names ordering the rows of ``em_comp_wt``/``em_comp_apfu``.
    em_comp_wt : numpy.ndarray
        Per-endmember oxide composition, weight fractions, shape
        ``(n_endmembers, n_oxides)``; rows sum to 1 (MAGEMin
        ``SS_vec[i].emComp_wt``).
    em_comp_apfu : numpy.ndarray
        Per-endmember atom counts per formula unit, same shape
        (MAGEMin ``SS_vec[i].emComp_apfu``; oxide columns hold atoms of
        that oxide's element — e.g. Al₂O₃ column = 2 for pyrope —
        plus a total-O column, giving exact endmember molar masses).
    oxide_names : sequence of str
        Oxide order of the output vector (the calculator's standardised
        bulk oxide list).

    Returns
    -------
    numpy.ndarray
        Unit-sum oxide weight fractions aligned to ``oxide_names``.

    Raises
    ------
    ValueError
        If shapes are inconsistent, an endmember name is unknown, the
        molar fractions are invalid, or any molar mass is non-positive.
    """
    names = tuple(str(name) for name in endmember_names)
    oxides = tuple(str(oxide) for oxide in oxide_names)
    comp_wt = np.asarray(em_comp_wt, dtype=float)
    apfu = np.asarray(em_comp_apfu, dtype=float)
    n_em, n_ox = len(names), len(oxides)
    if comp_wt.shape != (n_em, n_ox) or apfu.shape != (n_em, n_ox):
        raise ValueError(
            f"endmember composition matrices must have shape "
            f"({n_em}, {n_ox}); got {comp_wt.shape} and {apfu.shape}"
        )
    unknown = [key for key in fractions if str(key) not in names]
    if unknown:
        raise ValueError(
            f"Unknown garnet endmember(s) {sorted(unknown)}; "
            f"available names are {sorted(names)}"
        )
    f = np.array([float(fractions.get(name, 0.0)) for name in names])
    if not np.all(np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError("endmember fractions must be finite and non-negative")

    # Endmember molar masses: atom counts × atomic mass of each column's
    # element (the leading element of the oxide formula).
    masses = np.empty(n_ox)
    for index, oxide in enumerate(oxides):
        match = re.match(r"([A-Z][a-z]?)", oxide)
        element = match.group(1) if match else None
        if element is None or element not in atomic_mass_dict:
            raise ValueError(f"Cannot derive an atomic mass for oxide {oxide!r}")
        masses[index] = atomic_mass_dict[element]
    endmember_mw = apfu @ masses
    if not np.all(np.isfinite(endmember_mw)) or np.any(endmember_mw <= 0.0):
        raise ValueError(
            "Endmember molar masses must be positive; check emComp_apfu rows"
        )

    weights = f * endmember_mw
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Endmember fractions have no positive mass total")
    weights = weights / total

    oxide_wt = weights @ comp_wt
    oxide_sum = oxide_wt.sum()
    if not np.isfinite(oxide_sum) or oxide_sum <= 0.0:
        raise ValueError("Mixed oxide composition has no positive total")
    return oxide_wt / oxide_sum


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


__all__ = [
    "garnet_cations_from_apfu",
    "garnet_endmember_fractions_to_oxide_wt",
    "garnet_fractions_to_mg_fe_mn_ca",
    "kho_aware_exchange_potentials",
    "normalise_fe_basis",
    "normalise_garnet_cations",
]
