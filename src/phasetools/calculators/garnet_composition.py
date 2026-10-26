"""Shared garnet cation-basis conversions for growth and MDF workflows."""

from collections.abc import Mapping

import numpy as np

from ..core.phase_properties import get_oxide_apfu, get_phase_fe_split

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


__all__ = [
    "garnet_cations_from_apfu",
    "normalise_fe_basis",
    "normalise_garnet_cations",
]
