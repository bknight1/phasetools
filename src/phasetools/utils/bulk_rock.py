import numpy as np
from molmass import Formula

''' Molecular weights of oxides and elements '''

ref_ox = [
    "SiO2", "TiO2", "Al2O3", "FeO", "Fe2O3", "MnO", "MgO", "CaO", "Na2O", "K2O",
    "Cr2O3", "NiO", "P2O5", "H2O", "CO2", "O", "O2", "S", "SO3", "Cl", "F"
]

ref_elements = [
    "Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K", "Cr", "Ni", "P",
    "H", "C", "O", "S", "Cl", "F"
]

class FormulaMassDict(dict):
    """Dictionary that lazily resolves unknown component masses using molmass."""
    def __missing__(self, key):
        mass = Formula(key).mass
        self[key] = mass
        return mass

def _build_mass_dict(components):
    return FormulaMassDict({comp: Formula(comp).mass for comp in components})

molar_mass_dict = _build_mass_dict(ref_ox)
atomic_mass_dict = _build_mass_dict(ref_elements)

def get_molar_mass_dict():
    return molar_mass_dict

def get_atomic_mass_dict():
    return atomic_mass_dict

def convert_oxide_to_element_moles(oxide_moles, oxide, element):
    molar_mass_dict = get_molar_mass_dict()
    return oxide_moles * (molar_mass_dict[element] / molar_mass_dict[oxide])

def convert_mol_percent_to_wt_percent(mol_percents, components, mass_dict):
    """Generic conversion from mole percent to weight (mass) percent."""
    total_mass = 0
    for comp, mol in zip(components, mol_percents):
        total_mass += mol * mass_dict[comp]
    wt_percents = [(mol * mass_dict[comp] / total_mass) * 100 for comp, mol in zip(components, mol_percents)]
    return wt_percents

def atomic_frac_to_wt_frac(atomic_frac: dict[str, float], mass_dict: dict[str, float]) -> dict[str, float]:
    """Convert atomic (molar) site fractions to weight-based site fractions.

    Parameters
    ----------
    atomic_frac : dict
        Mapping of component names to their atomic fractions (summing to 1.0).
    mass_dict : dict
        Mapping of component names to their atomic/molecular masses.

    Returns
    -------
    dict
        Weight-based fractions (summing to 1.0).
    """
    total_mass = sum(atomic_frac[k] * mass_dict[k] for k in atomic_frac)
    return {k: atomic_frac[k] * mass_dict[k] / total_mass for k in atomic_frac}


def convert_wt_percent_to_mol_percent(wt_percents, components, mass_dict):
    """Generic conversion from weight (mass) percent to mole percent."""
    total_moles = 0
    for comp, wt in zip(components, wt_percents):
        total_moles += wt / mass_dict[comp]
    mol_percents = [((wt / mass_dict[comp]) / total_moles) * 100 for comp, wt in zip(components, wt_percents)]
    return mol_percents

def mol_fractions_to_wt_fractions(
    mol: float | list | np.ndarray, components: list[str], mass_dict: dict[str, float]
) -> float | list | np.ndarray:
    """Convert mole fractions to weight fractions (no normalisation).

    Unlike :func:`convert_mol_percent_to_wt_percent`, the output is not
    normalised to sum to 100. This means the function can be applied to a
    single component (e.g. a single oxide for an iron redox conversion) or
    a subset of a composition, as well as a full composition.

    Parameters
    ----------
    mol : float or array_like
        Mole fraction(s) of each component.
    components : list of str
        Component names corresponding to each input value.
    mass_dict : dict
        Mapping of component names to molecular masses.

    Returns
    -------
    float or list or numpy.ndarray
        Weight fraction(s) of each component. A scalar input returns a
        scalar, a list input returns a list, and any other array-like
        input returns a numpy array.

    Notes
    -----
    Scalar input assumes a single component -- pass ``components=[comp]``
    with the oxide name.
    """
    if np.isscalar(mol):
        return float(mol) * mass_dict[components[0]]
    mol_arr = np.asarray(mol, dtype=float)
    if mol_arr.ndim == 0:
        return float(mol_arr.item() * mass_dict[components[0]])
    masses = np.array([mass_dict[comp] for comp in components], dtype=float)
    result = mol_arr * masses
    return result.tolist() if isinstance(mol, list) else result

def wt_fractions_to_mol_fractions(
    wt: float | list | np.ndarray, components: list[str], mass_dict: dict[str, float]
) -> float | list | np.ndarray:
    """Convert weight fractions to mole fractions (no normalisation).

    Unlike :func:`convert_wt_percent_to_mol_percent`, the output is not
    normalised to sum to 100. This means the function can be applied to a
    single component (e.g. a single oxide for an iron redox conversion) or
    a subset of a composition, as well as a full composition.

    Parameters
    ----------
    wt : float or array_like
        Weight fraction(s) of each component.
    components : list of str
        Component names corresponding to each input value.
    mass_dict : dict
        Mapping of component names to molecular masses.

    Returns
    -------
    float or list or numpy.ndarray
        Mole fraction(s) of each component. A scalar input returns a
        scalar, a list input returns a list, and any other array-like
        input returns a numpy array.
    """
    if np.isscalar(wt):
        return float(wt) / mass_dict[components[0]]
    wt_arr = np.asarray(wt, dtype=float)
    if wt_arr.ndim == 0:
        return float(wt_arr.item() / mass_dict[components[0]])
    masses = np.array([mass_dict[comp] for comp in components], dtype=float)
    result = wt_arr / masses
    return result.tolist() if isinstance(wt, list) else result

def convert_wt_percent_to_moles(wt_percents, components, mass_dict, total_weight):
    """Convert weight (mass) percentages to moles."""
    moles = []
    for comp, wt in zip(components, wt_percents):
        moles.append((wt / 100 * total_weight) / mass_dict[comp])
    return moles

def convert_mol_percent_to_moles(mol_percent_dict, mass_dict, total_mass=100):
    """Convert mole percentages (given as a dictionary) to moles."""
    moles = {}
    for comp, mol_percent in mol_percent_dict.items():
        comp_mass = total_mass * (mol_percent / 100)
        moles[comp] = comp_mass / mass_dict[comp]
    return moles

def convert_moles_to_mol_percent(moles, components):
    """Convert absolute moles (given as a list or dict) to mole percentages."""
    if isinstance(moles, list):
        moles_dict = dict(zip(components, moles))
    else:
        moles_dict = moles
        
    total = sum(moles_dict.values())
    return {comp: (moles_dict[comp] / total) * 100 for comp in components}

def split_feot_to_feo_o(feot_moles: float, fe3_frac: float) -> tuple[float, float]:
    """
    Split total iron (FeOt) into the MAGEMin ``FeO + O`` redox pair at a
    target Fe3+/FeOt fraction, conserving the total iron budget.

    Parameters
    ----------
    feot_moles : float
        Total iron in mole units (atoms of Fe, equivalently the amount of
        FeO that would carry all iron as Fe2+).
    fe3_frac : float
        Target Fe3+/FeOt fraction in ``[0, 1]``.  0 = fully reduced (all
        Fe2+), 1 = fully oxidised (all Fe3+).

    Returns
    -------
    (feo, o) : tuple[float, float]
        ``feo`` is the total-iron column (all Fe expressed as FeO, equal to
        ``feot_moles``) and ``o`` is the excess oxygen required to oxidise
        the target fraction, per ``2FeO + O -> Fe2O3``.

    Notes
    -----
    Molar bookkeeping (each Fe2O3 carries 2 Fe atoms and needs 1 O):
        Fe3+ atoms = 2 * O  =>  O = fe3_frac * FeOt / 2
        Fe2+ atoms = FeOt - 2 * O  (implicitly held by the FeO column)
    """
    fe3_frac = float(fe3_frac)
    if not 0.0 <= fe3_frac <= 1.0:
        raise ValueError(f"fe3_frac must be in [0, 1], got {fe3_frac!r}")
    feot_moles = float(feot_moles)
    return feot_moles, fe3_frac * feot_moles / 2.0

def express_bulk_in_feo_o_basis(
    X: list[float],
    Xoxides: list[str],
    fe3_frac: float,
    feo_oxide: str = "FeO",
    fe2o3_oxide: str = "Fe2O3",
    o_oxide: str = "O",
) -> tuple[list[float], list[str]]:
    """
    Express a bulk composition in the MAGEMin ``FeO + O`` redox basis at a
    target Fe3+/FeOt fraction, conserving total iron.

    The ``FeO`` column is set to total iron (all Fe expressed as FeO,
    ``FeOt``) and ``O`` is set to the excess oxygen giving the requested
    Fe3+/FeOt partition.  Any ``Fe2O3`` component is removed (set to 0) and
    an existing ``O`` component is overwritten; missing components are
    appended.  The returned list preserves the input oxide order.

    Parameters
    ----------
    X : array-like
        Bulk composition values in MOLE units (mol fractions or mol%).  Use
        weight-based converters first if the input is in wt%.
    Xoxides : list of str
        Oxide names corresponding to ``X`` (may include 'FeO', 'Fe2O3', 'O'
        in any combination).
    fe3_frac : float
        Target Fe3+/FeOt fraction in ``[0, 1]``.
    feo_oxide, fe2o3_oxide, o_oxide : str
        Component names used in ``Xoxides``.

    Returns
    -------
    (X_new, Xoxides_new) : tuple[list, list]
        Composition in the ``FeO + O`` basis (unnormalised -- pass to
        ``convertBulk4MAGEMin`` or normalise afterwards).

    Examples
    --------
    >>> X = [50.0, 8.0, 0.5]              # SiO2, FeO, O  (mol%)
    >>> ox = ['SiO2', 'FeO', 'O']
    >>> X2, ox2 = express_bulk_in_feo_o_basis(X, ox, fe3_frac=0.1)
    >>> ox2
    ['SiO2', 'FeO', 'O']
    >>> X2[2] == 0.1 * X[1] / 2.0         # O = fe3_frac * FeOt / 2
    True
    """
    if len(Xoxides) != len(X):
        raise ValueError(
            f"X ({len(X)} items) and Xoxides ({len(Xoxides)} items) must have the same length"
        )
    seen = set()
    for ox in Xoxides:
        if ox in seen:
            raise ValueError(f"Duplicate oxide name in Xoxides: {ox!r}")
        seen.add(ox)

    X = [float(v) for v in X]
    Xoxides = list(Xoxides)
    feo_i = Xoxides.index(feo_oxide) if feo_oxide in Xoxides else None
    fe2o3_i = Xoxides.index(fe2o3_oxide) if fe2o3_oxide in Xoxides else None
    o_i = Xoxides.index(o_oxide) if o_oxide in Xoxides else None

    feo_mol = X[feo_i] if feo_i is not None else 0.0
    fe2o3_mol = X[fe2o3_i] if fe2o3_i is not None else 0.0
    feot = feo_mol + 2.0 * fe2o3_mol            # total Fe atoms (moles)

    feo_total, o_excess = split_feot_to_feo_o(feot, fe3_frac)

    if feo_i is not None:
        X[feo_i] = feo_total
    else:
        X.append(feo_total)
        Xoxides.append(feo_oxide)
    if fe2o3_i is not None:
        X[fe2o3_i] = 0.0
    if o_i is not None:
        X[o_i] = o_excess
    else:
        X.append(o_excess)
        Xoxides.append(o_oxide)
    return X, Xoxides
