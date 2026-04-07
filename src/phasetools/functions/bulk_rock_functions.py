# %%
import numpy as np
from molmass import Formula

# %%
### Useful functions for bulk rock conversions

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

### Get the mass dictionaries for use in conversion functions.
def get_molar_mass_dict():
    return molar_mass_dict

def get_atomic_mass_dict():
    return atomic_mass_dict

def convert_oxide_to_element_moles(oxide_moles, oxide, element):
    molar_mass_dict = get_molar_mass_dict()
    return oxide_moles * (molar_mass_dict[element] / molar_mass_dict[oxide])


def convert_mol_percent_to_wt_percent(mol_percents, components, mass_dict):
    """
    Generic conversion from mole percent to weight (mass) percent.
    
    Args:
        mol_percents (list): Mole percentages for each component.
        components (list): List of component names (e.g., oxides or elements).
        mass_dict (dict): Dictionary mapping each component to its mass (g/mol or atomic mass).
        
    Returns:
        list: Weight percentages for each component.
    """
    total_mass = 0
    for comp, mol in zip(components, mol_percents):
        total_mass += mol * mass_dict[comp]
    wt_percents = [(mol * mass_dict[comp] / total_mass) * 100 for comp, mol in zip(components, mol_percents)]
    return wt_percents


def convert_wt_percent_to_mol_percent(wt_percents, components, mass_dict):
    """
    Generic conversion from weight (mass) percent to mole percent.
    
    Args:
        wt_percents (list): Weight percentages for each component.
        components (list): List of component names.
        mass_dict (dict): Dictionary mapping each component to its mass (g/mol or atomic mass).
        
    Returns:
        list: Mole percentages for each component.
    """
    total_moles = 0
    for comp, wt in zip(components, wt_percents):
        total_moles += wt / mass_dict[comp]
    mol_percents = [((wt / mass_dict[comp]) / total_moles) * 100 for comp, wt in zip(components, wt_percents)]
    return mol_percents


def convert_wt_percent_to_moles(wt_percents, components, mass_dict, total_weight):
    """
    Convert weight (mass) percentages to moles.
    
    Args:
        wt_percents (list): Weight percentages for each component.
        components (list): List of component names.
        mass_dict (dict): Dictionary mapping each component to its mass.
        total_weight (float): Total weight of the sample.
        
    Returns:
        list: Moles for each component.
    """
    moles = []
    for comp, wt in zip(components, wt_percents):
        # (wt%/100 * total_weight) gives the weight of the component.
        moles.append((wt / 100 * total_weight) / mass_dict[comp])
    return moles


def convert_mol_percent_to_moles(mol_percent_dict, mass_dict, total_mass=100):
    """
    Convert mole percentages (given as a dictionary) to moles for each component.
    
    Args:
        mol_percent_dict (dict): Dictionary where keys are components and values are mole percentages.
        mass_dict (dict): Dictionary mapping each component to its mass.
        total_mass (float): Total mass (default 100 g) implied for the mole percentage conversion.
        
    Returns:
        dict: Dictionary of moles for each component.
    """
    moles = {}
    for comp, mol_percent in mol_percent_dict.items():
        # Assume total_mass is proportional to the sum of the percentages (e.g. 100 g)
        comp_mass = total_mass * (mol_percent / 100)
        moles[comp] = comp_mass / mass_dict[comp]
    return moles


def convert_moles_to_mol_percent(moles, components):
    """
    Convert absolute moles (given as a list or dict) to mole percentages.
    
    Args:
        moles (list or dict): Absolute moles for each component.
            If list, the order must correspond to 'components'.
        components (list): List of component names.
        
    Returns:
        dict: Dictionary of mole percentages for each component.
    """
    if isinstance(moles, list):
        moles_dict = dict(zip(components, moles))
    else:
        moles_dict = moles
        
    total = sum(moles_dict.values())
    return {comp: (moles_dict[comp] / total) * 100 for comp in components}




