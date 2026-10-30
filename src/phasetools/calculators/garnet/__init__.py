"""Garnet domain package: calculator, MDF/PTM solver and composition maths.

Import surface mirrors the old flat modules, so
``from phasetools.calculators.garnet import ...`` continues to work.
"""

from .calculator import MAGEMinGarnetCalculator
from .composition import (
    garnet_cations_from_apfu,
    garnet_endmember_fractions_to_oxide_wt,
    garnet_fractions_to_mg_fe_mn_ca,
    kho_aware_exchange_potentials,
    normalise_fe_basis,
    normalise_garnet_cations,
)
from .mdf import GarnetMDFCalculator, GarnetMDFGridResult, GarnetMDFResult

__all__ = [
    "GarnetMDFCalculator",
    "GarnetMDFGridResult",
    "GarnetMDFResult",
    "MAGEMinGarnetCalculator",
    "garnet_cations_from_apfu",
    "garnet_endmember_fractions_to_oxide_wt",
    "garnet_fractions_to_mg_fe_mn_ca",
    "kho_aware_exchange_potentials",
    "normalise_fe_basis",
    "normalise_garnet_cations",
]
