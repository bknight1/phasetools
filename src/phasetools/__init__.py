# ### Core imports and Public API

### Import juliacall to access MAGEMin_C
try:
    import juliacall
    from juliacall import Main as jl
    from juliacall import convert as jlconvert

    MAGEMin_C = juliacall.newmodule("MAGEMin_C")
    MAGEMin_C.seval("using MAGEMin_C")
except Exception as _e:  # noqa: BLE001
    import warnings as _warnings

    _warnings.warn(
        f"Julia/MAGEMin_C not available: {_e}. "
        "Phase calculations requiring MAGEMin will fail. "
        "Install Julia and MAGEMin_C: phasetools-julia-setup --install",
        stacklevel=2,
    )
    MAGEMin_C = None
    jl = None
    jlconvert = None

__all__ = [
    "GarnetGenerator",
    "MAGEMinAssemblageCalculator",
    "MAGEMinBase",
    "MAGEMinGarnetCalculator",
    "MAGEMinPTGridCalculator",
    "MAGEMin_C",
    "PhaseFunctions",
    "PhasePTEstimator",
    "bulk_rock",
    "calculate_kd_fe_mg",
    "extract_end_member",
    "generate_distribution",
    "get_oxide_apfu",
    "get_phase_chemistry",
    "get_phase_mg_number",
    "jl",
    "jlconvert",
    "phase_frac",
    "single_point_minimization_with_conversion",
]

# Expose Public API
from .calculators.assemblage import MAGEMinAssemblageCalculator
from .calculators.garnet import MAGEMinGarnetCalculator
from .calculators.phase_search import PhaseFunctions
from .calculators.pt_estimation import PhasePTEstimator
from .calculators.pt_grid import MAGEMinPTGridCalculator
from .core.base import MAGEMinBase
from .core.engine import single_point_minimization_with_conversion
from .core.phase_properties import (
    calculate_kd_fe_mg,
    extract_end_member,
    get_oxide_apfu,
    get_phase_chemistry,
    get_phase_mg_number,
    phase_frac,
)
from .models.garnet_growth import GarnetGenerator, generate_distribution
from .utils import bulk_rock
