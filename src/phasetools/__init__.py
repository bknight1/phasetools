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
    "CandidatePhaseGibbs",
    "GarnetGenerator",
    "GarnetMDFCalculator",
    "GarnetMDFGridResult",
    "GarnetMDFResult",
    "MAGEMinAssemblageCalculator",
    "MAGEMinBase",
    "MAGEMinGarnetCalculator",
    "MAGEMinPTGridCalculator",
    "MAGEMin_C",
    "PhaseFunctions",
    "PhasePTEstimator",
    "SolutionPhaseGibbsEvaluation",
    "StablePhaseGibbs",
    "ThermodynamicSnapshot",
    "bulk_rock",
    "calculate_kd_fe_mg",
    "evaluate_solution_phase_gibbs",
    "extract_candidate_phase_gibbs",
    "extract_end_member",
    "extract_gamma",
    "extract_stable_phase_gibbs",
    "extract_system_gibbs",
    "extract_thermodynamic_snapshot",
    "garnet_fractions_to_mg_fe_mn_ca",
    "generate_distribution",
    "get_oxide_apfu",
    "get_phase_chemistry",
    "get_phase_mg_number",
    "jl",
    "jlconvert",
    "kho_aware_exchange_potentials",
    "optimize_pt",
    "phase_frac",
    "single_point_minimization_with_conversion",
]

# Expose Public API
from .calculators.assemblage import MAGEMinAssemblageCalculator
from .calculators.garnet import MAGEMinGarnetCalculator
from .calculators.garnet_mdf import (
    GarnetMDFCalculator,
    GarnetMDFGridResult,
    GarnetMDFResult,
    garnet_fractions_to_mg_fe_mn_ca,
    kho_aware_exchange_potentials,
)
from .calculators.phase_search import PhaseFunctions
from .calculators.pt_estimation import PhasePTEstimator, optimize_pt
from .calculators.pt_grid import MAGEMinPTGridCalculator
from .core.base import MAGEMinBase
from .core.engine import (
    SolutionPhaseGibbsEvaluation,
    evaluate_solution_phase_gibbs,
    single_point_minimization_with_conversion,
)
from .core.phase_properties import (
    calculate_kd_fe_mg,
    extract_end_member,
    get_oxide_apfu,
    get_phase_chemistry,
    get_phase_mg_number,
    phase_frac,
)
from .core.thermodynamics import (
    CandidatePhaseGibbs,
    StablePhaseGibbs,
    ThermodynamicSnapshot,
    extract_candidate_phase_gibbs,
    extract_gamma,
    extract_stable_phase_gibbs,
    extract_system_gibbs,
    extract_thermodynamic_snapshot,
)
from .models.garnet_growth import GarnetGenerator, generate_distribution
from .utils import bulk_rock
