# ### Core imports and Public API

### Import juliacall to access MAGEMin_C
import juliacall

MAGEMin_C = juliacall.newmodule("MAGEMin_C")
MAGEMin_C.seval("using MAGEMin_C")

# Expose Public API
from .core.base import MAGEMinBase
from .core.engine import single_point_minimization_with_conversion
from .core.phase_properties import phase_frac, extract_end_member, get_oxide_apfu

from .calculators.garnet import MAGEMinGarnetCalculator
from .calculators.assemblage import MAGEMinAssemblageCalculator
from .calculators.pt_estimation import PhasePTEstimator
from .calculators.phase_search import PhaseFunctions

from .models.garnet_growth import GarnetGenerator, generate_distribution

from .utils import bulk_rock
