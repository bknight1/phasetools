# ### Import the functions folder

### Import juliacall to access MAGEMin_C
import juliacall
# from juliacall import Main as jl, convert as jlconvert

MAGEMin_C = juliacall.newmodule("MAGEMin_C")
MAGEMin_C.seval("using MAGEMin_C")

from .functions import bulk_rock_functions, garnet_growth, MAGEMin_functions, pt_estimation, utils
