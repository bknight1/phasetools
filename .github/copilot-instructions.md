# pyMAGEMin — AI Coding Assistant Instructions

This repo provides Python wrappers around the Julia MAGEMin_C thermodynamic minimizer to compute phase equilibria and garnet growth paths. These notes capture the project-specific patterns, workflows, and integration points to help you be productive fast.

## Architecture and Key Modules
- Core package: pyMAGEMin (Python, src layout). See [src/pyMAGEMin/__init__.py](src/pyMAGEMin/__init__.py) for `juliacall` initialization of the Julia module `MAGEMin_C` made globally available as `pyMAGEMin.MAGEMin_C`.
- MAGEMin wrappers and helpers:
  - [src/pyMAGEMin/functions/MAGEMin_functions.py](src/pyMAGEMin/functions/MAGEMin_functions.py): `MAGEMinGarnetCalculator` (grid, single-point, and path calculations, optional fractionation) and `PhaseFunctions` (solidus/liquidus search via `scipy.optimize.root_scalar`, batch fractionation).
  - [src/pyMAGEMin/functions/bulk_rock_functions.py](src/pyMAGEMin/functions/bulk_rock_functions.py): composition utilities (mol↔wt conversions, FeOt splitting, garnet endmember → element fractions).
  - [src/pyMAGEMin/functions/garnet_growth.py](src/pyMAGEMin/functions/garnet_growth.py): builds size distributions and radial profiles from P–T–t paths using results from `MAGEMinGarnetCalculator`.
  - [src/pyMAGEMin/functions/utils.py](src/pyMAGEMin/functions/utils.py): `create_PTt_path()` P–T–t interpolation with embedded original points.

## Julia Integration (critical)
- Requires a local Julia installation and the Julia package `MAGEMin_C`. The installer checks/installs both (see [setup.py](setup.py)).
- Cross-language data: use `from juliacall import Main as jl, convert as jlconvert` then convert to Julia types, e.g. `jlconvert(jl.Vector[jl.Float64], np_array)` and `jlconvert(jl.Vector[jl.String], list_of_names)`.
- Initialize MAGEMin once per session: `data = MAGEMin_C.Initialize_MAGEMin("ig", verbose=False)`; reuse `data` across calls for performance.

## Typical Call Flow
- Single or multi-point minimization: `MAGEMin_C.single_point_minimization(P, T, data, X=..., Xoxides=..., sys_in=...)` or `multi_point_minimization(P_vec, T_vec, data, ...)`.
- Higher-level APIs in `MAGEMinGarnetCalculator` wrap the above and compute endmember and element fractions for phase `"g"` (garnet). Helpers `phase_frac()` and `extract_end_member()` encapsulate MAGEMin_C output access patterns.
- Units and conventions:
  - `sys_in`: `'mol'`, `'wt'`, or `'vol'` where applicable; keep consistent across pipelines.
  - Garnet endmembers: `py`, `alm`, `spss`, `gr`, `kho` with element mapping defined in [bulk_rock_functions.py](src/pyMAGEMin/functions/bulk_rock_functions.py).
  - Phase keys: garnet is `'g'`; many helpers assume this.

## Minimal Usage Example
```python
import numpy as np
from pyMAGEMin import MAGEMin_C
from juliacall import Main as jl, convert as jlconvert

data = MAGEMin_C.Initialize_MAGEMin("ig", verbose=False)
P = jlconvert(jl.Vector[jl.Float64], np.linspace(1, 10, 5))
T = jlconvert(jl.Vector[jl.Float64], np.linspace(700, 900, 5))
Xox = jlconvert(jl.Vector[jl.String], ["SiO2","Al2O3","CaO","MgO","FeO","Fe2O3","K2O","Na2O","TiO2","Cr2O3","H2O"])
X   = jlconvert(jl.Vector[jl.Float64], [48.43,15.19,11.57,10.13,6.65,1.64,0.59,1.87,0.68,0.0,3.0])
out = MAGEMin_C.multi_point_minimization(P, T, data, X=X, Xoxides=Xox, sys_in="wt")
```

## Developer Workflows
- Install (ensures Julia and MAGEMin_C):
```bash
python -m pip install -e .
```
- Quick check Julia availability if install fails:
```bash
julia --version
julia -e 'using Pkg; Pkg.status("MAGEMin_C")'
```
- Parallel example (MPI): see [tests/parallel_test.py](tests/parallel_test.py). Run with e.g.:
```bash
mpiexec -n 4 python tests/parallel_test.py
```
- Tutorials and examples: notebooks under [Tutorials/](Tutorials) demonstrate workflows end-to-end.

## Patterns and Gotchas
- Always convert Python arrays/lists to Julia vectors with `jlconvert(...)` before calling MAGEMin_C; prefer contiguous NumPy arrays for speed.
- Reuse `data` returned by `Initialize_MAGEMin(...)` across calls to avoid re-initialization overhead.
- `PhaseFunctions.find_phase_in/saturation` use bisection with brackets; provide realistic `(T_low, T_high)` for robust convergence.
- Fractionation: `PhaseFunctions.fractionate_phase()` updates bulk composition between steps; honor `sys_in` and return value when chaining along paths.
- When computing element fractions from endmembers, prefer `calculate_molar_fractions(...)` in [bulk_rock_functions.py](src/pyMAGEMin/functions/bulk_rock_functions.py) and only convert to wt% at edges using `convert_mol_percent_to_wt_percent(...)`.

## Extending the Package
- New wrappers should follow the `MAGEMin_functions.py` pattern: convert inputs via `jlconvert`, call MAGEMin_C, then post-process into NumPy arrays and Python dicts.
- Keep phase keys and `sys_in` handling consistent; update helpers if introducing new phases or endmembers.
- Place general utilities in `functions/` and import in [src/pyMAGEMin/__init__.py](src/pyMAGEMin/__init__.py) if they should be top-level accessible.

