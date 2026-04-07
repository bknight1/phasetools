---
applyTo: '**'
---

# pyMAGEMin Memory (repo-local)

## Coding Preferences

- Use NumPy for arrays and `scipy.optimize.root_scalar` for scalar minimization/root finding.
- Follow existing patterns in `MAGEMinGarnetCalculator` and `PhaseFunctions`.
- Julia conversion: use `jlconvert(jl.Vector[...], data)` for lists/arrays before MAGEMin_C calls.
- Reuse `data = MAGEMin_C.Initialize_MAGEMin("ig", verbose=False)` across calls for speed.
- Keep edits minimal and localized in `src/pyMAGEMin/functions/`.

## Project Architecture

- **Core:** pyMAGEMin wraps Julia's MAGEMin_C thermodynamic minimizer.
- **Bootstrap:** `MAGEMin_C` is initialized in `src/pyMAGEMin/__init__.py`.
- **Key functions:** 
  - `MAGEMinGarnetCalculator.gt_along_path(...)` for garnet chemistry/fractionation along a P-T path
  - `MAGEMinGarnetCalculator.gt_single_point_calc_elements(...)` for single P-T points
  - `MAGEMinGarnetCalculator.generate_2D_grid_gt_elements(...)` for grid workflows
- **Element fractions:** Mg, Fe, Ca, Mn (returned in mol or wt basis depending on `sys_in`).
- **Bulk rock:** wt%, mol%, or vol% depending on `sys_in` (`'wt'`, `'mol'`, `'vol'`).

## Solutions Repository

### MAGEMin Workflow Notes
- **Path calculations:** `gt_along_path(...)` returns `X_along_path`; preserve it when `fractionate=True`.
- **Single-point checks:** use `gt_single_point_calc_elements(...)` for Fe-Mg-Mn-Ca outputs.
- **Grid checks:** use `generate_2D_grid_gt_endmembers(...)` or `generate_2D_grid_gt_elements(...)`.
