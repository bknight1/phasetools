# Calculators Submodule

Calculators are high-level wrappers designed for specific thermodynamic tasks, such as grid generation, phase searching, and inverse modelling.

## Key Components

### `pt_grid.py`
- **`MAGEMinPTGridCalculator`**: Core utility for calculating property grids over Pressure and Temperature.
- Supports structured extraction of end-members, cations, and oxides from large datasets.

### `garnet.py`
- **`MAGEMinGarnetCalculator`**: Specialized for garnet chemistry.
- Includes logic for site-specific cation fractions (X-site) and modelling garnet evolution along P-T paths (optionally with fractionation).
- When `sys_in='wt'`, fractionation uses the weight-basis garnet fraction (not the molar fraction) to stay consistent with the bulk composition units.
- `X_along_path` rows are normalised to sum to 1 regardless of the `sys_in` setting.
- **`fe_basis`** option controls the Fe basis of garnet X-site fractions: `'FeOt'` (default — all Fe treated as Fe²⁺, the standard community convention, e.g. Williams & Grambling 1990; Krogh Ravna 2000) or `'Fe2+'` (stoichiometric ferrous-only split, for garnets with significant Fe³⁺).

### `pt_estimation.py`
- **`PhasePTEstimator`**: A geothermobarometry engine.
- Minimises the misfit between observed phase chemistry (e.g., from EPMA) and equilibrium thermodynamic predictions using various `scipy.optimize` solvers.

### `assemblage.py`
- **`MAGEMinAssemblageCalculator`**: Maps the stability of multi-mineral assemblages.
- Generates boolean masks to identify P-T regions where specific minerals coexist.

### `phase_search.py`
- **`PhaseFunctions`**: Uses root-finding algorithms to locate phase boundaries.
- Precise determination of solidus, liquidus, or specific phase appearance/disappearance points.
- `fractionate_phase` handles both solution phases (`SS_vec`) and pure phases (`PP_vec`) transparently.  When a phase appears multiple times (e.g. solvus), only the first instance is removed.  Output is always normalised to sum to 1.
