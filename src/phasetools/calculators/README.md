# Calculators Submodule

Calculators are high-level wrappers designed for specific thermodynamic tasks, such as grid generation, phase searching, and inverse modelling.

## Key Components

### `pt_grid.py`
- **`MAGEMinPTGridCalculator`**: Core utility for calculating property grids over Pressure and Temperature.
- Supports structured extraction of end-members, cations, and oxides from large datasets.

### `garnet.py`
- **`MAGEMinGarnetCalculator`**: Specialized for garnet chemistry.
- Includes logic for site-specific cation fractions (X-site) and modelling garnet evolution along P-T paths (optionally with fractionation).

### `pt_estimation.py`
- **`PhasePTEstimator`**: A geothermobarometry engine.
- Minimises the misfit between observed phase chemistry (e.g., from EPMA) and equilibrium thermodynamic predictions using various `scipy.optimize` solvers.

### `assemblage.py`
- **`MAGEMinAssemblageCalculator`**: Maps the stability of multi-mineral assemblages.
- Generates boolean masks to identify P-T regions where specific minerals coexist.

### `phase_search.py`
- **`PhaseSearchCalculator`**: Uses root-finding algorithms to locate phase boundaries.
- Precise determination of solidus, liquidus, or specific phase appearance/disappearance points.
