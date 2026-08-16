# Core Submodule

The `core` submodule provides the foundational classes and low-level bridging logic required to communicate with the MAGEMin thermodynamic engine.

## Key Components

### `base.py`
- **`MAGEMinBase`**: The base class for all calculators and models. 
- Manages bulk composition initialisation and unit standardisation (e.g., converting wt% to molar fractions).
- Provides state-sharing utilities (`_copy_state_to`) to sync Julia objects between different instances.

### `engine.py`
- Low-level interface for Python-to-Julia type conversion.
- Handles the technical bridging to the `MAGEMin_C` Julia routines.

### `phase_properties.py`
- Standardised utility functions for extracting data from MAGEMin results:
    - `phase_frac`: Total phase fractions (handles solvi).
    - `get_oxide_apfu`: Cation counts (Atoms Per Formula Unit).
    - `get_phase_chemistry`: Oxide concentrations (wt% or mol%).
    - `get_phase_mg_number`: Phase-wide $Mg\#$ (Total Iron).
    - `get_phase_mg2_number`: Phase-wide $Mg\#$ (Divalent Iron only).
    - `get_phase_fe_split`: Heuristic splitting of total iron into $\text{Fe}^{2+}$ and $\text{Fe}^{3+}$.
    - `calculate_kd_fe_mg`: Distribution coefficients between phases.

### Solvus (multi-instance) handling
MAGEMin reports coexisting solvus limbs as repeated entries in `out.ph`
(e.g. two `dio` clinopyroxenes or two `amp` amphiboles). `phase_frac`
sums them; the low-level composition helpers in `phase_properties.py`
(`get_oxide_apfu`, `get_phase_chemistry`, `extract_end_member`,
`get_phase_mg_number`, `get_phase_mg2_number`, `get_phase_fe_split`)
take an `instance` argument:

- integer index (default `0`) — that instance, with a `UserWarning`
  noting how many other instances exist; out-of-range → zeros;
- `'all'` — one value per instance, returned as numpy arrays.

`MAGEMinPTGridCalculator.extract_from_grid` / `single_point_calc` /
`generate_2D_grid` return a **list of per-instance bundles**, indexed
`res[0]`, `res[1]`, ... :

```python
res = calc.extract_from_grid('dio', oxides=['Na2O'], mg_number=True)
c0, c1 = res                     # limb 0, limb 1
c0['mol_frac']                   # array over grid points (limb 0)
c0['ox_apfu_Na2O']               # array over grid points (limb 0)
total_dio = c0['mol_frac'] + c1['mol_frac']   # sum the limbs yourself
```

- Bundle keys are unsuffixed (`mol_frac`, `wt_frac`, `vol_frac`,
  `ox_apfu_*`, `chem_*`, `em_*`, `cat_*`, `Mg_number`, `Fe2`, `Fe3`).
- The shape is the **same for every phase** — a single-instance phase
  (e.g. garnet) has a one-element list, always at `res[0]`.
- Each bundle value is an array over the grid points, **NaN** where the
  phase (or that limb) is absent; if the phase never appears, the list
  is empty.  No totals are computed — sum the `mol_frac` bundles if
  you want the whole-phase fraction.
- Bundle `res[k]` is the k-th occurrence of the phase in that point's
  `out.ph`; solvus branch ordering may swap between grid points, so
  track both limbs when plotting isopleths.  `end_members='auto'`
  discovers end-members from the first occurrence (solvus limbs share
  the same solution model).  `MAGEMinGarnetCalculator.
  generate_2D_grid_gt_endmembers` returns the single-instance bundle
  directly, preserving the historical `em_py`, `mol_frac`, ... keys.
