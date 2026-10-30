# Core Submodule

The `core` submodule provides the foundational classes and low-level bridging logic required to communicate with the MAGEMin thermodynamic engine.

`MAGEMinBase.setup_bulk_composition(..., rm_list=None)` optionally removes
phases. An empty list (`rm_list=[]`) is treated as no removals and does not
call MAGEMin's `remove_phases`; a non-empty list is passed through unchanged.
Use `rm_list=['g']` (for example) when actual phase suppression is intended.

## Key Components

### `base.py`
- **`MAGEMinBase`**: The base class for all calculators and models. 
- Manages bulk composition initialisation and unit standardisation (e.g., converting wt% to molar fractions).
- Provides state-sharing utilities (`_copy_state_to`) to sync Julia objects between different instances. The target becomes a *non-owner* of the shared handle (`_owns_data = False`) and also receives `db`/`dataset`/`verbose`; if it previously owned a different handle (construct-then-copy), that handle is finalised automatically so its C-side memory is not leaked.
- **`finalize()`**: releases the ~40 MB C-side allocation behind `self.data` — Julia's GC never does, so call this when finished with an instance (idempotent). Only the *owning* instance may finalise; non-owners warn and skip.

### `engine.py`
- Low-level interface for Python-to-Julia type conversion.
- Handles the technical bridging to the `MAGEMin_C` Julia routines.
- `evaluate_solution_phase_gibbs(calculator, phase, fractions, pressure_kbar,
  temperature_c)` evaluates a supplied, normalised solution-phase endmember
  composition using MAGEMin's `p2x_convert` / `pc_convert` bridge. It is generic
  across solution phases; it does not run equilibrium or perform inversion.
- Returns a `SolutionPhaseGibbsEvaluation` containing `phase`, copied Python
  endmember fractions, `df`, `mu`, `dfx`, and the MAGEMin `sf_ok` flag.
- The optional `endmember_names` argument lets a calculator pass names
  discovered from a phase-bearing MAGEMin output; these labels document the
  returned vector order without imposing database-specific names in the bridge.

### `thermodynamics.py`
- Generic plain-Python extraction helpers and dataclasses for available
  thermodynamic output fields: `extract_gamma`, `extract_system_gibbs`,
  `extract_stable_phase_gibbs`, `extract_candidate_phase_gibbs`, and
  `extract_thermodynamic_snapshot`.
- MAGEMin output fields vary by version/calculation mode; unavailable fields
  are represented by `None` or empty tuples. Gamma stays in MAGEMin's native
  order; optional oxide names document that ordering without transforming it.

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
