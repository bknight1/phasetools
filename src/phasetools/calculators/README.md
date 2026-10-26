# Calculators Submodule

Calculators are high-level wrappers for thermodynamic tasks such as grid
generation, phase searching, geothermobarometry and garnet MDF calculations.

## Key components

### `pt_grid.py`
- **`MAGEMinPTGridCalculator`** calculates properties over pressure-temperature
  grids and extracts endmembers, cations, and oxides.

### `garnet.py`
- **`MAGEMinGarnetCalculator`** extracts garnet chemistry and models garnet
  evolution along P–T paths, optionally with fractionation.
- Growth cation outputs keep their historical `(Mg, Mn, Fe, Ca)` tuple (and
  optional weight conversion). Growth/APFU and MDF endmember projections share
  the atomic cation normaliser in `garnet_composition.py`; closed proportions
  are not APFU or site occupancies. In particular, `FeOt` includes ferric Fe
  assigned to kho's Y site, not only divalent X-site Fe.

### `pt_estimation.py`
- **`PhasePTEstimator`** minimises misfit between observed phase chemistry and
  equilibrium predictions using `scipy.optimize`.
- **`optimize_pt(objective, bounds, ...)`** is a shared P–T optimiser for a
  one-argument objective over `[P, T]`; it supports global and local methods,
  optional quick mode, and callback-path capture.

### `assemblage.py`
- **`MAGEMinAssemblageCalculator`** maps stability and coexistence of requested
  mineral assemblages over P–T conditions.

### `phase_search.py`
- **`PhaseFunctions`** uses root finding to locate phase boundaries and phase
  appearance or disappearance conditions.

### `garnet_mdf.py`
- **`GarnetMDFCalculator`** discovers the active garnet endmembers from
  garnet-stable equilibrium and dispatches from standardised molar bulk O.
  Configure bulk once with `setup_bulk_composition(oxides, values,
  sys_in='wt' or 'mol')`; `calculate_compositions(P, T)`,
  `calculate_matrix_state(P, T)`, and `scan_isopleths(P_range, T_range)` then
  use that saved state. A scan needs only one setup call.
  Absent/zero O solves the pyrope-referenced divalent exchanges over `py`,
  `alm`, optional `spss`, and `gr`, with `kho` fixed exactly to zero. Positive
  O solves the N−1 exchanges over the full supported `py`, `alm`, optional
  `spss`, `gr`, `kho` set.
  It inherits directly from `MAGEMinBase` and uses raw single-point MAGEMin_C
  minimisations rather than constructing P–T grids.
  Unsupported active components, including zero-fraction components, are
  rejected. Positive-O/kho exchange requires standardised matrix Gamma for FeO,
  excess `O`, and `Al2O3`; its matrix exchange potential is
  `2*Gamma_FeO + Gamma_O - Gamma_Al2O3`. MAGEMin standardises Fe2O3 input to
  FeO plus excess O, so neither raw FeO nor raw O is required when Fe2O3 is
  supplied. The result
  records the selected `solver_mode` (`"restricted"` or `"full"`). The ordinary
  equilibrium projection remains unrestricted and is reported independently.
- `calculate_compositions(...)` returns a `GarnetMDFResult` whose `mdf`
  and `equilibrium` mappings contain exactly `Mg`, `Fe`, `Mn`, and `Ca`. These
  are closed Mg-Fe-Mn-Ca cation proportions, not thermodynamic endmember
  fractions, APFU, or X-site proportions. The keyword `fe_basis` defaults to
  `"FeOt"` (case-insensitive; `"Fe2"` is also accepted as an alias). In both
  bases, py/alm/spss/gr contribute three corresponding cations. Kho contributes
  three Mg and two Fe for `"FeOt"`, but contributes no Fe for `"Fe2+"`;
  almandine Fe is retained on either basis. The selected canonical basis is
  recorded as `result.fe_basis`. When spessartine is absent, Mn is zero and the
  other values close to one. The basis changes these reported projections,
  not the MDF thermodynamic equations.
- Thermodynamic endmember fractions and closed cation comparisons are distinct.
  The result retains active MDF and ordinary-equilibrium fractions in private
  audit fields. `active_endmember_names`, `matrix_phases`, and
  `equilibrium_phases` identify the model and candidate assemblages.
- `scan_isopleths(...)` returns Mg/Fe/Mn/Ca arrays, residual norms, convergence
  flags, status messages, active endmember names, and matrix assemblages for
  each P–T point. Failed cells contain NaN compositions; successful but
  non-converged cells retain their compositions and diagnostics. Its `fe_basis`
  keyword uses the same default and conventions as `calculate_compositions`,
  and the canonical basis is recorded as `result.fe_basis`. The default
  Euclidean N−1 residual tolerance is `0.03 kJ/mol garnet formula`.
- `calculate_equilibrium(P, T)` requires a configured bulk, checks unrestricted
  minimisation and garnet stability, and discovers active model names.
  `calculate_matrix_state(P, T)` rechecks unrestricted garnet stability on the
  configured bulk, then temporarily removes garnet and returns matrix output,
  named Gamma values, and phase assemblage. The configured raw input and its
  `sys_in` basis remain authoritative; only phase-removal state is transient.
  Inspect the candidate matrix assemblage; numerical convergence does not
  establish geological validity.

The calculator accepts `sys_in='wt'` or `'mol'` for bulk setup. Dispatch uses
the standardised molar O actually sent to MAGEMin, not the raw input scale or
the unit-converted `self.X`. Internal
thermodynamic garnet fractions are molar; public closed cation proportions are
independent of the bulk basis. The supported tutorial is
`Tutorials/Garnet/MDF_single_point_workflow.py` and its paired notebook.

This calculator implements the garnet maximum-driving-force (MDF) workflow,
also described as the parallel-tangent method/model (PTM).
References: [10.1007/s00410-014-1059-6](https://doi.org/10.1007/s00410-014-1059-6),
[10.1016/j.chemgeo.2018.05.019](https://doi.org/10.1016/j.chemgeo.2018.05.019),
and [10.1016/j.chemgeo.2019.119323](https://doi.org/10.1016/j.chemgeo.2019.119323).
