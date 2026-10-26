# Garnet Tutorials

This directory contains tutorials focused on modelling garnet chemistry, growth, and geothermobarometry using the `MAGEMinGarnetCalculator` and related models.

## Tutorial Catalog

### [Single-Point Equilibrium and Garnet MDF](./MDF_single_point_workflow.ipynb)
Compares equilibrium and maximum-driving-force garnet at one P–T point, showing active thermodynamic fractions and comparable closed Mg-Fe(total)-Mn-Ca cation proportions on an EPMA all-Fe-as-FeO basis. The paired Jupytext source is [`MDF_single_point_workflow.py`](./MDF_single_point_workflow.py).

### [Synthetic-Target EBC Recovery with MDF](./MDF_synthetic_ebc_workflow.ipynb)
Bounded inversion of synthetic MDF targets for a restricted kho-absent model, a full kho-bearing thermodynamic-fraction target, and a four-component EPMA-style cation target. Every forward evaluation uses `GarnetMDFCalculator.calculate_compositions`: standardised molar O=0 selects restricted MDF (with kho exactly zero), while positive O selects full kho-aware MDF, without a kho target. Starting guesses are additive FeO/MnO/CaO changes in absolute wt-percentage points (including −10 and ±5 wt-pp starts), rather than relative-percent changes. Trial complete-bulk vectors are converted to mol% and normalised; normalised optimiser coordinates span [−1, 1], mapped to non-negative oxide amounts. Case A recovers target space from a large start but demonstrates non-unique bulk recovery; Case B recovers its source from the large start, while Case C retains a small non-zero target-space residual (~1.1 × 10⁻³) from its ≥5 wt-pp start. These are synthetic tests, not real EPMA inference or recovery of a published composition. The paired Jupytext source is [`MDF_synthetic_ebc_workflow.py`](./MDF_synthetic_ebc_workflow.py).

### [1. Functionality Overview](./1-Functionality_overview.ipynb)
**Objective:** Comprehensive guide to extracting garnet data and modelling growth.
- **Key Features:**
    - Extracting end-member and elemental fractions from single-point calculations.
    - Modelling garnet evolution along P-T paths with and without fractionation.
    - Generating complex garnet populations with radial zoning.
    - Visualising compositional profiles and size distributions.
    - Assessing garnet stability and chemistry during retrograde metamorphism.

### [2. Overlapping Isopleths](./2-Overlapping_isopleths.ipynb)
**Objective:** Estimating metamorphic P-T paths using inverse modelling of garnet profiles.
- **Key Features:**
    - Constructing large P-T grids of garnet composition.
    - Visualising compositional isopleths (e.g., Almandine, Grossular, Pyrope, Spessartine).
    - Implementing geothermobarometry by minimising the misfit between measured chemistry and thermodynamic predictions.
    - Recovering P-T trajectories from zoned garnet crystals.

### [3. Garnet-Cpx Thermometry](./3-Garnet_cpx_thermometry.ipynb)
**Objective:** Mapping the garnet–clinopyroxene Fe²⁺–Mg distribution coefficient (Kd) and comparing thermometer calibrations.
- **Key Features:**
    - Setting up the S10 eclogite bulk composition in the `mpe` database.
    - Extracting garnet and clinopyroxene Fe²⁺/Mg from APFU with dynamic cpx phase resolution (`dio`/`omph`/`aug`).
    - Masking absent and solvus-ambiguous grid cells.
     - Mapping Kd over a P–T grid and applying Mysen & Heier (1972), Ganguly (1979; explicitly labelled Yavuz & Yıldırım piecewise form), Krogh Ravna (2000), Ellis & Green (1979), and Räheim & Green (1974) calibrations.
     - Using the intentional traditional mixed Fe convention: garnet Fe from `ox_apfu_FeO` (all treated as Fe²⁺) and cpx Fe²⁺ from the excess-O heuristic split.

### Garnet MDF/PTM workflow

The single-point tutorial above demonstrates the supported MDF workflow using
`GarnetMDFCalculator`. Maximum-driving-force (MDF) and parallel-tangent
method/model (PTM) are related descriptions of this supported garnet
calculation. The calculator compares ordinary equilibrium with
a garnet-suppressed candidate matrix, solves kho-aware exchange equations over
the active supported garnet model, and reports closed Mg-Fe(total)-Mn-Ca
cation proportions. These outputs are not APFU, X-site fractions, or
thermodynamic endmember fractions. All Fe is reported as FeO-equivalent,
including ferric Fe in kho; inspect the matrix assemblage and convergence
diagnostics when interpreting results.

References: [10.1007/s00410-014-1059-6](https://doi.org/10.1007/s00410-014-1059-6),
[10.1016/j.chemgeo.2018.05.019](https://doi.org/10.1016/j.chemgeo.2018.05.019),
and [10.1016/j.chemgeo.2019.119323](https://doi.org/10.1016/j.chemgeo.2019.119323).

---
**Units Note:** All tutorials use `kbar` for pressure and `°C` for temperature.
