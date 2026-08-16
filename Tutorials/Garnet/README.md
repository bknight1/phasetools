# Garnet Tutorials

This directory contains tutorials focused on modelling garnet chemistry, growth, and geothermobarometry using the `MAGEMinGarnetCalculator` and related models.

## Tutorial Catalog

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

---
**Units Note:** All tutorials use `kbar` for pressure and `°C` for temperature.
