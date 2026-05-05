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

---
**Units Note:** All tutorials use `kbar` for pressure and `°C` for temperature.
