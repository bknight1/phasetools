# General Tutorials

This directory contains foundational tutorials for using `phasetools` to interact with the MAGEMin thermodynamic engine.

## Tutorial Catalog

### [1. Getting Started](./1-Getting_started.ipynb)
**Objective:** Introduction to the basic workflow of `phasetools`.
- **Key Features:**
    - Importing required modules.
    - Performing minimisations for predefined bulk rock compositions (e.g., KLB-1).
    - Setting up custom bulk compositions.
    - Changing thermodynamic datasets.
    - Excluding specific phases from calculations.
    - Basic multi-point (grid) minimisation.

### [2. Determining Fractions](./2-Determining_fractions.ipynb)
**Objective:** Using brute-force search to identify phase stability boundaries.
- **Key Features:**
    - Defining calculation parameters for igneous systems (`ig` database).
    - Locating solidus and liquidus temperatures at a given pressure.
    - Analysing phase fraction changes across temperature gradients.

### [3. Lunar Magma Ocean](./3-Lunar_Magma_Ocean.ipynb)
**Objective:** Replicating a complex planetary differentiation model (Johnson et al. 2021).
- **Key Features:**
    - Modelling multi-stage crystallisation of a rocky body (the Moon).
    - **Stage 0:** Equilibrium crystallisation.
    - **Stages 1-N:** Sequential fractional crystallisation in discrete volume steps.
    - Geophysical integration: Converting between radius, pressure, and depth.
    - Synthesising the formation of a flotation crust.

---
**Units Note:** All tutorials use `kbar` for pressure and `°C` for temperature.
