# AI Development Mandate: phasetools

This document defines the foundational requirements and architectural standards for the `phasetools` package. All AI agents (Gemini, Claude, etc.) must adhere to these rules.

## 1. Core Development Principles

### 1.1 Language and Localization
- **British English:** All code comments, docstrings, variable names, and user-facing strings MUST use British English.
  - *Correct:* `colour`, `standardise`, `modelling`, `crystallisation`, `organisation`, `recognise`.
  - *Incorrect:* `color`, `standardize`, `modeling`, `crystallization`, `organization`, `recognize`.

### 1.2 Scientific Principles: MAGEMin and Thermodynamics
Every calculation or model class MUST inherit from `MAGEMinBase`.
- **Standardised Setup:** Use `setup_bulk_composition` to handle chemical initialisation. This method automatically standardises the oxide list and ensures unit consistency via `convertBulk4MAGEMin`.
- **Iron and Oxygen Basis:** MAGEMin standardises inputs to a `FeO` and `O` (Excess Oxygen) basis for most databases (e.g., `ig`, `mp`). This redistribution (1 mol `Fe2O3` → 2 mol `FeO` + 1 mol `O`) correctly reflects the engine's stoichiometric requirements for redox modelling.
- **Mantle Iron (sb24):** For the `sb24` database, the chemical system is `NCFMASO-Cr` and includes pure stoichiometric iron phases (e.g., `Fe(a)`).
- **Units:** Pressure is strictly in `kbar` (10 kbar = 1 GPa) and Temperature is in `°C` for all user-facing inputs and calculator methods.
- **Numerical Stability:** To ensure solver convergence, core oxides (e.g., `SiO2`, `Al2O3`, `MgO`) are clamped to a minimum molar fraction of $10^{-4}$. Optional oxides (e.g., `TiO2`, `MnO`, `H2O`) are set to $0.0$ if their absolute value is below $2 \cdot 10^{-5}$.
- **Unit Normalisation:** Compositions are converted to molar fractions if provided in `wt%` and then normalised to sum to $100.0$.
- **Phase Property Extraction:** Agents MUST use the standardised utility functions in `src/phasetools/core/phase_properties.py`.
  - **Handling Solvi:** Use `phase_frac` to retrieve fractions; it automatically sums multiple instances of the same phase.
  - **Mg# Logic:** Use `get_phase_mg_number` to ensure atomic consistency across `FeO`, `Fe2O3`, and metal `Fe`.
  - **Kd Calculations:** Use `calculate_kd_fe_mg` for distribution coefficients.

### 1.3 Databases and Constraints
Agents MUST select the appropriate thermodynamic database, referring to the [official MAGEMin documentation](https://computationalthermodynamics.github.io/MAGEMin_C.jl/dev/) for calibration details.

#### 1.3.1 Standard Databases
- **Metapelite (mp):** `K2O-Na2O-CaO-FeO-MgO-Al2O3-SiO2-H2O-TiO2-O-MnO`
- **Ultramafic (um):** `SiO2-Al2O3-MgO-FeO-O-H2O-S`
- **Metabasite (mb):** `K2O-Na2O-CaO-FeO-MgO-Al2O3-SiO2-H2O-TiO2-O`
- **Mantle (mtl, sb11, sb21):** `Na2O-CaO-FeO-MgO-Al2O3-SiO2`
- **Igneous (ig):** `K2O-Na2O-CaO-FeO-MgO-Al2O3-SiO2-H2O-TiO2-O-Cr2O3`
- **Igneous alkaline (igad):** `K2O-Na2O-CaO-FeO-MgO-Al2O3-SiO2-TiO2-O-Cr2O3`
- **Mantle (sb24):** `Na2O-CaO-FeO-MgO-Al2O3-SiO2-O-Cr2O3`

#### 1.3.2 Extended Databases
Extended databases merge multiple standard datasets to allow for "cross-lithology" modelling and expand the list of potential stable mineral phases.
- **`mpe` (Metapelite Extended):** Combines `mp` with `mb` and other models.
- **`mbe` (Metabasite Extended):** Adds high-pressure/specialised solution models to `mb`.
- **`ume` (Ultramafic Extended):** Combines `um` with `mb` for mantle-crust interactions.

#### 1.3.3 Database References
- **Metapelite (mp/mpe):** White et al. (2014), *J. Metamorph. Geol.*, 32, 261-286.
- **Metabasite (mb/mbe):** Green et al. (2016), *J. Metamorph. Geol.*, 34, 845-869.
- **Ultramafic (um/ume):** Evans & Frost (2021), *J. Petrol.*, 62(3), egab016.
- **Igneous (ig):** Green et al. (2025), *J. Petrol.*, 66, egae079.
- **Igneous Alkaline (igad):** Weller et al. (2024), *J. Petrol.*, 65, egae098.
- **Mantle (mtl):** Holland et al. (2013), *J. Petrol.*, 54, 1901-1920.
- **Mantle (sb11/21/24):** Stixrude & Lithgow-Bertelloni (2011/2021/2024), *Geophys. J. Int.*

**Applicability Warning:** All datasets are calibrated for limited P-T-X ranges. Exceeding these limits leads to non-convergence or inaccuracies. For databases without an `O` component (e.g., `mtl`), `FeO` is treated as total iron.

---

## 2. Repository Architecture and Standards

### 2.1 Modular Structure
The package follows a strictly modular "library-style" structure.

#### Core Logic (`src/phasetools/core/`)
- **`base.py`:** Contains `MAGEMinBase`, the foundational class for all thermodynamic operations. It manages bulk composition initialisation, unit standardisation (molar fractions), and state sharing between instances (`_copy_state_to`).
- **`engine.py`:** Low-level bridging logic. Handles automatic Python-to-Julia type conversion for `MAGEMin_C` minimisation routines.
- **`phase_properties.py`:** Standard utilities for extracting data from MAGEMin output objects, including phase fractions, chemistry (APFU), Mg#, and Kd values.

#### Specialised Calculators (`src/phasetools/calculators/`)
- Specialised classes for specific tasks like Garnet chemistry, multi-phase assemblages, P-T grid generation, and geothermobarometry estimation.

#### Multi-step Models (`src/phasetools/models/`)
- Complex simulations that chain calculations, such as `GarnetGenerator` (fractional growth) and `MagmaOceanModel` (planetary differentiation).

#### Utilities (`src/phasetools/utils/`)
- **`bulk_rock.py`:** Chemistry and stoichiometry engine. Provides mass resolution via `molmass` and unit conversion helpers (e.g., wt% to mol% and vice versa).
- **`general.py`:** Mathematical helpers, including monotonic P-T-t path interpolation using PCHIP.

### 2.2 Public API
Maintain a clean public API in `src/phasetools/__init__.py`. High-level classes MUST be accessible directly via `from phasetools import ...`.

### 2.3 Technical Standards
- **Dependencies:** `juliacall`, `molmass`, `numpy`, `scipy`.
- **Julia Safety:** Avoid double-converting objects. Verify `juliacall` wrappers before using `jlconvert`.
- **State Sharing:** Use `self._copy_state_to(other_instance)` to transfer standardised Julia objects.
- **Type Safety:** Use type hints for all public methods.
- **Documentation:** Use the NumPy docstring format.
- **Testing:** New features MUST include a verification script in the `tests/` directory.

---

## 3. Calculators
Calculators are specialized classes for performing thermodynamic minimisations and extracting specific phase properties.

### 3.1 P-T Grid Calculator (`MAGEMinPTGridCalculator`)
Handles multi-point minimisations over pressure-temperature grids.
- **Property Extraction:** Use `extract_from_grid` to retrieve structured NumPy arrays for phase fractions, oxide chemistry (APFU), element concentrations, and end-members.
- **End-member Discovery:** Supports `end_members='auto'` to automatically discover stable end-members from the result object.

### 3.2 Garnet Calculator (`MAGEMinGarnetCalculator`)
Extends the grid calculator with high-level wrappers specifically for garnet chemistry.
- **X-Site Normalisation:** Uses a robust Fe-split method to strictly isolate divalent iron (`Fe2+`), ensuring that Mg-Mn-Fe-Ca fractions for the divalent (X) site sum to 1.0.
- **Path Calculations:** `gt_along_path` computes garnet evolution along P-T trajectories, optionally including fractionation.

### 3.3 P-T Estimator (`PhasePTEstimator`)
Generalized utility for geothermobarometry by minimizing the misfit between measured phase chemistry and equilibrium predictions.
- **Optimisation:** Supports multiple `scipy.optimize` solvers, including global methods (`differential_evolution`, `dual_annealing`) and local methods (`shgo`, `minimize`).

### 3.4 Assemblage Calculator (`MAGEMinAssemblageCalculator`)
Maps the stability fields of multi-mineral assemblages.
- **Coexistence Masking:** Returns boolean stability masks where all requested phases are simultaneously stable.

### 3.5 Phase Search (`PhaseSearchCalculator`)
Uses root-finding algorithms to locate precise phase stability boundaries.
- **Appearance/Saturation:** Finds conditions for phase appearance (solidus) or phase saturation (liquidus) given a pressure and temperature bracket.

---

## 4. Models
Models are complex multi-step simulations that chain thermodynamic calculations to simulate geological processes.

### 4.1 Garnet Growth Generator (`GarnetGenerator`)
Simulates the growth of synthetic garnet populations with compositional zoning.
- **Radial Zoning:** Models zoning across a user-defined number of radial shells.
- **Cohort Modelling:** Generates distributions of garnet sizes (size classes) and associated formation times.
- **Fractionation:** Optionally fractionates garnet from the bulk composition at each growth step to simulate Rayleigh-style fractionation.

### 4.2 Magma Ocean Model (`MagmaOceanModel`)
Simulates the crystallisation of a Magma Ocean on a rocky body (e.g., the Moon).
- **Multi-Stage Crystallisation:** Transitions from stage 0 (equilibrium crystallisation) to stages 1-N (fractional crystallisation in discrete volume steps).
- **Geophysical Integration:** Includes pressure-to-depth and radius-to-pressure conversions based on body-specific parameters (radius, gravity, density).

