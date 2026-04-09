# AI Development Mandate: phasetools

This document defines the foundational requirements and architectural standards for the `phasetools` package. All AI agents (Gemini, Claude, etc.) must adhere to these rules.

## 1. Language and Localization
- **British English:** All code comments, docstrings, variable names, and user-facing strings MUST use British English.
  - *Correct:* `colour`, `standardise`, `modelling`, `crystallisation`, `organisation`, `recognise`.
  - *Incorrect:* `color`, `standardize`, `modeling`, `crystallization`, `organization`, `recognize`.

## 2. Repository Architecture
The package follows a strictly modular "library-style" structure. Do not introduce flat scripts or "God Object" files.
- `src/phasetools/core/`: Low-level thermodynamic engine wrappers and base classes.
- `src/phasetools/calculators/`: Specialised classes for geological tasks (Garnet, Assemblage, P-T).
- `src/phasetools/models/`: Complex multi-step simulations (e.g., `GarnetGenerator`).
- `src/phasetools/utils/`: Chemistry, stoichiometry, and general math helpers.

## 3. Core Principles: MAGEMinBase
Every calculation or model class MUST inherit from `MAGEMinBase`.
- **Standardised Setup:** Use `setup_bulk_composition` to handle chemical initialisation. This method automatically standardises the oxide list and ensures unit consistency via `convertBulk4MAGEMin`.
- **State Sharing:** Never manually sync attributes between instances. Use the `self._copy_state_to(other_instance)` method to transfer standardised Julia objects and stoichiometry.
- **Julia Safety:** Avoid double-converting objects. Always check if a variable is already a `juliacall` wrapper before using `jlconvert`.

## 4. Technical Standards
- **Dependencies:** Core logic depends on `juliacall` (Julia bridge), `molmass` (stoichiometry), `numpy`, and `scipy`.
- **Type Safety:** Use type hints for all public methods.
- **Documentation:** Use the NumPy docstring format.
- **Testing:** New features MUST include a verification script or test case in the `tests/` directory.

## 5. Public API
Maintain a clean public API in `src/phasetools/__init__.py`. High-level classes should be accessible directly via `from phasetools import ...`.
