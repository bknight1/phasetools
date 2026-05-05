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
