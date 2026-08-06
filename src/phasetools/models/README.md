# Models Submodule

Models are complex, multi-step simulations that chain multiple thermodynamic calculations to simulate dynamic geological and planetary processes.

## Key Components

### `garnet_growth.py`
- **`GarnetGenerator`**: Simulates the fractional growth of garnet crystals.
- Handles radial shell zoning, cohort formation times, and size-frequency distributions.
- When `fractionate=True`, garnet is removed from the reactive bulk at each P-T step.  The fractionation amount is based on the appropriate unit basis (mol or wt) matching `sys_in`.
- `get_retrograde_concentrations` recalculates retrograde compositions using the bulk at last growth (fixed-bulk assumption — garnet resorption during retrograde is not modelled).

### `magma_ocean.py`
- **`MagmaOcean`**: Simulates the cooling and solidification of a planetary magma ocean.
- Transitions between equilibrium and fractional crystallisation stages.
- Integrates planetary-scale geophysical parameters (radius, gravity) to calculate pressure-depth relationships.
- `run_fractional_stages` preserves and restores the instance's bulk composition (`self.X`) after execution, so repeated calls do not mutate state.
