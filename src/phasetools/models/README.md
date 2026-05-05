# Models Submodule

Models are complex, multi-step simulations that chain multiple thermodynamic calculations to simulate dynamic geological and planetary processes.

## Key Components

### `garnet_growth.py`
- **`GarnetGenerator`**: Simulates the fractional growth of garnet crystals.
- Handles radial shell zoning, cohort formation times, and size-frequency distributions.

### `magma_ocean.py`
- **`MagmaOceanModel`**: Simulates the cooling and solidification of a planetary magma ocean.
- Transitions between equilibrium and fractional crystallisation stages.
- Integrates planetary-scale geophysical parameters (radius, gravity) to calculate pressure-depth relationships.
