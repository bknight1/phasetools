# Utilities Submodule

The `utils` submodule contains helper functions for chemistry, math, and general data processing.

## Key Components

### `bulk_rock.py`
- Core stoichiometric engine for managing bulk compositions.
- Provides molar mass lookups and unit conversion helpers (e.g., mol% to wt% and vice versa).
- Standardises oxide lists and manages cation-oxide mapping.

### `general.py`
- General mathematical utilities.
- Includes Monotonic Cubic Interpolation (PCHIP) for smooth P-T-t path generation and data resampling.
