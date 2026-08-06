# Utilities Submodule

The `utils` submodule contains helper functions for chemistry, math, and general data processing.

## Key Components

### `bulk_rock.py`
- Core stoichiometric engine for managing bulk compositions.
- Provides molar mass lookups and unit conversion helpers (e.g., mol% to wt% and vice versa).
- `mol_fractions_to_wt_fractions` / `wt_fractions_to_mol_fractions`: non-normalising converters for single components or subsets (e.g., FeO/Fe2O3/FeOt iron redox conversions).
- `split_feot_to_feo_o` / `express_bulk_in_feo_o_basis`: split total iron (FeOt) into the MAGEMin **FeO + O** redox basis at a target Fe³⁺/FeOt fraction, **conserving FeOt** (FeO column = total Fe; O = f·FeOt/2 per 2FeO + O → Fe₂O₃). Useful for redox sweeps and for converting measured (FeOt-only) bulks into MAGEMin format.
- Standardises oxide lists and manages cation-oxide mapping.

### `general.py`
- General mathematical utilities.
- Includes Monotonic Cubic Interpolation (PCHIP) for smooth P-T-t path generation and data resampling.
