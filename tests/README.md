# Phasetools Test Suite

Verification scripts and unit tests for the thermodynamic and chemical logic in `phasetools`. Most tests are mock-based (`unittest.mock.patch`) and require no live Julia runtime. Two files (`test_magma_ocean.py` and `test_phase_properties_live.py`) require Julia + MAGEMin_C and are skipped automatically when the runtime is unavailable.

---

## `test_phase_properties.py`

**What this file proves:** The Fe²⁺/Fe³⁺ splitting heuristics and FeOt-based redox calculations produce physically consistent results across multiple mineral phases, databases, and unit conventions.

### Key tests

- **Garnet Fe³⁺ splitting (excess-O basis):** Validates that garnet Fe³⁺ is correctly computed from the excess oxygen component. In MAGEMin's O basis, `Fe₂O₃ → 2FeO + O`, so garnet Fe³⁺ ∝ excess O allocated to garnet.
- **Clinopyroxene Fe³⁺ splitting:** Validates Fe³⁺ for cpx phases where it matters (e.g., acmite component NaFe³⁺Si₂O₆). Cpx Fe³⁺ affects Jadeite vs. Acmite end-member proportions.
- **Fe₂O₃ convention handling:** Tests that when databases or inputs use the traditional Fe₂O₃ convention (rather than O basis), the conversion preserves total Fe and Fe³⁺/FeOt.
- **sb24 iron handling:** Tests iron splitting for the sb24 mantle database, which includes metallic iron (`Fe(a)`) as a phase — requires different bookkeeping than silicate-only databases.
- **Mg# robustness:** Validates that Mg# = Mg/(Mg+Fe²⁺) is computed correctly across different Fe components and databases. Mg# is the primary compositional metric for comparing model output to natural samples.
- **K_D Fe-Mg logic:** Tests that the distribution coefficient K_D = (Fe²⁺/Mg)ᵠᵃʳⁿᵉᵗ / (Fe²⁺/Mg)ᶜᵖˣ is correctly calculated. K_D > 1 means garnet concentrates Fe²⁺ relative to cpx; K_D < 1 means the opposite.

---

## `test_garnet.py`

**What this file proves:** The garnet calculator's `fe_basis` flag, `gt_along_path` fractionation logic, and end-member extraction are correct.

### Key tests

- **FeOt vs Fe²⁺ basis:** Tests that `fe_basis='FeOt'` uses total iron (FeO + 2·Fe₂O₃) on the X-site and normalises to 1, while `fe_basis='Fe2+'` uses only the ferrous split. Both modes are needed depending on whether garnet Fe³⁺ is significant.
- **`gt_along_path` fractionation:** Validates that garnet fractionation along a P–T path produces correct cumulative modes and X-site compositions, and that `X_along_path` is properly row-normalised.
- **End-member extraction:** Confirms that garnet end-member fractions (pyrope, almandine, grossular, spessartine, kho) are correctly returned.

---

## `test_phase_search.py`

**What this file proves:** `PhaseFunctions.fractionate_phase` handles pure phases (PP_vec), solution phases (SS_vec), and zero-fraction edge cases without errors.

### Key tests

- **Pure-phase IndexError fix:** Validates that `fractionate_phase` does not crash when all phases in the assemblage are pure (no solution phases in SS_vec).
- **Solution-phase path:** Confirms that the standard solution-phase fractionation path still works correctly.
- **Zero fraction returns unchanged bulk:** When `frac_amount=0`, the bulk composition is returned unmodified.

---

## `test_grid_extraction.py`

**What this file proves:** Solvus (multi-instance phase) handling in the low-level composition helpers and grid-level extraction functions is correct.

### Key tests

- **Instance indexing:** Tests that `get_oxide_apfu`, `extract_end_member`, `get_phase_chemistry`, `get_phase_mg_number`, `get_phase_mg2_number`, and `get_phase_fe_split` correctly handle `instance` parameter (integer index or `'all'`).
- **`extract_from_grid` solvus bundles:** Validates that grid extraction returns a list of per-instance bundles (`res[0]`, `res[1]`), with unsuffixed keys and NaN where a phase or limb is absent.
- **`single_point_calc` solvus bundles:** Same bundle semantics for single-point calculations.
- **Phase index helper (`_phase_indices`):** Tests that repeated phase keys are correctly mapped to their SS_vec or PP_vec indices.

---

## `test_bulk_rock.py`

**What this file proves:** The non-normalising mole/weight fraction converters and FeOt conversion utilities are numerically correct.

### Key tests

- **Scalar and list converters:** Tests that `mol_fractions_to_wt_fractions` and `wt_fractions_to_mol_fractions` handle both scalar and list inputs, with correct molar mass multiplication and no unwanted normalisation.
- **Round-trip consistency:** Confirms that mol → wt → mol (and vice versa) recovers the original values.
- **FeOt conversion:** Tests `split_feot_to_feo_o` and `express_bulk_in_feo_o_basis` — splitting FeOt into FeO + O at a target Fe³⁺/FeOt ratio while conserving total FeOt.

---

## `test_magma_ocean.py` (live — requires Julia + MAGEMin_C)

**What this file proves:** The `MagmaOcean` model's crystallisation path — equilibrium stage 0 followed by fractional stages 1–N — reproduces expected lunar magma ocean (LMO) behaviour.

### Key tests

- **Stage 0 equilibrium:** Validates that the equilibrium crystallisation of a Taylor Whole Moon (TWM) composition produces reasonable melt and solid fractions over the 45–17 kbar range.
- **Fractional stages 1–10:** Checks that the 10-stage fractional crystallisation path produces the correct number of stages, expected base pressures, and appropriate phase appearances (spinel in intermediate stages, ilmenite in final stages).
- **Geometric evolution:** Verifies that the pressure-to-radius mapping produces physically plausible layer thicknesses for a 1737 km radius body with a 330 km core.

---

## `test_phase_properties_live.py` (live — requires Julia + MAGEMin_C)

**What this file proves:** The heuristic Fe²⁺/Fe³⁺ split matches independent thermodynamic references (end-member stoichiometry and site fractions) exactly, and that derived ratios (K_D, Mg#) are invariant to the bulk composition unit system (wt% vs. mol%).

### `TestSiteOccupancy` (mpe database, S10 eclogite composition)

**What this class proves:** For the legacy mpe database with a well-characterised eclogite composition, garnet and clinopyroxene heuristic Fe splitting is consistent with end-member stoichiometry across a range of P–T conditions.

#### Tests

- **`test_garnet_heuristic_across_pt`**
  - **Checks:** Garnet Fe²⁺ = 3 × almandine fraction and Fe³⁺ = 2 × kho fraction (exact equality) across five P–T points.
  - **Physical meaning:** Proves the heuristic correctly implements the a-x model's Fe split for garnet. In garnet (X₃Y₂Si₃O₁₂), the 3 Fe²⁺ atoms occupy the X-site and the 2 Fe³⁺ atoms occupy the Y-site. The heuristic recovers this site-specific partitioning without hard-coding end-member names.

- **`test_heuristic_mg_matches_site_fractions`**
  - **Checks:** Mg# computed from the heuristic matches Mg# computed from `siteFractions` (summed across all crystallographic sites) for both garnet and cpx, to floating-point precision.
  - **Physical meaning:** Mg# = Mg/(Mg+Fe²⁺) is the most commonly reported mineral compositional metric. Agreement proves the heuristic correctly partitions Fe between Mg and Fe at the site level, not just bulk level.

### `TestHeuristicAcrossDatabases` (ig, mp, mb)

**What this class proves:** The Fe²⁺ + Fe³⁺ = FeOt closure relation holds across the three main solution-model families — regardless of which database is used, total Fe is conserved by the heuristic.

### `TestMbN_MORB` (mb database, N-MORB composition — Gale et al., 2013)

**What this class proves:** For a standard mid-ocean ridge basalt composition, the heuristic Fe split exactly reproduces the a-x model's end-member fractions and site fractions, and these results are independent of whether the input is wt% or mol%.

#### Tests

- **`test_kd_mg_independent_of_units`**
  - **Checks:** K_D and Mg# are identical (within floating-point tolerance) when the same composition is run as both wt% and mol%.
  - **Physical meaning:** K_D and Mg# are intensive ratios — they depend on phase chemistry, not the bulk unit system. If they differ between wt% and mol%, there is a unit conversion bug.

---

## Running tests

```bash
# Run all tests (skips live tests when Julia is unavailable)
python3 -m unittest discover tests

# Run a specific file
python3 -m unittest tests.test_phase_properties -v

# Run a specific class
python3 -m unittest tests.test_phase_properties_live.TestMbN_MORB -v

# Run a specific test
python3 -m unittest tests.test_phase_properties_live.TestMbN_MORB.test_kd_mg_independent_of_units -v
```
