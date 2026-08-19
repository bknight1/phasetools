# AGENTS.md

## Quick start

**Conda / mamba (Linux & CI):**
```bash
mamba env create -f environment.yml
conda activate phasetools
phasetools-julia-setup --install
phasetools-julia-setup --check
```

**pip + juliaup (cross-platform, incl. macOS):**
```bash
python -m pip install -e .
phasetools-julia-setup --install  # install MAGEMin_C in Julia
phasetools-julia-setup --check    # verify Julia + MAGEMin_C
python3 -m unittest discover tests   # run all tests (mock-based, no Julia needed)
```

## Package structure

```
src/phasetools/
  __init__.py        — initialises MAGEMin_C via juliacall; exports public API
  julia_setup.py     — CLI for phasetools-julia-setup (check/install Julia + MAGEMin_C)
  core/
    base.py          — MAGEMinBase (parent class for all calculators/models)
    engine.py        — single_point_minimization_with_conversion (Python→Julia bridge)
    phase_properties.py  — phase_frac, extract_end_member, get_oxide_apfu,
                           get_phase_chemistry, get_phase_mg_number,
                           get_phase_mg2_number, get_phase_fe_split,
                           calculate_kd_fe_mg
  calculators/
    pt_grid.py       — MAGEMinPTGridCalculator (multi-point P-T grids)
    garnet.py        — MAGEMinGarnetCalculator (garnet-focused wrappers)
    phase_search.py  — PhaseFunctions (solidus/liquidus root-finding)
    pt_estimation.py — PhasePTEstimator (geothermobarometry)
    assemblage.py    — MAGEMinAssemblageCalculator (stability masks)
  models/
    garnet_growth.py — GarnetGenerator, generate_distribution
    magma_ocean.py   — MagmaOcean
  utils/
    bulk_rock.py     — mol↔wt conversions, molar mass dicts, atomic_frac_to_wt_frac,
                       non-normalising mol↔wt fraction converters (single-component
                       or subset conversions, e.g. FeO/Fe2O3/FeOt),
                       split_feot_to_feo_o / express_bulk_in_feo_o_basis
                       (FeOt→FeO+O split at a target Fe³⁺/FeOt, conserving FeOt)
    general.py       — PCHIP interpolation helpers
```

## Conventions (must follow)

- **British English** throughout: `colour`, `standardise`, `crystallisation`, `modelling`, `organisation`, `recognise`.
- **NumPy docstrings** and **type hints** on all public methods.
- **P in kbar, T in °C** for all user-facing inputs.
- Garnet phase key is always `'g'`.
- Every calculator or model class **must inherit from `MAGEMinBase`**.
- High-level classes accessible via `from phasetools import ...`.

## Julia bridge (critical)

```python
from juliacall import Main as jl, convert as jlconvert
```

- Convert arrays/lists before MAGEMin_C calls:
  `jlconvert(jl.Vector[jl.Float64], np_array)` or `jlconvert(jl.Vector[jl.String], list_of_names)`
- **Avoid double-converting** objects already in Julia format — verify the `juliacall` wrapper.
- Initialise once and **reuse the `data` handle**:
  `data = MAGEMin_C.Initialize_MAGEMin("ig", verbose=False)`
- Share state across instances: `self._copy_state_to(other)` transfers standardised Julia objects.

## Databases

### Standard databases

| Label | Chemical system |
|-------|----------------|
| `ig` (igneous) | `K2O-Na2O-CaO-FeO-MgO-Al2O3-SiO2-H2O-TiO2-O-Cr2O3` |
| `igad` (igneous alkaline) | `K2O-Na2O-CaO-FeO-MgO-Al2O3-SiO2-TiO2-O-Cr2O3` |
| `mp` (metapelite) | `K2O-Na2O-CaO-FeO-MgO-Al2O3-SiO2-H2O-TiO2-O-MnO` |
| `mb` (metabasite) | `K2O-Na2O-CaO-FeO-MgO-Al2O3-SiO2-H2O-TiO2-O` |
| `um` (ultramafic) | `SiO2-Al2O3-MgO-FeO-O-H2O-S` |
| `mtl` / `sb11` / `sb21` (mantle) | `Na2O-CaO-FeO-MgO-Al2O3-SiO2` |
| `sb24` (mantle) | `Na2O-CaO-FeO-MgO-Al2O3-SiO2-O-Cr2O3` (includes metal Fe phases like `Fe(a)`) |

### Extended databases

- **`mpe`**: mp + mb (cross-lithology metapelite)
- **`mbe`**: mb + high-pressure models
- **`ume`**: um + mb (mantle-crust interactions)

### Applicability warning

All datasets calibrated for limited P-T-X ranges — exceeding limits causes non-convergence or inaccuracies. For databases without `O` (e.g., `mtl`), `FeO` is treated as total iron.

## Bulk composition

Use `setup_bulk_composition(Xoxides, X, sys_in, rm_list=None)` (from `MAGEMinBase`). It calls `MAGEMin_C.convertBulk4MAGEMin` to standardise, builds a stoichiometry map, and optionally removes phases.

- If `sys_in='wt'`: standardised molar output is converted back to wt%.
- Otherwise stays in molar fractions, normalised to sum 100.0.
- Core oxides (SiO₂, Al₂O₃, MgO) clamped to min **1e-4** molar fraction.
- Optional oxides (TiO₂, MnO, H₂O etc.) set to 0.0 if < **2e-5**.
- `phase_frac()` handles solvi — automatically sums multiple instances of the same phase.

### Unit systems

When comparing MAGEMin model output to measured data (e.g., EPMA), **convert everything to a consistent unit system** using `MAGEMin_C.convertBulk4MAGEMin`:

```python
x_jl = jlconvert(jl.Vector[jl.Float64], wt_list)
ox_jl = jlconvert(jl.Vector[jl.String], ox_names)
x_mol_jl, xox_mol_jl = MAGEMin_C.convertBulk4MAGEMin(x_jl, ox_jl, "wt", db)
```

- Pad missing oxides with zeros when targets have fewer oxides than the bulk rock.
- Common approach: set `sys_in='mol'` on the calculator and pre-convert the bulk rock from wt% to mol% before calling `setup_bulk_composition`.
- X-site (cation) fractions are **always atomic/molar** — the `cat_*` fields from `extract_from_grid` return atomic fractions regardless of `sys_in`. Use `atomic_frac_to_wt_frac()` (in `bulk_rock.py`) to convert if weight-based fractions are needed.

## Redox model

- MAGEMin uses **FeO + O (excess oxygen)** basis for most databases (`ig`, `mp`, etc.).
- `1 mol Fe2O3 → 2 mol FeO + 1 mol O`
- `get_phase_fe_split(out, phase)` for heuristic Fe²⁺/Fe³⁺ partitioning.
- `get_phase_mg_number` atomic across Fe, FeO, Fe2O3 components.
- `calculate_kd_fe_mg(out, phase1, phase2)` for distribution coefficients.
- **Garnet X-site fractions default to FeOt** (all Fe as Fe²⁺): `MAGEMinGarnetCalculator` / `GarnetGenerator` take a `fe_basis` argument (`'FeOt'` default, `'Fe2+'` for the stoichiometric ferrous-only split). FeOt is the community convention for garnet end-members (e.g. Williams & Grambling 1990; Krogh Ravna 2000); use `'Fe2+'` only when garnet Fe³⁺ is known to be significant.
- sb24 uses pure stoichiometric iron (`Fe(a)`) with excess O.

## Endmember formulae

> **Source:** Extracted from MAGEMin source code at [`ComputationalThermodynamics/MAGEMin`](https://github.com/ComputationalThermodynamics/MAGEMin):
> - TC database endmember entries: [`src/TC_database/TC_endmembers.c`](https://github.com/ComputationalThermodynamics/MAGEMin/blob/main/src/TC_database/TC_endmembers.c)
> - SB database endmember entries: [`src/SB_database/SB_endmembers.c`](https://github.com/ComputationalThermodynamics/MAGEMin/blob/main/src/SB_database/SB_endmembers.c)
> - Endmember listing from Python: `calc.get_phase_endmembers(phase, grid_out)`

### Oxide index ordering

The `Comp[]` array in endmember database entries uses a **database-specific** oxide index ordering:

| Database family | Oxide ordering (index: oxide) | Used by |
|----------------|-------------------------------|---------|
| **TC** | `0:SiO₂, 1:Al₂O₃, 2:CaO, 3:MgO, 4:FeO, 5:K₂O, 6:Na₂O, 7:TiO₂, 8:O, 9:MnO, 10:Cr₂O₃, 11:H₂O, 12:CO₂` | `ig`, `igad`, `mp`, `mb`, `mpe`, `mbe`, `um`, `ume` |
| **SB** sb11/sb21 | `0:SiO₂, 1:CaO, 2:Al₂O₃, 3:FeO, 4:MgO, 5:Na₂O` | `mtl`, `sb11`, `sb21` |
| **SB** sb24 | `0:SiO₂, 1:CaO, 2:Al₂O₃, 3:MgO, 4:Na₂O, 5:O, 6:Cr₂O₃, 7:Fe` | `sb24` |

The last element of `Comp[]` (index 15 for TC, index 17 for SB) is the **total number of atoms per formula unit**.

### Key endmember formulae

```
Garnet (g):       py  = Mg3Al2Si3O12        alm = Fe3Al2Si3O12
                  gr  = Ca3Al2Si3O12        spss = Mn3Al2Si3O12

Cpx (dio):        jd  = NaAlSi2O6           di  = CaMgSi2O6
                  hed = CaFeSi2O6           acm = NaFe³⁺Si2O6  (with O)

Opx (opx):        en  = Mg2Si2O6            fs  = Fe2Si2O6
                  mgts = MgAlAlSiO6

Feldspar (fsp):   ab  = NaAlSi3O8           an  = CaAl2Si2O8
                  san = KAlSi3O8

Amphibole (amp):  tr  = Ca2Mg5Si8O22(OH)2
                  ts  = Ca2Mg3Al2Si6O22(OH)2
                  gl  = Na2Mg3Al2Si8O22(OH)2
                  parg = NaCa2Mg4Al3Si6O22(OH)2

Mica (bi):        phl = KMg3AlSi3O10(OH)2
                  ann = KFe3AlSi3O10(OH)2
                  east = KMg2Al3Si2O10(OH)2

Mica (mu):        mu  = KAl3Si3O10(OH)2
                  cel = KMgAlSi4O10(OH)2
                  fcel = KFeAlSi4O10(OH)2
```

### Critical pitfall: solution-model pseudo-endmembers

Some endmembers returned by `get_phase_endmembers(phase, out)` are **not in `TC_endmembers.c`**. They are **pseudo-endmembers** (ordering parameters / intermediate compositions) defined programmatically in the solution model C code (`src/TC_database/SS_xeos_PC_*.c`).

**Known pseudo-endmembers:**
| Phase | Pseudo-endmember | Notes |
|-------|-----------------|-------|
| cpx (dio) | `om` (omphacite) | ~Na₀.₅Ca₀.₅ intermediate — carries BOTH Na and Ca |
| cpx (dio) | `acmm` | Variant of acmite (NaFe³⁺Si₂O₆), model-specific |
| cpx (dio) | `cfm` | Ca-Fe-Mg ordering component |
| cpx (dio) | `jac` | Jadeite-acmite intermediate |

**APFU vs endmembers: when to use which:**
| Need | Use | Why |
|------|-----|-----|
| Element ratios (XJd, XMg, XAn) | **APFU** via `oxides=[...]` or `cations=[...]` | Reads structural formula directly; avoids pseudo-endmember misclassification |
| Activity/composition modelling | **Endmembers** via `end_members=[...]` | Endmembers are the thermodynamic mixing components |
| Verifying endmember assignments | **Both + least squares** | `APFU = Σ em_frac × Comp` — if the fit fails, your endmember set or formula assignment is wrong |

Since endmembers are just oxide arrays in the database, APFU already encodes the same information without the risk of summing the wrong subset. For composition comparison with measured data, **APFU is always simpler and more reliable.**

## Calculator API patterns

- **PTGrid**: `calculate_grid(P, T)` → `extract_from_grid(phase, end_members='auto')`
- **Garnet**: `gt_along_path(P, T, fractionate=True)` for evolution with zoning; X-site normalisation isolates Fe²⁺.
- **PTEstimator**: wraps `scipy.optimize` (global: `differential_evolution`, `dual_annealing`; local: `shgo`, `minimize`).
- **Assemblage**: returns boolean stability masks for requested phase coexistence.
- **PhaseSearch**: uses `scipy.optimize.root_scalar` with P,T brackets for solidus/liquidus.

## Comparing EPMA data with MAGEMin outputs

### Extraction methods from `extract_from_grid`

**Note:** `extract_from_grid` now returns a list of per-instance bundles (one per coexisting solvus limb). Each bundle `res[k]` contains the following keys. For phases that appear only once, `res[0]` is the only element. The `instance=` parameter from the intermediate API has been dropped — the list-of-bundles shape supersedes it.

| Key pattern | Source | Units | What it returns |
|-------------|--------|-------|----------------|
| `ox_apfu_{oxide}` | `oxides=[...]` → `get_oxide_apfu()` | Atoms per formula unit | Structural formula on a per-formula-unit basis (normalised to the phase's oxygen count, e.g. 6 O for cpx) |
| `chem_{oxide}` | `chemistry=[...]` → `get_phase_chemistry()` | mol% (if `sys_in='mol'`) or wt% (if `sys_in='wt'`) | Oxide concentration in the phase, on the same basis as `sys_in` |
| `cat_{cation}` | `cations=[...]` → `_extract_cations_from_apfu()` | Atomic fraction (0–1) | Cation site fractions, always atomic regardless of `sys_in` |

**For comparing with EPMA, use `ox_apfu_*` (APFu)** — this is the standard mineralogical normalisation (atoms per formula unit, typically 6 O for pyroxene, 12 O for garnet, 22 O for amphibole) and matches how EPMA data is conventionally reported.

### Scale-invariant ratios (always prefer these)

Ratios like XJd = Na/(Na+Ca) and Mg# = Mg/(Mg+Fe) are **independent of the normalisation basis** — the denominator cancels whether you use APFu, mol%, or wt%. They are the safest quantities to compare between model and measurement:

```python
# From APFu (model side)
xjd = ox_apfu_Na2O / (ox_apfu_Na2O + ox_apfu_CaO)

# From chemistry (model side) — same result
xjd = (2 * chem_Na2O) / (2 * chem_Na2O + chem_CaO)   # ×2 for Na atoms

# From mol% columns (EPMA side) — same result if done correctly
xjd = (2 * Na2O_mol%) / (2 * Na2O_mol% + CaO_mol%)
```

All three give identical values (verified to machine precision). The `×2` on Na₂O converts from oxide molecules to Na atoms (2 Na per Na₂O vs 1 Ca per CaO).

### Critical pitfall: oxide-molecule vs atom ratios

**Do not compute element ratios from oxide molecule fractions without accounting for stoichiometry.** Na₂O has 2 Na atoms per molecule; CaO has 1 Ca atom. The oxide-molecule ratio Na₂O/(Na₂O+CaO) is systematically ~half the correct cation ratio Na/(Na+Ca).

| Formula | Convention | XJd for typical omphacite |
|---------|-----------|---------------------------|
| `Na2O_mol% / (Na2O_mol% + CaO_mol%)` | Oxide molecules | **0.14–0.18** (WRONG) |
| `(2 × Na2O_mol%) / (2 × Na2O_mol% + CaO_mol%)` | Atoms (correct) | **0.25–0.31** |
| `ox_apfu_Na2O / (ox_apfu_Na2O + ox_apfu_CaO)` | APFu atoms (correct) | **0.25–0.31** |

Mg# = MgO/(MgO+FeO) is **not affected** because both oxides carry 1 cation per molecule.

### Critical pitfall: wt_to_apfu oxygen counting

When converting EPMA wt% to APFu, count **oxygen atoms per oxide molecule**, not the number of capital-O letters:

| Oxide | `ox.count("O")` (WRONG) | Correct O count |
|-------|--------------------------|-----------------|
| SiO₂ | 1 | **2** |
| TiO₂ | 1 | **2** |
| Al₂O₃ | 1 | **3** |
| MgO, CaO, FeO, Na₂O, K₂O, MnO | 1 | 1 |

The `ox.count("O")` bug undercounts oxygen, inflating the6-O scale factor by ~60% and corrupting all APFu values.

### Practical checklist for EPMA ↔ MAGEMin comparison

1. **Load EPMA data** and compute the measured ratios from the correct columns:
   - XJd = `2 * Na2O_mol% / (2 * Na2O_mol% + CaO_mol%)`  (not `Na2O/(Na2O+CaO)`)
   - Mg# = `MgO_mol% / (MgO_mol% + FeO_mol%)`
   - Or convert wt% → APFu with correct oxygen counts (2, 2, 3, 1, 1, 1, 2, 2, 1 for SiO₂…FeO)

2. **Run the model** grid over the same P–T box. Extract with `oxides=["Na2O","CaO","MgO","FeO"]` and `mg_number=True`.

3. **Compute model ratios** from APFu:
   - `XJd_model = ox_apfu_Na2O / (ox_apfu_Na2O + ox_apfu_CaO)`
   - `Mg#_model = get_phase_mg_number(out, phase)` or `ox_apfu_MgO / (ox_apfu_MgO + ox_apfu_FeO)`

4. **Compare ratios**, not absolute APFu. Absolute APFu depends on the normalisation basis and will differ between a6-O structural formula and a mol% concentration — but the ratios are identical.

5. **Verify agreement**: `np.allclose(xjd_model, xjd_chem, equal_nan=True)` should return `True` (use `equal_nan=True` because missing-phase points are NaN).

## Models

- **GarnetGenerator**: radial zoning across shells, cohort size distributions, Rayleigh-style fractionation at each step.
- **MagmaOcean**: equilibrium (stage 0) → fractional crystallisation (stages 1–N); pressure-to-depth/radius conversions for rocky bodies.

## Tests

- **unittest**, mock-based (`unittest.mock.patch`), **no live Julia runtime** needed.
- Run: `python3 -m unittest discover tests`
- Files: `test_redox_logic.py`, `test_site_occupancy.py`, `test_fe_basis.py`, `test_fractionation.py`, `test_iron_oxide_conversions.py`, `test_solvus_instances.py`, `test_lmo_fix.py`.
- New public functions must be documented in the **directory's README.md** (e.g., `calculators/README.md`, `core/README.md`) **and** the package structure above must be updated.

## Requirements (from `pyproject.toml`)

`pandas`, `numpy`, `matplotlib`, `scipy`, `molmass`, `juliacall`. Python ≥ 3.10.
Extras: `[jupyter]` (jupyterlab, ipykernel, nbconvert), `[dev]` (ruff).

## Tutorials

Jupyter notebooks in `Tutorials/` (garnet growth, magma ocean, general workflows).
