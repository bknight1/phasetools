# Phasetools Testing Suite

This directory contains verification scripts and unit tests to ensure the thermodynamic and chemical logic of `phasetools` remains consistent and accurate.

## Test Catalog

### `test_redox_logic.py`
**Objective:** Validates the iron-splitting and redox heuristics.
- **Key Tests:**
    - Garnet $\text{Fe}^{3+}$ splitting using the excess oxygen (O) basis.
    - Clinopyroxene $\text{Fe}^{3+}$ splitting (e.g., Acmite component).
    - Traditional $\text{Fe}_2\text{O}_3$ convention handling.
    - Iron handling for specific databases like `sb24` (metal iron + oxygen).
    - $Mg\#$ calculation robustness across different iron components.
    - $\text{Fe}-\text{Mg}$ distribution coefficient ($K_d$) logic.

### `test_site_occupancy.py`
**Objective:** Compares built-in chemical heuristics against explicit structural site occupancy.
- **Key Tests:**
    - **Garnet:** Compares the heuristic $\text{Fe}^{2+}/\text{Fe}^{3+}$ split against the stoichiometric site totals calculated from end-members (`py`, `alm`, `kho`). Verifies $Mg\#$ specifically for the structural X-site.
    - **Clinopyroxene:** Validates the $\text{Fe}^{2+}/\text{Fe}^{3+}$ split for ordered omphacitic pyroxenes against site occupancy. Verifies $Mg\#$ specifically for the **M1m** structural site.

---
**Run all tests:**
```bash
python3 -m unittest discover tests
```
