# phasetools

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Julia Version](https://img.shields.io/badge/julia-1.6+-purple.svg)](https://julialang.org/)
[![Thermodynamic Engine](https://img.shields.io/badge/engine-MAGEMin-orange.svg)](https://github.com/ComputationalThermodynamics/MAGEMin)

`phasetools` is a high-level Python library designed for advanced thermodynamic modelling and petrological simulations. It provides a seamless bridge to the **MAGEMin** C-backend via Julia, offering specialised tools for phase equilibria, geothermobarometry, and complex multi-step simulations like garnet growth and magma ocean differentiation.

## 🚀 Key Features

### 1. Specialised Calculators
- **P-T Grid Explorer:** Efficiently calculate phase properties (fractions, chemistry, end-members) across multi-dimensional pressure-temperature grids.
- **Geothermobarometry Estimator:** Estimate equilibrium P-T conditions by minimising chemical misfits between measured and predicted phase compositions using global and local optimisers.
- **Phase Search & Stability:** Precision root-finding to locate phase appearance (solidus) and saturation (liquidus) boundaries.
- **Assemblage Mapper:** Identify stability fields and coexistence regions for complex multi-mineral assemblages.

### 2. Advanced Petrological Models
- **Garnet Growth Simulator:** Model synthetic garnet populations with fractional growth, radial compositional zoning, and cohort-based size distributions.
- **Magma Ocean Differentiation:** Simulate the multi-stage cooling and crystallisation of planetary magma oceans (equilibrium vs. fractional stages) with integrated geophysical depth-pressure conversions.

### 3. Core Utility Engine
- **Robust Redox Handling:** Automatic redistribution of iron into $Fe^{2+}$ and $Fe^{3+}$ using excess oxygen heuristics, ensuring compatibility with thermodynamic databases.
- **Stoichiometry & Units:** Integrated mass resolution via `molmass` and flexible unit conversions between weight, molar, and volume systems.
- **Julia Bridge:** Managed lifecycle for the Julia environment and `MAGEMin_C` interface.

## 📦 Installation

### 1. Conda / mamba (Linux & CI)

```bash
mamba env create -f environment.yml
conda activate phasetools
phasetools-julia-setup --install
phasetools-julia-setup --check
```

*Note: On macOS (especially Apple Silicon), conda-forge's `julia` package is unavailable — use the pip + juliaup path below instead.*

### 2. pip + juliaup (cross-platform, incl. macOS)

```bash
python -m pip install -e .
phasetools-julia-setup --check
phasetools-julia-setup --install
```

To also install Jupyter (for tutorials) and dev tooling (ruff):

```bash
python -m pip install -e ".[jupyter,dev]"
```

*Note: If Julia is not installed on your system, please download it from [julialang.org](https://julialang.org/downloads/).*

## 🔬 Scientific Foundation

### Supported Databases
`phasetools` supports the full suite of Holland & Powell datasets implemented in MAGEMin, including:
- **Igneous:** `ig` (Green et al., 2025), `igad` (Alkaline; Weller et al., 2024).
- **Metamorphic:** `mp` (Metapelite), `mb` (Metabasite).
- **Ultramafic:** `um` (Evans & Frost, 2021).
- **Mantle:** `mtl` (Holland et al., 2013), `sb11/21/24` (Stixrude & Lithgow-Bertelloni).
- **Extended:** `mpe`, `mbe`, `ume` for cross-lithology modelling.

### Redox and Iron Partitioning
When using redox-active databases, `phasetools` employs a robust heuristic to partition total iron atoms ($Fe_{total}$) into divalent ($Fe^{2+}$) and trivalent ($Fe^{3+}$) states:

1.  **Excess Oxygen Detection:** Isolates the oxygen contribution added to oxidise iron from 2+ to 3+ by evaluating the baseline anion stoichiometry.
2.  **Ferric Calculation:** Each mole of excess oxygen balances two moles of ferric iron ($Fe^{3+} = 2 \times O_{excess}$).
3.  **Atomic Consistency:** Ensures that geothermometers targeting specific iron valencies are mathematically consistent with the underlying thermodynamic engine.

## 📖 Tutorials
Explore the `Tutorials/` directory for detailed Jupyter notebooks covering:
- **Garnet:** Compositional isopleths and growth workflows.
- **General:** Getting started, determining phase fractions, and Magma Ocean modelling.

## 📜 License & Citation
`phasetools` is developed for academic research. If you use this software in your work, please cite the underlying MAGEMin engine:

> Riel, N., Kaus, B. J. P., Green, E. C. R., & Berlie, N. (2022). MAGEMin, an Efficient Gibbs Energy Minimizer: Application to Igneous Systems. *Geochemistry, Geophysics, Geosystems*, 23, e2022GC010427. [https://doi.org/10.1029/2022GC010427](https://doi.org/10.1029/2022GC010427)
