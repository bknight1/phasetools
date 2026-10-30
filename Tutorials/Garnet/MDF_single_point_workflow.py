# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: uw3-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Single-point garnet equilibrium and MDF/PTM
#
# Compare ordinary equilibrium and maximum-driving-force (MDF) garnet at one
# P–T point. Parallel-tangent method/model (PTM) is a related description of
# this calculation. The comparable outputs are closed Mg-Fe(total)-Mn-Ca
# proportions.

# %%
from phasetools import GarnetMDFCalculator, garnet_fractions_to_mg_fe_mn_ca
import numpy as np

# %% [markdown]
# ## Configure the bulk and P–T point
#
# Configure the bulk once; subsequent calculations read this saved state.
# MAGEMin standardises Fe2O3 input to FeO plus excess O, so raw O (and even raw
# FeO) need not be supplied when ferric iron is entered as Fe2O3. The reported
# Fe(total) proportions count all kho iron as FeO-equivalent.

# %%
oxides = (
    "H2O",
    "SiO2",
    "Al2O3",
    "CaO",
    "MgO",
    "FeO",
    "K2O",
    "Na2O",
    "TiO2",
    "MnO",
    "O",
)
bulk_mol = (0.92, 54.57, 8.79, 11.20, 8.45, 12.89, 0.24, 2.24, 1.12, 0.22, 0.64)
pressure_kbar, temperature_c = 30.0, 700.0

calculator = GarnetMDFCalculator(db="mpe", dataset=636)
calculator.setup_bulk_composition(oxides, bulk_mol, sys_in="mol", rm_list=[])

# %% [markdown]
# ## Ordinary equilibrium garnet
#
# Keep MAGEMin's active thermodynamic fractions visible as a diagnostic; they
# are distinct from the closed cation proportions used for comparison below.

# %%
equilibrium_output = calculator.calculate_equilibrium(pressure_kbar, temperature_c)

garnet_index = next(
    index for index, phase in enumerate(equilibrium_output.ph) if str(phase) == "g"
)
garnet_solution = equilibrium_output.SS_vec[garnet_index]
active_names = tuple(str(name) for name in garnet_solution.emNames)
active_fractions = tuple(map(float, garnet_solution.emFrac))

if len(active_names) != len(active_fractions):
    raise RuntimeError("MAGEMin garnet emNames/emFrac lengths do not match")

    
equilibrium_fractions = dict(zip(active_names, active_fractions, strict=True))

print("Active equilibrium thermodynamic fractions:", equilibrium_fractions)
equilibrium_cats = garnet_fractions_to_mg_fe_mn_ca(equilibrium_fractions, fe_basis='fe2+')

# %% [markdown]
# ## Garnet-suppressed matrix and MDF/PTM garnet
#
# `calculate_matrix_state` returns the candidate matrix assemblage and oxide
# chemical potentials. The calculator solves kho-aware MDF on the active
# thermodynamic endmember simplex.

# %%
result = calculator.calculate_compositions(pressure_kbar, temperature_c, fe_basis='fe2+')
matrix_phases = result.matrix_phases

print("Matrix phase assemblage:", ", ".join(matrix_phases))
print("Automatic solver mode from standardised molar O:", result.solver_mode)
print("Active MDF thermodynamic fractions:", result._mdf_endmember_fractions)
print(
    f"MDF converged: {result.converged}; residual norm: "
    f"{result.residual_norm_kj_per_formula:.4g} kJ/mol formula; "
    f"residuals: {result.residual_kj_per_formula}"
)
if not result.converged:
    print("Solver message:", result.message)

mdf_cations = result.mdf
equilibrium_cations = result.equilibrium

# %%
for key in zip(equilibrium_cations):
    print(f'key: {key[0]}')
    print(f'Is close: {(np.isclose(equilibrium_cations[key[0]], equilibrium_cats[key[0]]))}')

# %% [markdown]
# ## Comparable closed cation proportions
#
# These are cation proportions—not thermodynamic endmember fractions or APFU.
# Fe includes ferrous and ferric iron, including the two Fe atoms in kho.

# %%
print("Closed Mg-Fe(2+)-Mn-Ca cation proportions (Fe2+ basis):")
for label, composition in (
    ("Ordinary equilibrium", equilibrium_cations),
    ("Maximum-driving-force", mdf_cations),
):
    values = ", ".join(f"{cation}={value:.4f}" for cation, value in composition.items())
    print(f"{label}: {values}")

# %%
# Closed Mg-Fe(total)-Mn-Ca cation proportions (EPMA all-Fe-as-FeO basis):
# Ordinary equilibrium: Mg=0.1982, Fe=0.5139, Mn=0.0107, Ca=0.2773
# Maximum-driving-force: Mg=0.0895, Fe=0.6523, Mn=0.1544, Ca=0.1038

# %%
