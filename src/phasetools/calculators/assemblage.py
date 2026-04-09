import numpy as np
import sys
from ..core.base import MAGEMinBase
from ..core.phase_properties import phase_frac
from phasetools import MAGEMin_C

class MAGEMinAssemblageCalculator(MAGEMinBase):
    """
    Calculator for identifying the stability field of a multi-mineral assemblage
    and calculating its total volume (or weight/molar) fraction across a P-T grid.
    """
    def __init__(self, db="ig", dataset=636, verbose=False):
        super().__init__(db, dataset, verbose)

    def calculate_assemblage_grid(self, P, T, phases):
        """
        Calculates the stability field and total fraction of a mineral assemblage.

        Args:
            P, T: Meshgrid-style flattened arrays for Pressure (kbar) and Temperature (C).
            phases: List of MAGEMin phase names (e.g., ['g', 'omph', 'q', 'zo']).

        Returns:
            stability_mask: Boolean array (True where all phases coexist).
            total_fraction: Float array (sum of fractions where all coexist; 0.0 otherwise).
        """
        out = MAGEMin_C.multi_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
        sys.stdout.flush()

        stability_mask = np.zeros_like(P, dtype=bool)
        total_fraction = np.zeros_like(P, dtype=float)

        for i in range(len(P)):
            # Check if all phases are present
            is_stable = all(p in out[i].ph for p in phases)
            stability_mask[i] = is_stable
            
            if is_stable:
                # Sum the fractions of all phases in the assemblage
                total_fraction[i] = sum(phase_frac(p, out[i], self.sys_in) for p in phases)
            else:
                total_fraction[i] = 0.0

        return stability_mask, total_fraction
