import numpy as np
import sys
from scipy import optimize
from typing import List, Dict, Any, Tuple, Optional
from ..core.base import MAGEMinBase
from ..core.phase_properties import phase_frac, get_phase_chemistry
from phasetools import MAGEMin_C
from juliacall import Main as jl, convert as jlconvert

class MagmaOcean(MAGEMinBase):
    """
    Generic model for Magma Ocean (MO) crystallisation on rocky bodies.
    Based on the multi-stage approach of Johnson et al. (2021).
    
    Stages:
    0: Equilibrium crystallisation (e.g. 0-50 vol.% solid).
    1-N: Fractional crystallisation in discrete steps.
    """
    
    def __init__(self, 
                 radius_km: float = 1737.1, 
                 core_radius_km: float = 330.0,
                 gravity: float = 1.62,
                 avg_density: float = 3350.0,
                 db: str = "ig", 
                 dataset: int = 636, 
                 verbose: bool = False):
        """
        Initialise the Magma Ocean model.
        
        Parameters
        ----------
        radius_km : float
            Radius of the rocky body (default: Moon).
        core_radius_km : float
            Radius of the core (base of the MO).
        gravity : float
            Surface gravity in m/s^2.
        avg_density : float
            Average mantle density in kg/m^3 for pressure-depth conversions.
        db : str
            MAGEMin database to use ('ig' for igneous).
        dataset : int
            MAGEMin dataset version.
        verbose : bool
            Enable verbose output.
        """
        super().__init__(db, dataset, verbose)
        self.radius_body = radius_km
        self.radius_core = core_radius_km
        self.g = gravity
        self.rho_avg = avg_density
        
    def pressure_to_depth(self, P_kbar: float) -> float:
        """Convert pressure (kbar) to depth (km)."""
        # P in kbar = 1e8 Pa; P = rho * g * h
        return (P_kbar * 1e8) / (self.rho_avg * self.g) / 1000.0

    def depth_to_pressure(self, depth_km: float) -> float:
        """Convert depth (km) to pressure (kbar)."""
        return (self.rho_avg * self.g * depth_km * 1000.0) / 1e8

    def radius_to_pressure(self, R_km: float) -> float:
        """Convert radius from center (km) to pressure (kbar)."""
        depth = self.radius_body - R_km
        return self.depth_to_pressure(depth)

    def pressure_to_radius(self, P_kbar: float) -> float:
        """Convert pressure (kbar) to radius from center (km)."""
        depth = self.pressure_to_depth(P_kbar)
        return self.radius_body - depth

    def get_volume_between_radii(self, r1: float, r2: float) -> float:
        """Calculate volume of a spherical shell between radii r1 and r2."""
        return (4.0/3.0) * np.pi * (np.abs(r1**3 - r2**3))

    def find_temperature_at_vol_frac(self, P: float, target_vol_frac: float, bracket: List[float] = [800, 3000]) -> float:
        """Find the temperature at which a specific volume fraction of solid is reached."""
        def func(T):
            out = MAGEMin_C.single_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
            # Total solid fraction = 1 - melt fraction (liq)
            liq_frac = phase_frac(phase='liq', MAGEMinOutput=out, sys_in='vol')
            solid_frac = 1.0 - liq_frac
            return solid_frac - target_vol_frac

        try:
            result = optimize.root_scalar(func, bracket=bracket, method='bisect', xtol=1.0)
            return float(result.root)
        except ValueError:
            f_low = func(bracket[0])
            f_high = func(bracket[1])
            return float(bracket[0] if abs(f_low) < abs(f_high) else bracket[1])

    def get_phase_chemistry_at_index(self, out, i: int) -> np.ndarray:
        """Extract the chemical composition vector of a phase at a specific index."""
        if i < out.n_SS:
            obj = out.SS_vec[i]
        else:
            obj = out.PP_vec[i - out.n_SS]
            
        if self.sys_in.casefold() == 'wt':
            return np.array(obj.Comp_wt, dtype=float)
        else:
            return np.array(obj.Comp, dtype=float)

    def run_stage_0(self, p_start: float, p_end: float, solid_frac: float = 0.5, p_intervals: int = 20) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Stage 0: Equilibrium crystallisation from 0 to target solid fraction.
        """
        pressures = np.linspace(p_start, p_end, p_intervals)
        
        results = {
            "p_base_cumulate": p_start,
            "p_top_cumulate": p_end,
            "P": pressures,
            "T": [],
            "modes": [],
            "densities": [],
            "melt_comps": [],
            "layer_modes": {}
        }
        
        melt_sum = np.zeros(len(self._Xoxides_py))
        layer_modes_sum = {}
        
        for P in pressures:
            T = self.find_temperature_at_vol_frac(P, solid_frac)
            results["T"].append(T)
            
            out = MAGEMin_C.single_point_minimization(P, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
            
            modes = {}
            densities = {}
            for i, ph_name in enumerate(out.ph):
                ph_str = str(ph_name)
                vfrac = float(out.ph_frac_vol[i])
                modes[ph_str] = modes.get(ph_str, 0.0) + vfrac
                
                if i < out.n_SS:
                    rho = float(out.SS_vec[i].rho)
                else:
                    rho = float(out.PP_vec[i - out.n_SS].rho)
                densities[ph_str] = rho
                
                if ph_str == 'liq':
                    melt_comp = self.get_phase_chemistry_at_index(out, i)
                    melt_sum += melt_comp
                else:
                    layer_modes_sum[ph_str] = layer_modes_sum.get(ph_str, 0.0) + vfrac
            
            results["modes"].append(modes)
            results["densities"].append(densities)
            
        avg_melt = melt_sum / p_intervals
        total_solid = sum(layer_modes_sum.values())
        results["layer_modes"] = {ph: val / total_solid for ph, val in layer_modes_sum.items()}
        
        return results, avg_melt

    def run_fractional_stages(self, 
                               starting_melt: np.ndarray, 
                               p_start: float, 
                               p_end: float, 
                               vol_step: float = 0.05,
                               starting_vol_frac: float = 0.5,
                               n_stages: int = 10,
                               float_phases: List[str] = ['pl', 'q', 'fsp', 'san', 'ksp']) -> List[Dict[str, Any]]:
        """
        Fractional crystallisation stages of the remaining melt.
        
        Following Johnson et al. (2021), each stage represents a 5% volume 
        increment of the total LMO. The mineralogy of the WHOLE liquid ocean 
        is calculated at the pressure of its base.
        
        Parameters
        ----------
        starting_melt : np.ndarray
            Initial melt composition.
        p_start : float
            Initial pressure at the base of the melt ocean.
        p_end : float
            Pressure at the top of the melt ocean (e.g. surface).
        vol_step : float
            Volume fraction of the TOTAL LMO to crystallise in each stage.
        starting_vol_frac : float
            Fraction of the total LMO that is liquid at the start of stage 1 (default: 0.5).
        n_stages : int
            Number of stages to run.
        float_phases : list[str]
            List of phases that float to form the crust.
        """
        all_stage_results = []
        current_melt_comp = starting_melt
        
        # Calculate volume of total LMO based on the initial melt ocean bounds
        v_total_mo_init = self.get_volume_between_radii(self.pressure_to_radius(p_start), self.pressure_to_radius(p_end))
        v_total_lmo = v_total_mo_init / starting_vol_frac
        v_step = vol_step * v_total_lmo
        
        r_bottom = self.pressure_to_radius(p_start)
        r_top = self.pressure_to_radius(p_end)
        
        current_liquid_vol_frac = starting_vol_frac
        
        for stage in range(1, n_stages + 1):
            # Set composition
            self.X = jlconvert(jl.Vector[jl.Float64], current_melt_comp)
            
            # Base pressure of the current liquid ocean
            p_base = self.radius_to_pressure(r_bottom)
            
            # Per Johnson et al. 2021: Stage 1 concludes when 5 vol% solid is reached 
            # for the WHOLE melt ocean. Target solid frac = vol_step / current_liquid_vol_frac.
            target_solid_frac = vol_step / current_liquid_vol_frac
            
            # Find temperature at base pressure for target solid fraction
            T = self.find_temperature_at_vol_frac(p_base, target_solid_frac)
            
            # Run minimization at the base pressure
            out = MAGEMin_C.single_point_minimization(p_base, T, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in, rm_list=self.rm_list)
            
            stage_results = {
                "stage": stage,
                "p_base": p_base,
                "p_top": self.radius_to_pressure(r_top),
                "T": T,
                "modes": {},
                "densities": {},
                "layer_modes": {}
            }
            
            layer_modes_sum = {}
            for i, ph_name in enumerate(out.ph):
                ph_str = str(ph_name)
                vfrac = float(out.ph_frac_vol[i])
                stage_results["modes"][ph_str] = vfrac
                
                if i < out.n_SS:
                    rho = float(out.SS_vec[i].rho)
                else:
                    rho = float(out.PP_vec[i - out.n_SS].rho)
                stage_results["densities"][ph_str] = rho
                
                if ph_str == 'liq':
                    current_melt_comp = self.get_phase_chemistry_at_index(out, i)
                else:
                    layer_modes_sum[ph_str] = layer_modes_sum.get(ph_str, 0.0) + vfrac
            
            # Normalise solid modes for the layer
            total_solid = sum(layer_modes_sum.values())
            stage_results["layer_modes"] = {ph: val / total_solid for ph, val in layer_modes_sum.items()}
            
            # Update Geometry: Sinking minerals raise the bottom radius, floating ones lower the top radius.
            pl_frac = sum(stage_results["layer_modes"].get(ph, 0.0) for ph in float_phases)
            v_float = v_step * pl_frac
            v_sink = v_step * (1.0 - pl_frac)
            
            r_bottom = np.power(r_bottom**3 + (3 * v_sink) / (4 * np.pi), 1/3)
            r_top = np.power(r_top**3 - (3 * v_float) / (4 * np.pi), 1/3)
            
            current_liquid_vol_frac -= vol_step
            all_stage_results.append(stage_results)
            
            if current_liquid_vol_frac <= 0: break
            
        return all_stage_results
            
        return all_stage_results
