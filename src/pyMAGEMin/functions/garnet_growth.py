import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



def generate_distribution(n_classes, r_min, dr, fnr, Gn, tGn):
    """
    Generates a distribution of radial sizes and associated garnet volumes and formation times.
    
    Parameters:
        n_classes (int): Number of classes.
        r_min (float): Minimum mineral/crystal radius.
        dr (float): Increment by which radii are increased.
        fnr (array-like): Array of normalized fractions for new volume added per class.
        Gn (array-like): Total volume at discrete time steps.
        tGn (array-like): Formation times corresponding to volumes in Gn.
    
    Returns:
        cumulative_volumes (np.array): Cumulative mineral volume for each class.
        formation_times (np.array): Formation time assigned to each class.
        radii (np.array): Final radius values for each class.
        radius_matrix (2D np.array): Matrix of radius values between classes.
    """
    # Calculate the initial volume for a garnet of minimum radius.
    initial_volume = 4/3 * np.pi * r_min**3

    # Initialize arrays for cumulative volumes, formation times, and radii.
    cumulative_volumes = np.zeros(n_classes)
    formation_times = np.zeros(n_classes)
    # Every new class starts with r_min.
    radii = np.full(n_classes, r_min, dtype=float)
    # Create an empty matrix for radii between classes.
    radius_matrix = np.full((n_classes, n_classes), np.nan)
    
    # Loop over each garnet class.
    for i in range(n_classes):
        if i == 0:
            # For the first class, scale the initial volume using the fraction.
            current_volume = initial_volume * fnr[i]
            cumulative_volumes[i] = current_volume
            # Save the volume increment for reference.
            vol_increment = initial_volume
        else:
            # For subsequent classes, compute the volume increment for each existing class.
            # This computes the change in volume when the radius increases by dr.
            volume_increments = 4/3 * np.pi * ((radii[:i] + dr)**3 - (radii[:i])**3)
            
            current_volume = initial_volume * fnr[i] + np.sum(volume_increments * fnr[:i])

            # Update cumulative volume: each class adds its computed volume.
            cumulative_volumes[i] = cumulative_volumes[i-1] + current_volume
        
        # Update the radii for the already processed classes by adding dr,
        # then reset the current class radius to the minimum.
        radii[:i] += dr
        radii[i] = r_min

        # Build the radius matrix: for column i, store the radii for previous classes.
        radius_matrix[:i, i] = radii[:i]
        radius_matrix[i, i] = radii[i]
        
        # Determine the formation time for this class by finding which 
        # time index in tGn corresponds to the cumulative volume.

        # Snap to the first time index where available volume meets/exceeds the need
        capped_volume = min(cumulative_volumes[i], Gn[-1])
        idx = np.searchsorted(Gn, capped_volume, side="left")
        if idx >= len(tGn):
            idx = len(tGn) - 1
        formation_times[i] = tGn[idx]


    
    return cumulative_volumes, formation_times, radii, radius_matrix



class GarnetGenerator:
    """
    Generate synthetic garnet populations with compositional zoning along P-T-t paths.
    
    This class models garnet crystallisation and growth during metamorphic evolution,
    producing populations of garnets with realistic core-to-rim compositional zonation
    that reflects changing pressure, temperature, and composition conditions.
    
    The model uses a multi-class approach where garnets of different sizes are treated
    as discrete cohorts that nucleate and grow at different times along the P-T-t path.
    Each garnet's internal zoning is determined by the conditions (P, T, composition)
    that existed when that garnet reached each radial position.

    """
    
    def __init__(self, Pi, Ti, ti, data, X, Xoxides, sys_in, rm_list=None,
                 r_min=10, r_max=100, garnet_classes=99, nR_diff=99, fractionate=False):
        """Initialize the GarnetGenerator.
        
        Parameters
        ----------
        Pi : array-like
            Pressure values along the P-T-t path (kbar).
        Ti : array-like
            Temperature values along the P-T-t path (°C).
        ti : array-like
            Time values along the P-T-t path (Myr).
        data : dict or object
            Bulk rock composition data.
        X : array-like
            Molar fractions of system components.
        Xoxides : array-like
            Oxide compositions or molar fractions.
        sys_in : str
            System name for MAGEMin (e.g., 'ig', 'mp', 'alk').
        rm_list : list, optional
            List of phases to remove from equilibrium calculations.
        r_min : float, default=10
            Minimum garnet radius (micrometers).
        r_max : float, default=100
            Maximum garnet radius (micrometers).
        garnet_classes : int, default=99
            Number of garnet size classes (cohorts) to model.
        nR_diff : int, default=99
            Number of radial shells per garnet for compositional zoning.
        fractionate : bool, default=False
            If True, fractionate garnet from the bulk composition as it grows.
        """
        
        from .MAGEMin_functions import MAGEMinGarnetCalculator
        self.garnet_generator = MAGEMinGarnetCalculator()


        self.Pi = Pi
        self.Ti = Ti
        self.ti = ti
        self.data = data
        self.X = X
        self.Xoxides = Xoxides
        self.sys_in = sys_in
        self.r_min = r_min
        self.r_max = r_max
        self.garnet_classes = garnet_classes
        self.nR_diff = nR_diff
        self.fractionate = fractionate
        self.rm_list = rm_list

        self.X_orig = X.copy()
        self.X_last_growth = None
        self.first_growth_index = None
        self.last_growth_index = None
        self.last_growth_time = None


        ### calculate the garnet data over path
        self.extract_garnet_data()

    def get_last_growth_bulk_composition(self):
        """Return bulk composition at ``last_growth_index`` (not path end)."""
        return self.X_last_growth

    def get_growth_indices(self):
        """Return (first_growth_index, last_growth_index)."""
        return self.first_growth_index, self.last_growth_index

    def extract_garnet_data(self):
        

        (self.gt_mol_frac, self.gt_wt_frac, self.gt_vol_frac,
         self.Mgi, self.Mni, self.Fei, self.Cai, self.X_along_path) = self.garnet_generator.gt_along_path(
            self.Pi, 
            self.Ti, 
            self.data, self.X, self.Xoxides,
            self.sys_in, fractionate=self.fractionate, rm_list=self.rm_list,
        )

        if self.gt_vol_frac[0] > 0:
            self.gt_vol_frac[...] = self.gt_vol_frac[...] - self.gt_vol_frac[0] ### sets the first value to 0

        GVi = np.array(self.gt_vol_frac, dtype=float)
        if GVi.size == 0 or np.max(GVi) <= 0:
            self.X_last_growth = None
            self.first_growth_index = None
            self.last_growth_index = None
            self.last_growth_time = None
            return

        GVn = self._compute_normalized_GVG(GVi)
        first_one_idx = self._first_one_index(GVn)

        try:
            last_zero_idx = self._last_zero_before(GVn, first_one_idx, strict=True)
            first_growth_idx = last_zero_idx + 1
        except IndexError:
            first_growth_idx = 0

        self.first_growth_index = int(first_growth_idx)
        self.last_growth_index = int(first_one_idx)
        self.last_growth_time = float(np.array(self.ti)[first_one_idx])

        if self.X_along_path is not None and len(self.X_along_path) > first_one_idx:
            self.X_last_growth = np.array(self.X_along_path[first_one_idx], dtype=float).copy()
        else:
            self.X_last_growth = None

    
    def _compute_normalized_GVG(self, GVi):
        """Compute and return normalized garnet volume sequence (GVG)"""
        GVG = [GVi[0]]
        for i in range(1, len(GVi)):
            if GVi[i] <= max(GVG):
                GVG.append(max(GVG))
            else:
                GVG.append(GVi[i])
        return np.array(GVG) / np.max(GVG)

    
    def _get_size_distribution(self, size_dist, r):
        """Compute the garnet size distribution based on size_dist input"""
        n_classes = len(r)
        if isinstance(size_dist, str):
            if size_dist == 'N':  # normal distribution
                mi = (self.r_min + self.r_max) / 2
                s = (mi - self.r_min) / 2
                finp = norm.pdf(r, loc=mi, scale=s)
            elif size_dist == 'U':  # uniform distribution
                finp = np.ones(n_classes)
            else:
                raise ValueError("When provided as a string, size_dist must be 'N' or 'U'")
        elif isinstance(size_dist, (list, np.ndarray)):
            user_dist = np.array(size_dist, dtype=float)
            if user_dist.shape[0] != n_classes:
                raise ValueError("User-defined distribution must have length equal to garnet_classes")
            finp = user_dist
        else:
            raise ValueError("size_dist must be a string ('N' or 'U') or a numeric array")
        return finp

    def _normalize_distribution(self, finp, r):
        """Normalize the size distribution using volume and return reversed array (fnr)."""
        v = 4/3 * np.pi * r**3
        V = np.sum(v * finp)
        fn = finp / V
        return fn[::-1]  # reverse order: largest garnet goes first

    # ---- Internal helpers to reduce repetition ----
    def _first_one_index(self, GVn):
        return np.where(GVn == 1)[0][0]

    def _last_zero_before(self, GVn, stop_idx, strict=True):
        idxs = np.where(GVn[:stop_idx] == 0)[0]
        if idxs.size == 0:
            if strict:
                raise IndexError("No zero found before first-one index in GVn")
            return -1
        return idxs[-1]

    def _slice_arrays(self, ind):
        tG = self.ti[ind]
        TG = self.Ti[ind]
        PG = self.Pi[ind]
        MnG = np.array(self.Mni)[ind]
        MgG = np.array(self.Mgi)[ind]
        FeG = np.array(self.Fei)[ind]
        CaG = np.array(self.Cai)[ind]
        return tG, TG, PG, MnG, MgG, FeG, CaG

    def _build_size_distribution(self, size_dist='N'):
        n_classes = self.garnet_classes
        r = np.linspace(self.r_min, self.r_max, n_classes, endpoint=True)
        dr = r[1] - r[0]
        finp = self._get_size_distribution(size_dist, r)
        fnr = self._normalize_distribution(finp, r)
        return n_classes, r, dr, finp, fnr

    def _interp(self, t_src, y_src, t_new):
        return np.interp(t_new, t_src, y_src)

    def get_prograde_concentrations(self, new_t=None):
        """Get the prograde concentrations of garnet-forming elements.

        Parameters:
            new_t (array-like, optional): New time values to interpolate the data
                and return the concentrations at these times. If None, the original
                data is returned. Uses a linear interpolation between datapoints.
            
        Returns:
            Concentrations (array): An array with the element concentrations and PTt data at each prograde step.
        """

        GVi = np.array(self.gt_vol_frac)
        GVn = self._compute_normalized_GVG(GVi)

        first_one_idx = self._first_one_index(GVn)
        try:
            last_zero_idx = self._last_zero_before(GVn, first_one_idx, strict=True)
        except IndexError:
            last_zero_idx = -1
        ind = np.arange(last_zero_idx+1, first_one_idx+1)

        tG, TG, PG, MnG, MgG, FeG, CaG = self._slice_arrays(ind)

        if new_t is not None:
            # interpolate onto new time values
            new_TG = self._interp(tG, TG, new_t)
            new_PG = self._interp(tG, PG, new_t)
            new_MnG = self._interp(tG, MnG, new_t)
            new_MgG = self._interp(tG, MgG, new_t)
            new_FeG = self._interp(tG, FeG, new_t)
            new_CaG = self._interp(tG, CaG, new_t)

            data = np.column_stack([new_t, new_TG, new_PG, new_MnG, new_MgG, new_FeG, new_CaG]).T

        else:
             data = np.column_stack([tG, TG, PG, MnG, MgG, FeG, CaG]).T

        return data
        
    def get_retrograde_concentrations(self, new_t=None):
        """Get the retrograde concentrations of garnet-forming elements.

        Parameters:
            new_t (array-like, optional): New time values to interpolate the data
                and return the concentrations at these times. If None, the original
                data is returned. Uses a linear interpolation between datapoints.
            
        Returns:
            Concentrations (array): An array with the element concentrations and PTt data at each retrograde step.
        """

        GVi = np.array(self.gt_vol_frac)
        GVn = self._compute_normalized_GVG(GVi)

        first_one_idx = self._first_one_index(GVn)
        ind = slice(first_one_idx, None)

        tG, TG, PG, MnG, MgG, FeG, CaG = self._slice_arrays(ind)

        if new_t is not None:
            t_eval = np.asarray(new_t, dtype=float)
            T_eval = self._interp(tG, TG, t_eval)
            P_eval = self._interp(tG, PG, t_eval)
        else:
            t_eval = np.asarray(tG, dtype=float)
            T_eval = np.asarray(TG, dtype=float)
            P_eval = np.asarray(PG, dtype=float)

        # If bulk composition at last growth stage differs from final path bulk
        # composition, recalculate retrograde EM fractions using the last-growth
        # bulk composition at each retrograde P-T point.
        do_recalc = False
        x_last_growth = None
        if self.X_along_path is not None and self.last_growth_index is not None and len(self.X_along_path) > 0:
            x_last_growth = np.asarray(self.X_along_path[self.last_growth_index], dtype=float)
            x_last_path = np.asarray(self.X_along_path[-1], dtype=float)
            if x_last_growth.shape == x_last_path.shape and not np.array_equal(x_last_growth, x_last_path):
                do_recalc = True

        if do_recalc:
            Mn_eval = np.zeros_like(t_eval, dtype=float)
            Mg_eval = np.zeros_like(t_eval, dtype=float)
            Fe_eval = np.zeros_like(t_eval, dtype=float)
            Ca_eval = np.zeros_like(t_eval, dtype=float)

            for i in range(len(t_eval)):
                (_gt_frac, _gt_wt, _gt_vol,
                 Mg_i, Mn_i, Fe_i, Ca_i, _out) = self.garnet_generator.gt_single_point_calc_elements(
                    P_eval[i], T_eval[i], self.data, x_last_growth, self.Xoxides, self.sys_in, self.rm_list
                )
                Mn_eval[i] = Mn_i
                Mg_eval[i] = Mg_i
                Fe_eval[i] = Fe_i
                Ca_eval[i] = Ca_i

            data = np.column_stack([t_eval, T_eval, P_eval, Mn_eval, Mg_eval, Fe_eval, Ca_eval]).T
            return data

        if new_t is not None:
            # interpolate onto new time values
            new_MnG = self._interp(tG, MnG, t_eval)
            new_MgG = self._interp(tG, MgG, t_eval)
            new_FeG = self._interp(tG, FeG, t_eval)
            new_CaG = self._interp(tG, CaG, t_eval)

            data = np.column_stack([t_eval, T_eval, P_eval, new_MnG, new_MgG, new_FeG, new_CaG]).T

        else:
             data = np.column_stack([tG, TG, PG, MnG, MgG, FeG, CaG]).T

        return data


    def generate_garnets(self, size_dist='N'):
        """Generates garnet distributions.

        Parameters:
            size_dist (str or array-like): 
                'N' for a normal distribution, 
                'U' for a uniform distribution, or 
                a user-defined numeric distribution array of length equal to garnet_classes.
            
        Returns:
            garnets (list): List of garnet data dictionaries.
        """

        GVi = np.array(self.gt_vol_frac)
        GVn = self._compute_normalized_GVG(GVi)

        first_one_idx = self._first_one_index(GVn)
        # safe last_zero like original generate_garnets
        try:
            last_zero_idx = self._last_zero_before(GVn, first_one_idx, strict=True)
        except IndexError:
            last_zero_idx = -1

        ind = np.arange(last_zero_idx+1, first_one_idx+1)

        GVG = GVn[ind]

        tG, TG, PG, MnG, MgG, FeG, CaG = self._slice_arrays(ind)

        # Generate radius classes and distributions
        n_classes, r, dr, finp, fnr = self._build_size_distribution(size_dist)

        Gn = GVG / np.max(GVG)
        tGn = tG

        G, t_arr, r_r, R = generate_distribution(n_classes, self.r_min, dr, fnr, Gn, tGn)

        # Interpolate physical properties along the garnet growth
        PGrw = self._interp(tG, PG, t_arr)
        TGrw = self._interp(tG, TG, t_arr)
        Mnrw = self._interp(tG, MnG, t_arr)
        Mgrw = self._interp(tG, MgG, t_arr)
        Ferw = self._interp(tG, FeG, t_arr)
        Carw = 1 - Mnrw - Mgrw - Ferw

        garnets = []
        for i in range(n_classes):
            ind_range = np.arange(i, n_classes)
            Rr1 = R[i, ind_range]
            tr1 = t_arr[ind_range]
            Pr1 = PGrw[ind_range]
            Tr1 = TGrw[ind_range]
            Mnr1 = Mnrw[ind_range]
            Mgr1 = Mgrw[ind_range]
            Fer1 = Ferw[ind_range]
            Car1 = 1 - Mnr1 - Mgr1 - Fer1

            dRr = Rr1[-1] / self.nR_diff
            Rrz = np.arange(dRr, Rr1[-1] + dRr, dRr)
            trz = np.interp(Rrz, Rr1, tr1)
            Prz = np.interp(Rrz, Rr1, Pr1)
            Trz = np.interp(Rrz, Rr1, Tr1)
            Mnrz = np.interp(Rrz, Rr1, Mnr1)
            Mgrz = np.interp(Rrz, Rr1, Mgr1)
            Ferz = np.interp(Rrz, Rr1, Fer1)
            Carz = 1 - Mnrz - Mgrz - Ferz

            Rr_full = np.concatenate([[0], Rrz])
            tr_full = np.concatenate([[trz[0]], trz])
            Pr_full = np.concatenate([[Prz[0]], Prz])
            Tr_full = np.concatenate([[Trz[0]], Trz])
            Mnr_full = np.concatenate([[Mnrz[0]], Mnrz])
            Mgr_full = np.concatenate([[Mgrz[0]], Mgrz])
            Fer_full = np.concatenate([[Ferz[0]], Ferz])
            Car_full = np.concatenate([[Carz[0]], Carz])

            garnet_population_data = {
                "Rr": Rr_full,
                "tr": tr_full,
                "Pr": Pr_full,
                "Tr": Tr_full,
                "Mnr": Mnr_full,
                "Mgr": Mgr_full,
                "Fer": Fer_full,
                "Car": Car_full
            }
            garnets.append(garnet_population_data)
        
        return garnets

    def plot_garnet_summary(self, size_dist='N', garnet_no=0, path=None, plot_fig=True):
        """
        Plot a summary of the garnet formation results.
        
        Parameters:
            size_dist (str or array-like): 
                'N' for a normal distribution, 'U' for uniform,
                or a user-defined numeric array (length=garnet_classes).
            path (str, optional): Path to save the figure.
        """

        GVi = np.array(self.gt_vol_frac)
        if GVi[0] > 0:
            GVi = GVi - GVi[0]
        GVn = self._compute_normalized_GVG(GVi)

        first_one_idx = self._first_one_index(GVn)
        try:
            last_zero_idx = self._last_zero_before(GVn, first_one_idx, strict=True)
        except IndexError:
            last_zero_idx = -1
        ind = np.arange(last_zero_idx+1, first_one_idx+1)

        GVG = GVn[ind]

        tG, TG, PG, MnG, MgG, FeG, CaG = self._slice_arrays(ind)

        n_classes, r, dr, finp, fnr = self._build_size_distribution(size_dist)

        Gn = GVG / np.max(GVG)
        tGn = tG

        G, t_arr, r_r, R = generate_distribution(n_classes, self.r_min, dr, fnr, Gn, tGn)
        
        # Interpolate physical properties along garnet growth
        PGrw = self._interp(tG, PG, t_arr)
        TGrw = self._interp(tG, TG, t_arr)
        Mnrw = self._interp(tG, MnG, t_arr)
        Mgrw = self._interp(tG, MgG, t_arr)
        Ferw = self._interp(tG, FeG, t_arr)
        Carw = self._interp(tG, CaG, t_arr)

        # --- Create the summary subplots ---
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        fig.suptitle('Garnet formation summary')

        # Subplot 1: Garnet size
        axs[0, 0].set_title('Garnet size')
        axs[0, 0].set_xlabel('r')
        axs[0, 0].set_ylabel('f')
        axs[0, 0].set_xlim([r_r.min(), r_r.max()])
        axs[0, 0].plot(r_r, finp, '-', label='Size Distribution')
        for i in range(n_classes):
            axs[0, 0].plot([r_r[i], r_r[i]], [0, finp[i]], '-')
        axs[0, 0].legend()

        # Subplot 2: Classes' birth place
        axs[0, 1].set_title('Classes birth place')
        axs[0, 1].set_xlabel('T')
        axs[0, 1].set_ylabel('P')
        axs[0, 1].plot(self.Ti, self.Pi, label='Path')
        axs[0, 1].plot(self.Ti, self.Pi, 'kx', label='Path Points')
        axs[0, 1].plot(TGrw, PGrw, 'r.', label='Garnet Formation')
        axs[0, 1].legend()

        # Subplot 3: Classes growth
        axs[1, 0].set_title('Classes growth')
        axs[1, 0].set_xlabel('t')
        axs[1, 0].set_ylabel('r')
        for i in range(0, n_classes, 10):
            axs[1, 0].plot(t_arr[i:], R[i, i:], 'r.-', label=f'Class {i}')
        # axs[1, 0].legend()

        # Subplot 4: Volume consumption
        axs[1, 1].set_title('Volume consumption')
        axs[1, 1].set_xlabel('t')
        axs[1, 1].set_ylabel('GV')
        axs[1, 1].set_ylim([0, 1.01])
        axs[1, 1].plot(tGn, Gn, 'k', drawstyle='steps-post', label='Normalized Volume')
        axs[1, 1].plot(t_arr, G, 'mx', label='Volume Growth')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        # Subplot 5: Elemental compositions
        axs[2, 0].set_title('Garnet Elemental Compositions')
        axs[2, 0].set_xlabel('r')
        axs[2, 0].set_ylabel('c')
        axs[2, 0].set_xlim([0, self.r_max])
        for i in np.arange(0, n_classes, 10):
            ind_local = np.arange(i, n_classes)
            rplt = R[i, :]
            axs[2, 0].plot(rplt[ind_local], Mnrw[ind_local], '-b', label='Mn')
            axs[2, 0].plot(rplt[ind_local], Mgrw[ind_local], '-g', label='Mg')
            axs[2, 0].plot(rplt[ind_local], Ferw[ind_local], '-r', label='Fe')
            axs[2, 0].plot(rplt[ind_local], (Carw[ind_local]), '-', c='gold', label='Ca')
        axs[2, 0].legend()

        # Subplot 6: Garnet composition
        i = garnet_no  # highlight the defined garnet number
        ind_local = np.arange(i, n_classes)
        rplt = R[i, :]
        for ax in [axs[2, 0], axs[2, 1]]:
            ax.set_title('Chosen Garnet Composition')
            ax.set_xlabel('r')
            ax.set_ylabel('c')
            ax.set_xlim([0, self.r_max])
            ax.plot(rplt[ind_local], Mnrw[ind_local], 'bx', label='Mn')
            ax.plot(rplt[ind_local], Mgrw[ind_local], 'gx', label='Mg')
            ax.plot(rplt[ind_local], Ferw[ind_local], 'rx', label='Fe')
            ax.plot(rplt[ind_local], (Carw[ind_local]), 'x', c='gold', linewidth=2, label='Ca')

        # Add a single legend for unique labels
        axs[2, 0].legend(['Mn', 'Mg', 'Fe', 'Ca'], loc='upper right')

        axs[2, 1].plot(rplt[ind_local], Mnrw[ind_local], 'b-', label='Mn')
        axs[2, 1].plot(rplt[ind_local], Mgrw[ind_local], 'g-', label='Mg')
        axs[2, 1].plot(rplt[ind_local], Ferw[ind_local], 'r-', label='Fe')
        axs[2, 1].plot(rplt[ind_local], (Carw[ind_local]), '-', c='gold', linewidth=2, label='Ca')
        
        # Add a single legend for unique labels
        axs[2, 1].legend(['Mn', 'Mg', 'Fe', 'Ca'], loc='upper right')

        axs[2, 1].grid(True)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        if path is not None:
            plt.savefig(path)

        if plot_fig == True:
            plt.show()
        else:
            plt.close()




