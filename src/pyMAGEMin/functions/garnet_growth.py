import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def generate_distribution_specified(n_classes, r_min, dr, fnr, Gn, tGn, formation_times_spec=None): 
    
    """ Generates garnet distribution based either on the volume interpolation or on specified formation times/PT points.
    Parameters:
    n_classes (int): Number of classes.
    r_min (float): Minimum radius.
    dr (float): Increment in radius.
    fnr (array-like): Normalized new volume fractions.
    Gn (array-like): Normalized cumulative garnet volumes.
    tGn (array-like): Default times corresponding to Gn.
    formation_times_spec (array-like, optional): If provided, an array of length n_classes specifying the
        desired formation times (or PT points) at which garnets should form.
        This lets you force garnets to grow at specific conditions along the path.

    Returns:
        cumulative_volumes (np.array): Cumulative volume for each class.
        formation_times (np.array): Formation times (or PT points) for each class.
        radii (np.array): Final radius values for each class.
        radius_matrix (2D np.array): Matrix of radius values between classes.
    """
    initial_volume = 4/3 * np.pi * r_min**3
    cumulative_volumes = np.zeros(n_classes)
    formation_times = np.zeros(n_classes)
    radii = np.full(n_classes, r_min, dtype=float)
    radius_matrix = np.full((n_classes, n_classes), np.nan)


    if formation_times_spec is not None: 
        formation_times_spec = np.asarray(formation_times_spec)
        nt = formation_times_spec.shape[0]
        if nt != n_classes:
            # Repeat the given formation times until we have n_classes entries and sort in order.
            formation_times_spec = np.sort( np.resize(formation_times_spec, n_classes) )

    
    for i in range(n_classes):
        if i == 0:
            current_volume = initial_volume * fnr[i]
            cumulative_volumes[i] = current_volume
        else:
            volume_increments = 4/3 * np.pi * ((radii[:i] + dr)**3 - (radii[:i])**3)
            current_volume = initial_volume * fnr[i] + np.sum(volume_increments * fnr[:i])
            cumulative_volumes[i] = cumulative_volumes[i-1] + current_volume

        radii[:i] += dr
        radii[i] = r_min
        radius_matrix[:i, i] = radii[:i]
        radius_matrix[i, i] = radii[i]

        # If a user-specified array is provided, use that directly for formation time (or PT point)
        if formation_times_spec is not None:
            formation_times[i] = formation_times_spec[i]
        else:
            # Otherwise use volume-based interpolation
            lower_inds = np.where(Gn <= cumulative_volumes[i])[0]
            upper_inds = np.where(Gn >= cumulative_volumes[i])[0]
            i_lower = lower_inds[-1] if lower_inds.size > 0 else upper_inds[0]
            i_upper = upper_inds[0] if upper_inds.size > 0 else lower_inds[-1]
            if i_lower == i_upper:
                formation_times[i] = tGn[i_upper]
            else:
                formation_times[i] = np.interp(cumulative_volumes[i],
                                            [Gn[i_lower], Gn[i_upper]],
                                            [tGn[i_lower], tGn[i_upper]])

    return cumulative_volumes, formation_times, radii, radius_matrix


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
        formation_times (np.array): Formation time for each class.
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
        lower_inds = np.where(Gn <= cumulative_volumes[i])[0]
        upper_inds = np.where(Gn >= cumulative_volumes[i])[0]
        i_lower = lower_inds[-1] if lower_inds.size > 0 else upper_inds[0]
        i_upper = upper_inds[0] if upper_inds.size > 0 else lower_inds[-1]
        
        # If the cumulative volume exactly matches one value, use that formation time;
        # otherwise, interpolate between the two nearest time points.
        if i_lower == i_upper:
            formation_times[i] = tGn[i_upper]
        else:
            formation_times[i] = np.interp(cumulative_volumes[i],
                                           [Gn[i_lower], Gn[i_upper]],
                                           [tGn[i_lower], tGn[i_upper]])


    ### IF we wanted evenly spaced formation times
    # formation_times = np.linspace(tGn[0], tGn[-1], n_classes)
    
    return cumulative_volumes, formation_times, radii, radius_matrix



class GarnetGenerator:
    def __init__(self, Pi, Ti, ti, data, X, Xoxides, sys_in, rm_list=None,
                 t_start_growth=None, t_end_growth=None,
                 r_min=10, r_max=100, garnet_classes=99, nR_diff=99, fractionate=False):
        self.Pi = Pi
        self.Ti = Ti
        self.ti = ti
        self.data = data
        self.X = X
        self.Xoxides = Xoxides
        self.sys_in = sys_in
        self.t_start_growth = t_start_growth
        self.t_end_growth = t_end_growth
        self.r_min = r_min
        self.r_max = r_max
        self.garnet_classes = garnet_classes
        self.nR_diff = nR_diff
        self.fractionate = fractionate
        self.rm_list = rm_list

        # if self.t_start_growth is not None and self.t_end_growth is not None:
        #     # If both t_start_growth and t_end_growth are specified, we only consider the data within this time range
        #     self._mask = (self.ti >= self.t_start_growth) & (self.ti <= self.t_end_growth)
        # elif self.t_start_growth is not None:
        #     # If only t_start_growth is specified, we consider data from this time onwards
        #     self._mask = self.ti >= self.t_start_growth
        # elif self.t_end_growth is not None:
        #     # If only t_end_growth is specified, we consider data up to this time
        #     self._mask = self.ti <= self.t_end_growth
        # else:
        #     # If neither is specified, we use the full range of indices
        #     self._mask = np.ones_like(self.ti, dtype=bool)

        ### calculate the garnet data over path
        self.extract_garnet_data()

    def extract_garnet_data(self):
        from .MAGEMin_functions import MAGEMinGarnetCalculator
        garnet_generator = MAGEMinGarnetCalculator()

        (self.gt_mol_frac, self.gt_wt_frac, self.gt_vol_frac,
         self.d_gt_mol_frac, self.d_gt_wt_frac,
         self.Mgi, self.Mni, self.Fei, self.Cai) = garnet_generator.gt_along_path(
            self.Pi, 
            self.Ti, 
            self.data, self.X, self.Xoxides,
            self.sys_in, fractionate=self.fractionate, rm_list=self.rm_list,
        )

        if self.gt_vol_frac[0] > 0:
            self.gt_vol_frac[...] = self.gt_vol_frac[...] - self.gt_vol_frac[0] ### sets the first value to 0

    def _reset_variables(self):
        ### reset everything to None
        (self.gt_mol_frac, self.gt_wt_frac, self.gt_vol_frac,
         self.d_gt_mol_frac, self.d_gt_wt_frac,
         self.Mgi, self.Mni, self.Fei, self.Cai) = [None]*9
    
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
                finp = np.exp(-(r - mi)**2 / (2 * s**2)) / (np.sqrt(2 * np.pi) * s)
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

    def get_prograde_concentrations(self, new_t=None):
        """Get the retrograde concentrations of garnet-forming elements.

        Parameters:
            new_t (array-like, optional): New time values to interpolate the data
                and return the concentrations at these times. If None, the original
                data is returned. Uses a linear interpolation between datapoints.
            
        Returns:
            Concentrations (array): An array with the element concentrations and PTt data at each retrograde step.
        """

        ### add in a mask based on self.t_start_growth and self.t_end_growth
        if self.t_start_growth is not None and self.t_end_growth is not None:
            time_mask = (self.ti >= self.t_start_growth) & (self.ti <= self.t_end_growth)
        elif self.t_start_growth is not None:
            # If only t_start_growth is specified, we consider data from this time onwards
            time_mask = self.ti >= self.t_start_growth
        elif self.t_end_growth is not None:
            time_mask = self.ti <= self.t_end_growth
        else:
            time_mask = np.ones_like(self.ti, dtype=bool)

        GVi = np.array(self.gt_vol_frac)[time_mask]
        # if GVi[0] > 0:
        #     GVi = GVi - GVi[0]
        GVn = self._compute_normalized_GVG(GVi)

        first_one_idx = np.where(GVn == 1)[0][0]
        last_zero_idx = np.where(GVn[:first_one_idx] == 0)[0][-1]
        
        ind = np.arange(last_zero_idx+1, first_one_idx+1)

        GVG = GVn[ind]

        tG = self.ti[time_mask][ind]
        TG = self.Ti[time_mask][ind]
        PG = self.Pi[time_mask][ind]
        MnG = np.array(self.Mni)[time_mask][ind]
        MgG = np.array(self.Mgi)[time_mask][ind]
        FeG = np.array(self.Fei)[time_mask][ind]
        CaG = np.array(self.Cai)[time_mask][ind]


        ### If we go outside garnet-in area, we need to fill the zero data with the last non-zero value
        non_zero_mask = MnG != 0
        # Use np.maximum.accumulate to propagate the last non-zero index
        # and fill with values from those indices
        last_non_zero = np.maximum.accumulate(non_zero_mask * np.arange(len(MnG)))
        MnG = MnG[last_non_zero]
        MgG = MgG[last_non_zero]
        FeG = FeG[last_non_zero]
        CaG = CaG[last_non_zero]
        
        if new_t is not None:
            # interpolate onto new time values
            interp_TG = interp1d(tG, TG, kind='linear', fill_value='extrapolate')
            interp_PG = interp1d(tG, PG, kind='linear', fill_value='extrapolate')
            interp_MnG = interp1d(tG, MnG, kind='linear', fill_value='extrapolate')
            interp_MgG = interp1d(tG, MgG, kind='linear', fill_value='extrapolate')
            interp_FeG = interp1d(tG, FeG, kind='linear', fill_value='extrapolate')
            interp_CaG = interp1d(tG, CaG, kind='linear', fill_value='extrapolate')
            interp_GVG = interp1d(tG, GVG, kind='linear', fill_value='extrapolate')

            new_TG = interp_TG(new_t)
            new_PG = interp_PG(new_t)
            new_MnG = interp_MnG(new_t)
            new_MgG = interp_MgG(new_t)
            new_FeG = interp_FeG(new_t)
            new_CaG = interp_CaG(new_t)
            new_GVG = interp_GVG(new_t)

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
        ### add in a mask based on self.t_start_growth and self.t_end_growth
        GVi = np.array(self.gt_vol_frac)
        GVn = self._compute_normalized_GVG(GVi)

        first_one_idx = np.where(GVn == 1)[0][0]
        last_zero_idx = np.where(GVn[:first_one_idx] == 0)[0][-1]

        if self.t_end_growth is not None:
            time_mask = self.ti >= self.t_end_growth
            ind = time_mask
        else:
            ind = slice(first_one_idx, None)



        GVG = GVn[ind]
        tG = self.ti[ind]
        TG = self.Ti[ind]
        PG = self.Pi[ind]
        MnG = np.array(self.Mni)[ind]
        MgG = np.array(self.Mgi)[ind]
        FeG = np.array(self.Fei)[ind]
        CaG = np.array(self.Cai)[ind]
        

        ### If we go outside garnet growth area, we need to fill the zero data with the last non-zero value
        non_zero_mask = MnG != 0
        # Use np.maximum.accumulate to propagate the last non-zero index
        # and fill with values from those indices
        last_non_zero = np.maximum.accumulate(non_zero_mask * np.arange(len(MnG)))
        MnG = MnG[last_non_zero]
        MgG = MgG[last_non_zero]
        FeG = FeG[last_non_zero]
        CaG = CaG[last_non_zero]
        
        if new_t is not None:
            # interpolate onto new time values
            interp_TG = interp1d(tG, TG, kind='linear', fill_value='extrapolate')
            interp_PG = interp1d(tG, PG, kind='linear', fill_value='extrapolate')
            interp_MnG = interp1d(tG, MnG, kind='linear', fill_value='extrapolate')
            interp_MgG = interp1d(tG, MgG, kind='linear', fill_value='extrapolate')
            interp_FeG = interp1d(tG, FeG, kind='linear', fill_value='extrapolate')
            interp_CaG = interp1d(tG, CaG, kind='linear', fill_value='extrapolate')
            interp_GVG = interp1d(tG, GVG, kind='linear', fill_value='extrapolate')

            new_TG = interp_TG(new_t)
            new_PG = interp_PG(new_t)
            new_MnG = interp_MnG(new_t)
            new_MgG = interp_MgG(new_t)
            new_FeG = interp_FeG(new_t)
            new_CaG = interp_CaG(new_t)
            new_GVG = interp_GVG(new_t)

            data = np.column_stack([new_t, new_TG, new_PG, new_MnG, new_MgG, new_FeG, new_CaG]).T

        else:
             data = np.column_stack([tG, TG, PG, MnG, MgG, FeG, CaG]).T



        return data



    def generate_garnets(self, size_dist='N', formation_times=None):
        """Generates garnet distributions.

        Parameters:
            size_dist (str or array-like): 
                'N' for a normal distribution, 
                'U' for a uniform distribution, or 
                a user-defined numeric distribution array of length equal to garnet_classes.
            formation_times (array-like, optional):
                If provided, an array of length n_classes specifying the desired formation times
                at which garnets should form. This lets you force garnets to grow at specific conditions along the path.
            
        Returns:
            garnets (list): List of garnet data dictionaries.
        """

        ### add in a mask based on self.t_start_growth and self.t_end_growth
        if self.t_start_growth is not None and self.t_end_growth is not None:
            time_mask = (self.ti >= self.t_start_growth) & (self.ti <= self.t_end_growth)
        elif self.t_start_growth is not None:
            # If only t_start_growth is specified, we consider data from this time onwards
            time_mask = self.ti >= self.t_start_growth
        elif self.t_end_growth is not None:
            time_mask = self.ti <= self.t_end_growth
        else:
            time_mask = np.ones_like(self.ti, dtype=bool)

        GVi = np.array(self.gt_vol_frac)[time_mask]
        # if GVi[0] > 0:
        #     GVi = GVi - GVi[0]
        GVn = self._compute_normalized_GVG(GVi)

        first_one_idx = np.where(GVn == 1)[0][0]
        last_zero_idx = np.where(GVn[:first_one_idx] == 0)[0][-1]
        
        ind = np.arange(last_zero_idx+1, first_one_idx+1)

        GVG = GVn[ind]

        tG = self.ti[time_mask][ind]
        TG = self.Ti[time_mask][ind]
        PG = self.Pi[time_mask][ind]
        MnG = np.array(self.Mni)[time_mask][ind]
        MgG = np.array(self.Mgi)[time_mask][ind]
        FeG = np.array(self.Fei)[time_mask][ind]
        CaG = np.array(self.Cai)[time_mask][ind]


        # Generate radius classes
        n_classes = self.garnet_classes
        r = np.linspace(self.r_min, self.r_max, n_classes, endpoint=True)
        dr = r[1] - r[0]

        # Determine distribution for garnet sizes

        finp = self._get_size_distribution(size_dist, r)
        

        # Normalize the distribution by volume
        fnr = self._normalize_distribution(finp, r)

        Gn = GVG / np.max(GVG)
        tGn = tG

        # G, t, r_r, R = generate_distribution(n_classes, self.r_min, dr, fnr, Gn, tGn)
        G, t_arr, r_r, R = generate_distribution_specified(n_classes, self.r_min, dr, fnr, Gn, tGn, formation_times_spec=formation_times)


        # Interpolate physical properties along the garnet growth
        PGrw = np.interp(t_arr, tG, PG)
        TGrw = np.interp(t_arr, tG, TG)
        Mnrw = np.interp(t_arr, tG, MnG)
        Mgrw = np.interp(t_arr, tG, MgG)
        Ferw = np.interp(t_arr, tG, FeG)
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

    def plot_garnet_summary(self, size_dist='N', garnet_no=0, path=None, formation_times=None, plot_fig=True):
        """
        Plot a summary of the garnet formation results.
        
        Parameters:
            size_dist (str or array-like): 
                'N' for a normal distribution, 'U' for uniform,
                or a user-defined numeric array (length=garnet_classes).
            path (str, optional): Path to save the figure.
        """

        ### add in a mask based on self.t_start_growth and self.t_end_growth
        if self.t_start_growth is not None and self.t_end_growth is not None:
            time_mask = (self.ti >= self.t_start_growth) & (self.ti <= self.t_end_growth)
        elif self.t_start_growth is not None:
            # If only t_start_growth is specified, we consider data from this time onwards
            time_mask = self.ti >= self.t_start_growth
        elif self.t_end_growth is not None:
            time_mask = self.ti <= self.t_end_growth
        else:
            time_mask = np.ones_like(self.ti, dtype=bool)

        GVi = np.array(self.gt_vol_frac)[time_mask]
        if GVi[0] > 0:
            GVi = GVi - GVi[0]
        GVn = self._compute_normalized_GVG(GVi)

        first_one_idx = np.where(GVn == 1)[0][0]
        last_zero_idx = np.where(GVn[:first_one_idx] == 0)[0][-1]
        
        ind = np.arange(last_zero_idx+1, first_one_idx+1)

        GVG = GVn[ind]

        tG = self.ti[time_mask][ind]
        TG = self.Ti[time_mask][ind]
        PG = self.Pi[time_mask][ind]
        MnG = np.array(self.Mni)[time_mask][ind]
        MgG = np.array(self.Mgi)[time_mask][ind]
        FeG = np.array(self.Fei)[time_mask][ind]
        CaG = np.array(self.Cai)[time_mask][ind]

        
        n_classes = self.garnet_classes
        r = np.linspace(self.r_min, self.r_max, n_classes, endpoint=True)
        dr = r[1] - r[0]

        finp = self._get_size_distribution(size_dist, r)

        
        # Normalize the distribution by volume
        fnr = self._normalize_distribution(finp, r)

        Gn = GVG / np.max(GVG)
        tGn = tG

        # Compute garnet distribution characteristics (using generate_distribution)
        G, t_arr, r_r, R = generate_distribution_specified(n_classes, self.r_min, dr, fnr, Gn, tGn, formation_times_spec=formation_times) #generate_distribution(n_classes, self.r_min, dr, fnr, Gn, tGn)
        
        # Interpolate physical properties along garnet growth
        PGrw = np.interp(t_arr, tG, PG)
        TGrw = np.interp(t_arr, tG, TG)
        Mnrw = np.interp(t_arr, tG, MnG)
        Mgrw = np.interp(t_arr, tG, MgG)
        Ferw = np.interp(t_arr, tG, FeG)
        Carw = np.interp(t_arr, tG, CaG)

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
        axs[1, 1].plot(tGn, Gn, 'k', label='Normalized Volume')
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
        for ax in [axs[2, 0], axs[2, 1]]:
            ax.set_title('Chosen Garnet Composition')
            ax.set_xlabel('r')
            ax.set_ylabel('c')
            ax.set_xlim([0, self.r_max])
            i = garnet_no  # highlight the defined garnet number
            ind_local = np.arange(i, n_classes)
            rplt = R[i, :]
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

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if path is not None:
            plt.savefig(path)

        if plot_fig == True:
            plt.show()
        else:
            plt.close()




