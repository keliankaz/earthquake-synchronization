import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import label
from matplotlib.colors import to_rgba
import glob
import pandas as pd
import re
from typing import Literal, Union
import pygmt
import seaborn as sns
import matplotlib.colors as mcolors

def cycles_colormap(m=None):
    """
    Original code by Sylvain Barbot (USC), ceded by Baoning Wu. Originally generated for Matlab.
    Generates a colormap for representing slip velocity during seismic cycles.
    """
    if m is None:
        m = plt.rcParams['image.lut']

    # normalized RGB values
    cpt = np.array([
        [-12.0, 0, 0, 0],
        [-11.0, 0, 0, 0],
        [-9.3, 106, 135, 196],
        [-8.8, 135, 164, 224],
        [-3.0, 247, 236, 44],
        [-1.0, 239, 64, 35],
        [0.0, 128, 21, 23],
        [1.0, 50, 21, 23]
    ])

    x = -12 + np.linspace(0, 1, m) * 13.0

    # rgb interpolation
    r_interp = interp1d(cpt[:, 0], cpt[:, 1] / 255, kind='linear')
    g_interp = interp1d(cpt[:, 0], cpt[:, 2] / 255, kind='linear')
    b_interp = interp1d(cpt[:, 0], cpt[:, 3] / 255, kind='linear')

    r = r_interp(x)
    g = g_interp(x)
    b = b_interp(x)

    colormap = np.stack([r, g, b], axis=1) 

    return colormap

def seconds_to_years(year_n):
    seconds_per_year = year_n/(365.25 * 24 * 3600)
    return seconds_per_year

def find_events(masked_grid,DX,N2,plotYN):
    binary_mask = ~np.isnan(masked_grid)
    labeled_fault_matrix, ngroups = label(binary_mask)
    start_timestep = []
    end_timestep = []
    timestep_event = [] 
    rupture_length_pixels = []  # length of the rupture in pixels (xaxis)
    group_id = []
    centroids = []
    #labeled_fault_matrix = np.flipud(labeled_fault_matrix)
    
    for groupi in range(1, ngroups + 1):
        group_indices = np.argwhere(labeled_fault_matrix == groupi)
        start_timestep.append(group_indices[0, 0])  # onset of rupture
        end_timestep.append(group_indices[-1, 0]) # end of rupture
        x_indices = group_indices[:, 1]
        rupture_length = np.ptp(x_indices) + 1  # ptp calculates the range, +1 to include both endpoints
        midpt = int((np.max(x_indices)+np.min(x_indices))/2)
        centroids.append(midpt)
        rupture_length_pixels.append(rupture_length)
        group_id.append(groupi)
    
    total_npixels = masked_grid.shape[1]
    pixel_size = (DX*N2)/total_npixels  # size of pixels
    # rate_weakening = 5000/pixel_size
    # mid_point = total_npixels/2
    # obs_edge1 = mid_point - 2300/pixel_size
    # obs_edge2 =  mid_point + 2300/pixel_size
    
    if plotYN == 'Yes':
        plt.figure(dpi=300)
        cmap = plt.get_cmap('tab20', ngroups + 1)  # Add 1 to accommodate event 0
        cmap.colors[0] = to_rgba('white')  # 
        plt.imshow(np.flipud(labeled_fault_matrix), cmap=cmap, aspect='auto', interpolation='none')
        plt.colorbar(label='Group ID')
        plt.xlabel('Position')  
        plt.ylabel('Time step #') 
        plt.xticks(rotation=45)  
        plt.gca().set_facecolor('white')     

        # plt.axvline(mid_point-rate_weakening/2,c='k',linestyle=':',linewidth=1)
        # plt.axvline(mid_point+rate_weakening/2,c='k',linestyle=':',linewidth=1,label='Rate weakening')
        # plt.axvline(mid_point,c='r',linestyle=':',linewidth=1,label='Mid point')
        # plt.axvline(obs_edge1,c='r',linestyle=':',linewidth=1,label='Obs patch 1')
        # plt.axvline(obs_edge2,c='r',linestyle=':',linewidth=1,label='Obs patch 2')
        # plt.legend(fontsize=8)
        
    return (
        start_timestep, 
        end_timestep, 
        rupture_length_pixels, 
        group_id, 
        total_npixels, 
        pixel_size,
        centroids
    )
    
def potency(grid,DX):
    # grid = np.flipud(grid)
    potency = []
    for row in grid:
        dsum = np.sum(row * DX) 
        potency.append(dsum)
    return potency

class MotorcycleModelOutput:

    cell_dict = {
        0.01: (40, 2048),
        0.008: (20, 4096),
        0.005: (10, 8192),
        0.003: (10, 8192),
        0.001: (5, 16384),
    }

    down_sample_factor = 4

    def __init__(self, file_path: str, Dc: float, time_step_rate: int = 20):

        self.file_path = file_path
        self.file = file_path.split("/")[-1] if "/" in file_path else file_path
        self.time_steps_grid = self.tau_grid.shape[1]
        self.Dc = Dc
        self.Dx, self.N2 = self.cell_dict[Dc]
        self.Nx = self.N2 // self.down_sample_factor
        self.fault_data = self.load_fault_data()
        self.time_step_rate = time_step_rate
        self.time_seconds = (
            self.fault_data["Index"].iloc[:: int(self.time_step_rate)].values
        )
        self.time_years = self.time_seconds / 365 / 24 / 60 / 60
        self._duration = self.time_years[-1] - self.time_years[0]

    @property
    def tau_grid(self):
        if not hasattr(self, "_tau_grid"):
            self._tau_grid = self.load_feature_grid("tau")
        return self._tau_grid

    @property
    def slip_grid(self):
        if not hasattr(self, "_slip_grid"):
            self._slip_grid = self.load_feature_grid("slip")
        return self._slip_grid

    @property
    def log10v_grid(self):
        if not hasattr(self, "_log10v_grid"):
            self._log10v_grid = self.load_feature_grid("log10v")
        return self._log10v_grid

    def load_fault_data(self):
        """
        Load the fault data from the file.
        """
        cols = [
            "Index",
            "Displacement",
            "Col3",
            "Col4",
            "Col5",
            "col",
            "cplb",
            "Slip Rate",
            "Col7",
            "Col8",
            "Col9",
            "Col10",
            "Col11",
        ]
        file_pattern = self.file_path + "/patch-01-*.dat"
        matching_files = glob.glob(file_pattern)
        fault_data = pd.read_csv(matching_files[0], sep="\s+", header=None, names=cols)

        return fault_data

    def load_feature_grid(self, feature: Literal["tau", "slip", "log10v"]):
        """
        Load a feature grid from the file.
        feature: str, the feature to load
        """
        return pygmt.load_dataarray(
            self.file_path + "/" + f"fault-01-{feature}.grd", engine="netcdf4"
        )

    def plot_grid(
        self,
        feature: Literal["tau", "slip", "log10v"],
        ax: plt.Axes = None,
        cmap: mcolors.ListedColormap = None,
        crop: Union[float, list] = 0.0,
    ):
        """
        Plot a feature grid from the file.
        feature: str, the feature to plot
        """
        if ax is None:
            _, ax = plt.subplots(dpi=200, figsize=(4,16))

        if cmap is None:
            m = 256  # Define the number of colors
            cmap = mcolors.ListedColormap(cycles_colormap(m))

        sns.heatmap(self.load_feature_grid(feature), cmap=cmap, ax=ax, cbar=False)
        if isinstance(crop, (list, tuple, np.ndarray)):
   
            ax.set_xlim(crop)
        else:
            ax.set_xlim(crop * self.Nx, (1 - crop) * self.Nx)
        ax.invert_yaxis()  # make time go upwards
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Distance along the fault")
        ax.set_ylabel("Time step #")
   

        return ax

    def make_catalog(self, threshold: float = -3):
        """
        Make a catalog of ruptures from the file.
        """
        mask = np.ma.masked_where(
            self.log10v_grid < threshold, self.log10v_grid
        ).filled(np.nan)

        (
            start_timestep,
            end_timestep,
            rupture_length_pixels,
            group_id,
            total_pixels,
            pixel_size,
            midpts,
        ) = find_events(mask, self.Dx, self.N2, "No")

        catalog = pd.DataFrame(
            {
                "start_time": np.array(self.time_years)[start_timestep],
                "end_time": np.array(self.time_years)[end_timestep],
                "length_km": np.array(rupture_length_pixels) * pixel_size / 1000,
                "midpoint_index": midpts,
                "length_pixels": rupture_length_pixels,
            }
        )
        
        return catalog

    def compute_stress_drop(self, dt: float = 0.001):

        catalog = self.make_catalog()

        average_stress_drop = []

        [
            self.calculate_average_stress_drop(
                t0 - dt, t1 + dt, i_midpoint, width
            )
            for t0,t1, i_midpoint, width in zip(
                catalog["start_time"].values,
                catalog["end_time"].values,
                catalog["midpoint_index"].values,
                catalog["length_pixels"].values,
            )
        ]

        return average_stress_drop

    def calculate_average_stress_drop(self, t1, t2, center_index, width, time_units: Literal["seconds", "years"] = "years"):
        
        t = self.time_years if time_units == "years" else self.time_seconds

        t1_index = np.argmin(np.abs(t - t1))
        t2_index = np.argmin(np.abs(t - t2))
        
        tau_0 = self.tau_grid[t1_index, center_index - width // 2 : center_index + width // 2]
        tau_1 = self.tau_grid[t2_index, center_index - width // 2 : center_index + width // 2]

        return np.mean(tau_1 - tau_0)

class CoplanarFaultModelOutput(MotorcycleModelOutput):
    """
    This class is used to analyze the outputs of Motorcycle models for two coplanar faults.
    """

    def __init__(self, file_path: str, Dc: float=None, time_step_rate: int = 20):
        """
        Initialize the CoplanarFaultModelOutput class.

        Expects a filename with the following format:
        output_Dc_XX/output_Dc_XX_L1_<L1>_m_L2_<L2>_m_S_<S>_m/

        """
        self.file = file_path.split("/")[-1] if "/" in file_path else file_path
        self.L1 = self.query_file_name("L1")
        self.L2 = self.query_file_name("L2")
        
        if Dc is None:
            self.Dc = self.query_file_name("Dc")
        else:
            self.Dc = Dc
        super().__init__(file_path, self.Dc)

    def query_file_name(self, pattern: Literal["L1", "L2", "Dc"]) -> str:
        match = re.search(f"{pattern}_(\d+)", self.file)
        if match:
            match = match.group(1)
            if pattern == "Dc":
                match = f"0.{match}"
        return float(match) if match else None

    def get_combined_catalog(self):

        catalog = super().make_catalog()

        fault_id = np.zeros(len(catalog))
        fault_id[catalog["midpoint_index"] < self.Nx // 2] = 1
        fault_id[catalog["midpoint_index"] > self.Nx // 2] = 2

        catalog["fault_id"] = fault_id

        return catalog

    def get_catalogs(self):

        catalog = super().make_catalog()

        catalog_1 = catalog[catalog["midpoint_index"] < self.Nx // 2]
        catalog_2 = catalog[catalog["midpoint_index"] > self.Nx // 2]

        return catalog_1, catalog_2

    def estimate_coupling(self, stressor: Literal["1_to_2", "2_to_1"] = "1_to_2", system_spanning: bool = True, system_spanning_threshold: float = 1):
        """
        Estimate the coupling between the two faults.
        """

        catalogs = [c for c in self.get_catalogs()] # list comprehension to unpack the two catalogs into list
        
        if system_spanning:
            catalogs[0] = catalogs[0][catalogs[0]["length_km"] > self.L1/1000 * system_spanning_threshold]
            catalogs[1] = catalogs[1][catalogs[1]["length_km"] > self.L2/1000 * system_spanning_threshold]
        
        if stressor == "1_to_2":
            stressor_catalog, stressee_catalog = catalogs[0], catalogs[1]
        elif stressor == "2_to_1":
            stressor_catalog, stressee_catalog = catalogs[1], catalogs[0]
        else:
            raise ValueError(f"Invalid stressor: {stressor}")
        
        t_stressor_start  = stressor_catalog["start_time"].values
        t_stressor_end = stressor_catalog["end_time"].values
        t_stressee_start = stressee_catalog["start_time"].values
        t_stressee_end = stressee_catalog["end_time"].values

        stress_transfer_list = []
        stress_drop_list = []
        coupling_list = []
        
        for ti, tf in zip(t_stressor_start, t_stressor_end):
            next_event_bool = t_stressee_start >= ti
            # break if no events
            if not np.any(next_event_bool):
                break
            
            next_event = np.where(next_event_bool)[0][0]

            stress_transfer = self.calculate_average_stress_drop(
                ti,
                tf, 
                stressee_catalog["midpoint_index"].values[next_event],
                stressee_catalog["length_pixels"].values[next_event],
            )

            stress_drop = self.calculate_average_stress_drop(
                t_stressee_start[next_event],
                t_stressee_end[next_event],
                stressee_catalog["midpoint_index"].values[next_event],
                stressee_catalog["length_pixels"].values[next_event],
            )

            stress_transfer_list.append(stress_transfer)
            stress_drop_list.append(stress_drop)
            coupling_list.append(stress_transfer/stress_drop)
            
        stress_transfer_array = np.concatenate([stress_transfer_list])
        stress_drop_array = np.concatenate([stress_drop_list])
        coupling_array = np.concatenate([coupling_list])
        
        return stress_transfer_array, stress_drop_array, coupling_array
        
    def plot_catalog(
        self,
        ax: plt.Axes = None,
    ):
        """
        Plot the catalog colored by fault id
        """
        if ax is None:
            _, ax = plt.subplots(dpi=200, figsize=(8,2))
            
        catalog1, catalog2 = self.get_catalogs()
        
        colors = ["mediumpurple", "darkseagreen"]
        
        for i, catalog in enumerate([catalog1, catalog2]):
            markerline, stemlines, baseline = ax.stem(catalog["start_time"], catalog["length_km"], label=f"Fault {i+1}") 
            plt.setp(markerline, color=colors[i],lw=0.1,ms=2)
            plt.setp(stemlines, color=colors[i],lw=0.5)
            plt.setp(baseline, color='white')
        
        ax.set(
            xlabel="Time (years)",
            ylabel="Length (km)",
            ylim=[0, ax.get_ylim()[1]],
        )

        return ax
