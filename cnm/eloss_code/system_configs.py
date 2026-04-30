import numpy as np
from glauber import SystemSpec

def get_rhic_pp_params():
    """
    Returns the hardcoded pp parameters used in RHIC F1_t and F2_t functions.
    Overrides global defaults.
    """
    return {
        'p0': 3.3,  # GeV
        'm': 4.8,   # pt dist exponent
        'n': 8.3    # y dist exponent
    }

class LHCConfig:
    """Configuration for pPb LHC 5.02 TeV (and 8.16 TeV placeholders)"""
    # System
    roots5 = 5023.0
    roots8 = 8160.0
    sigma_nn_5 = 67.6
    sigma_nn_8 = 71.0
    A = 208
    
    spec_5TeV = SystemSpec("pA", roots5, A, sigma_nn_mb=sigma_nn_5)
    
    # pp parameters (Charmonia @ 5.02 TeV default)
    pp_params = {'p0': 4.2, 'm': 3.5, 'n': 19.2}
    
    # Analysis
    # Note: notebook uses (60,80), (80,100) for plots, but (60,100) for Leff base calc
    cent_bins_plotting = [(0,20), (20,40), (40,60), (60,80), (80,100)]
    cent_bins_calc = [(0,20), (20,40), (40,60), (60,80), (80,100)]
    
    rapidity_windows = [(-4.46, -2.96), (-1.37, 0.43), (2.03, 3.53)]
    
    # Execution / Plotting
    pt_range_integrated = (0.0, 10.0) # Integration range for R_pA vs y (LHC)
    pT_edges = np.arange(0.0, 20.0 + 2.5, 2.5) # [0.0, ..., 20.0] - Calculation Range
    pT_plot_lim = (0.0, 20.0) # Truncate plots here to avoid edge effects
    
    q0_pair = (0.05, 0.09)
    p0_scale_pair = (0.9, 1.1)
    
    integration_steps = {'Ny_bin': 16, 'Npt_bin': 32}


class RHICConfig:
    """Configuration for dAu RHIC 200 GeV"""
    # System
    roots = 200.0
    sigma_nn = 42.0
    A = 197
    
    # Note: Notebook uses "dA" for main checks but sometimes "pA" for mb reference
    spec = SystemSpec("dA", roots, A, sigma_nn_mb=sigma_nn)
    spec_200GeV = spec # Alias for notebook compatibility
    
    # Custom pp parametrization overrides
    pp_params = get_rhic_pp_params()
    
    # Analysis
    cent_bins = [(0,20), (20,40), (40,60), (60,100)]
    cent_bins_plotting = [(0,20), (20,40), (40,60), (60,80), (80,100)] # Alias for consistency
    # Note: plotting might use same bins or different, notebook suggests standard set
    
    rapidity_windows = [(-2.2, -1.2), (-0.35, 0.35), (1.2, 2.2)]
    
    # Execution / Plotting
    # Calculation Range: y in [-5, 5], pT in [0, 20]
    y_edges = np.arange(-4.0, 4.0 + 0.5, 0.5) 
    pt_range_y_integrated = (0.0, 5.0)
    pt_range_integrated = pt_range_y_integrated # Alias for notebook usage (Cell 4)
    
    pT_edges = np.arange(0.0, 15.0 + 2.5, 2.5) # [0.0, ..., 20.0] - Calculation Range
    
    # Plotting Limits (Truncated)
    y_plot_lim = (-3.0, 3.0)
    pT_plot_lim = (0.0, 10.0)
    
    q0_pair = (0.05, 0.09)
    p0_scale_pair = (0.9, 1.1)
    
    integration_steps = {'Ny_bin': 12, 'Npt_bin': 24}


# =============================================================================
# AA Collision Configurations
# =============================================================================

class PbPbConfig:
    """Configuration for Pb-Pb at LHC 5.02 TeV"""
    roots = 5023.0
    sigma_nn = 67.6  # mb
    A = 208

    spec = SystemSpec("AA", roots, A, sigma_nn_mb=sigma_nn)

    # pp parameters (Bottomonia defaults — same as pPb LHC)
    pp_params = {'p0': 4.2, 'm': 3.5, 'n': 19.2}

    # Centrality
    cent_bins = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    rapidity_windows = [(-4.46, -2.96), (-1.37, 0.43), (2.03, 3.53)]

    # Execution
    pt_range_integrated = (0.0, 10.0)
    pT_edges = np.arange(0.0, 20.0 + 2.5, 2.5)
    pT_plot_lim = (0.0, 20.0)

    q0_pair = (0.05, 0.09)
    p0_scale_pair = (0.9, 1.1)
    integration_steps = {'Ny_bin': 16, 'Npt_bin': 32}


class AuAuConfig:
    """Configuration for Au-Au at RHIC 200 GeV"""
    roots = 200.0
    sigma_nn = 42.0  # mb
    A = 197

    spec = SystemSpec("AA", roots, A, sigma_nn_mb=sigma_nn)

    # pp parameters (RHIC kinematics)
    pp_params = get_rhic_pp_params()

    # Centrality
    cent_bins = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    rapidity_windows = [(-2.2, -1.2), (-0.35, 0.35), (1.2, 2.2)]

    # Execution
    pt_range_integrated = (0.0, 5.0)
    y_edges = np.arange(-4.0, 4.0 + 0.5, 0.5)
    pT_edges = np.arange(0.0, 15.0 + 2.5, 2.5)
    y_plot_lim = (-3.0, 3.0)
    pT_plot_lim = (0.0, 10.0)

    q0_pair = (0.05, 0.09)
    p0_scale_pair = (0.9, 1.1)
    integration_steps = {'Ny_bin': 12, 'Npt_bin': 24}


class OOConfig:
    """Configuration for O-O at LHC 5.36 TeV"""
    roots = 5360.0
    sigma_nn = 67.6  # mb (LHC energy, approximate)
    A = 16

    spec = SystemSpec("AA", roots, A, sigma_nn_mb=sigma_nn)

    # pp parameters (LHC kinematics — same as pPb)
    pp_params = {'p0': 4.2, 'm': 3.5, 'n': 19.2}

    # Centrality — fewer bins for light ion
    # Centrality — fewer bins for light ion
    cent_bins = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 100)]
    # Rapidities for CMS OO (@ LHC) 
    # Mid: |y| < 2.4, Forward: 2.5 to 4.0, Backward: -5.0 to -2.5
    rapidity_windows = [(-2.4, 2.4), (2.5, 4.0), (-5.0, -2.5)]

    # Execution
    pt_range_integrated = (0.0, 10.0)
    pT_edges = np.arange(0.0, 20.0 + 2.5, 2.5)
    pT_plot_lim = (0.0, 20.0)

    q0_pair = (0.05, 0.09)
    p0_scale_pair = (0.9, 1.1)
    integration_steps = {'Ny_bin': 16, 'Npt_bin': 32}
