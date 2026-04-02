#!/usr/bin/env python3
"""
Publication-Quality Plotting for QTraj Analysis

Generates plots for single-kappa and multi-kappa band results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple


def setup_publication_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def load_csv_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV with R_AA data."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    x = data[:, 0]  # Npart, pT, or y
    raa = data[:, 1:]  # R_AA for each state
    return x, raa


def plot_single_kappa(
    csv_path: str,
    output_path: str,
    states: List[str] = ['1S', '2S', '3S'],
    xlabel: str = r'$N_\mathrm{part}$',
    ylabel: str = r'$R_\mathrm{AA}$'
):
    """
    Plot single kappa results.
    
    Args:
        csv_path: Path to CSV file
        output_path: Output PDF path
        states: List of states to plot (default: 1S, 2S, 3S)
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    setup_publication_style()
    
    # Load data
    x, raa = load_csv_data(csv_path)
    
    # State mapping
    state_cols = {
        '1S': 0, '2S': 1, '1P0': 2, '1P1': 3, '1P2': 4,
        '3S': 5, '2P0': 6, '2P1': 7, '2P2': 8
    }
    
    colors = {'1S': 'blue', '2S': 'red', '3S': 'green'}
    
    fig, ax = plt.subplots()
    
    for state in states:
        if state in state_cols:
            col_idx = state_cols[state]
            color = colors.get(state, 'black')
            ax.plot(x, raa[:, col_idx], 'o-', label=rf'$\Upsilon({state})$', 
                   color=color, markersize=6)
    
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_ylim(0, 1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_bands(
    bands_csv: str,
    output_path: str,
    exp_data: Optional = None,
    states: List[str] = ['1S', '2S', '3S'],
    xlabel: str = r'$N_\mathrm{part}$',
    ylabel: str = r'$R_\mathrm{AA}$'
):
    """
    Plot multi-kappa uncertainty bands.
    
    Args:
        bands_csv: Path to bands CSV
        output_path: Output PDF path
        exp_data: ExperimentalDataset object (optional)
        states: States to plot
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    setup_publication_style()
    
    # Load bands
    data = np.loadtxt(bands_csv, delimiter=',', skiprows=1)
    x = data[:, 0]
    
    # Parse columns: x, central_1S, upper_1S, lower_1S, central_2S, ...
    state_mapping = {
        '1S': (1, 2, 3),    # (central, upper, lower) column indices
        '2S': (4, 5, 6),
        '1P0': (7, 8, 9),
        '1P1': (10, 11, 12),
        '1P2': (13, 14, 15),
        '3S': (16, 17, 18),
        '2P0': (19, 20, 21),
        '2P1': (22, 23, 24),
        '2P2': (25, 26, 27),
    }
    
    colors = {'1S': 'blue', '2S': 'red', '3S': 'green'}
    
    fig, ax = plt.subplots()
    
    for state in states:
        if state not in state_mapping:
            continue
        
        c_idx, u_idx, l_idx = state_mapping[state]
        color = colors.get(state, 'black')
        
        central = data[:, c_idx]
        upper = data[:, u_idx]
        lower = data[:, l_idx]
        
        # Plot band
        ax.fill_between(x, lower, upper, alpha=0.3, color=color, 
                        label=rf'$\Upsilon({state})$ theory')
        ax.plot(x, central, '-', color=color, linewidth=2)
    
    # Add experimental data if provided
    if exp_data is not None:
        x_exp, y_exp, err_low, err_high = exp_data.get_arrays()
        ax.errorbar(x_exp, y_exp, yerr=[err_low, err_high],
                   fmt='o', color='black', markersize=8,
                   label=f'{exp_data.experiment} data', capsize=5)
    
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_ylim(0, 1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    """Test plotting functions."""
    # Example usage
    test_file = Path('outputs/RHIC/AuAu/200GeV/bands/kappa4-5/data/raa_vs_npart_bands.csv')
    
    if test_file.exists():
        output = test_file.parent.parent / 'plots' / 'raa_vs_npart_bands.pdf'
        output.parent.mkdir(parents=True, exist_ok=True)
        
        plot_bands(
            str(test_file),
            str(output),
            exp_data=None,
            states=['1S', '2S', '3S']
        )
    else:
        print(f"Test file not found: {test_file}")


if __name__ == '__main__':
    main()
