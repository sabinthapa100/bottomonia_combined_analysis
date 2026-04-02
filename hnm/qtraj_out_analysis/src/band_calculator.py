#!/usr/bin/env python3
"""
Band Calculator for Multi-Kappa Uncertainty Bands

Combines results from multiple kappa values to produce uncertainty bands.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv


class BandCalculator:
    """
    Calculate uncertainty bands from multiple kappa datasets.
    
    Usage:
        calc = BandCalculator()
        bands = calc.compute_bands_from_files([
            'kappa3/raa_vs_npart.csv',
            'kappa4/raa_vs_npart.csv',
            'kappa5/raa_vs_npart.csv'
        ])
        calc.save_bands(bands, 'combined_bands.csv')
    """
    
    def load_csv(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CSV with R_AA data.
        
        Returns:
            x_values, raa_matrix (rows=bins, cols=states)
        """
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        x_vals = data[:, 0]  # First column (Npart, pT, or y)
        raa_vals = data[:, 1:]  # Remaining columns (R_AA for each state)
        return x_vals, raa_vals
    
    def compute_bands_from_files(
        self,
        filepaths: List[str],
        method: str = 'minmax',
        tolerance: float = 1e-3
    ) -> Dict:
        """
        Compute bands from multiple CSV files.
        
        Args:
            filepaths: List of CSV paths (2 or 3 files)
            method: 'minmax' (take min/max across kappa)
            tolerance: Tolerance for x-value matching
        
        Returns:
            Dictionary with x, central, upper, lower arrays
        """
        if len(filepaths) < 2:
            raise ValueError("Need at least 2 files for bands")
        
        # Load all files
        x_vals_list = []
        raa_list = []
        
        for fpath in filepaths:
            x, raa = self.load_csv(fpath)
            x_vals_list.append(x)
            raa_list.append(raa)
        
        # Use first file's x-values as reference
        x_vals = x_vals_list[0]
        
        # Interpolate other files to match if needed
        raa_interp = [raa_list[0]]
        for i, (x_other, raa_other) in enumerate(zip(x_vals_list[1:], raa_list[1:]), 1):
            if not np.allclose(x_other, x_vals, atol=tolerance, rtol=tolerance):
                print(f"  Warning: Interpolating file {i+1} to match x-grid")
                # Interpolate each state column
                raa_new = np.zeros((len(x_vals), raa_other.shape[1]))
                for j in range(raa_other.shape[1]):
                    raa_new[:, j] = np.interp(x_vals, x_other, raa_other[:, j])
                raa_interp.append(raa_new)
            else:
                raa_interp.append(raa_other)
        
        # Stack RAA values: shape (n_kappa, n_bins, n_states)
        raa_stack = np.stack(raa_interp, axis=0)
        
        # Compute bands
        if method == 'minmax':
            # Upper = max across kappa
            upper = np.max(raa_stack, axis=0)
            # Lower = min across kappa
            lower = np.min(raa_stack, axis=0)
            # Central = middle kappa (if 3) or mean (if 2)
            if len(filepaths) == 3:
                central = raa_stack[1]  # Middle one (kappa4)
            else:
                central = np.mean(raa_stack, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'x': x_vals,
            'central': central,
            'upper': upper,
            'lower': lower,
            'n_kappa': len(filepaths),
        }
    
    def save_bands(self, bands: Dict, output_path: str, x_label: str = 'x'):
        """
        Save bands to CSV.
        
        Format: x, central_state1, upper_state1, lower_state1, central_state2, ...
        """
        x = bands['x']
        central = bands['central']
        upper = bands['upper']
        lower = bands['lower']
        
        n_bins, n_states = central.shape
        
        # Build header
        state_names = ['1S', '2S', '1P0', '1P1', '1P2', '3S', '2P0', '2P1', '2P2']
        header_parts = [x_label]
        for i in range(n_states):
            state = state_names[i] if i < len(state_names) else f'state{i}'
            header_parts.extend([f'{state}_central', f'{state}_upper', f'{state}_lower'])
        
        # Build data array
        data_cols = [x]
        for i in range(n_states):
            data_cols.extend([central[:, i], upper[:, i], lower[:, i]])
        
        data = np.column_stack(data_cols)
        
        # Save
        np.savetxt(output_path, data, delimiter=',', 
                   header=','.join(header_parts), comments='')
        
        print(f"✓ Saved bands: {output_path}")


def main():
    """Test band calculator."""
    print("Band Calculator Test")
    print("=" * 80)
    
    # Example: compute bands if files exist
    test_dir = Path('agent_helper/results/kappa4_test')
    if test_dir.exists():
        print(f"\nLooking for test files in {test_dir}...")
        csv_file = test_dir / 'raa_vs_npart.csv'
        if csv_file.exists():
            print(f"Found: {csv_file}")
            
            calc = BandCalculator()
            x, raa = calc.load_csv(str(csv_file))
            print(f"Loaded: {len(x)} bins, {raa.shape[1]} states")
            print(f"R_AA range: {raa.min():.3f} - {raa.max():.3f}")
    else:
        print("\nNo test files found - run observable_calculator first")
    
    print("\n✅ Band calculator ready")


if __name__ == '__main__':
    main()
