#!/usr/bin/env python3
"""
Binning configuration helpers for qtraj analysis.

The legacy generic helpers remain available, but the canonical published
binning now comes from the observable registry so callers can request the
exact Mathematica x-grids used for the thesis comparison figures.
"""

import numpy as np
from typing import List, Tuple, Optional

from qtraj_analysis.observable_registry import (
    get_mathematica_bin_edges,
    get_mathematica_grid,
    get_mathematica_grid_values,
)


class BinningConfig:
    """Flexible binning configuration."""
    
    @staticmethod
    def create_bins(min_val: float, max_val: float, bin_size: float) -> np.ndarray:
        """Create bin edges."""
        return np.arange(min_val, max_val + bin_size/2, bin_size)
    
    @classmethod
    def default_y_bins(cls, max_y: float = 5.0, bin_size: float = 0.5) -> np.ndarray:
        """
        Default rapidity binning.
        
        Args:
            max_y: Maximum |y| (default: 5.0)
            bin_size: Bin width (default: 0.5)
        
        Returns:
            Bin edges: [-5, -4.5, -4, ..., 4.5, 5]
        """
        return cls.create_bins(-max_y, max_y, bin_size)
    
    @classmethod
    def default_pt_bins(cls, max_pt: float = 40.0, bin_size: float = 2.5) -> np.ndarray:
        """
        Default pT binning.
        
        Args:
            max_pt: Maximum pT (default: 40 GeV)
            bin_size: Bin width (default: 2.5 GeV)
        
        Returns:
            Bin edges: [0, 2.5, 5, ..., 37.5, 40]
        """
        return cls.create_bins(0.0, max_pt, bin_size)
    
    @classmethod
    def paper_pt_bins(cls) -> np.ndarray:
        """
        Canonical PbPb 5.02 TeV R_AA(pT) bin edges used in the published
        Mathematica comparison.
        """
        edges = get_mathematica_bin_edges("pbpb5023_raavspt")
        if edges is None:
            raise ValueError("Registry does not define exact pT bin edges for pbpb5023_raavspt")
        return edges

    @classmethod
    def paper_y_bins(cls) -> np.ndarray:
        """
        Canonical PbPb 5.02 TeV R_AA(y) Mathematica x-grid values.

        The PbPb 5.02 TeV rapidity export stores sample points rather than an
        explicit edge list, so this returns the exact exported x-grid.
        """
        return get_mathematica_grid_values("pbpb5023_raavsy")
    
    @classmethod
    def rhic_y_bins(cls) -> np.ndarray:
        """RHIC acceptance: |y| < 1."""
        return np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    @classmethod
    def lhcb_y_bins(cls) -> np.ndarray:
        """LHCb forward: 2 < y < 4.5."""
        return np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

    @classmethod
    def mathematica_grid_values(cls, observable_id: str) -> np.ndarray:
        """Return the exact Mathematica x-grid stored in the registry."""
        return get_mathematica_grid_values(observable_id)

    @classmethod
    def mathematica_bin_edges(cls, observable_id: str) -> Optional[np.ndarray]:
        """Return exact Mathematica bin edges when the registry knows them."""
        return get_mathematica_bin_edges(observable_id)

    @classmethod
    def mathematica_grid_spec(cls, observable_id: str):
        """Return the full registry grid specification for an observable id."""
        return get_mathematica_grid(observable_id)


if __name__ == '__main__':
    print("Binning Configurations")
    print("=" * 80)
    
    print("\nDefault y bins (±5, step 0.5):")
    y_bins = BinningConfig.default_y_bins()
    print(f"  {len(y_bins)-1} bins: {y_bins}")
    
    print("\nDefault pT bins (0-40, step 2.5):")
    pt_bins = BinningConfig.default_pt_bins()
    print(f"  {len(pt_bins)-1} bins: {pt_bins}")
    
    print("\nPaper pT bins:")
    print(f"  {BinningConfig.paper_pt_bins()}")
    
    print("\nPaper y bins:")
    print(f"  {BinningConfig.paper_y_bins()}")
