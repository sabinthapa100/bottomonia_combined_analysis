
"""
Production Run Script for CNM Effects (RHIC 200 GeV d+Au)
This script runs the full data generation and plotting pipeline without a notebook interface.
"""
import sys
import os
import numpy as np
from pathlib import Path

# Fix paths to include local modules
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "../cnm_combine"))
sys.path.append(str(HERE / "../eloss_code"))
sys.path.append(str(HERE / "../npdf_code"))

from cnm_combine import CNMCombine

def run_production():
    print("----------------------------------------------------------------")
    print(" Starting CNM Production Run: RHIC 200 GeV d+Au Charmonia")
    print("----------------------------------------------------------------")
    
    comb = CNMCombine.from_defaults(
        energy="200", 
        family="charmonia",
        y_edges=np.arange(-2.2, 2.3, 0.2), # Fine binning for line plots
        y_windows=[
            (-2.2, -1.2, "Backward"),
            (-0.35, 0.35, "Mid"),
            (1.2, 2.2, "Forward")
        ],
        cent_bins=[(0,20), (20,40), (40,60), (60,100)]
    )
    
    # Save outputs to outputs/RHIC/CNM_200_Production
    out_path = HERE.parent / "outputs" / "RHIC" / "CNM_200_Production"
    comb.run_and_save_production(outdir=out_path)
    
    print("----------------------------------------------------------------")
    print(f" Production Complete. Results saved to: {out_path}")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    run_production()
