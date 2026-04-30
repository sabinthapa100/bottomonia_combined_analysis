
"""
Production Run Script for CNM Effects (LHC 5.02 & 8.16 GeV p+Pb)
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
    for energy in ["5.02", "8.16"]:
        print("----------------------------------------------------------------")
        print(f" Starting CNM Production Run: LHC {energy} TeV p+Pb Charmonia")
        print("----------------------------------------------------------------")
        
        comb = CNMCombine.from_defaults(
            energy=energy, 
            family="charmonia",
            y_edges=np.arange(-5.0, 5.25, 0.5),
            y_windows=[
                (-4.46, -2.96, "Backward"),
                (-1.37, 0.43, "Mid"),
                (2.03, 3.53, "Forward")
            ],
            cent_bins=[(0,20), (20,40), (40,60), (60,80), (80,100)]
        )
        
        # Save outputs to outputs/LHC/CNM_{Energy}_Production
        out_path = HERE.parent / "outputs" / "LHC" / f"CNM_{energy}_Production"
        comb.run_and_save_production(outdir=out_path)
        
        print("----------------------------------------------------------------")
        print(f" Production Complete for {energy} TeV. Results saved to: {out_path}")
        print("----------------------------------------------------------------")

if __name__ == "__main__":
    run_production()
