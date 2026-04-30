import os
import sys
import numpy as np

# Add the cnm/quenching_integration/code directory to sys.path to import OpticalGlauber
# We need to reach it from /mnt/workstation/bottomonia_combined_analysis
# Absolute path: /mnt/workstation/bottomonia_combined_analysis/cnm/quenching_integration/code/
sys.path.insert(0, '/mnt/workstation/bottomonia_combined_analysis/cnm/quenching_integration/code')

from optical_glauber import SystemSpec, OpticalGlauber

# Fix for older numpy version without np.trapezoid
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

def main():
    # OO 5.36 TeV parameters from run_bottomonia_cnm_prim_OO.py
    roots = 5360.0
    A = 16
    sigma_nn = 68.0
    
    spec = SystemSpec(system="AA", roots_GeV=roots, A=A, sigma_nn_mb=sigma_nn)
    
    # Initialize Glauber
    gl = OpticalGlauber(spec, verbose=True)
    
    # Output directory
    outdir = "/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj_inputs/OxygenOxygen5360/glauber-data"
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Exporting OO Glauber tables to {outdir} ...")
    gl.export_tsv(outdir, kind="AA")
    print("Done.")

if __name__ == "__main__":
    main()
