import sys
import os

# Add code directory to path
sys.path.append(os.path.abspath("./")) # Run from project root

from eloss_code.system_configs import LHCConfig, RHICConfig
from eloss_code.glauber_wrapper import GlauberWrapper

def verify_lhc():
    print("--- Verifying LHC Config ---")
    spec = LHCConfig.spec_5TeV
    print(f"System: {spec.system}, Roots: {spec.roots_GeV}, A: {spec.A}, Sigma: {spec.sigma_nn_mb}")
    
    gl = GlauberWrapper(spec.system, spec.roots_GeV, spec.A, spec.sigma_nn_mb)
    
    # Check MinBias Leff
    leff_mb = gl.leff_minbias_pA()
    print(f"Leff MB (pA): {leff_mb:.4f} fm")
    
    # Check Binned Leff
    print("Leff Bins:")
    leff_bins = gl.leff_bins_pA(LHCConfig.cent_bins_plotting)
    for label, val in leff_bins.items():
        print(f"  {label}: {val:.4f} fm")
        
def verify_rhic():
    print("\n--- Verifying RHIC Config ---")
    spec = RHICConfig.spec
    print(f"System: {spec.system}, Roots: {spec.roots_GeV}, A: {spec.A}, Sigma: {spec.sigma_nn_mb}")
    
    gl = GlauberWrapper(spec.system, spec.roots_GeV, spec.A, spec.sigma_nn_mb)
    
    # Check MinBias Leff - notebook used "pA" call on "dA" object effectively?
    # Or maybe leff_minbias_dA? Let's see what wrapper does.
    try:
        leff_mb = gl.leff_minbias_pA() # Wrapper redirects to dA if system is dA
        print(f"Leff MB (via wrapper): {leff_mb:.4f} fm")
    except Exception as e:
        print(f"Error calculating MB Leff: {e}")

    # Check Binned Leff
    print("Leff Bins:")
    try:
        leff_bins = gl.leff_bins_pA(RHICConfig.cent_bins)
        for label, val in leff_bins.items():
            print(f"  {label}: {val:.4f} fm")
    except Exception as e:
        print(f"Error calculating Binned Leff: {e}")

if __name__ == "__main__":
    verify_lhc()
    verify_rhic()
