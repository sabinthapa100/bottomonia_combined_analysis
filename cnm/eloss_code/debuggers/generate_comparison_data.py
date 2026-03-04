
import sys
import os
import math
import csv
import numpy as np
import torch
from dataclasses import dataclass, replace

# Ensure local imports work
sys.path.append(os.getcwd())
try:
    import quenching_fast as QF
    from eloss_cronin_dAu import R_loss, R_broad, xA_scalar
    from particle import Particle, PPSpectrumParams
    from glauber import OpticalGlauber, SystemSpec
except ImportError:
    # Try adding subdirectory if running from root
    sys.path.append('eloss_code')
    import quenching_fast as QF
    from eloss_cronin_dAu import R_loss as R_loss_dAu
    from eloss_cronin_dAu import R_broad as R_broad_dAu
    from particle import Particle, PPSpectrumParams
    from glauber import OpticalGlauber, SystemSpec

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

OUTPUT_DIR = "eloss_code/generated_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# dAu 200 GeV Parameters
DAU_ROOTS = 200.0
# User-specified effective lengths for dAu (from Table 2 in arXiv:1304.0901)
DAU_LEFF_USER = {
    "MinBias": 3.27, # From arleo_peigne_fig3_data.py
    "0-20%": 12.87,
    "20-40%": 9.62,
    "40-60%": 7.17,
    "60-88%": 3.84
}
# Rapidity map for dAu
DAU_RAPIDITY = {
    "Backward": -1.7, # range [-2.2, -1.2]
    "Central": 0.0,   # range [-0.35, 0.35]
    "Forward": 1.7    # range [1.2, 2.2]
}

# pPb 5.02 TeV Parameters
PPB_ROOTS = 5020.0
# Rapidity map for pPb (Generic LHC Forward/Backward)
# Forward (p-going): Small x in Pb? No, y>0 usually p-going => low x_A => less suppression?
# Actually Arleo-Peigne predicts strong suppression at large x_F (Forward).
# We will use y values typical for LHCb
PPB_RAPIDITY = {
    "Backward": -3.5, # Pb-going direction (usually)
    "Mid": 0.0,
    "Forward": 3.5    # p-going direction (usually)
}

PT_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Physics parameters
QHAT0 = 0.075
LP_FM = 1.5

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def get_pPb_Leff_dict():
    """Calculate L_eff for pPb 5.02 TeV using Glauber model"""
    # Create SystemSpec for pPb
    spec = SystemSpec(system="pA", roots_GeV=PPB_ROOTS, A=208)
    gl = OpticalGlauber(spec, verbose=False)
    
    # Calculate centrality bins
    cent_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    leff_bins = gl.leff_bins_pA(cent_bins, method="optical", Lp_fm=LP_FM)
    
    # Add MinBias
    leff_bins["MinBias"] = gl.leff_minbias_pA(Lp_fm=LP_FM)
    
    print("Calculated pPb L_eff values:")
    for k, v in leff_bins.items():
        print(f"  {k}: {v:.3f} fm")
        
    return leff_bins

def run_calculation(system_name, roots, leff_dict, rapidity_dict, pT_list, use_AA_kind=False):
    """Run model calculations for a given system"""
    
    # Define Particle (J/psi)
    # pp parameters depend on energy slightly, but we use consistent defaults or scale them?
    # Arleo Peigne often use simple power laws. 
    # For 200 GeV: p0=3.3, m=4.3, n=8.3 (from validation script)
    # For 5.02 TeV: Parameters might differ.
    # We will use standard 200 GeV params for dAu.
    # For pPb, we might need updated params, but for R_pA ratios, exact pp params are less critical 
    # than the quenching logic, provided n is reasonable. We'll use the same class defaults 
    # but acknowledge this might be approximate for pPb spectral shape.
    
    if abs(roots - 200.0) < 1.0:
        pp_params = PPSpectrumParams(p0=3.3, m=4.3, n=8.3)
    else:
        # Generic high energy params (approximate)
        pp_params = PPSpectrumParams(p0=4.5, m=4.0, n=5.5) # Example LHC tune
        
    P = Particle(family="charmonia", state="Jpsi", mass_override_GeV=3.097, pp_params=pp_params)
    
    # QuenchParams
    qp = QF.QuenchParams(
        qhat0=QHAT0, 
        lp_fm=LP_FM, 
        roots_GeV=roots, 
        use_hard_cronin=True, 
        device="cpu"
    )
    
    results = []
    
    for cent_label, L_val in leff_dict.items():
        # Update L
        qp_L = replace(qp, LA_fm=float(L_val))
        
        for rap_label, y_val in rapidity_dict.items():
            for pT in pT_list:
                # Calculate
                try:
                    # R_loss
                    # Use dAu routine explicitly if available or standard
                    # The dAu routine in eloss_cronin_dAu is cleaner for single points
                    from eloss_cronin_dAu import R_loss, R_broad
                    
                    # Note: For dAu, we might want to ensure we use valid physics (AA vs pA).
                    # eloss_cronin_dAu.py is specifically tuned for the dAu analysis.
                    # For pPb, we can use the same functions as they reduce to pA logic 
                    # (xA calculation is slightly different? No, xA_scalar is generic).
                    
                    r_loss = R_loss(P, qp_L, float(y_val), float(pT))
                    r_broad = R_broad(P, qp_L, float(y_val), float(pT))
                    
                    # If system is dAu and user wants "AA" style two-sided quenching?
                    # The current simple R_loss script is one-sided (projectile through target).
                    # For dAu, Arleo Peigne usually treat it as p-A like (d passes through Au).
                    # So one-sided is likely correct for the standard result.
                    
                    r_total = r_loss * r_broad
                    
                    results.append({
                        "System": system_name,
                        "RapidityLabel": rap_label,
                        "y": y_val,
                        "Centrality": cent_label,
                        "L_eff_fm": L_val,
                        "pT": pT,
                        "R_loss": round(r_loss, 4),
                        "R_broad": round(r_broad, 4),
                        "R_total": round(r_total, 4)
                    })
                    
                except Exception as e:
                    print(f"Error for {system_name} {cent_label} {rap_label} pT={pT}: {e}")
                    
    return results

def save_to_csv(results, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', newline='') as f:
        keys = ["System", "RapidityLabel", "y", "Centrality", "L_eff_fm", "pT", "R_loss", "R_broad", "R_total"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} rows to {path}")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    print("Generating dAu 200 GeV Data...")
    dau_results = run_calculation("dAu 200 GeV", DAU_ROOTS, DAU_LEFF_USER, DAU_RAPIDITY, PT_VALUES)
    save_to_csv(dau_results, "dAu_200GeV_Model_Results.csv")
    
    print("\nGenerating pPb 5.02 TeV Data...")
    # Get L_eff for pPb
    ppb_leff = get_pPb_Leff_dict()
    ppb_results = run_calculation("pPb 5.02 TeV", PPB_ROOTS, ppb_leff, PPB_RAPIDITY, PT_VALUES)
    save_to_csv(ppb_results, "pPb_5TeV_Model_Results.csv")
    
    print("\nAll data generation complete.")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
