
import sys
import os
import math
import numpy as np
import torch
from dataclasses import dataclass, replace

sys.path.append('quenching_integration/code')
sys.path.append('.')

import quenching_fast as QF
import eloss_cronin_dAu as ECD
from particle import Particle, PPSpectrumParams

# Parameters
ROOTS_GEV = 200.0
# p0=3.3, m=4.3, n=8.3 (Arleo 200 GeV)
PP_PARAMS = PPSpectrumParams(p0=3.3, m=4.3, n=8.3)

def run_debug():
    P = Particle(family="charmonia", state="Jpsi", mass_override_GeV=3.097, pp_params=PP_PARAMS)
    
    qp = QF.QuenchParams(
        qhat0=0.075, lp_fm=1.5, roots_GeV=ROOTS_GEV, use_hard_cronin=True, device="cpu"
    )
    
    y = 1.7
    pT = 4.0
    L = 10.23 # MinBias
    qp = replace(qp, LA_fm=L)
    
    print(f"--- Debug Calculation: y={y}, pT={pT}, L={L} ---")
    
    # 1. R_broad
    rb = ECD.R_broad(P, qp, y, pT)
    # 2. R_loss
    rl = ECD.R_loss(P, qp, y, pT)
    # 3. R_total
    rt = rb * rl
    
    print(f"My Code Results:")
    print(f"  R_broad = {rb:.4f}")
    print(f"  R_loss  = {rl:.4f}")
    print(f"  R_total = {rt:.4f}")
    
    print(f"\nTarget (User Data):")
    print(f"  R_broad ~ 1.31")
    print(f"  R_total ~ 1.06")
    print(f"  Implied R_loss ~ {1.06/1.31:.4f}")
    
    # Check dpt
    xA = ECD.xA_scalar(P, qp, y, pT)
    xA_t = torch.tensor([xA], dtype=torch.float64)
    dpt = QF._dpt_from_xL_t(qp, xA_t, L, hard=True)[0].item()
    print(f"\nDiagnostics:")
    print(f"  xA = {xA:.5f}")
    print(f"  dpt = {dpt:.4f} GeV")

    print("\n--- Tuning Scenarios ---")
    
    # Scenario 1: Increase alpha_s to 0.6, Decrease qhat to 0.06
    qp1 = replace(qp, qhat0=0.06, alpha_of_mu=lambda mu: 0.6)
    rb1 = ECD.R_broad(P, qp1, y, pT)
    rl1 = ECD.R_loss(P, qp1, y, pT)
    rt1 = rb1 * rl1
    print(f"Scenario 1 (alpha=0.5, qhat=0.075):")
    print(f"  R_broad = {rb1:.4f} (Target 1.31)")
    print(f"  R_loss  = {rl1:.4f} (Target 0.81)")
    print(f"  R_total = {rt1:.4f} (Target 1.06)")
    
    # Scenario 2: Decrease lp to 0.7 (proton radius?), Decrease qhat to 0.055
    qp2 = replace(qp, qhat0=0.055, lp_fm=0.7) 
    rb2 = ECD.R_broad(P, qp2, y, pT)
    rl2 = ECD.R_loss(P, qp2, y, pT)
    rt2 = rb2 * rl2
    print(f"Scenario 2 (lp=0.7, qhat=0.055):")
    print(f"  R_broad = {rb2:.4f}")
    print(f"  R_loss  = {rl2:.4f}")
    print(f"  R_total = {rt2:.4f}")

    return

if __name__ == "__main__":
    run_debug()
