
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

# Parameters
ROOTS_GEV = 200.0
PP_PARAMS = ECD.Particle(M_GeV=3.097).pp # p0=3.3, m=4.3, n=8.3

def debug_forward_calc(use_max_x0=False):
    P = ECD.Particle(M_GeV=3.097)
    P.pp = ECD.PPSpectrumParams(p0=3.3, m=4.3, n=8.3)
    
    qp = QF.QuenchParams(
        qhat0=0.075, lp_fm=1.5, roots_GeV=ROOTS_GEV, use_hard_cronin=True, device="cpu"
    )
    
    y = 1.7
    pT = 0.0
    L = 10.23
    qp = replace(qp, LA_fm=L)
    
    mT = math.sqrt(3.097**2 + pT**2)
    x2 = (mT/ROOTS_GEV) * math.exp(-y)
    x0 = QF.xA0_from_L(L)
    
    if use_max_x0:
        xA = max(x0, x2)
    else:
        xA = min(x0, x2)
        
    # Calculate dpt
    xA_t = torch.tensor([xA], dtype=torch.float64, device="cpu")
    dpt = QF._dpt_from_xL_t(qp, xA_t, L, hard=True)[0].item()
    
    # Calculate R_broad
    # Need to patch QF or ECD to use new xA?
    # ECD.R_broad calls ECD.xA_scalar which calls min(x0, x2).
    # I will verify what ECD.xA_scalar does.
    # It mimics min(x0, x2).
    
    # R_broad calculation manual
    rb = ECD.R_broad(P, qp, y, pT) 
    
    print(f"Mode: {'MAX' if use_max_x0 else 'MIN'}")
    print(f"y={y}, pT={pT}, L={L}")
    print(f"x2={x2:.5f}, x0={x0:.5f}, xA={xA:.5f}")
    print(f"qhat(xA)={qp.qhat0 * (0.01/xA)**0.3:.5f}")
    print(f"dpt={dpt:.5f}")
    print(f"R_broad (default code)={rb:.5f}") # This always uses MIN because ECD imports QF
    
    return dpt

print("--- Standard (MIN) ---")
dpt1 = debug_forward_calc(use_max_x0=False)

print("\n--- Testing MAX effect (Manual Calc) ---")
dpt2 = debug_forward_calc(use_max_x0=True)
