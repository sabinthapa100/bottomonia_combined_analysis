#!/usr/bin/env python3
"""
Systematic search for correct L_eff configuration
Testing hypothesis: Energy loss and broadening may use DIFFERENT L_eff values
"""
import numpy as np
import sys
from dataclasses import replace
import torch

sys.path.append(".")
import quenching_fast as QF
from particle import Particle, PPSpectrumParams
from coupling import alpha_s_provider

# Reference data (Central rapidity, pT=5 GeV)
ref_eloss_at_5 = 0.928  # From user's table
ref_broad_at_5 = 1.25   # From user's table
ref_total_at_5 = 1.16   # From user's table

# Setup
root_s = 200.0
pp_rhic = PPSpectrumParams(p0=3.3, m=4.3, n=8.3)
P_psi = Particle(family="charmonia", state="avg", pp_params=pp_rhic)
alpha_cst = alpha_s_provider(mode="constant", alpha0=0.5)

qp_base = QF.QuenchParams(
    qhat0=0.075, lp_fm=1.5,
    lambdaQCD=0.25, roots_GeV=root_s, alpha_of_mu=alpha_cst,
    alpha_scale="mT", use_hard_cronin=True,
    device="cpu"
)

# Import functions
def F2_t(P, y, pT, roots):
    import torch
    n = P.pp.n
    M = P.M_GeV
    mT = torch.sqrt(torch.tensor(pT*pT + M*M))
    arg = 1.0 - (2.0*mT/roots)*torch.cosh(torch.tensor(y))
    return torch.clamp(arg, min=1e-30)**n

def xA_scalar(P, qp, y_eff, L_fm, pT):
    import math
    M = P.M_GeV
    mT = math.sqrt(M*M + pT*pT)
    x2 = (mT/qp.roots_GeV)*math.exp(-y_eff)
    x0 = QF.xA0_from_L(L_fm)
    return min(x0, x2)

def R_loss_simple(P, qp, y, pT, L_eff):
    """Compute R_loss with specific L_eff"""
    import math
    dev = "cpu"
    M = P.M_GeV; mT = math.sqrt(M*M + pT*pT)
    y_max = QF.y_max(qp.roots_GeV, mT)
    dymax = QF.dymax(y, y_max)
    if dymax <= QF.DY_EPS: return 1.0

    xA = xA_scalar(P, qp, y, L_eff, pT)
    xA_t = torch.tensor([xA], dtype=torch.float64, device=dev)
    y0 = torch.tensor([y], dtype=torch.float64, device=dev)
    pT0 = torch.tensor([pT], dtype=torch.float64, device=dev)
    
    qp_test = replace(qp, LA_fm=L_eff)
    denom = F2_t(P, y0[0].item(), pT0[0].item(), qp.roots_GeV)
    
    umin, umax = -30.0, math.log(dymax)
    u, wu = QF._gl_nodes_torch(umin, umax, 96, dev)
    dy = torch.exp(u); z = torch.expm1(dy).clamp_min(QF.Z_FLOOR)
    
    ph = QF.PhatA_t(z, mT, xA_t.expand_as(z), qp_test, pT=pT) * torch.exp(dy)

    nums = torch.tensor([float(F2_t(P, y + float(dy_v), pT, qp.roots_GeV)) for dy_v in dy])
    val = torch.sum(wu * torch.exp(u) * ph * (nums/denom))
    Zc  = torch.sum(wu * torch.exp(u) * ph)
    return float((torch.clamp_min(1.0 - Zc, 0.0) + val).item())

def R_broad_simple(P, qp, y, pT, L_eff):
    """Compute R_broad with specific L_eff"""
    import torch
    dev = "cpu"
    xA = xA_scalar(P, qp, y, L_eff, pT)
    xA_t = torch.tensor([xA], dtype=torch.float64, device=dev)
    
    qp_test = replace(qp, LA_fm=L_eff)
    dpt = QF._dpt_from_xL_t(qp_test, xA_t, L_eff, hard=True)[0]
    if abs(float(dpt)) < 1e-10: return 1.0

    phi, wphi, cphi, sphi = QF._phi_nodes_gl_torch(96, dev)
    pshift = QF._shift_pT_pA(pT, dpt, cphi, sphi)

    # F1
    p0, m = P.pp.p0, P.pp.m
    p0sq = p0*p0
    F1den = (p0sq / (p0sq + pT*pT))**m
    F1num = (p0sq / (p0sq + pshift*pshift))**m
    
    # F2
    F2den = float(F2_t(P, y, pT, qp.roots_GeV))
    F2num = torch.tensor([float(F2_t(P, y, float(p), qp.roots_GeV)) for p in pshift.cpu().numpy()])
    
    return float(torch.sum(wphi * (F1num/F1den) * (F2num/F2den)).item())

# Test configurations
print("="*80)
print("Systematic L_eff Search for d+Au Min-Bias")
print("="*80)
print(f"\nTest point: y=0, pT=5 GeV")
print(f"Reference: R_loss={ref_eloss_at_5:.3f}, R_broad={ref_broad_at_5:.3f}, R_total={ref_total_at_5:.3f}\n")

configs = [
    ("Same L for both (Table 2)", 2.95, 2.95),
    ("Same L for both (Page 5)", 10.23, 10.23),
    ("Energy Loss=Table2, Broad=Page5", 2.95, 10.23),
    ("Energy Loss=Page5, Broad=Table2", 10.23, 2.95),
    ("Energy Loss=5fm, Broad=Page5", 5.0, 10.23),
    ("Energy Loss=7fm, Broad=Page5", 7.0, 10.23),
    ("Energy Loss=3.5fm, Broad=10.23", 3.5, 10.23),
]

results = []
for name, L_eloss, L_broad in configs:
    rl = R_loss_simple(P_psi, qp_base, 0.0, 5.0, L_eloss)
    rb = R_broad_simple(P_psi, qp_base, 0.0, 5.0, L_broad)
    rt = rl * rb
    
    err_l = abs(rl - ref_eloss_at_5) / ref_eloss_at_5 * 100
    err_b = abs(rb - ref_broad_at_5) / ref_broad_at_5 * 100
    err_t = abs(rt - ref_total_at_5) / ref_total_at_5 * 100
    
    results.append((name, L_eloss, L_broad, rl, rb, rt, err_l, err_b, err_t))
    
    print(f"{name:35s} L_e={L_eloss:5.2f} L_b={L_broad:5.2f}")
    print(f"  R_loss={rl:.4f} (err:{err_l:5.1f}%)  R_broad={rb:.4f} (err:{err_b:5.1f}%)  R_total={rt:.4f} (err:{err_t:5.1f}%)")
    print()

# Find best
best = min(results, key=lambda x: x[6] + x[7] + x[8])  # Minimum total error
print("="*80)
print(f"BEST CONFIGURATION: {best[0]}")
print(f"  L_eloss = {best[1]:.2f} fm, L_broad = {best[2]:.2f} fm")
print(f"  Total error: {best[6] + best[7] + best[8]:.1f}%")
print("="*80)
