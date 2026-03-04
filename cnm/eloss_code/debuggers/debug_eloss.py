#!/usr/bin/env python3
"""
Debug energy loss calculation for d+Au
"""
import numpy as np
import torch
import math
import sys
sys.path.append(".")
import eloss_cronin_dAu as EDA
import quenching_fast as QF
from particle import Particle, PPSpectrumParams
from coupling import alpha_s_provider

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

# Test at y=0, pT=5 GeV
y_test, pT_test = 0.0, 5.0
L_Au, L_d = 10.23, 6.34

print("="*70)
print("Energy Loss Diagnostic for d+Au Min-Bias")
print("="*70)
print(f"\nTest point: y={y_test}, pT={pT_test} GeV")
print(f"L_Au = {L_Au} fm, L_d = {L_d} fm")

# Test ONE-SIDED (pA style) for comparison
from dataclasses import replace
qp_Au_only = replace(qp_base, LA_fm=L_Au, LB_fm=0.0)
rl_Au_only = EDA.R_loss(P_psi, qp_Au_only, y_test, pT_test)

print(f"\nONE-SIDED (pA-style, Au only):")
print(f"  R_loss(Au only) = {rl_Au_only:.4f}")

# Test TWO-SIDED (AB style)
qp_AB = replace(qp_base, LA_fm=L_Au, LB_fm=L_d)
rl_AB = EDA.R_loss_AB(P_psi, qp_AB, y_test, pT_test)

print(f"\nTWO-SIDED (AB-style, d+Au):")
print(f"  R_loss_AB(d+Au) = {rl_AB:.4f}")

# Expected from reference
print(f"\nREFERENCE (Arleo-Peigné):")
print(f"  R_loss(expected) ≈ 0.928 (at pT=5)")

# Check individual sides
M = P_psi.M_GeV; mT = math.sqrt(M*M + pT_test*pT_test)
y_max = QF.y_max(root_s, mT)
dymax_Au = QF.dymax(y_test, y_max)
dymax_d = QF.dymax(-y_test, y_max)

print(f"\nKinematic Limits:")
print(f"  y_max = {y_max:.3f}")
print(f"  dymax(Au, y={y_test}) = {dymax_Au:.3f}")
print(f"  dymax(d, y={-y_test}) = {dymax_d:.3f}")

# Check xA values
xA_Au = EDA.xA_scalar(P_psi, qp_Au_only, y_test, L_Au, pT_test)
xA_d = EDA.xA_scalar(P_psi, qp_AU, -y_test, L_d, pT_test)

print(f"\nBjorken-x values:")
print(f"  xA(Au) = {xA_Au:.6f}")
print(f"  xA(d)  = {xA_d:.6f}")

# Check qhat
qhat_Au = qp_base.qhat0 * (1e-2/xA_Au)**0.3
qhat_d = qp_base.qhat0 * (1e-2/xA_d)**0.3

print(f"\nTransport coefficients:")
print(f"  qhat(Au) = {qhat_Au:.4f} GeV²/fm")
print(f"  qhat(d)  = {qhat_d:.4f} GeV²/fm")

# Check Phat values
xA_t = torch.tensor([xA_Au])
dytest = 0.1  # Small test shift
z_test = math.expm1(dytest)
phat_Au = QF.PhatA_t(torch.tensor([z_test]), mT, xA_t, qp_Au_only, pT=pT_test)[0]

print(f"\nQuenching Weight Test (δy={dytest}):")
print(f"  z = exp(δy)-1 = {z_test:.4f}")
print(f"  PhatA(z, Au) = {float(phat_Au):.6f}")

# Analyze issue
print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if rl_AB > 0.99:
    print("⚠️  Energy loss is ZERO (R_loss ≈ 1.0)")
    print("\nPossible causes:")
    print("  1. Two-sided integration is canceling out")
    print("  2. L_eff values are too small (coherence length issue)")
    print("  3. Phat kernel returning zero")
    print("  4. Integration limits are wrong")
elif rl_Au_only > 0.99:
    print("⚠️  Even ONE-SIDED has no energy loss!")
    print("\nPossible causes:")
    print("  1. L_eff vs lp issue (L - lp too small)")
    print("  2. xA coherence cutoff too aggressive")
    print("  3. Phat kernel formula error")
else:
    print("✓  One-sided works, but two-sided doesn't")
    print("\nLikely cause:")
    print("  - Two-sided integration logic has an error")

print("="*70)
