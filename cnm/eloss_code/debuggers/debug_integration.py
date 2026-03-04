#!/usr/bin/env python3
"""
Debug the energy loss integration in detail
"""
import torch
import math
import sys
import numpy as np
sys.path.append(".")
import quenching_fast as QF
from particle import Particle, PPSpectrumParams
from coupling import alpha_s_provider

# Setup
root_s = 200.0
pp_rhic = PPSpectrumParams(p0=3.3, m=4.3, n=8.3)
P_psi = Particle(family="charmonia", state="avg", pp_params=pp_rhic)
alpha_cst = alpha_s_provider(mode="constant", alpha0=0.5)

qp = QF.QuenchParams(
    qhat0=0.075, lp_fm=1.5, LA_fm=10.23, LB_fm=0.0,
    lambdaQCD=0.25, roots_GeV=root_s, alpha_of_mu=alpha_cst,
    alpha_scale="mT", use_hard_cronin=True,
    device="cpu"
)

# Test point
y, pT = 0.0, 5.0
M = P_psi.M_GeV
mT = math.sqrt(M*M + pT*pT)

print("="*80)
print("Energy Loss Integration Diagnostic")
print("="*80)
print(f"\nTest: y={y}, pT={pT} GeV, L_Au={qp.LA_fm} fm")

# Compute integration limits
y_max = QF.y_max(root_s, mT)
dymax = QF.dymax(y, y_max)

print(f"\nKinematic limits:")
print(f"  y_max = {y_max:.3f}")
print(f"  dymax = {dymax:.3f}")
print(f"  Integration range for δy: [0, {dymax:.3f}]")

# Compute xA
x_val = mT/root_s * math.exp(-y)
x0 = QF.xA0_from_L(qp.LA_fm)
xA = min(x0, x_val)

print(f"\nBjorken-x:")
print(f"  x(kinematic) = {x_val:.6f}")
print(f"  x0(L={qp.LA_fm}) = {x0:.6f}")
print(f"  xA(used) = {xA:.6f}")

# Setup integration nodes
Ny = 96
umin, umax = -30.0, math.log(dymax)
u, wu = QF._gl_nodes_torch(umin, umax, Ny, "cpu")
dy = torch.exp(u)
z = torch.expm1(dy).clamp_min(QF.Z_FLOOR)

# Compute Phat
xA_t = torch.tensor([xA], dtype=torch.float64)
ph = QF.PhatA_t(z, mT, xA_t.expand_as(z), qp, pT=pT)

# Jacobian
ph_jac = ph * torch.exp(dy)  # This is the crucial part!

# F2 ratios
def F2_val(yy, pt):
    n = P_psi.pp.n
    arg = 1.0 - (2.0*math.sqrt(M*M + pt*pt)/root_s)*math.cosh(yy)
    return max(arg, 1e-30)**n

F2_den = F2_val(y, pT)
F2_num = np.array([F2_val(y + float(dy_i), pT) for dy_i in dy])
F2_num_t = torch.tensor(F2_num, dtype=torch.float64)

# Compute terms
term_full = wu * torch.exp(u) * ph_jac * (F2_num_t / F2_den)
Zc_contrib = wu * torch.exp(u) * ph_jac

# Integrate
val = float(torch.sum(term_full).item())
Zc = float(torch.sum(Zc_contrib).item())
p0 = max(0.0, 1.0 - Zc)
R_loss = p0 + val

print(f"\nIntegration results:")
print(f"  Zc (total quenching probability) = {Zc:.6f}")
print(f"  p0 (no-quench probability) = {p0:.6f}")
print(f"  Integral value = {val:.6f}")
print(f"  R_loss = p0 + integral = {R_loss:.6f}")

# Show sample of integrand
print(f"\nSample of integrand (first 10 nodes):")
print(f"{'i':>3} |  {'dy':>8}  {'z':>8} | {'Phat':>12} {'Phat*Jac':>12} | {'F2num/F2den':>12} | {'Integrand':>12}")
print("-" * 95)
for i in range(min(10, len(dy))):
    integrand_val = float(term_full[i])
    print(f"{i:3d} | {float(dy[i]):8.4f} {float(z[i]):8.4f} | {float(ph[i]):12.6e} {float(ph_jac[i]):12.6e} | {F2_num[i]/F2_den:12.6f} | {integrand_val:12.6e}")

print()
print("="*80)
print("DIAGNOSIS:")
print("="*80)

# Check if Phat is too small
max_phat = float(torch.max(ph).item())
max_phat_jac = float(torch.max(ph_jac).item())

print(f"\nPhat statistics:")
print(f"  Max Phat (no Jacobian) = {max_phat:.6e}")
print(f"  Max Phat×Jacobian = {max_phat_jac:.6e}")
print(f"  Zc = {Zc:.6f}")

if Zc < 0.01:
    print("\n⚠️  CRITICAL: Zc is TINY ({:.4f})".format(Zc))
    print("   This means almost NO quenching is happening!")
    print("   Possible causes:")
    print("   1. Phat values are too small")
    print("   2. Integration range (dymax) is too small")
    print("   3. F2 is killing the contrib (check if F2_num/F2_den → 0)")

if val < 0.1 * (1.0 - R_loss):
    print(f"\n⚠️ Integral contribution ({val:.4f}) is small compared to deficit")
    print(f"   Expected deficit: ~{1.0 - 0.928:.4f} (to match ref)")
    print(f"   Actual: {1.0 - R_loss:.4f}")

print("="*80)
