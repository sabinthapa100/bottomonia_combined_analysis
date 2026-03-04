
import math
import numpy as np
import torch
import logging
from dataclasses import replace
import quenching_fast as QF

# Setup module-level logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)
logger.setLevel(logging.INFO)

# Arleo & Peigne d+Au 200 GeV L_eff
# L_Au = 10.23 fm from paper (page 5, footnote)
DAU_LEFF_PAIRS = {
    "MinBias": (10.23, 0.0), # (LA, LB) -> Gold, Deuteron
    "0-20%":   (12.87, 0.0),
    "20-40%":  (9.62, 0.0),
    "40-60%":  (7.17, 0.0),
    "60-88%":  (3.84, 0.0)
}

def validate_physics_params(P, qp, y, pT):
    """Checks for unphysical parameters and kinematic limits."""
    M = float(P.M_GeV)
    mT = math.sqrt(M**2 + pT**2)
    y_max = math.log(max(qp.roots_GeV/mT, 1.0001))
    
    if abs(y) > y_max:
        logger.warning(f"Rapidity |y|={abs(y)} exceeds kinematic limit {y_max:.2f}")
        return False
    if qp.LA_fm < 0 or qp.LB_fm < 0:
        logger.error(f"Negative path length: LA={qp.LA_fm}, LB={qp.LB_fm}")
        return False
    return True

ARLEO_PEIGNE_LEFF_USER = {k: v[0] for k, v in DAU_LEFF_PAIRS.items()}

# ------------------------------------------------------------------------------
# BROADENING ONLY (EXACT copy from p+Pb eloss_cronin.py)
# ------------------------------------------------------------------------------
def R_broad(P, qp, y, pT, Nphi=64):
    """
    pT-broadening factor R^broad_pA(y, pT) - EXACT copy from p+Pb.
    """
    device = qp.device if hasattr(qp, "device") and qp.device else "cpu"
    M = float(P.M_GeV)
    mT = math.sqrt(M**2 + pT**2)
    
    # ARLEO-PEIGNE d+Au KINEMATICS:
    # Broadening depends on the target x (x2).
    # In the d+Au center-of-mass frame, the Au nucleus moves with negative rapidity.
    # The effective x probed in the target nucleus is proportional to exp(+y)
    # relative to the deuteron direction (y>0).
    xA_val = min(QF.xA0_from_L(qp.LA_fm), (mT/qp.roots_GeV)*math.exp(y))
    
    with torch.no_grad():
        xA = torch.tensor([xA_val], dtype=torch.float64, device=device)
        dpta = QF._dpt_from_xL_t(qp, xA, qp.LA_fm, hard=qp.use_hard_cronin)[0]
        
        if torch.abs(dpta) < 1e-8:
            return 1.0
        
        phi, wphi, cphi, sphi = QF._phi_nodes_gl_torch(Nphi, device)
        pshift = QF._shift_pT_pA(pT, dpta, cphi, sphi)
        
        # F1 ratio
        p0, m = P.pp.p0, P.pp.m
        p0sq = p0 * p0
        F1_den = (p0sq / (p0sq + pT**2))**m
        if F1_den <= 1e-30:
            return 1.0
        F1_num = torch.tensor([float((p0sq / (p0sq + p**2))**m) for p in pshift.cpu().numpy()],
                               dtype=torch.float64, device=device)
        R1 = F1_num / F1_den
        
        # F2 ratio
        n = P.pp.n
        M = float(P.M_GeV)
        arg_den = 1.0 - (2.0*mT/qp.roots_GeV)*math.cosh(y)
        F2_den = max(arg_den, 1e-30)**n
        if F2_den <= 1e-30:
            return 1.0
        F2_nums = []
        for p in pshift.cpu().numpy():
            mT_shift = math.sqrt(M**2 + float(p)**2)
            arg = 1.0 - (2.0*mT_shift/qp.roots_GeV)*math.cosh(y)
            F2_nums.append(max(arg, 1e-30)**n)
        F2_num = torch.tensor(F2_nums, dtype=torch.float64, device=device)
        R2 = F2_num / F2_den
        
        R_phi = R1 * R2
        R_broad = torch.sum(R_phi * wphi)
        return float(R_broad.item())

# ------------------------------------------------------------------------------
# TWO-SIDED BROADENING for d+Au (Effective Single Medium Approach)
# ------------------------------------------------------------------------------
def R_broad_AB(P, qp, y, pT, Nphi=64):
    """
    Computes broadening for d+Au.
    
    PHYSICS UPDATE:
    Per Arleo-Peigné (arXiv:1304.0901), d+Au is treated as a "Single Effective Medium"
    where the path length L_eff (e.g., 10.23 fm for MinBias) accounts for the 
    geometry of the Au nucleus averaged over the deuteron wave function.
    
    The deuteron itself is too dilute to act as a significant second medium 
    contributing to broadening sum. Therefore, we do NOT add a separate 
    term for the deuteron (LB=0). We calculate broadening using the single
    effective scale determines by LA_fm.
    
    The asymmetry (forward vs backward) arises naturally from the kinematic
    factors (Bjorken-x dependence of qhat and the F2 slope), not from 
    summing two different media.
    """
    # For d+Au Single Effective Medium: 
    # Broadening is determined by the single scale L_eff (stored in qp.LA_fm).
    # Effectively, this is the same as single-sided broadening with L = L_eff.
    return R_broad(P, qp, y, pT, Nphi=Nphi)



def R_loss_AB(P, qp, y, pT, Ny=64):
    """
    Computes energy loss for d+Au.
    
    PHYSICS UPDATE:
    Similar to broadening, d+Au energy loss is dominated by the propagation 
    through the effective medium (Au side). We use the Single Effective Medium 
    logic with L = L_eff (LA_fm).
    """
    # Use "dAu" mode if specific dAu logic is needed inside R_loss (e.g. specific x calculation)
    # But standard pA logic with correct L_eff is what Arleo-Peigne uses.
    return R_loss(P, qp, y, pT, Ny=Ny, calc_mode="pA") 


# ------------------------------------------------------------------------------
# ENERGY LOSS (using δy integration with Phat quenching weight)
# ------------------------------------------------------------------------------
def R_loss(P, qp, y, pT, Ny=96, calc_mode="pA"):
    """
    Energy-loss factor R^loss_pA(y, pT) for single-sided (p+A).
    Based on Arleo-Peigné Eq. 645.
    
    calc_mode: "pA" (standard) or "dAu" (uses L_Au for qhat, calculates x accordingly)
    """
    import torch
    
    device = qp.device if hasattr(qp, "device") and qp.device else "cpu"
    M = float(P.M_GeV)
    mT = math.sqrt(M**2 + pT**2)
    y_max_pt = QF.y_max(qp.roots_GeV, mT)
    
    # Safety Check
    if not validate_physics_params(P, qp, y, pT):
        return 1.0

    dym = QF.dymax(+y, y_max_pt)
    
    if dym <= QF.DY_EPS:
        return 1.0
    
    # ARLEO-PEIGNE d+Au KINEMATICS:
    # Consistent with broadening, we use the target-frame x definition (exp(y)).
    xA_val = min(QF.xA0_from_L(qp.LA_fm), (mT/qp.roots_GeV)*math.exp(y))
    xA_val = max(1e-12, min(0.99, xA_val))
    
    # Adaptive Ny based on dy range
    Ny_actual = QF._Ny_from_dymax(dym)
    
    with torch.no_grad():
        xA = torch.tensor([xA_val], dtype=torch.float64, device=device)
        
        # F2 ratio function
        # F2(y, pT) = (1 - 2mT/sqrt(s) cosh(y))^n
        # Needs to handle shift y -> y + dy
        p0, m, n = P.pp.p0, P.pp.m, P.pp.n
        
        arg_den = 1.0 - (2.0*mT/qp.roots_GeV)*math.cosh(y)
        F2_den = max(arg_den, 1e-12)**n # Use floor to avoid div zero
        
        def F2_ratio(yshift):
            if F2_den <= 1e-30:
                return torch.ones_like(yshift)
            
            F2_nums = []
            for ys in yshift.cpu().numpy():
                arg = 1.0 - (2.0*mT/qp.roots_GeV)*math.cosh(float(ys))
                F2_nums.append(max(arg, 1e-30)**n)
            F2_num = torch.tensor(F2_nums, dtype=torch.float64, device=device)
            return F2_num / F2_den
        
        # Exponential mapping: u = ln(δy)
        # z = exp(δy) - 1 => δy = ln(1+z)
        # Using built-in GL nodes from QF (u nodes for z integral or log-z)
        # The p+Pb code uses log-z integration for 'exp' mapping.
        
        zmax = math.expm1(dym)
        if zmax <= 1e-12:
            return 1.0
            
        umin, umax = -30.0, math.log(max(zmax, 1e-300))
        u, wu = QF._gl_nodes_torch(umin, umax, Ny_actual, device)
        
        z = torch.exp(u).clamp_min(1e-12) # z integration variable
        
        # Calculate Phat
        ph = QF.PhatA_t(z, mT, xA.expand_as(z), qp, pT=pT)
        
        if (ph <= 0).all():
            return 1.0
        
        # dy from z (magnitude of rapidity shift in relativistic limit)
        dy_mag = torch.log1p(z)
        
        # Energy Loss Shift Requirement:
        # The shift must always increase the rapidity in the direction of the probe's
        # parent parton. For d+Au, y -> y + delta_y consistently produces
        # suppression at forward rapidity and enhancement at backward rapidity.
        yshift = y + dy_mag
        
        ratio = F2_ratio(yshift)
        
        # Numerical Integration Jacobian:
        # We integrate over u = ln z.
        # The target integral is over delta_y.
        # d(delta_y) = dz / (1+z) = (z du) / (1+z).
        # jac_z = z (from u->z step).
        # inv1pz = 1/(1+z) (from z->delta_y step).
        # This is consistent with Arleo-Peigne math where "no explicit Jacobian"
        # refers to the energy variable, but we must handle the delta_y variable change.
        jac_z = torch.exp(u)
        inv1pz = 1.0 / (1.0 + z)

        
        val = torch.sum(wu * jac_z * ph * inv1pz * ratio)
        Zc = torch.sum(wu * jac_z * ph)
        Zc = torch.clamp(Zc, 0.0, 1.0)
        
        if float(Zc.item()) < 1e-12:
            return 1.0
        
        p0_contrib = torch.clamp(1.0 - Zc, 0.0, 1.0)
        R_loss_res = p0_contrib + val
        return float(R_loss_res.item())

# ------------------------------------------------------------------------------
# Helper Wrappers
# ------------------------------------------------------------------------------
def curves_vs_pT(P, roots_GeV, qp_base, Leff_dict, pT_grid, y_fix, mode="pA"):
    out = {}
    for tag, L in Leff_dict.items():
        LA, LB = (float(L), 0.0)
        if mode == "AB":
            if tag in DAU_LEFF_PAIRS:
                LA, LB = DAU_LEFF_PAIRS[tag]
            else:
                # Default to Single-Sided (L_d ~ 0) for d+Au to match Arleo-Peigne reference
                # Two-sided (0.62) overestimates Cronin peak significantly
                LA, LB = (float(L), 0.0)
        
        qp_tag = replace(qp_base, LA_fm=LA, LB_fm=LB)
        Rl, Rb, Rt = [], [], []
        
        for pT in pT_grid:
            rl_v = R_loss_AB(P, qp_tag, y_fix, pT) if mode=="AB" else R_loss(P, qp_tag, y_fix, pT)
            rb_v = R_broad_AB(P, qp_tag, y_fix, pT) if mode=="AB" else R_broad(P, qp_tag, y_fix, pT)
            Rl.append(rl_v)
            Rb.append(rb_v)
            Rt.append(rl_v * rb_v)
        
        out[tag] = (np.array(Rl), np.array(Rb), np.array(Rt))
    return out

def curves_vs_pT_binned_rap(P, roots_GeV, qp_base, Leff_dict, pT_grid, y_range, Ny_bin=12, mode="pA"):
    out = {}
    y_nodes = np.linspace(y_range[0], y_range[1], Ny_bin)
    
    for tag, L in Leff_dict.items():
        LA, LB = (float(L), 0.0)
        if mode == "AB":
            if tag in DAU_LEFF_PAIRS:
                LA, LB = DAU_LEFF_PAIRS[tag]
            else:
                # Default to Single-Sided (L_d ~ 0) for d+Au to match Arleo-Peigne reference
                # Two-sided (0.62) overestimates Cronin peak significantly
                LA, LB = (float(L), 0.0)
        
        qp_tag = replace(qp_base, LA_fm=LA, LB_fm=LB)
        Rl_avg, Rb_avg, Rt_avg = [], [], []
        
        for pT in pT_grid:
            # Weight by F2
            pT_t = torch.tensor([pT], device=qp_base.device, dtype=torch.float64).expand(len(y_nodes))
            y_t = torch.tensor(y_nodes, device=qp_base.device, dtype=torch.float64)
            
            # F2 for weighting
            def F2_weight(y_val, pt_val):
                n = P.pp.n
                M = P.M_GeV
                mT_val = math.sqrt(M*M + pt_val*pt_val)
                arg = 1.0 - (2.0*mT_val/roots_GeV)*math.cosh(y_val)
                return max(arg, 1e-30)**n
            
            weights = np.array([F2_weight(float(yy), pT) for yy in y_nodes])
            weights /= (np.sum(weights) if np.sum(weights) > 0 else 1.0)
            
            # Compute at each y node
            rl_pts = []
            rb_pts = []
            for y_v in y_nodes:
                if mode == "AB":
                    rl_v = R_loss_AB(P, qp_tag, y_v, pT)
                    rb_v = R_broad_AB(P, qp_tag, y_v, pT)
                else:
                    rl_v = R_loss(P, qp_tag, y_v, pT)
                    rb_v = R_broad(P, qp_tag, y_v, pT)
                rl_pts.append(rl_v)
                rb_pts.append(rb_v)
            
            Rl_avg.append(np.sum(weights * np.array(rl_pts)))
            Rb_avg.append(np.sum(weights * np.array(rb_pts)))
            Rt_avg.append(np.sum(weights * (np.array(rl_pts) * np.array(rb_pts))))
        
        out[tag] = (np.array(Rl_avg), np.array(Rb_avg), np.array(Rt_avg))
    return out

def curves_vs_y(P, roots_GeV, qp_base, Leff_dict, y_grid, pT, mode="pA"):
    out = {}
    for tag, L in Leff_dict.items():
        LA, LB = (float(L), 0.0)
        if mode == "AB":
            if tag in DAU_LEFF_PAIRS:
                LA, LB = DAU_LEFF_PAIRS[tag]
            else:
                LA, LB = (float(L), float(L) * 0.62)
        
        qp_tag = replace(qp_base, LA_fm=LA, LB_fm=LB)
        Rl, Rb, Rt = [], [], []
        
        for y_v in y_grid:
            if mode == "AB":
                rl_v = R_loss_AB(P, qp_tag, y_v, pT)
                rb_v = R_broad_AB(P, qp_tag, y_v, pT)
            else:
                rl_v = R_loss(P, qp_tag, y_v, pT)
                rb_v = R_broad(P, qp_tag, y_v, pT)
            Rl.append(rl_v)
            Rb.append(rb_v)
            Rt.append(rl_v * rb_v)
        
        out[tag] = (np.array(Rl), np.array(Rb), np.array(Rt))
    return out
