"""
eloss_cronin_centrality_test.py

Reconstructed module for Coherent Energy Loss + pT Broadening (Cronin).
Consolidates logic from 'Golden Standard' notebooks:
- 05_b_eloss_cronin_pA_LHC_final.ipynb (LHC)
- 06_b_eloss_cronin_dA_RHIC_final.ipynb (RHIC)

Uses `quenching_fast.py` for the core physics kernels.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Optional, Literal, Union

# Local imports
try:
    from particle import Particle, PPSpectrumParams
    from glauber import OpticalGlauber, SystemSpec
    import quenching_fast as QF
except ImportError:
    # Fallback for when running in different contexts or if paths allow it
    import sys
    sys.path.append(".")
    from particle import Particle, PPSpectrumParams
    from glauber import OpticalGlauber, SystemSpec
    import quenching_fast as QF

# Global constants from notebooks
F1_FLOOR = 1e-16
F2_FLOOR = 1e-12
Z_FLOOR = 1e-12
XMIN_SAFE = 1e-12
XMAX_SAFE = 0.99

# --- Device Helper ---
def _qp_device(qp) -> torch.device:
    """Infer torch device from QuenchParams."""
    dev_str = getattr(qp, "device", None)
    if dev_str is None:
        dev_str = "cuda" if (QF._HAS_TORCH and torch.cuda.is_available()) else "cpu"
    if dev_str == "cuda" and not torch.cuda.is_available():
        dev_str = "cpu"
    return torch.device(dev_str)

# --- PP Parameterization Functions ---

def particle_with_scaled_p0(P: Particle, scale: float) -> Particle:
    """
    Return a new Particle with pp.p0 -> scale * p0 (m,n unchanged).
    All other attributes (family, state, mass) are copied.
    """
    pp = P.pp
    new_pp = PPSpectrumParams(p0=pp.p0 * scale, m=pp.m, n=pp.n)
    return Particle(
        family=P.family,
        state=P.state,
        mass_override_GeV=P.mass_override_GeV,
        pp_params=new_pp,
    )

def F1_t(P: Particle, pT_t: torch.Tensor, pp_override: Optional[PPSpectrumParams] = None) -> torch.Tensor:
    """
    F1(p_T) = (p0^2 / (p0^2 + p_T^2))^m
    
    If pp_override is provided, use its p0, m. Otherwise use P.pp.
    """
    if pp_override:
        p0, m = pp_override.p0, pp_override.m
    else:
        p0, m = P.pp.p0, P.pp.m
        
    p0_sq = float(p0) * float(p0)
    return (p0_sq / (p0_sq + pT_t * pT_t))**m

def F2_t(P: Particle, y_t: torch.Tensor, pT_t: torch.Tensor, roots_GeV: float, pp_override: Optional[PPSpectrumParams] = None) -> torch.Tensor:
    """
    F2(y,p_T) = [1 - 2 M_T cosh(y) / sqrt(s)]^n, clamped >= 0.
    """
    if pp_override:
        n = pp_override.n
    else:
        n = P.pp.n
        
    M = float(P.M_GeV)
    roots = float(roots_GeV)
    pT_sq = pT_t * pT_t
    mT    = torch.sqrt(pT_sq + M*M)
    
    # 2*mT*cosh(y)/roots
    arg   = 1.0 - (2.0 * mT / roots) * torch.cosh(y_t)
    arg_clamped = torch.clamp(arg, min=1e-30)
    return arg_clamped**n

def F2_t_pt(P: Particle, y_val: float, pT_t: torch.Tensor, roots_GeV: float, device=None, pp_override: Optional[PPSpectrumParams] = None) -> torch.Tensor:
    """Helper to call F2 with fixed yScalar and variable pT Tensor."""
    if device is None:
        device = pT_t.device
    y_t = torch.full_like(pT_t, float(y_val), device=device)
    return F2_t(P, y_t, pT_t, roots_GeV, pp_override)

def _sigma_pp_weight(P: Particle, roots_GeV: float, table_or_none, y: float, pT: float) -> float:
    """
    sigma_pp(y,pT;sqrt(s)) used as weight.
    """
    if QF._HAS_TORCH and table_or_none is not None and hasattr(table_or_none, "device"):
        # Assume it's QF.TorchSigmaPPTable or similar
        dev = table_or_none.device
        with torch.no_grad():
            y_t = torch.tensor([y],  dtype=torch.float64, device=dev)
            p_t = torch.tensor([pT], dtype=torch.float64, device=dev)
            return float(table_or_none(y_t, p_t)[0, 0].item())
    else:
        # Fallback to analytical parametrization if P has it
        if hasattr(P, "d2sigma_pp"):
            return float(P.d2sigma_pp(float(y), float(pT), float(roots_GeV)))
        else:
            # Fallback to F1*F2 approx if d2sigma_pp not available on P directly
            # This is a bit rough but works for weighting purposes
            # d2sigma ~ F1(pT) * F2(y,pT)
            pT_t = torch.tensor([pT], dtype=torch.float64)
            y_t = torch.tensor([y], dtype=torch.float64)
            val = F1_t(P, pT_t) * F2_t(P, y_t, pT_t, roots_GeV)
            return float(val.item())

# --- xA Calculation ---

def xA_scalar(P: Particle, roots_GeV: float, qp, y: float, pT: float) -> float:
    """
    x_A = min( x0(L_A), x_target )
    
    Target kinematics:
      x_target = (mT/s) * exp(-y)
    """
    M  = float(P.M_GeV)
    mT = math.sqrt(M*M + float(pT)**2)
    # Both notebooks use exp(-y)
    x_target = (mT / float(roots_GeV)) * math.exp(-float(y))
    
    x0 = QF.xA0_from_L(qp.LA_fm)
    xA = min(x0, x_target)
    xA = max(XMIN_SAFE, min(XMAX_SAFE, xA))
    return float(xA)

# --- Core Wrappers ---

def R_pA_eloss(
    P: Particle,
    roots_GeV: float,
    qp,
    y: float,
    pT: float,
    Ny: int | None = None,
    pp_override: Optional[PPSpectrumParams] = None
) -> float:
    r"""
    Coherent energy-loss factor (Arleo-Peigne). Wraps quenching_fast.
    """
    if not QF._HAS_TORCH:
        raise RuntimeError("R_pA_eloss requires torch.")

    device = _qp_device(qp)
    M = float(P.M_GeV)
    pT0 = float(pT)
    mT = math.sqrt(M*M + pT0*pT0)

    # Upper limit for dy integration
    y_max_pt = QF.y_max(roots_GeV, mT)
    # Note: 05_b uses +y in dymax call
    dym = QF.dymax(y, y_max_pt)
    
    if dym <= QF.DY_EPS:
        return 1.0

    if Ny is None:
        Ny = QF._Ny_from_dymax(dym)

    zmax = math.expm1(dym)
    if zmax <= QF.Z_FLOOR:
        return 1.0

    xA_val = xA_scalar(P, roots_GeV, qp, y, pT0)

    with torch.no_grad():
        xA = torch.tensor([xA_val], dtype=torch.float64, device=device)
        y0_t = torch.tensor([y], dtype=torch.float64, device=device)
        pT0_t = torch.tensor([pT0], dtype=torch.float64, device=device)

        F2_den_t = F2_t(P, y0_t, pT0_t, roots_GeV, pp_override)[0]
        if F2_den_t <= F2_FLOOR:
            return 1.0

        mapping = getattr(qp, "mapping", "exp")

        if mapping == "exp":
            umin = -30.0
            umax = math.log(max(zmax, 1e-300))
            u, wu = QF._gl_nodes_torch(umin, umax, Ny, device)
            
            z = torch.exp(u).clamp_min(QF.Z_FLOOR)
            # PhatA_t from quenching_fast
            ph = QF.PhatA_t(z, mT, xA.expand_as(z), qp, pT=pT0)
            
            if (ph <= 0).all():
                return 1.0

            # dy = ln(1+z)
            dy = torch.log1p(z)
            yshift = y + dy
            
            F2_num = F2_t(P, yshift, pT0_t.expand_as(yshift), roots_GeV, pp_override)
            ratio = F2_num / F2_den_t
            ratio = torch.where(torch.isfinite(ratio) & (ratio >= 0.0), ratio, torch.zeros_like(ratio))
            
            jac_z = torch.exp(u)
            inv1pz = 1.0 / (1.0 + z)
            
            # Integral: \int dz P(z) * 1/(1+z) * Ratio
            # jac_z handles du -> dz
            val = torch.sum(wu * jac_z * ph * inv1pz * ratio)
            
            # Normalization 1-p0
            Zc = torch.sum(wu * jac_z * ph)
            
        else:
            # Linear mapping fallback
            z, wz = QF._gl_nodes_torch(0.0, float(zmax), Ny, device)
            z = z.clamp_min(QF.Z_FLOOR)
            
            ph = QF.PhatA_t(z, mT, xA.expand_as(z), qp, pT=pT0)
            if (ph <= 0).all():
                return 1.0
                
            dy = torch.log1p(z)
            yshift = y + dy
            
            F2_num = F2_t(P, yshift, pT0_t.expand_as(yshift), roots_GeV, pp_override)
            ratio = F2_num / F2_den_t
            
            inv1pz = 1.0 / (1.0 + z)
            val = torch.sum(wz * ph * inv1pz * ratio)
            Zc = torch.sum(wz * ph)

        Zc = torch.clamp(Zc, min=0.0, max=1.0)
        p0 = torch.clamp(1.0 - Zc, 0.0, 1.0)
        
        R_loss = p0 + val
        return float(R_loss.item())

def R_pA_broad(
    P: Particle,
    roots_GeV: float,
    qp,
    y: float,
    pT: float,
    Nphi: int = 128, 
    Nk: int = 32,
    pp_override: Optional[PPSpectrumParams] = None
) -> float:
    """
    Cronin/Broadening factor.
    """
    if not QF._HAS_TORCH:
        raise RuntimeError("R_pA_broad requires torch.")
    device = _qp_device(qp)
    
    broad_model = getattr(qp, "broadening_model", getattr(qp, "broad_model", "ring")).lower()
    use_ring = broad_model in ("ring", "hard", "fixed", "shift")
    
    with torch.no_grad():
        xA_val = xA_scalar(P, roots_GeV, qp, y, pT)
        xA = torch.tensor([xA_val], dtype=torch.float64, device=device)
        
        # Broadening scale
        dpt = QF._dpt_from_xL_t(qp, xA, qp.LA_fm, hard=use_ring)[0]
        dpt = torch.abs(dpt)
        
        if dpt < 1e-10:
            return 1.0
            
        pT0 = float(pT)
        pT0_t = torch.tensor([pT0], dtype=torch.float64, device=device)
        
        # Denominators
        F1_den = F1_t(P, pT0_t, pp_override)[0]
        if F1_den <= F1_FLOOR:
            return 1.0
            
        F2_den = F2_t_pt(P, y_val=y, pT_t=pT0_t, roots_GeV=roots_GeV, device=device, pp_override=pp_override)[0]
        if F2_den <= F2_FLOOR:
            return 1.0
            
        # Phi integration nodes
        phi, wphi, cphi, sphi = QF._phi_nodes_gl_torch(Nphi, device)
        
        if use_ring:
            # Fixed magnitude shift, average over phi
            dpt2d = dpt[None, None]
            c2d = cphi[None, :]
            
            # p_shifted^2 = pT^2 + dpt^2 - 2 pT dpt cos(phi)
            pshift = torch.sqrt(torch.clamp(
                pT0*pT0 + dpt2d*dpt2d - 2.0*pT0*dpt2d*c2d, min=0.0
            ))
            
            # F1 numerator
            F1_num = F1_t(P, pshift, pp_override)
            R1 = F1_num / F1_den
            
            # F2 numerator (at same y)
            F2_num = F2_t_pt(P, y_val=y, pT_t=pshift, roots_GeV=roots_GeV, device=device, pp_override=pp_override)
            R2 = F2_num / F2_den
            
            R = R1 * R2
            R = torch.where(torch.isfinite(R) & (R >= 0.0), R, torch.zeros_like(R))
            
            # Average
            R_broad = torch.sum(R * wphi[None, :], dim=1)[0]
            return float(R_broad.item())
            
        else:
            # Gaussian broadening (Nk nodes for k, Nphi for angle)
            # Delta = <k^2>
            Delta = dpt * dpt 
            
            u, wu = QF._gl_nodes_torch(0.0, 1.0, Nk, device)
            u = torch.clamp(u, min=1e-300, max=1.0)
            t = -torch.log(u)
            k = torch.sqrt(Delta * t) # shape (Nk,)
            
            k2d = k[:, None]    # (Nk, 1)
            c2d = cphi[None, :] # (1, Nphi)
            
            # pshift grid (Nk, Nphi)
            pshift = torch.sqrt(torch.clamp(
                pT0*pT0 + k2d*k2d - 2.0*pT0*k2d*c2d, min=0.0
            ))
            
            F1_num = F1_t(P, pshift, pp_override)
            R1 = F1_num / F1_den
            
            F2_num = F2_t_pt(P, y_val=y, pT_t=pshift, roots_GeV=roots_GeV, device=device, pp_override=pp_override)
            R2 = F2_num / F2_den
            
            R = R1 * R2
            R = torch.where(torch.isfinite(R) & (R >= 0.0), R, torch.zeros_like(R))
            
            # Average over phi first
            R_phiavg = torch.sum(R * wphi[None, :], dim=1) # (Nk,)
            
            # Integrate over k (t)
            R_broad = torch.sum(wu * R_phiavg)
            return float(R_broad.item())

def R_pA_factored(
    P: Particle,
    roots_GeV: float,
    qp,
    y: float,
    pT: float,
    Ny_eloss: int = 256,
    Nphi_broad: int = 256,
    pp_override: Optional[PPSpectrumParams] = None
) -> float:
    """
    R_pA ~ R_loss * R_broad
    """
    rl = R_pA_eloss(P, roots_GeV, qp, y, pT, Ny=Ny_eloss, pp_override=pp_override)
    rb = R_pA_broad(P, roots_GeV, qp, y, pT, Nphi=Nphi_broad, pp_override=pp_override)
    return rl * rb

# --- Bin Integration Logic ---

def R_binned_2D(
    R_func,                     # R_func(y,pT) -> float
    P: Particle, 
    roots_GeV: float,
    y_range: Tuple[float, float], 
    pt_range: Tuple[float, float],
    Ny_bin: int = 12, 
    Npt_bin: int = 24,
    weight_kind: str = "pp",    # "pp" or "flat"
    table_for_pp=None,
    weight_ref_y: float | str = "local",
):
    """
    Generic bin average of R(y,pT) over y_range x pt_range.
    """
    yl, yr = float(y_range[0]), float(y_range[1])
    pl, pr = float(pt_range[0]), float(pt_range[1])

    y_nodes, y_w = QF._gl_nodes_np(yl, yr, Ny_bin)
    p_nodes, p_w = QF._gl_nodes_np(pl, pr, Npt_bin)

    if isinstance(weight_ref_y, str) and weight_ref_y.lower() == "local":
        y_ref = None
    else:
        y_ref = float(weight_ref_y)

    acc_num, acc_den = 0.0, 0.0

    for yi, wy in zip(y_nodes, y_w):
        y_for_w = float(yi) if y_ref is None else y_ref
        for pj, wp in zip(p_nodes, p_w):
            pj_f = float(pj)
            R    = float(R_func(float(yi), pj_f))

            if weight_kind == "pp":
                wgt = _sigma_pp_weight(P, roots_GeV, table_for_pp, y_for_w, pj_f)
                wgt *= max(pj_f, 1e-8)   # sigma_pp * pT (jacobian)
            else:
                wgt = 1.0

            acc_num += wy * wp * R * wgt
            acc_den += wy * wp * wgt

    if acc_den <= 0:
        return acc_num
    return float(acc_num / acc_den)

def rpa_binned_vs_y(
    P: Particle, roots_GeV: float, qp_base,
    glauber, cent_bins,
    y_edges, pt_range,
    components=("eloss_broad",),
    Ny_bin: int = 12, Npt_bin: int = 24,
    table_for_pp=None,
    weight_kind: str = "pp",
    weight_ref_y: float | str = "local",
):
    """
    Returns y_centers, R_comp, labels.
    """
    y_edges = np.asarray(y_edges, float)
    assert y_edges.ndim == 1 and y_edges.size >= 2
    Ny = y_edges.size - 1
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    comps  = list(components)
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in cent_bins]

    L_by = glauber.leff_bins_pA(cent_bins, method="optical")
    Leff_dict = {lab: float(L_by[lab]) for lab in labels}

    # optical weights for MB
    w_bins = np.array(
        [QF._optical_bin_weight_pA(glauber, a, b) for (a,b) in cent_bins],
        float
    )
    w_bins = w_bins / max(w_bins.sum(), 1e-30)
    w_dict = {lab: w_bins[i] for i, lab in enumerate(labels)}

    R_comp = {comp: {lab: np.zeros(Ny, float) for lab in labels}
              for comp in comps}

    for i in range(Ny):
        y_r = (float(y_edges[i]), float(y_edges[i+1]))

        for lab in labels:
            L   = Leff_dict[lab]
            qpL = replace(qp_base, LA_fm=float(L), LB_fm=float(L))

            def R_loss(y, pT, qpL=qpL):
                return R_pA_eloss(P, roots_GeV, qpL, y, pT, Ny=None)

            def R_broad(y, pT, qpL=qpL):
                return R_pA_broad(P, roots_GeV, qpL, y, pT, Nphi=64)

            for comp in comps:
                if comp == "loss":
                    R_func = R_loss
                elif comp == "broad":
                    R_func = R_broad
                elif comp == "eloss_broad":
                    def R_func(y, pT, qpL=qpL):
                        return R_loss(y, pT, qpL=qpL) * R_broad(y, pT, qpL=qpL)
                else:
                    raise ValueError(f"Unknown component: {comp}")

                R_bar = R_binned_2D(
                    R_func, P, roots_GeV,
                    y_r, pt_range,
                    Ny_bin=Ny_bin, Npt_bin=Npt_bin,
                    weight_kind=weight_kind,
                    table_for_pp=table_for_pp,
                    weight_ref_y=weight_ref_y,
                )
                R_comp[comp][lab][i] = R_bar

    # min-bias
    for comp in comps:
        R_mat = np.vstack([R_comp[comp][lab] for lab in labels])  # (Ncent, Ny)
        w_arr = np.array([w_dict[lab] for lab in labels])
        R_MB  = np.average(R_mat, axis=0, weights=w_arr)
        R_comp[comp]["MB"] = R_MB

    return y_centers, R_comp, labels

def rpa_binned_vs_pT(
    P: Particle, roots_GeV: float, qp_base,
    glauber, cent_bins,
    pT_edges, y_range,
    components=("eloss_broad",),
    Ny_bin: int = 12, Npt_bin: int = 24,
    table_for_pp=None,
    weight_kind: str = "pp",
    weight_ref_y: float | str = "local",
):
    """
    Same as rpa_binned_vs_y but swapping y <-> pT roles.
    """
    pT_edges = np.asarray(pT_edges, float)
    assert pT_edges.ndim == 1 and pT_edges.size >= 2
    Np = pT_edges.size - 1
    pT_centers = 0.5 * (pT_edges[:-1] + pT_edges[1:])

    comps  = list(components)
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in cent_bins]

    L_by = glauber.leff_bins_pA(cent_bins, method="optical")
    Leff_dict = {lab: float(L_by[lab]) for lab in labels}

    w_bins = np.array(
        [QF._optical_bin_weight_pA(glauber, a, b) for (a,b) in cent_bins],
        float
    )
    w_bins = w_bins / max(w_bins.sum(), 1e-30)
    w_dict = {lab: w_bins[i] for i, lab in enumerate(labels)}

    R_comp = {comp: {lab: np.zeros(Np, float) for lab in labels}
              for comp in comps}

    for i in range(Np):
        pt_r = (float(pT_edges[i]), float(pT_edges[i+1]))

        for lab in labels:
            L   = Leff_dict[lab]
            qpL = replace(qp_base, LA_fm=float(L), LB_fm=float(L))

            def R_loss(y, pT, qpL=qpL):
                return R_pA_eloss(P, roots_GeV, qpL, y, pT, Ny=None)

            def R_broad(y, pT, qpL=qpL):
                return R_pA_broad(P, roots_GeV, qpL, y, pT, Nphi=64)

            for comp in comps:
                if comp == "loss":
                    R_func = R_loss
                elif comp == "broad":
                    R_func = R_broad
                elif comp == "eloss_broad":
                    def R_func(y, pT, qpL=qpL):
                        return R_loss(y, pT, qpL=qpL) * R_broad(y, pT, qpL=qpL)
                else:
                    raise ValueError(f"Unknown component: {comp}")

                R_bar = R_binned_2D(
                    R_func, P, roots_GeV,
                    y_range, pt_r,
                    Ny_bin=Ny_bin, Npt_bin=Npt_bin,
                    weight_kind=weight_kind,
                    table_for_pp=table_for_pp,
                    weight_ref_y=weight_ref_y,
                )
                R_comp[comp][lab][i] = R_bar

    for comp in comps:
        R_mat = np.vstack([R_comp[comp][lab] for lab in labels])  # (Ncent, Np)
        w_arr = np.array([w_dict[lab] for lab in labels])
        R_MB  = np.average(R_mat, axis=0, weights=w_arr)
        R_comp[comp]["MB"] = R_MB

    return pT_centers, R_comp, labels

# --- Band Calculation Helpers ---

def _two_point_band(R_lo: np.ndarray, R_hi: np.ndarray):
    """
    Given arrays R(q_min) and R(q_max), return
      Rc, Rlow, Rhigh  with symmetric error.
    """
    Rc = 0.5 * (R_lo + R_hi)
    dR = 0.5 * np.abs(R_hi - R_lo)
    return Rc, Rc - dR, Rc + dR

def combine_factorized_bands_1d(RL_c, RL_lo, RL_hi,
                                RB_c, RB_lo, RB_hi):
    """
    Combine loss & broad bands into total, assuming factorisation.
    """
    RT_c, RT_lo, RT_hi = {}, {}, {}
    for lab in RL_c.keys():
        Lc  = np.asarray(RL_c[lab])
        Llo = np.asarray(RL_lo[lab])
        Lhi = np.asarray(RL_hi[lab])

        Bc  = np.asarray(RB_c[lab])
        Blo = np.asarray(RB_lo[lab])
        Bhi = np.asarray(RB_hi[lab])

        dL = 0.5*np.abs(Lhi - Llo)
        dB = 0.5*np.abs(Bhi - Blo)

        Lc_safe = np.where(np.abs(Lc) > 1e-12, Lc, 1.0)
        Bc_safe = np.where(np.abs(Bc) > 1e-12, Bc, 1.0)

        Rc   = Lc * Bc
        rel2 = (dL/Lc_safe)**2 + (dB/Bc_safe)**2
        dR   = Rc * np.sqrt(rel2)

        RT_c[lab], RT_lo[lab], RT_hi[lab] = Rc, Rc - dR, Rc + dR

    return RT_c, RT_lo, RT_hi

# --- Band vs Y ---

def rpa_band_vs_y_eloss(
    P, roots_GeV: float,
    qp_base,
    glauber, cent_bins,
    y_edges, pt_range,
    q0_pair=(0.05, 0.09),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: str = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
):
    q0_lo, q0_hi = q0_pair
    qp_lo = replace(qp_base, qhat0=float(q0_lo))
    qp_hi = replace(qp_base, qhat0=float(q0_hi))

    y_cent_lo, R_lo, labels = rpa_binned_vs_y(
        P, roots_GeV, qp_lo,
        glauber, cent_bins,
        y_edges, pt_range,
        components=("loss",),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
    )
    y_cent_hi, R_hi, _ = rpa_binned_vs_y(
        P, roots_GeV, qp_hi,
        glauber, cent_bins,
        y_edges, pt_range,
        components=("loss",),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
    )
    y_cent = y_cent_lo

    RL_c, RL_lo, RL_hi = {}, {}, {}
    for lab in R_lo["loss"].keys():
        Rc, Rl, Rh = _two_point_band(R_lo["loss"][lab], R_hi["loss"][lab])
        RL_c[lab], RL_lo[lab], RL_hi[lab] = Rc, Rl, Rh

    return y_cent, RL_c, RL_lo, RL_hi, labels

def rpa_band_vs_y_broad(
    P, roots_GeV: float,
    qp_base,
    glauber, cent_bins,
    y_edges, pt_range,
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: str = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
):
    P_lo = particle_with_scaled_p0(P, p0_scale_pair[0])
    P_hi = particle_with_scaled_p0(P, p0_scale_pair[1])

    y_cent_lo, R_lo, labels = rpa_binned_vs_y(
        P_lo, roots_GeV, qp_base,
        glauber, cent_bins,
        y_edges, pt_range,
        components=("broad",),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
    )
    y_cent_hi, R_hi, _ = rpa_binned_vs_y(
        P_hi, roots_GeV, qp_base,
        glauber, cent_bins,
        y_edges, pt_range,
        components=("broad",),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
    )
    y_cent = y_cent_lo

    RB_c, RB_lo, RB_hi = {}, {}, {}
    for lab in R_lo["broad"].keys():
        Rc, Rl, Rh = _two_point_band(R_lo["broad"][lab], R_hi["broad"][lab])
        RB_c[lab], RB_lo[lab], RB_hi[lab] = Rc, Rl, Rh

    return y_cent, RB_c, RB_lo, RB_hi, labels

def rpa_band_vs_y(
    P, roots_GeV: float,
    qp_base,
    glauber, cent_bins,
    y_edges, pt_range,
    components=("loss","broad","eloss_broad"),
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: str = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
):
    """
    Full RpA band vs y.
    """
    # ---- eLoss band (q0) ----
    y_cent, RL_c, RL_lo, RL_hi, labels = rpa_band_vs_y_eloss(
        P, roots_GeV, qp_base,
        glauber, cent_bins,
        y_edges, pt_range,
        q0_pair=q0_pair,
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=table_for_pp,
    )

    # ---- broad band (p0) ----
    y_cent2, RB_c, RB_lo, RB_hi, _ = rpa_band_vs_y_broad(
        P, roots_GeV, qp_base,
        glauber, cent_bins,
        y_edges, pt_range,
        p0_scale_pair=p0_scale_pair,
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=table_for_pp,
    )

    # ---- total band ----
    RT_c, RT_lo, RT_hi = combine_factorized_bands_1d(
        RL_c, RL_lo, RL_hi,
        RB_c, RB_lo, RB_hi,
    )

    bands = {}
    if "loss" in components:
        bands["loss"] = (RL_c, RL_lo, RL_hi)
    if "broad" in components:
        bands["broad"] = (RB_c, RB_lo, RB_hi)
    if "eloss_broad" in components:
        bands["eloss_broad"] = (RT_c, RT_lo, RT_hi)

    return y_cent, bands, labels

# --- Band vs pT ---

def rpa_band_vs_pT_eloss(
    P, roots_GeV: float,
    qp_base,
    glauber, cent_bins,
    pT_edges, y_range,
    q0_pair=(0.05, 0.09),
    component="loss",
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: str = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
):
    q0_lo, q0_hi = q0_pair
    qp_lo = replace(qp_base, qhat0=float(q0_lo))
    qp_hi = replace(qp_base, qhat0=float(q0_hi))

    pT_cent_lo, R_lo, labels = rpa_binned_vs_pT(
        P, roots_GeV, qp_lo,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=table_for_pp,
    )
    pT_cent_hi, R_hi, _ = rpa_binned_vs_pT(
        P, roots_GeV, qp_hi,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=table_for_pp,
    )
    pT_cent = pT_cent_lo

    RL_c, RL_lo, RL_hi = {}, {}, {}
    for lab in R_lo[component].keys():
        Rc, Rl, Rh = _two_point_band(R_lo[component][lab], R_hi[component][lab])
        RL_c[lab], RL_lo[lab], RL_hi[lab] = Rc, Rl, Rh

    return pT_cent, RL_c, RL_lo, RL_hi, labels

def rpa_band_vs_pT_broad(
    P, roots_GeV: float,
    qp_base,
    glauber, cent_bins,
    pT_edges, y_range,
    p0_scale_pair=(0.9, 1.1),
    component="broad",
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: str = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
):
    P_lo = particle_with_scaled_p0(P, p0_scale_pair[0])
    P_hi = particle_with_scaled_p0(P, p0_scale_pair[1])

    pT_cent_lo, R_lo, labels = rpa_binned_vs_pT(
        P_lo, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=table_for_pp,
    )
    pT_cent_hi, R_hi, _ = rpa_binned_vs_pT(
        P_hi, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=table_for_pp,
    )
    pT_cent = pT_cent_lo

    RB_c, RB_lo, RB_hi = {}, {}, {}
    for lab in R_lo[component].keys():
        Rc, Rl, Rh = _two_point_band(R_lo[component][lab], R_hi[component][lab])
        RB_c[lab], RB_lo[lab], RB_hi[lab] = Rc, Rl, Rh

    return pT_cent, RB_c, RB_lo, RB_hi, labels

def rpa_band_vs_pT(
    P, roots_GeV: float,
    qp_base,
    glauber, cent_bins,
    pT_edges, y_range,
    components=("loss","broad","eloss_broad"),
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: str = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
):
    # loss band
    pT_cent, RL_c, RL_lo, RL_hi, labels = rpa_band_vs_pT_eloss(
        P, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        q0_pair=q0_pair,
        component="loss",
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=table_for_pp,
    )

    # broad band
    pT_cent2, RB_c, RB_lo, RB_hi, _ = rpa_band_vs_pT_broad(
        P, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        p0_scale_pair=p0_scale_pair,
        component="broad",
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=table_for_pp,
    )

    # total
    RT_c, RT_lo, RT_hi = combine_factorized_bands_1d(
        RL_c, RL_lo, RL_hi,
        RB_c, RB_lo, RB_hi,
    )

    bands = {}
    if "loss" in components:
        bands["loss"] = (RL_c, RL_lo, RL_hi)
    if "broad" in components:
        bands["broad"] = (RB_c, RB_lo, RB_hi)
    if "eloss_broad" in components:
        bands["eloss_broad"] = (RT_c, RT_lo, RT_hi)

    return pT_cent, bands, labels

# --- Band vs Centrality ---

def rpa_vs_centrality(
    P, roots_GeV: float, qp, glauber, cent_bins,
    y_range, pt_range,
    component="eloss_broad",
    Ny_bin=16, Npt_bin=32,
    weight_kind="pp", weight_ref_y="local",
    table_for_pp=None,
):
    """
    Compute single R_{pA} value for each centrality bin (by averaging over y, pT, b).
    Returns: labels, R_values, R_MB
    """
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in cent_bins]
    
    # 1. Optical weights for MB
    w_bins = np.array(
        [QF._optical_bin_weight_pA(glauber, a, b) for (a,b) in cent_bins],
        float
    )
    w_bins = w_bins / max(w_bins.sum(), 1e-30)
    
    # 2. Compute R for each bin
    L_by = glauber.leff_bins_pA(cent_bins, method="optical")
    R_vals = []
    
    for lab in labels:
        L = float(L_by[lab])
        qpL = replace(qp, LA_fm=L, LB_fm=L)
        
        def R_loss(y, pT, qpL=qpL):
            return R_pA_eloss(P, roots_GeV, qpL, y, pT, Ny=None)
        def R_broad(y, pT, qpL=qpL):
            return R_pA_broad(P, roots_GeV, qpL, y, pT, Nphi=64)
            
        if component == "loss":
            func = R_loss
        elif component == "broad":
            func = R_broad
        else: # eloss_broad
            def func(y, pT, qpL=qpL):
                return R_loss(y, pT, qpL) * R_broad(y, pT, qpL)
                
        val = R_binned_2D(
            func, P, roots_GeV, 
            y_range, pt_range,
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            weight_kind=weight_kind,
            weight_ref_y=weight_ref_y,
            table_for_pp=table_for_pp,
        )
        R_vals.append(val)
        
    R_vals = np.array(R_vals)
    R_MB = float(np.average(R_vals, weights=w_bins))
    
    return labels, R_vals, R_MB

def rpa_band_vs_centrality(
    P, roots_GeV: float, qp_base,
    glauber, cent_bins,
    y_range, pt_range,
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 16, Npt_bin: int = 32,
    weight_kind: str = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
):
    """
    Error bands vs centrality:
      loss band (q0), broad band (p0), total (quadrature).
    """
    # Drop "MB" bin if present (0-100)
    core_bins = [b for b in cent_bins if not (b[0] == 0 and b[1] == 100)]
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in core_bins]

    # --- loss band ---
    q0_lo, q0_hi = q0_pair
    RL_lo, RL_hi, RL_c = {}, {}, {}

    for q0, store in [(q0_lo, RL_lo), (q0_hi, RL_hi)]:
        qp_q = replace(qp_base, qhat0=float(q0))
        _, Rvals, _ = rpa_vs_centrality(
            P, roots_GeV, qp_q, glauber, core_bins,
            y_range, pt_range,
            component="loss",
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            weight_kind=weight_kind,
            weight_ref_y=weight_ref_y,
            table_for_pp=table_for_pp,
        )
        for lab, val in zip(labels, Rvals):
            store[lab] = val

    for lab in labels:
        Rc, Rl, Rh = _two_point_band(RL_lo[lab], RL_hi[lab])
        RL_c[lab], RL_lo[lab], RL_hi[lab] = Rc, Rl, Rh

    # --- broad band ---
    RB_lo, RB_hi, RB_c = {}, {}, {}
    for p0s, store in [(p0_scale_pair[0], RB_lo), (p0_scale_pair[1], RB_hi)]:
        P_sc = particle_with_scaled_p0(P, p0s)
        _, Rvals, _ = rpa_vs_centrality(
            P_sc, roots_GeV, qp_base, glauber, core_bins,
            y_range, pt_range,
            component="broad",
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            weight_kind=weight_kind,
            weight_ref_y=weight_ref_y,
            table_for_pp=table_for_pp,
        )
        for lab, val in zip(labels, Rvals):
            store[lab] = val

    for lab in labels:
        Rc, Rl, Rh = _two_point_band(RB_lo[lab], RB_hi[lab])
        RB_c[lab], RB_lo[lab], RB_hi[lab] = Rc, Rl, Rh

    # --- combine ---
    RT_c, RT_lo, RT_hi = combine_factorized_bands_1d(
        RL_c, RL_lo, RL_hi,
        RB_c, RB_lo, RB_hi
    )

    # MB values
    w_bins = np.array(
        [QF._optical_bin_weight_pA(glauber, a, b) for (a,b) in core_bins],
        float
    )
    w_bins = w_bins / max(w_bins.sum(), 1e-30)

    def _mb_from_dict(Dc, Dlo, Dhi):
        arr_c  = np.array([Dc[lab]  for lab in labels])
        arr_lo = np.array([Dlo[lab] for lab in labels])
        arr_hi = np.array([Dhi[lab] for lab in labels])
        Rc  = float(np.average(arr_c,  weights=w_bins))
        Rlo = float(np.average(arr_lo, weights=w_bins))
        Rhi = float(np.average(arr_hi, weights=w_bins))
        return Rc, Rlo, Rhi

    RMB_loss  = _mb_from_dict(RL_c, RL_lo, RL_hi)
    RMB_broad = _mb_from_dict(RB_c, RB_lo, RB_hi)
    RMB_tot   = _mb_from_dict(RT_c, RT_lo, RT_hi)

    # Return everything
    # Dicts keyed by labels
    results = {
        "labels": labels,
        "loss": (RL_c, RL_lo, RL_hi),
        "broad": (RB_c, RB_lo, RB_hi),
        "total": (RT_c, RT_lo, RT_hi),
        "MB": {
            "loss": RMB_loss,
            "broad": RMB_broad,
            "total": RMB_tot
        }
    }
    return results

# --- Helpers: Centrality & Weights ---

def make_centrality_weight_dict(cent_bins: List[Tuple[float, float]], c0: float = 0.25) -> Dict[str, float]:
    """
    Exponential weights exp(-c/c0) integrated over bins.
    """
    edges_frac = np.array([c[0] for c in cent_bins] + [cent_bins[-1][1]], float) / 100.0
    
    num = np.exp(-edges_frac[:-1] / c0) - np.exp(-edges_frac[1:] / c0)
    denom = 1.0 - np.exp(-1.0 / c0)
    w = num / denom
    
    w /= w.sum() # normalize
    
    tags = [f"{int(a)}-{int(b)}%" for (a,b) in cent_bins]
    return {tag: val for tag, val in zip(tags, w)}

# --- Batch Computation Helpers (Legacy/Raw) ---

def compute_curves(
    P: Particle,
    roots_GeV: float,
    qp_base,
    Leff_dict: Dict[str, float],
    variable: Literal["y", "pT"],
    grid: np.ndarray,
    fixed_val: float,
    Ny_eloss=256,
    Nphi_broad=256,
    pp_override: Optional[PPSpectrumParams] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute R_loss, R_broad, R_tot curves for each centrality bin in Leff_dict.
    RAW curves, no bin averaging.
    """
    results = {}
    
    for tag, Leff in Leff_dict.items():
        # Update L_eff in quench params
        qp = replace(qp_base, LA_fm=float(Leff), LB_fm=float(Leff))
        
        Rl, Rb, Rt = [], [], []
        
        for v in grid:
            if variable == "y":
                y, pT = v, fixed_val
            else:
                y, pT = fixed_val, v
                
            rl = R_pA_eloss(P, roots_GeV, qp, y, pT, Ny=Ny_eloss, pp_override=pp_override)
            rb = R_pA_broad(P, roots_GeV, qp, y, pT, Nphi=Nphi_broad, pp_override=pp_override)
            
            Rl.append(rl)
            Rb.append(rb)
            Rt.append(rl * rb)
            
        results[tag] = (np.array(Rl), np.array(Rb), np.array(Rt))
        
    return results

# --- Setup Helper ---

def get_default_parameters(
    system: Literal["pPb5", "pPb8", "dAu200"] = "pPb5",
    device=None,
) -> Dict:
    """
    Returns standard parameters for LHC/RHIC run.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Alpha_s constant as per notebooks
    def alpha_s_provider(*, mode="constant", alpha0=0.5):
        # Placeholder for simpler alpha_s passed to QP.
        # In actual notebook, it's a callable function of scale.
        # But QF.QuenchParams expects 'alpha_of_mu' to be a callable or float?
        # Notebook passed `alpha_cst` which was result of `alpha_s_provider` call?
        # Actually in notebook: alpha_cst = alpha_s_provider(mode="constant") -> returns a callable.
        class ConstantAlpha:
            def __init__(self, val): self.val = float(val)
            def __call__(self, mu): return self.val
        return ConstantAlpha(alpha0)

    alpha_cst = alpha_s_provider(alpha0=0.5)

    if system == "pPb5":
        roots = 5023.0
        A = 208
        spec = SystemSpec("pA", roots, A=A)
        glauber = OpticalGlauber(spec)
        Lmb = glauber.leff_minbias_pA()
        
        qp = QF.QuenchParams(
            qhat0=0.075,
            lp_fm=1.5,
            LA_fm=Lmb, LB_fm=Lmb,
            lambdaQCD=0.25,
            roots_GeV=roots,
            alpha_of_mu=alpha_cst,
            alpha_scale="mT",
            use_hard_cronin=True,
            mapping="exp",
            device=device
        )
        cent_bins = [(0,20), (20,40), (40,60), (60,100)]
        
    elif system == "pPb8":
        roots = 8160.0 # Standard LHCb value
        A = 208
        spec = SystemSpec("pA", roots, A=A)
        glauber = OpticalGlauber(spec)
        Lmb = glauber.leff_minbias_pA()
        
        qp = QF.QuenchParams(
            qhat0=0.075,
            lp_fm=1.5,
            LA_fm=Lmb, LB_fm=Lmb,
            lambdaQCD=0.25,
            roots_GeV=roots,
            alpha_of_mu=alpha_cst,
            alpha_scale="mT",
            use_hard_cronin=True,
            mapping="exp",
            device=device
        )
        cent_bins = [(0,20), (20,40), (40,60), (60,100)]
        
    elif system == "dAu200":
        roots = 200.0
        A = 197
        spec = SystemSpec("dA", roots, A=A)
        glauber = OpticalGlauber(spec)
        # For dAu, use 'binomial' leff by default or 'optical'?
        # Notebook 06 used 'binomial' for dAu explicitly?
        # Usually for RHIC small systems, binomial is standard for dAu
        Lmb = glauber.leff_minbias_dA(method="binomial")
        
        qp = QF.QuenchParams(
            qhat0=0.075,
            lp_fm=1.5,
            LA_fm=Lmb, LB_fm=Lmb,
            lambdaQCD=0.25,
            roots_GeV=roots,
            alpha_of_mu=alpha_cst,
            alpha_scale="mT",
            use_hard_cronin=True,
            mapping="exp", # or linear? Notebook 06 used exp? Check if unsure.
            device=device
        )
        cent_bins = [(0,20), (20,40), (40,60), (60,88)]
        
    else:
        raise ValueError(f"Unknown system: {system}")
        
    return {
        "roots_GeV": roots,
        "glauber": glauber,
        "qp_base": qp,
        "cent_bins": cent_bins,
        "cent_labels": [f"{int(a)}-{int(b)}%" for (a,b) in cent_bins]
    }
