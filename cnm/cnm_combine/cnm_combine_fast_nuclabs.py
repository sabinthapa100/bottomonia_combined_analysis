"""
cnm_combine_fast_nuclabs.py

Optimized CNM combination module WITH NUCLEAR ABSORPTION support.
nPDF (with optional absorption) × (eloss × Cronin pT broadening).
Vectorized implementation of energy loss and broadening calculations.

NEW in this version:
- Supports nuclear absorption via NuclearAbsorption class
- Enables testing of npdf-only, npdf+absorption, and cnm+absorption components
"""

from __future__ import annotations

import math
import numpy as np
import sys
from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence, Tuple, Literal, List
from pathlib import Path

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Import siblings
from particle import Particle, PPSpectrumParams
from glauber import OpticalGlauber, SystemSpec
import quenching_fast as QF
import eloss_cronin_centrality as EC

from npdf_data import NPDFSystem, RpAAnalysis
from gluon_ratio import EPPS21Ratio, GluonEPPSProvider

# Import the ABSORPTION-enabled nPDF module
import npdf_centrality_dAu as npdf_nuclabs
from npdf_centrality_dAu import (
    NuclearAbsorption,
    compute_df49_by_centrality,
    make_centrality_weight_dict,
    bin_rpa_vs_y,
    bin_rpa_vs_pT,
    bin_rpa_vs_centrality,
)

# ------------------------------------------------------------------
# Vectorized Physics Kernels
# ------------------------------------------------------------------

def xA_tensor(P, roots_GeV, qp, y_t: torch.Tensor, pT_t: torch.Tensor, y_sign_for_xA: int = -1):
    """
    Vectorized xA calculation.
    """
    M = float(P.M_GeV)
    roots = float(roots_GeV)
    
    mT = torch.sqrt(M*M + pT_t*pT_t)
    x_target = (mT / roots) * torch.exp(y_sign_for_xA * y_t)
    
    # x0 from L (qp.LA_fm is scalar here)
    x0 = QF.xA0_from_L(qp.LA_fm)
    
    xA = torch.minimum(torch.tensor(x0, device=x_target.device, dtype=x_target.dtype), x_target)
    xA = torch.clamp(xA, min=1e-12, max=0.99)
    return xA

def R_pA_eloss_batch(
    P: Particle,
    roots_GeV: float,
    qp,
    y_t: torch.Tensor,   # Shape (N,)
    pT_t: torch.Tensor,  # Shape (N,)
    Ny_int: int = 64,    # Integration points for z
    pp_override: Optional[PPSpectrumParams] = None,
    y_sign_for_xA: int = -1
) -> torch.Tensor:       # Shape (N,)
    """
    Batch calculation of R_loss for N points.
    Optimized to use a single grid for z-integration.
    """
    assert _HAS_TORCH
    device = y_t.device
    N = y_t.numel()
    
    P_eff = P
    if pp_override is not None:
        P_eff = replace(P, pp_params=pp_override)
    
    # Precompute denominators
    F2_den = EC.F2_t(P_eff, y_t, pT_t, roots_GeV)
    # Avoid div/0
    mask_valid = (F2_den > 1e-12)

    # Prepare integration nodes (exp mapping)
    # We use a fixed range logic suitable for batching.
    # dymax varies per point, but we can integrate up to a safe max 
    # and zero out contributions where z > zmax_local.
    # However, QF.PhatA_t needs meaningful z.
    
    # Simplify: Use fixed max z based on max dymax in batch or global max.
    # QF uses dy_max(y, y_max_pt).
    mT = torch.sqrt(float(P.M_GeV)**2 + pT_t*pT_t)
    y_max_pt = torch.log(roots_GeV/mT) # simplified
    dy_m = torch.clamp(y_max_pt - y_t, min=0.0, max=math.log(2.0))
    z_max_local = torch.expm1(dy_m)
    
    # Global max z for the grid
    z_max_global = z_max_local.max().item()
    if z_max_global <= 1e-12:
         return torch.ones_like(y_t)

    # Integration nodes u: -30 to log(zmax)
    umin = -30.0
    umax = math.log(max(z_max_global, 1e-300))
    u_node, wu_node = QF._gl_nodes_torch(umin, umax, Ny_int, device)
    
    # Shapes:
    # u_node: (Ny,)
    # y_t: (N,)
    # Broadcast to (N, Ny)
    u = u_node.unsqueeze(0).expand(N, -1)     # (N, Ny)
    wu = wu_node.unsqueeze(0).expand(N, -1)   # (N, Ny)
    
    z = torch.exp(u) # (N, Ny)
    # Mask out z > z_max_local
    mask_z = (z <= z_max_local.unsqueeze(1))
    z = torch.clamp(z, min=1e-12) # safe for Phat evaluation
    
    # xA calculation (N,) -> (N, 1)
    xA = xA_tensor(P, roots_GeV, qp, y_t, pT_t, y_sign_for_xA=y_sign_for_xA).unsqueeze(1)
    mT_exp = mT.unsqueeze(1)
    pT_exp = pT_t.unsqueeze(1)
    
    # Phat evaluation (N, Ny)
    # qp.LA_fm is scalar, used inside PhatA_t via qp
    ph = QF.PhatA_t(z, mT_exp, xA.expand_as(z), qp, pT=pT_exp)
    
    # Integrand
    # dy = ln(1+z)
    dy = torch.log1p(z)
    yshift = y_t.unsqueeze(1) + dy
    
    # F2 num: need broadcast pT
    F2_num = EC.F2_t(P_eff, yshift, pT_exp.expand_as(yshift), roots_GeV)
    ratio = F2_num / F2_den.unsqueeze(1)
    ratio = torch.where(torch.isfinite(ratio) & (ratio >= 0.0), ratio, torch.zeros_like(ratio))
    
    jac_z = torch.exp(u)
    inv1pz = 1.0 / (1.0 + z)
    
    # Weighted sum
    integrand = jac_z * ph * inv1pz * ratio
    norm_term = jac_z * ph
    
    # Apply mask (z <= zmax) by zeroing integrand/norm where invalid
    # Actually, if we just let it run, Phat should be valid but physics might be wrong if we integrate beyond limits
    # The mask is important.
    integrand = torch.where(mask_z, integrand, torch.zeros_like(integrand))
    norm_term = torch.where(mask_z, norm_term, torch.zeros_like(norm_term))
    
    val = torch.sum(wu * integrand, dim=1) # (N,)
    Zc  = torch.sum(wu * norm_term, dim=1) # (N,)
    
    Zc = torch.clamp(Zc, min=0.0, max=1.0)
    p0 = 1.0 - Zc
    
    R_loss = p0 + val
    
    # if den was 0, return 1.0
    return torch.where(mask_valid, R_loss, torch.ones_like(R_loss))

def R_pA_broad_batch(
    P: Particle,
    roots_GeV: float,
    qp,
    y_t: torch.Tensor,
    pT_t: torch.Tensor,
    Nphi: int = 64,
    pp_override: Optional[PPSpectrumParams] = None,
    y_sign_for_xA: int = -1
) -> torch.Tensor:
    assert _HAS_TORCH
    device = y_t.device
    N = y_t.numel()
    
    # Denominators
    P_eff = P
    if pp_override is not None:
        P_eff = replace(P, pp_params=pp_override)
    
    F1_den = EC.F1_t(P_eff, pT_t, roots_GeV)
    F2_den = EC.F2_t(P_eff, y_t, pT_t, roots_GeV)
    denom = F1_den * F2_den
    mask_valid = (denom > 1e-12)
    
    # xA and dpt
    xA = xA_tensor(P, roots_GeV, qp, y_t, pT_t, y_sign_for_xA=y_sign_for_xA)
    
    # dpt calculation (vectorized?)
    # QF._dpt_from_xL_t takes x, L.
    # L is scalar qp.LA_fm. x is (N,).
    # Returns (N,)
    broad_model = getattr(qp, "broadening_model", "ring").lower()
    use_ring = broad_model in ("ring", "hard", "fixed", "shift")
    
    dpt = QF._dpt_from_xL_t(qp, xA, qp.LA_fm, hard=use_ring)
    dpt = torch.abs(dpt)
    
    # Phi nodes (one set)
    phi, wphi, cphi, sphi = QF._phi_nodes_gl_torch(Nphi, device)
    # cphi shape (Nphi,)
    
    # Broadcast
    # We want result (N,)
    # p_shifted needs (N, Nphi)
    
    pT_exp = pT_t.unsqueeze(1)   # (N, 1)
    dpt_exp = dpt.unsqueeze(1)   # (N, 1)
    cphi_exp = cphi.unsqueeze(0) # (1, Nphi)
    
    # p_shifted^2
    psh_sq = pT_exp**2 + dpt_exp**2 - 2*pT_exp*dpt_exp*cphi_exp
    pshift = torch.sqrt(torch.clamp(psh_sq, min=0.0))
    
    # F1 num (N, Nphi)
    F1_num = EC.F1_t(P_eff, pshift, roots_GeV)
    
    # F2 num (N, Nphi) - y is constant in phi integral
    y_exp = y_t.unsqueeze(1).expand(N, Nphi)
    F2_num = EC.F2_t(P_eff, y_exp, pshift, roots_GeV)
    
    num = F1_num * F2_num
    
    # Average over phi
    avg_num = torch.sum(num * wphi.unsqueeze(0), dim=1) # (N,)
    
    R = avg_num / denom
    R = torch.where(torch.isfinite(R) & (R >= 0.0), R, torch.zeros_like(R))
    return torch.where(mask_valid, R, torch.ones_like(R))


# ------------------------------------------------------------------
# Batch Driver
# ------------------------------------------------------------------

def rpa_binned_batch_driver(
    P: Particle, roots_GeV: float, qp_base,
    glauber, cent_bins,
    bins_a: np.ndarray, # y_edges or pT_edges (primary axis)
    bins_b_range: Tuple[float, float], # Secondary axis integration range
    mode: Literal["vs_y", "vs_pT"],
    components=("eloss_broad",),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: str = "pp",
    table_for_pp=None,
    weight_ref_y: float | str = "local",
    pp_override=None,
    y_sign_for_xA: int = -1,
    debug: bool = False
):
    """
    Computes binned results for ALL bins and ALL centralities using batch processing.
    """
    if not _HAS_TORCH:
        raise RuntimeError("cnm_combine_fast: torch required for vectorized kernels.")
    
    device = EC._qp_device(qp_base)
    
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in cent_bins]
    L_by = glauber.leff_bins_pA(cent_bins, method="optical")
    Leff = [float(L_by[lab]) for lab in labels]
    
    # Weights for MB
    w_bins = np.array([QF._optical_bin_weight_pA(glauber, a, b) for (a,b) in cent_bins], float)
    w_bins /= max(w_bins.sum(), 1e-30)
    w_dict = {lab: w_bins[i] for i, lab in enumerate(labels)}

    n_primary = len(bins_a) - 1
    
    # Pre-generate quadrature grids for primary bins
    # We will accumulate numerators and denominators for each bin.
    # To batch effectively, we construct a flattened list of All Points for All Bins.
    # Then for each centrality, we evaluate.
    
    # 1. Construct integration points for all bins
    # For vs_y: bin i has y_nodes (Ny), p_nodes (Npt over range).
    # For vs_pT: bin i has p_nodes (Npt), y_nodes (Ny over range).
    
    # We can perform a "Super Grid" if bins are contiguous?
    # But usually we integrate specific bins.
    
    all_y = []
    all_p = []
    all_w = [] # Combined weight (wy * wp * jacobians)
    bin_indices = [] # Map point -> bin index
    
    # Secondary nodes (fixed range)
    if mode == "vs_y":
        sec_nodes, sec_w = QF._gl_nodes_np(bins_b_range[0], bins_b_range[1], Npt_bin)
    else:
        sec_nodes, sec_w = QF._gl_nodes_np(bins_b_range[0], bins_b_range[1], Ny_bin)

    for i in range(n_primary):
        a0, a1 = bins_a[i], bins_a[i+1]
        if mode == "vs_y":
            prim_nodes, prim_w = QF._gl_nodes_np(a0, a1, Ny_bin)
            # Meshgrid
            Y, P_ = np.meshgrid(prim_nodes, sec_nodes, indexing='ij')
            WY, WP = np.meshgrid(prim_w, sec_w, indexing='ij')
            W = WY * WP
            
            flat_y = Y.ravel()
            flat_p = P_.ravel()
            flat_w = W.ravel()
        else:
            prim_nodes, prim_w = QF._gl_nodes_np(a0, a1, Npt_bin)
            P_, Y = np.meshgrid(prim_nodes, sec_nodes, indexing='ij')
            WP, WY = np.meshgrid(prim_w, sec_w, indexing='ij')
            W = WY * WP
            
            flat_y = Y.ravel()
            flat_p = P_.ravel()
            flat_w = W.ravel()

        all_y.append(flat_y)
        all_p.append(flat_p)
        all_w.append(flat_w)
        bin_indices.append(np.full(flat_y.shape, i, dtype=int))

    # Concatenate all
    batch_y_np = np.concatenate(all_y)
    batch_p_np = np.concatenate(all_p)
    batch_w_np = np.concatenate(all_w)
    batch_idx = np.concatenate(bin_indices)
    
    if debug:
        print(f"[DEBUG] Batch size: {len(batch_y_np)} points across {n_primary} bins.")
    
    # Calculate pp weights (outside torch loop usually, or inside)
    # sigma_pp * pT
    # If using table, we can batch this too.
    # EC._sigma_pp_weight handles one by one. I should use vectorized if possible.
    
    batch_sigma_w = np.ones_like(batch_w_np)
    if weight_kind == "pp":
         # Attempt batch evaluation
         if hasattr(table_for_pp, "device"):
             # Torch table
             ty = torch.tensor(batch_y_np, device=device)
             tp = torch.tensor(batch_p_np, device=device)
             with torch.no_grad():
                 spp = table_for_pp(ty, tp).cpu().numpy()
         else:
             # Loop or scalar
             # Fallback to manual loop is slow.
             # Assume analytic if table is None
             # Or if table is None, EC uses F1*F2 approximation.
             # We can vectorize that easily.
             if table_for_pp is None:
                  # Use F1*F2 approximation
                  ty = torch.tensor(batch_y_np, device=device)
                  tp = torch.tensor(batch_p_np, device=device)
                  
                  P_eff = P
                  if pp_override is not None:
                      P_eff = replace(P, pp_params=pp_override)
                  
                  f1 = EC.F1_t(P_eff, tp, roots_GeV)
                  f2 = EC.F2_t(P_eff, ty, tp, roots_GeV)
                  spp = (f1 * f2).cpu().numpy()
             else:
                  # Fallback to loop for generic callable
                  spp = np.array([float(table_for_pp(y,p)) for y,p in zip(batch_y_np, batch_p_np)])
        
         # Multiply by pT (jacobian d2pT -> dpT dy) and weight assignment
         batch_sigma_w = spp * np.maximum(batch_p_np, 1e-8)

    final_weights = batch_w_np * batch_sigma_w
    
    # Move to GPU
    y_t = torch.tensor(batch_y_np, device=device, dtype=torch.float64)
    p_t = torch.tensor(batch_p_np, device=device, dtype=torch.float64)
    w_t = torch.tensor(final_weights, device=device, dtype=torch.float64)
    # Denominator (weight sum) per bin
    # We can precompute this since it doesn't depend on bands
    
    # Accumulators
    # results[comp][cent_idx][bin_idx]
    
    R_comp = {comp: {lab: np.zeros(n_primary, float) for lab in labels} for comp in components}
    
    # Loop centralities (L varies)
    for c_idx, lab in enumerate(labels):
        L = Leff[c_idx]
        qpL = replace(qp_base, LA_fm=float(L), LB_fm=float(L))
        
        is_AA = (getattr(qp_base, "system", "pA") == "AA")
        
        # Calculate components
        # R_loss
        need_loss = any(k in components for k in ["loss", "eloss_broad"])
        need_broad = any(k in components for k in ["broad", "eloss_broad"])
        
        val_loss = None
        val_broad = None
        
        if need_loss:
            val_loss = R_pA_eloss_batch(P, roots_GeV, qpL, y_t, p_t, Ny_int=64, pp_override=pp_override, y_sign_for_xA=y_sign_for_xA)
            if is_AA:
                # Multiply by loss from the other nucleus (flipped rapidity)
                val_loss_flipped = R_pA_eloss_batch(P, roots_GeV, qpL, -y_t, p_t, Ny_int=64, pp_override=pp_override, y_sign_for_xA=y_sign_for_xA)
                val_loss = val_loss * val_loss_flipped

        if need_broad:
            val_broad = R_pA_broad_batch(P, roots_GeV, qpL, y_t, p_t, Nphi=64, pp_override=pp_override, y_sign_for_xA=y_sign_for_xA)
            if is_AA:
                # Multiply by broadening from the other nucleus (flipped rapidity)
                val_broad_flipped = R_pA_broad_batch(P, roots_GeV, qpL, -y_t, p_t, Nphi=64, pp_override=pp_override, y_sign_for_xA=y_sign_for_xA)
                val_broad = val_broad * val_broad_flipped
            
        for comp in components:
            if comp == "loss":
                R_vec = val_loss
            elif comp == "broad":
                R_vec = val_broad
            elif comp == "eloss_broad":
                R_vec = val_loss * val_broad
            else:
                continue
                
            # Aggregate by bin
            # Weighted average: sum(R * w) / sum(w)
            num = R_vec * w_t
            
            # Scatter add using numpy (faster for simple reduction than implementing scatter on gpu for small bins? 
            # No, GPU scatter is good. But returning to CPU is fine.)
            num_np = num.cpu().numpy()
            
            # Using bincount/add.at
            denom = np.bincount(batch_idx, weights=final_weights, minlength=n_primary)
            numer = np.bincount(batch_idx, weights=num_np, minlength=n_primary)
            
            safe_denom = np.where(denom > 0, denom, 1.0)
            avg = numer / safe_denom
            R_comp[comp][lab] = avg

    # Compute MB
    for comp in components:
        R_mat = np.vstack([R_comp[comp][lab] for lab in labels])
        w_arr = np.array([w_dict[lab] for lab in labels])
        R_MB = np.average(R_mat, axis=0, weights=w_arr)
        R_comp[comp]["MB"] = R_MB

    centers = 0.5*(bins_a[:-1] + bins_a[1:])
    return centers, R_comp, labels, batch_idx, final_weights


# ------------------------------------------------------------------
# Band Drivers (Wrappers)
# ------------------------------------------------------------------

def optimized_rpa_band_common(
    P, roots, qp_base, gl, cent_bins,
    bins_a, bins_b_range, mode,
    components,
    q0_pair, p0_scale_pair,
    Ny_bin, Npt_bin,
    weight_kind, table_for_pp,
    y_sign_for_xA: int = -1,
    debug: bool = False
):
    # This replaces rpa_band_vs_y/pT
    # Logic:
    # 1. Calculate Loss band (q0_lo, q0_hi). Parallel over cents? 
    #    Actually we loop qp variation -> calls driver.
    
    bands = {}
    
    # 0. BASELINE (central)
    _, R_base, labels, _, _ = rpa_binned_batch_driver(
         P, roots, qp_base, gl, cent_bins, bins_a, bins_b_range, mode,
         components=("loss", "broad"), Ny_bin=Ny_bin, Npt_bin=Npt_bin,
         weight_kind=weight_kind, table_for_pp=table_for_pp,
         y_sign_for_xA=y_sign_for_xA, debug=debug
    )

    # 1. LOSS BAND (q0 scan)
    if any(k in components for k in ["loss", "eloss_broad"]):
        q0_lo, q0_hi = q0_pair
        qp_lo = replace(qp_base, qhat0=float(q0_lo))
        qp_hi = replace(qp_base, qhat0=float(q0_hi))
        
        _, R_lo, _, _, _ = rpa_binned_batch_driver(
             P, roots, qp_lo, gl, cent_bins, bins_a, bins_b_range, mode,
             components=("loss",), Ny_bin=Ny_bin, Npt_bin=Npt_bin,
             weight_kind=weight_kind, table_for_pp=table_for_pp,
             y_sign_for_xA=y_sign_for_xA, debug=debug
        )
        _, R_hi, _, _, _ = rpa_binned_batch_driver(
             P, roots, qp_hi, gl, cent_bins, bins_a, bins_b_range, mode,
             components=("loss",), Ny_bin=Ny_bin, Npt_bin=Npt_bin,
             weight_kind=weight_kind, table_for_pp=table_for_pp,
             y_sign_for_xA=y_sign_for_xA, debug=debug
        )
        
        RL_c, RL_lo, RL_hi = {}, {}, {}
        for lab in labels + ["MB"]:
            rc, rl, rh = EC._two_point_band(R_lo["loss"][lab], R_hi["loss"][lab], R_base["loss"][lab])
            RL_c[lab], RL_lo[lab], RL_hi[lab] = rc, rl, rh
        bands["loss"] = (RL_c, RL_lo, RL_hi)

    # 2. BROAD BAND (p0 scan)
    if any(k in components for k in ["broad", "eloss_broad"]):
         P_lo = EC.particle_with_scaled_p0(P, p0_scale_pair[0])
         P_hi = EC.particle_with_scaled_p0(P, p0_scale_pair[1])
         _, R_lo_b, _, _, _ = rpa_binned_batch_driver(
             P_lo, roots, qp_base, gl, cent_bins, bins_a, bins_b_range, mode,
             components=("broad",), Ny_bin=Ny_bin, Npt_bin=Npt_bin,
             weight_kind=weight_kind, table_for_pp=table_for_pp,
             pp_override=P_lo.pp, y_sign_for_xA=y_sign_for_xA, debug=debug
         )
         _, R_hi_b, _, _, _ = rpa_binned_batch_driver(
             P_hi, roots, qp_base, gl, cent_bins, bins_a, bins_b_range, mode,
             components=("broad",), Ny_bin=Ny_bin, Npt_bin=Npt_bin,
             weight_kind=weight_kind, table_for_pp=table_for_pp,
             pp_override=P_hi.pp, y_sign_for_xA=y_sign_for_xA, debug=debug
         )

         RB_c, RB_lo, RB_hi = {}, {}, {}
         for lab in labels + ["MB"]:
            rc, rl, rh = EC._two_point_band(R_lo_b["broad"][lab], R_hi_b["broad"][lab], R_base["broad"][lab])
            RB_c[lab], RB_lo[lab], RB_hi[lab] = rc, rl, rh
         bands["broad"] = (RB_c, RB_lo, RB_hi)

    # COMBINE
    if "eloss_broad" in components:
        RT_c, RT_lo, RT_hi = EC.combine_factorized_bands_1d(
            bands["loss"][0], bands["loss"][1], bands["loss"][2],
            bands["broad"][0], bands["broad"][1], bands["broad"][2]
        )
        bands["eloss_broad"] = (RT_c, RT_lo, RT_hi)

    # Get centers from one of the drivers (they are same)
    centers = 0.5*(bins_a[:-1] + bins_a[1:])
    return centers, bands, labels


# ------------------------------------------------------------------
# Main Class
# ------------------------------------------------------------------

from cnm_combine import CNMCombine as CNMCombineBase, DEFAULT_PT_RANGE, _tags_for_cent_bins, combine_two_bands_1d

@dataclass
class CNMCombineFast(CNMCombineBase):
    """
    Optimized subclass of CNMCombine WITH NUCLEAR ABSORPTION support.
    Overrides the calculation methods to use vectorized kernels.
    
    New parameter:
    --------------
    absorption : NuclearAbsorption or None
        If provided, nuclear absorption is applied to the nPDF calculation.
        Default None means no absorption (backward compatible).
    """
    y_sign_for_xA: int = -1
    debug: bool = False
    absorption: Optional[NuclearAbsorption] = None  # NEW: nuclear absorption support
    
    @property
    def system(self):
        return self.spec.system

    def _calc_eloss_broad_band_vs_y(self, y_edges, pt_range_avg, components):
        # Maps to rpa_band_vs_y logic but using Fast
        return optimized_rpa_band_common(
            self.particle, self.sqrt_sNN, self.qp_base, self.gl,
            self.cent_bins, y_edges, pt_range_avg, "vs_y",
            components, self.q0_pair, self.p0_scale_pair,
            Ny_bin=16, Npt_bin=24, # Use slightly higher quality
            weight_kind="pp", table_for_pp=None,
            y_sign_for_xA=self.y_sign_for_xA, debug=self.debug
        )

    def _calc_eloss_broad_band_vs_pT(self, pt_edges, y_window, components):
        return optimized_rpa_band_common(
            self.particle, self.sqrt_sNN, self.qp_base, self.gl,
            self.cent_bins, pt_edges, y_window, "vs_pT",
            components, self.q0_pair, self.p0_scale_pair,
            Ny_bin=16, Npt_bin=24,
            weight_kind="pp", table_for_pp=None,
            y_sign_for_xA=self.y_sign_for_xA, debug=self.debug
        )

    # Override cnm_vs_y to use optimized calc
    def cnm_vs_y(self, y_edges=None, pt_range_avg=None, include_mb=True, components=("npdf","eloss","broad","eloss_broad","cnm"), **kwargs):
        # We invoke the base logic but intercept the EC calls?
        # No, base calls EC.rpa_band_vs_y.
        # I should reimplement cnm_vs_y OR monkeypatch EC?
        # Using subclass override is cleaner.
        
        # COPY of cnm_vs_y code but using self._calc...
        # ... actually, I can just copy the method and change the EC call.
        # Or better: CNMCombineBase calls EC directly.
        # To reuse code, I will copy and adapt.
        
        if y_edges is None: y_edges = self.y_edges
        if pt_range_avg is None: pt_range_avg = self.pt_range_avg
        
        # 1. Call nPDF part (reusing base tools)
        if self.npdf_ctx is not None:
            wcent = make_centrality_weight_dict(self.cent_bins, c0=self.cent_c0) if include_mb else None
            npdf_bins_y = bin_rpa_vs_y(
                self.npdf_ctx["df49_by_cent"], self.npdf_ctx["df_pp"], self.npdf_ctx["df_pa"], self.npdf_ctx["gluon"],
                cent_bins=self.cent_bins, y_edges=y_edges, pt_range_avg=pt_range_avg, weight_mode=self.weight_mode, y_ref=self.y_ref,
                pt_floor_w=self.pt_floor_w, wcent_dict=wcent, include_mb=include_mb
            )
        else:
            npdf_bins_y = None
        
        # 2. Call FAST eloss
        y_cent, bands_y, labels = self._calc_eloss_broad_band_vs_y(y_edges, pt_range_avg, ["loss","broad","eloss_broad"])
        
        # 3. Combine
        final_bands = self._combine_bands_generic(components, npdf_bins_y, bands_y, labels, include_mb, mode="y")
        return y_cent, labels, final_bands

    def cnm_vs_pT(self, y_window, pt_edges=None, components=("npdf","eloss","broad","eloss_broad","cnm"), include_mb=True, **kwargs):
        if pt_edges is None: pt_edges = self.p_edges
        if len(y_window)==3: y0,y1,_=y_window
        else: y0,y1=y_window
        
        if self.npdf_ctx is not None:
            wcent = make_centrality_weight_dict(self.cent_bins, c0=self.cent_c0) if include_mb else None
            npdf_bins = bin_rpa_vs_pT(
                self.npdf_ctx["df49_by_cent"], self.npdf_ctx["df_pp"], self.npdf_ctx["df_pa"], self.npdf_ctx["gluon"],
                cent_bins=self.cent_bins, pt_edges=pt_edges, y_window=(y0,y1), weight_mode=self.weight_mode, y_ref=self.y_ref,
                pt_floor_w=self.pt_floor_w, wcent_dict=wcent, include_mb=include_mb
            )
        else:
            npdf_bins = None
        
        pT_cent, bands_pt, labels = self._calc_eloss_broad_band_vs_pT(pt_edges, (y0,y1), ["loss","broad","eloss_broad"])
        
        final_bands = self._combine_bands_generic(components, npdf_bins, bands_pt, labels, include_mb, mode="pT")
        return pT_cent, labels, final_bands

    def cnm_vs_centrality(self, y_window, pt_range_avg=None, components=("npdf","eloss","broad","eloss_broad","cnm"), include_mb=True, **kwargs):
        # We need a fast centrality driver.
        # EC.rpa_band_vs_centrality just loops buckets.
        # We can implement a fast version too if needed, but usually centrality bin count is small (5).
        # However, the INTGEGRATION inside each bin is what's slow.
        # Reuse optimized_rpa_band_common logic?
        # rpa_band_vs_centrality returns 1 point per bin.
        # This is equivalent to rpa_band_vs_y with 1 huge Y-bin? No, Y-window.
        # It's simply one integrated number per centrality.
        # We can use rpa_binned_batch_driver with ONE bin (the whole window).
        
        if pt_range_avg is None: pt_range_avg = self.pt_range_avg
        if len(y_window)==3: y0,y1,_=y_window
        else: y0,y1=y_window
        
        # Use existing Fast driver with 1 bin
        # We trick it by passing bins_a as [y_min, y_max].
        # But we need result per separate centrality bin.
        # The driver returns {comp: {cent_lab: array_of_bins}}.
        # Here array_of_bins has length 1.
        
        # nPDF
        if self.npdf_ctx is not None:
            wcent = make_centrality_weight_dict(self.cent_bins, c0=self.cent_c0)
            width_weights = np.array([wcent[f"{int(a)}-{int(b)}%"] for (a, b) in self.cent_bins], float)
            npdf_cent = bin_rpa_vs_centrality(
                 self.npdf_ctx["df49_by_cent"], self.npdf_ctx["df_pp"], self.npdf_ctx["df_pa"], self.npdf_ctx["gluon"],
                 cent_bins=self.cent_bins, y_window=(y0,y1), pt_range_avg=pt_range_avg, weight_mode=self.weight_mode, y_ref=self.y_ref,
                 pt_floor_w=self.pt_floor_w, width_weights=width_weights
            )
        else:
            npdf_cent = None
        
        # Fast ELOSS
        # We compute for 'vs_y' but with a single bin cover y0..y1
        dummy_y_edges = np.array([y0, y1])
        _, bands, labels = optimized_rpa_band_common(
            self.particle, self.sqrt_sNN, self.qp_base, self.gl,
            self.cent_bins, dummy_y_edges, pt_range_avg, "vs_y",
            ["loss","broad","eloss_broad"], self.q0_pair, self.p0_scale_pair,
            Ny_bin=16, Npt_bin=24, weight_kind="pp", table_for_pp=None,
            y_sign_for_xA=self.y_sign_for_xA, debug=self.debug
        )
        
        # bands[comp] -> (Rc, Rlo, Rhi) -> dict[lab] -> array len 1.
        # We need to restructure to cnm_vs_centrality format detailed in cnm_combine.py
        
        cnm_cent = {}
        tags = _tags_for_cent_bins(self.cent_bins, include_mb=include_mb) # e.g. "0-20%", ..., "MB"
        
        if "npdf" in components and npdf_cent is not None:
            # nPDF
            Rc_n = np.asarray(npdf_cent["r_central"], float)
            Rlo_n = np.asarray(npdf_cent["r_lo"], float)
            Rhi_n = np.asarray(npdf_cent["r_hi"], float)
            mb_n = (float(npdf_cent["mb_r_central"]), float(npdf_cent["mb_r_lo"]), float(npdf_cent["mb_r_hi"]))
            cnm_cent["npdf"] = (Rc_n, Rlo_n, Rhi_n, *mb_n)
        else:
            Rc_n = Rlo_n = Rhi_n = None
            mb_n = (1.0, 1.0, 1.0) # Dummy for combine

        # process local bands
        for comp in ["loss", "broad", "eloss_broad"]:
            if comp == "loss" and "eloss" not in components: continue
            if comp == "broad" and "broad" not in components: continue
            if comp == "eloss_broad" and "eloss_broad" not in components: continue
            
            target_key = "eloss" if comp=="loss" else comp
            
            Bc, Blo, Bhi = bands[comp]
            
            # Arrays
            vals_c = np.array([float(Bc[lab][0]) for lab in labels])
            vals_lo = np.array([float(Blo[lab][0]) for lab in labels])
            vals_hi = np.array([float(Bhi[lab][0]) for lab in labels])
            
            mb_c = float(Bc["MB"][0])
            mb_lo = float(Blo["MB"][0])
            mb_hi = float(Bhi["MB"][0])
            
            cnm_cent[target_key] = (vals_c, vals_lo, vals_hi, mb_c, mb_lo, mb_hi)
        
        # Compute CNM
        if "cnm" in components:
            (Rt_c, Rt_lo, Rt_hi) = bands["eloss_broad"]
            
            # Centrality part
            Rtot_c = np.array([float(Rt_c[lab][0]) for lab in labels])
            Rtot_lo = np.array([float(Rt_lo[lab][0]) for lab in labels])
            Rtot_hi = np.array([float(Rt_hi[lab][0]) for lab in labels])

            if Rc_n is not None:
                Rc_cnm, Rlo_cnm, Rhi_cnm = combine_two_bands_1d(
                    Rc_n, Rlo_n, Rhi_n, Rtot_c, Rtot_lo, Rtot_hi
                )
                # MB part
                rmb_c, rmb_lo, rmb_hi = combine_two_bands_1d(
                    np.array([mb_n[0]]), np.array([mb_n[1]]), np.array([mb_n[2]]),
                    np.array([float(Rt_c["MB"][0])]), np.array([float(Rt_lo["MB"][0])]), np.array([float(Rt_hi["MB"][0])])
                )
                cnm_cent["cnm"] = (Rc_cnm, Rlo_cnm, Rhi_cnm, float(rmb_c[0]), float(rmb_lo[0]), float(rmb_hi[0]))
            else:
                # CNM is just ELoss+Broad if nPDF missing
                cnm_cent["cnm"] = (Rtot_c, Rtot_lo, Rtot_hi, float(Rt_c["MB"][0]), float(Rt_lo["MB"][0]), float(Rt_hi["MB"][0]))

        return cnm_cent

    def _combine_bands_generic(self, components, npdf_data, band_data, labels, include_mb, mode="y"):
        
        final_bands = {}
        tags = labels + (["MB"] if include_mb else [])
        
        # Build structure
        for comp in components:
            Dc, Dlo, Dhi = {}, {}, {}
            for tag in tags:
                if comp == "npdf":
                    d = npdf_data[tag]
                    Dc[tag] = np.asarray(d["r_central"], float)
                    Dlo[tag] = np.asarray(d["r_lo"], float)
                    Dhi[tag] = np.asarray(d["r_hi"], float)
                elif comp == "cnm":
                    # Need npdf and eloss_broad
                    rtc, rtlo, rthi = band_data["eloss_broad"]
                    rtc, rtlo, rthi = rtc[tag], rtlo[tag], rthi[tag]

                    if npdf_data is not None and tag in npdf_data:
                        dn = npdf_data[tag]
                        rnc, rnlo, rnhi = np.asarray(dn["r_central"]), np.asarray(dn["r_lo"]), np.asarray(dn["r_hi"])
                        rc, rlo, rhi = combine_two_bands_1d(rnc, rnlo, rnhi, rtc, rtlo, rthi)
                        Dc[tag], Dlo[tag], Dhi[tag] = rc, rlo, rhi
                    else:
                        # eloss only
                        Dc[tag], Dlo[tag], Dhi[tag] = rtc, rtlo, rthi
                else:
                    # Map loop to eloss keys
                    # cnm_combine uses "eloss" key but EC returns "loss"
                    key = comp
                    if comp == "eloss": key = "loss"
                    
                    if key in band_data:
                         bc, blo, bhi = band_data[key]
                         Dc[tag], Dlo[tag], Dhi[tag] = bc[tag], blo[tag], bhi[tag]
            
            if Dc:
                final_bands[comp] = (Dc, Dlo, Dhi)
        return final_bands

    @classmethod
    def from_defaults(
        cls,
        energy: Literal["200", "5.02", "8.16"] = "8.16",
        family: Literal["charmonia", "bottomonia"] = "charmonia",
        particle_state: str = "avg",
        m_state_for_np: float | str | None = None,
        cent_bins: Sequence[Tuple[float, float]] = None,
        y_edges: np.ndarray = None,
        p_edges: np.ndarray = None,
        y_windows: Sequence[Tuple[float, float, str]] = None,
        pt_range_avg: Tuple[float, float] = None,
        pt_floor_w: float = 1.0,
        weight_mode: str = "pp@local",
        y_ref: float = 0.0,
        cent_c0: float = 0.25,
        q0_pair: Tuple[float, float] = (0.05, 0.09),
        p0_scale_pair: Tuple[float, float] = (0.9, 1.1),
        nb_bsamples: int = 5,
        y_shift_fraction: Optional[float] = None, # None means use system default
        alpha_s_mode: Literal["constant", "running"] = "running",
        alpha0: float = 0.5,
        debug: bool = False,
        # NEW: Nuclear absorption parameters
        enable_absorption: bool = False,
        abs_sigma_mb: Optional[float] = None,  # None = use energy-dependent default
        abs_mode: str = "dA_avg_TA",  # or "avg_TA" for simpler pA-like
        absorption_obj: Optional[NuclearAbsorption] = None,  # optional override (e.g. feed-down combo)
    ) -> "CNMCombineFast":
        from cnm_combine import SQRTS, SIG_NN, EPPS_DIR, DAU_DIR, P5_DIR, P8_DIR, DEFAULT_Y_EDGES, DEFAULT_P_EDGES, DEFAULT_Y_WINDOWS, DEFAULT_CENT_BINS, alpha_s_provider
        
        energy = str(energy)
        if energy not in SQRTS:
            raise ValueError(f"Unknown energy {energy}")

        sqrt_sNN = SQRTS[energy]
        sigma_nn_mb = SIG_NN[energy]
        system_name = "dA" if energy == "200" else "pA"

        if cent_bins is None: cent_bins = DEFAULT_CENT_BINS
        if y_edges is None: y_edges = DEFAULT_Y_EDGES
        if p_edges is None: p_edges = DEFAULT_P_EDGES
        if y_windows is None: y_windows = DEFAULT_Y_WINDOWS
        
        if pt_range_avg is None:
            pt_range_avg = (0.0, 5.0) if energy == "200" else (0.0, 10.0)

        if y_shift_fraction is None:
            y_shift_fraction = 1.0 if energy == "200" else 2.0

        particle = Particle(family=family, state=particle_state)
        input_dir = DAU_DIR if energy=="200" else (P5_DIR if energy=="5.02" else P8_DIR)
        target_A = 197 if energy=="200" else 208
        
        m_state_for_np_used: float | str
        if m_state_for_np is None:
            m_state_for_np_used = "charmonium" if family=="charmonia" else ("bottomonium" if family=="bottomonia" else particle.M_GeV)
        else:
            m_state_for_np_used = m_state_for_np

        epps_ratio = EPPS21Ratio(A=target_A, path=str(EPPS_DIR))
        ## For pA or dA, xA is always target side (A=Pb)
        y_sign_for_xA = -1
        
        # NEW: Nuclear absorption setup
        absorption = None
        if enable_absorption:
            if absorption_obj is not None:
                absorption = absorption_obj
            else:
                # Set default sigma_abs based on energy if not provided
                if abs_sigma_mb is None:
                    if energy == "200":
                        abs_sigma_mb = 4.2
                    else:
                        abs_sigma_mb = 0.0

                # τ-modes require a charmonium-state radius (1S/1P/2S). We infer it
                # from the particle_state when possible to keep notebook usage minimal.
                abs_state = None
                if str(abs_mode).endswith("_tau") and (family == "charmonia"):
                    pst = str(particle_state).strip()
                    abs_state = "1S" if pst in ("avg", "", "None") else pst

                if abs_state is not None:
                    absorption = NuclearAbsorption(
                        mode=abs_mode,
                        sigma_abs_mb=abs_sigma_mb,
                        state=abs_state,
                    )
                else:
                    absorption = NuclearAbsorption(
                        mode=abs_mode,
                        sigma_abs_mb=abs_sigma_mb,
                    )
            if debug:
                sigma_dbg = getattr(absorption, "sigma_abs_mb", abs_sigma_mb)
                print(f"[CNMCombineFast] Nuclear absorption enabled: mode={abs_mode}, σ_abs={sigma_dbg} mb (override={absorption_obj is not None})")
        
        # Build the gluon provider
        gluon = GluonEPPSProvider(
            epps_ratio,
            sqrt_sNN_GeV=sqrt_sNN,
            m_state_GeV=m_state_for_np_used,
            y_sign_for_xA=y_sign_for_xA,
        ).with_geometry()

        gl_spec = SystemSpec(system_name, sqrt_sNN, A=target_A, sigma_nn_mb=sigma_nn_mb)
        gl_ana = OpticalGlauber(gl_spec, verbose=False)

        ana = RpAAnalysis()
        name_str = f"{system_name} {energy}"
        sys_npdf = NPDFSystem.from_folder(str(input_dir), kick="pp", name=name_str)

        base_df, r0, M = ana.compute_rpa_members(
            sys_npdf.df_pp, sys_npdf.df_pa, sys_npdf.df_errors,
            join="intersect", lowpt_policy="drop", pt_shift_min=pt_floor_w
        )

        # Compute nPDF context with absorption support
        # NOTE: We always use npdf_nuc labs module now (absorption-enabled)
        df49_by_cent, K_by_cent, SA_all, Y_SHIFT = compute_df49_by_centrality(
            base_df, r0, M, gluon, gl_ana,
            cent_bins=cent_bins, nb_bsamples=nb_bsamples, y_shift_fraction=y_shift_fraction,
            absorption=absorption  # Pass absorption (None for NO absorption)
        )

        npdf_ctx = dict(
            df49_by_cent=df49_by_cent,
            df_pp=sys_npdf.df_pp,
            df_pa=sys_npdf.df_pa,
            gluon=gluon,
        )

        alpha_s = alpha_s_provider(mode=alpha_s_mode, alpha0=alpha0, LambdaQCD=0.25)
        Lmb = gl_ana.leff_minbias_pA() if system_name == "pA" else gl_ana.leff_minbias_dA()
        
        device = "cpu"
        if _HAS_TORCH and torch.cuda.is_available(): device = "cuda"

        qp_base = QF.QuenchParams(
            qhat0=0.075, lp_fm=1.5, LA_fm=Lmb,
            LB_fm=1.5 if system_name == "pA" else Lmb,
            system=system_name if energy == "200" else "pPb",
            lambdaQCD=0.25, roots_GeV=sqrt_sNN,
            alpha_of_mu=alpha_s, alpha_scale="mT",
            use_hard_cronin=True, mapping="exp", device=device,
        )

        return cls(
            energy=energy,family=family, particle_state=particle_state,
            sqrt_sNN=sqrt_sNN, sigma_nn_mb=sigma_nn_mb,
            cent_bins=cent_bins, y_edges=np.asarray(y_edges, float),
            p_edges=np.asarray(p_edges, float), y_windows=y_windows,
            pt_range_avg=pt_range_avg, pt_floor_w=pt_floor_w,
            weight_mode=weight_mode, y_ref=y_ref, cent_c0=cent_c0,
            q0_pair=q0_pair, p0_scale_pair=p0_scale_pair,
            nb_bsamples=nb_bsamples, y_shift_fraction=y_shift_fraction,
            particle=particle, gl=gl_ana, qp_base=qp_base, npdf_ctx=npdf_ctx,
            y_sign_for_xA=y_sign_for_xA, spec=gl_spec, debug=debug,
            absorption=absorption  # NEW: pass absorption to instance
        )
