# eloss_cronin_centrality.py
"""
Coherent energy loss + pT broadening (Cronin) with centrality dependence.

This module is the notebook logic refactored into reusable functions.

Core physics:
    * R_pA_eloss(P, roots_GeV, qp, y, pT)
    * R_pA_broad(P, roots_GeV, qp, y, pT)
    * R_pA_factored = R_pA_eloss * R_pA_broad

Binned averages (no band):
    * R_binned_2D(...)
    * rpa_binned_vs_y(...)
    * rpa_binned_vs_pT(...)
    * rpa_vs_centrality(...)

Bands (two-point scans in q0 and p0):
    * rpa_band_vs_y_eloss / rpa_band_vs_y_broad / rpa_band_vs_y
    * rpa_band_vs_pT_eloss / rpa_band_vs_pT_broad / rpa_band_vs_pT
    * rpa_band_vs_centrality

Plot helpers (publication style, optional):
    * plot_RpA_vs_y_components_per_centrality(...)
    * plot_RpA_vs_pT_components_per_centrality(...)
    * plot_RpA_vs_centrality_components_band(...)

Min-bias centrality weights:
    * By default we use an exponential scheme w(c) ∝ exp(-c/c0) in c∈[0,1]
      integrated over each centrality bin ("exp" mode).
    * You can switch to optical-Glauber weights ("optical" mode).
    * Or supply your own custom weights ("custom" mode).

All interfaces are designed so you can:
  - call from a notebook with a single energy (e.g. 5.02 TeV),
  - later overlay additional energies on top,
  - later multiply with other factors if needed.

Author: Sabin  refactor.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Iterable, Literal, Sequence, Tuple, List

import numpy as np

# torch import is required for quenching_fast
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# --- local physics modules ---
from particle import Particle, PPSpectrumParams
from glauber import OpticalGlauber
import quenching_fast as QF

import matplotlib.ticker as ticker

# ------------------------------
# Global numerical knobs / floors
# ------------------------------
F1_FLOOR = 1e-16
F2_FLOOR = 1e-12   # for kinematic edge protection
ZC_EPS   = 1e-12

XMIN_SAFE = 1e-12
XMAX_SAFE = 0.99

# Arleo & Peigne Reference L_eff values
# d+Au 200 GeV (arXiv:1304.0901)
AP_LEFF_DAU_200 = {
    "MinBias": 10.23,
    "0-20%":   12.87,
    "20-40%":  9.62,
    "40-60%":  7.17,
    "60-88%":  3.84
}

# p+Pb 5.02 TeV (arXiv:1212.0434)
AP_LEFF_PPB_5020 = {
    "MinBias": 9.38, # Approximate MB value
    "0-20%":   12.86,
    "20-40%":  9.61,
    "40-60%":  7.15,
    "60-100%": 4.31
}


# ----------------------------------------------------------------------
# Error Handling & Validation Infrastructure
# ----------------------------------------------------------------------
class RpACalculationError(Exception):
    """Raised when R_pA calculation fails validation checks or encounters errors."""
    pass


def validate_inputs(P: Particle, roots_GeV: float, qp, y: float, pT: float) -> None:
    """
    Comprehensive input validation with helpful error messages.
    
    Raises
    ------
    RpACalculationError
        If any input is invalid with detailed explanation
    """
    # Particle mass must be physical
    M = P.M_GeV
    if M <= 0:
        raise RpACalculationError(f"Particle mass must be positive, got M={M} GeV")
    
    # Center-of-mass energy must exceed twice the particle mass
    if roots_GeV <= 2 * M:
        raise RpACalculationError(
            f"Center-of-mass energy √s={roots_GeV} GeV too low for particle mass M={M} GeV. "
            f"Need √s > 2M = {2*M} GeV for production."
        )
    
    # pT must be non-negative
    if pT < 0:
        raise RpACalculationError(f"Transverse momentum pT must be non-negative, got pT={pT} GeV")
    
    # Rapidity must be within kinematic limits
    mT = math.sqrt(M**2 + pT**2)
    y_max_kinematic = QF.y_max(roots_GeV, mT)
    if abs(y) > y_max_kinematic:
        raise RpACalculationError(
            f"Rapidity |y|={abs(y)} exceeds kinematic limit {y_max_kinematic:.2f} "
            f"for √s={roots_GeV} GeV, M={M} GeV, pT={pT} GeV"
        )
    
    # QuenchParams validation
    if qp.qhat0 <= 0:
        raise RpACalculationError(f"qhat0 must be positive, got {qp.qhat0}")
    
    # Relaxed: LA must be at least lp for energy loss to occur.
    # If LA <= lp, energy loss is physically zero/suppressed. 
    # We allow it here and handle it in the quenching functions by returning 1.0.
    pass
    
    if qp.lambdaQCD <= 0:
        raise RpACalculationError(f"λ_QCD must be positive, got {qp.lambdaQCD} GeV")


def _check_numerical_stability(array: torch.Tensor, name: str, context: str) -> None:
    """
    Check tensor for NaN, Inf, or unexpected values.
    
    Parameters
    ----------
    array : torch.Tensor
        Tensor to check
    name : str
        Variable name for error message
    context : str
        Calculation context (e.g., "R_pA_eloss integration")
        
    Raises
    ------
    RpACalculationError
        If numerical instability detected
    """
    if _HAS_TORCH:
        if torch.isnan(array).any():
            raise RpACalculationError(f"NaN detected in {name} during {context}")
        
        if torch.isinf(array).any():
            raise RpACalculationError(f"Inf detected in {name} during {context}")


# ----------------------------------------------------------------------
# Centrality weights for MB (w(c) scheme)
# ----------------------------------------------------------------------
def make_centrality_weight_dict(cent_bins: List[Tuple[float, float]],
                                c0: float = 0.25
                                ) -> Dict[str, float]:
    """
    Return dict[tag] -> W_bin for each centrality bin, with tags "0-20%",...

    Uses w(c) ∝ exp(-c/c0) on c∈[0,1], integrated over each bin.

    Parameters
    ----------
    cent_bins : list of (c0,c1) in %, e.g. [(0,20),(20,40),...]
    c0        : float, exponential scale in w(c)

    Returns
    -------
    WCENT : dict mapping tag -> weight (normalized to sum to 1)
    """
    edges_frac = np.array(
        [cent_bins[0][0]] + [b for (_, b) in cent_bins],
        float
    ) / 100.0  # e.g. [0.0,0.2,0.4,0.6,0.8,1.0]

    num   = np.exp(-edges_frac[:-1] / c0) - np.exp(-edges_frac[1:] / c0)
    denom = 1.0 - np.exp(-1.0 / c0)
    w = num / max(denom, 1e-30)

    w = np.clip(w, 0.0, None)
    s = np.sum(w)
    if s > 0.0:
        w /= s
    else:
        w = np.ones_like(w) / len(w)

    tags = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]
    return {tag: float(wi) for tag, wi in zip(tags, w)}


def _get_mb_weight_array(
    cent_bins: List[Tuple[float, float]],
    glauber: OpticalGlauber,
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
) -> np.ndarray:
    """
    Helper: return normalized MB weights array w_i for each centrality bin
    (same order as cent_bins).

    Modes
    -----
    "exp"     : use make_centrality_weight_dict(cent_bins, c0=mb_c0)
    "optical" : use QF._optical_bin_weight_pA(glauber, a, b)
    "custom"  : use mb_weights_custom[tag] for tag="a-b%"
    """
    labels = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]

    if mb_weight_mode == "exp":
        w_dict = make_centrality_weight_dict(cent_bins, c0=mb_c0)
        w_arr = np.array([float(w_dict[lab]) for lab in labels], float)

    elif mb_weight_mode == "optical":
        # Check system type if available
        is_dA = False
        if hasattr(glauber, 'spec') and hasattr(glauber.spec, 'system'):
            if glauber.spec.system == 'dA':
                is_dA = True
        
        if is_dA:
            w_arr = np.array(
                [QF._optical_bin_weight_dA(glauber, a, b) for (a, b) in cent_bins],
                float
            )
        else:
            w_arr = np.array(
                [QF._optical_bin_weight_pA(glauber, a, b) for (a, b) in cent_bins],
                float
            )

    elif mb_weight_mode == "custom":
        if mb_weights_custom is None:
            raise ValueError(
                "mb_weights_custom must be provided when mb_weight_mode='custom'"
            )
        w_arr = np.array(
            [float(mb_weights_custom.get(lab, 0.0)) for lab in labels],
            float
        )

    else:
        raise ValueError(f"Unknown mb_weight_mode='{mb_weight_mode}'")

    s = float(w_arr.sum())
    if s > 0.0:
        w_arr /= s
    else:
        w_arr = np.ones_like(w_arr) / max(len(w_arr), 1)
    return w_arr


# ------------------------------------------------
# Small device helper (CPU / GPU, robust)
# ------------------------------------------------
def _qp_device(qp) -> torch.device:
    """
    Infer torch.device from QuenchParams.qp.device when possible.
    Fall back to GPU if available, else CPU.
    """
    dev_str = getattr(qp, "device", None)
    if dev_str is None:
        dev_str = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    if dev_str == "cuda" and not torch.cuda.is_available():
        dev_str = "cpu"
    if not _HAS_TORCH:
        return dev_str # Return string if torch not available
    return torch.device(dev_str)


# ------------------------------------------------
# Optional helper: scale p0 in pp spectrum
# ------------------------------------------------
def particle_with_scaled_p0(P: Particle, scale: float) -> Particle:
    """
    Return a new Particle with pp.p0 → scale * p0 (m,n unchanged).
    Used to define the Cronin (broadening) band.

    All other attributes (family, state, mass) are copied as is.
    """
    pp = P.pp
    new_pp = PPSpectrumParams(p0=pp.p0 * scale, m=pp.m, n=pp.n)
    return Particle(
        family=P.family,
        state=P.state,
        mass_override_GeV=P.mass_override_GeV,
        pp_params=new_pp,
    )


# ------------------------------------------------
# pp parametrisation: F1(pT) * F2(y,pT)
# ------------------------------------------------
def F1_t(P: Particle, pT_t: torch.Tensor, roots_GeV: float) -> torch.Tensor:
    """
    F1(p_T) = (p0^2 / (p0^2 + p_T^2))^m
    """
    pp = P.get_pp(roots_GeV)
    p0, m = pp.p0, pp.m
    p0_sq = float(p0) * float(p0)
    return (p0_sq / (p0_sq + pT_t * pT_t)) ** m


def F2_t(
    P: Particle,
    y_t: torch.Tensor,
    pT_t: torch.Tensor,
    roots_GeV: float,
) -> torch.Tensor:
    """
    F2(y,p_T) = [1 - 2 M_T cosh(y) / sqrt(s)]^n, clamped ≥ 0.
    """
    pp = P.get_pp(roots_GeV)
    n = pp.n
    M = float(P.M_GeV)
    roots = float(roots_GeV)
    pT_sq = pT_t * pT_t
    mT    = torch.sqrt(pT_sq + M * M)
    arg   = 1.0 - (2.0 * mT / roots) * torch.cosh(y_t)
    arg_clamped = torch.clamp(arg, min=1e-30)
    return arg_clamped ** n


def F2_t_pt(
    P: Particle,
    y_val: float,
    pT_t: torch.Tensor,
    roots_GeV: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    F2(y_val, p_T) for a tensor of p_T values.
    """
    if device is None:
        device = pT_t.device
    y_t = torch.full_like(pT_t, float(y_val), device=device)
    return F2_t(P, y_t, pT_t, roots_GeV)


# ------------------------------------------------
# x_A (coherence) with safety clamp
# ------------------------------------------------
def _xA_tensor(P: Particle, roots_GeV: float, qp, y_t: torch.Tensor, pT_t: torch.Tensor) -> torch.Tensor:
    """
    Vectorized x_A calculation using torch tensors.
    """
    system = getattr(qp, "system", "pA")
    M  = float(P.M_GeV)
    mT = torch.sqrt(M * M + pT_t ** 2)
    
    # Kinematic x fraction in the target (Au or Pb)
    # LHC p+Pb convention from 1212.0434 and RHIC d+Au
    x_target = (mT / float(roots_GeV)) * torch.exp(-y_t)
        
    x0 = QF.xA0_from_L(qp.LA_fm)
    
    # xA = min(x0, x_target), then clamped
    xA = torch.minimum(torch.tensor(x0, device=y_t.device, dtype=y_t.dtype), x_target)
    xA = torch.clamp(xA, min=XMIN_SAFE, max=XMAX_SAFE)
    return xA

def xA_scalar(P: Particle, roots_GeV: float, qp, y: float, pT: float) -> float:
    """
    x_A = min( x0(L_A), x_target ) with
      x_target = (m_T / sqrt(s)) e^{-y} for p+A (LHC)
      x_target = (m_T / sqrt(s)) e^{+y} for d+A (RHIC, Arleo-Peigne convention)
      x0(L_A) = ħc / (2 m_p L_A).

    Then clamped to [XMIN_SAFE, XMAX_SAFE].
    """
    system = getattr(qp, "system", "pA")
    M  = float(P.M_GeV)
    mT = math.sqrt(M * M + float(pT) ** 2)
    
    x_target = (mT / float(roots_GeV)) * math.exp(-float(y))
        
    x0 = QF.xA0_from_L(qp.LA_fm)
    xA = min(x0, x_target)
    xA = max(XMIN_SAFE, min(XMAX_SAFE, xA))
    return float(xA)


# ------------------------------------------------
# Coherent energy-loss factor R_pA^loss
# ------------------------------------------------
def R_pA_eloss(
    P: Particle,
    roots_GeV: float,
    qp,
    y,
    pT,
    Ny: int | None = None,
) -> float | np.ndarray:
    r"""
    Coherent energy-loss factor (Arleo–Peigné).
    Vectorized to accept scalar or array inputs for y and pT.
    """
    if not QF._HAS_TORCH:
        raise RuntimeError("R_pA_eloss: torch (double precision) required.")

    device = _qp_device(qp)
    M   = float(P.M_GeV)
    
    # Prepare inputs as tensors (Batch size B)
    # This automatically handles scalars by treating them as 1-element arrays
    y_in = np.atleast_1d(y)
    pT_in = np.atleast_1d(pT)
    
    # Simple broadcasting check: if sizes differ and neither is 1, it's an error
    # But usually we call this with matching sizes or one scalar.
    # For R_binned_2D vectorization, y and pT will be same-size flattened arrays.
    
    y_t  = torch.tensor(y_in, dtype=torch.float64, device=device)
    pT_t = torch.tensor(pT_in, dtype=torch.float64, device=device)
    
    # Ensure they broadcast to a common shape if needed, but for now assuming 
    # they match or broadcast naturally in element-wise ops.
    
    mT  = torch.sqrt(M*M + pT_t*pT_t)

    # 1. Determine integration limit (zmax/dym)
    # We use the maximum required range across the batch to vectorize the integration grid
    # or just a safe upper bound. The kernel handles z > zmax by returning 0 or we clamp.
    # Here, we compute dym per point.
    
    y_max_pt = torch.log(torch.clamp(roots_GeV / mT, min=1.0 + 1e-12))
    
    # Only support p+A convention for dy_max logic in current vectorization for speed?
    # Actually QF.dymax is scalar. Let's make a tensor version inline.
    # dym = max(0, min(log2, y_max - y))
    
    # Note: Arleo-Peigne usually take y > 0 for pA? Or sign matters?
    # QF.dymax uses (+y). Let's stick to that.
    
    dym_t = y_max_pt - y_t
    dym_t = torch.clamp(dym_t, min=0.0)
    dym_t = torch.minimum(dym_t, torch.tensor(math.log(2.0), device=device, dtype=torch.float64))
    
    # If all dym are tiny, return 1.0
    if (dym_t <= QF.DY_EPS).all():
        return np.ones_like(y_in, dtype=float) if y_in.size > 1 else 1.0

    # Max dym across the batch to set the integration grid
    max_dym = float(dym_t.max())
    
    if Ny is None:
        Ny = QF._Ny_from_dymax(max_dym)

    # z_max global for the grid
    zmax_global = math.expm1(max_dym)
    if zmax_global <= QF.Z_FLOOR:
        return np.ones_like(y_in, dtype=float) if y_in.size > 1 else 1.0

    # 2. Get xA for the batch
    xA = _xA_tensor(P, roots_GeV, qp, y_t, pT_t)

    with torch.no_grad():
        # F2 denominator: F2(y, pT)
        F2_den_t = F2_t(P, y_t, pT_t, roots_GeV)
        # Avoid division by zero
        F2_den_t = torch.where(F2_den_t > F2_FLOOR, F2_den_t, torch.ones_like(F2_den_t))

        mapping = getattr(qp, "mapping", "exp")

        # Integration Logic
        # We need to broadcast the integration nodes (N_nodes) against the batch (B)
        # Nodes: (N_nodes, 1)
        # Data:  (1, B)
        
        if mapping == "exp":
            umin = -30.0
            umax = math.log(max(zmax_global, 1e-300))
            u, wu = QF._gl_nodes_torch(umin, umax, Ny, device)
            
            # Reshape for broadcasting
            u  = u.unsqueeze(1)   # (Ny, 1)
            wu = wu.unsqueeze(1)  # (Ny, 1)
            
            z = torch.exp(u).clamp_min(QF.Z_FLOOR) # (Ny, 1)
            
            # Expand data for batching
            # Shape (1, B)
            mT_batch = mT.unsqueeze(0)
            xA_batch = xA.unsqueeze(0)
            y_batch  = y_t.unsqueeze(0)
            pT_batch = pT_t.unsqueeze(0)
            dym_batch= dym_t.unsqueeze(0)
            F2_den_batch = F2_den_t.unsqueeze(0)
            
            # Compute Phat: result is (Ny, B)
            # PhatA_t handles broadcasting automatically
            ph = QF.PhatA_t(z, mT_batch, xA_batch, qp, pT=pT_batch)
            
            # Mask out points where z > zmax_local (i.e. where dy > dym_local)
            # dy nodes
            dy = torch.log1p(z) # (Ny, 1)
            
            # Mask: dy <= dym_batch
            # (Ny, 1) <= (1, B)  --> (Ny, B)
            mask = (dy <= dym_batch)
            ph = ph * mask.double()

            # F2 numerator: F2(y + dy, pT)
            yshift = y_batch + dy
            
            # IMPORTANT: pT_batch must expand to match yshift for F2_t
            F2_num = F2_t(P, yshift, pT_batch.expand_as(yshift), roots_GeV)
            
            ratio = F2_num / F2_den_batch
            ratio = torch.where(torch.isfinite(ratio) & (ratio >= 0.0), ratio, torch.zeros_like(ratio))
            
            jac_z = torch.exp(u)
            inv1pz = 1.0 / (1.0 + z)
            
            # Integrands
            # Normalization integral: 1 - p0 = ∫ dz ph
            integrand_Zc = wu * jac_z * ph
            
            # Expectation integral: ∫ dz ph * [1/(1+z)] * ratio
            integrand_val = wu * jac_z * ph * inv1pz * ratio
            
            # Sum over nodes (dim 0) -> result shape (B,)
            Zc = torch.sum(integrand_Zc, dim=0)
            val = torch.sum(integrand_val, dim=0)

        else:
            # Linear mapping (fallback)
            z, wz = QF._gl_nodes_torch(0.0, float(zmax_global), Ny, device)
            z  = z.unsqueeze(1).clamp_min(QF.Z_FLOOR)
            wz = wz.unsqueeze(1)
            
            mT_batch = mT.unsqueeze(0)
            xA_batch = xA.unsqueeze(0)
            y_batch  = y_t.unsqueeze(0)
            pT_batch = pT_t.unsqueeze(0)
            dym_batch= dym_t.unsqueeze(0)
            F2_den_batch = F2_den_t.unsqueeze(0)

            ph = QF.PhatA_t(z, mT_batch, xA_batch, qp, pT=pT_batch)
            
            dy = torch.log1p(z)
            mask = (dy <= dym_batch)
            ph = ph * mask.double()
            
            yshift = y_batch + dy
            F2_num = F2_t(P, yshift, pT_batch.expand_as(yshift), roots_GeV)
            
            ratio = F2_num / F2_den_batch
            ratio = torch.where(torch.isfinite(ratio) & (ratio >= 0.0), ratio, torch.zeros_like(ratio))
            
            inv1pz = 1.0 / (1.0 + z)
            
            Zc  = torch.sum(wz * ph, dim=0)
            val = torch.sum(wz * ph * inv1pz * ratio, dim=0)

        # p0 and R_loss
        Zc = torch.clamp(Zc, min=0.0, max=1.0)
        p0 = torch.clamp(1.0 - Zc, 0.0, 1.0)
        
        R_loss = p0 + val
    
    # Return result consistent with input
    res = R_loss.detach().cpu().numpy()
    
    # If input was scalar, return scalar
    if np.isscalar(y) and np.isscalar(pT):
        return float(res[0])
        
    return res


def R_pA_broad(
    P: Particle,
    roots_GeV: float,
    qp,
    y,
    pT,
    Nphi: int = 128,
    Nk: int = 32,
) -> float | np.ndarray:
    r"""
    Cronin/broadening factor. Vectorized for speed.
    """
    if not QF._HAS_TORCH:
        raise RuntimeError("R_pA_broad: torch missing.")
    device = _qp_device(qp)

    broad_model = getattr(qp, "broadening_model", getattr(qp, "broad_model", "ring")).lower()
    use_ring = broad_model in ("ring", "hard", "fixed", "shift")

    # Inputs to Tensors
    y_in = np.atleast_1d(y)
    pT_in = np.atleast_1d(pT)
    
    y_t  = torch.tensor(y_in, dtype=torch.float64, device=device)
    pT_t = torch.tensor(pT_in, dtype=torch.float64, device=device)

    with torch.no_grad():
        # --- xA and broadening scale ---
        # Vectorized xA
        xA = _xA_tensor(P, roots_GeV, qp, y_t, pT_t)

        # Hard/ring broadening
        dpt = QF._dpt_from_xL_t(qp, xA, qp.LA_fm, hard=use_ring) # result is tensor (B,)
        dpt = torch.abs(dpt)

        # Quick check: if all dpt small, return 1
        if (dpt < 1e-10).all():
            return np.ones_like(y_in, dtype=float) if y_in.size > 1 else 1.0

        # Denominators F1, F2
        F1_den = F1_t(P, pT_t, roots_GeV)
        F1_den = torch.where(F1_den > F1_FLOOR, F1_den, torch.ones_like(F1_den))

        F2_den = F2_t(P, y_t, pT_t, roots_GeV)
        F2_den = torch.where(F2_den > F2_FLOOR, F2_den, torch.ones_like(F2_den))

        # --- phi nodes ---
        phi, wphi, cphi, sphi = QF._phi_nodes_gl_torch(Nphi, device)
        # Shapes: cphi (Nphi,)

        if use_ring:
            # Broadcast setup
            # dpt: (B,) -> (1, B)
            # pT_t:(B,) -> (1, B)
            # y_t: (B,) -> (1, B)
            # cphi:(Nphi) -> (Nphi, 1)
            
            dpt2d = dpt.unsqueeze(0)
            pT2d  = pT_t.unsqueeze(0)
            y2d   = y_t.unsqueeze(0)
            c2d   = cphi.unsqueeze(1)
            
            # pshift: (Nphi, B)
            pshift = torch.sqrt(torch.clamp(
                pT2d*pT2d + dpt2d*dpt2d - 2.0*pT2d*dpt2d*c2d, min=0.0
            ))

            # Numerators
            F1_num = F1_t(P, pshift, roots_GeV)
            
            # F2 numerator: y is constant for phi integration, but pT scans
            # We expand y to (Nphi, B)
            F2_num = F2_t(P, y2d.expand_as(pshift), pshift, roots_GeV)
            
            # Ratios
            # F1_den, F2_den are (B,), reshape to (1, B)
            R1 = F1_num / F1_den.unsqueeze(0)
            R2 = F2_num / F2_den.unsqueeze(0)
            
            R = R1 * R2
            R = torch.where(torch.isfinite(R) & (R >= 0.0), R, torch.zeros_like(R))
            
            # Angle average
            # wphi: (Nphi,), broadcast to (Nphi, 1)
            R_broad = torch.sum(R * wphi.unsqueeze(1), dim=0) # (B,)
            
        else:
            # Gaussian
            Delta = dpt * dpt # (B,)
            
            u, wu = QF._gl_nodes_torch(0.0, 1.0, Nk, device)
            u = torch.clamp(u, min=1e-300, max=1.0)
            t = -torch.log(u)
            
            # k integration grid: k^2 = Delta * t
            # k = sqrt(Delta * t)
            # Shapes: Delta (B), t (Nk) -> k (Nk, B)
            k = torch.sqrt(Delta.unsqueeze(0) * t.unsqueeze(1))
            
            # Full grid: (Nk, Nphi, B)
            # pT: (1, 1, B)
            pT3d = pT_t.unsqueeze(0).unsqueeze(0)
            y3d  = y_t.unsqueeze(0).unsqueeze(0)
            
            k3d  = k.unsqueeze(1)      # (Nk, 1, B)
            c3d  = cphi.view(1, -1, 1) # (1, Nphi, 1)
            
            pshift = torch.sqrt(torch.clamp(
                pT3d*pT3d + k3d*k3d - 2.0*pT3d*k3d*c3d, min=0.0
            ))
            
            F1_num = F1_t(P, pshift, roots_GeV)
            F2_num = F2_t(P, y3d.expand_as(pshift), pshift, roots_GeV)
            
            # Denom: (B) -> (1,1,B)
            R1 = F1_num / F1_den.view(1,1,-1)
            R2 = F2_num / F2_den.view(1,1,-1)
            
            R = R1 * R2
            R = torch.where(torch.isfinite(R) & (R >= 0.0), R, torch.zeros_like(R))
            
            # 1. Phi average (dim 1)
            # wphi: (Nphi) -> (1, Nphi, 1)
            R_phiavg = torch.sum(R * wphi.view(1, -1, 1), dim=1) # (Nk, B)
            
            # 2. k (t) average (dim 0)
            # wu: (Nk) -> (Nk, 1)
            R_broad = torch.sum(wu.unsqueeze(1) * R_phiavg, dim=0) # (B,)

    res = R_broad.detach().cpu().numpy()
    if np.isscalar(y) and np.isscalar(pT):
        return float(res[0])
    return res


def R_pA_factored(
    P: Particle,
    roots_GeV: float,
    qp,
    y: float,
    pT: float,
    Ny_eloss: int = 256,
    Nphi_broad: int = 128,
    Nk_broad: int = 32,
) -> float:
    """
    Factorised Arleo–Peigné approximation:
      R_pA ≃ R_pA^loss · R_pA^broad
    """
    Rloss  = R_pA_eloss(P, roots_GeV, qp, y, pT, Ny=Ny_eloss)
    Rbroad = R_pA_broad(P, roots_GeV, qp, y, pT, Nphi=Nphi_broad, Nk=Nk_broad)
    return Rloss * Rbroad



# ------------------------------------------------
# σ_pp weight from table or parametrisation
# ------------------------------------------------
def _sigma_pp_weight(P, roots_GeV: float, table_or_none, y: float, pT: float) -> float:
    """
    σ_pp(y,pT;√s) used as weight.
    """
    if QF._HAS_TORCH and isinstance(table_or_none, QF.TorchSigmaPPTable):
        dev = table_or_none.device
        with torch.no_grad():
            y_t = torch.tensor([y],  dtype=torch.float64, device=dev)
            p_t = torch.tensor([pT], dtype=torch.float64, device=dev)
            return float(table_or_none(y_t, p_t)[0, 0].item())
    else:
        return float(P.d2sigma_pp(float(y), float(pT), float(roots_GeV)))


# ----------------------------------------------------------------
# Generic 2D bin average
# ----------------------------------------------------------------
def R_binned_2D(
    R_func,                     # R_func(y,pT) -> float
    P, roots_GeV: float,
    y_range, pt_range,
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    table_for_pp=None,
    weight_ref_y: float | str = "local",
) -> float:
    """
    Generic bin average of R(y,pT) over y_range × pt_range.
    Vectorized implementation for speed.
    """
    # Ensure ranges are tuples even if scalars passed (fixes TypeError)
    if not isinstance(y_range, (tuple, list, np.ndarray)):
        y_range = (float(y_range), float(y_range))
    if not isinstance(pt_range, (tuple, list, np.ndarray)):
        pt_range = (float(pt_range), float(pt_range))

    yl, yr = y_range
    pl, pr = pt_range

    # Handle zero-width bins gracefully for Gaussian-Legendre
    Ny = Ny_bin if abs(yr - yl) > 1e-9 else 1
    Np = Npt_bin if abs(pr - pl) > 1e-9 else 1

    y_nodes, y_w = QF._gl_nodes_np(yl, yr, Ny)
    p_nodes, p_w = QF._gl_nodes_np(pl, pr, Np)

    if isinstance(weight_ref_y, str) and weight_ref_y.lower() == "local":
        y_ref = None
    else:
        y_ref = float(weight_ref_y)

    # VECTORIZATION START
    # Create meshgrid of nodes to call R_func once for (Ny * Np) points
    # Y shape: (Ny, Np), P shape: (Ny, Np)
    Y_mesh, P_mesh = np.meshgrid(y_nodes, p_nodes, indexing='ij')
    
    # Flatten to list of points
    y_flat = Y_mesh.ravel()
    p_flat = P_mesh.ravel()
    
    # Batched R call (assumes R_func supports arrays/tensors now)
    # The modified R_pA_eloss and R_pA_broad handle this efficiently using torch
    R_flat = R_func(y_flat, p_flat)
    
    # Reshape back to grid
    R_grid = R_flat.reshape(Ny, Np)
    
    # Calculate weights
    # y_w shape: (Ny,), p_w shape: (Np,) -> W_int shape: (Ny, Np)
    W_int = y_w[:, None] * p_w[None, :]
    
    # Physics Weights (sigma_pp)
    if weight_kind == "pp":
        # Vectorized weight calc or loops (usually fast compared to R)
        if y_ref is not None:
            # fixed y reference
            Y_for_w = np.full_like(Y_mesh, y_ref)
        else:
            Y_for_w = Y_mesh
            
        # We assume _sigma_pp_weight is scalar or can handle element-wise
        # For safety/simplicity we can create it via loop if table lookup isn't vectorized
        # But let's try strict vectorization if table available, else fallback
        
        wgt_grid = np.zeros_like(R_grid)
        
        # NOTE: If we really want max speed, we should vectorize sigma_pp too.
        # But doing this loop is O(300) cheap calls vs O(300) expensive calls.
        # This keeps compatibility with generic P.d2sigma_pp.
        for i in range(Ny):
            for j in range(Np):
                yv = Y_for_w[i,j]
                pv = P_mesh[i,j]
                dsig = _sigma_pp_weight(P, roots_GeV, table_for_pp, yv, pv)
                wgt_grid[i,j] = dsig * max(pv, 1e-8)
    else:
        wgt_grid = np.ones_like(R_grid)

    # Weighted Sum
    acc_num = np.sum(R_grid * wgt_grid * W_int)
    acc_den = np.sum(wgt_grid * W_int)

    if acc_den <= 0:
        return float(acc_num)
    return float(acc_num / acc_den)


# ------------------------------------------------
# RpA(y): binned in y, integrated over pT
# ------------------------------------------------
def rpa_binned_vs_y(
    P, roots_GeV: float, qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_edges, pt_range,
    components: Sequence[Literal["loss", "broad", "eloss_broad"]] = ("eloss_broad",),
    Ny_bin: int = 12, Npt_bin: int = 24,
    table_for_pp=None,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
    kind: Literal["pA", "AA"] = "pA",
):
    """
    RpA vs y (binned in y, integrated over pT) for each centrality bin + MB.

    MB can use:
      * mb_weight_mode="exp"     → exponential w(c) scheme (default)
      * mb_weight_mode="optical" → optical Glauber weights
      * mb_weight_mode="custom"  → user-provided mb_weights_custom[tag]
    """
    y_edges = np.asarray(y_edges, float)
    assert y_edges.ndim == 1 and y_edges.size >= 2
    Ny_bins = y_edges.size - 1
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    comps  = list(components)
    labels = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]

    # System detection for L_eff
    is_dA = getattr(glauber, "spec", None) and getattr(glauber.spec, "system", "pA") == "dA"
    
    if is_dA:
        Leff_dict = {lab: float(glauber.leff_bin_dA(a, b, method="binomial")) 
                     for lab, (a, b) in zip(labels, cent_bins)}
    else:
        L_by = glauber.leff_bins_pA(cent_bins, method="optical")
        Leff_dict = {lab: float(L_by[lab]) for lab in labels}

    # MB weights (array over cent_bins)
    w_arr_mb = _get_mb_weight_array(
        cent_bins, glauber,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    w_dict = {lab: w_arr_mb[i] for i, lab in enumerate(labels)}

    R_comp = {comp: {lab: np.zeros(Ny_bins, float) for lab in labels}
              for comp in comps}

    for i in range(Ny_bins):
        y_range = (float(y_edges[i]), float(y_edges[i+1]))

        for lab in labels:
            L   = Leff_dict[lab]
            qpL = replace(qp_base, LA_fm=float(L))

            def R_loss(y, pT, qpL=qpL):
                return R_pA_eloss(P, roots_GeV, qpL, y, pT, Ny=256)

            def R_broad(y, pT, qpL=qpL):
                return R_pA_broad(P, roots_GeV, qpL, y, pT, Nphi=128, Nk=32)

            for comp in comps:
                if comp == "loss":
                    if kind == "AA":
                         def R_func(y, pT, qpL=qpL):
                             return R_loss(y, pT, qpL=qpL) * R_loss(-y, pT, qpL=qpL)
                    else:
                         R_func = R_loss
                elif comp == "broad":
                    if kind == "AA":
                         def R_func(y, pT, qpL=qpL):
                             return R_broad(y, pT, qpL=qpL) * R_broad(-y, pT, qpL=qpL)
                    else:
                         R_func = R_broad
                elif comp == "eloss_broad":
                    def R_func(y, pT, qpL=qpL):
                        val = R_loss(y, pT, qpL=qpL) * R_broad(y, pT, qpL=qpL)
                        if kind == "AA":
                             val *= R_loss(-y, pT, qpL=qpL) * R_broad(-y, pT, qpL=qpL)
                        return val
                else:
                    raise ValueError(f"Unknown component: {comp}")

                R_bar = R_binned_2D(
                    R_func, P, roots_GeV,
                    y_range, pt_range,
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


# ------------------------------------------------
# RpA(pT): binned in pT, integrated over y
# ------------------------------------------------
def rpa_binned_vs_pT(
    P, roots_GeV: float, qp_base,
    glauber: OpticalGlauber, cent_bins,
    pT_edges, y_range,
    components: Sequence[Literal["loss", "broad", "eloss_broad"]] = ("eloss_broad",),
    Ny_bin: int = 12, Npt_bin: int = 24,
    table_for_pp=None,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
    kind: Literal["pA", "AA"] = "pA",
):
    """
    Same as rpa_binned_vs_y but swapping y ↔ pT roles.

    Returns
    -------
      pT_centers : array
      R_comp     : dict[component][tag] -> array
      labels     : list of centrality labels (no MB).
    """
    pT_edges = np.asarray(pT_edges, float)
    assert pT_edges.ndim == 1 and pT_edges.size >= 2
    Np = pT_edges.size - 1
    pT_centers = 0.5 * (pT_edges[:-1] + pT_edges[1:])

    comps  = list(components)
    labels = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]

    # System detection for L_eff
    is_dA = getattr(glauber, "spec", None) and getattr(glauber.spec, "system", "pA") == "dA"
    
    if is_dA:
        Leff_dict = {lab: float(glauber.leff_bin_dA(a, b, method="binomial")) 
                     for lab, (a, b) in zip(labels, cent_bins)}
    else:
        L_by = glauber.leff_bins_pA(cent_bins, method="optical")
        Leff_dict = {lab: float(L_by[lab]) for lab in labels}

    w_arr_mb = _get_mb_weight_array(
        cent_bins, glauber,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    w_dict = {lab: w_arr_mb[i] for i, lab in enumerate(labels)}

    R_comp = {comp: {lab: np.zeros(Np, float) for lab in labels}
              for comp in comps}

    for i in range(Np):
        pt_range = (float(pT_edges[i]), float(pT_edges[i+1]))

        for lab in labels:
            L   = Leff_dict[lab]
            qpL = replace(qp_base, LA_fm=float(L))

            def R_loss(y, pT, qpL=qpL):
                return R_pA_eloss(P, roots_GeV, qpL, y, pT, Ny=256)

            def R_broad(y, pT, qpL=qpL):
                return R_pA_broad(P, roots_GeV, qpL, y, pT, Nphi=128, Nk=32)

            for comp in comps:
                if comp == "loss":
                    if kind == "AA":
                         def R_func(y, pT, qpL=qpL):
                             return R_loss(y, pT, qpL=qpL) * R_loss(-y, pT, qpL=qpL)
                    else:
                         R_func = R_loss
                elif comp == "broad":
                    if kind == "AA":
                         def R_func(y, pT, qpL=qpL):
                             return R_broad(y, pT, qpL=qpL) * R_broad(-y, pT, qpL=qpL)
                    else:
                         R_func = R_broad
                elif comp == "eloss_broad":
                    def R_func(y, pT, qpL=qpL):
                        val = R_loss(y, pT, qpL=qpL) * R_broad(y, pT, qpL=qpL)
                        if kind == "AA":
                             val *= R_loss(-y, pT, qpL=qpL) * R_broad(-y, pT, qpL=qpL)
                        return val
                else:
                    raise ValueError(f"Unknown component: {comp}")

                R_bar = R_binned_2D(
                    R_func, P, roots_GeV,
                    y_range, pt_range,
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


# ------------------------------------------------
# Centrality dependence: RpA(cent)
# ------------------------------------------------
def rpa_vs_centrality(
    P, roots_GeV: float, qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_range, pt_range,
    component: Literal["loss", "broad", "eloss_broad"] = "eloss_broad",
    Ny_bin: int = 16, Npt_bin: int = 32,
    table_for_pp=None,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
    kind: Literal["pA", "AA"] = "pA",
):
    """
    Centrality dependence of R (pA or AA):

      <R>_bin(a–b%) = ⟨R(y,pT)⟩_{y_range × pt_range} in that bin.

    For kind='AA', L_eff uses leff_bin_AA (optical T_AA/rho0) and
    R_AA is computed as the factored approximation:
      R_AA(y,pT) ≈ R_pA(y; L) × R_pA(-y; L)    [loss or broad]
      R_AA        ≈ R_pA_loss(y)·R_pA_broad(y) × R_pA_loss(-y)·R_pA_broad(-y)

    MB is computed from per-bin values using the same centrality weights
    as other functions (mb_weight_mode).
    """
    assert component in ("loss", "broad", "eloss_broad")
    labels = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]

    # ---- L_eff dispatch: AA / dA / pA ----
    is_dA = getattr(glauber, "spec", None) and getattr(glauber.spec, "system", "pA") == "dA"
    is_AA = (kind == "AA")

    if is_AA:
        if not hasattr(glauber, "leff_bin_AA"):
            raise RuntimeError("glauber.leff_bin_AA not found — upgrade to latest glauber.py")
        Leff_dict = {lab: float(glauber.leff_bin_AA(a / 100.0, b / 100.0))
                     for lab, (a, b) in zip(labels, cent_bins)}
    elif is_dA:
        Leff_dict = {lab: float(glauber.leff_bin_dA(a, b, method="binomial"))
                     for lab, (a, b) in zip(labels, cent_bins)}
    else:
        L_by = glauber.leff_bins_pA(cent_bins, method="optical")
        Leff_dict = {lab: float(L_by[lab]) for lab in labels}

    R_vals = []

    for lab, (a, b) in zip(labels, cent_bins):
        L   = Leff_dict[lab]
        qpL = replace(qp_base, LA_fm=L)

        def R_loss(y, pT, qpL=qpL):
            return R_pA_eloss(P, roots_GeV, qpL, y, pT, Ny=256)

        def R_broad(y, pT, qpL=qpL):
            return R_pA_broad(P, roots_GeV, qpL, y, pT, Nphi=128, Nk=32)

        if component == "loss":
            if is_AA:
                def R_func(y, pT, qpL=qpL):
                    return R_loss(y, pT, qpL=qpL) * R_loss(-y, pT, qpL=qpL)
            else:
                R_func = R_loss
        elif component == "broad":
            if is_AA:
                def R_func(y, pT, qpL=qpL):
                    return R_broad(y, pT, qpL=qpL) * R_broad(-y, pT, qpL=qpL)
            else:
                R_func = R_broad
        else:  # eloss_broad
            def R_func(y, pT, qpL=qpL):
                val = R_loss(y, pT, qpL=qpL) * R_broad(y, pT, qpL=qpL)
                if is_AA:
                    val *= R_loss(-y, pT, qpL=qpL) * R_broad(-y, pT, qpL=qpL)
                return val

        R_bin = R_binned_2D(
            R_func, P, roots_GeV,
            y_range, pt_range,
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            weight_kind=weight_kind,
            table_for_pp=table_for_pp,
            weight_ref_y=weight_ref_y,
        )
        R_vals.append(R_bin)

    R_vals = np.array(R_vals)

    # MB over centralities using chosen weights
    w_bins = _get_mb_weight_array(
        cent_bins, glauber,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    R_MB = float(np.average(R_vals, weights=w_bins))

    return labels, R_vals, R_MB


# ------------------------------------------------
# Two-point bands + combination
# ------------------------------------------------
def _two_point_band(R_lo: np.ndarray, R_hi: np.ndarray, R_base: Optional[np.ndarray] = None):
    """
    Given arrays R(min) and R(max), return
    Rc, Rlow, Rhigh.
    If R_base is provided, it is used as the central value (asymmetric band).
    Otherwise, we return a symmetric band centered on the average.
    """
    if R_base is not None:
        Rc = R_base
        # In case the scan is not monotonic, we take the envelope
        R_min = np.minimum(R_lo, R_hi)
        R_max = np.maximum(R_lo, R_hi)
        return Rc, np.minimum(Rc, R_min), np.maximum(Rc, R_max)
    
    Rc = 0.5 * (R_lo + R_hi)
    dR = 0.5 * np.abs(R_hi - R_lo)
    return Rc, Rc - dR, Rc + dR


def combine_factorized_bands_1d(
    RL_c, RL_lo, RL_hi,
    RB_c, RB_lo, RB_hi,
):
    """
    Combine loss & broad bands into eloss_broad, assuming factorisation:
      R_tot = R_L * R_B
    Propagates asymmetric uncertainties correctly.
    """
    RT_c, RT_lo, RT_hi = {}, {}, {}
    for lab in RL_c.keys():
        Lc  = np.asarray(RL_c[lab])
        Llo = np.asarray(RL_lo[lab])
        Lhi = np.asarray(RL_hi[lab])

        Bc  = np.asarray(RB_c[lab])
        Blo = np.asarray(RB_lo[lab])
        Bhi = np.asarray(RB_hi[lab])

        # Relative deviations
        Lc_safe = np.where(np.abs(Lc) > 1e-12, Lc, np.nan)
        Bc_safe = np.where(np.abs(Bc) > 1e-12, Bc, np.nan)

        rel_lo_L = (Lc - Llo) / Lc_safe
        rel_lo_B = (Bc - Blo) / Bc_safe
        rel_lo_tot = np.sqrt(np.maximum(0, rel_lo_L**2 + rel_lo_B**2))

        rel_hi_L = (Lhi - Lc) / Lc_safe
        rel_hi_B = (Bhi - Bc) / Bc_safe
        rel_hi_tot = np.sqrt(np.maximum(0, rel_hi_L**2 + rel_hi_B**2))

        Rc = Lc * Bc
        RT_c[lab] = Rc
        RT_lo[lab] = Rc * (1.0 - rel_lo_tot)
        RT_hi[lab] = Rc * (1.0 + rel_hi_tot)

    return RT_c, RT_lo, RT_hi


# ------------------------------------------------
# Bands vs y
# ------------------------------------------------
def rpa_band_vs_y_eloss(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_edges, pt_range,
    q0_pair=(0.05, 0.09),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
    kind: Literal["pA", "AA"] = "pA",
):
    """
    Binned R_pA^loss(y) band from q0_pair.

    Returns
    -------
      y_cent
      RL_c[lab], RL_lo[lab], RL_hi[lab]   (lab includes "MB")
      labels   (centrality labels)
    """
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
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
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
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
    )
    assert np.allclose(y_cent_lo, y_cent_hi)
    y_cent = y_cent_lo

    RL_c, RL_lo, RL_hi = {}, {}, {}
    for lab in R_lo["loss"].keys():   # cent bins + "MB"
        Rc, Rl, Rh = _two_point_band(R_lo["loss"][lab],
                                     R_hi["loss"][lab])
        RL_c[lab], RL_lo[lab], RL_hi[lab] = Rc, Rl, Rh

    return y_cent, RL_c, RL_lo, RL_hi, labels


def rpa_band_vs_y_broad(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_edges, pt_range,
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
    kind: Literal["pA", "AA"] = "pA",
):
    """
    Binned R_pA^broad(y) band from scaling p0 in the pp spectrum.

    Returns
    -------
      y_cent
      RB_c[lab], RB_lo[lab], RB_hi[lab]
      labels
    """
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
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
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
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
    )
    assert np.allclose(y_cent_lo, y_cent_hi)
    y_cent = y_cent_lo

    RB_c, RB_lo, RB_hi = {}, {}, {}
    for lab in R_lo["broad"].keys():
        Rc, Rl, Rh = _two_point_band(R_lo["broad"][lab],
                                     R_hi["broad"][lab])
        RB_c[lab], RB_lo[lab], RB_hi[lab] = Rc, Rl, Rh

    return y_cent, RB_c, RB_lo, RB_hi, labels


def rpa_band_vs_y(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_edges, pt_range,
    components=("loss", "broad", "eloss_broad"),
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
    kind: Literal["pA", "AA"] = "pA",
):
    """
    Full RpA band vs y:

      • eloss band from q0_pair
      • broad band from p0_scale_pair
      • eloss_broad band from factorised combination in quadrature.
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
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
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
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
    )
    assert np.allclose(y_cent, y_cent2)

    # ---- eloss_broad band from loss ⊗ broad (quadrature) ----
    RT_c, RT_lo, RT_hi = combine_factorized_bands_1d(
        RL_c, RL_lo, RL_hi,
        RB_c, RB_lo, RB_hi,
    )

    bands: dict[str, tuple[dict, dict, dict]] = {}
    if "loss" in components:
        bands["loss"] = (RL_c, RL_lo, RL_hi)
    if "broad" in components:
        bands["broad"] = (RB_c, RB_lo, RB_hi)
    if "eloss_broad" in components:
        bands["eloss_broad"] = (RT_c, RT_lo, RT_hi)

    return y_cent, bands, labels


# ------------------------------------------------
# Bands vs pT
# ------------------------------------------------
def rpa_band_vs_pT_eloss(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    pT_edges, y_range,
    q0_pair=(0.05, 0.09),
    component="loss",
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
    kind: Literal["pA", "AA"] = "pA",
    table_for_pp=None,
):
    assert component in ("loss", "eloss_broad")

    q0_lo, q0_hi = q0_pair
    qp_lo = replace(qp_base, qhat0=float(q0_lo))
    qp_hi = replace(qp_base, qhat0=float(q0_hi))

    pT_cent_lo, R_lo, labels = rpa_binned_vs_pT(
        P, roots_GeV, qp_lo,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
    )
    pT_cent_hi, R_hi, _ = rpa_binned_vs_pT(
        P, roots_GeV, qp_hi,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
    )
    assert np.allclose(pT_cent_lo, pT_cent_hi)
    pT_cent = pT_cent_lo

    RL_c, RL_lo, RL_hi = {}, {}, {}
    for lab in R_lo[component].keys():   # cent bins + "MB"
        Rc, Rl, Rh = _two_point_band(R_lo[component][lab],
                                     R_hi[component][lab])
        RL_c[lab], RL_lo[lab], RL_hi[lab] = Rc, Rl, Rh

    return pT_cent, RL_c, RL_lo, RL_hi, labels


def rpa_band_vs_pT_broad(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    pT_edges, y_range,
    p0_scale_pair=(0.9, 1.1),
    component="broad",
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
    table_for_pp=None,
    kind: Literal["pA", "AA"] = "pA",
):
    assert component in ("broad", "eloss_broad")

    P_lo = particle_with_scaled_p0(P, p0_scale_pair[0])
    P_hi = particle_with_scaled_p0(P, p0_scale_pair[1])

    pT_cent_lo, R_lo, labels = rpa_binned_vs_pT(
        P_lo, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
    )
    pT_cent_hi, R_hi, _ = rpa_binned_vs_pT(
        P_hi, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
    )
    assert np.allclose(pT_cent_lo, pT_cent_hi)
    pT_cent = pT_cent_lo

    RB_c, RB_lo, RB_hi = {}, {}, {}
    for lab in R_lo[component].keys():   # cent bins + "MB"
        Rc, Rl, Rh = _two_point_band(R_lo[component][lab],
                                     R_hi[component][lab])
        RB_c[lab], RB_lo[lab], RB_hi[lab] = Rc, Rl, Rh

    return pT_cent, RB_c, RB_lo, RB_hi, labels


def rpa_band_vs_pT(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    pT_edges, y_range,
    components=("loss", "broad", "eloss_broad"),
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    table_for_pp=None,
    mb_weights_custom: Dict[str, float] | None = None,
    kind: Literal["pA", "AA"] = "pA",
):
    """
    Full RpA band vs pT (y-integrated).

      • eloss band from q0_pair
      • broad band from p0_scale_pair
      • eloss_broad band from factorised combination in quadrature.
    """
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
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
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
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
        kind=kind,
    )
    assert np.allclose(pT_cent, pT_cent2)

    # eloss_broad
    RT_c, RT_lo, RT_hi = combine_factorized_bands_1d(
        RL_c, RL_lo, RL_hi,
        RB_c, RB_lo, RB_hi,
    )

    bands: dict[str, tuple[dict, dict, dict]] = {}
    if "loss" in components:
        bands["loss"] = (RL_c, RL_lo, RL_hi)
    if "broad" in components:
        bands["broad"] = (RB_c, RB_lo, RB_hi)
    if "eloss_broad" in components:
        bands["eloss_broad"] = (RT_c, RT_lo, RT_hi)

    return pT_cent, bands, labels


# ------------------------------------------------
# Bands vs centrality
# ------------------------------------------------
def rpa_band_vs_centrality(
    P, roots_GeV: float, qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_range, pt_range,
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 16, Npt_bin: int = 32,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    table_for_pp=None,
    mb_weights_custom: Dict[str, float] | None = None,
    kind: Literal["pA", "AA"] = "pA",
):
    """
    Error bands vs centrality:

      • q0_pair       → energy-loss band
      • p0_scale_pair → pp(p0) scale band for Cronin (broadening)
      • eloss_broad band    → quadrature combination of eloss + broad.

    We do NOT treat the (0,100) "MB bin" as a separate optical bin here;
    MB is computed from chosen centrality weights over the genuine bins.
    """
    # Drop any explicit 0-100 bin from the core averaging:
    core_bins = [b for b in cent_bins if not (b[0] == 0 and b[1] == 100)]
    labels    = [f"{int(a)}-{int(b)}%" for (a, b) in core_bins]

    # ---------- loss band (q0 scan) ----------
    q0_lo, q0_hi = q0_pair
    RL_lo, RL_hi, RL_c = {}, {}, {}

    for q0, store in [(q0_lo, RL_lo), (q0_hi, RL_hi)]:
        qp_q = replace(qp_base, qhat0=float(q0))
        _, Rvals_q, _ = rpa_vs_centrality(
            P, roots_GeV, qp_q, glauber, core_bins,
            y_range, pt_range,
            component="loss",
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            table_for_pp=table_for_pp,
            weight_kind=weight_kind,
            weight_ref_y=weight_ref_y,
            mb_weight_mode=mb_weight_mode,
            mb_c0=mb_c0,
            mb_weights_custom=mb_weights_custom,
            kind=kind,
        )
        for lab, val in zip(labels, Rvals_q):
            store[lab] = val

    RL_base = {}
    _, Rvals_base, _ = rpa_vs_centrality(
        P, roots_GeV, qp_base, glauber, core_bins,
        y_range, pt_range,
        component="loss",
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind, weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode, mb_c0=mb_c0, mb_weights_custom=mb_weights_custom,
        kind=kind,
    )
    for lab, val in zip(labels, Rvals_base):
        RL_base[lab] = val

    for lab in labels:
        rc, rl, rh = _two_point_band(RL_lo[lab], RL_hi[lab], RL_base[lab])
        RL_c[lab], RL_lo[lab], RL_hi[lab] = rc, rl, rh

    # ---------- broad band (p0 scan) ----------
    RB_lo, RB_hi, RB_c = {}, {}, {}

    for p0_scale, store in [(p0_scale_pair[0], RB_lo),
                            (p0_scale_pair[1], RB_hi)]:
        P_scaled = particle_with_scaled_p0(P, p0_scale)
        _, Rvals_q, _ = rpa_vs_centrality(
            P_scaled, roots_GeV, qp_base, glauber, core_bins,
            y_range, pt_range,
            component="broad",
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            table_for_pp=table_for_pp,
            weight_kind=weight_kind,
            weight_ref_y=weight_ref_y,
            mb_weight_mode=mb_weight_mode,
            mb_c0=mb_c0,
            mb_weights_custom=mb_weights_custom,
            kind=kind,
        )
        for lab, val in zip(labels, Rvals_q):
            store[lab] = val

    RB_base = {}
    _, Rvals_base, _ = rpa_vs_centrality(
        P, roots_GeV, qp_base, glauber, core_bins,
        y_range, pt_range,
        component="broad",
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind, weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode, mb_c0=mb_c0, mb_weights_custom=mb_weights_custom,
        kind=kind,
    )
    for lab, val in zip(labels, Rvals_base):
        RB_base[lab] = val

    for lab in labels:
        rc, rl, rh = _two_point_band(RB_lo[lab], RB_hi[lab], RB_base[lab])
        RB_c[lab], RB_lo[lab], RB_hi[lab] = rc, rl, rh

    # ---------- combine loss + broad in quadrature ----------
    RT_c, RT_lo, RT_hi = {}, {}, {}
    for lab in labels:
        Lc, Llo, Lhi = RL_c[lab], RL_lo[lab], RL_hi[lab]
        Bc, Blo, Bhi = RB_c[lab], RB_lo[lab], RB_hi[lab]

        dL = 0.5 * abs(Lhi - Llo)
        dB = 0.5 * abs(Bhi - Blo)

        Lc_safe = Lc if abs(Lc) > 1e-10 else 1.0
        Bc_safe = Bc if abs(Bc) > 1e-10 else 1.0

        Rc   = Lc * Bc
        rel2 = (dL / Lc_safe) ** 2 + (dB / Bc_safe) ** 2
        dR   = Rc * math.sqrt(rel2)

        RT_c[lab]  = Rc
        RT_lo[lab] = Rc - dR
        RT_hi[lab] = Rc + dR

    # ---------- MB values (centrality weights over core bins only) ----------
    w_bins = _get_mb_weight_array(
        core_bins, glauber,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )

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

    # eloss_broad MB via factorised combination in quadrature
    RcL_MB, RloL_MB, RhiL_MB = RMB_loss
    RcB_MB, RloB_MB, RhiB_MB = RMB_broad
    dL_MB = 0.5 * abs(RhiL_MB - RloL_MB)
    dB_MB = 0.5 * abs(RhiB_MB - RloB_MB)
    Rc_MB = RcL_MB * RcB_MB
    rel2_MB = (dL_MB / max(abs(RcL_MB), 1e-12)) ** 2 + (dB_MB / max(abs(RcB_MB), 1e-12)) ** 2
    dR_MB = Rc_MB * math.sqrt(rel2_MB)
    RMB_tot = (Rc_MB, Rc_MB - dR_MB, Rc_MB + dR_MB)

    return (labels,
            RL_c, RL_lo, RL_hi,
            RB_c, RB_lo, RB_hi,
            RT_c, RT_lo, RT_hi,
            RMB_loss, RMB_broad, RMB_tot)


# ============================================================
# Plot helpers (optional)
# ============================================================
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _step_from_centers(x_cent, vals):
    """
    Given bin centers x_cent and values vals (same length, uniform spacing),
    build (x_edges, y_step) so that

        plt.step(x_edges, y_step, where="post")

    gives a flat segment per bin.

    Assumes constant bin width.
    """
    x_cent = np.asarray(x_cent, float)
    vals   = np.asarray(vals, float)
    assert x_cent.size == vals.size

    if x_cent.size > 1:
        dx = np.diff(x_cent)
        dx0 = dx[0]
        if not np.allclose(dx, dx0):
            raise ValueError("x_cent not uniformly spaced – can't stepify safely.")
    else:
        # single bin; choose arbitrary width 1
        dx0 = 1.0

    x_edges = np.concatenate(([x_cent[0] - 0.5 * dx0],
                              x_cent + 0.5 * dx0))
    y_step  = np.concatenate([vals, vals[-1:]])
    return x_edges, y_step


def centrality_step_arrays(cent_bins, vals):
    """
    Build step-plot arrays from centrality bins [(0,20), ...] and
    values array of length len(cent_bins).

    Returns
    -------
    x_edges : array, shape (Nbins+1,)
    y_step  : array, shape (Nbins+1,)
    """
    vals = np.asarray(vals, float)
    assert len(vals) == len(cent_bins)

    edges = [cent_bins[0][0]] + [b for (_, b) in cent_bins]
    x_edges = np.array(edges, float)          # e.g. [0,20,40,60,80,100]
    y_step  = np.concatenate([vals, vals[-1:]])
    return x_edges, y_step


def plot_RpA_vs_y_components_per_centrality(
    P, roots_GeV, qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_edges, pt_range,
    show_components=("loss", "broad", "eloss_broad"),
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    include_MB: bool = True,
    ncols: int = 3,
    step: bool = True,
    suptitle: str | None = None,
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
    extra_bands: Dict[str, Tuple[Dict, Dict, Dict]] | None = None,
    table_for_pp=None,
):
    """
    Make a grid of subplots, one per centrality bin (+ optional MB),
    with different components (loss, broad, eloss_broad, plus optional extra) shown as
    different colours + legend entries.

    extra_bands: optional dict {'custom_label': (Rc_dict, Rlo_dict, Rhi_dict), ...}
    """
    # Nomenclature mapping (internal show_components <-> extra_bands keys)
    comp_map = {
        "loss": "eloss",
        "broad": "broad",
        "eloss_broad": "eloss_broad",
    }

    needed_eloss = []
    if extra_bands:
        for c in show_components:
            mapped_c = comp_map.get(c, c)
            if mapped_c in ("eloss", "broad", "eloss_broad") and mapped_c not in extra_bands:
                needed_eloss.append(mapped_c)
    else:
        for c in show_components:
            mapped_c = comp_map.get(c, c)
            if mapped_c in ("eloss", "broad", "eloss_broad"):
                needed_eloss.append(mapped_c)

    if needed_eloss:
        y_cent, bands, labels = rpa_band_vs_y(
            P, roots_GeV, qp_base,
            glauber, cent_bins,
            y_edges, pt_range,
            components=tuple(sorted(set(needed_eloss))),
            q0_pair=q0_pair,
            p0_scale_pair=p0_scale_pair,
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            weight_kind=weight_kind,
            weight_ref_y=weight_ref_y,
            table_for_pp=table_for_pp,
            mb_weight_mode=mb_weight_mode,
            mb_c0=mb_c0,
            mb_weights_custom=mb_weights_custom,
        )
    else:
        y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
        bands = {}
        tags = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins if (a, b) != (0, 100)]
        labels = tags # fallback

    # Merge in extra bands if provided
    if extra_bands:
        for k, v in extra_bands.items():
            bands[k] = v

    # Which centrality tags to show as panels
    cent_tags = [f"{a}-{b}%" for (a, b) in cent_bins if (a, b) != (0, 100)]
    if include_MB:
        # add MB if present in bands
        any_comp = next(iter(bands.values()))
        Rc_dict_any = any_comp[0]
        if "MB" in Rc_dict_any:
            cent_tags.append("MB")

    # Colours & labels per component (consistent across panels)
    comp_colors = {
        "eloss": "C0",
        "broad": "C1",
        "eloss_broad": "C3",
    }
    comp_labels = {
        "eloss": r"ELoss only",
        "broad": r"Broadening only",
        "eloss_broad": r"Total ($R_{\mathrm{loss}} \times R_{\mathrm{broad}}$)",
    }

    # Common pT-range note
    try:
        p1, p2 = pt_range
        note = rf"$p_T\in[{p1:.1f},{p2:.1f}]$ GeV"
    except:
        note = rf"$p_T \approx {pt_range:.1f}$ GeV"

    #set_publication_style()
    
    # Figure / axes
    n_panels = len(cent_tags)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 3.4 * nrows),
        dpi=140,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for iax, (ax, tag) in enumerate(zip(axes, cent_tags)):
        for comp in show_components:
            if comp not in bands:
                continue
            
            # Robust unpacking to avoid ValueError
            data_item = bands[comp]
            if isinstance(data_item, dict):
                # Format: dict[tag] -> (c, lo, hi)
                if tag not in data_item: continue
                Rc_arr, Rlo_arr, Rhi_arr = data_item[tag]
            else:
                # Format: (Rc_dict, Rlo_dict, Rhi_dict)
                Rc_dict, Rlo_dict, Rhi_dict = data_item
                if tag not in Rc_dict: continue
                Rc_arr, Rlo_arr, Rhi_arr = Rc_dict[tag], Rlo_dict[tag], Rhi_dict[tag]
            
            Rc  = np.asarray(Rc_arr)
            Rlo = np.asarray(Rlo_arr)
            Rhi = np.asarray(Rhi_arr)

            col   = comp_colors.get(comp, "k")
            label = comp_labels.get(comp, comp) if iax == 0 else None

            if step:
                x_edges, y_c  = _step_from_centers(y_cent, Rc)
                _,       y_lo = _step_from_centers(y_cent, Rlo)
                _,       y_hi = _step_from_centers(y_cent, Rhi)

                ax.step(x_edges, y_c, where="post",
                        color=col, lw=1.6, label=label)
                ax.fill_between(
                    x_edges, y_lo, y_hi,
                    step="post", color=col, alpha=0.25, linewidth=0.0
                )
            else:
                ax.plot(y_cent, Rc, color=col, lw=1.6, label=label)
                ax.fill_between(
                    y_cent, Rlo, Rhi,
                    color=col, alpha=0.25, linewidth=0.0
                )

        # Horizontal R=1 line
        ax.axhline(1.0, color="k", ls=":", lw=0.8)

        if tag == "MB":
            ax.set_title("Min-Bias", loc="right", fontsize=10, fontweight="bold")
        else:
            ax.set_title(tag, loc="right", fontsize=10, fontweight="bold")

        ax.set_xlabel(r"$y$")
        ax.grid(False)

        # Note inside each panel
        ax.text(
            0.03, 0.97, note,
            transform=ax.transAxes,
            fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

        if iax == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower left")

        if hasattr(ax, "tick_params"):
            ax.tick_params(direction="in", top=True, right=True, which="both")
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Consistency
    for ax in axes:
        ax.set_ylim(0.0, 2.0)
    
    if roots_GeV < 500: # RHIC
        plt.xlim(-3.0, 3.0)
    else: # LHC
        plt.xlim(-5.0, 5.0)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)

    plt.tight_layout()
    return fig, axes


def plot_RpA_vs_pT_components_per_centrality(
    P, roots_GeV, qp_base,
    glauber: OpticalGlauber, cent_bins,
    pT_edges, y_range,
    show_components=("loss", "broad", "eloss_broad"),
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    include_MB: bool = True,
    ncols: int = 3,
    step: bool = True,
    suptitle: str | None = None,
    ylabel: str = r"$R_{pA}(p_T)$",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
    table_for_pp=None,
    extra_bands: Dict[str, Tuple[Dict, Dict, Dict]] | None = None,
):
    """
    Grid of subplots: one panel per centrality bin (+ optional MB),
    curves = components (loss, broad, eloss_broad, plus optional extra) vs pT.

    extra_bands: optional dict {'custom_label': (Rc_dict, Rlo_dict, Rhi_dict), ...}
    """
    comp_map = {
        "loss": "eloss",
        "broad": "broad",
        "eloss_broad": "eloss_broad",
    }

    needed_eloss = []
    if extra_bands:
        for c in show_components:
            mapped_c = comp_map.get(c, c)
            if mapped_c in ("eloss", "broad", "eloss_broad") and mapped_c not in extra_bands:
                needed_eloss.append(mapped_c)
    else:
        for c in show_components:
            mapped_c = comp_map.get(c, c)
            if mapped_c in ("eloss", "broad", "eloss_broad"):
                needed_eloss.append(mapped_c)

    if needed_eloss:
        pT_cent, bands, labels = rpa_band_vs_pT(
            P, roots_GeV, qp_base,
            glauber, cent_bins,
            pT_edges, y_range,
            components=tuple(sorted(set(needed_eloss))),
            q0_pair=q0_pair,
            p0_scale_pair=p0_scale_pair,
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            weight_kind=weight_kind,
            weight_ref_y=weight_ref_y,
            mb_weight_mode=mb_weight_mode,
            table_for_pp=table_for_pp,
            mb_c0=mb_c0,
            mb_weights_custom=mb_weights_custom,
        )
    else:
        pT_cent = 0.5 * (pT_edges[:-1] + pT_edges[1:])
        bands = {}
        tags = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins if (a, b) != (0, 100)]
        labels = tags

    if extra_bands:
        for k, v in extra_bands.items():
            bands[k] = v

    # Panels: centrality tags (+ MB)
    cent_tags = [f"{a}-{b}%" for (a, b) in cent_bins if (a, b) != (0, 100)]
    if include_MB:
        any_comp = next(iter(bands.values()))
        Rc_dict_any = any_comp[0]
        if "MB" in Rc_dict_any:
            cent_tags.append("MB")

    # Colours & labels per component (consistent with y-plots)
    comp_colors = {
        "eloss": "C0",
        "broad": "C1",
        "eloss_broad": "C3",
    }
    comp_labels = {
        "eloss": r"ELoss only",
        "broad": r"Broadening only",
        "eloss_broad": r"Total ($R_{\mathrm{loss}} \times R_{\mathrm{broad}}$)",
    }

    # Note inside each panel: y-range + pT-range
    try:
        y1, y2 = y_range
        y_note = rf"${y1:.2f}<y<{y2:.2f}$"
    except:
        y_note = rf"$y \approx {y_range:.2f}$"
    
    note = (
        y_note + "\n"
        rf"$p_T\in[{pT_edges[0]:.1f},{pT_edges[-1]:.1f}]$ GeV"
    )

    set_publication_style()
    
    # Figure / axes layout
    n_panels = len(cent_tags)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 3.4 * nrows),
        dpi=140,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for iax, (ax, tag) in enumerate(zip(axes, cent_tags)):
        for comp in show_components:
            if comp not in bands: continue
            
            # Robust unpacking to avoid ValueError
            data_item = bands[comp]
            if isinstance(data_item, dict):
                # Format: dict[tag] -> (c, lo, hi)
                if tag not in data_item: continue
                Rc_arr, Rlo_arr, Rhi_arr = data_item[tag]
            else:
                # Format: (Rc_dict, Rlo_dict, Rhi_dict)
                Rc_dict, Rlo_dict, Rhi_dict = data_item
                if tag not in Rc_dict: continue
                Rc_arr, Rlo_arr, Rhi_arr = Rc_dict[tag], Rlo_dict[tag], Rhi_dict[tag]

            Rc  = np.asarray(Rc_arr)
            Rlo = np.asarray(Rlo_arr)
            Rhi = np.asarray(Rhi_arr)

            col   = comp_colors.get(comp, "k")
            # Only first panel gets legend labels
            label = comp_labels.get(comp, comp) if iax == 0 else None

            if step:
                x_edges, y_c  = _step_from_centers(pT_cent, Rc)
                _,       y_lo = _step_from_centers(pT_cent, Rlo)
                _,       y_hi = _step_from_centers(pT_cent, Rhi)

                ax.step(x_edges, y_c, where="post",
                        lw=1.6, color=col, label=label)
                ax.fill_between(
                    x_edges, y_lo, y_hi,
                    step="post", alpha=0.25, color=col, linewidth=0.0
                )
            else:
                ax.plot(pT_cent, Rc, lw=1.6, color=col, label=label)
                ax.fill_between(
                    pT_cent, Rlo, Rhi,
                    alpha=0.25, color=col, linewidth=0.0
                )

        # R=1 reference line
        ax.axhline(1.0, color="k", ls=":", lw=0.8)

        # Panel title = centrality / MB
        title_right = "Min-Bias" if tag == "MB" else tag
        ax.set_title(title_right, loc="right", fontsize=10, fontweight="bold")
        ax.set_title(y_note, loc="left", fontsize=10, color="navy")

        # Left column gets y-label
        if iax % ncols == 0:
            ax.set_ylabel(ylabel)

        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_xlim(pT_edges[0], pT_edges[-1])
        ax.grid(False)

        # Note inside panel
        ax.text(
            0.03, 0.97, note,
            transform=ax.transAxes,
            fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

        if iax == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower left")

        if hasattr(ax, "tick_params"):
            ax.tick_params(direction="in", top=True, right=True, which="both")
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Remove any unused axes
    for j in range(n_panels, len(axes)):
        fig.delaxes(axes[j])

    # Consistency
    for ax in axes:
        ax.set_ylim(0.0, 2.0)
    
    # x-axis for PT
    if roots_GeV < 500: # RHIC
        plt.xlim(0.0, 10.0)
    else: # LHC
        plt.xlim(0.0, 15.0)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)

    plt.tight_layout()
    return fig, axes


def plot_RpA_vs_centrality_components_band(
    cent_bins, labels,
    RL_c=None, RL_lo=None, RL_hi=None, RMB_loss=None,
    RB_c=None, RB_lo=None, RB_hi=None, RMB_broad=None,
    RT_c=None, RT_lo=None, RT_hi=None, RMB_tot=None,
    show=("eloss_broad",),                  # e.g. ("loss","broad","eloss_broad")
    ax=None,
    ylabel=r"$R_{pA}(\mathrm{cent})$",
    note: str | None = None,
    system_label: str | None = None,  # e.g. r"$5.02$ TeV p+Pb"
):
    """
    Step-style RpA vs centrality, with optional bands for
    loss, broad, and eloss_broad components, plus MB horizontal band.

    Parameters
    ----------
    cent_bins : list of (a,b) centrality edges.
    labels    : list of matching strings "a-b%".
    RL_*      : dict[lab] -> scalar, loss band (central, low, high).
    RB_*      : dict[lab] -> scalar, broad band.
    RT_*      : dict[lab] -> scalar, eloss_broad band.
    RMB_*     : (Rc_MB, Rlo_MB, Rhi_MB) tuples per component.
    system_label : string appended to legend label for "eloss_broad".
    """
    set_publication_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=140)
    else:
        fig = ax.figure

    # colours & labels per component
    comp_color = {
        "loss":  "C0",
        "broad": "C1",
        "eloss_broad": "C3",
    }
    comp_label = {
        "loss":  r"eloss",
        "broad": r"$p_T$ broadening",
        "eloss_broad": r"eloss $\times$ $p_T$ broadening",
    }
    if system_label is not None:
        comp_label["eloss_broad"] = system_label  # e.g. "5.02 TeV p+Pb"

    # helper to step-plot one component
    def _plot_comp(comp, data_or_dict):
        if isinstance(data_or_dict, dict):
            # Format: {tag: (c, lo, hi)}
            Cc = {k: v[0] for k, v in data_or_dict.items()}
            Clo= {k: v[1] for k, v in data_or_dict.items()}
            Chi= {k: v[2] for k, v in data_or_dict.items()}
        else:
            Cc, Clo, Chi = data_or_dict

        vals_c  = np.array([Cc[lab]  for lab in labels])
        vals_lo = np.array([Clo[lab] for lab in labels])
        vals_hi = np.array([Chi[lab] for lab in labels])

        x_edges, y_c  = centrality_step_arrays(cent_bins, vals_c)
        _,       y_lo = centrality_step_arrays(cent_bins, vals_lo)
        _,       y_hi = centrality_step_arrays(cent_bins, vals_hi)

        col = comp_color[comp]
        lab = comp_label[comp]

        ax.step(x_edges, y_c, where="post", lw=2.0, color=col, label=lab)
        ax.fill_between(x_edges, y_lo, y_hi,
                        step="post", alpha=0.25, color=col, linewidth=0.0)

    # loss
    if "loss" in show and RL_c is not None:
        _plot_comp("loss", RL_c if isinstance(RL_c, dict) else (RL_c, RL_lo, RL_hi))

        if RMB_loss is not None:
            Rc_MB, Rlo_MB, Rhi_MB = RMB_loss
            x_band = np.array([cent_bins[0][0], cent_bins[-1][1]], float)
            ax.hlines(Rc_MB, x_band[0], x_band[1],
                      colors=comp_color["loss"], linestyles="--",
                      linewidth=1.2, label=r"MB loss")
            ax.fill_between(
                x_band,
                [Rlo_MB, Rlo_MB],
                [Rhi_MB, Rhi_MB],
                color=comp_color["loss"], alpha=0.12, linewidth=0.0,
            )

    # broad
    if "broad" in show and RB_c is not None:
        _plot_comp("broad", RB_c if isinstance(RB_c, dict) else (RB_c, RB_lo, RB_hi))

        if RMB_broad is not None:
            Rc_MB, Rlo_MB, Rhi_MB = RMB_broad
            x_band = np.array([cent_bins[0][0], cent_bins[-1][1]], float)
            ax.hlines(Rc_MB, x_band[0], x_band[1],
                      colors=comp_color["broad"], linestyles="--",
                      linewidth=1.2, label=r"MB broad")
            ax.fill_between(
                x_band,
                [Rlo_MB, Rlo_MB],
                [Rhi_MB, Rhi_MB],
                color=comp_color["broad"], alpha=0.12, linewidth=0.0,
            )

    # eloss_broad (this is usually the main one, with darker MB)
    if "eloss_broad" in show and RT_c is not None:
        _plot_comp("eloss_broad", RT_c if isinstance(RT_c, dict) else (RT_c, RT_lo, RT_hi))

        if RMB_tot is not None:
            Rc_MB, Rlo_MB, Rhi_MB = RMB_tot
            x_band = np.array([cent_bins[0][0], cent_bins[-1][1]], float)
            ax.hlines(Rc_MB, x_band[0], x_band[1],
                      colors=comp_color["eloss_broad"], linestyles="--",
                      linewidth=1.6, label=r"MB eloss_broad (eloss$\times$broad)")
            ax.fill_between(
                x_band,
                [Rlo_MB, Rlo_MB],
                [Rhi_MB, Rhi_MB],
                color="gray", alpha=0.15, linewidth=0.0,
            )

    pass


    pass


    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("centrality [%]")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 2.0)
    ax.set_xlim(cent_bins[0][0], cent_bins[-1][1])
    ax.grid(False)
    
    # After plotting everything:
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    if note is not None:
        ax.text(
            0.03, 0.97, note,
            transform=ax.transAxes,
            fontsize=9,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

    return fig, ax

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# THE "FRONT DOOR": compute_eloss_cronin
# ------------------------------------------------------------------------------
def compute_eloss_cronin(
    system: str,
    roots_GeV: float,
    particle: Particle,
    y_regions: Dict[str, Tuple[float, float] | float],
    pT_grid: np.ndarray,
    centrality_bins: List[Tuple[float, float]],
    *,
    leff_mode: Literal["glauber", "AP"] = "glauber",
    qhat0: float = 0.075,
    lp_fm: float = 1.5,
    band_spec: Optional[Dict] = None,
    return_components: bool = True,
    device: str | None = None,
    glauber_custom: OpticalGlauber | None = None,
) -> Dict:
    """
    A stable unified interface to compute CNM eloss + broadening.
    Supports both pA (LHC) and dA (RHIC) systems.
    
    Returns
    -------
    results : dict
        Structured as:
        {
            "meta": {...},
            "pT": pT_grid,
            "y_regions": {
                label: {
                    "cent_labels": [...],
                    "MinBias": { "eloss_broad": [...], "loss": [...], "broad": [...] },
                    ...
                }
            }
        }
    """
    from copy import deepcopy
    
    # 1. Setup Master QuenchParams
    qp_master = QF.QuenchParams(
        qhat0=qhat0,
        lp_fm=lp_fm,
        roots_GeV=roots_GeV,
        system=system,
        device=device or ("cuda" if (QF._HAS_TORCH and torch.cuda.is_available()) else "cpu")
    )
    
    # 2. Resolve L_eff
    if leff_mode == "AP":
        if "dA" in str(system):
            leff_ref = AP_LEFF_DAU_200
        else:
            leff_ref = AP_LEFF_PPB_5020
            
        # We might need to map the requested centrality_bins to labels in leff_ref
        # For simplicity, if leff_mode=="AP", we'll just use the standard bins if they match.
        Leff_dict_all = deepcopy(leff_ref)
    else:
        # Use Glauber
        if glauber_custom is not None:
            gl = glauber_custom
        else:
            from glauber import SystemSpec
            A_target = 197 if "dA" in str(system) else 208
            gl = OpticalGlauber(SystemSpec(system[:2], roots_GeV, A=A_target))
        
        labels = [f"{int(a)}-{int(b)}%" for (a, b) in centrality_bins]
        if "dA" in str(system):
            # Arleo-Peigne d+Au paper uses binomial for centrality
            Leff_dict_all = {lab: float(gl.leff_bin_dA(a, b, method="binomial")) 
                             for lab, (a, b) in zip(labels, centrality_bins)}
            Leff_dict_all["MinBias"] = float(gl.leff_minbias_dA())
        else:
            Leff_dict_all = gl.leff_bins_pA(centrality_bins, method="optical")
            Leff_dict_all["MinBias"] = float(gl.leff_minbias_pA())

    # 3. Execution Loop
    results = {
        "meta": {
            "system": system,
            "roots_GeV": roots_GeV,
            "leff_mode": leff_mode,
            "qhat0": qhat0,
            "lp_fm": lp_fm,
            "particle": particle.tag
        },
        "pT": pT_grid,
        "y_regions": {}
    }
    
    comps = ["loss", "broad", "eloss_broad"] if return_components else ["eloss_broad"]
    
    for y_label, y_val in y_regions.items():
        y_data = {"cent_labels": [lab for lab in Leff_dict_all.keys() if lab != "MinBias"]}
        
        # We'll use a local helper to compute the curves for a given L
        def compute_one_L(L_val, y_spec):
            qp = replace(qp_master, LA_fm=float(L_val))
            rl, rb, rt = [], [], []
            
            # handle binned vs fixed y
            is_binned = isinstance(y_spec, (list, tuple))
            
            for pt in pT_grid:
                if is_binned:
                    # use R_binned_2D
                    def R_loss_fn(yy, ptt):
                        return R_pA_eloss(particle, roots_GeV, qp, yy, ptt)
                    def R_broad_fn(yy, ptt):
                        return R_pA_broad(particle, roots_GeV, qp, yy, ptt)
                    
                    l_v = R_binned_2D(R_loss_fn, particle, roots_GeV, y_spec, (pt, pt), Ny_bin=12, Npt_bin=1)
                    b_v = R_binned_2D(R_broad_fn, particle, roots_GeV, y_spec, (pt, pt), Ny_bin=12, Npt_bin=1)
                else:
                    l_v = R_pA_eloss(particle, roots_GeV, qp, y_spec, pt)
                    b_v = R_pA_broad(particle, roots_GeV, qp, y_spec, pt)
                
                rl.append(l_v); rb.append(b_v); rt.append(l_v * b_v)
            
            out = {"eloss_broad": np.array(rt)}
            if return_components:
                out["loss"] = np.array(rl)
                out["broad"] = np.array(rb)
            return out

        # Compute for each centrality
        for lab, L in Leff_dict_all.items():
            y_data[lab] = compute_one_L(L, y_val)
            
        # Band calculation if requested
        if band_spec:
            # band_spec = {"qhat0": (lo, hi), "p0_scale": (lo, hi)}
            # This would double/triple the calculation. 
            # Implement if needed, for now we leave hooks.
            pass
            
        results["y_regions"][y_label] = y_data
        
    return results

def set_publication_style():
    """
    Sets global matplotlib RC parameters for publication-quality plots.
    Consistent with user's 'signature style':
    - Serif font (Times New Roman)
    - Inward minor ticks
    - No grid, no titles
    - Legend consistency
    """
    import matplotlib.pyplot as plt
    params = {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "font.size": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.edgecolor": "black",
        "patch.force_edgecolor": True,
    }
    plt.rcParams.update(params)

# ----------------------------------------------------------------------
# Default Centrality Colors
# ----------------------------------------------------------------------
DEFAULT_CENTRALITY_COLORS = {
    "0-10%":   "tab:red",
    "0-20%":   "tab:blue",
    "20-40%":  "tab:orange",
    "40-60%":  "tab:green",
    "60-80%":  "tab:red",
    "60-100%": "tab:red",
    "MB":      "black",
}

# ----------------------------------------------------------------------
# Missing Plotting Functions (Ported from Notebooks)
# ----------------------------------------------------------------------

def plot_RpA_vs_y_band(
    y_cent, Rc_dict, Rlow_dict, Rhigh_dict,
    tags_order,
    component_label=r"$R_{pA}(y)$",
    ax=None,
    step: bool = True,
    note: str | None = None,
    colors_dict=None,
):
    """
    RpA vs y with bands for each centrality + MB.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=130)
    else:
        fig = ax.figure

    use_colors = colors_dict if colors_dict is not None else DEFAULT_CENTRALITY_COLORS

    for tag in tags_order:
        if tag not in Rc_dict: continue
        
        Rc  = np.asarray(Rc_dict[tag])
        Rlo = np.asarray(Rlow_dict[tag])
        Rhi = np.asarray(Rhigh_dict[tag])
        col = use_colors.get(tag, "tab:gray")

        if step:
            x_edges, y_c  = _step_from_centers(y_cent, Rc)
            _,       y_lo = _step_from_centers(y_cent, Rlo)
            _,       y_hi = _step_from_centers(y_cent, Rhi)

            ax.step(x_edges, y_c, where="post", color=col, lw=1.5, label=tag)
            ax.fill_between(x_edges, y_lo, y_hi,
                            step="post", color=col, alpha=0.25, linewidth=0.0)
        else:
            ax.plot(y_cent, Rc, color=col, lw=1.5, label=tag)
            ax.fill_between(y_cent, Rlo, Rhi,
                            color=col, alpha=0.25, linewidth=0.0)

    # MB as thick black line + grey band
    if "MB" in Rc_dict:
        Rc  = np.asarray(Rc_dict["MB"])
        Rlo = np.asarray(Rlow_dict["MB"])
        Rhi = np.asarray(Rhigh_dict["MB"])

        if step:
            x_edges, y_c  = _step_from_centers(y_cent, Rc)
            _,       y_lo = _step_from_centers(y_cent, Rlo)
            _,       y_hi = _step_from_centers(y_cent, Rhi)

            ax.step(x_edges, y_c, where="post", color="k", lw=2.0, label="MB")
            ax.fill_between(x_edges, y_lo, y_hi,
                            step="post", color="gray", alpha=0.30, linewidth=0.0)
        else:
            ax.plot(y_cent, Rc, color="k", lw=2.0, label="MB")
            ax.fill_between(y_cent, Rlo, Rhi,
                            color="gray", alpha=0.30, linewidth=0.0)

    # Only first axis gets the legend (if strictly adhering to previous style, but here we just put it on the ax passed)
    # The previous code had specific logic for grids. Here we assume single ax usage or loop.
    
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(component_label)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(False)

    if note is not None:
        ax.text(
            0.03, 0.97, note,
            transform=ax.transAxes,
            fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

    return fig, ax


def plot_RpA_vs_pT_band(
    pT_cent,
    Rc_dict, Rlow_dict, Rhigh_dict,
    tags_order,
    component_label=r"$R_{pA}(p_T)$",
    ax=None,
    step: bool = True,
    note: str | None = None,
    colors_dict=None,
    xlim=None,
):
    """
    RpA vs pT with centrality bands.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=130)
    else:
        fig = ax.figure

    use_colors = colors_dict if colors_dict is not None else DEFAULT_CENTRALITY_COLORS

    for tag in tags_order:
        if tag not in Rc_dict: continue
        
        Rc  = np.asarray(Rc_dict[tag])
        Rlo = np.asarray(Rlow_dict[tag])
        Rhi = np.asarray(Rhigh_dict[tag])
        col = use_colors.get(tag, "tab:gray")

        if step:
            x_edges, y_c  = _step_from_centers(pT_cent, Rc)
            _,       y_lo = _step_from_centers(pT_cent, Rlo)
            _,       y_hi = _step_from_centers(pT_cent, Rhi)

            ax.step(x_edges, y_c, where="post", lw=1.5, color=col, label=tag)
            ax.fill_between(x_edges, y_lo, y_hi,
                            step="post", alpha=0.25, color=col, linewidth=0.0)
        else:
            ax.plot(pT_cent, Rc, lw=1.5, color=col, label=tag)
            ax.fill_between(pT_cent, Rlo, Rhi,
                            alpha=0.25, color=col, linewidth=0.0)

    if "MB" in Rc_dict:
        Rc  = np.asarray(Rc_dict["MB"])
        Rlo = np.asarray(Rlow_dict["MB"])
        Rhi = np.asarray(Rhigh_dict["MB"])

        if step:
            x_edges, y_c  = _step_from_centers(pT_cent, Rc)
            _,       y_lo = _step_from_centers(pT_cent, Rlo)
            _,       y_hi = _step_from_centers(pT_cent, Rhi)

            ax.step(x_edges, y_c, where="post", lw=2.0, color="k", label="MB")
            ax.fill_between(x_edges, y_lo, y_hi,
                            step="post", alpha=0.3, color="gray", linewidth=0.0)
        else:
            ax.plot(pT_cent, Rc, lw=2.0, color="k", label="MB")
            ax.fill_between(pT_cent, Rlo, Rhi,
                            alpha=0.3, color="gray", linewidth=0.0)

    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel(component_label)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(False)
    
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0,15)

    if note is not None:
        ax.text(
            0.03, 0.97, note,
            transform=ax.transAxes,
            fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

    return fig, ax
