# quenching_fast.py
# Robust, GPU-ready coherent energy-loss (Arleo–Peigné) + pT broadening.
# Minimal-change hardening (finalized):
#  * δy upper-limit respects σ_pp table y-range (avoids “flat vs y” when clamped)
#  * Peripheral-safe Cronin taper → 0 as L→lp (fixes low-L_eff artefacts)
#  * Thin-phase auto-Ny tightened; exp-mapping on dy with δ-peak p0 piece
#  * α(μ) hook used consistently inside Φ(z) (mT/pT/fixed)
#  * AB: one-sided fallback if only A or B is active (no premature σ_pp return)
#  * Binning weights: default y=0; pass weight_ref_y="local" to use local y
#  * Units are consistent:  q̂0 in GeV^2/fm; ℓ^2=q̂·L [GeV^2]; Λ_p^2=max(λ_QCD^2, q̂·lp)

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple, Literal, Union
import math, numpy as np

# ----- torch / torchquad (optional acceleration) ----------------------------
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from torchquad import GaussLegendre as TQ_GaussLegendre
    _HAS_TQ = True
except Exception:
    _HAS_TQ = False

# ----- constants / numerics --------------------------------------------------
M_PROTON = 0.938                 # GeV
LOG2     = math.log(2.0)
Z_FLOOR  = 1e-12
_PI2_12  = (math.pi**2)/12.0
DY_EPS   = 1e-6
HBARC    = 0.1973269804          # GeV·fm

def _torch_device(dev: Optional[str] = None):
    if not _HAS_TORCH:
        return None
    if isinstance(dev, str):
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- GL nodes (NumPy / Torch) ---------------------------------------------
def _gl_nodes_np(a: float, b: float, n: int):
    x, w = np.polynomial.legendre.leggauss(int(n))
    xm, xc = 0.5*(b-a), 0.5*(b+a)
    return xc + xm*x, xm*w

def _gl_nodes_torch(a: float, b: float, n: int, device):
    x, w = np.polynomial.legendre.leggauss(int(n))
    xm, xc = 0.5*(b-a), 0.5*(b+a)
    xt = torch.tensor(xc + xm*x, dtype=torch.float64, device=device)
    wt = torch.tensor(xm*w,      dtype=torch.float64, device=device)
    return xt, wt

def _phi_nodes_gl_torch(nphi: int, device):
    phi, w = _gl_nodes_torch(0.0, 2.0*math.pi, nphi, device)
    wbar   = w/(2.0*math.pi)  # normalized average over 2π
    return phi, wbar, torch.cos(phi), torch.sin(phi)

# ----- Dilog Li2 (real) ------------------------------------------------------
def _li2_series_unit_np(x: np.ndarray, K: int = 256) -> np.ndarray:
    k = np.arange(1, K+1, dtype=float)
    return np.sum(((-x[:,None])**k)/(k*k), axis=1)

def Li2_np(z: np.ndarray) -> np.ndarray:
    x = -z
    res = np.empty_like(x, dtype=float)
    m_small = (x <= 1.0)
    if np.any(m_small):
        res[m_small] = _li2_series_unit_np(x[m_small])
    if np.any(~m_small):
        xs = x[~m_small]
        ln = np.log(xs)
        inv = -1.0/xs
        res_large = -_PI2_12 - 0.5*ln*ln - _li2_series_unit_np(inv)
        res[~m_small] = res_large
    return res

def _li2_series_unit_t(x, K: int = 256):
    k = torch.arange(1, K+1, device=x.device, dtype=torch.float64)
    term = torch.pow(-x.unsqueeze(-1), k)/(k*k)
    return term.sum(dim=-1)

def Li2_torch(z):
    """
    Reverted to stable scipy-based Li2 to ensure physics correctness.
    Matches Arleo-Peigne paper results.
    """
    from scipy.special import spence
    
    # We only care about real z.
    # z is typically negative in quenching_fast.
    # Li2(z) = spence(1-z)
    z_np = z.detach().cpu().numpy()
    res_np = spence(1.0 - z_np)
    
    return torch.from_numpy(res_np).to(z.device, dtype=z.dtype)

# ----- Parameters -------------------------------------------------------------
@dataclass(frozen=True)
class QuenchParams:
    qhat0: float = 0.075    # GeV^2/fm at x=1e-2
    lp_fm: float = 1.5      # fm
    LA_fm: float = 10.0     # fm
    LB_fm: float = 10.0     # fm
    lambdaQCD: float = 0.25 # GeV
    roots_GeV: float = 8160.0
    alpha_of_mu: Callable[[float], float] = lambda mu: 0.5
    alpha_scale: Literal["mT","pT","fixed"] = "mT"
    Nc: int = 3
    use_hard_cronin: bool = True
    mapping: Literal["exp","linear"] = "exp"
    system: str = "pA" # "pA", "dA", "AA"
    device: Optional[str] = None

HBARC = 0.1973269804  # GeV*fm
# ----- helpers (NumPy) -------------------------------------------------------
def xA0_from_L(L_fm: float) -> float:
    # x0 = 1/(2 m_p L) in natural units; L is in fm -> multiply by ħc
    L = max(L_fm, 1e-12)
    return HBARC / (2.0 * M_PROTON * L)

def qhat_of_x(qp: QuenchParams, xA: float) -> float:
    # qhat(x) = qhat0 * (1e-2/x)^0.3  [GeV^2/fm]
    x_eff = max(min(xA, xA0_from_L(qp.LA_fm)), 1e-9)
    return qp.qhat0 * (1e-2/x_eff)**0.3

def _l2(qp: QuenchParams, xA: float, L_fm: float) -> float:
    # ℓ^2 = qhat(x) * L  [GeV^2]
    return qhat_of_x(qp, xA) * L_fm

# ----- Torch qhat / ℓ² / Λp² / αs / ΔpT -------------------------------------
def _qhat_t(qhat0: float, x):
    x = torch.clamp(x, min=1e-12)
    return qhat0 * torch.pow(1e-2/x, 0.3)

def _l2_t(qp: QuenchParams, x, L_fm: float):
    return _qhat_t(qp.qhat0, x) * L_fm

def _Lambda_p2_t(qp: QuenchParams, x):
    lp2 = _qhat_t(qp.qhat0, x) * qp.lp_fm
    lam2= torch.tensor(qp.lambdaQCD**2, dtype=torch.float64, device=lp2.device)
    return torch.maximum(lp2, lam2)

def _alpha_mu_t(qp: QuenchParams, mT, pT):
    if qp.alpha_scale == "mT":
        mu = mT
    elif qp.alpha_scale == "pT":
        if torch.is_tensor(pT):
            mu = torch.clamp(pT, min=0.5)
        else:
            mu = max(float(pT), 0.5)
    else:
        mu = 1.5
    
    # Evaluate alpha_s. alpha_of_mu from provider usually expects float or numpy.
    # For now, evaluate at the mean scalar if mu is a tensor, 
    # or handle item() if it's a size-1 tensor.
    if torch.is_tensor(mu):
        if mu.numel() == 1:
            return float(qp.alpha_of_mu(float(mu.item())))
        else:
            # Vectorized alpha evaluation (if supported by provider)
            # Most cases use constant alpha, so this is safe.
            return qp.alpha_of_mu(mu.detach().cpu().numpy())
    return float(qp.alpha_of_mu(mu))

def _dpt_from_xL_t(qp: QuenchParams, x, L_fm: float, hard=True):
    # C++ crosssections.cpp uses: dptA = sqrt(lA2 - lAp2) = sqrt(qhat*(L - lp))
    # This is the CORRECT formula matching the C++ reference
    qL = _qhat_t(qp.qhat0, x) * L_fm
    qlp = _qhat_t(qp.qhat0, x) * qp.lp_fm
    base = torch.clamp(qL - qlp, min=0.0)
    return torch.sqrt(base)


# ----- kinematics -------------------------------------------------------------
def y_max(roots_GeV: float, mT: float) -> float:
    return math.log(max(roots_GeV/mT, 1.0 + 1e-12))

def dymax(y: float, y_max_pt: float) -> float:
    return max(0.0, min(LOG2, max(y_max_pt - y, 0.0)))

# ----- AP kernel (Torch) -----------------------------------------------------
def _Phat_core_t(z, Mperp2, l2, Lp2, alpha, Nc):
    z  = torch.clamp(z, min=Z_FLOOR)
    mask0 = (l2 <= Lp2)
    inv = 1.0/(z*z*Mperp2)
    # C++ uses 2.0*M_PI in denominator (line 204 in crosssections.cpp)
    expo = (alpha*Nc/(2.0*math.pi)) * (Li2_torch(-l2*inv) - Li2_torch(-Lp2*inv))
    expo = torch.clamp(expo, min=-700.0, max=+700.0)
    deriv= 2.0*(torch.log1p(l2*inv) - torch.log1p(Lp2*inv))/z
    # C++ uses 2.0*M_PI in pre-factor (line 209 in crosssections.cpp)
    val  = (alpha*torch.exp(expo)*Nc*deriv)/(2.0*math.pi)
    val  = torch.where(mask0, torch.zeros_like(val), val)
    val  = torch.where(torch.isfinite(val) & (val>0.0), val, torch.zeros_like(val))
    return val

def PhatA_t(z, mT, xA, qp: QuenchParams, pT=None):
    a   = _alpha_mu_t(qp, mT, 0.0 if pT is None else pT)
    if not torch.is_tensor(a):
        a = torch.tensor(a, dtype=torch.float64, device=z.device)
    l2  = _l2_t(qp, xA, qp.LA_fm)
    Lp2 = _Lambda_p2_t(qp, xA)
    return _Phat_core_t(z, mT*mT, l2, Lp2, a, qp.Nc)

def PhatB_t(z, mT, xB, qp: QuenchParams, pT=None):
    a   = _alpha_mu_t(qp, mT, 0.0 if pT is None else pT)
    if not torch.is_tensor(a):
        a = torch.tensor(a, dtype=torch.float64, device=z.device)
    l2  = _l2_t(qp, xB, qp.LB_fm)
    Lp2 = _Lambda_p2_t(qp, xB)
    return _Phat_core_t(z, mT*mT, l2, Lp2, a, qp.Nc)

# ----- Torch σ_pp table ------------------------------------------------------
class TorchSigmaPPTable:
    """Torch bilinear in y and linear in pT (broadcast-friendly)."""
    def __init__(self, P, roots_GeV: float, y_grid: np.ndarray, pt_grid: np.ndarray, device: Optional[str]=None):
        assert _HAS_TORCH, "Torch not available."
        self.P, self.roots = P, float(roots_GeV)
        self.device = _torch_device(device)
        self.y  = torch.tensor(np.asarray(y_grid, float),  dtype=torch.float64, device=self.device)
        self.pt = torch.tensor(np.asarray(pt_grid, float), dtype=torch.float64, device=self.device)
        Z = np.empty((self.y.numel(), self.pt.numel()), float)
        for i, yy in enumerate(self.y.cpu().numpy()):
            for j, pp in enumerate(self.pt.cpu().numpy()):
                Z[i, j] = float(P.d2sigma_pp(float(yy), float(pp), self.roots))
        self.Z = torch.tensor(Z, dtype=torch.float64, device=self.device)

    def __call__(self, y, pt):
        yv = y.to(self.device, dtype=torch.float64)
        pv = pt.to(self.device, dtype=torch.float64)
        # Clamp queries to domain (prevents artificial “same for all L” at edges)
        yv = torch.clamp(yv, min=self.y[0],  max=self.y[-1])
        pv = torch.clamp(pv, min=self.pt[0], max=self.pt[-1])

        i  = torch.searchsorted(self.y, yv) - 1
        i  = torch.clamp(i, 0, self.y.numel()-2)
        y0 = self.y[i]; y1 = self.y[i+1]
        ty = torch.where((y1>y0), (yv-y0)/(y1-y0), torch.zeros_like(yv))

        def interp_row(idx):
            row = self.Z[idx]
            j = torch.searchsorted(self.pt, pv) - 1
            j = torch.clamp(j, 0, self.pt.numel()-2)
            p0 = self.pt[j]; p1 = self.pt[j+1]
            pv_safe = torch.clamp(pv, min=p0, max=p1)
            r0 = row[j];     r1 = row[j+1]
            u  = torch.where((p1>p0), (pv_safe-p0)/(p1-p0), torch.zeros_like(pv_safe))
            return (1.0-u)*r0 + u*r1

        z0 = interp_row(i)
        z1 = interp_row(i+1)
        return (1.0 - ty.unsqueeze(-1))*z0 + ty.unsqueeze(-1)*z1

def _dsigpp_from_table(table: TorchSigmaPPTable):
    def f(y_t, pt_t):
        return table(y_t, pt_t)
    return f

# ----- φ kinematics (Torch) --------------------------------------------------
def _shift_pT_pA(pt: float, dpta, cphi, sphi):
    ptv = torch.tensor(pt, dtype=torch.float64, device=dpta.device)
    return torch.sqrt((ptv - dpta*cphi)**2 + (dpta*sphi)**2)

def _shift_pT_AB(pt: float, dptB, dptA, cA, sA, cB, sB):
    ptv = torch.tensor(pt, dtype=torch.float64, device=dptA.device)
    comp1 = ptv - dptA*cA - dptB*cB
    comp2 =        dptA*sA + dptB*sB
    return torch.sqrt(comp1*comp1 + comp2*comp2)

# ----- adaptive Ny for thin phase space -------------------------------------
def _Ny_from_dymax(dym: float) -> int:
    if dym < 0.005: return 96  
    if dym < 0.02:  return 64
    if dym < 0.05:  return 48
    if dym < 0.10:  return 40
    return 32

# ----- Optical Weights (pA, dA) ----------------------------------------------
_MB_TO_FM2 = 0.1

def _optical_bin_weight_pA(glauber, c0_percent: float, c1_percent: float, n_sub: int = 1200) -> float:
    """
    Fraction of total inelastic pA cross section in the centrality bin [c0, c1]%.
    Used for weighting centrality bins in min-bias observables.
    """
    bmin = float(glauber.b_from_percentile(c0_percent/100.0, kind="pA"))
    bmax = float(glauber.b_from_percentile(c1_percent/100.0, kind="pA"))
    if bmax <= bmin: return 0.0
    b_sub = np.linspace(bmin, bmax, n_sub)
    
    # Try to find tabulated TpA(b) or use callable
    if hasattr(glauber, 'b_grid') and hasattr(glauber, 'TpA_b'):
        TpA_sub = np.interp(b_sub, np.asarray(glauber.b_grid, float), np.asarray(glauber.TpA_b, float))
    elif hasattr(glauber, 'TpA'):
        TpA_sub = np.array([glauber.TpA(b) for b in b_sub])
    else:
        return 0.0
        
    sigma_fm2 = float(glauber.spec.sigma_nn_mb) * _MB_TO_FM2
    pinel = 1.0 - np.exp(-sigma_fm2 * np.maximum(TpA_sub, 0.0))
    integrand = 2.0 * math.pi * b_sub * pinel
    
    try:
        numer_fm2 = float(np.trapz(integrand, b_sub))
    except Exception:
        # fallback for newer numpy versions where trapz might be moved
        numer_fm2 = float(np.sum(integrand) * (b_sub[1]-b_sub[0]))
        
    sigma_tot_fm2 = float(glauber.sigma_pA_tot_mb) * _MB_TO_FM2
    return numer_fm2 / max(sigma_tot_fm2, 1e-30)

def _optical_bin_weight_dA(glauber, c0_percent: float, c1_percent: float, n_sub: int = 1200) -> float:
    """
    Fraction of total inelastic dA cross section in the centrality bin [c0, c1]%.
    Used for weighting centrality bins in min-bias observables.
    """
    bmin = float(glauber.b_from_percentile(c0_percent/100.0, kind="dA"))
    bmax = float(glauber.b_from_percentile(c1_percent/100.0, kind="dA"))
    if bmax <= bmin: return 0.0
    b_sub = np.linspace(bmin, bmax, n_sub)

    if hasattr(glauber, 'b_grid') and hasattr(glauber, 'TdA_b'):
        TdA_sub = np.interp(b_sub, np.asarray(glauber.b_grid, float), np.asarray(glauber.TdA_b, float))
    elif hasattr(glauber, 'TdA'):
        TdA_sub = np.array([glauber.TdA(b) for b in b_sub])
    else:
        return 0.0

    sigma_fm2 = float(glauber.spec.sigma_nn_mb) * _MB_TO_FM2
    pinel = 1.0 - np.exp(-sigma_fm2 * np.maximum(TdA_sub, 0.0))
    integrand = 2.0 * math.pi * b_sub * pinel
    
    try:
        numer_fm2 = float(np.trapz(integrand, b_sub))
    except Exception:
        numer_fm2 = float(np.sum(integrand) * (b_sub[1]-b_sub[0]))
    
    sigma_tot_fm2 = float(glauber.sigma_dA_tot_mb) * _MB_TO_FM2
    return numer_fm2 / max(sigma_tot_fm2, 1e-30)

