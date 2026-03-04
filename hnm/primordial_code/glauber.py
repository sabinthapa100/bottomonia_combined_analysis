# glauber.py — Optical + Monte Carlo Glauber for pA / dA / AA + L_eff (Arleo–Peigné)
# ---------------------------------------------------------------------------------
# Keeps p+Pb (pA) behavior intact (optical Tp⊗TA + binomial L_eff) and adds d+Au (dA):
#   - Optical dA: T_dA = T_d ⊗ T_A, centrality via inelastic CDF, <Ncoll>, <Npart>
#   - Binomial dA L_eff: Appendix-B style with Hulthén + p2 correction
#   - Monte Carlo Glauber (minimal): event-by-event Ncoll/Npart for pA and dA
#
# Dependencies: numpy only (matplotlib optional for your own plotting)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple
import math
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

MB_TO_FM2 = 0.1
FM2_TO_MB = 10.0

DEFAULT_RHO0 = 0.17   # fm^-3
DEFAULT_LP_FM = 1.5   # fm

_SIGMA_NN_BY_ROOTS_MB = {
    200.0: 42.0,     # RHIC (commonly 40–42 mb; override if you want 40)
    2760.0: 62.0,
    5023.0: 67.6,
    8160.0: 71.0,
}

_DIFFUSENESS_BY_ROOTS = {
    200.0: 0.535,
    2760.0: 0.549,
    5023.0: 0.549,
    8160.0: 0.549,
}

# ----------------------------- Numerics --------------------------------------

def _leggauss(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(int(n))

def _gl_integrate_2d(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ax: float, bx: float, ay: float, by: float, nx: int = 64, ny: int = 64
) -> float:
    x, wx = _leggauss(nx)
    y, wy = _leggauss(ny)
    xm, xc = 0.5*(bx-ax), 0.5*(bx+ax)
    ym, yc = 0.5*(by-ay), 0.5*(by+ay)
    X = xm * np.broadcast_to(x[:, None], (nx, ny)) + xc
    Y = ym * np.broadcast_to(y[None, :], (nx, ny)) + yc
    W = (xm*ym) * (wx[:, None] * wy[None, :])
    F = f(X, Y)
    return float(np.sum(W * F))

def _interp1(x: np.ndarray, y: np.ndarray, xq: np.ndarray | float) -> np.ndarray | float:
    return np.interp(xq, x, y, left=y[0], right=y[-1])

def _bilinear(Z: np.ndarray, x: np.ndarray, y: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    xq = np.asarray(xq, float); yq = np.asarray(yq, float)
    xi = np.clip(np.searchsorted(x, xq) - 1, 0, len(x)-2)
    yi = np.clip(np.searchsorted(y, yq) - 1, 0, len(y)-2)
    x0, x1 = x[xi], x[xi+1]
    y0, y1 = y[yi], y[yi+1]
    z00 = Z[xi, yi];     z10 = Z[xi+1, yi]
    z01 = Z[xi, yi+1];   z11 = Z[xi+1, yi+1]
    tx = (xq - x0)/np.maximum(x1-x0, 1e-15)
    ty = (yq - y0)/np.maximum(y1-y0, 1e-15)
    return ((1-tx)*(1-ty)*z00 + tx*(1-ty)*z10 + (1-tx)*ty*z01 + tx*ty*z11)

# ----------------------------- Densities -------------------------------------

@dataclass(frozen=True)
class WoodsSaxon:
    A: int
    rho0: float = DEFAULT_RHO0
    d_fm: float = 0.549
    rmax_fm: float = 50.0
    dr_fm: float = 0.02
    zmax_fm: float = 50.0
    nz: int = 200

    def radius_rn(self) -> float:
        a13 = self.A ** (1/3)
        return 1.12*a13 - 0.86/a13

    def rho(self, r: np.ndarray) -> np.ndarray:
        R = self.radius_rn()
        return self.rho0 / (1.0 + np.exp((r - R)/self.d_fm))

    def tabulate_T_of_r(self) -> Tuple[np.ndarray, np.ndarray]:
        r = np.arange(0.0, self.rmax_fm + 1e-12, self.dr_fm)
        x, w = _leggauss(self.nz)
        z = self.zmax_fm * x  # [-zmax, +zmax]
        T = np.empty_like(r)
        for i, ri in enumerate(r):
            rr = np.sqrt(ri*ri + z*z)
            T[i] = self.zmax_fm * np.sum(w * self.rho(rr))  # correct mapping
        return r, T

@dataclass(frozen=True)
class ProtonProfile:
    # Kept identical to your existing p+Pb optical projectile shape
    def T_p(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.400905 * np.exp(-1.28022 * np.power(x*x + y*y, 0.925))

@dataclass(frozen=True)
class HulthenProfile:
    # Orientation-averaged deuteron thickness from Hulthén; normalized to 2 nucleons.
    alpha: float = 0.228
    beta: float = 1.18
    zmax_fm: float = 50.0
    nz: int = 400

    def _psi2_relative(self, r: np.ndarray) -> np.ndarray:
        a, b = self.alpha, self.beta
        N2 = (a*b*(a+b)) / (2.0*math.pi*(b-a)**2)
        r_safe = np.where(r < 1e-12, 1e-12, r)
        return N2 * (np.exp(-a*r_safe) - np.exp(-b*r_safe))**2 / (r_safe*r_safe)

    def rho_cm(self, r: np.ndarray) -> np.ndarray:
        # ρ_cm(r) = 16 |ψ(2r)|² gives ∫d³r ρ_cm = 2
        return 16.0 * self._psi2_relative(2.0*r)

    def T_d(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        b = np.sqrt(x*x + y*y).reshape(-1)
        nodes, weights = _leggauss(self.nz)
        z = self.zmax_fm * nodes
        Td = np.empty_like(b)
        for i, bi in enumerate(b):
            rr = np.sqrt(bi*bi + z*z)
            Td[i] = self.zmax_fm * np.sum(weights * self.rho_cm(rr))
        return Td.reshape(x.shape)

# ----------------------------- System spec -----------------------------------

@dataclass(frozen=True)
class SystemSpec:
    system: Literal["AA", "pA", "dA"]
    roots_GeV: float
    A: int
    sigma_nn_mb: Optional[float] = None
    diffuseness_fm: Optional[float] = None

    def resolve(self) -> Tuple[float, float]:
        sigma = self.sigma_nn_mb if self.sigma_nn_mb is not None else _SIGMA_NN_BY_ROOTS_MB.get(self.roots_GeV, 67.6)
        dval  = self.diffuseness_fm if self.diffuseness_fm is not None else _DIFFUSENESS_BY_ROOTS.get(self.roots_GeV, 0.549)
        return float(sigma), float(dval)

# =============================================================================
# Optical Glauber
# =============================================================================

class OpticalGlauber:
    def __init__(
        self,
        spec: SystemSpec,
        *,
        verbose: bool = True,
        xylim_fm: float = 20.0,
        nx: int = 160,
        ny: int = 160,
        # pA convolution window (keeps your p+Pb behavior)
        pa_x_half_width_fm: float = 5.0,
        pa_y_half_width_fm: float = 15.0,
        nx_pa: int = 160,
        ny_pa: int = 160,
        # b-grid
        bmax_fm: float = 20.0,
        nb: int = 201,
    ) -> None:
        self.spec = spec
        self.verbose = bool(verbose)
        self.sigma_nn_mb, self.d_fm = spec.resolve()
        self.sigma_nn_fm2 = self.sigma_nn_mb * MB_TO_FM2

        # Ensure downstream code can always read resolved values from `self.spec`
        # (some callers expect `glauber.spec.sigma_nn_mb` to be a float, not None)
        self.spec = SystemSpec(
            system=spec.system,
            roots_GeV=spec.roots_GeV,
            A=spec.A,
            sigma_nn_mb=self.sigma_nn_mb,
            diffuseness_fm=self.d_fm,
        )

        # target thickness
        self.ws = WoodsSaxon(A=spec.A, d_fm=self.d_fm)
        self.r_grid, self.TA_r_lut = self.ws.tabulate_T_of_r()

        # TA(x,y) grid for fast bilinear
        self.xylim_fm = float(xylim_fm)
        self.nx = int(nx); self.ny = int(ny)
        self.x_grid = np.linspace(-self.xylim_fm, self.xylim_fm, self.nx)
        self.y_grid = np.linspace(-self.xylim_fm, self.xylim_fm, self.ny)
        X, Y = np.meshgrid(self.x_grid, self.y_grid, indexing="ij")
        self.TA_xy_grid = _interp1(self.r_grid, self.TA_r_lut, np.hypot(X, Y))

        if self.verbose:
            integ = np.trapz(np.trapz(self.TA_xy_grid, self.y_grid, axis=1), self.x_grid, axis=0)
            print(f"[OpticalGlauber] Target A={spec.A}, d={self.d_fm:.3f} fm, σ_nn={self.sigma_nn_mb:.2f} mb")
            print(f"[OpticalGlauber] ∫ d²s T_A(s) ≈ {float(integ):.3f}  (should be ~A={spec.A})")

        self.proton = ProtonProfile()
        self.deuteron = HulthenProfile()

        self.pa_x_hw = float(pa_x_half_width_fm)
        self.pa_y_hw = float(pa_y_half_width_fm)
        self.nx_pa = int(nx_pa); self.ny_pa = int(ny_pa)

        self.bmax_fm = float(bmax_fm)
        self.b_grid = np.linspace(0.0, self.bmax_fm, int(nb))
        self.db_fm = float(self.b_grid[1] - self.b_grid[0])

        # Caches to avoid repeating expensive binomial precomputations across many bins.
        self._cache_pA_binomial = {}
        self._cache_dA_binomial = {}

        # ------------------------------------------------------------
        # Deuteron radial thickness LUT for NuclearAbsorption(dA_avg_TA)
        # Provides: Td_r(r) [fm^-2], with ∫ d^2s Td(s) ≈ 2
        # (Does not affect any existing calculation unless called.)
        # ------------------------------------------------------------
        self._rTd = np.arange(0.0, self.bmax_fm + 1e-12, 0.05)  # fm
        # Evaluate Td at (x=r, y=0) because Td depends only on sqrt(x^2+y^2)
        self._Td_lut = self.deuteron.T_d(self._rTd, np.zeros_like(self._rTd)).astype(float)

        if self.verbose:
            integ_td = 2.0 * math.pi * np.trapz(self._rTd * self._Td_lut, self._rTd)
            print(f"[OpticalGlauber] ∫ d²s T_d(s) ≈ {float(integ_td):.4f}  (should be ~2)")

        if self.verbose:
            print("[OpticalGlauber] Tabulating overlaps T_AA(b), T_pA(b), T_dA(b)...")
        self.TAA_b = np.array([self._TAA_of_b(b) for b in self.b_grid], float)
        self.TpA_b = np.array([self._TpA_conv_of_b(b, proj="p") for b in self.b_grid], float)
        self.TdA_b = np.array([self._TpA_conv_of_b(b, proj="d") for b in self.b_grid], float)

        self.sigma_AA_tot_mb = self._sigma_tot_mb(self.TAA_b)
        self.sigma_pA_tot_mb = self._sigma_tot_mb(self.TpA_b)
        self.sigma_dA_tot_mb = self._sigma_tot_mb(self.TdA_b)

        self.cdf_AA = self._cdf(self.TAA_b, self.sigma_AA_tot_mb)
        self.cdf_pA = self._cdf(self.TpA_b, self.sigma_pA_tot_mb)
        self.cdf_dA = self._cdf(self.TdA_b, self.sigma_dA_tot_mb)

        if self.verbose:
            print(f"[OpticalGlauber] σ_tot: AA={self.sigma_AA_tot_mb:.1f} mb, "
                  f"pA={self.sigma_pA_tot_mb:.1f} mb, dA={self.sigma_dA_tot_mb:.1f} mb")

    # --------- TA accessors ---------

    def TA_r(self, r: np.ndarray | float) -> np.ndarray | float:
        r = np.asarray(r, float)
        out = np.zeros_like(r)
        m = r <= self.r_grid[-1]
        out[m] = _interp1(self.r_grid, self.TA_r_lut, r[m])
        return float(out) if out.shape == () else out

    def Td_r(self, r: np.ndarray | float) -> np.ndarray | float:
        """
        Deuteron thickness T_d(r) [fm^-2], orientation-averaged Hulthén.
        Normalized such that ∫ d²s T_d(s) ≈ 2.

        Required by NuclearAbsorption(mode='dA_avg_TA').
        """
        r = np.asarray(r, float)
        out = np.zeros_like(r)
        m = r <= self._rTd[-1]
        out[m] = _interp1(self._rTd, self._Td_lut, r[m])
        return float(out) if out.shape == () else out

    def TA_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float); y = np.asarray(y, float)
        r = np.hypot(x, y)
        out = np.empty_like(r)
        inside = (np.abs(x) <= self.xylim_fm) & (np.abs(y) <= self.xylim_fm)
        if np.any(inside):
            out[inside] = _bilinear(self.TA_xy_grid, self.x_grid, self.y_grid, x[inside], y[inside])
        if np.any(~inside):
            out[~inside] = _interp1(self.r_grid, self.TA_r_lut, r[~inside])
        return out

    # --------- overlap integrals ---------

    def _TAA_of_b(self, b_fm: float) -> float:
        bx = b_fm/2.0
        def f(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return self.TA_xy(X+bx, Y) * self.TA_xy(X-bx, Y)
        lim = 15.0
        return _gl_integrate_2d(f, -lim, lim, -lim, lim, nx=120, ny=120)

    def _TpA_conv_of_b(self, b_fm: float, proj: Literal["p","d"]="p") -> float:
        bx = b_fm/2.0
        hw_x = self.pa_x_hw if proj == "p" else max(self.pa_x_hw, 7.0)
        xa, xb = b_fm - hw_x, b_fm + hw_x
        ya, yb = -self.pa_y_hw, self.pa_y_hw

        def f(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            TA = self.TA_xy(X+bx, Y)
            if proj == "p":
                Tp = self.proton.T_p(X-bx, Y)
            else:
                # Use cached Td(r) from look-up table for speed
                Tp = self.Td_r(np.hypot(X-bx, Y)) 
            return TA * Tp

        return _gl_integrate_2d(f, xa, xb, ya, yb, nx=self.nx_pa, ny=self.ny_pa)

    # --------- σ_tot and CDF ---------

    def _sigma_tot_mb(self, T_b: np.ndarray) -> float:
        integrand = self.b_grid * (1.0 - np.exp(-self.sigma_nn_fm2 * np.maximum(T_b, 0.0)))
        sig_fm2 = 2.0*math.pi * np.trapz(integrand, self.b_grid)
        return float(sig_fm2 * FM2_TO_MB)

    def _cdf(self, T_b: np.ndarray, sig_tot_mb: float) -> np.ndarray:
        integrand = self.b_grid * (1.0 - np.exp(-self.sigma_nn_fm2 * np.maximum(T_b, 0.0)))
        cum = 2.0*math.pi * np.cumsum(integrand) * self.db_fm
        return (cum * FM2_TO_MB) / max(sig_tot_mb, 1e-30)

    def b_from_percentile(self, c: float, kind: Literal["AA","pA","dA"]) -> float:
        c = float(np.clip(c, 0.0, 1.0))
        if kind == "AA":
            return float(_interp1(self.cdf_AA, self.b_grid, c))
        if kind == "pA":
            return float(_interp1(self.cdf_pA, self.b_grid, c))
        return float(_interp1(self.cdf_dA, self.b_grid, c))

    def _bin_edges_b(self, c0: float, c1: float, kind: Literal["pA","dA","AA"]) -> Tuple[float,float]:
        return self.b_from_percentile(c0, kind), self.b_from_percentile(c1, kind)

    # =============================================================================
    # Optical means: <Ncoll>, <Npart> for dA
    # =============================================================================

    def ncoll_mean_bin_pA_optical(self, c0: float, c1: float, n_sub: int = 1200) -> float:
        bmin, bmax = self._bin_edges_b(c0, c1, "pA")
        if bmax <= bmin:
            return 0.0
        b = np.linspace(bmin, bmax, n_sub)
        T = np.interp(b, self.b_grid, self.TpA_b)
        lam = self.sigma_nn_fm2 * np.maximum(T, 0.0)
        pinel = 1.0 - np.exp(-lam)
        Kcond = np.where(pinel > 0, lam/pinel, 0.0)
        w = 2.0*math.pi*b*pinel
        return float(np.trapz(Kcond*w, b) / max(np.trapz(w, b), 1e-30))

    def ncoll_mean_bin_dA_optical(self, c0: float, c1: float, n_sub: int = 1200) -> float:
        bmin, bmax = self._bin_edges_b(c0, c1, "dA")
        if bmax <= bmin:
            return 0.0
        b = np.linspace(bmin, bmax, n_sub)
        T = np.interp(b, self.b_grid, self.TdA_b)
        lam = self.sigma_nn_fm2 * np.maximum(T, 0.0)
        pinel = 1.0 - np.exp(-lam)
        Kcond = np.where(pinel > 0, lam/pinel, 0.0)
        w = 2.0*math.pi*b*pinel
        return float(np.trapz(Kcond*w, b) / max(np.trapz(w, b), 1e-30))

    def npart_dA_at_b_optical(self, b: float) -> float:
        """
        Optical AB formula with T_d (proj) and T_A (target):
          Npart^A(b) = ∫ d²s T_A(s) [1 - exp(-σ T_d(b-s))]
          Npart^d(b) = ∫ d²s T_d(s) [1 - exp(-σ T_A(b-s))]
        Uses the same square region as TA grid.
        """
        lim = self.xylim_fm
        bx = b/2.0

        xs = self.x_grid
        ys = self.y_grid
        X, Y = np.meshgrid(xs, ys, indexing="ij")

        TA = self.TA_xy_grid
        Td_shift = self.deuteron.T_d((bx - X), -Y)
        part_targ = TA * (1.0 - np.exp(-self.sigma_nn_fm2 * np.maximum(Td_shift, 0.0)))

        TA_shift = self.TA_xy((X + bx), Y)
        Td = self.deuteron.T_d(X - bx, Y)
        part_proj = Td * (1.0 - np.exp(-self.sigma_nn_fm2 * np.maximum(TA_shift, 0.0)))

        dA = float(np.trapz(np.trapz(part_targ, ys, axis=1), xs, axis=0))
        dd = float(np.trapz(np.trapz(part_proj, ys, axis=1), xs, axis=0))
        return float(dA + dd)

    def npart_mean_bin_dA_optical(self, c0: float, c1: float, n_b: int = 60) -> float:
        bmin, bmax = self._bin_edges_b(c0, c1, "dA")
        if bmax <= bmin:
            return 0.0
        b = np.linspace(bmin, bmax, n_b)

        T = np.interp(b, self.b_grid, self.TdA_b)
        lam = self.sigma_nn_fm2 * np.maximum(T, 0.0)
        pinel = 1.0 - np.exp(-lam)
        w = 2.0*math.pi*b*pinel

        nparts = np.array([self.npart_dA_at_b_optical(float(bi)) for bi in b], float)
        return float(np.trapz(nparts*w, b) / max(np.trapz(w, b), 1e-30))

    # =============================================================================
    # pA L_eff (binomial + optical) — keep your working logic
    # =============================================================================

    def leff_bin_pA(self, c0: float, c1: float, *,
                    rho0_fm3: float = DEFAULT_RHO0,
                    Lp_fm: float = DEFAULT_LP_FM,
                    method: Literal["binomial","optical"]="binomial") -> float:
        if method == "optical":
            return self._leff_bin_pA_optical(c0, c1, rho0_fm3=rho0_fm3, Lp_fm=Lp_fm)
        return self._leff_bin_pA_binomial(c0, c1, rho0_fm3=rho0_fm3, Lp_fm=Lp_fm)

    def leff_minbias_pA(self, *, rho0_fm3: float = DEFAULT_RHO0, Lp_fm: float = DEFAULT_LP_FM) -> float:
        A = float(self.spec.A)
        TA2 = np.trapz(np.trapz(self.TA_xy_grid*self.TA_xy_grid, self.y_grid, axis=1), self.x_grid, axis=0)
        return float(Lp_fm + ((A-1.0)/(A*A*rho0_fm3))*TA2)

    def _leff_bin_pA_optical(self, c0: float, c1: float, *, rho0_fm3: float, Lp_fm: float) -> float:
        bmin, bmax = self._bin_edges_b(c0, c1, "pA")
        if bmax <= bmin:
            return float(Lp_fm)
        b = np.linspace(bmin, bmax, 1200)
        TpA = np.interp(b, self.b_grid, self.TpA_b)
        lam = self.sigma_nn_fm2 * np.maximum(TpA, 0.0)
        w = 2.0*math.pi*b
        num = np.trapz(lam*lam*w, b)
        den = np.trapz(lam*w, b)
        if den <= 0:
            return float(Lp_fm)
        return float(Lp_fm + num/(self.sigma_nn_fm2*rho0_fm3*den))

    def _leff_bin_pA_binomial(self, c0: float, c1: float, *, rho0_fm3: float, Lp_fm: float) -> float:
        A = int(self.spec.A)
        sigma = self.sigma_nn_fm2

        cache_key = ("sigmaN_F", A, float(sigma), float(self.bmax_fm))
        cached = self._cache_pA_binomial.get(cache_key)

        if cached is None:
            b = np.linspace(0.0, self.bmax_fm, 1200)
            db = b[1] - b[0]
            TA = np.maximum(self.TA_r(b), 0.0)
            p = np.clip(sigma * TA / float(A), 0.0, 1.0 - 1e-15)

            logC = np.array(
                [math.lgamma(A + 1) - math.lgamma(N + 1) - math.lgamma(A - N + 1) for N in range(A + 1)],
                float,
            )
            sigmaN = np.zeros(A + 1, float)
            jac = 2.0 * math.pi * b * db

            lp = np.where(p > 0, np.log(p), -1e30)
            lq = np.where(p < 1, np.log1p(-p), -1e30)

            for N in range(1, A + 1):
                wN = np.exp(logC[N] + N * lp + (A - N) * lq)
                sigmaN[N] = float(np.sum(jac * wN))

            sig_inel = float(np.sum(sigmaN[1:]))
            if sig_inel <= 0:
                return float(Lp_fm)

            # tail CDF F[N] = (Σ_{k>=N} σ_k) / σ_inel
            tail = 0.0
            F = np.zeros(A + 2, float)
            for N in range(A, 0, -1):
                tail += sigmaN[N]
                F[N] = tail / sig_inel
            F[A + 1] = 0.0

            cached = (logC, sigmaN, F, sig_inel)
            self._cache_pA_binomial[cache_key] = cached
        else:
            logC, sigmaN, F, sig_inel = cached


        def N_from_c(c: float) -> int:
            c = float(np.clip(c, 0.0, 1.0))
            if c <= 0.0:
                return A+1
            if c >= 1.0:
                return 1
            for N in range(A, 0, -1):
                if F[N] >= c and F[N+1] < c:
                    return N
            return 1

        N1 = max(1, N_from_c(c1))
        N2 = min(A, (A if c0 <= 0.0 else N_from_c(c0)-1))
        if N1 > N2:
            return float(Lp_fm)

        Num = sum(N*(N-1)*sigmaN[N] for N in range(N1, N2+1))
        Den = sum(N*sigmaN[N] for N in range(N1, N2+1))
        if Den <= 0:
            return float(Lp_fm)
        return float(max(Lp_fm, Lp_fm + Num/(sigma*rho0_fm3*Den)))

    # =============================================================================
    # dA L_eff (binomial + optical analogue)
    # =============================================================================

    @staticmethod
    def _p_profile_regge(b: np.ndarray, *, alpha_fm2: float = 1.05, Npar: float = 1.1) -> np.ndarray:
        b2 = np.asarray(b, float)**2
        return 1.0 - np.exp(-2.0*Npar*np.exp(-b2/alpha_fm2))

    def _tabulate_I_of_r(self, rmax: float = 20.0, dr: float = 0.05,
                         *, alpha_fm2: float = 1.05, Npar: float = 1.1,
                         ns: int = 240, nphi: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        r_grid = np.arange(0.0, rmax+1e-12, dr)
        smax = 12.0
        s_nodes, s_w = _leggauss(ns)
        sm, sc = 0.5*smax, 0.5*smax
        s = sc + sm*s_nodes
        ws = sm*s_w

        phi_nodes, phi_w = _leggauss(nphi)
        pm, pc = math.pi, math.pi
        phi = pc + pm*phi_nodes   # [0,2π]
        wphi = pm*phi_w

        p_s = self._p_profile_regge(s, alpha_fm2=alpha_fm2, Npar=Npar)
        out = np.empty_like(r_grid)
        for i, r in enumerate(r_grid):
            sr2 = s[:, None]**2 + r*r + 2.0*s[:, None]*r*np.cos(phi[None, :])
            p_sr = self._p_profile_regge(np.sqrt(np.maximum(sr2, 0.0)), alpha_fm2=alpha_fm2, Npar=Npar)
            integrand = (s[:, None] * p_s[:, None] * p_sr)
            out[i] = float(np.sum(ws[:, None]*wphi[None, :]*integrand))
        return r_grid, out

    def _prepare_Pd_lookup(self, rmax: float = 20.0, dr: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        # Build Pd(r) = ∫ dz |ψ(√(r²+z²))|² and normalize so ∫ d²r Pd = 1
        hul = HulthenProfile(alpha=self.deuteron.alpha, beta=self.deuteron.beta, zmax_fm=50.0, nz=300)
        r = np.arange(0.0, rmax+1e-12, dr)
        nodes, weights = _leggauss(300)
        zmax = 50.0
        z = zmax*nodes
        wz = zmax*weights

        Pd_raw = np.empty_like(r)
        for i, ri in enumerate(r):
            rr = np.sqrt(ri*ri + z*z)
            Pd_raw[i] = float(np.sum(wz * hul._psi2_relative(rr)))

        norm = 2.0*math.pi * np.trapz(r*Pd_raw, r)
        Pd = Pd_raw / max(norm, 1e-30)
        return r, Pd

    def leff_bin_dA(
        self, c0: float, c1: float, *,
        rho0_fm3: float = DEFAULT_RHO0,
        Lp_fm: float = DEFAULT_LP_FM,
        method: Literal["binomial","optical"]="binomial",
        alpha_fm2: float = 1.05,
        Npar: float = 1.1
    ) -> float:
        if method == "optical":
            return self._leff_bin_dA_optical(c0, c1, rho0_fm3=rho0_fm3, Lp_fm=Lp_fm)
        return self._leff_bin_dA_binomial(c0, c1, rho0_fm3=rho0_fm3, Lp_fm=Lp_fm,
                                          alpha_fm2=alpha_fm2, Npar=Npar)

    def leff_minbias_dA(self, *, rho0_fm3: float = DEFAULT_RHO0, Lp_fm: float = DEFAULT_LP_FM,
                        method: Literal["binomial","optical"]="binomial",
                        alpha_fm2: float = 1.05, Npar: float = 1.1) -> float:
        return self.leff_bin_dA(0.0, 1.0, rho0_fm3=rho0_fm3, Lp_fm=Lp_fm, method=method,
                                alpha_fm2=alpha_fm2, Npar=Npar)

    def _leff_bin_dA_optical(self, c0: float, c1: float, *, rho0_fm3: float, Lp_fm: float) -> float:
        bmin, bmax = self._bin_edges_b(c0, c1, "dA")
        if bmax <= bmin:
            return float(Lp_fm)
        b = np.linspace(bmin, bmax, 1200)
        TdA = np.interp(b, self.b_grid, self.TdA_b)
        lam = self.sigma_nn_fm2 * np.maximum(TdA, 0.0)
        w = 2.0*math.pi*b
        num = np.trapz(lam*lam*w, b)
        den = np.trapz(lam*w, b)
        if den <= 0:
            return float(Lp_fm)
        # Factor 0.5 accounts for averaging TdA (which integrates to 2A) over deuteron wavefunction
        return float(Lp_fm + 0.5 * num/(self.sigma_nn_fm2*rho0_fm3*den))

    def _leff_bin_dA_binomial(
        self, c0: float, c1: float, *,
        rho0_fm3: float, Lp_fm: float,
        alpha_fm2: float, Npar: float
    ) -> float:
        A = int(self.spec.A)

        # Cache all expensive precomputations that do NOT depend on (c0,c1).
        cache_key = ("dA_pre", int(A), float(alpha_fm2), float(Npar), float(self.bmax_fm))
        cached = self._cache_dA_binomial.get(cache_key)
        if cached is None:
            rmax = 20.0
            dr = 0.05
            r_grid, Pd = self._prepare_Pd_lookup(rmax=rmax, dr=dr)
            rI, I = self._tabulate_I_of_r(rmax=rmax, dr=dr, alpha_fm2=alpha_fm2, Npar=Npar)
            if len(rI) != len(r_grid) or abs((rI[1] - rI[0]) - (r_grid[1] - r_grid[0])) > 1e-12:
                I = np.interp(r_grid, rI, I)

            Nb = 700
            b = np.linspace(0.0, self.bmax_fm, Nb)
            db = b[1] - b[0]
            TA_b = np.maximum(self.TA_r(b), 0.0)

            logC = np.array(
                [math.lgamma(A + 1) - math.lgamma(N + 1) - math.lgamma(A - N + 1) for N in range(A + 1)],
                float,
            )
            sigmaN = np.zeros(A + 1, float)

            jac_b = 2.0 * math.pi * b * db
            jac_r = 2.0 * math.pi * r_grid * (r_grid[1] - r_grid[0])

            # angle-average between b and r
            nphi = 32
            phi_nodes, phi_w = _leggauss(nphi)
            pm, pc = math.pi, math.pi
            phi = pc + pm * phi_nodes
            wphi = pm * phi_w  # integrates [0,2π]

            def TA_shift_avg(bmag: float, rmag: float) -> Tuple[float, float]:
                rp2 = (0.5 * rmag) ** 2
                m1 = bmag * bmag + rp2 + bmag * rmag * np.cos(phi)
                m2 = bmag * bmag + rp2 - bmag * rmag * np.cos(phi)
                t1 = self.TA_r(np.sqrt(np.maximum(m1, 0.0)))
                t2 = self.TA_r(np.sqrt(np.maximum(m2, 0.0)))
                avg1 = float(np.sum(wphi * t1) / (2.0 * math.pi))
                avg2 = float(np.sum(wphi * t2) / (2.0 * math.pi))
                return avg1, avg2

            # build p_dA(b,r)
            p_dA = np.zeros((Nb, len(r_grid)), float)
            for ib, bmag in enumerate(b):
                TA0 = float(TA_b[ib])
                for ir, rmag in enumerate(r_grid):
                    tplus, tminus = TA_shift_avg(float(bmag), float(rmag))
                    p1 = self.sigma_nn_fm2 * tplus / float(A)
                    p2 = self.sigma_nn_fm2 * tminus / float(A)
                    p2_term = (TA0 * float(I[ir])) / float(A)
                    p = p1 + p2 - p2_term
                    p_dA[ib, ir] = float(np.clip(p, 0.0, 1.0 - 1e-12))

            # Vectorized sigmaN calculation
            # p_dA is (Nb, Nr)
            # logC is (A+1,)
            # jac_b (Nb,), jac_r (Nr,), Pd (Nr,)
            
            # Reduce jac_r * Pd to single vector
            w_r = jac_r * Pd  # (Nr,)
            
            for N in range(1, A + 1):
                lnC = logC[N]
                # Calculate wN for all b, r
                # Broadcasting: (Nb, Nr)
                ln = lnC + N * np.log(p_dA) + (A - N) * np.log1p(-p_dA)
                wN = np.exp(np.clip(ln, -800.0, 50.0))
                
                # Inner integral over r: sum(wN * w_r, axis=1) -> (Nb,)
                inner = np.sum(wN * w_r[None, :], axis=1)
                
                # Outer integral over b: sum(jac_b * inner)
                sigmaN[N] = float(np.sum(jac_b * inner))

            sig_inel = float(np.sum(sigmaN[1:]))
            if sig_inel <= 0:
                return float(Lp_fm)

            tail = 0.0
            F = np.zeros(A + 2, float)
            for N in range(A, 0, -1):
                tail += sigmaN[N]
                F[N] = tail / sig_inel
            F[A + 1] = 0.0

            cached = (r_grid, Pd, I, b, TA_b, logC, sigmaN, F, jac_b, jac_r, sig_inel, p_dA)
            self._cache_dA_binomial[cache_key] = cached
        else:
            (r_grid, Pd, I, b, TA_b, logC, sigmaN, F, jac_b, jac_r, sig_inel, p_dA) = cached
        
        Nb = len(b)

        def N_from_c(c: float) -> int:
            c = float(np.clip(c, 0.0, 1.0))
            if c <= 0.0:
                return A+1
            if c >= 1.0:
                return 1
            for N in range(A, 0, -1):
                if F[N] >= c and F[N+1] < c:
                    return N
            return 1

        N1 = max(1, N_from_c(c1))
        N2 = min(A, (A if c0 <= 0.0 else N_from_c(c0)-1))
        if N1 > N2:
            return float(Lp_fm)

        # Eq.(LeffRhic) numerator/denominator
        Num = 0.0
        Den = 0.0
        
        # Vectorized L_eff loop
        w_r = jac_r * Pd  # (Nr,)
        
        # We need to iterate from N1 to N2.
        # This is fast if just adding scalars, but we need to integrate over b, r again.
        
        # Precompute TA term
        # (Nb,)
        TA0 = TA_b
        # Mask for TA0 > 0 already handled implicitly as p_dA would be small, but let's be safe
        
        term_num_b = np.zeros(Nb, float)
        term_den_b = np.zeros(Nb, float)

        for N in range(N1, N2+1):
            lnC = logC[N]
            
            # (Nb, Nr)
            ln = lnC + N * np.log(p_dA) + (A - N) * np.log1p(-p_dA)
            
            # Need N-2 and N-1 powers. 
            # w_num corresponds to p^(N-2) * (1-p)^(A-N)
            # w_den corresponds to p^(N-1) * (1-p)^(A-N)
            # Our precomputed 'ln' is for p^N. 
            # So ln_num = ln - 2*log(p)
            # ln_den = ln - 1*log(p)
            
            # Note: handle p=0 case. p_dA is clipped to >0 for logs if needed, but if p_dA is miniscule...
            # The original code did:
            # ln_num = lnC + (N-2)*np.log(pb) + (A-N)*np.log1p(-pb)
            # This is safer than subtracting from 'ln' if p is very small.
            
            ln_common = lnC + (A-N)*np.log1p(-p_dA) # (Nb, Nr)
            ln_p = np.log(p_dA)
            
            ln_num = ln_common + (N-2)*ln_p
            ln_den = ln_common + (N-1)*ln_p
            
            w_num = np.exp(np.clip(ln_num, -800.0, 50.0))
            w_den = np.exp(np.clip(ln_den, -800.0, 50.0))
            
            # Integrate over r: (Nb,)
            I_num_r = np.sum(w_num * w_r[None, :], axis=1)
            I_den_r = np.sum(w_den * w_r[None, :], axis=1)
            
            # Add to accumulators (weighted by jac_b and TA0 terms)
            # Contribution to Num: N(N-1) * integral(TA^2 * ...)
            # Contribution to Den: N * integral(TA * ...)
            
            term_num_b += (N*(N-1)) * (TA0*TA0 * I_num_r)
            term_den_b += N * (TA0 * I_den_r)

        # Final integrate over b
        TotalNum = np.sum(jac_b * term_num_b)
        TotalDen = np.sum(jac_b * term_den_b)

        if TotalDen <= 0:
            return float(Lp_fm)
        return float(max(Lp_fm, Lp_fm + TotalNum/(rho0_fm3*TotalDen)))

    # =============================================================================
    # Backwards-compatible helpers expected by some notebooks/modules
    # =============================================================================

    @staticmethod
    def _parse_cent_bins(
        cent: Sequence[float] | Sequence[Tuple[float, float]]
    ) -> Tuple[List[Tuple[float, float]], List[str]]:
        """Accept either edges [0,10,20,...] or explicit bins [(0,10),(10,20),...]."""
        if len(cent) == 0:
            return [], []
        first = cent[0]
        bins: List[Tuple[float, float]]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            bins = [(float(a), float(b)) for (a, b) in cent]  # type: ignore[arg-type]
        else:
            edges = [float(x) for x in cent]  # type: ignore[arg-type]
            bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        labels = [f"{int(round(a))}-{int(round(b))}%" for (a, b) in bins]
        return bins, labels

    def leff_bins_pA(
        self,
        cent: Sequence[float] | Sequence[Tuple[float, float]],
        *,
        method: Literal["binomial", "optical"] = "binomial",
        rho0_fm3: float = DEFAULT_RHO0,
        Lp_fm: float = DEFAULT_LP_FM,
    ) -> Dict[str, float]:
        """Return {"0-10%": L_eff, ...} for pA bins (percent)."""
        bins, labels = self._parse_cent_bins(cent)
        out: Dict[str, float] = {}
        for (a, b), lab in zip(bins, labels):
            out[lab] = float(
                self.leff_bin_pA(
                    a / 100.0,
                    b / 100.0,
                    rho0_fm3=rho0_fm3,
                    Lp_fm=Lp_fm,
                    method=method,
                )
            )
        return out

    def leff_bins_dA(
        self,
        cent: Sequence[float] | Sequence[Tuple[float, float]],
        *,
        method: Literal["binomial", "optical"] = "binomial",
        rho0_fm3: float = DEFAULT_RHO0,
        Lp_fm: float = DEFAULT_LP_FM,
        alpha_fm2: float = 1.05,
        Npar: float = 1.1,
    ) -> Dict[str, float]:
        """Return {"0-10%": L_eff, ...} for dA bins (percent)."""
        bins, labels = self._parse_cent_bins(cent)
        out: Dict[str, float] = {}
        for (a, b), lab in zip(bins, labels):
            out[lab] = float(
                self.leff_bin_dA(
                    a / 100.0,
                    b / 100.0,
                    rho0_fm3=rho0_fm3,
                    Lp_fm=Lp_fm,
                    method=method,
                    alpha_fm2=alpha_fm2,
                    Npar=Npar,
                )
            )
        return out

    # =============================================================================
    # Convenience printers
    # =============================================================================

    def print_table_pA(self, edges_percent: Sequence[float], *, method_leff: Literal["binomial","optical"]="binomial") -> None:
        print("--------------------------------------------------------------------------")
        print(f"pA centrality |  <Ncoll> (optical)  |  L_eff [fm]   (L_eff: {method_leff})")
        print("--------------------------------------------------------------------------")
        for i in range(len(edges_percent)-1):
            a = edges_percent[i]/100.0
            b = edges_percent[i+1]/100.0
            ncoll = self.ncoll_mean_bin_pA_optical(a,b)
            leff  = self.leff_bin_pA(a,b,method=method_leff)
            print(f" {edges_percent[i]:>3.0f}-{edges_percent[i+1]:<3.0f}%    | "
                  f"{ncoll:10.3f}        | {leff:8.3f}")
        print("--------------------------------------------------------------------------")

    def print_table_dA(self, edges_percent: Sequence[float], *, method_leff: Literal["binomial","optical"]="binomial") -> None:
        print("-------------------------------------------------------------------------------")
        print(f"dA centrality |  <Ncoll> (optical)  |  <Npart> (optical)  |  L_eff [fm] ({method_leff})")
        print("-------------------------------------------------------------------------------")
        for i in range(len(edges_percent)-1):
            a = edges_percent[i]/100.0
            b = edges_percent[i+1]/100.0
            ncoll = self.ncoll_mean_bin_dA_optical(a,b)
            npart = self.npart_mean_bin_dA_optical(a,b)
            leff  = self.leff_bin_dA(a,b,method=method_leff)
            print(f" {edges_percent[i]:>3.0f}-{edges_percent[i+1]:<3.0f}%    | "
                  f"{ncoll:10.3f}        | {npart:10.3f}        | {leff:8.3f}")
        print("-------------------------------------------------------------------------------")

# =============================================================================
# Monte Carlo Glauber (minimal)
# =============================================================================

@dataclass(frozen=True)
class MCConfig:
    n_events: int = 20000
    bmax_fm: float = 20.0
    sigma_nn_mb: float = 42.0
    A: int = 197
    d_fm: float = 0.535
    rho0: float = DEFAULT_RHO0
    seed: int = 12345
    hulthen_alpha: float = 0.228
    hulthen_beta: float = 1.18

@dataclass(frozen=True)
class HulthenWavefunction:
    alpha: float = 0.228
    beta: float = 1.18

    def pdf_r(self, r: np.ndarray) -> np.ndarray:
        a, b = self.alpha, self.beta
        N2 = (a*b*(a+b)) / (2.0*math.pi*(b-a)**2)
        r = np.asarray(r, float)
        r_safe = np.where(r < 1e-12, 1e-12, r)
        psi2 = N2 * (np.exp(-a*r_safe) - np.exp(-b*r_safe))**2 / (r_safe*r_safe)
        return 4.0*math.pi*r*r*psi2

    def sample_r(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        rmax = 30.0
        rg = np.linspace(0.0, rmax, 5000)
        pg = self.pdf_r(rg)
        pmax = float(pg.max()) * 1.05
        out = []
        while len(out) < n:
            r = float(rng.uniform(0.0, rmax))
            u = float(rng.uniform(0.0, pmax))
            if u <= float(self.pdf_r(np.array([r]))[0]):
                out.append(r)
        return np.array(out, float)

class MonteCarloGlauber:
    def __init__(self, cfg: MCConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.seed))
        self.sigma_nn_fm2 = cfg.sigma_nn_mb * MB_TO_FM2
        self.d_max = math.sqrt(self.sigma_nn_fm2 / math.pi)
        self.ws = WoodsSaxon(A=cfg.A, rho0=cfg.rho0, d_fm=cfg.d_fm)
        self.hul = HulthenWavefunction(alpha=cfg.hulthen_alpha, beta=cfg.hulthen_beta)

    def _sample_WS_positions(self) -> np.ndarray:
        A = self.cfg.A
        Rmax = self.ws.radius_rn() + 10.0*self.ws.d_fm
        rg = np.linspace(0.0, Rmax, 2000)
        fg = rg*rg*self.ws.rho(rg)
        fmax = float(fg.max())*1.05

        out = np.empty((A,3), float)
        n = 0
        while n < A:
            r = float(self.rng.uniform(0.0, Rmax))
            u = float(self.rng.uniform(0.0, fmax))
            if u > r*r*float(self.ws.rho(np.array([r]))[0]):
                continue
            cost = float(self.rng.uniform(-1.0, 1.0))
            sint = math.sqrt(max(0.0, 1.0-cost*cost))
            phi = float(self.rng.uniform(0.0, 2.0*math.pi))
            out[n] = (r*sint*math.cos(phi), r*sint*math.sin(phi), r*cost)
            n += 1
        return out

    def _sample_deuteron_transverse(self) -> np.ndarray:
        r = float(self.hul.sample_r(self.rng, 1)[0])
        cost = float(self.rng.uniform(-1.0, 1.0))
        sint = math.sqrt(max(0.0, 1.0-cost*cost))
        phi = float(self.rng.uniform(0.0, 2.0*math.pi))
        rx = r*sint*math.cos(phi)
        ry = r*sint*math.sin(phi)
        return np.array([[+0.5*rx, +0.5*ry], [-0.5*rx, -0.5*ry]], float)

    def run(self, system: Literal["pA","dA"]="dA") -> Dict[str, np.ndarray]:
        cfg = self.cfg
        n = int(cfg.n_events)
        bmax = float(cfg.bmax_fm)

        Ncoll = np.zeros(n, int)
        Npart = np.zeros(n, int)
        bvals = np.zeros(n, float)

        for ievt in range(n):
            u = float(self.rng.uniform(0.0, 1.0))
            b = bmax*math.sqrt(u)   # uniform in area
            bvals[ievt] = b

            targ = self._sample_WS_positions()
            targ_xy = targ[:, :2]

            if system == "pA":
                proj_xy = np.array([[b, 0.0]], float)
            else:
                rel = self._sample_deuteron_transverse()
                proj_xy = rel + np.array([b, 0.0])[None, :]

            diff = proj_xy[:, None, :] - targ_xy[None, :, :]
            d2 = np.sum(diff*diff, axis=-1)
            hit = (d2 < self.d_max*self.d_max)

            Ncoll[ievt] = int(np.sum(hit))
            targ_part = np.any(hit, axis=0)
            proj_part = np.any(hit, axis=1)
            Npart[ievt] = int(np.sum(targ_part) + np.sum(proj_part))

        return dict(b=bvals, Ncoll=Ncoll, Npart=Npart)

    @staticmethod
    def centrality_slices(values: np.ndarray, edges_percent: Sequence[float]) -> List[np.ndarray]:
        vals = np.asarray(values)
        order = np.argsort(vals)[::-1]  # central = larger value
        n = len(vals)
        masks = []
        for i in range(len(edges_percent)-1):
            a = edges_percent[i]/100.0
            b = edges_percent[i+1]/100.0
            i0 = int(round(a*n))
            i1 = int(round(b*n))
            sel = order[i0:i1]
            m = np.zeros(n, bool)
            m[sel] = True
            masks.append(m)
        return masks

    @staticmethod
    def mean_in_bins(values: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        vals = np.asarray(values)
        out = np.zeros(len(masks), float)
        for i, m in enumerate(masks):
            out[i] = float(vals[m].mean()) if np.any(m) else 0.0
        return out
