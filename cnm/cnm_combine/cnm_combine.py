# cnm_combine.py
"""
Robust CNM combination module: nPDF × (eloss × Cronin pT broadening)
vs y, vs pT, vs centrality – closely mirroring the original notebook.

Key features
------------
* Handles both 5.02 and 8.16 TeV.
* Centrality bins, y-edges, pT-edges, y-windows can be given from outside.
* Uses the existing npdf_centrality + eloss_cronin_centrality modules.
* Returns clean dict-of-arrays results so you can:
    - save to CSV via helpers, and
    - reuse in later calculations (e.g. multiply with primordial bands).
* Fixes the Rb_hi bug that caused the UnboundLocalError.

Exports
-------
- CNMCombine
- DEFAULT_CENT_BINS, DEFAULT_Y_EDGES, DEFAULT_P_EDGES,
  DEFAULT_Y_WINDOWS, DEFAULT_PT_RANGE_AVG
- combine_two_bands_1d
- cnm_vs_y_to_dataframe, cnm_vs_pT_to_dataframe, cnm_vs_cent_to_dataframe
- demo_plots() (optional quick-look plotting)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
import sys

# ------------------------------------------------------------
# Paths / imports
# ------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

NPDF_CODE_DIR = ROOT / "npdf_code"
ELOSS_CODE_DIR = ROOT / "eloss_code"

if str(NPDF_CODE_DIR) not in sys.path:
    sys.path.append(str(NPDF_CODE_DIR))
if str(ELOSS_CODE_DIR) not in sys.path:
    sys.path.append(str(ELOSS_CODE_DIR))

from npdf_data import NPDFSystem, RpAAnalysis  # type: ignore
from gluon_ratio import EPPS21Ratio, GluonEPPSProvider  # type: ignore
from glauber import OpticalGlauber, SystemSpec  # type: ignore
from npdf_centrality import (  # type: ignore
    compute_df49_by_centrality,
    make_centrality_weight_dict,
    bin_rpa_vs_y,
    bin_rpa_vs_pT,
    bin_rpa_vs_centrality,
)

from particle import Particle  # type: ignore
from coupling import alpha_s_provider  # type: ignore
import quenching_fast as QF  # type: ignore
# Use the FAST vectorized test module as requested
import eloss_cronin_centrality_test as EC  # type: ignore

# ------------------------------------------------------------
# nPDF input locations
# ------------------------------------------------------------
NPDF_INPUT_DIR = ROOT / "input" / "npdf"
P5_DIR = NPDF_INPUT_DIR / "pPb5TeV"
P8_DIR = NPDF_INPUT_DIR / "pPb8TeV"
DAU_DIR = NPDF_INPUT_DIR / "dAu200GeV"
EPPS_DIR = NPDF_INPUT_DIR / "nPDFs"

SQRTS = {"200": 200.0, "5.02": 5023.0, "8.16": 8160.0}
SIG_NN = {"200": 42.0, "5.02": 67.0, "8.16": 71.0}

# ------------------------------------------------------------
# Defaults (centrality, y, pT): LHC
# ------------------------------------------------------------
DEFAULT_CENT_BINS: Sequence[Tuple[float, float]] = [
    (0, 20),
    (20, 40),
    (40, 60),
    (60, 80),
    (80, 100),
]

# CMS-like y windows (you can override in notebook)
DEFAULT_Y_WINDOWS: Sequence[Tuple[float, float, str]] = [
    (-4.46, -2.96, "-4.46 < y < -2.96"),
    (-1.37, 0.43, "-1.37 < y < 0.43"),
    (2.03, 3.53, "2.03 < y < 3.53"),
]

# Binning grids
DEFAULT_PT_RANGE: Tuple[float, float] = (0.0, 20.0)
DEFAULT_PT_RANGE_AVG: Tuple[float, float] = (0.0, 15.0)
DEFAULT_PT_FLOOR_W: float = 1.0

DEFAULT_Y_EDGES: np.ndarray = np.arange(-5.0, 5.0 + 0.25, 0.5)
DEFAULT_P_EDGES: np.ndarray = np.arange(0.0, 20.0 + 2.5, 2.5)

# Weighting & centrality
DEFAULT_WEIGHT_MODE: str = "pp@local"
DEFAULT_Y_REF: float = 0.0
DEFAULT_CENT_EXP_C0: float = 0.25  # parameter in exp-weights for centrality

# eloss / broadening parameter bands
DEFAULT_Q0_PAIR: Tuple[float, float] = (0.05, 0.09)
DEFAULT_P0_SCALE_PAIR: Tuple[float, float] = (0.9, 1.1)

DEFAULT_NB_BSAMPLES: int = 5
DEFAULT_Y_SHIFT_FRACTION: float = 2.0


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def combine_two_bands_1d(
    R1_c,
    R1_lo,
    R1_hi,
    R2_c,
    R2_lo,
    R2_hi,
    eps: float = 1e-12,
):
    """
    Multiply two RpA bands: R_tot = R1 * R2 with standard
    quadrature error propagation on relative uncertainties.
    Handles asymmetric bands by propagating low and high deviations separately.
    """
    R1_c = np.asarray(R1_c, float)
    R1_lo = np.asarray(R1_lo, float)
    R1_hi = np.asarray(R1_hi, float)

    R2_c = np.asarray(R2_c, float)
    R2_lo = np.asarray(R2_lo, float)
    R2_hi = np.asarray(R2_hi, float)

    Rc = np.full_like(R1_c, np.nan, dtype=float)
    Rlo = np.full_like(R1_c, np.nan, dtype=float)
    Rhi = np.full_like(R1_c, np.nan, dtype=float)

    mask = np.isfinite(R1_c) & np.isfinite(R2_c)
    if not np.any(mask):
        return Rc, Rlo, Rhi

    # Fractional deviations
    R1_safe = np.where(np.abs(R1_c) > eps, R1_c, np.nan)
    R2_safe = np.where(np.abs(R2_c) > eps, R2_c, np.nan)

    # Relative errors on low side
    rel_lo_1 = (R1_c - R1_lo) / R1_safe
    rel_lo_2 = (R2_c - R2_lo) / R2_safe
    rel_lo_tot = np.sqrt(np.maximum(0, rel_lo_1**2 + rel_lo_2**2))

    # Relative errors on high side
    rel_hi_1 = (R1_hi - R1_c) / R1_safe
    rel_hi_2 = (R2_hi - R2_c) / R2_safe
    rel_hi_tot = np.sqrt(np.maximum(0, rel_hi_1**2 + rel_hi_2**2))

    R_tot_c = R1_c * R2_c
    
    Rc[mask] = R_tot_c[mask]
    Rlo[mask] = R_tot_c[mask] * (1.0 - rel_lo_tot[mask])
    Rhi[mask] = R_tot_c[mask] * (1.0 + rel_hi_tot[mask])
    
    return Rc, Rlo, Rhi


def _tags_for_cent_bins(
    cent_bins: Sequence[Tuple[float, float]],
    include_mb: bool = True,
) -> Sequence[str]:
    tags = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]
    if include_mb:
        tags.append("MB")
    return tags


# ------------------------------------------------------------
# Main container
# ------------------------------------------------------------
@dataclass
class CNMCombine:
    # configuration
    energy: str
    family: str
    particle_state: str

    sqrt_sNN: float
    sigma_nn_mb: float

    cent_bins: Sequence[Tuple[float, float]]
    y_edges: np.ndarray
    p_edges: np.ndarray
    y_windows: Sequence[Tuple[float, float, str]]
    pt_range_avg: Tuple[float, float]
    pt_floor_w: float

    weight_mode: str
    y_ref: float
    cent_c0: float

    q0_pair: Tuple[float, float]
    p0_scale_pair: Tuple[float, float]
    nb_bsamples: int
    y_shift_fraction: float

    # derived
    particle: Particle
    npdf_ctx: Dict[str, object]
    gl: OpticalGlauber
    qp_base: object
    spec: SystemSpec

    # -------------------------
    # Constructor from defaults
    # -------------------------
    @classmethod
    def from_defaults(
        cls,
        energy: Literal["200", "5.02", "8.16"] = "8.16",
        family: Literal["charmonia", "bottomonia"] = "charmonia",
        particle_state: str = "avg",
        cent_bins: Sequence[Tuple[float, float]] = None,
        y_edges: np.ndarray = None,
        p_edges: np.ndarray = None,
        y_windows: Sequence[Tuple[float, float, str]] = None,
        pt_range_avg: Tuple[float, float] = None,
        pt_floor_w: float = DEFAULT_PT_FLOOR_W,
        weight_mode: str = DEFAULT_WEIGHT_MODE,
        y_ref: float = DEFAULT_Y_REF,
        cent_c0: float = DEFAULT_CENT_EXP_C0,
        q0_pair: Tuple[float, float] = DEFAULT_Q0_PAIR,
        p0_scale_pair: Tuple[float, float] = DEFAULT_P0_SCALE_PAIR,
        nb_bsamples: int = DEFAULT_NB_BSAMPLES,
        y_shift_fraction: float = DEFAULT_Y_SHIFT_FRACTION,
    ) -> "CNMCombine":

        energy = str(energy)
        if energy not in SQRTS:
            raise ValueError("energy must be '5.02' or '8.16'")

        sqrt_sNN = SQRTS[energy]
        sigma_nn_mb = SIG_NN[energy]

        if cent_bins is None:
            cent_bins = DEFAULT_CENT_BINS
        if y_edges is None:
            y_edges = DEFAULT_Y_EDGES
        if p_edges is None:
            p_edges = DEFAULT_P_EDGES
        if y_windows is None:
            y_windows = DEFAULT_Y_WINDOWS
        if pt_range_avg is None:
            if energy == "200":
                pt_range_avg = (0.0, 5.0)
            else:
                pt_range_avg = (0.0, 10.0)

        # quarkonium state (family + 'avg' or specific)
        particle = Particle(family=family, state=particle_state)

        # nPDF input dir
        if energy == "200":
            input_dir = DAU_DIR
            target_A = 197
            system_name = "dA"
        elif energy == "5.02":
            input_dir = P5_DIR
            target_A = 208
            system_name = "pA"
        else:
            input_dir = P8_DIR
            target_A = 208
            system_name = "pA"

        # ----------------------------------------------------
        # nPDF side: GluonEPPSProvider + RpAAnalysis
        # ----------------------------------------------------
        if family == "charmonia":
            m_state_for_np = "charmonium"
        elif family == "bottomonia":
            m_state_for_np = "bottomonium"
        else:
            # fallback: numeric mass if family name is non-standard
            m_state_for_np = particle.M_GeV

        epps_ratio = EPPS21Ratio(A=target_A, path=str(EPPS_DIR))
        gluon = GluonEPPSProvider(
            epps_ratio,
            sqrt_sNN_GeV=sqrt_sNN,
            m_state_GeV=m_state_for_np,
            y_sign_for_xA=-1 if system_name == "pA" else +1, # Arleo-Peigne convention for xA
        )

        gl_ana = OpticalGlauber(
            SystemSpec(system_name, sqrt_sNN, A=target_A, sigma_nn_mb=sigma_nn_mb)
        )

        ana = RpAAnalysis()
        sys_npdf = NPDFSystem.from_folder(
            str(input_dir),
            kick="pp",
            name=f"{system_name} {energy} GeV" if energy == "200" else f"p+Pb {energy} TeV",
        )

        base, r0, M = ana.compute_rpa_members(
            sys_npdf.df_pp,
            sys_npdf.df_pa,
            sys_npdf.df_errors,
            join="intersect",
            lowpt_policy="drop",
            pt_shift_min=pt_floor_w,
            shift_if_r_below=0.0,
        )

        # dAu specific: ensure we use the correct weight mode
        # For eloss code, we might need 'optical' weights if 'exp' is not calibrated for dAu
        # But here we default to what was requested.
        
        df49_by_cent, K_by_cent, SA_all, Y_SHIFT = compute_df49_by_centrality(
            base,
            r0,
            M,
            gluon,
            gl_ana,
            cent_bins=cent_bins,
            nb_bsamples=nb_bsamples,
            y_shift_fraction=y_shift_fraction,
        )

        npdf_ctx = dict(
            df49_by_cent=df49_by_cent,
            df_pp=sys_npdf.df_pp,
            df_pa=sys_npdf.df_pa,
            gluon=gluon,
        )

        # ----------------------------------------------------
        # eloss + Cronin: QF.QuenchParams, alpha_s, etc.
        # ----------------------------------------------------
        alpha_s = alpha_s_provider(mode="running", LambdaQCD=0.25)
        gl_eloss = gl_ana
        Lmb = gl_eloss.leff_minbias_pA() if system_name == "pA" else gl_eloss.leff_minbias_dA()

        device = "cpu"
        try:
            import torch  # type: ignore

            if QF._HAS_TORCH and torch.cuda.is_available():  # type: ignore
                device = "cuda"
        except Exception:
            pass

        qp_base = QF.QuenchParams(
            qhat0=0.075,
            lp_fm=1.5,
            LA_fm=Lmb,
            LB_fm=1.5 if system_name == "pA" else Lmb, # For dA, projectile (d) also has L_eff
            system=system_name if energy == "200" else "pPb",
            lambdaQCD=0.25,
            roots_GeV=sqrt_sNN,
            alpha_of_mu=alpha_s,
            alpha_scale="mT",
            use_hard_cronin=True,
            mapping="exp",
            device=device,
        )

        return cls(
            energy=energy,
            family=family,
            particle_state=particle_state,
            sqrt_sNN=sqrt_sNN,
            sigma_nn_mb=sigma_nn_mb,
            cent_bins=cent_bins,
            y_edges=np.asarray(y_edges, float),
            p_edges=np.asarray(p_edges, float),
            y_windows=y_windows,
            pt_range_avg=pt_range_avg,
            pt_floor_w=pt_floor_w,
            weight_mode=weight_mode,
            y_ref=y_ref,
            cent_c0=cent_c0,
            q0_pair=q0_pair,
            p0_scale_pair=p0_scale_pair,
            nb_bsamples=nb_bsamples,
            y_shift_fraction=y_shift_fraction,
            particle=particle,
            npdf_ctx=npdf_ctx,
            gl=gl_eloss,
            qp_base=qp_base,
            spec=gl_eloss.spec,
        )

    @property
    def P(self):
        return self.particle

    @property
    def roots(self):
        return self.sqrt_sNN

    @property
    def qp(self):
        return self.qp_base

    def _calc_eloss_broad_band_vs_centrality(self, cent_bins, y_window, pt_range):
        """Helper for notebooks to get raw band data using FAST module."""
        return EC.rpa_band_vs_centrality(
            self.particle,
            self.sqrt_sNN,
            self.qp_base,
            self.gl,
            cent_bins,
            y_window,
            pt_range,
            q0_pair=self.q0_pair,
            p0_scale_pair=self.p0_scale_pair,
            Ny_bin=16,
            Npt_bin=32,
            weight_kind="pp",
            weight_ref_y="local",
        )

    # --------------------------------------------------------
    # RpA vs rapidity
    # --------------------------------------------------------
    def cnm_vs_y(
        self,
        y_edges: Optional[np.ndarray] = None,
        pt_range_avg: Optional[Tuple[float, float]] = None,
        include_mb: bool = True,
        components: Sequence[str] = ("npdf", "eloss", "broad", "eloss_broad", "cnm"),
        mb_weight_mode: str = "exp",
        mb_c0: Optional[float] = None,
    ):
        if y_edges is None:
            y_edges = self.y_edges
        if pt_range_avg is None:
            pt_range_avg = self.pt_range_avg
        if mb_c0 is None:
            mb_c0 = self.cent_c0

        # MB weights for nPDF side
        wcent = (
            make_centrality_weight_dict(self.cent_bins, c0=mb_c0)
            if include_mb
            else None
        )

        # nPDF-only RpA vs y
        npdf_bins_y = bin_rpa_vs_y(
            self.npdf_ctx["df49_by_cent"],
            self.npdf_ctx["df_pp"],
            self.npdf_ctx["df_pa"],
            self.npdf_ctx["gluon"],
            cent_bins=self.cent_bins,
            y_edges=y_edges,
            pt_range_avg=pt_range_avg,
            weight_mode=self.weight_mode,
            y_ref=self.y_ref,
            pt_floor_w=self.pt_floor_w,
            wcent_dict=wcent,
            include_mb=include_mb,
        )

        y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])

        # eloss + broad + total vs y using FAST module
        y_cent_eloss, bands_y, labels_y = EC.rpa_band_vs_y(
            self.particle,
            self.sqrt_sNN,
            self.qp_base,
            self.gl,
            self.cent_bins,
            y_edges,
            pt_range_avg,
            components=("loss", "broad", "eloss_broad"),
            q0_pair=self.q0_pair,
            p0_scale_pair=self.p0_scale_pair,
            Ny_bin=12,
            Npt_bin=24,
            weight_kind="pp",
            weight_ref_y="local",
            table_for_pp=None,
        )

        if not np.allclose(y_cent, y_cent_eloss):
            raise RuntimeError("y grid mismatch between nPDF and eloss bands")

        RL_c, RL_lo, RL_hi = bands_y["loss"]
        RB_c, RB_lo, RB_hi = bands_y["broad"]
        RT_c, RT_lo, RT_hi = bands_y["eloss_broad"]

        tags = _tags_for_cent_bins(self.cent_bins, include_mb=include_mb)

        cnm_y: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            comp: {} for comp in components
        }

        for tag in tags:
            # nPDF
            npdf_data = npdf_bins_y[tag]
            Rn_c = np.asarray(npdf_data["r_central"], float)
            Rn_lo = np.asarray(npdf_data["r_lo"], float)
            Rn_hi = np.asarray(npdf_data["r_hi"], float)

            # eloss
            Rloss_c = np.asarray(RL_c[tag], float)
            Rloss_lo = np.asarray(RL_lo[tag], float)
            Rloss_hi = np.asarray(RL_hi[tag], float)

            # broadening (BUG FIX: use RB_hi here)
            Rb_c = np.asarray(RB_c[tag], float)
            Rb_lo = np.asarray(RB_lo[tag], float)
            Rb_hi = np.asarray(RB_hi[tag], float)

            # total eloss×broad
            Rtot_c = np.asarray(RT_c[tag], float)
            Rtot_lo = np.asarray(RT_lo[tag], float)
            Rtot_hi = np.asarray(RT_hi[tag], float)

            # combined CNM = nPDF × total (total is eloss x pT broad)
            Rcnm_c, Rcnm_lo, Rcnm_hi = combine_two_bands_1d(
                Rn_c, Rn_lo, Rn_hi,
                Rtot_c, Rtot_lo, Rtot_hi,
            )

        # Re-structure into canonical bands format:
        # dict[comp] -> (Rc_dict, Rlo_dict, Rhi_dict)
        final_bands = {}
        for comp in components:
            Dc, Dlo, Dhi = {}, {}, {}
            for tag in tags:
                val = cnm_y[comp][tag]
                Dc[tag], Dlo[tag], Dhi[tag] = val
            final_bands[comp] = (Dc, Dlo, Dhi)

        return y_cent, tags, final_bands

    # --------------------------------------------------------
    # RpA vs pT: binned + bands
    # --------------------------------------------------------
    def cnm_vs_pT(
        self,
        y_window: Tuple[float, float] | Tuple[float, float, str],
        pt_edges: Optional[np.ndarray] = None,
        components: Sequence[str] = ("npdf", "eloss", "broad", "eloss_broad", "cnm"),
        include_mb: bool = True,
        mb_weight_mode: str = "exp",
        mb_c0: Optional[float] = None,
    ):
        if len(y_window) == 3:
            y0, y1, _ = y_window
        else:
            y0, y1 = y_window

        if pt_edges is None:
            pt_edges = self.p_edges
        if mb_c0 is None:
            mb_c0 = self.cent_c0

        wcent = (
            make_centrality_weight_dict(self.cent_bins, c0=mb_c0)
            if include_mb
            else None
        )

        # nPDF vs pT
        npdf_bins_pt = bin_rpa_vs_pT(
            self.npdf_ctx["df49_by_cent"],
            self.npdf_ctx["df_pp"],
            self.npdf_ctx["df_pa"],
            self.npdf_ctx["gluon"],
            cent_bins=self.cent_bins,
            pt_edges=pt_edges,
            y_window=(y0, y1),
            weight_mode=self.weight_mode,
            y_ref=self.y_ref,
            pt_floor_w=self.pt_floor_w,
            wcent_dict=wcent,
            include_mb=include_mb,
        )

        pT_cent = 0.5 * (pt_edges[:-1] + pt_edges[1:])

        # eloss + broad + total vs pT using FAST module
        pT_cent_eloss, bands_pt, labels_pt = EC.rpa_band_vs_pT(
            self.particle,
            self.sqrt_sNN,
            self.qp_base,
            self.gl,
            self.cent_bins,
            pt_edges,
            (y0, y1),
            components=("loss", "broad", "eloss_broad"),
            q0_pair=self.q0_pair,
            p0_scale_pair=self.p0_scale_pair,
            Ny_bin=12,
            Npt_bin=24,
            weight_kind="pp",
            weight_ref_y="local",
            table_for_pp=None,
        )

        if not np.allclose(pT_cent, pT_cent_eloss):
            raise RuntimeError("pT grid mismatch between nPDF and eloss bands")

        RL_c, RL_lo, RL_hi = bands_pt["loss"]
        RB_c, RB_lo, RB_hi = bands_pt["broad"]
        RT_c, RT_lo, RT_hi = bands_pt["eloss_broad"]

        tags = _tags_for_cent_bins(self.cent_bins, include_mb=include_mb)

        cnm_pt: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            comp: {} for comp in components
        }

        for tag in tags:
            d = npdf_bins_pt[tag]
            Rn_c = np.asarray(d["r_central"], float)
            Rn_lo = np.asarray(d["r_lo"], float)
            Rn_hi = np.asarray(d["r_hi"], float)

            Rloss_c = np.asarray(RL_c[tag], float)
            Rloss_lo = np.asarray(RL_lo[tag], float)
            Rloss_hi = np.asarray(RL_hi[tag], float)

            # BUG FIX here too: use RB_hi
            Rb_c = np.asarray(RB_c[tag], float)
            Rb_lo = np.asarray(RB_lo[tag], float)
            Rb_hi = np.asarray(RB_hi[tag], float)

            Rtot_c = np.asarray(RT_c[tag], float)
            Rtot_lo = np.asarray(RT_lo[tag], float)
            Rtot_hi = np.asarray(RT_hi[tag], float)

            Rcnm_c, Rcnm_lo, Rcnm_hi = combine_two_bands_1d(
                Rn_c, Rn_lo, Rn_hi,
                Rtot_c, Rtot_lo, Rtot_hi,
            )

        # Re-structure into canonical bands format
        final_bands = {}
        for comp in components:
            Dc, Dlo, Dhi = {}, {}, {}
            for tag in tags:
                val = cnm_pt[comp][tag]
                Dc[tag], Dlo[tag], Dhi[tag] = val
            final_bands[comp] = (Dc, Dlo, Dhi)

        return pT_cent, tags, final_bands

    # --------------------------------------------------------
    # RpA vs centrality (with MB point)
    # --------------------------------------------------------
    def cnm_vs_centrality(
        self,
        y_window: Tuple[float, float] | Tuple[float, float, str],
        pt_range_avg: Optional[Tuple[float, float]] = None,
        components: Sequence[str] = ("npdf", "eloss", "broad", "eloss_broad", "cnm"),
        mb_weight_mode: str = "exp",
        mb_c0: Optional[float] = None,
    ):
        if len(y_window) == 3:
            y0, y1, _ = y_window
        else:
            y0, y1 = y_window

        if pt_range_avg is None:
            pt_range_avg = self.pt_range_avg
        if mb_c0 is None:
            mb_c0 = self.cent_c0

        # nPDF centrality-averaged result
        wcent = make_centrality_weight_dict(self.cent_bins, c0=mb_c0)
        width_weights = np.array(
            [wcent[f"{int(a)}-{int(b)}%"] for (a, b) in self.cent_bins],
            float,
        )

        npdf_cent = bin_rpa_vs_centrality(
            self.npdf_ctx["df49_by_cent"],
            self.npdf_ctx["df_pp"],
            self.npdf_ctx["df_pa"],
            self.npdf_ctx["gluon"],
            cent_bins=self.cent_bins,
            y_window=(y0, y1),
            pt_range_avg=pt_range_avg,
            weight_mode=self.weight_mode,
            y_ref=self.y_ref,
            pt_floor_w=self.pt_floor_w,
            width_weights=width_weights,
        )

        Rc_n = np.asarray(npdf_cent["r_central"], float)
        Rlo_n = np.asarray(npdf_cent["r_lo"], float)
        Rhi_n = np.asarray(npdf_cent["r_hi"], float)

        mb_n_c = float(npdf_cent["mb_r_central"])
        mb_n_lo = float(npdf_cent["mb_r_lo"])
        mb_n_hi = float(npdf_cent["mb_r_hi"])

        labels_cent = [f"{int(a)}-{int(b)}%" for (a, b) in self.cent_bins]

        # eloss + broadening centrality bands using FAST module
        (
            labels_el,
            band_loss,
            band_broad,
            band_tot
        ) = EC.rpa_band_vs_centrality(
            self.particle,
            self.sqrt_sNN,
            self.qp_base,
            self.gl,
            self.cent_bins,
            (y0, y1),
            pt_range_avg,
            q0_pair=self.q0_pair,
            p0_scale_pair=self.p0_scale_pair,
            Ny_bin=16,
            Npt_bin=32,
            weight_kind="pp",
            weight_ref_y="local",
        )
        
        # Unpack FAST module results
        # Format: (Rc, Rlo, Rhi) dictionaries
        RL_c, RL_lo, RL_hi = band_loss
        RB_c, RB_lo, RB_hi = band_broad
        RT_c, RT_lo, RT_hi = band_tot
        
        # NOTE: The FAST module does not return explicit MB tuple in the same way.
        # It includes 'MB' in the dictionaries.
        # We need to extract them.
        
        RMB_loss = (RL_c["MB"], RL_lo["MB"], RL_hi["MB"])
        RMB_broad = (RB_c["MB"], RB_lo["MB"], RB_hi["MB"])
        RMB_tot = (RT_c["MB"], RT_lo["MB"], RT_hi["MB"])

        if labels_el != labels_cent:
            raise RuntimeError("Centrality label mismatch between nPDF and eloss bands")

        Rloss_c = np.array([RL_c[lab] for lab in labels_cent], float)
        Rloss_lo = np.array([RL_lo[lab] for lab in labels_cent], float)
        Rloss_hi = np.array([RL_hi[lab] for lab in labels_cent], float)

        Rb_c = np.array([RB_c[lab] for lab in labels_cent], float)
        Rb_lo = np.array([RB_lo[lab] for lab in labels_cent], float)
        Rb_hi = np.array([RB_hi[lab] for lab in labels_cent], float)

        Rtot_c = np.array([RT_c[lab] for lab in labels_cent], float)
        Rtot_lo = np.array([RT_lo[lab] for lab in labels_cent], float)
        Rtot_hi = np.array([RT_hi[lab] for lab in labels_cent], float)

        Rc_tot_MB, Rlo_tot_MB, Rhi_tot_MB = RMB_tot

        # combined CNM vs centrality
        Rcnm_c, Rcnm_lo, Rcnm_hi = combine_two_bands_1d(
            Rc_n, Rlo_n, Rhi_n,
            Rtot_c, Rtot_lo, Rtot_hi,
        )

        # MB CNM point
        Rcnm_MB_c, Rcnm_MB_lo, Rcnm_MB_hi = combine_two_bands_1d(
            np.array([mb_n_c]),
            np.array([mb_n_lo]),
            np.array([mb_n_hi]),
            np.array([Rc_tot_MB]),
            np.array([Rlo_tot_MB]),
            np.array([Rhi_tot_MB]),
        )
        Rcnm_MB_c = float(Rcnm_MB_c[0])
        Rcnm_MB_lo = float(Rcnm_MB_lo[0])
        Rcnm_MB_hi = float(Rcnm_MB_hi[0])

        # component → (centrality array, MB triple)
        cnm_cent: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]] = {}

        if "npdf" in components:
            cnm_cent["npdf"] = (Rc_n, Rlo_n, Rhi_n, mb_n_c, mb_n_lo, mb_n_hi)
        if "eloss" in components:
            Rc_MB_loss, Rlo_MB_loss, Rhi_MB_loss = RMB_loss
            cnm_cent["eloss"] = (Rloss_c, Rloss_lo, Rloss_hi,
                                 Rc_MB_loss, Rlo_MB_loss, Rhi_MB_loss)
        if "broad" in components:
            Rc_MB_broad, Rlo_MB_broad, Rhi_MB_broad = RMB_broad
            cnm_cent["broad"] = (Rb_c, Rb_lo, Rb_hi,
                                 Rc_MB_broad, Rlo_MB_broad, Rhi_MB_broad)
        if "eloss_broad" in components:
            cnm_cent["eloss_broad"] = (Rtot_c, Rtot_lo, Rtot_hi,
                                       Rc_tot_MB, Rlo_tot_MB, Rhi_tot_MB)
        if "cnm" in components:
            cnm_cent["cnm"] = (Rcnm_c, Rcnm_lo, Rcnm_hi,
                               Rcnm_MB_c, Rcnm_MB_lo, Rcnm_MB_hi)

        return cnm_cent

    def run_and_save_production(self, outdir: Optional[Path] = None):
        """
        Runs the full analysis for y, pT, and centrality,
        saves results to CSV and generates high-quality figures.
        """
        import matplotlib.pyplot as plt
        from eloss_cronin_centrality import (
            plot_RpA_vs_y_components_per_centrality,
            plot_RpA_vs_pT_components_per_centrality,
            plot_RpA_vs_centrality_components_band
        )

        if outdir is None:
            outdir = ROOT / "outputs" / ("RHIC" if self.energy == "200" else "LHC") / f"CNM_{self.energy}"
        outdir.mkdir(exist_ok=True, parents=True)

        system_label = f"{self.energy} GeV {self.energy == '200' and 'd+Au' or 'p+Pb'}"
        
        # 1. CNM vs y
        print(f"[CNMCombine] Computing vs y...")
        y_cent, tags_y, cnm_y = self.cnm_vs_y()
        df_y = cnm_vs_y_to_dataframe(y_cent, tags_y, cnm_y, "cnm")
        df_y.to_csv(outdir / f"cnm_vs_y_{self.family}_{self.particle_state}.csv", index=False)

        # Prepare extra bands for plotting (npdf and cnm)
        extra_y = {
            "npdf": (cnm_y["npdf"]),
            "cnm":  (cnm_y["cnm"]),
        }

        fig_y, _ = plot_RpA_vs_y_components_per_centrality(
            self.particle, self.sqrt_sNN, self.qp_base, self.gl, self.cent_bins,
            self.y_edges, self.pt_range_avg, 
            show_components=("loss", "broad", "total", "npdf", "cnm"), ## total is (eloss x pT broad)
            mb_weight_mode="exp",
            extra_bands=extra_y
        )
        fig_y.savefig(outdir / f"plot_cnm_vs_y_{self.family}.png", dpi=150)
        plt.show()
        
        # 2. CNM vs pT (for each y window)
        print(f"[CNMCombine] Computing vs pT...")
        for y0, y1, name in self.y_windows:
            pT_cent, tags_pt, cnm_pt = self.cnm_vs_pT((y0, y1, name))
            df_pt = cnm_vs_pT_to_dataframe(pT_cent, tags_pt, cnm_pt, "cnm")
            safe_name = name.replace(" ", "_").replace("<", "lt").replace(">", "gt").replace("/", "p")
            df_pt.to_csv(outdir / f"cnm_vs_pT_{safe_name}_{self.family}_{self.particle_state}.csv", index=False)

            fig_pt, _ = plot_RpA_vs_pT_components_per_centrality(
                self.particle, self.sqrt_sNN, self.qp_base, self.gl, self.cent_bins,
                self.p_edges, (y0, y1),
                show_components=("loss", "broad", "total", "npdf", "cnm"),
                extra_bands={
                    "npdf": cnm_pt["npdf"],
                    "cnm":  cnm_pt["cnm"]
                }
            )
            fig_pt.savefig(outdir / f"plot_cnm_vs_pT_{safe_name}_{self.family}.png", dpi=150)
            plt.show()

        # 3. CNM vs centrality (for each y window)
        print(f"[CNMCombine] Computing vs centrality...")
        for y0, y1, name in self.y_windows:
            cnm_cent = self.cnm_vs_centrality((y0, y1, name))
            df_cent = cnm_vs_cent_to_dataframe(self.cent_bins, cnm_cent, "cnm")
            safe_name = name.replace(" ", "_").replace("<", "lt").replace(">", "gt").replace("/", "p")
            df_cent.to_csv(outdir / f"cnm_vs_cent_{safe_name}_{self.family}_{self.particle_state}.csv", index=False)

            # High-level total CNM band plot
            # Extract all components
            labels = [f"{int(a)}-{int(b)}%" for (a, b) in self.cent_bins]

            def map_comp(comp_key):
                rc, rlo, rhi, mb_c, mb_lo, mb_hi = cnm_cent[comp_key]
                RT_c_d  = {l: val for l, val in zip(labels, rc)}
                RT_lo_d = {l: val for l, val in zip(labels, rlo)}
                RT_hi_d = {l: val for l, val in zip(labels, rhi)}
                RMB = (float(mb_c), float(mb_lo), float(mb_hi))
                return RT_c_d, RT_lo_d, RT_hi_d, RMB

            RL_c, RL_lo, RL_hi, RL_mb = map_comp("loss")
            RB_c, RB_lo, RB_hi, RB_mb = map_comp("broad")
            RT_c, RT_lo, RT_hi, RT_mb = map_comp("total")
            RNP_c, RNP_lo, RNP_hi, RNP_mb = map_comp("npdf")
            RCNM_c, RCNM_lo, RCNM_hi, RCNM_mb = map_comp("cnm")

            fig_cent, _ = plot_RpA_vs_centrality_components_band(
                self.cent_bins, labels,
                RL_c=RL_c, RL_lo=RL_lo, RL_hi=RL_hi, RMB_loss=RL_mb,
                RB_c=RB_c, RB_lo=RB_lo, RB_hi=RB_hi, RMB_broad=RB_mb,
                RT_c=RT_c, RT_lo=RT_lo, RT_hi=RT_hi, RMB_tot=RT_mb,
                RNP_c=RNP_c, RNP_lo=RNP_lo, RNP_hi=RNP_hi, RMB_npdf=RNP_mb,
                RCNM_c=RCNM_c, RCNM_lo=RCNM_lo, RCNM_hi=RCNM_hi, RMB_cnm=RCNM_mb,
                show=("loss", "broad", "total", "npdf", "cnm"),
                system_label=f"{self.particle} {system_label}",
                note=f"${y0:.2f} < y < {y1:.2f}$"
            )
            fig_cent.savefig(outdir / f"plot_cnm_vs_cent_{safe_name}_{self.family}.png", dpi=150)
            plt.show()
            
        print(f"[CNMCombine] All data and plots saved to {outdir}")


# ------------------------------------------------------------
# DataFrame converters (for CSV / plotting)
# ------------------------------------------------------------
def cnm_vs_y_to_dataframe(
    y_cent: np.ndarray,
    tags: Sequence[str],
    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    component: str,
) -> pd.DataFrame:
    """
    Long-form DataFrame for a single component vs y across centralities.
    """
    rows = []
    comp_dict = result[component]

    for tag in tags:
        Rc, Rlo, Rhi = comp_dict[tag]
        for yi, Rc_i, Rlo_i, Rhi_i in zip(y_cent, Rc, Rlo, Rhi):
            rows.append(
                dict(
                    y_center=float(yi),
                    centrality=tag,
                    is_MB=(tag == "MB"),
                    R_central=float(Rc_i),
                    R_lo=float(Rlo_i),
                    R_hi=float(Rhi_i),
                )
            )

    return pd.DataFrame(rows)


def cnm_vs_pT_to_dataframe(
    pT_cent: np.ndarray,
    tags: Sequence[str],
    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    component: str,
) -> pd.DataFrame:
    """
    Long-form DataFrame for a single component vs pT across centralities.
    """
    rows = []
    comp_dict = result[component]

    for tag in tags:
        Rc, Rlo, Rhi = comp_dict[tag]
        for pi, Rc_i, Rlo_i, Rhi_i in zip(pT_cent, Rc, Rlo, Rhi):
            rows.append(
                dict(
                    pT_center=float(pi),
                    centrality=tag,
                    is_MB=(tag == "MB"),
                    R_central=float(Rc_i),
                    R_lo=float(Rlo_i),
                    R_hi=float(Rhi_i),
                )
            )

    return pd.DataFrame(rows)


def cnm_vs_cent_to_dataframe(
    cent_bins: Sequence[Tuple[float, float]],
    result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]],
    component: str,
) -> pd.DataFrame:
    """
    Long-form DataFrame for a component vs centrality, including MB point.
    """
    Rc, Rlo, Rhi, Rc_MB, Rlo_MB, Rhi_MB = result[component]

    rows = []
    for (cL, cR), Rc_i, Rlo_i, Rhi_i in zip(cent_bins, Rc, Rlo, Rhi):
        lab = f"{int(cL)}-{int(cR)}%"
        rows.append(
            dict(
                cent_left=float(cL),
                cent_right=float(cR),
                cent_label=lab,
                is_MB=False,
                R_central=float(Rc_i),
                R_lo=float(Rlo_i),
                R_hi=float(Rhi_i),
            )
        )

    # MB entry
    rows.append(
        dict(
            cent_left=float(cent_bins[0][0]),
            cent_right=float(cent_bins[-1][1]),
            cent_label="MB",
            is_MB=True,
            R_central=float(Rc_MB),
            R_lo=float(Rlo_MB),
            R_hi=float(Rhi_MB),
        )
    )

    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# Light demo plotting (optional)
# ----------------------------------------------------------------------


def _step_from_centers(x_cent: np.ndarray, vals: np.ndarray):
    x_cent = np.asarray(x_cent, float)
    vals = np.asarray(vals, float)
    assert x_cent.size == vals.size

    if x_cent.size > 1:
        dx = np.diff(x_cent)
        dx0 = dx[0]
        if not np.allclose(dx, dx0):
            raise ValueError("x_cent not uniformly spaced")
    else:
        dx0 = 1.0

    x_edges = np.concatenate(([x_cent[0] - 0.5 * dx0], x_cent + 0.5 * dx0))
    y_step = np.concatenate([vals, vals[-1:]])
    return x_edges, y_step


def demo_plots(
    energy: str = "8.16",
    family: str = "charmonia",
    particle_state: str = "avg",
    outdir: Optional[Path] = None,
):
    """
    Quick-look plots (vs y, vs pT, vs centrality) using this module.

    This is deliberately simpler than the publication-style notebook
    but is good for sanity checks.
    """
    import matplotlib.pyplot as plt

    if outdir is None:
        outdir = HERE / "output-cnm"
    outdir.mkdir(exist_ok=True, parents=True)

    comb = CNMCombine.from_defaults(
        energy=energy,
        family=family,
        particle_state=particle_state,
    )

    components = ("npdf", "eloss", "broad", "eloss_broad", "cnm")
    comp_colors = {
        "npdf": "tab:blue",
        "eloss": "tab:orange",
        "broad": "tab:green",
        "eloss_broad": "tab:purple",
        "cnm": "k",
    }

    # ---- RpA vs y ----
    y_cent, tags_y, cnm_y = comb.cnm_vs_y()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, tag in enumerate(tags_y):
        ax = axes[i]
        for comp in components:
            Rc, Rlo, Rhi = cnm_y[comp][tag]
            x_edges, y_c = _step_from_centers(y_cent, Rc)
            _, y_lo = _step_from_centers(y_cent, Rlo)
            _, y_hi = _step_from_centers(y_cent, Rhi)

            ax.step(
                x_edges, y_c, where="post",
                lw=1.6, color=comp_colors[comp],
                label=comp if i == 0 else None,
            )
            ax.fill_between(
                x_edges, y_lo, y_hi,
                step="post", alpha=0.2, color=comp_colors[comp],
            )

        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_title(tag)
        ax.set_xlabel("y")
        ax.set_ylabel(r"$R_{pA}$")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(f"CNM vs y, sqrt(sNN)={comb.sqrt_sNN/1000:.2f} TeV")
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(outdir / f"demo_RpA_CNM_vs_y_{energy.replace('.','p')}TeV.png", dpi=150)
    plt.show()

    # ---- RpA vs pT (first y-window) ----
    y0, y1, label = comb.y_windows[0]
    pT_cent, tags_pt, cnm_pt = comb.cnm_vs_pT((y0, y1, label))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, tag in enumerate(tags_pt):
        ax = axes[i]
        for comp in components:
            Rc, Rlo, Rhi = cnm_pt[comp][tag]
            x_edges, y_c = _step_from_centers(pT_cent, Rc)
            _, y_lo = _step_from_centers(pT_cent, Rlo)
            _, y_hi = _step_from_centers(pT_cent, Rhi)

            ax.step(
                x_edges, y_c, where="post",
                lw=1.6, color=comp_colors[comp],
                label=comp if i == 0 else None,
            )
            ax.fill_between(
                x_edges, y_lo, y_hi,
                step="post", alpha=0.2, color=comp_colors[comp],
            )

        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_title(tag)
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_ylabel(r"$R_{pA}$")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(
        f"CNM vs pT, {label}, sqrt(sNN)={comb.sqrt_sNN/1000:.2f} TeV"
    )
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(
        outdir
        / f"demo_RpA_CNM_vs_pT_{label.replace(' ','_')}_{energy.replace('.','p')}TeV.png",
        dpi=150,
    )
    plt.show()

    # ---- RpA vs centrality (all y-windows) ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (y0, y1, name) in zip(axes, comb.y_windows):
        cnm_cent = comb.cnm_vs_centrality((y0, y1, name))
        edges = [comb.cent_bins[0][0]] + [b for (_, b) in comb.cent_bins]
        x_edges = np.array(edges, float)

        for comp in components:
            Rc, Rlo, Rhi, Rc_MB, Rlo_MB, Rhi_MB = cnm_cent[comp]
            y_step = np.concatenate([Rc, Rc[-1:]])
            y_lo = np.concatenate([Rlo, Rlo[-1:]])
            y_hi = np.concatenate([Rhi, Rhi[-1:]])

            ax.step(
                x_edges, y_step, where="post",
                lw=1.8, color=comp_colors[comp],
                label=comp if ax is axes[0] else None,
            )
            ax.fill_between(
                x_edges, y_lo, y_hi,
                step="post", alpha=0.2, color=comp_colors[comp],
            )

            if comp == "cnm":
                x_mb = np.array([comb.cent_bins[0][0], comb.cent_bins[-1][1]], float)
                ax.hlines(Rc_MB, x_mb[0], x_mb[1],
                          colors="k", linestyles="--", lw=1.8)

        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("Centrality [%]")
        ax.set_ylabel(r"$R_{pA}$")
        ax.set_title(name)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(f"CNM vs centrality, sqrt(sNN)={comb.sqrt_sNN/1000:.2f} TeV")
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(
        outdir
        / f"demo_RpA_CNM_vs_centrality_{energy.replace('.','p')}TeV.png",
        dpi=150,
    )
    plt.show()


if __name__ == "__main__":
    # Run quick sanity-check plots for 8.16 TeV charmonia
    demo_plots()

