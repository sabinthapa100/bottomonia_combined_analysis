#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_bottomonia_cnm_prim_OO.py
================================
Final combined Upsilon CNM + Primordial production for Oxygen-Oxygen collisions @ 5.36 TeV.
Publication-ready framework with proper binning, error propagation, and CSV export.

Combination:
  R_AA^Total = R_AA^CNM * R_AA^Primordial
  Errors add in quadrature (asymmetric bands handled separately).

Output Structure:
  outputs/cnm_prim/
  ├── min_bias/        # For now: min bias primordial only
  │   └── OO_5p36TeV/
  │       ├── binned_data/    # HEPData-style CSVs
  │       └── plots/           # Publication plots (PDF + PNG)
  └── centrality/       # Future: once centrality-dependent primordial is available
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2] 

# List of directories to add to sys.path
paths_to_add = [
    "cnm/eloss_code",
    "cnm/npdf_code",
    "cnm/cnm_combine",
    "hnm/primordial_code",
    "cnm/cnm_scripts",
    "cnm/npdf_code/deps",
]

for d in reversed(paths_to_add):
    p = str(ROOT / d)
    if p not in sys.path:
        sys.path.insert(0, p)

from cnm_combine_fast_nuclabs import CNMCombineFast
from cnm_combine import combine_two_bands_1d, alpha_s_provider
from gluon_ratio import EPPS21Ratio, GluonEPPSProvider
from glauber import SystemSpec, OpticalGlauber
from particle import Particle
from npdf_module import NPDFSystem, RpAAnalysis
from eloss_cronin_centrality import _qp_device
from npdf_centrality import compute_df49_by_centrality
from npdf_OO_data import load_OO_dat, build_OO_rpa_grid
import eloss_cronin_centrality as EC
import quenching_fast as QF

from ups_particle import make_bottomonia_system
from prim_band import PrimordialBand
from prim_io import load_pair
from prim_analysis import PrimordialAnalysis

# ── Physics config ──────────────────────────────────────────────────
SQRTS_NN = {"5.36": 5360.0}

INPUT_BASE = ROOT / "inputs" / "primordial"
PRIM_INPUTS = {
    "5.36": {
        "NPWLC": {
            "lower": INPUT_BASE / "output_OxOx5360_NPWLC" / "output-lower" / "datafile.gz",
            "upper": INPUT_BASE / "output_OxOx5360_NPWLC" / "output-upper" / "datafile.gz",
        },
        "Pert": {
            "lower": INPUT_BASE / "output_OxOx5360_Pert" / "output-lower" / "datafile.gz",
            "upper": INPUT_BASE / "output_OxOx5360_Pert" / "output-upper" / "datafile.gz",
        },
    }
}

DPI = 150
ALPHA_BAND = 0.20

# ── Global Plot Settings ─────────────────────────────────────────────
Y_LIM_RPA = (0.2, 1.2)
X_LIM_PT = (0, 15)

CALC_COMPS = ["cnm", "prim_NPWLC", "prim_Pert", "total_NPWLC", "total_Pert"]
COMPONENTS_TO_PLOT = ["cnm", "total_NPWLC", "total_Pert"]

COLORS = {
    'cnm':          '#7B2D8B',   # purple
    'prim_NPWLC':   '#FF8C00',   # orange (consistent with CNM script)
    'prim_Pert':    '#00BFFF',   # deep sky blue
    'total_NPWLC':  '#DC143C',   # crimson red
    'total_Pert':   '#0066CC',   # dark blue
}

LABELS = {
    "cnm":          "Total CNM",
    "prim_NPWLC":   "Primordial (NPWLC)",
    "prim_Pert":    "Primordial (Pert)",
    "total_NPWLC":  "CNM + Primordial (NPWLC)",
    "total_Pert":   "CNM + Primordial (Pert)",
}

STATES = ["ups1S", "ups2S", "ups3S"]
STATE_NAMES = {
    "ups1S": r"$\Upsilon(1S)$",
    "ups2S": r"$\Upsilon(2S)$",
    "ups3S": r"$\Upsilon(3S)$",
}

CENT_BINS = [(0,20),(20,40),(40,60),(60,80),(80,100)]
MB_C0 = 0.25

# Kinematic binning
Y_EDGES = np.arange(-5.0, 5.0 + 0.5, 0.5)
P_EDGES = np.arange(0.0, 20.0 + 1.0, 1.0)
PT_RANGE_AVG = (0.0, 15.0)

# Rapidity windows for CMS (3 rapidity regions)
Y_WINDOWS = [
    (-5.0, -2.4, r"$-5.0 < y < -2.4$ (Backward)"),
    (-2.4,  2.4, r"$-2.4 < y < 2.4$ (Midrapidity)"),
    ( 2.4,  4.5, r"$2.4 < y < 4.5$ (Forward)"),
]

# Rapidity bins for pT plots
Y_BINS_FOR_PT = [(y0, y1) for y0, y1, _ in Y_WINDOWS]

# ── Helper Functions ────────────────────────────────────────────────
def step_from_centers(centers, values):
    """Convert bin centers + values to step function edges."""
    centers = np.asarray(centers, float)
    values = np.asarray(values, float)
    if len(centers) < 2:
        hw = 1.0
    else:
        hw = (centers[1] - centers[0]) / 2.0
    edges = np.append(centers - hw, centers[-1] + hw)
    step_vals = np.append(values, values[-1])
    return edges, step_vals

def export_to_csv(filename, df):
    """Export combined result to HEPData-compatible CSV."""
    df.to_csv(filename, index=False, float_format='%.6e')
    print(f"    ✓ {filename.name}")

# ── Factory Helpers ─────────────────────────────────────────────────
def build_cnm_context(energy):
    print(f"\n[INFO] Loading CNM context for O+O @ {energy} TeV ...", flush=True)

    sqrt_sNN = SQRTS_NN[energy]
    sigma_nn_mb = 68.0

    particle = Particle(family="bottomonia", state="avg", mass_override_GeV=9.46)
    epps_ratio = EPPS21Ratio(A=16, path=str(ROOT / "cnm" / "npdf_code" / "nPDFs"))
    
    gluon = GluonEPPSProvider(
        epps_ratio, sqrt_sNN_GeV=sqrt_sNN, m_state_GeV=10.01, y_sign_for_xA=-1
    ).with_geometry()

    gl_spec = SystemSpec("AA", sqrt_sNN, A=16, sigma_nn_mb=sigma_nn_mb)
    gl_ana = OpticalGlauber(gl_spec, nx_pa=64, ny_pa=64, verbose=False)

    OO_DAT = ROOT / "inputs" / "npdf" / "OxygenOxygen5360" / "nPDF_OO.dat"
    data = load_OO_dat(str(OO_DAT))
    grid = build_OO_rpa_grid(data, pt_max=20.0)

    r0 = grid["r_central"].to_numpy()
    M = grid[[f"r_mem_{i:03d}" for i in range(1, 49)]].to_numpy().T
    SA_all = np.vstack([r0[None, :], M])

    df49_by_cent, K_by_cent, _, Y_SHIFT = compute_df49_by_centrality(
        grid, r0, M, gluon, gl_ana,
        cent_bins=CENT_BINS, nb_bsamples=5, kind="AA", SA_all=SA_all
    )

    npdf_ctx = dict(df49_by_cent=df49_by_cent, df_pp=grid, df_pa=grid, gluon=gluon)

    alpha_s = alpha_s_provider(mode="running", LambdaQCD=0.25)
    Lmb = gl_ana.leff_minbias_AA()
    
    device = EC._qp_device(None)
    qp_base = QF.QuenchParams(
        qhat0=0.075, lp_fm=1.5, LA_fm=Lmb, LB_fm=Lmb,
        system="AA", lambdaQCD=0.25, roots_GeV=sqrt_sNN,
        alpha_of_mu=alpha_s, alpha_scale="mT",
        use_hard_cronin=True, mapping="exp", device=device,
    )

    return CNMCombineFast(
        energy=energy,family="bottomonia", particle_state="avg",
        sqrt_sNN=sqrt_sNN, sigma_nn_mb=sigma_nn_mb,
        cent_bins=CENT_BINS, y_edges=Y_EDGES,
        p_edges=P_EDGES, y_windows=Y_WINDOWS,
        pt_range_avg=PT_RANGE_AVG, pt_floor_w=1.0,
        weight_mode="flat", y_ref=0.0, cent_c0=MB_C0,
        q0_pair=(0.05, 0.09), p0_scale_pair=(0.9, 1.1),
        nb_bsamples=5, y_shift_fraction=1.0,
        particle=particle, gl=gl_ana, qp_base=qp_base, npdf_ctx=npdf_ctx,
        y_sign_for_xA=-1, spec=gl_spec, debug=False, absorption=None
    )


def build_primordial_band(energy, model):
    print(f"\n[INFO] Loading Primordial {model} context for O+O @ {energy} TeV ...", flush=True)
    sqrts = SQRTS_NN[energy]
    system = make_bottomonia_system(sqrts_pp_GeV=sqrts)
    paths = PRIM_INPUTS[energy][model]
    
    if not paths["lower"].exists() or not paths["upper"].exists():
        print(f"  [WARN] Missing primordial input files for {model} at {energy} TeV.")
        return None
        
    df_lo, df_hi = load_pair(str(paths["lower"]), str(paths["upper"]), system, debug=False)
    ana_lo = PrimordialAnalysis(df_lo, system, with_feeddown=True)
    ana_hi = PrimordialAnalysis(df_hi, system, with_feeddown=True)
    return PrimordialBand(lower=ana_lo, upper=ana_hi, include_run_errors=True)


# ── Plotting Utilities ──────────────────────────────────────────────
def step_from_centers(centers, values):
    hw = (centers[1] - centers[0]) / 2.0
    edges = np.append(centers - hw, centers[-1] + hw)
    step_vals = np.append(values, values[-1])
    return edges, step_vals

def cent_step_arrays(cent_bins, vals):
    edges = [b[0] for b in cent_bins] + [cent_bins[-1][1]]
    step_vals = np.append(vals, vals[-1])
    return np.array(edges), step_vals

# ── Global Plot: R_pA vs y ──────────────────────────────────────────
def plot_rpa_vs_y_grid(cnm, energy, bands_npwlc, bands_pert):
    """
    cnm.cnm_vs_y returns components. We pull Total CNM.
    bands_model is PrimordialBand object.
    We plot the product of CNM and Primordial bands.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=DPI, sharey=True)
    plt.subplots_adjust(wspace=0)

    # 1. Get CNM values for min_bias vs y
    yc, tags, dict_cnm = cnm.cnm_vs_y(include_mb=True)
    cnm_c, cnm_lo, cnm_hi = dict_cnm["cnm"][0]["MB"], dict_cnm["cnm"][1]["MB"], dict_cnm["cnm"][2]["MB"]

    # 2. Get Primordial values vs y
    y_bins = list(zip(Y_EDGES[:-1], Y_EDGES[1:]))
    prim_np_c, prim_np_band = bands_npwlc.vs_y(PT_RANGE_AVG, y_bins)
    prim_pe_c, prim_pe_band = bands_pert.vs_y(PT_RANGE_AVG, y_bins)

    # Interpolate primordial onto CNM y-centers 
    # (they should already match if Y_EDGES translates to the exact same 'y' centers)
    yp = prim_np_c["y"].to_numpy()

    sys_note = rf"$\mathbf{{O+O @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$" + "\n" + r"$p_T$-integrated: $p_T \in [0, 15]$ GeV"

    for i, state in enumerate(STATES):
        ax = axes[i]

    for i, state in enumerate(STATES):
        ax = axes[i]

        # CNM
        if "cnm" in COMPONENTS_TO_PLOT:
            xe, yc_s = step_from_centers(yc, cnm_c)
            ax.step(xe, yc_s, where="post", lw=2, color=COLORS["cnm"], ls="-", label=LABELS["cnm"])
            ax.fill_between(xe, step_from_centers(yc, cnm_lo)[1], step_from_centers(yc, cnm_hi)[1], step="post", color=COLORS["cnm"], alpha=ALPHA_BAND, lw=0)

        # NPWLC Combo
        p_c = prim_np_c[state].to_numpy()
        p_lo = prim_np_band[f"{state}_lo"].to_numpy()
        p_hi = prim_np_band[f"{state}_hi"].to_numpy()
        
        if "prim_NPWLC" in COMPONENTS_TO_PLOT:
            xe, yc_s = step_from_centers(yc, p_c)
            ax.step(xe, yc_s, where="post", lw=1.5, ls="--", color=COLORS["prim_NPWLC"], label=LABELS["prim_NPWLC"])
            ax.fill_between(xe, step_from_centers(yc, p_lo)[1], step_from_centers(yc, p_hi)[1], step="post", color=COLORS["prim_NPWLC"], alpha=0.1, lw=0)

        if "total_NPWLC" in COMPONENTS_TO_PLOT:
            tot_np_c, tot_np_lo, tot_np_hi = combine_two_bands_1d(cnm_c, cnm_lo, cnm_hi, p_c, p_lo, p_hi)
            xe, yc_s = step_from_centers(yc, tot_np_c)
            ax.step(xe, yc_s, where="post", lw=2.2, color=COLORS["total_NPWLC"], ls="-", label=LABELS["total_NPWLC"])
            ax.fill_between(xe, step_from_centers(yc, tot_np_lo)[1], step_from_centers(yc, tot_np_hi)[1], step="post", color=COLORS["total_NPWLC"], alpha=ALPHA_BAND, lw=0)

        # Pert Combo
        p_c = prim_pe_c[state].to_numpy()
        p_lo = prim_pe_band[f"{state}_lo"].to_numpy()
        p_hi = prim_pe_band[f"{state}_hi"].to_numpy()

        if "prim_Pert" in COMPONENTS_TO_PLOT:
            xe, yc_s = step_from_centers(yc, p_c)
            ax.step(xe, yc_s, where="post", lw=1.5, ls="--", color=COLORS["prim_Pert"], label=LABELS["prim_Pert"])
            ax.fill_between(xe, step_from_centers(yc, p_lo)[1], step_from_centers(yc, p_hi)[1], step="post", color=COLORS["prim_Pert"], alpha=0.1, lw=0)

        if "total_Pert" in COMPONENTS_TO_PLOT:
            tot_pe_c, tot_pe_lo, tot_pe_hi = combine_two_bands_1d(cnm_c, cnm_lo, cnm_hi, p_c, p_lo, p_hi)
            xe, yc_s = step_from_centers(yc, tot_pe_c)
            ax.step(xe, yc_s, where="post", lw=2.2, color=COLORS["total_Pert"], ls="-", label=LABELS["total_Pert"])
            ax.fill_between(xe, step_from_centers(yc, tot_pe_lo)[1], step_from_centers(yc, tot_pe_hi)[1], step="post", color=COLORS["total_Pert"], alpha=ALPHA_BAND, lw=0)

        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.text(0.96, 0.96, STATE_NAMES[state], transform=ax.transAxes, ha="right", va="top", weight="bold", fontsize=12)
        
        if i == 0:
            ax.text(0.04, 0.05, sys_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
            ax.legend(loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(0.0, 0.20))
            
        ax.set_xlim(-5.0, 5.0); ax.set_ylim(*Y_LIM_RPA)
        ax.set_xlabel(r"$y_{CM}$", fontsize=14)
        if i == 0: ax.set_ylabel(r"$R^{\Upsilon}_{AA}$", fontsize=16)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=11)
        ax.label_outer()

    return fig

# ── Global Plot: R_pA vs pT ─────────────────────────────────────────
def plot_rpa_vs_pT_grid(cnm, energy, bands_npwlc, bands_pert):
    """
    3 rows (y-windows) x 3 columns (states 1S, 2S, 3S)
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=DPI, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    sys_note = rf"$\mathbf{{O+O @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$"

    for row, (y0, y1, yname) in enumerate(Y_WINDOWS):
        pc, tags, dict_cnm = cnm.cnm_vs_pT((y0, y1), include_mb=True)
        cnm_c, cnm_lo, cnm_hi = dict_cnm["cnm"][0]["MB"], dict_cnm["cnm"][1]["MB"], dict_cnm["cnm"][2]["MB"]

        # Primordial 
        pt_bins = list(zip(P_EDGES[:-1], P_EDGES[1:]))
        prim_np_c, prim_np_band = bands_npwlc.vs_pt((y0, y1), pt_bins)
        prim_pe_c, prim_pe_band = bands_pert.vs_pt((y0, y1), pt_bins)

        for col, state in enumerate(STATES):
            ax = axes[row, col]

            # CNM
            if "cnm" in COMPONENTS_TO_PLOT:
                xe, yc_s = step_from_centers(pc, cnm_c)
                ax.step(xe, yc_s, where="post", lw=2, color=COLORS["cnm"], ls="-", label=LABELS["cnm"])
                ax.fill_between(xe, step_from_centers(pc, cnm_lo)[1], step_from_centers(pc, cnm_hi)[1], step="post", color=COLORS["cnm"], alpha=ALPHA_BAND, lw=0)

            # NPWLC Combo
            p_c = prim_np_c[state].to_numpy()
            p_lo = prim_np_band[f"{state}_lo"].to_numpy()
            p_hi = prim_np_band[f"{state}_hi"].to_numpy()

            if "prim_NPWLC" in COMPONENTS_TO_PLOT:
                xe, yc_s = step_from_centers(pc, p_c)
                ax.step(xe, yc_s, where="post", lw=1.5, ls="--", color=COLORS["prim_NPWLC"], label=LABELS["prim_NPWLC"])
                ax.fill_between(xe, step_from_centers(pc, p_lo)[1], step_from_centers(pc, p_hi)[1], step="post", color=COLORS["prim_NPWLC"], alpha=0.1, lw=0)

            if "total_NPWLC" in COMPONENTS_TO_PLOT:
                tot_np_c, tot_np_lo, tot_np_hi = combine_two_bands_1d(cnm_c, cnm_lo, cnm_hi, p_c, p_lo, p_hi)
                xe, yc_s = step_from_centers(pc, tot_np_c)
                ax.step(xe, yc_s, where="post", lw=2.2, color=COLORS["total_NPWLC"], ls="-", label=LABELS["total_NPWLC"])
                ax.fill_between(xe, step_from_centers(pc, tot_np_lo)[1], step_from_centers(pc, tot_np_hi)[1], step="post", color=COLORS["total_NPWLC"], alpha=ALPHA_BAND, lw=0)

            # Pert Combo
            p_c = prim_pe_c[state].to_numpy()
            p_lo = prim_pe_band[f"{state}_lo"].to_numpy()
            p_hi = prim_pe_band[f"{state}_hi"].to_numpy()

            if "prim_Pert" in COMPONENTS_TO_PLOT:
                xe, yc_s = step_from_centers(pc, p_c)
                ax.step(xe, yc_s, where="post", lw=1.5, ls="--", color=COLORS["prim_Pert"], label=LABELS["prim_Pert"])
                ax.fill_between(xe, step_from_centers(pc, p_lo)[1], step_from_centers(pc, p_hi)[1], step="post", color=COLORS["prim_Pert"], alpha=0.1, lw=0)

            if "total_Pert" in COMPONENTS_TO_PLOT:
                tot_pe_c, tot_pe_lo, tot_pe_hi = combine_two_bands_1d(cnm_c, cnm_lo, cnm_hi, p_c, p_lo, p_hi)
                xe, yc_s = step_from_centers(pc, tot_pe_c)
                ax.step(xe, yc_s, where="post", lw=2.2, color=COLORS["total_Pert"], ls="-", label=LABELS["total_Pert"])
                ax.fill_between(xe, step_from_centers(pc, tot_pe_lo)[1], step_from_centers(pc, tot_pe_hi)[1], step="post", color=COLORS["total_Pert"], alpha=ALPHA_BAND, lw=0)

            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
            ax.text(0.96, 0.96, STATE_NAMES[state], transform=ax.transAxes, ha="right", va="top", weight="bold", fontsize=12)
            ax.text(0.04, 0.96, yname, transform=ax.transAxes, ha="left", va="top", color="navy", fontsize=11, fontweight="bold")
            
            if row == 0 and col == 0:
                ax.text(0.04, 0.05, sys_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
                ax.legend(loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(0.0, 0.15))
                
            ax.set_xlim(*X_LIM_PT); ax.set_ylim(*Y_LIM_RPA)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune="both"))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=9, prune="both"))
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=11)
            ax.label_outer()

    fig.text(0.5, 0.05, r"$p_T$ (GeV)", ha="center", fontsize=16)
    fig.text(0.06, 0.5, r"$R^{\Upsilon}_{AA}$", va="center", rotation="vertical", fontsize=16)
    return fig

# ── Main Generator ──────────────────────────────────────────────────
def run_energy(energy):
    cnm = build_cnm_context(energy)
    band_np = build_primordial_band(energy, "NPWLC")
    band_pe = build_primordial_band(energy, "Pert")
    
    if band_np is None or band_pe is None:
        print(f"Skipping {energy}")
        return

    etag = energy.replace('.','p')
    OUTDIR = ROOT / "outputs" / "cnm_prim" / "min_bias" / f"OO_{etag}TeV"
    OUTDIR.mkdir(exist_ok=True, parents=True)

    print(f"  [PLOT] Combinations vs y ...", flush=True)
    fig_y = plot_rpa_vs_y_grid(cnm, energy, band_np, band_pe)
    fig_y.savefig(OUTDIR / f"Upsilon_RAA_CNM_Prim_vs_y_MB_OO_{etag}TeV.pdf", bbox_inches="tight")
    plt.close(fig_y)

    print(f"  [PLOT] Combinations vs pT ...", flush=True)
    fig_pt = plot_rpa_vs_pT_grid(cnm, energy, band_np, band_pe)
    fig_pt.savefig(OUTDIR / f"Upsilon_RAA_CNM_Prim_vs_pT_Grid_OO_{etag}TeV.pdf", bbox_inches="tight")
    plt.close(fig_pt)

if __name__ == "__main__":
    for e in ["5.36"]:
        print(f"{'='*60}\nProcessing O+O {e} TeV")
        run_energy(e)
    print("DONE O+O Combined Prod Runs.")
