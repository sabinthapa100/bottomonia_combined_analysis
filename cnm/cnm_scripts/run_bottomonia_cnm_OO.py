#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bottomonia_cnm_OO.py
========================
O-O Upsilon production script integrating Total CNM:
nPDF + Energy Loss + pT Broadening for 5.36 TeV.

Matches publication style: outputs categorized into min_bias/ and centrality/.
"""

from pathlib import Path
import sys
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
    "cnm/cnm_combine",
    "cnm/npdf_code",
]

for d in reversed(paths_to_add):
    p = str(ROOT / d)
    if p not in sys.path:
        sys.path.insert(0, p)

from npdf_OO_data import load_OO_dat, build_OO_rpa_grid
from glauber import OpticalGlauber, SystemSpec
from gluon_ratio import GluonEPPSProvider, EPPS21Ratio
from npdf_centrality import compute_df49_by_centrality
from particle import Particle
from coupling import alpha_s_provider
import quenching_fast as QF
from cnm_combine_fast_nuclabs import CNMCombineFast

# ── Config ───────────────────────────────────────────────────────────
ENERGIES = ["5.36"]
SQRTS_GEV = {"5.36": 5360.0}
SIG_NN_MB = {"5.36": 68.0}
M_UPSILON = 9.46
M_UPSILON_AVG = 10.01  # For nPDF matching

# OUTPUT PATHS (ROOT/outputs/cnm/...)
OUTDIR_CENT = ROOT / "outputs" / "cnm" / "centrality" / "OO_5p36TeV"
OUTDIR_MB   = ROOT / "outputs" / "cnm" / "min_bias" / "OO_5p36TeV"

DPI = 150
ALPHA_BAND = 0.22

# PLOTTING LIMITS (STRICTLY REQUESTED) - set to None to compute from data
Y_LIM_RPA = None  # (0.4, 1.2)  # dynamic or override
X_LIM_PT = (0, 15)  # Truncated to 15 GeV by default

CALC_COMPS = ("npdf", "eloss", "broad", "eloss_broad", "cnm")
COMPONENTS_TO_PLOT = ["npdf", "eloss", "broad", "eloss_broad", "cnm"]

# Color convention
COLORS = {
    "npdf":       "#E69F00",   # orange
    "eloss":      "#F4A0A0",   # pink/salmon
    "broad":      "#56B4E9",   # light blue
    "eloss_broad":"#404040",   # dark gray
    "cnm":        "#606060",   # Gray (Total CNM)
}

LABELS = {
    "npdf":       "nPDF (EPPS21)",
    "eloss":      "ELoss",
    "broad":      r"$p_T$-Broadening",
    "eloss_broad":r"ELoss + $p_T$-Broad",
    "cnm":        "Total CNM",
}

CENT_BINS = [(0,20),(20,40),(40,60),(60,80),(80,100)]
MB_C0 = 0.25

Y_EDGES = np.arange(-5.0, 5.0 + 0.5, 0.5)
P_EDGES = np.arange(0.0, 20.0 + 1.0, 1.0)
PT_RANGE_AVG = (0.0, 15.0)

Y_WINDOWS = [
    (-5.0, -2.5, r"$-5.0 < y < -2.5$"),
    (-2.4,  2.4, r"$-2.4 < y < 2.4$"),
    ( 2.5,  4.0, r"$2.5 < y < 4.0$"),
]

# Eloss params
Q0_PAIR = (0.05, 0.09)
P0_SCALE_PAIR = (0.9, 1.1)

# ── Factory ──────────────────────────────────────────────────────────
def build_eloss_context(energy="5.36"):
    print(f"\n[INFO] Loading O+O @ {energy} TeV ...", flush=True)
    SQRT_SNN = SQRTS_GEV[energy]
    SIG_NN = SIG_NN_MB[energy]
    
    OO_DAT = ROOT / "inputs" / "npdf" / "OxygenOxygen5360" / "nPDF_OO.dat"
    data = load_OO_dat(str(OO_DAT))
    grid = build_OO_rpa_grid(data, pt_max=20.0)
    
    gl = OpticalGlauber(SystemSpec("AA", SQRT_SNN, A=16, sigma_nn_mb=SIG_NN), nx_pa=64, ny_pa=64, verbose=False)
    
    r0 = grid["r_central"].to_numpy()
    M = grid[[f"r_mem_{i:03d}" for i in range(1, 49)]].to_numpy().T
    SA_all = np.vstack([r0[None, :], M])
    
    epps_wrapper = EPPS21Ratio(A=16, path=str(ROOT / "inputs" / "npdf" / "nPDFs"))
    gluon_provider = GluonEPPSProvider(epps_wrapper, SQRT_SNN, m_state_GeV=M_UPSILON_AVG)
    
    df49_by_cent, K_by_cent, _, Y_SHIFT = compute_df49_by_centrality(
        grid, r0, M, gluon_provider, gl,
        cent_bins=CENT_BINS, nb_bsamples=5, kind="AA", SA_all=SA_all
    )
    
    npdf_ctx = dict(df49_by_cent=df49_by_cent, df_pp=grid, df_pa=grid, gluon=gluon_provider)
    particle = Particle(family="bottomonia", state="avg", mass_override_GeV=M_UPSILON)
    alpha_s = alpha_s_provider(mode="running", LambdaQCD=0.25)
    Lmb = gl.leff_minbias_AA()
    
    device = "cpu"
    qp_base = QF.QuenchParams(
        qhat0=0.075, lp_fm=1.5,
        LA_fm=Lmb, LB_fm=Lmb,
        system="AA", lambdaQCD=0.25, roots_GeV=SQRT_SNN,
        alpha_of_mu=alpha_s, alpha_scale="mT",
        use_hard_cronin=True, mapping="exp", device=device
    )
    
    cnm = CNMCombineFast(
        energy=energy, family="bottomonia", particle_state="avg",
        sqrt_sNN=SQRT_SNN, sigma_nn_mb=SIG_NN,
        cent_bins=CENT_BINS, y_edges=Y_EDGES, p_edges=P_EDGES,
        y_windows=Y_WINDOWS, pt_range_avg=PT_RANGE_AVG, pt_floor_w=1.0,
        weight_mode="flat", y_ref=0.0, cent_c0=MB_C0,
        q0_pair=Q0_PAIR, p0_scale_pair=P0_SCALE_PAIR, nb_bsamples=5,
        y_shift_fraction=1.0, particle=particle,
        npdf_ctx=npdf_ctx, gl=gl, qp_base=qp_base, spec=gl.spec
    )
    return cnm

# ── Helpers ──────────────────────────────────────────────────────────
def step_from_centers(xc, vals):
    xc = np.asarray(xc, float); vals = np.asarray(vals, float)
    dx = np.diff(xc)
    dx0 = dx[0] if dx.size else 1.0
    xe = np.concatenate(([xc[0]-0.5*dx0], xc+0.5*dx0))
    ys = np.concatenate([vals, vals[-1:]])
    return xe, ys

def cent_step_arrays(cb, vals):
    vals = np.asarray(vals, float)
    edges = [cb[0][0]] + [b for (_, b) in cb]
    return np.asarray(edges, float), np.concatenate([vals, vals[-1:]])

def edges_from_centers(xc):
    xc = np.asarray(xc, float)
    if xc.size < 2: return np.array([xc[0]-0.5, xc[0]+0.5])
    mids = 0.5*(xc[:-1]+xc[1:])
    return np.concatenate([[xc[0]-(mids[0]-xc[0])], mids, [xc[-1]+(xc[-1]-mids[-1])]])

# ── CSV savers ──────────────────────────────────────────────────────
def save_csv_y(outdir, yc, bands_all, tags, energy, suffix=""):
    # bands_all: {comp: (Rc_dict, Rlo_dict, Rhi_dict)}
    for comp, (Rc_dict, Rlo_dict, Rhi_dict) in bands_all.items():
        for tag in tags:
            Rc, Rlo, Rhi = Rc_dict[tag], Rlo_dict[tag], Rhi_dict[tag]
            df = pd.DataFrame({"y_center": yc, "R_central": Rc, "R_lo": Rlo, "R_hi": Rhi})
            st = tag.replace("%","pct").replace(" ","")
            df.to_csv(outdir / f"Upsilon_RAA_{comp}_vs_y_{st}_OO_{energy.replace('.','p')}TeV{suffix}.csv", index=False)

def save_csv_pT(outdir, pc, bands_all, tags, energy, yname=""):
    for comp, (Rc_dict, Rlo_dict, Rhi_dict) in bands_all.items():
        for tag in tags:
            Rc, Rlo, Rhi = Rc_dict[tag], Rlo_dict[tag], Rhi_dict[tag]
            df = pd.DataFrame({"pT_center": pc, "R_central": Rc, "R_lo": Rlo, "R_hi": Rhi})
            st = tag.replace("%","pct").replace(" ","")
            yn = yname.replace(" ","").replace("<","").replace(">","").replace("$","").replace("\\","")
            df.to_csv(outdir / f"Upsilon_RAA_{comp}_vs_pT_{st}_{yn}_OO_{energy.replace('.','p')}TeV.csv", index=False)

def save_csv_cent(outdir, cnm, bands_cent, yname, energy):
    # bands_cent: {comp: (Rc, Rlo, Rhi, mbc, mblo, mbhi)}
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in cnm.cent_bins]
    
    # Ncoll is same for all components
    gl = cnm.gl
    Ncoll_cent = [gl.ncoll_mean_bin_AA_optical(a/100.0, b/100.0) for (a,b) in cnm.cent_bins]
    Ncoll_MB = gl.ncoll_mean_bin_AA_optical(0.0, 1.0)

    for comp, (Rc, Rlo, Rhi, mbc, mblo, mbhi) in bands_cent.items():
        rows = []
        for i, ((cL,cR), lab) in enumerate(zip(cnm.cent_bins, labels)):
            rows.append(dict(cent_left=float(cL), cent_right=float(cR), label=lab,
                             Ncoll=float(Ncoll_cent[i]), R_central=float(Rc[i]),
                             R_lo=float(Rlo[i]), R_hi=float(Rhi[i]), is_MB=False))
        rows.append(dict(cent_left=0, cent_right=100, label="MB",
                         Ncoll=float(Ncoll_MB), R_central=float(mbc),
                         R_lo=float(mblo), R_hi=float(mbhi), is_MB=True))
        
        yn = yname.replace(" ","").replace("<","").replace(">","").replace("$","").replace("\\","")
        pd.DataFrame(rows).to_csv(
            outdir / f"Upsilon_RAA_{comp}_vs_cent_{yn}_OO_{energy.replace('.','p')}TeV.csv", index=False)

# ── Core Plotting ────────────────────────────────────────────────────
def compute_ylim_from_bands(bands_dict, margin=0.05):
    """Compute y-axis limits from a bands dictionary used in CNM plotting.
    """
    vals = []
    for comp_bands in bands_dict.values():
        # comp_bands can be (Dc, Dlo, Dhi) or (vals_c, vals_lo, vals_hi, mb_c, mb_lo, mb_hi)
        # We try to extract values for both MB and centrality bins
        if isinstance(comp_bands, (list, tuple)):
            for item in comp_bands:
                if isinstance(item, dict):
                    for v in item.values():
                        vals.append(np.asarray(v, float))
                elif isinstance(item, (np.ndarray, list, float, int)):
                    vals.append(np.asarray(item, float))
        elif isinstance(comp_bands, dict):
             for v in comp_bands.values():
                 vals.append(np.asarray(v, float))
    if not vals:
        return (0.0, 1.0)
    allvals = np.concatenate([v.flatten() for v in vals if v.size])
    allvals = allvals[np.isfinite(allvals)]
    if allvals.size == 0:
        return (0.0, 1.0)
    mn = allvals.min(); mx = allvals.max()
    rng = mx - mn
    if rng <= 0:
        return (mn - 0.1, mx + 0.1)
    return (max(0.0, mn - margin*rng), mx + margin*rng)


def plot_rpa_vs_y_grid(cnm, yc, tags, bands, energy):
    n_tags = len(tags)
    n_cols = 4
    n_rows = (n_tags + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5*n_rows), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)
    axes_flat = axes.flatten()

    sys_note = rf"$\mathbf{{O+O @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$"

    for ip, tag in enumerate(tags):
        ax = axes_flat[ip]
        plotted_handles = []
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands: continue
            Rc, Rlo, Rhi = bands[comp][0][tag], bands[comp][1][tag], bands[comp][2][tag]
            xe, yc_s = step_from_centers(yc, Rc)
            color = COLORS.get(comp, "black")
            ls = "--" if comp == "npdf" else "-"
            lw = 2.2 if comp == "cnm" else 1.5
            line, = ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
            ax.fill_between(xe, step_from_centers(yc, Rlo)[1], step_from_centers(yc, Rhi)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)
            if ip == 0: plotted_handles.append(line)

        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.text(0.96, 0.96, tag, transform=ax.transAxes, ha="right", va="top", weight="bold", fontsize=11)
        
        if ip == 0:
            ax.text(0.95, 0.90, rf"$p_T \in [{PT_RANGE_AVG[0]:.0f},\,{PT_RANGE_AVG[1]:.0f}]$ GeV", transform=ax.transAxes, ha="right", va="top", fontsize=9)
        if ip == 1:
            ax.text(0.05, 0.90, sys_note, transform=ax.transAxes, ha="left", va="top", fontsize=10)

        ax.set_xlim(-5, 5)
        if Y_LIM_RPA is None:
            lims = compute_ylim_from_bands(bands)
            ax.set_ylim(*lims)
        else:
            ax.set_ylim(*Y_LIM_RPA)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
        ax.label_outer()

    if n_tags < len(axes_flat):
         axes_flat[n_tags-1].legend(handles=plotted_handles, loc="lower center", frameon=False, fontsize=10)
    
    for j in range(n_tags, len(axes_flat)): fig.delaxes(axes_flat[j])
    fig.text(0.5, 0.02, r"$y$", ha="center", fontsize=18)
    fig.text(0.04, 0.5, r"$R^{\Upsilon}_{AA}$", va="center", rotation="vertical", fontsize=18)
    return fig

def plot_rpa_vs_pT_grid(cnm, energy, y_windows, p_edges):
    pc, tags, bands = cnm.cnm_vs_pT((y_windows[0][0], y_windows[0][1]), p_edges, components=CALC_COMPS, include_mb=True)
    n_rows = len(y_windows); n_cols = len(tags)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.0*n_rows), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)

    if n_rows == 1: axes = [axes]
    sys_note = rf"$\mathbf{{O+O @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$"

    for row, (y0,y1,yname) in enumerate(y_windows):
        pc, tags_pt, bands_pt = cnm.cnm_vs_pT((y0,y1), p_edges, components=CALC_COMPS, include_mb=True)
        for col, tag in enumerate(tags_pt):
            ax = axes[row][col]
            for comp in COMPONENTS_TO_PLOT:
                if comp not in bands_pt: continue
                Rc, Rlo, Rhi = bands_pt[comp][0][tag], bands_pt[comp][1][tag], bands_pt[comp][2][tag]
                xe, yc_s = step_from_centers(pc, Rc)
                color = COLORS.get(comp, "black")
                ls = "--" if comp == "npdf" else "-"
                lw = 2.2 if comp == "cnm" else 1.5
                ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
                ax.fill_between(xe, step_from_centers(pc, Rlo)[1], step_from_centers(pc, Rhi)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)

            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
            ax.text(0.96, 0.96, tag, transform=ax.transAxes, ha="right", va="top", weight="bold", fontsize=10)
            ax.text(0.04, 0.96, yname, transform=ax.transAxes, ha="left", va="top", color="navy", fontsize=9, fontweight="bold")
            
            if row == 0 and col == 0:
                ax.text(0.05, 0.10, sys_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)
                
            ax.set_xlim(*X_LIM_PT)
            if Y_LIM_RPA is None:
                lims = compute_ylim_from_bands(bands)
                ax.set_ylim(*lims)
            else:
                ax.set_ylim(*Y_LIM_RPA)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune="both"))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=9, prune="both"))
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=10)
            ax.label_outer()
            if row == 0 and col == 1:
                ax.legend(loc="lower left", frameon=False, fontsize=8)

    fig.text(0.5, 0.02, r"$p_T$ (GeV)", ha="center", fontsize=20)
    fig.text(0.04, 0.5, r"$R^{\Upsilon}_{AA}$", va="center", rotation="vertical", fontsize=20)
    return fig

def plot_rpa_vs_centrality(cnm, energy, y_windows, pt_range_avg):
    gl = cnm.gl
    nc_bins = [gl.ncoll_mean_bin_AA_optical(a/100.0, b/100.0) for (a,b) in cnm.cent_bins]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=DPI, sharey=True)
    plt.subplots_adjust(wspace=0)

    sys_note = rf"$\mathbf{{O+O @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$"

    for i, (y0,y1,yname) in enumerate(y_windows):
        bands_cent = cnm.cnm_vs_centrality((y0,y1), pt_range_avg, components=CALC_COMPS)
        ax = axes[i]
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands_cent: continue
            Rc, Rlo, Rhi, mbc, mblo, mbhi = bands_cent[comp]
            color = COLORS.get(comp, "black")
            ls = "-" if comp == "cnm" else "--" if comp == "npdf" else "-"
            lw = 2.0 if comp == "cnm" else 1.5
            
            xe, yc_s = cent_step_arrays(cnm.cent_bins, Rc)
            ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
            ax.fill_between(xe, cent_step_arrays(cnm.cent_bins, Rlo)[1], cent_step_arrays(cnm.cent_bins, Rhi)[1], step="post", color=color, alpha=0.15, lw=0)
            
            # MB band
            ax.fill_between([0,100], [mblo]*2, [mbhi]*2, color=color, alpha=0.08, hatch="//", lw=0)
            ax.hlines(mbc, 0, 100, colors=color, linestyles=":", linewidth=1.5)

        ax.text(0.92, 0.94, yname, transform=ax.transAxes, ha="right", va="top", fontsize=11, color="navy", fontweight="bold")
        if i == 0:
            ax.text(0.95, 0.05, sys_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
        if i == 1:
            ax.legend(loc="lower left", frameon=False, fontsize=10)

        ax.set_xlim(0, 100)
        if Y_LIM_RPA is None:
            lims = compute_ylim_from_bands(bands_cent)
            ax.set_ylim(*lims)
        else:
            ax.set_ylim(*Y_LIM_RPA)

        ax.set_xlabel("Centrality [%]", fontsize=14)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.axhline(1.0, color="k", ls="-", lw=1.0)
        if i == 0: ax.set_ylabel(r"$R^{\Upsilon}_{AA}$", fontsize=16)

    fig.tight_layout()
    return fig

# ── Execution ────────────────────────────────────────────────────────
def run_energy(energy):
    cnm = build_eloss_context(energy)
    etag = energy.replace('.','p')
    OUTDIR_CENT.mkdir(parents=True, exist_ok=True)
    OUTDIR_MB.mkdir(parents=True, exist_ok=True)
    
    # R_AA vs y
    print(f"  [PLOT] R_AA vs y ...", flush=True)
    yc, tags_y, bands_y = cnm.cnm_vs_y(y_edges=Y_EDGES, pt_range_avg=PT_RANGE_AVG, components=CALC_COMPS, include_mb=True)
    fig_y = plot_rpa_vs_y_grid(cnm, yc, tags_y, bands_y, energy)
    fig_y.savefig(OUTDIR_CENT / f"Upsilon_RAA_CNM_vs_y_OO_{etag}TeV_Grid.pdf", bbox_inches="tight")
    fig_y.savefig(OUTDIR_CENT / f"Upsilon_RAA_CNM_vs_y_OO_{etag}TeV_Grid.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_y)
    save_csv_y(OUTDIR_CENT, yc, bands_y, tags_y, energy)
    
    # MB R_AA vs y (Subfigure style)
    fig_mb_y, ax_mb_y = plt.subplots(figsize=(8,6), dpi=DPI)
    for comp in COMPONENTS_TO_PLOT:
        if comp not in bands_y: continue
        Rc = bands_y[comp][0]["MB"]
        xe, yc_s = step_from_centers(yc, Rc)
        ls = "--" if comp == "npdf" else "-"
        lw = 2.4 if comp == "cnm" else 1.8
        ax_mb_y.step(xe, yc_s, where="post", lw=lw, ls=ls, color=COLORS[comp], label=LABELS[comp])
        ax_mb_y.fill_between(xe, step_from_centers(yc, bands_y[comp][1]["MB"])[1], step_from_centers(yc, bands_y[comp][2]["MB"])[1], step="post", color=COLORS[comp], alpha=0.1, lw=0)

    ax_mb_y.set_xlim(-5, 5)
    if Y_LIM_RPA is None:
        lims = compute_ylim_from_bands(bands_y)
        ax_mb_y.set_ylim(*lims)
    else:
        ax_mb_y.set_ylim(*Y_LIM_RPA)
    ax_mb_y.axhline(1.0, color="k", ls="-", lw=0.8)
    ax_mb_y.set_xlabel(r"$y$", fontsize=14); ax_mb_y.set_ylabel(r"$R^{\Upsilon}_{AA}$", fontsize=14)
    ax_mb_y.legend(loc="lower right", frameon=False, fontsize=11)
    ax_mb_y.text(0.05, 0.95, rf"$\mathbf{{O+O @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$ (Min. Bias)", transform=ax_mb_y.transAxes, ha="left", va="top", fontsize=12)
    fig_mb_y.savefig(OUTDIR_MB / f"Upsilon_RAA_CNM_vs_y_MB_OO_{etag}TeV.pdf", bbox_inches="tight")
    fig_mb_y.savefig(OUTDIR_MB / f"Upsilon_RAA_CNM_vs_y_MB_OO_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_mb_y)
    
    # R_AA vs pT
    print(f"  [PLOT] R_AA vs pT ...", flush=True)
    fig_pt = plot_rpa_vs_pT_grid(cnm, energy, Y_WINDOWS, P_EDGES)
    fig_pt.savefig(OUTDIR_CENT / f"Upsilon_RAA_CNM_vs_pT_Grid_OO_{etag}TeV.pdf", bbox_inches="tight")
    fig_pt.savefig(OUTDIR_CENT / f"Upsilon_RAA_CNM_vs_pT_Grid_OO_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_pt)

    for y0, y1, yname in Y_WINDOWS:
        pc, tags_pT, bands_pT = cnm.cnm_vs_pT((y0, y1), P_EDGES, components=CALC_COMPS, include_mb=True)
        save_csv_pT(OUTDIR_CENT, pc, bands_pT, tags_pT, energy, yname)

    # MB-only pT (for multiple windows)
    fig_mb_pt, axes_mb_pt = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI, sharey=True)
    plt.subplots_adjust(wspace=0)
    for i, (y0, y1, yname) in enumerate(Y_WINDOWS):
        ax = axes_mb_pt[i]
        pc_mb, tags_pt_mb, bands_pt_mb = cnm.cnm_vs_pT((y0, y1), P_EDGES, components=CALC_COMPS, include_mb=True)
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands_pt_mb: continue
            Rc_mb = bands_pt_mb[comp][0]["MB"]
            xe, yc_s = step_from_centers(pc_mb, Rc_mb)
            ls = "--" if comp == "npdf" else "-"
            lw = 2.4 if comp == "cnm" else 1.8
            ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=COLORS.get(comp, "black"), label=LABELS.get(comp, comp) if i==1 else None)
            ax.fill_between(xe, step_from_centers(pc_mb, bands_pt_mb[comp][1]["MB"])[1], step_from_centers(pc_mb, bands_pt_mb[comp][2]["MB"])[1], step="post", color=COLORS.get(comp, "black"), alpha=0.1, lw=0)
        
        ax.axhline(1.0, color="k", ls="-", lw=0.8)
        ax.set_xlim(*X_LIM_PT)
        if Y_LIM_RPA is None:
            lims = compute_ylim_from_bands(bands_pt_mb)
            ax.set_ylim(*lims)
        else:
            ax.set_ylim(*Y_LIM_RPA)
        ax.text(0.5, 0.92, yname, transform=ax.transAxes, ha="center", va="top", fontsize=11, color="navy", fontweight="bold")
        ax.set_xlabel(r"$p_T$ (GeV)", fontsize=12)
        if i == 0:
            ax.set_ylabel(r"$R^{\Upsilon}_{AA}$ (CNM)", fontsize=14)
            ax.text(0.05, 0.05, rf"$\mathbf{{O+O @ {energy} TeV}}$"+"\nMin. Bias", transform=ax.transAxes, fontsize=10)
    axes_mb_pt[1].legend(loc="lower right", frameon=False, fontsize=10)
    fig_mb_pt.tight_layout()
    fig_mb_pt.savefig(OUTDIR_MB / f"Upsilon_RAA_CNM_vs_pT_MB_OO_{etag}TeV.pdf", bbox_inches="tight")
    fig_mb_pt.savefig(OUTDIR_MB / f"Upsilon_RAA_CNM_vs_pT_MB_OO_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_mb_pt)
    
    # R_AA vs Centrality
    print(f"  [PLOT] R_AA vs centrality ...", flush=True)
    fig_C = plot_rpa_vs_centrality(cnm, energy, Y_WINDOWS, PT_RANGE_AVG)
    fig_C.savefig(OUTDIR_CENT / f"Upsilon_RAA_CNM_vs_centrality_OO_{etag}TeV.pdf", bbox_inches="tight")
    fig_C.savefig(OUTDIR_CENT / f"Upsilon_RAA_CNM_vs_centrality_OO_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_C)

    for y0, y1, yname in Y_WINDOWS:
        bands_cent = cnm.cnm_vs_centrality((y0,y1), PT_RANGE_AVG, components=CALC_COMPS)
        save_csv_cent(OUTDIR_CENT, cnm, bands_cent, yname, energy)
    
    print(f"  ✓ DONE — O+O @ {energy} TeV")

def main():
    for e in ENERGIES: run_energy(e)

if __name__ == "__main__":
    main()
