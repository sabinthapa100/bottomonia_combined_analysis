#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_upsilon_npdf_OO.py
======================
Upsilon nPDF (min-bias + centrality) analysis for O+O @ 5.36 TeV.

Uses pre-computed gluon nPDF ratios from nPDF_OO.dat (49 EPPS21 sets).
  R_AA^{nPDF}(y, pT) = Rg1(x1) * Rg2(x2)

For centrality dependence, we treat the combined min-bias ratio as a single effective
nPDF modification and apply standard spatial weighting logic via npdf_centrality.py.

Outputs:
  outputs/npdf/min_bias/OO_5p36TeV/  — MB results
  outputs/npdf/centrality/OO_5p36TeV/ — Centrality dependent results
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path("/home/sawin/Desktop/bottomonia_combined_analysis/")
NPDF_CODE_DIR = ROOT / "cnm" / "npdf_code"
if str(NPDF_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(NPDF_CODE_DIR))

from npdf_OO_data import load_OO_dat, build_OO_rpa_grid, bin_rpa_vs_y_OO, bin_rpa_vs_pT_OO
from gluon_ratio import GluonEPPSProvider, EPPS21Ratio
from glauber import OpticalGlauber, SystemSpec
from npdf_centrality import (
    compute_df49_by_centrality,
    make_centrality_weight_dict,
    bin_rpa_vs_y,
    bin_rpa_vs_pT,
    bin_rpa_vs_centrality,
)

# ── Config ───────────────────────────────────────────────────────────
OO_DAT = ROOT / "inputs" / "npdf" / "OxygenOxygen5360" / "nPDF_OO.dat"
OUTDIR_MB = ROOT / "outputs" / "npdf" / "min_bias" / "OO_5p36TeV"
OUTDIR_CENT = ROOT / "outputs" / "npdf" / "centrality" / "OO_5p36TeV"
OUTDIR_MB.mkdir(exist_ok=True, parents=True)
OUTDIR_CENT.mkdir(exist_ok=True, parents=True)

DPI = 150
ALPHA_BAND = 0.22
PT_MAX = 15.0

SQRT_SNN = 5360.0  # GeV
SIG_NN_MB = 68.0  # approx for 5.36 TeV
SYSTEM_LABEL = r"O+O @ $\sqrt{s_{NN}}=5.36$ TeV"

Y_EDGES = np.arange(-5.0, 5.0 + 0.5, 0.5)
P_EDGES = np.arange(0.0, 15.0 + 1.0, 1.0)
PT_RANGE_AVG = (0.0, 15.0)
PT_FLOOR_W = 1.0

CENT_BINS = [(0,20),(20,40),(40,60),(60,80),(80,100)]
NB_BSAMPLES = 5
MB_C0 = 0.25

# Requested Rapidity Windows
Y_WINDOWS = [
    (-5.0, -2.5, r"$-5.0 < y < -2.5$"),
    (-2.4,  2.4, r"$-2.4 < y < 2.4$"),
    ( 2.5,  4.0, r"$2.5 < y < 4.0$"),
]

# ── Helpers ──────────────────────────────────────────────────────────
def tags_for_cent_bins(cb, include_mb=True):
    t = [f"{int(a)}-{int(b)}%" for (a,b) in cb]
    if include_mb: t.append("MB")
    return t

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

def ncoll_by_cent_bins(gl, cent_bins):
    nc = [gl.ncoll_mean_bin_AA_optical(a/100.0, b/100.0) for (a,b) in cent_bins]
    nc_mb = gl.ncoll_mean_bin_AA_optical(0.0, 1.0)
    return np.asarray(nc, float), float(nc_mb)

# ── Analysis Wrappers ────────────────────────────────────────────────
def npdf_vs_y_wrapper(ctx, y_edges, pt_range_avg, include_mb=True):
    wcent = make_centrality_weight_dict(ctx["cent_bins"], c0=MB_C0) if include_mb else None
    out = bin_rpa_vs_y(
        ctx["df49_by_cent"], ctx["df_pp"], ctx["df_pa"], ctx["gluon"],
        cent_bins=ctx["cent_bins"], y_edges=y_edges, pt_range_avg=pt_range_avg,
        weight_mode="flat", pt_floor_w=PT_FLOOR_W,
        wcent_dict=wcent, include_mb=include_mb)
    yc = 0.5*(y_edges[:-1]+y_edges[1:])
    tags = tags_for_cent_bins(ctx["cent_bins"], include_mb=include_mb)
    bands = {t: (np.asarray(out[t]["r_central"],float),
                 np.asarray(out[t]["r_lo"],float),
                 np.asarray(out[t]["r_hi"],float)) for t in tags}
    return yc, tags, bands

def npdf_vs_pT_wrapper(ctx, y_window, pt_edges, include_mb=True):
    y0, y1 = y_window
    wcent = make_centrality_weight_dict(ctx["cent_bins"], c0=MB_C0) if include_mb else None
    out = bin_rpa_vs_pT(
        ctx["df49_by_cent"], ctx["df_pp"], ctx["df_pa"], ctx["gluon"],
        cent_bins=ctx["cent_bins"], pt_edges=pt_edges, y_window=(y0,y1),
        weight_mode="flat", pt_floor_w=PT_FLOOR_W,
        wcent_dict=wcent, include_mb=include_mb)
    pc = 0.5*(pt_edges[:-1]+pt_edges[1:])
    tags = tags_for_cent_bins(ctx["cent_bins"], include_mb=include_mb)
    bands = {t: (np.asarray(out[t]["r_central"],float),
                 np.asarray(out[t]["r_lo"],float),
                 np.asarray(out[t]["r_hi"],float)) for t in tags}
    return pc, tags, bands

def npdf_vs_centrality_wrapper(ctx, y_window, pt_range_avg):
    y0, y1 = y_window
    wcent = make_centrality_weight_dict(ctx["cent_bins"], c0=MB_C0)
    ww = np.array([wcent[f"{int(a)}-{int(b)}%"] for (a,b) in ctx["cent_bins"]], float)
    out = bin_rpa_vs_centrality(
        ctx["df49_by_cent"], ctx["df_pp"], ctx["df_pa"], ctx["gluon"],
        cent_bins=ctx["cent_bins"], y_window=(y0,y1), pt_range_avg=pt_range_avg,
        weight_mode="flat", pt_floor_w=PT_FLOOR_W,
        width_weights=ww)
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in ctx["cent_bins"]]
    Rc = np.asarray(out["r_central"], float)
    Rlo= np.asarray(out["r_lo"], float)
    Rhi= np.asarray(out["r_hi"], float)
    mb = (float(out["mb_r_central"]), float(out["mb_r_lo"]), float(out["mb_r_hi"]))
    return labels, Rc, Rlo, Rhi, mb

# ── CSV savers (Consolidated HEPData Style) ─────────────────────────
def save_consolidated_y_csv(filepath, yc, bands_y, tags):
    ye = edges_from_centers(yc)
    rows = []
    for tag in tags:
        Rc, Rlo, Rhi = bands_y[tag]
        for i in range(len(yc)):
            rows.append({
                "Variable": "Rapidity (y)",
                "Bin_Low": float(ye[i]), "Bin_High": float(ye[i+1]),
                "Centrality": tag,
                "RAA_Central": float(Rc[i]), "RAA_Err_Lo": float(Rlo[i]), "RAA_Err_Hi": float(Rhi[i])
            })
    pd.DataFrame(rows).to_csv(filepath, index=False)

def save_consolidated_pT_csv(filepath, pc, all_bands_pT, tags):
    # all_bands_pT: {yname: bands_dict}
    pe = edges_from_centers(pc)
    rows = []
    for yname, bands_pt in all_bands_pT.items():
        for tag in tags:
            Rc, Rlo, Rhi = bands_pt[tag]
            for i in range(len(pc)):
                rows.append({
                    "Rapidity_Window": yname,
                    "pT_Low": float(pe[i]), "pT_High": float(pe[i+1]),
                    "Centrality": tag,
                    "RAA_Central": float(Rc[i]), "RAA_Err_Lo": float(Rlo[i]), "RAA_Err_Hi": float(Rhi[i])
                })
    pd.DataFrame(rows).to_csv(filepath, index=False)

def save_consolidated_cent_csv(filepath, ctx, cent_data_all, Ncoll_cent, Ncoll_MB):
    # cent_data_all: {yname: (labels, Rc, Rlo, Rhi, mb_tuple)}
    rows = []
    for yname, (labels, Rc, Rlo, Rhi, mb) in cent_data_all.items():
        # Centrality bins
        for i, ((cL,cR), lab) in enumerate(zip(ctx["cent_bins"], labels)):
            rows.append({
                "Rapidity_Window": yname,
                "Cent_Low": float(cL), "Cent_High": float(cR),
                "Ncoll": float(Ncoll_cent[i]),
                "RAA_Central": float(Rc[i]), "RAA_Err_Lo": float(Rlo[i]), "RAA_Err_Hi": float(Rhi[i]),
                "is_MB": False
            })
        # MB bin
        rows.append({
            "Rapidity_Window": yname,
            "Cent_Low": 0.0, "Cent_High": 100.0,
            "Ncoll": float(Ncoll_MB),
            "RAA_Central": float(mb[0]), "RAA_Err_Lo": float(mb[1]), "RAA_Err_Hi": float(mb[2]),
            "is_MB": True
        })
    pd.DataFrame(rows).to_csv(filepath, index=False)

# ── Plotters (Style parity with pPb) ──────────────────────────────────
def plot_rpa_vs_y_grid(ctx, yc, tags, bands):
    n_tags = len(tags)
    n_cols = 4
    n_rows = (n_tags + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5*n_rows),
                             sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)
    axes_flat = axes.flatten()

    for ip, tag in enumerate(tags):
        ax = axes_flat[ip]
        Rc, Rlo, Rhi = bands[tag]
        xe, yc_s = step_from_centers(yc, Rc)
        _, ylo_s = step_from_centers(yc, Rlo)
        _, yhi_s = step_from_centers(yc, Rhi)
        ax.step(xe, yc_s, where="post", lw=1.8, color="tab:blue")
        ax.fill_between(xe, ylo_s, yhi_s, step="post", color="tab:blue",
                        alpha=ALPHA_BAND, linewidth=0)
        ax.axhline(1.0, color="k", ls="-", lw=0.8)
        ax.text(0.95, 0.92, tag, transform=ax.transAxes, ha="right", va="top",
                weight="bold", fontsize=11)
        if ip == 0:
            ax.text(0.28, 0.88, SYSTEM_LABEL, transform=ax.transAxes,
                    weight="bold", fontsize=10)
            ax.text(0.95, 0.08,
                    rf"$p_T \in [{PT_RANGE_AVG[0]:.0f},\,{PT_RANGE_AVG[1]:.0f}]$ GeV",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
        ax.set_xlim(-5, 5); ax.set_ylim(0.0, 1.50)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
        ax.label_outer()

    for j in range(n_tags, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.text(0.5, 0.02, r"$y$", ha="center", fontsize=18)
    fig.text(0.07, 0.5, r"$R^{\Upsilon}_{AA}$ (nPDF)", va="center",
             rotation="vertical", fontsize=18)
    return fig

def plot_rpa_vs_pT_grid(ctx, y_windows):
    _, tags_temp, _ = npdf_vs_pT_wrapper(ctx, y_windows[0][:2], P_EDGES, include_mb=True)
    n_rows = len(y_windows); n_cols = len(tags_temp)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.0*n_rows),
                             sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)

    for row, (y0,y1,yname) in enumerate(y_windows):
        pc, tags_pt, npdf_pt = npdf_vs_pT_wrapper(ctx, (y0,y1), P_EDGES, include_mb=True)
        for col, tag in enumerate(tags_pt):
            if col >= n_cols: break
            ax = axes[row, col]
            Rc, Rlo, Rhi = npdf_pt[tag]
            xe, yc_s = step_from_centers(pc, Rc)
            _, ylo_s = step_from_centers(pc, Rlo)
            _, yhi_s = step_from_centers(pc, Rhi)
            ax.step(xe, yc_s, where="post", lw=1.5, color="tab:blue")
            ax.fill_between(xe, ylo_s, yhi_s, step="post", color="tab:blue",
                            alpha=0.3, linewidth=0)
            ax.axhline(1.0, color="gray", ls="--", lw=0.8)
            ax.text(0.08, 0.08, yname, transform=ax.transAxes, color="navy",
                    fontsize=10, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
            ax.text(0.92, 0.88, tag, transform=ax.transAxes, ha="right",
                    weight="bold", fontsize=10)
            if row == 0 and col == 0:
                ax.text(0.28, 0.40, SYSTEM_LABEL, transform=ax.transAxes,
                        weight="bold", fontsize=10)
            ax.set_xlim(0, 15); ax.set_ylim(0.0, 1.50)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune="both"))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=9, prune="both"))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(axis="both", which="both", direction="in",
                           top=True, right=True, labelsize=11)
            ax.label_outer()

    fig.text(0.5, 0.02, r"$p_T$ (GeV)", ha="center", fontsize=20)
    fig.text(0.07, 0.5, r"$R^{\Upsilon}_{AA}$ (nPDF)", va="center",
             rotation="vertical", fontsize=20)
    return fig

def plot_rpa_vs_centrality(ctx, Ncoll_cent, Ncoll_MB, npdf_cent_all):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=DPI, sharey=False)

    for ax, (y0,y1,yname) in zip(axes, Y_WINDOWS):
        labels, Rc, Rlo, Rhi, mb = npdf_cent_all[yname]
        xe, yc_s = cent_step_arrays(ctx["cent_bins"], Rc)
        _, ylo_s = cent_step_arrays(ctx["cent_bins"], Rlo)
        _, yhi_s = cent_step_arrays(ctx["cent_bins"], Rhi)
        ax.step(xe, yc_s, where="post", lw=2.0, color="tab:blue",
                label="nPDF" if ax is axes[0] else None)
        ax.fill_between(xe, ylo_s, yhi_s, step="post", color="tab:blue",
                        alpha=ALPHA_BAND, linewidth=0)
        mb_c, mb_lo, mb_hi = mb
        ax.fill_between([0,100],[mb_lo]*2,[mb_hi]*2, color="tab:blue",
                        alpha=0.12, hatch="//", linewidth=0)
        ax.hlines(mb_c, 0, 100, colors="tab:blue", linestyles="--", linewidth=1.8)
        ax.text(0.92, 0.94, yname, transform=ax.transAxes, ha="right",
                va="top", fontsize=10)
        if ax is axes[0]:
            ax.text(0.03, 0.05, SYSTEM_LABEL, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.axhline(1.0, color="k", ls="-", lw=1.0)
        ax.set_xlabel("Centrality [%]")
        ax.set_ylabel(r"$R^{\Upsilon}_{AA}$ (nPDF)")
        ax.set_xlim(0, 100); ax.set_ylim(0.35, 1.25)
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig

def plot_rpa_vs_Ncoll(ctx, Ncoll_cent, Ncoll_MB, npdf_cent_all):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=DPI, sharey=False)
    xN_edges = edges_from_centers(Ncoll_cent)

    for ax, (y0,y1,yname) in zip(axes, Y_WINDOWS):
        labels, Rc, Rlo, Rhi, mb = npdf_cent_all[yname]
        mb_c, mb_lo, mb_hi = mb
        yc_s = np.concatenate([Rc, Rc[-1:]])
        ylo_s= np.concatenate([Rlo,Rlo[-1:]])
        yhi_s= np.concatenate([Rhi,Rhi[-1:]])
        ax.step(xN_edges, yc_s, where="post", lw=2.0, color="tab:blue",
                label="nPDF" if ax is axes[0] else None)
        ax.fill_between(xN_edges, ylo_s, yhi_s, step="post", color="tab:blue",
                        alpha=ALPHA_BAND, linewidth=0)
        ax.fill_between([xN_edges[0],xN_edges[-1]],[mb_lo]*2,[mb_hi]*2,
                        color="tab:blue", alpha=0.12, hatch="//", linewidth=0)
        ax.hlines(mb_c, xN_edges[0], xN_edges[-1], colors="tab:blue",
                  linestyles="--", linewidth=1.8)
        ax.text(0.92, 0.94, yname, transform=ax.transAxes, ha="right",
                va="top", fontsize=10)
        if ax is axes[0]:
            ax.text(0.03, 0.05, SYSTEM_LABEL, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.axhline(1.0, color="k", ls="-", lw=1.0)
        ax.set_xlabel(r"$\langle N_{\rm coll}\rangle$")
        ax.set_ylabel(r"$R^{\Upsilon}_{AA}$ (nPDF)")
        ax.set_ylim(0.35, 1.25)
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig

# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Upsilon nPDF — O+O @ 5.36 TeV  (Min-Bias + Centrality)")
    print("  STYLE: PARITY WITH pPb ANALYSIS")
    print("="*60)

    # 1. Load Data
    print("\n[1] Loading nPDF_OO.dat ...")
    data = load_OO_dat(str(OO_DAT))
    grid = build_OO_rpa_grid(data, pt_max=PT_MAX)
    print(f"    ✓ Grid: {len(grid)} points loaded")

    # 2. Build Glauber for O-O
    print("\n[2] Building Optical Glauber (O-O) ...")
    gl = OpticalGlauber(SystemSpec("AA", SQRT_SNN, A=16, sigma_nn_mb=SIG_NN_MB), verbose=False)
    Ncoll_cent, Ncoll_MB = ncoll_by_cent_bins(gl, CENT_BINS)
    print(f"    ✓ Ncoll (MB): {Ncoll_MB:.2f}, bins: {np.round(Ncoll_cent, 2)}")

    # 3. Centrality Dependence
    print("\n[3] Computing Centrality Dependence Grid ...")
    r0 = grid["r_central"].to_numpy()
    M = grid[[f"r_mem_{i:03d}" for i in range(1, 49)]].to_numpy().T # (48, N)
    
    # CRITICAL FIX for AA symmetry: Build SA_all directly from the grid 
    # (which has Rg1*Rg2 from the file) instead of recomputing it.
    SA_all = np.vstack([r0[None, :], M]) # shape (49, N)

    epps_wrapper = EPPS21Ratio(A=16, path=str(ROOT / "inputs" / "npdf" / "nPDFs"))
    gluon_provider = GluonEPPSProvider(epps_wrapper, SQRT_SNN, m_state_GeV=10.01)

    df49_by_cent, K_by_cent, _, Y_SHIFT = compute_df49_by_centrality(
        grid, r0, M, gluon_provider, gl,
        cent_bins=CENT_BINS, nb_bsamples=NB_BSAMPLES, kind="AA",
        SA_all=SA_all
    )

    ctx = dict(cent_bins=CENT_BINS, df49_by_cent=df49_by_cent, df_pp=grid, df_pa=grid,
               gluon=gluon_provider, gl=gl, sqrt_sNN=SQRT_SNN)

    # ── Plots ──
    print("\n[4] Generating Figures (pPb-parity) ...")

    # (A) R_AA vs y Grid
    print(f"  [PLOT] R_AA vs y — full grid ...")
    yc, tags_y, bands_y = npdf_vs_y_wrapper(ctx, Y_EDGES, PT_RANGE_AVG, include_mb=True)
    # Ensure MB band in result uses the file data directly
    mb_file_y = bin_rpa_vs_y_OO(grid, Y_EDGES, pt_range=PT_RANGE_AVG)
    bands_y["MB"] = (mb_file_y["r_central"].to_numpy(), mb_file_y["r_lo"].to_numpy(), mb_file_y["r_hi"].to_numpy())

    fig = plot_rpa_vs_y_grid(ctx, yc, tags_y, bands_y)
    fig.savefig(OUTDIR_CENT / "Upsilon_RAA_nPDF_vs_y_OO_5p36TeV_FullGrid.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / "Upsilon_RAA_nPDF_vs_y_OO_5p36TeV_FullGrid.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    save_consolidated_y_csv(OUTDIR_CENT / "Table_RAA_vs_y_Grid_OO_5p36TeV.csv", yc, bands_y, tags_y)

    # (B) MB-only vs y
    print(f"  [PLOT] MB-only vs y ...")
    Rc_mb_y, Rlo_mb_y, Rhi_mb_y = bands_y["MB"]
    fig_mb_y, ax_mb_y = plt.subplots(figsize=(8,5), dpi=DPI)
    xe_mb_y, yc_sm = step_from_centers(yc, Rc_mb_y)
    ax_mb_y.step(xe_mb_y, yc_sm, where="post", lw=2, color="tab:blue", label="nPDF (EPPS21)")
    ax_mb_y.fill_between(xe_mb_y, step_from_centers(yc, Rlo_mb_y)[1], step_from_centers(yc, Rhi_mb_y)[1],
                         step="post", color="tab:blue", alpha=ALPHA_BAND, linewidth=0)
    ax_mb_y.axhline(1.0, color="k", ls="-", lw=0.8)
    ax_mb_y.set_xlabel(r"$y$", fontsize=14); ax_mb_y.set_ylabel(r"$R^{\Upsilon}_{AA}$ (nPDF)", fontsize=14)
    ax_mb_y.set_xlim(-5, 5); ax_mb_y.set_ylim(0.4, 1.2)
    ax_mb_y.legend(loc="lower right", frameon=False, fontsize=12)
    ax_mb_y.text(0.03, 0.95, SYSTEM_LABEL + " (Min. Bias)", transform=ax_mb_y.transAxes, ha="left", va="top", fontsize=13, fontweight="bold")
    fig_mb_y.savefig(OUTDIR_MB / "Upsilon_RAA_nPDF_vs_y_OO_5p36TeV.pdf", bbox_inches="tight")
    fig_mb_y.savefig(OUTDIR_MB / "Upsilon_RAA_nPDF_vs_y_OO_5p36TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_mb_y)

    # (C) R_AA vs pT Full Grid
    print(f"  [PLOT] R_AA vs pT — full grid ...")
    fig = plot_rpa_vs_pT_grid(ctx, Y_WINDOWS)
    fig.savefig(OUTDIR_CENT / "Upsilon_RAA_nPDF_vs_pT_OO_5p36TeV_FullGrid.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / "Upsilon_RAA_nPDF_vs_pT_OO_5p36TeV_FullGrid.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    
    all_bands_pT = {}
    for y0, y1, yname in Y_WINDOWS:
        pc, tags_pt, bands_pt = npdf_vs_pT_wrapper(ctx, (y0,y1), P_EDGES, include_mb=True)
        all_bands_pT[yname] = bands_pt
    save_consolidated_pT_csv(OUTDIR_CENT / "Table_RAA_vs_pT_Grid_OO_5p36TeV.csv", pc, all_bands_pT, tags_pt)

    # (D) MB-only vs pT (3 windows)
    print(f"  [PLOT] MB-only vs pT (3 windows) ...")
    fig_mb_pt, axes_mb_pt = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI, sharey=True)
    for ax, (y0, y1, yname) in zip(axes_mb_pt, Y_WINDOWS):
        res_pt_mb = bin_rpa_vs_pT_OO(grid, P_EDGES, y_range=(y0, y1))
        pc_mb = 0.5 * (P_EDGES[:-1] + P_EDGES[1:])
        Rc, Rlo, Rhi = res_pt_mb["r_central"], res_pt_mb["r_lo"], res_pt_mb["r_hi"]
        xe, yc_sp = step_from_centers(pc_mb, Rc)
        ax.step(xe, yc_sp, where="post", lw=2, color="tab:blue", label="nPDF (EPPS21)")
        ax.fill_between(xe, step_from_centers(pc_mb, Rlo)[1], step_from_centers(pc_mb, Rhi)[1],
                        step="post", color="tab:blue", alpha=ALPHA_BAND, linewidth=0)
        ax.axhline(1.0, color="k", ls="-", lw=0.8)
        ax.text(0.5, 0.92, yname, transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$p_T$ (GeV)", fontsize=12)
        if ax == axes_mb_pt[0]:
            ax.set_ylabel(r"$R^{\Upsilon}_{AA}$ (nPDF)", fontsize=12)
            ax.text(0.05, 0.05, SYSTEM_LABEL + "\nMin. Bias", transform=ax.transAxes, fontsize=10)
        ax.set_xlim(0, 15); ax.set_ylim(0.4, 1.2)
    fig_mb_pt.tight_layout()
    fig_mb_pt.savefig(OUTDIR_MB / "Upsilon_RAA_nPDF_vs_pT_OO_5p36TeV.pdf", bbox_inches="tight")
    fig_mb_pt.savefig(OUTDIR_MB / "Upsilon_RAA_nPDF_vs_pT_OO_5p36TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_mb_pt)

    # (E) R_AA vs Centrality %
    print(f"  [PLOT] R_AA vs centrality ...")
    npdf_cent_all = {}
    for y0, y1, yname in Y_WINDOWS:
        res = npdf_vs_centrality_wrapper(ctx, (y0,y1), PT_RANGE_AVG)
        npdf_cent_all[yname] = res
    fig = plot_rpa_vs_centrality(ctx, Ncoll_cent, Ncoll_MB, npdf_cent_all)
    fig.savefig(OUTDIR_CENT / "Upsilon_RAA_nPDF_vs_centrality_OO_5p36TeV.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / "Upsilon_RAA_nPDF_vs_centrality_OO_5p36TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # (F) R_AA vs Ncoll
    print(f"  [PLOT] R_AA vs Ncoll ...")
    fig = plot_rpa_vs_Ncoll(ctx, Ncoll_cent, Ncoll_MB, npdf_cent_all)
    fig.savefig(OUTDIR_CENT / "Upsilon_RAA_nPDF_vs_Ncoll_OO_5p36TeV.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / "Upsilon_RAA_nPDF_vs_Ncoll_OO_5p36TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    
    save_consolidated_cent_csv(OUTDIR_CENT / "Table_RAA_vs_Centrality_OO_5p36TeV.csv", ctx, npdf_cent_all, Ncoll_cent, Ncoll_MB)

    print("\n" + "="*60)
    print("  ✓ ALL DONE — O+O nPDF analysis complete (Style Parity with pPb)")
    print("  Outputs in:", OUTDIR_CENT.relative_to(ROOT))
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
