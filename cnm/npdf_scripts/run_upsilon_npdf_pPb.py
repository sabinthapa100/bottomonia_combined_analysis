#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_upsilon_npdf_pPb.py
=======================
Upsilon nPDF centrality + min-bias analysis for pPb @ 5.02 and 8.16 TeV.

Usage:
    python run_upsilon_npdf_pPb.py          # runs both energies
    python run_upsilon_npdf_pPb.py 5.02     # only 5.02 TeV
    python run_upsilon_npdf_pPb.py 8.16     # only 8.16 TeV

Outputs go to:
    outputs/npdf/min_bias/    — MB-only R_pA vs y (per energy)
    outputs/npdf/centrality/  — all centrality + MB plots and CSV

NO nuclear absorption. Pure nPDF (EPPS21) shadowing/anti-shadowing.
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
ROOT = Path("/home/sawin/Desktop/bottomonia_combined_analysis/")
NPDF_CODE_DIR = ROOT / "cnm" / "npdf_code"
if str(NPDF_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(NPDF_CODE_DIR))

from npdf_data import NPDFSystem, RpAAnalysis
from gluon_ratio import EPPS21Ratio, GluonEPPSProvider
from glauber import OpticalGlauber, SystemSpec
from npdf_centrality import (
    compute_df49_by_centrality,
    make_centrality_weight_dict,
    bin_rpa_vs_y,
    bin_rpa_vs_pT,
    bin_rpa_vs_centrality,
)

# ── Physics config ──────────────────────────────────────────────────
M_UPSILON_AVG = 10.01  # GeV, average bottomonium mass

SQRTS_GEV   = {"5.02": 5023.0, "8.16": 8160.0}
SIG_NN_MB   = {"5.02": 67.0,   "8.16": 71.0}
PREFIX_MAP  = {"5.02": "upsppb5_", "8.16": "upsppb_"}

NPDF_INPUT_DIR = ROOT / "inputs" / "npdf"
P5_DIR   = NPDF_INPUT_DIR / "pPb5TeV"
P8_DIR   = NPDF_INPUT_DIR / "pPb8TeV"
EPPS_DIR = NPDF_INPUT_DIR / "nPDFs"

OUTDIR_BASE_MB   = ROOT / "outputs" / "npdf" / "min_bias"
OUTDIR_BASE_CENT = ROOT / "outputs" / "npdf" / "centrality"

# ── Binning ─────────────────────────────────────────────────────────
CENT_BINS = [(0,10),(10,20),(20,40),(40,60),(60,80),(80,100)]

Y_EDGES = np.arange(-5.5, 5.0 + 0.5, 0.5)
P_EDGES = np.arange(0.0, 20.0 + 2.5, 2.5)

Y_WINDOWS = [
    (-4.46, -2.96, r"$-4.46 < y < -2.96$"),
    (-1.37,  0.43, r"$-1.37 < y < 0.43$"),
    ( 2.03,  3.53, r"$2.03 < y < 3.53$"),
]

PT_RANGE_AVG = (0.0, 15.0)
PT_FLOOR_W   = 1.0

WEIGHT_MODE      = "pp@local"
Y_REF            = 0.0
NB_BSAMPLES      = 5
Y_SHIFT_FRACTION = 2.0
MB_C0            = 0.25

DPI = 150
ALPHA_BAND = 0.22

# ── Builder ─────────────────────────────────────────────────────────
def build_npdf_context(energy):
    sqrt_sNN   = SQRTS_GEV[energy]
    sigma_nn   = SIG_NN_MB[energy]
    input_dir  = P5_DIR if energy == "5.02" else P8_DIR
    prefix     = PREFIX_MAP[energy]

    print(f"\n{'='*60}")
    print(f"  Upsilon nPDF  —  p+Pb @ {energy} TeV")
    print(f"  M_Upsilon = {M_UPSILON_AVG} GeV  (average)")
    print(f"  sigma_NN  = {sigma_nn} mb")
    print(f"  Input dir = {input_dir}")
    print(f"  Prefix    = {prefix}")
    print(f"{'='*60}")

    epps = EPPS21Ratio(A=208, path=str(EPPS_DIR))
    gluon = GluonEPPSProvider(epps, sqrt_sNN_GeV=sqrt_sNN,
                               m_state_GeV=M_UPSILON_AVG, y_sign_for_xA=-1)

    gl = OpticalGlauber(SystemSpec("pA", sqrt_sNN, A=208, sigma_nn_mb=sigma_nn),
                        verbose=False)

    sys_npdf = NPDFSystem.from_folder(str(input_dir), kick="pp",
                                       name=f"p+Pb {energy} TeV", prefix=prefix)
    print(f"  [OK] Loaded {len(sys_npdf.error_paths)+2} TOP files ({prefix}*)")

    ana = RpAAnalysis()
    base, r0, M = ana.compute_rpa_members(
        sys_npdf.df_pp, sys_npdf.df_pa, sys_npdf.df_errors,
        join="intersect", lowpt_policy="drop",
        pt_shift_min=PT_FLOOR_W, shift_if_r_below=0.0,
    )
    print(f"  [OK] RpA grid: {len(base)} points, {M.shape[0]} Hessian members")

    df49_by_cent, K_by_cent, SA_all, Y_SHIFT = compute_df49_by_centrality(
        base, r0, M, gluon, gl,
        cent_bins=CENT_BINS, nb_bsamples=NB_BSAMPLES,
        y_shift_fraction=Y_SHIFT_FRACTION,
    )
    print(f"  [OK] Centrality tables built, Y_SHIFT = {Y_SHIFT}")

    return dict(energy=energy, sqrt_sNN=sqrt_sNN, sigma_nn_mb=sigma_nn,
                cent_bins=CENT_BINS, df49_by_cent=df49_by_cent,
                df_pp=sys_npdf.df_pp, df_pa=sys_npdf.df_pa,
                gluon=gluon, gl=gl, K_by_cent=K_by_cent,
                SA_all=SA_all, Y_SHIFT=Y_SHIFT)

# ── Helpers ─────────────────────────────────────────────────────────
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
    edges = [cb[0][0]] + [b for (_,b) in cb]
    return np.asarray(edges, float), np.concatenate([vals, vals[-1:]])

def edges_from_centers(xc):
    xc = np.asarray(xc, float)
    if xc.size < 2: return np.array([xc[0]-0.5, xc[0]+0.5])
    mids = 0.5*(xc[:-1]+xc[1:])
    return np.concatenate([[xc[0]-(mids[0]-xc[0])], mids, [xc[-1]+(xc[-1]-mids[-1])]])

def ncoll_by_cent_bins(ctx, optical=True):
    gl = ctx["gl"]
    fn = gl.ncoll_mean_bin_pA_optical if optical else gl.ncoll_mean_bin_pA
    nc = [fn(a/100.0, b/100.0) for (a,b) in ctx["cent_bins"]]
    nc_mb = fn(0.0, 1.0)
    return np.asarray(nc, float), float(nc_mb)

def npdf_vs_centrality(ctx, y_window, pt_range_avg, mb_c0=MB_C0):
    y0, y1 = y_window
    wcent = make_centrality_weight_dict(ctx["cent_bins"], c0=mb_c0)
    ww = np.array([wcent[f"{int(a)}-{int(b)}%"] for (a,b) in ctx["cent_bins"]], float)
    out = bin_rpa_vs_centrality(
        ctx["df49_by_cent"], ctx["df_pp"], ctx["df_pa"], ctx["gluon"],
        cent_bins=ctx["cent_bins"], y_window=(y0,y1), pt_range_avg=pt_range_avg,
        weight_mode=WEIGHT_MODE, y_ref=Y_REF, pt_floor_w=PT_FLOOR_W,
        width_weights=ww)
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in ctx["cent_bins"]]
    Rc = np.asarray(out["r_central"], float)
    Rlo= np.asarray(out["r_lo"], float)
    Rhi= np.asarray(out["r_hi"], float)
    mb = (float(out["mb_r_central"]), float(out["mb_r_lo"]), float(out["mb_r_hi"]))
    return labels, Rc, Rlo, Rhi, mb

def npdf_vs_y(ctx, y_edges, pt_range_avg, include_mb=True, mb_c0=MB_C0):
    wcent = make_centrality_weight_dict(ctx["cent_bins"], c0=mb_c0) if include_mb else None
    out = bin_rpa_vs_y(
        ctx["df49_by_cent"], ctx["df_pp"], ctx["df_pa"], ctx["gluon"],
        cent_bins=ctx["cent_bins"], y_edges=y_edges, pt_range_avg=pt_range_avg,
        weight_mode=WEIGHT_MODE, y_ref=Y_REF, pt_floor_w=PT_FLOOR_W,
        wcent_dict=wcent, include_mb=include_mb)
    yc = 0.5*(y_edges[:-1]+y_edges[1:])
    tags = tags_for_cent_bins(ctx["cent_bins"], include_mb=include_mb)
    bands = {t: (np.asarray(out[t]["r_central"],float),
                 np.asarray(out[t]["r_lo"],float),
                 np.asarray(out[t]["r_hi"],float)) for t in tags}
    return yc, tags, bands

def npdf_vs_pT(ctx, y_window, pt_edges, include_mb=True, mb_c0=MB_C0):
    y0, y1 = y_window
    wcent = make_centrality_weight_dict(ctx["cent_bins"], c0=mb_c0) if include_mb else None
    out = bin_rpa_vs_pT(
        ctx["df49_by_cent"], ctx["df_pp"], ctx["df_pa"], ctx["gluon"],
        cent_bins=ctx["cent_bins"], pt_edges=pt_edges, y_window=(y0,y1),
        weight_mode=WEIGHT_MODE, y_ref=Y_REF, pt_floor_w=PT_FLOOR_W,
        wcent_dict=wcent, include_mb=include_mb)
    pc = 0.5*(pt_edges[:-1]+pt_edges[1:])
    tags = tags_for_cent_bins(ctx["cent_bins"], include_mb=include_mb)
    bands = {t: (np.asarray(out[t]["r_central"],float),
                 np.asarray(out[t]["r_lo"],float),
                 np.asarray(out[t]["r_hi"],float)) for t in tags}
    return pc, tags, bands

# ── CSV savers ──────────────────────────────────────────────────────
def save_csv_y(outdir, yc, bands, tags, energy, suffix=""):
    for tag in tags:
        Rc, Rlo, Rhi = bands[tag]
        df = pd.DataFrame({"y_center": yc, "R_central": Rc, "R_lo": Rlo, "R_hi": Rhi})
        st = tag.replace("%","pct").replace(" ","")
        df.to_csv(outdir / f"Upsilon_RpA_nPDF_vs_y_{st}_{energy.replace('.','p')}TeV{suffix}.csv", index=False)

def save_csv_pT(outdir, pc, bands, tags, energy, yname=""):
    for tag in tags:
        Rc, Rlo, Rhi = bands[tag]
        df = pd.DataFrame({"pT_center": pc, "R_central": Rc, "R_lo": Rlo, "R_hi": Rhi})
        st = tag.replace("%","pct").replace(" ","")
        yn = yname.replace(" ","").replace("<","").replace(">","").replace("$","").replace("\\","")
        df.to_csv(outdir / f"Upsilon_RpA_nPDF_vs_pT_{st}_{yn}_{energy.replace('.','p')}TeV.csv", index=False)

def save_csv_cent(outdir, ctx, labels, Rc, Rlo, Rhi, mb, Ncoll_cent, Ncoll_MB, yname, energy):
    rows = []
    for i, ((cL,cR), lab) in enumerate(zip(ctx["cent_bins"], labels)):
        rows.append(dict(cent_left=float(cL), cent_right=float(cR), label=lab,
                         Ncoll=float(Ncoll_cent[i]), R_central=float(Rc[i]),
                         R_lo=float(Rlo[i]), R_hi=float(Rhi[i]), is_MB=False))
    rows.append(dict(cent_left=0, cent_right=100, label="MB",
                     Ncoll=float(Ncoll_MB), R_central=float(mb[0]),
                     R_lo=float(mb[1]), R_hi=float(mb[2]), is_MB=True))
    yn = yname.replace(" ","").replace("<","").replace(">","").replace("$","").replace("\\","")
    pd.DataFrame(rows).to_csv(
        outdir / f"Upsilon_RpA_nPDF_vs_cent_{yn}_{energy.replace('.','p')}TeV.csv", index=False)

# ============= PLOTTING FUNCTIONS ====================================

def plot_rpa_vs_y_grid(ctx, yc, tags, bands, energy):
    """Full grid: one panel per centrality + MB as last panel."""
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
            elabel = rf"$\sqrt{{s_{{NN}}}}={ctx['sqrt_sNN']/1000:.2f}$ TeV"
            ax.text(0.28, 0.88, f"p+Pb @{elabel}", transform=ax.transAxes,
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
    fig.text(0.07, 0.5, r"$R^{\Upsilon}_{pA}$ (nPDF)", va="center",
             rotation="vertical", fontsize=18)
    return fig

def plot_rpa_vs_pT_grid(ctx, y_windows, energy):
    """Full grid: rows=rapidity windows, cols=centrality+MB."""
    _, tags_temp, _ = npdf_vs_pT(ctx, y_windows[0][:2], P_EDGES, include_mb=True)
    n_rows = len(y_windows); n_cols = len(tags_temp)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.0*n_rows),
                             sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)

    for row, (y0,y1,yname) in enumerate(y_windows):
        pc, tags_pt, npdf_pt = npdf_vs_pT(ctx, (y0,y1), P_EDGES, include_mb=True)
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
                elabel = rf"$\sqrt{{s_{{NN}}}}={ctx['sqrt_sNN']/1000:.2f}$ TeV"
                ax.text(0.28, 0.40, f"p+Pb @{elabel}", transform=ax.transAxes,
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
    fig.text(0.07, 0.5, r"$R^{\Upsilon}_{pA}$ (nPDF)", va="center",
             rotation="vertical", fontsize=20)
    return fig

def plot_rpa_vs_centrality(ctx, Ncoll_cent, Ncoll_MB, npdf_cent_all, energy):
    """Step plots vs centrality % with dashed MB overlay."""
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
            ax.text(0.03, 0.05,
                    rf"$\sqrt{{s_{{NN}}}}={ctx['sqrt_sNN']/1000:.2f}$ TeV",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.axhline(1.0, color="k", ls="-", lw=1.0)
        ax.set_xlabel("Centrality [%]")
        ax.set_ylabel(r"$R^{\Upsilon}_{pA}$ (nPDF)")
        ax.set_xlim(0, 100); ax.set_ylim(0.35, 1.25)
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig

def plot_rpa_vs_Ncoll(ctx, Ncoll_cent, Ncoll_MB, npdf_cent_all, energy):
    """Step plots vs <Ncoll> with dashed MB overlay."""
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
            ax.text(0.03, 0.05,
                    rf"$\sqrt{{s_{{NN}}}}={ctx['sqrt_sNN']/1000:.2f}$ TeV",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.axhline(1.0, color="k", ls="-", lw=1.0)
        ax.set_xlabel(r"$\langle N_{\rm coll}\rangle$")
        ax.set_ylabel(r"$R^{\Upsilon}_{pA}$ (nPDF)")
        ax.set_ylim(0.35, 1.25)
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig

# ============= MAIN PIPELINE ========================================

def run_energy(energy):
    ctx = build_npdf_context(energy)
    etag = energy.replace(".","p")
    OUTDIR_MB   = OUTDIR_BASE_MB   / f"pPb_{etag}TeV"
    OUTDIR_CENT = OUTDIR_BASE_CENT / f"pPb_{etag}TeV"
    OUTDIR_MB.mkdir(exist_ok=True, parents=True)
    OUTDIR_CENT.mkdir(exist_ok=True, parents=True)
    Ncoll_cent, Ncoll_MB = ncoll_by_cent_bins(ctx)
    print(f"  [Ncoll] bins: {np.round(Ncoll_cent,3)},  MB: {Ncoll_MB:.3f}")
    print(f"  [OUT]   min_bias  → {OUTDIR_MB}")
    print(f"  [OUT]   centrality → {OUTDIR_CENT}")

    # ── 1. R_pA vs y (centrality + MB grid) ──────────────────
    print(f"  [PLOT] R_pA vs y — full grid ...")
    yc, tags_y, bands_y = npdf_vs_y(ctx, Y_EDGES, PT_RANGE_AVG, include_mb=True)
    fig = plot_rpa_vs_y_grid(ctx, yc, tags_y, bands_y, energy)
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_y_{etag}TeV_FullGrid.pdf",
                bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_y_{etag}TeV_FullGrid.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    save_csv_y(OUTDIR_CENT, yc, bands_y, tags_y, energy)
    print(f"    → saved FullGrid y-plots + CSV in centrality/")

    # ── 1b. MB-only R_pA vs y (separate) ────────────────
    if "MB" in bands_y:
        Rc_mb, Rlo_mb, Rhi_mb = bands_y["MB"]
        fig_mb, ax_mb = plt.subplots(figsize=(8,5), dpi=DPI)
        xe, yc_s = step_from_centers(yc, Rc_mb)
        _, ylo_s = step_from_centers(yc, Rlo_mb)
        _, yhi_s = step_from_centers(yc, Rhi_mb)
        ax_mb.step(xe, yc_s, where="post", lw=2, color="tab:blue", label="nPDF (EPPS21)")
        ax_mb.fill_between(xe, ylo_s, yhi_s, step="post", color="tab:blue",
                           alpha=ALPHA_BAND, linewidth=0)
        ax_mb.axhline(1.0, color="k", ls="-", lw=0.8)
        ax_mb.set_xlabel(r"$y$", fontsize=14)
        ax_mb.set_ylabel(r"$R^{\Upsilon}_{pA}$ (nPDF)", fontsize=14)
        ax_mb.set_xlim(-5, 5); ax_mb.set_ylim(0.4, 1.3)
        ax_mb.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax_mb.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax_mb.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_mb.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_mb.tick_params(which="both", direction="in", top=True, right=True)
        ax_mb.legend(loc="lower right", frameon=False, fontsize=12)
        elabel = rf"p+Pb @ $\sqrt{{s_{{NN}}}}={ctx['sqrt_sNN']/1000:.2f}$ TeV"
        ax_mb.text(0.03, 0.95, elabel + "  (Min. Bias)", transform=ax_mb.transAxes,
                   ha="left", va="top", fontsize=13, fontweight="bold")
        fig_mb.tight_layout()
        fig_mb.savefig(OUTDIR_MB / f"Upsilon_RpA_nPDF_vs_y_MB_{etag}TeV.pdf",
                       bbox_inches="tight")
        fig_mb.savefig(OUTDIR_MB / f"Upsilon_RpA_nPDF_vs_y_MB_{etag}TeV.png",
                       dpi=DPI, bbox_inches="tight")
        pd.DataFrame({"y_center": yc, "R_central": Rc_mb, "R_lo": Rlo_mb, "R_hi": Rhi_mb})\
          .to_csv(OUTDIR_MB / f"Upsilon_RpA_nPDF_vs_y_MB_{etag}TeV.csv", index=False)
        plt.close(fig_mb)
        print(f"    → saved MB-only y-plot + CSV in min_bias/")

    # ── 2. R_pA vs pT (full grid) ────────────────────────
    print(f"  [PLOT] R_pA vs pT — full grid ...")
    fig = plot_rpa_vs_pT_grid(ctx, Y_WINDOWS, energy)
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_pT_{etag}TeV_FullGrid.pdf",
                bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_pT_{etag}TeV_FullGrid.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    for y0,y1,yname in Y_WINDOWS:
        pc, tags_pt, npdf_pt = npdf_vs_pT(ctx, (y0,y1), P_EDGES, include_mb=True)
        save_csv_pT(OUTDIR_CENT, pc, npdf_pt, tags_pt, energy, yname)
    print(f"    → saved FullGrid pT-plots + CSV in centrality/")

    # ── 3. R_pA vs centrality % ──────────────────────────
    print(f"  [PLOT] R_pA vs centrality ...")
    npdf_cent_all = {}
    for y0,y1,yname in Y_WINDOWS:
        labels, Rc, Rlo, Rhi, mb = npdf_vs_centrality(ctx, (y0,y1), PT_RANGE_AVG)
        npdf_cent_all[yname] = (labels, Rc, Rlo, Rhi, mb)
        save_csv_cent(OUTDIR_CENT, ctx, labels, Rc, Rlo, Rhi, mb,
                      Ncoll_cent, Ncoll_MB, yname, energy)
    fig = plot_rpa_vs_centrality(ctx, Ncoll_cent, Ncoll_MB, npdf_cent_all, energy)
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_centrality_{etag}TeV.pdf",
                bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_centrality_{etag}TeV.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved centrality plots + CSV in centrality/")

    # ── 4. R_pA vs <Ncoll> ───────────────────────────────
    print(f"  [PLOT] R_pA vs Ncoll ...")
    fig = plot_rpa_vs_Ncoll(ctx, Ncoll_cent, Ncoll_MB, npdf_cent_all, energy)
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_Ncoll_{etag}TeV.pdf",
                bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_Ncoll_{etag}TeV.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved Ncoll plots in centrality/")

    print(f"\n  ✓ DONE — p+Pb @ {energy} TeV\n")


# ── Entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    energies = sys.argv[1:] if len(sys.argv) > 1 else ["5.02", "8.16"]
    for e in energies:
        if e not in SQRTS_GEV:
            print(f"[ERROR] Unknown energy '{e}'. Use 5.02 or 8.16.")
            continue
        run_energy(e)
    print("="*60)
    print("  ALL DONE — outputs in outputs/npdf/{min_bias,centrality}")
    print("="*60)
