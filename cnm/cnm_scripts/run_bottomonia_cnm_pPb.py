#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_bottomonia_cnm_pPb.py
========================
Upsilon CNM (nPDF + ELoss + pT Broadening) production for pPb @ 5.02 and 8.16 TeV.
Upgraded to match publication style and organization.
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
    "cnm/cnm_combine",
    "cnm/npdf_code",
]

for d in reversed(paths_to_add):
    p = str(ROOT / d)
    if p not in sys.path:
        sys.path.insert(0, p)

from cnm_combine_fast_nuclabs import CNMCombineFast

# ── Physics config ──────────────────────────────────────────────────
M_UPSILON_AVG = 10.01
SQRTS_GEV = {"5.02": 5023.0, "8.16": 8160.0}

DPI = 150
ALPHA_BAND = 0.20

# ── Global Plot Settings ─────────────────────────────────────────────
Y_LIM_RPA = (0.4, 1.2)
X_LIM_PT = (0, 20)

COLORS = {
    'npdf':       '#E69F00',   # orange
    'eloss':      '#F4A0A0',   # pink/salmon
    'broad':      '#56B4E9',   # light blue
    'eloss_broad':'#000000',   # black
    'cnm':        '#606060',   # Gray (Total CNM)
}

LABELS = {
    "npdf":       "nPDF (EPPS21)",
    "eloss":      "ELoss",
    "broad":      r"$p_T$-Broadening",
    "eloss_broad":r"ELoss + $p_T$-Broad",
    "cnm":        "Total CNM",
}

CALC_COMPS = ["npdf", "eloss", "broad", "eloss_broad", "cnm"]
COMPONENTS_TO_PLOT = ["npdf", "eloss", "broad", "eloss_broad", "cnm"]

CENT_BINS = [(0,10),(10,20),(20,40),(40,60),(60,80),(80,100)]
Y_EDGES = np.arange(-5.5, 5.0 + 0.5, 0.5)
P_EDGES = np.arange(0.0, 20.0 + 1.0, 1.0) # More granular
PT_RANGE_AVG = (0.0, 20.0)

Y_WINDOWS = [
    (-4.46, -2.96, r"$-4.46 < y < -2.96$"),
    (-1.37,  0.43, r"$-1.37 < y < 0.43$"),
    ( 2.03,  3.53, r"$2.03 < y < 3.53$"),
]

def build_cnm_context(energy):
    print(f"\n[INFO] Loading p+Pb @ {energy} TeV ...", flush=True)
    cnm = CNMCombineFast.from_defaults(
        energy=energy, family="bottomonia", particle_state="avg",
        cent_bins=CENT_BINS, y_edges=Y_EDGES, p_edges=P_EDGES,
        y_windows=Y_WINDOWS, pt_range_avg=PT_RANGE_AVG,
        alpha_s_mode='constant', alpha0=0.5 # Consistent with standard
    )
    return cnm

# ── Helpers ─────────────────────────────────────────────────────────
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

# ── CSV savers ──────────────────────────────────────────────────────
def save_csv_y(outdir, yc, bands_all, tags, energy, suffix=""):
    # bands_all: {comp: (Rc_dict, Rlo_dict, Rhi_dict)}
    for comp, (Rc_dict, Rlo_dict, Rhi_dict) in bands_all.items():
        for tag in tags:
            Rc, Rlo, Rhi = Rc_dict[tag], Rlo_dict[tag], Rhi_dict[tag]
            df = pd.DataFrame({"y_center": yc, "R_central": Rc, "R_lo": Rlo, "R_hi": Rhi})
            st = tag.replace("%","pct").replace(" ","")
            df.to_csv(outdir / f"Upsilon_RpA_{comp}_vs_y_{st}_{energy.replace('.','p')}TeV{suffix}.csv", index=False)

def save_csv_pT(outdir, pc, bands_all, tags, energy, yname=""):
    for comp, (Rc_dict, Rlo_dict, Rhi_dict) in bands_all.items():
        for tag in tags:
            Rc, Rlo, Rhi = Rc_dict[tag], Rlo_dict[tag], Rhi_dict[tag]
            df = pd.DataFrame({"pT_center": pc, "R_central": Rc, "R_lo": Rlo, "R_hi": Rhi})
            st = tag.replace("%","pct").replace(" ","")
            yn = yname.replace(" ","").replace("<","").replace(">","").replace("$","").replace("\\","")
            df.to_csv(outdir / f"Upsilon_RpA_{comp}_vs_pT_{st}_{yn}_{energy.replace('.','p')}TeV.csv", index=False)

def save_csv_cent(outdir, cnm, bands_cent, yname, energy):
    # bands_cent: {comp: (Rc, Rlo, Rhi, mbc, mblo, mbhi)}
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in cnm.cent_bins]
    
    # Ncoll is same for all components
    gl = cnm.gl
    Ncoll_cent = [gl.ncoll_mean_bin_pA_optical(a/100.0, b/100.0) for (a,b) in cnm.cent_bins]
    Ncoll_MB = gl.ncoll_mean_bin_pA_optical(0.0, 1.0)

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
            outdir / f"Upsilon_RpA_{comp}_vs_cent_{yn}_{energy.replace('.','p')}TeV.csv", index=False)

# ── Plotting ────────────────────────────────────────────────────────
def plot_rpa_vs_y_grid(cnm, yc, tags, bands, energy):
    n_tags = len(tags)
    n_cols = 4
    n_rows = (n_tags + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5*n_rows), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)
    axes_flat = axes.flatten()

    sys_note = rf"$\mathbf{{p+Pb @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$"

    for ip, tag in enumerate(tags):
        ax = axes_flat[ip]
        plotted_handles = []
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands: continue
            Rc, Rlo, Rhi = bands[comp][0][tag], bands[comp][1][tag], bands[comp][2][tag]
            xe, yc_s = step_from_centers(yc, Rc)
            color = COLORS.get(comp, "black")
            ls = "--" if comp in ("npdf", "eloss_broad") else "-"
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

        ax.set_xlim(-5, 5); ax.set_ylim(*Y_LIM_RPA)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
        ax.label_outer()

    if n_tags < len(axes_flat):
         axes_flat[n_tags-1].legend(handles=plotted_handles, loc="lower center", frameon=False, fontsize=10)
    
    for j in range(n_tags, len(axes_flat)): fig.delaxes(axes_flat[j])
    fig.text(0.5, 0.02, r"$y$", ha="center", fontsize=18)
    fig.text(0.04, 0.5, r"$R^{\Upsilon}_{pA}$", va="center", rotation="vertical", fontsize=18)
    return fig

def plot_rpa_vs_pT_grid(cnm, energy, y_windows, p_edges):
    # Get bands for first window to get tags
    _, tags, _ = cnm.cnm_vs_pT((y_windows[0][0], y_windows[0][1]), p_edges, include_mb=True)
    n_rows = len(y_windows); n_cols = len(tags)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.0*n_rows), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)

    sys_note = rf"$\mathbf{{p+Pb @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$"

    for row, (y0,y1,yname) in enumerate(y_windows):
        pc, tags_pt, bands_pt = cnm.cnm_vs_pT((y0,y1), p_edges, include_mb=True)
        for col, tag in enumerate(tags_pt):
            ax = axes[row][col]
            for comp in COMPONENTS_TO_PLOT:
                if comp not in bands_pt: continue
                Rc, Rlo, Rhi = bands_pt[comp][0][tag], bands_pt[comp][1][tag], bands_pt[comp][2][tag]
                xe, yc_s = step_from_centers(pc, Rc)
                color = COLORS.get(comp, "black")
                ls = "--" if comp in ("npdf", "eloss_broad") else "-"
                lw = 2.2 if comp == "cnm" else 1.5
                ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
                ax.fill_between(xe, step_from_centers(pc, Rlo)[1], step_from_centers(pc, Rhi)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)

            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
            ax.text(0.96, 0.96, tag, transform=ax.transAxes, ha="right", va="top", weight="bold", fontsize=10)
            ax.text(0.04, 0.96, yname, transform=ax.transAxes, ha="left", va="top", color="navy", fontsize=9, fontweight="bold")
            
            if row == 0 and col == 0:
                ax.text(0.05, 0.10, sys_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)
                
            ax.set_xlim(*X_LIM_PT); ax.set_ylim(*Y_LIM_RPA)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune="both"))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=9, prune="both"))
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=10)
            ax.label_outer()
            if row == 0 and col == 1:
                ax.legend(loc="lower left", frameon=False, fontsize=8)

    fig.text(0.5, 0.02, r"$p_T$ (GeV)", ha="center", fontsize=20)
    fig.text(0.04, 0.5, r"$R^{\Upsilon}_{pA}$", va="center", rotation="vertical", fontsize=20)
    return fig

def plot_rpa_vs_centrality(cnm, energy, y_windows, pt_range_avg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=DPI, sharey=True)
    plt.subplots_adjust(wspace=0)

    sys_note = rf"$\mathbf{{p+Pb @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$"

    for i, (y0,y1,yname) in enumerate(y_windows):
        bands_cent = cnm.cnm_vs_centrality((y0,y1), pt_range_avg)
        ax = axes[i]
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands_cent: continue
            Rc, Rlo, Rhi, mbc, mblo, mbhi = bands_cent[comp]
            color = COLORS.get(comp, "black")
            ls = "-" if comp == "cnm" else "--" if comp in ("npdf", "eloss_broad") else "-"
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

        ax.set_xlim(0, 100); ax.set_ylim(*Y_LIM_RPA)
        ax.set_xlabel("Centrality [%]", fontsize=14)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.axhline(1.0, color="k", ls="-", lw=1.0)
        if i == 0: ax.set_ylabel(r"$R^{\Upsilon}_{pA}$", fontsize=16)

    fig.tight_layout()
    return fig

# ── Main ────────────────────────────────────────────────────────────
def run_energy(energy):
    cnm = build_cnm_context(energy)
    etag = energy.replace('.','p')
    
    OUTDIR_CENT = ROOT / "outputs" / "cnm" / "centrality" / f"pPb_{etag}TeV"
    OUTDIR_MB   = ROOT / "outputs" / "cnm" / "min_bias" / f"pPb_{etag}TeV"
    OUTDIR_CENT.mkdir(exist_ok=True, parents=True)
    OUTDIR_MB.mkdir(exist_ok=True, parents=True)

    # 1. R_pA vs y (grid)
    print(f"  [PLOT] R_pA vs y ...", flush=True)
    yc, tags_y, bands_y = cnm.cnm_vs_y(y_edges=Y_EDGES, pt_range_avg=PT_RANGE_AVG, components=CALC_COMPS, include_mb=True)
    fig = plot_rpa_vs_y_grid(cnm, yc, tags_y, bands_y, energy)
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_CNM_vs_y_{etag}TeV_Grid.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_CNM_vs_y_{etag}TeV_Grid.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    save_csv_y(OUTDIR_CENT, yc, bands_y, tags_y, energy)

    # MB-only y
    fig_mb, ax_mb = plt.subplots(figsize=(8,6), dpi=DPI)
    for comp in COMPONENTS_TO_PLOT:
        if comp not in bands_y: continue
        Rc_mb = bands_y[comp][0]["MB"]
        xe, yc_s = step_from_centers(yc, Rc_mb)
        ls = "--" if comp in ("npdf", "eloss_broad") else "-"
        lw = 2.4 if comp == "cnm" else 1.8
        ax_mb.step(xe, yc_s, where="post", lw=lw, ls=ls, color=COLORS.get(comp, "black"), label=LABELS.get(comp, comp))
        ax_mb.fill_between(xe, step_from_centers(yc, bands_y[comp][1]["MB"])[1], step_from_centers(yc, bands_y[comp][2]["MB"])[1], step="post", color=COLORS.get(comp, "black"), alpha=0.1, lw=0)

    ax_mb.axhline(1.0, color="k", ls="-", lw=0.8)
    ax_mb.set_xlim(-5, 5); ax_mb.set_ylim(*Y_LIM_RPA)
    ax_mb.set_xlabel(r"$y$", fontsize=14); ax_mb.set_ylabel(r"$R^{\Upsilon}_{pA}$", fontsize=14)
    ax_mb.legend(loc="lower right", frameon=False, fontsize=11)
    ax_mb.text(0.05, 0.95, rf"$\mathbf{{p+Pb @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$ (Min. Bias)", transform=ax_mb.transAxes, ha="left", va="top", fontsize=12)
    fig_mb.savefig(OUTDIR_MB / f"Upsilon_RpA_CNM_vs_y_MB_{etag}TeV.pdf", bbox_inches="tight")
    fig_mb.savefig(OUTDIR_MB / f"Upsilon_RpA_CNM_vs_y_MB_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_mb)

    # 2. R_pA vs pT
    print(f"  [PLOT] R_pA vs pT ...", flush=True)
    fig = plot_rpa_vs_pT_grid(cnm, energy, Y_WINDOWS, P_EDGES)
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_CNM_vs_pT_Grid_{etag}TeV.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_CNM_vs_pT_Grid_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    for y0, y1, yname in Y_WINDOWS:
        pc, tags_pT, bands_pT = cnm.cnm_vs_pT((y0, y1), P_EDGES, components=CALC_COMPS, include_mb=True)
        save_csv_pT(OUTDIR_CENT, pc, bands_pT, tags_pT, energy, yname)

    # MB-only pT (for middle window)
    y0_mb, y1_mb, yname_mb = Y_WINDOWS[1]
    pc_mb, tags_pt_mb, bands_pt_mb = cnm.cnm_vs_pT((y0_mb, y1_mb), P_EDGES, components=CALC_COMPS, include_mb=True)
    fig_mb_pt, ax_mb_pt = plt.subplots(figsize=(8,6), dpi=DPI)
    for comp in COMPONENTS_TO_PLOT:
        if comp not in bands_pt_mb: continue
        Rc_mb = bands_pt_mb[comp][0]["MB"]
        xe, yc_s = step_from_centers(pc_mb, Rc_mb)
        ls = "--" if comp in ("npdf", "eloss_broad") else "-"
        lw = 2.4 if comp == "cnm" else 1.8
        ax_mb_pt.step(xe, yc_s, where="post", lw=lw, ls=ls, color=COLORS.get(comp, "black"), label=LABELS.get(comp, comp))
        ax_mb_pt.fill_between(xe, step_from_centers(pc_mb, bands_pt_mb[comp][1]["MB"])[1], step_from_centers(pc_mb, bands_pt_mb[comp][2]["MB"])[1], step="post", color=COLORS.get(comp, "black"), alpha=0.1, lw=0)
    
    ax_mb_pt.axhline(1.0, color="k", ls="-", lw=0.8)
    ax_mb_pt.set_xlim(*X_LIM_PT); ax_mb_pt.set_ylim(*Y_LIM_RPA)
    ax_mb_pt.set_xlabel(r"$p_T$ (GeV)", fontsize=14); ax_mb_pt.set_ylabel(r"$R^{\Upsilon}_{pA}$", fontsize=14)
    ax_mb_pt.legend(loc="lower right", frameon=False, fontsize=11)
    ax_mb_pt.text(0.05, 0.95, rf"$\mathbf{{p+Pb @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$ (Min. Bias)", transform=ax_mb_pt.transAxes, ha="left", va="top", fontsize=12)
    ax_mb_pt.text(0.05, 0.88, yname_mb, transform=ax_mb_pt.transAxes, ha="left", va="top", fontsize=11, color="navy", fontweight="bold")
    fig_mb_pt.savefig(OUTDIR_MB / f"Upsilon_RpA_CNM_vs_pT_MB_{etag}TeV.pdf", bbox_inches="tight")
    fig_mb_pt.savefig(OUTDIR_MB / f"Upsilon_RpA_CNM_vs_pT_MB_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_mb_pt)

    # 3. R_pA vs centrality
    print(f"  [PLOT] R_pA vs centrality ...", flush=True)
    fig = plot_rpa_vs_centrality(cnm, energy, Y_WINDOWS, PT_RANGE_AVG)
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_CNM_vs_centrality_{etag}TeV.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_CNM_vs_centrality_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    for y0, y1, yname in Y_WINDOWS:
        bands_cent = cnm.cnm_vs_centrality((y0,y1), PT_RANGE_AVG, components=CALC_COMPS)
        save_csv_cent(OUTDIR_CENT, cnm, bands_cent, yname, energy)

    print(f"  ✓ DONE — p+Pb @ {energy} TeV\n", flush=True)

if __name__ == "__main__":
    energies = sys.argv[1:] if len(sys.argv) > 1 else ["5.02", "8.16"]
    for e in energies: run_energy(e)
