#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_upsilon_eloss_pPb.py
========================
Standalone p-Pb Upsilon production script integrating:
Energy Loss + pT Broadening (EXCLUDING nPDF).

Matches publication style: outputs categorized into min_bias/ and centrality/.
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
from glauber import OpticalGlauber, SystemSpec
from particle import Particle
from coupling import alpha_s_provider
import quenching_fast as QF

# ── Physics config ──────────────────────────────────────────────────
M_UPSILON_AVG = 10.01
SQRTS_GEV = {"5.02": 5023.0, "8.16": 8160.0}
SIG_NN_MB = {"5.02": 67.0,   "8.16": 71.0}

DPI = 150
ALPHA_BAND = 0.20

# ── Global Plot Settings ─────────────────────────────────────────────
Y_LIM_RPA = (0.65, 1.1)
X_LIM_PT = (0, 20.0)

COLORS = {
    'eloss':      '#F4A0A0',   # pink/salmon
    'broad':      '#56B4E9',   # light blue
    'eloss_broad':'#404040',   # dark gray
}

LABELS = {
    "eloss":      "ELoss",
    "broad":      r"$p_T$-Broadening",
    "eloss_broad":r"ELoss + $p_T$-Broad",
}

CALC_COMPS = ["eloss", "broad", "eloss_broad"]
COMPONENTS_TO_PLOT = ["eloss", "broad", "eloss_broad"]

CENT_BINS = [(0,10),(10,20),(20,40),(40,60),(60,80),(80,100)]
Y_EDGES = np.arange(-5.5, 5.0 + 0.5, 0.5)
P_EDGES = np.arange(0.0, 20.0 + 1.0, 1.0)
PT_RANGE_AVG = (0.0, 15.0)

Y_WINDOWS = [
    (-4.46, -2.96, r"$-4.46 < y < -2.96$"),
    (-1.37,  0.43, r"$-1.37 < y < 0.43$"),
    ( 2.03,  3.53, r"$2.03 < y < 3.53$"),
]

Q0_PAIR = (0.05, 0.09)
P0_SCALE_PAIR = (0.9, 1.1)

def build_eloss_context(energy):
    print(f"\n[INFO] Loading p+Pb @ {energy} TeV (ELoss Only)...", flush=True)
    
    sqrt_sNN = SQRTS_GEV[energy]
    sigma_nn_mb = SIG_NN_MB[energy]
    
    # Glauber geometry
    gl_spec = SystemSpec("pA", sqrt_sNN, A=208, sigma_nn_mb=sigma_nn_mb)
    gl = OpticalGlauber(gl_spec, nx_pa=64, ny_pa=64, verbose=False)
    
    # Quenching params
    alpha_s = alpha_s_provider(mode="running", LambdaQCD=0.25)
    Lmb = gl.leff_minbias_pA()
    device = "cpu"
    qp_base = QF.QuenchParams(
        qhat0=0.075, lp_fm=1.5, LA_fm=Lmb, LB_fm=Lmb,
        system="pA", lambdaQCD=0.25, roots_GeV=sqrt_sNN,
        alpha_of_mu=alpha_s, alpha_scale="mT",
        use_hard_cronin=True, mapping="exp", device=device,
    )
    
    # Build CNM combiner with NPDF disabled
    cnm = CNMCombineFast(
        energy=energy, family="bottomonia", particle_state="avg",
        sqrt_sNN=sqrt_sNN, sigma_nn_mb=sigma_nn_mb,
        cent_bins=CENT_BINS, y_edges=Y_EDGES, p_edges=P_EDGES,
        y_windows=Y_WINDOWS, pt_range_avg=PT_RANGE_AVG, pt_floor_w=1.0,
        weight_mode="flat", y_ref=0.0, cent_c0=0.25,
        q0_pair=Q0_PAIR, p0_scale_pair=P0_SCALE_PAIR, nb_bsamples=5,
        y_shift_fraction=1.0,
        particle=Particle(family="bottomonia", state="avg", mass_override_GeV=9.46),
        npdf_ctx=None,  # DISABLED EXPLICITLY
        gl=gl, qp_base=qp_base, spec=gl_spec
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
    edges = [cb[0][0]] + [b for (_,b) in cb]
    return np.asarray(edges, float), np.concatenate([vals, vals[-1:]])

# ── CSV savers ──────────────────────────────────────────────────────
def save_csv_y(outdir, yc, bands_all, tags, energy, suffix=""):
    # bands_all: {comp: (Rc_dict, Rlo_dict, Rhi_dict)}
    for comp, (Rc_dict, Rlo_dict, Rhi_dict) in bands_all.items():
        for tag in tags:
            Rc, Rlo, Rhi = Rc_dict[tag], Rlo_dict[tag], Rhi_dict[tag]
            df = pd.DataFrame({"y_center": yc, "R_central": Rc, "R_lo": Rlo, "R_hi": Rhi})
            st = tag.replace("%","pct").replace(" ","")
            df.to_csv(outdir / f"Upsilon_RpA_ELoss_{comp}_vs_y_{st}_{energy.replace('.','p')}TeV{suffix}.csv", index=False)

def save_csv_pT(outdir, pc, bands_all, tags, energy, yname=""):
    for comp, (Rc_dict, Rlo_dict, Rhi_dict) in bands_all.items():
        for tag in tags:
            Rc, Rlo, Rhi = Rc_dict[tag], Rlo_dict[tag], Rhi_dict[tag]
            df = pd.DataFrame({"pT_center": pc, "R_central": Rc, "R_lo": Rlo, "R_hi": Rhi})
            st = tag.replace("%","pct").replace(" ","")
            yn = yname.replace(" ","").replace("<","").replace(">","").replace("$","").replace("\\","")
            df.to_csv(outdir / f"Upsilon_RpA_ELoss_{comp}_vs_pT_{st}_{yn}_{energy.replace('.','p')}TeV.csv", index=False)

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
            outdir / f"Upsilon_RpA_ELoss_{comp}_vs_cent_{yn}_{energy.replace('.','p')}TeV.csv", index=False)

# ── Plotting ────────────────────────────────────────────────────────
def plot_rpa_vs_y_grid(cnm, yc, tags, bands, energy):
    n_tags = len(tags)
    n_cols = 4
    n_rows = (n_tags + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5*n_rows), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)
    axes_flat = axes.flatten()

    sys_note = rf"$\mathbf{{p+Pb @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$ (ELoss Only)"

    for ip, tag in enumerate(tags):
        ax = axes_flat[ip]
        plotted_handles = []
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands: continue
            Rc, Rlo, Rhi = bands[comp][0][tag], bands[comp][1][tag], bands[comp][2][tag]
            xe, yc_s = step_from_centers(yc, Rc)
            color = COLORS.get(comp, "black")
            line, = ax.step(xe, yc_s, where="post", lw=1.8, color=color, label=LABELS.get(comp, comp))
            ax.fill_between(xe, step_from_centers(yc, Rlo)[1], step_from_centers(yc, Rhi)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)
            if ip == 0: plotted_handles.append(line)

        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.text(0.96, 0.94, tag, transform=ax.transAxes, ha="right", va="top", weight="bold", fontsize=11)
        if ip == 1: ax.text(0.05, 0.90, sys_note, transform=ax.transAxes, ha="left", va="top", fontsize=10)

        ax.set_xlim(-5, 5); ax.set_ylim(*Y_LIM_RPA)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
        ax.label_outer()

    if n_tags < len(axes_flat):
         axes_flat[n_tags-1].legend(handles=plotted_handles, loc="lower center", frameon=False, fontsize=10)
    for j in range(n_tags, len(axes_flat)): fig.delaxes(axes_flat[j])
    fig.text(0.5, 0.02, r"$y$", ha="center", fontsize=18)
    fig.text(0.04, 0.5, r"$R^{\Upsilon}_{pA}$ (ELoss Only)", va="center", rotation="vertical", fontsize=18)
    return fig

def plot_rpa_vs_pT_grid(cnm, energy, y_windows, p_edges):
    _, tags, _ = cnm.cnm_vs_pT((y_windows[0][0], y_windows[0][1]), p_edges, components=CALC_COMPS, include_mb=True)
    n_rows = len(y_windows); n_cols = len(tags)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.0*n_rows), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)

    sys_note = rf"$\mathbf{{p+Pb @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$ (ELoss Only)"

    for row, (y0,y1,yname) in enumerate(y_windows):
        pc, tags_pt, bands_pt = cnm.cnm_vs_pT((y0,y1), p_edges, components=CALC_COMPS, include_mb=True)
        for col, tag in enumerate(tags_pt):
            ax = axes[row][col]
            for comp in COMPONENTS_TO_PLOT:
                if comp not in bands_pt: continue
                Rc, Rlo, Rhi = bands_pt[comp][0][tag], bands_pt[comp][1][tag], bands_pt[comp][2][tag]
                xe, yc_s = step_from_centers(pc, Rc)
                color = COLORS.get(comp, "black")
                ax.step(xe, yc_s, where="post", lw=1.8, color=color, label=LABELS.get(comp, comp))
                ax.fill_between(xe, step_from_centers(pc, Rlo)[1], step_from_centers(pc, Rhi)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)

            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
            ax.text(0.96, 0.94, tag, transform=ax.transAxes, ha="right", va="top", weight="bold", fontsize=10)
            ax.text(0.04, 0.94, yname, transform=ax.transAxes, ha="left", va="top", color="navy", fontsize=9, fontweight="bold")
            if row == 0 and col == 0: ax.text(0.05, 0.10, sys_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)
            ax.set_xlim(*X_LIM_PT); ax.set_ylim(*Y_LIM_RPA)
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=10)
            ax.label_outer()
            if row == 0 and col == 1: ax.legend(loc="lower left", frameon=False, fontsize=8)

    fig.text(0.5, 0.02, r"$p_T$ (GeV)", ha="center", fontsize=20)
    fig.text(0.04, 0.5, r"$R^{\Upsilon}_{pA}$ (ELoss Only)", va="center", rotation="vertical", fontsize=20)
    return fig

def plot_rpa_vs_centrality(cnm, energy, y_windows, pt_range_avg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=DPI, sharey=True)
    plt.subplots_adjust(wspace=0)

    for i, (y0,y1,yname) in enumerate(y_windows):
        bands_cent = cnm.cnm_vs_centrality((y0,y1), pt_range_avg, components=CALC_COMPS)
        ax = axes[i]
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands_cent: continue
            Rc, Rlo, Rhi, mbc, mblo, mbhi = bands_cent[comp]
            color = COLORS.get(comp, "black")
            xe, yc_s = cent_step_arrays(cnm.cent_bins, Rc)
            ax.step(xe, yc_s, where="post", lw=1.8, color=color, label=LABELS.get(comp, comp))
            ax.fill_between(xe, cent_step_arrays(cnm.cent_bins, Rlo)[1], cent_step_arrays(cnm.cent_bins, Rhi)[1], step="post", color=color, alpha=0.15, lw=0)
            ax.hlines(mbc, 0, 100, colors=color, linestyles=":", linewidth=1.2)

        ax.text(0.92, 0.94, yname, transform=ax.transAxes, ha="right", va="top", fontsize=11, color="navy", fontweight="bold")
        if i == 1: ax.legend(loc="lower left", frameon=False, fontsize=10)
        ax.set_xlim(0, 100); ax.set_ylim(*Y_LIM_RPA)
        ax.set_xlabel("Centrality [%]", fontsize=14)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.axhline(1.0, color="k", ls="-", lw=1.0)
        if i == 0: ax.set_ylabel(r"$R^{\Upsilon}_{pA}$ (ELoss Only)", fontsize=16)

    fig.tight_layout()
    return fig

# ── Execution ────────────────────────────────────────────────────────
def run_energy(energy):
    cnm = build_eloss_context(energy)
    etag = energy.replace('.','p')
    
    OUTDIR_CENT = ROOT / "outputs" / "eloss" / "centrality" / f"pPb_{etag}TeV"
    OUTDIR_MB   = ROOT / "outputs" / "eloss" / "min_bias" / f"pPb_{etag}TeV"
    OUTDIR_CENT.mkdir(exist_ok=True, parents=True)
    OUTDIR_MB.mkdir(exist_ok=True, parents=True)

    print(f"  [PLOT] R_pA vs y (ELoss Only) ...", flush=True)
    yc, tags_y, bands_y = cnm.cnm_vs_y(y_edges=Y_EDGES, pt_range_avg=PT_RANGE_AVG, components=CALC_COMPS, include_mb=True)
    fig_y = plot_rpa_vs_y_grid(cnm, yc, tags_y, bands_y, energy)
    fig_y.savefig(OUTDIR_CENT / f"Upsilon_RpA_ELossOnly_vs_y_{etag}TeV_Grid.pdf", bbox_inches="tight")
    fig_y.savefig(OUTDIR_CENT / f"Upsilon_RpA_ELossOnly_vs_y_{etag}TeV_Grid.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_y)
    save_csv_y(OUTDIR_CENT, yc, bands_y, tags_y, energy)

    # MB y
    fig_mb_y, ax_mb_y = plt.subplots(figsize=(8,6), dpi=DPI)
    for comp in COMPONENTS_TO_PLOT:
        if comp not in bands_y: continue
        Rc = bands_y[comp][0]["MB"]
        xe, yc_s = step_from_centers(yc, Rc)
        ax_mb_y.step(xe, yc_s, where="post", lw=2.4, color=COLORS[comp], label=LABELS[comp])
        ax_mb_y.fill_between(xe, step_from_centers(yc, bands_y[comp][1]["MB"])[1], step_from_centers(yc, bands_y[comp][2]["MB"])[1], step="post", color=COLORS[comp], alpha=0.1, lw=0)
    ax_mb_y.set_xlim(-5, 5); ax_mb_y.set_ylim(*Y_LIM_RPA); ax_mb_y.axhline(1.0, color="k", ls="-", lw=0.8)
    ax_mb_y.set_xlabel(r"$y$", fontsize=14); ax_mb_y.set_ylabel(r"$R^{\Upsilon}_{pA}$ (ELoss Only)", fontsize=14)
    ax_mb_y.legend(loc="lower right", frameon=False, fontsize=11)
    ax_mb_y.text(0.05, 0.95, rf"$\mathbf{{p+Pb @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$ (Min. Bias)", transform=ax_mb_y.transAxes, ha="left", va="top", fontsize=12)
    fig_mb_y.savefig(OUTDIR_MB / f"Upsilon_RpA_ELossOnly_vs_y_MB_{etag}TeV.pdf", bbox_inches="tight")
    fig_mb_y.savefig(OUTDIR_MB / f"Upsilon_RpA_ELossOnly_vs_y_MB_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_mb_y)

    print(f"  [PLOT] R_pA vs pT (ELoss Only) ...", flush=True)
    fig_pt = plot_rpa_vs_pT_grid(cnm, energy, Y_WINDOWS, P_EDGES)
    fig_pt.savefig(OUTDIR_CENT / f"Upsilon_RpA_ELossOnly_vs_pT_Grid_{etag}TeV.pdf", bbox_inches="tight")
    fig_pt.savefig(OUTDIR_CENT / f"Upsilon_RpA_ELossOnly_vs_pT_Grid_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_pt)

    for row, (y0,y1,yname) in enumerate(Y_WINDOWS):
        pc, tags_pt, bands_pt = cnm.cnm_vs_pT((y0,y1), P_EDGES, components=CALC_COMPS, include_mb=True)
        save_csv_pT(OUTDIR_CENT, pc, bands_pt, tags_pt, energy, yname)
    
    # MB pT (middle window)
    y0_mb, y1_mb, yname_mb = Y_WINDOWS[1]
    pc_mb, tags_pt_mb, bands_pt_mb = cnm.cnm_vs_pT((y0_mb, y1_mb), P_EDGES, components=CALC_COMPS, include_mb=True)
    fig_mb_pt, ax_mb_pt = plt.subplots(figsize=(8,6), dpi=DPI)
    for comp in COMPONENTS_TO_PLOT:
        if comp not in bands_pt_mb: continue
        Rc_mb = bands_pt_mb[comp][0]["MB"]
        xe, yc_s = step_from_centers(pc_mb, Rc_mb)
        ax_mb_pt.step(xe, yc_s, where="post", lw=2.4, color=COLORS[comp], label=LABELS[comp])
        ax_mb_pt.fill_between(xe, step_from_centers(pc_mb, bands_pt_mb[comp][1]["MB"])[1], step_from_centers(pc_mb, bands_pt_mb[comp][2]["MB"])[1], step="post", color=COLORS[comp], alpha=0.1, lw=0)
    ax_mb_pt.set_xlim(*X_LIM_PT); ax_mb_pt.set_ylim(*Y_LIM_RPA); ax_mb_pt.axhline(1.0, color="k", ls="-", lw=0.8)
    ax_mb_pt.set_xlabel(r"$p_T$ (GeV)", fontsize=14); ax_mb_pt.set_ylabel(r"$R^{\Upsilon}_{pA}$ (ELoss Only)", fontsize=14)
    ax_mb_pt.legend(loc="lower right", frameon=False, fontsize=11); ax_mb_pt.text(0.05, 0.95, rf"$\mathbf{{p+Pb @ \sqrt{{s_{{NN}}}} = {energy} TeV}}$ (Min. Bias)", transform=ax_mb_pt.transAxes, ha="left", va="top", fontsize=12)
    fig_mb_pt.savefig(OUTDIR_MB / f"Upsilon_RpA_ELossOnly_vs_pT_MB_{etag}TeV.pdf", bbox_inches="tight")
    fig_mb_pt.savefig(OUTDIR_MB / f"Upsilon_RpA_ELossOnly_vs_pT_MB_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_mb_pt)

    print(f"  [PLOT] R_pA vs centrality (ELoss Only) ...", flush=True)
    fig_c = plot_rpa_vs_centrality(cnm, energy, Y_WINDOWS, PT_RANGE_AVG)
    fig_c.savefig(OUTDIR_CENT / f"Upsilon_RpA_ELossOnly_vs_centrality_{etag}TeV.pdf", bbox_inches="tight")
    fig_c.savefig(OUTDIR_CENT / f"Upsilon_RpA_ELossOnly_vs_centrality_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_c)
    for y0, y1, yname in Y_WINDOWS:
        bands_cent = cnm.cnm_vs_centrality((y0,y1), PT_RANGE_AVG, components=CALC_COMPS)
        save_csv_cent(OUTDIR_CENT, cnm, bands_cent, yname, energy)

    print(f"  ✓ DONE (ELoss Only) — p+Pb @ {energy} TeV\n", flush=True)

if __name__ == "__main__":
    energies = sys.argv[1:] if len(sys.argv) > 1 else ["5.02", "8.16"]
    for e in energies: run_energy(e)
