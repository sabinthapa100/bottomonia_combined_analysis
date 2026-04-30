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
OUTDIR_PUB = ROOT / "outputs" / "cnm" / "publication_OO_5p36TeV"

DPI = 150
ALPHA_BAND = 0.22

# PLOTTING LIMITS (STRICTLY REQUESTED) - set to None to compute from data
Y_LIM_RPA = (0.2, 1.2)
PT_MIN = 0.5
# EPPS21 grid in nPDF_OO.dat tabulates up to pT=20 GeV. The unphysical blowup
# is confined to |y|~4.75; since Y_WINDOWS is capped at |y|<=4.0, pT=20 GeV
# is well-behaved (Rg1*Rg2 stays ~0.8 at y=+/-3.75 for pT up to 20 GeV).
PT_MAX_REQUESTED = 20.0
PT_BIN_WIDTH = 1.0
Y_BIN_WIDTH = 0.5
X_LIM_PT = (0, PT_MAX_REQUESTED)
X_LIM_Y = (-4.0, 4.0)  # Truncated to avoid nPDF edge artefacts at |y|>4

CALC_COMPS = ("npdf", "eloss", "broad", "eloss_broad", "cnm")
COMPONENTS_TO_PLOT = ["npdf", "eloss_broad", "cnm"]  # Show only these components

# Color convention
COLORS = {
    "npdf":       "#E69F00",   # orange
    "eloss":      "#F4A0A0",   # pink/salmon
    "broad":      "#56B4E9",   # light blue
    "eloss_broad":"#000000",   # black
    "cnm":        "#606060",   # Gray (Total CNM)
}

LABELS = {
    "npdf":       "nPDF (EPPS21)",
    "eloss_broad":r"ELoss + $p_T$-Broad",
    "cnm":        "CNM",
}

CENT_BINS = [(0,10),(10,30),(30,50),(50,70),(70,100)]
MB_C0 = 0.25

# Y-grid capped at |y| <= 4.0: the |y|=4.75 ring of nPDF_OO.dat contains
# EPPS21 extrapolation artefacts (R_g blows up at large pT).
Y_EDGES = np.arange(-4.0, 4.0 + Y_BIN_WIDTH, Y_BIN_WIDTH)
P_EDGES = np.arange(PT_MIN, PT_MAX_REQUESTED + PT_BIN_WIDTH, PT_BIN_WIDTH)
PT_RANGE_AVG = (PT_MIN, PT_MAX_REQUESTED)

# Rapidity windows for pT plots and centrality plots. Narrowed to |y| <= 4.0
# so that integration never touches the diseased y=+/-4.75 grid edge.
Y_WINDOWS = [
    (-4.0, -2.5, r"$-4.0 < y < -2.5$"),
    (-2.4,  2.4, r"$-2.4 < y < 2.4$"),
    ( 2.5,  4.0, r"$2.5 < y < 4.0$"),
]

# Optional extended windows used for dedicated pT comparison outputs.
EXTENDED_PT_WINDOWS = [
    (-4.5, -2.5, r"$-4.5 < y < -2.5$"),
    ( 2.5,  4.5, r"$2.5 < y < 4.5$"),
]

NOTE_BOX = dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5)

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
    data_pt_max = float(data[1]["pt"].max())
    pt_max = min(PT_MAX_REQUESTED, data_pt_max)
    if pt_max < PT_MAX_REQUESTED:
        print(
            f"  [WARN] nPDF_OO.dat only reaches pT={data_pt_max:g} GeV; "
            f"using pT<= {pt_max:g} GeV for integrated and pT-binned CNM.",
            flush=True,
        )
    pt_range_avg = (PT_MIN, pt_max)
    p_edges = np.arange(PT_MIN, pt_max + PT_BIN_WIDTH, PT_BIN_WIDTH)
    grid = build_OO_rpa_grid(data, pt_max=pt_max)

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
        cent_bins=CENT_BINS, y_edges=Y_EDGES, p_edges=p_edges,
        y_windows=Y_WINDOWS, pt_range_avg=pt_range_avg, pt_floor_w=1.0,
        weight_mode="flat", y_ref=0.0, cent_c0=MB_C0,
        q0_pair=Q0_PAIR, p0_scale_pair=P0_SCALE_PAIR, nb_bsamples=5,
        y_shift_fraction=1.0, particle=particle,
        npdf_ctx=npdf_ctx, gl=gl, qp_base=qp_base, spec=gl.spec
    )
    cnm.publication_p_edges = p_edges
    cnm.publication_pt_range_avg = pt_range_avg
    cnm.publication_pt_support_max = pt_max
    return cnm

# ── Helpers ──────────────────────────────────────────────────────────
def fmt_axis_number(x):
    x = float(x)
    return f"{int(x)}" if abs(x - round(x)) < 1e-9 else f"{x:g}"

def pt_range_label(pt_range):
    # Display the integration range as [0, pt_max] for publication clarity
    # (data grid starts at 0.5 GeV; first bin is [0.5, 1.5] -> center 1 GeV).
    lo = 0.0 if pt_range[0] <= 0.5 + 1e-9 else pt_range[0]
    return rf"$p_T \in [{fmt_axis_number(lo)},\,{fmt_axis_number(pt_range[1])}]$ GeV"

def y_window_file_tag(y0, y1):
    def one(v):
        prefix = "n" if v < 0 else "p"
        return prefix + f"{abs(v):.1f}".replace(".", "p")
    return f"y_{one(y0)}_{one(y1)}"

def y_window_csv_tag(y0, y1):
    return f"y_{fmt_axis_number(y0)}_to_{fmt_axis_number(y1)}".replace("-", "m").replace(".", "p")

def step_from_centers(xc, vals, x_start=None):
    xc = np.asarray(xc, float); vals = np.asarray(vals, float)
    dx = np.diff(xc)
    dx0 = dx[0] if dx.size else 1.0
    xe = np.concatenate(([xc[0]-0.5*dx0], xc+0.5*dx0))
    if x_start is not None:
        xe[0] = x_start
    ys = np.concatenate([vals, vals[-1:]])
    return xe, ys

def cent_step_arrays(cb, vals):
    vals = np.asarray(vals, float)
    edges = [cb[0][0]] + [b for (_, b) in cb]
    return np.asarray(edges, float), np.concatenate([vals, vals[-1:]])

def edges_from_centers(xc):
    xc = np.asarray(xc, float)
    if xc.size < 2:
        return np.array([xc[0]-0.5, xc[0]+0.5])
    mids = 0.5*(xc[:-1]+xc[1:])
    return np.concatenate([[xc[0]-(mids[0]-xc[0])], mids, [xc[-1]+(xc[-1]-mids[-1])]])

def pin_edge_bins(bands_y, yc):
    """Pin outermost bins to neighbors to avoid nPDF edge artefacts."""
    if yc.size < 3: return
    for comp in bands_y:
        # bands_y[comp] is (Rc_dict, Rlo_dict, Rhi_dict)
        for i in range(3):
            d = bands_y[comp][i]
            for tag in d:
                # Modifies in-place
                vals = np.asarray(d[tag], float)
                vals[0] = vals[1]
                vals[-1] = vals[-2]
                d[tag] = vals

# ── CSV savers (Consolidated) ────────────────────────────────────
def save_csv_y_consolidated(outdir, yc, bands_all, tags, energy):
    """Save one consolidated CSV per centrality bin + MB with all components."""
    # bands_all: {comp: (Rc_dict, Rlo_dict, Rhi_dict)}
    for tag in tags:
        rows = []
        for y_val in yc:
            row = {"y_center": float(y_val)}
            for comp in ["npdf", "eloss_broad", "cnm"]:
                if comp in bands_all:
                    Rc_dict, Rlo_dict, Rhi_dict = bands_all[comp]
                    row[f"{comp}_central"] = float(Rc_dict[tag][np.argmin(np.abs(yc - y_val))])
                    row[f"{comp}_lo"] = float(Rlo_dict[tag][np.argmin(np.abs(yc - y_val))])
                    row[f"{comp}_hi"] = float(Rhi_dict[tag][np.argmin(np.abs(yc - y_val))])
            rows.append(row)
        
        st = tag.replace("%","pct").replace(" ","")
        df = pd.DataFrame(rows)
        df.to_csv(outdir / f"Upsilon_RAA_vs_y_{st}_OO_{energy.replace('.','p')}TeV.csv", index=False)

def save_csv_pT_consolidated(outdir, pc, bands_all, tags, energy, yname=""):
    """Save one consolidated CSV per rapidity window with all components and centrality bins."""
    # bands_all: {comp: (Rc_dict, Rlo_dict, Rhi_dict)}
    rows = []
    for tag in tags:
        for p_val in pc:
            p_idx = np.argmin(np.abs(pc - p_val))
            row = {"pT_center": float(p_val), "centrality": str(tag)}
            for comp in ["npdf", "eloss_broad", "cnm"]:
                if comp in bands_all:
                    Rc_dict, Rlo_dict, Rhi_dict = bands_all[comp]
                    row[f"{comp}_central"] = float(Rc_dict[tag][p_idx])
                    row[f"{comp}_lo"] = float(Rlo_dict[tag][p_idx])
                    row[f"{comp}_hi"] = float(Rhi_dict[tag][p_idx])
            rows.append(row)
    
    yn = yname.replace(" ","").replace("<","").replace(">","").replace("$","").replace("\\","")
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"Upsilon_RAA_vs_pT_{yn}_OO_{energy.replace('.','p')}TeV.csv", index=False)

def save_csv_cent_consolidated(outdir, cnm, bands_cent, yname, energy):
    """Save one consolidated CSV per rapidity window with all components and centrality bins."""
    # bands_cent: {comp: (Rc, Rlo, Rhi, mbc, mblo, mbhi)}
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in cnm.cent_bins]

    # Ncoll is same for all components
    gl = cnm.gl
    Ncoll_cent = [gl.ncoll_mean_bin_AA_optical(a/100.0, b/100.0) for (a,b) in cnm.cent_bins]
    Ncoll_MB = gl.ncoll_mean_bin_AA_optical(0.0, 1.0)

    rows = []
    for i, ((cL,cR), lab) in enumerate(zip(cnm.cent_bins, labels)):
        row = dict(cent_left=float(cL), cent_right=float(cR), label=lab, Ncoll=float(Ncoll_cent[i]))
        for comp in ["npdf", "eloss_broad", "cnm"]:
            if comp in bands_cent:
                Rc, Rlo, Rhi, mbc, mblo, mbhi = bands_cent[comp]
                row[f"{comp}_central"] = float(Rc[i])
                row[f"{comp}_lo"] = float(Rlo[i])
                row[f"{comp}_hi"] = float(Rhi[i])
        rows.append(row)
    
    # Add MB row
    mb_row = dict(cent_left=0, cent_right=100, label="MB", Ncoll=float(Ncoll_MB))
    for comp in ["npdf", "eloss_broad", "cnm"]:
        if comp in bands_cent:
            Rc, Rlo, Rhi, mbc, mblo, mbhi = bands_cent[comp]
            mb_row[f"{comp}_central"] = float(mbc)
            mb_row[f"{comp}_lo"] = float(mblo)
            mb_row[f"{comp}_hi"] = float(mbhi)
    rows.append(mb_row)

    yn = yname.replace(" ","").replace("<","").replace(">","").replace("$","").replace("\\","")
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"Upsilon_RAA_vs_cent_{yn}_OO_{energy.replace('.','p')}TeV.csv", index=False)

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
    mn = allvals.min()
    mx = allvals.max()
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

    sys_note = rf"$\mathbf{{O+O, \sqrt{{s_{{\rm NN}}}} = {energy} TeV}}$"

    for ip, tag in enumerate(tags):
        ax = axes_flat[ip]
        plotted_handles = []
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands:
                continue
            Rc, Rlo, Rhi = bands[comp][0][tag], bands[comp][1][tag], bands[comp][2][tag]
            xe, yc_s = step_from_centers(yc, Rc)
            color = COLORS.get(comp, "black")
            ls = "--" if comp in ("npdf", "eloss_broad") else "-"
            lw = 2.2 if comp == "cnm" else 1.5
            line, = ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
            ax.fill_between(xe, step_from_centers(yc, Rlo)[1], step_from_centers(yc, Rhi)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)
            if ip == 0:
                plotted_handles.append(line)

        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.text(0.96, 0.96, tag, transform=ax.transAxes, ha="right", va="top", weight="bold", fontsize=11)

        if ip == 0:
            ax.text(0.95, 0.90, rf"$p_T \in [{PT_RANGE_AVG[0]:.0f},\,{PT_RANGE_AVG[1]:.0f}]$ GeV",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)
        if ip == 1:
            ax.text(0.05, 0.90, sys_note, transform=ax.transAxes, ha="left", va="top", fontsize=10)

        ax.set_xlim(*X_LIM_Y)
        if Y_LIM_RPA is None:
            lims = compute_ylim_from_bands(bands)
            ax.set_ylim(*lims)
        else:
            ax.set_ylim(*Y_LIM_RPA)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
        ax.label_outer()

    if n_tags < len(axes_flat):
        axes_flat[n_tags-1].legend(handles=plotted_handles, loc="lower center", frameon=False, fontsize=10)

    for j in range(n_tags, len(axes_flat)):
        fig.delaxes(axes_flat[j])
    fig.text(0.5, 0.02, r"$y$", ha="center", fontsize=18)
    fig.text(0.04, 0.5, r"$R^{\Upsilon}_{AA}$", va="center", rotation="vertical", fontsize=18)
    return fig

def plot_rpa_vs_pT_grid(cnm, energy, y_windows, p_edges):
    pc, tags, bands = cnm.cnm_vs_pT((y_windows[0][0], y_windows[0][1]), p_edges, components=CALC_COMPS, include_mb=True)
    n_rows = len(y_windows)
    n_cols = len(tags)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.0*n_rows), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)

    if n_rows == 1:
        axes = [axes]
    sys_note = rf"$\mathbf{{O+O, \sqrt{{s_{{\rm NN}}}} = {energy} TeV}}$"

    for row, (y0, y1, yname) in enumerate(y_windows):
        pc, tags_pt, bands_pt = cnm.cnm_vs_pT((y0, y1), p_edges, components=CALC_COMPS, include_mb=True)
        for col, tag in enumerate(tags_pt):
            ax = axes[row][col]
            for comp in COMPONENTS_TO_PLOT:
                if comp not in bands_pt:
                    continue
                Rc, Rlo, Rhi = bands_pt[comp][0][tag], bands_pt[comp][1][tag], bands_pt[comp][2][tag]
                xe, yc_s = step_from_centers(pc, Rc, x_start=0.0)
                color = COLORS.get(comp, "black")
                ls = "--" if comp in ("npdf", "eloss_broad") else "-"
                lw = 2.2 if comp == "cnm" else 1.5
                ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
                ax.fill_between(xe, step_from_centers(pc, Rlo, x_start=0.0)[1], step_from_centers(pc, Rhi, x_start=0.0)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)

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
            ax.minorticks_on()
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

    sys_note = rf"$\mathbf{{O+O, \sqrt{{s_{{\rm NN}}}} = {energy} TeV}}$"

    for i, (y0, y1, yname) in enumerate(y_windows):
        bands_cent = cnm.cnm_vs_centrality((y0, y1), pt_range_avg, components=CALC_COMPS)
        ax = axes[i]
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands_cent:
                continue
            Rc, Rlo, Rhi, mbc, mblo, mbhi = bands_cent[comp]
            color = COLORS.get(comp, "black")
            ls = "-" if comp == "cnm" else "--" if comp in ("npdf", "eloss_broad") else "-"
            lw = 2.0 if comp == "cnm" else 1.5

            xe, yc_s = cent_step_arrays(cnm.cent_bins, Rc)
            ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
            ax.fill_between(xe, cent_step_arrays(cnm.cent_bins, Rlo)[1], cent_step_arrays(cnm.cent_bins, Rhi)[1], step="post", color=color, alpha=0.15, lw=0)

            # MB band
            ax.fill_between([0, 100], [mblo]*2, [mbhi]*2, color=color, alpha=0.08, hatch="//", lw=0)
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
        ax.yaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.axhline(1.0, color="k", ls="-", lw=1.0)
        if i == 0:
            ax.set_ylabel(r"$R^{\Upsilon}_{AA}$", fontsize=16)

    fig.tight_layout()
    return fig

def plot_mb_summary(cnm, yc, bands_y_mb, bands_pt_mb_list, energy):
    """
    Consolidated MB Summary: y (1 panel) + pT (3 panels) in 2x2 grid.
    bands_pt_mb_list: list of (yname, pc, bands_pt)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=DPI)
    axes_flat = axes.flatten()

    # System label: O+O, \sqrt{s_{\rm NN}} = 5.36 TeV (MB)
    sys_note = rf"$\mathbf{{O+O, \sqrt{{s_{{\rm NN}}}} = {energy} \, TeV}}$ (MB)"

    # [0] R_AA vs y
    ax_y = axes_flat[0]
    for comp in COMPONENTS_TO_PLOT:
        if comp not in bands_y_mb:
            continue
        Rc = bands_y_mb[comp][0]["MB"]
        xe, yc_s = step_from_centers(yc, Rc)
        ls = "--" if comp in ("npdf", "eloss_broad") else "-"
        lw = 2.4 if comp == "cnm" else 1.8
        color = COLORS.get(comp, "black")
        ax_y.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
        ax_y.fill_between(
            xe,
            step_from_centers(yc, bands_y_mb[comp][1]["MB"])[1],
            step_from_centers(yc, bands_y_mb[comp][2]["MB"])[1],
            step="post", color=color, alpha=0.1, lw=0
        )

    ax_y.set_xlim(*X_LIM_Y)
    ax_y.set_ylim(*Y_LIM_RPA)
    ax_y.axhline(1.0, color="k", ls="-", lw=0.8)
    ax_y.set_xlabel(r"$y$", fontsize=16)
    ax_y.set_ylabel(r"$R_{AA}$", fontsize=16)
    ax_y.minorticks_on()
    ax_y.tick_params(which="both", direction="in", top=True, right=True, labelsize=12)
    ax_y.text(0.05, 0.95, sys_note, transform=ax_y.transAxes, ha="left", va="top", fontsize=14)
    ax_y.text(
        0.95, 0.95,
        rf"$p_T \in [{PT_RANGE_AVG[0]:.0f},\,{PT_RANGE_AVG[1]:.0f}]$ GeV",
        transform=ax_y.transAxes, ha="right", va="top", fontsize=12
    )

    # [1, 2, 3] R_AA vs pT for 3 windows
    for i, (yname, pc, bands_pt) in enumerate(bands_pt_mb_list):
        ax = axes_flat[1+i]
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands_pt:
                continue
            Rc = bands_pt[comp][0]["MB"]
            xe, yc_s = step_from_centers(pc, Rc, x_start=0.0)
            ls = "--" if comp in ("npdf", "eloss_broad") else "-"
            lw = 2.4 if comp == "cnm" else 1.8
            color = COLORS.get(comp, "black")
            ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
            ax.fill_between(
                xe,
                step_from_centers(pc, bands_pt[comp][1]["MB"], x_start=0.0)[1],
                step_from_centers(pc, bands_pt[comp][2]["MB"], x_start=0.0)[1],
                step="post", color=color, alpha=0.1, lw=0
            )

        ax.axhline(1.0, color="k", ls="-", lw=0.8)
        ax.set_xlim(*X_LIM_PT)
        ax.set_ylim(*Y_LIM_RPA)
        ax.set_xlabel(r"$p_T$ (GeV)", fontsize=16)
        ax.set_ylabel(r"$R_{AA}$", fontsize=16)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=12)
        ax.text(
            0.04, 0.94, yname,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=15, color="#202020", fontweight="bold"
        )

        # Legend only in second panel (top-right panel overall)
        if i == 0:
            ax.legend(loc="lower left", frameon=False, fontsize=11)

    plt.tight_layout()
    return fig

# ── Publication Plotting Functions ────────────────────────────────────

def plot_centrality_with_mb_overlay(cnm, energy, y_windows, pt_range_avg):
    """Publication-ready R_AA vs centrality with MB overlay bands."""
    fig, axes = plt.subplots(1, len(y_windows), figsize=(5 * len(y_windows), 5), sharey=True)
    if len(y_windows) == 1:
        axes = [axes]
    # No title

    sys_note = rf"$\mathbf{{O+O, \sqrt{{s_{{\rm NN}}}} = {energy} \, TeV}}$"

    for i, (y0, y1, yname) in enumerate(y_windows):
        ax = axes[i]
        # No subplot title

        # Get CNM bands for this rapidity window
        bands = cnm.cnm_vs_centrality((y0, y1), pt_range_avg, components=CALC_COMPS)
        
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands:
                continue
            Rc, Rlo, Rhi, mbc, mblo, mbhi = bands[comp]
            color = COLORS.get(comp, "black")
            ls = "--" if comp in ("npdf", "eloss_broad") else "-"
            lw = 2.2 if comp == "cnm" else 1.5

            # Step plot for centrality
            xe, yc_s = cent_step_arrays(cnm.cent_bins, Rc)
            ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
            ax.fill_between(xe, cent_step_arrays(cnm.cent_bins, Rlo)[1], cent_step_arrays(cnm.cent_bins, Rhi)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)

            # MB band
            ax.fill_between([0, 100], [mblo]*2, [mbhi]*2, color=color, alpha=0.08, hatch="//", lw=0)
            ax.hlines(mbc, 0, 100, colors=color, linestyles=":", linewidth=1.5)

        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        # Put text inside plot
        ax.text(0.05, 0.95, yname, transform=ax.transAxes, ha="left", va="top", fontsize=11, color="navy", fontweight="bold", bbox=NOTE_BOX)
        if i == 0:
            ax.text(0.05, 0.10, sys_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, bbox=NOTE_BOX)
        ax.text(0.95, 0.05, pt_range_label(pt_range_avg), transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox=NOTE_BOX)
        if i == 1:
            ax.legend(loc="lower left", frameon=False, fontsize=10)

        ax.set_xlim(0, 100)
        ax.set_ylim(*Y_LIM_RPA)
        # Explicit ticks at 0, 20, 40, 60, 80, 100%
        ax.xaxis.set_major_locator(ticker.FixedLocator([0, 20, 40, 60, 80, 100]))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
        ax.label_outer()

    axes[0].set_ylabel(r"$R^{\Upsilon}_{AA}$", fontsize=16)
    axes[len(y_windows) // 2].set_xlabel("Centrality [%]", fontsize=14)

    plt.tight_layout(pad=0.8, w_pad=0.8)
    return fig

def plot_centrality_with_mb_overlay_selected(cnm, energy, y_windows, pt_range_avg):
    """Publication-style R_AA vs centrality for a selected subset of rapidity windows."""
    ncols = len(y_windows)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    sys_note = rf"$\mathbf{{O+O, \sqrt{{s_{{\rm NN}}}} = {energy} \, TeV}}$"

    for i, ((y0, y1, yname), ax) in enumerate(zip(y_windows, axes)):
        bands = cnm.cnm_vs_centrality((y0, y1), pt_range_avg, components=CALC_COMPS)

        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands:
                continue
            Rc, Rlo, Rhi, mbc, mblo, mbhi = bands[comp]
            color = COLORS.get(comp, "black")
            ls = "--" if comp in ("npdf", "eloss_broad") else "-"
            lw = 2.2 if comp == "cnm" else 1.5

            xe, yc_s = cent_step_arrays(cnm.cent_bins, Rc)
            ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
            ax.fill_between(xe, cent_step_arrays(cnm.cent_bins, Rlo)[1], cent_step_arrays(cnm.cent_bins, Rhi)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)

            ax.fill_between([0, 100], [mblo] * 2, [mbhi] * 2, color=color, alpha=0.08, hatch="//", lw=0)
            ax.hlines(mbc, 0, 100, colors=color, linestyles=":", linewidth=1.5)

        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.text(0.05, 0.95, yname, transform=ax.transAxes, ha="left", va="top", fontsize=11, color="navy", fontweight="bold", bbox=NOTE_BOX)
        if i == 0:
            ax.text(0.05, 0.10, sys_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, bbox=NOTE_BOX)
            ax.set_ylabel(r"$R^{\Upsilon}_{AA}$", fontsize=16)
        ax.text(0.95, 0.05, pt_range_label(pt_range_avg), transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox=NOTE_BOX)
        if i == min(1, ncols - 1):
            ax.legend(loc="lower left", frameon=False, fontsize=10)

        ax.set_xlim(0, 100)
        ax.set_ylim(*Y_LIM_RPA)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0, 20, 40, 60, 80, 100]))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
        ax.set_xlabel("Centrality [%]", fontsize=14)
        ax.label_outer()

    plt.tight_layout(pad=0.8, w_pad=0.8)
    return fig

def plot_y_by_centrality_bins(cnm, yc, bands_y, energy, pt_range_avg):
    """Publication-ready R_AA vs y in centrality bins + MB."""
    cent_bins = [(0,10),(10,30),(30,50),(50,70),(70,100)]
    n_cent = len(cent_bins)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    # No suptitle

    sys_note = rf"$\mathbf{{O+O, \sqrt{{s_{{\rm NN}}}} = {energy} \, TeV}}$"

    for i, (c0, c1) in enumerate(cent_bins):
        ax = axes[i]
        # No subplot title

        # Plot R_AA vs y for this centrality bin
        tag = f"{int(c0)}-{int(c1)}%"
        
        for comp in COMPONENTS_TO_PLOT:
            if comp not in bands_y:
                continue
            Rc = bands_y[comp][0][tag]  # Central values
            Rlo = bands_y[comp][1][tag]  # Lower errors
            Rhi = bands_y[comp][2][tag]  # Upper errors

            color = COLORS.get(comp, "black")
            ls = "--" if comp in ("npdf", "eloss_broad") else "-"
            lw = 2.2 if comp == "cnm" else 1.5

            xe, yc_s = step_from_centers(yc, Rc)
            ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
            ax.fill_between(xe, step_from_centers(yc, Rlo)[1], step_from_centers(yc, Rhi)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)

        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        # Put text inside plot
        ax.text(0.05, 0.95, f"{int(c0)}-{int(c1)}%", transform=ax.transAxes, ha="left", va="top", fontsize=11, weight="bold", bbox=NOTE_BOX)
        ax.text(0.05, 0.05, pt_range_label(pt_range_avg), transform=ax.transAxes, ha="left", va="bottom", fontsize=9, bbox=NOTE_BOX)

        ax.set_xlim(*X_LIM_Y)
        ax.set_ylim(*Y_LIM_RPA)
        # No grid
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
        ax.label_outer()

    # Last subplot for MB
    ax = axes[-1]
    # No title

    for comp in COMPONENTS_TO_PLOT:
        if comp not in bands_y:
            continue
        Rc_mb = bands_y[comp][0]["MB"]
        Rlo_mb = bands_y[comp][1]["MB"]
        Rhi_mb = bands_y[comp][2]["MB"]

        color = COLORS.get(comp, "black")
        ls = "--" if comp in ("npdf", "eloss_broad") else "-"
        lw = 2.2 if comp == "cnm" else 1.5

        xe, yc_s = step_from_centers(yc, Rc_mb)
        ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
        ax.fill_between(xe, step_from_centers(yc, Rlo_mb)[1], step_from_centers(yc, Rhi_mb)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)

    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.text(0.05, 0.95, "Min Bias", transform=ax.transAxes, ha="left", va="top", fontsize=11, weight="bold", bbox=NOTE_BOX)
    ax.text(0.05, 0.05, pt_range_label(pt_range_avg), transform=ax.transAxes, ha="left", va="bottom", fontsize=9, bbox=NOTE_BOX)

    ax.set_xlim(*X_LIM_Y)
    ax.set_ylim(*Y_LIM_RPA)
    # No grid
    ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
    ax.label_outer()

    # Set axis labels
    for ax in axes[-3:]:
        ax.set_xlabel(r"$y$", fontsize=14)
    for ax in axes[::3]:
        ax.set_ylabel(r"$R^{\Upsilon}_{AA}$", fontsize=14)

    # In-panel legend (last / MB panel has most room at upper-right)
    axes[-1].legend(loc="upper right", frameon=False, fontsize=10)

    # System note inside first panel
    axes[0].text(0.5, 0.08, sys_note, transform=axes[0].transAxes, ha="center", va="bottom", fontsize=10, bbox=NOTE_BOX)

    plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=0.8)
    return fig

def plot_pt_by_rapidity_and_centrality(cnm, energy, y_windows, p_edges):
    """Publication-ready R_AA vs pT in 3 figures (one per rapidity window)."""
    cent_bins = [(0,10),(10,30),(30,50),(50,70),(70,100)]

    figures = []
    pt_results = []
    for yi, (y0, y1, yname) in enumerate(y_windows):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
        axes = axes.flatten()
        # No suptitle

        sys_note = rf"$\mathbf{{O+O, \sqrt{{s_{{\rm NN}}}} = {energy} \, TeV}}$"

        # Get pT data for this rapidity window (all centrality bins)
        pc, tags_pt, bands_pt = cnm.cnm_vs_pT((y0, y1), p_edges, components=CALC_COMPS, include_mb=True)
        pt_results.append((y0, y1, yname, pc, tags_pt, bands_pt))

        for i, tag in enumerate(tags_pt):
            ax = axes[i]
            # No subplot title

            for comp in COMPONENTS_TO_PLOT:
                if comp not in bands_pt:
                    continue
                Rc = bands_pt[comp][0][tag]  # Central values
                Rlo = bands_pt[comp][1][tag]  # Lower errors
                Rhi = bands_pt[comp][2][tag]  # Upper errors

                color = COLORS.get(comp, "black")
                ls = "--" if comp in ("npdf", "eloss_broad") else "-"
                lw = 2.2 if comp == "cnm" else 1.5

                xe, yc_s = step_from_centers(pc, Rc, x_start=0.0)
                ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=color, label=LABELS.get(comp, comp))
                ax.fill_between(xe, step_from_centers(pc, Rlo, x_start=0.0)[1], step_from_centers(pc, Rhi, x_start=0.0)[1], step="post", color=color, alpha=ALPHA_BAND, lw=0)

            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
            # Put text inside plot
            if i < len(cent_bins):
                c0, c1 = cent_bins[i]
                ax.text(0.05, 0.95, f"{int(c0)}-{int(c1)}%", transform=ax.transAxes, ha="left", va="top", fontsize=11, weight="bold", bbox=NOTE_BOX)
            else:
                ax.text(0.05, 0.95, "Min Bias", transform=ax.transAxes, ha="left", va="top", fontsize=11, weight="bold", bbox=NOTE_BOX)
            ax.text(0.05, 0.05, yname, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color="navy", fontweight="bold", bbox=NOTE_BOX)

            ax.set_xlim(*X_LIM_PT)
            ax.set_ylim(*Y_LIM_RPA)
            # Explicit ticks at 0, 4, 8, 12, 16, 20 GeV (include endpoints)
            ax.xaxis.set_major_locator(ticker.FixedLocator([0, 4, 8, 12, 16, 20]))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(prune="both"))
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
            ax.label_outer()

        # Set axis labels
        for ax in axes[-3:]:
            ax.set_xlabel(r"$p_T$ (GeV)", fontsize=14)
        for ax in axes[::3]:
            ax.set_ylabel(r"$R^{\Upsilon}_{AA}$", fontsize=14)

        # Hide any trailing empty subplots
        for ax in axes[len(tags_pt):]:
            ax.set_visible(False)

        # In-panel legend (MB panel has most room on right)
        mb_panel_idx = len(tags_pt) - 1
        axes[mb_panel_idx].legend(loc="upper right", frameon=False, fontsize=10)

        # System note inside first panel (centered bottom)
        axes[0].text(0.5, 0.08, sys_note, transform=axes[0].transAxes, ha="center", va="bottom", fontsize=10, bbox=NOTE_BOX)

        plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=0.8)
        figures.append(fig)

    return figures, pt_results

# ── Execution ────────────────────────────────────────────────────────
def run_energy(energy):
    cnm = build_eloss_context(energy)
    p_edges = getattr(cnm, "publication_p_edges", P_EDGES)
    pt_range_avg = getattr(cnm, "publication_pt_range_avg", PT_RANGE_AVG)
    etag = energy.replace('.', 'p')
    OUTDIR_PUB.mkdir(parents=True, exist_ok=True)
    
    # Clean old outputs
    print(f"  [CLEANUP] Removing old output files ...", flush=True)
    for f in OUTDIR_PUB.glob("*.pdf"):
        f.unlink()
    for f in OUTDIR_PUB.glob("*.png"):
        f.unlink()
    for f in OUTDIR_PUB.glob("*.csv"):
        f.unlink()

    # Get data for all plots (using test parameters for faster computation)
    yc, tags_y, bands_y = cnm.cnm_vs_y(y_edges=Y_EDGES, pt_range_avg=pt_range_avg, components=CALC_COMPS, include_mb=True)

    # 1. R_AA vs Centrality with MB overlay (1×3 subplots)
    print(f"  [PLOT] R_AA vs centrality with MB overlay ...", flush=True)
    fig_cent_mb = plot_centrality_with_mb_overlay(cnm, energy, Y_WINDOWS, pt_range_avg)
    fig_cent_mb.savefig(OUTDIR_PUB / f"Upsilon_RAA_CNM_vs_centrality_MB_overlay_OO_{etag}TeV.pdf", bbox_inches="tight")
    fig_cent_mb.savefig(OUTDIR_PUB / f"Upsilon_RAA_CNM_vs_centrality_MB_overlay_OO_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_cent_mb)

    # Custom publication variant requested: extend the forward arm to y=4.5
    # for the dedicated mid+forward centrality overlay figure only.
    selected_y_windows = [Y_WINDOWS[1], (2.5, 4.5, r"$2.5 < y < 4.5$")]
    fig_cent_mb_mid_fwd = plot_centrality_with_mb_overlay_selected(cnm, energy, selected_y_windows, pt_range_avg)
    fig_cent_mb_mid_fwd.savefig(OUTDIR_PUB / f"Upsilon_RAA_CNM_vs_centrality_MB_overlay_mid_forward_OO_{etag}TeV.pdf", bbox_inches="tight")
    fig_cent_mb_mid_fwd.savefig(OUTDIR_PUB / f"Upsilon_RAA_CNM_vs_centrality_MB_overlay_mid_forward_OO_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_cent_mb_mid_fwd)

    # Save centrality CSV data
    print(f"  [CSV] Saving centrality data ...", flush=True)
    for y0, y1, yname in Y_WINDOWS:
        bands_cent = cnm.cnm_vs_centrality((y0, y1), pt_range_avg, components=CALC_COMPS)
        yn = y_window_csv_tag(y0, y1)
        save_csv_cent_consolidated(OUTDIR_PUB, cnm, bands_cent, yn, energy)

    # 2. R_AA vs y by centrality bins + MB (2×3 grid)
    print(f"  [PLOT] R_AA vs y by centrality bins ...", flush=True)
    fig_y_cent = plot_y_by_centrality_bins(cnm, yc, bands_y, energy, pt_range_avg)
    fig_y_cent.savefig(OUTDIR_PUB / f"Upsilon_RAA_CNM_vs_y_by_centrality_OO_{etag}TeV.pdf", bbox_inches="tight")
    fig_y_cent.savefig(OUTDIR_PUB / f"Upsilon_RAA_CNM_vs_y_by_centrality_OO_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_y_cent)

    # Save y CSV data
    print(f"  [CSV] Saving rapidity data ...", flush=True)
    save_csv_y_consolidated(OUTDIR_PUB, yc, bands_y, tags_y, energy)

    # 3. R_AA vs pT by rapidity and centrality (3 figures, one per rapidity window)
    print(f"  [PLOT] R_AA vs pT by rapidity and centrality ...", flush=True)
    figures_pt, pt_results = plot_pt_by_rapidity_and_centrality(cnm, energy, Y_WINDOWS, p_edges)

    for i, (y0, y1, yname, pc, tags_pT, bands_pT) in enumerate(pt_results):
        yname_file = y_window_file_tag(y0, y1)
        fig = figures_pt[i]
        fig.savefig(OUTDIR_PUB / f"Upsilon_RAA_CNM_vs_pT_{yname_file}_OO_{etag}TeV.pdf", bbox_inches="tight")
        fig.savefig(OUTDIR_PUB / f"Upsilon_RAA_CNM_vs_pT_{yname_file}_OO_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # Save pT CSV data for all centrality bins and rapidity windows
    print(f"  [CSV] Saving pT data ...", flush=True)
    for y0, y1, yname, pc, tags_pT, bands_pT in pt_results:
        yname_file = y_window_file_tag(y0, y1)
        save_csv_pT_consolidated(OUTDIR_PUB, pc, bands_pT, tags_pT, energy, yname_file)

    # Additional pT outputs with extended forward/backward rapidity acceptance.
    print(f"  [PLOT] R_AA vs pT for extended rapidity windows ...", flush=True)
    figures_pt_ext, pt_results_ext = plot_pt_by_rapidity_and_centrality(cnm, energy, EXTENDED_PT_WINDOWS, p_edges)
    for i, (y0, y1, _yname, pc, tags_pT, bands_pT) in enumerate(pt_results_ext):
        yname_file = y_window_file_tag(y0, y1)
        fig = figures_pt_ext[i]
        fig.savefig(OUTDIR_PUB / f"Upsilon_RAA_CNM_vs_pT_{yname_file}_OO_{etag}TeV.pdf", bbox_inches="tight")
        fig.savefig(OUTDIR_PUB / f"Upsilon_RAA_CNM_vs_pT_{yname_file}_OO_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        save_csv_pT_consolidated(OUTDIR_PUB, pc, bands_pT, tags_pT, energy, yname_file)

    print(f"  ✓ DONE — O+O @ {energy} TeV publication plots")

def main():
    for e in ENERGIES:
        run_energy(e)

if __name__ == "__main__":
    main()
