import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ------------------------------------------------------------------
# Project paths (edit ONLY if your folder layout differs)
# ------------------------------------------------------------------
ROOT = Path("/home/sawin/Desktop/bottomonia_combined_analysis/")                   # where "npdf_code/" and "input/" live
NPDF_CODE_DIR = ROOT / "cnm" / "npdf_code" # must contain: npdf_data.py, gluon_ratio.py, glauber.py, npdf_centrality.py

if str(NPDF_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(NPDF_CODE_DIR))

# ------------------------------------------------------------------
# npdf_code imports
# ------------------------------------------------------------------
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

print("[OK] Imported npdf_code modules.")

# ------------------------------------------------------------------
# Output toggles
# ------------------------------------------------------------------
SAVE_PDF   = True
SAVE_CSV   = True
DPI        = 150
ALPHA_BAND = 0.22

# ------------------------------------------------------------------
# Energy choice
# ------------------------------------------------------------------
ENERGY = "5.02"  # "5.02" or "8.16"

# ------------------------------------------------------------------
# Centrality bins (edit freely)
# ------------------------------------------------------------------
CENT_BINS = [(0,10),(10,20),(20,40),(40,60),(60,80),(80,100)]
# CENT_BINS = [(0,20),(20,40),(40,60),(60,80),(80,100)]

# ------------------------------------------------------------------
# Binning (edit freely)
# ------------------------------------------------------------------
Y_EDGES = np.arange(-5.5, 5.0 + 0.5, 0.5)
P_EDGES = np.arange(0.0, 20.0 + 2.5, 2.5)

Y_WINDOWS = [
    (-4.46, -2.96, "-4.46 < y < -2.96"),
    (-1.37,  0.43, "-1.37 < y < 0.43"),
    ( 2.03,  3.53, "2.03 < y < 3.53"),
]

PT_RANGE_AVG = (0.0, 15.0)  # used for RpA(y) and RpA(cent)
PT_FLOOR_W   = 1.0          # low-pT weight floor (same idea as your CNM notebook)

# ------------------------------------------------------------------
# nPDF weighting knobs (KEEP FLEXIBLE)
# ------------------------------------------------------------------
WEIGHT_MODE      = "pp@local"
Y_REF            = 0.0
NB_BSAMPLES      = 5
Y_SHIFT_FRACTION = 2.0       # IMPORTANT knob you requested
MB_C0            = 0.25      # exp-weight parameter used for MB (same style as CNM notebook)

# ------------------------------------------------------------------
# Input locations (match your cnm_combine defaults)
# ------------------------------------------------------------------
NPDF_INPUT_DIR = ROOT / "inputs" / "npdf"
P5_DIR   = NPDF_INPUT_DIR / "pPb5TeV"
P8_DIR   = NPDF_INPUT_DIR / "pPb8TeV"
EPPS_DIR = NPDF_INPUT_DIR / "nPDFs"

SQRTS_GEV = {"5.02": 5023.0, "8.16": 8160.0}
SIG_NN_MB = {"5.02": 67.0,   "8.16": 71.0}

OUTDIR = ROOT / f"outputs/npdf/"
OUTDIR.mkdir(exist_ok=True, parents=True)

print(f"[CFG] ENERGY={ENERGY} TeV, OUTDIR={OUTDIR}")

## Loaders and builders
def build_npdf_context(
    energy: str,
    cent_bins,
    nb_bsamples: int = NB_BSAMPLES,
    y_shift_fraction: float = Y_SHIFT_FRACTION,
    pt_floor_w: float = PT_FLOOR_W,
    m_state_for_np=10.01,   # Bottomonia average mass
):
    if energy not in SQRTS_GEV:
        raise ValueError("energy must be '5.02' or '8.16'")

    sqrt_sNN = SQRTS_GEV[energy]
    sigma_nn_mb = SIG_NN_MB[energy]
    input_dir = P5_DIR if energy == "5.02" else P8_DIR

    # EPPS21 ratio + gluon provider
    epps_ratio = EPPS21Ratio(A=208, path=str(EPPS_DIR))
    gluon = GluonEPPSProvider(
        epps_ratio,
        sqrt_sNN_GeV=sqrt_sNN,
        m_state_GeV=m_state_for_np,
        y_sign_for_xA=-1,
    )

    # Optical Glauber for pA
    gl_pA = OpticalGlauber(
        SystemSpec("pA", sqrt_sNN, A=208, sigma_nn_mb=sigma_nn_mb),
        verbose=False,
    )

    # Load TopDrawer nPDF tables
    sys_npdf = NPDFSystem.from_folder(
        str(input_dir),
        kick="pp",
        name=f"p+Pb {energy} TeV",
        prefix="upsppb5_" if energy == "5.02" else "upsppb_"
    )

    # Build RpA grid + Hessian members
    ana = RpAAnalysis()
    base, r0, M = ana.compute_rpa_members(
        sys_npdf.df_pp,
        sys_npdf.df_pa,
        sys_npdf.df_errors,
        join="intersect",
        lowpt_policy="drop",
        pt_shift_min=pt_floor_w,
        shift_if_r_below=0.0,
    )

    # Centrality dependence: df49 for each centrality bin
    df49_by_cent, K_by_cent, SA_all, Y_SHIFT = compute_df49_by_centrality(
        base, r0, M,
        gluon, gl_pA,
        cent_bins=cent_bins,
        nb_bsamples=nb_bsamples,
        y_shift_fraction=y_shift_fraction,
    )

    return dict(
        energy=energy,
        sqrt_sNN=sqrt_sNN,
        sigma_nn_mb=sigma_nn_mb,
        cent_bins=cent_bins,
        df49_by_cent=df49_by_cent,
        df_pp=sys_npdf.df_pp,
        df_pa=sys_npdf.df_pa,
        gluon=gluon,
        gl=gl_pA,
        K_by_cent=K_by_cent,
        SA_all=SA_all,
        Y_SHIFT=Y_SHIFT,
    )

ctx = build_npdf_context(ENERGY, CENT_BINS)
print(f"[OK] Built nPDF context for √sNN={ctx['sqrt_sNN']/1000:.2f} TeV, bins={len(CENT_BINS)}, Y_SHIFT={ctx['Y_SHIFT']}")


## Helpers
def tags_for_cent_bins(cent_bins, include_mb=True):
    tags = [f"{int(a)}-{int(b)}%" for (a,b) in cent_bins]
    if include_mb:
        tags.append("MB")
    return tags

def step_from_centers(x_cent, vals):
    x_cent = np.asarray(x_cent, float)
    vals = np.asarray(vals, float)
    assert x_cent.size == vals.size
    if x_cent.size > 1:
        dx = np.diff(x_cent)
        if not np.allclose(dx, dx[0]):
            raise ValueError("x_cent not uniformly spaced; provide edges instead.")
        dx0 = dx[0]
    else:
        dx0 = 1.0
    x_edges = np.concatenate(([x_cent[0] - 0.5*dx0], x_cent + 0.5*dx0))
    y_step  = np.concatenate([vals, vals[-1:]])
    return x_edges, y_step

def cent_step_arrays(cent_bins, vals):
    vals = np.asarray(vals, float)
    edges = [cent_bins[0][0]] + [b for (_, b) in cent_bins]
    x_edges = np.asarray(edges, float)
    y_step = np.concatenate([vals, vals[-1:]])
    return x_edges, y_step

def ncoll_by_cent_bins(ctx, optical=True):
    gl = ctx["gl"]
    fn = gl.ncoll_mean_bin_pA_optical if optical else gl.ncoll_mean_bin_pA
    ncoll = [fn(a/100.0, b/100.0) for (a,b) in ctx["cent_bins"]]
    ncoll_mb = fn(0.0, 1.0)
    return np.asarray(ncoll, float), float(ncoll_mb)

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
    rows = []
    for yname, (labels, Rc, Rlo, Rhi, mb) in cent_data_all.items():
        for i, ((cL,cR), lab) in enumerate(zip(ctx["cent_bins"], labels)):
            rows.append({
                "Rapidity_Window": yname,
                "Cent_Low": float(cL), "Cent_High": float(cR),
                "Ncoll": float(Ncoll_cent[i]),
                "RAA_Central": float(Rc[i]), "RAA_Err_Lo": float(Rlo[i]), "RAA_Err_Hi": float(Rhi[i]),
                "is_MB": False
            })
        rows.append({
            "Rapidity_Window": yname,
            "Cent_Low": 0.0, "Cent_High": 100.0,
            "Ncoll": float(Ncoll_MB),
            "RAA_Central": float(mb[0]), "RAA_Err_Lo": float(mb[1]), "RAA_Err_Hi": float(mb[2]),
            "is_MB": True
        })
    pd.DataFrame(rows).to_csv(filepath, index=False)

# ── Calculation Helpers ─────────────────────────────────────────────
def edges_from_centers(xc):
    xc = np.asarray(xc, float)
    if xc.size < 2:
        dx = 1.0
        return np.array([xc[0]-0.5*dx, xc[0]+0.5*dx], float)
    mids = 0.5*(xc[:-1] + xc[1:])
    left = xc[0] - (mids[0] - xc[0])
    right = xc[-1] + (xc[-1] - mids[-1])
    return np.concatenate([[left], mids, [right]])

def npdf_vs_y(ctx, y_edges, pt_range_avg, include_mb=True, mb_c0=MB_C0):
    wcent = make_centrality_weight_dict(ctx["cent_bins"], c0=mb_c0) if include_mb else None
    out = bin_rpa_vs_y(
        ctx["df49_by_cent"], ctx["df_pp"], ctx["df_pa"], ctx["gluon"],
        cent_bins=ctx["cent_bins"], y_edges=y_edges, pt_range_avg=pt_range_avg,
        weight_mode=WEIGHT_MODE, y_ref=Y_REF, pt_floor_w=PT_FLOOR_W,
        wcent_dict=wcent, include_mb=include_mb
    )
    y_cent = 0.5*(y_edges[:-1] + y_edges[1:])
    tags = tags_for_cent_bins(ctx["cent_bins"], include_mb=include_mb)
    bands = {tag: (np.asarray(out[tag]["r_central"], float),
                   np.asarray(out[tag]["r_lo"], float),
                   np.asarray(out[tag]["r_hi"], float)) for tag in tags}
    return y_cent, tags, bands

def npdf_vs_pT(ctx, y_window, pt_edges, include_mb=True, mb_c0=MB_C0):
    wcent = make_centrality_weight_dict(ctx["cent_bins"], c0=mb_c0) if include_mb else None
    out = bin_rpa_vs_pT(
        ctx["df49_by_cent"], ctx["df_pp"], ctx["df_pa"], ctx["gluon"],
        cent_bins=ctx["cent_bins"], pt_edges=pt_edges, y_window=y_window,
        weight_mode=WEIGHT_MODE, y_ref=Y_REF, pt_floor_w=PT_FLOOR_W,
        wcent_dict=wcent, include_mb=include_mb
    )
    pT_cent = 0.5*(pt_edges[:-1] + pt_edges[1:])
    tags = tags_for_cent_bins(ctx["cent_bins"], include_mb=include_mb)
    bands = {tag: (np.asarray(out[tag]["r_central"], float),
                   np.asarray(out[tag]["r_lo"], float),
                   np.asarray(out[tag]["r_hi"], float)) for tag in tags}
    return pT_cent, tags, bands

def npdf_vs_centrality(ctx, y_window, pt_range_avg, mb_c0=MB_C0):
    wcent = make_centrality_weight_dict(ctx["cent_bins"], c0=mb_c0)
    width_weights = np.array([wcent[f"{int(a)}-{int(b)}%"] for (a,b) in ctx["cent_bins"]], float)
    out = bin_rpa_vs_centrality(
        ctx["df49_by_cent"], ctx["df_pp"], ctx["df_pa"], ctx["gluon"],
        cent_bins=ctx["cent_bins"], y_window=y_window, pt_range_avg=pt_range_avg,
        weight_mode=WEIGHT_MODE, y_ref=Y_REF, pt_floor_w=PT_FLOOR_W,
        width_weights=width_weights
    )
    labels = [f"{int(a)}-{int(b)}%" for (a,b) in ctx["cent_bins"]]
    Rc, Rlo, Rhi = np.asarray(out["r_central"]), np.asarray(out["r_lo"]), np.asarray(out["r_hi"])
    mb = (float(out["mb_r_central"]), float(out["mb_r_lo"]), float(out["mb_r_hi"]))
    return labels, Rc, Rlo, Rhi, mb

# ── Main Run ────────────────────────────────────────────────────────
def run_main():
    etag = ENERGY.replace('.','p')
    OUTDIR_MB   = ROOT / "outputs" / "npdf" / "min_bias" / f"pPb_{etag}TeV"
    OUTDIR_CENT = ROOT / "outputs" / "npdf" / "centrality" / f"pPb_{etag}TeV"
    OUTDIR_MB.mkdir(exist_ok=True, parents=True)
    OUTDIR_CENT.mkdir(exist_ok=True, parents=True)

    print(f"  [OUT]   min_bias   → {OUTDIR_MB}")
    print(f"  [OUT]   centrality  → {OUTDIR_CENT}")

    # Ncoll mapping (optical default)
    Ncoll_cent, Ncoll_MB = ncoll_by_cent_bins(ctx, optical=True)

    # (A) R_pA vs y (Grid)
    print(f"  [PLOT] R_pA vs y — full grid ...")
    y_cent, tags_y, bands_y = npdf_vs_y(ctx, Y_EDGES, PT_RANGE_AVG, include_mb=True)
    
    # Grid Plot (Gold Standard)
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)
    axes_flat = axes.flatten()
    for ip, tag in enumerate(tags_y):
        ax = axes_flat[ip]
        Rc, Rlo, Rhi = bands_y[tag]
        xe, yc_s = step_from_centers(y_cent, Rc)
        ax.step(xe, yc_s, where="post", lw=1.8, color="tab:blue")
        ax.fill_between(xe, step_from_centers(y_cent, Rlo)[1], step_from_centers(y_cent, Rhi)[1], step="post", color="tab:blue", alpha=0.25, lw=0)
        ax.axhline(1.0, color="k", ls="-", lw=0.8)
        ax.text(0.95, 0.92, tag, transform=ax.transAxes, ha="right", va="top", weight='bold', fontsize=11)
        ax.set_xlim(-5.0, 5.0); ax.set_ylim(0.0, 1.50)
        ax.label_outer()
    fig.text(0.5, 0.04, r'$y$', ha='center', fontsize=18)
    fig.text(0.08, 0.5, r'$R^{\Upsilon}_{pA} (nPDF)$', va='center', rotation='vertical', fontsize=18)
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_y_{etag}TeV_FullGrid.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_y_{etag}TeV_FullGrid.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    save_consolidated_y_csv(OUTDIR_CENT / f"Table_RAA_vs_y_Grid_{etag}TeV.csv", y_cent, bands_y, tags_y)

    # MB-only y plot
    Rc_mb, Rlo_mb, Rhi_mb = bands_y["MB"]
    fig_mb, ax_mb = plt.subplots(figsize=(8,5), dpi=DPI)
    xe, yc_s = step_from_centers(y_cent, Rc_mb)
    ax_mb.step(xe, yc_s, where="post", lw=2, color="tab:blue", label="nPDF (EPPS21)")
    ax_mb.fill_between(xe, step_from_centers(y_cent, Rlo_mb)[1], step_from_centers(y_cent, Rhi_mb)[1], step="post", color="tab:blue", alpha=ALPHA_BAND, lw=0)
    ax_mb.axhline(1.0, color="k", ls="-", lw=0.8)
    ax_mb.set_xlabel(r"$y$", fontsize=14); ax_mb.set_ylabel(r"$R^{\Upsilon}_{pA}$ (nPDF)", fontsize=14)
    ax_mb.set_xlim(-5, 5); ax_mb.set_ylim(0.4, 1.3)
    ax_mb.legend(loc="lower right", frameon=False, fontsize=12)
    ax_mb.text(0.03, 0.95, rf"p+Pb @ {ENERGY} TeV  (Min. Bias)", transform=ax_mb.transAxes, ha="left", va="top", fontsize=13, fontweight="bold")
    fig_mb.savefig(OUTDIR_MB / f"Upsilon_RpA_nPDF_vs_y_MB_{etag}TeV.pdf", bbox_inches="tight")
    fig_mb.savefig(OUTDIR_MB / f"Upsilon_RpA_nPDF_vs_y_MB_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_mb)

    # (B) R_pA vs pT (Grid)
    print(f"  [PLOT] R_pA vs pT — full grid ...")
    all_bands_pT = {}
    n_rows, n_cols = len(Y_WINDOWS), len(tags_y)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.0*n_rows), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(hspace=0, wspace=0)
    for row_idx, (y0, y1, yname) in enumerate(Y_WINDOWS):
        pc, tags_pt, bands_pt = npdf_vs_pT(ctx, (y0, y1), P_EDGES, include_mb=True)
        all_bands_pT[yname] = bands_pt
        for col_idx, tag in enumerate(tags_pt):
            ax = axes[row_idx, col_idx]
            Rc, Rlo, Rhi = bands_pt[tag]
            xe, yc_s = step_from_centers(pc, Rc)
            ax.step(xe, yc_s, where="post", lw=1.5, color="tab:blue")
            ax.fill_between(xe, step_from_centers(pc, Rlo)[1], step_from_centers(pc, Rhi)[1], step="post", color="tab:blue", alpha=0.3, lw=0)
            ax.axhline(1.0, color="gray", ls="--", lw=0.8)
            ax.text(0.92, 0.88, tag, transform=ax.transAxes, ha="right", weight='bold', fontsize=10)
            ax.text(0.08, 0.08, yname, transform=ax.transAxes, color="navy", fontsize=11, fontweight='bold')
            ax.set_xlim(0, 15.0); ax.set_ylim(0.0, 1.50)
            ax.label_outer()
    fig.text(0.5, 0.04, r'$p_T$ (GeV)', ha='center', fontsize=20)
    fig.text(0.08, 0.5, r'$R^{\Upsilon}_{pA} (nPDF)$', va='center', rotation='vertical', fontsize=20)
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_pT_{etag}TeV_FullGrid.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR_CENT / f"Upsilon_RpA_nPDF_vs_pT_{etag}TeV_FullGrid.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    save_consolidated_pT_csv(OUTDIR_CENT / f"Table_RAA_vs_pT_Grid_{etag}TeV.csv", pc, all_bands_pT, tags_pt)

    # MB-only pT (3 windows)
    fig_mb_pt, axes_mb_pt = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI, sharey=True)
    for ax, (y0, y1, yname) in zip(axes_mb_pt, Y_WINDOWS):
        pc_mb, tags_pt_mb, bands_pt_mb = npdf_vs_pT(ctx, (y0, y1), P_EDGES, include_mb=True)
        Rc, Rlo, Rhi = bands_pt_mb["MB"]
        xe, yc_s = step_from_centers(pc_mb, Rc)
        ax.step(xe, yc_s, where="post", lw=2, color="tab:blue", label="nPDF (EPPS21)")
        ax.fill_between(xe, step_from_centers(pc_mb, Rlo)[1], step_from_centers(pc_mb, Rhi)[1], step="post", color="tab:blue", alpha=ALPHA_BAND, lw=0)
        ax.axhline(1.0, color="k", ls="-", lw=0.8)
        ax.text(0.5, 0.92, yname, transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$p_T$ (GeV)", fontsize=12)
        if ax == axes_mb_pt[0]:
            ax.set_ylabel(r"$R^{\Upsilon}_{pA}$ (nPDF)", fontsize=12)
            ax.text(0.05, 0.05, f"p+Pb @ {ENERGY} TeV\nMin. Bias", transform=ax.transAxes, fontsize=10)
        ax.set_xlim(0, 15); ax.set_ylim(0.4, 1.3)
    fig_mb_pt.tight_layout()
    fig_mb_pt.savefig(OUTDIR_MB / f"Upsilon_RpA_nPDF_vs_pT_MB_{etag}TeV.pdf", bbox_inches="tight")
    fig_mb_pt.savefig(OUTDIR_MB / f"Upsilon_RpA_nPDF_vs_pT_MB_{etag}TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_mb_pt)

    # (C) R_pA vs Centrality
    print(f"  [PLOT] R_pA vs centrality ...")
    npdf_cent_all_dict = {}
    for y0, y1, yname in Y_WINDOWS:
        res = npdf_vs_centrality(ctx, (y0, y1), PT_RANGE_AVG)
        npdf_cent_all_dict[yname] = res
    
    # Re-use existing plotters if possible, or simple inline
    save_consolidated_cent_csv(OUTDIR_CENT / f"Table_RAA_vs_Centrality_{etag}TeV.csv", ctx, npdf_cent_all_dict, Ncoll_cent, Ncoll_MB)

if __name__ == "__main__":
    run_main()
    print("\n DONE — Consolidated pPb nPDF Outputs.")

