#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_compare_RpO_vs_RpPb.py
==========================
Comparison plot: R_pO (= R_g^O = Rg2) vs R_pPb (5.02 & 8.16 TeV) — Min-Bias.

Physics:
  R_pO  is extracted from nPDF_OO.dat column "Rg2" — the gluon nPDF ratio for
  one oxygen nucleus (effectively R_pO, what Ramona calls "Rg2").
  R_pPb is computed fresh from the EPPS21 .top files (same pipeline as
  run_upsilon_npdf_pPb.py) for 5.02 TeV and 8.16 TeV.

Both are min-bias results binned vs y and vs pT, with full Hessian error bands.

Output saved to:
  outputs/npdf/min_bias/compare_RpO_vs_RpPb/

Usage:
    conda run -n research python cnm/npdf_scripts/run_compare_RpO_vs_RpPb.py
"""
import sys, re, warnings
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

# ── Physics / Input Paths ─────────────────────────────────────────────
M_UPSILON_AVG = 10.01   # GeV  (average bottomonium mass)

OO_DAT   = ROOT / "inputs" / "npdf" / "OxygenOxygen5360" / "nPDF_OO.dat"
P5_DIR   = ROOT / "inputs" / "npdf" / "pPb5TeV"
P8_DIR   = ROOT / "inputs" / "npdf" / "pPb8TeV"
OUTDIR   = ROOT / "outputs" / "npdf" / "min_bias" / "compare_RpO_vs_RpPb"
OUTDIR.mkdir(exist_ok=True, parents=True)

# ── Binning ──────────────────────────────────────────────────────────
Y_EDGES    = np.arange(-5.0, 5.0 + 0.5, 0.5)        # same as OO script
P_EDGES    = np.arange(0.0,  15.0 + 1.0, 1.0)        # same as OO script
PT_RANGE   = (0.0, 15.0)
PT_FLOOR_W = 1.0
PT_MAX_OO  = 15.0                                      # cut for OO dat grid

# ── Styling ───────────────────────────────────────────────────────────
DPI         = 150
ALPHA_BAND  = 0.20

COLOR_PO    = "tab:orange"     # R_pO  (Rg2 from OO dat)
COLOR_P5    = "tab:blue"       # R_pPb 5.02 TeV
COLOR_P8    = "tab:green"      # R_pPb 8.16 TeV

# Labels — energy shown for all, no 'nPDF_O'
LABEL_PO    = r"$R_{pO}$ @ $\sqrt{s_{NN}}=5.36$ TeV (O)"
LABEL_P5    = r"$R_{pPb}$ @ $\sqrt{s_{NN}}=5.02$ TeV"
LABEL_P8    = r"$R_{pPb}$ @ $\sqrt{s_{NN}}=8.16$ TeV"

# Rapidity windows for the 3-panel pT plot
Y_WINDOWS_PT = [
    (-5.0, -2.4, r"$-5.0 < y < -2.4$"),
    (-2.4,  2.4, r"$|y| < 2.4$"),
    ( 2.4,  5.0, r"$2.4 < y < 5.0$"),
]

# ── Helper: step function from bin centers ─────────────────────────────
def step_from_centers(xc, vals):
    xc   = np.asarray(xc,   float)
    vals = np.asarray(vals, float)
    dx   = np.diff(xc)
    dx0  = dx[0] if dx.size else 1.0
    xe   = np.concatenate(([xc[0] - 0.5*dx0], xc + 0.5*dx0))
    ys   = np.concatenate([vals, vals[-1:]])
    return xe, ys

# ======================================================================
# 1.  R_pO (Rg2) extraction from nPDF_OO.dat
# ======================================================================

def load_RpO_grid(filepath: Path, pt_max: float = PT_MAX_OO) -> pd.DataFrame:
    """
    Parse nPDF_OO.dat and build an R_pO (Rg2) grid with Hessian bands.

    Columns in nPDF_OO.dat per EPPS21 set:
        y  pt  x1  x2  Rg1  Rg2  Rg1*Rg2

    Rg2 is the gluon nuclear modification from ONE oxygen nucleus (the
    "heavy-nucleus" side) — this is effectively R_pO = R_g^O.

    Returns
    -------
    DataFrame with columns: y, pt, r_central, r_lo, r_hi, r_mem_001..r_mem_048
    """
    text = filepath.read_text()
    set_pat = re.compile(r"^\s*EPPS21\s+set\s+(\d+)\s*$", re.MULTILINE)
    matches = list(set_pat.finditer(text))

    sets = {}
    for i, m in enumerate(matches):
        sid   = int(m.group(1))
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        rows  = []
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("y"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                vals = [float(p) for p in parts[:7]]
                rows.append(vals)
            except ValueError:
                continue
        if rows:
            df = pd.DataFrame(rows,
                              columns=["y", "pt", "x1", "x2", "Rg1", "Rg2", "Rg1Rg2"])
            sets[sid] = df

    # central set (set 1), filter pT <= pt_max
    central = sets[1].copy()
    central = central[central["pt"] <= pt_max].reset_index(drop=True)
    r0      = central["Rg2"].to_numpy()     # <-- Rg2 = R_pO
    N       = len(r0)

    # Hessian members (sets 2..49)
    mems = []
    for sid in range(2, 50):
        df_e = sets[sid].copy()
        df_e = df_e[df_e["pt"] <= pt_max].reset_index(drop=True)
        mems.append(df_e["Rg2"].to_numpy()[:N])

    M = np.stack(mems, axis=0)   # shape (48, N)

    # pairwise Hessian band
    D = M[0::2, :] - M[1::2, :]
    h = 0.5 * np.sqrt(np.sum(D * D, axis=0))

    out = central[["y", "pt"]].copy()
    out["r_central"] = r0
    out["r_lo"]      = r0 - h
    out["r_hi"]      = r0 + h
    for j in range(M.shape[0]):
        out[f"r_mem_{j+1:03d}"] = M[j]

    print(f"  [RpO grid] {len(out)} points, pT in [{out['pt'].min():.1f}, {out['pt'].max():.1f}] GeV")
    return out


def bin_RpO_vs_y(grid: pd.DataFrame,
                 y_edges: np.ndarray,
                 pt_range: tuple = (0.0, 15.0)) -> pd.DataFrame:
    """Average R_pO over pT range, bin in y."""
    mask = (grid["pt"] >= pt_range[0]) & (grid["pt"] <= pt_range[1])
    g    = grid[mask]
    mem_cols = [c for c in grid.columns if c.startswith("r_mem_")]
    y_cents  = 0.5 * (y_edges[:-1] + y_edges[1:])
    rows = []
    for i in range(len(y_edges) - 1):
        sel = g[(g["y"] >= y_edges[i]) & (g["y"] < y_edges[i + 1])]
        if len(sel) == 0:
            continue
        r0 = float(sel["r_central"].mean())
        M  = sel[mem_cols].mean(axis=0).to_numpy()          # (48,)
        D  = M[0::2] - M[1::2]
        h  = 0.5 * np.sqrt(np.sum(D * D))
        rows.append({"y_center":   float(y_cents[i]),
                     "r_central":  r0,
                     "r_lo":       r0 - h,
                     "r_hi":       r0 + h})
    return pd.DataFrame(rows)


def bin_RpO_vs_pT(grid: pd.DataFrame,
                  pt_edges: np.ndarray,
                  y_range: tuple = (-4.5, 4.5)) -> pd.DataFrame:
    """Average R_pO over y range, bin in pT."""
    mask = (grid["y"] >= y_range[0]) & (grid["y"] <= y_range[1])
    g    = grid[mask]
    mem_cols = [c for c in grid.columns if c.startswith("r_mem_")]
    pt_cents = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    rows = []
    for i in range(len(pt_edges) - 1):
        sel = g[(g["pt"] >= pt_edges[i]) & (g["pt"] < pt_edges[i + 1])]
        if len(sel) == 0:
            continue
        r0 = float(sel["r_central"].mean())
        M  = sel[mem_cols].mean(axis=0).to_numpy()
        D  = M[0::2] - M[1::2]
        h  = 0.5 * np.sqrt(np.sum(D * D))
        rows.append({"pt_center":  float(pt_cents[i]),
                     "r_central":  r0,
                     "r_lo":       r0 - h,
                     "r_hi":       r0 + h})
    return pd.DataFrame(rows)


# ======================================================================
# 2.  R_pPb extraction — reuse existing NPDFSystem pipeline (MB only)
# ======================================================================

def load_RpPb_MB(energy: str,
                 y_edges: np.ndarray,
                 pt_edges: np.ndarray,
                 pt_range: tuple = (0.0, 15.0)) -> dict:
    """
    Compute min-bias R_pPb vs y and vs pT using the EPPS21 .top files.
    Returns dict with keys 'y_df' and 'pt_df' (DataFrames with r_central/r_lo/r_hi).
    """
    from gluon_ratio import EPPS21Ratio, GluonEPPSProvider
    from npdf_data import NPDFSystem, RpAAnalysis

    SQRTS  = {"5.02": 5023.0, "8.16": 8160.0}
    PREFIX = {"5.02": "upsppb5_", "8.16": "upsppb_"}
    INDIR  = {  "5.02": P5_DIR,     "8.16": P8_DIR}

    sqrt_sNN  = SQRTS[energy]
    prefix    = PREFIX[energy]
    input_dir = INDIR[energy]

    print(f"\n  [pPb {energy} TeV] Loading EPPS21 .top files ...")
    sys_npdf = NPDFSystem.from_folder(str(input_dir), kick="pp",
                                      name=f"p+Pb {energy} TeV", prefix=prefix)
    print(f"    {len(sys_npdf.error_paths)+2} files loaded.")

    ana  = RpAAnalysis()
    base, r0, M = ana.compute_rpa_members(
        sys_npdf.df_pp, sys_npdf.df_pa, sys_npdf.df_errors,
        join="intersect", lowpt_policy="drop",
        pt_shift_min=PT_FLOOR_W, shift_if_r_below=0.0,
    )
    # M shape: (N_members, N_grid_points)

    # ── bin vs y (average over pT) ─────────────────────────────────
    # base has columns: y, pt  (from compute_rpa_members)
    mask_y = (base["pt"] >= pt_range[0]) & (base["pt"] <= pt_range[1])
    g_y    = base[mask_y]
    r0_y   = r0[mask_y]
    M_y    = M[:, mask_y]

    y_cents = 0.5 * (y_edges[:-1] + y_edges[1:])
    rows_y  = []
    with np.errstate(all="ignore"):
        for i in range(len(y_edges) - 1):
            sel_mask = (g_y["y"].to_numpy() >= y_edges[i]) & (g_y["y"].to_numpy() < y_edges[i + 1])
            if sel_mask.sum() == 0:
                continue
            r_c = float(np.nanmean(r0_y[sel_mask]))
            if not np.isfinite(r_c):
                continue
            m_c = np.nanmean(M_y[:, sel_mask], axis=1)   # (n_members,)
            D   = m_c[0::2] - m_c[1::2]
            h   = 0.5 * np.sqrt(np.nansum(D * D))
            rows_y.append({"y_center":  float(y_cents[i]),
                            "r_central": r_c,
                            "r_lo":      r_c - h,
                            "r_hi":      r_c + h})

    # ── bin vs pT (average over full y) ───────────────────────────
    pt_arr   = base["pt"].to_numpy()
    pt_cents = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    rows_pt  = []
    for i in range(len(pt_edges) - 1):
        sel_mask = (pt_arr >= pt_edges[i]) & (pt_arr < pt_edges[i + 1])
        if sel_mask.sum() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_c = float(np.nanmean(r0[sel_mask]))
        if not np.isfinite(r_c):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m_c = np.nanmean(M[:, sel_mask], axis=1)
        D   = m_c[0::2] - m_c[1::2]
        h   = 0.5 * np.sqrt(np.nansum(D * D))
        rows_pt.append({"pt_center": float(pt_cents[i]),
                         "r_central": r_c,
                         "r_lo":      r_c - h,
                         "r_hi":      r_c + h})

    print(f"    binned: {len(rows_y)} y-bins, {len(rows_pt)} pT-bins")
    return {"y_df":  pd.DataFrame(rows_y),
            "pt_df": pd.DataFrame(rows_pt),
            "raw":   (base, r0, M)}  # raw arrays for per-window rebinning


def bin_RpPb_pT_window(raw, y0: float, y1: float,
                       pt_edges: np.ndarray) -> pd.DataFrame:
    """
    Bin R_pPb vs pT for a specific rapidity window [y0, y1].

    Parameters
    ----------
    raw : tuple (base, r0, M)  from load_RpPb_MB["raw"]
    y0, y1 : rapidity limits
    pt_edges : array of pT bin edges
    """
    base, r0, M = raw
    y_arr  = base["y"].to_numpy()
    pt_arr = base["pt"].to_numpy()
    mask   = (y_arr >= y0) & (y_arr <= y1)

    r0_w = r0[mask]
    M_w  = M[:, mask]
    pt_w = pt_arr[mask]

    pt_cents = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    rows = []
    for i in range(len(pt_edges) - 1):
        sel = (pt_w >= pt_edges[i]) & (pt_w < pt_edges[i + 1])
        if sel.sum() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_c = float(np.nanmean(r0_w[sel]))
        if not np.isfinite(r_c):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m_c = np.nanmean(M_w[:, sel], axis=1)
        D = m_c[0::2] - m_c[1::2]
        h = 0.5 * np.sqrt(np.nansum(D * D))
        rows.append({"pt_center": float(pt_cents[i]),
                      "r_central": r_c,
                      "r_lo":      r_c - h,
                      "r_hi":      r_c + h})
    return pd.DataFrame(rows)




def _draw_band(ax, xc, Rc, Rlo, Rhi, color, label, lw=2.0, alpha=ALPHA_BAND):
    xe, yc_s  = step_from_centers(xc, Rc)
    _, ylo_s  = step_from_centers(xc, Rlo)
    _, yhi_s  = step_from_centers(xc, Rhi)
    ax.step(xe, yc_s,  where="post", lw=lw, color=color, label=label)
    ax.fill_between(xe, ylo_s, yhi_s, step="post", color=color,
                    alpha=alpha, linewidth=0)


def _style_ax(ax, xlabel, ylabel, xlim, ylim,
              xmaj=1.0, ymaj=0.1,
              xticks=ticker.MultipleLocator,
              yticks=ticker.MultipleLocator):
    ax.axhline(1.0, color="k", ls="-", lw=0.8)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xmaj))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ymaj))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)


# ======================================================================
# 4.  R_pO vs y (3-panel: one for each rapidity sub-range + global)
# ======================================================================

def plot_RpO_vs_RpPb_y(rpo_y, rpPb5_y, rpPb8_y, full_y_range=True):
    """
    Plot R_pO (Rg2, orange) vs R_pPb 5.02 (blue) and 8.16 TeV (green)
    as function of rapidity — min-bias bands.

    Produces TWO figures:
      (a) Full y range on single axes
      (b) 3-panel: negative / mid / positive rapidity
    """
    figures = []

    # ── (a) Single-panel full y range ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=DPI)

    _draw_band(ax,
               rpo_y["y_center"], rpo_y["r_central"],
               rpo_y["r_lo"], rpo_y["r_hi"],
               COLOR_PO, LABEL_PO)
    _draw_band(ax,
               rpPb5_y["y_center"], rpPb5_y["r_central"],
               rpPb5_y["r_lo"], rpPb5_y["r_hi"],
               COLOR_P5, LABEL_P5)
    _draw_band(ax,
               rpPb8_y["y_center"], rpPb8_y["r_central"],
               rpPb8_y["r_lo"], rpPb8_y["r_hi"],
               COLOR_P8, LABEL_P8)

    _style_ax(ax, r"$y$", r"$R$ (nPDF, EPPS21)", xlim=(-5, 5), ylim=(0.35, 1.30),
              xmaj=1, ymaj=0.1)

    ax.text(0.03, 0.96,
            r"$\Upsilon$ production — Min. Bias (nPDF only)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold")
    ax.text(0.03, 0.88,
            r"$p_T \in [0,\,15]$ GeV",
            transform=ax.transAxes, ha="left", va="top", fontsize=11)

    ax.legend(loc="lower center", frameon=True, fontsize=10,
              framealpha=0.85, edgecolor="gray")

    fig.tight_layout()
    figures.append(("RpO_vs_RpPb_vs_y_MB", fig))

    return figures


def plot_RpO_vs_RpPb_pT(windows_data):
    """
    3-panel pT plot: one panel per rapidity window.

    Parameters
    ----------
    windows_data : list of (yname, rpo_pt, rpPb5_pt, rpPb8_pt)
        Each entry is one panel.  *_pt DataFrames have columns
        pt_center, r_central, r_lo, r_hi.
    """
    n = len(windows_data)
    fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5.2), dpi=DPI,
                             sharey=True)
    plt.subplots_adjust(wspace=0)

    for col, (ax, (yname, rpo_pt, rpPb5_pt, rpPb8_pt)) in \
            enumerate(zip(axes, windows_data)):

        _draw_band(ax,
                   rpo_pt["pt_center"],  rpo_pt["r_central"],
                   rpo_pt["r_lo"],        rpo_pt["r_hi"],
                   COLOR_PO, LABEL_PO if col == 0 else None)
        _draw_band(ax,
                   rpPb5_pt["pt_center"], rpPb5_pt["r_central"],
                   rpPb5_pt["r_lo"],      rpPb5_pt["r_hi"],
                   COLOR_P5, LABEL_P5 if col == 0 else None)
        _draw_band(ax,
                   rpPb8_pt["pt_center"], rpPb8_pt["r_central"],
                   rpPb8_pt["r_lo"],      rpPb8_pt["r_hi"],
                   COLOR_P8, LABEL_P8 if col == 0 else None)

        _style_ax(ax,
                  xlabel=r"$p_T$ (GeV)",
                  ylabel=(r"$R$ (nPDF)" if col == 0 else ""),
                  xlim=(0, 15), ylim=(0.35, 1.55),
                  xmaj=2, ymaj=0.1)

        # rapidity label in the panel
        ax.text(0.97, 0.97, yname, transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold")

        # EPPS21 label — only on first panel
        if col == 0:
            ax.text(0.04, 0.97, "EPPS21",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=11, fontstyle="italic")

        ax.tick_params(labelleft=(col == 0))

    # shared legend on the leftmost panel
    axes[0].legend(loc="lower right", frameon=True, fontsize=9.5,
                   framealpha=0.88, edgecolor="gray")

    fig.text(0.5, -0.01, r"$p_T$ (GeV)", ha="center", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 1])
    return fig


# ======================================================================
# 5.  Save CSV helpers
# ======================================================================

def save_csv(outdir, df, stem):
    df.to_csv(outdir / f"{stem}.csv", index=False)


# ======================================================================
# 6.  Main
# ======================================================================

def main():
    print("=" * 65)
    print("  Compare R_pO (Rg2) vs R_pPb  (Min-Bias, nPDF = EPPS21)")
    print("  R_pO  @ O+O 5.36 TeV   | R_pPb @ 5.02 TeV & 8.16 TeV")
    print("=" * 65)

    # ── Step 1: Load R_pO grid (Rg2) ──────────────────────────────
    print("\n[1] Building R_pO (Rg2) grid from nPDF_OO.dat ...")
    rpo_grid = load_RpO_grid(OO_DAT, pt_max=PT_MAX_OO)

    rpo_y  = bin_RpO_vs_y(rpo_grid,  Y_EDGES, pt_range=PT_RANGE)
    rpo_pt = bin_RpO_vs_pT(rpo_grid, P_EDGES, y_range=(-4.5, 4.5))
    print(f"  ✓ R_pO: {len(rpo_y)} y-bins, {len(rpo_pt)} pT-bins")

    save_csv(OUTDIR, rpo_y,  "Upsilon_RpO_Rg2_vs_y_OO_5p36TeV_MB")
    save_csv(OUTDIR, rpo_pt, "Upsilon_RpO_Rg2_vs_pT_OO_5p36TeV_MB")

    # ── Step 2: Load R_pPb (5.02 and 8.16 TeV) ────────────────────
    rpPb5 = load_RpPb_MB("5.02", Y_EDGES, P_EDGES, pt_range=PT_RANGE)
    rpPb8 = load_RpPb_MB("8.16", Y_EDGES, P_EDGES, pt_range=PT_RANGE)

    save_csv(OUTDIR, rpPb5["y_df"],  "Upsilon_RpPb_vs_y_5p02TeV_MB")
    save_csv(OUTDIR, rpPb5["pt_df"], "Upsilon_RpPb_vs_pT_5p02TeV_MB")
    save_csv(OUTDIR, rpPb8["y_df"],  "Upsilon_RpPb_vs_y_8p16TeV_MB")
    save_csv(OUTDIR, rpPb8["pt_df"], "Upsilon_RpPb_vs_pT_8p16TeV_MB")

    # ── Step 3: Plots ──────────────────────────────────────────────
    print("\n[3] Generating comparison plots ...")

    # (A) vs y — unchanged single-panel
    figs_y = plot_RpO_vs_RpPb_y(rpo_y, rpPb5["y_df"], rpPb8["y_df"])
    for stem, fig in figs_y:
        fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(OUTDIR / f"{stem}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"    → {stem}.pdf/.png")

    # (B) vs pT — 3-panel (one per rapidity window)
    print("  [PLOT] R vs pT — 3-panel rapidity windows ...")
    windows_data = []
    for y0, y1, yname in Y_WINDOWS_PT:
        rpo_pt_w  = bin_RpO_vs_pT(rpo_grid, P_EDGES, y_range=(y0, y1))
        rpPb5_pt_w = bin_RpPb_pT_window(rpPb5["raw"], y0, y1, P_EDGES)
        rpPb8_pt_w = bin_RpPb_pT_window(rpPb8["raw"], y0, y1, P_EDGES)
        windows_data.append((yname, rpo_pt_w, rpPb5_pt_w, rpPb8_pt_w))
        # save CSVs per window
        ys = yname.replace("$","").replace("\\","").replace(" ","").replace("<","to").replace("|","")
        save_csv(OUTDIR, rpo_pt_w,  f"Upsilon_RpO_vs_pT_{ys}_MB")
        save_csv(OUTDIR, rpPb5_pt_w, f"Upsilon_RpPb5_vs_pT_{ys}_MB")
        save_csv(OUTDIR, rpPb8_pt_w, f"Upsilon_RpPb8_vs_pT_{ys}_MB")

    fig_pt = plot_RpO_vs_RpPb_pT(windows_data)
    fig_pt.savefig(OUTDIR / "RpO_vs_RpPb_vs_pT_MB.pdf", bbox_inches="tight")
    fig_pt.savefig(OUTDIR / "RpO_vs_RpPb_vs_pT_MB.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_pt)
    print("    → RpO_vs_RpPb_vs_pT_MB.pdf/.png  (3-panel)")

    print("\n" + "=" * 65)
    print("  ✓ ALL DONE")
    print(f"  Output dir: {OUTDIR.relative_to(ROOT)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
