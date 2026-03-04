#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_upsilon_npdf_OO.py
======================
Upsilon nPDF (min-bias) analysis for O+O @ 5.36 TeV.

Uses pre-computed gluon nPDF ratios from nPDF_OO.dat (49 EPPS21 sets).
  R_AA^{nPDF}(y, pT) = Rg1(x1) * Rg2(x2)

No pp cross-section needed — ratios are directly provided.
No nuclear absorption. No glauber changes.

Outputs:
  outputs/npdf/min_bias/  — R_AA vs y, R_AA vs pT (per y-window) + CSV
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

# ── Config ───────────────────────────────────────────────────────────
OO_DAT = ROOT / "inputs" / "npdf" / "OxygenOxygen5360" / "nPDF_OO.dat"
OUTDIR = ROOT / "outputs" / "npdf" / "min_bias" / "OO_5p36TeV"
OUTDIR.mkdir(exist_ok=True, parents=True)

DPI = 150
ALPHA_BAND = 0.22
PT_MAX = 15.0

SQRT_SNN = 5360.0  # GeV
SYSTEM_LABEL = r"O+O @ $\sqrt{s_{NN}}=5.36$ TeV"

Y_EDGES = np.arange(-5.0, 5.0 + 0.5, 0.5)
PT_EDGES = np.arange(0.0, PT_MAX + 0.5, 0.5)
PT_RANGE_MB = (0.0, PT_MAX)

Y_WINDOWS = [
    (-4.5, -2.5, r"$-4.5 < y < -2.5$"),
    (-2.5,  0.0, r"$-2.5 < y < 0$"),
    ( 0.0,  2.5, r"$0 < y < 2.5$"),
    ( 2.5,  4.5, r"$2.5 < y < 4.5$"),
]

# ── Helpers ──────────────────────────────────────────────────────────
def step_from_centers(xc, vals):
    xc = np.asarray(xc, float); vals = np.asarray(vals, float)
    dx = np.diff(xc)
    dx0 = dx[0] if dx.size else 1.0
    xe = np.concatenate(([xc[0]-0.5*dx0], xc+0.5*dx0))
    ys = np.concatenate([vals, vals[-1:]])
    return xe, ys


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Upsilon nPDF — O+O @ 5.36 TeV  (Min-Bias)")
    print("  Using pre-computed gluon ratios from nPDF_OO.dat")
    print("  49 EPPS21 sets (1 central + 48 Hessian)")
    print("="*60)

    # 1. Load
    print("\n[1] Loading nPDF_OO.dat ...")
    data = load_OO_dat(str(OO_DAT))
    n_sets = len(data)
    n_rows = len(data[1])
    y_min, y_max = data[1]['y'].min(), data[1]['y'].max()
    pt_min, pt_max = data[1]['pt'].min(), data[1]['pt'].max()
    print(f"    ✓ {n_sets} EPPS21 sets loaded")
    print(f"    ✓ {n_rows} data points per set")
    print(f"    ✓ y in [{y_min:.2f}, {y_max:.2f}],  pT in [{pt_min:.1f}, {pt_max:.1f}] GeV")

    # 2. Build R_AA grid
    print("\n[2] Building R_AA grid (Hessian error bands) ...")
    grid = build_OO_rpa_grid(data, pt_max=PT_MAX)
    print(f"    ✓ Grid: {len(grid)} points after pT < {PT_MAX} GeV cut")

    # Spot checks
    for yval in [0.0, 2.0, -3.0]:
        sel = grid[(grid['y'].abs() - abs(yval) < 0.3) & (grid['pt'] < 3.0)]
        if len(sel) > 0:
            print(f"    [check] y≈{yval:+.1f}, pT<3: R_AA = {sel['r_central'].mean():.4f} "
                  f"± [{sel['r_lo'].mean():.4f}, {sel['r_hi'].mean():.4f}]")

    # 3. R_AA vs y
    print("\n[3] Plotting R_AA vs y (pT-averaged, min-bias) ...")
    rpa_y = bin_rpa_vs_y_OO(grid, Y_EDGES, pt_range=PT_RANGE_MB)
    yc = rpa_y["y_center"].to_numpy()
    Rc = rpa_y["r_central"].to_numpy()
    Rlo = rpa_y["r_lo"].to_numpy()
    Rhi = rpa_y["r_hi"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    xe, yc_s = step_from_centers(yc, Rc)
    _, ylo_s = step_from_centers(yc, Rlo)
    _, yhi_s = step_from_centers(yc, Rhi)
    ax.step(xe, yc_s, where="post", lw=2.0, color="tab:red", label="nPDF (EPPS21)")
    ax.fill_between(xe, ylo_s, yhi_s, step="post", color="tab:red",
                    alpha=ALPHA_BAND, linewidth=0)
    ax.axhline(1.0, color="k", ls="-", lw=0.8)
    ax.set_xlabel(r"$y$", fontsize=14)
    ax.set_ylabel(r"$R^{\Upsilon}_{AA}$ (nPDF)", fontsize=14)
    ax.set_xlim(-5, 5); ax.set_ylim(0.4, 1.15)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    ax.text(0.03, 0.95, SYSTEM_LABEL + "  (Min. Bias)", transform=ax.transAxes,
            ha="left", va="top", fontsize=13, fontweight="bold")
    ax.text(0.03, 0.87, rf"$p_T \in [{PT_RANGE_MB[0]:.0f},\,{PT_RANGE_MB[1]:.0f}]$ GeV",
            transform=ax.transAxes, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Upsilon_RAA_nPDF_vs_y_OO_5p36TeV.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "Upsilon_RAA_nPDF_vs_y_OO_5p36TeV.png", dpi=DPI, bbox_inches="tight")
    rpa_y.to_csv(OUTDIR / "Upsilon_RAA_nPDF_vs_y_OO_5p36TeV.csv", index=False)
    plt.close()
    print(f"    ✓ Saved: Upsilon_RAA_nPDF_vs_y_OO_5p36TeV.{{pdf,png,csv}}")

    # 4. R_AA vs pT — panel per y-window
    print("\n[4] Plotting R_AA vs pT (per y-window) ...")
    n_win = len(Y_WINDOWS)
    fig, axes = plt.subplots(1, n_win, figsize=(5*n_win, 5), dpi=DPI, sharey=True)
    if n_win == 1: axes = [axes]

    for ax, (y0, y1, label) in zip(axes, Y_WINDOWS):
        rpa_pt = bin_rpa_vs_pT_OO(grid, PT_EDGES, y_range=(y0, y1))
        pc = rpa_pt["pt_center"].to_numpy()
        rc_pt = rpa_pt["r_central"].to_numpy()
        rlo_pt = rpa_pt["r_lo"].to_numpy()
        rhi_pt = rpa_pt["r_hi"].to_numpy()

        xe, yc_s = step_from_centers(pc, rc_pt)
        _, ylo_s = step_from_centers(pc, rlo_pt)
        _, yhi_s = step_from_centers(pc, rhi_pt)
        ax.step(xe, yc_s, where="post", lw=2.0, color="tab:red")
        ax.fill_between(xe, ylo_s, yhi_s, step="post", color="tab:red",
                        alpha=ALPHA_BAND, linewidth=0)
        ax.axhline(1.0, color="k", ls="-", lw=0.8)
        ax.set_xlabel(r"$p_T$ (GeV)", fontsize=13)
        ax.text(0.95, 0.95, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold")
        ax.set_xlim(0, PT_MAX); ax.set_ylim(0.4, 1.15)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)

        # Save per-window CSV
        safe_name = label.replace("$","").replace(" ","").replace("<","").replace(">","").replace("\\","")
        rpa_pt.to_csv(OUTDIR / f"Upsilon_RAA_nPDF_vs_pT_OO_{safe_name}_5p36TeV.csv", index=False)

    axes[0].set_ylabel(r"$R^{\Upsilon}_{AA}$ (nPDF)", fontsize=13)
    axes[0].text(0.05, 0.05, SYSTEM_LABEL, transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTDIR / "Upsilon_RAA_nPDF_vs_pT_OO_5p36TeV.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "Upsilon_RAA_nPDF_vs_pT_OO_5p36TeV.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved: Upsilon_RAA_nPDF_vs_pT_OO_5p36TeV.{{pdf,png,csv}}")

    # 5. Summary
    mid = grid[(grid["y"].abs() < 0.5) & (grid["pt"] <= 5.0)]
    fwd = grid[(grid["y"] > 2.0) & (grid["y"] < 4.0) & (grid["pt"] <= 5.0)]
    bwd = grid[(grid["y"] < -2.0) & (grid["y"] > -4.0) & (grid["pt"] <= 5.0)]
    print("\n" + "="*60)
    print("  SUMMARY — O+O 5.36 TeV  nPDF  min-bias  R_AA  (pT<5 GeV)")
    print("="*60)
    print(f"  Mid-rapidity |y|<0.5:  {mid['r_central'].mean():.4f} "
          f"[{mid['r_lo'].mean():.4f}, {mid['r_hi'].mean():.4f}]")
    print(f"  Forward  2<y<4:        {fwd['r_central'].mean():.4f} "
          f"[{fwd['r_lo'].mean():.4f}, {fwd['r_hi'].mean():.4f}]")
    print(f"  Backward -4<y<-2:      {bwd['r_central'].mean():.4f} "
          f"[{bwd['r_lo'].mean():.4f}, {bwd['r_hi'].mean():.4f}]")
    print("="*60)
    print("  ✓ DONE — O+O nPDF analysis complete\n")


if __name__ == "__main__":
    main()
