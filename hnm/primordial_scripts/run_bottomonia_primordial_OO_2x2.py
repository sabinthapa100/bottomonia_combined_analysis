#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bottomonia_primordial_OO_2x2.py
===================================
Production script for Primordial Υ suppression in O+O 5.36 TeV.
Generates a 2x2 grid (R_pA vs y, and R_pA vs pT in 3 windows).

Requirements:
- Two rates: NPWLC (solid) and Pert (dashed).
- No grid, no panel titles.
- Legend consolidated.
- y-range: [-4.5, 4.0].
- pT-range: [0, 20] GeV.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# ---- Path setup ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(os.path.dirname(_HERE), 'primordial_code')
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from ups_particle    import make_bottomonia_system
from prim_io         import load_pair
from prim_analysis   import PrimordialAnalysis
from prim_band       import PrimordialBand
from export_utils    import save_hepdata_csv

# ===========================================================================
# Configuration
# ===========================================================================
SQRTS_NN = 5360.0
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
INPUT_BASE = os.path.join(_REPO_ROOT, "inputs", "primordial")
OUT_DIR = os.path.join(_REPO_ROOT, "outputs", "cnm_hnm", "primordial_OO")

INPUTS = {
    "NPWLC": {
        "lower": os.path.join(INPUT_BASE, "output_OxOx5360_NPWLC", "output-lower", "datafile.gz"),
        "upper": os.path.join(INPUT_BASE, "output_OxOx5360_NPWLC", "output-upper", "datafile.gz"),
    },
    "Pert": {
        "lower": os.path.join(INPUT_BASE, "output_OxOx5360_Pert", "output-lower", "datafile.gz"),
        "upper": os.path.join(INPUT_BASE, "output_OxOx5360_Pert", "output-upper", "datafile.gz"),
    },
}

# Kinematics
Y_BINS = [(-5.0, -4.5), (-4.5, -4.0), (-4.0, -3.0), (-3.0, -2.0), (-2.0, -1.0),
          (-1.0,  0.0), ( 0.0,  1.0), ( 1.0,  2.0), ( 2.0,  3.0), ( 3.0,  4.0),
          ( 4.0,  4.5), ( 4.5,  5.0)]

PT_BINS = [(i*1.0, (i+1)*1.0) for i in range(20)]
PT_WINDOW_FOR_Y = (0.0, 20.0)

Y_WINDOWS = [
    (-2.4,  2.4),   # mid
    (-5.0, -2.4),   # back
    ( 2.4,  5.0),   # forw
]

# Styling
mpl.rcParams.update({
    "axes.linewidth":   1.4,
    "axes.labelsize":   15,
    "font.family":      "DejaVu Sans",
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":  12,
    "figure.autolayout": True,
})

COLORS = {
    "ups1S": plt.cm.tab10.colors[0],
    "ups2S": plt.cm.tab10.colors[1],
    "ups3S": plt.cm.tab10.colors[2],
}

# ===========================================================================
# Helpers
# ===========================================================================

def build_band(model: str) -> PrimordialBand:
    system = make_bottomonia_system(sqrts_pp_GeV=SQRTS_NN)
    paths  = INPUTS[model]
    df_lo, df_hi = load_pair(paths["lower"], paths["upper"], system)
    ana_lo = PrimordialAnalysis(df_lo, system, with_feeddown=True)
    ana_hi = PrimordialAnalysis(df_hi, system, with_feeddown=True)
    return PrimordialBand(lower=ana_lo, upper=ana_hi, include_run_errors=True)

def draw_band(ax, x, cen, lo, hi, color, ls='-', step=True):
    if step:
        ax.fill_between(x, lo, hi, alpha=0.2, facecolor=color, edgecolor="none", step="mid")
        ax.step(x, cen, color=color, lw=2.0, ls=ls, where="mid")
    else:
        ax.fill_between(x, lo, hi, alpha=0.2, facecolor=color, edgecolor="none")
        ax.plot(x, cen, color=color, lw=2.0, ls=ls)

# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("--> Loading bands...")
    band_npwlc = build_band("NPWLC")
    band_pert  = build_band("Pert")
    
    bands = {
        "NPWLC": (band_npwlc, '-'),
        "Pert":  (band_pert,  '--'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    # 1. R_AA vs y (top left)
    ax_y = axes[0, 0]
    for lbl, (band, ls) in bands.items():
        cen, bd = band.vs_y(PT_WINDOW_FOR_Y, Y_BINS, flip_y=False)
        xs = cen["y"].to_numpy()
        for st in ["ups1S", "ups2S", "ups3S"]:
            mu = cen[st].to_numpy()
            lo = bd[f"{st}_lo"].to_numpy()
            hi = bd[f"{st}_hi"].to_numpy()
            draw_band(ax_y, xs, mu, lo, hi, COLORS[st], ls=ls)
        
        # Save CSV for vs_y
        csv_name = os.path.join(OUT_DIR, f"Upsilon_RAA_Primordial_vs_y_{lbl}_OO_5p36TeV.csv")
        save_hepdata_csv(cen, bd, "y", Y_BINS, csv_name)
    
    ax_y.set_xlabel(r"$y$")
    ax_y.set_ylabel(r"$R_{AA}$")
    ax_y.set_xlim(-5.0, 5.0)
    ax_y.set_ylim(0, 1.2)
    ax_y.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax_y.text(0.05, 0.95, r"O+O 5.36 TeV (MB)", transform=ax_y.transAxes, va='top', fontweight='bold', fontsize=12)
    ax_y.text(0.05, 0.88, rf"$p_T \in [0, 20]$ GeV", transform=ax_y.transAxes, va='top', fontsize=11)
    
    # Tick decorations
    ax_y.tick_params(which='both', direction='in', top=True, right=True)
    ax_y.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax_y.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax_y.grid(False)

    # 2-4. R_AA vs pT (subpanels)
    target_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    y_win_labels = ["mid", "back", "forw"]
    for i, yw in enumerate(Y_WINDOWS):
        ax = target_axes[i]
        ylbl = y_win_labels[i]
        for lbl, (band, ls) in bands.items():
            cen, bd = band.vs_pt(yw, PT_BINS)
            xs = cen["pt"].to_numpy()
            for st in ["ups1S", "ups2S", "ups3S"]:
                mu = cen[st].to_numpy()
                lo = bd[f"{st}_lo"].to_numpy()
                hi = bd[f"{st}_hi"].to_numpy()
                draw_band(ax, xs, mu, lo, hi, COLORS[st], ls=ls)
            
            # Save CSV for vs_pt in this window
            csv_name = os.path.join(OUT_DIR, f"Upsilon_RAA_Primordial_vs_pt_{ylbl}_{lbl}_OO_5p36TeV.csv")
            save_hepdata_csv(cen, bd, "pt", PT_BINS, csv_name)
        
        ax.set_xlabel(r"$p_T$ [GeV]")
        if i == 1: # Left panel of bottom row
            ax.set_ylabel(r"$R_{AA}$")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 1.2)
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax.text(0.95, 0.95, rf"${yw[0]:.1f} < y < {yw[1]:.1f}$", transform=ax.transAxes, va='top', ha='right', fontweight='bold', fontsize=11, color='blue')
        
        # Tick decorations
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(False)

    # Legends - Consolidated on top-left panel
    state_handles = [Line2D([0], [0], color=COLORS[s], lw=2, label=rf"$\Upsilon({s[3:5]})$") for s in ["ups1S", "ups2S", "ups3S"]]
    model_handles = [
        Line2D([0], [0], color="black", lw=1.5, ls="-", label="TAMU-NP (Prim)"),
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="TAMU-P (Prim)")
    ]
    
    # Place state legend
    leg1 = ax_y.legend(handles=state_handles, loc="lower right", frameon=False, fontsize=12)
    ax_y.add_artist(leg1)
    # Place model legend
    ax_y.legend(handles=model_handles, loc="lower left", frameon=False, fontsize=12)

    # Save
    out_name = os.path.join(OUT_DIR, "Upsilon_RAA_Primordial_Summary_OO_5p36TeV")
    fig.savefig(out_name + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_name + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved plot: {out_name}.png")
    plt.close(fig)
    print(f"--> Saved plot: {out_name}.png")

if __name__ == "__main__":
    main()
