#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_oo5360_bottomonia.py
========================
Primordial Υ suppression analysis for O+O collisions at √s_NN = 5.36 TeV.

Uses both NPWLC and Pert TAMU outputs (upper + lower bounds each) to produce:
  1. R_pA vs y  (full rapidity range, all Υ states on one plot)
  2. R_pA vs pT (three rapidity windows as sub-panels)
  3. Comparison overlay: NPWLC vs Pert on the same axes

Rapidity regions (CMS O+O 5.36 TeV):
  - Midrapidity: |y| < 2.4
  - Backward   : -5.0 < y < -2.4
  - Forward    :  2.0 < y < 4.5

Run:
    conda run -n research python agents/run_oo5360_bottomonia.py

All outputs saved to:  agents/output_primordial_OO5360/
"""

from __future__ import annotations

import os
import sys

# ---- Path setup: allow running from repo root ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(os.path.dirname(_HERE), 'primordial_code')
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import numpy as np

from ups_particle    import make_bottomonia_system
from prim_io         import load_pair
from prim_analysis   import PrimordialAnalysis
from prim_band       import PrimordialBand
from plot_primordial import (
    plot_rpa_vs_y_and_pt,
    plot_rpa_comparison_vs_y,
    plot_rpa_vs_pt_three_windows,
)
from export_utils import save_hepdata_csv

# ===========================================================================
# Configuration
# ===========================================================================

# O+O √s_NN [GeV]
SQRTS_NN = 5360.0

# Input directories (relative to repo root; adjust if running from elsewhere)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
INPUT_BASE = os.path.join(_REPO_ROOT, "inputs", "primordial")

_NPWLC = os.path.join(INPUT_BASE, "output_OxOx5360_NPWLC")
_PERT  = os.path.join(INPUT_BASE, "output_OxOx5360_Pert")

INPUTS = {
    "NPWLC": {
        "lower": os.path.join(_NPWLC, "output-lower", "datafile.gz"),
        "upper": os.path.join(_NPWLC, "output-upper", "datafile.gz"),
    },
    "Pert": {
        "lower": os.path.join(_PERT, "output-lower", "datafile.gz"),
        "upper": os.path.join(_PERT, "output-upper", "datafile.gz"),
    },
}

# Output directory
OUT_DIR = os.path.join(_REPO_ROOT, "outputs", "hnm", "primordial", "oxygenoxygen5360")

# Feed-down on by default
WITH_FEEDDOWN = True

# ---------------------------------------------------------------------------
# Kinematic bins (LHC O+O)
# ---------------------------------------------------------------------------

# Full y range present in the data
Y_BINS = [(-5.0, -4.0), (-4.0, -3.0), (-3.0, -2.0), (-2.0, -1.0),
          (-1.0,  0.0), (-0.5,  0.5), ( 0.0,  1.0),
          ( 1.0,  2.0), ( 2.0,  3.0), ( 3.0,  4.0), ( 4.0,  5.0)]

# Compact y bins used for summary
Y_BINS_COARSE = [(-4.0, -2.5), (-2.5, -1.0), (-1.0, 1.0), (1.0, 2.5), (2.5, 4.0)]

# pT windows for rapidity overview
PT_WINDOW_FOR_Y = (0.0, 20.0)

# Rapidity windows for pT plots
Y_WINDOWS_FOR_PT = [
    (-5.0, -2.4),   # backward
    (-2.4,  2.4),   # midrapidity
    ( 2.0,  4.5),   # forward
]

# pT bins for pT plots
PT_BINS = [(i*1.0, (i+1)*1.0) for i in range(20)]  # 0-20 GeV in 1 GeV steps

# ===========================================================================
# Main
# ===========================================================================

def build_band(model: str, feeddown: bool = True) -> PrimordialBand:
    """Load lower + upper for a given model and return PrimordialBand."""
    system = make_bottomonia_system(sqrts_pp_GeV=SQRTS_NN)
    paths  = INPUTS[model]

    print(f"\n[{model}] Loading lower: {paths['lower']}")
    print(f"[{model}] Loading upper: {paths['upper']}")

    df_lo, df_hi = load_pair(paths["lower"], paths["upper"], system, debug=False)

    print(f"  → lower: {len(df_lo):,} rows | b unique: {list(np.unique(df_lo['b']))}")
    print(f"  → upper: {len(df_hi):,} rows | b unique: {list(np.unique(df_hi['b']))}")

    ana_lo = PrimordialAnalysis(df_lo, system, with_feeddown=feeddown)
    ana_hi = PrimordialAnalysis(df_hi, system, with_feeddown=feeddown)

    return PrimordialBand(lower=ana_lo, upper=ana_hi, include_run_errors=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== Primordial Υ suppression  O+O √s_NN = {SQRTS_NN/1000:.3f} TeV ===")
    print(f"    Feed-down: {'ON' if WITH_FEEDDOWN else 'OFF'}")
    print(f"    Outputs   : {OUT_DIR}")

    # ---- Build bands ----
    band_npwlc = build_band("NPWLC", feeddown=WITH_FEEDDOWN)
    band_pert  = build_band("Pert",  feeddown=WITH_FEEDDOWN)

    # ---- Quick sanity: print central values vs y for Υ(1S) ----
    print("\n--- Sanity check: Υ(1S) R_pA vs y [NPWLC, lower] ---")
    cen, bd = band_npwlc.vs_y(
        pt_window=PT_WINDOW_FOR_Y,
        y_bins=Y_BINS_COARSE,
        flip_y=False,
    )
    print(cen[["y", "ups1S", "ups2S", "ups3S"]].to_string(index=False))

    # ----------------------------------------------------------------
    # Figure 1: R_pA vs y + pT panels  — NPWLC only
    # ----------------------------------------------------------------
    plot_rpa_vs_y_and_pt(
        band_dict={"NPWLC": (band_npwlc, "bottomonia", "-")},
        system_name="bottomonia",
        y_bins=Y_BINS,
        pt_window_for_y=PT_WINDOW_FOR_Y,
        y_windows_for_pt=Y_WINDOWS_FOR_PT,
        pt_bins=PT_BINS,
        ylim=(0.0, 1.15),
        xlim_y=(-5.0, 5.0),
        xlim_pt=(0.0, 20.0),
        flip_y=False,
        step=True,
        suptitle=r"O+O, $\sqrt{s_{\rm NN}}$ = 5.36 TeV (MB) — NPWLC",
        save_dir=OUT_DIR,
        tag="rpa_NPWLC",
        legend_loc="lower left",
        text_loc=(0.03, 0.05),
    )

    # ---- Export Data ----
    cen_y, bd_y = band_npwlc.vs_y(PT_WINDOW_FOR_Y, Y_BINS, flip_y=False)
    save_hepdata_csv(
        cen_y, bd_y, "y", Y_BINS, 
        os.path.join(OUT_DIR, "rpa_vs_y_NPWLC_hepdata.csv")
    )
    
    for yw in Y_WINDOWS_FOR_PT:
        yw_str = f"{yw[0]}_{yw[1]}".replace(".","p")
        cen_pt, bd_pt = band_npwlc.vs_pt(yw, PT_BINS)
        save_hepdata_csv(
            cen_pt, bd_pt, "pt", PT_BINS,
            os.path.join(OUT_DIR, f"rpa_vs_pt_NPWLC_y_{yw_str}_hepdata.csv")
        )

    # ----------------------------------------------------------------
    # Figure 2: R_pA vs y + pT panels  — Pert only
    # ----------------------------------------------------------------
    plot_rpa_vs_y_and_pt(
        band_dict={"Pert": (band_pert, "bottomonia", "--")},
        system_name="bottomonia",
        y_bins=Y_BINS,
        pt_window_for_y=PT_WINDOW_FOR_Y,
        y_windows_for_pt=Y_WINDOWS_FOR_PT,
        pt_bins=PT_BINS,
        ylim=(0.0, 1.15),
        xlim_y=(-5.0, 5.0),
        xlim_pt=(0.0, 20.0),
        flip_y=False,
        step=True,
        suptitle=r"O+O, $\sqrt{s_{\rm NN}}$ = 5.36 TeV (MB) — Pert",
        save_dir=OUT_DIR,
        tag="rpa_Pert",
        legend_loc="lower left",
        text_loc=(0.03, 0.05),
    )
    
    # ---- Export Data ----
    cen_y_p, bd_y_p = band_pert.vs_y(PT_WINDOW_FOR_Y, Y_BINS, flip_y=False)
    save_hepdata_csv(
        cen_y_p, bd_y_p, "y", Y_BINS, 
        os.path.join(OUT_DIR, "rpa_vs_y_Pert_hepdata.csv")
    )

    for yw in Y_WINDOWS_FOR_PT:
        yw_str = f"{yw[0]}_{yw[1]}".replace(".","p")
        cen_pt, bd_pt = band_pert.vs_pt(yw, PT_BINS)
        save_hepdata_csv(
            cen_pt, bd_pt, "pt", PT_BINS,
            os.path.join(OUT_DIR, f"rpa_vs_pt_Pert_y_{yw_str}_hepdata.csv")
        )

    # ----------------------------------------------------------------
    # Figure 3: Overlay comparison — NPWLC vs Pert
    # ----------------------------------------------------------------
    entry_dict = {
        "NPWLC": (band_npwlc, "bottomonia", "-"),
        "Pert":  (band_pert,  "bottomonia", "--"),
    }

    plot_rpa_comparison_vs_y(
        entries=entry_dict,
        system_name="bottomonia",
        y_bins=Y_BINS,
        pt_window=PT_WINDOW_FOR_Y,
        flip_y=False,
        ylim=(0.0, 1.15),
        xlim_y=(-5.0, 5.0),
        step=True,
        suptitle=r"O+O, $\sqrt{s_{\rm NN}}$ = 5.36 TeV: NPWLC vs Pert (MB)",
        save_dir=OUT_DIR,
        tag="comparison_vs_y",
        legend_loc_model="lower left",
    )

    # ----------------------------------------------------------------
    # Figure 4: R_pA vs pT — three rapidity windows, NPWLC vs Pert
    # ----------------------------------------------------------------
    plot_rpa_vs_pt_three_windows(
        entries=entry_dict,
        system_name="bottomonia",
        pt_bins=PT_BINS,
        y_windows=Y_WINDOWS_FOR_PT,
        ylim=(0.0, 1.15),
        xlim_pt=(0.0, 20.0),
        step=True,
        suptitle=r"O+O, $\sqrt{s_{\rm NN}}$ = 5.36 TeV (MB)",
        save_dir=OUT_DIR,
        tag="rpa_pt_windows_compare",
        legend_loc_model="lower left",
        text_loc=(0.03, 0.05),
    )

    print(f"\n=== Done! All plots in: {OUT_DIR} ===")


if __name__ == "__main__":
    main()
