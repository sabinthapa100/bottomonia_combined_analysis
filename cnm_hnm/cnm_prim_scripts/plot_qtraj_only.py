#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_qtraj_only.py
==================
Reads outputs from QTraj-NLO (wReg and noReg) previously processed,
and plots the *HNM Only* R_AA comparisons against the TAMU HNM exact results 
for 5.36 TeV O+O.

This script is purely analytical and does not interfere with the core TAMU Primordial engine.
"""

import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[2] 
paths_to_add = [
    "cnm/eloss_code",
    "cnm/npdf_code",
    "cnm/cnm_combine",
    "hnm/tamu-traj/data_analysis/primordial_code",
    "cnm/cnm_scripts",
    "cnm_hnm/cnm_prim_scripts"
]
for p in paths_to_add:
    sys.path.insert(0, str(ROOT / p))

from run_bottomonia_cnm_prim_OO import build_primordial_band, STATES, STATE_NAMES

P_STATES = ["chi_b1P", "chi_b2P"]
P_STATE_NAMES = {
    "chi_b1P": r"$\chi_{b}(1P)$",
    "chi_b2P": r"$\chi_{b}(2P)$"
}


DPI = 300

COLORS = {
    'prim_NPWLC': '#2ca02c',   # Green
    'prim_Pert':  '#7f7f7f',   # Gray
    'ups1S_wReg':  '#d62728',  # Red
    'ups1S_noReg': '#e377c2',  # Pink
    'ups2S_wReg':  '#ff7f0e',  # Orange
    'ups2S_noReg': '#1f77b4',  # Blue
    'ups3S_wReg':  '#9467bd',  # Purple
    'ups3S_noReg': '#8c564b',  # Brown
}

LABELS = {
    'prim_NPWLC': 'TAMU NPWLC (HNM)',
    'prim_Pert':  'TAMU Pert (HNM)'
}

Y_WINDOWS = [
    (-2.4, 2.4, "Mid-Rapidity (|y|<2.4)"),
    ( 2.5, 4.0, "Forward Rapidity (2.5<y<4.0)") # Kept same exact label
]
ALPHA_BAND = 0.2

Y_EDGES = np.linspace(-5.0, 5.0, 21)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
P_EDGES = np.linspace(0.0, 20.0, 21)
P_CENTERS = 0.5 * (P_EDGES[:-1] + P_EDGES[1:])
PT_RANGE_AVG = (0.0, 20.0)

def step_from_centers(centers, values):
    dx = centers[1] - centers[0]
    edges = np.zeros(len(centers) + 1)
    edges[:-1] = centers - dx/2
    edges[-1] = centers[-1] + dx/2
    ys = np.zeros(len(centers) + 1)
    ys[:-1] = values
    ys[-1] = values[-1]
    return edges, ys

def align_to_grid(band_df, grid_col, new_centers):
    # TAMU functions return dataframe indexed by pt or y.
    out = pd.DataFrame({grid_col: new_centers})
    for col in band_df.columns:
        if col != grid_col:
            out[col] = np.interp(new_centers, band_df[grid_col], band_df[col])
    return out

def load_qtraj_res_pt(tag):
    out = {}
    base_dir = ROOT / "outputs" / "qtraj_nlo" / "OO_536" / tag
    for (y0, y1, yname) in Y_WINDOWS:
        label = "mid" if "Mid" in yname else "forw"
        if y0 == -2.5 and y1 == 4.0: label = "forw"
        f = base_dir / f"qtraj_vs_pt_{label}.csv"
        if not f.exists(): continue
        df = pd.read_csv(f)
        for st in STATES + P_STATES: # "ups1S", "ups2S", "ups3S", "chi_b1P", "chi_b2P"
            if st in df.columns:
                if yname not in out: out[yname] = {}
                out[yname][st] = df[["pt", st, f"{st}_err"]].rename(columns={st: "R_AA_val", f"{st}_err": "R_AA_err"})
    return out

def load_qtraj_res_y(tag):
    out = {}
    base_dir = ROOT / "outputs" / "qtraj_nlo" / "OO_536" / tag
    f = base_dir / "qtraj_vs_y.csv"
    if not f.exists(): return out
    df = pd.read_csv(f)
    for st in STATES + P_STATES:
        if st in df.columns:
            out[st] = df[["y", st, f"{st}_err"]].rename(columns={st: "R_AA_val", f"{st}_err": "R_AA_err"})
    return out

print("Loading QTraj Outputs...")
QTRAJ_RESULTS = {
    'wReg': {'y': load_qtraj_res_y('wReg'), 'pt': load_qtraj_res_pt('wReg')},
    'noReg': {'y': load_qtraj_res_y('noReg'), 'pt': load_qtraj_res_pt('noReg')}
}

def get_tamu_state_key(state):
    if state == "chi_b1P": return "chibJ_1P"
    if state == "chi_b2P": return "chibJ_2P"
    return state

def _plot_qtraj_pt(ax, p_centers, yname, state, show_label=True):
    for tag in ['wReg', 'noReg']:
        if yname in QTRAJ_RESULTS[tag]['pt'] and state in QTRAJ_RESULTS[tag]['pt'][yname]:
            df = QTRAJ_RESULTS[tag]['pt'][yname][state]
            post = "w/ regeneration" if tag == 'wReg' else "w/o regeneration"
            fmt = 'o' if tag == 'wReg' else 's'
            color = COLORS[f'ups1S_{tag}'] if state.startswith('ups') else COLORS[f'ups1S_{tag}'] 
            ax.errorbar(df["pt"], df["R_AA_val"], yerr=df["R_AA_err"], fmt=fmt,
                        color=color, label=f"QTraj {post}" if show_label else None,
                        markersize=5, capsize=3, elinewidth=1.5, ls='none')

def _plot_qtraj_y(ax, y_centers, state, show_label=True):
    for tag in ['wReg', 'noReg']:
        if state in QTRAJ_RESULTS[tag]['y']:
            df = QTRAJ_RESULTS[tag]['y'][state]
            post = "w/ regeneration" if tag == 'wReg' else "w/o regeneration"
            fmt = 'o' if tag == 'wReg' else 's'
            color = COLORS[f'ups1S_{tag}'] if state.startswith('ups') else COLORS[f'ups1S_{tag}'] 
            ax.errorbar(df["y"], df["R_AA_val"], yerr=df["R_AA_err"], fmt=fmt,
                        color=color, label=f"QTraj {post}" if show_label else None,
                        markersize=5, capsize=3, elinewidth=1.5, ls='none')


def plot_hnm_only(band_np, band_pe):
    # Y-Plots
    fig_y, axes_y = plt.subplots(1, 3, figsize=(15, 5), sharey=True, dpi=DPI)
    plt.subplots_adjust(wspace=0)
    y_grid = list(zip(Y_EDGES[:-1], Y_EDGES[1:]))
    prim_np_c_raw, prim_np_band_raw = band_np.vs_y(PT_RANGE_AVG, y_grid)
    prim_pe_c_raw, prim_pe_band_raw = band_pe.vs_y(PT_RANGE_AVG, y_grid)
    prim_np_c = align_to_grid(prim_np_c_raw, "y", Y_CENTERS)
    prim_np_band = align_to_grid(prim_np_band_raw, "y", Y_CENTERS)
    prim_pe_c = align_to_grid(prim_pe_c_raw, "y", Y_CENTERS)
    prim_pe_band = align_to_grid(prim_pe_band_raw, "y", Y_CENTERS)

    for i, state in enumerate(STATES):
        ax = axes_y[i]
        _plot_qtraj_y(ax, Y_CENTERS, state, show_label=(i==0))
        
        ax.axhline(1.0, color='gray', ls=':', lw=0.8)
        ax.text(0.95, 0.95, fr"$\mathbf{{{STATE_NAMES[state][1:-1]}}}$", transform=ax.transAxes, ha='right', va='top', fontsize=14)
        if i == 0:
            ax.set_ylabel(r"$R_{pA}$ (HNM Only)", fontsize=16)
            ax.legend(loc='lower left', frameon=False, fontsize=9)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(0, 1.2)
        ax.set_xlabel(r"$y$", fontsize=15)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune='both'))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which='both', direction='in', top=True, right=True)

    # PT-Plots
    fig_pt, axes_pt = plt.subplots(len(Y_WINDOWS), 3, figsize=(15, 12), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(wspace=0, hspace=0)
    for row, (y0, y1, yname) in enumerate(Y_WINDOWS):
        pt_grid = list(zip(P_EDGES[:-1], P_EDGES[1:]))
        prim_np_c_raw, prim_np_band_raw = band_np.vs_pt((y0, y1), pt_grid)
        prim_pe_c_raw, prim_pe_band_raw = band_pe.vs_pt((y0, y1), pt_grid)
        prim_np_c = align_to_grid(prim_np_c_raw, "pt", P_CENTERS)
        prim_np_band = align_to_grid(prim_np_band_raw, "pt", P_CENTERS)
        prim_pe_c = align_to_grid(prim_pe_c_raw, "pt", P_CENTERS)
        prim_pe_band = align_to_grid(prim_pe_band_raw, "pt", P_CENTERS)

        for col, state in enumerate(STATES):
            ax = axes_pt[row, col]
            show = (row==0 and col==0)
            _plot_qtraj_pt(ax, P_CENTERS, yname, state, show_label=show)
            
            ax.axhline(1.0, color='gray', ls=':', lw=0.8)
            ax.text(0.95, 0.95, fr"$\mathbf{{{STATE_NAMES[state][1:-1]}}}$", transform=ax.transAxes, ha='right', va='top', fontsize=14)
            ax.text(0.05, 0.95, yname, transform=ax.transAxes, ha='left', va='top', color='blue', fontsize=10, fontweight='bold')
            ax.set_xlim(0, 20); ax.set_ylim(0, 1.2)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(which='both', direction='in', top=True, right=True)
            if show: ax.legend(loc='lower left', frameon=False, fontsize=8)
            if row == len(Y_WINDOWS)-1: ax.set_xlabel(r"$p_T$ [GeV]", fontsize=15)
            if col == 0: ax.set_ylabel(r"$R_{pA}$ (HNM Only)", fontsize=15)

    return fig_y, fig_pt



def plot_pwave_hnm_only(band_np, band_pe):
    # Y-Plots
    fig_y, axes_y = plt.subplots(1, 2, figsize=(10, 5), sharey=True, dpi=DPI)
    plt.subplots_adjust(wspace=0)
    y_grid = list(zip(Y_EDGES[:-1], Y_EDGES[1:]))
    prim_np_c_raw, prim_np_band_raw = band_np.vs_y(PT_RANGE_AVG, y_grid)
    prim_pe_c_raw, prim_pe_band_raw = band_pe.vs_y(PT_RANGE_AVG, y_grid)
    prim_np_c = align_to_grid(prim_np_c_raw, "y", Y_CENTERS)
    prim_np_band = align_to_grid(prim_np_band_raw, "y", Y_CENTERS)
    prim_pe_c = align_to_grid(prim_pe_c_raw, "y", Y_CENTERS)
    prim_pe_band = align_to_grid(prim_pe_band_raw, "y", Y_CENTERS)

    for i, state in enumerate(P_STATES):
        ax = axes_y[i]
        t_key = get_tamu_state_key(state)
        
        _plot_qtraj_y(ax, Y_CENTERS, state, show_label=(i==0))
        
        ax.axhline(1.0, color='gray', ls=':', lw=0.8)
        ax.text(0.95, 0.95, fr"$\mathbf{{{P_STATE_NAMES[state][1:-1]}}}$", transform=ax.transAxes, ha='right', va='top', fontsize=14)
        if i == 0:
            ax.set_ylabel(r"$R_{pA}$ (HNM Only)", fontsize=16)
            ax.legend(loc='lower left', frameon=False, fontsize=9)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(0, 1.2)
        ax.set_xlabel(r"$y$", fontsize=15)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune='both'))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which='both', direction='in', top=True, right=True)

    # PT-Plots
    fig_pt, axes_pt = plt.subplots(len(Y_WINDOWS), 2, figsize=(10, 12), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(wspace=0, hspace=0)
    for row, (y0, y1, yname) in enumerate(Y_WINDOWS):
        pt_grid = list(zip(P_EDGES[:-1], P_EDGES[1:]))
        prim_np_c_raw, prim_np_band_raw = band_np.vs_pt((y0, y1), pt_grid)
        prim_pe_c_raw, prim_pe_band_raw = band_pe.vs_pt((y0, y1), pt_grid)
        prim_np_c = align_to_grid(prim_np_c_raw, "pt", P_CENTERS)
        prim_np_band = align_to_grid(prim_np_band_raw, "pt", P_CENTERS)
        prim_pe_c = align_to_grid(prim_pe_c_raw, "pt", P_CENTERS)
        prim_pe_band = align_to_grid(prim_pe_band_raw, "pt", P_CENTERS)

        for col, state in enumerate(P_STATES):
            ax = axes_pt[row, col]
            show = (row==0 and col==0)
            t_key = get_tamu_state_key(state)
            _plot_qtraj_pt(ax, P_CENTERS, yname, state, show_label=show)
            
            ax.axhline(1.0, color='gray', ls=':', lw=0.8)
            ax.text(0.95, 0.95, fr"$\mathbf{{{P_STATE_NAMES[state][1:-1]}}}$", transform=ax.transAxes, ha='right', va='top', fontsize=14)
            ax.text(0.05, 0.95, yname, transform=ax.transAxes, ha='left', va='top', color='blue', fontsize=10, fontweight='bold')
            ax.set_xlim(0, 20); ax.set_ylim(0, 1.2)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(which='both', direction='in', top=True, right=True)
            if show: ax.legend(loc='lower left', frameon=False, fontsize=8)
            if row == len(Y_WINDOWS)-1: ax.set_xlabel(r"$p_T$ [GeV]", fontsize=15)
            if col == 0: ax.set_ylabel(r"$R_{pA}$ (HNM Only)", fontsize=15)

    return fig_y, fig_pt

if __name__ == "__main__":
    print("Building Primordial HNM Bands ...")
    band_np = build_primordial_band("NPWLC")
    band_pe = build_primordial_band("Pert")

    OUT_DIR = ROOT / "outputs" / "cnm_hnm" / "integrated_qtraj_only"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating HNM QTraj vs TAMU comparison figures ...")
    fig_y, fig_pt = plot_hnm_only(band_np, band_pe)
    
    fig_y.savefig(OUT_DIR / "Upsilon_RAA_HNM_QTraj_vs_y_OO_5p36TeV.png", dpi=DPI, bbox_inches='tight')
    fig_y.savefig(OUT_DIR / "Upsilon_RAA_HNM_QTraj_vs_y_OO_5p36TeV.pdf", bbox_inches='tight')
    
    fig_pt.savefig(OUT_DIR / "Upsilon_RAA_HNM_QTraj_vs_pT_OO_5p36TeV.png", dpi=DPI, bbox_inches='tight')
    fig_pt.savefig(OUT_DIR / "Upsilon_RAA_HNM_QTraj_vs_pT_OO_5p36TeV.pdf", bbox_inches='tight')

    print("Generating HNM (P-Wave) comparison figures ...")
    fig_py, fig_ppt = plot_pwave_hnm_only(band_np, band_pe)
    
    fig_py.savefig(OUT_DIR / "Chib_RAA_HNM_QTraj_vs_y_OO_5p36TeV.png", dpi=DPI, bbox_inches='tight')
    fig_py.savefig(OUT_DIR / "Chib_RAA_HNM_QTraj_vs_y_OO_5p36TeV.pdf", bbox_inches='tight')
    
    fig_ppt.savefig(OUT_DIR / "Chib_RAA_HNM_QTraj_vs_pT_OO_5p36TeV.png", dpi=DPI, bbox_inches='tight')
    fig_ppt.savefig(OUT_DIR / "Chib_RAA_HNM_QTraj_vs_pT_OO_5p36TeV.pdf", bbox_inches='tight')

    print(f"DONE. Plots saved to {OUT_DIR}")
    
    # Generate cleanly formatted CSV files
    all_rows = []
    for tag in QTRAJ_RESULTS:
        for st in QTRAJ_RESULTS[tag]['y']:
            df = QTRAJ_RESULTS[tag]['y'][st]
            df = df.copy()
            df["state"] = st
            df["type"] = tag
            all_rows.append(df)
    if all_rows:
        pd.concat(all_rows).to_csv(OUT_DIR / "QTraj_HNM_vs_y_OO_5p36TeV.csv", index=False)
    
    print(f"DONE. Unified CSVs saved to {OUT_DIR}")
