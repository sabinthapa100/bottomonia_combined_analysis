#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bottomonia_cnm_prim_OO.py
============================
Final combined Upsilon CNM + Primordial production for O+O @ 5.36 TeV.

Combination:
  R_AA^Total = R_AA^CNM * R_AA^Primordial
  Errors add in quadrature.
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
    "cnm/npdf_code",
    "cnm/cnm_combine",
    "hnm/primordial_code",
    "cnm/cnm_scripts",
]

for d in reversed(paths_to_add):
    p = str(ROOT / d)
    if p not in sys.path:
        sys.path.insert(0, p)

from cnm_combine_fast_nuclabs import CNMCombineFast
from cnm_combine import combine_two_bands_1d
from ups_particle import make_bottomonia_system
from prim_band import PrimordialBand
from prim_io import load_pair
from prim_analysis import PrimordialAnalysis
from npdf_OO_data import load_OO_dat, build_OO_rpa_grid
from glauber import OpticalGlauber, SystemSpec
from gluon_ratio import GluonEPPSProvider, EPPS21Ratio
from npdf_centrality import compute_df49_by_centrality
from particle import Particle
from coupling import alpha_s_provider
import quenching_fast as QF

# ── Physics config ──────────────────────────────────────────────────
SQRTS_NN = 5360.0
SIG_NN_MB = 68.0
M_UPS_AVG = 10.01

INPUT_BASE = ROOT / "inputs" / "primordial"
PRIM_INPUTS = {
    "NPWLC": {
        "lower": INPUT_BASE / "output_OxOx5360_NPWLC" / "output-lower" / "datafile.gz",
        "upper": INPUT_BASE / "output_OxOx5360_NPWLC" / "output-upper" / "datafile.gz",
    },
    "Pert": {
        "lower": INPUT_BASE / "output_OxOx5360_Pert" / "output-lower" / "datafile.gz",
        "upper": INPUT_BASE / "output_OxOx5360_Pert" / "output-upper" / "datafile.gz",
    },
}

DPI = 150
ALPHA_BAND = 0.20

COMPONENTS_TO_PLOT = ["cnm", "total_NPWLC", "total_Pert"]

COLORS = {
    'cnm':          '#606060',   # gray
    'prim_NPWLC':   'tab:orange', 
    'prim_Pert':    'tab:cyan',
    'total_NPWLC':  'tab:red',    # red
    'total_Pert':   'tab:green',  # green
}

LABELS = {
    "cnm":          "CNM (nPDF x Eloss x $p_T-$broad)",
    "prim_NPWLC":   "TAMU-NP (Prim)",
    "prim_Pert":    "TAMU-P (Prim)",
    "total_NPWLC":  "CNM x TAMU-NP (Prim)",
    "total_Pert":   "CNM x TAMU-P (Prim)",
}

STATES = ["ups1S", "ups2S", "ups3S"]
STATE_NAMES = {
    "ups1S": r"$\Upsilon(1S)$",
    "ups2S": r"$\Upsilon(2S)$",
    "ups3S": r"$\Upsilon(3S)$",
}

CENT_BINS = [(0,20),(20,40),(40,60),(60,100)]
Y_EDGES = np.arange(-5.0, 5.0 + 0.5, 0.5)
P_EDGES = np.arange(0.0, 20.0 + 1.0, 1.0) 
PT_RANGE_AVG = (0.0, 20.0)

Y_WINDOWS = [
    (-2.4,  2.4, r"$-2.4 < y < 2.4$"),
    (-4.5, -2.4, r"$-4.5 < y < -2.4$"),
    ( 2.4,  4.0, r"$2.4 < y < 4.0$"),
]

# ── Factory Helpers ─────────────────────────────────────────────────
def build_cnm_context():
    print(f"\n[INFO] Loading CNM context for O+O @ 5.36 TeV ...", flush=True)
    OO_DAT = ROOT / "inputs" / "npdf" / "OxygenOxygen5360" / "nPDF_OO.dat"
    data = load_OO_dat(str(OO_DAT))
    grid = build_OO_rpa_grid(data, pt_max=20.0)
    gl = OpticalGlauber(SystemSpec("AA", SQRTS_NN, A=16, sigma_nn_mb=SIG_NN_MB), verbose=False)

    r0 = grid["r_central"].to_numpy()
    M = grid[[f"r_mem_{i:03d}" for i in range(1, 49)]].to_numpy().T
    SA_all = np.vstack([r0[None, :], M])

    epps_wrapper = EPPS21Ratio(A=16, path=str(ROOT / "inputs" / "npdf" / "nPDFs"))
    gluon_provider = GluonEPPSProvider(epps_wrapper, SQRTS_NN, m_state_GeV=M_UPS_AVG)

    df49_by_cent, _, _, _ = compute_df49_by_centrality(
        grid, r0, M, gluon_provider, gl,
        cent_bins=CENT_BINS, nb_bsamples=5, kind="AA", SA_all=SA_all
    )

    npdf_ctx = dict(df49_by_cent=df49_by_cent, df_pp=grid, df_pa=grid, gluon=gluon_provider)
    particle = Particle(family="bottomonia", state="avg", mass_override_GeV=9.46)
    alpha_s = alpha_s_provider(mode="running", LambdaQCD=0.25)
    Lmb = gl.leff_minbias_AA()

    qp_base = QF.QuenchParams(
        qhat0=0.075, lp_fm=1.5, LA_fm=Lmb, LB_fm=Lmb,
        system="AA", lambdaQCD=0.25, roots_GeV=SQRTS_NN,
        alpha_of_mu=alpha_s, alpha_scale="mT",
        use_hard_cronin=True, mapping="exp", device="cpu"
    )

    cnm = CNMCombineFast(
        energy="5.36", family="bottomonia", particle_state="avg",
        sqrt_sNN=SQRTS_NN, sigma_nn_mb=SIG_NN_MB,
        cent_bins=CENT_BINS, y_edges=Y_EDGES, p_edges=P_EDGES,
        y_windows=Y_WINDOWS, pt_range_avg=PT_RANGE_AVG, pt_floor_w=1.0,
        weight_mode="flat", y_ref=0.0, cent_c0=0.25,
        q0_pair=(0.05, 0.09), p0_scale_pair=(0.9, 1.1), nb_bsamples=5,
        y_shift_fraction=1.0, particle=particle, 
        npdf_ctx=npdf_ctx, gl=gl, qp_base=qp_base, spec=gl.spec
    )
    return cnm

def build_primordial_band(model):
    print(f"\n[INFO] Loading Primordial {model} context for O+O @ 5.36 TeV ...", flush=True)
    system = make_bottomonia_system(sqrts_pp_GeV=SQRTS_NN)
    paths = PRIM_INPUTS[model]
    df_lo, df_hi = load_pair(str(paths["lower"]), str(paths["upper"]), system, debug=False)
    ana_lo = PrimordialAnalysis(df_lo, system, with_feeddown=True)
    ana_hi = PrimordialAnalysis(df_hi, system, with_feeddown=True)
    return PrimordialBand(lower=ana_lo, upper=ana_hi, include_run_errors=True)

def align_to_grid(df, x_col, x_expected):
    df_grid = pd.DataFrame({x_col: x_expected})
    merged = pd.merge(df_grid, df, on=x_col, how="left")
    state_cols = [c for c in df.columns if c != x_col and not c.endswith("_lo") and not c.endswith("_hi") and not c.endswith("_err")]
    err_cols = [c for c in df.columns if c.endswith("_err") or c.endswith("_lo") or c.endswith("_hi")]
    merged[state_cols] = merged[state_cols].fillna(1.0)
    merged[err_cols] = merged[err_cols].fillna(0.0)
    return merged

def step_from_centers(centers, values):
    hw = (centers[1] - centers[0]) / 2.0
    edges = np.append(centers - hw, centers[-1] + hw)
    step_vals = np.append(values, values[-1])
    return edges, step_vals

def pin_edge_bins(bands_y, yc):
    """Pin outermost bins to neighbors to avoid nPDF edge artefacts."""
    if yc.size < 3: return
    for comp in bands_y:
        # bands_y[comp] is (Rc_dict, Rlo_dict, Rhi_dict)
        for i in range(3):
            d = bands_y[comp][i]
            for tag in d:
                vals = np.asarray(d[tag], float)
                vals[0] = vals[1]
                vals[-1] = vals[-2]
                d[tag] = vals

# ── Plotting ────────────────────────────────────────────────────────
def plot_results(cnm, band_np, band_pe):
    # R_AA vs y
    fig_y, axes_y = plt.subplots(1, 3, figsize=(15, 5), sharey=True, dpi=DPI)
    plt.subplots_adjust(wspace=0)

    yc, labels, dict_cnm = cnm.cnm_vs_y(include_mb=True)
    pin_edge_bins(dict_cnm, yc) # Fix nPDF edges
    cnm_c, cnm_lo, cnm_hi = dict_cnm["cnm"][0]["MB"], dict_cnm["cnm"][1]["MB"], dict_cnm["cnm"][2]["MB"]

    y_grid = list(zip(Y_EDGES[:-1], Y_EDGES[1:]))
    prim_np_c_raw, prim_np_band_raw = band_np.vs_y(PT_RANGE_AVG, y_grid)
    prim_pe_c_raw, prim_pe_band_raw = band_pe.vs_y(PT_RANGE_AVG, y_grid)

    prim_np_c = align_to_grid(prim_np_c_raw, "y", yc)
    prim_np_band = align_to_grid(prim_np_band_raw, "y", yc)
    prim_pe_c = align_to_grid(prim_pe_c_raw, "y", yc)
    prim_pe_band = align_to_grid(prim_pe_band_raw, "y", yc)

    sys_note = r"$\mathbf{O+O, \sqrt{s_{NN}} = 5.36 \ TeV \ (MB)}$"

    for i, state in enumerate(STATES):
        ax = axes_y[i]
        xe, yc_s = step_from_centers(yc, cnm_c)
        ax.step(xe, yc_s, lw=2, color=COLORS['cnm'], label=LABELS['cnm'])
        ax.fill_between(xe, step_from_centers(yc, cnm_lo)[1], step_from_centers(yc, cnm_hi)[1], color=COLORS['cnm'], alpha=0.1)

        # NPWLC
        p_c = prim_np_c[state].to_numpy()
        p_lo = prim_np_band[f"{state}_lo"].to_numpy()
        p_hi = prim_np_band[f"{state}_hi"].to_numpy()
        tot_c, tot_lo, tot_hi = combine_two_bands_1d(cnm_c, cnm_lo, cnm_hi, p_c, p_lo, p_hi)
        xe, ys = step_from_centers(yc, tot_c)
        ax.step(xe, ys, lw=2, color=COLORS['total_NPWLC'], label=LABELS['total_NPWLC'], where='post')
        ax.fill_between(xe, step_from_centers(yc, tot_lo)[1], step_from_centers(yc, tot_hi)[1], 
                        color=COLORS['total_NPWLC'], alpha=ALPHA_BAND, step='post')

        # Pert
        p_c = prim_pe_c[state].to_numpy()
        p_lo = prim_pe_band[f"{state}_lo"].to_numpy()
        p_hi = prim_pe_band[f"{state}_hi"].to_numpy()
        tot_c, tot_lo, tot_hi = combine_two_bands_1d(cnm_c, cnm_lo, cnm_hi, p_c, p_lo, p_hi)
        xe, ys = step_from_centers(yc, tot_c)
        ax.step(xe, ys, lw=2, ls='--', color=COLORS['total_Pert'], label=LABELS['total_Pert'], where='post')
        ax.fill_between(xe, step_from_centers(yc, tot_lo)[1], step_from_centers(yc, tot_hi)[1], 
                        color=COLORS['total_Pert'], alpha=ALPHA_BAND, step='post')

        ax.axhline(1.0, color='gray', ls=':', lw=0.8)
        ax.text(0.95, 0.95, fr"$\mathbf{{{STATE_NAMES[state][1:-1]}}}$", transform=ax.transAxes, ha='right', va='top', fontsize=14)
        
        if i == 0:
            ax.text(0.05, 0.95, sys_note, transform=ax.transAxes, ha='left', va='top', fontsize=11)
            ax.set_ylabel(r"$R_{AA}$", fontsize=16)
            ax.legend(loc='lower left', frameon=False, fontsize=10)

        ax.set_xlim(-4.5, 4.5); ax.set_ylim(0, 1.2)
        ax.set_xlabel(r"$y$", fontsize=15)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune='both'))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which='both', direction='in', top=True, right=True)

    # R_AA vs pT
    fig_pt, axes_pt = plt.subplots(len(Y_WINDOWS), 3, figsize=(15, 12), sharex=True, sharey=True, dpi=DPI)
    plt.subplots_adjust(wspace=0, hspace=0)

    for row, (y0, y1, yname) in enumerate(Y_WINDOWS):
        pc, labels, dict_cnm = cnm.cnm_vs_pT((y0, y1), include_mb=True)
        cnm_c, cnm_lo, cnm_hi = dict_cnm["cnm"][0]["MB"], dict_cnm["cnm"][1]["MB"], dict_cnm["cnm"][2]["MB"]

        pt_grid = list(zip(P_EDGES[:-1], P_EDGES[1:]))
        prim_np_c_raw, prim_np_band_raw = band_np.vs_pt((y0, y1), pt_grid)
        prim_pe_c_raw, prim_pe_band_raw = band_pe.vs_pt((y0, y1), pt_grid)

        prim_np_c = align_to_grid(prim_np_c_raw, "pt", pc)
        prim_np_band = align_to_grid(prim_np_band_raw, "pt", pc)
        prim_pe_c = align_to_grid(prim_pe_c_raw, "pt", pc)
        prim_pe_band = align_to_grid(prim_pe_band_raw, "pt", pc)

        for col, state in enumerate(STATES):
            ax = axes_pt[row, col]
            xe, yc_s = step_from_centers(pc, cnm_c)
            ax.step(xe, yc_s, lw=2, color=COLORS['cnm'], label=LABELS['cnm'] if (row==0 and col==0) else None)
            ax.fill_between(xe, step_from_centers(pc, cnm_lo)[1], step_from_centers(pc, cnm_hi)[1], color=COLORS['cnm'], alpha=0.1)

            # NPWLC
            p_c = prim_np_c[state].to_numpy()
            p_lo = prim_np_band[f"{state}_lo"].to_numpy()
            p_hi = prim_np_band[f"{state}_hi"].to_numpy()
            tot_c, tot_lo, tot_hi = combine_two_bands_1d(cnm_c, cnm_lo, cnm_hi, p_c, p_lo, p_hi)
            xe, ys = step_from_centers(pc, tot_c)
            ax.step(xe, ys, lw=2, color=COLORS['total_NPWLC'], label=LABELS['total_NPWLC'] if (row==0 and col==0) else None, where='post')
            ax.fill_between(xe, step_from_centers(pc, tot_lo)[1], step_from_centers(pc, tot_hi)[1], 
                            color=COLORS['total_NPWLC'], alpha=ALPHA_BAND, step='post')

            # Pert
            p_c = prim_pe_c[state].to_numpy()
            p_lo = prim_pe_band[f"{state}_lo"].to_numpy()
            p_hi = prim_pe_band[f"{state}_hi"].to_numpy()
            tot_c, tot_lo, tot_hi = combine_two_bands_1d(cnm_c, cnm_lo, cnm_hi, p_c, p_lo, p_hi)
            xe, ys = step_from_centers(pc, tot_c)
            ax.step(xe, ys, lw=2, ls='--', color=COLORS['total_Pert'], label=LABELS['total_Pert'] if (row==0 and col==0) else None, where='post')
            ax.fill_between(xe, step_from_centers(pc, tot_lo)[1], step_from_centers(pc, tot_hi)[1], 
                            color=COLORS['total_Pert'], alpha=ALPHA_BAND, step='post')

            ax.axhline(1.0, color='gray', ls=':', lw=0.8)
            ax.text(0.95, 0.95, fr"$\mathbf{{{STATE_NAMES[state][1:-1]}}}$", transform=ax.transAxes, ha='right', va='top', fontsize=14)
            ax.text(0.05, 0.95, yname, transform=ax.transAxes, ha='left', va='top', color='blue', fontsize=10, fontweight='bold')
            ax.set_xlim(0, 20); ax.set_ylim(0, 1.2)
            
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(which='both', direction='in', top=True, right=True)

            if row == 0 and col == 0:
                ax.legend(loc='lower left', frameon=False, fontsize=9)
            
            if row == len(Y_WINDOWS)-1: ax.set_xlabel(r"$p_T$ [GeV]", fontsize=15)
            if col == 0: ax.set_ylabel(r"$R_{AA}$", fontsize=15)

    return fig_y, fig_pt

# ── Integration & Export ───────────────────────────────────────────
def run_and_save_csv(cnm, band_np, band_pe, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    # R_AA vs y
    yc, labels, dict_cnm = cnm.cnm_vs_y(include_mb=True)
    pin_edge_bins(dict_cnm, yc) # Fix nPDF edges
    cnm_c, cnm_lo, cnm_hi = dict_cnm["cnm"][0]["MB"], dict_cnm["cnm"][1]["MB"], dict_cnm["cnm"][2]["MB"]

    y_grid = list(zip(Y_EDGES[:-1], Y_EDGES[1:]))
    models = {"NPWLC": band_np, "Pert": band_pe}
    
    for mname, band in models.items():
        prim_c_raw, prim_band_raw = band.vs_y(PT_RANGE_AVG, y_grid)
        p_c = align_to_grid(prim_c_raw, "y", yc)
        p_band = align_to_grid(prim_band_raw, "y", yc)
        
        for state in STATES:
            pc = p_c[state].to_numpy()
            plo = p_band[f"{state}_lo"].to_numpy()
            phi = p_band[f"{state}_hi"].to_numpy()
            tot_c, tot_lo, tot_hi = combine_two_bands_1d(cnm_c, cnm_lo, cnm_hi, pc, plo, phi)
            df = pd.DataFrame({
                "y": yc,
                "R_AA_val": np.round(tot_c, 5),
                "R_AA_lo": np.round(tot_lo, 5),
                "R_AA_hi": np.round(tot_hi, 5)
            })
            df.to_csv(out_dir / f"Upsilon_RAA_Integrated_vs_y_{state}_{mname}_OO_5p36TeV.csv", index=False)

    # R_AA vs pT
    for row, (y0, y1, yname) in enumerate(Y_WINDOWS):
        pc, labels, dict_cnm = cnm.cnm_vs_pT((y0, y1), include_mb=True)
        cnm_c, cnm_lo, cnm_hi = dict_cnm["cnm"][0]["MB"], dict_cnm["cnm"][1]["MB"], dict_cnm["cnm"][2]["MB"]
        ylbl = ["mid", "back", "forw"][row]
        
        for mname, band in models.items():
            pt_grid = list(zip(P_EDGES[:-1], P_EDGES[1:]))
            prim_c_raw, prim_band_raw = band.vs_pt((y0, y1), pt_grid)
            p_c = align_to_grid(prim_c_raw, "pt", pc)
            p_band = align_to_grid(prim_band_raw, "pt", pc)
            
            for state in STATES:
                p_v = p_c[state].to_numpy()
                p_l = p_band[f"{state}_lo"].to_numpy()
                p_h = p_band[f"{state}_hi"].to_numpy()
                tot_c, tot_lo, tot_hi = combine_two_bands_1d(cnm_c, cnm_lo, cnm_hi, p_v, p_l, p_h)
                
                df = pd.DataFrame({
                    "pt": pc,
                    "R_AA_val": np.round(tot_c, 5),
                    "R_AA_lo": np.round(tot_lo, 5),
                    "R_AA_hi": np.round(tot_hi, 5)
                })
                df.to_csv(out_dir / f"Upsilon_RAA_Integrated_vs_pt_{ylbl}_{state}_{mname}_OO_5p36TeV.csv", index=False)

def main():
    cnm = build_cnm_context()
    band_np = build_primordial_band("NPWLC")
    band_pe = build_primordial_band("Pert")

    OUT_DIR = ROOT / "outputs" / "cnm_hnm" / "integrated_OO"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  [PLOT] Integrated results ...")
    fig_y, fig_pt = plot_results(cnm, band_np, band_pe)
    fig_y.savefig(OUT_DIR / "Upsilon_RAA_Integrated_vs_y_OO_5p36TeV.pdf", bbox_inches='tight')
    fig_y.savefig(OUT_DIR / "Upsilon_RAA_Integrated_vs_y_OO_5p36TeV.png", dpi=DPI, bbox_inches='tight')
    fig_pt.savefig(OUT_DIR / "Upsilon_RAA_Integrated_vs_pT_Grid_OO_5p36TeV.pdf", bbox_inches='tight')
    fig_pt.savefig(OUT_DIR / "Upsilon_RAA_Integrated_vs_pT_Grid_OO_5p36TeV.png", dpi=DPI, bbox_inches='tight')
    plt.close('all')

    print(f"  [CSV] Exporting integrated results ...")
    run_and_save_csv(cnm, band_np, band_pe, OUT_DIR)

    print(f"\n[OK] Integrated output in: {OUT_DIR}")

if __name__ == "__main__":
    main()
