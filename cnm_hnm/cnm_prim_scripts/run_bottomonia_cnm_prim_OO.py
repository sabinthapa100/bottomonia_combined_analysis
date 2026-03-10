#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_bottomonia_cnm_prim_OO_v2.py
=================================
Final combined Upsilon CNM + Primordial production for Oxygen-Oxygen @ 5.36 TeV.
Production-ready framework with binned output, error propagation, and HEPData export.

Combination: R_AA^Total = R_AA^CNM * R_AA^Primordial
Errors: Quadrature (asymmetric bands)

Output Structure:
  outputs/cnm_prim/min_bias/OO_5p36TeV/
  ├── binned_data/          # HEPData CSVs
  │   ├── Upsilon_RAA_vs_y_MB_OO_5p36TeV.csv
  │   ├── Upsilon_RAA_vs_pT_Backward_MB_OO_5p36TeV.csv
  │   └── ... (for each state & model)
  └── plots/                 # Publication plots
      ├── Upsilon_RAA_vs_y_MB_OO_5p36TeV.pdf
      └── ... (state-specific plots)

CURRENT: Min-bias primordial only (both NPWLC and Pert models)
FUTURE: Centrality-dependent primordial data when available
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Path setup ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
paths_to_add = [
    "cnm/eloss_code",
    "cnm/npdf_code",
    "cnm/cnm_combine",
    "hnm/primordial_code",
    "cnm/cnm_scripts",
    "cnm/npdf_code/deps",
]
for d in reversed(paths_to_add):
    p = str(ROOT / d)
    if p not in sys.path:
        sys.path.insert(0, p)

from cnm_combine_fast_nuclabs import CNMCombineFast
from cnm_combine import combine_two_bands_1d, alpha_s_provider
from gluon_ratio import EPPS21Ratio, GluonEPPSProvider
from glauber import SystemSpec, OpticalGlauber
from particle import Particle
from npdf_centrality import compute_df49_by_centrality
from npdf_OO_data import load_OO_dat, build_OO_rpa_grid
import eloss_cronin_centrality as EC
import quenching_fast as QF

from ups_particle import make_bottomonia_system
from prim_band import PrimordialBand
from prim_io import load_pair
from prim_analysis import PrimordialAnalysis

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

SQRTS_NN = 5360.0  # GeV
ENERGY_TAG = "5.36"  # TeV label

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

# Output directories
OUTDIR_BASE = ROOT / "outputs" / "cnm_prim" / "min_bias" / f"OO_{ENERGY_TAG.replace('.','p')}TeV"
OUTDIR_BINNED = OUTDIR_BASE / "binned_data"
OUTDIR_PLOTS = OUTDIR_BASE / "plots"

# Plotting
DPI = 150
ALPHA_BAND = 0.20
# initial placeholder, will be overridden by dynamic limits
Y_LIM_RPA = None  # computed from data
X_LIM_PT = (0, 15)

# Colors (updated per user request and to avoid stacking)
COLORS = {
    'npdf':        '#56B4E9',       # light blue
    'eloss':       '#F4A0A0',       # pink/salmon
    'broad':       '#E69F00',       # orange (distinct from before)
    'eloss_broad': '#404040',       # dark gray
    'cnm':         '#808080',       # gray
    # primordial only bands (lighter shades)
    'prim_NPWLC':  '#FF6347',       # tomato
    'prim_Pert':   '#90EE90',       # light green
    # combined results (darker)
    'total_NPWLC': '#8B0000',       # dark red
    'total_Pert':  '#006400',       # dark green
}

LABELS = {
    "npdf":         "nPDF (EPPS21)",
    "eloss":        "ELoss",
    "broad":        r"$p_T$-Broadening",
    "eloss_broad":  r"ELoss + $p_T$-Broad",
    "cnm":          "Total CNM",
    "prim_NPWLC":   "Primordial (NPWLC)",
    "prim_Pert":    "Primordial (Pert)",
    "total_NPWLC":  "CNM + Primordial (NPWLC)",
    "total_Pert":   "CNM + Primordial (Pert)",
}

# Bottomonia states (differential in primordial)
STATES = ["ups1S", "ups2S", "ups3S"]
STATE_NAMES = {
    "ups1S": r"$\Upsilon(1S)$",
    "ups2S": r"$\Upsilon(2S)$",
    "ups3S": r"$\Upsilon(3S)$",
}

# Kinematic binning
Y_EDGES = np.arange(-5.0, 5.0 + 0.5, 0.5)
P_EDGES = np.arange(0.0, 20.0 + 1.0, 1.0)
PT_RANGE_AVG = (0.0, 15.0)

# Centrality bins (for future use)
CENT_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
MB_C0 = 0.25

# CMS rapidity windows (3 regions)
Y_WINDOWS = [
    (-5.0, -2.4, "backward"),
    (-2.4,  2.4, "midrapidity"),
    ( 2.4,  4.5, "forward"),
]

# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def step_from_centers(centers, values):
    """Convert bin centers to step function edges."""
    centers = np.asarray(centers, float)
    values = np.asarray(values, float)
    if len(centers) < 2:
        hw = 1.0
    else:
        hw = (centers[1] - centers[0]) / 2.0
    edges = np.append(centers - hw, centers[-1] + hw)
    step_vals = np.append(values, values[-1])
    return edges, step_vals

def export_to_csv(filename, df):
    """Export DataFrame to CSV in scientific notation."""
    df.to_csv(filename, index=False, float_format='%.6e')
    print(f"      ✓ {filename.name}")

# ═══════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def build_cnm_context():
    """Build CNM (nPDF + eloss + pT broadening) context for O+O @ 5.36 TeV."""
    print(f"\n  [INFO] Building CNM context for O+O @ {ENERGY_TAG} TeV ...", flush=True)

    sqrt_sNN = SQRTS_NN
    sigma_nn_mb = 68.0

    # Load nPDF grid
    particle = Particle(family="bottomonia", state="avg", mass_override_GeV=9.46)
    epps_ratio = EPPS21Ratio(A=16, path=str(ROOT / "cnm" / "npdf_code" / "nPDFs"))
    gluon = GluonEPPSProvider(
        epps_ratio, sqrt_sNN_GeV=sqrt_sNN, m_state_GeV=10.01, y_sign_for_xA=-1
    ).with_geometry()

    # Glauber geometry
    gl_spec = SystemSpec("AA", sqrt_sNN, A=16, sigma_nn_mb=sigma_nn_mb)
    gl_ana = OpticalGlauber(gl_spec, nx_pa=64, ny_pa=64, verbose=False)

    # Load O+O nPDF data
    OO_DAT = ROOT / "inputs" / "npdf" / "OxygenOxygen5360" / "nPDF_OO.dat"
    data = load_OO_dat(str(OO_DAT))
    grid = build_OO_rpa_grid(data, pt_max=20.0)

    r0 = grid["r_central"].to_numpy()
    M = grid[[f"r_mem_{i:03d}" for i in range(1, 49)]].to_numpy().T
    SA_all = np.vstack([r0[None, :], M])

    df49_by_cent, K_by_cent, _, Y_SHIFT = compute_df49_by_centrality(
        grid, r0, M, gluon, gl_ana,
        cent_bins=CENT_BINS, nb_bsamples=5, kind="AA", SA_all=SA_all
    )

    npdf_ctx = dict(df49_by_cent=df49_by_cent, df_pp=grid, df_pa=grid, gluon=gluon)

    # Quenching params
    alpha_s = alpha_s_provider(mode="running", LambdaQCD=0.25)
    Lmb = gl_ana.leff_minbias_AA()
    device = "cpu"
    qp_base = QF.QuenchParams(
        qhat0=0.075, lp_fm=1.5, LA_fm=Lmb, LB_fm=Lmb,
        system="AA", lambdaQCD=0.25, roots_GeV=sqrt_sNN,
        alpha_of_mu=alpha_s, alpha_scale="mT",
        use_hard_cronin=True, mapping="exp", device=device,
    )

    # Build CNM combiner
    cnm = CNMCombineFast(
        energy=ENERGY_TAG, family="bottomonia", particle_state="avg",
        sqrt_sNN=sqrt_sNN, sigma_nn_mb=sigma_nn_mb,
        cent_bins=CENT_BINS, y_edges=Y_EDGES, p_edges=P_EDGES,
        y_windows=[(y0, y1) for y0, y1, _ in Y_WINDOWS],
        pt_range_avg=PT_RANGE_AVG, pt_floor_w=1.0,
        weight_mode="flat", y_ref=0.0, cent_c0=MB_C0,
        q0_pair=(0.05, 0.09), p0_scale_pair=(0.9, 1.1),
        nb_bsamples=5, y_shift_fraction=1.0,
        particle=particle, gl=gl_ana, qp_base=qp_base, npdf_ctx=npdf_ctx,
        y_sign_for_xA=-1, spec=gl_spec, debug=False, absorption=None
    )

    return cnm


def build_primordial_band(model):
    """Load primordial band for given model (NPWLC or Pert)."""
    print(f"\n  [INFO] Loading Primordial {model} ...", flush=True)

    system = make_bottomonia_system(sqrts_pp_GeV=SQRTS_NN)
    paths = PRIM_INPUTS[model]

    if not paths["lower"].exists():
        print(f"    [WARN] Missing: {paths['lower']}")
        return None
    if not paths["upper"].exists():
        print(f"    [WARN] Missing: {paths['upper']}")
        return None

    df_lo, df_hi = load_pair(str(paths["lower"]), str(paths["upper"]), system, debug=False)
    ana_lo = PrimordialAnalysis(df_lo, system, with_feeddown=True)
    ana_hi = PrimordialAnalysis(df_hi, system, with_feeddown=True)

    band = PrimordialBand(lower=ana_lo, upper=ana_hi, include_run_errors=True)
    print(f"    ✓ Loaded {model} band")
    return band


# ═══════════════════════════════════════════════════════════════════════
# COMBINATION & DATA EXPORT
# ═══════════════════════════════════════════════════════════════════════

def combine_and_export_vs_y(cnm, band_npwlc, band_pert):
    """
    Combine CNM + Primordial vs rapidity for each state.
    Export to CSV and return data for plotting.
    """
    print("\n  [DATA] R_AA vs y ...", flush=True)

    y_bins = list(zip(Y_EDGES[:-1], Y_EDGES[1:]))
    
    # Get CNM values (all components)
    y_centers = (Y_EDGES[:-1] + Y_EDGES[1:]) / 2.0
    yc, tags, dict_cnm = cnm.cnm_vs_y(components=["npdf", "eloss", "broad", "eloss_broad", "cnm"], include_mb=True)
    cnm_c = dict_cnm["cnm"][0]["MB"]
    cnm_lo = dict_cnm["cnm"][1]["MB"]
    cnm_hi = dict_cnm["cnm"][2]["MB"]
    
    # Get individual CNM components
    npdf_c = dict_cnm["npdf"][0]["MB"]
    npdf_lo = dict_cnm["npdf"][1]["MB"]
    npdf_hi = dict_cnm["npdf"][2]["MB"]
    eloss_c = dict_cnm["eloss"][0]["MB"]
    eloss_lo = dict_cnm["eloss"][1]["MB"]
    eloss_hi = dict_cnm["eloss"][2]["MB"]
    broad_c = dict_cnm["broad"][0]["MB"]
    broad_lo = dict_cnm["broad"][1]["MB"]
    broad_hi = dict_cnm["broad"][2]["MB"]
    eloss_broad_c = dict_cnm["eloss_broad"][0]["MB"]
    eloss_broad_lo = dict_cnm["eloss_broad"][1]["MB"]
    eloss_broad_hi = dict_cnm["eloss_broad"][2]["MB"]

    # Get primordial values for each state
    prim_np_c, prim_np_band = band_npwlc.vs_y(PT_RANGE_AVG, y_bins)
    prim_pe_c, prim_pe_band = band_pert.vs_y(PT_RANGE_AVG, y_bins)

    # Store combined results
    combined_data = {}

    # Combine for each state
    for state in STATES:
        p_np_c = prim_np_c[state].to_numpy()
        p_np_lo = prim_np_band[f"{state}_lo"].to_numpy()
        p_np_hi = prim_np_band[f"{state}_hi"].to_numpy()

        p_pe_c = prim_pe_c[state].to_numpy()
        p_pe_lo = prim_pe_band[f"{state}_lo"].to_numpy()
        p_pe_hi = prim_pe_band[f"{state}_hi"].to_numpy()

        # Combine CNM with NPWLC primordial
        tot_np_c, tot_np_lo, tot_np_hi = combine_two_bands_1d(
            cnm_c, cnm_lo, cnm_hi, p_np_c, p_np_lo, p_np_hi
        )

        # Combine CNM with Pert primordial
        tot_pe_c, tot_pe_lo, tot_pe_hi = combine_two_bands_1d(
            cnm_c, cnm_lo, cnm_hi, p_pe_c, p_pe_lo, p_pe_hi
        )

        combined_data[state] = {
            "npdf": (npdf_c, npdf_lo, npdf_hi),
            "eloss": (eloss_c, eloss_lo, eloss_hi),
            "broad": (broad_c, broad_lo, broad_hi),
            "eloss_broad": (eloss_broad_c, eloss_broad_lo, eloss_broad_hi),
            "cnm": (cnm_c, cnm_lo, cnm_hi),
            "prim_NPWLC": (p_np_c, p_np_lo, p_np_hi),
            "prim_Pert": (p_pe_c, p_pe_lo, p_pe_hi),
            "total_NPWLC": (tot_np_c, tot_np_lo, tot_np_hi),
            "total_Pert": (tot_pe_c, tot_pe_lo, tot_pe_hi),
        }

        # Export to CSV
        df_export = pd.DataFrame({
            "y": y_centers,
            "R_npdf": npdf_c,
            "R_npdf_lo": npdf_lo,
            "R_npdf_hi": npdf_hi,
            "R_eloss": eloss_c,
            "R_eloss_lo": eloss_lo,
            "R_eloss_hi": eloss_hi,
            "R_broad": broad_c,
            "R_broad_lo": broad_lo,
            "R_broad_hi": broad_hi,
            "R_eloss_broad": eloss_broad_c,
            "R_eloss_broad_lo": eloss_broad_lo,
            "R_eloss_broad_hi": eloss_broad_hi,
            "R_cnm": cnm_c,
            "R_cnm_lo": cnm_lo,
            "R_cnm_hi": cnm_hi,
            "R_prim_NPWLC": p_np_c,
            "R_prim_NPWLC_lo": p_np_lo,
            "R_prim_NPWLC_hi": p_np_hi,
            "R_prim_Pert": p_pe_c,
            "R_prim_Pert_lo": p_pe_lo,
            "R_prim_Pert_hi": p_pe_hi,
            "R_total_NPWLC": tot_np_c,
            "R_total_NPWLC_lo": tot_np_lo,
            "R_total_NPWLC_hi": tot_np_hi,
            "R_total_Pert": tot_pe_c,
            "R_total_Pert_lo": tot_pe_lo,
            "R_total_Pert_hi": tot_pe_hi,
        })

        fname = OUTDIR_BINNED / f"Upsilon_{state}_RAA_vs_y_MB_OO_{ENERGY_TAG.replace('.','p')}TeV.csv"
        export_to_csv(fname, df_export)

    return y_centers, combined_data


def combine_and_export_vs_pt(cnm, band_npwlc, band_pert):
    """
    Combine CNM + Primordial vs pT for each rapidity window and state.
    Export to CSV and return data for plotting.
    """
    print("\n  [DATA] R_AA vs pT ...", flush=True)

    pt_bins = list(zip(P_EDGES[:-1], P_EDGES[1:]))
    pt_centers = (P_EDGES[:-1] + P_EDGES[1:]) / 2.0

    combined_data = {}

    for y0, y1, y_label in Y_WINDOWS:
        # Get CNM values for this y window (all components)
        pc, tags, dict_cnm = cnm.cnm_vs_pT((y0, y1), components=["npdf", "eloss", "broad", "eloss_broad", "cnm"], include_mb=True)
        cnm_c = dict_cnm["cnm"][0]["MB"]
        cnm_lo = dict_cnm["cnm"][1]["MB"]
        cnm_hi = dict_cnm["cnm"][2]["MB"]
        
        # Get individual CNM components
        npdf_c = dict_cnm["npdf"][0]["MB"]
        npdf_lo = dict_cnm["npdf"][1]["MB"]
        npdf_hi = dict_cnm["npdf"][2]["MB"]
        eloss_c = dict_cnm["eloss"][0]["MB"]
        eloss_lo = dict_cnm["eloss"][1]["MB"]
        eloss_hi = dict_cnm["eloss"][2]["MB"]
        broad_c = dict_cnm["broad"][0]["MB"]
        broad_lo = dict_cnm["broad"][1]["MB"]
        broad_hi = dict_cnm["broad"][2]["MB"]
        eloss_broad_c = dict_cnm["eloss_broad"][0]["MB"]
        eloss_broad_lo = dict_cnm["eloss_broad"][1]["MB"]
        eloss_broad_hi = dict_cnm["eloss_broad"][2]["MB"]

        # Get primordial values
        prim_np_c, prim_np_band = band_npwlc.vs_pt((y0, y1), pt_bins)
        prim_pe_c, prim_pe_band = band_pert.vs_pt((y0, y1), pt_bins)

        y_key = f"{y0:.1f}_to_{y1:.1f}"
        combined_data[y_key] = {}

        # Combine for each state
        for state in STATES:
            p_np_c = prim_np_c[state].to_numpy()
            p_np_lo = prim_np_band[f"{state}_lo"].to_numpy()
            p_np_hi = prim_np_band[f"{state}_hi"].to_numpy()

            p_pe_c = prim_pe_c[state].to_numpy()
            p_pe_lo = prim_pe_band[f"{state}_lo"].to_numpy()
            p_pe_hi = prim_pe_band[f"{state}_hi"].to_numpy()

            # Combine
            tot_np_c, tot_np_lo, tot_np_hi = combine_two_bands_1d(
                cnm_c, cnm_lo, cnm_hi, p_np_c, p_np_lo, p_np_hi
            )
            tot_pe_c, tot_pe_lo, tot_pe_hi = combine_two_bands_1d(
                cnm_c, cnm_lo, cnm_hi, p_pe_c, p_pe_lo, p_pe_hi
            )

            combined_data[y_key][state] = {
                "npdf": (npdf_c, npdf_lo, npdf_hi),
                "eloss": (eloss_c, eloss_lo, eloss_hi),
                "broad": (broad_c, broad_lo, broad_hi),
                "eloss_broad": (eloss_broad_c, eloss_broad_lo, eloss_broad_hi),
                "cnm": (cnm_c, cnm_lo, cnm_hi),
                "prim_NPWLC": (p_np_c, p_np_lo, p_np_hi),
                "prim_Pert": (p_pe_c, p_pe_lo, p_pe_hi),
                "total_NPWLC": (tot_np_c, tot_np_lo, tot_np_hi),
                "total_Pert": (tot_pe_c, tot_pe_lo, tot_pe_hi),
            }

            # Export to CSV
            df_export = pd.DataFrame({
                "pT": pt_centers,
                "R_npdf": npdf_c,
                "R_npdf_lo": npdf_lo,
                "R_npdf_hi": npdf_hi,
                "R_eloss": eloss_c,
                "R_eloss_lo": eloss_lo,
                "R_eloss_hi": eloss_hi,
                "R_broad": broad_c,
                "R_broad_lo": broad_lo,
                "R_broad_hi": broad_hi,
                "R_eloss_broad": eloss_broad_c,
                "R_eloss_broad_lo": eloss_broad_lo,
                "R_eloss_broad_hi": eloss_broad_hi,
                "R_cnm": cnm_c,
                "R_cnm_lo": cnm_lo,
                "R_cnm_hi": cnm_hi,
                "R_prim_NPWLC": p_np_c,
                "R_prim_NPWLC_lo": p_np_lo,
                "R_prim_NPWLC_hi": p_np_hi,
                "R_prim_Pert": p_pe_c,
                "R_prim_Pert_lo": p_pe_lo,
                "R_prim_Pert_hi": p_pe_hi,
                "R_total_NPWLC": tot_np_c,
                "R_total_NPWLC_lo": tot_np_lo,
                "R_total_NPWLC_hi": tot_np_hi,
                "R_total_Pert": tot_pe_c,
                "R_total_Pert_lo": tot_pe_lo,
                "R_total_Pert_hi": tot_pe_hi,
            })

            fname = OUTDIR_BINNED / f"Upsilon_{state}_RAA_vs_pT_{y_label}_MB_OO_{ENERGY_TAG.replace('.','p')}TeV.csv"
            export_to_csv(fname, df_export)

    return pt_centers, combined_data


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_ylim_from_data(data_dict, margin=0.05):
    """Compute y-axis limits from combined_data structure.

    ``data_dict`` has the same form as ``combined_data`` in the script;
    each entry is another dict keyed by component/model and whose value is a
    triplet (central, lo, hi).
    """
    all_vals = []
    for entry in data_dict.values():
        for comp_vals in entry.values():
            arr = np.asarray(comp_vals[0], float)
            all_vals.append(arr)
            # include lo/hi as well in case they extend further
            all_vals.append(np.asarray(comp_vals[1], float))
            all_vals.append(np.asarray(comp_vals[2], float))
    if not all_vals:
        return (0.0, 1.0)
    flat = np.concatenate(all_vals)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return (0.0, 1.0)
    mn = flat.min()
    mx = flat.max()
    rng = mx - mn
    if rng <= 0:
        return (mn - 0.1, mx + 0.1)
    return (max(0.0, mn - margin * rng), mx + margin * rng)


def plot_raa_vs_y(y_centers, combined_data):
    """
    Plot R_AA vs rapidity for each state (3 columns).
    """
    print("\n  [PLOT] R_AA vs y ...", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=DPI, sharey=True)
    plt.subplots_adjust(wspace=0.05)

    sys_note = rf"$\mathbf{{O+O @ \sqrt{{s_{{NN}}}} = {ENERGY_TAG} \, \text{{TeV}}}}$" + "\n" + r"$p_T \in [0, 15]$ GeV"

    for col, state in enumerate(STATES):
        ax = axes[col]
        data = combined_data[state]

        # Plot CNM components
        components_to_plot = ["npdf", "eloss", "broad", "cnm"]
        for comp in components_to_plot:
            if comp in data:
                comp_c, comp_lo, comp_hi = data[comp]
                xe, yc_s = step_from_centers(y_centers, comp_c)
                ls = "--" if comp == "npdf" else "-"
                lw = 2.2 if comp == "cnm" else 1.5
                ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=COLORS[comp], label=LABELS[comp])
                ax.fill_between(xe, *step_from_centers(y_centers, comp_lo)[1:], step="post",
                                 color=COLORS[comp], alpha=ALPHA_BAND, lw=0)

        # Plot primordial-only results (lighter colours to distinguish)
        for key, label in [("prim_NPWLC", LABELS["prim_NPWLC"]),
                           ("prim_Pert", LABELS["prim_Pert"])]:
            prim_c, prim_lo, prim_hi = data[key]
            xe, yc_s = step_from_centers(y_centers, prim_c)
            ax.step(xe, yc_s, where="post", lw=1.8, color=COLORS[key], label=label)
            ax.fill_between(xe, *step_from_centers(y_centers, prim_lo)[1:], step="post",
                            color=COLORS[key], alpha=ALPHA_BAND/1.5, lw=0)

        # Plot combined results
        for key, label in [("total_NPWLC", LABELS["total_NPWLC"]),
                           ("total_Pert", LABELS["total_Pert"])]:
            tot_c, tot_lo, tot_hi = data[key]
            xe, yc_s = step_from_centers(y_centers, tot_c)
            ax.step(xe, yc_s, where="post", lw=2.2, color=COLORS[key], label=label)
            ax.fill_between(xe, *step_from_centers(y_centers, tot_lo)[1:], step="post",
                            color=COLORS[key], alpha=ALPHA_BAND, lw=0)

        ax.axhline(1.0, color="gray", ls=":", lw=1.0)
        ax.text(0.98, 0.98, STATE_NAMES[state], transform=ax.transAxes, ha="right", va="top",
                fontsize=13, fontweight="bold")

        ax.set_xlim(-5.0, 5.0)
        # dynamic y-limits
        if Y_LIM_RPA is None:
            low, high = compute_ylim_from_data({state: combined_data[state]})
            ax.set_ylim(low, high)
        else:
            ax.set_ylim(*Y_LIM_RPA)
        ax.set_xlabel(r"$y$", fontsize=14)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=10)

        if col == 0:
            ax.set_ylabel(r"$R_{AA}^{\Upsilon}$", fontsize=15)
            ax.text(0.02, 0.05, sys_note, transform=ax.transAxes, ha="left", va="bottom", 
                   fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
            ax.legend(loc="lower left", frameon=False, fontsize=9)
        else:
            ax.set_yticklabels([])

    plt.tight_layout()
    return fig


def plot_raa_vs_pt_grid(pt_centers, combined_data):
    """
    Plot R_AA vs pT: 3 rows (rapidity windows) x 3 cols (states).
    """
    print("\n  [PLOT] R_AA vs pT (3x3 grid) ...", flush=True)

    # compute global y-limits over all windows/states
    if Y_LIM_RPA is None:
        # flatten all data
        bigdict = {}
        for ykey, sub in combined_data.items():
            for st, stdata in sub.items():
                bigdict[f"{ykey}_{st}"] = stdata
        ylims_computed = compute_ylim_from_data(bigdict)
        ylims = ylims_computed
    else:
        ylims = Y_LIM_RPA

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=DPI, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.08)

    sys_note = rf"$O+O @ {ENERGY_TAG} \, \text{{TeV}}$"

    for row, (y0, y1, y_label) in enumerate(Y_WINDOWS):
        y_key = f"{y0:.1f}_to_{y1:.1f}"
        y_window_data = combined_data[y_key]

        for col, state in enumerate(STATES):
            ax = axes[row, col]
            data = y_window_data[state]

            # Plot CNM components (only in first column to avoid clutter)
            if col == 0:
                components_to_plot = ["npdf", "cnm"]
                for comp in components_to_plot:
                    if comp in data:
                        comp_c, comp_lo, comp_hi = data[comp]
                        xe, yc_s = step_from_centers(pt_centers, comp_c)
                        ls = "--" if comp == "npdf" else "-"
                        lw = 2.0 if comp == "cnm" else 1.5
                        ax.step(xe, yc_s, where="post", lw=lw, ls=ls, color=COLORS[comp], label=LABELS[comp])
                        ax.fill_between(xe, *step_from_centers(pt_centers, comp_lo)[1:], step="post",
                                       color=COLORS[comp], alpha=ALPHA_BAND, lw=0)

            # Plot primordial (only in first column)
            if col == 0:
                for key, label in [("prim_NPWLC", LABELS["prim_NPWLC"]),
                                   ("prim_Pert", LABELS["prim_Pert"])]:
                    prim_c, prim_lo, prim_hi = data[key]
                    xe, yc_s = step_from_centers(pt_centers, prim_c)
                    ax.step(xe, yc_s, where="post", lw=1.8, color=COLORS[key], label=label)
                    ax.fill_between(xe, *step_from_centers(pt_centers, prim_lo)[1:], step="post",
                                    color=COLORS[key], alpha=ALPHA_BAND, lw=0)

            # Plot combined results
            for key, label in [("total_NPWLC", LABELS["total_NPWLC"]),
                               ("total_Pert", LABELS["total_Pert"])]:
                tot_c, tot_lo, tot_hi = data[key]
                xe, yc_s = step_from_centers(pt_centers, tot_c)
                ax.step(xe, yc_s, where="post", lw=2.0, color=COLORS[key], label=label)
                ax.fill_between(xe, *step_from_centers(pt_centers, tot_lo)[1:], step="post",
                               color=COLORS[key], alpha=ALPHA_BAND, lw=0)

            ax.axhline(1.0, color="gray", ls=":", lw=1.0)

            # Annotations
            ax.text(0.98, 0.98, STATE_NAMES[state], transform=ax.transAxes, 
                   ha="right", va="top", fontsize=11, fontweight="bold")
            ax.text(0.02, 0.98, y_label, transform=ax.transAxes, ha="left", va="top",
                   fontsize=10, color="navy", fontweight="bold")

            ax.set_xlim(*X_LIM_PT)
            ax.set_ylim(*ylims)
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=9)
            ax.label_outer()

            # Legend (only first panel)
            if row == 0 and col == 0:
                ax.legend(loc="upper left", frameon=False, fontsize=8)
                ax.text(0.98, 0.50, sys_note, transform=ax.transAxes, ha="right", va="top", fontsize=10)

    # Global labels
    fig.text(0.5, 0.02, r"$p_T$ (GeV)", ha="center", fontsize=15)
    fig.text(0.02, 0.5, r"$R_{AA}^{\Upsilon}$", va="center", rotation="vertical", fontsize=15)

    return fig


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def run_production():
    """Main production run."""
    print(f"\n{'='*70}")
    print(f"  CNM + Primordial Combination: O+O @ {ENERGY_TAG} TeV")
    print(f"  Bottomonia States: 1S, 2S, 3S")
    print(f"  Primordial Models: NPWLC, Pert")
    print(f"{'='*70}")

    # Setup output directories
    OUTDIR_BASE.mkdir(parents=True, exist_ok=True)
    OUTDIR_BINNED.mkdir(parents=True, exist_ok=True)
    OUTDIR_PLOTS.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output: {OUTDIR_BASE}")

    # Load contexts
    print("\n[1/4] Building contexts ...", flush=True)
    cnm = build_cnm_context()
    band_npwlc = build_primordial_band("NPWLC")
    band_pert = build_primordial_band("Pert")

    if band_npwlc is None or band_pert is None:
        print("\n  [ERROR] Failed to load primordial data. Aborting.")
        return

    # Combine and export data
    print("\n[2/4] Combining CNM + Primordial and exporting data ...", flush=True)
    y_centers, y_combined = combine_and_export_vs_y(cnm, band_npwlc, band_pert)
    pt_centers, pt_combined = combine_and_export_vs_pt(cnm, band_npwlc, band_pert)

    # Generate plots
    print("\n[3/4] Generating plots ...", flush=True)
    fig_y = plot_raa_vs_y(y_centers, y_combined)
    fig_y.savefig(OUTDIR_PLOTS / f"Upsilon_RAA_vs_y_MB_OO_{ENERGY_TAG.replace('.','p')}TeV.pdf",
                  bbox_inches="tight", dpi=DPI)
    fig_y.savefig(OUTDIR_PLOTS / f"Upsilon_RAA_vs_y_MB_OO_{ENERGY_TAG.replace('.','p')}TeV.png",
                  bbox_inches="tight", dpi=DPI)
    plt.close(fig_y)
    print(f"      ✓ R_AA vs y plot saved")

    fig_pt = plot_raa_vs_pt_grid(pt_centers, pt_combined)
    fig_pt.savefig(OUTDIR_PLOTS / f"Upsilon_RAA_vs_pT_Grid_OO_{ENERGY_TAG.replace('.','p')}TeV.pdf",
                   bbox_inches="tight", dpi=DPI)
    fig_pt.savefig(OUTDIR_PLOTS / f"Upsilon_RAA_vs_pT_Grid_OO_{ENERGY_TAG.replace('.','p')}TeV.png",
                   bbox_inches="tight", dpi=DPI)
    plt.close(fig_pt)
    print(f"      ✓ R_AA vs pT grid saved")

    print(f"\n[4/4] Summary", flush=True)
    print(f"  ✓ Binned data:     {len(list(OUTDIR_BINNED.glob('*.csv')))} CSV files")
    print(f"  ✓ Plots:           {len(list(OUTDIR_PLOTS.glob('*.pdf')))} PDF files")
    print(f"\n  ✅ Production complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_production()
