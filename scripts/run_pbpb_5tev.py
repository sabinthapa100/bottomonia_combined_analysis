#!/usr/bin/env python3
"""
run_pbpb_5tev.py  —  PbPb sqrt(s_NN) = 5.023 TeV  QTraj-NLO Production
==========================================================================

Reproduces the Mathematica notebook (raaCalculator-trajectories-qtavg-lhc-5tev.nb)
results exactly.  Theory band from kappa=3 to kappa=4.

Experimental data overlay:
  • CMS 2019  (Y1S, Y2S, Y3S  — Npart and pT)
  • ALICE     (Y1S, Y2S Npart;  Y1S pT)
  • ATLAS     (Y1S, Y2S Npart;  Y1S, Y2S pT)

Produces:
  1.  R_AA vs N_part   – Υ(1S), Υ(2S), Υ(3S)            [all three experiments]
  2.  R_AA vs N_part   – χ_b(1P), χ_b(2P)                [theory predictions only]
  3.  R_AA vs p_T      – Υ(1S), Υ(2S), Υ(3S)            [CMS + ALICE + ATLAS]
  4.  R_AA vs y        – Υ(1S), Υ(2S), Υ(3S)            [theory bands]
  5.  Double ratios vs N_part: Υ(2S)/Υ(1S), Υ(3S)/Υ(1S)
  6.  Double ratios vs p_T:    Υ(2S)/Υ(1S), Υ(3S)/Υ(2S)

Usage:
    python scripts/run_pbpb_5tev.py

Output:  outputs/qtraj_nlo/PbPb5023/
"""

import os, sys, logging
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "hnm", "qtraj-nlo", "qtraj_out_analysis", "src"))

from qtraj_analysis.io import read_whitespace_table, parse_records
from qtraj_analysis.matching import build_observables
from qtraj_analysis.binning import compute_raa_vs_b, compute_raa_vs_pt, compute_raa_vs_y
from qtraj_analysis.feeddown import (
    build_feeddown_matrix, solve_primordial_sigmas, apply_feeddown_to_raa6,
)
from qtraj_analysis.glauber import load_glauber, GlauberInterpolator
from qtraj_analysis.survival_probability import compute_raa_inclusive

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("PbPb5TeV")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (exact values from Mathematica notebook)
# ══════════════════════════════════════════════════════════════════════════════

INPUT_BASE = os.path.join(REPO_ROOT, "inputs", "qtraj_inputs", "PbPb5023")
OUTDIR     = os.path.join(REPO_ROOT, "outputs", "qtraj_nlo", "PbPb5023")

KAPPA_CONFIGS = {
    "k3": {
        "datafile": os.path.join(INPUT_BASE, "lhc3d-k3", "datafile-avg.gz"),
        "label": r"$\kappa = 3$",
    },
    "k4": {
        "datafile": os.path.join(INPUT_BASE, "lhc3d-k4", "datafile-avg.gz"),
        "label": r"$\kappa = 4$",
    },
}

GLAUBER_BVSC = os.path.join(INPUT_BASE, "glauber-data", "bvscData.tsv")
GLAUBER_NBIN = os.path.join(INPUT_BASE, "glauber-data", "nbinvsbData.tsv")

# b-values used in the simulation (from Mathematica notebook)
BVALS = np.array([
    0.0, 2.32326, 4.24791, 6.00746, 7.77937, 9.21228,
    10.4493, 11.5541, 12.5619, 13.4945, 14.3815, 15.6555,
], dtype=np.float64)

# N_part at each b (from Glauber-AA-CMS-Sabin.nb)
NPART_VALS = np.array([
    406.12232235261763, 374.9882778251423, 315.8591222901291,
    243.53796282970384, 168.50283662224285, 112.39007027735359,
    70.7871426138262,   41.05093670919672,  21.292761520545895,
    9.667790836200798,   3.8095328022884103,  0.970598659138994,
], dtype=np.float64)

# Approximate centrality class labels for each b-value
CENTRALITY_LABELS = [
    "0%", "0-5%", "5-10%", "10-20%", "20-30%", "30-40%",
    "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]

# pp inclusive cross-sections (nb, PDG-based, from Mathematica notebook)
# Order: 1S, 2S, chi_b0(1P), chi_b1(1P), chi_b2(1P), 3S, chi_b0(2P), chi_b1(2P), chi_b2(2P)
SIGMAS_EXP = np.array([57.6, 19.0, 3.72, 13.69, 16.1, 6.8, 3.27, 12.0, 14.15],
                       dtype=np.float64)

# State indices in the 9-state basis
IDX_1S, IDX_2S, IDX_3S = 0, 1, 5
IDX_1P0, IDX_1P1, IDX_1P2 = 2, 3, 4
IDX_2P0, IDX_2P1, IDX_2P2 = 6, 7, 8

# pT and rapidity binning
PT_EDGES = np.arange(0.0, 30.0 + 2.0, 2.0)
Y_EDGES  = np.array([-2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4])
Y_WINDOW = (-2.4, 2.4)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL DATA  (extracted from Mathematica plotMaker-new.nb)
# ══════════════════════════════════════════════════════════════════════════════

# ── CMS 2019  (existing TSV files, format: x RAA stat+ stat- syst+ syst-) ──
CMS_DATA_DIR = os.path.join(INPUT_BASE, "data")

# CMS Y3S vs Npart (from newDataNpart[3] in notebook)
CMS_Y3S_NPART = dict(
    npart = np.array([11.43, 42.81, 109.1, 269.1]),
    raa   = np.array([0.534, 0.290, 0.109, 0.051]),
    stat  = np.array([0.194, 0.065, 0.035, 0.019]),
    syst  = np.zeros(4),   # not separately resolved in the notebook
)

# ── ALICE (extracted from ALICEraaC + ALICEraaSys + ALICEraaStat) ──
# ALICEraaSys  = RAA + syst_err  (systematic upper bound)
# ALICEraaStat = RAA + sqrt(stat^2 + syst^2)  (total upper bound)
_alice_raa_c   = np.array([0.8641449, 0.45022938, 0.45482674, 0.4100887, 0.4003135,
                             0.31416014, 0.32513508, 0.3360841, 0.3151856])
_alice_raa_sys = np.array([0.94372195, 0.47251055, 0.47870088, 0.42600524, 0.42577913,
                             0.33962578, 0.33786717, 0.35836527, 0.33110362])
_alice_raa_stat = np.array([1.0344381, 0.5059323, 0.50416505, 0.44351333, 0.42896217,
                              0.34758332, 0.3505993, 0.36632285, 0.34224275])
_alice_syst = _alice_raa_sys - _alice_raa_c
_alice_total = _alice_raa_stat - _alice_raa_c
_alice_stat  = np.sqrt(np.maximum(_alice_total**2 - _alice_syst**2, 0.0))

ALICE_Y1S_NPART = dict(
    npart = np.array([11.762, 42.446, 87.0, 131.4, 189.2, 241.15, 282.78, 333.3, 384.3]),
    raa   = _alice_raa_c,
    stat  = _alice_stat,
    syst  = _alice_syst,
)

# ALICE Y2S vs Npart (2 centrality bins — from ALICEnpartExpt5020Y2Sdata1 output)
ALICE_Y2S_NPART = dict(
    npart = np.array([53.98, 268.98]),
    raa   = np.array([0.2274, 0.1040]),
    stat  = np.array([0.0642, 0.0338]),
    syst  = np.zeros(2),
)

# ALICE Y1S vs pT (from ALICEptExpt5020Y1S: {{pT, RAA}, {stat, syst}})
ALICE_Y1S_PT = dict(
    pt   = np.array([1.0, 3.0, 5.0, 10.5]),
    raa  = np.array([0.37, 0.41, 0.38, 0.45]),
    stat = np.array([0.04, 0.03, 0.04, 0.04]),
    syst = np.array([0.05, 0.03, 0.05, 0.05]),
)

# ── ATLAS (from ATLASpts1s, ATLASpts2s, ATLASpts1spt, ATLASpts2spt) ──
# ATLAS quotes combined stat+syst in these data — no separate breakdown
ATLAS_Y1S_NPART = dict(
    npart = np.array([22.845, 70.259, 131.034, 189.224, 242.672, 285.345, 333.19, 384.483]),
    raa   = np.array([0.8384, 0.6008, 0.5069, 0.3633, 0.3232, 0.2818, 0.3163, 0.2403]),
    stat  = np.array([0.124, 0.073, 0.079, 0.056, 0.051, 0.050, 0.038, 0.041]),
    syst  = np.zeros(8),
)

ATLAS_Y2S_NPART = dict(
    npart = np.array([47.242, 160.552, 264.341]),
    raa   = np.array([0.3612, 0.1527, 0.0881]),
    stat  = np.array([0.096, 0.064, 0.061]),
    syst  = np.zeros(3),
)

ATLAS_Y1S_PT = dict(
    pt   = np.array([1.0, 3.0, 5.0, 7.5, 10.5, 21.0]),
    raa  = np.array([0.3014, 0.2819, 0.3652, 0.3918, 0.3475, 0.3280]),
    stat = np.array([0.056, 0.049, 0.057, 0.061, 0.057, 0.038]),
    syst = np.zeros(6),
)

ATLAS_Y2S_PT = dict(
    pt   = np.array([2.0, 6.5, 19.5]),
    raa  = np.array([0.1170, 0.0957, 0.0904]),
    stat = np.array([0.052, 0.059, 0.049]),
    syst = np.zeros(3),
)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_cms_data():
    """Load CMS 2019 PbPb 5.023 TeV experimental data from TSV files."""
    files = {
        "y1s_npart": "CMS2019-Y1s-npart.tsv",
        "y2s_npart": "CMS2019-Y2s-npart.tsv",
        "y1s_pt":    "CMS2019-Y1s-pt.tsv",
        "y2s_pt":    "CMS2019-Y2s-pt.tsv",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(CMS_DATA_DIR, fname)
        if os.path.exists(path):
            arr = np.loadtxt(path)
            data[key] = arr
            logger.info("Loaded CMS data: %s  (%d points)", fname, arr.shape[0])
        else:
            logger.warning("CMS data not found: %s", path)
    return data


def run_single_kappa(kappa_key, config):
    """
    Full analysis pipeline for one kappa value.

    Returns dict with:
      npart        : (n_b,)
      raa9_incl    : (n_b, 9)   — inclusive R_AA vs N_part
      raa9_sem     : (n_b, 9)
      pt_centers   : (n_pt,)
      raa9_pt      : (n_pt, 9)  — min-bias R_AA vs p_T
      y_centers    : (n_y,)
      raa9_y       : (n_y, 9)   — min-bias R_AA vs y
    """
    logger.info("── %s ──", config["label"])
    datafile = config["datafile"]
    if not os.path.exists(datafile):
        logger.error("Datafile not found: %s", datafile)
        return None

    # 1) Parse avg file → matched trajectory observables
    table   = read_whitespace_table(datafile, logger)
    records = parse_records(table, logger)
    obs     = build_observables(records, logger)

    # 2) Per-b survival averages
    raa_vs_b = compute_raa_vs_b(obs, logger)
    logger.info("b-values found: %s", np.round(raa_vs_b.bvals, 3))

    # 3) Glauber model
    gm      = load_glauber(GLAUBER_BVSC, GLAUBER_NBIN, BVALS, NPART_VALS, logger)
    glauber = GlauberInterpolator(gm)
    nbin    = glauber.b_to_nbin(raa_vs_b.bvals)
    npart   = glauber.b_to_npart(raa_vs_b.bvals)

    # 4) Feeddown + inclusive R_AA vs N_part
    feeddown    = build_feeddown_matrix()
    sigmas_prim = solve_primordial_sigmas(feeddown, SIGMAS_EXP)
    raa9_incl, raa9_sem = compute_raa_inclusive(
        raa_vs_b.raa6_mean, raa_vs_b.raa6_sem, sigmas_prim, nbin, feeddown
    )

    bf_1s = sigmas_prim[0] / SIGMAS_EXP[0]
    logger.info("Υ(1S) direct fraction σ_dir/σ_exp = %.4f  (Mathematica: 0.7468)", bf_1s)

    # 5) R_AA vs p_T  (min-bias, |y| < 2.4)
    pt_centers, raa6_pt, sem6_pt = compute_raa_vs_pt(obs, PT_EDGES, Y_WINDOW, logger)
    raa9_pt = np.full((len(pt_centers), 9), np.nan)
    for i in range(len(pt_centers)):
        if not np.isnan(raa6_pt[i, 0]):
            r9, _ = apply_feeddown_to_raa6(raa6_pt[i], sem6_pt[i], feeddown, sigmas_prim)
            raa9_pt[i] = r9

    # 6) R_AA vs y  (min-bias, all b-values)
    y_centers, raa6_y, sem6_y = compute_raa_vs_y(obs, Y_EDGES, logger)
    raa9_y = np.full((len(y_centers), 9), np.nan)
    for i in range(len(y_centers)):
        if not np.isnan(raa6_y[i, 0]):
            r9, _ = apply_feeddown_to_raa6(raa6_y[i], sem6_y[i], feeddown, sigmas_prim)
            raa9_y[i] = r9

    return dict(
        npart=npart, raa9_incl=raa9_incl, raa9_sem=raa9_sem,
        pt_centers=pt_centers, raa9_pt=raa9_pt,
        y_centers=y_centers,   raa9_y=raa9_y,
    )


def save_csvs(outdir, kappa_key, res):
    os.makedirs(outdir, exist_ok=True)
    hdr9 = "Npart,RAA_1S,RAA_2S,RAA_1P0,RAA_1P1,RAA_1P2,RAA_3S,RAA_2P0,RAA_2P1,RAA_2P2"

    idx = np.argsort(res["npart"])
    np.savetxt(os.path.join(outdir, f"raavsnpart_{kappa_key}.csv"),
               np.column_stack([res["npart"][idx], res["raa9_incl"][idx]]),
               delimiter=",", header=hdr9, comments="", fmt="%.8f")

    pt_hdr = "pt," + hdr9.split(",", 1)[1]
    valid = ~np.isnan(res["raa9_pt"][:, 0])
    np.savetxt(os.path.join(outdir, f"raavspt_mb_{kappa_key}.csv"),
               np.column_stack([res["pt_centers"][valid], res["raa9_pt"][valid]]),
               delimiter=",", header=pt_hdr, comments="", fmt="%.8f")

    y_hdr = "y," + hdr9.split(",", 1)[1]
    valid_y = ~np.isnan(res["raa9_y"][:, 0])
    np.savetxt(os.path.join(outdir, f"raavsy_mb_{kappa_key}.csv"),
               np.column_stack([res["y_centers"][valid_y], res["raa9_y"][valid_y]]),
               delimiter=",", header=y_hdr, comments="", fmt="%.8f")

    logger.info("Saved CSVs for %s to %s", kappa_key, outdir)


# ══════════════════════════════════════════════════════════════════════════════
# THESIS-QUALITY PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

_THESIS_RC = {
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
    "axes.unicode_minus":  False,
    "font.size":           13,
    "axes.labelsize":      15,
    "xtick.labelsize":     12,
    "ytick.labelsize":     12,
    "legend.fontsize":     10,
    "figure.dpi":          150,
    "axes.grid":           False,
    "axes.linewidth":      1.0,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
}

# Theory band colors
_BAND_COLOR  = "#1f77b4"
_BAND_ALPHA  = 0.25
_BAND_LW     = 1.8

# Experiment styles
_EXP_STYLES = {
    "CMS":   dict(color="black",    marker="s",  ms=4.5, label="CMS (2019)"),
    "ALICE": dict(color="#d62728",  marker="o",  ms=4.5, label="ALICE"),
    "ATLAS": dict(color="#2ca02c",  marker="^",  ms=4.5, label="ATLAS"),
}
_SYST_ALPHA = 0.20

# State labels
_STATE_TEX = {
    IDX_1S:  r"$\Upsilon(1S)$",
    IDX_2S:  r"$\Upsilon(2S)$",
    IDX_3S:  r"$\Upsilon(3S)$",
    IDX_1P1: r"$\chi_b(1P)$",
    IDX_2P1: r"$\chi_b(2P)$",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _draw_exp(ax, x, y, stat, syst, style, label=None, box_width=None):
    """Draw experiment error bars (stat) + systematic boxes."""
    from matplotlib.patches import Rectangle
    lbl = label if label is not None else style.get("label", "")
    ax.errorbar(x, y,
                yerr=stat,
                fmt=style["marker"], ms=style["ms"],
                color=style["color"], elinewidth=1.2,
                capsize=3, capthick=1.2,
                label=lbl, zorder=10)
    if np.any(syst > 0):
        w = box_width if box_width else max(np.diff(x).min() * 0.15, 3.0) if len(x) > 1 else 5.0
        for xi, yi, su in zip(x, y, syst):
            rect = Rectangle((xi - w, yi - su), 2*w, 2*su,
                              linewidth=0.7, edgecolor=style["color"],
                              facecolor=style["color"], alpha=_SYST_ALPHA, zorder=9)
            ax.add_patch(rect)


def _theory_band(ax, x3, y3, x4, y4, label=None):
    """Fill kappa=3..4 band on a common grid; also draw mid-line."""
    xmin = max(x3.min(), x4.min())
    xmax = min(x3.max(), x4.max())
    xg = np.linspace(xmin, xmax, 400)
    lo  = np.minimum(np.interp(xg, x3, y3), np.interp(xg, x4, y4))
    hi  = np.maximum(np.interp(xg, x3, y3), np.interp(xg, x4, y4))
    mid = 0.5 * (lo + hi)
    ax.fill_between(xg, lo, hi, alpha=_BAND_ALPHA, color=_BAND_COLOR, zorder=2,
                    label=label)
    ax.plot(xg, mid, "-", color=_BAND_COLOR, lw=_BAND_LW, zorder=3)


def _theory_step_band(ax, edges, y3, y4, label=None):
    """Step-style theory band for binned observables (pT, y)."""
    lo  = np.minimum(y3, y4)
    hi  = np.maximum(y3, y4)
    mid = 0.5 * (lo + hi)
    # Extend for step plot
    lo2  = np.append(lo,  lo[-1])
    hi2  = np.append(hi,  hi[-1])
    mid2 = np.append(mid, mid[-1])
    ax.fill_between(edges, lo2, hi2, alpha=_BAND_ALPHA, color=_BAND_COLOR,
                    step="post", zorder=2, label=label)
    ax.step(edges, mid2, color=_BAND_COLOR, lw=_BAND_LW, where="post", zorder=3)


def _finish_ax(ax, xlabel, xlim, ylim=(0.0, 1.25)):
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(1.0, lw=0.7, ls="--", color="gray", zorder=1)


def _save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    for ext in (".pdf", ".png"):
        path = os.path.join(outdir, stem + ext)
        dpi = 300 if ext == ".png" else None
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        logger.info("Saved: %s", path)


# ─── 1)  R_AA vs N_part  (1S | 2S | 3S  with all experiments) ───────────────

def plot_raa_vs_npart(results, cms_data, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_THESIS_RC)
    r3, r4 = results.get("k3"), results.get("k4")

    states = [
        (IDX_1S, "CMS",   cms_data.get("y1s_npart"), "ALICE", ALICE_Y1S_NPART, "ATLAS", ATLAS_Y1S_NPART),
        (IDX_2S, "CMS",   cms_data.get("y2s_npart"), "ALICE", ALICE_Y2S_NPART, "ATLAS", ATLAS_Y2S_NPART),
        (IDX_3S, "CMS",   None,                       "ALICE", None,             "ATLAS", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    plt.subplots_adjust(wspace=0.04)

    for col, (sidx, _, cms_arr, _, alice_d, _, atlas_d) in enumerate(states):
        ax = axes[col]
        first = (col == 0)

        # Theory band
        if r3 and r4:
            o3, o4 = np.argsort(r3["npart"]), np.argsort(r4["npart"])
            _theory_band(ax, r3["npart"][o3], r3["raa9_incl"][o3, sidx],
                         r4["npart"][o4], r4["raa9_incl"][o4, sidx],
                         label=r"QTraj NLO $\kappa\!\in\![3,4]$" if first else None)

        # CMS TSV data (Y1S, Y2S)
        if cms_arr is not None:
            d = cms_arr[:, :6]
            _draw_exp(ax, d[:, 0], d[:, 1],
                      0.5*(np.abs(d[:, 2]) + np.abs(d[:, 3])),
                      0.5*(np.abs(d[:, 4]) + np.abs(d[:, 5])),
                      _EXP_STYLES["CMS"], label="CMS (2019)" if first else "_")
        elif sidx == IDX_3S:
            d = CMS_Y3S_NPART
            _draw_exp(ax, d["npart"], d["raa"], d["stat"], d["syst"],
                      _EXP_STYLES["CMS"], label="CMS (2019)" if first else "_")

        # ALICE
        if alice_d is not None:
            _draw_exp(ax, alice_d["npart"], alice_d["raa"],
                      alice_d["stat"], alice_d["syst"],
                      _EXP_STYLES["ALICE"], label="ALICE" if first else "_")

        # ATLAS
        if atlas_d is not None:
            _draw_exp(ax, atlas_d["npart"], atlas_d["raa"],
                      atlas_d["stat"], atlas_d["syst"],
                      _EXP_STYLES["ATLAS"], label="ATLAS" if first else "_")

        ax.text(0.97, 0.95, _STATE_TEX[sidx], transform=ax.transAxes,
                ha="right", va="top", fontsize=14)
        _finish_ax(ax, r"$\langle N_\mathrm{part}\rangle$", (0, 425))

    axes[0].set_ylabel(r"$R_{AA}$", labelpad=2)
    axes[0].legend(loc="lower left", framealpha=0.9, edgecolor="none", fontsize=9)

    _save(fig, outdir, "raa_vs_npart_pbpb5tev")
    plt.close(fig)


# ─── 2)  R_AA vs N_part  (chi_b predictions) ─────────────────────────────────

def plot_raa_vs_npart_chib(results, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_THESIS_RC)
    r3, r4 = results.get("k3"), results.get("k4")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0), sharey=True)
    plt.subplots_adjust(wspace=0.04)

    for col, sidx in enumerate([IDX_1P1, IDX_2P1]):
        ax = axes[col]
        if r3 and r4:
            o3, o4 = np.argsort(r3["npart"]), np.argsort(r4["npart"])
            _theory_band(ax, r3["npart"][o3], r3["raa9_incl"][o3, sidx],
                         r4["npart"][o4], r4["raa9_incl"][o4, sidx],
                         label=r"QTraj NLO $\kappa\!\in\![3,4]$" if col == 0 else None)
        ax.text(0.97, 0.95, _STATE_TEX[sidx], transform=ax.transAxes,
                ha="right", va="top", fontsize=14)
        _finish_ax(ax, r"$\langle N_\mathrm{part}\rangle$", (0, 425))

    axes[0].set_ylabel(r"$R_{AA}$", labelpad=2)
    axes[0].legend(loc="lower left", framealpha=0.9, edgecolor="none", fontsize=9)
    _save(fig, outdir, "raa_vs_npart_chib_pbpb5tev")
    plt.close(fig)


# ─── 3)  R_AA vs p_T  (1S | 2S | 3S, step bands + experiments) ──────────────

def plot_raa_vs_pt(results, cms_data, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_THESIS_RC)
    r3, r4 = results.get("k3"), results.get("k4")

    states = [
        (IDX_1S, cms_data.get("y1s_pt"), ALICE_Y1S_PT, ATLAS_Y1S_PT),
        (IDX_2S, cms_data.get("y2s_pt"), None,          ATLAS_Y2S_PT),
        (IDX_3S, None,                    None,          None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    plt.subplots_adjust(wspace=0.04)

    for col, (sidx, cms_arr, alice_d, atlas_d) in enumerate(states):
        ax = axes[col]
        first = (col == 0)

        # Theory step band
        if r3 and r4:
            v3 = ~np.isnan(r3["raa9_pt"][:, sidx])
            v4 = ~np.isnan(r4["raa9_pt"][:, sidx])
            if v3.sum() > 0 and v4.sum() > 0:
                # Align on common valid edges
                n = min(v3.sum(), v4.sum())
                edges = PT_EDGES[:n+1]
                y3 = r3["raa9_pt"][v3, sidx][:n]
                y4 = r4["raa9_pt"][v4, sidx][:n]
                _theory_step_band(ax, edges, y3, y4,
                                  label=r"QTraj NLO $\kappa\!\in\![3,4]$" if first else None)

        # CMS
        if cms_arr is not None:
            d = cms_arr[:, :6]
            _draw_exp(ax, d[:, 0], d[:, 1],
                      0.5*(np.abs(d[:, 2]) + np.abs(d[:, 3])),
                      0.5*(np.abs(d[:, 4]) + np.abs(d[:, 5])),
                      _EXP_STYLES["CMS"], label="CMS (2019)" if first else "_")

        # ALICE
        if alice_d is not None:
            _draw_exp(ax, alice_d["pt"], alice_d["raa"],
                      alice_d["stat"], alice_d["syst"],
                      _EXP_STYLES["ALICE"], label="ALICE" if first else "_")

        # ATLAS
        if atlas_d is not None:
            _draw_exp(ax, atlas_d["pt"], atlas_d["raa"],
                      atlas_d["stat"], atlas_d["syst"],
                      _EXP_STYLES["ATLAS"], label="ATLAS" if first else "_")

        ax.text(0.97, 0.95, _STATE_TEX[sidx], transform=ax.transAxes,
                ha="right", va="top", fontsize=14)
        _finish_ax(ax, r"$p_T\ [\mathrm{GeV}]$", (0, 28))

    axes[0].set_ylabel(r"$R_{AA}$", labelpad=2)
    axes[0].legend(loc="lower right", framealpha=0.9, edgecolor="none", fontsize=9)
    _save(fig, outdir, "raa_vs_pt_pbpb5tev")
    plt.close(fig)


# ─── 4)  R_AA vs y  (1S | 2S | 3S, theory bands) ────────────────────────────

def plot_raa_vs_y(results, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_THESIS_RC)
    r3, r4 = results.get("k3"), results.get("k4")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    plt.subplots_adjust(wspace=0.04)

    for col, sidx in enumerate([IDX_1S, IDX_2S, IDX_3S]):
        ax = axes[col]
        first = (col == 0)

        if r3 and r4:
            v3 = ~np.isnan(r3["raa9_y"][:, sidx])
            v4 = ~np.isnan(r4["raa9_y"][:, sidx])
            if v3.sum() > 0 and v4.sum() > 0:
                n = min(v3.sum(), v4.sum())
                edges = Y_EDGES[:n+1]
                y3 = r3["raa9_y"][v3, sidx][:n]
                y4 = r4["raa9_y"][v4, sidx][:n]
                _theory_step_band(ax, edges, y3, y4,
                                  label=r"QTraj NLO $\kappa\!\in\![3,4]$" if first else None)

        ax.text(0.97, 0.95, _STATE_TEX[sidx], transform=ax.transAxes,
                ha="right", va="top", fontsize=14)
        _finish_ax(ax, r"$y$", (-2.6, 2.6))

    axes[0].set_ylabel(r"$R_{AA}$", labelpad=2)
    axes[0].legend(loc="lower center", framealpha=0.9, edgecolor="none", fontsize=9)
    _save(fig, outdir, "raa_vs_y_pbpb5tev")
    plt.close(fig)


# ─── 5)  Double ratios vs N_part ─────────────────────────────────────────────

def plot_double_ratio_npart(results, cms_data, outdir):
    """Υ(2S)/Υ(1S) and Υ(3S)/Υ(1S) vs N_part."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_THESIS_RC)
    r3, r4 = results.get("k3"), results.get("k4")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0), sharey=True)
    plt.subplots_adjust(wspace=0.04)

    combos = [
        (IDX_2S, IDX_1S, r"$R_{AA}[\Upsilon(2S)]/R_{AA}[\Upsilon(1S)]$",
         "y2s_npart", "y1s_npart"),
        (IDX_3S, IDX_1S, r"$R_{AA}[\Upsilon(3S)]/R_{AA}[\Upsilon(1S)]$",
         None, "y1s_npart"),
    ]

    for col, (num_idx, den_idx, ylabel, cms_num_key, cms_den_key) in enumerate(combos):
        ax = axes[col]
        first = (col == 0)

        # Theory band
        if r3 and r4:
            for res, ls in [(r3, "-"), (r4, "--")]:
                o = np.argsort(res["npart"])
                raa_num = res["raa9_incl"][o, num_idx]
                raa_den = res["raa9_incl"][o, den_idx]
                valid = raa_den > 0
                ratio = np.divide(raa_num, raa_den, where=valid,
                                  out=np.full_like(raa_num, np.nan))

            o3 = np.argsort(r3["npart"]); o4 = np.argsort(r4["npart"])
            rat3 = np.where(r3["raa9_incl"][o3, den_idx] > 0,
                            r3["raa9_incl"][o3, num_idx] / r3["raa9_incl"][o3, den_idx], np.nan)
            rat4 = np.where(r4["raa9_incl"][o4, den_idx] > 0,
                            r4["raa9_incl"][o4, num_idx] / r4["raa9_incl"][o4, den_idx], np.nan)
            valid3 = ~np.isnan(rat3); valid4 = ~np.isnan(rat4)
            if valid3.sum() > 1 and valid4.sum() > 1:
                _theory_band(ax, r3["npart"][o3][valid3], rat3[valid3],
                             r4["npart"][o4][valid4], rat4[valid4],
                             label=r"QTraj NLO $\kappa\!\in\![3,4]$" if first else None)

        # CMS data ratio (from TSV)
        if cms_num_key and cms_num_key in cms_data and cms_den_key in cms_data:
            dn = cms_data[cms_num_key][:, :6]
            dd = cms_data[cms_den_key][:, :6]
            # Match by Npart (they may have different lengths — interpolate denominator)
            npart_n = dn[:, 0]; raa_n = dn[:, 1]; stat_n = np.abs(dn[:, 2])
            npart_d = dd[:, 0]; raa_d = dd[:, 1]; stat_d = np.abs(dd[:, 2])
            # Use common Npart points
            common = np.intersect1d(np.round(npart_n, 0), np.round(npart_d, 0))
            if len(common) >= 2:
                mask_n = np.isin(np.round(npart_n, 0), common)
                mask_d = np.isin(np.round(npart_d, 0), common)
                r_n = raa_n[mask_n]; r_d = raa_d[mask_d]
                s_n = stat_n[mask_n]; s_d = stat_d[mask_d]
                ratio_cms = np.where(r_d > 0, r_n / r_d, np.nan)
                # Error propagation: δ(A/B) ≈ A/B * sqrt((δA/A)^2 + (δB/B)^2)
                ratio_err = np.abs(ratio_cms) * np.sqrt(
                    np.where(r_n > 0, (s_n/r_n)**2, 0) +
                    np.where(r_d > 0, (s_d/r_d)**2, 0))
                _draw_exp(ax, npart_n[mask_n], ratio_cms, ratio_err, np.zeros_like(ratio_err),
                          _EXP_STYLES["CMS"], label="CMS (2019)" if first else "_")

        ax.text(0.97, 0.95, ylabel, transform=ax.transAxes,
                ha="right", va="top", fontsize=11)
        _finish_ax(ax, r"$\langle N_\mathrm{part}\rangle$", (0, 425), ylim=(0.0, 1.1))

    axes[0].set_ylabel("Double ratio", labelpad=2)
    axes[0].legend(loc="upper right", framealpha=0.9, edgecolor="none", fontsize=9)
    _save(fig, outdir, "double_ratio_npart_pbpb5tev")
    plt.close(fig)


# ─── 6)  Double ratios vs p_T ─────────────────────────────────────────────────

def plot_double_ratio_pt(results, cms_data, outdir):
    """Υ(2S)/Υ(1S) and Υ(3S)/Υ(2S) vs p_T."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_THESIS_RC)
    r3, r4 = results.get("k3"), results.get("k4")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0), sharey=True)
    plt.subplots_adjust(wspace=0.04)

    combos = [
        (IDX_2S, IDX_1S, r"$R_{AA}[\Upsilon(2S)]/R_{AA}[\Upsilon(1S)]$",
         "y2s_pt", "y1s_pt"),
        (IDX_3S, IDX_2S, r"$R_{AA}[\Upsilon(3S)]/R_{AA}[\Upsilon(2S)]$",
         None, None),
    ]

    for col, (num_idx, den_idx, ylabel, cms_num_key, cms_den_key) in enumerate(combos):
        ax = axes[col]
        first = (col == 0)

        # Theory step band
        if r3 and r4:
            vn3 = ~np.isnan(r3["raa9_pt"][:, num_idx])
            vd3 = ~np.isnan(r3["raa9_pt"][:, den_idx])
            vn4 = ~np.isnan(r4["raa9_pt"][:, num_idx])
            vd4 = ~np.isnan(r4["raa9_pt"][:, den_idx])
            v3 = vn3 & vd3; v4 = vn4 & vd4
            if v3.sum() > 0 and v4.sum() > 0:
                n = min(v3.sum(), v4.sum())
                edges = PT_EDGES[:n+1]
                rat3 = np.where(r3["raa9_pt"][v3, den_idx] > 0,
                                r3["raa9_pt"][v3, num_idx] / r3["raa9_pt"][v3, den_idx],
                                np.nan)[:n]
                rat4 = np.where(r4["raa9_pt"][v4, den_idx] > 0,
                                r4["raa9_pt"][v4, num_idx] / r4["raa9_pt"][v4, den_idx],
                                np.nan)[:n]
                valid = ~(np.isnan(rat3) | np.isnan(rat4))
                if valid.sum() > 0:
                    _theory_step_band(ax, edges[:valid.sum()+1],
                                      rat3[valid], rat4[valid],
                                      label=r"QTraj NLO $\kappa\!\in\![3,4]$" if first else None)

        # CMS data ratio
        if cms_num_key and cms_num_key in cms_data and cms_den_key in cms_data:
            dn = cms_data[cms_num_key][:, :6]
            dd = cms_data[cms_den_key][:, :6]
            nmin = min(len(dn), len(dd))
            r_n = dn[:nmin, 1]; r_d = dd[:nmin, 1]
            s_n = np.abs(dn[:nmin, 2]); s_d = np.abs(dd[:nmin, 2])
            pt_c = dn[:nmin, 0]
            ratio_cms = np.where(r_d > 0, r_n / r_d, np.nan)
            ratio_err = np.abs(ratio_cms) * np.sqrt(
                np.where(r_n > 0, (s_n/r_n)**2, 0) +
                np.where(r_d > 0, (s_d/r_d)**2, 0))
            _draw_exp(ax, pt_c, ratio_cms, ratio_err, np.zeros_like(ratio_err),
                      _EXP_STYLES["CMS"], label="CMS (2019)" if first else "_")

        ax.text(0.97, 0.95, ylabel, transform=ax.transAxes,
                ha="right", va="top", fontsize=11)
        _finish_ax(ax, r"$p_T\ [\mathrm{GeV}]$", (0, 28), ylim=(0.0, 1.1))

    axes[0].set_ylabel("Double ratio", labelpad=2)
    axes[0].legend(loc="upper right", framealpha=0.9, edgecolor="none", fontsize=9)
    _save(fig, outdir, "double_ratio_pt_pbpb5tev")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load CMS data from TSV files
    cms_data = load_cms_data()

    # Run analysis for each kappa
    results = {}
    for kappa_key, config in KAPPA_CONFIGS.items():
        res = run_single_kappa(kappa_key, config)
        if res is not None:
            results[kappa_key] = res
            save_csvs(OUTDIR, kappa_key, res)

    if not results:
        logger.error("No results — check datafile paths.")
        return

    # Generate all plots
    logger.info("Generating plots...")
    plot_raa_vs_npart(results, cms_data, OUTDIR)
    plot_raa_vs_npart_chib(results, OUTDIR)
    plot_raa_vs_pt(results, cms_data, OUTDIR)
    plot_raa_vs_y(results, OUTDIR)
    plot_double_ratio_npart(results, cms_data, OUTDIR)
    plot_double_ratio_pt(results, cms_data, OUTDIR)

    logger.info("All done — outputs in %s", OUTDIR)


if __name__ == "__main__":
    main()
