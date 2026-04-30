#!/usr/bin/env python3
"""
run_oo_5p36tev.py  —  O+O sqrt(s_NN) = 5.36 TeV  QTraj-NLO Production
========================================================================

Physics fix for wReg data:
  processEvents.py uses array[:-2] which drops column d6 (primordial
  singlet survival) from the 8-column raw trajectory format.  This script
  reads the raw datafile.gz and adds d6 back to the correct surv6 slot:
    L=0  → surv6[0] += mean(d6_prim)     (total 1S = regen + primordial)
    L=1  → surv6[2] += mean(d6_prim)     (total 1P = regen + primordial)

  Without this fix: wReg RAA < noReg (physically impossible).
  With this fix:    wReg RAA > noReg (regeneration enhances survival). ✓

Output: outputs/qtraj_nlo/OO5p36TeV/
"""

import os, sys, gzip, logging
import numpy as np
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "hnm", "qtraj-nlo", "qtraj_out_analysis"))

from qtraj_analysis.schema import Record, TrajectoryObs
from qtraj_analysis.io import read_whitespace_table, parse_records
from qtraj_analysis.matching import build_observables
from qtraj_analysis.binning import compute_raa_vs_pt, compute_raa_vs_y
from qtraj_analysis.feeddown import (
    build_feeddown_matrix, solve_primordial_sigmas, apply_feeddown_to_raa6,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("OO5p36")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

INPUT_BASE = os.path.join(REPO_ROOT, "inputs", "qtraj_inputs", "OxygenOxygen5360")
OUTDIR     = os.path.join(REPO_ROOT, "outputs", "qtraj_nlo", "OO5p36TeV")

NOREG_FILE = os.path.join(INPUT_BASE, "qtraj_nlo_run1_OO_5.36_kap6_noReg", "datafile_partial.gz")
WREG_RAW   = os.path.join(INPUT_BASE, "qtraj-nlo-run2-00-5.36-kap6-wReg",  "datafile.gz")

# pp cross sections (nb) at √s = 5 TeV — used as baseline for both 5.02 and 5.36 TeV
# Order: 1S, 2S, chi_b0(1P), chi_b1(1P), chi_b2(1P), 3S, chi_b0(2P), chi_b1(2P), chi_b2(2P)
SIGMAS_EXP = np.array([57.6, 19.0, 3.72, 13.69, 16.1, 6.8, 3.27, 12.0, 14.15],
                       dtype=np.float64)

# Binning
PT_EDGES = np.array([0, 2, 4, 6, 8, 12, 16, 20], dtype=np.float64)
Y_EDGES  = np.arange(-2.4, 2.4 + 0.4, 0.4)
Y_WINDOW = (-2.4, 2.4)

# State indices (9-state basis)
IDX_1S = 0
IDX_2S = 1
IDX_3S = 5


# ══════════════════════════════════════════════════════════════════════════════
# WREG RAW FILE READER  (the core fix)
# ══════════════════════════════════════════════════════════════════════════════

def read_wreg_raw_with_primordial(rawfile: str) -> list:
    """
    Read raw wReg datafile.gz; group quantum trajectories per physical trajectory;
    average all columns; apply primordial (d6) correction to surv6.

    Raw 8-column format per quantum trajectory:
      L=0:  [d0_regen_1S, d1_regen_2S, d2, d3_regen_3S, d4, d5, d6_prim_1S, L=0]
      L=1:  [d0, d1, d2_regen_1P, d3, d4_regen_2P, d5, d6_prim_1P, L=1]

    processEvents.py drops column 6 (d6_prim) via [:-2], losing the primordial piece.

    Fix applied:
      L=0 → surv6[0] = mean(d0_regen) + mean(d6_prim)   [total 1S]
      L=1 → surv6[2] = mean(d2_regen) + mean(d6_prim)   [total 1P]
    """
    groups: dict = defaultdict(list)

    with gzip.open(rawfile, "rt") as f:
        for line in f:
            meta_str = line.rstrip("\n")
            data     = list(map(float, next(f).split()))
            L        = int(data[-1])
            groups[(meta_str, L)].append(data)

    logger.info("wReg raw: %d unique (meta, L) physical trajectories", len(groups))

    recs = []
    for (meta_str, L), rows in groups.items():
        arr = np.array(rows, dtype=np.float64)  # shape (N_qtraj, 8)
        m   = np.mean(arr, axis=0)              # mean over quantum trajectories

        if L == 0:
            # total 1S = regenerated (m[0]) + primordial (m[6])
            six = np.array([m[0] + m[6], m[1], m[2], m[3], m[4], m[5]], dtype=np.float64)
        else:
            # total 1P = regenerated (m[2]) + primordial (m[6])
            six = np.array([m[0], m[1], m[2] + m[6], m[3], m[4], m[5]], dtype=np.float64)

        meta_arr = np.array(list(map(float, meta_str.split())), dtype=np.float64)
        qweight  = float(arr.shape[0])
        vec      = np.concatenate([six, [float(L), qweight]])
        recs.append(Record(meta=meta_arr, vec=vec))

    logger.info("wReg: built %d corrected records", len(recs))
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_oo_analysis(label: str, obs: list) -> dict:
    """
    Compute inclusive R_AA vs pT and vs y from TrajectoryObs list.

    All at fixed b=4.49691 fm (the only simulated impact parameter in OO).
    Nbin appears in numerator and denominator and cancels in the RAA ratio,
    so we work directly with the feeddown-weighted average of surv9.
    """
    feeddown    = build_feeddown_matrix()
    sigmas_prim = solve_primordial_sigmas(feeddown, SIGMAS_EXP)

    # ── RAA vs pT  (|y| < 2.4) ──────────────────────────────────────────────
    pt_centers, raa6_pt, sem6_pt = compute_raa_vs_pt(obs, PT_EDGES, Y_WINDOW, logger)
    raa9_pt  = np.full((len(pt_centers), 9), np.nan)
    err9_pt  = np.full((len(pt_centers), 9), np.nan)
    for i in range(len(pt_centers)):
        if not np.isnan(raa6_pt[i, 0]):
            r9, e9 = apply_feeddown_to_raa6(raa6_pt[i], sem6_pt[i], feeddown, sigmas_prim)
            raa9_pt[i] = r9
            err9_pt[i] = e9

    # ── RAA vs y ────────────────────────────────────────────────────────────
    y_centers, raa6_y, sem6_y = compute_raa_vs_y(obs, Y_EDGES, logger)
    raa9_y = np.full((len(y_centers), 9), np.nan)
    for i in range(len(y_centers)):
        if not np.isnan(raa6_y[i, 0]):
            r9, _ = apply_feeddown_to_raa6(raa6_y[i], sem6_y[i], feeddown, sigmas_prim)
            raa9_y[i] = r9

    # ── Integrated (all trajectories, |y|<2.4) ─────────────────────────────
    obs_mid = [o for o in obs if Y_WINDOW[0] <= o.y <= Y_WINDOW[1]]
    all_surv6 = np.vstack([o.surv6 for o in obs_mid])
    mean6 = np.mean(all_surv6, axis=0)
    sem6  = np.std(all_surv6, axis=0) / np.sqrt(len(obs_mid))
    raa9_int, err9_int = apply_feeddown_to_raa6(mean6, sem6, feeddown, sigmas_prim)

    logger.info(
        "%s  |y|<2.4  integrated RAA: 1S=%.4f  2S=%.4f  3S=%.4f",
        label, raa9_int[IDX_1S], raa9_int[IDX_2S], raa9_int[IDX_3S]
    )

    return dict(
        pt_centers=pt_centers, raa9_pt=raa9_pt, err9_pt=err9_pt,
        y_centers=y_centers,   raa9_y=raa9_y,
        raa9_int=raa9_int,     err9_int=err9_int,
    )


def save_csvs(outdir: str, label: str, res: dict):
    os.makedirs(outdir, exist_ok=True)
    hdr9 = "pt,RAA_1S,RAA_2S,RAA_1P0,RAA_1P1,RAA_1P2,RAA_3S,RAA_2P0,RAA_2P1,RAA_2P2"
    valid = ~np.isnan(res["raa9_pt"][:, 0])
    np.savetxt(
        os.path.join(outdir, f"raavspt_OO_5p36TeV_{label}.csv"),
        np.column_stack([res["pt_centers"][valid], res["raa9_pt"][valid]]),
        delimiter=",", header=hdr9, comments="", fmt="%.8f",
    )
    hdr_y = hdr9.replace("pt,", "y,")
    valid_y = ~np.isnan(res["raa9_y"][:, 0])
    np.savetxt(
        os.path.join(outdir, f"raavsy_OO_5p36TeV_{label}.csv"),
        np.column_stack([res["y_centers"][valid_y], res["raa9_y"][valid_y]]),
        delimiter=",", header=hdr_y, comments="", fmt="%.8f",
    )
    logger.info("Saved CSVs for %s", label)


# ══════════════════════════════════════════════════════════════════════════════
# THESIS-QUALITY PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

_THESIS_RC = {
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
    "axes.unicode_minus":  False,   # avoid glyph-missing warning for minus signs
    "font.size":           14,
    "axes.labelsize":      16,
    "xtick.labelsize":     13,
    "ytick.labelsize":     13,
    "legend.fontsize":     11,
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

_COL_WREG  = "#1f77b4"   # blue  (with regeneration)
_COL_NOREG = "#d62728"   # red   (no regeneration)
_STATE_TEX = {IDX_1S: r"$\Upsilon(1S)$",
              IDX_2S: r"$\Upsilon(2S)$",
              IDX_3S: r"$\Upsilon(3S)$"}


def _finish_ax(ax, xlabel, xlim, ylim=(0.0, 1.6)):
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(1.0, lw=0.7, ls="--", color="gray", zorder=1)


def _save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    for ext in (".pdf", ".png"):
        p = os.path.join(outdir, stem + ext)
        fig.savefig(p, bbox_inches="tight", dpi=300 if ext == ".png" else None)
        logger.info("Saved: %s", p)


# ─── RAA vs pT ────────────────────────────────────────────────────────────────

def plot_raa_vs_pt(res_w, res_n, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_THESIS_RC)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    plt.subplots_adjust(wspace=0.05)

    for col, sidx in enumerate([IDX_1S, IDX_2S, IDX_3S]):
        ax = axes[col]

        for res, color, label in [
            (res_w, _COL_WREG,  "wReg (regen ON)"),
            (res_n, _COL_NOREG, "noReg"),
        ]:
            pt = res["pt_centers"]
            y  = res["raa9_pt"][:, sidx]
            e  = res["err9_pt"][:, sidx]
            valid = ~np.isnan(y)
            ax.errorbar(pt[valid], y[valid], yerr=e[valid],
                        fmt="o-", color=color, lw=1.6, ms=5, capsize=3,
                        label=label if col == 0 else None, zorder=5)

        ax.text(0.95, 0.93, _STATE_TEX[sidx], transform=ax.transAxes,
                ha="right", va="top", fontsize=15)
        _finish_ax(ax, r"$p_T\ [\mathrm{GeV}]$", (0, 20))

    axes[0].set_ylabel(r"$R_{AA}$")
    axes[0].legend(loc="lower right", framealpha=0.85, edgecolor="none")

    # System annotation on first panel
    axes[0].text(0.04, 0.07,
                 r"O-O $\sqrt{s_{NN}} = 5.36$ TeV, $|y| < 2.4$" "\n"
                 r"QTraj NLO, $\kappa = 6$",
                 transform=axes[0].transAxes, fontsize=10, va="bottom")

    _save(fig, outdir, "raa_vs_pt_OO_5p36TeV")
    plt.close(fig)


# ─── RAA vs y ─────────────────────────────────────────────────────────────────

def plot_raa_vs_y(res_w, res_n, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_THESIS_RC)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    plt.subplots_adjust(wspace=0.05)

    for col, sidx in enumerate([IDX_1S, IDX_2S, IDX_3S]):
        ax = axes[col]

        for res, color, label in [
            (res_w, _COL_WREG,  "wReg (regen ON)"),
            (res_n, _COL_NOREG, "noReg"),
        ]:
            yc = res["y_centers"]
            y  = res["raa9_y"][:, sidx]
            valid = ~np.isnan(y)
            ax.plot(yc[valid], y[valid], "o-", color=color, lw=1.6, ms=5,
                    label=label if col == 0 else None, zorder=5)

        ax.text(0.95, 0.93, _STATE_TEX[sidx], transform=ax.transAxes,
                ha="right", va="top", fontsize=15)
        _finish_ax(ax, r"$y$", (-2.6, 2.6))

    axes[0].set_ylabel(r"$R_{AA}$")
    axes[0].legend(loc="lower center", framealpha=0.85, edgecolor="none")

    axes[0].text(0.04, 0.07,
                 r"O-O $\sqrt{s_{NN}} = 5.36$ TeV" "\n"
                 r"QTraj NLO, $\kappa = 6$, min-bias",
                 transform=axes[0].transAxes, fontsize=10, va="bottom")

    _save(fig, outdir, "raa_vs_y_OO_5p36TeV")
    plt.close(fig)


# ─── Summary bar chart ────────────────────────────────────────────────────────

def plot_integrated_summary(res_w, res_n, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_THESIS_RC)

    states = [IDX_1S, IDX_2S, IDX_3S]
    labels = [r"$\Upsilon(1S)$", r"$\Upsilon(2S)$", r"$\Upsilon(3S)$"]
    x = np.arange(len(states))

    fig, ax = plt.subplots(figsize=(7, 5))

    vals_w = [res_w["raa9_int"][s] for s in states]
    errs_w = [res_w["err9_int"][s] for s in states]
    vals_n = [res_n["raa9_int"][s] for s in states]
    errs_n = [res_n["err9_int"][s] for s in states]

    w = 0.35
    ax.bar(x - w/2, vals_w, w, yerr=errs_w, label="wReg",  color=_COL_WREG,
           alpha=0.8, capsize=5, error_kw={"elinewidth": 1.5})
    ax.bar(x + w/2, vals_n, w, yerr=errs_n, label="noReg", color=_COL_NOREG,
           alpha=0.8, capsize=5, error_kw={"elinewidth": 1.5})

    ax.axhline(1.0, lw=0.7, ls="--", color="gray")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel(r"$R_{AA}$"); ax.set_ylim(0, 1.8)
    ax.legend(framealpha=0.85, edgecolor="none")
    ax.text(0.97, 0.97,
            r"O-O $\sqrt{s_{NN}} = 5.36$ TeV" "\n"
            r"$|y| < 2.4$, min-bias",
            transform=ax.transAxes, ha="right", va="top", fontsize=11)

    _save(fig, outdir, "raa_integrated_OO_5p36TeV")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # ── 1) noReg: standard pipeline ──────────────────────────────────────────
    logger.info("Loading noReg data ...")
    table_n  = read_whitespace_table(NOREG_FILE, logger)
    recs_n   = parse_records(table_n, logger)
    obs_n    = build_observables(recs_n, logger)
    logger.info("noReg: %d matched physical trajectories", len(obs_n))

    # ── 2) wReg: raw file with primordial fix ─────────────────────────────────
    logger.info("Loading wReg raw data with primordial correction ...")
    recs_w = read_wreg_raw_with_primordial(WREG_RAW)
    obs_w  = build_observables(recs_w, logger)
    logger.info("wReg: %d matched physical trajectories", len(obs_w))

    # ── 3) Physics sanity check ───────────────────────────────────────────────
    mean_1s_w = np.mean([o.surv6[0] for o in obs_w])
    mean_1s_n = np.mean([o.surv6[0] for o in obs_n])
    ratio = mean_1s_w / mean_1s_n if mean_1s_n > 0 else float("inf")
    logger.info(
        "PHYSICS CHECK  mean_surv_1S:  wReg=%.4f  noReg=%.4f  ratio=%.4f  %s",
        mean_1s_w, mean_1s_n, ratio,
        "✓ (wReg > noReg)" if ratio > 1 else "✗ PHYSICS VIOLATION"
    )
    assert ratio > 1.0, (
        f"Physics violation: wReg 1S survival ({mean_1s_w:.4f}) "
        f"is NOT greater than noReg ({mean_1s_n:.4f})"
    )

    # ── 4) Analysis ───────────────────────────────────────────────────────────
    logger.info("Running noReg analysis ...")
    res_n = run_oo_analysis("noReg", obs_n)

    logger.info("Running wReg analysis ...")
    res_w = run_oo_analysis("wReg",  obs_w)

    # ── 5) Save CSVs ──────────────────────────────────────────────────────────
    save_csvs(OUTDIR, "wReg",  res_w)
    save_csvs(OUTDIR, "noReg", res_n)

    # ── 6) Thesis plots ───────────────────────────────────────────────────────
    logger.info("Generating thesis figures ...")
    plot_raa_vs_pt(res_w, res_n, OUTDIR)
    plot_raa_vs_y( res_w, res_n, OUTDIR)
    plot_integrated_summary(res_w, res_n, OUTDIR)

    logger.info("═" * 55)
    logger.info("O+O 5.36 TeV complete  →  %s", OUTDIR)
    logger.info("═" * 55)


if __name__ == "__main__":
    main()
