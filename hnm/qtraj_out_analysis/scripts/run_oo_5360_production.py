#!/usr/bin/env python3
"""
O+O 5.36 TeV HNM production.

Two modes:
  --mode noReg   : quantum jumps OFF  (no regeneration)
  --mode wReg    : quantum jumps ON   (with regeneration)

Single min-bias impact parameter b = 4.49691 fm.
Produces R_AA vs y, R_AA vs pT, and R_AA vs N_part (all at this single b).

Datafile format: records alternate S-wave (v1,v2,v4,v6) and P-wave (v3,v5).
Both rows have L=0 in column 7, so we pair by consecutive rows.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np

from qtraj_analysis.feeddown import (
    build_feeddown_matrix,
    split_hyperfine_6_to_9,
    solve_primordial_sigmas,
)
from qtraj_analysis.glauber import GlauberInterpolator, load_glauber
from qtraj_analysis.stats import mean_and_sem

# ── Constants ───────────────────────────────────────────────────────
B_MINBIAS = 4.49691  # fm

SIGMAS_EXP_OO = np.array(
    [
        1.0,
        0.25,
        0.05,
        0.15,
        0.10,
        0.02,
        0.005,
        0.015,
        0.010,
    ]
)

MODE_CONFIGS = {
    "noReg": {
        "label": "noReg",
        "description": "noRegeneration",
        "datafile": "inputs/qtraj_inputs/OxygenOxygen5360/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile_partial.gz",
    },
    "wReg": {
        "label": "wReg",
        "description": "withRegeneration",
        "datafile": "inputs/qtraj_inputs/OxygenOxygen5360/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile-avg.gz",
    },
}


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "inputs").exists() and (parent / "hnm").exists():
            return parent
    raise RuntimeError(f"Could not infer repo root from {here}")


REPO_ROOT = _find_repo_root()


def read_oo_datafile(datafile: str, logger: logging.Logger) -> np.ndarray:
    """
    Read OO datafile and pair S/P records.

    Format (noReg): 2-row records, 7-col meta + 8-col data.
      Records alternate S-wave (v1,v2,v4,v6 nonzero) and P-wave (v3,v5 nonzero).
      We pair consecutive (S, P) rows.

    Format (wReg): 4-row records, 7-col meta + 14-col data + 7-col meta + 14-col data.
      First data row: S-wave (cols 0-5 = surv6 S-wave states)
      Second data row: S-wave cont. (cols 0-5 again or P-wave)
      The 14-col format: [S_1S, S_2S, S_1P, S_3S, S_2P, S_1D, <other>, ..., L, qweight?]

    Returns list of dicts with surv6, pt, y, qweight.
    """
    if datafile.endswith(".gz"):
        opener = lambda: gzip.open(datafile, "rt")
    else:
        opener = lambda: open(datafile, "r")

    with opener() as f:
        lines = [line.strip() for line in f if line.strip()]

    # Detect format from first few lines
    ncols_data = None
    for i in range(1, min(10, len(lines))):
        cols = lines[i].split()
        if len(cols) >= 8:
            ncols_data = len(cols)
            break

    logger.info("Detected data row width: %d columns", ncols_data or 0)

    obs_list = []

    if ncols_data == 8:
        # noReg format: 2-row records, alternate S/P
        i = 0
        while i < len(lines) - 1:
            meta_cols = lines[i].split()
            data_cols = lines[i + 1].split()

            if len(meta_cols) != 7 or len(data_cols) != 8:
                i += 1
                continue

            v3 = float(data_cols[2])
            v5 = float(data_cols[4])

            if v3 > 0.001 or v5 > 0.001:
                # P-wave row — skip, we'll get it in the next pair
                i += 2
                continue

            s_meta = [float(x) for x in meta_cols]
            s_data = [float(x) for x in data_cols]

            # Look ahead for P-wave pair
            p_found = False
            if i + 3 < len(lines):
                pm = lines[i + 2].split()
                pd = lines[i + 3].split()
                if len(pm) == 7 and len(pd) == 8:
                    pv3 = float(pd[2])
                    pv5 = float(pd[4])
                    if pv3 > 0.001 or pv5 > 0.001:
                        surv6 = np.array(
                            [
                                s_data[0],
                                s_data[1],
                                float(pd[2]),
                                s_data[3],
                                float(pd[4]),
                                s_data[5],
                            ],
                            dtype=np.float64,
                        )
                        obs_list.append(
                            {
                                "surv6": surv6,
                                "pt": s_meta[4],
                                "y": s_meta[6],
                                "qweight": s_data[7],
                            }
                        )
                        p_found = True
                        i += 4

            if not p_found:
                surv6 = np.array(
                    [
                        s_data[0],
                        s_data[1],
                        0.0,
                        s_data[3],
                        0.0,
                        s_data[5],
                    ],
                    dtype=np.float64,
                )
                obs_list.append(
                    {
                        "surv6": surv6,
                        "pt": s_meta[4],
                        "y": s_meta[6],
                        "qweight": s_data[7],
                    }
                )
                i += 2

    elif ncols_data == 14:
        # wReg format: 4-row records.
        # Structure: (meta, data, meta_copy, data_zero)
        # data (first data row) 14 columns:
        #   cols 0-5:  direct evolution survival (6 states)
        #   cols 6-11: regeneration contribution (6 states)
        #   col 12:    qweight
        # Total R_AA = cols 0-5 + cols 6-11
        # data_zero (second data row) is mostly zeros — ignore it.
        i = 0
        while i < len(lines) - 3:
            m1 = lines[i].split()
            d1 = lines[i + 1].split()

            if len(m1) != 7 or len(d1) != 14:
                i += 1
                continue

            meta = [float(x) for x in m1]
            direct = np.array([float(x) for x in d1[:6]], dtype=np.float64)
            regen = np.array([float(x) for x in d1[6:12]], dtype=np.float64)
            surv6 = direct + regen
            qweight = float(d1[12]) if len(d1) > 12 else 1.0

            obs_list.append(
                {
                    "surv6": surv6,
                    "pt": meta[4],
                    "y": meta[6],
                    "qweight": qweight,
                }
            )
            i += 4

    else:
        logger.error("Unsupported datafile format (ncols=%s)", ncols_data)

    logger.info("Read %d trajectory observables from %s", len(obs_list), datafile)
    return np.array(obs_list, dtype=object) if obs_list else np.array([], dtype=object)


def compute_raa_inclusive(
    surv6_mean: np.ndarray,
    surv6_sem: np.ndarray,
    sigmas_prim: np.ndarray,
    feeddown: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inclusive R_AA (9 states) from primordial 6-state survival."""
    surv9 = split_hyperfine_6_to_9(surv6_mean)
    sem9 = split_hyperfine_6_to_9(surv6_sem)

    num = feeddown @ (sigmas_prim * surv9)
    den = feeddown @ sigmas_prim

    raa_inc = np.divide(num, den, out=np.full_like(num, np.nan), where=(den != 0))
    raa_inc_sem = np.divide(
        feeddown @ (sigmas_prim * sem9),
        den,
        out=np.full_like(sem9, 0.0),
        where=(den != 0),
    )
    return raa_inc, raa_inc_sem


def run_oo_production(mode: str, outdir: str, logger: logging.Logger) -> int:
    config = MODE_CONFIGS[mode]
    datafile = str(REPO_ROOT / config["datafile"])

    if not os.path.exists(datafile):
        logger.error("Datafile not found: %s", datafile)
        return 1

    os.makedirs(outdir, exist_ok=True)

    # 1) Read datafile
    obs = read_oo_datafile(datafile, logger)
    if len(obs) == 0:
        logger.error("No observables read from datafile!")
        return 1

    # Stack all survival vectors
    surv6_all = np.vstack([o["surv6"] for o in obs])
    pt_all = np.array([o["pt"] for o in obs])
    y_all = np.array([o["y"] for o in obs])

    # 2) Overall mean survival (all trajectories at single b)
    surv6_mean, surv6_sem = mean_and_sem(surv6_all)
    logger.info("Overall survival at b=%.4f:", B_MINBIAS)
    for idx, (m, s) in enumerate(zip(surv6_mean, surv6_sem)):
        logger.info("  state[%d] = %.6f ± %.6f", idx, m, s)

    # 3) Load Glauber
    from scipy.interpolate import interp1d as _interp1d

    glauber_dir = (
        REPO_ROOT / "inputs" / "qtraj_inputs" / "OxygenOxygen5360" / "glauber-data"
    )
    npart_table = np.loadtxt(str(glauber_dir / "npartvsbData.tsv"))
    nbin_table = np.loadtxt(str(glauber_dir / "nbinvsbData.tsv"))

    npart = float(
        _interp1d(npart_table[:, 0], npart_table[:, 1], kind="linear")(B_MINBIAS)
    )
    nbin = float(
        _interp1d(nbin_table[:, 0], nbin_table[:, 1], kind="linear")(B_MINBIAS)
    )
    logger.info("Glauber: N_part=%.2f, N_bin=%.2f", npart, nbin)

    # Also load via the Glauber loader for pT/y binning (N_bin only)
    bvsc_path = str(glauber_dir / "bvscData.tsv")
    nbin_path = str(glauber_dir / "nbinvsbData.tsv")
    bvals = np.array([B_MINBIAS])
    npart_vals = np.array([npart])

    glauber_model = load_glauber(bvsc_path, nbin_path, bvals, npart_vals, logger)
    glauber = GlauberInterpolator(glauber_model)

    # 4) Feeddown
    feeddown = build_feeddown_matrix()
    sigmas_prim = solve_primordial_sigmas(feeddown, SIGMAS_EXP_OO)

    raa9, raa9_sem = compute_raa_inclusive(surv6_mean, surv6_sem, sigmas_prim, feeddown)
    logger.info("Inclusive R_AA: 1S=%.4f 2S=%.4f 3S=%.4f", raa9[0], raa9[1], raa9[5])

    # 5) R_AA vs pT
    pt_edges = np.array([0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    raa6_pt_mean = []
    raa6_pt_sem = []
    for i in range(len(pt_edges) - 1):
        mask = (pt_all >= pt_edges[i]) & (pt_all < pt_edges[i + 1])
        if np.any(mask):
            mu, se = mean_and_sem(surv6_all[mask])
        else:
            mu = np.full(6, np.nan)
            se = np.full(6, 0.0)
        raa6_pt_mean.append(mu)
        raa6_pt_sem.append(se)
    raa6_pt_mean = np.vstack(raa6_pt_mean)
    raa6_pt_sem = np.vstack(raa6_pt_sem)

    raa9_pt = []
    raa9_pt_sem = []
    for i in range(len(pt_centers)):
        r9, r9s = compute_raa_inclusive(
            raa6_pt_mean[i], raa6_pt_sem[i], sigmas_prim, feeddown
        )
        raa9_pt.append(r9)
        raa9_pt_sem.append(r9s)
    raa9_pt = np.vstack(raa9_pt)
    raa9_pt_sem = np.vstack(raa9_pt_sem)

    # 6) R_AA vs y
    y_edges = np.array([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    raa6_y_mean = []
    raa6_y_sem = []
    for i in range(len(y_edges) - 1):
        mask = (y_all >= y_edges[i]) & (y_all < y_edges[i + 1])
        if np.any(mask):
            mu, se = mean_and_sem(surv6_all[mask])
        else:
            mu = np.full(6, np.nan)
            se = np.full(6, 0.0)
        raa6_y_mean.append(mu)
        raa6_y_sem.append(se)
    raa6_y_mean = np.vstack(raa6_y_mean)
    raa6_y_sem = np.vstack(raa6_y_sem)

    raa9_y = []
    raa9_y_sem = []
    for i in range(len(y_centers)):
        r9, r9s = compute_raa_inclusive(
            raa6_y_mean[i], raa6_y_sem[i], sigmas_prim, feeddown
        )
        raa9_y.append(r9)
        raa9_y_sem.append(r9s)
    raa9_y = np.vstack(raa9_y)
    raa9_y_sem = np.vstack(raa9_y_sem)

    # 7) Save CSVs
    state_labels = [
        "1S",
        "2S",
        "chi_b0(1P)",
        "chi_b1(1P)",
        "chi_b2(1P)",
        "3S",
        "chi_b0(2P)",
        "chi_b1(2P)",
        "chi_b2(2P)",
    ]
    label = config["label"]

    # N_part (single point)
    npart_path = os.path.join(outdir, f"oo5360_{label}_raavsnpart.csv")
    header = "N_part," + ",".join(f"RAA_{s}" for s in state_labels)
    arr = np.concatenate([[npart], raa9]).reshape(1, -1)
    np.savetxt(npart_path, arr, delimiter=",", header=header, comments="")
    logger.info("Saved: %s", npart_path)

    # pT
    pt_path = os.path.join(outdir, f"oo5360_{label}_raavspt.csv")
    header = "pT," + ",".join(f"RAA_{s}" for s in state_labels)
    arr = np.column_stack([pt_centers, raa9_pt])
    np.savetxt(pt_path, arr, delimiter=",", header=header, comments="")
    logger.info("Saved: %s", pt_path)

    # y
    y_path = os.path.join(outdir, f"oo5360_{label}_raavsy.csv")
    header = "y," + ",".join(f"RAA_{s}" for s in state_labels)
    arr = np.column_stack([y_centers, raa9_y])
    np.savetxt(y_path, arr, delimiter=",", header=header, comments="")
    logger.info("Saved: %s", y_path)

    # Primordial 6-state
    prim_path = os.path.join(outdir, f"oo5360_{label}_raa6_primordial.csv")
    header = "b,RAA_1S,RAA_2S,RAA_1P,RAA_3S,RAA_2P,RAA_1D,SEM_1S,SEM_2S,SEM_1P,SEM_3S,SEM_2P,SEM_1D"
    arr = np.concatenate([[B_MINBIAS], surv6_mean, surv6_sem]).reshape(1, -1)
    np.savetxt(prim_path, arr, delimiter=",", header=header, comments="")
    logger.info("Saved: %s", prim_path)

    # 8) Plot
    try:
        _plot_oo(
            label,
            config["description"],
            npart,
            raa9,
            raa9_sem,
            pt_centers,
            raa9_pt,
            raa9_pt_sem,
            y_centers,
            raa9_y,
            raa9_y_sem,
            outdir,
            logger,
        )
    except Exception as e:
        logger.warning("Plot generation failed: %s", e)

    logger.info("OO %s production complete.", mode)
    return 0


def _plot_oo(
    label,
    description,
    npart,
    raa_npart,
    raa_npart_sem,
    pt_centers,
    raa_pt,
    raa_pt_sem,
    y_centers,
    raa_y,
    raa_y_sem,
    outdir,
    logger,
):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_idx = [0, 1, 5]  # 1S, 2S, 3S
    state_labels_3 = [r"$\Upsilon(1S)$", r"$\Upsilon(2S)$", r"$\Upsilon(3S)$"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # N_part (single point)
    ax = axes[0]
    for si, sl in zip(state_idx, state_labels_3):
        ax.errorbar(
            [npart],
            [raa_npart[si]],
            yerr=[raa_npart_sem[si]],
            fmt="o",
            ms=8,
            capsize=4,
            label=sl,
        )
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel(r"$\langle N_{\mathrm{part}} \rangle$")
    ax.set_ylabel(r"$R_{\mathrm{AA}}$")
    ax.set_title(f"O+O 5.36 TeV ({label})\nb={B_MINBIAS} fm")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.5)
    ax.grid(alpha=0.2)

    # pT
    ax = axes[1]
    for si, sl in zip(state_idx, state_labels_3):
        ax.errorbar(
            pt_centers,
            raa_pt[:, si],
            yerr=raa_pt_sem[:, si],
            fmt="o",
            ms=6,
            capsize=3,
            label=sl,
        )
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel(r"$R_{\mathrm{AA}}$")
    ax.set_title(f"R_AA vs pT ({label})")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.5)
    ax.grid(alpha=0.2)

    # y
    ax = axes[2]
    for si, sl in zip(state_idx, state_labels_3):
        ax.errorbar(
            y_centers,
            raa_y[:, si],
            yerr=raa_y_sem[:, si],
            fmt="o",
            ms=6,
            capsize=3,
            label=sl,
        )
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$R_{\mathrm{AA}}$")
    ax.set_title(f"R_AA vs y ({label})")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.5)
    ax.grid(alpha=0.2)

    fig.suptitle(f"O+O 5.36 TeV — {description}", fontsize=14, y=1.02)
    fig.tight_layout()

    pdf_path = os.path.join(outdir, f"oo5360_{label}_raa_summary.pdf")
    png_path = os.path.join(outdir, f"oo5360_{label}_raa_summary.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plots: %s, %s", pdf_path, png_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="O+O 5.36 TeV HNM production")
    parser.add_argument("--mode", choices=["noReg", "wReg", "both"], default="both")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("qtraj_out_analysis.oo5360")

    modes = ["noReg", "wReg"] if args.mode == "both" else [args.mode]
    rc = 0

    for mode in modes:
        if args.outdir:
            outdir = args.outdir
        else:
            outdir = str(
                REPO_ROOT
                / "outputs"
                / "qtraj_outputs"
                / "OO"
                / "5p36TeV"
                / mode
                / "production"
            )

        logger.info("=" * 60)
        logger.info("Running OO 5.36 TeV mode=%s", mode)
        logger.info("=" * 60)

        result = run_oo_production(mode, outdir, logger)
        if result != 0:
            rc = result

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
