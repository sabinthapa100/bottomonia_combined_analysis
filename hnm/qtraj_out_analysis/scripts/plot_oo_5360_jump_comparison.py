#!/usr/bin/env python3
"""
OO 5.36 TeV fixed-b comparison: Jump OFF (noReg) vs Jump ON (wReg).

Requested kinematics:
  - R_AA vs pT: pT in [0, 20] GeV, y in [-2.4, 2.4]
  - R_AA vs y : y in [-2.4, 2.4], integrate pT up to 20 GeV

Statewise outputs:
  - 1S, 2S, 3S, 1P (chi_b1 1P), 2P (chi_b1 2P)
  - one dashed (Jump OFF) + one solid (Jump ON) curve per plot
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qtraj_analysis.binning import compute_raa_vs_pt, compute_raa_vs_y  # noqa: E402
from qtraj_analysis.feeddown import (  # noqa: E402
    apply_feeddown_to_raa6,
    build_feeddown_matrix,
    solve_primordial_sigmas,
)
from qtraj_analysis.io import load_qtraj_table, parse_records  # noqa: E402
from qtraj_analysis.kinematics_presets import SIGMAS_EXP_OO_5360  # noqa: E402
from qtraj_analysis.matching import build_observables  # noqa: E402


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "inputs").exists() and (parent / "hnm").exists():
            return parent
    raise RuntimeError(f"Could not infer repo root from {here}")


REPO_ROOT = _find_repo_root()
B_FIXED = 4.49691

MODE_FILES = {
    "Jump OFF (noReg)": "inputs/qtraj_inputs/OxygenOxygen5360/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile.gz",
    "Jump ON (wReg)": "inputs/qtraj_inputs/OxygenOxygen5360/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile.gz",
}

STATE_MAP = (
    (0, "1S", r"$\Upsilon(1S)$"),
    (1, "2S", r"$\Upsilon(2S)$"),
    (5, "3S", r"$\Upsilon(3S)$"),
    (3, "1P", r"$\chi_{b1}(1P)$"),
    (7, "2P", r"$\chi_{b1}(2P)$"),
)


@dataclass(frozen=True)
class ModeResult:
    label: str
    pt_centers: np.ndarray
    raa9_pt: np.ndarray
    sem9_pt: np.ndarray
    y_centers: np.ndarray
    raa9_y: np.ndarray
    sem9_y: np.ndarray


def _apply_feeddown_binned(raa6: np.ndarray, sem6: np.ndarray, feeddown: np.ndarray, sigmas_prim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raa9 = np.full((raa6.shape[0], 9), np.nan, dtype=np.float64)
    sem9 = np.full((raa6.shape[0], 9), np.nan, dtype=np.float64)
    for i in range(raa6.shape[0]):
        if np.isnan(raa6[i, 0]):
            continue
        r, e = apply_feeddown_to_raa6(raa6[i], sem6[i], feeddown, sigmas_prim)
        raa9[i] = r
        sem9[i] = e
    return raa9, sem9


def _weighted_mean_sem_surv6(bin_obs: list) -> tuple[np.ndarray, np.ndarray]:
    if not bin_obs:
        return np.full(6, np.nan, dtype=np.float64), np.full(6, 0.0, dtype=np.float64)
    X = np.vstack([o.surv6 for o in bin_obs]).astype(np.float64)
    w = np.asarray([o.qweight for o in bin_obs], dtype=np.float64)
    if not np.isfinite(w).all() or np.sum(w) <= 0:
        w = np.ones(len(bin_obs), dtype=np.float64)
    wsum = float(np.sum(w))
    mu = (w[:, None] * X).sum(axis=0) / wsum
    var = (w[:, None] * (X - mu) ** 2).sum(axis=0) / wsum
    neff = (wsum * wsum) / float(np.sum(w * w))
    sem = np.sqrt(np.maximum(var, 0.0)) / np.sqrt(max(neff, 1.0))
    return mu, sem


def _compute_raa_vs_pt_weighted(obs: list, pt_edges: np.ndarray, y_window: tuple[float, float], logger: logging.Logger) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0, y1 = y_window
    chosen = [o for o in obs if y0 <= o.y <= y1]
    logger.info("Weighted binning: y window [%.2f, %.2f] -> %d trajectories", y0, y1, len(chosen))
    centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    means = []
    sems = []
    for i in range(len(pt_edges) - 1):
        p0, p1 = pt_edges[i], pt_edges[i + 1]
        bin_obs = [o for o in chosen if p0 <= o.pt < p1]
        mu, se = _weighted_mean_sem_surv6(bin_obs)
        means.append(mu)
        sems.append(se)
    return centers, np.vstack(means), np.vstack(sems)


def _compute_raa_vs_y_weighted(obs: list, y_edges: np.ndarray, pt_max_for_y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = [o for o in obs if o.pt <= pt_max_for_y]
    centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    means = []
    sems = []
    for i in range(len(y_edges) - 1):
        y0, y1 = y_edges[i], y_edges[i + 1]
        bin_obs = [o for o in chosen if y0 <= o.y < y1]
        mu, se = _weighted_mean_sem_surv6(bin_obs)
        means.append(mu)
        sems.append(se)
    return centers, np.vstack(means), np.vstack(sems)


def _analyze_mode(
    label: str,
    datafile: Path,
    pt_edges: np.ndarray,
    y_edges: np.ndarray,
    y_window_pt: tuple[float, float],
    pt_max_for_y: float,
    logger: logging.Logger,
    *,
    weighted_binning: bool,
) -> ModeResult:
    table = load_qtraj_table(str(datafile), logger)
    records = parse_records(table, logger)
    obs = build_observables(records, logger)

    feeddown = build_feeddown_matrix()
    sigmas_prim = solve_primordial_sigmas(feeddown, SIGMAS_EXP_OO_5360)

    if weighted_binning:
        pt_centers, raa6_pt, sem6_pt = _compute_raa_vs_pt_weighted(obs, pt_edges, y_window_pt, logger)
    else:
        pt_centers, raa6_pt, sem6_pt = compute_raa_vs_pt(obs, pt_edges, y_window=y_window_pt, logger=logger)
    raa9_pt, sem9_pt = _apply_feeddown_binned(raa6_pt, sem6_pt, feeddown, sigmas_prim)

    if weighted_binning:
        y_centers, raa6_y, sem6_y = _compute_raa_vs_y_weighted(obs, y_edges, pt_max_for_y)
    else:
        obs_y = [o for o in obs if o.pt <= pt_max_for_y]
        y_centers, raa6_y, sem6_y = compute_raa_vs_y(obs_y, y_edges, logger=logger)
    raa9_y, sem9_y = _apply_feeddown_binned(raa6_y, sem6_y, feeddown, sigmas_prim)

    return ModeResult(
        label=label,
        pt_centers=pt_centers,
        raa9_pt=raa9_pt,
        sem9_pt=sem9_pt,
        y_centers=y_centers,
        raa9_y=raa9_y,
        sem9_y=sem9_y,
    )


def _save_csv(mode: ModeResult, out_data: Path, mode_tag: str) -> None:
    out_data.mkdir(parents=True, exist_ok=True)
    header = (
        "x,RAA_1S,RAA_2S,RAA_1P0,RAA_1P1,RAA_1P2,RAA_3S,RAA_2P0,RAA_2P1,RAA_2P2,"
        "SEM_1S,SEM_2S,SEM_1P0,SEM_1P1,SEM_1P2,SEM_3S,SEM_2P0,SEM_2P1,SEM_2P2"
    )
    np.savetxt(
        out_data / "oo5360_raavspt_midrapidity.csv",
        np.column_stack([mode.pt_centers, mode.raa9_pt, mode.sem9_pt]),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10g",
    )
    np.savetxt(
        out_data / "oo5360_raavsy.csv",
        np.column_stack([mode.y_centers, mode.raa9_y, mode.sem9_y]),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10g",
    )


def _plot_statewise(
    no_reg: ModeResult,
    w_reg: ModeResult,
    out_fig_compare: Path,
    out_fig_noreg: Path,
    out_fig_wreg: Path,
    y_window_pt: tuple[float, float],
    pt_max_for_y: float,
    logger: logging.Logger,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_fig_compare.mkdir(parents=True, exist_ok=True)
    out_fig_noreg.mkdir(parents=True, exist_ok=True)
    out_fig_wreg.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    for idx, tag, label_tex in STATE_MAP:
        fig_pt, ax_pt = plt.subplots(figsize=(6.8, 4.8))
        ax_pt.errorbar(
            no_reg.pt_centers, no_reg.raa9_pt[:, idx], yerr=no_reg.sem9_pt[:, idx],
            fmt="o", ms=4, lw=1.5, ls="--", capsize=2, color="#1f77b4", label="Jump OFF (noReg)"
        )
        ax_pt.errorbar(
            w_reg.pt_centers, w_reg.raa9_pt[:, idx], yerr=w_reg.sem9_pt[:, idx],
            fmt="o", ms=4, lw=1.7, ls="-", capsize=2, color="#d62728", label="Jump ON (wReg)"
        )
        ax_pt.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
        ax_pt.set_ylabel(r"$R_{AA}$")
        ax_pt.set_xlim(0.0, 20.0)
        ax_pt.set_ylim(0.0, 1.6)
        ax_pt.grid(alpha=0.25, lw=0.5)
        ax_pt.axhline(1.0, ls="--", lw=0.8, color="0.5")
        ax_pt.set_title(f"{label_tex}  |  " + r"$-2 \leq y \leq 2$")
        ax_pt.legend(loc="best", framealpha=0.95)
        fig_pt.tight_layout()
        p_pt = out_fig_compare / f"oo5360_raavspt_{tag}__jump_on_off"
        fig_pt.savefig(str(p_pt) + ".pdf", bbox_inches="tight")
        fig_pt.savefig(str(p_pt) + ".png", dpi=220, bbox_inches="tight")
        plt.close(fig_pt)

        fig_y, ax_y = plt.subplots(figsize=(6.8, 4.8))
        ax_y.errorbar(
            no_reg.y_centers, no_reg.raa9_y[:, idx], yerr=no_reg.sem9_y[:, idx],
            fmt="o", ms=4, lw=1.5, ls="--", capsize=2, color="#1f77b4", label="Jump OFF (noReg)"
        )
        ax_y.errorbar(
            w_reg.y_centers, w_reg.raa9_y[:, idx], yerr=w_reg.sem9_y[:, idx],
            fmt="o", ms=4, lw=1.7, ls="-", capsize=2, color="#d62728", label="Jump ON (wReg)"
        )
        ax_y.set_xlabel(r"$y$")
        ax_y.set_ylabel(r"$R_{AA}$")
        ax_y.set_xlim(-5.0, 5.0)
        ax_y.set_ylim(0.0, 1.6)
        ax_y.grid(alpha=0.25, lw=0.5)
        ax_y.axhline(1.0, ls="--", lw=0.8, color="0.5")
        ax_y.set_title(f"{label_tex}  |  " + rf"$p_T \leq {pt_max_for_y:.0f}\,\mathrm{{GeV}}$")
        ax_y.legend(loc="best", framealpha=0.95)
        fig_y.tight_layout()
        p_y = out_fig_compare / f"oo5360_raavsy_{tag}__jump_on_off"
        fig_y.savefig(str(p_y) + ".pdf", bbox_inches="tight")
        fig_y.savefig(str(p_y) + ".png", dpi=220, bbox_inches="tight")
        plt.close(fig_y)
        logger.info("Saved plots for state %s", tag)

    # Production-style multipanel summaries
    fig_pt, axes_pt = plt.subplots(3, 2, figsize=(12.5, 11.0), sharex=True, sharey=True)
    fig_y, axes_y = plt.subplots(3, 2, figsize=(12.5, 11.0), sharex=True, sharey=True)
    axp = axes_pt.ravel()
    axy = axes_y.ravel()

    for i, (idx, _tag, label_tex) in enumerate(STATE_MAP):
        ax1 = axp[i]
        ax1.errorbar(
            no_reg.pt_centers, no_reg.raa9_pt[:, idx], yerr=no_reg.sem9_pt[:, idx],
            fmt="o", ms=3.2, lw=1.3, ls="--", capsize=1.8, color="#1f77b4", label="Jump OFF (noReg)"
        )
        ax1.errorbar(
            w_reg.pt_centers, w_reg.raa9_pt[:, idx], yerr=w_reg.sem9_pt[:, idx],
            fmt="o", ms=3.2, lw=1.5, ls="-", capsize=1.8, color="#d62728", label="Jump ON (wReg)"
        )
        ax1.set_xlim(0.0, 20.0)
        ax1.set_ylim(0.0, 1.6)
        ax1.axhline(1.0, ls="--", lw=0.7, color="0.55")
        ax1.grid(alpha=0.2, lw=0.5)
        ax1.set_title(label_tex)

        ax2 = axy[i]
        ax2.errorbar(
            no_reg.y_centers, no_reg.raa9_y[:, idx], yerr=no_reg.sem9_y[:, idx],
            fmt="o", ms=3.2, lw=1.3, ls="--", capsize=1.8, color="#1f77b4", label="Jump OFF (noReg)"
        )
        ax2.errorbar(
            w_reg.y_centers, w_reg.raa9_y[:, idx], yerr=w_reg.sem9_y[:, idx],
            fmt="o", ms=3.2, lw=1.5, ls="-", capsize=1.8, color="#d62728", label="Jump ON (wReg)"
        )
        ax2.set_xlim(-5.0, 5.0)
        ax2.set_ylim(0.0, 1.6)
        ax2.axhline(1.0, ls="--", lw=0.7, color="0.55")
        ax2.grid(alpha=0.2, lw=0.5)
        ax2.set_title(label_tex)

    # Hide empty 6th panel
    axp[-1].axis("off")
    axy[-1].axis("off")

    for j in (0, 2, 4):
        axp[j].set_ylabel(r"$R_{AA}$")
        axy[j].set_ylabel(r"$R_{AA}$")
    for j in (4,):
        axp[j].set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
        axy[j].set_xlabel(r"$y$")

    axp[0].legend(loc="lower right", framealpha=0.95)
    axy[0].legend(loc="lower right", framealpha=0.95)

    fig_pt.suptitle(r"O+O $\sqrt{s_{NN}}=5.36$ TeV  |  $-2 \leq y \leq 2$", y=0.995, fontsize=15)
    fig_y.suptitle(r"O+O $\sqrt{s_{NN}}=5.36$ TeV  |  $p_T \leq 20$ GeV", y=0.995, fontsize=15)
    fig_pt.tight_layout(rect=(0, 0, 1, 0.98))
    fig_y.tight_layout(rect=(0, 0, 1, 0.98))

    fig_pt.savefig(out_fig_compare / "oo5360_raavspt_stategrid__jump_on_off.pdf", bbox_inches="tight")
    fig_pt.savefig(out_fig_compare / "oo5360_raavspt_stategrid__jump_on_off.png", dpi=250, bbox_inches="tight")
    fig_y.savefig(out_fig_compare / "oo5360_raavsy_stategrid__jump_on_off.pdf", bbox_inches="tight")
    fig_y.savefig(out_fig_compare / "oo5360_raavsy_stategrid__jump_on_off.png", dpi=250, bbox_inches="tight")
    plt.close(fig_pt)
    plt.close(fig_y)

    # Standalone per-mode band + binned-point plots
    def plot_single_mode(mode: ModeResult, mode_tag: str, color: str, linestyle: str, out_fig_mode: Path) -> None:
        for idx, tag, label_tex in STATE_MAP:
            fig1, ax1 = plt.subplots(figsize=(6.8, 4.8))
            y = mode.raa9_pt[:, idx]
            e = mode.sem9_pt[:, idx]
            m = np.isfinite(y)
            ax1.fill_between(mode.pt_centers[m], y[m] - e[m], y[m] + e[m], color=color, alpha=0.22, linewidth=0.0)
            ax1.plot(mode.pt_centers[m], y[m], linestyle=linestyle, color=color, lw=2.0, label=mode.label)
            ax1.scatter(mode.pt_centers[m], y[m], s=18, color=color, zorder=3)
            ax1.set_xlim(0.0, 20.0)
            ax1.set_ylim(0.0, 1.6)
            ax1.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
            ax1.set_ylabel(r"$R_{AA}$")
            ax1.axhline(1.0, ls="--", lw=0.8, color="0.5")
            ax1.grid(alpha=0.2, lw=0.5)
            ax1.set_title(f"{label_tex}  |  " + r"$-2 \leq y \leq 2$")
            ax1.legend(loc="best", framealpha=0.95)
            fig1.tight_layout()
            fig1.savefig(out_fig_mode / f"oo5360_raavspt_{tag}.pdf", bbox_inches="tight")
            fig1.savefig(out_fig_mode / f"oo5360_raavspt_{tag}.png", dpi=230, bbox_inches="tight")
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(6.8, 4.8))
            y2 = mode.raa9_y[:, idx]
            e2 = mode.sem9_y[:, idx]
            m2 = np.isfinite(y2)
            ax2.fill_between(mode.y_centers[m2], y2[m2] - e2[m2], y2[m2] + e2[m2], color=color, alpha=0.22, linewidth=0.0)
            ax2.plot(mode.y_centers[m2], y2[m2], linestyle=linestyle, color=color, lw=2.0, label=mode.label)
            ax2.scatter(mode.y_centers[m2], y2[m2], s=18, color=color, zorder=3)
            ax2.set_xlim(-5.0, 5.0)
            ax2.set_ylim(0.0, 1.6)
            ax2.set_xlabel(r"$y$")
            ax2.set_ylabel(r"$R_{AA}$")
            ax2.axhline(1.0, ls="--", lw=0.8, color="0.5")
            ax2.grid(alpha=0.2, lw=0.5)
            ax2.set_title(f"{label_tex}  |  " + rf"$p_T \leq {pt_max_for_y:.0f}\,\mathrm{{GeV}}$")
            ax2.legend(loc="best", framealpha=0.95)
            fig2.tight_layout()
            fig2.savefig(out_fig_mode / f"oo5360_raavsy_{tag}.pdf", bbox_inches="tight")
            fig2.savefig(out_fig_mode / f"oo5360_raavsy_{tag}.png", dpi=230, bbox_inches="tight")
            plt.close(fig2)

        # Triplet overlay: 1S, 2S, 3S together (single mode)
        triplet = (
            (0, r"$\Upsilon(1S)$", "#1f77b4"),
            (1, r"$\Upsilon(2S)$", "#ff7f0e"),
            (5, r"$\Upsilon(3S)$", "#2ca02c"),
        )

        fig3, ax3 = plt.subplots(figsize=(7.2, 5.0))
        for idx_t, lbl, c in triplet:
            y = mode.raa9_pt[:, idx_t]
            e = mode.sem9_pt[:, idx_t]
            m = np.isfinite(y)
            ax3.fill_between(mode.pt_centers[m], y[m] - e[m], y[m] + e[m], color=c, alpha=0.16, linewidth=0.0)
            ax3.plot(mode.pt_centers[m], y[m], linestyle=linestyle, color=c, lw=2.0, label=lbl)
            ax3.scatter(mode.pt_centers[m], y[m], s=16, color=c, zorder=3)
        ax3.set_xlim(0.0, 20.0)
        ax3.set_ylim(0.0, 1.6)
        ax3.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
        ax3.set_ylabel(r"$R_{AA}$")
        ax3.axhline(1.0, ls="--", lw=0.8, color="0.5")
        ax3.grid(alpha=0.2, lw=0.5)
        ax3.set_title(f"{mode.label}  |  " + r"$-2 \leq y \leq 2$")
        ax3.legend(loc="best", framealpha=0.95, title="States")
        fig3.tight_layout()
        fig3.savefig(out_fig_mode / "oo5360_raavspt_triplet_1S2S3S.pdf", bbox_inches="tight")
        fig3.savefig(out_fig_mode / "oo5360_raavspt_triplet_1S2S3S.png", dpi=240, bbox_inches="tight")
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(7.2, 5.0))
        for idx_t, lbl, c in triplet:
            y2 = mode.raa9_y[:, idx_t]
            e2 = mode.sem9_y[:, idx_t]
            m2 = np.isfinite(y2)
            ax4.fill_between(mode.y_centers[m2], y2[m2] - e2[m2], y2[m2] + e2[m2], color=c, alpha=0.16, linewidth=0.0)
            ax4.plot(mode.y_centers[m2], y2[m2], linestyle=linestyle, color=c, lw=2.0, label=lbl)
            ax4.scatter(mode.y_centers[m2], y2[m2], s=16, color=c, zorder=3)
        ax4.set_xlim(-5.0, 5.0)
        ax4.set_ylim(0.0, 1.6)
        ax4.set_xlabel(r"$y$")
        ax4.set_ylabel(r"$R_{AA}$")
        ax4.axhline(1.0, ls="--", lw=0.8, color="0.5")
        ax4.grid(alpha=0.2, lw=0.5)
        ax4.set_title(f"{mode.label}  |  " + rf"$p_T \leq {pt_max_for_y:.0f}\,\mathrm{{GeV}}$")
        ax4.legend(loc="best", framealpha=0.95, title="States")
        fig4.tight_layout()
        fig4.savefig(out_fig_mode / "oo5360_raavsy_triplet_1S2S3S.pdf", bbox_inches="tight")
        fig4.savefig(out_fig_mode / "oo5360_raavsy_triplet_1S2S3S.png", dpi=240, bbox_inches="tight")
        plt.close(fig4)

    plot_single_mode(no_reg, "noreg", "#1f77b4", "--", out_fig_noreg)
    plot_single_mode(w_reg, "wreg", "#d62728", "-", out_fig_wreg)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        default="outputs/qtraj_outputs/LHC/OxygenOxygen5p36TeV/jump_on_off_statewise",
        help="Repo-relative or absolute output root for data+figures.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output root before regenerating organized outputs.",
    )
    parser.add_argument(
        "--unweighted-binning",
        action="store_true",
        help="Use plain (unweighted) trajectory means in bins; default is qweight-weighted binning.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("plot_oo_5360_jump_comparison")

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    if args.clean and out_root.exists():
        import shutil

        shutil.rmtree(out_root)

    out_data_noreg = out_root / "noreg" / "data"
    out_data_wreg = out_root / "wreg" / "data"
    out_fig_noreg = out_root / "noreg" / "figures"
    out_fig_wreg = out_root / "wreg" / "figures"
    out_data_compare = out_root / "comparison" / "data"
    out_fig_compare = out_root / "comparison" / "figures"
    out_data_compare.mkdir(parents=True, exist_ok=True)

    pt_edges = np.arange(0.0, 21.0, 1.0, dtype=np.float64)   # 0..20 (1 GeV bins)
    y_edges = np.arange(-2.4, 2.4 + 0.4, 0.4, dtype=np.float64)
    y_window_pt = (-2.4, 2.4)
    pt_max_for_y = 20.0

    no_reg = _analyze_mode(
        "Jump OFF (noReg)",
        (REPO_ROOT / MODE_FILES["Jump OFF (noReg)"]).resolve(),
        pt_edges,
        y_edges,
        y_window_pt,
        pt_max_for_y,
        logger,
        weighted_binning=not args.unweighted_binning,
    )
    w_reg = _analyze_mode(
        "Jump ON (wReg)",
        (REPO_ROOT / MODE_FILES["Jump ON (wReg)"]).resolve(),
        pt_edges,
        y_edges,
        y_window_pt,
        pt_max_for_y,
        logger,
        weighted_binning=not args.unweighted_binning,
    )

    _save_csv(no_reg, out_data_noreg, "noreg")
    _save_csv(w_reg, out_data_wreg, "wreg")
    _save_csv(no_reg, out_data_compare, "noreg")
    _save_csv(w_reg, out_data_compare, "wreg")
    _plot_statewise(
        no_reg,
        w_reg,
        out_fig_compare,
        out_fig_noreg,
        out_fig_wreg,
        y_window_pt,
        pt_max_for_y,
        logger,
    )

    logger.info("Done. b is fixed by OO inputs at ~%.5f fm", B_FIXED)
    logger.info("Outputs: %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
