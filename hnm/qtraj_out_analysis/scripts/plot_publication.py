#!/usr/bin/env python3
"""
Publication-quality plots matching Mathematica style from arXiv-2305.17841v2.

Style:
- Theory: step function (Mathematica Join[..., {y[[-1]]}] trick)
- Theory band: shaded envelope between k3 and k4
- Data: error bars with experiment-specific markers
- ALICE: open circles, ATLAS: open squares, CMS: open triangles
- Dashed horizontal line at R_AA = 1
- Grid lines at 0.2 opacity
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYSTEM_BASES = {
    ("PbPb", "5.02 TeV"): "outputs/qtraj_outputs/LHC/PbPb5p02TeV/production",
    ("PbPb", "2.76 TeV"): "outputs/qtraj_outputs/LHC/PbPb2p76TeV/production",
    ("AuAu", "200 GeV"): "outputs/qtraj_outputs/RHIC/AuAu200GeV/production",
}

EXPERIMENT_STYLES = {
    "alice": {
        "marker": "o",
        "mfc": "none",
        "mec": "#0066CC",
        "ecolor": "#0066CC",
        "ms": 6,
        "mew": 1.5,
    },
    "atlas": {
        "marker": "s",
        "mfc": "none",
        "mec": "#2CA02C",
        "ecolor": "#2CA02C",
        "ms": 6,
        "mew": 1.5,
    },
    "cms": {
        "marker": "^",
        "mfc": "none",
        "mec": "#D62728",
        "ecolor": "#D62728",
        "ms": 7,
        "mew": 1.5,
    },
    "star": {
        "marker": "D",
        "mfc": "none",
        "mec": "#9467BD",
        "ecolor": "#9467BD",
        "ms": 6,
        "mew": 1.5,
    },
}

KAPPA_COLORS = {
    "k3": "#1f77b4",
    "k4": "#ff7f0e",
    "kappa4": "#ff7f0e",
    "kappa5": "#2ca02c",
}

logger = logging.getLogger("plot_publication")


def read_csv(path: str) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)


def draw_theory_envelope(ax, x, y_lo, y_hi, color="#888888", alpha=0.18, label=None):
    """Draw shaded theory envelope."""
    ax.fill_between(x, y_lo, y_hi, alpha=alpha, color=color, label=label, zorder=1)


def draw_theory_line(ax, x, y, label, color, lw=1.8, ls="-", alpha=0.9, zorder=2):
    """Draw theory curve."""
    ax.plot(
        x,
        y,
        color=color,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


def draw_experiment(ax, x, y, yerr, xerr=None, experiment="cms", label=""):
    """Draw experiment data with error bars."""
    style = EXPERIMENT_STYLES.get(experiment.lower(), EXPERIMENT_STYLES["cms"])
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt=style["marker"],
        mfc=style["mfc"],
        mec=style["mec"],
        ecolor=style["ecolor"],
        ms=style["ms"],
        mew=style["mew"],
        capsize=3,
        linestyle="none",
        label=label,
        zorder=5,
    )


def load_theory_pair(base, obs_id, state):
    """Load k3/k4 theory CSVs for a state."""
    result = {}
    theory_dir = os.path.join(base, "data", "comparison", "theory")
    if not os.path.isdir(theory_dir):
        return result
    for f in os.listdir(theory_dir):
        if f.startswith(f"{obs_id}__{state}_") and f.endswith("_theory.csv"):
            kappa = f.replace(f"{obs_id}__{state}_", "").replace("_theory.csv", "")
            result[kappa] = read_csv(os.path.join(theory_dir, f))
    return result


def load_envelope(base, obs_id, state):
    """Load envelope CSV if available."""
    env_dir = os.path.join(base, "data", "comparison", "theory_envelopes")
    path = os.path.join(env_dir, f"{obs_id}__{state}__theory_envelope.csv")
    if os.path.exists(path):
        return read_csv(path)
    return None


def load_experiments(base, obs_id):
    """Load all experimental CSVs for an observable."""
    exp_dir = os.path.join(base, "data", "comparison", "experiment")
    if not os.path.isdir(exp_dir):
        return []
    experiments = []
    for f in sorted(os.listdir(exp_dir)):
        if f.startswith(obs_id) and f.endswith("__exp.csv"):
            d = read_csv(os.path.join(exp_dir, f))
            name = f.replace(f"{obs_id}__", "").replace("__exp.csv", "")
            short = next(
                (k for k in ["alice", "atlas", "cms", "star"] if k in name.lower()),
                name,
            )
            experiments.append((d, name, short))
    return experiments


def get_x_label(obs_type):
    m = {
        "RAA_vs_npart": r"$\langle N_{\mathrm{part}} \rangle$",
        "RAA_vs_pt": r"$p_T$ [GeV]",
        "RAA_vs_y": r"$|y|$",
        "double_ratio_vs_npart": r"$\langle N_{\mathrm{part}} \rangle$",
        "double_ratio_vs_pt": r"$p_T$ [GeV]",
    }
    return m.get(obs_type, obs_type)


def get_y_label(obs_type):
    if obs_type.startswith("RAA"):
        return r"$R_{\mathrm{AA}}$"
    if obs_type.startswith("double_ratio"):
        return "Double ratio"
    return obs_type


def get_state_label(state):
    """Short LaTeX label for a state."""
    s = (
        state.lower()
        .replace("_", " ")
        .replace("s", "S")
        .replace("1s", "1S")
        .replace("2s", "2S")
        .replace("3s", "3S")
    )
    if "2s 1s" in s:
        return r"$\Upsilon(2S)/\Upsilon(1S)$"
    if "3s 1s" in s:
        return r"$\Upsilon(3S)/\Upsilon(1S)$"
    if "3s 2s" in s:
        return r"$\Upsilon(3S)/\Upsilon(2S)$"
    if "1s" in s:
        return r"$\Upsilon(1S)$"
    if "2s" in s:
        return r"$\Upsilon(2S)$"
    if "3s" in s:
        return r"$\Upsilon(3S)$"
    return state


def plot_single_observable(base, obs_id, obs_type, system, energy, states, outdir):
    """Create publication plot for one observable."""
    n_panels = len(states)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5), squeeze=False)
    axes_flat = axes.ravel()

    for ax, state in zip(axes_flat, states):
        # Theory envelope
        envelope = load_envelope(base, obs_id, state)
        if envelope is not None:
            x = envelope["x"]
            draw_theory_envelope(ax, x, envelope["y_lower"], envelope["y_upper"])

        # Individual theory curves
        theory = load_theory_pair(base, obs_id, state)
        for kappa, data in theory.items():
            x = data["x"]
            y = data["y_center"]
            color = KAPPA_COLORS.get(kappa, "#333333")
            draw_theory_line(ax, x, y, label=kappa, color=color)

        # Experiments
        experiments = load_experiments(base, obs_id)
        for exp_data, exp_name, exp_short in experiments:
            ncols = exp_data.dtype.names
            x = exp_data["x"]
            y = exp_data["y"]
            yerr_lo = exp_data["yerr_low"] if "yerr_low" in ncols else np.zeros(len(x))
            yerr_hi = (
                exp_data["yerr_high"] if "yerr_high" in ncols else np.zeros(len(x))
            )
            xerr = None
            if "x_low" in ncols and "x_high" in ncols:
                xerr = np.vstack([x - exp_data["x_low"], exp_data["x_high"] - x])
            draw_experiment(
                ax, x, y, yerr_hi, xerr=xerr, experiment=exp_short, label=exp_name
            )

        # Formatting
        ax.set_xlabel(get_x_label(obs_type), fontsize=11)
        ax.set_ylabel(get_y_label(obs_type), fontsize=11)
        ax.set_title(get_state_label(state), fontsize=12)
        ax.axhline(1.0, color="0.5", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.legend(fontsize=7, loc="best", framealpha=0.9)

    fig.suptitle(f"{system} {energy}", fontsize=13, y=1.02)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, f"{obs_id}.pdf")
    png_path = os.path.join(outdir, f"{obs_id}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_all_systems(repo_root):
    repo = Path(repo_root)
    observables = [
        # PbPb 5.02 TeV
        ("pbpb5023_raavsnpart", "RAA_vs_npart", "PbPb", "5.02 TeV", ["1s", "2s", "3s"]),
        ("pbpb5023_raavspt", "RAA_vs_pt", "PbPb", "5.02 TeV", ["1s", "2s", "3s"]),
        ("pbpb5023_raavsy", "RAA_vs_y", "PbPb", "5.02 TeV", ["1s", "2s", "3s"]),
        (
            "pbpb5023_ratio21vsnpart",
            "double_ratio_vs_npart",
            "PbPb",
            "5.02 TeV",
            ["2s_1s"],
        ),
        ("pbpb5023_ratio21vspt", "double_ratio_vs_pt", "PbPb", "5.02 TeV", ["2s_1s"]),
        (
            "pbpb5023_ratio31vsnpart",
            "double_ratio_vs_npart",
            "PbPb",
            "5.02 TeV",
            ["3s_1s"],
        ),
        ("pbpb5023_ratio32vspt", "double_ratio_vs_pt", "PbPb", "5.02 TeV", ["3s_2s"]),
        # PbPb 2.76 TeV
        ("pbpb2760_raavsnpart", "RAA_vs_npart", "PbPb", "2.76 TeV", ["1s", "2s", "3s"]),
        ("pbpb2760_raavspt", "RAA_vs_pt", "PbPb", "2.76 TeV", ["1s", "2s", "3s"]),
        ("pbpb2760_raavsy", "RAA_vs_y", "PbPb", "2.76 TeV", ["1s", "2s", "3s"]),
        # AuAu 200 GeV
        ("auau200_raavsnpart", "RAA_vs_npart", "AuAu", "200 GeV", ["1s", "2s", "3s"]),
        ("auau200_raavspt", "RAA_vs_pt", "AuAu", "200 GeV", ["1s", "2s"]),
        ("auau200_raavsy", "RAA_vs_y", "AuAu", "200 GeV", ["1s", "2s", "3s"]),
    ]

    total = 0
    for obs_id, obs_type, system, energy, states in observables:
        key = (system, energy)
        base = str(repo / SYSTEM_BASES.get(key, ""))
        if not os.path.exists(base):
            logger.warning(f"Base not found: {base} for {obs_id}")
            continue

        # Output directory
        sys_tag = system.replace(" ", "").replace("PbPb", "LHC").replace("AuAu", "RHIC")
        eng_tag = energy.replace(" ", "").replace(".", "p")
        if system == "PbPb":
            outdir = str(
                repo
                / "outputs"
                / "qtraj_outputs"
                / "LHC"
                / eng_tag
                / "publication"
                / "figures"
            )
        else:
            outdir = str(
                repo
                / "outputs"
                / "qtraj_outputs"
                / "RHIC"
                / eng_tag
                / "publication"
                / "figures"
            )

        try:
            pdf, png = plot_single_observable(
                base, obs_id, obs_type, system, energy, states, outdir
            )
            logger.info(f"✓ {obs_id} → {pdf}")
            total += 1
        except Exception as e:
            logger.error(f"✗ {obs_id}: {e}")

    logger.info(f"Generated {total} publication plots")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "inputs").exists():
            repo_root = str(parent)
            break
    else:
        repo_root = str(here.parents[1])
    plot_all_systems(repo_root)
