#!/usr/bin/env python3
"""
Publication-ready plots for PbPb 5.02 TeV, PbPb 2.76 TeV, and AuAu 200 GeV.

Features:
- Step-function theory curves (Mathematica style)
- Proper experiment legends: "CMS Upsilon(1S)" with correct color/marker
- Error bars with bin edges for pT (LHC=2.5 GeV, RHIC=1.5 GeV bins)
- Combined double-ratio figures (2S/1S and 3S/1S together)
- Step and smooth Y plots
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Constants ────────────────────────────────────────────────────────
REPO_ROOT = None
for p in Path(__file__).resolve().parents:
    if (p / "outputs").exists():
        REPO_ROOT = p
        break
if REPO_ROOT is None:
    REPO_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_BASES = {
    "pbpb5023": str(
        REPO_ROOT / "outputs" / "qtraj_outputs" / "LHC" / "PbPb5p02TeV" / "production"
    ),
    "pbpb2760": str(
        REPO_ROOT / "outputs" / "qtraj_outputs" / "LHC" / "PbPb2p76TeV" / "production"
    ),
    "auau200": str(
        REPO_ROOT / "outputs" / "qtraj_outputs" / "RHIC" / "AuAu200GeV" / "production"
    ),
}

# Experiment colors and markers (consistent across all plots)
EXP_STYLES = {
    "alice": {"color": "#0066CC", "marker": "o", "ms": 6, "mew": 1.3, "mfc": "none"},
    "atlas": {"color": "#2CA02C", "marker": "s", "ms": 6, "mew": 1.3, "mfc": "none"},
    "cms": {"color": "#D62728", "marker": "^", "ms": 7, "mew": 1.3, "mfc": "none"},
    "star": {"color": "#9467BD", "marker": "D", "ms": 6, "mew": 1.3, "mfc": "none"},
}

# State display names
STATE_LABELS = {
    "1s": r"$\Upsilon(1S)$",
    "2s": r"$\Upsilon(2S)$",
    "3s": r"$\Upsilon(3S)$",
    "2s_1s": r"$\Upsilon(2S)/\Upsilon(1S)$",
    "3s_1s": r"$\Upsilon(3S)/\Upsilon(1S)$",
    "3s_2s": r"$\Upsilon(3S)/\Upsilon(2S)$",
}

# Theory curve colors per kappa
KAPPA_COLORS = {
    "k3": "#1f77b4",
    "k4": "#ff7f0e",
    "kappa4": "#ff7f0e",
    "kappa5": "#2ca02c",
}


def read_theory(path):
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return d


def read_exp(path):
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return d


def get_exp_label(filename, obs_id):
    """Build legend label like 'CMS Upsilon(1S)' from filename."""
    name = filename.replace(obs_id + "__", "").replace("__exp.csv", "")
    parts = name.split("_")
    # Find experiment name
    exp = parts[0].upper() if parts else name
    # Find state
    if "upsilon_3s" in name:
        state = r"$\Upsilon(3S)$"
    elif "upsilon_2s" in name:
        state = r"$\Upsilon(2S)$"
    elif "upsilon_1s" in name:
        state = r"$\Upsilon(1S)$"
    elif "star_y_3s" in name:
        state = r"$\Upsilon(3S)$"
    elif "star_y_2s" in name:
        state = r"$\Upsilon(2S)$"
    elif "star_y_1s" in name:
        state = r"$\Upsilon(1S)$"
    elif "2s_upsilon_1s" in name:
        state = r"$\Upsilon(2S)/\Upsilon(1S)$"
    elif "3s_upsilon_1s" in name:
        state = r"$\Upsilon(3S)/\Upsilon(1S)$"
    elif "3s_upsilon_2s" in name:
        state = r"$\Upsilon(3S)/\Upsilon(2S)$"
    else:
        state = name
    return f"{exp} {state}"


def get_exp_short(filename):
    """Get experiment short name from filename."""
    name = filename.lower()
    if "alice" in name:
        return "alice"
    if "atlas" in name:
        return "atlas"
    if "cms" in name:
        return "cms"
    if "star" in name:
        return "star"
    return "unknown"


def draw_theory_step(ax, x, y, label, color, lw=1.8, ls="-", alpha=0.9):
    """Draw step-function theory curve (Mathematica Join[..., {y[[-1]]}])."""
    x_s = np.concatenate([x, [x[-1]]])
    y_s = np.concatenate([y, [y[-1]]])
    ax.plot(
        x_s,
        y_s,
        color=color,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        label=label,
        zorder=3,
        drawstyle="steps-post",
    )


def draw_theory_envelope_step(ax, x, y_lo, y_hi, color="#aaaaaa", alpha=0.15):
    """Draw step-function theory envelope."""
    x_s = np.concatenate([x, [x[-1]]])
    lo_s = np.concatenate([y_lo, [y_lo[-1]]])
    hi_s = np.concatenate([y_hi, [y_hi[-1]]])
    ax.fill_between(x_s, lo_s, hi_s, alpha=alpha, color=color, step="post", zorder=1)


def draw_exp(ax, x, y, yerr_lo, yerr_hi, exp_short, label, xerr_lo=None, xerr_hi=None):
    """Draw experiment with error bars and proper marker."""
    s = EXP_STYLES.get(exp_short, EXP_STYLES["cms"])
    if xerr_lo is not None and xerr_hi is not None:
        xerr = np.vstack([xerr_lo, xerr_hi])
    else:
        xerr = None
    yerr = np.vstack([yerr_lo, yerr_hi])
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt=s["marker"],
        color=s["color"],
        ecolor=s["color"],
        mfc=s["mfc"],
        mec=s["color"],
        ms=s["ms"],
        mew=s["mew"],
        capsize=3,
        capthick=1,
        linestyle="none",
        linewidth=1,
        label=label,
        zorder=5,
    )


def find_theory(base, obs_id, state):
    """Find theory CSVs for a state."""
    th_dir = os.path.join(base, "data", "comparison", "theory")
    if not os.path.isdir(th_dir):
        return {}
    result = {}
    for f in os.listdir(th_dir):
        if f.startswith(f"{obs_id}__{state}_") and f.endswith("_theory.csv"):
            kappa = f.replace(f"{obs_id}__{state}_", "").replace("_theory.csv", "")
            result[kappa] = read_theory(os.path.join(th_dir, f))
    return result


def find_envelope(base, obs_id, state):
    """Find envelope CSV."""
    env_dir = os.path.join(base, "data", "comparison", "theory_envelopes")
    path = os.path.join(env_dir, f"{obs_id}__{state}__theory_envelope.csv")
    if os.path.exists(path):
        return read_theory(path)
    return None


def find_experiments(base, obs_id):
    """Find all experimental CSVs."""
    exp_dir = os.path.join(base, "data", "comparison", "experiment")
    if not os.path.isdir(exp_dir):
        return []
    result = []
    for f in sorted(os.listdir(exp_dir)):
        if f.startswith(obs_id) and f.endswith("__exp.csv"):
            result.append(f)
    return result


def plot_raa_npart(base, obs_id, title, outdir):
    """R_AA vs N_part: 3-panel (1S, 2S, 3S) with step functions."""
    states = ["1s", "2s", "3s"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, state in zip(axes, states):
        # Theory envelope
        env = find_envelope(base, obs_id, state)
        if env is not None:
            draw_theory_envelope_step(ax, env["x"], env["y_lower"], env["y_upper"])

        # Theory curves (step)
        theory = find_theory(base, obs_id, state)
        for kappa, d in theory.items():
            draw_theory_step(
                ax,
                d["x"],
                d["y_center"],
                label=f"qtraj-NLO {kappa}",
                color=KAPPA_COLORS.get(kappa, "#333"),
                lw=1.8,
            )

        # Experiments
        exp_files = find_experiments(base, obs_id)
        for ef in exp_files:
            d = read_exp(os.path.join(base, "data", "comparison", "experiment", ef))
            short = get_exp_short(ef)
            label = get_exp_label(ef, obs_id)
            draw_exp(ax, d["x"], d["y"], d["yerr_low"], d["yerr_high"], short, label)

        ax.axhline(1.0, color="0.5", lw=0.7, ls="--", zorder=0)
        ax.set_xlabel(r"$\langle N_{\mathrm{part}} \rangle$", fontsize=12)
        ax.set_ylabel(r"$R_{\mathrm{AA}}$", fontsize=12)
        ax.set_title(STATE_LABELS.get(state, state), fontsize=13)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.15, lw=0.5)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=1)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, outdir, obs_id)


def plot_raa_pt(base, obs_id, title, outdir):
    """R_AA vs pT: 3-panel with step functions and bin-edge error bars."""
    states = ["1s", "2s", "3s"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, state in zip(axes, states):
        # Theory envelope
        env = find_envelope(base, obs_id, state)
        if env is not None:
            draw_theory_envelope_step(ax, env["x"], env["y_lower"], env["y_upper"])

        # Theory curves (step)
        theory = find_theory(base, obs_id, state)
        for kappa, d in theory.items():
            draw_theory_step(
                ax,
                d["x"],
                d["y_center"],
                label=f"qtraj-NLO {kappa}",
                color=KAPPA_COLORS.get(kappa, "#333"),
                lw=1.8,
            )

        # Experiments (with bin edges)
        exp_files = find_experiments(base, obs_id)
        for ef in exp_files:
            d = read_exp(os.path.join(base, "data", "comparison", "experiment", ef))
            short = get_exp_short(ef)
            label = get_exp_label(ef, obs_id)
            xerr_lo = d["x"] - d["x_low"]
            xerr_hi = d["x_high"] - d["x"]
            draw_exp(
                ax,
                d["x"],
                d["y"],
                d["yerr_low"],
                d["yerr_high"],
                short,
                label,
                xerr_lo=xerr_lo,
                xerr_hi=xerr_hi,
            )

        ax.axhline(1.0, color="0.5", lw=0.7, ls="--", zorder=0)
        ax.set_xlabel(r"$p_T$ [GeV]", fontsize=12)
        ax.set_ylabel(r"$R_{\mathrm{AA}}$", fontsize=12)
        ax.set_title(STATE_LABELS.get(state, state), fontsize=13)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.15, lw=0.5)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=1)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, outdir, obs_id)


def plot_raa_y(base, obs_id, title, outdir):
    """R_AA vs y: 3-panel with step functions."""
    states = ["1s", "2s", "3s"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, state in zip(axes, states):
        # Theory envelope
        env = find_envelope(base, obs_id, state)
        if env is not None:
            draw_theory_envelope_step(ax, env["x"], env["y_lower"], env["y_upper"])

        # Theory curves (step)
        theory = find_theory(base, obs_id, state)
        for kappa, d in theory.items():
            draw_theory_step(
                ax,
                d["x"],
                d["y_center"],
                label=f"qtraj-NLO {kappa}",
                color=KAPPA_COLORS.get(kappa, "#333"),
                lw=1.8,
            )

        # Experiments
        exp_files = find_experiments(base, obs_id)
        for ef in exp_files:
            d = read_exp(os.path.join(base, "data", "comparison", "experiment", ef))
            short = get_exp_short(ef)
            label = get_exp_label(ef, obs_id)
            draw_exp(ax, d["x"], d["y"], d["yerr_low"], d["yerr_high"], short, label)

        ax.axhline(1.0, color="0.5", lw=0.7, ls="--", zorder=0)
        ax.set_xlabel(r"$|y|$", fontsize=12)
        ax.set_ylabel(r"$R_{\mathrm{AA}}$", fontsize=12)
        ax.set_title(STATE_LABELS.get(state, state), fontsize=13)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.15, lw=0.5)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=1)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, outdir, obs_id)


def plot_double_ratios(
    base, obs_21_npart, obs_21_pt, obs_31_npart, obs_32_pt, title, outdir
):
    """Combined double-ratio figure: 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        (axes[0, 0], obs_21_npart, "2s_1s", r"$\langle N_{\mathrm{part}} \rangle$"),
        (axes[0, 1], obs_21_pt, "2s_1s", r"$p_T$ [GeV]"),
        (axes[1, 0], obs_31_npart, "3s_1s", r"$\langle N_{\mathrm{part}} \rangle$"),
        (axes[1, 1], obs_32_pt, "3s_2s", r"$p_T$ [GeV]"),
    ]

    for ax, obs_id, state, xlabel in panels:
        if not obs_id:
            ax.set_visible(False)
            continue

        # Theory envelope
        env = find_envelope(base, obs_id, state)
        if env is not None:
            draw_theory_envelope_step(ax, env["x"], env["y_lower"], env["y_upper"])

        # Theory curves
        theory = find_theory(base, obs_id, state)
        for kappa, d in theory.items():
            draw_theory_step(
                ax,
                d["x"],
                d["y_center"],
                label=f"qtraj-NLO {kappa}",
                color=KAPPA_COLORS.get(kappa, "#333"),
                lw=1.8,
            )

        # Experiments
        exp_files = find_experiments(base, obs_id)
        for ef in exp_files:
            d = read_exp(os.path.join(base, "data", "comparison", "experiment", ef))
            short = get_exp_short(ef)
            label = get_exp_label(ef, obs_id)
            xerr_lo = d["x"] - d["x_low"] if "x_low" in d.dtype.names else None
            xerr_hi = d["x_high"] - d["x"] if "x_high" in d.dtype.names else None
            draw_exp(
                ax,
                d["x"],
                d["y"],
                d["yerr_low"],
                d["yerr_high"],
                short,
                label,
                xerr_lo=xerr_lo,
                xerr_hi=xerr_hi,
            )

        ax.axhline(1.0, color="0.5", lw=0.7, ls="--", zorder=0)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Double ratio", fontsize=11)
        ax.set_title(STATE_LABELS.get(state, state), fontsize=12)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.15, lw=0.5)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    _save(
        fig,
        outdir,
        f"{obs_21_npart.split('__')[0]}_double_ratios"
        if obs_21_npart
        else "double_ratios",
    )


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f"{name}.pdf")
    png = os.path.join(outdir, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name} → {pdf}")


def run_all():
    logging.basicConfig(level=logging.WARNING)

    systems = [
        {
            "key": "pbpb5023",
            "title": "PbPb 5.02 TeV",
            "outdir": str(
                REPO_ROOT
                / "outputs"
                / "qtraj_outputs"
                / "LHC"
                / "PbPb5p02TeV"
                / "publication"
                / "figures"
            ),
            "plots": {
                "npart": ("pbpb5023_raavsnpart", plot_raa_npart),
                "pt": ("pbpb5023_raavspt", plot_raa_pt),
                "y": ("pbpb5023_raavsy", plot_raa_y),
            },
            "double_ratios": {
                "obs_21_npart": "pbpb5023_ratio21vsnpart",
                "obs_21_pt": "pbpb5023_ratio21vspt",
                "obs_31_npart": "pbpb5023_ratio31vsnpart",
                "obs_32_pt": "pbpb5023_ratio32vspt",
            },
        },
        {
            "key": "pbpb2760",
            "title": "PbPb 2.76 TeV",
            "outdir": str(
                REPO_ROOT
                / "outputs"
                / "qtraj_outputs"
                / "LHC"
                / "PbPb2p76TeV"
                / "publication"
                / "figures"
            ),
            "plots": {
                "npart": ("pbpb2760_raavsnpart", plot_raa_npart),
                "pt": ("pbpb2760_raavspt", plot_raa_pt),
                "y": ("pbpb2760_raavsy", plot_raa_y),
            },
        },
        {
            "key": "auau200",
            "title": "AuAu 200 GeV",
            "outdir": str(
                REPO_ROOT
                / "outputs"
                / "qtraj_outputs"
                / "RHIC"
                / "AuAu200GeV"
                / "publication"
                / "figures"
            ),
            "plots": {
                "npart": ("auau200_raavsnpart", plot_raa_npart),
                "pt": ("auau200_raavspt", plot_raa_pt),
                "y": ("auau200_raavsy", plot_raa_y),
            },
        },
    ]

    for sys_cfg in systems:
        base = SYSTEM_BASES[sys_cfg["key"]]
        outdir = sys_cfg["outdir"]
        title = sys_cfg["title"]
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

        for plot_name, (obs_id, plot_func) in sys_cfg["plots"].items():
            try:
                plot_func(base, obs_id, f"{title} — {obs_id.replace('_', ' ')}", outdir)
            except Exception as e:
                print(f"  ✗ {obs_id}: {e}")

        # Double ratios (only for PbPb 5.02 TeV)
        if "double_ratios" in sys_cfg:
            dr = sys_cfg["double_ratios"]
            try:
                plot_double_ratios(
                    base,
                    dr.get("obs_21_npart", ""),
                    dr.get("obs_21_pt", ""),
                    dr.get("obs_31_npart", ""),
                    dr.get("obs_32_pt", ""),
                    f"{title} — Double Ratios",
                    outdir,
                )
            except Exception as e:
                print(f"  ✗ double_ratios: {e}")

    print(f"\n{'=' * 60}")
    print("  DONE — all publication plots generated")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_all()
