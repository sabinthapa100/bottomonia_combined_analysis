#!/usr/bin/env python3
"""
plot_style.py - Shared plotting style for PhD thesis figures.

Conventions:
- No titles on plots (use legends and axis labels)
- Clean, publication-ready styling
- Consistent color palette and markers
- Notes inside plots for context
- HEPData-compatible formatting
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# ============================================================
# Style Configuration
# ============================================================
PLOT_STYLE = {
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.linewidth": 1.5,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
}

# Color palette (colorblind-friendly, publication-ready)
COLORS = {
    "1S": "#1f77b4",  # blue
    "2S": "#ff7f0e",  # orange
    "3S": "#2ca02c",  # green
    "1P": "#d62728",  # red
    "2P": "#9467bd",  # purple
    "OctS": "#8c564b",  # brown
    "OctP": "#e377c2",  # pink
    "munich": "#1f77b4",
    "ksu_iso": "#ff7f0e",
    "ksu_aniso": "#2ca02c",
    "noJumps": "#1f77b4",
    "withJumps": "#d62728",
}

# Line styles for different conditions
LINE_STYLES = {
    "noJumps": "-",
    "withJumps": "--",
    "munich": "-",
    "ksu_iso": "-.",
    "ksu_aniso": ":",
}

# Markers for different NQTRAJ values
MARKERS = {
    10: "o",
    20: "s",
    40: "^",
    100: "D",
    200: "v",
    500: "*",
}

# State labels for legends
STATE_LABELS = {
    "1S": r"$\Upsilon(1S)$",
    "2S": r"$\Upsilon(2S)$",
    "3S": r"$\Upsilon(3S)$",
    "1P": r"$\chi_{b}(1P)$",
    "2P": r"$\chi_{b}(2P)$",
    "OctS": r"Octet $S$",
    "OctP": r"Octet $P$",
}

# Physics constants
GEV_TO_FMC = 0.1973269804  # 1/GeV to fm/c
HBAR_C = 0.1973269804  # GeV*fm


def apply_style():
    """Apply the thesis plotting style."""
    plt.rcParams.update(PLOT_STYLE)
    plt.rcParams.update(
        {
            "text.usetex": False,  # Set True if LaTeX is available
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
            "mathtext.fontset": "dejavusans",
        }
    )


def setup_axes(ax, xlabel, ylabel, xlim=None, ylim=None, grid=True):
    """Configure axes with consistent styling."""
    ax.set_xlabel(xlabel, fontsize=16, fontweight="normal")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="normal")

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if grid:
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    ax.tick_params(
        direction="in", top=True, right=True, labelsize=13, width=1.2, length=6
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def add_physics_note(ax, text, loc="upper right", fontsize=10):
    """Add a physics context note inside the plot."""
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85
        ),
    )


def save_fig(fig, name, output_dir, formats=("png", "pdf")):
    """Save figure in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(
            path,
            dpi=300 if fmt == "png" else None,
            bbox_inches="tight",
            transparent=(fmt == "pdf"),
        )
        print(f"    Saved: {path}")


def survival_color(state_name):
    """Get color for a state."""
    return COLORS.get(state_name, "#333333")


def survival_label(state_name):
    """Get legend label for a state."""
    return STATE_LABELS.get(state_name, state_name)


def format_time_axis(ax, unit="fm/c"):
    """Format time axis with proper units."""
    if unit == "fm/c":
        ax.set_xlabel(r"$\tau$ [fm/c]")
    elif unit == "1/GeV":
        ax.set_xlabel(r"$\tau$ [1/GeV]")


def create_survival_legend(ax, states, jump_mode="noJumps"):
    """Create a clean legend for survival probability plots."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper right",
            framealpha=0.9,
            edgecolor="0.7",
            fontsize=12,
        )


def add_cms_label(ax, text="PbPb $\sqrt{s_{NN}}$ = 5 TeV"):
    """Add collision system label."""
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
        ),
    )


def add_potential_label(ax, potential_name):
    """Add potential type label."""
    labels = {
        "munich": "Munich Potential",
        "ksu_iso": "KSU (Isotropic)",
        "ksu_aniso": "KSU (Anisotropic)",
    }
    ax.text(
        0.02,
        0.02,
        labels.get(potential_name, potential_name),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85
        ),
    )


def create_multi_panel_figure(nrows, ncols, figsize=None):
    """Create a multi-panel figure with shared styling."""
    if figsize is None:
        figsize = (8 * ncols, 6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.atleast_2d(axes)
    return fig, axes
