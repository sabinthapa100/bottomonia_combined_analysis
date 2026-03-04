# -*- coding: utf-8 -*-
"""
plot_primordial.py
==================
Plotting helpers for primordial R_pA analysis (bottomonia / charmonia).

Functions
---------
plot_rpa_vs_y_and_pt(band_dict, *, ...)
    1×(1+N_rapdity_windows) figure:
      - Left panel : R_pA vs y  (full y range, all Υ states on one plot)
      - Right N    : R_pA vs pT one panel per rapidity window

plot_rpa_vs_y_comparison(band_dict, ...)
    One panel, comparing species across y for each model.

All functions write PNGs to `save_dir` if provided, otherwise call plt.show().
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

# ---------------------------------------------------------------------------
# Colours / labels for main observable states
# ---------------------------------------------------------------------------

# Upsilon states we always show
UPSILON_PLOT_STATES = [
    ("ups1S",  r"$\Upsilon(1S)$"),
    ("ups2S",  r"$\Upsilon(2S)$"),
    ("ups3S",  r"$\Upsilon(3S)$"),
]

# Charmonia states we always show
JPSI_PLOT_STATES = [
    ("jpsi_1S", r"$J/\psi(1S)$"),
    ("psi_2S",  r"$\psi(2S)$"),
]

_COLORS_UPS  = None  # filled lazily
_COLORS_JPSI = None


def _ups_colors() -> Dict[str, tuple]:
    global _COLORS_UPS
    if _HAVE_PLT and _COLORS_UPS is None:
        base = plt.cm.tab10.colors
        _COLORS_UPS = {
            "ups1S":  base[0],
            "ups2S":  base[1],
            "ups3S":  base[2],
        }
    return _COLORS_UPS or {}


def _jpsi_colors() -> Dict[str, tuple]:
    global _COLORS_JPSI
    if _HAVE_PLT and _COLORS_JPSI is None:
        base = plt.cm.tab10.colors
        _COLORS_JPSI = {
            "jpsi_1S": base[0],
            "psi_2S":  base[1],
        }
    return _COLORS_JPSI or {}


def _plot_states_for(system_name: str):
    if system_name == "bottomonia":
        return UPSILON_PLOT_STATES
    return JPSI_PLOT_STATES


def _colors_for(system_name: str):
    if system_name == "bottomonia":
        return _ups_colors()
    return _jpsi_colors()


# ---------------------------------------------------------------------------
# Step-histogram helpers
# ---------------------------------------------------------------------------

def _draw_band_and_line(ax, x, center, lo, hi, color, lw=2.0, label=None, step=True):
    if step:
        # Use matplotlib's step functionality directly
        ax.fill_between(x, lo, hi, alpha=0.22, facecolor=color, edgecolor="none", step="mid")
        ax.step(x, center, color=color, lw=lw, label=label, solid_capstyle="round", where="mid")
    else:
        ax.fill_between(x, lo, hi, alpha=0.22, facecolor=color, edgecolor="none")
        ax.plot(x, center, color=color, lw=lw, label=label)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _apply_style():
    if not _HAVE_PLT:
        return
    mpl.rcParams.update({
        "axes.linewidth":   1.4,
        "axes.titlesize":   13,
        "axes.labelsize":   13,
        "font.family":      "DejaVu Sans",
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
        "lines.linewidth":  2.0,
        "figure.autolayout": True,
    })


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_rpa_vs_y_and_pt(
    band_dict: Dict[str, Tuple["BandLike", str, str]],
    *,
    system_name: str = "bottomonia",
    # y-panel settings
    y_bins: Sequence[Tuple[float, float]],
    pt_window_for_y: Tuple[float, float] = (0.0, 20.0),
    y_window_full: Tuple[float, float] = (-5.0, 5.0),
    flip_y: bool = False,
    # pT-panel settings
    y_windows_for_pt: Sequence[Tuple[float, float]] = (
        (-5.0, -2.5), (-2.5, 0.0), (1.5, 4.0)
    ),
    pt_bins: Sequence[Tuple[float, float]] = (),
    pt_window_for_pt: Tuple[float, float] = (0.0, 20.0),
    # cosmetics
    ylim: Tuple[float, float] = (0.0, 1.1),
    xlim_y:  Optional[Tuple[float, float]] = None,
    xlim_pt: Optional[Tuple[float, float]] = None,
    step: bool = True,
    figure_size: Optional[Tuple[float, float]] = None,
    suptitle: str = "",
    # output
    save_dir: Optional[str] = None,
    tag: str = "primordial",
) -> None:
    """
    Create a 1×(1+N_pt_windows) figure for each entry in band_dict.

    Parameters
    ----------
    band_dict    : { label: (band_obj, system_name, linestyle) }
                   `band_obj` must have .vs_y(...) and .vs_pt(...) returning
                   (df_center, df_band).
    system_name  : 'bottomonia' or 'charmonia'
    y_bins       : rapidity bins for the left (R_pA vs y) panel
    pt_window_for_y : pT window for the y panel
    y_windows_for_pt: rapidity windows for the right pT panels
    pt_bins      : pT bins for pT panels (required if y_windows_for_pt is non-empty)
    ylim         : y-axis limits
    flip_y       : flip rapidity sign (p-going → positive)
    step         : use step histograms
    suptitle     : super-title for the figure
    save_dir     : if given, save PNG there; else show interactively
    tag          : filename prefix
    """
    if not _HAVE_PLT:
        raise RuntimeError("matplotlib not available.")
    _apply_style()

    plot_states = _plot_states_for(system_name)
    colors      = _colors_for(system_name)

    n_pt_panels = len(y_windows_for_pt)
    n_cols      = 1 + n_pt_panels
    figsize     = figure_size or (4.0 * n_cols + 0.5, 4.5)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for model_label, (band_obj, sys_nm, ls) in band_dict.items():
        fig, axes = plt.subplots(1, n_cols, figsize=figsize, constrained_layout=True)
        if n_cols == 1:
            axes = [axes]

        # ---- left panel: R_pA vs y ----
        ax_y = axes[0]
        cen_y, bd_y = band_obj.vs_y(
            pt_window=pt_window_for_y,
            y_bins=list(y_bins),
            flip_y=flip_y,
        )
        x_y = cen_y["y"].to_numpy()
        for st, label in plot_states:
            if st not in cen_y.columns:
                continue
            c   = colors.get(st, "gray")
            mu  = cen_y[st].to_numpy()
            lo  = bd_y.get(f"{st}_lo", pd.Series(mu)).to_numpy()
            hi  = bd_y.get(f"{st}_hi", pd.Series(mu)).to_numpy()
            _draw_band_and_line(ax_y, x_y, mu, lo, hi, c, label=label, step=step)

        ax_y.set_ylim(*ylim)
        if xlim_y:
            ax_y.set_xlim(*xlim_y)
        ax_y.set_xlabel(r"$y$")
        ax_y.set_ylabel(r"$R_{pA}$")
        ax_y.axhline(1.0, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax_y.text(
            0.03, 0.97,
            rf"$p_T \in [{pt_window_for_y[0]:.0f},{pt_window_for_y[1]:.0f}]$ GeV",
            transform=ax_y.transAxes, va="top", ha="left", fontsize=10,
        )
        ax_y.legend(frameon=False, fontsize=9)

        # ---- right panels: R_pA vs pT at each rapidity window ----
        for ki, yw in enumerate(y_windows_for_pt):
            ax_pt = axes[1 + ki]
            if not pt_bins:
                ax_pt.set_visible(False)
                continue

            cen_pt, bd_pt = band_obj.vs_pt(y_window=yw, pt_bins=list(pt_bins))
            x_pt = cen_pt["pt"].to_numpy()
            for st, label in plot_states:
                if st not in cen_pt.columns:
                    continue
                c  = colors.get(st, "gray")
                mu = cen_pt[st].to_numpy()
                lo = bd_pt.get(f"{st}_lo", pd.Series(mu)).to_numpy()
                hi = bd_pt.get(f"{st}_hi", pd.Series(mu)).to_numpy()
                _draw_band_and_line(ax_pt, x_pt, mu, lo, hi, c, label=label, step=step)

            ax_pt.set_ylim(*ylim)
            if xlim_pt:
                ax_pt.set_xlim(*xlim_pt)
            ax_pt.set_xlabel(r"$p_T$ [GeV]")
            ax_pt.axhline(1.0, color="gray", lw=0.7, ls="--", alpha=0.5)
            ax_pt.text(
                0.03, 0.97,
                rf"${yw[0]:.1f} < y < {yw[1]:.1f}$",
                transform=ax_pt.transAxes, va="top", ha="left", fontsize=10,
            )

        if suptitle:
            fig.suptitle(f"{suptitle}  [{model_label}]", fontsize=12, y=1.01)

        fname = f"{tag}_{model_label.replace(' ','_')}"
        if save_dir:
            fpath_png = os.path.join(save_dir, fname + ".png")
            fpath_pdf = os.path.join(save_dir, fname + ".pdf")
            fig.savefig(fpath_png, dpi=180, bbox_inches="tight")
            fig.savefig(fpath_pdf, bbox_inches="tight")
            plt.close(fig)
            print(f"[plot] saved → {fpath_png}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Overlay comparison: multiple models on the same axes
# ---------------------------------------------------------------------------

def plot_rpa_comparison_vs_y(
    entries: Dict[str, Tuple["BandLike", str, str]],
    *,
    system_name: str = "bottomonia",
    y_bins: Sequence[Tuple[float, float]],
    pt_window: Tuple[float, float] = (0.0, 20.0),
    flip_y: bool = False,
    states_to_show: Optional[List[str]] = None,
    ylim: Tuple[float, float] = (0.0, 1.1),
    xlim_y: Optional[Tuple[float, float]] = None,
    step: bool = True,
    suptitle: str = "",
    save_dir: Optional[str] = None,
    tag: str = "comparison_vs_y",
) -> None:
    """
    Compare multiple band objects (e.g. NPWLC vs Pert) on the same R_pA(y) panel
    for each observable state.
    """
    if not _HAVE_PLT:
        raise RuntimeError("matplotlib not available.")
    _apply_style()

    plot_states = _plot_states_for(system_name)
    if states_to_show:
        plot_states = [(s, l) for s, l in plot_states if s in states_to_show]

    colors   = _colors_for(system_name)
    linestyles = ["-", "--", ":", "-."]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    model_handles = []
    for mi, (model_label, (band_obj, sys_nm, _)) in enumerate(entries.items()):
        ls  = linestyles[mi % len(linestyles)]
        cen, bd = band_obj.vs_y(
            pt_window=pt_window,
            y_bins=list(y_bins),
            flip_y=flip_y,
        )
        x = cen["y"].to_numpy()
        for st, state_label in plot_states:
            if st not in cen.columns:
                continue
            c  = colors.get(st, "gray")
            mu = cen[st].to_numpy()
            lo = bd.get(f"{st}_lo", pd.Series(mu)).to_numpy()
            hi = bd.get(f"{st}_hi", pd.Series(mu)).to_numpy()
            _draw_band_and_line(
                ax, x, mu, lo, hi, c, lw=2.0, step=step,
                label=f"{state_label} [{model_label}]"
            )
            # Override linestyle
            if ax.lines:
                ax.lines[-1].set_linestyle(ls)

        model_handles.append(
            Line2D([0], [0], color="0.35", lw=2.0, ls=ls, label=model_label)
        )

    # combined legend
    state_handles = [
        Line2D([0], [0], color=colors.get(s, "gray"), lw=2.0, label=l)
        for s, l in plot_states
    ]
    first_legend  = ax.legend(handles=state_handles, loc="upper right", frameon=False, fontsize=9)
    ax.add_artist(first_legend)
    ax.legend(handles=model_handles, loc="lower left", frameon=False, fontsize=9)

    ax.set_ylim(*ylim)
    if xlim_y:
        ax.set_xlim(*xlim_y)
    ax.set_xlabel(r"$y$"); ax.set_ylabel(r"$R_{pA}$")
    ax.axhline(1.0, color="gray", lw=0.7, ls="--", alpha=0.5)
    if suptitle:
        ax.set_title(suptitle, fontsize=12)

    fname = f"{tag}"
    if save_dir:
        fpath_png = os.path.join(save_dir, fname + ".png")
        fpath_pdf = os.path.join(save_dir, fname + ".pdf")
        fig.savefig(fpath_png, dpi=180, bbox_inches="tight")
        fig.savefig(fpath_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved → {fpath_png}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# R_pA vs pT sub-panels (3 rapidity windows), both models side-by-side
# ---------------------------------------------------------------------------

def plot_rpa_vs_pt_three_windows(
    entries: Dict[str, Tuple["BandLike", str, str]],
    *,
    system_name: str = "bottomonia",
    pt_bins: Sequence[Tuple[float, float]],
    y_windows: Sequence[Tuple[float, float]],
    states_to_show: Optional[List[str]] = None,
    ylim: Tuple[float, float] = (0.0, 1.1),
    xlim_pt: Optional[Tuple[float, float]] = None,
    step: bool = True,
    suptitle: str = "",
    save_dir: Optional[str] = None,
    tag: str = "rpa_pt_windows",
) -> None:
    """
    Create a 1×N_windows figure comparing multiple models (e.g. NPWLC and Pert)
    in each rapidity window panel, showing Υ(1S,2S,3S) together.
    """
    if not _HAVE_PLT:
        raise RuntimeError("matplotlib not available.")
    _apply_style()

    plot_states = _plot_states_for(system_name)
    if states_to_show:
        plot_states = [(s, l) for s, l in plot_states if s in states_to_show]

    colors     = _colors_for(system_name)
    linestyles = ["-", "--", ":", "-."]
    n_win = len(y_windows)
    figsize = (4.5 * n_win, 4.5)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, n_win, figsize=figsize, constrained_layout=True)
    if n_win == 1:
        axes = [axes]

    for ki, yw in enumerate(y_windows):
        ax = axes[ki]
        for mi, (model_label, (band_obj, sys_nm, _)) in enumerate(entries.items()):
            ls  = linestyles[mi % len(linestyles)]
            cen, bd = band_obj.vs_pt(y_window=yw, pt_bins=list(pt_bins))
            x = cen["pt"].to_numpy()
            for st, state_label in plot_states:
                if st not in cen.columns:
                    continue
                c  = colors.get(st, "gray")
                mu = cen[st].to_numpy()
                lo = bd.get(f"{st}_lo", pd.Series(mu)).to_numpy()
                hi = bd.get(f"{st}_hi", pd.Series(mu)).to_numpy()
                _draw_band_and_line(ax, x, mu, lo, hi, c, lw=2.0, step=step)
                if ax.lines:
                    ax.lines[-1].set_linestyle(ls)

        ax.set_ylim(*ylim)
        if xlim_pt:
            ax.set_xlim(*xlim_pt)
        ax.set_xlabel(r"$p_T$ [GeV]")
        if ki == 0:
            ax.set_ylabel(r"$R_{pA}$")
        ax.axhline(1.0, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.text(
            0.03, 0.97,
            rf"${yw[0]:.1f} < y < {yw[1]:.1f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
        )

    # Combined legend on first panel
    ax0 = axes[0]
    state_handles = [
        Line2D([0], [0], color=colors.get(s, "gray"), lw=2.0, label=l)
        for s, l in plot_states
    ]
    model_handles = [
        Line2D([0], [0], color="0.3", lw=2.0, ls=linestyles[i % len(linestyles)],
               label=ml)
        for i, ml in enumerate(entries.keys())
    ]
    leg1 = ax0.legend(handles=state_handles, loc="upper right", frameon=False, fontsize=9)
    ax0.add_artist(leg1)
    ax0.legend(handles=model_handles, loc="lower left", frameon=False, fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)

    fname = f"{tag}"
    if save_dir:
        fpath_png = os.path.join(save_dir, fname + ".png")
        fpath_pdf = os.path.join(save_dir, fname + ".pdf")
        fig.savefig(fpath_png, dpi=180, bbox_inches="tight")
        fig.savefig(fpath_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved → {fpath_png}")
    else:
        plt.show()
