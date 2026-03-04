"""
plotting_utils.py

Standard plotting utilities for eloss/broadening analysis.
Extracted from 05_b_eloss_cronin_pA_LHC_final.ipynb.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Standard colors for centrality bins
COLORS = {
    "0-20%":  "C0",
    "20-40%": "C1",
    "40-60%": "C2",
    "60-80%": "C3",
    "60-100%": "C3",
    "80-100%": "C4",
    "0-100%": "gray", # Changed to gray for Total integrated if needed
    "MB": "k"
}

def format_publication_ax(ax, xlabel, ylabel, ylim=(0, 2.0)):
    """Standard formatting for publication plots."""
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(ylim)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1, top=True, right=True)
    ax.grid(False) 
    # Ensure minus signs are rendered as hyphens for compatibility if desired, 
    # but matplotlib usually handles this. User requested specific look.
    return ax

def plot_components_panel(
    ax, x_data, 
    bands_loss, bands_broad, bands_total,
    label_key, 
    feature_label="",
    xlabel=None, ylabel=None,
    step=True
):
    """
    Plots Loss, Broad, and Total components for a SINGLE centrality/bin on one axis.
    """
    # Helper to unpack
    def get_dat(bands, key):
        if not bands or key not in bands[0]: return None, None, None
        return bands[0][key], bands[1][key], bands[2][key] # c, lo, hi

    # Plot Loss (Dashed, Blue-ish or defined)
    Lc, Llo, Lhi = get_dat(bands_loss, label_key)
    if Lc is not None:
        col = "C1" # Orange
        lbl = "eloss"
        if step:
            x_edge, y_c = _step_from_centers(x_data, Lc)
            _, y_lo = _step_from_centers(x_data, Llo)
            _, y_hi = _step_from_centers(x_data, Lhi)
            ax.step(x_edge, y_c, where='post', color=col, ls='--', lw=1.5, label=lbl)
            ax.fill_between(x_edge, y_lo, y_hi, step='post', color=col, alpha=0.2, lw=0)
        else:
            ax.plot(x_data, Lc, color=col, ls='--', lw=1.5, label=lbl)
            ax.fill_between(x_data, Llo, Lhi, color=col, alpha=0.2, lw=0)

    # Plot Broad (Dotted, Green-ish)
    Bc, Blo, Bhi = get_dat(bands_broad, label_key)
    if Bc is not None:
        col = "C2" # Green
        lbl = "broad"
        if step:
            x_edge, y_c = _step_from_centers(x_data, Bc)
            _, y_lo = _step_from_centers(x_data, Blo)
            _, y_hi = _step_from_centers(x_data, Bhi)
            ax.step(x_edge, y_c, where='post', color=col, ls=':', lw=1.5, label=lbl)
            ax.fill_between(x_edge, y_lo, y_hi, step='post', color=col, alpha=0.2, lw=0)
        else:
            ax.plot(x_data, Bc, color=col, ls=':', lw=1.5, label=lbl)
            ax.fill_between(x_data, Blo, Bhi, color=col, alpha=0.2, lw=0)

    # Plot Total (Solid, Red/Black, with band)
    Tc, Tlo, Thi = get_dat(bands_total, label_key)
    if Tc is not None:
        col = "k" if label_key == "MB" else "C3" # Red for cent, Black for MB
        lbl = f"eloss x broad" # ({label_key}) - label key is usually title
        if step:
            x_edge, y_c = _step_from_centers(x_data, Tc)
            _, y_lo = _step_from_centers(x_data, Tlo)
            _, y_hi = _step_from_centers(x_data, Thi)
            ax.step(x_edge, y_c, where='post', color=col, ls='-', lw=2.0, label=lbl)
            ax.fill_between(x_edge, y_lo, y_hi, step='post', color=col, alpha=0.2, lw=0)
        else:
            ax.plot(x_data, Tc, color=col, ls='-', lw=2.0, label=lbl)
            ax.fill_between(x_data, Tlo, Thi, color=col, alpha=0.2, lw=0)

    # Internal Text
    if feature_label:
        # User requested bold top right
        ax.text(0.96, 0.96, feature_label, transform=ax.transAxes,
                va='top', ha='right', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

    if xlabel and ylabel:
        format_publication_ax(ax, xlabel, ylabel)
    
def _step_from_centers(x_cent, vals):
    """
    Given bin centers x_cent and values vals (same length, uniform spacing),
    build (x_edges, y_step) so that plt.step(x_edges, y_step, where="post")
    gives a flat segment per bin.
    """
    x_cent = np.asarray(x_cent, float)
    vals   = np.asarray(vals, float)
    
    if x_cent.size > 1:
        dx = np.diff(x_cent)
        dx0 = dx[0]
        # Allow small tolerance
        if not np.allclose(dx, dx0, rtol=1e-3):
             # Non-uniform binning fallback: use midpoints
             edges = np.concatenate([
                 [x_cent[0] - dx0/2],
                 0.5*(x_cent[:-1] + x_cent[1:]),
                 [x_cent[-1] + dx0/2]
             ])
             # Repeat values to make step plot work with 'post'
             # actually standard 'post' step needs edges and values
             # But here we want visual "bins".
             # Better approach for non-uniform:
             x_edges = []
             y_step = []
             # This helper was specific to uniform bins in the notebook.
             # We'll stick to uniform assumption or simple extension.
             pass 

    # Default uniform assumption logic from notebook
    if x_cent.size > 1:
        dx0 = x_cent[1] - x_cent[0]
    else:
        dx0 = 1.0

    x_edges = np.concatenate(([x_cent[0] - 0.5*dx0],
                              x_cent + 0.5*dx0))
    # Extend vals to match edges for 'post' step
    y_step  = np.concatenate([vals, vals[-1:]])
    return x_edges, y_step

def plot_RpA_vs_y_binned(
    y_cent, R_dict, labels, component="eloss_broad",
    show_MB=True, ax=None, colors=COLORS
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=130)
    else:
        fig = ax.figure

    for lab in labels:
        if lab == "0-100%":
            continue
        if lab in R_dict[component]:
            R = R_dict[component][lab]
            col = colors.get(lab, None)
            ax.plot(y_cent, R, marker="o", ms=3, lw=1.2, label=lab, color=col)

    if show_MB and "MB" in R_dict[component]:
        Rmb = R_dict[component]["MB"]
        ax.plot(y_cent, Rmb, lw=2.0, ls="-", color="k", label="MB")

    ax.set_ylim(0.6, 1.3)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(rf"$R^{{{component}}}_{{pA, \psi}}(y)$")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(False)
    return fig, ax

def plot_RpA_vs_pT_binned(
    pT_cent, R_dict, labels, component="eloss_broad",
    show_MB=True, ax=None, colors=COLORS
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=130)
    else:
        fig = ax.figure

    for lab in labels:
        if lab == "0-100%":
            continue
        if lab in R_dict[component]:
            R = R_dict[component][lab]
            col = colors.get(lab, None)
            ax.plot(pT_cent, R, marker="o", ms=3, lw=1.2, label=lab, color=col)

    if show_MB and "MB" in R_dict[component]:
        Rmb = R_dict[component]["MB"]
        ax.plot(pT_cent, Rmb, lw=2.0, ls="-", color="k", label="MB")

    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel(r"$R_{pA}(p_T)$")
    ax.legend(frameon=False, fontsize=7)
    return fig, ax

def plot_RpA_vs_y_band(
    y_cent, Rc_dict, Rlow_dict, Rhigh_dict,
    tags_order,
    comp_name="eloss_broad", # which component is this dictionary for?
    ax=None,
    step: bool = True,
    note: str | None = None,
    colors=COLORS,
    label_suffix=""
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=130)
    else:
        fig = ax.figure

    linestyle = "-"
    if comp_name == "loss": linestyle = "--"
    elif comp_name == "broad": linestyle = ":"
    
    # If plotting multiple components on same axis, might want distinct colors/styles logic.
    # But current usage usually plots ONE component vs y for multiple centralities.
    # If user wants all components, we might call this function multiple times on same ax?
    # Or we redesign this to take ALL dictionaries.
    
    # Based on user request: "with eloss, pt broad, eloss_broad components visible -- all shown"
    # This implies we likely want to plot ALL components for a GIVEN centrality?
    # OR all centralities for ALL components? That would be very crowded.
    # The original notebooks plotted:
    # 1. RpA vs y (eloss+broad) for all centralities
    # 2. RpA vs pT (eloss+broad) for all centralities
    # 3. RpA vs Cent (showing loss, broad, total)
    
    # If user wants "all components visible" on RpA vs y plot?
    # That usually means for a SPECIFIC centrality (e.g. MB) showing breakdown.
    # Let's stick to the notebook style for RpA vs y/pT (Total RpA for multiple cents),
    # BUT allow overlay if needed.
    
    for tag in tags_order:
        if tag not in Rc_dict: continue
        
        Rc  = np.asarray(Rc_dict[tag])
        Rlo = np.asarray(Rlow_dict[tag])
        Rhi = np.asarray(Rhigh_dict[tag])
        col = colors.get(tag, "k")
        
        lbl = f"{tag}{label_suffix}"

        if step:
            x_edges, y_c  = _step_from_centers(y_cent, Rc)
            _,       y_lo = _step_from_centers(y_cent, Rlo)
            _,       y_hi = _step_from_centers(y_cent, Rhi)

            ax.step(x_edges, y_c, where="post", color=col, lw=1.5, label=lbl, linestyle=linestyle)
            if comp_name == "eloss_broad": # Only shade total band to avoid mess
                ax.fill_between(x_edges, y_lo, y_hi,
                                step="post", color=col, alpha=0.25, linewidth=0.0)
        else:
            ax.plot(y_cent, Rc, color=col, lw=1.5, label=lbl, linestyle=linestyle)
            if comp_name == "eloss_broad":
                ax.fill_between(y_cent, Rlo, Rhi,
                                color=col, alpha=0.25, linewidth=0.0)

    if "MB" in Rc_dict:
        Rc  = np.asarray(Rc_dict["MB"])
        Rlo = np.asarray(Rlow_dict["MB"])
        Rhi = np.asarray(Rhigh_dict["MB"])
        
        lbl = f"MB{label_suffix}"

        if step:
            x_edges, y_c  = _step_from_centers(y_cent, Rc)
            _,       y_lo = _step_from_centers(y_cent, Rlo)
            _,       y_hi = _step_from_centers(y_cent, Rhi)
            
            # Thick black line for MB
            ax.step(x_edges, y_c, where="post", color="k", lw=2.0, label=lbl, linestyle=linestyle)
            if comp_name == "eloss_broad":
                ax.fill_between(x_edges, y_lo, y_hi,
                                step="post", color="k", alpha=0.15, linewidth=0.0)
        else:
            ax.plot(y_cent, Rc, color="k", lw=2.0, label=lbl, linestyle=linestyle)
            if comp_name == "eloss_broad":
                ax.fill_between(y_cent, Rlo, Rhi,
                                color="k", alpha=0.15, linewidth=0.0)
            
    if note:
        ax.text(0.05, 0.95, note, transform=ax.transAxes,
                va="top", ha="left", fontsize=9, 
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$R_{pA}(y)$")
    ax.set_ylim(0, 1.4) 
    
    # Clean up legend (remove duplicates)
    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    # ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8, loc="lower left")
    
    return fig, ax

def plot_RpA_vs_pT_band(
    pT_cent,
    Rc_dict, Rlow_dict, Rhigh_dict,
    tags_order,
    comp_name="eloss_broad",
    ax=None,
    step: bool = True,
    note: str | None = None,
    colors=COLORS,
    label_suffix=""
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=130)
    else:
        fig = ax.figure

    linestyle = "-"
    if comp_name == "loss": linestyle = "--"
    elif comp_name == "broad": linestyle = ":"

    for tag in tags_order:
        if tag not in Rc_dict: continue
        
        Rc  = np.asarray(Rc_dict[tag])
        Rlo = np.asarray(Rlow_dict[tag])
        Rhi = np.asarray(Rhigh_dict[tag])
        col = colors.get(tag, "k")
        
        lbl = f"{tag}{label_suffix}"

        if step:
            x_edges, y_c  = _step_from_centers(pT_cent, Rc)
            _,       y_lo = _step_from_centers(pT_cent, Rlo)
            _,       y_hi = _step_from_centers(pT_cent, Rhi)

            ax.step(x_edges, y_c, where="post", lw=1.5, color=col, label=lbl, linestyle=linestyle)
            if comp_name == "eloss_broad":
                ax.fill_between(x_edges, y_lo, y_hi,
                                step="post", alpha=0.25, color=col, linewidth=0.0)
        else:
            ax.plot(pT_cent, Rc, lw=1.5, color=col, label=lbl, linestyle=linestyle)
            if comp_name == "eloss_broad":
                ax.fill_between(pT_cent, Rlo, Rhi,
                                alpha=0.25, color=col, linewidth=0.0)

    if "MB" in Rc_dict:
        Rc  = np.asarray(Rc_dict["MB"])
        Rlo = np.asarray(Rlow_dict["MB"])
        Rhi = np.asarray(Rhigh_dict["MB"])
        
        lbl = f"MB{label_suffix}"
            
        if step:
            x_edges, y_c  = _step_from_centers(pT_cent, Rc)
            _,       y_lo = _step_from_centers(pT_cent, Rlo)
            _,       y_hi = _step_from_centers(pT_cent, Rhi)
            ax.step(x_edges, y_c, where="post", color="k", lw=2.0, label=lbl, linestyle=linestyle)
            if comp_name == "eloss_broad":
                ax.fill_between(x_edges, y_lo, y_hi,
                                step="post", color="k", alpha=0.15, linewidth=0.0)
        else:
            ax.plot(pT_cent, Rc, color="k", lw=2.0, label=lbl, linestyle=linestyle)
            if comp_name == "eloss_broad":
                ax.fill_between(pT_cent, Rlo, Rhi,
                                color="k", alpha=0.15, linewidth=0.0)

    if note:
        ax.text(0.05, 0.95, note, transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel(r"$R_{pA}(p_T)$")
    
    # Clean up legend (remove duplicates)
    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    # ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
    
    return fig, ax

def plot_RpA_vs_centrality_components_band(
    cent_bins, labels,
    RL_c=None, RL_lo=None, RL_hi=None, RMB_loss=None,
    RB_c=None, RB_lo=None, RB_hi=None, RMB_broad=None,
    RT_c=None, RT_lo=None, RT_hi=None, RMB_tot=None,
    show=("eloss_broad",),
    ax=None,
    ylabel=r"$R_{pA}(\mathrm{cent})$",
    note: str | None = None,
    system_label: str | None = None,
):
    """
    Step-style RpA vs centrality, with optional bands for
    loss, broad, total.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.5), dpi=130)
    else:
        fig = ax.figure

    comp_color = {
        "loss":  "C1",
        "broad": "C2",
        "eloss_broad": "C0",
    }
    comp_label = {
        "loss":  r"loss",
        "broad": r"broad",
        "eloss_broad": r"eloss_broad",
    }
    if system_label is not None:
        comp_label["eloss_broad"] = system_label

    # Cent bins -> x axis
    # We'll plot steps from edge a to b
    
    def _plot_comp(name, Dic_c, Dic_lo, Dic_hi, MB_tuple):
        color = comp_color.get(name, "k")
        label = comp_label.get(name, name)
        
        # We construct a step plot across all bins
        # x-axis: 0..100
        # values: constant in each bin
        
        # sort bins by start
        # Assuming ordered labels matching cent_bins
        # cent_bins list of (a,b)
        
        # Build piecewise arrays
        xs = []
        ys = []
        ylos = []
        yhis = []
        
        # Iterate over bins
        for i, (a, b) in enumerate(cent_bins):
            lab = labels[i]
            if lab not in Dic_c: continue
            
            val = Dic_c[lab]
            vlo = Dic_lo[lab]
            vhi = Dic_hi[lab]
            
            # append standard step points
            # For "post", we want x=a -> val, x=b -> val
            # But matplotlib step "post" means interval [x[i], x[i+1]] has value y[i]
            # So we supply edges
            pass 

        # Using loop to plot disjoint steps (visual preference) or one continuous?
        # Typically one continuous if bins are contiguous. 
        # But here bins might not be contiguous, usually they are.
        
        # Let's just plot horizontal lines for each bin + bands
        first = True
        for i, (a, b) in enumerate(cent_bins):
            lab = labels[i]
            if lab not in Dic_c: continue
            
            val = Dic_c[lab]
            vlo = Dic_lo[lab]
            vhi = Dic_hi[lab]
            
            lbl = label if first else None
            
            # Line
            ax.hlines(val, a, b, colors=color, lw=2, label=lbl)
            
            # Band
            ax.fill_between([a, b], [vlo, vlo], [vhi, vhi],
                            color=color, alpha=0.25, linewidth=0.0)
            
            if first: first = False

        # MB band as background strip
        if MB_tuple is not None:
            mc, mlo, mhi = MB_tuple
            ax.axhline(mc, color=color, linestyle="--", alpha=0.4, lw=1.0)
            ax.axhspan(mlo, mhi, color=color, alpha=0.1, linewidth=0.0)

    if "loss" in show and RL_c:
        _plot_comp("loss", RL_c, RL_lo, RL_hi, RMB_loss)

    if "broad" in show and RB_c:
        _plot_comp("broad", RB_c, RB_lo, RB_hi, RMB_broad)
        
    if "eloss_broad" in show and RT_c:
        _plot_comp("eloss_broad", RT_c, RT_lo, RT_hi, RMB_tot)

    if note:
        ax.text(0.05, 0.95, note, transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax.set_xlabel("Centrality (%)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="best")
    
    return fig, ax
