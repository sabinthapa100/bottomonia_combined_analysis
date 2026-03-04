#!/usr/bin/env python
# coding: utf-8
"""
eloss_cronin_unified.py
=======================

A unified script to compute and plot coherent energy loss and pT broadening
for Quarkonia (Charmonia / Bottomonia) in pA, dA, and AA collisions.

Designed for flexibility and correctness by default.

Manual / Common Configurations:
-------------------------------
1. Bottomonia in Oxygen-Oxygen @ 5.36 TeV (Default)
   python eloss_cronin_unified.py
   (Uses --oxygen_density HarmonicOscillator by default)

2. Charmonia in p-Pb @ 5.02 TeV
   python eloss_cronin_unified.py --particle charmonia --system pA --energy 5023

3. Bottomonia in Au-Au @ 200 GeV
   python eloss_cronin_unified.py --particle bottomonia --system AA --energy 200

4. Bottomonia in d-Au @ 200 GeV
   python eloss_cronin_unified.py --particle bottomonia --system dA --energy 200

Outputs are automatically organized into:
    ../../output/primordial/<system><energy>/
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "eloss_code"))

import eloss_cronin_centrality as EC
import plotting_utils as PU
import system_configs as SC
from glauber import OpticalGlauber
from particle import Particle, PPSpectrumParams
import quenching_fast as QF
import coupling as CPL

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams["axes.grid"] = False 
plt.rcParams["legend.frameon"] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False 

def save_bands_to_csv(base_path, file_prefix, x_cent, bands_dict, labels):
    """
    Saves calculation results to CSV files. 
    One file per component (loss, broad, total).
    Each CSV contains the central values and bands for all centralities.
    """
    import pandas as pd
    for comp, (c_dict, lo_dict, hi_dict) in bands_dict.items():
        data = {'x': x_cent}
        # MB first if it exists
        order = (["MB"] if "MB" in c_dict else []) + [l for l in labels if l != "MB"]
        for lab in order:
            if lab in c_dict:
                # Ensure we handle potentially scalar values from extract()
                val_c = np.atleast_1d(c_dict[lab])
                val_lo = np.atleast_1d(lo_dict[lab])
                val_hi = np.atleast_1d(hi_dict[lab])
                
                data[f'R_{lab}_cent'] = val_c
                data[f'R_{lab}_low'] = val_lo
                data[f'R_{lab}_high'] = val_hi
        
        df = pd.DataFrame(data)
        csv_name = f"{file_prefix}_{comp}.csv"
        csv_path = os.path.join(base_path, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")

def save_centrality_to_csv(base_path, file_prefix, cent_bins, labels, bands_dict, filter_labels=None):
    """
    Saves centrality results to CSV files. 
    One file per component (loss, broad, total).
    Rows: MB, 0-10%, 10-20%, ...
    """
    import pandas as pd
    for comp, (c_dict, lo_dict, hi_dict) in bands_dict.items():
        rows = []
        # Add MB if it exists and we're not filtering it out
        if "MB" in c_dict and (filter_labels is None or "MB" in filter_labels):
            rows.append({
                'label': 'MB', 'bin_min': 0, 'bin_max': 100,
                'R_cent': float(c_dict['MB']), 'R_low': float(lo_dict['MB']), 'R_high': float(hi_dict['MB'])
            })
        
        for i, (a, b) in enumerate(cent_bins):
            if i >= len(labels): break
            lab = labels[i]
            if filter_labels is not None and lab not in filter_labels: continue
            if lab in c_dict:
                rows.append({
                    'label': lab, 'bin_min': a, 'bin_max': b,
                    'R_cent': float(c_dict[lab]), 'R_low': float(lo_dict[lab]), 'R_high': float(hi_dict[lab])
                })
        
        if not rows: continue
        df = pd.DataFrame(rows)
        csv_name = f"{file_prefix}_{comp}.csv"
        csv_path = os.path.join(base_path, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")

def get_config(system, energy, particle_family):
    """Returns a configuration object based on system, energy, and particle."""
    if system == "pA":
        # Usually pPb 5.02 or 8.16 TeV
        from glauber import SystemSpec
        cfg = SC.LHCConfig
        cfg.spec = cfg.spec_5TeV if energy < 6000 else SystemSpec("pA", energy, 208, sigma_nn_mb=71.0)
        cfg.roots = energy
    elif system == "dA":
        cfg = SC.RHICConfig
        cfg.roots = energy
    elif system == "AA":
        if abs(energy - 5023) < 10 or abs(energy - 5.02) < 0.1:
            cfg = SC.PbPbConfig
        elif abs(energy - 200) < 10:
            cfg = SC.AuAuConfig
        elif abs(energy - 5360) < 10 or abs(energy - 5.36) < 0.1:
            cfg = SC.OOConfig
        else:
            # Generic fallback for other AA systems (like PbPb 2.76 TeV or 2.25 TeV)
            from glauber import SystemSpec
            cfg = SC.PbPbConfig
            cfg.roots = energy
            cfg.spec = SystemSpec("AA", energy, 208, sigma_nn_mb=np.interp(energy, [200, 2760, 5023], [42.0, 61.8, 67.6]))
    else:
        raise ValueError(f"Unknown system {system}")
        
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Unified eloss/broadening computer and plotter.")
    parser.add_argument("--particle", type=str, choices=["charmonia", "bottomonia"], default="bottomonia",
                        help="Particle family to simulate (default: bottomonia).")
    parser.add_argument("--state", type=str, default="1S",
                        help="Quarkonium state (e.g. 1S).")
    parser.add_argument("--system", type=str, choices=["pA", "dA", "AA"], default="AA",
                        help="Collision system type (default: AA).")
    parser.add_argument("--energy", type=float, default=5360.0,
                        help="Collision energy in GeV (default: 5360.0).")
    parser.add_argument("--oxygen_density", type=str, choices=["HarmonicOscillator", "ThreeParamFermi", "WoodsSaxon"],
                        default="HarmonicOscillator", help="Density profile for Oxygen-16 (default: HarmonicOscillator).")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Unified Execution: {args.particle} {args.state} in {args.system} @ {args.energy} GeV")
    print("=" * 60)

    # Determine output directory based on conventions
    if args.system == "AA":
        if abs(args.energy - 5360) < 10: sys_folder = "oxygenoxygen5360"
        elif abs(args.energy - 5023) < 10: sys_folder = "PbPb5023"
        elif abs(args.energy - 200) < 10: sys_folder = "AuAu200"
        else: sys_folder = f"AA{int(args.energy)}"
    elif args.system == "pA":
        if abs(args.energy - 5023) < 10: sys_folder = "pPb5023"
        elif abs(args.energy - 8160) < 10: sys_folder = "pPb8160"
        else: sys_folder = f"pA{int(args.energy)}"
    elif args.system == "dA":
        sys_folder = f"dAu{int(args.energy)}"
    else:
        sys_folder = f"{args.system}{int(args.energy)}"

    base_out_dir = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "outputs", "eloss_ptbraod", sys_folder))
    minbias_dir = os.path.join(base_out_dir, "minbias")
    centralities_dir = os.path.join(base_out_dir, "centralities")
    os.makedirs(minbias_dir, exist_ok=True)
    os.makedirs(centralities_dir, exist_ok=True)
    
    print(f"Outputs will be saved nicely to: {base_out_dir}")
    print(f"  - MinBias results: {minbias_dir}")
    print(f"  - Centrality results: {centralities_dir}")

    cfg = get_config(args.system, args.energy, args.particle)
    roots = cfg.roots
    cent_bins = getattr(cfg, "cent_bins_plotting", getattr(cfg, "cent_bins", None))
    rap_windows = cfg.rapidity_windows

    print("\nInitializing Glauber...")
    gl = OpticalGlauber(cfg.spec, density_name=args.oxygen_density, verbose=True)

    # L_eff minbias and bin prints
    print(f"\nComputing L_eff for {args.system}...")
    if args.system == "AA":
        Lmb = gl.leff_minbias_AA()
        leff_bins = gl.leff_bins_AA(cent_bins)
        kind = "AA"
    elif args.system == "dA":
        Lmb = gl.leff_minbias_dA()
        leff_bins = gl.leff_bins_dA(cent_bins)
        kind = "dA"
    else:
        Lmb = gl.leff_minbias_pA()
        leff_bins = gl.leff_bins_pA(cent_bins)
        kind = "pA"

    print(f"  L_eff (MinBias) = {Lmb:.3f} fm")
    for lab, L in leff_bins.items():
        print(f"    {lab}: L_eff = {L:.3f} fm")

    # Particle setup
    pp_params_obj = None
    if hasattr(cfg, "pp_params"):
        pp_params_obj = PPSpectrumParams(p0=cfg.pp_params['p0'], 
                                         m=cfg.pp_params['m'], 
                                         n=cfg.pp_params['n'])
    
    mass_ov = 3.097 if args.particle == "charmonia" else 9.460
    P = Particle(family=args.particle, state=args.state, mass_override_GeV=mass_ov,
                 pp_params=pp_params_obj)

    # QuenchParams setup
    alpha_cst = CPL.alpha_s_provider(mode="constant", alpha0=0.5)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    qp_base = QF.QuenchParams(
        qhat0=0.075 if args.particle=="charmonia" else 0.07,
        lp_fm=1.5, 
        LA_fm=Lmb, LB_fm=Lmb if args.system == "AA" else 0.0, 
        lambdaQCD=0.25, roots_GeV=roots, 
        alpha_of_mu=alpha_cst, alpha_scale="mT",
        use_hard_cronin=True, mapping="exp",
        system=kind,
        device=torch.device(device)
    )

    file_prefix = f"{args.particle}_{args.state}_{args.system}_{int(args.energy)}GeV"
    
    # ---------------------------------------------------------
    # 1. RpX vs y (Integrated over pT)
    # ---------------------------------------------------------
    y_edges = getattr(cfg, "y_edges", np.linspace(-5.0, 5.0, 41))
    pt_range = getattr(cfg, "pt_range_integrated", getattr(cfg, "pt_range_y_integrated", (0.0, 10.0)))
    y_plot_lim = getattr(cfg, "y_plot_lim", (-5.2, 5.2))
    
    print(f"\nComputing vs y for pT range: {pt_range}...")
    y_cent, bands_y, labels_y = EC.rpa_band_vs_y(
        P, roots, qp_base, gl, cent_bins, 
        y_edges, pt_range, 
        q0_pair=cfg.q0_pair,
        components=("loss","broad","eloss_broad"),
        Ny_bin=cfg.integration_steps['Ny_bin'], 
        Npt_bin=cfg.integration_steps['Npt_bin'],
        kind=kind
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    axes = axes.flatten()
    plot_labels = ["MB"] + [l for l in labels_y if l != "MB"]

    sys_title = f"{args.system} @ $\\sqrt{{s_{{NN}}}}$={args.energy} GeV"
    for i, lab in enumerate(plot_labels):
        if i >= len(axes): break
        ax = axes[i]
        feature_label = f"{lab}\n$p_T \\in [{pt_range[0]}, {pt_range[1]}]$ GeV"
        PU.plot_components_panel(
            ax, y_cent, 
            bands_y["loss"], bands_y["broad"], bands_y["eloss_broad"],
            label_key=lab, feature_label=feature_label,
            xlabel=r"$y_{COM}$", ylabel=rf"$R_{{{args.system}}}$"
        )
        ax.set_ylim(0., 2.0 if args.system == "AA" else 1.5)
        ax.set_xlim(y_plot_lim[0], y_plot_lim[1])

        if i == 0:
            ax.legend(fontsize=9, loc='lower center', frameon=False)
            ax.text(0.05, 0.95, sys_title, transform=ax.transAxes, 
                    fontsize=11, fontweight='bold', va='top')

    if len(plot_labels) < len(axes):
        for j in range(len(plot_labels), len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    # 1a. Save full grid to centralities
    fig.savefig(os.path.join(centralities_dir, f"{file_prefix}_vs_y.png"), dpi=150)
    fig.savefig(os.path.join(centralities_dir, f"{file_prefix}_vs_y.pdf"))
    plt.close(fig)
    
    # 1b. Save MB-only to minbias
    fig_mb, ax_mb = plt.subplots(figsize=(6, 5), dpi=150)
    feat_mb = f"MB\n$p_T \\in [{pt_range[0]}, {pt_range[1]}]$ GeV"
    PU.plot_components_panel(
        ax_mb, y_cent, 
        bands_y["loss"], bands_y["broad"], bands_y["eloss_broad"],
        label_key="MB", feature_label=feat_mb,
        xlabel=r"$y_{COM}$", ylabel=rf"$R_{{{args.system}}}$"
    )
    ax_mb.set_ylim(0., 2.0 if args.system == "AA" else 1.5)
    ax_mb.set_xlim(y_plot_lim[0], y_plot_lim[1])
    ax_mb.legend(fontsize=9, loc='lower center', frameon=False)
    ax_mb.text(0.05, 0.95, sys_title, transform=ax_mb.transAxes, fontsize=11, fontweight='bold', va='top')
    plt.tight_layout()
    fig_mb.savefig(os.path.join(minbias_dir, f"{file_prefix}_vs_y_MB.png"), dpi=150)
    fig_mb.savefig(os.path.join(minbias_dir, f"{file_prefix}_vs_y_MB.pdf"))
    plt.close(fig_mb)

    print(f"Saved vs_y plots: grid in centralities, MB-only in minbias.")

    # Save CSVs
    save_bands_to_csv(minbias_dir, f"{file_prefix}_vs_y_MB", y_cent, bands_y, ["MB"])
    save_bands_to_csv(centralities_dir, f"{file_prefix}_vs_y", y_cent, bands_y, labels_y)

    # ---------------------------------------------------------
    # 2. RpX vs pT (in Rapidity Windows)
    # ---------------------------------------------------------
    pT_edges = cfg.pT_edges
    pT_plot_lim = getattr(cfg, "pT_plot_lim", (0.0, 15.0))
    pT_cent_list = []
    mb_pt_results = [] # To store (y_win, bands_pt)
    
    for y_win in rap_windows:
        print(f"\nComputing vs pT for y in {y_win}...")
        pT_cent, bands_pt, labels_pt = EC.rpa_band_vs_pT(
            P, roots, qp_base, gl, cent_bins, 
            pT_edges, y_win, 
            q0_pair=cfg.q0_pair,
            components=("loss","broad","eloss_broad"),
            Ny_bin=cfg.integration_steps['Ny_bin'], 
            Npt_bin=cfg.integration_steps['Npt_bin'],
            kind=kind
        )
        pT_cent_list.append(pT_cent)
        mb_pt_results.append((y_win, bands_pt))

        # 2a. Save full grid to centralities
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
        axes = axes.flatten()
        plot_labels = ["MB"] + [l for l in labels_pt if l != "MB"]

        for i, lab in enumerate(plot_labels):
            if i >= len(axes): break
            ax = axes[i]
            feature_label = f"{lab}\n${y_win[0]} < y < {y_win[1]}$"
            PU.plot_components_panel(
                ax, pT_cent, 
                bands_pt["loss"], bands_pt["broad"], bands_pt["eloss_broad"],
                label_key=lab, feature_label=feature_label,
                xlabel=r"$p_T$ [GeV]", ylabel=rf"$R_{{{args.system}}}$"
            )
            ax.set_ylim(0., 2.0)
            ax.set_xlim(pT_plot_lim[0], pT_plot_lim[1])

            if i == 0:
                ax.legend(fontsize=9, loc='lower right', frameon=False)
                ax.text(0.05, 0.95, sys_title, transform=ax.transAxes, 
                        fontsize=11, fontweight='bold', va='top')

        if len(plot_labels) < len(axes):
            for j in range(len(plot_labels), len(axes)):
                axes[j].axis('off')

        plt.tight_layout()
        y_label = f"y_{y_win[0]}_{y_win[1]}"
        fig.savefig(os.path.join(centralities_dir, f"{file_prefix}_vs_pT_{y_label}.png"), dpi=150)
        fig.savefig(os.path.join(centralities_dir, f"{file_prefix}_vs_pT_{y_label}.pdf"))
        plt.close(fig)
        
        # Save CSV to centralities (all) and minbias (MB only)
        save_bands_to_csv(minbias_dir, f"{file_prefix}_vs_pT_{y_label}_MB", pT_cent, bands_pt, ["MB"])
        save_bands_to_csv(centralities_dir, f"{file_prefix}_vs_pT_{y_label}", pT_cent, bands_pt, labels_pt)

    # 2b. MinBias Rapidity Comparison Plot
    print("\nCreating MB Rapidity Comparison Plot...")
    fig_comp, ax_comp = plt.subplots(figsize=(7, 6), dpi=150)
    colors = ['#D32F2F', '#1976D2', '#388E3C', '#FBC02D', '#7B1FA2']
    
    for idx, (y_win, b_pt) in enumerate(mb_pt_results):
        pT_c = pT_cent_list[idx]
        y_str = f"${y_win[0]} < y < {y_win[1]}$"
        c_vals, lo_vals, hi_vals = b_pt["eloss_broad"]
        ax_comp.plot(pT_c, c_vals["MB"], color=colors[idx % len(colors)], lw=2, label=y_str)
        ax_comp.fill_between(pT_c, lo_vals["MB"], hi_vals["MB"], color=colors[idx % len(colors)], alpha=0.2)

    PU.format_publication_ax(ax_comp, r"$p_T$ [GeV]", rf"$R_{{{args.system}}}$ (Total)", ylim=(0, 2.0))
    ax_comp.set_xlim(pT_plot_lim[0], pT_plot_lim[1])
    ax_comp.legend(fontsize=10, loc='lower right', frameon=False, title="MB Rapidity Comparison")
    ax_comp.text(0.05, 0.95, sys_title, transform=ax_comp.transAxes, fontsize=12, fontweight='bold', va='top')
    plt.tight_layout()
    fig_comp.savefig(os.path.join(minbias_dir, f"{file_prefix}_vs_pT_rapidity_comparison_MB.png"), dpi=150)
    fig_comp.savefig(os.path.join(minbias_dir, f"{file_prefix}_vs_pT_rapidity_comparison_MB.pdf"))
    plt.close(fig_comp)

    # ---------------------------------------------------------
    # 3. RpX vs Centrality (in Rapidity Windows)
    # ---------------------------------------------------------
    pT_int = np.array([pt_range[0], pt_range[1]])
    
    fig, axes = plt.subplots(1, len(rap_windows), figsize=(5 * len(rap_windows), 4.5), dpi=150)
    if len(rap_windows) == 1: axes = [axes]
    
    for i, y_win in enumerate(rap_windows):
        ax = axes[i]
        print(f"\nComputing vs Cent for y in {y_win}...")
        _, bands_all, labels_all = EC.rpa_band_vs_pT(
            P, roots, qp_base, gl, cent_bins, 
            pT_int, y_win, 
            q0_pair=cfg.q0_pair,
            components=("loss","broad","eloss_broad"),
            Ny_bin=cfg.integration_steps['Ny_bin'], 
            Npt_bin=cfg.integration_steps['Npt_bin'],
            kind=kind
        )

        def extract(dic, comp_name):
            c, lo, hi = {}, {}, {}
            src_c, src_lo, src_hi = dic[comp_name]
            for k in src_c.keys():
                 c[k] = src_c[k][0]
                 lo[k] = src_lo[k][0]
                 hi[k] = src_hi[k][0]
            return c, lo, hi

        RL_c, RL_lo, RL_hi = extract(bands_all, "loss")
        RB_c, RB_lo, RB_hi = extract(bands_all, "broad")
        RT_c, RT_lo, RT_hi = extract(bands_all, "eloss_broad")

        RMB_loss = (RL_c.get("MB",0), RL_lo.get("MB",0), RL_hi.get("MB",0))
        RMB_broad = (RB_c.get("MB",0), RB_lo.get("MB",0), RB_hi.get("MB",0))
        RMB_tot = (RT_c.get("MB",0), RT_lo.get("MB",0), RT_hi.get("MB",0))

        PU.plot_RpA_vs_centrality_components_band(
            cent_bins, labels_all,
            RL_c, RL_lo, RL_hi, RMB_loss,
            RB_c, RB_lo, RB_hi, RMB_broad,
            RT_c, RT_lo, RT_hi, RMB_tot,
            show=("loss","broad","eloss_broad"),
            ax=ax,
            note=f"${y_win[0]} < y < {y_win[1]}$\n{sys_title}"
        )
        PU.format_publication_ax(ax, "Centrality (%)", rf"$R_{{{args.system}}}$", ylim=(0, 1.5))

        # Save CSV for vs_centrality in this window
        y_label = f"y_{y_win[0]}_{y_win[1]}"
        save_centrality_to_csv(minbias_dir, f"{file_prefix}_vs_centrality_{y_label}", 
                               cent_bins, labels_all, bands_all, filter_labels=["MB"])
        save_centrality_to_csv(centralities_dir, f"{file_prefix}_vs_centrality_{y_label}", 
                               cent_bins, labels_all, bands_all)

    plt.tight_layout()
    # Save centrality plots ONLY to centralities_dir
    fig.savefig(os.path.join(centralities_dir, f"{file_prefix}_vs_centrality.png"), dpi=150)
    fig.savefig(os.path.join(centralities_dir, f"{file_prefix}_vs_centrality.pdf"))
    
    plt.close(fig)
    print(f"Saved vs_centrality plots to centralities folder.")
    print("\nAll done!")

if __name__ == "__main__":
    main()
