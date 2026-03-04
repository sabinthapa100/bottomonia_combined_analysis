#!/usr/bin/env python
# coding: utf-8

# # LHC O-O Energy Loss & Broadening (Verified Module)
# 
# This notebook computes $R_{pA}$ for O-O at $\sqrt{s_{NN}} = 5.02$ TeV using `eloss_cronin_centrality_test`.
# It implements strict publication-quality plotting standards:
# - Visualizing Energy Loss, pT Broadening, and Total effects.
# - Consistent axes (0-2), minor ticks, no grid.
# - Flexible pT ranges and rapidity windows.
# - **Calculation**: Full phase space ($y \in [-5,5], p_T \in [0,20]$).
# - **Plotting**: Truncated ranges to avoid edge artifacts.
# 

# In[ ]:


import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import importlib

# Add module path
sys.path.append(os.path.abspath("../eloss_code"))

import eloss_cronin_centrality_test as EC
import plotting_utils as PU
import system_configs as SC
importlib.reload(PU)
importlib.reload(SC)

from system_configs import OOConfig
from glauber_wrapper import GlauberWrapper
from particle import Particle, PPSpectrumParams

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

print("Modules imported and plotting style set. Calculation ranges extended.")


# ## 1. Setup System

# In[ ]:


# System Config
spec = OOConfig.spec_5TeV
roots = 5360.0
cent_bins = OOConfig.cent_bins_plotting
rap_windows = OOConfig.rapidity_windows

# Glauber
gl_wrapper = GlauberWrapper("pA", spec.roots_GeV, spec.A, spec.sigma_nn_mb)

# Particle
Upsilon = Particle(family="bottomonia", state="1S", mass_override_GeV=9.46,
                pp_params=PPSpectrumParams(p0=OOConfig.pp_params['p0'], 
                                         m=OOConfig.pp_params['m'], 
                                         n=OOConfig.pp_params['n']))

# Params
import quenching_fast as QF
import coupling as CPL
alpha_cst = CPL.alpha_s_provider(mode="constant", alpha0=0.5)
qp_base = QF.QuenchParams(
    qhat0=0.075, lp_fm=1.5, 
    LA_fm=10.0, LB_fm=10.0, 
    lambdaQCD=0.25, roots_GeV=roots, 
    alpha_of_mu=alpha_cst, alpha_scale="mT",
    use_hard_cronin=True, mapping="exp",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


# ## 2. $R_{pO}$ vs $y$ (Flexible pT, Component Visualization)

# In[ ]:


y_edges = np.linspace(-5.0, 5.0, 41)
pt_range = OOConfig.pt_range_integrated 

print(f"Computing vs y for pT range: {pt_range}...")
y_cent, bands_y, labels_y = EC.rpa_band_vs_y(
    Upsilon, roots, qp_base, gl_wrapper, cent_bins, 
    y_edges, pt_range, 
    q0_pair=OOConfig.q0_pair,
    components=("loss","broad","eloss_broad"),
    Ny_bin=OOConfig.integration_steps['Ny_bin'], 
    Npt_bin=OOConfig.integration_steps['Npt_bin']
)

# --- PLOT: Components for each Centrality ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
axes = axes.flatten()

plot_labels = ["MB"] + [l for l in labels_y if l != "MB"]

for i, lab in enumerate(plot_labels):
    if i >= len(axes): break
    ax = axes[i]
    PU.plot_components_panel(
        ax, y_cent, 
        bands_y["loss"], bands_y["broad"], bands_y["eloss_broad"],
        label_key=lab,
        feature_label=fr"{lab}" + "\n" + fr"$p_T \in [{pt_range[0]}, {pt_range[1]}]$ GeV",
        xlabel=r"$y$", ylabel=r"$R_{pO}$"
    )
    ax.set_ylim(0., 2.0)
    ax.set_xlim(-5.2, 5.2)

    if i == 0:
        ax.legend(fontsize=9, loc='lower center', frameon=False)
        ax.text(0.05, 0.95, "p+O @$\\sqrt{s_{NN}}=5.36$ TeV", transform=ax.transAxes, 
                fontsize=11, fontweight='bold', va='top')

plt.tight_layout()
plt.show()


# ## 3. $R_{pO}$ vs $p_T$ (in Rapidity Windows)
# 
# Calculation uses expanded `pT_edges` (up to 20 GeV), but plots are truncated to `pT_plot_lim` (e.g. 15 GeV).

# In[ ]:


pT_edges = OOConfig.pT_edges # Now covers 0-20 GeV

for y_win in rap_windows:
    print(f"Computing vs pT for y in {y_win}...")
    pT_cent, bands_pt, labels_pt = EC.rpa_band_vs_pT(
        Upsilon, roots, qp_base, gl_wrapper, cent_bins, 
        pT_edges, y_win, 
        q0_pair=OOConfig.q0_pair,
        components=("loss","broad","eloss_broad"),
        Ny_bin=OOConfig.integration_steps['Ny_bin'], 
        Npt_bin=OOConfig.integration_steps['Npt_bin']
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    axes = axes.flatten()
    plot_labels = ["MB"] + [l for l in labels_pt if l != "MB"]

    for i, lab in enumerate(plot_labels):
        if i >= len(axes): break
        ax = axes[i]
        PU.plot_components_panel(
            ax, pT_cent, 
            bands_pt["loss"], bands_pt["broad"], bands_pt["eloss_broad"],
            label_key=lab,
            feature_label=f"{lab}\n${y_win[0]} < y < {y_win[1]}$",
            xlabel=r"$p_T$ [GeV]", ylabel=r"$R_{pO}$"
        )
        ax.set_ylim(0., 2.0)
        # Use plotting limits from config (e.g. 0-15)
        ax.set_xlim(OOConfig.pT_plot_lim[0], OOConfig.pT_plot_lim[1])

        if i == 0:
             ax.legend(fontsize=9, loc='lower right', frameon=False)

    plt.tight_layout()
    plt.show()


# ## 4. $R_{pO}$ vs Centrality

# In[ ]:


# Integrated pT for Centrality dependence
pT_int = np.array([OOConfig.pt_range_integrated[0], OOConfig.pt_range_integrated[1]])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

for i, y_win in enumerate(rap_windows):
    ax = axes[i]
    print(f"Computing vs Cent for y={y_win}...")

    _, bands_all, labels_all = EC.rpa_band_vs_pT(
        Upsilon, roots, qp_base, gl_wrapper, cent_bins, 
        pT_int, y_win, 
        q0_pair=OOConfig.q0_pair,
        components=("loss","broad","eloss_broad"),
        Ny_bin=OOConfig.integration_steps['Ny_bin'], 
        Npt_bin=OOConfig.integration_steps['Npt_bin']
    )

    def extract(dic, comp_name):
        c, lo, hi = {}, {}, {}
        src_c, src_lo, src_hi = bands_all[comp_name]
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
        note=f"${y_win[0]} < y < {y_win[1]}$"
    )
    PU.format_publication_ax(ax, "Centrality (%)", r"$R_{pO}$", ylim=(0, 2.0))

plt.tight_layout()
plt.show()

