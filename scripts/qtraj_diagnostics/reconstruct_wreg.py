import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
from pathlib import Path
import os
import matplotlib.ticker as ticker

ROOT = Path('/mnt/workstation/bottomonia_combined_analysis')
DPI = 300
STATES = ['ups1S', 'ups2S', 'ups3S']
P_STATES = ['chi_b1P', 'chi_b2P']
Y_EDGES = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
P_EDGES = [0, 2, 4, 6, 8, 12, 16, 20]

def step_from_centers(centers, values):
    dx = centers[1] - centers[0]
    xe = np.zeros(len(centers) + 1)
    xe[:-1] = centers - dx/2.0
    xe[-1] = centers[-1] + dx/2.0
    ys = np.zeros(len(values) + 1)
    ys[:-1] = values
    ys[-1] = values[-1]
    return xe, ys

def load_data(fpath):
    rows = []
    with gzip.open(fpath, 'rt') as f:
        meta = None
        for i, line in enumerate(f):
            if i % 2 == 0:
                meta = list(map(float, line.split()))
            else:
                data = list(map(float, line.split()))
                if len(data) == 8:
                    if data[-1] == 0:
                        rows.append({'x0': meta[1], 'pt': meta[4], 'y': meta[6], 'initL': data[-1],
                                     'ups1S': data[0], 'ups2S': data[1], 'ups3S': data[3], 'chi_b1P': 0.0, 'chi_b2P': 0.0})
                    elif data[-1] == 1:
                        rows.append({'x0': meta[1], 'pt': meta[4], 'y': meta[6], 'initL': data[-1],
                                     'ups1S': 0.0, 'ups2S': 0.0, 'ups3S': 0.0, 'chi_b1P': data[0], 'chi_b2P': data[1]})
                elif len(data) == 14:
                    if data[-1] == 0:
                        rows.append({'x0': meta[1], 'pt': meta[4], 'y': meta[6], 'initL': data[-1],
                                     'ups1S': data[0], 'ups2S': data[2], 'ups3S': data[6], 'chi_b1P': 0.0, 'chi_b2P': 0.0})
                    elif data[-1] == 1:
                        rows.append({'x0': meta[1], 'pt': meta[4], 'y': meta[6], 'initL': data[-1],
                                     'ups1S': 0.0, 'ups2S': 0.0, 'ups3S': 0.0, 'chi_b1P': data[0], 'chi_b2P': data[2]})
    return pd.DataFrame(rows)

print("Loading raw datastores ...")
df_no = load_data(ROOT / 'inputs' / 'qtraj_nlo_run1_OO_5.36_kap6_noReg' / 'datafile_partial.gz')
df_w  = load_data(ROOT / 'inputs' / 'qtraj-nlo-run2-00-5.36-kap6-wReg' / 'datafile-avg.gz')

df_no_core = df_no[np.abs(df_no['x0']) < 0.2]

def binned_means(df, var, edges, initL):
    subs = df[df['initL'] == initL].copy()
    subs['bin'] = pd.cut(subs[var], bins=edges)
    return subs.groupby('bin', observed=False).mean(numeric_only=True)

y_centers = 0.5 * (np.array(Y_EDGES[:-1]) + np.array(Y_EDGES[1:]))
pt_centers = 0.5 * (np.array(P_EDGES[:-1]) + np.array(P_EDGES[1:]))

res_y = {}
for st in STATES + P_STATES:
    iL = 0 if st in STATES else 1
    no_full = binned_means(df_no, 'y', Y_EDGES, iL)[st].values
    no_core = binned_means(df_no_core, 'y', Y_EDGES, iL)[st].values
    w_core = binned_means(df_w, 'y', Y_EDGES, iL)[st].values
    
    # Differential enhancement
    epsilon = w_core / np.where(no_core == 0, 1, no_core)
    epsilon = np.nan_to_num(epsilon, nan=1.0)
    w_rebuilt = no_full * epsilon
    
    res_y[st] = {'noReg': no_full, 'wReg': w_rebuilt, 'noRegCore': no_core, 'wRegCore': w_core}

res_pt = {}
for st in STATES + P_STATES:
    iL = 0 if st in STATES else 1
    df_no_cut = df_no[np.abs(df_no['y']) <= 2.4]
    df_no_core_cut = df_no_core[np.abs(df_no_core['y']) <= 2.4]
    df_w_cut = df_w[np.abs(df_w['y']) <= 2.4]
    
    no_full = binned_means(df_no_cut, 'pt', P_EDGES, iL)[st].values
    no_core = binned_means(df_no_core_cut, 'pt', P_EDGES, iL)[st].values
    w_core = binned_means(df_w_cut, 'pt', P_EDGES, iL)[st].values
    
    epsilon = w_core / np.where(no_core == 0, 1, no_core)
    epsilon = np.nan_to_num(epsilon, nan=1.0)
    w_rebuilt = no_full * epsilon
    
    res_pt[st] = {'noReg': no_full, 'wReg': w_rebuilt}

OUT_DIR = ROOT / 'outputs' / 'cnm_hnm' / 'integrated_qtraj_only'
os.makedirs(OUT_DIR, exist_ok=True)

print("Exporting rebuilt CSVs ...")
y_rows = []
for st in STATES + P_STATES:
    for tag in ['noReg', 'wReg']:
        for i, val in enumerate(res_y[st][tag]):
            y_rows.append({'state': st, 'type': tag, 'y_min': Y_EDGES[i], 'y_max': Y_EDGES[i+1], 'R_AA': val})
pd.DataFrame(y_rows).to_csv(OUT_DIR / "QTraj_HNM_vs_y_OO_5p36TeV_Rebuilt.csv", index=False)

pt_rows = []
for st in STATES + P_STATES:
    for tag in ['noReg', 'wReg']:
        for i, val in enumerate(res_pt[st][tag]):
            pt_rows.append({'state': st, 'type': tag, 'pt_min': P_EDGES[i], 'pt_max': P_EDGES[i+1], 'R_AA': val})
pd.DataFrame(pt_rows).to_csv(OUT_DIR / "QTraj_HNM_vs_pT_OO_5p36TeV_Rebuilt.csv", index=False)

def plot_rebuilt(states, name):
    fig_y, axes_y = plt.subplots(1, len(states), figsize=(5*len(states), 5), sharey=True, dpi=DPI)
    if len(states) == 1: axes_y = [axes_y]
    plt.subplots_adjust(wspace=0)
    for i, st in enumerate(states):
        ax = axes_y[i]
        
        xe, ys = step_from_centers(y_centers, res_y[st]['noReg'])
        ax.step(xe, ys, lw=2, color='red', ls='--', label='noReg (Full 2D)' if i==0 else None, where='post')
        xe, ys = step_from_centers(y_centers, res_y[st]['wReg'])
        ax.step(xe, ys, lw=2, color='blue', label='wReg (2D Extrapolated)' if i==0 else None, where='post')
        
        ax.text(0.95, 0.95, rf"$\mathbf{{{st}}}$", transform=ax.transAxes, ha='right', va='top', fontsize=14)
        ax.set_xlabel(r"$y$", fontsize=15)
        ax.set_ylim(0, 1.2); ax.set_xlim(-4.5, 4.5)
        if i == 0:
            ax.set_ylabel(r"$R_{pA}$ (Rebuilt)", fontsize=16)
            ax.legend(loc='lower left', frameon=False)
            
    fig_y.savefig(OUT_DIR / f"{name}_RAA_HNM_QTraj_vs_y_Rebuilt.png", bbox_inches='tight')
    
    fig_pt, axes_pt = plt.subplots(1, len(states), figsize=(5*len(states), 5), sharey=True, dpi=DPI)
    if len(states) == 1: axes_pt = [axes_pt]
    plt.subplots_adjust(wspace=0)
    for i, st in enumerate(states):
        ax = axes_pt[i]
        
        xe, ys = step_from_centers(pt_centers, res_pt[st]['noReg'])
        ax.step(xe, ys, lw=2, color='red', ls='--', label='noReg' if i==0 else None, where='post')
        xe, ys = step_from_centers(pt_centers, res_pt[st]['wReg'])
        ax.step(xe, ys, lw=2, color='blue', label='wReg' if i==0 else None, where='post')
        
        ax.text(0.95, 0.95, rf"$\mathbf{{{st}}}$", transform=ax.transAxes, ha='right', va='top', fontsize=14)
        ax.set_xlabel(r"$p_T$ [GeV]", fontsize=15)
        ax.set_ylim(0, 1.2); ax.set_xlim(0, 20)
        if i == 0:
            ax.set_ylabel(r"$R_{pA}$ (Rebuilt)", fontsize=16)
            ax.legend(loc='lower left', frameon=False)
            
    fig_pt.savefig(OUT_DIR / f"{name}_RAA_HNM_QTraj_vs_pT_Rebuilt.png", bbox_inches='tight')

print("Plotting figures...")
plot_rebuilt(STATES, "Upsilon")
plot_rebuilt(P_STATES, "Chib")
print("DONE!")
