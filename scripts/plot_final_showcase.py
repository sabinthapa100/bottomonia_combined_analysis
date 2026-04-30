import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

ROOT = Path('/mnt/workstation/bottomonia_combined_analysis')
SMOOTH_DIR = ROOT / 'outputs' / 'cnm_hnm' / 'integrated_OO'
STEP_DIR = ROOT / 'outputs' / 'cnm_hnm' / 'final_qtraj_analysis_event_weighted'
OUT_DIR = ROOT / 'outputs' / 'final_results_summary_event_weighted'
os.makedirs(OUT_DIR, exist_ok=True)

STATES = ['raa1S', 'raa2S', 'raa3S']
STATE_LABELS = {
    'raa1S': r'$\Upsilon(1S)$',
    'raa2S': r'$\Upsilon(2S)$',
    'raa3S': r'$\Upsilon(3S)$'
}

def load_smooth(state_label, var):
    # Map state label to TAMU filename state
    smap = {'raa1S': 'ups1S', 'raa2S': 'ups2S', 'raa3S': 'ups3S'}
    state = smap[state_label]
    suff = "OO_5p36TeV.csv"
    vname = 'pt_mid' if var == 'pT' else var
    pre = f"{state}_RAA_HNM_only_vs"
    try:
        df_np = pd.read_csv(SMOOTH_DIR / f"{pre}_{vname}_NPWLC_{suff}")
        df_pe = pd.read_csv(SMOOTH_DIR / f"{pre}_{vname}_Pert_{suff}")
        return df_np, df_pe
    except: return None, None

def plot_variable(var, xlabel):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    plt.subplots_adjust(wspace=0.1)
    
    no_reg = pd.read_csv(STEP_DIR / f"QTraj_EventWeighted_RAA_vs_{var.lower()}_noReg.csv")
    w_reg = pd.read_csv(STEP_DIR / f"QTraj_EventWeighted_RAA_vs_{var.lower()}_wReg.csv")
    
    for i, state in enumerate(STATES):
        ax = axes[i]
        
        # Smooth Bands (TAMU MB)
        df_np, df_pe = load_smooth(state, var)
        if df_np is not None:
            x = df_np.iloc[:, 0].values
            y_np = df_np.iloc[:, 1].values
            y_pe = df_pe.iloc[:, 1].values
            ax.fill_between(x, np.minimum(y_np, y_pe), np.maximum(y_np, y_pe), color='orange', alpha=0.3, label='TAMU MB Band' if i==0 else None)

        # Step Bands (QTraj b=4.5 Rebuilt)
        # Note: CSV index is y_bin or pt_bin. We recover centers.
        vcol = 'y_bin' if var == 'y' else 'pt_bin'
        
        def get_xy(df, col):
            # Parse "(min, max]"
            raw = df[col].values
            centers = []
            for r in raw:
                r = r.strip('()[]')
                lo, hi = map(float, r.split(','))
                centers.append(0.5*(lo + hi))
            return np.array(centers), df[state].values

        xc, yc_no = get_xy(no_reg, vcol)
        xc, yc_w = get_xy(w_reg, vcol)
        
        # Step plot
        dx = xc[1] - xc[0]
        xe = np.concatenate([xc - dx/2.0, [xc[-1] + dx/2.0]])
        ys_no = np.concatenate([yc_no, [yc_no[-1]]])
        ys_w = np.concatenate([yc_w, [yc_w[-1]]])
        
        ax.step(xe, ys_no, where='post', color='red', ls='--', lw=2, label='QTraj noReg (b=4.5)' if i==0 else None)
        ax.step(xe, ys_w, where='post', color='blue', lw=2, label='QTraj wReg (b=4.5)' if i==0 else None)

        ax.set_ylim(0, 1.25)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.text(0.95, 0.95, STATE_LABELS[state], transform=ax.transAxes, ha='right', va='top', fontsize=16, fontweight='bold')
        if i == 0:
            ax.set_ylabel(r'$R_{AA}^{HNM}$', fontsize=16)
            ax.legend(loc='lower left', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"Fixed-Target Comparison: QTraj (b=4.5) vs TAMU (Min Bias MB)", fontsize=18)
    fig.savefig(OUT_DIR / f"Final_Weighted_Comparison_{var}.png", bbox_inches='tight', dpi=300)

print("Planting final plots ...")
plot_variable('y', 'Rapidity y')
plot_variable('pT', r'$p_T$ [GeV]')
print(f"DONE. Images in {OUT_DIR}")
