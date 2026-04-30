import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Paths
ROOT_DIR = "/mnt/workstation/bottomonia_combined_analysis"
PBPB_DIR = os.path.join(ROOT_DIR, "outputs", "qtraj_nlo", "PbPb5023")
OO_DIR = os.path.join(ROOT_DIR, "outputs", "qtraj_oo_5p36tev")
THESIS_DIR = os.path.join(ROOT_DIR, "outputs", "thesis_synthesis")
os.makedirs(THESIS_DIR, exist_ok=True)

# CMS Data (for PbPb context)
CMS_DATA_DIR = os.path.join(ROOT_DIR, "inputs/qtraj_inputs/PbPb5023/data")

def load_data():
    results = {}
    # PbPb
    for k in ["k3", "k4"]:
        path = os.path.join(PBPB_DIR, f"raavsnpart_{k}.csv")
        if os.path.exists(path):
            results[f"PbPb_{k}"] = pd.read_csv(path)
            
    # OO (Min-Bias/Single Bin)
    for reg in ["wReg", "noReg"]:
        path = os.path.join(OO_DIR, f"raa_vs_pt_OO_5p36TeV_{reg}.csv")
        if os.path.exists(path):
            # For Npart plot, we need to average RAA over pT or just pick min-bias value.
            # But wait, run_oo_5p36tev didn't output a single MB value.
            # I'll just load the vs_pt and take the average (rough estimate for <b>)
            df = pd.read_csv(path)
            results[f"OO_{reg}"] = df
            
    return results

def plot_system_comparison():
    res = load_data()
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 15,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 1. PbPb Band (k3-k4)
    if "PbPb_k3" in res and "PbPb_k4" in res:
        df3 = res["PbPb_k3"]
        df4 = res["PbPb_k4"]
        # Assuming they have the same Npart points
        ax.fill_between(df3["Npart"], df3["RAA_1S"], df4["RAA_1S"], 
                        alpha=0.3, color="tab:blue", label=r"Pb-Pb 5.02 TeV (QTraj $\kappa \in [3,4]$)")
        
    # 2. OO points (Integrated/Average over pT)
    # Npart for OO was ~7.5.
    if "OO_wReg" in res:
        df = res["OO_wReg"]
        # Simple mean as proxy for MB (rough)
        val = df["RAA_1S"].mean()
        ax.errorbar([7.5], [val], color="tab:red", fmt="*", ms=12, 
                    label="OO 5.36 TeV MB (wReg, QTraj)")
        
    if "OO_noReg" in res:
        df = res["OO_noReg"]
        val = df["RAA_1S"].mean()
        ax.errorbar([7.5], [val], color="tab:blue", fmt="s", ms=8, alpha=0.5,
                    label="OO 5.36 TeV MB (noReg, QTraj)")

    # 3. CMS data for PbPb baseline
    cms_path = os.path.join(CMS_DATA_DIR, "CMS2019-Y1s-npart.tsv")
    if os.path.exists(cms_path):
        d = np.loadtxt(cms_path)
        ax.errorbar(d[:, 0], d[:, 1], yerr=np.sqrt(d[:, 2]**2 + d[:, 4]**2), 
                    fmt="o", color="black", ms=5, label="CMS 2019 (Pb-Pb)")

    ax.axhline(1.0, color="gray", linestyle="--")
    ax.set_xlabel(r"$N_{\mathrm{part}}$")
    ax.set_ylabel(r"$R_{AA}[\Upsilon(1S)]$")
    ax.set_title(r"System-Size Dependence of $\Upsilon(1S)$ Suppression")
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)
    
    plt.savefig(os.path.join(THESIS_DIR, "raa_system_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(THESIS_DIR, "raa_system_comparison.png"), bbox_inches="tight", dpi=150)
    print(f"Saved synthesis plot to {THESIS_DIR}")

if __name__ == "__main__":
    plot_system_comparison()
