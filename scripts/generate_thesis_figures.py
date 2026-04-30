import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = "/mnt/workstation/bottomonia_combined_analysis"
PBPB_DIR = os.path.join(ROOT_DIR, "outputs/qtraj_nlo/PbPb5023")
OO_DIR = os.path.join(ROOT_DIR, "outputs/qtraj_oo_5p36tev")
FINAL_OUT = os.path.join(ROOT_DIR, "outputs/thesis_final_figures")
os.makedirs(FINAL_OUT, exist_ok=True)

def plot_raa_pt_comparison():
    # Load PbPb Min-Bias
    pbpb_k3 = pd.read_csv(os.path.join(PBPB_DIR, "raavspt_mb_k3.csv"))
    pbpb_k4 = pd.read_csv(os.path.join(PBPB_DIR, "raavspt_mb_k4.csv"))
    
    # Load OO
    oo_w = pd.read_csv(os.path.join(OO_DIR, "raavspt_OO_5p36TeV_wReg.csv"))
    oo_n = pd.read_csv(os.path.join(OO_DIR, "raavspt_OO_5p36TeV_noReg.csv"))
    
    plt.figure(figsize=(10, 7))
    plt.axhline(1.0, color='black', alpha=0.3)
    
    # PbPb Band
    plt.fill_between(pbpb_k3['pt'], pbpb_k3['RAA_1S'], pbpb_k4['RAA_1S'], alpha=0.2, label='Pb-Pb 5.02 TeV (0-100%)', color='tab:blue')
    
    # OO points
    plt.plot(oo_w['pt'], oo_w['RAA_1S'], 'r-o', label='O-O 5.36 TeV (wReg)')
    plt.plot(oo_n['pt'], oo_n['RAA_1S'], 'b--s', label='O-O 5.36 TeV (noReg)', alpha=0.6)
    
    plt.xlabel(r"$p_T$ [GeV]")
    plt.ylabel(r"$R_{AA}$")
    plt.title(r"$\Upsilon(1S)$ Nuclear Modification Factor vs $p_T$")
    plt.legend()
    plt.grid(True, alpha=0.1)
    plt.savefig(os.path.join(FINAL_OUT, "comparison_raa_pt.pdf"))
    plt.savefig(os.path.join(FINAL_OUT, "comparison_raa_pt.png"), dpi=200)

def plot_raa_y_oo():
    oo_w = pd.read_csv(os.path.join(OO_DIR, "raavsy_OO_5p36TeV_wReg.csv"))
    
    plt.figure(figsize=(8, 6))
    plt.plot(oo_w['y'], oo_w['RAA_1S'], 'r-o', label=r'$\Upsilon(1S)$')
    plt.plot(oo_w['y'], oo_w['RAA_2S'], 'g-s', label=r'$\Upsilon(2S)$')
    plt.plot(oo_w['y'], oo_w['RAA_3S'], 'b-^', label=r'$\Upsilon(3S)$')
    
    plt.axhline(1.0, color='black', alpha=0.2)
    plt.xlabel("Rapidity $y$")
    plt.ylabel(r"$R_{AA}$")
    plt.title("O-O 5.36 TeV: Rapidity Dependence (wReg)")
    plt.legend()
    plt.savefig(os.path.join(FINAL_OUT, "oo_raa_y.pdf"))
    plt.savefig(os.path.join(FINAL_OUT, "oo_raa_y.png"), dpi=200)

if __name__ == "__main__":
    plot_raa_pt_comparison()
    plot_raa_y_oo()
    print(f"Thesis final figures generated in {FINAL_OUT}")
