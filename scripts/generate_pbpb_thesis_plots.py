import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directories
OUT_DIR = "/mnt/workstation/bottomonia_combined_analysis/outputs/thesis_pbpb"
os.makedirs(OUT_DIR, exist_ok=True)
PBPB_RESULTS = "/mnt/workstation/bottomonia_combined_analysis/outputs/qtraj_nlo/PbPb5023"

# --- EXPERIMENTAL DATA (PbPb 5.02 TeV CMS) ---

# Npart Data (CMS 2018)
npart_1s = np.array([11.43, 31.21, 54.42, 87.19, 131.0, 188.2, 262.3, 331.5, 382.3])
raa_1s = np.array([0.792, 0.922, 0.609, 0.524, 0.485, 0.402, 0.324, 0.321, 0.319])
err_1s = np.array([0.177, 0.150, 0.083, 0.058, 0.045, 0.047, 0.027, 0.029, 0.027])

npart_2s = np.array([11.43, 31.21, 54.42, 87.19, 131.0, 188.2, 262.3, 331.5, 382.3])
raa_2s = np.array([0.596, 0.439, 0.250, 0.177, 0.131, 0.096, 0.100, 0.073, 0.078])
err_2s = np.array([0.128, 0.082, 0.046, 0.030, 0.024, 0.022, 0.019, 0.021, 0.020])

npart_3s = np.array([11.43, 42.81, 109.1, 269.1])
raa_3s = np.array([0.534, 0.290, 0.109, 0.051])
err_3s = np.array([0.194, 0.065, 0.035, 0.019])

# pT Data (CMS 2018 - MB 0-100%)
pt_1s = np.array([1.0, 3.0, 5.0, 7.5, 10.5, 21.0])
raa_pt_1s = np.array([0.301, 0.362, 0.388, 0.402, 0.422, 0.425])
err_pt_1s = np.array([0.126, 0.045, 0.048, 0.042, 0.057, 0.044])

pt_2s = np.array([1.5, 4.5, 7.5, 12.0, 22.5])
raa_pt_2s = np.array([0.087, 0.122, 0.098, 0.126, 0.140])
err_pt_2s = np.array([0.019, 0.019, 0.021, 0.021, 0.022])

# Plotting Function
def create_npart_plot():
    # Load Theory
    k3 = pd.read_csv(os.path.join(PBPB_RESULTS, "raavsnpart_k3.csv"))
    k4 = pd.read_csv(os.path.join(PBPB_RESULTS, "raavsnpart_k4.csv"))
    
    plt.figure(figsize=(9, 7))
    plt.yscale("log")
    
    # THEORY BANDS
    plt.fill_between(k3['Npart'], k3['RAA_1S'], k4['RAA_1S'], color='tab:blue', alpha=0.2)
    plt.fill_between(k3['Npart'], k3['RAA_2S'], k4['RAA_2S'], color='tab:red', alpha=0.2)
    plt.fill_between(k3['Npart'], k3['RAA_3S'], k4['RAA_3S'], color='tab:green', alpha=0.2)
    
    # THEORY LINES (central)
    plt.plot(k3['Npart'], (k3['RAA_1S']+k4['RAA_1S'])/2, color='tab:blue', lw=1.5, label=r'$\Upsilon(1S)$ - Theory')
    plt.plot(k3['Npart'], (k3['RAA_2S']+k4['RAA_2S'])/2, color='tab:red', lw=1.5, label=r'$\Upsilon(2S)$ - Theory')
    plt.plot(k3['Npart'], (k3['RAA_3S']+k4['RAA_3S'])/2, color='tab:green', lw=1.5, label=r'$\Upsilon(3S)$ - Theory')
    
    # EXPERIMENTAL POINTS
    plt.errorbar(npart_1s, raa_1s, yerr=err_1s, fmt='bo', markersize=6, capsize=3, label='CMS 1S Exp')
    plt.errorbar(npart_2s, raa_2s, yerr=err_2s, fmt='rs', markersize=6, capsize=3, label='CMS 2S Exp')
    plt.errorbar(npart_3s, raa_3s, yerr=err_3s, fmt='g^', markersize=6, capsize=3, label='CMS 3S Exp')
    
    plt.xlabel(r"$N_{\rm part}$", fontsize=16)
    plt.ylabel(r"$R_{AA}$", fontsize=16)
    plt.xlim(0, 420)
    plt.ylim(0.01, 1.5)
    plt.legend(frameon=False, loc="lower left", fontsize=12)
    plt.tick_params(direction='in', top=True, right=True, labelsize=14)
    # No title, No grid
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pbpb_5tev_raa_vs_npart_thesis.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "pbpb_5tev_raa_vs_npart_thesis.png"), dpi=300)
    plt.close()

def create_pt_plot():
    # Load Theory
    k3 = pd.read_csv(os.path.join(PBPB_RESULTS, "raavspt_mb_k3.csv"))
    k4 = pd.read_csv(os.path.join(PBPB_RESULTS, "raavspt_mb_k4.csv"))
    
    plt.figure(figsize=(9, 7))
    plt.yscale("log")
    
    # THEORY BANDS
    plt.fill_between(k3['pt'], k3['RAA_1S'], k4['RAA_1S'], color='tab:blue', alpha=0.2)
    plt.fill_between(k3['pt'], k3['RAA_2S'], k4['RAA_2S'], color='tab:red', alpha=0.2)
    plt.fill_between(k3['pt'], k3['RAA_3S'], k4['RAA_3S'], color='tab:green', alpha=0.2)
    
    # THEORY LINES (central)
    plt.plot(k3['pt'], (k3['RAA_1S']+k4['RAA_1S'])/2, color='tab:blue', lw=1.5, label=r'$\Upsilon(1S)$ - Theory')
    plt.plot(k3['pt'], (k3['RAA_2S']+k4['RAA_2S'])/2, color='tab:red', lw=1.5, label=r'$\Upsilon(2S)$ - Theory')
    plt.plot(k3['pt'], (k3['RAA_3S']+k4['RAA_3S'])/2, color='tab:green', lw=1.5, label=r'$\Upsilon(3S)$ - Theory')
    
    # EXPERIMENTAL POINTS
    plt.errorbar(pt_1s, raa_pt_1s, yerr=err_pt_1s, fmt='bo', markersize=6, capsize=3, label='CMS 1S Exp')
    plt.errorbar(pt_2s, raa_pt_2s, yerr=err_pt_2s, fmt='rs', markersize=6, capsize=3, label='CMS 2S Exp')
    
    plt.xlabel(r"$p_T$ [GeV]", fontsize=16)
    plt.ylabel(r"$R_{AA}$", fontsize=16)
    plt.xlim(0, 30)
    plt.ylim(0.01, 1.2)
    plt.legend(frameon=False, loc="lower right", fontsize=12)
    plt.tick_params(direction='in', top=True, right=True, labelsize=14)
    # No title, No grid
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pbpb_5tev_raa_vs_pt_thesis.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "pbpb_5tev_raa_vs_pt_thesis.png"), dpi=300)
    plt.close()

def create_ratio_plot():
    # Load Theory
    k3 = pd.read_csv(os.path.join(PBPB_RESULTS, "raavsnpart_k3.csv"))
    k4 = pd.read_csv(os.path.join(PBPB_RESULTS, "raavsnpart_k4.csv"))
    
    # Double Ratio: RAA[2S] / RAA[1S]
    ratio_k3 = k3['RAA_2S'] / k3['RAA_1S']
    ratio_k4 = k4['RAA_2S'] / k4['RAA_1S']
    
    plt.figure(figsize=(9, 7))
    
    plt.fill_between(k3['Npart'], ratio_k3, ratio_k4, color='tab:red', alpha=0.2)
    plt.plot(k3['Npart'], (ratio_k3+ratio_k4)/2, color='tab:red', lw=1.5, label=r'$\Upsilon(2S)/\Upsilon(1S)$ - Theory')
    
    # Exp Data for ratio
    exp_ratio_2s1s = raa_2s / raa_1s
    exp_err_ratio = exp_ratio_2s1s * np.sqrt((err_2s/raa_2s)**2 + (err_1s/raa_1s)**2)
    
    plt.errorbar(npart_1s, exp_ratio_2s1s, yerr=exp_err_ratio, fmt='rd', markersize=6, capsize=3, label='CMS 2S/1S Exp')
    
    plt.xlabel(r"$N_{\rm part}$", fontsize=16)
    plt.ylabel(r"Double Ratio $R_{AA}(2S)/R_{AA}(1S)$", fontsize=16)
    plt.xlim(0, 420)
    plt.ylim(0, 1)
    plt.legend(frameon=False, loc="upper right", fontsize=12)
    plt.tick_params(direction='in', top=True, right=True, labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pbpb_5tev_double_ratio_thesis.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "pbpb_5tev_double_ratio_thesis.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    create_npart_plot()
    create_pt_plot()
    create_ratio_plot()
    print(f"Thesis plots for PbPb 5 TeV generated in {OUT_DIR}")
