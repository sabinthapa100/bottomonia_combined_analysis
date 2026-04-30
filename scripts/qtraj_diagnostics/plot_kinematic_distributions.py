import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
import os

def load_kinematics(fpath):
    coords = []
    with gzip.open(fpath, 'rt') as f:
        meta = None
        for i, line in enumerate(f):
            if i % 2 == 0:
                meta = list(map(float, line.split()))
            else:
                data = list(map(float, line.split()))
                if meta[-1] == 0 or (len(data) == 14 and data[-1] == 0) or (len(data) == 8 and data[-1] == 0): 
                    # For safety, checking data[-1]==0 covers S-wave
                    # Only collect coordinates if it's evaluated
                    if (len(data) == 8 and data[-1] == 0) or (len(data) == 14 and data[-1] == 0):
                        coords.append({'pt': meta[4], 'y': meta[6]})
    return pd.DataFrame(coords)

print("Loading noReg (~80k)...")
df_no = load_kinematics('/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile_partial.gz')

print("Loading wReg (300k)...")
df_w = load_kinematics('/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile-avg.gz')

# Create Output Dirs
out_dir = "./outputs/cnm_hnm/integrated_qtraj_only"
os.makedirs(out_dir, exist_ok=True)

# pT Distribution
plt.figure(figsize=(8, 6))
plt.hist(df_w['pt'], bins=50, range=(0, 20), density=True, alpha=0.5, color='blue', label=f'wReg (N={len(df_w)})')
plt.hist(df_no['pt'], bins=50, range=(0, 20), density=True, alpha=0.5, color='red', label=f'noReg (N={len(df_no)})', histtype='step', lw=2)
plt.xlabel(r"$p_T$ [GeV]", fontsize=14)
plt.ylabel("Normalized Density", fontsize=14)
plt.title(r"Initital $p_T$ Sampling Distribution", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/pt_distribution.png", dpi=300)

# y Distribution
plt.figure(figsize=(8, 6))
plt.hist(df_w['y'], bins=50, range=(-5, 5), density=True, alpha=0.5, color='blue', label='wReg')
plt.hist(df_no['y'], bins=50, range=(-5, 5), density=True, alpha=0.5, color='red', label='noReg', histtype='step', lw=2)
plt.xlabel(r"Rapidity $y$", fontsize=14)
plt.ylabel("Normalized Density", fontsize=14)
plt.title(r"Initital Rapidity $y$ Sampling Distribution", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/y_distribution.png", dpi=300)

print(f"DONE! Distributions saved to {out_dir}")
