import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt

def load_qtraj(fpath):
    rows = []
    with gzip.open(fpath, 'rt') as f:
        meta = None
        for i, line in enumerate(f):
            if i % 2 == 0:
                meta = list(map(float, line.split()))
            else:
                data = list(map(float, line.split()))
                if data[-1] == 0: # S-wave init only
                    x0, y0 = meta[1], meta[2]
                    r = np.sqrt(x0**2 + y0**2)
                    rows.append({'r': r, 'ups1S': data[0]})
    return pd.DataFrame(rows)

print("Loading noReg (~80k)...")
df_no = load_qtraj('/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile_partial.gz')
print("Loading wReg (300k)...")
df_w = load_qtraj('/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile-avg.gz')

# Bin by r
bins = np.linspace(0, 4.0, 20)
df_no['r_bin'] = pd.cut(df_no['r'], bins)
df_w['r_bin'] = pd.cut(df_w['r'], bins)

no_mean = df_no.groupby('r_bin', observed=False)['ups1S'].mean().values
w_mean = df_w.groupby('r_bin', observed=False)['ups1S'].mean().values
centers = 0.5 * (bins[:-1] + bins[1:])

# Plot
plt.figure(figsize=(8, 6))
plt.plot(centers, w_mean, 'bo-', label='wReg (Regeneration ON)', lw=2)
plt.plot(centers, no_mean, 'ro--', label='noReg (Regeneration OFF)', lw=2)
plt.xlabel(r"Initial Distance from Center, $r = \sqrt{x_0^2 + y_0^2}$ [fm]", fontsize=14)
plt.ylabel(r"1S Survival Probability ($R_{AA}$)", fontsize=14)
plt.title(r"$\Upsilon$(1S) Survival vs. Geometry ($O+O$ $5.36$ TeV, $b=4.5$ fm)", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('/home/sawin/.gemini/antigravity/brain/573fe68a-0296-4647-bb8c-edc17bf8cd8b/artifacts/RAA_vs_radius.png', dpi=300)
print("Plot saved.")
