import gzip
import pandas as pd

def load_data(fpath, n_cols=8):
    rows = []
    with gzip.open(fpath, 'rt') as f:
        meta = None
        for i, line in enumerate(f):
            if i % 2 == 0:
                meta = list(map(float, line.split()))
            else:
                data = list(map(float, line.split()))
                if len(data) == 8:
                    if data[-1] == 0: # S-wave init
                        rows.append({'pt': meta[4], 'y': meta[6], 'ups1S': data[0], 'is_peripheral': (data[0] >= 0.999)})
                elif len(data) == 14:
                    if data[-1] == 0: # S-wave init
                        rows.append({'pt': meta[4], 'y': meta[6], 'ups1S': data[0], 'is_peripheral': (data[0] >= 0.999)})
    return pd.DataFrame(rows)

df_no = load_data('/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile_partial.gz')
df_w  = load_data('/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile-avg.gz')

print(f"Total noReg mean R_AA: {df_no['ups1S'].mean():.4f}")
print(f"noReg mean pT: {df_no['pt'].mean():.4f}")
print(f"noReg mean y: {df_no['y'].mean():.4f}")
print(f"noReg purely unsuppressed fraction: {df_no['is_peripheral'].mean():.4f}")

print("--------------------------------------------------")

print(f"Total wReg mean R_AA: {df_w['ups1S'].mean():.4f}")
print(f"wReg mean pT: {df_w['pt'].mean():.4f}")
print(f"wReg mean y: {df_w['y'].mean():.4f}")
print(f"wReg purely unsuppressed fraction: {df_w['is_peripheral'].mean():.4f}")

