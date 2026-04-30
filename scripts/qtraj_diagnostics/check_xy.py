import pandas as pd
import gzip

def get_xy(fpath):
    coords = []
    with gzip.open(fpath, 'rt') as f:
        meta = None
        for i, line in enumerate(f):
            if i % 2 == 0:
                meta = list(map(float, line.split()))
            else:
                data = list(map(float, line.split()))
                if data[-1] == 0:
                    coords.append({'x0': meta[1], 'y0': meta[2], 'b': meta[0], 'pt': meta[4]})
    return pd.DataFrame(coords)

df_no = get_xy('/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile_partial.gz')
df_w = get_xy('/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile-avg.gz')

print(f"noReg mean r: {(df_no['x0']**2 + df_no['y0']**2).mean():.4f}")
print(f"wReg mean r: {(df_w['x0']**2 + df_w['y0']**2).mean():.4f}")
print(f"noReg mean b: {df_no['b'].mean():.4f}")
print(f"wReg mean b: {df_w['b'].mean():.4f}")
