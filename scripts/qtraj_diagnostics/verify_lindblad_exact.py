import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path("/mnt/workstation/bottomonia_combined_analysis")
sys.path.append(str(ROOT / "cnm_hnm/cnm_prim_scripts"))
sys.path.append(str(ROOT / "hnm/qtraj-nlo/qtraj_out_analysis/src"))

from io import read_qtraj_file

f_noreg = ROOT / "inputs" / "qtraj-nlo-run2-00-5.36-kap6-noreg" / "datafile_partial.gz"
f_wreg = ROOT / "inputs" / "qtraj-nlo-run2-00-5.36-kap6-wReg" / "datafile-avg.gz"

print("Loading noReg...")
df_no = read_qtraj_file(f_noreg, 8, "mathematica", -5.0, 5.0)

print("Loading wReg...")
df_w = read_qtraj_file(f_wreg, 14, "qtavg", -5.0, 5.0)

# Merge on exact physical paths: x0, y0, pt, y
df_n = df_no.groupby(['x0', 'y0', 'pt', 'y']).mean().reset_index()
df_w = df_w.groupby(['x0', 'y0', 'pt', 'y']).mean().reset_index()

merged = pd.merge(df_n, df_w, on=['x0', 'y0', 'pt', 'y'], suffixes=('_no', '_w'))
print(f"Matched {len(merged)} tracks perfectly out of {len(df_n)} noReg and {len(df_w)} wReg.")

print(f"wReg mean R_AA: {merged['ups1S_w'].mean()}")
print(f"noReg mean R_AA: {merged['ups1S_no'].mean()}")

if merged['ups1S_w'].mean() < merged['ups1S_no'].mean():
    print("WARNING: wReg is STILL lower than noReg even when strictly matched!")
    print("THIS MEANS THERE IS A PHYSICAL ERROR OR PROCESSING BUG IN WREG.")
else:
    print("SUCCESS: wReg >= noReg on identical paths. The discrepancy is purely global sampling.")

merged.to_csv("reg_check.csv", index=False)
