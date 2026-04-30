import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import os
import sys

ROOT = Path('/mnt/workstation/bottomonia_combined_analysis')
OUT_DIR = ROOT / 'outputs' / 'cnm_hnm' / 'final_qtraj_analysis_event_weighted'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Physics Parameters (from ups_particle.py / Mathematica) ──────────────────
# 9-state basis: 1S, 2S, 1P0, 1P1, 1P2, 3S, 2P0, 2P1, 2P2
sigmas_exp = np.array([57.6, 19.0, 3.72, 13.69, 16.1, 6.8, 3.27, 12.0, 14.15])
F = np.array([
    [1.0,    0.2645, 0.0194, 0.352,  0.18,   0.0657, 0.0038, 0.1153, 0.077 ],
    [0.0,    1.0,    0.0,    0.0,    0.0,    0.106,  0.0138, 0.181,  0.089 ],
    [0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0   ],
    [0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0091, 0.0   ],
    [0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0051],
    [0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0   ],
    [0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0   ],
    [0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0   ],
    [0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0   ],
])
F_inv = np.linalg.inv(F)
sigmas_dir = F_inv @ sigmas_exp

# ── Event-Level Recombination ────────────────────────────────────────────────
def process_datafile(fpath):
    # Group trajectories by metadata (b, x0, y0, phi, pt, eta, yrap)
    # The datafile.gz rows: (meta) then (data)
    events = {} # key: tuple(meta), val: [S_row, P_row]
    
    with gzip.open(fpath, 'rt') as f:
        meta = None
        for i, line in enumerate(f):
            try:
                line_data = list(map(float, line.split()))
                if i % 2 == 0:
                    meta = tuple(line_data)
                else:
                    data = line_data
                    if meta not in events: events[meta] = [None, None]
                    initL = int(data[-1])
                    if initL == 0: events[meta][0] = data
                    elif initL == 1: events[meta][1] = data
            except: pass
    
    processed_events = []
    for meta, components in events.items():
        s_row, p_row = components
        if s_row is None and p_row is None: continue
        
        # 1S, 2S, 3S from S-wave init; 1P, 2P from P-wave init
        # (Assuming the system starts as a mixture of vacuum states)
        o0, o1, o2, o3, o4 = 0, 0, 0, 0, 0
        if s_row:
            o0, o1, o3 = s_row[0], s_row[1], s_row[3]
        if p_row:
            o2, o4 = p_row[2], p_row[4]
            
        # Build 9-state base vector
        v = np.array([o0, o1, o2, o2, o2, o3, o4, o4, o4])
        
        # Observed R_AA (Feeddown weighted by direct sigmas)
        num = F @ (sigmas_dir * v)
        den = sigmas_exp # (F @ sigmas_dir)
        raa = num / den
        
        r = np.sqrt(meta[1]**2 + meta[2]**2)
        processed_events.append({
            'r': r, 'x0': meta[1], 'pt': meta[4], 'y': meta[6],
            'raa1S': raa[0], 'raa2S': raa[1], 'raa3S': raa[5]
        })
    return pd.DataFrame(processed_events)

print("Loading and recombining raw 8-column QTraj events ...")
df_no = process_datafile(ROOT / 'inputs' / 'qtraj_nlo_run1_OO_5.36_kap6_noReg' / 'datafile_partial.gz')
print(f"  noReg: {len(df_no)} unique events.")

df_w  = process_datafile(ROOT / 'inputs' / 'qtraj-nlo-run2-00-5.36-kap6-wReg' / 'datafile.gz')
print(f"  wReg: {len(df_w)} unique events.")

# ── Geometric Normalization ───────────────────────────────────────────────────
df_no_core = df_no[np.abs(df_no['x0']) < 0.2].copy()
edges = np.linspace(0, 5, 21)
df_w['bin'] = pd.cut(df_w['r'], edges)
df_no_core['bin'] = pd.cut(df_no_core['r'], edges)

e_map = df_w.groupby('bin', observed=False)[['raa1S', 'raa2S', 'raa3S']].mean() / \
        df_no_core.groupby('bin', observed=False)[['raa1S', 'raa2S', 'raa3S']].mean().replace(0,1)
e_map = e_map.fillna(1.0).clip(lower=1.0)

print("Rebuilding unbiased 2D wReg result ...")
df_no['bin'] = pd.cut(df_no['r'], edges)
df_w_rebuilt = df_no.copy()
for state in ['raa1S', 'raa2S', 'raa3S']:
    df_w_rebuilt[state] *= df_w_rebuilt['bin'].map(e_map[state].to_dict()).fillna(1.0)

# Export
Y_EDGES = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
PT_EDGES = [0, 2, 4, 6, 8, 12, 16, 20]

def export_results(df, tag):
    df['y_bin'] = pd.cut(df['y'], Y_EDGES)
    res_y = df.groupby('y_bin', observed=False)[['raa1S', 'raa2S', 'raa3S']].mean()
    res_y.to_csv(OUT_DIR / f"QTraj_EventWeighted_RAA_vs_y_{tag}.csv")
    
    sel = df[np.abs(df['y']) <= 2.4].copy()
    sel['pt_bin'] = pd.cut(sel['pt'], PT_EDGES)
    res_pt = sel.groupby('pt_bin', observed=False)[['raa1S', 'raa2S', 'raa3S']].mean()
    res_pt.to_csv(OUT_DIR / f"QTraj_EventWeighted_RAA_vs_pt_{tag}.csv")

export_results(df_no, "noReg")
export_results(df_w_rebuilt, "wReg")
print(f"DONE. Results in {OUT_DIR}")
