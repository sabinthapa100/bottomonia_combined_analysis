import sys
import os
import gzip
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add directory containing qtraj_analysis symlink to path
ROOT = Path("/mnt/workstation/bottomonia_combined_analysis")
sys.path.insert(0, str(ROOT / "hnm/qtraj-nlo/qtraj_out_analysis"))

from qtraj_analysis.io import read_whitespace_table, parse_records
from qtraj_analysis.matching import build_observables

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Rapidity windows from run_bottomonia_cnm_prim_OO.py
Y_WINDOWS = [
    (-2.4,  2.4, "mid"),
    (-4.5, -2.4, "back"),
    ( 2.4,  4.0, "forw"),
]

# pT bins (1 GeV width)
PT_EDGES = np.arange(0.0, 20.0 + 1.0, 1.0)
PT_CENTERS = 0.5 * (PT_EDGES[:-1] + PT_EDGES[1:])

# Y bins (0.5 width)
Y_EDGES = np.arange(-5.0, 5.0 + 0.5, 0.5)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])

STATE_NAMES = ["ups1S", "ups2S", "chi_b1P", "ups3S", "chi_b2P", "ups1D"]


def run_analysis(datafile: Path, output_dir: Path, label: str):
    """Run QTraj analysis on a single datafile and output CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{label}] Reading data from {datafile}...")
    table = read_whitespace_table(str(datafile), logger)
    records = parse_records(table, logger)
    obs_list = build_observables(records, logger)
    logger.info(f"[{label}] Successfully loaded {len(obs_list)} matched trajectories.")

    # 1. R_AA vs y (Integrated over pT 0-20)
    logger.info(f"[{label}] Computing R_AA vs y...")
    y_results = []
    for yc in Y_CENTERS:
        y_min, y_max = yc - 0.25, yc + 0.25
        subset = [o for o in obs_list if y_min <= o.y < y_max]
        row = {"y": yc}
        if subset:
            for s_idx, state in enumerate(STATE_NAMES):
                vals = np.array([o.surv6[s_idx] for o in subset])
                row[state] = np.mean(vals)
                row[f"{state}_err"] = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        else:
            for state in STATE_NAMES:
                row[state] = 1.0
                row[f"{state}_err"] = 0.0
        y_results.append(row)
    
    df_y = pd.DataFrame(y_results)
    df_y.to_csv(output_dir / "qtraj_vs_y.csv", index=False)
    logger.info(f"[{label}] Saved y-results to {output_dir / 'qtraj_vs_y.csv'}")

    # 2. R_AA vs pT for each y-window
    logger.info(f"[{label}] Computing R_AA vs pT for each y-window...")
    for y0, y1, yname in Y_WINDOWS:
        logger.info(f"[{label}] Processing window: {yname} ({y0} < y < {y1})")
        pt_results = []
        y_subset = [o for o in obs_list if y0 <= o.y < y1]
        
        for pc in PT_CENTERS:
            p_min, p_max = pc - 0.5, pc + 0.5
            subset = [o for o in y_subset if p_min <= o.pt < p_max]
            row = {"pt": pc}
            if subset:
                for s_idx, state in enumerate(STATE_NAMES):
                    vals = np.array([o.surv6[s_idx] for o in subset])
                    row[state] = np.mean(vals)
                    row[f"{state}_err"] = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            else:
                for state in STATE_NAMES:
                    row[state] = 1.0
                    row[f"{state}_err"] = 0.0
            pt_results.append(row)
            
        df_pt = pd.DataFrame(pt_results)
        df_pt.to_csv(output_dir / f"qtraj_vs_pt_{yname}.csv", index=False)
        logger.info(f"[{label}] Saved pT-results for {yname}")


def main():
    # --- noReg (run1) ---
    noReg_datafile = ROOT / "inputs/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile_partial.gz"
    noReg_output   = ROOT / "outputs/qtraj_nlo/OO_536/noReg"
    run_analysis(noReg_datafile, noReg_output, "noReg")

    # --- wReg (run2, pre-averaged) ---
    wReg_datafile = ROOT / "inputs/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile-avg.gz"
    wReg_output   = ROOT / "outputs/qtraj_nlo/OO_536/wReg"
    run_analysis(wReg_datafile, wReg_output, "wReg")


if __name__ == "__main__":
    main()
