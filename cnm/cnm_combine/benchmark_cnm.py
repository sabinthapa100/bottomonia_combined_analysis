
import sys
import time
import numpy as np
from pathlib import Path

# Add paths
base_dir = Path("/home/sawin/Desktop/Charmonia/charmonia_combined_analysis")
sys.path.append(str(base_dir / "eloss_code"))
sys.path.append(str(base_dir / "npdf_code"))
sys.path.append(str(base_dir / "cnm_combine"))

try:
    from cnm_combine import CNMCombine
    print("CNMCombine imported successfully.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def benchmark_system(energy="5.02", system="LHC"):
    print(f"\n--- Benchmarking {system} (Energy={energy}) ---")
    
    start_init = time.time()
    if system == "LHC":
        cnm = CNMCombine.from_defaults(energy="5.02", family="charmonia")
    else: # RHIC
        # RHIC bins: 0-20, 20-40, 40-60, 60-88
        rhic_bins = [(0,20), (20,40), (40,60), (60,88)]
        # y-window: -3 to 3
        y_wins = [(-3.0, 3.0, "-3 < y < 3")]
        cnm = CNMCombine.from_defaults(
            energy="200", 
            family="charmonia",
            cent_bins=rhic_bins,
            y_edges=np.linspace(-3.0, 3.0, 31),
            p_edges=np.linspace(0.0, 10.0, 11),
            y_windows=y_wins,
            pt_range_avg=(0.0, 5.0)
        )
    print(f"Init time: {time.time() - start_init:.2f} s")

    # 1. RpA vs y
    t0 = time.time()
    res_y = cnm.cnm_vs_y(include_mb=True)
    print(f"RpA vs y time: {time.time() - t0:.2f} s")
    
    # 2. RpA vs pT (first window)
    t0 = time.time()
    win = cnm.y_windows[0]
    res_pt = cnm.cnm_vs_pT(y_window=win, include_mb=True)
    print(f"RpA vs pT time ({win[2]}): {time.time() - t0:.2f} s")

    # 3. RpA vs Centrality
    t0 = time.time()
    res_cent = cnm.cnm_vs_centrality(y_window=win, include_mb=True)
    print(f"RpA vs Cent time ({win[2]}): {time.time() - t0:.2f} s")

if __name__ == "__main__":
    benchmark_system("5.02", "LHC")
    benchmark_system("200", "RHIC")
