# -*- coding: utf-8 -*-
"""
verify_oo5360.py
================
Senior Scientist Debugger & Verifier for OOP 5.36 TeV Primordial Analysis.

This script runs the Python primordial module on the OxOx_5360_NPWLC/Pert data
and explicitly compares the generated R_pA vs y and R_pA vs pT against the
pre-existing Mathematica `.m` outputs from `hnm/output-oxygen-oxygen-5.36/`.

Provides detailed, physics-focused logging of intermediate steps.
"""

from __future__ import annotations

import os
import re
import sys
import numpy as np
import pandas as pd

# Allow relative imports from agents folder for now
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(os.path.dirname(_HERE), 'primordial_code')
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from ups_particle import make_bottomonia_system
from prim_band import PrimordialBand
from prim_io import load_pair
from prim_analysis import PrimordialAnalysis

# ---------------------------------------------------------------------------
# Mathematica Parser
# ---------------------------------------------------------------------------

def parse_mathematica_m(filepath: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Parses a Mathematica .m file containing lists of lists:
    { {x1, Around[mu1, err1]}, {x2, Around[mu2, err2]}, ... }
    
    Returns a list of tuples: (x_arr, mu_arr, err_arr)
    One tuple per particle state. By convention: [1S, 2S, 3S].
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] Missing Mathematica verification file: {filepath}")
        return []
        
    with open(filepath, 'r') as f:
        content = f.read()

    # Match inner `{x, Around[mu, err]}`
    pattern = r"\{([\-\d\.]+),\s*Around\[([\-\d\.]+),\s*([\d\.]+)\]\}"
    matches = re.findall(pattern, content)

    # Split into separate states when x drops (new x-axis sequence)
    states_data = []
    current_state = []
    last_x = float('inf')

    for match in matches:
        x, mu, err = map(float, match)
        if x < last_x and current_state:
            states_data.append(current_state)
            current_state = []
        current_state.append((x, mu, err))
        last_x = x

    if current_state:
        states_data.append(current_state)

    # Convert to numpy arrays
    result = []
    for st_data in states_data:
        arr = np.array(st_data)
        result.append((arr[:,0], arr[:,1], arr[:,2]))

    return result

# ---------------------------------------------------------------------------
# Verifier Core
# ---------------------------------------------------------------------------

def compare_arrays(name: str, x_py: np.ndarray, y_py: np.ndarray, err_py: np.ndarray, 
                   x_math: np.ndarray, y_math: np.ndarray, err_math: np.ndarray, 
                   tolerance: float = 1e-4) -> bool:
    """Compares Python arrays against Mathematica arrays with interpolation if needed."""
    print(f"\n  [Verify] {name}: Checking {len(x_py)} Python points vs {len(x_math)} Math points.")
    
    # Check x-axis alignment
    if len(x_py) != len(x_math) or not np.allclose(x_py, x_math, atol=1e-3):
        print("    [WARN] x-axis mismatch between Python and Mathematica!")
        print("    Python x :", np.round(x_py, 3))
        print("    Math x   :", np.round(x_math, 3))
        
        # We can interpolate Math to Python x to do the check
        y_math_interp = np.interp(x_py, x_math, y_math)
        err_math_interp = np.interp(x_py, x_math, err_math)
    else:
        y_math_interp = y_math
        err_math_interp = err_math

    mu_diff = np.abs(y_py - y_math_interp)
    err_diff = np.abs(err_py - err_math_interp)
    
    mu_max_idx = np.argmax(mu_diff)
    
    print(f"    Max rel diff in mean : {mu_diff[mu_max_idx]/np.maximum(1e-9, y_math_interp[mu_max_idx]):.2%}")
    print(f"    Max abs diff in mean : {np.max(mu_diff):.2e}")
    # print(f"    Max abs diff in error: {np.max(err_diff):.2e}")

    if np.max(mu_diff) < tolerance:
        print(f"    ✅ PASS: {name} matches Mathematica within {tolerance}")
        return True
    else:
        print(f"    ❌ FAIL: {name} exceeds tolerance {tolerance}")
        print(f"       Worst idx={mu_max_idx}: py={y_py[mu_max_idx]:.5f} | mma={y_math_interp[mu_max_idx]:.5f}")
        return False

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def verify_model(model_name: str, lower_gz: str, upper_gz: str, math_dir: str):
    print(f"\n{'='*70}")
    print(f"🔬 SENIOR SCIENTIST DEBUGGER: Verifying {model_name}...")
    print(f"{'='*70}")
    
    # 1. Setup physics system
    print("[1] Building 9-state Bottomonia system for O+O 5.36 TeV")
    system = make_bottomonia_system(sqrts_pp_GeV=5360.0)
    print("    [Debug] σ_dir:", np.round(system.sigma_dir, 3))
    print("    [Debug] Feeding matrix determinant:", np.linalg.det(system.F))
    
    # 2. Loading
    print(f"\n[2] Loading simulation outputs...")
    df_lo, df_hi = load_pair(lower_gz, upper_gz, system, debug=False)
    print(f"    Lower shape: {df_lo.shape}, b_mean: {df_lo['b'].mean():.4f}")
    
    # 3. Analyze bounds
    print(f"\n[3] Computing R_pA bounds with Feed-Down=ON ...")
    ana_lo = PrimordialAnalysis(df_lo, system, with_feeddown=True)
    # The Mathematica files we are checking against are specifically for the "lower" band
    # Or at least, the folder is `output-lower`. So we verify `ana_lo` against them.
    
    math_y_file = os.path.join(math_dir, "rpavsy.m")
    math_pt_cen_file = os.path.join(math_dir, "rpavspt-central.m")
    math_pt_fwd_file = os.path.join(math_dir, "rpavspt-forward.m")
    math_pt_bwd_file = os.path.join(math_dir, "rpavspt-backward.m")
    
    # Define analysis bins mimicking Mathematica bounds
    # Notice the bin edges used in previous script:
    # 20 bins for y: -5 to +5 -> bin width 0.5
    y_bin_edges = np.linspace(-5.0, 5.0, 21)
    y_bins = [(y_bin_edges[i], y_bin_edges[i+1]) for i in range(20)]
    
    # For pT: 12 bins from 0 to 30 -> bin width 2.5? Let's check Math files:
    # math x: 1.25, 3.75, 6.25 ... 28.75. Yes, width 2.5.
    pt_bin_edges = np.linspace(0.0, 30.0, 13)
    pt_bins = [(pt_bin_edges[i], pt_bin_edges[i+1]) for i in range(12)]

    # Rapidity windows
    Y_CEN = (-1.93, 1.93)
    Y_BWD = (-5.0, -2.5)
    Y_FWD = (1.5, 4.0)
    
    estados = ["ups1S", "ups2S", "ups3S"]
    estados_nome = ["Υ(1S)", "Υ(2S)", "Υ(3S)"]

    print(f"\n[4] Verifying vs Y integration...")
    if os.path.exists(math_y_file):
        # We must ensure we cover all pT to match Mathematica's selectYweighted (which had no pT cut)
        pt_integration_window = (-np.inf, np.inf)
        py_y = ana_lo.rpa_vs_y(
            pt_window=pt_integration_window,
            y_bins=y_bins,
            flip_y=True
        )
        math_y_data = parse_mathematica_m(math_y_file)
        
        for st_idx, (st_key, st_name) in enumerate(zip(estados, estados_nome)):
            if st_idx < len(math_y_data):
                mx, my, merr = math_y_data[st_idx]
                px = py_y['y'].to_numpy()
                py = py_y[st_key].to_numpy()
                perr = py_y[f'{st_key}_err'].to_numpy()
                compare_arrays(f"R_pA vs Y [{st_name}]", px, py, perr, mx, my, merr)

    print(f"\n[5] Verifying vs pT [Central {Y_CEN}]...")
    if os.path.exists(math_pt_cen_file):
        py_pt_cen = ana_lo.rpa_vs_pt(y_window=Y_CEN, pt_bins=pt_bins)
        math_cen_data = parse_mathematica_m(math_pt_cen_file)
        for st_idx, (st_key, st_name) in enumerate(zip(estados, estados_nome)):
            if st_idx < len(math_cen_data):
                mx, my, merr = math_cen_data[st_idx]
                px = py_pt_cen['pt'].to_numpy()
                py = py_pt_cen[st_key].to_numpy()
                perr = py_pt_cen[f'{st_key}_err'].to_numpy()
                compare_arrays(f"R_pA vs pT Central [{st_name}]", px, py, perr, mx, my, merr)

    print(f"\n[6] Verifying vs pT [Forward {Y_FWD}]...")
    if os.path.exists(math_pt_fwd_file):
        py_pt_fwd = ana_lo.rpa_vs_pt(y_window=Y_FWD, pt_bins=pt_bins)
        math_fwd_data = parse_mathematica_m(math_pt_fwd_file)
        for st_idx, (st_key, st_name) in enumerate(zip(estados, estados_nome)):
            if st_idx < len(math_fwd_data):
                mx, my, merr = math_fwd_data[st_idx]
                px = py_pt_fwd['pt'].to_numpy()
                py = py_pt_fwd[st_key].to_numpy()
                perr = py_pt_fwd[f'{st_key}_err'].to_numpy()
                compare_arrays(f"R_pA vs pT Forward [{st_name}]", px, py, perr, mx, my, merr)

if __name__ == "__main__":
    _REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
    
    NPWLC_LOWER = os.path.join(_REPO_ROOT, "inputs", "primordial", "output_OxOx5360_NPWLC", "output-lower", "datafile.gz")
    NPWLC_UPPER = os.path.join(_REPO_ROOT, "inputs", "primordial", "output_OxOx5360_NPWLC", "output-upper", "datafile.gz")
    NPWLC_MATH  = os.path.join(_REPO_ROOT, "hnm", "output-oxygen-oxygen-5.36", "output_OxOx5360_NPWLC", "output-lower")
    
    verify_model("NPWLC (Lower Band)", NPWLC_LOWER, NPWLC_UPPER, NPWLC_MATH)
