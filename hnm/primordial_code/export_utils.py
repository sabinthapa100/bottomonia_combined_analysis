# -*- coding: utf-8 -*-
"""
export_utils.py
===============
Utilities for exporting primordial analysis central/band results to 
publication-ready CSV files (HEPData style).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Optional

def save_hepdata_csv(
    df_center: pd.DataFrame,
    df_band: pd.DataFrame,
    xcol: str,
    bins: Sequence[Tuple[float, float]],
    save_path: str,
    states: Optional[Sequence[str]] = None,
):
    """
    Saves a CSV with bin boundaries and separate stat/sys errors.
    
    Columns:
      [xcol]_min, [xcol]_max, [xcol]_mid,
      [state]_val, [state]_stat, [state]_sys_lo, [state]_sys_hi
    """
    if states is None:
        # Infer states from df_center columns
        states = [c for c in df_center.columns if c != xcol and not c.endswith("_err")]

    # Create base dataframe with bin edges
    data = {
        f"{xcol}_min": [b[0] for b in bins],
        f"{xcol}_max": [b[1] for b in bins],
        f"{xcol}_mid": df_center[xcol].to_numpy(),
    }

    for s in states:
        if s not in df_center.columns:
            continue
            
        val = df_center[s].to_numpy()
        stat = df_center[f"{s}_err"].to_numpy() if f"{s}_err" in df_center.columns else np.zeros_like(val)
        
        # Systematic band from df_band
        lo = df_band[f"{s}_lo"].to_numpy() if f"{s}_lo" in df_band.columns else val
        hi = df_band[f"{s}_hi"].to_numpy() if f"{s}_hi" in df_band.columns else val
        
        data[f"{s}_val"] = val
        data[f"{s}_stat"] = stat
        data[f"{s}_sys_lo"] = np.abs(val - lo)
        data[f"{s}_sys_hi"] = np.abs(hi - val)

    out_df = pd.DataFrame(data)
    out_df.to_csv(save_path, index=False)
    print(f"[export] HEPData CSV saved to: {save_path}")
