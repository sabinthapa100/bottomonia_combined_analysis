# -*- coding: utf-8 -*-
"""
npdf_OO_data.py
===============
Loader and analysis utilities for Oxygen-Oxygen nPDF_OO.dat file.

The file contains pre-computed gluon nPDF ratios R_g^O(x1) * R_g^O(x2)
for EPPS21 sets 1..49 (1 = central, 2..49 = 48 Hessian error members).

Columns per set: y, pt, x1, x2, Rg1, Rg2, Rg1*Rg2

For minimum-bias AA (O-O), the nPDF modification factor is simply Rg1*Rg2.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_OO_dat(filepath: str) -> Dict[int, pd.DataFrame]:
    """
    Parse nPDF_OO.dat into a dict of {set_id: DataFrame}.

    Each DataFrame has columns: y, pt, x1, x2, Rg1, Rg2, Rg1Rg2

    Parameters
    ----------
    filepath : str
        Path to nPDF_OO.dat

    Returns
    -------
    dict  : {1: df_central, 2: df_err1, ..., 49: df_err48}
    """
    filepath = Path(filepath)
    text = filepath.read_text()

    # Split by "EPPS21 set" header lines
    set_pat = re.compile(r"^\s*EPPS21\s+set\s+(\d+)\s*$", re.MULTILINE)
    matches = list(set_pat.finditer(text))

    result = {}
    for i, m in enumerate(matches):
        set_id = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        block = text[start:end]
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("y"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                vals = [float(p) for p in parts[:7]]
                rows.append(vals)
            except ValueError:
                continue

        if rows:
            df = pd.DataFrame(rows, columns=["y", "pt", "x1", "x2", "Rg1", "Rg2", "Rg1Rg2"])
            result[set_id] = df

    return result


def build_OO_rpa_grid(data: Dict[int, pd.DataFrame],
                      pt_max: float = 15.0) -> pd.DataFrame:
    """
    Build an R_AA grid with Hessian error bands from the 49 EPPS21 sets.

    The central ratio is Rg1*Rg2 from set 1.
    Error bands are computed from the 48 Hessian members (sets 2..49).

    Parameters
    ----------
    data : dict from load_OO_dat
    pt_max : float
        Maximum pT to include (high-pT rows often have unphysical ratios)

    Returns
    -------
    DataFrame with columns: y, pt, r_central, r_lo, r_hi, r_mem_001..r_mem_048
    """
    central = data[1].copy()
    central = central[central["pt"] <= pt_max].reset_index(drop=True)

    r0 = central["Rg1Rg2"].to_numpy()
    N = len(r0)

    # Collect members (aligned to central grid)
    mems = []
    for sid in range(2, 50):
        df_e = data[sid].copy()
        df_e = df_e[df_e["pt"] <= pt_max].reset_index(drop=True)
        mems.append(df_e["Rg1Rg2"].to_numpy()[:N])

    M = np.stack(mems, axis=0)  # (48, N)

    # Hessian band (pairwise)
    D = M[0::2, :] - M[1::2, :]
    h = 0.5 * np.sqrt(np.sum(D * D, axis=0))

    out = central[["y", "pt"]].copy()
    out["r_central"] = r0
    out["r_lo"] = r0 - h
    out["r_hi"] = r0 + h

    for j in range(M.shape[0]):
        out[f"r_mem_{j+1:03d}"] = M[j]

    return out


def bin_rpa_vs_y_OO(grid: pd.DataFrame,
                    y_edges: np.ndarray,
                    pt_range: Tuple[float, float] = (0.0, 15.0)) -> pd.DataFrame:
    """
    Bin R_AA vs rapidity, averaging over pT range.

    Returns DataFrame with columns: y_center, r_central, r_lo, r_hi
    """
    mask = (grid["pt"] >= pt_range[0]) & (grid["pt"] <= pt_range[1])
    g = grid[mask].copy()

    y_cents = 0.5 * (y_edges[:-1] + y_edges[1:])
    rows = []
    for i in range(len(y_edges) - 1):
        sel = g[(g["y"] >= y_edges[i]) & (g["y"] < y_edges[i + 1])]
        if len(sel) == 0:
            continue
        rows.append({
            "y_center": float(y_cents[i]),
            "r_central": float(sel["r_central"].mean()),
            "r_lo": float(sel["r_lo"].mean()),
            "r_hi": float(sel["r_hi"].mean()),
        })
    return pd.DataFrame(rows)


def bin_rpa_vs_pT_OO(grid: pd.DataFrame,
                     pt_edges: np.ndarray,
                     y_range: Tuple[float, float] = (-4.5, 4.5)) -> pd.DataFrame:
    """
    Bin R_AA vs pT, averaging over y range.

    Returns DataFrame with columns: pt_center, r_central, r_lo, r_hi
    """
    mask = (grid["y"] >= y_range[0]) & (grid["y"] <= y_range[1])
    g = grid[mask].copy()

    pt_cents = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    rows = []
    for i in range(len(pt_edges) - 1):
        sel = g[(g["pt"] >= pt_edges[i]) & (g["pt"] < pt_edges[i + 1])]
        if len(sel) == 0:
            continue
        rows.append({
            "pt_center": float(pt_cents[i]),
            "r_central": float(sel["r_central"].mean()),
            "r_lo": float(sel["r_lo"].mean()),
            "r_hi": float(sel["r_hi"].mean()),
        })
    return pd.DataFrame(rows)
