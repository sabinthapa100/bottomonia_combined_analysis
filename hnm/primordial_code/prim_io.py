# -*- coding: utf-8 -*-
"""
prim_io.py
==========
I/O layer for TAMU primordial `datafile.gz` outputs.

Handles:
  - Bottomonia: 5, 6, 9, or 10 suppression columns → normalizes to 9 (STATE9).
  - Charmonia : 5 suppression columns → STATE5.
  - Flexible column mapping for meta rows (b, pT, y).
  - Single or multiple impact-parameter (b) files.

Key class
---------
PrimordialReader  – reads one datafile.gz for a given QuarkoniumSystem.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ups_particle import QuarkoniumSystem, BOTTOM_STATE_NAMES

# ---------------------------------------------------------------------------
# Internal: normalise suppression rows for bottomonia
# ---------------------------------------------------------------------------

def _split_hf_5to9(v5: np.ndarray) -> np.ndarray:
    """5-column: [1S, 2S, 1P, 3S, 2P] → 9-column hyperfine split."""
    a = np.asarray(v5, float).reshape(-1)
    if a.size != 5:
        raise ValueError(f"Expected 5 suppression entries, got {a.size}")
    # 1P → [chib0_1P, chib1_1P, chib2_1P] all equal; same for 2P
    return np.array([a[0], a[1], a[2], a[2], a[2], a[3], a[4], a[4], a[4]], float)


def _normalize_bottomonia_row(row: np.ndarray, debug: bool = False) -> np.ndarray:
    n = row.size
    if n == 9:
        if debug: print(f"  [io] 9→9")
        return row[:9]
    if n == 5:
        if debug: print(f"  [io] 5→9 (hyperfine split)")
        return _split_hf_5to9(row)
    if n == 6:
        if debug: print(f"  [io] 6→5 (drop trailing) →9 (hyperfine split)")
        return _split_hf_5to9(row[:5])
    if n >= 10:
        if debug: print(f"  [io] {n}→9 (take first 9)")
        return row[:9]
    # fallback: use first 5
    if debug: print(f"  [io] {n}→5 (fallback) →9 (hyperfine split)")
    return _split_hf_5to9(row[:5])


# ---------------------------------------------------------------------------
# PrimordialReader
# ---------------------------------------------------------------------------

@dataclass
class PrimordialReader:
    """
    Read TAMU primordial datafile.gz and return a DataFrame.

    Parameters
    ----------
    path       : path to datafile.gz (or plain text)
    system     : QuarkoniumSystem (defines state names and n_states)
    meta_idx_b : column index in meta row for impact parameter b  (default 0)
    meta_idx_pt: column index in meta row for pT                   (default 4)
    meta_idx_y : column index in meta row for rapidity y           (default 6)
    debug      : print diagnostic info
    """
    path:        str
    system:      QuarkoniumSystem
    meta_idx_b:  int = 0
    meta_idx_pt: int = 4
    meta_idx_y:  int = 6
    debug:       bool = False

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"datafile not found: {self.path}")

        def _iter_rows(p):
            opener = gzip.open if str(p).endswith(".gz") else open
            with opener(p, "rt") as f:
                for ln, line in enumerate(f, 1):
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        yield [float(x) for x in s.split()]
                    except Exception:
                        if self.debug:
                            print(f"[io WARN] Non-numeric line {ln} skipped.")

        rows = list(_iter_rows(self.path))
        if len(rows) < 2:
            raise RuntimeError(f"datafile too short: {self.path}")
        if len(rows) % 2 != 0:
            if self.debug:
                print("[io WARN] Odd row count; dropping last row.")
            rows = rows[:-1]

        meta = np.asarray(rows[0::2], float)
        sup  = np.asarray(rows[1::2], float)

        if meta.shape[0] != sup.shape[0]:
            raise RuntimeError("Meta/suppression row count mismatch (corrupted file?).")

        bi  = self.meta_idx_b
        pi  = self.meta_idx_pt
        yi  = self.meta_idx_y

        if meta.shape[1] <= max(bi, pi, yi):
            raise IndexError(
                f"Meta rows have only {meta.shape[1]} columns; need indices {bi},{pi},{yi}."
            )

        b  = meta[:, bi]
        pt = meta[:, pi]
        y  = meta[:, yi]

        if self.debug:
            print(f"[io INFO] {os.path.basename(self.path)} | rows={len(sup)} | "
                  f"sup_cols={sup.shape[1]} | system={self.system.name}")
            print(f"          First sup row: {np.array2string(sup[0], precision=5, threshold=20)}")

        # Normalise suppression columns
        if self.system.name == "bottomonia":
            S = np.vstack([_normalize_bottomonia_row(sup[i], self.debug)
                           for i in range(sup.shape[0])])
        else:
            # charmonia: take first 5 columns directly
            n_st = self.system.n_states  # 5
            if sup.shape[1] < n_st:
                raise IndexError(
                    f"Suppression rows have {sup.shape[1]} cols; need ≥{n_st} for {self.system.name}."
                )
            S = sup[:, :n_st]

        data = {"b": b, "pt": pt, "y": y}
        for j, nm in enumerate(self.system.state_names):
            data[nm] = S[:, j]

        df = pd.DataFrame(data)
        if self.debug:
            print(f"[io INFO] Parsed → {len(df):,} rows | "
                  f"b:[{b.min():.2f},{b.max():.2f}] "
                  f"pt:[{pt.min():.2f},{pt.max():.2f}] "
                  f"y:[{y.min():.2f},{y.max():.2f}]")
        return df


# ---------------------------------------------------------------------------
# Convenience: load a pair (lower, upper) → list of DataFrames
# ---------------------------------------------------------------------------

def load_pair(
    lower_path: str,
    upper_path: str,
    system: QuarkoniumSystem,
    *,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load lower and upper bound files, return (df_lower, df_upper)."""
    df_lo = PrimordialReader(lower_path, system, debug=debug).load()
    df_hi = PrimordialReader(upper_path, system, debug=debug).load()
    return df_lo, df_hi


def load_single(
    path: str,
    system: QuarkoniumSystem,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """Load a single datafile.gz."""
    return PrimordialReader(path, system, debug=debug).load()
