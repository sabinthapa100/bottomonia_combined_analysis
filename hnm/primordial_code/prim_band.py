# -*- coding: utf-8 -*-
"""
prim_band.py
============
Band (uncertainty envelope) combiner for two primordial runs (lower + upper).

Given two PrimordialAnalysis objects (one for each TAMU output folder),
combines them into:
  - central value = mean of (lower, upper)
  - band          = envelope [min, max] (optionally expanded by ±SEM)

Public API
----------
PrimordialBand(lower, upper, include_run_errors=True)
    .vs_b(y_window, pt_window)             → (df_center, df_band)
    .vs_y(pt_window, y_bins, flip_y=False) → (df_center, df_band)
    .vs_pt(y_window, pt_bins)              → (df_center, df_band)
    .vs_y_at_b(b, ...)                     → (df_center, df_band)
    .vs_pt_at_b(b, ...)                    → (df_center, df_band)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from prim_analysis import PrimordialAnalysis


# ---------------------------------------------------------------------------
# Envelope engine
# ---------------------------------------------------------------------------

def _envelope(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    xcol: str,
    *,
    include_run_errors: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge two runs on xcol and build center + band DataFrames.

    Returns
    -------
    center : DataFrame  with columns [xcol, state, state_err, ...]
    band   : DataFrame  with columns [xcol, state_lo, state_hi, ...]
    """
    # Outer merge keeps all x points from both runs
    dfA = dfA.copy(); dfB = dfB.copy()
    dfA.columns = [xcol] + [
        f"{c}:A" if c != xcol else xcol for c in dfA.columns if c != xcol
    ]
    # rebuild properly
    colsA = list(dfA.columns)
    colsB_raw = [c for c in dfB.columns if c != xcol]
    dfB.columns = [xcol] + [f"{c}:B" for c in colsB_raw]

    merged = pd.merge(dfA, dfB, on=xcol, how="outer", sort=True)

    # Interpolate NaNs from outer join
    for c in merged.columns:
        if c == xcol:
            continue
        if merged[c].dtype.kind in "fiu":
            merged[c] = merged[c].interpolate("linear", limit_direction="both")

    # State names = columns in A that are not xcol and not _err
    base_states = [
        c[:-2] for c in colsA
        if c != xcol and c.endswith(":A") and not c.endswith("_err:A")
    ]

    center_data = {xcol: merged[xcol].to_numpy()}
    band_data   = {xcol: merged[xcol].to_numpy()}

    for s in base_states:
        muA = merged[f"{s}:A"].to_numpy()
        muB = merged[f"{s}:B"].to_numpy()

        if include_run_errors:
            eA_col = f"{s}_err:A"; eB_col = f"{s}_err:B"
            eA = merged[eA_col].to_numpy() if eA_col in merged.columns else np.zeros_like(muA)
            eB = merged[eB_col].to_numpy() if eB_col in merged.columns else np.zeros_like(muB)
            lo = np.minimum(muA - eA, muB - eB)
            hi = np.maximum(muA + eA, muB + eB)
        else:
            lo = np.minimum(muA, muB)
            hi = np.maximum(muA, muB)

        mu = 0.5 * (muA + muB)
        eA_col2 = f"{s}_err:A"; eB_col2 = f"{s}_err:B"
        eA2 = merged[eA_col2].to_numpy() if eA_col2 in merged.columns else np.zeros_like(muA)
        eB2 = merged[eB_col2].to_numpy() if eB_col2 in merged.columns else np.zeros_like(muB)
        e   = 0.5 * np.sqrt(eA2**2 + eB2**2)

        center_data[s]            = mu
        center_data[f"{s}_err"]   = e
        band_data[f"{s}_lo"]      = lo
        band_data[f"{s}_hi"]      = hi

    return pd.DataFrame(center_data), pd.DataFrame(band_data)


# ---------------------------------------------------------------------------
# PrimordialBand
# ---------------------------------------------------------------------------

@dataclass
class PrimordialBand:
    """
    Combines two PrimordialAnalysis runs (lower + upper) into a band.

    Parameters
    ----------
    lower              : PrimordialAnalysis for the lower (or first) run
    upper              : PrimordialAnalysis for the upper (or second) run
    include_run_errors : expand band by ±SEM of each run (default True)
    """
    lower: PrimordialAnalysis
    upper: PrimordialAnalysis
    include_run_errors: bool = True

    def _combine(
        self, fn_name: str, xcol: str, **kw
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfA = getattr(self.lower, fn_name)(**kw)
        dfB = getattr(self.upper, fn_name)(**kw)
        return _envelope(dfA, dfB, xcol, include_run_errors=self.include_run_errors)

    def vs_b(
        self,
        y_window: Tuple[float, float],
        pt_window: Tuple[float, float],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._combine("rpa_vs_b", "b", y_window=y_window, pt_window=pt_window)

    def vs_y(
        self,
        pt_window: Tuple[float, float],
        y_bins: Sequence[Tuple[float, float]],
        flip_y: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._combine(
            "rpa_vs_y", "y",
            pt_window=pt_window, y_bins=y_bins, flip_y=flip_y,
        )

    def vs_pt(
        self,
        y_window: Tuple[float, float],
        pt_bins: Sequence[Tuple[float, float]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._combine("rpa_vs_pt", "pt", y_window=y_window, pt_bins=pt_bins)

    def vs_y_at_b(
        self,
        b: float,
        pt_window: Tuple[float, float],
        y_bins: Sequence[Tuple[float, float]],
        flip_y: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfA = self.lower.rpa_vs_y_at_b(b, pt_window=pt_window, y_bins=y_bins, flip_y=flip_y)
        dfB = self.upper.rpa_vs_y_at_b(b, pt_window=pt_window, y_bins=y_bins, flip_y=flip_y)
        return _envelope(dfA, dfB, "y", include_run_errors=self.include_run_errors)

    def vs_pt_at_b(
        self,
        b: float,
        y_window: Tuple[float, float],
        pt_bins: Sequence[Tuple[float, float]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfA = self.lower.rpa_vs_pt_at_b(b, y_window=y_window, pt_bins=pt_bins)
        dfB = self.upper.rpa_vs_pt_at_b(b, y_window=y_window, pt_bins=pt_bins)
        return _envelope(dfA, dfB, "pt", include_run_errors=self.include_run_errors)

    def b_values(self) -> np.ndarray:
        """Impact parameters common to both runs."""
        bA = self.lower.b_values()
        bB = self.upper.b_values()
        # return union
        return np.unique(np.concatenate([bA, bB]))
