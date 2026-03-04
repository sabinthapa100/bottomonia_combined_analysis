# -*- coding: utf-8 -*-
"""
prim_analysis.py
================
Core per-run R_pA analysis for quarkonia primordial suppression.

Works with any QuarkoniumSystem (bottomonia 9-state or charmonia 5-state).

Public API
----------
PrimordialAnalysis(df, system, with_feeddown=True)
    .rpa_vs_b(y_window, pt_window)
    .rpa_vs_y(pt_window, y_bins, flip_y=False)
    .rpa_vs_pt(y_window, pt_bins)
    .rpa_vs_y_at_b(b, pt_window, y_bins, flip_y=False)
    .rpa_vs_pt_at_b(b, y_window, pt_bins)

All methods return a tidy DataFrame with columns:
    [xcol, state, state_err, ...]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ups_particle import QuarkoniumSystem

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _nanmean_sem(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    mu  = float(np.nanmean(x))
    sem = float(np.nanstd(x, ddof=1) / math.sqrt(x.size))
    return mu, sem


# ---------------------------------------------------------------------------
# PrimordialAnalysis
# ---------------------------------------------------------------------------

@dataclass
class PrimordialAnalysis:
    """
    Compute R_pA from one TAMU primordial datafile output.

    Parameters
    ----------
    df            : DataFrame from PrimordialReader.load()
    system        : QuarkoniumSystem (carries feed-down matrix and sigmas)
    with_feeddown : if True (default), apply feed-down when computing R_pA
    norm_factor   : overall normalisation factor (default 1.0; used for Nbin weight if needed)
    """
    df:            pd.DataFrame
    system:        QuarkoniumSystem
    with_feeddown: bool  = True
    norm_factor:   float = 1.0

    # ---- private feed-down application ----
    def _apply_feeddown(
        self,
        Rdir: np.ndarray,
        Edir: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map direct suppression → observed suppression via feed-down.

        R_obs[i] = Σ_j F[i,j] σ_dir[j] R_dir[j] / (Σ_j F[i,j] σ_dir[j])

        The error is propagated via Jacobian linearisation.
        """
        sig  = self.system.sigma_dir  # direct cross sections
        num  = self.system.F @ (sig * Rdir)
        den  = self.system.F @ sig
        with np.errstate(divide="ignore", invalid="ignore"):
            Robs = np.where(den > 0, num / den, np.nan)

        cov_dir = np.diag(Edir**2)
        J       = self.system.F @ np.diag(sig)
        cov_num = J @ cov_dir @ J.T
        var_obs = np.diag(cov_num) / np.where(den > 0, den**2, np.nan)
        return Robs, np.sqrt(np.maximum(0.0, var_obs))

    # ---- block average ----
    def _block_stats(self, block: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        means, errs = [], []
        for nm in self.system.state_names:
            m, e = _nanmean_sem(block[nm].to_numpy())
            means.append(m); errs.append(e)
        return np.array(means, float), np.array(errs, float)

    # ---- selection helpers ----
    def _sel(self, df: pd.DataFrame,
             y_lo=None, y_hi=None,
             pt_lo=None, pt_hi=None) -> pd.DataFrame:
        m = pd.Series(True, index=df.index)
        if y_lo  is not None: m &= df["y"]  >= y_lo
        if y_hi  is not None: m &= df["y"]  <= y_hi
        if pt_lo is not None: m &= df["pt"] >= pt_lo
        if pt_hi is not None: m &= df["pt"] <= pt_hi
        return df.loc[m]

    # ---- row builder ----
    def _make_row(self, key: str, val: float, block: pd.DataFrame) -> dict:
        Rdir, Edir = self._block_stats(block)
        if self.with_feeddown:
            Robs, Eobs = self._apply_feeddown(Rdir, Edir)
        else:
            Robs, Eobs = Rdir, Edir
        row = {key: val}
        for k, nm in enumerate(self.system.state_names):
            row[nm]           = float(Robs[k])
            row[f"{nm}_err"]  = float(Eobs[k])
        return row

    # ---- inject χbJ averages (bottomonia only) ----
    def _inject_chibJ(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.system.name != "bottomonia":
            return df
        for grp, cols in [
            ("chibJ_1P", ["chib0_1P","chib1_1P","chib2_1P"]),
            ("chibJ_2P", ["chib0_2P","chib1_2P","chib2_2P"]),
        ]:
            present = [c for c in cols if c in df.columns]
            if len(present) == 3:
                av = df[present].to_numpy(float)
                df[grp] = np.nanmean(av, axis=1)
                ecols = [f"{c}_err" for c in present]
                if all(ec in df.columns for ec in ecols):
                    ee = df[ecols].to_numpy(float)
                    df[f"{grp}_err"] = np.sqrt(np.nansum(ee**2, axis=1)) / 3.0
        # inject χcJ average (charmonia)
        if self.system.name == "charmonia":
            ccols = ["chic0_1P","chic1_1P","chic2_1P"]
            if all(c in df.columns for c in ccols):
                df["chicJ_1P"] = df[ccols].to_numpy(float).mean(axis=1)
        return df

    # ================================================================
    # Public analysis methods
    # ================================================================

    def rpa_vs_b(
        self,
        y_window: Tuple[float, float],
        pt_window: Tuple[float, float],
    ) -> pd.DataFrame:
        """R_pA vs impact parameter b, integrated over y_window × pt_window."""
        ly, hy = y_window
        lp, hp = pt_window
        d = self._sel(self.df, y_lo=ly, y_hi=hy, pt_lo=lp, pt_hi=hp)
        rows = []
        for bval, blk in d.groupby("b", sort=True):
            rows.append(self._make_row("b", float(bval), blk))
        out = pd.DataFrame(rows).sort_values("b").reset_index(drop=True)
        return self._inject_chibJ(out)

    def rpa_vs_y(
        self,
        pt_window: Tuple[float, float],
        y_bins: Sequence[Tuple[float, float]],
        flip_y: bool = False,
    ) -> pd.DataFrame:
        """R_pA vs rapidity in bins, integrated over pt_window."""
        lp, hp = pt_window
        rows = []
        for lo, hi in y_bins:
            blk = self._sel(self.df, y_lo=lo, y_hi=hi, pt_lo=lp, pt_hi=hp)
            if blk.empty:
                continue
            ymid = 0.5 * (lo + hi)
            rows.append(self._make_row("y", (-ymid if flip_y else ymid), blk))
        out = pd.DataFrame(rows).sort_values("y").reset_index(drop=True)
        return self._inject_chibJ(out)

    def rpa_vs_pt(
        self,
        y_window: Tuple[float, float],
        pt_bins: Sequence[Tuple[float, float]],
    ) -> pd.DataFrame:
        """R_pA vs transverse momentum in bins, integrated over y_window."""
        ly, hy = y_window
        rows = []
        for lo, hi in pt_bins:
            blk = self._sel(self.df, y_lo=ly, y_hi=hy, pt_lo=lo, pt_hi=hi)
            if blk.empty:
                continue
            rows.append(self._make_row("pt", 0.5 * (lo + hi), blk))
        out = pd.DataFrame(rows).sort_values("pt").reset_index(drop=True)
        return self._inject_chibJ(out)

    # ---- per-b versions ----

    def rpa_vs_y_at_b(
        self,
        b: float,
        pt_window: Tuple[float, float],
        y_bins: Sequence[Tuple[float, float]],
        flip_y: bool = False,
    ) -> pd.DataFrame:
        """R_pA vs y at a fixed impact parameter b."""
        dfb = self.df[np.isclose(self.df["b"], b)]
        lp, hp = pt_window
        rows = []
        for lo, hi in y_bins:
            blk = self._sel(dfb, y_lo=lo, y_hi=hi, pt_lo=lp, pt_hi=hp)
            if blk.empty:
                continue
            ymid = 0.5 * (lo + hi)
            row = self._make_row("y", (-ymid if flip_y else ymid), blk)
            row["b"] = float(b)
            rows.append(row)
        out = pd.DataFrame(rows).sort_values("y").reset_index(drop=True)
        return self._inject_chibJ(out)

    def rpa_vs_pt_at_b(
        self,
        b: float,
        y_window: Tuple[float, float],
        pt_bins: Sequence[Tuple[float, float]],
    ) -> pd.DataFrame:
        """R_pA vs pT at a fixed impact parameter b."""
        dfb = self.df[np.isclose(self.df["b"], b)]
        ly, hy = y_window
        rows = []
        for lo, hi in pt_bins:
            blk = self._sel(dfb, y_lo=ly, y_hi=hy, pt_lo=lo, pt_hi=hi)
            if blk.empty:
                continue
            row = self._make_row("pt", 0.5 * (lo + hi), blk)
            row["b"] = float(b)
            rows.append(row)
        out = pd.DataFrame(rows).sort_values("pt").reset_index(drop=True)
        return self._inject_chibJ(out)

    # ---- list of available b values in data ----
    def b_values(self) -> np.ndarray:
        return np.unique(self.df["b"].to_numpy())
