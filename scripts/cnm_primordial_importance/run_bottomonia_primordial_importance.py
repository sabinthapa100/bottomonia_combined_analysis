#!/usr/bin/env python3
"""Weighted bottomonia primordial old-vs-new analysis.

This script is deliberately self-contained under scripts/cnm_primordial_importance.
It does not modify the existing CNM, HNM, or CNM x HNM publication workflows.

New TAMU importance-sampled files have an 8th trajectory metadata column:

    b  x0  y0  phi  pT  phi_p  y  weight

Old files have only the first seven metadata columns and are treated as uniform
weight samples.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
TAMU_LIB = REPO_ROOT / "hnm" / "tamu-traj" / "data_analysis" / "primordial_code"
if str(TAMU_LIB) not in sys.path:
    sys.path.insert(0, str(TAMU_LIB))

from ups_particle import make_bottomonia_system  # noqa: E402


STATE_DIRECT: Tuple[str, ...] = (
    "ups1S", "ups2S", "chib0_1P", "chib1_1P", "chib2_1P",
    "ups3S", "chib0_2P", "chib1_2P", "chib2_2P",
)
STATE_MAIN: Tuple[str, ...] = ("ups1S", "ups2S", "ups3S")
STATE_LABEL: Mapping[str, str] = {
    "ups1S": "1S",
    "ups2S": "2S",
    "ups3S": "3S",
}
STATE_TEX: Mapping[str, str] = {
    "ups1S": r"$\Upsilon(1S)$",
    "ups2S": r"$\Upsilon(2S)$",
    "ups3S": r"$\Upsilon(3S)$",
}
STATE_COLORS: Mapping[str, str] = {
    "ups1S": "#2ca02c",
    "ups2S": "#1f77b4",
    "ups3S": "#d62728",
}
STATE_LS: Mapping[str, object] = {
    "ups1S": "--",
    "ups2S": (0, (4, 2, 1, 2)),
    "ups3S": ":",
}
PRIM_HATCH = "//"

CENT_BINS: Tuple[Tuple[float, float], ...] = (
    (0.0, 10.0), (10.0, 20.0), (20.0, 40.0),
    (40.0, 60.0), (60.0, 80.0), (80.0, 100.0),
)
PT_EDGES_FULL = np.arange(0.5, 20.0 + 0.5 + 1e-9, 1.0, dtype=np.float64)
PT_EDGES_SMOKE = np.array([0.5, 5.5, 10.5, 20.5], dtype=np.float64)
MB_C0 = 0.25


@dataclass(frozen=True)
class SystemConfig:
    key: str
    label: str
    sqrts_gev: float
    input_tag: str
    b_grid: Tuple[float, ...]
    y_edges: np.ndarray
    y_edges_smoke: np.ndarray
    y_windows: Tuple[Tuple[str, Tuple[float, float], str], ...]
    mb_window: Tuple[float, float]
    pt_max: float


SYSTEMS: Mapping[str, SystemConfig] = {
    "lhc_oo5360": SystemConfig(
        key="lhc_oo5360",
        label=r"O+O, $\sqrt{s_{\mathrm{NN}}}=5.36$ TeV",
        sqrts_gev=5360.0,
        input_tag="OO5360",
        b_grid=(0.0, 1.3908, 2.5429, 3.5962, 4.6575, 5.5436, 6.7204),
        y_edges=np.arange(-4.0, 4.0 + 0.5, 0.5, dtype=np.float64),
        y_edges_smoke=np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float64),
        y_windows=(
            ("midrapidity", (-2.4, 2.4), "|y| < 2.4"),
            ("forward", (2.5, 4.0), "2.5 < y < 4"),
        ),
        mb_window=(0.0, 100.0),
        pt_max=20.0,
    ),
    "rhic_oo200": SystemConfig(
        key="rhic_oo200",
        label=r"O+O, $\sqrt{s_{\mathrm{NN}}}=200$ GeV",
        sqrts_gev=200.0,
        input_tag="OO200",
        b_grid=(0.0, 1.3225, 2.4181, 3.4198, 4.4307, 5.2924, 6.4887),
        y_edges=np.arange(-2.0, 2.0 + 0.5, 0.5, dtype=np.float64),
        y_edges_smoke=np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64),
        y_windows=(
            ("midrapidity", (-1.0, 1.0), "|y| < 1"),
            ("forward", (1.0, 2.0), "1 < y < 2"),
        ),
        mb_window=(0.0, 80.0),
        pt_max=15.0,
    ),
}

VARIANTS: Mapping[str, Tuple[str, str]] = {
    "nonp": ("bottomonia_nonp_{tag}_opt_production", "TAMU-NP"),
    "pert": ("bottomonia_pert_{tag}_opt_production", "TAMU-P"),
}
GREEN = "#2ca02c"
CNM_SCRIPT_ROOT = REPO_ROOT / "cnm" / "cnm_scripts"
CNM_CODE_PATHS = (
    REPO_ROOT / "cnm" / "eloss_code",
    REPO_ROOT / "cnm" / "cnm_combine",
    REPO_ROOT / "cnm" / "npdf_code",
)
M_UPSILON = 9.46
M_UPSILON_AVG = 10.01
RHIC_ABS_SIGMA_MB = 4.2


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    paths: Mapping[str, Path]
    central_policy: str
    fixed_b_reference: bool = False


def _cent_tag(c0: float, c1: float) -> str:
    return f"{int(c0)}-{int(c1)}%"


def _round_b(v: float) -> float:
    return round(float(v), 4)


def _normalize_bottomonia_row(row: Sequence[float]) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float64).reshape(-1)
    if arr.size >= 9:
        return arr[:9]
    if arr.size >= 5:
        v = arr[:5]
        return np.array([v[0], v[1], v[2], v[2], v[2], v[3], v[4], v[4], v[4]], dtype=np.float64)
    raise ValueError(f"Suppression row has {arr.size} columns; need at least 5.")


def read_tamu_datafile(path: Path, *, max_events_per_b: int | None = None) -> pd.DataFrame:
    """Read alternating TAMU metadata/suppression rows.

    Metadata column 8, when present, is the physical importance-sampling weight.
    Old 7-column metadata rows receive weight 1.0.
    """
    data: Dict[str, List[float]] = {"b": [], "pt": [], "y": [], "weight": []}
    for name in STATE_DIRECT:
        data[name] = []

    counts: Dict[float, int] = {}
    n_pairs = 0
    pending_meta: List[float] | None = None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                row = [float(x) for x in s.split()]
            except ValueError:
                continue
            if pending_meta is None:
                pending_meta = row
                continue
            meta, sup = pending_meta, row
            pending_meta = None
            n_pairs += 1

            if len(meta) < 7:
                raise RuntimeError(f"metadata row has {len(meta)} columns; need >= 7 in {path}")
            b_key = _round_b(meta[0])
            if max_events_per_b is not None:
                n_seen = counts.get(b_key, 0)
                if n_seen >= max_events_per_b:
                    continue
                counts[b_key] = n_seen + 1
            weight = float(meta[7]) if len(meta) >= 8 else 1.0
            surv = _normalize_bottomonia_row(sup)
            data["b"].append(float(meta[0]))
            data["pt"].append(float(meta[4]))
            data["y"].append(float(meta[6]))
            data["weight"].append(weight)
            for j, name in enumerate(STATE_DIRECT):
                data[name].append(float(surv[j]))

    if n_pairs == 0:
        raise RuntimeError(f"No metadata/suppression pairs parsed from {path}")
    if pending_meta is not None:
        # Odd numeric row at EOF: ignore it, matching the legacy reader behavior.
        pass

    return pd.DataFrame(data)


def _weighted_direct_mean(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.full(len(STATE_DIRECT), np.nan, dtype=np.float64)
    values = df[list(STATE_DIRECT)].to_numpy(dtype=np.float64)
    weights = df["weight"].to_numpy(dtype=np.float64)
    out = np.full(len(STATE_DIRECT), np.nan, dtype=np.float64)
    for j in range(len(STATE_DIRECT)):
        m = np.isfinite(values[:, j]) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(m):
            continue
        wsum = float(np.sum(weights[m]))
        if wsum > 0.0 and np.isfinite(wsum):
            out[j] = float(np.sum(weights[m] * values[m, j]) / wsum)
    return out


def _apply_feeddown(direct: np.ndarray, system) -> np.ndarray:
    if not np.all(np.isfinite(direct)):
        return np.full(len(STATE_DIRECT), np.nan, dtype=np.float64)
    sig = system.sigma_dir
    num = system.F @ (sig * direct)
    den = system.F @ sig
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0.0, num / den, np.nan)


def _main_from_direct(direct: np.ndarray, system) -> np.ndarray:
    obs = _apply_feeddown(direct, system)
    return np.asarray([obs[system.index(name)] for name in STATE_MAIN], dtype=np.float64)


def _select(df: pd.DataFrame,
            *,
            b_value: float | None = None,
            y_window: Tuple[float, float] | None = None,
            pt_window: Tuple[float, float] | None = None) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    if b_value is not None:
        mask &= np.isclose(df["b"].to_numpy(dtype=float), float(b_value), atol=1e-3)
    if y_window is not None:
        y0, y1 = y_window
        y = df["y"].to_numpy(dtype=float)
        mask &= (y >= float(y0)) & (y <= float(y1))
    if pt_window is not None:
        p0, p1 = pt_window
        pt = df["pt"].to_numpy(dtype=float)
        mask &= (pt >= float(p0)) & (pt <= float(p1))
    return df.loc[mask]


def _centrality_weights(cfg: SystemConfig) -> Dict[str, float]:
    edges = np.asarray([CENT_BINS[0][0]] + [b for _, b in CENT_BINS], dtype=np.float64) / 100.0
    raw = np.exp(-edges[:-1] / MB_C0) - np.exp(-edges[1:] / MB_C0)
    mb0, mb1 = cfg.mb_window[0] / 100.0, cfg.mb_window[1] / 100.0
    keep = np.asarray([
        (a / 100.0 >= mb0 - 1e-12) and (b / 100.0 <= mb1 + 1e-12)
        for a, b in CENT_BINS
    ], dtype=bool)
    raw = raw * keep
    total = float(np.sum(raw))
    if total <= 0.0:
        raise RuntimeError(f"Empty MB centrality window for {cfg.key}")
    return {_cent_tag(a, b): float(w / total) for (a, b), w in zip(CENT_BINS, raw)}


def _direct_for_centrality_bin(df: pd.DataFrame,
                               cfg: SystemConfig,
                               cent_index: int,
                               *,
                               y_window: Tuple[float, float] | None,
                               pt_window: Tuple[float, float] | None) -> np.ndarray:
    b_value = cfg.b_grid[cent_index + 1]
    return _weighted_direct_mean(_select(df, b_value=b_value, y_window=y_window, pt_window=pt_window))


def _direct_for_mb(df: pd.DataFrame,
                   cfg: SystemConfig,
                   *,
                   y_window: Tuple[float, float] | None,
                   pt_window: Tuple[float, float] | None) -> np.ndarray:
    weights = _centrality_weights(cfg)
    acc = np.zeros(len(STATE_DIRECT), dtype=np.float64)
    used = 0.0
    for cent_index, (c0, c1) in enumerate(CENT_BINS):
        tag = _cent_tag(c0, c1)
        w = weights.get(tag, 0.0)
        if w <= 0.0:
            continue
        direct = _direct_for_centrality_bin(
            df, cfg, cent_index, y_window=y_window, pt_window=pt_window,
        )
        if not np.all(np.isfinite(direct)):
            continue
        acc += w * direct
        used += w
    if used <= 0.0:
        return np.full(len(STATE_DIRECT), np.nan, dtype=np.float64)
    return acc / used


def _run_vs_y(df: pd.DataFrame, cfg: SystemConfig, y_edges: np.ndarray, system) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        y_center = 0.5 * (float(y0) + float(y1))
        for cent_index, (c0, c1) in enumerate(CENT_BINS):
            direct = _direct_for_centrality_bin(
                df, cfg, cent_index, y_window=(float(y0), float(y1)), pt_window=(0.0, cfg.pt_max),
            )
            values = _main_from_direct(direct, system)
            row = {
                "y_center": y_center,
                "centrality": _cent_tag(c0, c1),
                "mb": False,
                "cent_left": c0,
                "cent_right": c1,
            }
            for k, name in enumerate(STATE_MAIN):
                row[name] = float(values[k])
            rows.append(row)
        direct = _direct_for_mb(df, cfg, y_window=(float(y0), float(y1)), pt_window=(0.0, cfg.pt_max))
        values = _main_from_direct(direct, system)
        row = {
            "y_center": y_center,
            "centrality": "MB",
            "mb": True,
            "cent_left": cfg.mb_window[0],
            "cent_right": cfg.mb_window[1],
        }
        for k, name in enumerate(STATE_MAIN):
            row[name] = float(values[k])
        rows.append(row)
    return pd.DataFrame(rows)


def _run_vs_pt(df: pd.DataFrame,
               cfg: SystemConfig,
               pt_edges: np.ndarray,
               window: Tuple[float, float],
               system) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for p0, p1 in zip(pt_edges[:-1], pt_edges[1:]):
        p_center = 0.5 * (float(p0) + float(p1))
        for cent_index, (c0, c1) in enumerate(CENT_BINS):
            direct = _direct_for_centrality_bin(
                df, cfg, cent_index, y_window=window, pt_window=(float(p0), float(p1)),
            )
            values = _main_from_direct(direct, system)
            row = {
                "pT_center": p_center,
                "centrality": _cent_tag(c0, c1),
                "mb": False,
                "cent_left": c0,
                "cent_right": c1,
                "y_min": window[0],
                "y_max": window[1],
            }
            for k, name in enumerate(STATE_MAIN):
                row[name] = float(values[k])
            rows.append(row)
        direct = _direct_for_mb(df, cfg, y_window=window, pt_window=(float(p0), float(p1)))
        values = _main_from_direct(direct, system)
        row = {
            "pT_center": p_center,
            "centrality": "MB",
            "mb": True,
            "cent_left": cfg.mb_window[0],
            "cent_right": cfg.mb_window[1],
            "y_min": window[0],
            "y_max": window[1],
        }
        for k, name in enumerate(STATE_MAIN):
            row[name] = float(values[k])
        rows.append(row)
    return pd.DataFrame(rows)


def _run_vs_centrality(df: pd.DataFrame,
                       cfg: SystemConfig,
                       window: Tuple[float, float],
                       system) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cent_index, (c0, c1) in enumerate(CENT_BINS):
        direct = _direct_for_centrality_bin(
            df, cfg, cent_index, y_window=window, pt_window=(0.0, cfg.pt_max),
        )
        values = _main_from_direct(direct, system)
        row = {
            "cent_left": c0,
            "cent_right": c1,
            "cent_center": 0.5 * (c0 + c1),
            "centrality": _cent_tag(c0, c1),
            "mb": False,
            "y_min": window[0],
            "y_max": window[1],
        }
        for k, name in enumerate(STATE_MAIN):
            row[name] = float(values[k])
        rows.append(row)
    direct = _direct_for_mb(df, cfg, y_window=window, pt_window=(0.0, cfg.pt_max))
    values = _main_from_direct(direct, system)
    row = {
        "cent_left": cfg.mb_window[0],
        "cent_right": cfg.mb_window[1],
        "cent_center": 0.5 * (cfg.mb_window[0] + cfg.mb_window[1]),
        "centrality": "MB",
        "mb": True,
        "y_min": window[0],
        "y_max": window[1],
    }
    for k, name in enumerate(STATE_MAIN):
        row[name] = float(values[k])
    rows.append(row)
    return pd.DataFrame(rows)


def _combine_band(run_tables: Mapping[str, pd.DataFrame],
                  *,
                  key_cols: Sequence[str],
                  central_policy: str) -> pd.DataFrame:
    labels = list(run_tables)
    if not labels:
        raise RuntimeError("No run tables to combine.")
    merged = None
    for label, table in run_tables.items():
        rename = {name: f"{name}__{label}" for name in STATE_MAIN}
        sub = table[list(key_cols) + list(STATE_MAIN)].rename(columns=rename)
        merged = sub if merged is None else merged.merge(sub, on=list(key_cols), how="outer")
    assert merged is not None
    for name in STATE_MAIN:
        vals = np.stack(
            [merged[f"{name}__{label}"].to_numpy(dtype=np.float64) for label in labels],
            axis=1,
        )
        if central_policy == "center" and "center" in labels:
            center = merged[f"{name}__center"].to_numpy(dtype=np.float64)
        else:
            finite = np.isfinite(vals)
            count = np.sum(finite, axis=1)
            total = np.sum(np.where(finite, vals, 0.0), axis=1)
            center = np.full(vals.shape[0], np.nan, dtype=np.float64)
            center[count > 0] = total[count > 0] / count[count > 0]
        merged[f"{name}_central"] = center
        finite = np.isfinite(vals)
        has_any = np.any(finite, axis=1)
        lo = np.full(vals.shape[0], np.nan, dtype=np.float64)
        hi = np.full(vals.shape[0], np.nan, dtype=np.float64)
        lo[has_any] = np.min(np.where(finite[has_any], vals[has_any], np.inf), axis=1)
        hi[has_any] = np.max(np.where(finite[has_any], vals[has_any], -np.inf), axis=1)
        merged[f"{name}_lo"] = lo
        merged[f"{name}_hi"] = hi
    keep = list(key_cols)
    for name in STATE_MAIN:
        keep += [f"{name}_central", f"{name}_lo", f"{name}_hi"]
    return _sort_output_table(merged[keep])


def _sort_output_table(df: pd.DataFrame) -> pd.DataFrame:
    if "centrality" not in df.columns:
        return df
    ordered_tags = [_cent_tag(a, b) for a, b in CENT_BINS] + ["MB"]
    order = {tag: i for i, tag in enumerate(ordered_tags)}
    out = df.copy()
    out["__cent_order"] = out["centrality"].map(order).fillna(999).astype(int)
    if "y_center" in out.columns:
        sort_cols = ["y_center", "__cent_order"]
    elif "pT_center" in out.columns:
        sort_cols = ["pT_center", "__cent_order"]
    elif "cent_center" in out.columns:
        sort_cols = ["__cent_order"]
    else:
        sort_cols = ["__cent_order"]
    return out.sort_values(sort_cols).drop(columns=["__cent_order"]).reset_index(drop=True)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _pt_edges_for_cfg(cfg: SystemConfig, *, smoke: bool) -> np.ndarray:
    if smoke:
        return np.array([0.5, 5.5, 10.5, float(cfg.pt_max) + 0.5], dtype=np.float64)
    return np.arange(0.5, float(cfg.pt_max) + 0.5 + 1e-9, 1.0, dtype=np.float64)


def _interp_series(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(xp) & np.isfinite(fp)
    if not np.any(m):
        return np.full_like(x, np.nan, dtype=np.float64)
    order = np.argsort(xp[m])
    return np.interp(x, xp[m][order], fp[m][order], left=fp[m][order][0], right=fp[m][order][-1])


def _nan_triplet_like(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    nan = np.full_like(x, np.nan, dtype=np.float64)
    return nan.copy(), nan.copy(), nan.copy()


def _ensure_cnm_paths() -> None:
    for path in (CNM_SCRIPT_ROOT, *CNM_CODE_PATHS):
        p = str(path)
        while p in sys.path:
            sys.path.remove(p)
    for path in reversed((CNM_SCRIPT_ROOT, *CNM_CODE_PATHS)):
        sys.path.insert(0, str(path))


def _cent_tags(include_mb: bool = True) -> List[str]:
    tags = [_cent_tag(a, b) for a, b in CENT_BINS]
    if include_mb:
        tags.append("MB")
    return tags


def _table_from_cnm_bands(xcol: str, centers: np.ndarray, tags: Sequence[str],
                          bands: Mapping[str, Tuple[Mapping[str, np.ndarray],
                                                    Mapping[str, np.ndarray],
                                                    Mapping[str, np.ndarray]]]) -> pd.DataFrame:
    Rc, Rlo, Rhi = bands["cnm"]
    rows: List[Dict[str, object]] = []
    for i, xval in enumerate(np.asarray(centers, dtype=np.float64)):
        for tag in tags:
            if tag not in Rc:
                continue
            rows.append({
                xcol: float(xval),
                "centrality": tag,
                "cnm_central": float(np.asarray(Rc[tag], dtype=np.float64)[i]),
                "cnm_lo": float(np.asarray(Rlo[tag], dtype=np.float64)[i]),
                "cnm_hi": float(np.asarray(Rhi[tag], dtype=np.float64)[i]),
            })
    return pd.DataFrame(rows)


def _table_from_cnm_centrality(win_key: str, cfg: SystemConfig, bands: Mapping[str, Tuple]) -> pd.DataFrame:
    Rc, Rlo, Rhi, mb_c, mb_lo, mb_hi = bands["cnm"]
    rows: List[Dict[str, object]] = []
    y_window = dict((k, w) for k, w, _ in cfg.y_windows)[win_key]
    for i, (c0, c1) in enumerate(CENT_BINS):
        rows.append({
            "cent_left": c0,
            "cent_right": c1,
            "cent_center": 0.5 * (c0 + c1),
            "centrality": _cent_tag(c0, c1),
            "cnm_central": float(np.asarray(Rc, dtype=np.float64)[i]),
            "cnm_lo": float(np.asarray(Rlo, dtype=np.float64)[i]),
            "cnm_hi": float(np.asarray(Rhi, dtype=np.float64)[i]),
            "y_min": y_window[0],
            "y_max": y_window[1],
        })
    rows.append({
        "cent_left": cfg.mb_window[0],
        "cent_right": cfg.mb_window[1],
        "cent_center": 0.5 * (cfg.mb_window[0] + cfg.mb_window[1]),
        "centrality": "MB",
        "cnm_central": float(mb_c),
        "cnm_lo": float(mb_lo),
        "cnm_hi": float(mb_hi),
        "y_min": y_window[0],
        "y_max": y_window[1],
    })
    return pd.DataFrame(rows)


def _recompute_mb_rows(table: pd.DataFrame, cfg: SystemConfig, xcol: str | None) -> pd.DataFrame:
    weights = _centrality_weights(cfg)
    finite = table[table["centrality"] != "MB"].copy()
    mb_rows: List[Dict[str, object]] = []
    group_cols = [xcol] if xcol is not None else ["__all"]
    if xcol is None:
        finite["__all"] = 0
    for _, grp in finite.groupby(group_cols, dropna=False):
        row: Dict[str, object] = {
            "centrality": "MB",
            "cnm_source": "",
        }
        if xcol is not None:
            row[xcol] = float(grp[xcol].iloc[0])
        for col in ("cnm_central", "cnm_lo", "cnm_hi"):
            num = 0.0
            den = 0.0
            for r in grp.itertuples(index=False):
                tag = str(getattr(r, "centrality"))
                val = float(getattr(r, col))
                w = float(weights.get(tag, 0.0))
                if w > 0.0 and np.isfinite(val):
                    num += w * val
                    den += w
            row[col] = float(num / den) if den > 0.0 else float("nan")
        for col in ("y_min", "y_max", "cent_left", "cent_right", "cent_center"):
            if col in grp.columns and col not in row:
                vals = grp[col].dropna().unique()
                if vals.size == 1:
                    row[col] = float(vals[0])
        if xcol is None:
            row["cent_left"] = cfg.mb_window[0]
            row["cent_right"] = cfg.mb_window[1]
            row["cent_center"] = 0.5 * (cfg.mb_window[0] + cfg.mb_window[1])
        mb_rows.append(row)
    out = table[table["centrality"] != "MB"].copy()
    if "__all" in out.columns:
        out = out.drop(columns=["__all"])
    if mb_rows:
        out = pd.concat([out, pd.DataFrame(mb_rows)], ignore_index=True)
    if "__all" in out.columns:
        out = out.drop(columns=["__all"])
    return _sort_output_table(out)


def _pin_cnm_edge_nans(table: pd.DataFrame, xcol: str) -> pd.DataFrame:
    """Fill unsupported edge CNM bins from the nearest finite CNM bin.

    This is only an edge-support guard for plotting/composition on requested
    axes. Interior NaNs remain NaNs because those would indicate a real hole.
    """
    out = table.copy()
    cols = ["cnm_central", "cnm_lo", "cnm_hi"]
    if xcol not in out.columns or not set(cols).issubset(out.columns):
        return out
    for _tag, idx in out.groupby("centrality").groups.items():
        sub = out.loc[idx].sort_values(xcol)
        order = sub.index.to_numpy()
        finite = np.ones(len(order), dtype=bool)
        for col in cols:
            finite &= np.isfinite(out.loc[order, col].to_numpy(dtype=np.float64))
        if not np.any(finite):
            continue
        first_finite = int(np.flatnonzero(finite)[0])
        last_finite = int(np.flatnonzero(finite)[-1])
        for pos in range(0, first_finite):
            for col in cols:
                out.loc[order[pos], col] = out.loc[order[first_finite], col]
        for pos in range(last_finite + 1, len(order)):
            for col in cols:
                out.loc[order[pos], col] = out.loc[order[last_finite], col]
    return _sort_output_table(out)


def _hold_cnm_above_pt(table: pd.DataFrame, hold_above: float) -> pd.DataFrame:
    out = table.copy()
    cols = ["cnm_central", "cnm_lo", "cnm_hi"]
    if "pT_center" not in out.columns or not set(cols).issubset(out.columns):
        return out
    for _tag, idx in out.groupby("centrality").groups.items():
        sub = out.loc[idx].sort_values("pT_center")
        ref = sub[sub["pT_center"] <= float(hold_above) + 1e-12].tail(1)
        if ref.empty:
            continue
        high_idx = sub[sub["pT_center"] > float(hold_above) + 1e-12].index
        for col in cols:
            out.loc[high_idx, col] = float(ref.iloc[0][col])
    return _sort_output_table(out)


def _combine_relative_quadrature(prim_c: np.ndarray, prim_lo: np.ndarray, prim_hi: np.ndarray,
                                 cnm_c: np.ndarray, cnm_lo: np.ndarray, cnm_hi: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prim_c = np.asarray(prim_c, dtype=np.float64)
    prim_lo = np.asarray(prim_lo, dtype=np.float64)
    prim_hi = np.asarray(prim_hi, dtype=np.float64)
    cnm_c = np.asarray(cnm_c, dtype=np.float64)
    cnm_lo = np.asarray(cnm_lo, dtype=np.float64)
    cnm_hi = np.asarray(cnm_hi, dtype=np.float64)
    central = prim_c * cnm_c
    with np.errstate(divide="ignore", invalid="ignore"):
        p_lo = np.where(np.abs(prim_c) > 1e-12, np.maximum(prim_c - prim_lo, 0.0) / np.abs(prim_c), 0.0)
        p_hi = np.where(np.abs(prim_c) > 1e-12, np.maximum(prim_hi - prim_c, 0.0) / np.abs(prim_c), 0.0)
        c_lo = np.where(np.abs(cnm_c) > 1e-12, np.maximum(cnm_c - cnm_lo, 0.0) / np.abs(cnm_c), 0.0)
        c_hi = np.where(np.abs(cnm_c) > 1e-12, np.maximum(cnm_hi - cnm_c, 0.0) / np.abs(cnm_c), 0.0)
    rel_lo = np.sqrt(p_lo * p_lo + c_lo * c_lo)
    rel_hi = np.sqrt(p_hi * p_hi + c_hi * c_hi)
    lo = central * (1.0 - rel_lo)
    hi = central * (1.0 + rel_hi)
    lo = np.where(np.isfinite(lo), np.maximum(lo, 0.0), np.nan)
    hi = np.where(np.isfinite(hi), hi, np.nan)
    central = np.where(np.isfinite(central), central, np.nan)
    return central, lo, hi


@dataclass(frozen=True)
class CNMTables:
    y: pd.DataFrame
    pt: Mapping[str, pd.DataFrame]
    centrality: Mapping[str, pd.DataFrame]
    source: str


_CNM_TABLE_CACHE: Dict[Tuple[str, Tuple[float, ...], Tuple[float, ...]], CNMTables] = {}


def _pin_cnm_edge_bins(bands_y, yc: np.ndarray) -> None:
    if yc.size < 3:
        return
    for comp in bands_y:
        for i in range(3):
            d = bands_y[comp][i]
            for tag in d:
                vals = np.asarray(d[tag], dtype=np.float64).copy()
                vals[0] = vals[1]
                vals[-1] = vals[-2]
                d[tag] = vals


def _patch_module_attrs(module, updates: Mapping[str, object]):
    old = {name: getattr(module, name) for name in updates if hasattr(module, name)}
    for name, value in updates.items():
        setattr(module, name, value)
    return old


def _restore_module_attrs(module, old: Mapping[str, object]) -> None:
    for name, value in old.items():
        setattr(module, name, value)


def _build_lhc_cnm_context(cfg: SystemConfig, y_edges: np.ndarray, pt_edges: np.ndarray):
    _ensure_cnm_paths()
    import run_bottomonia_cnm_OO as cnm_oo  # noqa: E402

    y_windows = [(float(y0), float(y1), rf"${tex}$") for _, (y0, y1), tex in cfg.y_windows]
    updates = {
        "CENT_BINS": [(float(a), float(b)) for a, b in CENT_BINS],
        "Y_EDGES": np.asarray(y_edges, dtype=np.float64),
        "P_EDGES": np.asarray(pt_edges, dtype=np.float64),
        "PT_RANGE_AVG": (0.0, float(cfg.pt_max)),
        "Y_WINDOWS": y_windows,
    }
    old = _patch_module_attrs(cnm_oo, updates)
    try:
        cnm = cnm_oo.build_eloss_context("5.36")
        cnm.cent_bins = [(float(a), float(b)) for a, b in CENT_BINS]
        cnm.y_edges = np.asarray(y_edges, dtype=np.float64)
        cnm.p_edges = np.asarray(pt_edges, dtype=np.float64)
        cnm.pt_range_avg = (0.0, float(cfg.pt_max))
        cnm.y_windows = y_windows
        return cnm, "live:cnm/cnm_scripts/run_bottomonia_cnm_OO.py"
    finally:
        _restore_module_attrs(cnm_oo, old)


def _build_oo_npdf_grid_from_epps(cfg: SystemConfig, y_edges: np.ndarray, pt_edges: np.ndarray) -> pd.DataFrame:
    _ensure_cnm_paths()
    from gluon_ratio import EPPS21Ratio, GluonEPPSProvider  # noqa: E402

    y_centers = 0.5 * (np.asarray(y_edges[:-1], dtype=np.float64) + np.asarray(y_edges[1:], dtype=np.float64))
    pt_centers = 0.5 * (np.asarray(pt_edges[:-1], dtype=np.float64) + np.asarray(pt_edges[1:], dtype=np.float64))
    yy, pp = np.meshgrid(y_centers, pt_centers, indexing="ij")
    y_flat = yy.reshape(-1)
    pt_flat = pp.reshape(-1)

    epps = EPPS21Ratio(A=16, path=str(REPO_ROOT / "inputs" / "npdf" / "nPDFs"))
    g_forward = GluonEPPSProvider(epps, cfg.sqrts_gev, m_state_GeV=M_UPSILON_AVG, y_sign_for_xA=1)
    g_backward = GluonEPPSProvider(epps, cfg.sqrts_gev, m_state_GeV=M_UPSILON_AVG, y_sign_for_xA=-1)

    members: List[np.ndarray] = []
    for sid in range(1, 50):
        r1 = np.asarray(g_forward.SA_ypt_set(y_flat, pt_flat, set_id=sid), dtype=np.float64)
        r2 = np.asarray(g_backward.SA_ypt_set(y_flat, pt_flat, set_id=sid), dtype=np.float64)
        members.append(r1 * r2)
    r0 = members[0]
    M = np.stack(members[1:], axis=0)
    D = M[0::2, :] - M[1::2, :]
    h = 0.5 * np.sqrt(np.sum(D * D, axis=0))
    out = pd.DataFrame({
        "y": y_flat,
        "pt": pt_flat,
        "r_central": r0,
        "r_lo": r0 - h,
        "r_hi": r0 + h,
    })
    for j, arr in enumerate(M, start=1):
        out[f"r_mem_{j:03d}"] = arr
    return out


def _build_rhic_cnm_context(cfg: SystemConfig, y_edges: np.ndarray, pt_edges: np.ndarray):
    _ensure_cnm_paths()
    from glauber import OpticalGlauber, SystemSpec  # noqa: E402
    from gluon_ratio import EPPS21Ratio, GluonEPPSProvider  # noqa: E402
    from npdf_centrality import compute_df49_by_centrality  # noqa: E402
    from particle import Particle  # noqa: E402
    from coupling import alpha_s_provider  # noqa: E402
    import quenching_fast as QF  # noqa: E402
    from cnm_combine_fast_nuclabs import CNMCombineFast  # noqa: E402

    grid = _build_oo_npdf_grid_from_epps(cfg, y_edges, pt_edges)
    gl = OpticalGlauber(
        SystemSpec("AA", cfg.sqrts_gev, A=16, sigma_nn_mb=42.0),
        nx_pa=64, ny_pa=64, verbose=False,
    )
    r0 = grid["r_central"].to_numpy(dtype=np.float64)
    M = grid[[f"r_mem_{i:03d}" for i in range(1, 49)]].to_numpy(dtype=np.float64).T
    SA_all = np.vstack([r0[None, :], M])

    epps = EPPS21Ratio(A=16, path=str(REPO_ROOT / "inputs" / "npdf" / "nPDFs"))
    gluon = GluonEPPSProvider(epps, cfg.sqrts_gev, m_state_GeV=M_UPSILON_AVG)
    df49_by_cent, _K_by_cent, _SA_all, _y_shift = compute_df49_by_centrality(
        grid, r0, M, gluon, gl,
        cent_bins=[(float(a), float(b)) for a, b in CENT_BINS],
        nb_bsamples=5, y_shift_fraction=0.0, kind="AA", SA_all=SA_all,
    )
    particle = Particle(family="bottomonia", state="avg", mass_override_GeV=M_UPSILON)
    alpha_s = alpha_s_provider(mode="running", LambdaQCD=0.25)
    Lmb = gl.leff_minbias_AA()
    qp_base = QF.QuenchParams(
        qhat0=0.075, lp_fm=1.5,
        LA_fm=Lmb, LB_fm=Lmb,
        system="AA", lambdaQCD=0.25, roots_GeV=cfg.sqrts_gev,
        alpha_of_mu=alpha_s, alpha_scale="mT",
        use_hard_cronin=True, mapping="exp", device="cpu",
    )
    cnm = CNMCombineFast(
        energy="0.200", family="bottomonia", particle_state="avg",
        sqrt_sNN=cfg.sqrts_gev, sigma_nn_mb=42.0,
        cent_bins=[(float(a), float(b)) for a, b in CENT_BINS],
        y_edges=np.asarray(y_edges, dtype=np.float64),
        p_edges=np.asarray(pt_edges, dtype=np.float64),
        y_windows=[(float(y0), float(y1), tex) for _, (y0, y1), tex in cfg.y_windows],
        pt_range_avg=(0.0, float(cfg.pt_max)), pt_floor_w=1.0,
        weight_mode="flat", y_ref=0.0, cent_c0=MB_C0,
        q0_pair=(0.05, 0.09), p0_scale_pair=(0.9, 1.1), nb_bsamples=5,
        y_shift_fraction=0.0, particle=particle,
        npdf_ctx=dict(df49_by_cent=df49_by_cent, df_pp=grid, df_pa=grid, gluon=gluon),
        gl=gl, qp_base=qp_base, spec=gl.spec,
    )
    return cnm, f"live:OO200 EPPS21 A=16 + eloss/broad + absorption sigma_abs={RHIC_ABS_SIGMA_MB:g} mb"


def _absorption_factor_for_cent(glauber, c0: float, c1: float, *, sigma_abs_mb: float) -> float:
    sigma_abs_fm2 = 0.1 * float(sigma_abs_mb)
    ps = np.linspace(float(c0) / 100.0, float(c1) / 100.0, 5)
    vals = []
    for p in ps:
        b = float(glauber.b_from_percentile(float(p), kind="AA"))
        if hasattr(glauber, "TA_r"):
            thickness = 2.0 * float(glauber.TA_r(b))
        else:
            thickness = 0.0
        vals.append(math.exp(-sigma_abs_fm2 * max(thickness, 0.0)))
    return float(np.mean(vals))


def _apply_rhic_absorption_to_table(table: pd.DataFrame, cfg: SystemConfig, glauber) -> pd.DataFrame:
    out = table.copy()
    if cfg.key != "rhic_oo200":
        return out
    factors = {
        _cent_tag(a, b): _absorption_factor_for_cent(glauber, a, b, sigma_abs_mb=RHIC_ABS_SIGMA_MB)
        for a, b in CENT_BINS
    }
    weights = _centrality_weights(cfg)
    mb_num = sum(weights.get(tag, 0.0) * val for tag, val in factors.items())
    mb_den = sum(weights.get(tag, 0.0) for tag in factors)
    factors["MB"] = mb_num / mb_den if mb_den > 0.0 else float("nan")
    fac = out["centrality"].astype(str).map(factors).to_numpy(dtype=np.float64)
    for col in ("cnm_central", "cnm_lo", "cnm_hi"):
        out[col] = out[col].to_numpy(dtype=np.float64) * fac
    return out


def _compute_cnm_tables(cfg: SystemConfig, y_edges: np.ndarray, pt_edges: np.ndarray,
                        logger: logging.Logger) -> CNMTables:
    key = (
        cfg.key,
        tuple(np.asarray(y_edges, dtype=np.float64).round(8)),
        tuple(np.asarray(pt_edges, dtype=np.float64).round(8)),
    )
    cached = _CNM_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    if cfg.key == "lhc_oo5360":
        logger.info("[%s] computing CNM live via bottomonia OO CNM script", cfg.key)
        cnm, source = _build_lhc_cnm_context(cfg, y_edges, pt_edges)
    elif cfg.key == "rhic_oo200":
        logger.info("[%s] computing CNM live from OO200 CNM modules", cfg.key)
        cnm, source = _build_rhic_cnm_context(cfg, y_edges, pt_edges)
    else:
        raise ValueError(f"CNM runtime is not configured for {cfg.key}")

    yc, tags_y, bands_y = cnm.cnm_vs_y(
        y_edges=np.asarray(y_edges, dtype=np.float64),
        pt_range_avg=(0.0, float(cfg.pt_max)),
        components=("cnm",),
        include_mb=True,
    )
    _pin_cnm_edge_bins(bands_y, np.asarray(yc, dtype=np.float64))
    y_table = _table_from_cnm_bands("y_center", np.asarray(yc, dtype=np.float64), tags_y, bands_y)
    y_table = _recompute_mb_rows(y_table, cfg, "y_center")

    pt_tables: Dict[str, pd.DataFrame] = {}
    cent_tables: Dict[str, pd.DataFrame] = {}
    for win_key, y_window, _ in cfg.y_windows:
        pc, tags_pt, bands_pt = cnm.cnm_vs_pT(
            y_window,
            np.asarray(pt_edges, dtype=np.float64),
            components=("cnm",),
            include_mb=True,
        )
        pt_table = _table_from_cnm_bands("pT_center", np.asarray(pc, dtype=np.float64), tags_pt, bands_pt)
        pt_table = _recompute_mb_rows(pt_table, cfg, "pT_center")
        pt_table = _pin_cnm_edge_nans(pt_table, "pT_center")
        pt_table["y_min"] = float(y_window[0])
        pt_table["y_max"] = float(y_window[1])
        pt_tables[win_key] = pt_table

        bands_cent = cnm.cnm_vs_centrality(
            y_window,
            pt_range_avg=(0.0, float(cfg.pt_max)),
            components=("cnm",),
            include_mb=True,
        )
        cent_table = _table_from_cnm_centrality(win_key, cfg, bands_cent)
        cent_table = _recompute_mb_rows(cent_table, cfg, None)
        if "y_min" not in cent_table.columns:
            cent_table["y_min"] = float(y_window[0])
            cent_table["y_max"] = float(y_window[1])
        cent_tables[win_key] = cent_table

    if cfg.key == "rhic_oo200":
        y_table = _apply_rhic_absorption_to_table(y_table, cfg, cnm.gl)
        y_table = _recompute_mb_rows(y_table, cfg, "y_center")
        y_window_by_key = {wk: ww for wk, ww, _ in cfg.y_windows}
        rhic_pt_tables: Dict[str, pd.DataFrame] = {}
        for k, v in pt_tables.items():
            vv = _recompute_mb_rows(_apply_rhic_absorption_to_table(v, cfg, cnm.gl), cfg, "pT_center")
            y0, y1 = y_window_by_key[k]
            if float(y0) * float(y1) > 0.0:
                vv = _hold_cnm_above_pt(vv, 8.5)
            rhic_pt_tables[k] = _pin_cnm_edge_nans(vv, "pT_center")
        pt_tables = rhic_pt_tables
        cent_tables = {
            k: _recompute_mb_rows(_apply_rhic_absorption_to_table(v, cfg, cnm.gl), cfg, None)
            for k, v in cent_tables.items()
        }

    for table in [y_table, *pt_tables.values(), *cent_tables.values()]:
        table["cnm_source"] = source

    out = CNMTables(y=y_table, pt=pt_tables, centrality=cent_tables, source=source)
    _CNM_TABLE_CACHE[key] = out
    return out


def _merge_cnm_columns(df: pd.DataFrame, cnm: pd.DataFrame, *,
                       key_cols: Sequence[str]) -> pd.DataFrame:
    out = df.merge(
        cnm[list(key_cols) + ["cnm_central", "cnm_lo", "cnm_hi", "cnm_source"]],
        on=list(key_cols),
        how="left",
    )
    for name in STATE_MAIN:
        c, lo, hi = _combine_relative_quadrature(
            out[f"{name}_central"].to_numpy(dtype=np.float64),
            out[f"{name}_lo"].to_numpy(dtype=np.float64),
            out[f"{name}_hi"].to_numpy(dtype=np.float64),
            out["cnm_central"].to_numpy(dtype=np.float64),
            out["cnm_lo"].to_numpy(dtype=np.float64),
            out["cnm_hi"].to_numpy(dtype=np.float64),
        )
        out[f"{name}_cnm_central"] = c
        out[f"{name}_cnm_lo"] = lo
        out[f"{name}_cnm_hi"] = hi
    return _sort_output_table(out)


def _configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 1.0,
    })
    return plt


def _style_axis(ax, xlabel: str) -> None:
    ax.axhline(1.0, color="0.55", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$R_{AA}$")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True,
                   labelbottom=True, labelleft=True)


def _edges_from_centers(centers: np.ndarray, *, x_min: float | None = None,
                        x_max: float | None = None) -> np.ndarray:
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 0:
        return np.array([], dtype=np.float64)
    if c.size == 1:
        width = 1.0
        left = c[0] - 0.5 * width if x_min is None else float(x_min)
        right = c[0] + 0.5 * width if x_max is None else float(x_max)
        return np.array([left, right], dtype=np.float64)
    mids = 0.5 * (c[:-1] + c[1:])
    left = c[0] - (mids[0] - c[0]) if x_min is None else float(x_min)
    right = c[-1] + (c[-1] - mids[-1]) if x_max is None else float(x_max)
    return np.r_[left, mids, right].astype(np.float64)


def _contiguous_true_segments(mask: np.ndarray) -> List[np.ndarray]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    return [seg for seg in np.split(idx, breaks) if seg.size]


def _step_dashed_hatched(ax, centers: np.ndarray, central: np.ndarray,
                         lo: np.ndarray, hi: np.ndarray, *,
                         color: str, label: str | None,
                         linestyle: object = "--",
                         x_min: float | None = None,
                         x_max: float | None = None,
                         zorder: int = 3,
                         fill_alpha: float = 0.10,
                         hatch: str | None = None,
                         fill_color: str | None = None) -> None:
    centers = np.asarray(centers, dtype=np.float64)
    central = np.asarray(central, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    mask = np.isfinite(centers) & np.isfinite(central) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(mask):
        return
    for nseg, idx in enumerate(_contiguous_true_segments(mask)):
        start, stop = int(idx[0]), int(idx[-1])
        c = centers[start:stop + 1]
        edges = _edges_from_centers(c, x_min=x_min if start == 0 else None,
                                    x_max=x_max if stop == centers.size - 1 else None)
        y = central[start:stop + 1]
        ylo = lo[start:stop + 1]
        yhi = hi[start:stop + 1]
        ax.step(edges, np.r_[y, y[-1]], where="post", color=color, lw=1.9,
                ls=linestyle, label=label if nseg == 0 else None, zorder=zorder)
        if hatch:
            ax.fill_between(edges, np.r_[ylo, ylo[-1]], np.r_[yhi, yhi[-1]],
                            step="post", facecolor="none", edgecolor=fill_color or color,
                            hatch=hatch, linewidth=0.0, alpha=0.85,
                            zorder=zorder - 1)
        else:
            ax.fill_between(edges, np.r_[ylo, ylo[-1]], np.r_[yhi, yhi[-1]],
                            step="post", facecolor=fill_color or color, edgecolor="none",
                            alpha=fill_alpha, zorder=zorder - 1)


def _step_old_reference(ax, centers: np.ndarray, values: np.ndarray, *,
                        color: str, x_min: float | None = None,
                        x_max: float | None = None) -> None:
    centers = np.asarray(centers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(centers) & np.isfinite(values)
    if not np.any(mask):
        return
    idx = np.flatnonzero(mask)
    start, stop = int(idx[0]), int(idx[-1])
    c = centers[start:stop + 1]
    edges = _edges_from_centers(c, x_min=x_min if start == 0 else None,
                                x_max=x_max if stop == centers.size - 1 else None)
    y = values[start:stop + 1]
    ax.step(edges, np.r_[y, y[-1]], where="post", color=color,
            linestyle="-.", linewidth=1.15, alpha=0.90, zorder=6)


def _step_cnm_reference(ax, df: pd.DataFrame, xcol: str, *,
                        x_min: float | None = None,
                        x_max: float | None = None,
                        label: str | None = "CNM") -> None:
    if not {"cnm_central", "cnm_lo", "cnm_hi"}.issubset(df.columns):
        return
    x = df[xcol].to_numpy(dtype=np.float64)
    c = df["cnm_central"].to_numpy(dtype=np.float64)
    lo = df["cnm_lo"].to_numpy(dtype=np.float64)
    hi = df["cnm_hi"].to_numpy(dtype=np.float64)
    _step_dashed_hatched(
        ax, x, c, lo, hi,
        color="#555555", label=label, linestyle="-.",
        x_min=x_min, x_max=x_max, zorder=2,
        fill_alpha=0.22, fill_color="#9a9a9a",
    )


def _state_label(state: str, layer_label: str) -> str:
    return rf"{STATE_TEX[state]} {layer_label}"


def _cnm_legend_label(cfg: SystemConfig) -> str:
    base = r"CNM: nPDF $\times$ ELoss $\times$ $p_T$ broad"
    if cfg.key == "rhic_oo200":
        return base + r" $\times$ Nucl Abs"
    return base


def _finite_band(df: pd.DataFrame, xcol: str) -> pd.DataFrame:
    cols = [xcol]
    for name in STATE_MAIN:
        cols += [f"{name}_central", f"{name}_lo", f"{name}_hi"]
    mask = np.ones(len(df), dtype=bool)
    for col in cols:
        if col in df.columns:
            mask &= np.isfinite(df[col].to_numpy(dtype=np.float64))
    return df.loc[mask].copy().sort_values(xcol)


def _plot_step_panel(ax, df: pd.DataFrame, xcol: str, *,
                     old_ref: pd.DataFrame | None = None,
                     old_x_col: str | None = None,
                     x_min: float | None = None,
                     x_max: float | None = None,
                     variant_label: str = "TAMU-NP",
                     show_cnm_reference: bool = True,
                     cnm_label: str = "CNM") -> None:
    x = df[xcol].to_numpy(dtype=np.float64)
    if show_cnm_reference:
        _step_cnm_reference(ax, df, xcol, x_min=x_min, x_max=x_max, label=cnm_label)
    for name in STATE_MAIN:
        color = STATE_COLORS[name]
        c = df[f"{name}_central"].to_numpy(dtype=np.float64)
        lo = df[f"{name}_lo"].to_numpy(dtype=np.float64)
        hi = df[f"{name}_hi"].to_numpy(dtype=np.float64)
        _step_dashed_hatched(ax, x, c, lo, hi, color=color,
                             label=_state_label(name, f"{variant_label} Prim"),
                             linestyle=STATE_LS.get(name, "--"),
                             x_min=x_min, x_max=x_max,
                             fill_alpha=0.08)
        ccol = f"{name}_cnm_central"
        if ccol in df.columns:
            cc = df[ccol].to_numpy(dtype=np.float64)
            clo = df[f"{name}_cnm_lo"].to_numpy(dtype=np.float64)
            chi = df[f"{name}_cnm_hi"].to_numpy(dtype=np.float64)
            _step_dashed_hatched(ax, x, cc, clo, chi, color=color,
                                 label=_state_label(name, f"CNM x {variant_label} Prim"),
                                 linestyle="-", x_min=x_min, x_max=x_max,
                                 zorder=5, fill_alpha=0.20)
        if old_ref is not None and old_x_col is not None and f"{name}_central" in old_ref:
            ox = old_ref[old_x_col].to_numpy(dtype=np.float64)
            oy = old_ref[f"{name}_central"].to_numpy(dtype=np.float64)
            _step_old_reference(ax, ox, oy, color=color, x_min=x_min, x_max=x_max)


def _common_ylim(axes: Sequence) -> Tuple[float, float]:
    ymin = np.inf
    ymax = -np.inf
    for ax in axes:
        if not ax.get_visible():
            continue
        lo, hi = ax.get_ylim()
        ymin = min(ymin, lo)
        ymax = max(ymax, hi)
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        return 0.0, 1.15
    span = ymax - ymin
    return max(0.0, ymin - 0.05 * span), ymax + 0.18 * span


def _legend_panel(ax, handles, labels, cfg: SystemConfig, *,
                  y_window_tex: str | None = None,
                  old_ref: bool = False) -> None:
    ax.axis("off")
    if old_ref:
        from matplotlib.lines import Line2D
        handles = list(handles) + [
            Line2D([0], [0], color="0.25", linestyle="-.", linewidth=1.2),
        ]
        labels = list(labels) + ["old fixed-b Prim"]
    ncol = 2 if len(labels) > 4 else 1
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
              frameon=False, fontsize=8.2, ncol=ncol,
              columnspacing=0.9, handlelength=2.3, handletextpad=0.55)


def _annotate_energy(ax, cfg: SystemConfig, *, x: float = 0.05, y: float = 0.06) -> None:
    ax.text(x, y, cfg.label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12.5, fontweight="bold")


def _annotate_y_window(ax, y_window_tex: str, *, x: float = 0.95, y: float = 0.93,
                       ha: str = "right", va: str = "top") -> None:
    ax.text(x, y, y_window_tex, transform=ax.transAxes,
            ha=ha, va=va, fontsize=11.0, fontweight="bold")


def plot_vs_y_grid(df: pd.DataFrame,
                   cfg: SystemConfig,
                   out: Path,
                   title: str,
                   old_ref: pd.DataFrame | None = None,
                   variant_label: str = "TAMU-NP") -> None:
    plt = _configure_matplotlib()
    tags = [_cent_tag(a, b) for a, b in CENT_BINS] + ["MB"]
    n_panels = len(tags) + 1
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                             sharex=True, sharey=True, dpi=160,
                             layout="constrained")
    axes_flat = list(axes.ravel())
    data_axes = []
    for i, tag in enumerate(tags):
        ax = axes_flat[i]
        sub = df[df["centrality"] == tag].copy().sort_values("y_center")
        if sub.empty:
            ax.set_visible(False)
            continue
        old = old_ref if tag == "MB" else None
        _plot_step_panel(ax, sub, "y_center", old_ref=old, old_x_col="y_center",
                         x_min=float(cfg.y_edges[0]), x_max=float(cfg.y_edges[-1]),
                         variant_label=variant_label,
                         cnm_label=_cnm_legend_label(cfg))
        _style_axis(ax, r"$y$")
        ax.set_xlim(float(cfg.y_edges[0]), float(cfg.y_edges[-1]))
        ax.text(0.05, 0.93, tag, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, fontweight="bold")
        if i == 0:
            _annotate_energy(ax, cfg)
        data_axes.append(ax)
    if data_axes:
        ylim = _common_ylim(data_axes)
        for ax in data_axes:
            ax.set_ylim(*ylim)
        handles, labels = data_axes[0].get_legend_handles_labels()
        _legend_panel(axes_flat[len(tags)], handles, labels, cfg,
                      old_ref=old_ref is not None)
    for j in range(len(tags) + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_vs_pt_grid(df: pd.DataFrame,
                    cfg: SystemConfig,
                    out: Path,
                    title: str,
                    old_ref: pd.DataFrame | None = None,
                    variant_label: str = "TAMU-NP") -> None:
    plt = _configure_matplotlib()
    tags = [_cent_tag(a, b) for a, b in CENT_BINS] + ["MB"]
    n_panels = len(tags) + 1
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                             sharex=True, sharey=True, dpi=160,
                             layout="constrained")
    axes_flat = list(axes.ravel())
    data_axes = []
    y_window_tex = None
    if {"y_min", "y_max"}.issubset(df.columns) and not df.empty:
        y0 = float(df["y_min"].iloc[0])
        y1 = float(df["y_max"].iloc[0])
        y_window_tex = rf"$\mathbf{{{y0:g}<y<{y1:g}}}$"
    for i, tag in enumerate(tags):
        ax = axes_flat[i]
        sub = df[df["centrality"] == tag].copy().sort_values("pT_center")
        if sub.empty:
            ax.set_visible(False)
            continue
        old = old_ref if tag == "MB" else None
        _plot_step_panel(ax, sub, "pT_center", old_ref=old, old_x_col="pT_center",
                         x_min=0.0, x_max=cfg.pt_max, variant_label=variant_label,
                         cnm_label=_cnm_legend_label(cfg))
        _style_axis(ax, r"$p_T$ [GeV]")
        ax.set_xlim(0.0, cfg.pt_max)
        try:
            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(5.0))
            ax.xaxis.set_minor_locator(MultipleLocator(1.0))
        except Exception:
            pass
        ax.text(0.05, 0.93, tag, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, fontweight="bold")
        if y_window_tex is not None:
            _annotate_y_window(ax, y_window_tex)
        if i == 0:
            _annotate_energy(ax, cfg)
        data_axes.append(ax)
    if data_axes:
        ylim = _common_ylim(data_axes)
        for ax in data_axes:
            ax.set_ylim(*ylim)
        handles, labels = data_axes[0].get_legend_handles_labels()
        _legend_panel(axes_flat[len(tags)], handles, labels, cfg,
                      old_ref=old_ref is not None)
    for j in range(len(tags) + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_vs_centrality(df: pd.DataFrame,
                       cfg: SystemConfig,
                       out: Path,
                       title: str,
                       *,
                       y_window_tex: str | None = None,
                       variant_label: str = "TAMU-NP") -> None:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True,
                             dpi=160, layout="constrained")
    for panel_index, (ax, name) in enumerate(zip(axes, STATE_MAIN)):
        sub = df[df["centrality"] != "MB"].sort_values("cent_center")
        c = sub[f"{name}_central"].to_numpy(dtype=np.float64)
        lo = sub[f"{name}_lo"].to_numpy(dtype=np.float64)
        hi = sub[f"{name}_hi"].to_numpy(dtype=np.float64)
        edges = np.r_[
            sub["cent_left"].to_numpy(dtype=np.float64)[0],
            sub["cent_right"].to_numpy(dtype=np.float64),
        ]
        color = GREEN
        if {"cnm_central", "cnm_lo", "cnm_hi"}.issubset(sub.columns):
            cnm_c = sub["cnm_central"].to_numpy(dtype=np.float64)
            cnm_lo = sub["cnm_lo"].to_numpy(dtype=np.float64)
            cnm_hi = sub["cnm_hi"].to_numpy(dtype=np.float64)
            if np.any(np.isfinite(cnm_c) & np.isfinite(cnm_lo) & np.isfinite(cnm_hi)):
                ax.step(edges, np.r_[cnm_c, cnm_c[-1]], where="post",
                        color="#555555", lw=1.5, ls="-.", label=_cnm_legend_label(cfg))
                ax.fill_between(edges, np.r_[cnm_lo, cnm_lo[-1]], np.r_[cnm_hi, cnm_hi[-1]],
                                step="post", facecolor="#9a9a9a", edgecolor="none",
                                linewidth=0.0, alpha=0.22)
        m = np.isfinite(c) & np.isfinite(lo) & np.isfinite(hi)
        if np.any(m):
            ax.step(edges, np.r_[c, c[-1]], where="post", color=color,
                    lw=1.9, ls="--", label=f"{variant_label} Prim")
            ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]],
                            step="post", facecolor=color, edgecolor="none",
                            linewidth=0.0, alpha=0.08)
        ccol = f"{name}_cnm_central"
        if ccol in sub.columns:
            cc = sub[ccol].to_numpy(dtype=np.float64)
            clo = sub[f"{name}_cnm_lo"].to_numpy(dtype=np.float64)
            chi = sub[f"{name}_cnm_hi"].to_numpy(dtype=np.float64)
            m_cnm = np.isfinite(cc) & np.isfinite(clo) & np.isfinite(chi)
            if np.any(m_cnm):
                ax.step(edges, np.r_[cc, cc[-1]], where="post", color=color,
                        lw=2.0, ls="-", label=f"CNM x {variant_label} Prim")
                ax.fill_between(edges, np.r_[clo, clo[-1]], np.r_[chi, chi[-1]],
                                step="post", facecolor=color, edgecolor="none",
                                linewidth=0.0, alpha=0.20)
        _style_axis(ax, "centrality [%]")
        ax.set_xlim(0.0, 100.0)
        ax.text(0.95, 0.93, STATE_TEX[name], transform=ax.transAxes,
                ha="right", va="top", fontsize=12, fontweight="bold")
        if y_window_tex is not None:
            _annotate_y_window(ax, y_window_tex, x=0.05, y=0.93, ha="left")
        if panel_index == 0:
            _annotate_energy(ax, cfg)
    ylim = _common_ylim(list(axes))
    for ax in axes:
        ax.set_ylim(*ylim)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="lower right", frameon=False,
                       fontsize=9.0, handlelength=2.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_vs_centrality_windows(by_window: Mapping[str, pd.DataFrame],
                               cfg: SystemConfig,
                               out: Path,
                               title: str,
                               *,
                               variant_label: str = "TAMU-NP") -> None:
    plt = _configure_matplotlib()
    n = len(cfg.y_windows)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), sharey=True,
                             dpi=160, layout="constrained")
    axes = np.atleast_1d(axes).flatten()
    data_axes = []
    for panel_index, (ax, (win_key, _, win_tex)) in enumerate(zip(axes, cfg.y_windows)):
        df = by_window[win_key]
        sub = df[df["centrality"] != "MB"].sort_values("cent_center")
        if sub.empty:
            ax.set_visible(False)
            continue
        edges = np.r_[
            sub["cent_left"].to_numpy(dtype=np.float64)[0],
            sub["cent_right"].to_numpy(dtype=np.float64),
        ]
        if {"cnm_central", "cnm_lo", "cnm_hi"}.issubset(sub.columns):
            cnm_c = sub["cnm_central"].to_numpy(dtype=np.float64)
            cnm_lo = sub["cnm_lo"].to_numpy(dtype=np.float64)
            cnm_hi = sub["cnm_hi"].to_numpy(dtype=np.float64)
            if np.any(np.isfinite(cnm_c) & np.isfinite(cnm_lo) & np.isfinite(cnm_hi)):
                ax.step(edges, np.r_[cnm_c, cnm_c[-1]], where="post",
                        color="#555555", lw=1.5, ls="-.", label=_cnm_legend_label(cfg))
                ax.fill_between(edges, np.r_[cnm_lo, cnm_lo[-1]], np.r_[cnm_hi, cnm_hi[-1]],
                                step="post", facecolor="#9a9a9a", edgecolor="none",
                                linewidth=0.0, alpha=0.22)
        for name in STATE_MAIN:
            c = sub[f"{name}_central"].to_numpy(dtype=np.float64)
            lo = sub[f"{name}_lo"].to_numpy(dtype=np.float64)
            hi = sub[f"{name}_hi"].to_numpy(dtype=np.float64)
            color = STATE_COLORS[name]
            ax.step(edges, np.r_[c, c[-1]], where="post", color=color,
                    lw=1.9, ls=STATE_LS.get(name, "--"),
                    label=_state_label(name, f"{variant_label} Prim"))
            ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]],
                            step="post", facecolor=color, edgecolor="none",
                            linewidth=0.0, alpha=0.08)
            ccol = f"{name}_cnm_central"
            if ccol in sub.columns:
                cc = sub[ccol].to_numpy(dtype=np.float64)
                clo = sub[f"{name}_cnm_lo"].to_numpy(dtype=np.float64)
                chi = sub[f"{name}_cnm_hi"].to_numpy(dtype=np.float64)
                ax.step(edges, np.r_[cc, cc[-1]], where="post", color=color,
                        lw=2.0, ls="-",
                        label=_state_label(name, f"CNM x {variant_label} Prim"))
                ax.fill_between(edges, np.r_[clo, clo[-1]], np.r_[chi, chi[-1]],
                                step="post", facecolor=color, edgecolor="none",
                                linewidth=0.0, alpha=0.20)
            mb = df[df["centrality"] == "MB"]
            if not mb.empty:
                row = mb.iloc[-1]
                ax.hlines(float(row[f"{name}_central"]),
                          float(row["cent_left"]), float(row["cent_right"]),
                          color=color, ls=STATE_LS.get(name, "--"),
                          lw=1.05, alpha=0.85)
        _style_axis(ax, "centrality [%]")
        ax.set_xlim(0.0, 100.0)
        _annotate_y_window(ax, win_tex, x=0.04, y=0.94, ha="left")
        if panel_index == 0:
            _annotate_energy(ax, cfg, x=0.04, y=0.06)
        data_axes.append(ax)
    if data_axes:
        ylim = _common_ylim(data_axes)
        for ax in data_axes:
            ax.set_ylim(*ylim)
        handles, labels = data_axes[0].get_legend_handles_labels()
        data_axes[-1].legend(handles, labels, loc="lower right", frameon=False,
                             fontsize=9.5, handlelength=2.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _new_dataset_spec(cfg: SystemConfig, variant: str) -> DatasetSpec:
    fmt, label = VARIANTS[variant]
    run_name = fmt.format(tag=cfg.input_tag)
    base = REPO_ROOT / "inputs" / "primordial_tamu_new" / f"OxOx{cfg.sqrts_gev:g}_optical"
    if cfg.key == "lhc_oo5360":
        base = REPO_ROOT / "inputs" / "primordial_tamu_new" / "OxOx5360_optical"
    elif cfg.key == "rhic_oo200":
        base = REPO_ROOT / "inputs" / "primordial_tamu_new" / "OxOx200_optical"
    return DatasetSpec(
        label=label,
        paths={
            "lower": base / run_name / "runs" / "lower" / "datafile.gz",
            "upper": base / run_name / "runs" / "upper" / "datafile.gz",
        },
        central_policy="midpoint",
    )


def _old_lhc_nonp_spec() -> DatasetSpec:
    base = REPO_ROOT / "inputs" / "primordial" / "output_tamu_npwlc_oo5.36"
    return DatasetSpec(
        label="Old fixed-b TAMU-NP",
        paths={
            "lower": base / "output-lower" / "datafile.gz",
            "center": base / "output-center" / "datafile.gz",
            "upper": base / "output-upper" / "datafile.gz",
        },
        central_policy="center",
        fixed_b_reference=True,
    )


def _fixed_b_vs_y(df: pd.DataFrame, y_edges: np.ndarray, system) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        direct = _weighted_direct_mean(_select(df, y_window=(float(y0), float(y1)), pt_window=(0.0, 20.0)))
        values = _main_from_direct(direct, system)
        row = {"y_center": 0.5 * (float(y0) + float(y1))}
        for k, name in enumerate(STATE_MAIN):
            row[name] = float(values[k])
        rows.append(row)
    return pd.DataFrame(rows)


def _fixed_b_vs_pt(df: pd.DataFrame, pt_edges: np.ndarray, window: Tuple[float, float], system) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for p0, p1 in zip(pt_edges[:-1], pt_edges[1:]):
        direct = _weighted_direct_mean(_select(df, y_window=window, pt_window=(float(p0), float(p1))))
        values = _main_from_direct(direct, system)
        row = {"pT_center": 0.5 * (float(p0) + float(p1))}
        for k, name in enumerate(STATE_MAIN):
            row[name] = float(values[k])
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_weights(df: pd.DataFrame) -> Dict[str, object]:
    w = df["weight"].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(df)),
        "weight_min": float(np.nanmin(w)) if w.size else float("nan"),
        "weight_max": float(np.nanmax(w)) if w.size else float("nan"),
        "weight_mean": float(np.nanmean(w)) if w.size else float("nan"),
        "all_weights_one": bool(np.allclose(w, 1.0)) if w.size else False,
        "b_values": [float(v) for v in np.unique(np.round(df["b"].to_numpy(dtype=float), 4))],
    }


def analyze_dataset(cfg: SystemConfig,
                    spec: DatasetSpec,
                    *,
                    outdir: Path,
                    pt_edges: np.ndarray,
                    y_edges: np.ndarray,
                    max_events_per_b: int | None,
                    include_cnm: bool,
                    make_plots: bool,
                    logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    system = make_bottomonia_system(sqrts_pp_GeV=cfg.sqrts_gev)
    run_tables_y: Dict[str, pd.DataFrame] = {}
    run_tables_pt: Dict[str, Dict[str, pd.DataFrame]] = {key: {} for key, _, _ in cfg.y_windows}
    run_tables_cent: Dict[str, Dict[str, pd.DataFrame]] = {key: {} for key, _, _ in cfg.y_windows}
    summaries: List[Dict[str, object]] = []

    for run_label, path in spec.paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        logger.info("[%s/%s/%s] reading %s", cfg.key, spec.label, run_label, path)
        df = read_tamu_datafile(path, max_events_per_b=max_events_per_b)
        summary = {"run": run_label, "path": str(path.relative_to(REPO_ROOT))}
        summary.update(_summarize_weights(df))
        summaries.append(summary)
        run_tables_y[run_label] = _run_vs_y(df, cfg, y_edges, system)
        for win_key, y_window, _ in cfg.y_windows:
            run_tables_pt[win_key][run_label] = _run_vs_pt(df, cfg, pt_edges, y_window, system)
            run_tables_cent[win_key][run_label] = _run_vs_centrality(df, cfg, y_window, system)

    csv_dir = outdir / "csv"
    fig_dir = outdir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    with (outdir / "input_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["run", "path", "rows", "weight_min", "weight_max", "weight_mean", "all_weights_one", "b_values"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            row = dict(row)
            row["b_values"] = " ".join(f"{v:g}" for v in row["b_values"])
            writer.writerow(row)

    out: Dict[str, pd.DataFrame] = {}
    cnm_tables: CNMTables | None = None
    if include_cnm:
        cnm_tables = _compute_cnm_tables(cfg, y_edges, pt_edges, logger)

    band_y = _combine_band(
        run_tables_y,
        key_cols=("y_center", "centrality", "mb", "cent_left", "cent_right"),
        central_policy=spec.central_policy,
    )
    if cnm_tables is not None:
        band_y = _merge_cnm_columns(
            band_y,
            cnm_tables.y,
            key_cols=("y_center", "centrality"),
        )
    _write_csv(csv_dir / "raa_vs_y_all_centralities.csv", band_y)
    out["y"] = band_y

    for win_key, _, _ in cfg.y_windows:
        band_pt = _combine_band(
            run_tables_pt[win_key],
            key_cols=("pT_center", "centrality", "mb", "cent_left", "cent_right", "y_min", "y_max"),
            central_policy=spec.central_policy,
        )
        if cnm_tables is not None:
            band_pt = _merge_cnm_columns(
                band_pt,
                cnm_tables.pt[win_key],
                key_cols=("pT_center", "centrality"),
            )
        _write_csv(csv_dir / f"raa_vs_pt_{win_key}_all_centralities.csv", band_pt)
        out[f"pt_{win_key}"] = band_pt

        band_cent = _combine_band(
            run_tables_cent[win_key],
            key_cols=("cent_left", "cent_right", "cent_center", "centrality", "mb", "y_min", "y_max"),
            central_policy=spec.central_policy,
        )
        if cnm_tables is not None:
            band_cent = _merge_cnm_columns(
                band_cent,
                cnm_tables.centrality[win_key],
                key_cols=("centrality",),
            )
        _write_csv(csv_dir / f"raa_vs_centrality_{win_key}.csv", band_cent)
        out[f"centrality_{win_key}"] = band_cent

    if make_plots:
        plot_vs_y_grid(
            band_y, cfg, fig_dir / "raa_vs_y_all_centralities",
            f"{cfg.label}  {spec.label}: primordial only",
            variant_label=spec.label,
        )
        plot_vs_centrality_windows(
            {win_key: out[f"centrality_{win_key}"] for win_key, _, _ in cfg.y_windows},
            cfg,
            fig_dir / "raa_vs_centrality_windows",
            f"{cfg.label}  {spec.label}: primordial only",
            variant_label=spec.label,
        )
        for win_key, _, win_tex in cfg.y_windows:
            plot_vs_pt_grid(
                out[f"pt_{win_key}"], cfg, fig_dir / f"raa_vs_pt_{win_key}_all_centralities",
                f"{cfg.label}  {spec.label}: primordial only, {win_tex}",
                variant_label=spec.label,
            )
            plot_vs_centrality(
                out[f"centrality_{win_key}"], cfg, fig_dir / f"raa_vs_centrality_{win_key}",
                f"{cfg.label}  {spec.label}: primordial only, {win_tex}",
                y_window_tex=win_tex,
                variant_label=spec.label,
            )
    return out


def analyze_old_lhc_reference(cfg: SystemConfig,
                              *,
                              outdir: Path,
                              pt_edges: np.ndarray,
                              y_edges: np.ndarray,
                              max_events_per_b: int | None,
                              logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    spec = _old_lhc_nonp_spec()
    system = make_bottomonia_system(sqrts_pp_GeV=cfg.sqrts_gev)
    y_runs: Dict[str, pd.DataFrame] = {}
    pt_runs: Dict[str, Dict[str, pd.DataFrame]] = {key: {} for key, _, _ in cfg.y_windows}
    summaries: List[Dict[str, object]] = []
    for run_label, path in spec.paths.items():
        logger.info("[old reference/%s] reading %s", run_label, path)
        df = read_tamu_datafile(path, max_events_per_b=max_events_per_b)
        summary = {"run": run_label, "path": str(path.relative_to(REPO_ROOT))}
        summary.update(_summarize_weights(df))
        summaries.append(summary)
        y_runs[run_label] = _fixed_b_vs_y(df, y_edges, system)
        for win_key, y_window, _ in cfg.y_windows:
            pt_runs[win_key][run_label] = _fixed_b_vs_pt(df, pt_edges, y_window, system)

    outdir.mkdir(parents=True, exist_ok=True)
    csv_dir = outdir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    with (outdir / "input_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["run", "path", "rows", "weight_min", "weight_max", "weight_mean", "all_weights_one", "b_values"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            row = dict(row)
            row["b_values"] = " ".join(f"{v:g}" for v in row["b_values"])
            writer.writerow(row)

    band_y = _combine_band(y_runs, key_cols=("y_center",), central_policy="center")
    _write_csv(csv_dir / "old_fixed_b_raa_vs_y.csv", band_y)
    out = {"y": band_y}
    for win_key, _, _ in cfg.y_windows:
        band_pt = _combine_band(pt_runs[win_key], key_cols=("pT_center",), central_policy="center")
        _write_csv(csv_dir / f"old_fixed_b_raa_vs_pt_{win_key}.csv", band_pt)
        out[f"pt_{win_key}"] = band_pt
    return out


def overlay_old_on_new_figures(new_results: Mapping[str, pd.DataFrame],
                               old_results: Mapping[str, pd.DataFrame],
                               cfg: SystemConfig,
                               spec_label: str,
                               *,
                               outdir: Path) -> None:
    fig_dir = outdir / "figures"
    plot_vs_y_grid(
        new_results["y"], cfg, fig_dir / "raa_vs_y_all_centralities_with_old_fixed_b",
        f"{cfg.label}  {spec_label}: new weighted vs old fixed-b",
        old_ref=old_results.get("y"),
        variant_label=spec_label,
    )
    for win_key, _, win_tex in cfg.y_windows:
        plot_vs_pt_grid(
            new_results[f"pt_{win_key}"], cfg,
            fig_dir / f"raa_vs_pt_{win_key}_all_centralities_with_old_fixed_b",
            f"{cfg.label}  {spec_label}: new weighted vs old fixed-b, {win_tex}",
            old_ref=old_results.get(f"pt_{win_key}"),
            variant_label=spec_label,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=sorted(SYSTEMS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=sorted(VARIANTS))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "cnm_primordial_importance")
    parser.add_argument("--smoke", action="store_true", help="Use fewer bins and a capped number of events per b.")
    parser.add_argument("--max-events-per-b", type=int, default=None,
                        help="Cap parsed events per impact parameter for quick checks.")
    parser.add_argument("--no-cnm", action="store_true",
                        help="Do not attach CNM tables; write primordial-only bands.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSVs only.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(levelname)s: %(message)s")
    logger = logging.getLogger("bottomonia_primordial_importance")

    max_events = args.max_events_per_b
    if args.smoke and max_events is None:
        max_events = 3000
    make_plots = not args.no_plots

    old_lhc_nonp: Dict[str, pd.DataFrame] | None = None
    if "lhc_oo5360" in args.systems and "nonp" in args.variants:
        cfg_lhc = SYSTEMS["lhc_oo5360"]
        old_lhc_nonp = analyze_old_lhc_reference(
            cfg_lhc,
            outdir=args.output_dir / "old_lhc_oo5360_nonp_fixed_b",
            pt_edges=_pt_edges_for_cfg(cfg_lhc, smoke=args.smoke),
            y_edges=cfg_lhc.y_edges_smoke if args.smoke else cfg_lhc.y_edges,
            max_events_per_b=max_events,
            logger=logger,
        )

    for system_key in args.systems:
        cfg = SYSTEMS[system_key]
        y_edges = cfg.y_edges_smoke if args.smoke else cfg.y_edges
        pt_edges = _pt_edges_for_cfg(cfg, smoke=args.smoke)
        for variant in args.variants:
            spec = _new_dataset_spec(cfg, variant)
            run_out = args.output_dir / cfg.key / variant
            results = analyze_dataset(
                cfg,
                spec,
                outdir=run_out,
                pt_edges=pt_edges,
                y_edges=y_edges,
                max_events_per_b=max_events,
                include_cnm=not args.no_cnm,
                make_plots=make_plots,
                logger=logger,
            )
            if make_plots and system_key == "lhc_oo5360" and variant == "nonp" and old_lhc_nonp:
                overlay_old_on_new_figures(results, old_lhc_nonp, cfg, spec.label, outdir=run_out)

    logger.info("Done. Outputs written under %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
