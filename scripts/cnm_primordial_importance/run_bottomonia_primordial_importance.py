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
PT_EDGES_FULL = np.arange(0.0, 20.0 + 1.0, 1.0, dtype=np.float64)
PT_EDGES_SMOKE = np.array([0.0, 5.0, 10.0, 20.0], dtype=np.float64)
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
        y_edges=np.arange(-2.5, 2.5 + 0.5, 0.5, dtype=np.float64),
        y_edges_smoke=np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float64),
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
CNM_CENT_DIR = REPO_ROOT / "outputs" / "cnm" / "centralities"
CNM_MINBIAS_DIR = REPO_ROOT / "outputs" / "cnm" / "minbias" / "005.36TeV" / "csv"
CNM_SOURCE_BINS = {
    "0-10%": (0.0, 10.0),
    "10-30%": (10.0, 30.0),
    "30-50%": (30.0, 50.0),
    "50-70%": (50.0, 70.0),
    "70-100%": (70.0, 100.0),
}
CNM_PT_FILES = {
    "midrapidity": CNM_CENT_DIR / "Upsilon_RAA_vs_pT_y_n2p4_p2p4_OO_5p36TeV.csv",
    "forward": CNM_CENT_DIR / "Upsilon_RAA_vs_pT_y_p2p5_p4p0_OO_5p36TeV.csv",
}
CNM_MB_PT_FILES = {
    "midrapidity": CNM_MINBIAS_DIR / "Upsilon_minbias_MB_OO_005p36TeV_vs_pT_mid.csv",
    "forward": CNM_MINBIAS_DIR / "Upsilon_minbias_MB_OO_005p36TeV_vs_pT_forward.csv",
}
CNM_MB_Y_FILE = CNM_MINBIAS_DIR / "Upsilon_minbias_MB_OO_005p36TeV_vs_y.csv"
CNM_CENT_FILES = {
    "midrapidity": CNM_CENT_DIR / "Upsilon_RAA_vs_cent_y_m2p4_to_2p4_OO_5p36TeV.csv",
    "forward": CNM_CENT_DIR / "Upsilon_RAA_vs_cent_y_2p5_to_4_OO_5p36TeV.csv",
}


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
            center = np.nanmean(vals, axis=1)
        merged[f"{name}_central"] = center
        merged[f"{name}_lo"] = np.nanmin(vals, axis=1)
        merged[f"{name}_hi"] = np.nanmax(vals, axis=1)
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
        return np.array([0.0, 5.0, 10.0, float(cfg.pt_max)], dtype=np.float64)
    return np.arange(0.0, float(cfg.pt_max) + 1.0, 1.0, dtype=np.float64)


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


def _read_csv_or_none(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _parse_cent_tag(tag: str) -> Tuple[float, float] | None:
    if tag == "MB":
        return None
    clean = str(tag).replace("%", "").strip()
    try:
        left, right = clean.split("-", 1)
        return float(left), float(right)
    except ValueError:
        return None


def _overlap_weights_for_tag(tag: str) -> Dict[str, float]:
    """Map a requested centrality bin onto the publication CNM bins.

    The bottomonia CNM publication package uses 0-10, 10-30, 30-50, 50-70,
    70-100% bins.  The importance workflow uses charmonia-style 0-10, 10-20,
    20-40, 40-60, 60-80, 80-100% bins.  Reusing publication CNM therefore
    requires overlap-weighted bin averages, not interpolation in bin center.
    """
    target = _parse_cent_tag(tag)
    if target is None:
        return {}
    t0, t1 = target
    overlaps: Dict[str, float] = {}
    for src_tag, (s0, s1) in CNM_SOURCE_BINS.items():
        width = max(0.0, min(t1, s1) - max(t0, s0))
        if width > 0.0:
            overlaps[src_tag] = width
    total = sum(overlaps.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in overlaps.items()}


def _weighted_arrays(parts: Sequence[Tuple[float, np.ndarray]], template: np.ndarray) -> np.ndarray:
    if not parts:
        return np.full_like(np.asarray(template, dtype=np.float64), np.nan, dtype=np.float64)
    vals = np.stack([np.asarray(arr, dtype=np.float64) for _, arr in parts], axis=0)
    weights = np.asarray([w for w, _ in parts], dtype=np.float64)[:, None]
    finite = np.isfinite(vals)
    denom = np.sum(np.where(finite, weights, 0.0), axis=0)
    numer = np.sum(np.where(finite, vals * weights, 0.0), axis=0)
    out = np.full(vals.shape[1], np.nan, dtype=np.float64)
    ok = denom > 0.0
    out[ok] = numer[ok] / denom[ok]
    return out


def _cnm_by_y(tag: str, y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def load_one(path_tag: str) -> pd.DataFrame:
        safe = path_tag.replace("%", "pct")
        path = CNM_CENT_DIR / f"Upsilon_RAA_vs_y_{safe}_OO_5p36TeV.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    y_values = np.asarray(y_values, dtype=np.float64)
    if tag == "MB":
        src = _read_csv_or_none(CNM_MB_Y_FILE)
        if src is None:
            src = _read_csv_or_none(CNM_CENT_DIR / "Upsilon_RAA_vs_y_MB_OO_5p36TeV.csv")
        if src is None:
            return _nan_triplet_like(y_values)
        return (
            _interp_series(y_values, src["y_center"].to_numpy(float), src["cnm_central"].to_numpy(float)),
            _interp_series(y_values, src["y_center"].to_numpy(float), src["cnm_lo"].to_numpy(float)),
            _interp_series(y_values, src["y_center"].to_numpy(float), src["cnm_hi"].to_numpy(float)),
        )

    if tag in CNM_SOURCE_BINS:
        src = load_one(tag)
        return (
            _interp_series(y_values, src["y_center"].to_numpy(float), src["cnm_central"].to_numpy(float)),
            _interp_series(y_values, src["y_center"].to_numpy(float), src["cnm_lo"].to_numpy(float)),
            _interp_series(y_values, src["y_center"].to_numpy(float), src["cnm_hi"].to_numpy(float)),
        )

    weights = _overlap_weights_for_tag(tag)
    if not weights:
        return _nan_triplet_like(y_values)
    vals_c, vals_lo, vals_hi = [], [], []
    for src_tag, w in weights.items():
        src = load_one(src_tag)
        vals_c.append((w, _interp_series(y_values, src["y_center"].to_numpy(float), src["cnm_central"].to_numpy(float))))
        vals_lo.append((w, _interp_series(y_values, src["y_center"].to_numpy(float), src["cnm_lo"].to_numpy(float))))
        vals_hi.append((w, _interp_series(y_values, src["y_center"].to_numpy(float), src["cnm_hi"].to_numpy(float))))
    return (
        _weighted_arrays(vals_c, y_values),
        _weighted_arrays(vals_lo, y_values),
        _weighted_arrays(vals_hi, y_values),
    )


def _cnm_by_pt(win_key: str, tag: str, pt_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt_values = np.asarray(pt_values, dtype=np.float64)
    if tag == "MB":
        mb_path = CNM_MB_PT_FILES.get(win_key)
        src = _read_csv_or_none(mb_path) if mb_path is not None else None
        if src is None:
            path = CNM_PT_FILES.get(win_key)
            src = _read_csv_or_none(path) if path is not None else None
            if src is not None:
                src = src[src["centrality"] == "MB"]
        if src is None or src.empty:
            return _nan_triplet_like(pt_values)
        return (
            _interp_series(pt_values, src["pT_center"].to_numpy(float), src["cnm_central"].to_numpy(float)),
            _interp_series(pt_values, src["pT_center"].to_numpy(float), src["cnm_lo"].to_numpy(float)),
            _interp_series(pt_values, src["pT_center"].to_numpy(float), src["cnm_hi"].to_numpy(float)),
        )

    path = CNM_PT_FILES.get(win_key)
    if path is None or not path.exists():
        return _nan_triplet_like(pt_values)
    src = pd.read_csv(path)
    if tag in set(src["centrality"]):
        sub = src[src["centrality"] == tag]
        return (
            _interp_series(pt_values, sub["pT_center"].to_numpy(float), sub["cnm_central"].to_numpy(float)),
            _interp_series(pt_values, sub["pT_center"].to_numpy(float), sub["cnm_lo"].to_numpy(float)),
            _interp_series(pt_values, sub["pT_center"].to_numpy(float), sub["cnm_hi"].to_numpy(float)),
        )

    weights = _overlap_weights_for_tag(tag)
    if not weights:
        return _nan_triplet_like(pt_values)
    vals_c, vals_lo, vals_hi = [], [], []
    for src_tag, w in weights.items():
        sub = src[src["centrality"] == src_tag]
        if sub.empty:
            continue
        vals_c.append((w, _interp_series(pt_values, sub["pT_center"].to_numpy(float), sub["cnm_central"].to_numpy(float))))
        vals_lo.append((w, _interp_series(pt_values, sub["pT_center"].to_numpy(float), sub["cnm_lo"].to_numpy(float))))
        vals_hi.append((w, _interp_series(pt_values, sub["pT_center"].to_numpy(float), sub["cnm_hi"].to_numpy(float))))
    return (
        _weighted_arrays(vals_c, pt_values),
        _weighted_arrays(vals_lo, pt_values),
        _weighted_arrays(vals_hi, pt_values),
    )


def _cnm_by_centrality(win_key: str, cent_values: np.ndarray, tag_values: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = CNM_CENT_FILES.get(win_key)
    if path is None or not path.exists():
        return _nan_triplet_like(cent_values)
    src = pd.read_csv(path)
    non_mb = src[src["label"] != "MB"].copy() if "label" in src.columns else src.copy()
    out_c = np.full(len(cent_values), np.nan, dtype=np.float64)
    out_lo = np.full(len(cent_values), np.nan, dtype=np.float64)
    out_hi = np.full(len(cent_values), np.nan, dtype=np.float64)
    mb = src[src["label"] == "MB"] if "label" in src.columns else src.iloc[0:0]
    for i, tag in enumerate(tag_values):
        if tag == "MB" and not mb.empty:
            row = mb.iloc[-1]
            out_c[i] = float(row["cnm_central"])
            out_lo[i] = float(row["cnm_lo"])
            out_hi[i] = float(row["cnm_hi"])
            continue
        weights = _overlap_weights_for_tag(tag)
        if not weights:
            continue
        c_parts, lo_parts, hi_parts = [], [], []
        for src_tag, w in weights.items():
            row = non_mb[non_mb["label"] == src_tag]
            if row.empty:
                continue
            r = row.iloc[-1]
            c_parts.append((w, float(r["cnm_central"])))
            lo_parts.append((w, float(r["cnm_lo"])))
            hi_parts.append((w, float(r["cnm_hi"])))
        if c_parts:
            out_c[i] = sum(w * v for w, v in c_parts) / sum(w for w, _ in c_parts)
            out_lo[i] = sum(w * v for w, v in lo_parts) / sum(w for w, _ in lo_parts)
            out_hi[i] = sum(w * v for w, v in hi_parts) / sum(w for w, _ in hi_parts)
    return out_c, out_lo, out_hi


def attach_lhc_cnm(df: pd.DataFrame, *, cfg: SystemConfig, kind: str,
                   win_key: str | None = None) -> pd.DataFrame:
    """Add publication bottomonia CNM and CNM x primordial columns.

    Source of CNM truth:
      * outputs/cnm/centralities/ from cnm/cnm_scripts/run_bottomonia_cnm_OO_publication.py
        for finite centrality bins and centrality dependence.
      * outputs/cnm/minbias/005.36TeV/csv/ from scripts/cnm/run_upsilon_oo5360_minbias_cnm_summary.py
        for MB y and pT projections.

    Requested charmonia-style bins are not identical to the publication CNM
    bins, so finite centrality CNM is overlap-weighted in centrality percentile
    space. MB is never treated as a finite bin.
    """
    out = df.copy()
    if cfg.key != "lhc_oo5360":
        return out
    cnm_c = np.full(len(out), np.nan, dtype=np.float64)
    cnm_lo = np.full(len(out), np.nan, dtype=np.float64)
    cnm_hi = np.full(len(out), np.nan, dtype=np.float64)
    if kind == "y":
        for tag, idx in out.groupby("centrality").groups.items():
            yc = out.loc[idx, "y_center"].to_numpy(float)
            c, lo, hi = _cnm_by_y(str(tag), yc)
            cnm_c[list(idx)] = c
            cnm_lo[list(idx)] = lo
            cnm_hi[list(idx)] = hi
    elif kind == "pt" and win_key is not None:
        for tag, idx in out.groupby("centrality").groups.items():
            pt = out.loc[idx, "pT_center"].to_numpy(float)
            c, lo, hi = _cnm_by_pt(win_key, str(tag), pt)
            cnm_c[list(idx)] = c
            cnm_lo[list(idx)] = lo
            cnm_hi[list(idx)] = hi
    elif kind == "centrality" and win_key is not None:
        c, lo, hi = _cnm_by_centrality(
            win_key,
            out["cent_center"].to_numpy(float),
            out["centrality"].astype(str).tolist(),
        )
        cnm_c, cnm_lo, cnm_hi = c, lo, hi
    else:
        return out

    out["cnm_central"] = cnm_c
    out["cnm_lo"] = cnm_lo
    out["cnm_hi"] = cnm_hi
    out["cnm_source"] = "bottomonia_publication_cnm_overlap_weighted"
    for name in STATE_MAIN:
        out[f"{name}_cnm_central"] = out[f"{name}_central"].to_numpy(float) * cnm_c
        out[f"{name}_cnm_lo"] = out[f"{name}_lo"].to_numpy(float) * cnm_lo
        out[f"{name}_cnm_hi"] = out[f"{name}_hi"].to_numpy(float) * cnm_hi
    return out


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


def _step_dashed_hatched(ax, centers: np.ndarray, central: np.ndarray,
                         lo: np.ndarray, hi: np.ndarray, *,
                         color: str, label: str | None,
                         linestyle: object = "--",
                         x_min: float | None = None,
                         x_max: float | None = None,
                         zorder: int = 3) -> None:
    centers = np.asarray(centers, dtype=np.float64)
    central = np.asarray(central, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    mask = np.isfinite(centers) & np.isfinite(central) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(mask):
        return
    idx = np.flatnonzero(mask)
    start, stop = int(idx[0]), int(idx[-1])
    c = centers[start:stop + 1]
    edges = _edges_from_centers(c, x_min=x_min if start == 0 else None,
                                x_max=x_max if stop == centers.size - 1 else None)
    y = central[start:stop + 1]
    ylo = lo[start:stop + 1]
    yhi = hi[start:stop + 1]
    ax.step(edges, np.r_[y, y[-1]], where="post", color=color, lw=1.9,
            ls=linestyle, label=label, zorder=zorder)
    ax.fill_between(edges, np.r_[ylo, ylo[-1]], np.r_[yhi, yhi[-1]],
                    step="post", facecolor="none", edgecolor=color,
                    hatch=PRIM_HATCH, linewidth=0.0, alpha=0.90,
                    zorder=zorder - 1)


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
    )


def _state_label(state: str, layer_label: str) -> str:
    return rf"{STATE_TEX[state]} {layer_label}"


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
                     show_cnm_reference: bool = True) -> None:
    x = df[xcol].to_numpy(dtype=np.float64)
    if show_cnm_reference:
        _step_cnm_reference(ax, df, xcol, x_min=x_min, x_max=x_max)
    for name in STATE_MAIN:
        color = STATE_COLORS[name]
        c = df[f"{name}_central"].to_numpy(dtype=np.float64)
        lo = df[f"{name}_lo"].to_numpy(dtype=np.float64)
        hi = df[f"{name}_hi"].to_numpy(dtype=np.float64)
        _step_dashed_hatched(ax, x, c, lo, hi, color=color,
                             label=_state_label(name, f"{variant_label} Prim"),
                             linestyle=STATE_LS.get(name, "--"),
                             x_min=x_min, x_max=x_max)
        ccol = f"{name}_cnm_central"
        if ccol in df.columns:
            cc = df[ccol].to_numpy(dtype=np.float64)
            clo = df[f"{name}_cnm_lo"].to_numpy(dtype=np.float64)
            chi = df[f"{name}_cnm_hi"].to_numpy(dtype=np.float64)
            _step_dashed_hatched(ax, x, cc, clo, chi, color=color,
                                 label=_state_label(name, f"CNM x {variant_label} Prim"),
                                 linestyle="-", x_min=x_min, x_max=x_max,
                                 zorder=5)
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
        sub = _finite_band(df[df["centrality"] == tag], "y_center")
        if sub.empty:
            ax.set_visible(False)
            continue
        old = old_ref if tag == "MB" else None
        _plot_step_panel(ax, sub, "y_center", old_ref=old, old_x_col="y_center",
                         x_min=float(cfg.y_edges[0]), x_max=float(cfg.y_edges[-1]),
                         variant_label=variant_label)
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
        sub = _finite_band(df[df["centrality"] == tag], "pT_center")
        if sub.empty:
            ax.set_visible(False)
            continue
        old = old_ref if tag == "MB" else None
        _plot_step_panel(ax, sub, "pT_center", old_ref=old, old_x_col="pT_center",
                         x_min=0.0, x_max=cfg.pt_max, variant_label=variant_label)
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
                        color="#555555", lw=1.5, ls="-.", label="CNM")
                ax.fill_between(edges, np.r_[cnm_lo, cnm_lo[-1]], np.r_[cnm_hi, cnm_hi[-1]],
                                step="post", facecolor="none", edgecolor="#555555",
                                hatch=PRIM_HATCH, linewidth=0.0, alpha=0.55)
        m = np.isfinite(c) & np.isfinite(lo) & np.isfinite(hi)
        if np.any(m):
            ax.step(edges, np.r_[c, c[-1]], where="post", color=color,
                    lw=1.9, ls="--", label=f"{variant_label} Prim")
            ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]],
                            step="post", facecolor="none", edgecolor=color,
                            hatch=PRIM_HATCH, linewidth=0.0, alpha=0.90)
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
                                step="post", facecolor="none", edgecolor=color,
                                hatch=PRIM_HATCH, linewidth=0.0, alpha=0.70)
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
                        color="#555555", lw=1.5, ls="-.", label="CNM")
                ax.fill_between(edges, np.r_[cnm_lo, cnm_lo[-1]], np.r_[cnm_hi, cnm_hi[-1]],
                                step="post", facecolor="none", edgecolor="#555555",
                                hatch=PRIM_HATCH, linewidth=0.0, alpha=0.55)
        for name in STATE_MAIN:
            c = sub[f"{name}_central"].to_numpy(dtype=np.float64)
            lo = sub[f"{name}_lo"].to_numpy(dtype=np.float64)
            hi = sub[f"{name}_hi"].to_numpy(dtype=np.float64)
            color = STATE_COLORS[name]
            ax.step(edges, np.r_[c, c[-1]], where="post", color=color,
                    lw=1.9, ls=STATE_LS.get(name, "--"),
                    label=_state_label(name, f"{variant_label} Prim"))
            ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]],
                            step="post", facecolor="none", edgecolor=color,
                            hatch=PRIM_HATCH, linewidth=0.0, alpha=0.90)
            ccol = f"{name}_cnm_central"
            if ccol in sub.columns:
                cc = sub[ccol].to_numpy(dtype=np.float64)
                clo = sub[f"{name}_cnm_lo"].to_numpy(dtype=np.float64)
                chi = sub[f"{name}_cnm_hi"].to_numpy(dtype=np.float64)
                ax.step(edges, np.r_[cc, cc[-1]], where="post", color=color,
                        lw=2.0, ls="-",
                        label=_state_label(name, f"CNM x {variant_label} Prim"))
                ax.fill_between(edges, np.r_[clo, clo[-1]], np.r_[chi, chi[-1]],
                                step="post", facecolor="none", edgecolor=color,
                                hatch=PRIM_HATCH, linewidth=0.0, alpha=0.70)
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
    band_y = _combine_band(
        run_tables_y,
        key_cols=("y_center", "centrality", "mb", "cent_left", "cent_right"),
        central_policy=spec.central_policy,
    )
    if include_cnm:
        band_y = attach_lhc_cnm(band_y, cfg=cfg, kind="y")
    _write_csv(csv_dir / "raa_vs_y_all_centralities.csv", band_y)
    out["y"] = band_y

    for win_key, _, _ in cfg.y_windows:
        band_pt = _combine_band(
            run_tables_pt[win_key],
            key_cols=("pT_center", "centrality", "mb", "cent_left", "cent_right", "y_min", "y_max"),
            central_policy=spec.central_policy,
        )
        if include_cnm:
            band_pt = attach_lhc_cnm(band_pt, cfg=cfg, kind="pt", win_key=win_key)
        _write_csv(csv_dir / f"raa_vs_pt_{win_key}_all_centralities.csv", band_pt)
        out[f"pt_{win_key}"] = band_pt

        band_cent = _combine_band(
            run_tables_cent[win_key],
            key_cols=("cent_left", "cent_right", "cent_center", "centrality", "mb", "y_min", "y_max"),
            central_policy=spec.central_policy,
        )
        if include_cnm:
            band_cent = attach_lhc_cnm(band_cent, cfg=cfg, kind="centrality", win_key=win_key)
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
