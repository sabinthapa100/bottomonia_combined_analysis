#!/usr/bin/env python3
"""
Central-only OO 5.36 TeV QTraj comparison and envelope builder.

This script follows the Mathematica/qtraj_analysis pipeline:

1. Load a qtraj `datafile.gz` or `datafile-avg.gz`.
2. If the file is raw, average quantum trajectories in memory exactly like
   `processEvents.py`.
3. Match L=0 and L=1 rows into the six-state basis
   {1S, 2S, 1P, 3S, 2P, 1D}.
4. Compute fixed-b central values in the OO acceptance:
   - R_AA vs pT for |y| <= 2.4
   - R_AA vs y  for pT <= 30 GeV over -5 <= y <= 5
   - integrated R_AA for |y| <= 2.4 and pT <= 30 GeV
5. Apply hyperfine splitting + feeddown to obtain inclusive 9-state R_AA.

The raw 8-column OO `wReg` layout is ``[v1..v6, rand_tag, L]`` where
``rand_tag`` is the qtraj trajectory random-number tag used only for
deduplication, NOT a physics observable. ``processEvents.average_raw_datafile``
explicitly drops the last two columns (`[:-2]`) during averaging. Earlier
versions of this script misinterpreted the 7th column as a primordial
``d6`` amplitude and folded it into the 1S / 1P survival, which inflated
R_AA(1S) above unity. The primordial-fix path has been removed; loading
now always goes through the documented processEvents-equivalent pipeline.

Primary use cases:
  - compare kappa=5 vs kappa=6 central values only
  - later build an envelope from any set of OO runs, e.g. 5 and 7
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "hnm" / "qtraj_out_analysis"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qtraj_analysis.feeddown import (  # noqa: E402
    apply_feeddown_to_raa6,
    build_feeddown_matrix,
    solve_primordial_sigmas,
)
from qtraj_analysis.binning import binned_step_series  # noqa: E402
from qtraj_analysis.io import load_qtraj_table, parse_records  # noqa: E402
from qtraj_analysis.kinematics_presets import (  # noqa: E402
    OO_INTEGRATED_Y_WINDOW,
    OO_PT_EDGES,
    OO_PT_MAX_FOR_Y,
    SIGMAS_EXP_OO_5360,
    Y_EDGES,
)
from qtraj_analysis.matching import build_observables  # noqa: E402


STATE_NAMES_9: Tuple[str, ...] = (
    "1S",
    "2S",
    "1P0",
    "1P1",
    "1P2",
    "3S",
    "2P0",
    "2P1",
    "2P2",
)
MAIN_STATE_INDICES: Tuple[int, ...] = (0, 1, 5)
MAIN_STATE_NAMES: Tuple[str, ...] = ("1S", "2S", "3S")
MAIN_STATE_TEX: Tuple[str, ...] = (
    r"$\Upsilon(1S)$",
    r"$\Upsilon(2S)$",
    r"$\Upsilon(3S)$",
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "CMS_Collab_OxygenOxygen" / "OxygenOxygen5p36TeV"
)
RUN_COLORS: Tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
)
FINAL_PT_PLOT_MAX = 22.0
FINAL_Y_PLOT_MAX = 4.5
FINAL_Y_EDGES = Y_EDGES.copy()


@dataclass(frozen=True)
class RunSpec:
    label: str
    datafile: Path


@dataclass(frozen=True)
class RunResult:
    label: str
    datafile: Path
    load_mode: str
    n_observables: int
    pt_centers: np.ndarray
    raa9_pt: np.ndarray
    sem9_pt: np.ndarray
    y_centers: np.ndarray
    raa9_y: np.ndarray
    sem9_y: np.ndarray
    mean6_int: np.ndarray
    sem6_int: np.ndarray
    raa9_int: np.ndarray
    sem9_int: np.ndarray


def _default_oo_wreg_path(kappa: int) -> Path:
    base = REPO_ROOT / "inputs" / "qtraj_inputs" / "OxygenOxygen5360" / f"qtraj-nlo-run2-00-5.36-kap{kappa}-wReg"
    raw = base / "datafile.gz"
    avg = base / "datafile-avg.gz"
    if raw.exists():
        return raw
    return avg


def _parse_run_arg(spec: str) -> RunSpec:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid --run {spec!r}; expected LABEL=PATH."
        )
    label, path_str = spec.split("=", 1)
    label = label.strip()
    path = Path(path_str.strip())
    if not label:
        raise argparse.ArgumentTypeError("Run label must not be empty.")
    return RunSpec(label=label, datafile=path)


def _weighted_mean_sem_surv6(obs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    if not obs:
        nan = np.full(6, np.nan, dtype=np.float64)
        return nan, nan

    X = np.vstack([o.surv6 for o in obs]).astype(np.float64)
    q = np.asarray([o.qweight for o in obs], dtype=np.float64)
    if (not np.isfinite(q).all()) or float(np.sum(q)) <= 0.0:
        q = np.ones(X.shape[0], dtype=np.float64)
    qsum = float(np.sum(q))
    mean = (X.T @ q) / qsum
    if X.shape[0] <= 1:
        sem = np.zeros(6, dtype=np.float64)
    else:
        var = (q[:, None] * (X - mean) ** 2).sum(axis=0) / qsum
        neff = (qsum * qsum) / float(np.sum(q * q))
        sem = np.sqrt(np.maximum(var, 0.0)) / np.sqrt(max(neff, 1.0))
    return mean, sem


def _raa9_with_sem(
    mean6: np.ndarray,
    sem6: np.ndarray,
    feeddown: np.ndarray,
    sigmas_prim: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(mean6[0]):
        nan = np.full(9, np.nan, dtype=np.float64)
        return nan, nan
    return apply_feeddown_to_raa6(mean6, sem6, feeddown, sigmas_prim)


def _load_observables(datafile: Path, logger: logging.Logger) -> Tuple[Sequence, str]:
    # Always use the documented processEvents-equivalent pipeline.
    # The 7th column in raw 8-col wReg rows is a random-number tag, not a
    # physics observable; it is dropped by average_raw_datafile. Do NOT
    # reintroduce a primordial `d6` fix here — that was a misidentification
    # of the rand_tag column and produced unphysical R_AA(1S) > 1.
    rows = load_qtraj_table(str(datafile), logger)
    records = parse_records(rows, logger)
    return build_observables(records, logger), "generic_loader"


def _compute_vs_pt(
    obs: Sequence,
    feeddown: np.ndarray,
    sigmas_prim: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (OO_PT_EDGES[:-1] + OO_PT_EDGES[1:])
    out = np.full((centers.shape[0], 9), np.nan, dtype=np.float64)
    sem = np.full((centers.shape[0], 9), np.nan, dtype=np.float64)
    y0, y1 = OO_INTEGRATED_Y_WINDOW
    selected = [o for o in obs if y0 <= o.y <= y1]
    for i in range(len(OO_PT_EDGES) - 1):
        p0, p1 = OO_PT_EDGES[i], OO_PT_EDGES[i + 1]
        bin_obs = [o for o in selected if p0 <= o.pt < p1]
        mean6, sem6 = _weighted_mean_sem_surv6(bin_obs)
        out[i], sem[i] = _raa9_with_sem(mean6, sem6, feeddown, sigmas_prim)
    return centers, out, sem


def _compute_vs_y(
    obs: Sequence,
    feeddown: np.ndarray,
    sigmas_prim: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (FINAL_Y_EDGES[:-1] + FINAL_Y_EDGES[1:])
    out = np.full((centers.shape[0], 9), np.nan, dtype=np.float64)
    sem = np.full((centers.shape[0], 9), np.nan, dtype=np.float64)
    selected = [o for o in obs if o.pt <= OO_PT_MAX_FOR_Y]
    for i in range(len(FINAL_Y_EDGES) - 1):
        y0, y1 = FINAL_Y_EDGES[i], FINAL_Y_EDGES[i + 1]
        bin_obs = [o for o in selected if y0 <= o.y < y1]
        mean6, sem6 = _weighted_mean_sem_surv6(bin_obs)
        out[i], sem[i] = _raa9_with_sem(mean6, sem6, feeddown, sigmas_prim)
    return centers, out, sem


def _compute_integrated(
    obs: Sequence,
    feeddown: np.ndarray,
    sigmas_prim: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y0, y1 = OO_INTEGRATED_Y_WINDOW
    selected = [o for o in obs if y0 <= o.y <= y1 and o.pt <= OO_PT_MAX_FOR_Y]
    mean6, sem6 = _weighted_mean_sem_surv6(selected)
    raa9, sem9 = _raa9_with_sem(mean6, sem6, feeddown, sigmas_prim)
    return mean6, sem6, raa9, sem9


def analyze_run(spec: RunSpec, logger: logging.Logger) -> RunResult:
    path = spec.datafile
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    logger.info("Loading %s from %s", spec.label, path)
    obs, load_mode = _load_observables(path, logger)
    feeddown = build_feeddown_matrix()
    sigmas_prim = solve_primordial_sigmas(feeddown, SIGMAS_EXP_OO_5360)

    pt_centers, raa9_pt, sem9_pt = _compute_vs_pt(obs, feeddown, sigmas_prim)
    y_centers, raa9_y, sem9_y = _compute_vs_y(obs, feeddown, sigmas_prim)
    mean6_int, sem6_int, raa9_int, sem9_int = _compute_integrated(obs, feeddown, sigmas_prim)

    logger.info(
        "%s: load=%s  nobs=%d  integrated RAA(1S)=%.6f  RAA(2S)=%.6f  RAA(3S)=%.6f",
        spec.label,
        load_mode,
        len(obs),
        raa9_int[0],
        raa9_int[1],
        raa9_int[5],
    )
    return RunResult(
        label=spec.label,
        datafile=path,
        load_mode=load_mode,
        n_observables=len(obs),
        pt_centers=pt_centers,
        raa9_pt=raa9_pt,
        sem9_pt=sem9_pt,
        y_centers=y_centers,
        raa9_y=raa9_y,
        sem9_y=sem9_y,
        mean6_int=mean6_int,
        sem6_int=sem6_int,
        raa9_int=raa9_int,
        sem9_int=sem9_int,
    )


def _compute_dr(raa9: np.ndarray) -> np.ndarray:
    # return dr of shape (..., 3) for 2S/1S, 3S/1S, 3S/2S
    dr = np.full(raa9.shape[:-1] + (3,), np.nan, dtype=np.float64)
    mask_1S = raa9[..., 0] > 0
    mask_2S = raa9[..., 1] > 0
    # 2S / 1S
    dr[..., 0] = np.where(mask_1S, raa9[..., 1] / raa9[..., 0], np.nan)
    # 3S / 1S
    dr[..., 1] = np.where(mask_1S, raa9[..., 5] / raa9[..., 0], np.nan)
    # 3S / 2S
    dr[..., 2] = np.where(mask_2S, raa9[..., 5] / raa9[..., 1], np.nan)
    return dr


def _ratio_with_err(
    num: np.ndarray,
    den: np.ndarray,
    num_err: np.ndarray,
    den_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    num_err = np.asarray(num_err, dtype=np.float64)
    den_err = np.asarray(den_err, dtype=np.float64)

    ratio = np.full_like(num, np.nan, dtype=np.float64)
    err = np.full_like(num, np.nan, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den) & np.isfinite(num_err) & np.isfinite(den_err) & (den > 0.0)
    if not np.any(ok):
        return ratio, err

    ratio[ok] = num[ok] / den[ok]
    err[ok] = np.sqrt(
        (num_err[ok] / den[ok]) ** 2
        + ((num[ok] * den_err[ok]) / (den[ok] ** 2)) ** 2
    )
    return ratio, err


def _compute_dr_err(raa9: np.ndarray, sem9: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dr = np.full(raa9.shape[:-1] + (3,), np.nan, dtype=np.float64)
    err = np.full_like(dr, np.nan, dtype=np.float64)
    dr[..., 0], err[..., 0] = _ratio_with_err(raa9[..., 1], raa9[..., 0], sem9[..., 1], sem9[..., 0])
    dr[..., 1], err[..., 1] = _ratio_with_err(raa9[..., 5], raa9[..., 0], sem9[..., 5], sem9[..., 0])
    dr[..., 2], err[..., 2] = _ratio_with_err(raa9[..., 5], raa9[..., 1], sem9[..., 5], sem9[..., 1])
    return dr, err


def _envelope(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    high = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    mid = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    for idx in np.ndindex(stack.shape[1:]):
        vals = stack[(slice(None),) + idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        low[idx] = float(np.min(vals))
        high[idx] = float(np.max(vals))
        mid[idx] = 0.5 * (low[idx] + high[idx])
    return mid, low, high


def _stat_augmented_envelope(
    center_stack: np.ndarray,
    sem_stack: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.full(center_stack.shape[1:], np.nan, dtype=np.float64)
    high = np.full(center_stack.shape[1:], np.nan, dtype=np.float64)
    max_sem = np.full(center_stack.shape[1:], np.nan, dtype=np.float64)
    for idx in np.ndindex(center_stack.shape[1:]):
        centers = center_stack[(slice(None),) + idx]
        sems = sem_stack[(slice(None),) + idx]
        ok = np.isfinite(centers)
        if not np.any(ok):
            continue
        sems = np.where(np.isfinite(sems), sems, 0.0)
        low[idx] = float(np.min(centers[ok] - sems[ok]))
        high[idx] = float(np.max(centers[ok] + sems[ok]))
        max_sem[idx] = float(np.max(sems[ok]))
    return low, high, max_sem


def _contiguous_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    spans: List[Tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for val in idx[1:]:
        cur = int(val)
        if cur != prev + 1:
            spans.append((start, prev))
            start = cur
        prev = cur
    spans.append((start, prev))
    return spans


def _step_band_asymmetric(
    ax,
    bin_edges: np.ndarray,
    y_mid: np.ndarray,
    y_low: np.ndarray,
    y_high: np.ndarray,
    *,
    facecolor: str,
    linecolor: str,
    label: str | None,
    alpha: float = 0.22,
    linewidth: float = 2.0,
) -> None:
    y_mid = np.asarray(y_mid, dtype=np.float64)
    y_low = np.asarray(y_low, dtype=np.float64)
    y_high = np.asarray(y_high, dtype=np.float64)
    mask = np.isfinite(y_mid) & np.isfinite(y_low) & np.isfinite(y_high)
    for i, (start, stop) in enumerate(_contiguous_spans(mask)):
        edges = np.asarray(bin_edges, dtype=np.float64)[start : stop + 2]
        mid_seg = y_mid[start : stop + 1]
        low_seg = y_low[start : stop + 1]
        high_seg = y_high[start : stop + 1]
        x = edges
        y_step = np.concatenate([mid_seg, [mid_seg[-1]]])
        y_lo = np.concatenate([low_seg, [low_seg[-1]]])
        y_hi = np.concatenate([high_seg, [high_seg[-1]]])
        ax.fill_between(
            x,
            y_lo,
            y_hi,
            step="post",
            color=facecolor,
            alpha=alpha,
            linewidth=0.0,
            zorder=1,
        )
        ax.step(
            x,
            y_step,
            where="post",
            color=linecolor,
            linewidth=linewidth,
            label=label if i == 0 else None,
            zorder=2,
        )


def _step_line(
    ax,
    bin_edges: np.ndarray,
    y_center: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str | None,
    alpha: float = 0.95,
) -> None:
    y_center = np.asarray(y_center, dtype=np.float64)
    mask = np.isfinite(y_center)
    for i, (start, stop) in enumerate(_contiguous_spans(mask)):
        edges = np.asarray(bin_edges, dtype=np.float64)[start : stop + 2]
        center = y_center[start : stop + 1]
        x, y_step, _, _ = binned_step_series(edges, center, np.zeros_like(center))
        ax.step(
            x,
            y_step,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label if i == 0 else None,
            zorder=3,
        )


def _line_plot(
    ax,
    x: np.ndarray,
    y_mid: np.ndarray,
    y_low: np.ndarray,
    y_high: np.ndarray,
    *,
    facecolor: str,
    linecolor: str,
    label: str | None,
    alpha: float = 0.22,
    linewidth: float = 2.0,
) -> None:
    x = np.asarray(x, dtype=np.float64)
    y_mid = np.asarray(y_mid, dtype=np.float64)
    y_low = np.asarray(y_low, dtype=np.float64)
    y_high = np.asarray(y_high, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y_mid) & np.isfinite(y_low) & np.isfinite(y_high)
    if not np.any(mask):
        return
    ax.fill_between(x[mask], y_low[mask], y_high[mask], color=facecolor, alpha=alpha, linewidth=0.0, zorder=1)
    ax.plot(x[mask], y_mid[mask], color=linecolor, linewidth=linewidth, label=label, zorder=2)


def _line_overlay(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str | None,
    alpha: float = 0.95,
) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return
    ax.plot(
        x[mask],
        y[mask],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=3,
    )


def _center_sem_band(center: np.ndarray, sem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float64)
    sem = np.asarray(sem, dtype=np.float64)
    low = np.full_like(center, np.nan, dtype=np.float64)
    high = np.full_like(center, np.nan, dtype=np.float64)
    ok = np.isfinite(center)
    sem = np.where(np.isfinite(sem), sem, 0.0)
    low[ok] = center[ok] - sem[ok]
    high[ok] = center[ok] + sem[ok]
    return low, high


def _compute_ylim(*arrays: np.ndarray) -> Tuple[float, float]:
    ymax = 0.0
    for arr in arrays:
        vals = np.asarray(arr, dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            ymax = max(ymax, float(np.max(finite)))
    upper = max(0.9, min(1.2, 1.12 * ymax + 0.05))
    return 0.0, upper


def _result_by_label(results: Sequence[RunResult], label: str) -> RunResult:
    for result in results:
        if result.label == label:
            return result
    labels = ", ".join(r.label for r in results)
    raise KeyError(f"reference label {label!r} not found in runs: {labels}")


def _reference_band(
    center_stack: np.ndarray,
    sem_stack: np.ndarray,
    ref_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(center_stack[ref_index], dtype=np.float64).copy()
    stat = np.asarray(sem_stack[ref_index], dtype=np.float64).copy()
    shape = center.shape
    model_low = np.full(shape, np.nan, dtype=np.float64)
    model_high = np.full(shape, np.nan, dtype=np.float64)
    total_low = np.full(shape, np.nan, dtype=np.float64)
    total_high = np.full(shape, np.nan, dtype=np.float64)

    other_indices = [i for i in range(center_stack.shape[0]) if i != ref_index]
    others = center_stack[other_indices] if other_indices else np.empty((0,) + shape, dtype=np.float64)

    for idx in np.ndindex(shape):
        c = float(center[idx]) if np.isfinite(center[idx]) else np.nan
        if not np.isfinite(c):
            continue
        s = float(stat[idx]) if np.isfinite(stat[idx]) else 0.0
        vals = others[(slice(None),) + idx] if others.size else np.array([], dtype=np.float64)
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            lo = c
            hi = c
        else:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
        model_low[idx] = lo
        model_high[idx] = hi
        sys_lo = max(c - lo, 0.0)
        sys_hi = max(hi - c, 0.0)
        total_low[idx] = c - np.sqrt(s * s + sys_lo * sys_lo)
        total_high[idx] = c + np.sqrt(s * s + sys_hi * sys_hi)

    return center, model_low, model_high, total_low, total_high, stat


def _configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    return plt


def _plot_single_run_publication(result: RunResult, outdir: Path) -> None:
    plt = _configure_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pt_low, pt_high = _center_sem_band(result.raa9_pt, result.sem9_pt)
    y_low, y_high = _center_sem_band(result.raa9_y, result.sem9_y)
    int_low, int_high = _center_sem_band(result.raa9_int, result.sem9_int)

    dr_pt, dr_pt_err = _compute_dr_err(result.raa9_pt, result.sem9_pt)
    dr_y, dr_y_err = _compute_dr_err(result.raa9_y, result.sem9_y)
    dr_int, dr_int_err = _compute_dr_err(result.raa9_int, result.sem9_int)
    dr_pt_low, dr_pt_high = _center_sem_band(dr_pt, dr_pt_err)
    dr_y_low, dr_y_high = _center_sem_band(dr_y, dr_y_err)
    dr_int_low, dr_int_high = _center_sem_band(dr_int, dr_int_err)

    band_color = "#9aa0a6"
    mid_color = "#111111"
    label = f"QTraj-NLO ({result.label}): stat band"
    cms_handle = Line2D([0], [0], color="#9467bd", marker="s", lw=0, markersize=6)

    state_sel = np.asarray(MAIN_STATE_INDICES, dtype=int)
    common_ylim = _compute_ylim(pt_high[:, state_sel], y_high[:, state_sel], int_high[state_sel])

    fig_pt, axes_pt = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (state_idx, label_tex) in enumerate(zip(MAIN_STATE_INDICES, MAIN_STATE_TEX)):
        ax = axes_pt[panel]
        _step_band_asymmetric(
            ax,
            OO_PT_EDGES,
            result.raa9_pt[:, state_idx],
            pt_low[:, state_idx],
            pt_high[:, state_idx],
            facecolor=band_color,
            linecolor=mid_color,
            label=result.label if panel == 0 else None,
        )
        ax.set_title(label_tex)
        ax.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
        ax.set_xlim(0.0, FINAL_PT_PLOT_MAX)
        ax.set_ylim(*common_ylim)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
        ax.grid(alpha=0.2, lw=0.5)
    axes_pt[0].set_ylabel(r"$R_{AA}$")
    axes_pt[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$b \simeq 4.497\ \mathrm{{fm}},\ -2.4 \leq y \leq 2.4$" "\n"
        + rf"{result.label.replace('_', ' ')}",
        transform=axes_pt[0].transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
    )
    axes_pt[0].legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        [label],
        loc="lower left",
        framealpha=0.95,
    )
    fig_pt.tight_layout()
    fig_pt.savefig(outdir / f"oo5360_raavspt__{result.label}__triplet.pdf", bbox_inches="tight")
    fig_pt.savefig(outdir / f"oo5360_raavspt__{result.label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_pt)

    fig_y, axes_y = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (state_idx, label_tex) in enumerate(zip(MAIN_STATE_INDICES, MAIN_STATE_TEX)):
        ax = axes_y[panel]
        _line_plot(
            ax,
            result.y_centers,
            result.raa9_y[:, state_idx],
            y_low[:, state_idx],
            y_high[:, state_idx],
            facecolor=band_color,
            linecolor=mid_color,
            label=result.label if panel == 0 else None,
        )
        ax.set_title(label_tex)
        ax.set_xlabel(r"$y$")
        ax.set_xlim(-FINAL_Y_PLOT_MAX, FINAL_Y_PLOT_MAX)
        ax.set_ylim(*common_ylim)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
        ax.grid(alpha=0.2, lw=0.5)
    axes_y[0].set_ylabel(r"$R_{AA}$")
    axes_y[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$b \simeq 4.497\ \mathrm{{fm}},\ p_T \leq {OO_PT_MAX_FOR_Y:.0f}\ \mathrm{{GeV}}$" "\n"
        + rf"{result.label.replace('_', ' ')}",
        transform=axes_y[0].transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
    )
    axes_y[0].legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        [label],
        loc="lower left",
        framealpha=0.95,
    )
    fig_y.tight_layout()
    fig_y.savefig(outdir / f"oo5360_raavsy__{result.label}__triplet.pdf", bbox_inches="tight")
    fig_y.savefig(outdir / f"oo5360_raavsy__{result.label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_y)

    fig_int, ax_int = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(MAIN_STATE_NAMES), dtype=np.float64)
    width = 0.62
    for i, state_idx in enumerate(MAIN_STATE_INDICES):
        ax_int.fill_between(
            [x[i] - width / 2.0, x[i] + width / 2.0],
            [int_low[state_idx], int_low[state_idx]],
            [int_high[state_idx], int_high[state_idx]],
            color=band_color,
            alpha=0.22,
            linewidth=0.0,
            zorder=1,
        )
        ax_int.hlines(
            result.raa9_int[state_idx],
            x[i] - width / 2.0,
            x[i] + width / 2.0,
            color=mid_color,
            linewidth=2.2,
            zorder=2,
        )
    ax_int.set_xticks(x, MAIN_STATE_TEX)
    ax_int.set_ylabel(r"$R_{AA}$")
    ax_int.set_ylim(*common_ylim)
    ax_int.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    ax_int.grid(axis="y", alpha=0.2, lw=0.5)
    ax_int.legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        [label],
        loc="upper right",
        framealpha=0.95,
    )
    fig_int.tight_layout()
    fig_int.savefig(outdir / f"oo5360_integrated__{result.label}__triplet.pdf", bbox_inches="tight")
    fig_int.savefig(outdir / f"oo5360_integrated__{result.label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_int)

    cms_3s2s_pt_x = [1.0, 7.0, 21.0]
    cms_3s2s_pt_y = [0.75, 0.58, 0.48]
    cms_3s2s_pt_stat = [0.26, 0.22, 0.19]
    cms_3s2s_pt_sys = [0.08, 0.06, 0.03]
    cms_dr_int_y = [0.66, 0.39, 0.60]
    cms_dr_int_stat = [0.05, 0.07, 0.13]
    cms_dr_int_sys = [0.02, 0.03, 0.03]

    dr_common_ylim = _compute_ylim(
        dr_pt_high,
        dr_y_high,
        dr_int_high,
        np.array(cms_3s2s_pt_y) + np.array(cms_3s2s_pt_stat) + np.array(cms_3s2s_pt_sys),
    )
    dr_labels = (
        r"$\Upsilon(2S) / \Upsilon(1S)$",
        r"$\Upsilon(3S) / \Upsilon(1S)$",
        r"$\Upsilon(3S) / \Upsilon(2S)$",
    )
    dr_state_indices = (0, 1, 2)

    fig_dr_pt, axes_dr_pt = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (dr_idx, label_tex) in enumerate(zip(dr_state_indices, dr_labels)):
        ax = axes_dr_pt[panel]
        _step_band_asymmetric(
            ax,
            OO_PT_EDGES,
            dr_pt[:, dr_idx],
            dr_pt_low[:, dr_idx],
            dr_pt_high[:, dr_idx],
            facecolor=band_color,
            linecolor=mid_color,
            label=result.label if panel == 0 else None,
        )
        if dr_idx == 2:
            ax.errorbar(cms_3s2s_pt_x, cms_3s2s_pt_y, yerr=cms_3s2s_pt_stat, fmt="s", color="#9467bd", zorder=5)
            for cx, cy, cey in zip(cms_3s2s_pt_x, cms_3s2s_pt_y, cms_3s2s_pt_sys):
                ax.add_patch(plt.Rectangle((cx - 0.7, cy - cey), 1.4, 2 * cey, facecolor="none", edgecolor="#9467bd", alpha=0.5, zorder=4))
        ax.set_title(label_tex)
        ax.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
        ax.set_xlim(0.0, FINAL_PT_PLOT_MAX)
        ax.set_ylim(*dr_common_ylim)
        ax.grid(alpha=0.2, lw=0.5)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    axes_dr_pt[0].set_ylabel("Double Ratio")
    axes_dr_pt[0].legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5), cms_handle],
        [label, "CMS Prelim"],
        loc="upper right",
        framealpha=0.95,
    )
    fig_dr_pt.tight_layout()
    fig_dr_pt.savefig(outdir / f"oo5360_drvspt__{result.label}__triplet.pdf", bbox_inches="tight")
    fig_dr_pt.savefig(outdir / f"oo5360_drvspt__{result.label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_dr_pt)

    fig_dr_y, axes_dr_y = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (dr_idx, label_tex) in enumerate(zip(dr_state_indices, dr_labels)):
        ax = axes_dr_y[panel]
        _line_plot(
            ax,
            result.y_centers,
            dr_y[:, dr_idx],
            dr_y_low[:, dr_idx],
            dr_y_high[:, dr_idx],
            facecolor=band_color,
            linecolor=mid_color,
            label=result.label if panel == 0 else None,
        )
        ax.set_title(label_tex)
        ax.set_xlabel(r"$y$")
        ax.set_xlim(-FINAL_Y_PLOT_MAX, FINAL_Y_PLOT_MAX)
        ax.set_ylim(*dr_common_ylim)
        ax.grid(alpha=0.2, lw=0.5)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    axes_dr_y[0].set_ylabel("Double Ratio")
    axes_dr_y[0].legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        [label],
        loc="upper right",
        framealpha=0.95,
    )
    fig_dr_y.tight_layout()
    fig_dr_y.savefig(outdir / f"oo5360_drvsy__{result.label}__triplet.pdf", bbox_inches="tight")
    fig_dr_y.savefig(outdir / f"oo5360_drvsy__{result.label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_dr_y)

    fig_dr_int, ax_dr_int = plt.subplots(figsize=(8.0, 5.0))
    x_dr = np.arange(len(dr_labels), dtype=np.float64)
    width = 0.62
    for i, dr_idx in enumerate(dr_state_indices):
        ax_dr_int.fill_between(
            [x_dr[i] - width / 2.0, x_dr[i] + width / 2.0],
            [dr_int_low[dr_idx], dr_int_low[dr_idx]],
            [dr_int_high[dr_idx], dr_int_high[dr_idx]],
            color=band_color,
            alpha=0.22,
            linewidth=0.0,
            zorder=1,
        )
        ax_dr_int.hlines(
            dr_int[dr_idx],
            x_dr[i] - width / 2.0,
            x_dr[i] + width / 2.0,
            color=mid_color,
            linewidth=2.2,
            zorder=2,
        )
        ax_dr_int.errorbar([x_dr[i]], [cms_dr_int_y[i]], yerr=[cms_dr_int_stat[i]], fmt="s", color="#9467bd", zorder=5)
        ax_dr_int.add_patch(plt.Rectangle((x_dr[i] - 0.15, cms_dr_int_y[i] - cms_dr_int_sys[i]), 0.3, 2 * cms_dr_int_sys[i], facecolor="none", edgecolor="#9467bd", alpha=0.5, zorder=4))
    ax_dr_int.set_xticks(x_dr, dr_labels)
    ax_dr_int.set_ylabel("Double Ratio")
    ax_dr_int.set_ylim(*dr_common_ylim)
    ax_dr_int.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    ax_dr_int.grid(axis="y", alpha=0.2, lw=0.5)
    ax_dr_int.legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5), cms_handle],
        [label, "CMS Prelim"],
        loc="upper right",
        framealpha=0.95,
    )
    fig_dr_int.tight_layout()
    fig_dr_int.savefig(outdir / f"oo5360_integrated_dr__{result.label}__triplet.pdf", bbox_inches="tight")
    fig_dr_int.savefig(outdir / f"oo5360_integrated_dr__{result.label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_dr_int)


def _plot_publication(results: Sequence[RunResult], outdir: Path) -> None:
    if len(results) < 2:
        return

    plt = _configure_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pt_stack = np.stack([r.raa9_pt for r in results], axis=0)
    pt_sem_stack = np.stack([r.sem9_pt for r in results], axis=0)
    y_stack = np.stack([r.raa9_y for r in results], axis=0)
    y_sem_stack = np.stack([r.sem9_y for r in results], axis=0)
    int_stack = np.stack([r.raa9_int for r in results], axis=0)
    int_sem_stack = np.stack([r.sem9_int for r in results], axis=0)
    pt_center, pt_low, pt_high = _envelope(pt_stack)
    y_center, y_low, y_high = _envelope(y_stack)
    int_center, int_low, int_high = _envelope(int_stack)
    pt_total_low, pt_total_high, _ = _stat_augmented_envelope(pt_stack, pt_sem_stack)
    y_total_low, y_total_high, _ = _stat_augmented_envelope(y_stack, y_sem_stack)
    int_total_low, int_total_high, _ = _stat_augmented_envelope(int_stack, int_sem_stack)

    dr_pt_stack = np.stack([_compute_dr_err(r.raa9_pt, r.sem9_pt)[0] for r in results], axis=0)
    dr_pt_err_stack = np.stack([_compute_dr_err(r.raa9_pt, r.sem9_pt)[1] for r in results], axis=0)
    dr_y_stack = np.stack([_compute_dr_err(r.raa9_y, r.sem9_y)[0] for r in results], axis=0)
    dr_y_err_stack = np.stack([_compute_dr_err(r.raa9_y, r.sem9_y)[1] for r in results], axis=0)
    dr_int_stack = np.stack([_compute_dr_err(r.raa9_int, r.sem9_int)[0] for r in results], axis=0)
    dr_int_err_stack = np.stack([_compute_dr_err(r.raa9_int, r.sem9_int)[1] for r in results], axis=0)

    dr_pt_center, dr_pt_low, dr_pt_high = _envelope(dr_pt_stack)
    dr_y_center, dr_y_low, dr_y_high = _envelope(dr_y_stack)
    dr_int_center, dr_int_low, dr_int_high = _envelope(dr_int_stack)
    dr_pt_total_low, dr_pt_total_high, _ = _stat_augmented_envelope(dr_pt_stack, dr_pt_err_stack)
    dr_y_total_low, dr_y_total_high, _ = _stat_augmented_envelope(dr_y_stack, dr_y_err_stack)
    dr_int_total_low, dr_int_total_high, _ = _stat_augmented_envelope(dr_int_stack, dr_int_err_stack)

    band_color = "#9aa0a6"
    mid_color = "#111111"
    kappa_text = ",".join(r.label.replace("kappa", "") for r in results)
    band_text = rf"Quantum jumps ON; $\kappa={kappa_text}$ envelope + traj. SEM"
    state_sel = np.asarray(MAIN_STATE_INDICES, dtype=int)
    common_ylim = _compute_ylim(
        pt_total_high[:, state_sel],
        y_total_high[:, state_sel],
        int_total_high[state_sel],
    )

    fig_pt, axes_pt = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (state_idx, label_tex) in enumerate(zip(MAIN_STATE_INDICES, MAIN_STATE_TEX)):
        ax = axes_pt[panel]
        _step_band_asymmetric(
            ax,
            OO_PT_EDGES,
            pt_center[:, state_idx],
            pt_total_low[:, state_idx],
            pt_total_high[:, state_idx],
            facecolor=band_color,
            linecolor=mid_color,
            label="Envelope central" if panel == 0 else None,
        )
        ax.set_title(label_tex)
        ax.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
        ax.set_xlim(0.0, FINAL_PT_PLOT_MAX)
        ax.set_ylim(*common_ylim)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
        ax.grid(alpha=0.2, lw=0.5)
    axes_pt[0].set_ylabel(r"$R_{AA}$")
    axes_pt[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$b \simeq 4.497\ \mathrm{{fm}},\ -2.4 \leq y \leq 2.4$" "\n"
        + band_text,
        transform=axes_pt[0].transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
    )
    axes_pt[0].legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        ["QTraj-NLO (wReg): total band"],
        loc="lower left",
        framealpha=0.95,
    )
    fig_pt.tight_layout()
    fig_pt.savefig(outdir / "oo5360_raavspt__band_triplet.pdf", bbox_inches="tight")
    fig_pt.savefig(outdir / "oo5360_raavspt__band_triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_pt)

    fig_y, axes_y = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (state_idx, label_tex) in enumerate(zip(MAIN_STATE_INDICES, MAIN_STATE_TEX)):
        ax = axes_y[panel]
        _line_plot(
            ax,
            results[0].y_centers,
            y_center[:, state_idx],
            y_total_low[:, state_idx],
            y_total_high[:, state_idx],
            facecolor=band_color,
            linecolor=mid_color,
            label="Envelope central" if panel == 0 else None,
        )
        ax.set_title(label_tex)
        ax.set_xlabel(r"$y$")
        ax.set_xlim(-FINAL_Y_PLOT_MAX, FINAL_Y_PLOT_MAX)
        ax.set_ylim(*common_ylim)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
        ax.grid(alpha=0.2, lw=0.5)
    axes_y[0].set_ylabel(r"$R_{AA}$")
    axes_y[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$b \simeq 4.497\ \mathrm{{fm}},\ p_T \leq {OO_PT_MAX_FOR_Y:.0f}\ \mathrm{{GeV}}$" "\n"
        + band_text,
        transform=axes_y[0].transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
    )
    axes_y[0].legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        ["QTraj-NLO (wReg): total band"],
        loc="lower left",
        framealpha=0.95,
    )
    fig_y.tight_layout()
    fig_y.savefig(outdir / "oo5360_raavsy__band_triplet.pdf", bbox_inches="tight")
    fig_y.savefig(outdir / "oo5360_raavsy__band_triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_y)

    fig_int, ax_int = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(MAIN_STATE_NAMES), dtype=np.float64)
    width = 0.62
    for i, state_name in enumerate(MAIN_STATE_NAMES):
        ax_int.fill_between(
            [x[i] - width / 2.0, x[i] + width / 2.0],
            [int_total_low[MAIN_STATE_INDICES[i]], int_total_low[MAIN_STATE_INDICES[i]]],
            [int_total_high[MAIN_STATE_INDICES[i]], int_total_high[MAIN_STATE_INDICES[i]]],
            color=band_color,
            alpha=0.22,
            linewidth=0.0,
            zorder=1,
        )
        ax_int.hlines(
            int_center[MAIN_STATE_INDICES[i]],
            x[i] - width / 2.0,
            x[i] + width / 2.0,
            color=mid_color,
            linewidth=2.2,
            zorder=2,
        )
    ax_int.set_xticks(x, MAIN_STATE_TEX)
    ax_int.set_ylabel(r"$R_{AA}$")
    ax_int.set_ylim(*common_ylim)
    ax_int.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    ax_int.grid(axis="y", alpha=0.2, lw=0.5)
    ax_int.text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$b \simeq 4.497\ \mathrm{{fm}},\ -2.4 \leq y \leq 2.4,\ p_T \leq {OO_PT_MAX_FOR_Y:.0f}\ \mathrm{{GeV}}$",
        transform=ax_int.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
    )
    ax_int.legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        ["QTraj-NLO (wReg): total band"],
        loc="upper right",
        framealpha=0.95,
    )
    fig_int.tight_layout()
    fig_int.savefig(outdir / "oo5360_integrated__band_triplet.pdf", bbox_inches="tight")
    fig_int.savefig(outdir / "oo5360_integrated__band_triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_int)

    dr_labels = (r"$\Upsilon(2S) / \Upsilon(1S)$", r"$\Upsilon(3S) / \Upsilon(1S)$", r"$\Upsilon(3S) / \Upsilon(2S)$")
    dr_state_indices = (0, 1, 2)

    cms_3s2s_pt_x = [1.0, 7.0, 21.0]
    cms_3s2s_pt_y = [0.75, 0.58, 0.48]
    cms_3s2s_pt_stat = [0.26, 0.22, 0.19]
    cms_3s2s_pt_sys = [0.08, 0.06, 0.03]
    
    cms_dr_int_y = [0.66, 0.39, 0.60]
    cms_dr_int_stat = [0.05, 0.07, 0.13]
    cms_dr_int_sys = [0.02, 0.03, 0.03]

    cms_handle = Line2D([0], [0], color="#9467bd", marker="s", lw=0, markersize=6)

    # DR vs pT
    dr_common_ylim = _compute_ylim(
        dr_pt_total_high,
        dr_y_total_high,
        dr_int_total_high,
        np.array(cms_3s2s_pt_y) + np.array(cms_3s2s_pt_stat) + np.array(cms_3s2s_pt_sys),
    )
    fig_dr_pt, axes_dr_pt = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (dr_idx, label_tex) in enumerate(zip(dr_state_indices, dr_labels)):
        ax = axes_dr_pt[panel]
        _step_band_asymmetric(
            ax,
            OO_PT_EDGES,
            dr_pt_center[:, dr_idx],
            dr_pt_total_low[:, dr_idx],
            dr_pt_total_high[:, dr_idx],
            facecolor=band_color, linecolor=mid_color, label="Envelope central" if panel == 0 else None,
        )
        if dr_idx == 2:
            ax.errorbar(cms_3s2s_pt_x, cms_3s2s_pt_y, yerr=cms_3s2s_pt_stat, fmt="s", color="#9467bd", label="CMS Prelim", zorder=5)
            for cx, cy, cey in zip(cms_3s2s_pt_x, cms_3s2s_pt_y, cms_3s2s_pt_sys):
                ax.add_patch(plt.Rectangle((cx-0.7, cy-cey), 1.4, 2*cey, facecolor="none", edgecolor="#9467bd", alpha=0.5, zorder=4))

        ax.set_title(label_tex)
        ax.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
        ax.set_xlim(0.0, FINAL_PT_PLOT_MAX)
        ax.set_ylim(*dr_common_ylim)
        ax.grid(alpha=0.2, lw=0.5)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)

    axes_dr_pt[0].set_ylabel("Double Ratio")
    axes_dr_pt[0].text(0.03, 0.97, r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n" + rf"$b \simeq 4.497\ \mathrm{{fm}},\ -2.4 \leq y \leq 2.4$" "\n" + band_text, transform=axes_dr_pt[0].transAxes, va="top", ha="left", fontsize=10.5)
    axes_dr_pt[0].legend([Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5), cms_handle], ["QTraj-NLO (wReg): total band", "CMS Prelim"], loc="upper right" if dr_pt_center[0,0] < 0.5 else "lower left", framealpha=0.95)
    fig_dr_pt.tight_layout()
    fig_dr_pt.savefig(outdir / "oo5360_drvspt__band_triplet.pdf", bbox_inches="tight")
    fig_dr_pt.savefig(outdir / "oo5360_drvspt__band_triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_dr_pt)

    # DR vs y
    fig_dr_y, axes_dr_y = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (dr_idx, label_tex) in enumerate(zip(dr_state_indices, dr_labels)):
        ax = axes_dr_y[panel]
        _line_plot(
            ax,
            results[0].y_centers,
            dr_y_center[:, dr_idx],
            dr_y_total_low[:, dr_idx],
            dr_y_total_high[:, dr_idx],
            facecolor=band_color, linecolor=mid_color, label="Envelope central" if panel == 0 else None,
        )
        ax.set_title(label_tex)
        ax.set_xlabel(r"$y$")
        ax.set_xlim(-FINAL_Y_PLOT_MAX, FINAL_Y_PLOT_MAX)
        ax.set_ylim(*dr_common_ylim)
        ax.grid(alpha=0.2, lw=0.5)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)

    axes_dr_y[0].set_ylabel("Double Ratio")
    axes_dr_y[0].text(0.03, 0.97, r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n" + rf"$b \simeq 4.497\ \mathrm{{fm}},\ p_T \leq {OO_PT_MAX_FOR_Y:.0f}\ \mathrm{{GeV}}$" "\n" + band_text, transform=axes_dr_y[0].transAxes, va="top", ha="left", fontsize=10.5)
    axes_dr_y[0].legend([Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)], ["QTraj-NLO (wReg): total band"], loc="upper right" if dr_y_center[0,0] < 0.5 else "lower left", framealpha=0.95)
    fig_dr_y.tight_layout()
    fig_dr_y.savefig(outdir / "oo5360_drvsy__band_triplet.pdf", bbox_inches="tight")
    fig_dr_y.savefig(outdir / "oo5360_drvsy__band_triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_dr_y)

    # DR Integrated
    fig_dr_int, ax_dr_int = plt.subplots(figsize=(8.0, 5.0))
    x_dr = np.arange(len(dr_labels), dtype=np.float64)
    width = 0.62
    for i, dr_idx in enumerate(dr_state_indices):
        ax_dr_int.fill_between(
            [x_dr[i] - width / 2.0, x_dr[i] + width / 2.0],
            [dr_int_total_low[dr_idx]] * 2,
            [dr_int_total_high[dr_idx]] * 2,
            color=band_color,
            alpha=0.22,
            linewidth=0.0,
            zorder=1,
        )
        ax_dr_int.hlines(dr_int_center[dr_idx], x_dr[i] - width / 2.0, x_dr[i] + width / 2.0, color=mid_color, linewidth=2.2, zorder=2)
        # CMS overlay
        ax_dr_int.errorbar([x_dr[i]], [cms_dr_int_y[i]], yerr=[cms_dr_int_stat[i]], fmt="s", color="#9467bd", zorder=5)
        ax_dr_int.add_patch(plt.Rectangle((x_dr[i]-0.15, cms_dr_int_y[i]-cms_dr_int_sys[i]), 0.3, 2*cms_dr_int_sys[i], facecolor="none", edgecolor="#9467bd", alpha=0.5, zorder=4))

    ax_dr_int.set_xticks(x_dr, dr_labels)
    ax_dr_int.set_ylabel("Double Ratio")
    ax_dr_int.set_ylim(*dr_common_ylim)
    ax_dr_int.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    ax_dr_int.grid(axis="y", alpha=0.2, lw=0.5)
    ax_dr_int.text(0.03, 0.97, r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n" + rf"$b \simeq 4.497\ \mathrm{{fm}},\ -2.4 \leq y \leq 2.4,\ p_T \leq {OO_PT_MAX_FOR_Y:.0f}\ \mathrm{{GeV}}$", transform=ax_dr_int.transAxes, va="top", ha="left", fontsize=10.5)
    ax_dr_int.legend([Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5), cms_handle], ["QTraj-NLO (wReg): total band", "CMS Prelim"], loc="upper right", framealpha=0.95)
    fig_dr_int.tight_layout()
    fig_dr_int.savefig(outdir / "oo5360_integrated_dr__band_triplet.pdf", bbox_inches="tight")
    fig_dr_int.savefig(outdir / "oo5360_integrated_dr__band_triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_dr_int)


def _plot_reference_publication(
    results: Sequence[RunResult],
    ref_label: str,
    outdir: Path,
) -> None:
    plt = _configure_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ref_index = next(i for i, r in enumerate(results) if r.label == ref_label)
    ref_result = results[ref_index]

    pt_stack = np.stack([r.raa9_pt for r in results], axis=0)
    pt_sem_stack = np.stack([r.sem9_pt for r in results], axis=0)
    y_stack = np.stack([r.raa9_y for r in results], axis=0)
    y_sem_stack = np.stack([r.sem9_y for r in results], axis=0)
    int_stack = np.stack([r.raa9_int for r in results], axis=0)
    int_sem_stack = np.stack([r.sem9_int for r in results], axis=0)

    pt_center, pt_model_low, pt_model_high, pt_total_low, pt_total_high, _ = _reference_band(pt_stack, pt_sem_stack, ref_index)
    y_center, y_model_low, y_model_high, y_total_low, y_total_high, _ = _reference_band(y_stack, y_sem_stack, ref_index)
    int_center, int_model_low, int_model_high, int_total_low, int_total_high, _ = _reference_band(int_stack, int_sem_stack, ref_index)

    dr_pt_stack = np.stack([_compute_dr_err(r.raa9_pt, r.sem9_pt)[0] for r in results], axis=0)
    dr_pt_err_stack = np.stack([_compute_dr_err(r.raa9_pt, r.sem9_pt)[1] for r in results], axis=0)
    dr_y_stack = np.stack([_compute_dr_err(r.raa9_y, r.sem9_y)[0] for r in results], axis=0)
    dr_y_err_stack = np.stack([_compute_dr_err(r.raa9_y, r.sem9_y)[1] for r in results], axis=0)
    dr_int_stack = np.stack([_compute_dr_err(r.raa9_int, r.sem9_int)[0] for r in results], axis=0)
    dr_int_err_stack = np.stack([_compute_dr_err(r.raa9_int, r.sem9_int)[1] for r in results], axis=0)

    dr_pt_center, dr_pt_model_low, dr_pt_model_high, dr_pt_total_low, dr_pt_total_high, _ = _reference_band(dr_pt_stack, dr_pt_err_stack, ref_index)
    dr_y_center, dr_y_model_low, dr_y_model_high, dr_y_total_low, dr_y_total_high, _ = _reference_band(dr_y_stack, dr_y_err_stack, ref_index)
    dr_int_center, dr_int_model_low, dr_int_model_high, dr_int_total_low, dr_int_total_high, _ = _reference_band(dr_int_stack, dr_int_err_stack, ref_index)

    band_color = "#9aa0a6"
    mid_color = "#111111"
    model_label = rf"central={ref_label}, model=({', '.join(r.label for r in results if r.label != ref_label)})"
    cms_handle = Line2D([0], [0], color="#9467bd", marker="s", lw=0, markersize=6)

    state_sel = np.asarray(MAIN_STATE_INDICES, dtype=int)
    common_ylim = _compute_ylim(pt_total_high[:, state_sel], y_total_high[:, state_sel], int_total_high[state_sel])

    fig_pt, axes_pt = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (state_idx, label_tex) in enumerate(zip(MAIN_STATE_INDICES, MAIN_STATE_TEX)):
        ax = axes_pt[panel]
        _step_band_asymmetric(
            ax,
            OO_PT_EDGES,
            pt_center[:, state_idx],
            pt_total_low[:, state_idx],
            pt_total_high[:, state_idx],
            facecolor=band_color,
            linecolor=mid_color,
            label=ref_label if panel == 0 else None,
        )
        ax.set_title(label_tex)
        ax.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
        ax.set_xlim(0.0, FINAL_PT_PLOT_MAX)
        ax.set_ylim(*common_ylim)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
        ax.grid(alpha=0.2, lw=0.5)
    axes_pt[0].set_ylabel(r"$R_{AA}$")
    axes_pt[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$b \simeq 4.497\ \mathrm{{fm}},\ -2.4 \leq y \leq 2.4$" "\n"
        + rf"central={ref_label}, total = stat({ref_label}) $\oplus$ model(5,7)",
        transform=axes_pt[0].transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
    )
    axes_pt[0].legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        [model_label],
        loc="lower left",
        framealpha=0.95,
    )
    fig_pt.tight_layout()
    fig_pt.savefig(outdir / f"oo5360_raavspt__ref_{ref_label}__triplet.pdf", bbox_inches="tight")
    fig_pt.savefig(outdir / f"oo5360_raavspt__ref_{ref_label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_pt)

    fig_y, axes_y = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (state_idx, label_tex) in enumerate(zip(MAIN_STATE_INDICES, MAIN_STATE_TEX)):
        ax = axes_y[panel]
        _line_plot(
            ax,
            ref_result.y_centers,
            y_center[:, state_idx],
            y_total_low[:, state_idx],
            y_total_high[:, state_idx],
            facecolor=band_color,
            linecolor=mid_color,
            label=ref_label if panel == 0 else None,
        )
        ax.set_title(label_tex)
        ax.set_xlabel(r"$y$")
        ax.set_xlim(-FINAL_Y_PLOT_MAX, FINAL_Y_PLOT_MAX)
        ax.set_ylim(*common_ylim)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
        ax.grid(alpha=0.2, lw=0.5)
    axes_y[0].set_ylabel(r"$R_{AA}$")
    axes_y[0].legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        [model_label],
        loc="lower left",
        framealpha=0.95,
    )
    fig_y.tight_layout()
    fig_y.savefig(outdir / f"oo5360_raavsy__ref_{ref_label}__triplet.pdf", bbox_inches="tight")
    fig_y.savefig(outdir / f"oo5360_raavsy__ref_{ref_label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_y)

    fig_int, ax_int = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(MAIN_STATE_NAMES), dtype=np.float64)
    width = 0.62
    for i, state_idx in enumerate(MAIN_STATE_INDICES):
        ax_int.fill_between(
            [x[i] - width / 2.0, x[i] + width / 2.0],
            [int_total_low[state_idx], int_total_low[state_idx]],
            [int_total_high[state_idx], int_total_high[state_idx]],
            color=band_color,
            alpha=0.22,
            linewidth=0.0,
            zorder=1,
        )
        ax_int.hlines(
            int_center[state_idx],
            x[i] - width / 2.0,
            x[i] + width / 2.0,
            color=mid_color,
            linewidth=2.2,
            zorder=2,
        )
    ax_int.set_xticks(x, MAIN_STATE_TEX)
    ax_int.set_ylabel(r"$R_{AA}$")
    ax_int.set_ylim(*common_ylim)
    ax_int.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    ax_int.grid(axis="y", alpha=0.2, lw=0.5)
    ax_int.legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        [model_label],
        loc="upper right",
        framealpha=0.95,
    )
    fig_int.tight_layout()
    fig_int.savefig(outdir / f"oo5360_integrated__ref_{ref_label}__triplet.pdf", bbox_inches="tight")
    fig_int.savefig(outdir / f"oo5360_integrated__ref_{ref_label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_int)

    cms_3s2s_pt_x = [1.0, 7.0, 21.0]
    cms_3s2s_pt_y = [0.75, 0.58, 0.48]
    cms_3s2s_pt_stat = [0.26, 0.22, 0.19]
    cms_3s2s_pt_sys = [0.08, 0.06, 0.03]
    cms_dr_int_y = [0.66, 0.39, 0.60]
    cms_dr_int_stat = [0.05, 0.07, 0.13]
    cms_dr_int_sys = [0.02, 0.03, 0.03]

    dr_common_ylim = _compute_ylim(
        dr_pt_total_high,
        dr_y_total_high,
        dr_int_total_high,
        np.array(cms_3s2s_pt_y) + np.array(cms_3s2s_pt_stat) + np.array(cms_3s2s_pt_sys),
    )
    dr_labels = (
        r"$\Upsilon(2S) / \Upsilon(1S)$",
        r"$\Upsilon(3S) / \Upsilon(1S)$",
        r"$\Upsilon(3S) / \Upsilon(2S)$",
    )
    dr_state_indices = (0, 1, 2)

    fig_dr_pt, axes_dr_pt = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (dr_idx, label_tex) in enumerate(zip(dr_state_indices, dr_labels)):
        ax = axes_dr_pt[panel]
        _step_band_asymmetric(
            ax,
            OO_PT_EDGES,
            dr_pt_center[:, dr_idx],
            dr_pt_total_low[:, dr_idx],
            dr_pt_total_high[:, dr_idx],
            facecolor=band_color,
            linecolor=mid_color,
            label=ref_label if panel == 0 else None,
        )
        if dr_idx == 2:
            ax.errorbar(cms_3s2s_pt_x, cms_3s2s_pt_y, yerr=cms_3s2s_pt_stat, fmt="s", color="#9467bd", zorder=5)
            for cx, cy, cey in zip(cms_3s2s_pt_x, cms_3s2s_pt_y, cms_3s2s_pt_sys):
                ax.add_patch(plt.Rectangle((cx - 0.7, cy - cey), 1.4, 2 * cey, facecolor="none", edgecolor="#9467bd", alpha=0.5, zorder=4))
        ax.set_title(label_tex)
        ax.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
        ax.set_xlim(0.0, FINAL_PT_PLOT_MAX)
        ax.set_ylim(*dr_common_ylim)
        ax.grid(alpha=0.2, lw=0.5)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    axes_dr_pt[0].set_ylabel("Double Ratio")
    axes_dr_pt[0].legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5), cms_handle],
        [model_label, "CMS Prelim"],
        loc="upper right",
        framealpha=0.95,
    )
    fig_dr_pt.tight_layout()
    fig_dr_pt.savefig(outdir / f"oo5360_drvspt__ref_{ref_label}__triplet.pdf", bbox_inches="tight")
    fig_dr_pt.savefig(outdir / f"oo5360_drvspt__ref_{ref_label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_dr_pt)

    fig_dr_y, axes_dr_y = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for panel, (dr_idx, label_tex) in enumerate(zip(dr_state_indices, dr_labels)):
        ax = axes_dr_y[panel]
        _line_plot(
            ax,
            ref_result.y_centers,
            dr_y_center[:, dr_idx],
            dr_y_total_low[:, dr_idx],
            dr_y_total_high[:, dr_idx],
            facecolor=band_color,
            linecolor=mid_color,
            label=ref_label if panel == 0 else None,
        )
        ax.set_title(label_tex)
        ax.set_xlabel(r"$y$")
        ax.set_xlim(-FINAL_Y_PLOT_MAX, FINAL_Y_PLOT_MAX)
        ax.set_ylim(*dr_common_ylim)
        ax.grid(alpha=0.2, lw=0.5)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    axes_dr_y[0].set_ylabel("Double Ratio")
    axes_dr_y[0].legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5)],
        [model_label],
        loc="upper right",
        framealpha=0.95,
    )
    fig_dr_y.tight_layout()
    fig_dr_y.savefig(outdir / f"oo5360_drvsy__ref_{ref_label}__triplet.pdf", bbox_inches="tight")
    fig_dr_y.savefig(outdir / f"oo5360_drvsy__ref_{ref_label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_dr_y)

    fig_dr_int, ax_dr_int = plt.subplots(figsize=(8.0, 5.0))
    x_dr = np.arange(len(dr_labels), dtype=np.float64)
    width = 0.62
    for i, dr_idx in enumerate(dr_state_indices):
        ax_dr_int.fill_between(
            [x_dr[i] - width / 2.0, x_dr[i] + width / 2.0],
            [dr_int_total_low[dr_idx], dr_int_total_low[dr_idx]],
            [dr_int_total_high[dr_idx], dr_int_total_high[dr_idx]],
            color=band_color,
            alpha=0.22,
            linewidth=0.0,
            zorder=1,
        )
        ax_dr_int.hlines(
            dr_int_center[dr_idx],
            x_dr[i] - width / 2.0,
            x_dr[i] + width / 2.0,
            color=mid_color,
            linewidth=2.2,
            zorder=2,
        )
        ax_dr_int.errorbar([x_dr[i]], [cms_dr_int_y[i]], yerr=[cms_dr_int_stat[i]], fmt="s", color="#9467bd", zorder=5)
        ax_dr_int.add_patch(plt.Rectangle((x_dr[i] - 0.15, cms_dr_int_y[i] - cms_dr_int_sys[i]), 0.3, 2 * cms_dr_int_sys[i], facecolor="none", edgecolor="#9467bd", alpha=0.5, zorder=4))
    ax_dr_int.set_xticks(x_dr, dr_labels)
    ax_dr_int.set_ylabel("Double Ratio")
    ax_dr_int.set_ylim(*dr_common_ylim)
    ax_dr_int.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    ax_dr_int.grid(axis="y", alpha=0.2, lw=0.5)
    ax_dr_int.legend(
        [Patch(facecolor=band_color, alpha=0.5, edgecolor=mid_color, linewidth=1.5), cms_handle],
        [model_label, "CMS Prelim"],
        loc="upper right",
        framealpha=0.95,
    )
    fig_dr_int.tight_layout()
    fig_dr_int.savefig(outdir / f"oo5360_integrated_dr__ref_{ref_label}__triplet.pdf", bbox_inches="tight")
    fig_dr_int.savefig(outdir / f"oo5360_integrated_dr__ref_{ref_label}__triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig_dr_int)


def _write_per_run_csv(result: RunResult, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    pt_header = ["pt_center"] + [f"RAA_{name}" for name in STATE_NAMES_9] + [f"SEM_{name}" for name in STATE_NAMES_9]
    pt_rows = np.column_stack([result.pt_centers, result.raa9_pt, result.sem9_pt])
    np.savetxt(
        outdir / f"oo5360_raavspt__{result.label}__central.csv",
        pt_rows,
        delimiter=",",
        header=",".join(pt_header),
        comments="",
        fmt="%.10g",
    )

    y_header = ["y_center"] + [f"RAA_{name}" for name in STATE_NAMES_9] + [f"SEM_{name}" for name in STATE_NAMES_9]
    y_rows = np.column_stack([result.y_centers, result.raa9_y, result.sem9_y])
    np.savetxt(
        outdir / f"oo5360_raavsy__{result.label}__central.csv",
        y_rows,
        delimiter=",",
        header=",".join(y_header),
        comments="",
        fmt="%.10g",
    )

    int_header = [f"mean6_{name}" for name in ("1S", "2S", "1P", "3S", "2P", "1D")]
    int_header += [f"sem6_{name}" for name in ("1S", "2S", "1P", "3S", "2P", "1D")]
    int_header += [f"RAA_{name}" for name in STATE_NAMES_9]
    int_header += [f"SEM_{name}" for name in STATE_NAMES_9]
    int_row = np.concatenate([result.mean6_int, result.sem6_int, result.raa9_int, result.sem9_int]).reshape(1, -1)
    np.savetxt(
        outdir / f"oo5360_integrated__{result.label}__central.csv",
        int_row,
        delimiter=",",
        header=",".join(int_header),
        comments="",
        fmt="%.10g",
    )

    dr_pt, dr_pt_err = _compute_dr_err(result.raa9_pt, result.sem9_pt)
    dr_pt_header = ["pt_center", "DR_2S1S", "ERR_DR_2S1S", "DR_3S1S", "ERR_DR_3S1S", "DR_3S2S", "ERR_DR_3S2S"]
    dr_pt_rows = np.column_stack([result.pt_centers, dr_pt[:, 0], dr_pt_err[:, 0], dr_pt[:, 1], dr_pt_err[:, 1], dr_pt[:, 2], dr_pt_err[:, 2]])
    np.savetxt(outdir / f"oo5360_drvspt__{result.label}__central.csv", dr_pt_rows, delimiter=",", header=",".join(dr_pt_header), comments="", fmt="%.10g")

    dr_y, dr_y_err = _compute_dr_err(result.raa9_y, result.sem9_y)
    dr_y_header = ["y_center", "DR_2S1S", "ERR_DR_2S1S", "DR_3S1S", "ERR_DR_3S1S", "DR_3S2S", "ERR_DR_3S2S"]
    dr_y_rows = np.column_stack([result.y_centers, dr_y[:, 0], dr_y_err[:, 0], dr_y[:, 1], dr_y_err[:, 1], dr_y[:, 2], dr_y_err[:, 2]])
    np.savetxt(outdir / f"oo5360_drvsy__{result.label}__central.csv", dr_y_rows, delimiter=",", header=",".join(dr_y_header), comments="", fmt="%.10g")

    dr_int, dr_int_err = _compute_dr_err(result.raa9_int, result.sem9_int)
    with open(outdir / f"oo5360_integrated_dr__{result.label}__central.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ratio", "value", "stat_err"])
        for ratio_name, idx in zip(("2S/1S", "3S/1S", "3S/2S"), (0, 1, 2)):
            writer.writerow([ratio_name, dr_int[idx], dr_int_err[idx]])


def _write_reference_band_csvs(
    results: Sequence[RunResult],
    ref_label: str,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ref_index = next(i for i, r in enumerate(results) if r.label == ref_label)

    pt_stack = np.stack([r.raa9_pt for r in results], axis=0)
    pt_sem_stack = np.stack([r.sem9_pt for r in results], axis=0)
    y_stack = np.stack([r.raa9_y for r in results], axis=0)
    y_sem_stack = np.stack([r.sem9_y for r in results], axis=0)
    int_stack = np.stack([r.raa9_int for r in results], axis=0)
    int_sem_stack = np.stack([r.sem9_int for r in results], axis=0)

    pt_center, pt_model_low, pt_model_high, pt_total_low, pt_total_high, pt_stat = _reference_band(pt_stack, pt_sem_stack, ref_index)
    y_center, y_model_low, y_model_high, y_total_low, y_total_high, y_stat = _reference_band(y_stack, y_sem_stack, ref_index)
    int_center, int_model_low, int_model_high, int_total_low, int_total_high, int_stat = _reference_band(int_stack, int_sem_stack, ref_index)

    header = ["x"]
    for name in MAIN_STATE_NAMES:
        header.extend(
            [
                f"center_{name}",
                f"model_min_{name}",
                f"model_max_{name}",
                f"total_min_{name}",
                f"total_max_{name}",
                f"stat_{name}",
            ]
        )

    pt_rows = [results[0].pt_centers]
    for idx in MAIN_STATE_INDICES:
        pt_rows.extend([pt_center[:, idx], pt_model_low[:, idx], pt_model_high[:, idx], pt_total_low[:, idx], pt_total_high[:, idx], pt_stat[:, idx]])
    np.savetxt(outdir / f"oo5360_raavspt__ref_{ref_label}.csv", np.column_stack(pt_rows), delimiter=",", header=",".join(header), comments="", fmt="%.10g")

    y_rows = [results[0].y_centers]
    for idx in MAIN_STATE_INDICES:
        y_rows.extend([y_center[:, idx], y_model_low[:, idx], y_model_high[:, idx], y_total_low[:, idx], y_total_high[:, idx], y_stat[:, idx]])
    np.savetxt(outdir / f"oo5360_raavsy__ref_{ref_label}.csv", np.column_stack(y_rows), delimiter=",", header=",".join(header), comments="", fmt="%.10g")

    with open(outdir / f"oo5360_integrated__ref_{ref_label}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "center", "model_min", "model_max", "total_min", "total_max", "stat"])
        for state_name, idx in zip(MAIN_STATE_NAMES, MAIN_STATE_INDICES):
            writer.writerow([state_name, int_center[idx], int_model_low[idx], int_model_high[idx], int_total_low[idx], int_total_high[idx], int_stat[idx]])

    dr_pt_stack = np.stack([_compute_dr_err(r.raa9_pt, r.sem9_pt)[0] for r in results], axis=0)
    dr_pt_err_stack = np.stack([_compute_dr_err(r.raa9_pt, r.sem9_pt)[1] for r in results], axis=0)
    dr_y_stack = np.stack([_compute_dr_err(r.raa9_y, r.sem9_y)[0] for r in results], axis=0)
    dr_y_err_stack = np.stack([_compute_dr_err(r.raa9_y, r.sem9_y)[1] for r in results], axis=0)
    dr_int_stack = np.stack([_compute_dr_err(r.raa9_int, r.sem9_int)[0] for r in results], axis=0)
    dr_int_err_stack = np.stack([_compute_dr_err(r.raa9_int, r.sem9_int)[1] for r in results], axis=0)

    dr_pt_center, dr_pt_model_low, dr_pt_model_high, dr_pt_total_low, dr_pt_total_high, dr_pt_stat = _reference_band(dr_pt_stack, dr_pt_err_stack, ref_index)
    dr_y_center, dr_y_model_low, dr_y_model_high, dr_y_total_low, dr_y_total_high, dr_y_stat = _reference_band(dr_y_stack, dr_y_err_stack, ref_index)
    dr_int_center, dr_int_model_low, dr_int_model_high, dr_int_total_low, dr_int_total_high, dr_int_stat = _reference_band(dr_int_stack, dr_int_err_stack, ref_index)

    dr_header = ["x"]
    for name in ("2S1S", "3S1S", "3S2S"):
        dr_header.extend(
            [
                f"center_DR_{name}",
                f"model_min_DR_{name}",
                f"model_max_DR_{name}",
                f"total_min_DR_{name}",
                f"total_max_DR_{name}",
                f"stat_DR_{name}",
            ]
        )

    dr_pt_rows = [results[0].pt_centers]
    for idx in (0, 1, 2):
        dr_pt_rows.extend([dr_pt_center[:, idx], dr_pt_model_low[:, idx], dr_pt_model_high[:, idx], dr_pt_total_low[:, idx], dr_pt_total_high[:, idx], dr_pt_stat[:, idx]])
    np.savetxt(outdir / f"oo5360_drvspt__ref_{ref_label}.csv", np.column_stack(dr_pt_rows), delimiter=",", header=",".join(dr_header), comments="", fmt="%.10g")

    dr_y_rows = [results[0].y_centers]
    for idx in (0, 1, 2):
        dr_y_rows.extend([dr_y_center[:, idx], dr_y_model_low[:, idx], dr_y_model_high[:, idx], dr_y_total_low[:, idx], dr_y_total_high[:, idx], dr_y_stat[:, idx]])
    np.savetxt(outdir / f"oo5360_drvsy__ref_{ref_label}.csv", np.column_stack(dr_y_rows), delimiter=",", header=",".join(dr_header), comments="", fmt="%.10g")

    with open(outdir / f"oo5360_integrated_dr__ref_{ref_label}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ratio", "center", "model_min", "model_max", "total_min", "total_max", "stat"])
        for ratio_name, idx in zip(("2S/1S", "3S/1S", "3S/2S"), (0, 1, 2)):
            writer.writerow([ratio_name, dr_int_center[idx], dr_int_model_low[idx], dr_int_model_high[idx], dr_int_total_low[idx], dr_int_total_high[idx], dr_int_stat[idx]])

    manifest = {
        "method": (
            f"Reference-kappa OO combination with central run {ref_label}. "
            "The central value is the reference run; the model band is the envelope from the non-reference runs; "
            "the total band combines the reference statistical SEM with the model spread in quadrature."
        ),
        "reference_label": ref_label,
        "runs": [
            {
                "label": r.label,
                "datafile": str(r.datafile),
                "load_mode": r.load_mode,
                "n_observables": r.n_observables,
            }
            for r in results
        ],
        "pt_edges": OO_PT_EDGES.tolist(),
        "y_edges": FINAL_Y_EDGES.tolist(),
        "integrated_y_window": list(OO_INTEGRATED_Y_WINDOW),
        "pt_max_for_y": OO_PT_MAX_FOR_Y,
    }
    (outdir / f"oo5360_manifest__ref_{ref_label}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_band_csvs(results: Sequence[RunResult], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    labels = [r.label for r in results]

    pt_stack = np.stack([r.raa9_pt for r in results], axis=0)
    pt_sem_stack = np.stack([r.sem9_pt for r in results], axis=0)
    y_stack = np.stack([r.raa9_y for r in results], axis=0)
    y_sem_stack = np.stack([r.sem9_y for r in results], axis=0)
    int_stack = np.stack([r.raa9_int for r in results], axis=0)
    int_sem_stack = np.stack([r.sem9_int for r in results], axis=0)

    pt_center, pt_low, pt_high = _envelope(pt_stack)
    y_center, y_low, y_high = _envelope(y_stack)
    int_center, int_low, int_high = _envelope(int_stack)
    pt_total_low, pt_total_high, pt_max_sem = _stat_augmented_envelope(pt_stack, pt_sem_stack)
    y_total_low, y_total_high, y_max_sem = _stat_augmented_envelope(y_stack, y_sem_stack)
    int_total_low, int_total_high, int_max_sem = _stat_augmented_envelope(int_stack, int_sem_stack)

    band_header = ["x"]
    for name in MAIN_STATE_NAMES:
        band_header.extend(
            [
                f"central_{name}",
                f"kappa_min_{name}",
                f"kappa_max_{name}",
                f"total_min_{name}",
                f"total_max_{name}",
                f"max_sem_{name}",
            ]
        )

    pt_rows = [result.pt_centers for result in results][0]
    pt_table = [pt_rows]
    for idx in MAIN_STATE_INDICES:
        pt_table.extend([
            pt_center[:, idx],
            pt_low[:, idx],
            pt_high[:, idx],
            pt_total_low[:, idx],
            pt_total_high[:, idx],
            pt_max_sem[:, idx],
        ])
    np.savetxt(
        outdir / "oo5360_raavspt__band.csv",
        np.column_stack(pt_table),
        delimiter=",",
        header=",".join(band_header),
        comments="",
        fmt="%.10g",
    )

    y_rows = [result.y_centers for result in results][0]
    y_table = [y_rows]
    for idx in MAIN_STATE_INDICES:
        y_table.extend([
            y_center[:, idx],
            y_low[:, idx],
            y_high[:, idx],
            y_total_low[:, idx],
            y_total_high[:, idx],
            y_max_sem[:, idx],
        ])
    np.savetxt(
        outdir / "oo5360_raavsy__band.csv",
        np.column_stack(y_table),
        delimiter=",",
        header=",".join(band_header),
        comments="",
        fmt="%.10g",
    )

    with open(outdir / "oo5360_integrated__band.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "central", "kappa_min", "kappa_max", "total_min", "total_max", "max_sem"])
        for state_name, idx in zip(MAIN_STATE_NAMES, MAIN_STATE_INDICES):
            writer.writerow([state_name, int_center[idx], int_low[idx], int_high[idx], int_total_low[idx], int_total_high[idx], int_max_sem[idx]])

    dr_pt_stack = np.stack([_compute_dr_err(r.raa9_pt, r.sem9_pt)[0] for r in results], axis=0)
    dr_pt_err_stack = np.stack([_compute_dr_err(r.raa9_pt, r.sem9_pt)[1] for r in results], axis=0)
    dr_y_stack = np.stack([_compute_dr_err(r.raa9_y, r.sem9_y)[0] for r in results], axis=0)
    dr_y_err_stack = np.stack([_compute_dr_err(r.raa9_y, r.sem9_y)[1] for r in results], axis=0)
    dr_int_stack = np.stack([_compute_dr_err(r.raa9_int, r.sem9_int)[0] for r in results], axis=0)
    dr_int_err_stack = np.stack([_compute_dr_err(r.raa9_int, r.sem9_int)[1] for r in results], axis=0)
    dr_pt_center, dr_pt_low, dr_pt_high = _envelope(dr_pt_stack)
    dr_y_center, dr_y_low, dr_y_high = _envelope(dr_y_stack)
    dr_int_center, dr_int_low, dr_int_high = _envelope(dr_int_stack)
    dr_pt_total_low, dr_pt_total_high, dr_pt_max_err = _stat_augmented_envelope(dr_pt_stack, dr_pt_err_stack)
    dr_y_total_low, dr_y_total_high, dr_y_max_err = _stat_augmented_envelope(dr_y_stack, dr_y_err_stack)
    dr_int_total_low, dr_int_total_high, dr_int_max_err = _stat_augmented_envelope(dr_int_stack, dr_int_err_stack)

    dr_band_header = ["x"]
    for name in ("2S1S", "3S1S", "3S2S"):
        dr_band_header.extend(
            [
                f"central_DR_{name}",
                f"kappa_min_DR_{name}",
                f"kappa_max_DR_{name}",
                f"total_min_DR_{name}",
                f"total_max_DR_{name}",
                f"max_err_DR_{name}",
            ]
        )

    dr_pt_table = [pt_rows]
    for idx in (0, 1, 2):
        dr_pt_table.extend([
            dr_pt_center[:, idx],
            dr_pt_low[:, idx],
            dr_pt_high[:, idx],
            dr_pt_total_low[:, idx],
            dr_pt_total_high[:, idx],
            dr_pt_max_err[:, idx],
        ])
    np.savetxt(outdir / "oo5360_drvspt__band.csv", np.column_stack(dr_pt_table), delimiter=",", header=",".join(dr_band_header), comments="", fmt="%.10g")

    dr_y_table = [y_rows]
    for idx in (0, 1, 2):
        dr_y_table.extend([
            dr_y_center[:, idx],
            dr_y_low[:, idx],
            dr_y_high[:, idx],
            dr_y_total_low[:, idx],
            dr_y_total_high[:, idx],
            dr_y_max_err[:, idx],
        ])
    np.savetxt(outdir / "oo5360_drvsy__band.csv", np.column_stack(dr_y_table), delimiter=",", header=",".join(dr_band_header), comments="", fmt="%.10g")

    with open(outdir / "oo5360_integrated_dr__band.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ratio", "central", "kappa_min", "kappa_max", "total_min", "total_max", "max_err"])
        for ratio_name, idx in zip(("2S/1S", "3S/1S", "3S/2S"), (0, 1, 2)):
            writer.writerow([ratio_name, dr_int_center[idx], dr_int_low[idx], dr_int_high[idx], dr_int_total_low[idx], dr_int_total_high[idx], dr_int_max_err[idx]])

    manifest = {
        "method": (
            "Mathematica-consistent OO fixed-b central-only envelope. "
            "Each run is reduced to a central RAA curve with propagated trajectory SEM. "
            "The pure-kappa band is the pointwise min/max envelope across run central values, "
            "the reported 'central' is 0.5*(kappa_min+kappa_max), and the total band is "
            "[min_i(center_i-sem_i), max_i(center_i+sem_i)]."
        ),
        "runs": [
            {
                "label": r.label,
                "datafile": str(r.datafile),
                "load_mode": r.load_mode,
                "n_observables": r.n_observables,
            }
            for r in results
        ],
        "pt_edges": OO_PT_EDGES.tolist(),
        "y_edges": FINAL_Y_EDGES.tolist(),
        "integrated_y_window": list(OO_INTEGRATED_Y_WINDOW),
        "pt_max_for_y": OO_PT_MAX_FOR_Y,
        "band_labels": labels,
    }
    (outdir / "oo5360_band_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _print_summary(results: Sequence[RunResult]) -> None:
    print("\n" + "=" * 96)
    print("OO 5.36 TeV QTraj-NLO Central Comparison")
    print("=" * 96)
    print(f"{'run':<18} {'loader':<28} {'RAA(1S)':>14} {'RAA(2S)':>14} {'RAA(3S)':>14} {'n_obs':>12}")
    print("-" * 96)
    for result in results:
        print(
            f"{result.label:<18} {result.load_mode:<28} {result.raa9_int[0]:>14.6f} "
            f"{result.raa9_int[1]:>14.6f} {result.raa9_int[5]:>14.6f} {result.n_observables:>12d}"
        )
    if len(results) >= 2:
        stack = np.stack([r.raa9_int for r in results], axis=0)
        mid, low, high = _envelope(stack)
        print("-" * 96)
        print(f"{'band central':<18} {mid[0]:>14.6f} {mid[1]:>14.6f} {mid[5]:>14.6f}")
        print(f"{'band min':<18} {low[0]:>14.6f} {low[1]:>14.6f} {low[5]:>14.6f}")
        print(f"{'band max':<18} {high[0]:>14.6f} {high[1]:>14.6f} {high[5]:>14.6f}")

        # Double Ratio summary
        dr_int_stack = np.stack([_compute_dr(r.raa9_int) for r in results], axis=0)
        dr_mid, dr_low, dr_high = _envelope(dr_int_stack)
        print("-" * 96)
        print(f"{'DR 2S/1S central':<18} {dr_mid[0]:>14.6f}")
        print(f"{'DR 2S/1S min':<18} {dr_low[0]:>14.6f}")
        print(f"{'DR 2S/1S max':<18} {dr_high[0]:>14.6f}")
        print(f"{'DR 3S/1S central':<18} {dr_mid[1]:>14.6f}")
        print(f"{'DR 3S/1S min':<18} {dr_low[1]:>14.6f}")
        print(f"{'DR 3S/1S max':<18} {dr_high[1]:>14.6f}")
        print(f"{'DR 3S/2S central':<18} {dr_mid[2]:>14.6f}")
        print(f"{'DR 3S/2S min':<18} {dr_low[2]:>14.6f}")
        print(f"{'DR 3S/2S max':<18} {dr_high[2]:>14.6f}")

    print("=" * 96 + "\n")


def _print_reference_summary(results: Sequence[RunResult], ref_label: str) -> None:
    ref_index = next(i for i, r in enumerate(results) if r.label == ref_label)
    ref = results[ref_index]
    print("\n" + "=" * 96)
    print(f"OO 5.36 TeV QTraj-NLO Reference Combination (central={ref_label})")
    print("=" * 96)
    print(f"{'run':<18} {'loader':<28} {'RAA(1S)':>14} {'RAA(2S)':>14} {'RAA(3S)':>14} {'n_obs':>12}")
    print("-" * 96)
    for result in results:
        print(
            f"{result.label:<18} {result.load_mode:<28} {result.raa9_int[0]:>14.6f} "
            f"{result.raa9_int[1]:>14.6f} {result.raa9_int[5]:>14.6f} {result.n_observables:>12d}"
        )

    stack = np.stack([r.raa9_int for r in results], axis=0)
    sem_stack = np.stack([r.sem9_int for r in results], axis=0)
    center, model_low, model_high, total_low, total_high, stat = _reference_band(stack, sem_stack, ref_index)
    print("-" * 96)
    print(f"{'central':<18} {center[0]:>14.6f} {center[1]:>14.6f} {center[5]:>14.6f}")
    print(f"{'model min':<18} {model_low[0]:>14.6f} {model_low[1]:>14.6f} {model_low[5]:>14.6f}")
    print(f"{'model max':<18} {model_high[0]:>14.6f} {model_high[1]:>14.6f} {model_high[5]:>14.6f}")
    print(f"{'total min':<18} {total_low[0]:>14.6f} {total_low[1]:>14.6f} {total_low[5]:>14.6f}")
    print(f"{'total max':<18} {total_high[0]:>14.6f} {total_high[1]:>14.6f} {total_high[5]:>14.6f}")
    print(f"{'stat only':<18} {stat[0]:>14.6f} {stat[1]:>14.6f} {stat[5]:>14.6f}")

    dr_stack = np.stack([_compute_dr_err(r.raa9_int, r.sem9_int)[0] for r in results], axis=0)
    dr_err_stack = np.stack([_compute_dr_err(r.raa9_int, r.sem9_int)[1] for r in results], axis=0)
    dr_center, dr_model_low, dr_model_high, dr_total_low, dr_total_high, dr_stat = _reference_band(dr_stack, dr_err_stack, ref_index)
    print("-" * 96)
    for ratio_name, idx in zip(("2S/1S", "3S/1S", "3S/2S"), (0, 1, 2)):
        print(f"{ratio_name + ' center':<18} {dr_center[idx]:>14.6f}")
        print(f"{ratio_name + ' model min':<18} {dr_model_low[idx]:>14.6f}")
        print(f"{ratio_name + ' model max':<18} {dr_model_high[idx]:>14.6f}")
        print(f"{ratio_name + ' total min':<18} {dr_total_low[idx]:>14.6f}")
        print(f"{ratio_name + ' total max':<18} {dr_total_high[idx]:>14.6f}")
        print(f"{ratio_name + ' stat':<18} {dr_stat[idx]:>14.6f}")
    print("=" * 96 + "\n")


def _resolve_run_specs(args: argparse.Namespace) -> List[RunSpec]:
    if args.run:
        specs = list(args.run)
    else:
        kappas = args.kappas or [5, 7]
        specs = [
            RunSpec(label=f"kappa{kappa}", datafile=_default_oo_wreg_path(kappa))
            for kappa in kappas
        ]
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kappas",
        nargs="+",
        type=int,
        help="OO wReg kappa values to load from inputs/qtraj_inputs/OxygenOxygen5360 (default: 5 7).",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run_arg,
        help="Explicit LABEL=PATH run specification. Can be repeated. Overrides --kappas.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write only CSV/JSON outputs and skip publication-style figure generation.",
    )
    parser.add_argument(
        "--reference-label",
        type=str,
        default="",
        help="If set, use this run as the central kappa and combine its stat error with the model envelope from the other runs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("compare_oo_kappa")

    specs = _resolve_run_specs(args)
    results: List[RunResult] = []
    for spec in specs:
        results.append(analyze_run(spec, logger))

    for result in results:
        _write_per_run_csv(result, args.output_dir)
    if args.reference_label:
        _result_by_label(results, args.reference_label)
        if len(results) < 2:
            raise ValueError("--reference-label requires at least two runs.")
        _write_reference_band_csvs(results, args.reference_label, args.output_dir)
        if not args.skip_plots:
            _plot_reference_publication(results, args.reference_label, args.output_dir)
        _print_reference_summary(results, args.reference_label)
    elif len(results) >= 2:
        _write_band_csvs(results, args.output_dir)
        if not args.skip_plots:
            _plot_publication(results, args.output_dir)
        _print_summary(results)
    elif len(results) == 1 and not args.skip_plots:
        _plot_single_run_publication(results[0], args.output_dir)
        _print_summary(results)
    else:
        _print_summary(results)
    logger.info("Wrote central comparison outputs to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
