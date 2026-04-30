#!/usr/bin/env python3
"""
OO 5.36 TeV bottomonia minimum-bias package:
QTraj HNM (kappa=5,7) + OO CNM + bin-by-bin combination.

This script computes the QTraj hot-medium branch directly from the bundled OO
`kap5` / `kap7` inputs, computes the OO minimum-bias CNM branch from the
existing CNM machinery, and combines the two with asymmetric relative errors
added in quadrature.
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
from typing import Dict, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
QTRAJ_PACKAGE_ROOT = REPO_ROOT / "hnm" / "qtraj_out_analysis"
CNM_SCRIPT_ROOT = REPO_ROOT / "cnm" / "cnm_scripts"

if str(QTRAJ_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(QTRAJ_PACKAGE_ROOT))

from qtraj_analysis.feeddown import (  # noqa: E402
    apply_feeddown_to_raa6,
    build_feeddown_matrix,
    solve_primordial_sigmas,
)
from qtraj_analysis.io import load_qtraj_table, parse_records  # noqa: E402
from qtraj_analysis.kinematics_presets import SIGMAS_EXP_OO_5360  # noqa: E402
from qtraj_analysis.matching import build_observables  # noqa: E402

if str(CNM_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(CNM_SCRIPT_ROOT))

import run_bottomonia_cnm_OO as cnm_oo  # noqa: E402


MAIN_STATE_INDICES: Tuple[int, ...] = (0, 1, 5)
MAIN_STATE_NAMES: Tuple[str, ...] = ("1S", "2S", "3S")
MAIN_STATE_TEX: Tuple[str, ...] = (
    r"$\Upsilon(1S)$",
    r"$\Upsilon(2S)$",
    r"$\Upsilon(3S)$",
)
WINDOW_KEYS: Tuple[str, ...] = ("backward", "midrapidity", "forward")

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "cnm_hnm" / "min_bias" / "OO_5p36TeV" / "qtraj_kappa57"
)
CMS_PRELIM_DOUBLE_RATIO_CSV = (
    REPO_ROOT
    / "inputs"
    / "experimental_data"
    / "lhc"
    / "OxygenOxygen5p36TeV"
    / "cms_preliminary_approx_upsilon_double_ratios.csv"
)
BLUE = "#1f77b4"
Y_EXPORT_MIN = -4.5
Y_EXPORT_MAX = 4.5


@dataclass(frozen=True)
class WindowSpec:
    key: str
    y_window: Tuple[float, float]
    label: str
    pt_edges: np.ndarray
    pt_bin_width: float


DEFAULT_PT_BIN_WIDTHS: Dict[str, float] = {
    "backward": 2.0,
    "midrapidity": 1.0,
    "forward": 2.0,
}

DEFAULT_PT_EDGE_OVERRIDES: Dict[str, np.ndarray] = {
    # Publication-oriented tail binning: keep substructure up to 14 GeV and
    # merge only the sparsest high-pT bins. Use the same binning in the
    # forward/backward arms so the symmetric O+O comparison is clean.
    "backward": np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 18.0, 20.0], dtype=np.float64),
    "forward": np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 18.0, 20.0], dtype=np.float64),
}


DEFAULT_Y_WINDOWS: Dict[str, Tuple[float, float, str]] = {
    # Mirror the publication-safe CNM acceptance: avoid the sparse |y|>4 edge
    # while keeping the forward/backward comparison symmetric in O+O.
    "backward":    (-4.0, -2.5, r"$-4.0 < y < -2.5$"),
    "midrapidity": (-2.4,  2.4, r"$-2.4 < y < 2.4$"),
    "forward":     ( 2.5,  4.0, r"$2.5 < y < 4.0$"),
}


DEFAULT_Y_PT_MIN: float = 0.0
DEFAULT_Y_PT_MAX: float = float(cnm_oo.PT_RANGE_AVG[1])
DEFAULT_PT_PLOT_MIN: float = 0.0
# Plot-only fractional half-band cap for sparse high-pT tails.
# This never changes numerical outputs in CSV; it only limits rendered band size.
DEFAULT_PT_MAX_REL_BAND_FOR_PLOT: float | None = 0.60
DEFAULT_INTEGRATED_Y_MIN: float = -2.4
DEFAULT_INTEGRATED_Y_MAX: float = 2.4
DEFAULT_INTEGRATED_PT_MIN: float = 1.0
DEFAULT_INTEGRATED_PT_MAX: float = 30.0
CNM_PT_MAX: float = float(cnm_oo.PT_RANGE_AVG[1])


@dataclass(frozen=True)
class SpectralResult:
    centers: np.ndarray
    raa9: np.ndarray
    sem9: np.ndarray


@dataclass(frozen=True)
class RunResult:
    label: str
    datafile: Path
    load_mode: str
    n_observables: int
    b_values: Tuple[float, ...]
    y_result: SpectralResult
    pt_results: Dict[str, SpectralResult]


@dataclass(frozen=True)
class HNMBand:
    edges: np.ndarray
    centers: np.ndarray
    kap5: np.ndarray
    kap7: np.ndarray
    central: np.ndarray
    low: np.ndarray
    high: np.ndarray


@dataclass(frozen=True)
class CNMBand:
    edges: np.ndarray
    centers: np.ndarray
    central: np.ndarray
    low: np.ndarray
    high: np.ndarray


@dataclass(frozen=True)
class CombinedBand:
    edges: np.ndarray
    centers: np.ndarray
    central: np.ndarray
    low: np.ndarray
    high: np.ndarray


def _default_oo_wreg_path(kappa: int) -> Path:
    base = (
        REPO_ROOT
        / "inputs"
        / "qtraj_inputs"
        / "OxygenOxygen5360"
        / f"qtraj-nlo-run2-00-5.36-kap{kappa}-wReg"
    )
    raw = base / "datafile.gz"
    avg = base / "datafile-avg.gz"
    if raw.exists():
        return raw
    return avg


def _make_pt_edges(bin_width: float) -> np.ndarray:
    pt_min = float(cnm_oo.PT_RANGE_AVG[0])
    pt_max = float(cnm_oo.PT_RANGE_AVG[1])
    bw = float(bin_width)
    if bw <= 0.0:
        raise ValueError(f"pt bin width must be positive, got {bw}")
    n_bins = int(round((pt_max - pt_min) / bw))
    if n_bins <= 0:
        raise ValueError(f"pt bin width {bw} produces no bins on [{pt_min},{pt_max}]")
    return np.linspace(pt_min, pt_min + n_bins * bw, n_bins + 1)


def _window_specs(
    pt_bin_widths: Dict[str, float] | None = None,
    y_windows: Dict[str, Tuple[float, float, str]] | None = None,
) -> Tuple[WindowSpec, ...]:
    widths = dict(DEFAULT_PT_BIN_WIDTHS)
    if pt_bin_widths:
        widths.update({k: float(v) for k, v in pt_bin_widths.items()})
    yw = {k: tuple(v) for k, v in DEFAULT_Y_WINDOWS.items()}
    if y_windows:
        yw.update({k: tuple(v) for k, v in y_windows.items()})
    specs = []
    for key in WINDOW_KEYS:
        y0, y1, label = yw[key]
        bw = float(widths[key])
        pt_edges = np.asarray(
            DEFAULT_PT_EDGE_OVERRIDES.get(key, _make_pt_edges(bw)),
            dtype=np.float64,
        )
        specs.append(
            WindowSpec(
                key=key,
                y_window=(float(y0), float(y1)),
                label=str(label),
                pt_edges=pt_edges,
                pt_bin_width=bw,
            )
        )
    return tuple(specs)


def _warn_if_custom_edges_override_bin_widths(
    pt_bin_widths: Dict[str, float],
    logger: logging.Logger,
) -> None:
    for key, edges in DEFAULT_PT_EDGE_OVERRIDES.items():
        requested = float(pt_bin_widths[key])
        default = float(DEFAULT_PT_BIN_WIDTHS[key])
        if abs(requested - default) > 1e-12:
            logger.warning(
                "%s pT bin width %.3f GeV ignored because custom edges %s are active.",
                key,
                requested,
                np.asarray(edges, dtype=np.float64).tolist(),
            )


def _resolve_path(pathlike: Path | str) -> Path:
    path = Path(pathlike)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_observables(datafile: Path, logger: logging.Logger):
    rows = load_qtraj_table(str(datafile), logger)
    records = parse_records(rows, logger)
    obs = build_observables(records, logger)
    mode = "raw_input" if datafile.name == "datafile.gz" else "avg_input"
    return obs, mode


def _weighted_mean_sem_surv6(obs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    if not obs:
        nan = np.full(6, np.nan, dtype=np.float64)
        return nan, nan

    values = np.vstack([entry.surv6 for entry in obs]).astype(np.float64)
    qweights = np.asarray([entry.qweight for entry in obs], dtype=np.float64)
    if (not np.isfinite(qweights).all()) or float(np.sum(qweights)) <= 0.0:
        qweights = np.ones(values.shape[0], dtype=np.float64)

    qsum = float(np.sum(qweights))
    mean = (values.T @ qweights) / qsum
    if values.shape[0] <= 1:
        sem = np.zeros(6, dtype=np.float64)
    else:
        var = (qweights[:, None] * (values - mean) ** 2).sum(axis=0) / qsum
        neff = (qsum * qsum) / float(np.sum(qweights * qweights))
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


def _compute_vs_y(
    obs: Sequence,
    y_edges: np.ndarray,
    pt_max: float,
    feeddown: np.ndarray,
    sigmas_prim: np.ndarray,
    pt_min: float = 0.0,
) -> SpectralResult:
    centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    raa9 = np.full((centers.shape[0], 9), np.nan, dtype=np.float64)
    sem9 = np.full((centers.shape[0], 9), np.nan, dtype=np.float64)
    selected = [
        entry for entry in obs
        if (entry.pt >= float(pt_min)) and (entry.pt <= float(pt_max))
    ]
    for index in range(len(y_edges) - 1):
        y0 = float(y_edges[index])
        y1 = float(y_edges[index + 1])
        bin_obs = [entry for entry in selected if y0 <= entry.y < y1]
        mean6, sem6 = _weighted_mean_sem_surv6(bin_obs)
        raa9[index], sem9[index] = _raa9_with_sem(mean6, sem6, feeddown, sigmas_prim)
    return SpectralResult(centers=centers, raa9=raa9, sem9=sem9)


def _compute_vs_pt(
    obs: Sequence,
    pt_edges: np.ndarray,
    y_window: Tuple[float, float],
    feeddown: np.ndarray,
    sigmas_prim: np.ndarray,
) -> SpectralResult:
    centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    raa9 = np.full((centers.shape[0], 9), np.nan, dtype=np.float64)
    sem9 = np.full((centers.shape[0], 9), np.nan, dtype=np.float64)
    y0, y1 = y_window
    selected = [entry for entry in obs if y0 <= entry.y <= y1]
    for index in range(len(pt_edges) - 1):
        p0 = float(pt_edges[index])
        p1 = float(pt_edges[index + 1])
        bin_obs = [entry for entry in selected if p0 <= entry.pt < p1]
        mean6, sem6 = _weighted_mean_sem_surv6(bin_obs)
        raa9[index], sem9[index] = _raa9_with_sem(mean6, sem6, feeddown, sigmas_prim)
    return SpectralResult(centers=centers, raa9=raa9, sem9=sem9)


def _analyze_qtraj_run(
    spec_label: str,
    datafile: Path,
    logger: logging.Logger,
    specs: Tuple[WindowSpec, ...],
    y_pt_range: Tuple[float, float],
) -> RunResult:
    path = _resolve_path(datafile)
    if not path.exists():
        raise FileNotFoundError(path)

    logger.info("Loading QTraj %s from %s", spec_label, path)
    obs, load_mode = _load_observables(path, logger)
    b_values = tuple(
        float(val) for val in np.unique(np.round(np.array([entry.b for entry in obs], dtype=np.float64), 6))
    )

    feeddown = build_feeddown_matrix()
    sigmas_prim = solve_primordial_sigmas(feeddown, SIGMAS_EXP_OO_5360)

    y_result = _compute_vs_y(
        obs,
        np.asarray(cnm_oo.Y_EDGES, dtype=np.float64),
        float(y_pt_range[1]),
        feeddown,
        sigmas_prim,
        pt_min=float(y_pt_range[0]),
    )
    pt_results: Dict[str, SpectralResult] = {}
    for spec in specs:
        pt_results[spec.key] = _compute_vs_pt(
            obs,
            np.asarray(spec.pt_edges, dtype=np.float64),
            spec.y_window,
            feeddown,
            sigmas_prim,
        )

    logger.info(
        "%s: mode=%s nobs=%d",
        spec_label,
        load_mode,
        len(obs),
    )
    return RunResult(
        label=spec_label,
        datafile=path,
        load_mode=load_mode,
        n_observables=len(obs),
        b_values=b_values,
        y_result=y_result,
        pt_results=pt_results,
    )


def _envelope(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    high = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    mid = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    for idx in np.ndindex(stack.shape[1:]):
        values = stack[(slice(None),) + idx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        low[idx] = float(np.min(values))
        high[idx] = float(np.max(values))
        mid[idx] = 0.5 * (low[idx] + high[idx])
    return mid, low, high


def _stat_augmented_envelope(
    center_stack: np.ndarray,
    sem_stack: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    low = np.full(center_stack.shape[1:], np.nan, dtype=np.float64)
    high = np.full(center_stack.shape[1:], np.nan, dtype=np.float64)
    for idx in np.ndindex(center_stack.shape[1:]):
        centers = center_stack[(slice(None),) + idx]
        sems = sem_stack[(slice(None),) + idx]
        mask = np.isfinite(centers)
        if not np.any(mask):
            continue
        sems = np.where(np.isfinite(sems), sems, 0.0)
        low[idx] = float(np.min(centers[mask] - sems[mask]))
        high[idx] = float(np.max(centers[mask] + sems[mask]))
    return low, high


def _build_hnm_band(
    edges: np.ndarray,
    run5_result: SpectralResult,
    run7_result: SpectralResult,
) -> HNMBand:
    kap5 = np.asarray(run5_result.raa9[:, MAIN_STATE_INDICES], dtype=np.float64)
    kap7 = np.asarray(run7_result.raa9[:, MAIN_STATE_INDICES], dtype=np.float64)

    center_stack = np.stack([kap5, kap7], axis=0)
    central, low, high = _envelope(center_stack)

    return HNMBand(
        edges=np.asarray(edges, dtype=np.float64),
        centers=np.asarray(run5_result.centers, dtype=np.float64),
        kap5=kap5,
        kap7=kap7,
        central=central,
        low=low,
        high=high,
    )


def _compute_cnm_bands(
    logger: logging.Logger,
    specs: Tuple[WindowSpec, ...],
    y_pt_range: Tuple[float, float],
) -> Tuple[CNMBand, Dict[str, CNMBand], object]:
    logger.info("Building OO CNM minimum-bias context from run_bottomonia_cnm_OO.py")
    cnm = cnm_oo.build_eloss_context("5.36")
    y_centers, _tags_y, bands_y = cnm.cnm_vs_y(
        y_edges=cnm_oo.Y_EDGES,
        pt_range_avg=(float(y_pt_range[0]), float(y_pt_range[1])),
        components=("cnm",),
        include_mb=True,
    )
    cnm_oo.pin_edge_bins(bands_y, np.asarray(y_centers, dtype=np.float64))
    y_band = CNMBand(
        edges=np.asarray(cnm_oo.Y_EDGES, dtype=np.float64),
        centers=np.asarray(y_centers, dtype=np.float64),
        central=np.asarray(bands_y["cnm"][0]["MB"], dtype=np.float64),
        low=np.asarray(bands_y["cnm"][1]["MB"], dtype=np.float64),
        high=np.asarray(bands_y["cnm"][2]["MB"], dtype=np.float64),
    )

    pt_bands: Dict[str, CNMBand] = {}
    for spec in specs:
        pt_centers, _tags_pt, bands_pt = cnm.cnm_vs_pT(
            spec.y_window,
            np.asarray(spec.pt_edges, dtype=np.float64),
            components=("cnm",),
            include_mb=True,
        )
        pt_bands[spec.key] = CNMBand(
            edges=np.asarray(spec.pt_edges, dtype=np.float64),
            centers=np.asarray(pt_centers, dtype=np.float64),
            central=np.asarray(bands_pt["cnm"][0]["MB"], dtype=np.float64),
            low=np.asarray(bands_pt["cnm"][1]["MB"], dtype=np.float64),
            high=np.asarray(bands_pt["cnm"][2]["MB"], dtype=np.float64),
        )
    return y_band, pt_bands, cnm


def _assert_aligned(label: str, centers_a: np.ndarray, centers_b: np.ndarray) -> None:
    if centers_a.shape != centers_b.shape or not np.allclose(centers_a, centers_b, atol=1e-9, rtol=0.0):
        raise ValueError(f"{label} centers are not aligned between HNM and CNM")


def _combine_two_bands_1d(
    r1_c: np.ndarray,
    r1_lo: np.ndarray,
    r1_hi: np.ndarray,
    r2_c: np.ndarray,
    r2_lo: np.ndarray,
    r2_hi: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1_c = np.asarray(r1_c, dtype=np.float64)
    r1_lo = np.asarray(r1_lo, dtype=np.float64)
    r1_hi = np.asarray(r1_hi, dtype=np.float64)
    r2_c = np.asarray(r2_c, dtype=np.float64)
    r2_lo = np.asarray(r2_lo, dtype=np.float64)
    r2_hi = np.asarray(r2_hi, dtype=np.float64)

    combined = np.full_like(r1_c, np.nan, dtype=np.float64)
    low = np.full_like(r1_c, np.nan, dtype=np.float64)
    high = np.full_like(r1_c, np.nan, dtype=np.float64)

    mask = np.isfinite(r1_c) & np.isfinite(r2_c)
    if not np.any(mask):
        return combined, low, high

    r1_safe = np.where(np.abs(r1_c) > eps, r1_c, np.nan)
    r2_safe = np.where(np.abs(r2_c) > eps, r2_c, np.nan)

    rel_lo_1 = (r1_c - r1_lo) / r1_safe
    rel_lo_2 = (r2_c - r2_lo) / r2_safe
    rel_hi_1 = (r1_hi - r1_c) / r1_safe
    rel_hi_2 = (r2_hi - r2_c) / r2_safe

    rel_lo = np.sqrt(np.maximum(0.0, rel_lo_1**2 + rel_lo_2**2))
    rel_hi = np.sqrt(np.maximum(0.0, rel_hi_1**2 + rel_hi_2**2))
    product = r1_c * r2_c

    combined[mask] = product[mask]
    low[mask] = product[mask] * (1.0 - rel_lo[mask])
    high[mask] = product[mask] * (1.0 + rel_hi[mask])
    return combined, low, high


def _build_combined_band(hnm_band: HNMBand, cnm_band: CNMBand) -> CombinedBand:
    _assert_aligned("combined", hnm_band.centers, cnm_band.centers)

    central = np.full_like(hnm_band.central, np.nan, dtype=np.float64)
    low = np.full_like(hnm_band.low, np.nan, dtype=np.float64)
    high = np.full_like(hnm_band.high, np.nan, dtype=np.float64)

    for state_index in range(len(MAIN_STATE_NAMES)):
        central[:, state_index], low[:, state_index], high[:, state_index] = _combine_two_bands_1d(
            hnm_band.central[:, state_index],
            hnm_band.low[:, state_index],
            hnm_band.high[:, state_index],
            cnm_band.central,
            cnm_band.low,
            cnm_band.high,
        )

    return CombinedBand(
        edges=np.asarray(hnm_band.edges, dtype=np.float64),
        centers=np.asarray(hnm_band.centers, dtype=np.float64),
        central=central,
        low=low,
        high=high,
    )


def _masked_bin_edges(edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.asarray(edges[indices[0] : indices[-1] + 2], dtype=np.float64)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _band_rows_hnm(axis_name: str, band: HNMBand, mask: np.ndarray | None = None) -> Tuple[Sequence[str], Sequence[Dict[str, object]]]:
    active_mask = np.ones(band.centers.shape[0], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    rows = []
    for index in np.flatnonzero(active_mask):
        row: Dict[str, object] = {
            f"{axis_name}_low": float(band.edges[index]),
            f"{axis_name}_high": float(band.edges[index + 1]),
            f"{axis_name}_center": float(band.centers[index]),
        }
        for state_index, state_name in enumerate(MAIN_STATE_NAMES):
            row[f"kap5_{state_name}"] = float(band.kap5[index, state_index])
            row[f"kap7_{state_name}"] = float(band.kap7[index, state_index])
            row[f"hnm_central_{state_name}"] = float(band.central[index, state_index])
            row[f"hnm_band_lo_{state_name}"] = float(band.low[index, state_index])
            row[f"hnm_band_hi_{state_name}"] = float(band.high[index, state_index])
        rows.append(row)

    fields = [f"{axis_name}_low", f"{axis_name}_high", f"{axis_name}_center"]
    for state_name in MAIN_STATE_NAMES:
        fields.extend(
            [
                f"kap5_{state_name}",
                f"kap7_{state_name}",
                f"hnm_central_{state_name}",
                f"hnm_band_lo_{state_name}",
                f"hnm_band_hi_{state_name}",
            ]
        )
    return fields, rows


def _band_rows_combined(
    axis_name: str,
    hnm_band: HNMBand,
    cnm_band: CNMBand,
    combined_band: CombinedBand,
    mask: np.ndarray | None = None,
) -> Tuple[Sequence[str], Sequence[Dict[str, object]]]:
    active_mask = np.ones(hnm_band.centers.shape[0], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    rows = []
    for index in np.flatnonzero(active_mask):
        row: Dict[str, object] = {
            f"{axis_name}_low": float(hnm_band.edges[index]),
            f"{axis_name}_high": float(hnm_band.edges[index + 1]),
            f"{axis_name}_center": float(hnm_band.centers[index]),
        }
        for state_index, state_name in enumerate(MAIN_STATE_NAMES):
            row[f"kap5_{state_name}"] = float(hnm_band.kap5[index, state_index])
            row[f"kap7_{state_name}"] = float(hnm_band.kap7[index, state_index])
            row[f"hnm_central_{state_name}"] = float(hnm_band.central[index, state_index])
            row[f"hnm_band_lo_{state_name}"] = float(hnm_band.low[index, state_index])
            row[f"hnm_band_hi_{state_name}"] = float(hnm_band.high[index, state_index])
            row[f"cnm_central_{state_name}"] = float(cnm_band.central[index])
            row[f"cnm_band_lo_{state_name}"] = float(cnm_band.low[index])
            row[f"cnm_band_hi_{state_name}"] = float(cnm_band.high[index])
            row[f"combined_central_{state_name}"] = float(combined_band.central[index, state_index])
            row[f"combined_band_lo_{state_name}"] = float(combined_band.low[index, state_index])
            row[f"combined_band_hi_{state_name}"] = float(combined_band.high[index, state_index])
        rows.append(row)

    fields = [f"{axis_name}_low", f"{axis_name}_high", f"{axis_name}_center"]
    for state_name in MAIN_STATE_NAMES:
        fields.extend(
            [
                f"kap5_{state_name}",
                f"kap7_{state_name}",
                f"hnm_central_{state_name}",
                f"hnm_band_lo_{state_name}",
                f"hnm_band_hi_{state_name}",
                f"cnm_central_{state_name}",
                f"cnm_band_lo_{state_name}",
                f"cnm_band_hi_{state_name}",
                f"combined_central_{state_name}",
                f"combined_band_lo_{state_name}",
                f"combined_band_hi_{state_name}",
            ]
        )
    return fields, rows


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


def _contiguous_spans(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    spans = []
    start = int(indices[0])
    previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            spans.append((start, previous))
            start = current
        previous = current
    spans.append((start, previous))
    return spans


def _step_band_asymmetric(
    ax,
    bin_edges: np.ndarray,
    y_mid: np.ndarray,
    y_low: np.ndarray,
    y_high: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    label: str | None,
) -> None:
    y_mid = np.asarray(y_mid, dtype=np.float64)
    y_low = np.asarray(y_low, dtype=np.float64)
    y_high = np.asarray(y_high, dtype=np.float64)
    mask = np.isfinite(y_mid) & np.isfinite(y_low) & np.isfinite(y_high)
    for span_index, (start, stop) in enumerate(_contiguous_spans(mask)):
        edges = np.asarray(bin_edges, dtype=np.float64)[start : stop + 2]
        mid = y_mid[start : stop + 1]
        low = y_low[start : stop + 1]
        high = y_high[start : stop + 1]
        x = edges
        y_step = np.concatenate([mid, [mid[-1]]])
        y_low_step = np.concatenate([low, [low[-1]]])
        y_high_step = np.concatenate([high, [high[-1]]])
        ax.fill_between(
            x,
            y_low_step,
            y_high_step,
            step="post",
            color=color,
            alpha=alpha,
            linewidth=0.0,
            zorder=1 if linestyle == "--" else 2,
        )
        ax.step(
            x,
            y_step,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label if span_index == 0 else None,
            zorder=3 if linestyle == "--" else 4,
        )


def _compute_ylim(*arrays: np.ndarray) -> Tuple[float, float]:
    ymax = 0.0
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            ymax = max(ymax, float(np.max(finite)))
    upper = max(0.9, min(1.25, 1.12 * ymax + 0.05))
    return 0.0, upper


def _masked_copy(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    bool_mask = np.broadcast_to(np.asarray(mask, dtype=bool), arr.shape)
    return np.where(bool_mask, arr, np.nan)


def _cap_band_for_plot(
    central: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    max_rel_band: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap asymmetric half-bands for plotting only; keep central values fixed."""
    c = np.asarray(central, dtype=np.float64).copy()
    lo = np.asarray(low, dtype=np.float64).copy()
    hi = np.asarray(high, dtype=np.float64).copy()
    if max_rel_band is None:
        return c, lo, hi
    cap = float(max_rel_band)
    if cap <= 0.0:
        return c, lo, hi
    safe = np.where(np.abs(c) > 1e-12, np.abs(c), np.nan)
    half_lo = np.maximum(0.0, c - lo)
    half_hi = np.maximum(0.0, hi - c)
    cap_abs = cap * safe
    lo = c - np.minimum(half_lo, cap_abs)
    hi = c + np.minimum(half_hi, cap_abs)
    return c, lo, hi


def _is_uniform_edges(edges: np.ndarray, atol: float = 1e-9) -> bool:
    diffs = np.diff(np.asarray(edges, dtype=np.float64))
    return bool(diffs.size == 0 or np.allclose(diffs, diffs[0], atol=atol, rtol=0.0))


def _plot_stability_mask(
    hnm_band: HNMBand,
    *,
    pt_min: float,
    max_rel_band: float | None,
    state_index: int = 0,
) -> np.ndarray:
    mask = np.asarray(hnm_band.centers >= float(pt_min), dtype=bool)
    if max_rel_band is None:
        return mask

    central = np.asarray(hnm_band.central[:, state_index], dtype=np.float64)
    low = np.asarray(hnm_band.low[:, state_index], dtype=np.float64)
    high = np.asarray(hnm_band.high[:, state_index], dtype=np.float64)
    safe = np.where(np.abs(central) > 1e-12, np.abs(central), np.nan)
    frac = np.maximum((central - low) / safe, (high - central) / safe)
    stable = np.isfinite(frac) & (frac <= float(max_rel_band))
    return mask & stable


def _ratio_plot_stability_mask(
    band: RatioBand,
    *,
    pt_min: float,
    max_rel_band: float | None,
    ratio_index: int = 0,
) -> np.ndarray:
    mask = np.asarray(band.centers >= float(pt_min), dtype=bool)
    if max_rel_band is None:
        return mask

    central = np.asarray(band.central[:, ratio_index], dtype=np.float64)
    low = np.asarray(band.low[:, ratio_index], dtype=np.float64)
    high = np.asarray(band.high[:, ratio_index], dtype=np.float64)
    safe = np.where(np.abs(central) > 1e-12, np.abs(central), np.nan)
    frac = np.maximum((central - low) / safe, (high - central) / safe)
    stable = np.isfinite(frac) & (frac <= float(max_rel_band))
    return mask & stable


CNM_GRAY = "#555555"


def _plot_triplet(
    outpath_stem: Path,
    *,
    edges: np.ndarray,
    hnm_band: HNMBand,
    combined_band: CombinedBand,
    cnm_band: CNMBand,
    xlabel: str,
    xlim: Tuple[float, float],
    header_label: str,
    extra_label: str | None,
    y_mask: np.ndarray | None = None,
    max_rel_band_for_plot: float | None = None,
) -> None:
    plt = _configure_matplotlib()

    if y_mask is None:
        mask = np.ones(hnm_band.centers.shape[0], dtype=bool)
    else:
        mask = np.asarray(y_mask, dtype=bool)

    cnm_central_2d = np.broadcast_to(
        cnm_band.central[:, None], hnm_band.central.shape
    )
    cnm_low_2d = np.broadcast_to(cnm_band.low[:, None], hnm_band.low.shape)
    cnm_high_2d = np.broadcast_to(cnm_band.high[:, None], hnm_band.high.shape)

    ylim = _compute_ylim(
        _masked_copy(hnm_band.high, mask[:, None]),
        _masked_copy(combined_band.high, mask[:, None]),
        _masked_copy(cnm_high_2d, mask[:, None]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
    for panel, (state_index, label_tex) in enumerate(zip(range(len(MAIN_STATE_NAMES)), MAIN_STATE_TEX)):
        ax = axes[panel]
        cnm_c_plot, cnm_l_plot, cnm_h_plot = _cap_band_for_plot(
            _masked_copy(cnm_central_2d[:, state_index], mask),
            _masked_copy(cnm_low_2d[:, state_index], mask),
            _masked_copy(cnm_high_2d[:, state_index], mask),
            max_rel_band=max_rel_band_for_plot,
        )
        _step_band_asymmetric(
            ax,
            edges,
            cnm_c_plot,
            cnm_l_plot,
            cnm_h_plot,
            color=CNM_GRAY,
            linestyle="-",
            linewidth=1.6,
            alpha=0.18,
            label="CNM only" if panel == 0 else None,
        )
        hnm_c_plot, hnm_l_plot, hnm_h_plot = _cap_band_for_plot(
            _masked_copy(hnm_band.central[:, state_index], mask),
            _masked_copy(hnm_band.low[:, state_index], mask),
            _masked_copy(hnm_band.high[:, state_index], mask),
            max_rel_band=max_rel_band_for_plot,
        )
        _step_band_asymmetric(
            ax,
            edges,
            hnm_c_plot,
            hnm_l_plot,
            hnm_h_plot,
            color=BLUE,
            linestyle="--",
            linewidth=2.0,
            alpha=0.12,
            label="KSU-Munich" if panel == 0 else None,
        )
        comb_c_plot, comb_l_plot, comb_h_plot = _cap_band_for_plot(
            _masked_copy(combined_band.central[:, state_index], mask),
            _masked_copy(combined_band.low[:, state_index], mask),
            _masked_copy(combined_band.high[:, state_index], mask),
            max_rel_band=max_rel_band_for_plot,
        )
        _step_band_asymmetric(
            ax,
            edges,
            comb_c_plot,
            comb_l_plot,
            comb_h_plot,
            color=BLUE,
            linestyle="-",
            linewidth=2.0,
            alpha=0.22,
            label=r"KSU-Munich $\times$ CNM" if panel == 0 else None,
        )
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axhline(1.0, color="0.55", linestyle="--", linewidth=0.8, zorder=0)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.text(
            0.97,
            0.96,
            label_tex,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=14,
            fontweight="bold",
        )
    axes[0].set_ylabel(r"$R_{AA}$")
    axes[0].text(
        0.03,
        0.97,
        header_label,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    if extra_label:
        axes[0].text(
            0.03,
            0.88,
            extra_label,
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=11,
        )
    axes[0].legend(loc="lower left", framealpha=0.95)
    fig.tight_layout(w_pad=0.25)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(outpath_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outpath_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


RATIO_PAIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (2, 0), (2, 1))
RATIO_KEYS: Tuple[str, ...] = ("2S_over_1S", "3S_over_1S", "3S_over_2S")
RATIO_TEX: Tuple[str, ...] = (
    r"$\Upsilon(2S)/\Upsilon(1S)$",
    r"$\Upsilon(3S)/\Upsilon(1S)$",
    r"$\Upsilon(3S)/\Upsilon(2S)$",
)
DOUBLE_RATIO_YLABEL = (
    r"$\dfrac{[\Upsilon(nS)/\Upsilon(mS)]_{\rm O+O}}"
    r"{[\Upsilon(nS)/\Upsilon(mS)]_{\rm pp}}$"
)


@dataclass(frozen=True)
class RatioBand:
    edges: np.ndarray
    centers: np.ndarray
    central: np.ndarray
    low: np.ndarray
    high: np.ndarray


@dataclass(frozen=True)
class IntegratedRatios:
    central: np.ndarray
    low: np.ndarray
    high: np.ndarray
    raa_kap5: np.ndarray
    raa_kap7: np.ndarray
    sem_kap5: np.ndarray
    sem_kap7: np.ndarray
    n_obs_kap5: int
    n_obs_kap7: int
    y_range: Tuple[float, float]
    pt_range: Tuple[float, float]


@dataclass(frozen=True)
class CmsPreliminaryDoubleRatios:
    integrated: Dict[str, Tuple[float, float]]
    pt_3s2s: list[Tuple[float, float, float]]


def _per_kappa_ratio(
    raa_i: float,
    raa_j: float,
    sem_i: float,
    sem_j: float,
) -> Tuple[float, float]:
    if not (np.isfinite(raa_i) and np.isfinite(raa_j)) or abs(raa_j) < 1e-12 or abs(raa_i) < 1e-12:
        return float("nan"), float("nan")
    ratio = raa_i / raa_j
    s_i = sem_i if np.isfinite(sem_i) else 0.0
    s_j = sem_j if np.isfinite(sem_j) else 0.0
    rel = float(np.sqrt((s_i / raa_i) ** 2 + (s_j / raa_j) ** 2))
    return float(ratio), float(abs(ratio) * rel)


def _envelope_two(values: Tuple[float, float], sems: Tuple[float, float]) -> Tuple[float, float, float]:
    v5, v7 = values
    s5, s7 = sems
    if not (np.isfinite(v5) and np.isfinite(v7)):
        return float("nan"), float("nan"), float("nan")
    central = 0.5 * (min(v5, v7) + max(v5, v7))
    low = float(min(v5 - s5, v7 - s7))
    high = float(max(v5 + s5, v7 + s7))
    return central, low, high


def _load_cms_preliminary_double_ratios(path: Path = CMS_PRELIM_DOUBLE_RATIO_CSV) -> CmsPreliminaryDoubleRatios | None:
    if not path.exists():
        return None
    integrated: Dict[str, Tuple[float, float]] = {}
    pt_points: list[Tuple[float, float, float]] = []
    obs_to_key = {
        "ratio_2S_1S": "2S_over_1S",
        "ratio_3S_1S": "3S_over_1S",
        "ratio_3S_2S": "3S_over_2S",
    }
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obs = str(row.get("observable", "")).strip()
            x_type = str(row.get("x_type", "")).strip().lower()
            key = obs_to_key.get(obs)
            if key is None:
                continue
            value = float(row.get("value", "nan"))
            stat = float(row.get("stat_err", "0") or 0.0)
            syst = float(row.get("syst_err", "0") or 0.0)
            err = float(np.sqrt(max(stat, 0.0) ** 2 + max(syst, 0.0) ** 2))
            if x_type == "integrated":
                integrated[key] = (value, err)
            elif x_type == "pt" and key == "3S_over_2S":
                x_center = float(row.get("x_center_gev", "nan"))
                pt_points.append((x_center, value, err))
    pt_points.sort(key=lambda x: x[0])
    if not integrated and not pt_points:
        return None
    return CmsPreliminaryDoubleRatios(integrated=integrated, pt_3s2s=pt_points)


def _compute_ratio_band_from_runs(
    edges: np.ndarray,
    run5_result: SpectralResult,
    run7_result: SpectralResult,
) -> RatioBand:
    centers = np.asarray(run5_result.centers, dtype=np.float64)
    n_bins = centers.shape[0]
    n_ratios = len(RATIO_PAIRS)
    central = np.full((n_bins, n_ratios), np.nan, dtype=np.float64)
    low = np.full_like(central, np.nan)
    high = np.full_like(central, np.nan)

    for r_idx, (i_main, j_main) in enumerate(RATIO_PAIRS):
        i9 = MAIN_STATE_INDICES[i_main]
        j9 = MAIN_STATE_INDICES[j_main]
        for b_idx in range(n_bins):
            ratio5, sem5 = _per_kappa_ratio(
                run5_result.raa9[b_idx, i9],
                run5_result.raa9[b_idx, j9],
                run5_result.sem9[b_idx, i9],
                run5_result.sem9[b_idx, j9],
            )
            ratio7, sem7 = _per_kappa_ratio(
                run7_result.raa9[b_idx, i9],
                run7_result.raa9[b_idx, j9],
                run7_result.sem9[b_idx, i9],
                run7_result.sem9[b_idx, j9],
            )
            c, lo, hi = _envelope_two((ratio5, ratio7), (sem5, sem7))
            central[b_idx, r_idx] = c
            low[b_idx, r_idx] = lo
            high[b_idx, r_idx] = hi

    return RatioBand(
        edges=np.asarray(edges, dtype=np.float64),
        centers=centers,
        central=central,
        low=low,
        high=high,
    )


def _build_integrated_band(
    *,
    kap5_obs_path: Path,
    kap7_obs_path: Path,
    feeddown: np.ndarray,
    sigmas_prim: np.ndarray,
    logger: logging.Logger,
    y_range: Tuple[float, float],
    pt_range: Tuple[float, float],
) -> IntegratedRatios:
    obs5, _ = _load_observables(kap5_obs_path, logger)
    obs7, _ = _load_observables(kap7_obs_path, logger)
    y_min, y_max = float(y_range[0]), float(y_range[1])
    pt_min, pt_max = float(pt_range[0]), float(pt_range[1])
    sel5 = [
        e for e in obs5
        if (y_min <= e.y <= y_max) and (pt_min <= e.pt <= pt_max)
    ]
    sel7 = [
        e for e in obs7
        if (y_min <= e.y <= y_max) and (pt_min <= e.pt <= pt_max)
    ]
    mean5_6, sem5_6 = _weighted_mean_sem_surv6(sel5)
    mean7_6, sem7_6 = _weighted_mean_sem_surv6(sel7)
    raa5_9, ssem5_9 = _raa9_with_sem(mean5_6, sem5_6, feeddown, sigmas_prim)
    raa7_9, ssem7_9 = _raa9_with_sem(mean7_6, sem7_6, feeddown, sigmas_prim)

    raa5_main = np.asarray(raa5_9[list(MAIN_STATE_INDICES)], dtype=np.float64)
    raa7_main = np.asarray(raa7_9[list(MAIN_STATE_INDICES)], dtype=np.float64)
    sem5_main = np.asarray(ssem5_9[list(MAIN_STATE_INDICES)], dtype=np.float64)
    sem7_main = np.asarray(ssem7_9[list(MAIN_STATE_INDICES)], dtype=np.float64)

    central = np.full(len(RATIO_PAIRS), np.nan, dtype=np.float64)
    low = np.full_like(central, np.nan)
    high = np.full_like(central, np.nan)
    for r_idx, (i_main, j_main) in enumerate(RATIO_PAIRS):
        ratio5, sem5 = _per_kappa_ratio(
            float(raa5_main[i_main]),
            float(raa5_main[j_main]),
            float(sem5_main[i_main]),
            float(sem5_main[j_main]),
        )
        ratio7, sem7 = _per_kappa_ratio(
            float(raa7_main[i_main]),
            float(raa7_main[j_main]),
            float(sem7_main[i_main]),
            float(sem7_main[j_main]),
        )
        c, lo, hi = _envelope_two((ratio5, ratio7), (sem5, sem7))
        central[r_idx] = c
        low[r_idx] = lo
        high[r_idx] = hi

    return IntegratedRatios(
        central=central,
        low=low,
        high=high,
        raa_kap5=raa5_main,
        raa_kap7=raa7_main,
        sem_kap5=sem5_main,
        sem_kap7=sem7_main,
        n_obs_kap5=len(sel5),
        n_obs_kap7=len(sel7),
        y_range=(y_min, y_max),
        pt_range=(pt_min, pt_max),
    )


def _ratio_band_rows(
    axis_name: str,
    band: RatioBand,
    mask: np.ndarray | None = None,
) -> Tuple[Sequence[str], Sequence[Dict[str, object]]]:
    active = np.ones(band.centers.shape[0], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    rows = []
    for index in np.flatnonzero(active):
        row: Dict[str, object] = {
            f"{axis_name}_low": float(band.edges[index]),
            f"{axis_name}_high": float(band.edges[index + 1]),
            f"{axis_name}_center": float(band.centers[index]),
        }
        for r_idx, key in enumerate(RATIO_KEYS):
            row[f"central_{key}"] = float(band.central[index, r_idx])
            row[f"band_lo_{key}"] = float(band.low[index, r_idx])
            row[f"band_hi_{key}"] = float(band.high[index, r_idx])
        rows.append(row)

    fields = [f"{axis_name}_low", f"{axis_name}_high", f"{axis_name}_center"]
    for key in RATIO_KEYS:
        fields.extend([f"central_{key}", f"band_lo_{key}", f"band_hi_{key}"])
    return fields, rows


def _ratio_ylim(*arrays: np.ndarray) -> Tuple[float, float]:
    lo = np.inf
    hi = -np.inf
    for arr in arrays:
        values = np.asarray(arr, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            lo = min(lo, float(np.min(finite)))
            hi = max(hi, float(np.max(finite)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.2
    span = max(hi - lo, 0.05)
    return max(0.0, lo - 0.1 * span), hi + 0.1 * span


def _plot_ratio_triplet(
    outpath_stem: Path,
    *,
    edges: np.ndarray,
    band: RatioBand,
    xlabel: str,
    xlim: Tuple[float, float],
    header_label: str,
    extra_label: str | None,
    mask: np.ndarray | None = None,
    max_rel_band_for_plot: float | None = None,
    cms_pt_3s2s: list[Tuple[float, float, float]] | None = None,
) -> None:
    plt = _configure_matplotlib()

    if mask is None:
        sel = np.ones(band.centers.shape[0], dtype=bool)
    else:
        sel = np.asarray(mask, dtype=bool)

    ylim = _ratio_ylim(
        _masked_copy(band.high, sel[:, None]),
        _masked_copy(band.low, sel[:, None]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
    for panel, label_tex in enumerate(RATIO_TEX):
        ax = axes[panel]
        ratio_c_plot, ratio_l_plot, ratio_h_plot = _cap_band_for_plot(
            _masked_copy(band.central[:, panel], sel),
            _masked_copy(band.low[:, panel], sel),
            _masked_copy(band.high[:, panel], sel),
            max_rel_band=max_rel_band_for_plot,
        )
        _step_band_asymmetric(
            ax,
            edges,
            ratio_c_plot,
            ratio_l_plot,
            ratio_h_plot,
            color=BLUE,
            linestyle="--",
            linewidth=2.0,
            alpha=0.16,
            label="KSU-Munich" if panel == 0 else None,
        )
        if panel == 2 and cms_pt_3s2s:
            cms_x = np.asarray([p[0] for p in cms_pt_3s2s], dtype=np.float64)
            cms_y = np.asarray([p[1] for p in cms_pt_3s2s], dtype=np.float64)
            cms_e = np.asarray([p[2] for p in cms_pt_3s2s], dtype=np.float64)
            ax.errorbar(
                cms_x,
                cms_y,
                yerr=cms_e,
                fmt="o",
                color="black",
                ecolor="black",
                elinewidth=1.4,
                capsize=3,
                markersize=5.5,
                label="CMS preliminary" if panel == 2 else None,
            )
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axhline(1.0, color="0.55", linestyle="--", linewidth=0.8, zorder=0)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.text(
            0.97,
            0.96,
            label_tex,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=14,
            fontweight="bold",
        )
    axes[0].set_ylabel(DOUBLE_RATIO_YLABEL)
    axes[0].text(
        0.03,
        0.97,
        header_label,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    if extra_label:
        axes[0].text(
            0.03,
            0.88,
            extra_label,
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=11,
        )
    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[2].get_legend_handles_labels()
    legend_handles = list(handles0)
    legend_labels = list(labels0)
    if "CMS preliminary" in labels2 and "CMS preliminary" not in legend_labels:
        idx = labels2.index("CMS preliminary")
        legend_handles.append(handles2[idx])
        legend_labels.append("CMS preliminary")
    axes[0].legend(legend_handles, legend_labels, loc="lower left", framealpha=0.95)
    fig.tight_layout(w_pad=0.25)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(outpath_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outpath_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_integrated_ratio(
    outpath_stem: Path,
    *,
    integrated: IntegratedRatios,
    header_label: str,
    cms_integrated: Dict[str, Tuple[float, float]] | None = None,
) -> None:
    plt = _configure_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))

    x_positions = np.arange(len(RATIO_PAIRS), dtype=np.float64)
    central = integrated.central
    err_lo = np.maximum(0.0, central - integrated.low)
    err_hi = np.maximum(0.0, integrated.high - central)
    yerr = np.vstack([err_lo, err_hi])

    ax.errorbar(
        x_positions,
        central,
        yerr=yerr,
        fmt="s",
        color=BLUE,
        ecolor=BLUE,
        elinewidth=2.0,
        capsize=6,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="KSU-Munich",
    )
    if cms_integrated:
        cms_vals = np.full(len(RATIO_KEYS), np.nan, dtype=np.float64)
        cms_errs = np.full(len(RATIO_KEYS), np.nan, dtype=np.float64)
        for i, key in enumerate(RATIO_KEYS):
            if key in cms_integrated:
                cms_vals[i], cms_errs[i] = cms_integrated[key]
        valid = np.isfinite(cms_vals) & np.isfinite(cms_errs)
        if np.any(valid):
            ax.errorbar(
                x_positions[valid] + 0.08,
                cms_vals[valid],
                yerr=cms_errs[valid],
                fmt="o",
                color="black",
                ecolor="black",
                elinewidth=1.4,
                capsize=3,
                markersize=5.5,
                label="CMS preliminary",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(RATIO_TEX, fontsize=13)
    ax.set_ylabel(DOUBLE_RATIO_YLABEL)
    ax.axhline(1.0, color="0.55", linestyle="--", linewidth=0.8, zorder=0)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    ax.text(
        0.03,
        0.97,
        header_label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    accept = (
        rf"$y \in [{integrated.y_range[0]:.1f},\,{integrated.y_range[1]:.1f}]$, "
        rf"$p_T \in [{integrated.pt_range[0]:.0f},\,{integrated.pt_range[1]:.0f}]$ GeV"
    )
    ax.text(
        0.03,
        0.88,
        accept,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    ax.set_xlim(-0.5, len(RATIO_PAIRS) - 0.5)
    finite_lows = integrated.low[np.isfinite(integrated.low)]
    finite_highs = integrated.high[np.isfinite(integrated.high)]
    if cms_integrated:
        cms_vals = []
        cms_errs = []
        for key in RATIO_KEYS:
            if key in cms_integrated:
                v, e = cms_integrated[key]
                if np.isfinite(v) and np.isfinite(e):
                    cms_vals.append(float(v))
                    cms_errs.append(float(e))
        if cms_vals:
            cms_vals_arr = np.asarray(cms_vals, dtype=np.float64)
            cms_errs_arr = np.asarray(cms_errs, dtype=np.float64)
            finite_lows = np.concatenate([finite_lows, cms_vals_arr - cms_errs_arr])
            finite_highs = np.concatenate([finite_highs, cms_vals_arr + cms_errs_arr])
    if finite_lows.size and finite_highs.size:
        lo = float(np.min(finite_lows))
        hi = float(np.max(finite_highs))
        span = max(hi - lo, 0.05)
        ax.set_ylim(max(0.0, lo - 0.15 * span), hi + 0.15 * span)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(outpath_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outpath_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_double_ratios(
    outdir: Path,
    *,
    run5: RunResult,
    run7: RunResult,
    y_hnm: HNMBand,
    pt_hnm: Dict[str, HNMBand],
    integrated_results: Dict[str, IntegratedRatios],
    specs: Tuple[WindowSpec, ...],
    pt_plot_min: float,
    pt_max_rel_band_for_plot: float | None,
) -> None:
    """Compute and emit double-ratio CSVs and figures.

    CNM is state-independent in this script's OO MB construction, so it
    cancels exactly in R_AA(nS)/R_AA(mS).  The ratio uses only the
    per-kappa HNM SpectralResults from run5/run7.
    """

    data_dir = outdir / "data"
    fig_dir = outdir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    header_label = r"$\mathbf{O+O\ (MB),\ \sqrt{s_{NN}} = 5.36\ TeV}$"
    pt_acceptance = rf"$p_T \in [{cnm_oo.PT_RANGE_AVG[0]:.0f},\,{cnm_oo.PT_RANGE_AVG[1]:.0f}]$ GeV"
    cms_data = _load_cms_preliminary_double_ratios()

    # ---- vs y --------------------------------------------------------
    y_mask = (y_hnm.centers >= Y_EXPORT_MIN) & (y_hnm.centers <= Y_EXPORT_MAX)
    y_ratio = _compute_ratio_band_from_runs(y_hnm.edges, run5.y_result, run7.y_result)
    fields, rows = _ratio_band_rows("y", y_ratio, y_mask)
    _write_csv(data_dir / "qtraj_double_ratio_vs_y.csv", rows, fields)

    masked_y_edges = _masked_bin_edges(y_ratio.edges, y_mask)
    _plot_ratio_triplet(
        fig_dir / "qtraj_double_ratio_vs_y_triplet",
        edges=masked_y_edges,
        band=RatioBand(
            edges=masked_y_edges,
            centers=y_ratio.centers[y_mask],
            central=y_ratio.central[y_mask],
            low=y_ratio.low[y_mask],
            high=y_ratio.high[y_mask],
        ),
        xlabel=r"$y$",
        xlim=(Y_EXPORT_MIN, Y_EXPORT_MAX),
        header_label=header_label,
        extra_label=pt_acceptance,
    )

    # ---- vs pT (per window) ------------------------------------------
    for spec in specs:
        pt_ratio = _compute_ratio_band_from_runs(
            pt_hnm[spec.key].edges,
            run5.pt_results[spec.key],
            run7.pt_results[spec.key],
        )
        fields, rows = _ratio_band_rows("pT", pt_ratio, None)
        _write_csv(
            data_dir / f"qtraj_double_ratio_vs_pT__{spec.key}.csv",
            rows,
            fields,
        )
        pt_ratio_plot_mask = np.asarray(pt_ratio.centers >= float(pt_plot_min), dtype=bool)
        extra_bits = [spec.label]
        if float(pt_plot_min) > 0.0:
            extra_bits.append(rf"shown for $p_T \geq {pt_plot_min:.1f}$ GeV")
        x_hi = float(pt_ratio.edges[-1])
        if cms_data is not None and spec.key == "midrapidity" and cms_data.pt_3s2s:
            x_hi = max(x_hi, max(p[0] for p in cms_data.pt_3s2s) + 1.0)
        _plot_ratio_triplet(
            fig_dir / f"qtraj_double_ratio_vs_pT__{spec.key}__triplet",
            edges=pt_ratio.edges,
            band=pt_ratio,
            xlabel=r"$p_T\ [\mathrm{GeV}]$",
            xlim=(float(pt_ratio.edges[0]), x_hi),
            header_label=header_label,
            extra_label="; ".join(extra_bits),
            mask=pt_ratio_plot_mask,
            max_rel_band_for_plot=pt_max_rel_band_for_plot,
            cms_pt_3s2s=(cms_data.pt_3s2s if (cms_data is not None and spec.key == "midrapidity") else None),
        )

    # ---- integrated --------------------------------------------------
    integ_fields = [
        "ratio",
        "central",
        "band_lo",
        "band_hi",
        "raa_kap5_num",
        "raa_kap5_den",
        "sem_kap5_num",
        "sem_kap5_den",
        "raa_kap7_num",
        "raa_kap7_den",
        "sem_kap7_num",
        "sem_kap7_den",
        "y_min",
        "y_max",
        "pt_min",
        "pt_max",
        "n_obs_kap5",
        "n_obs_kap7",
    ]
    for tag, integrated in integrated_results.items():
        integ_rows = []
        for r_idx, (i_main, j_main) in enumerate(RATIO_PAIRS):
            integ_rows.append(
                {
                    "ratio": RATIO_KEYS[r_idx],
                    "central": float(integrated.central[r_idx]),
                    "band_lo": float(integrated.low[r_idx]),
                    "band_hi": float(integrated.high[r_idx]),
                    "raa_kap5_num": float(integrated.raa_kap5[i_main]),
                    "raa_kap5_den": float(integrated.raa_kap5[j_main]),
                    "sem_kap5_num": float(integrated.sem_kap5[i_main]),
                    "sem_kap5_den": float(integrated.sem_kap5[j_main]),
                    "raa_kap7_num": float(integrated.raa_kap7[i_main]),
                    "raa_kap7_den": float(integrated.raa_kap7[j_main]),
                    "sem_kap7_num": float(integrated.sem_kap7[i_main]),
                    "sem_kap7_den": float(integrated.sem_kap7[j_main]),
                    "y_min": float(integrated.y_range[0]),
                    "y_max": float(integrated.y_range[1]),
                    "pt_min": float(integrated.pt_range[0]),
                    "pt_max": float(integrated.pt_range[1]),
                    "n_obs_kap5": integrated.n_obs_kap5,
                    "n_obs_kap7": integrated.n_obs_kap7,
                }
            )
        _write_csv(data_dir / f"qtraj_double_ratio_integrated__{tag}.csv", integ_rows, integ_fields)
        _plot_integrated_ratio(
            fig_dir / f"qtraj_double_ratio_integrated__{tag}",
            integrated=integrated,
            header_label=header_label,
            cms_integrated=(cms_data.integrated if cms_data is not None else None),
        )


def _write_manifest(
    outdir: Path,
    *,
    run5: RunResult,
    run7: RunResult,
    y_band_hnm: HNMBand,
    pt_bands_hnm: Dict[str, HNMBand],
    specs: Tuple[WindowSpec, ...],
) -> None:
    manifest = {
        "description": (
            "OO 5.36 TeV bottomonia minimum-bias package built from QTraj kap5/kap7 HNM "
            "inputs and the OO CNM MB engine. CNM is computed from run_bottomonia_cnm_OO.py; "
            "CNM x HNM is combined with asymmetric relative errors in quadrature."
        ),
        "qtraj_inputs": [
            {
                "label": run5.label,
                "datafile": str(run5.datafile),
                "load_mode": run5.load_mode,
                "n_observables": run5.n_observables,
            },
            {
                "label": run7.label,
                "datafile": str(run7.datafile),
                "load_mode": run7.load_mode,
                "n_observables": run7.n_observables,
            },
        ],
        "cnm_source_module": "cnm/cnm_scripts/run_bottomonia_cnm_OO.py",
        "kinematics": {
            "y_edges": np.asarray(cnm_oo.Y_EDGES, dtype=np.float64).tolist(),
            "pt_range_avg_GeV": list(cnm_oo.PT_RANGE_AVG),
            "rapidity_windows_for_pT": [
                {
                    "key": spec.key,
                    "y_min": spec.y_window[0],
                    "y_max": spec.y_window[1],
                    "label": spec.label,
                    "pt_bin_width_GeV": spec.pt_bin_width if _is_uniform_edges(spec.pt_edges) else None,
                    "pt_binning": "uniform" if _is_uniform_edges(spec.pt_edges) else "custom_tail_merged",
                    "pt_edges_GeV": np.asarray(spec.pt_edges, dtype=np.float64).tolist(),
                }
                for spec in specs
            ],
            "exported_y_center_range": [Y_EXPORT_MIN, Y_EXPORT_MAX],
        },
        "combination_formulae": {
            "hnm_central": "0.5 * (min(kap5, kap7) + max(kap5, kap7))",
            "hnm_low": "min(kap5, kap7)",
            "hnm_high": "max(kap5, kap7)",
            "combined": "R_total = R_HNM * R_CNM with asymmetric relative uncertainties added in quadrature",
        },
        "outputs": {
            "data_dir": str((outdir / "data").relative_to(REPO_ROOT)),
            "figure_dir": str((outdir / "figures").relative_to(REPO_ROOT)),
        },
        "state_order": list(MAIN_STATE_NAMES),
        "hnm_note": "No Glauber weighting is applied on the QTraj HNM side.",
        "grid_lengths": {
            "y_bins_full": int(len(y_band_hnm.centers)),
            "y_bins_exported": int(np.count_nonzero((y_band_hnm.centers >= Y_EXPORT_MIN) & (y_band_hnm.centers <= Y_EXPORT_MAX))),
            "pT_bins": {key: int(len(value.centers)) for key, value in pt_bands_hnm.items()},
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _save_outputs(
    outdir: Path,
    *,
    y_hnm: HNMBand,
    y_cnm: CNMBand,
    y_combined: CombinedBand,
    pt_hnm: Dict[str, HNMBand],
    pt_cnm: Dict[str, CNMBand],
    pt_combined: Dict[str, CombinedBand],
    specs: Tuple[WindowSpec, ...],
    y_pt_range: Tuple[float, float],
    pt_plot_min: float,
    pt_max_rel_band_for_plot: float | None,
) -> None:
    data_dir = outdir / "data"
    fig_dir = outdir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    y_mask = (y_hnm.centers >= Y_EXPORT_MIN) & (y_hnm.centers <= Y_EXPORT_MAX)

    hnm_fields, hnm_rows = _band_rows_hnm("y", y_hnm, y_mask)
    _write_csv(data_dir / "qtraj_hnm_vs_y.csv", hnm_rows, hnm_fields)
    combined_fields, combined_rows = _band_rows_combined("y", y_hnm, y_cnm, y_combined, y_mask)
    _write_csv(data_dir / "qtraj_hnm_cnm_vs_y.csv", combined_rows, combined_fields)

    for spec in specs:
        hnm_fields, hnm_rows = _band_rows_hnm("pT", pt_hnm[spec.key], None)
        _write_csv(data_dir / f"qtraj_hnm_vs_pT__{spec.key}.csv", hnm_rows, hnm_fields)
        combined_fields, combined_rows = _band_rows_combined(
            "pT",
            pt_hnm[spec.key],
            pt_cnm[spec.key],
            pt_combined[spec.key],
            None,
        )
        _write_csv(data_dir / f"qtraj_hnm_cnm_vs_pT__{spec.key}.csv", combined_rows, combined_fields)

    header_label = r"$\mathbf{O+O\ (MB),\ \sqrt{s_{NN}} = 5.36\ TeV}$"
    extra_y = None
    _plot_triplet(
        fig_dir / "qtraj_hnm_cnm_vs_y_triplet",
        edges=_masked_bin_edges(y_hnm.edges, y_mask),
        hnm_band=HNMBand(
            edges=_masked_bin_edges(y_hnm.edges, y_mask),
            centers=y_hnm.centers[y_mask],
            kap5=y_hnm.kap5[y_mask],
            kap7=y_hnm.kap7[y_mask],
            central=y_hnm.central[y_mask],
            low=y_hnm.low[y_mask],
            high=y_hnm.high[y_mask],
        ),
        combined_band=CombinedBand(
            edges=_masked_bin_edges(y_combined.edges, y_mask),
            centers=y_combined.centers[y_mask],
            central=y_combined.central[y_mask],
            low=y_combined.low[y_mask],
            high=y_combined.high[y_mask],
        ),
        cnm_band=CNMBand(
            edges=_masked_bin_edges(y_cnm.edges, y_mask),
            centers=y_cnm.centers[y_mask],
            central=y_cnm.central[y_mask],
            low=y_cnm.low[y_mask],
            high=y_cnm.high[y_mask],
        ),
        xlabel=r"$y$",
        xlim=(Y_EXPORT_MIN, Y_EXPORT_MAX),
        header_label=header_label,
        extra_label=extra_y,
        max_rel_band_for_plot=None,
    )

    pt_axis_xlim = (float(cnm_oo.PT_RANGE_AVG[0]), float(cnm_oo.PT_RANGE_AVG[1]))
    for spec in specs:
        pt_plot_mask = np.asarray(pt_hnm[spec.key].centers >= float(pt_plot_min), dtype=bool)
        extra_bits = [spec.label]
        if float(pt_plot_min) > 0.0:
            extra_bits.append(rf"shown for $p_T \geq {pt_plot_min:.1f}$ GeV")
        extra = "; ".join(extra_bits)
        _plot_triplet(
            fig_dir / f"qtraj_hnm_cnm_vs_pT__{spec.key}__triplet",
            edges=pt_hnm[spec.key].edges,
            hnm_band=pt_hnm[spec.key],
            combined_band=pt_combined[spec.key],
            cnm_band=pt_cnm[spec.key],
            xlabel=r"$p_T\ [\mathrm{GeV}]$",
            xlim=pt_axis_xlim,
            header_label=header_label,
            extra_label=extra,
            y_mask=pt_plot_mask,
            max_rel_band_for_plot=pt_max_rel_band_for_plot,
        )


def _print_spot_check(
    y_hnm: HNMBand,
    y_cnm: CNMBand,
    y_combined: CombinedBand,
    pt_hnm: HNMBand,
    pt_cnm: CNMBand,
    pt_combined: CombinedBand,
) -> None:
    y_indices = np.flatnonzero((y_hnm.centers >= Y_EXPORT_MIN) & (y_hnm.centers <= Y_EXPORT_MAX))
    if y_indices.size:
        index = int(y_indices[len(y_indices) // 2])
        print(
            "Spot check y-bin: "
            f"y={y_hnm.centers[index]:.2f}  "
            f"1S HNM=({y_hnm.central[index, 0]:.6f}, {y_hnm.low[index, 0]:.6f}, {y_hnm.high[index, 0]:.6f})  "
            f"CNM=({y_cnm.central[index]:.6f}, {y_cnm.low[index]:.6f}, {y_cnm.high[index]:.6f})  "
            f"Combined=({y_combined.central[index, 0]:.6f}, {y_combined.low[index, 0]:.6f}, {y_combined.high[index, 0]:.6f})"
        )

    if pt_hnm.centers.size:
        index = int(len(pt_hnm.centers) // 2)
        print(
            "Spot check pT-bin: "
            f"pT={pt_hnm.centers[index]:.2f}  "
            f"1S HNM=({pt_hnm.central[index, 0]:.6f}, {pt_hnm.low[index, 0]:.6f}, {pt_hnm.high[index, 0]:.6f})  "
            f"CNM=({pt_cnm.central[index]:.6f}, {pt_cnm.low[index]:.6f}, {pt_cnm.high[index]:.6f})  "
            f"Combined=({pt_combined.central[index, 0]:.6f}, {pt_combined.low[index, 0]:.6f}, {pt_combined.high[index, 0]:.6f})"
        )


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    logger = logging.getLogger("run_bottomonia_qtraj_hnm_cnm_minbias")

    pt_bin_widths = {
        "backward": float(args.pt_bin_backward),
        "midrapidity": float(args.pt_bin_midrapidity),
        "forward": float(args.pt_bin_forward),
    }
    _warn_if_custom_edges_override_bin_widths(pt_bin_widths, logger)
    specs = _window_specs(pt_bin_widths)
    y_pt_range = (float(args.y_pt_min), float(args.y_pt_max))
    pt_plot_min = float(args.pt_plot_min)
    pt_max_rel_band_for_plot = (
        None
        if args.pt_max_rel_band_for_plot is None or float(args.pt_max_rel_band_for_plot) < 0.0
        else float(args.pt_max_rel_band_for_plot)
    )

    run5 = _analyze_qtraj_run("kap5", _resolve_path(args.kap5_path), logger, specs, y_pt_range)
    run7 = _analyze_qtraj_run("kap7", _resolve_path(args.kap7_path), logger, specs, y_pt_range)

    y_hnm = _build_hnm_band(np.asarray(cnm_oo.Y_EDGES, dtype=np.float64), run5.y_result, run7.y_result)
    pt_hnm: Dict[str, HNMBand] = {}
    for spec in specs:
        pt_hnm[spec.key] = _build_hnm_band(
            np.asarray(spec.pt_edges, dtype=np.float64),
            run5.pt_results[spec.key],
            run7.pt_results[spec.key],
        )

    y_cnm, pt_cnm, _cnm_context = _compute_cnm_bands(logger, specs, y_pt_range)
    _assert_aligned("y", y_hnm.centers, y_cnm.centers)
    for spec in specs:
        _assert_aligned(f"pT {spec.key}", pt_hnm[spec.key].centers, pt_cnm[spec.key].centers)

    y_combined = _build_combined_band(y_hnm, y_cnm)
    pt_combined: Dict[str, CombinedBand] = {}
    for spec in specs:
        pt_combined[spec.key] = _build_combined_band(pt_hnm[spec.key], pt_cnm[spec.key])

    outdir = _resolve_path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        outdir,
        run5=run5,
        run7=run7,
        y_band_hnm=y_hnm,
        pt_bands_hnm=pt_hnm,
        specs=specs,
    )
    _save_outputs(
        outdir,
        y_hnm=y_hnm,
        y_cnm=y_cnm,
        y_combined=y_combined,
        pt_hnm=pt_hnm,
        pt_cnm=pt_cnm,
        pt_combined=pt_combined,
        specs=specs,
        y_pt_range=y_pt_range,
        pt_plot_min=pt_plot_min,
        pt_max_rel_band_for_plot=pt_max_rel_band_for_plot,
    )

    feeddown = build_feeddown_matrix()
    sigmas_prim = solve_primordial_sigmas(feeddown, SIGMAS_EXP_OO_5360)
    integrated_results = {
        "pt_0_20": _build_integrated_band(
            kap5_obs_path=_resolve_path(args.kap5_path),
            kap7_obs_path=_resolve_path(args.kap7_path),
            feeddown=feeddown,
            sigmas_prim=sigmas_prim,
            logger=logger,
            y_range=(DEFAULT_INTEGRATED_Y_MIN, DEFAULT_INTEGRATED_Y_MAX),
            pt_range=(0.0, float(cnm_oo.PT_RANGE_AVG[1])),
        ),
        "pt_0_30": _build_integrated_band(
            kap5_obs_path=_resolve_path(args.kap5_path),
            kap7_obs_path=_resolve_path(args.kap7_path),
            feeddown=feeddown,
            sigmas_prim=sigmas_prim,
            logger=logger,
            y_range=(DEFAULT_INTEGRATED_Y_MIN, DEFAULT_INTEGRATED_Y_MAX),
            pt_range=(0.0, 30.0),
        ),
    }

    _save_double_ratios(
        outdir,
        run5=run5,
        run7=run7,
        y_hnm=y_hnm,
        pt_hnm=pt_hnm,
        integrated_results=integrated_results,
        specs=specs,
        pt_plot_min=pt_plot_min,
        pt_max_rel_band_for_plot=pt_max_rel_band_for_plot,
    )

    _print_spot_check(
        y_hnm,
        y_cnm,
        y_combined,
        pt_hnm["midrapidity"],
        pt_cnm["midrapidity"],
        pt_combined["midrapidity"],
    )

    print(f"Wrote outputs to {outdir}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute OO bottomonia QTraj HNM (kap5,7), OO CNM MB, and combined CNM x HNM."
    )
    parser.add_argument(
        "--kap5-path",
        type=Path,
        default=_default_oo_wreg_path(5),
        help="Path to the OO kap5 qtraj bundle input (raw or averaged).",
    )
    parser.add_argument(
        "--kap7-path",
        type=Path,
        default=_default_oo_wreg_path(7),
        help="Path to the OO kap7 qtraj bundle input (raw or averaged).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, figures, and manifest.",
    )
    parser.add_argument(
        "--pt-bin-backward",
        type=float,
        default=DEFAULT_PT_BIN_WIDTHS["backward"],
        help="pT bin width (GeV) for the backward window.",
    )
    parser.add_argument(
        "--pt-bin-midrapidity",
        type=float,
        default=DEFAULT_PT_BIN_WIDTHS["midrapidity"],
        help="pT bin width (GeV) for the midrapidity window.",
    )
    parser.add_argument(
        "--pt-bin-forward",
        type=float,
        default=DEFAULT_PT_BIN_WIDTHS["forward"],
        help="pT bin width (GeV) for the forward window.",
    )
    parser.add_argument(
        "--y-pt-min",
        type=float,
        default=DEFAULT_Y_PT_MIN,
        help="Lower pT bound (GeV) used for the R_AA(y) projection on both HNM and CNM sides.",
    )
    parser.add_argument(
        "--y-pt-max",
        type=float,
        default=DEFAULT_Y_PT_MAX,
        help="Upper pT bound (GeV) used for the R_AA(y) projection on both HNM and CNM sides.",
    )
    parser.add_argument(
        "--pt-plot-min",
        type=float,
        default=DEFAULT_PT_PLOT_MIN,
        help="Plot-only lower pT bound (GeV) for the R_AA(pT) figures. CSV outputs are unaffected.",
    )
    parser.add_argument(
        "--pt-max-rel-band-for-plot",
        type=float,
        default=DEFAULT_PT_MAX_REL_BAND_FOR_PLOT,
        help="Plot-only cap on fractional half-band for pT bins (bands are clipped, bins are not hidden); set negative to disable.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
