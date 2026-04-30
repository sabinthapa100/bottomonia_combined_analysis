#!/usr/bin/env python3
"""
O+O sqrt(s_NN)=5.36 TeV (kappa_hat=6) QTraj-NLO HNM production (theory only).

Single impact parameter b (one value in the bundled OO files). At fixed b, R_AA
is built by **averaging over trajectories inside each kinematic bin** (half-open
pT and y bins in `kinematics_presets.OO_*`), then applying feed-down — no
averaging over b (there is only one b in these bundles).

Kinematics (OO-specific, see `kinematics_presets.OO_*`):
  - R_AA vs pT: 0–30 GeV (1 GeV bins), **CMS-style midrapidity |y| ≤ 2.4**.
  - R_AA vs y: y in **[-2.4, 2.4]**, integrating trajectories with **pT ≤ 30 GeV**.
  - Double ratios: 2S/1S, 3S/1S, and 3S/2S vs pT, vs y, and integrated at fixed b.

Quantum jumps OFF (noReg) vs ON (wReg):
  These are **different QTraj-NLO runs** (different input datafiles). The labels
  refer to the stochastic-jump / quantum-jump algorithm in the MCWF evolution,
  not “heavy-ion regeneration” in the thermal sense.

  **Observing R_AA(wReg) < R_AA(noReg) is not automatically a bug.** Turning
  quantum jumps on changes the Lindblad evolution (additional transition
  channels / decoherence). That can **increase** in-medium suppression of the
  projected survival and therefore **lower** R_AA relative to the no-jump
  truncation. Cross-check: trajectory counts, file provenance, and convergence
  of SEM in each bin (`oo5360_diagnostics.json`).

Important note about the qtraj raw format:
  For temperatureEvolution=3 output, the final two columns are documented in
  qtraj-fftw.tex as:
    <RAND> <INIT_L>
  i.e. the seventh numeric column is the initial random number, not a hidden
  primordial singlet contribution. This production runner therefore accepts the
  raw `datafile.gz` directly and lets `load_qtraj_table()` perform the same
  averaging as `processEvents.py` in memory when needed.

Outputs (repo-relative):
  outputs/qtraj_outputs/LHC/OxygenOxygen5p36TeV/production/
    data/
    figures/theory_only/

This script intentionally does NOT use any `glauber-data/*.tsv` because the OO
bundle is a single-b min-bias prediction and R_AA does not require N_bin.
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
from typing import Iterable, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qtraj_analysis.binning import (  # noqa: E402
    binned_step_series,
)
from qtraj_analysis.feeddown import (  # noqa: E402
    apply_feeddown_to_raa6,
    build_feeddown_matrix,
    solve_primordial_sigmas,
)
from qtraj_analysis.io import load_qtraj_table, parse_records  # noqa: E402
from qtraj_analysis.kinematics_presets import (  # noqa: E402
    OO_INTEGRATED_Y_WINDOW,
    OO_PT_EDGES as PT_EDGES,
    OO_PT_MAX_FOR_Y as PT_MAX_FOR_Y,
    OO_PT_RAPIDITY_WINDOWS as PT_RAPIDITY_WINDOWS,
    OO_Y_EDGES as Y_EDGES,
    SIGMAS_EXP_OO_5360,
)
from qtraj_analysis.matching import build_observables  # noqa: E402


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "inputs").exists() and (parent / "hnm").exists():
            return parent
    raise RuntimeError(f"Could not infer repo root from {here}")


REPO_ROOT = _find_repo_root()

B_MINBIAS = 4.49691  # fm (the only simulated b in the OO bundle)
KAPPA_HAT = 6
CMS_PRELIMINARY_OO_APPROX = (
    REPO_ROOT
    / "inputs"
    / "experimental_data"
    / "lhc"
    / "OxygenOxygen5p36TeV"
    / "cms_preliminary_approx_upsilon_double_ratios.json"
)
CNM_OO_5360_DIR = REPO_ROOT / "outputs" / "cnm" / "min_bias" / "OO_5p36TeV"
CNM_OO_5360_PT = CNM_OO_5360_DIR / "Upsilon_RAA_cnm_vs_pT_MB_-2.4y2.4_OO_5p36TeV.csv"
CNM_OO_5360_Y = CNM_OO_5360_DIR / "Upsilon_RAA_cnm_vs_y_MB_OO_5p36TeV.csv"

# OO-only binning from kinematics_presets.OO_* (not the PbPb default PT_EDGES).
SIGMAS_EXP_PP_5TEV = SIGMAS_EXP_OO_5360
INTEGRATED_Y_WINDOW = OO_INTEGRATED_Y_WINDOW
INTEGRATED_Y_LABEL = PT_RAPIDITY_WINDOWS[0][2]


MODE_CONFIGS = {
    "noReg": {
        "label": "noReg",
        "description": "Quantum jumps OFF",
        "datafile": "inputs/qtraj_inputs/OxygenOxygen5360/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile.gz",
        "linestyle": "--",
    },
    "wReg": {
        "label": "wReg",
        "description": "Quantum jumps ON",
        "datafile": "inputs/qtraj_inputs/OxygenOxygen5360/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile.gz",
        "linestyle": "-",
    },
}

DOUBLE_RATIO_SPECS = (
    ("ratio_2S_1S", "err_2S_1S", 1, 0, r"$\Upsilon(2S)/\Upsilon(1S)$"),
    ("ratio_3S_1S", "err_3S_1S", 5, 0, r"$\Upsilon(3S)/\Upsilon(1S)$"),
    ("ratio_3S_2S", "err_3S_2S", 5, 1, r"$\Upsilon(3S)/\Upsilon(2S)$"),
)


@dataclass(frozen=True)
class ModeResult:
    label: str
    description: str
    linestyle: str
    n_trajectories: int
    pt_all: np.ndarray
    y_all: np.ndarray
    surv6_all: np.ndarray
    qweights_all: np.ndarray
    pt_results: Tuple["PtWindowResult", ...]
    y_centers: np.ndarray
    raa9_y: np.ndarray
    sem9_y: np.ndarray
    mean6_int: np.ndarray
    sem6_int: np.ndarray
    raa9_int: np.ndarray
    sem9_int: np.ndarray


@dataclass(frozen=True)
class PtWindowResult:
    key: str
    y_window: Tuple[float, float]
    y_label: str
    pt_centers: np.ndarray
    raa9_pt: np.ndarray
    sem9_pt: np.ndarray


@dataclass(frozen=True)
class DoubleRatioSeries:
    x: np.ndarray
    ratio_2S_1S: np.ndarray
    err_2S_1S: np.ndarray
    ratio_3S_1S: np.ndarray
    err_3S_1S: np.ndarray
    ratio_3S_2S: np.ndarray
    err_3S_2S: np.ndarray


STATE_SPECS_MAIN = (
    (0, "Upsilon(1S)", r"$\Upsilon(1S)$", "#1f77b4"),
    (1, "Upsilon(2S)", r"$\Upsilon(2S)$", "#ff7f0e"),
    (5, "Upsilon(3S)", r"$\Upsilon(3S)$", "#2ca02c"),
)


def _weighted_mean_sem_matrix(
    X: np.ndarray,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if X.size == 0:
        return np.full(6, np.nan, dtype=np.float64), np.full(6, np.nan, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 6:
        raise ValueError("X must have shape (N, 6).")
    if q.ndim != 1 or q.shape[0] != X.shape[0]:
        raise ValueError("q must be length N for X shape (N, 6).")
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


def _load_observables(datafile: Path, logger: logging.Logger):
    table = load_qtraj_table(str(datafile), logger)
    records = parse_records(table, logger)
    return build_observables(records, logger)


def _apply_feeddown_binned(
    raa6: np.ndarray,
    sem6: np.ndarray,
    feeddown: np.ndarray,
    sigmas_prim: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    raa9 = np.full((raa6.shape[0], 9), np.nan, dtype=np.float64)
    sem9 = np.full((raa6.shape[0], 9), np.nan, dtype=np.float64)
    for i in range(raa6.shape[0]):
        if np.isnan(raa6[i, 0]):
            continue
        r9, e9 = apply_feeddown_to_raa6(raa6[i], sem6[i], feeddown, sigmas_prim)
        raa9[i] = r9
        sem9[i] = e9
    return raa9, sem9


def _weighted_mean_sem_surv6(obs: list) -> Tuple[np.ndarray, np.ndarray]:
    if not obs:
        return np.full(6, np.nan, dtype=np.float64), np.full(6, np.nan, dtype=np.float64)

    X = np.vstack([o.surv6 for o in obs]).astype(np.float64)
    q = np.asarray([o.qweight for o in obs], dtype=np.float64)
    return _weighted_mean_sem_matrix(X, q)


def _compute_raa_vs_pt_fixed_b(
    obs: list,
    pt_edges: np.ndarray,
    y_window: Tuple[float, float],
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0, y1 = y_window
    chosen = [o for o in obs if y0 <= o.y <= y1]
    logger.info(
        "Fixed-b binning with qweights: y window [%.2f, %.2f] -> %d trajectories",
        y0,
        y1,
        len(chosen),
    )
    centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    means = []
    sems = []
    for i in range(len(pt_edges) - 1):
        p0, p1 = pt_edges[i], pt_edges[i + 1]
        bin_obs = [o for o in chosen if p0 <= o.pt < p1]
        mu, se = _weighted_mean_sem_surv6(bin_obs)
        means.append(mu)
        sems.append(se)
    return centers, np.vstack(means), np.vstack(sems)


def _compute_raa_vs_y_fixed_b(
    obs: list,
    y_edges: np.ndarray,
    pt_max_for_y: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = [o for o in obs if o.pt <= pt_max_for_y]
    centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    means = []
    sems = []
    for i in range(len(y_edges) - 1):
        y0, y1 = y_edges[i], y_edges[i + 1]
        bin_obs = [o for o in chosen if y0 <= o.y < y1]
        mu, se = _weighted_mean_sem_surv6(bin_obs)
        means.append(mu)
        sems.append(se)
    return centers, np.vstack(means), np.vstack(sems)


def _compute_scaled_raa_vs_pt_from_arrays(
    pt_all: np.ndarray,
    y_all: np.ndarray,
    surv6_all: np.ndarray,
    qweights_all: np.ndarray,
    pt_edges: np.ndarray,
    y_window: Tuple[float, float],
    factor_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0, y1 = y_window
    select = (y0 <= y_all) & (y_all <= y1) & np.isfinite(factor_all)
    centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    means = []
    sems = []
    for i in range(len(pt_edges) - 1):
        p0, p1 = pt_edges[i], pt_edges[i + 1]
        mask = select & (p0 <= pt_all) & (pt_all < p1)
        X = surv6_all[mask] * factor_all[mask, None]
        q = qweights_all[mask]
        mu, se = _weighted_mean_sem_matrix(X, q)
        means.append(mu)
        sems.append(se)
    return centers, np.vstack(means), np.vstack(sems)


def _compute_scaled_raa_vs_y_from_arrays(
    pt_all: np.ndarray,
    y_all: np.ndarray,
    surv6_all: np.ndarray,
    qweights_all: np.ndarray,
    y_edges: np.ndarray,
    pt_max_for_y: float,
    factor_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    select = (pt_all <= pt_max_for_y) & np.isfinite(factor_all)
    centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    means = []
    sems = []
    for i in range(len(y_edges) - 1):
        y0, y1 = y_edges[i], y_edges[i + 1]
        mask = select & (y0 <= y_all) & (y_all < y1)
        X = surv6_all[mask] * factor_all[mask, None]
        q = qweights_all[mask]
        mu, se = _weighted_mean_sem_matrix(X, q)
        means.append(mu)
        sems.append(se)
    return centers, np.vstack(means), np.vstack(sems)


def _compute_scaled_integrated_raa_from_arrays(
    pt_all: np.ndarray,
    y_all: np.ndarray,
    surv6_all: np.ndarray,
    qweights_all: np.ndarray,
    y_window: Tuple[float, float],
    pt_max_for_y: float,
    factor_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y0, y1 = y_window
    mask = (y0 <= y_all) & (y_all <= y1) & (pt_all <= pt_max_for_y) & np.isfinite(factor_all)
    X = surv6_all[mask] * factor_all[mask, None]
    q = qweights_all[mask]
    return _weighted_mean_sem_matrix(X, q)


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
    ratio_err = np.full_like(num, np.nan, dtype=np.float64)

    ok = np.isfinite(num) & np.isfinite(den) & np.isfinite(num_err) & np.isfinite(den_err) & (den != 0.0)
    if not np.any(ok):
        return ratio, ratio_err

    ratio[ok] = num[ok] / den[ok]
    ratio_err[ok] = np.sqrt(
        (num_err[ok] / den[ok]) ** 2 + ((num[ok] * den_err[ok]) / (den[ok] ** 2)) ** 2
    )
    return ratio, ratio_err


def _compute_double_ratio_series(
    x: np.ndarray,
    raa9: np.ndarray,
    sem9: np.ndarray,
) -> DoubleRatioSeries:
    x = np.asarray(x, dtype=np.float64)
    raa9 = np.asarray(raa9, dtype=np.float64)
    sem9 = np.asarray(sem9, dtype=np.float64)
    if raa9.ndim == 1:
        raa9 = raa9.reshape(1, -1)
    if sem9.ndim == 1:
        sem9 = sem9.reshape(1, -1)
    if raa9.shape != sem9.shape:
        raise ValueError("raa9 and sem9 must have the same shape.")
    if raa9.shape[0] != x.shape[0]:
        raise ValueError("x length must match the first dimension of raa9/sem9.")

    ratio_2S_1S, err_2S_1S = _ratio_with_err(raa9[:, 1], raa9[:, 0], sem9[:, 1], sem9[:, 0])
    ratio_3S_1S, err_3S_1S = _ratio_with_err(raa9[:, 5], raa9[:, 0], sem9[:, 5], sem9[:, 0])
    ratio_3S_2S, err_3S_2S = _ratio_with_err(raa9[:, 5], raa9[:, 1], sem9[:, 5], sem9[:, 1])
    return DoubleRatioSeries(
        x=x,
        ratio_2S_1S=ratio_2S_1S,
        err_2S_1S=err_2S_1S,
        ratio_3S_1S=ratio_3S_1S,
        err_3S_1S=err_3S_1S,
        ratio_3S_2S=ratio_3S_2S,
        err_3S_2S=err_3S_2S,
    )


def _load_cms_preliminary_approx() -> Optional[dict]:
    if not CMS_PRELIMINARY_OO_APPROX.exists():
        return None
    return json.loads(CMS_PRELIMINARY_OO_APPROX.read_text(encoding="utf-8"))


def _analyze_mode(mode: str, logger: logging.Logger) -> ModeResult:
    cfg = MODE_CONFIGS[mode]
    datafile = (REPO_ROOT / cfg["datafile"]).resolve()
    if not datafile.exists():
        raise FileNotFoundError(f"Datafile not found: {datafile}")

    feeddown = build_feeddown_matrix()
    sigmas_prim = solve_primordial_sigmas(feeddown, SIGMAS_EXP_PP_5TEV)

    obs = _load_observables(datafile, logger)

    # RAA(pT) in the three requested rapidity windows.
    pt_results = []
    for key, y_window, y_label in PT_RAPIDITY_WINDOWS:
        pt_centers, raa6_pt, sem6_pt = _compute_raa_vs_pt_fixed_b(
            obs, PT_EDGES, y_window=y_window, logger=logger
        )
        raa9_pt, sem9_pt = _apply_feeddown_binned(raa6_pt, sem6_pt, feeddown, sigmas_prim)
        pt_results.append(
            PtWindowResult(
                key=key,
                y_window=y_window,
                y_label=y_label,
                pt_centers=pt_centers,
                raa9_pt=raa9_pt,
                sem9_pt=sem9_pt,
            )
        )

    # RAA(y) integrating pT up to PT_MAX_FOR_Y.
    y_centers, raa6_y, sem6_y = _compute_raa_vs_y_fixed_b(obs, Y_EDGES, pt_max_for_y=PT_MAX_FOR_Y)
    raa9_y, sem9_y = _apply_feeddown_binned(raa6_y, sem6_y, feeddown, sigmas_prim)

    # Integrated mid-rapidity summary: same y bounds as `compute_raa_vs_pt` y_window (inclusive).
    obs_int = [
        o
        for o in obs
        if (INTEGRATED_Y_WINDOW[0] <= o.y <= INTEGRATED_Y_WINDOW[1]) and (o.pt <= PT_MAX_FOR_Y)
    ]
    if not obs_int:
        mean6 = np.full(6, np.nan, dtype=np.float64)
        sem6 = np.full(6, np.nan, dtype=np.float64)
    else:
        mean6, sem6 = _weighted_mean_sem_surv6(obs_int)
    raa9_int, sem9_int = apply_feeddown_to_raa6(mean6, sem6, feeddown, sigmas_prim)

    logger.info(
        "OO 5.36 TeV (%s): integrated %s, pT<=%.1f -> RAA(1S)=%.4f RAA(2S)=%.4f RAA(3S)=%.4f",
        cfg["label"],
        INTEGRATED_Y_LABEL.replace("$", ""),
        PT_MAX_FOR_Y,
        raa9_int[0],
        raa9_int[1],
        raa9_int[5],
    )

    return ModeResult(
        label=cfg["label"],
        description=cfg["description"],
        linestyle=cfg["linestyle"],
        n_trajectories=len(obs),
        pt_all=np.array([o.pt for o in obs], dtype=np.float64),
        y_all=np.array([o.y for o in obs], dtype=np.float64),
        surv6_all=np.vstack([o.surv6 for o in obs]).astype(np.float64),
        qweights_all=np.array([o.qweight for o in obs], dtype=np.float64),
        pt_results=tuple(pt_results),
        y_centers=y_centers,
        raa9_y=raa9_y,
        sem9_y=sem9_y,
        mean6_int=mean6,
        sem6_int=sem6,
        raa9_int=raa9_int,
        sem9_int=sem9_int,
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_mode_diagnostics(
    out_data_dir: Path,
    mode_key: str,
    result: ModeResult,
    logger: logging.Logger,
) -> None:
    """Single-b summary: trajectory count, integrated R_AA, kinematic grid (fixed b = B_MINBIAS)."""
    cfg = MODE_CONFIGS[mode_key]
    datafile = (REPO_ROOT / cfg["datafile"]).resolve()
    integrated_ratios = _compute_double_ratio_series(
        np.array([B_MINBIAS], dtype=np.float64),
        result.raa9_int,
        result.sem9_int,
    )
    integrated_mask = (
        (INTEGRATED_Y_WINDOW[0] <= result.y_all)
        & (result.y_all <= INTEGRATED_Y_WINDOW[1])
        & (result.pt_all <= PT_MAX_FOR_Y)
    )
    integrated_qsum = float(np.sum(result.qweights_all[integrated_mask]))
    integrated_examples = result.surv6_all[integrated_mask][:3].tolist()
    payload = {
        "mode_key": mode_key,
        "label": cfg["label"],
        "description": cfg["description"],
        "datafile": str(datafile),
        "b_fixed_fm": B_MINBIAS,
        "single_b_only": True,
        "centrality_or_npart_scan_available": False,
        "averaging": (
            "At fixed b, R_AA in each bin is the qweight-weighted mean over trajectories in that bin, "
            "followed by feed-down."
        ),
        "n_trajectories": result.n_trajectories,
        "qweight_summary": {
            "min": float(np.min(result.qweights_all)),
            "max": float(np.max(result.qweights_all)),
            "mean": float(np.mean(result.qweights_all)),
            "all_equal": bool(np.allclose(result.qweights_all, result.qweights_all[0])),
        },
        "kinematics": {
            "pt_edges_gev": PT_EDGES.tolist(),
            "y_edges": Y_EDGES.tolist(),
            "pt_max_for_raa_vs_y": PT_MAX_FOR_Y,
            "mid_rapidity_window_for_raa_vs_pt": list(PT_RAPIDITY_WINDOWS[0][1]),
        },
        "method": {
            "fixed_b_average": "qweight-weighted mean survival in each selected bin",
            "feeddown": "apply_feeddown_to_raa6 on the weighted six-state survival mean and SEM",
            "bin_definitions": "half-open pT bins [pT_i, pT_{i+1}) and half-open y bins [y_i, y_{i+1})",
        },
        "integrated_selection": {
            "y_window": list(INTEGRATED_Y_WINDOW),
            "pt_max_gev": PT_MAX_FOR_Y,
            "n_trajectories": int(np.sum(integrated_mask)),
            "qsum": integrated_qsum,
            "mean_surv6": result.mean6_int.tolist(),
            "sem_surv6": result.sem6_int.tolist(),
            "example_surv6_rows_first3": integrated_examples,
        },
        "raa9_integrated_midrapidity_pt": {
            "Upsilon_1S": float(result.raa9_int[0]),
            "Upsilon_2S": float(result.raa9_int[1]),
            "Upsilon_3S": float(result.raa9_int[5]),
        },
        "double_ratios_integrated_midrapidity_pt": {
            "ratio_2S_1S": float(integrated_ratios.ratio_2S_1S[0]),
            "err_2S_1S": float(integrated_ratios.err_2S_1S[0]),
            "ratio_3S_1S": float(integrated_ratios.ratio_3S_1S[0]),
            "err_3S_1S": float(integrated_ratios.err_3S_1S[0]),
            "ratio_3S_2S": float(integrated_ratios.ratio_3S_2S[0]),
            "err_3S_2S": float(integrated_ratios.err_3S_2S[0]),
        },
    }
    path = out_data_dir / f"oo5360_diagnostics__{cfg['label']}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def _write_comparison_diagnostics(
    out_data_dir: Path,
    no_reg: ModeResult,
    w_reg: ModeResult,
    logger: logging.Logger,
) -> None:
    """Highlight wReg vs noReg difference on integrated 1S (not a physical proof of correctness)."""
    d1s = float(w_reg.raa9_int[0] - no_reg.raa9_int[0])
    no_reg_ratios = _compute_double_ratio_series(
        np.array([B_MINBIAS], dtype=np.float64),
        no_reg.raa9_int,
        no_reg.sem9_int,
    )
    w_reg_ratios = _compute_double_ratio_series(
        np.array([B_MINBIAS], dtype=np.float64),
        w_reg.raa9_int,
        w_reg.sem9_int,
    )
    payload = {
        "note": (
            "wReg - noReg for integrated mid-rapidity R_AA(1S). "
            "Negative means lower R_AA with quantum jumps ON — can occur from altered MCWF evolution "
            "(see module docstring), not necessarily from a bug."
        ),
        "delta_raa1S_integrated": d1s,
        "noReg_raa1S_integrated": float(no_reg.raa9_int[0]),
        "wReg_raa1S_integrated": float(w_reg.raa9_int[0]),
        "delta_ratio_2S_1S_integrated": float(w_reg_ratios.ratio_2S_1S[0] - no_reg_ratios.ratio_2S_1S[0]),
        "delta_ratio_3S_1S_integrated": float(w_reg_ratios.ratio_3S_1S[0] - no_reg_ratios.ratio_3S_1S[0]),
        "delta_ratio_3S_2S_integrated": float(w_reg_ratios.ratio_3S_2S[0] - no_reg_ratios.ratio_3S_2S[0]),
    }
    path = out_data_dir / "oo5360_diagnostics__wReg_minus_noReg.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def _save_mode_csvs(result: ModeResult, out_data_dir: Path, logger: logging.Logger) -> None:
    _ensure_dir(out_data_dir)

    header = (
        "x,RAA_1S,RAA_2S,RAA_1P0,RAA_1P1,RAA_1P2,RAA_3S,RAA_2P0,RAA_2P1,RAA_2P2,"
        "SEM_1S,SEM_2S,SEM_1P0,SEM_1P1,SEM_1P2,SEM_3S,SEM_2P0,SEM_2P1,SEM_2P2"
    )

    for pt_result in result.pt_results:
        pt_path = out_data_dir / f"oo5360_raavspt__{pt_result.key}__{result.label}.csv"
        pt_arr = np.column_stack([pt_result.pt_centers, pt_result.raa9_pt, pt_result.sem9_pt])
        np.savetxt(pt_path, pt_arr, delimiter=",", header=header, comments="", fmt="%.10g")
        logger.info("Saved %s", pt_path)

    y_path = out_data_dir / f"oo5360_raavsy__{result.label}.csv"
    y_arr = np.column_stack([result.y_centers, result.raa9_y, result.sem9_y])
    np.savetxt(y_path, y_arr, delimiter=",", header=header, comments="", fmt="%.10g")
    logger.info("Saved %s", y_path)

    int_path = out_data_dir / f"oo5360_integrated__{result.label}.csv"
    int_header = "RAA_1S,RAA_2S,RAA_1P0,RAA_1P1,RAA_1P2,RAA_3S,RAA_2P0,RAA_2P1,RAA_2P2," \
        "SEM_1S,SEM_2S,SEM_1P0,SEM_1P1,SEM_1P2,SEM_3S,SEM_2P0,SEM_2P1,SEM_2P2"
    int_arr = np.concatenate([result.raa9_int, result.sem9_int]).reshape(1, -1)
    np.savetxt(int_path, int_arr, delimiter=",", header=int_header, comments="", fmt="%.10g")
    logger.info("Saved %s", int_path)


def _save_double_ratio_csvs(result: ModeResult, out_data_dir: Path, logger: logging.Logger) -> None:
    _ensure_dir(out_data_dir)

    header = "x,ratio_2S_1S,err_2S_1S,ratio_3S_1S,err_3S_1S,ratio_3S_2S,err_3S_2S"

    for pt_result in result.pt_results:
        series = _compute_double_ratio_series(pt_result.pt_centers, pt_result.raa9_pt, pt_result.sem9_pt)
        pt_path = out_data_dir / f"oo5360_double_ratios_vspt__{pt_result.key}__{result.label}.csv"
        pt_arr = np.column_stack(
            [
                series.x,
                series.ratio_2S_1S,
                series.err_2S_1S,
                series.ratio_3S_1S,
                series.err_3S_1S,
                series.ratio_3S_2S,
                series.err_3S_2S,
            ]
        )
        np.savetxt(pt_path, pt_arr, delimiter=",", header=header, comments="", fmt="%.10g")
        logger.info("Saved %s", pt_path)

    y_series = _compute_double_ratio_series(result.y_centers, result.raa9_y, result.sem9_y)
    y_path = out_data_dir / f"oo5360_double_ratios_vsy__{result.label}.csv"
    y_arr = np.column_stack(
        [
            y_series.x,
            y_series.ratio_2S_1S,
            y_series.err_2S_1S,
            y_series.ratio_3S_1S,
            y_series.err_3S_1S,
            y_series.ratio_3S_2S,
            y_series.err_3S_2S,
        ]
    )
    np.savetxt(y_path, y_arr, delimiter=",", header=header, comments="", fmt="%.10g")
    logger.info("Saved %s", y_path)

    int_series = _compute_double_ratio_series(np.array([B_MINBIAS], dtype=np.float64), result.raa9_int, result.sem9_int)
    int_path = out_data_dir / f"oo5360_double_ratios_integrated__{result.label}.csv"
    int_header = "b_fm,ratio_2S_1S,err_2S_1S,ratio_3S_1S,err_3S_1S,ratio_3S_2S,err_3S_2S"
    int_arr = np.column_stack(
        [
            int_series.x,
            int_series.ratio_2S_1S,
            int_series.err_2S_1S,
            int_series.ratio_3S_1S,
            int_series.err_3S_1S,
            int_series.ratio_3S_2S,
            int_series.err_3S_2S,
        ]
    )
    np.savetxt(int_path, int_arr, delimiter=",", header=int_header, comments="", fmt="%.10g")
    logger.info("Saved %s", int_path)


def _load_cnm_curve(path: Path, x_key: str) -> dict:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    return {
        "x": np.asarray(arr[x_key], dtype=np.float64),
        "central": np.asarray(arr["R_central"], dtype=np.float64),
        "lo": np.asarray(arr["R_lo"], dtype=np.float64),
        "hi": np.asarray(arr["R_hi"], dtype=np.float64),
    }


def _interp_curve(values: np.ndarray, curve: dict, *, hold_edges: bool) -> np.ndarray:
    left = float(curve["central"][0]) if hold_edges else np.nan
    right = float(curve["central"][-1]) if hold_edges else np.nan
    return np.interp(values, curve["x"], curve["central"], left=left, right=right)


def _interp_curve_variant(values: np.ndarray, curve: dict, key: str, *, hold_edges: bool) -> np.ndarray:
    left = float(curve[key][0]) if hold_edges else np.nan
    right = float(curve[key][-1]) if hold_edges else np.nan
    return np.interp(values, curve["x"], curve[key], left=left, right=right)


def _curve_support_mask(x: np.ndarray, curve: dict) -> np.ndarray:
    return (x >= float(curve["x"][0]) - 1e-12) & (x <= float(curve["x"][-1]) + 1e-12)


def _total_band_from_curves(
    central: np.ndarray,
    stat: np.ndarray,
    lo_curve: np.ndarray,
    hi_curve: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    central = np.asarray(central, dtype=np.float64)
    stat = np.asarray(stat, dtype=np.float64)
    lo_curve = np.asarray(lo_curve, dtype=np.float64)
    hi_curve = np.asarray(hi_curve, dtype=np.float64)
    sys_lo = np.maximum(central - lo_curve, 0.0)
    sys_hi = np.maximum(hi_curve - central, 0.0)
    total_lo = central - np.sqrt(np.maximum(stat, 0.0) ** 2 + sys_lo ** 2)
    total_hi = central + np.sqrt(np.maximum(stat, 0.0) ** 2 + sys_hi ** 2)
    return total_lo, total_hi, sys_lo, sys_hi


def _compute_noReg_qgp_cnm_package(mode_result: ModeResult, logger: logging.Logger) -> dict:
    if not CNM_OO_5360_PT.exists() or not CNM_OO_5360_Y.exists():
        raise FileNotFoundError(
            f"Expected CNM OO inputs not found: {CNM_OO_5360_PT} and/or {CNM_OO_5360_Y}"
        )

    feeddown = build_feeddown_matrix()
    sigmas_prim = solve_primordial_sigmas(feeddown, SIGMAS_EXP_PP_5TEV)

    pt_curve = _load_cnm_curve(CNM_OO_5360_PT, "pT_center")
    y_curve = _load_cnm_curve(CNM_OO_5360_Y, "y_center")

    # Differential pT combination uses the CNM support as tabulated; no extrapolation in the exported curve.
    pt_factor_c = _interp_curve_variant(mode_result.pt_all, pt_curve, "central", hold_edges=True)
    pt_factor_lo = _interp_curve_variant(mode_result.pt_all, pt_curve, "lo", hold_edges=True)
    pt_factor_hi = _interp_curve_variant(mode_result.pt_all, pt_curve, "hi", hold_edges=True)
    pt_centers_all, pt_mean6_c, pt_sem6_c = _compute_scaled_raa_vs_pt_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        PT_EDGES,
        INTEGRATED_Y_WINDOW,
        pt_factor_c,
    )
    _, pt_mean6_lo, _ = _compute_scaled_raa_vs_pt_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        PT_EDGES,
        INTEGRATED_Y_WINDOW,
        pt_factor_lo,
    )
    _, pt_mean6_hi, _ = _compute_scaled_raa_vs_pt_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        PT_EDGES,
        INTEGRATED_Y_WINDOW,
        pt_factor_hi,
    )
    pt_raa9_c_all, pt_sem9_c_all = _apply_feeddown_binned(pt_mean6_c, pt_sem6_c, feeddown, sigmas_prim)
    pt_raa9_lo_all, _ = _apply_feeddown_binned(pt_mean6_lo, np.zeros_like(pt_mean6_lo), feeddown, sigmas_prim)
    pt_raa9_hi_all, _ = _apply_feeddown_binned(pt_mean6_hi, np.zeros_like(pt_mean6_hi), feeddown, sigmas_prim)
    pt_support_mask = _curve_support_mask(pt_centers_all, pt_curve)
    pt_low_all, pt_high_all, pt_sys_lo_all, pt_sys_hi_all = _total_band_from_curves(
        pt_raa9_c_all,
        pt_sem9_c_all,
        pt_raa9_lo_all,
        pt_raa9_hi_all,
    )

    # Differential y combination is defined across the full exported y range.
    y_factor_c = _interp_curve_variant(mode_result.y_all, y_curve, "central", hold_edges=False)
    y_factor_lo = _interp_curve_variant(mode_result.y_all, y_curve, "lo", hold_edges=False)
    y_factor_hi = _interp_curve_variant(mode_result.y_all, y_curve, "hi", hold_edges=False)
    y_centers, y_mean6_c, y_sem6_c = _compute_scaled_raa_vs_y_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        Y_EDGES,
        PT_MAX_FOR_Y,
        y_factor_c,
    )
    _, y_mean6_lo, _ = _compute_scaled_raa_vs_y_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        Y_EDGES,
        PT_MAX_FOR_Y,
        y_factor_lo,
    )
    _, y_mean6_hi, _ = _compute_scaled_raa_vs_y_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        Y_EDGES,
        PT_MAX_FOR_Y,
        y_factor_hi,
    )
    y_raa9_c, y_sem9_c = _apply_feeddown_binned(y_mean6_c, y_sem6_c, feeddown, sigmas_prim)
    y_raa9_lo, _ = _apply_feeddown_binned(y_mean6_lo, np.zeros_like(y_mean6_lo), feeddown, sigmas_prim)
    y_raa9_hi, _ = _apply_feeddown_binned(y_mean6_hi, np.zeros_like(y_mean6_hi), feeddown, sigmas_prim)
    y_low, y_high, y_sys_lo, y_sys_hi = _total_band_from_curves(y_raa9_c, y_sem9_c, y_raa9_lo, y_raa9_hi)

    # Integrated combinations: keep both pT- and y-projection prescriptions explicit.
    pt_factor_c_int = _interp_curve_variant(mode_result.pt_all, pt_curve, "central", hold_edges=True)
    pt_factor_lo_int = _interp_curve_variant(mode_result.pt_all, pt_curve, "lo", hold_edges=True)
    pt_factor_hi_int = _interp_curve_variant(mode_result.pt_all, pt_curve, "hi", hold_edges=True)
    mean6_pt_int_c, sem6_pt_int_c = _compute_scaled_integrated_raa_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        INTEGRATED_Y_WINDOW,
        PT_MAX_FOR_Y,
        pt_factor_c_int,
    )
    mean6_pt_int_lo, _ = _compute_scaled_integrated_raa_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        INTEGRATED_Y_WINDOW,
        PT_MAX_FOR_Y,
        pt_factor_lo_int,
    )
    mean6_pt_int_hi, _ = _compute_scaled_integrated_raa_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        INTEGRATED_Y_WINDOW,
        PT_MAX_FOR_Y,
        pt_factor_hi_int,
    )
    raa9_pt_int_c, sem9_pt_int_c = apply_feeddown_to_raa6(mean6_pt_int_c, sem6_pt_int_c, feeddown, sigmas_prim)
    raa9_pt_int_lo, _ = apply_feeddown_to_raa6(mean6_pt_int_lo, np.zeros_like(mean6_pt_int_lo), feeddown, sigmas_prim)
    raa9_pt_int_hi, _ = apply_feeddown_to_raa6(mean6_pt_int_hi, np.zeros_like(mean6_pt_int_hi), feeddown, sigmas_prim)
    int_pt_low, int_pt_high, int_pt_sys_lo, int_pt_sys_hi = _total_band_from_curves(
        raa9_pt_int_c,
        sem9_pt_int_c,
        raa9_pt_int_lo,
        raa9_pt_int_hi,
    )

    y_factor_c_int = _interp_curve_variant(mode_result.y_all, y_curve, "central", hold_edges=True)
    y_factor_lo_int = _interp_curve_variant(mode_result.y_all, y_curve, "lo", hold_edges=True)
    y_factor_hi_int = _interp_curve_variant(mode_result.y_all, y_curve, "hi", hold_edges=True)
    mean6_y_int_c, sem6_y_int_c = _compute_scaled_integrated_raa_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        INTEGRATED_Y_WINDOW,
        PT_MAX_FOR_Y,
        y_factor_c_int,
    )
    mean6_y_int_lo, _ = _compute_scaled_integrated_raa_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        INTEGRATED_Y_WINDOW,
        PT_MAX_FOR_Y,
        y_factor_lo_int,
    )
    mean6_y_int_hi, _ = _compute_scaled_integrated_raa_from_arrays(
        mode_result.pt_all,
        mode_result.y_all,
        mode_result.surv6_all,
        mode_result.qweights_all,
        INTEGRATED_Y_WINDOW,
        PT_MAX_FOR_Y,
        y_factor_hi_int,
    )
    raa9_y_int_c, sem9_y_int_c = apply_feeddown_to_raa6(mean6_y_int_c, sem6_y_int_c, feeddown, sigmas_prim)
    raa9_y_int_lo, _ = apply_feeddown_to_raa6(mean6_y_int_lo, np.zeros_like(mean6_y_int_lo), feeddown, sigmas_prim)
    raa9_y_int_hi, _ = apply_feeddown_to_raa6(mean6_y_int_hi, np.zeros_like(mean6_y_int_hi), feeddown, sigmas_prim)
    int_y_low, int_y_high, int_y_sys_lo, int_y_sys_hi = _total_band_from_curves(
        raa9_y_int_c,
        sem9_y_int_c,
        raa9_y_int_lo,
        raa9_y_int_hi,
    )

    integrated_mask = (
        (INTEGRATED_Y_WINDOW[0] <= mode_result.y_all)
        & (mode_result.y_all <= INTEGRATED_Y_WINDOW[1])
        & (mode_result.pt_all <= PT_MAX_FOR_Y)
    )
    tail_fraction = float(np.mean(mode_result.pt_all[integrated_mask] > float(pt_curve["x"][-1])))
    logger.info(
        "noReg QGPxCNM: pT-table support ends at %.1f GeV; %.3f of accepted trajectories are above that edge",
        float(pt_curve["x"][-1]),
        tail_fraction,
    )

    return {
        "pt_support_max_gev": float(pt_curve["x"][-1]),
        "pt_tail_fraction_for_integrated": tail_fraction,
        "pt": {
            "x": pt_centers_all[pt_support_mask],
            "edges": PT_EDGES[: int(np.sum(pt_support_mask)) + 1].copy(),
            "raa9_central": pt_raa9_c_all[pt_support_mask],
            "sem9_stat": pt_sem9_c_all[pt_support_mask],
            "raa9_lo_curve": pt_raa9_lo_all[pt_support_mask],
            "raa9_hi_curve": pt_raa9_hi_all[pt_support_mask],
            "band_lo": pt_low_all[pt_support_mask],
            "band_hi": pt_high_all[pt_support_mask],
            "sys_lo": pt_sys_lo_all[pt_support_mask],
            "sys_hi": pt_sys_hi_all[pt_support_mask],
        },
        "y": {
            "x": y_centers,
            "edges": Y_EDGES.copy(),
            "raa9_central": y_raa9_c,
            "sem9_stat": y_sem9_c,
            "raa9_lo_curve": y_raa9_lo,
            "raa9_hi_curve": y_raa9_hi,
            "band_lo": y_low,
            "band_hi": y_high,
            "sys_lo": y_sys_lo,
            "sys_hi": y_sys_hi,
        },
        "integrated": {
            "qgp_only": {
                "raa9": mode_result.raa9_int,
                "sem9_stat": mode_result.sem9_int,
            },
            "qgp_cnm_pt_projection": {
                "raa9": raa9_pt_int_c,
                "sem9_stat": sem9_pt_int_c,
                "band_lo": int_pt_low,
                "band_hi": int_pt_high,
                "sys_lo": int_pt_sys_lo,
                "sys_hi": int_pt_sys_hi,
            },
            "qgp_cnm_y_projection": {
                "raa9": raa9_y_int_c,
                "sem9_stat": sem9_y_int_c,
                "band_lo": int_y_low,
                "band_hi": int_y_high,
                "sys_lo": int_y_sys_lo,
                "sys_hi": int_y_sys_hi,
            },
        },
    }


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_noReg_hepdata_tables(
    mode_result: ModeResult,
    cnm_package: dict,
    out_data_dir: Path,
    logger: logging.Logger,
) -> None:
    _ensure_dir(out_data_dir)
    pt_result = {entry.key: entry for entry in mode_result.pt_results}["midrapidity"]
    pt_ratios = _compute_double_ratio_series(pt_result.pt_centers, pt_result.raa9_pt, pt_result.sem9_pt)
    y_ratios = _compute_double_ratio_series(mode_result.y_centers, mode_result.raa9_y, mode_result.sem9_y)
    int_ratios = _compute_double_ratio_series(
        np.array([B_MINBIAS], dtype=np.float64),
        mode_result.raa9_int,
        mode_result.sem9_int,
    )

    fieldnames = [
        "source",
        "state",
        "axis",
        "x_low",
        "x_high",
        "x_center",
        "value",
        "stat_err_minus",
        "stat_err_plus",
        "syst_err_minus",
        "syst_err_plus",
        "b_fm",
        "y_min",
        "y_max",
        "pt_max",
        "notes",
    ]
    pt_rows = []
    for idx, state_name, _, _ in STATE_SPECS_MAIN:
        for ibin in range(len(PT_EDGES) - 1):
            pt_rows.append(
                {
                    "source": "QGP_only",
                    "state": state_name,
                    "axis": "pT",
                    "x_low": float(PT_EDGES[ibin]),
                    "x_high": float(PT_EDGES[ibin + 1]),
                    "x_center": float(pt_result.pt_centers[ibin]),
                    "value": float(pt_result.raa9_pt[ibin, idx]),
                    "stat_err_minus": float(pt_result.sem9_pt[ibin, idx]),
                    "stat_err_plus": float(pt_result.sem9_pt[ibin, idx]),
                    "syst_err_minus": 0.0,
                    "syst_err_plus": 0.0,
                    "b_fm": B_MINBIAS,
                    "y_min": INTEGRATED_Y_WINDOW[0],
                    "y_max": INTEGRATED_Y_WINDOW[1],
                    "pt_max": PT_EDGES[-1],
                    "notes": "Fixed-b OO QTraj noReg; statistical SEM only",
                }
            )
        for ibin, x_center in enumerate(cnm_package["pt"]["x"]):
            pt_rows.append(
                {
                    "source": "QGPxCNM",
                    "state": state_name,
                    "axis": "pT",
                    "x_low": float(cnm_package["pt"]["edges"][ibin]),
                    "x_high": float(cnm_package["pt"]["edges"][ibin + 1]),
                    "x_center": float(x_center),
                    "value": float(cnm_package["pt"]["raa9_central"][ibin, idx]),
                    "stat_err_minus": float(cnm_package["pt"]["sem9_stat"][ibin, idx]),
                    "stat_err_plus": float(cnm_package["pt"]["sem9_stat"][ibin, idx]),
                    "syst_err_minus": float(cnm_package["pt"]["sys_lo"][ibin, idx]),
                    "syst_err_plus": float(cnm_package["pt"]["sys_hi"][ibin, idx]),
                    "b_fm": B_MINBIAS,
                    "y_min": INTEGRATED_Y_WINDOW[0],
                    "y_max": INTEGRATED_Y_WINDOW[1],
                    "pt_max": float(cnm_package["pt_support_max_gev"]),
                    "notes": "QGPxCNM; CNM support currently tabulated only through 19.5 GeV",
                }
            )
    pt_path = out_data_dir / "oo5360_hepdata_raa_vspt__noReg.csv"
    _write_csv_rows(pt_path, fieldnames, pt_rows)
    logger.info("Wrote %s", pt_path)

    y_rows = []
    for idx, state_name, _, _ in STATE_SPECS_MAIN:
        for ibin in range(len(Y_EDGES) - 1):
            y_rows.append(
                {
                    "source": "QGP_only",
                    "state": state_name,
                    "axis": "y",
                    "x_low": float(Y_EDGES[ibin]),
                    "x_high": float(Y_EDGES[ibin + 1]),
                    "x_center": float(mode_result.y_centers[ibin]),
                    "value": float(mode_result.raa9_y[ibin, idx]),
                    "stat_err_minus": float(mode_result.sem9_y[ibin, idx]),
                    "stat_err_plus": float(mode_result.sem9_y[ibin, idx]),
                    "syst_err_minus": 0.0,
                    "syst_err_plus": 0.0,
                    "b_fm": B_MINBIAS,
                    "y_min": Y_EDGES[0],
                    "y_max": Y_EDGES[-1],
                    "pt_max": PT_MAX_FOR_Y,
                    "notes": "Fixed-b OO QTraj noReg; statistical SEM only",
                }
            )
            y_rows.append(
                {
                    "source": "QGPxCNM",
                    "state": state_name,
                    "axis": "y",
                    "x_low": float(Y_EDGES[ibin]),
                    "x_high": float(Y_EDGES[ibin + 1]),
                    "x_center": float(cnm_package["y"]["x"][ibin]),
                    "value": float(cnm_package["y"]["raa9_central"][ibin, idx]),
                    "stat_err_minus": float(cnm_package["y"]["sem9_stat"][ibin, idx]),
                    "stat_err_plus": float(cnm_package["y"]["sem9_stat"][ibin, idx]),
                    "syst_err_minus": float(cnm_package["y"]["sys_lo"][ibin, idx]),
                    "syst_err_plus": float(cnm_package["y"]["sys_hi"][ibin, idx]),
                    "b_fm": B_MINBIAS,
                    "y_min": Y_EDGES[0],
                    "y_max": Y_EDGES[-1],
                    "pt_max": PT_MAX_FOR_Y,
                    "notes": "QGPxCNM from the OO min-bias CNM rapidity curve",
                }
            )
    y_path = out_data_dir / "oo5360_hepdata_raavsy__noReg.csv"
    _write_csv_rows(y_path, fieldnames, y_rows)
    logger.info("Wrote %s", y_path)

    int_rows = []
    tail_pct = 100.0 * float(cnm_package["pt_tail_fraction_for_integrated"])
    for idx, state_name, _, _ in STATE_SPECS_MAIN:
        int_rows.append(
            {
                "source": "QGP_only",
                "state": state_name,
                "axis": "integrated",
                "x_low": "",
                "x_high": "",
                "x_center": "",
                "value": float(mode_result.raa9_int[idx]),
                "stat_err_minus": float(mode_result.sem9_int[idx]),
                "stat_err_plus": float(mode_result.sem9_int[idx]),
                "syst_err_minus": 0.0,
                "syst_err_plus": 0.0,
                "b_fm": B_MINBIAS,
                "y_min": INTEGRATED_Y_WINDOW[0],
                "y_max": INTEGRATED_Y_WINDOW[1],
                "pt_max": PT_MAX_FOR_Y,
                "notes": "Integrated fixed-b OO QTraj noReg",
            }
        )
        for source, payload, note in (
            (
                "QGPxCNM_ptProjection",
                cnm_package["integrated"]["qgp_cnm_pt_projection"],
                f"Uses CNM(pT); last CNM bin held above 19.5 GeV for the {tail_pct:.2f}% high-pT tail",
            ),
            (
                "QGPxCNM_yProjection",
                cnm_package["integrated"]["qgp_cnm_y_projection"],
                "Uses CNM(y); current 1D CNM inputs do not define a unique 2D integrated factor",
            ),
        ):
            int_rows.append(
                {
                    "source": source,
                    "state": state_name,
                    "axis": "integrated",
                    "x_low": "",
                    "x_high": "",
                    "x_center": "",
                    "value": float(payload["raa9"][idx]),
                    "stat_err_minus": float(payload["sem9_stat"][idx]),
                    "stat_err_plus": float(payload["sem9_stat"][idx]),
                    "syst_err_minus": float(payload["sys_lo"][idx]),
                    "syst_err_plus": float(payload["sys_hi"][idx]),
                    "b_fm": B_MINBIAS,
                    "y_min": INTEGRATED_Y_WINDOW[0],
                    "y_max": INTEGRATED_Y_WINDOW[1],
                    "pt_max": PT_MAX_FOR_Y,
                    "notes": note,
                }
            )
    int_path = out_data_dir / "oo5360_hepdata_integrated__noReg.csv"
    _write_csv_rows(int_path, fieldnames, int_rows)
    logger.info("Wrote %s", int_path)

    ratio_fieldnames = [
        "ratio",
        "axis",
        "x_low",
        "x_high",
        "x_center",
        "value",
        "stat_err_minus",
        "stat_err_plus",
        "b_fm",
        "notes",
    ]
    ratio_rows_pt = []
    for ibin in range(len(PT_EDGES) - 1):
        for ratio_key, err_key, _, _, label_tex in DOUBLE_RATIO_SPECS:
            ratio_rows_pt.append(
                {
                    "ratio": label_tex,
                    "axis": "pT",
                    "x_low": float(PT_EDGES[ibin]),
                    "x_high": float(PT_EDGES[ibin + 1]),
                    "x_center": float(pt_ratios.x[ibin]),
                    "value": float(getattr(pt_ratios, ratio_key)[ibin]),
                    "stat_err_minus": float(getattr(pt_ratios, err_key)[ibin]),
                    "stat_err_plus": float(getattr(pt_ratios, err_key)[ibin]),
                    "b_fm": B_MINBIAS,
                    "notes": "CNM tables are common multiplicative Upsilon factors, so these double ratios are unchanged by QGPxCNM",
                }
            )
    ratio_pt_path = out_data_dir / "oo5360_hepdata_double_ratios_vspt__noReg.csv"
    _write_csv_rows(ratio_pt_path, ratio_fieldnames, ratio_rows_pt)
    logger.info("Wrote %s", ratio_pt_path)

    ratio_rows_y = []
    for ibin in range(len(Y_EDGES) - 1):
        for ratio_key, err_key, _, _, label_tex in DOUBLE_RATIO_SPECS:
            ratio_rows_y.append(
                {
                    "ratio": label_tex,
                    "axis": "y",
                    "x_low": float(Y_EDGES[ibin]),
                    "x_high": float(Y_EDGES[ibin + 1]),
                    "x_center": float(y_ratios.x[ibin]),
                    "value": float(getattr(y_ratios, ratio_key)[ibin]),
                    "stat_err_minus": float(getattr(y_ratios, err_key)[ibin]),
                    "stat_err_plus": float(getattr(y_ratios, err_key)[ibin]),
                    "b_fm": B_MINBIAS,
                    "notes": "CNM tables are common multiplicative Upsilon factors, so these double ratios are unchanged by QGPxCNM",
                }
            )
    ratio_y_path = out_data_dir / "oo5360_hepdata_double_ratios_vsy__noReg.csv"
    _write_csv_rows(ratio_y_path, ratio_fieldnames, ratio_rows_y)
    logger.info("Wrote %s", ratio_y_path)

    ratio_rows_int = []
    for ratio_key, err_key, _, _, label_tex in DOUBLE_RATIO_SPECS:
        ratio_rows_int.append(
            {
                "ratio": label_tex,
                "axis": "integrated",
                "x_low": "",
                "x_high": "",
                "x_center": "",
                "value": float(getattr(int_ratios, ratio_key)[0]),
                "stat_err_minus": float(getattr(int_ratios, err_key)[0]),
                "stat_err_plus": float(getattr(int_ratios, err_key)[0]),
                "b_fm": B_MINBIAS,
                "notes": "Integrated fixed-b OO QTraj noReg; unchanged under the current state-independent CNM factor",
            }
        )
    ratio_int_path = out_data_dir / "oo5360_hepdata_double_ratios_integrated__noReg.csv"
    _write_csv_rows(ratio_int_path, ratio_fieldnames, ratio_rows_int)
    logger.info("Wrote %s", ratio_int_path)


def _save_noReg_qgp_cnm_csvs(
    cnm_package: dict,
    out_data_dir: Path,
    logger: logging.Logger,
) -> None:
    _ensure_dir(out_data_dir)
    pt = cnm_package["pt"]
    y = cnm_package["y"]
    idxs = [idx for idx, _, _, _ in STATE_SPECS_MAIN]
    pt_cols = [pt["x"]]
    pt_header = ["x"]
    y_cols = [y["x"]]
    y_header = ["x"]
    for idx, state_name, _, _ in STATE_SPECS_MAIN:
        short = state_name.replace("Upsilon(", "").replace(")", "")
        pt_cols.extend(
            [
                pt["raa9_central"][:, idx],
                pt["sem9_stat"][:, idx],
                pt["band_lo"][:, idx],
                pt["band_hi"][:, idx],
            ]
        )
        pt_header.extend(
            [
                f"RAA_{short}",
                f"STAT_{short}",
                f"BAND_LO_{short}",
                f"BAND_HI_{short}",
            ]
        )
        y_cols.extend(
            [
                y["raa9_central"][:, idx],
                y["sem9_stat"][:, idx],
                y["band_lo"][:, idx],
                y["band_hi"][:, idx],
            ]
        )
        y_header.extend(
            [
                f"RAA_{short}",
                f"STAT_{short}",
                f"BAND_LO_{short}",
                f"BAND_HI_{short}",
            ]
        )
    pt_path = out_data_dir / "oo5360_raavspt__midrapidity__noReg_qgp_cnm.csv"
    np.savetxt(
        pt_path,
        np.column_stack(pt_cols),
        delimiter=",",
        header=",".join(pt_header),
        comments="",
        fmt="%.10g",
    )
    logger.info("Saved %s", pt_path)

    y_path = out_data_dir / "oo5360_raavsy__noReg_qgp_cnm.csv"
    np.savetxt(
        y_path,
        np.column_stack(y_cols),
        delimiter=",",
        header=",".join(y_header),
        comments="",
        fmt="%.10g",
    )
    logger.info("Saved %s", y_path)

    int_rows = []
    for source_key in ("qgp_cnm_pt_projection", "qgp_cnm_y_projection"):
        payload = cnm_package["integrated"][source_key]
        row = [payload["raa9"][idx] for idx in idxs]
        row += [payload["sem9_stat"][idx] for idx in idxs]
        row += [payload["band_lo"][idx] for idx in idxs]
        row += [payload["band_hi"][idx] for idx in idxs]
        int_rows.append(row)
    int_path = out_data_dir / "oo5360_integrated__noReg_qgp_cnm.csv"
    int_header = (
        "source,"
        "RAA_1S,RAA_2S,RAA_3S,"
        "STAT_1S,STAT_2S,STAT_3S,"
        "BAND_LO_1S,BAND_LO_2S,BAND_LO_3S,"
        "BAND_HI_1S,BAND_HI_2S,BAND_HI_3S"
    )
    with int_path.open("w", encoding="utf-8") as fh:
        fh.write(int_header + "\n")
        sources = ("QGPxCNM_ptProjection", "QGPxCNM_yProjection")
        for source, row in zip(sources, int_rows):
            fh.write(source + "," + ",".join(f"{float(v):.10g}" for v in row) + "\n")
    logger.info("Saved %s", int_path)


def _plot_overlay(noReg: ModeResult, wReg: ModeResult, out_fig_dir: Path, logger: logging.Logger) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _ensure_dir(out_fig_dir)

    # Indices on the 9-state basis.
    IDX_UPS = [(0, r"$\Upsilon(1S)$"), (1, r"$\Upsilon(2S)$"), (5, r"$\Upsilon(3S)$")]
    IDX_CHI = [(3, r"$\chi_{b1}(1P)$"), (7, r"$\chi_{b1}(2P)$")]

    colors = {
        0: "#1f77b4",
        1: "#ff7f0e",
        5: "#2ca02c",
        3: "#9467bd",
        7: "#d62728",
    }

    def draw_step_band(ax, bin_edges, centers, errs, *, color, ls, alpha, lw):
        x, y_step, y_min, y_max = binned_step_series(bin_edges, centers, errs)
        ax.fill_between(x, y_min, y_max, step="post", color=color, alpha=alpha, linewidth=0.0, zorder=1)
        ax.step(x, y_step, where="post", color=color, linestyle=ls, linewidth=lw, zorder=2)

    def decorate(ax, *, xlabel, xlim, ylim):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$R_{\mathrm{AA}}$")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axhline(1.0, color="0.5", ls="--", lw=0.8, zorder=0)
        ax.grid(alpha=0.2, lw=0.5)

    state_handles = [Line2D([0], [0], color=colors[i], lw=2.0, ls="-") for i, _ in IDX_UPS] + [
        Line2D([0], [0], color=colors[i], lw=2.0, ls="-") for i, _ in IDX_CHI
    ]
    state_labels = [lab for _, lab in IDX_UPS] + [lab for _, lab in IDX_CHI]

    mode_handles = [
        Line2D([0], [0], color="k", lw=2.0, ls=wReg.linestyle),
        Line2D([0], [0], color="k", lw=2.0, ls=noReg.linestyle),
    ]
    mode_labels = [wReg.description, noReg.description]
    no_reg_pt = {pt_result.key: pt_result for pt_result in noReg.pt_results}
    w_reg_pt = {pt_result.key: pt_result for pt_result in wReg.pt_results}

    for key, _, y_label in PT_RAPIDITY_WINDOWS:
        no_reg_window = no_reg_pt[key]
        w_reg_window = w_reg_pt[key]

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
        ax_u, ax_c = axes

        for idx, _ in IDX_UPS:
            color = colors[idx]
            draw_step_band(
                ax_u,
                PT_EDGES,
                no_reg_window.raa9_pt[:, idx],
                no_reg_window.sem9_pt[:, idx],
                color=color,
                ls=noReg.linestyle,
                alpha=0.10,
                lw=1.7,
            )
            draw_step_band(
                ax_u,
                PT_EDGES,
                w_reg_window.raa9_pt[:, idx],
                w_reg_window.sem9_pt[:, idx],
                color=color,
                ls=wReg.linestyle,
                alpha=0.16,
                lw=1.9,
            )

        for idx, _ in IDX_CHI:
            color = colors[idx]
            draw_step_band(
                ax_c,
                PT_EDGES,
                no_reg_window.raa9_pt[:, idx],
                no_reg_window.sem9_pt[:, idx],
                color=color,
                ls=noReg.linestyle,
                alpha=0.10,
                lw=1.7,
            )
            draw_step_band(
                ax_c,
                PT_EDGES,
                w_reg_window.raa9_pt[:, idx],
                w_reg_window.sem9_pt[:, idx],
                color=color,
                ls=wReg.linestyle,
                alpha=0.16,
                lw=1.9,
            )

        annotate = (
            r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
            + rf"$\hat{{\kappa}}={KAPPA_HAT},\ b={B_MINBIAS:.5g}\ \mathrm{{fm}}$" "\n"
            + y_label
        )
        ax_u.text(0.03, 0.97, annotate, transform=ax_u.transAxes, va="top", ha="left", fontsize=11)

        decorate(ax_u, xlabel=r"$p_T\ [\mathrm{GeV}]$", xlim=(PT_EDGES[0], PT_EDGES[-1]), ylim=(0.0, 1.6))
        decorate(ax_c, xlabel=r"$p_T\ [\mathrm{GeV}]$", xlim=(PT_EDGES[0], PT_EDGES[-1]), ylim=(0.0, 1.6))
        ax_u.set_title("S-wave states")
        ax_c.set_title("P-wave states")

        leg_states = ax_u.legend(state_handles, state_labels, loc="upper right", framealpha=0.95, title="States")
        ax_u.add_artist(leg_states)
        ax_u.legend(mode_handles, mode_labels, loc="lower right", framealpha=0.95, title="Mode")

        fig.tight_layout()
        pt_pdf = out_fig_dir / f"oo5360_raavspt__{key}__wReg_vs_noReg.pdf"
        pt_png = out_fig_dir / f"oo5360_raavspt__{key}__wReg_vs_noReg.png"
        fig.savefig(pt_pdf, bbox_inches="tight")
        fig.savefig(pt_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", pt_pdf)

    # y overlay
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    ax_u, ax_c = axes

    for idx, _ in IDX_UPS:
        color = colors[idx]
        draw_step_band(
            ax_u,
            Y_EDGES,
            noReg.raa9_y[:, idx],
            noReg.sem9_y[:, idx],
            color=color,
            ls=noReg.linestyle,
            alpha=0.10,
            lw=1.7,
        )
        draw_step_band(
            ax_u,
            Y_EDGES,
            wReg.raa9_y[:, idx],
            wReg.sem9_y[:, idx],
            color=color,
            ls=wReg.linestyle,
            alpha=0.16,
            lw=1.9,
        )

    for idx, _ in IDX_CHI:
        color = colors[idx]
        draw_step_band(
            ax_c,
            Y_EDGES,
            noReg.raa9_y[:, idx],
            noReg.sem9_y[:, idx],
            color=color,
            ls=noReg.linestyle,
            alpha=0.10,
            lw=1.7,
        )
        draw_step_band(
            ax_c,
            Y_EDGES,
            wReg.raa9_y[:, idx],
            wReg.sem9_y[:, idx],
            color=color,
            ls=wReg.linestyle,
            alpha=0.16,
            lw=1.9,
        )

    annotate = (
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$\hat{{\kappa}}={KAPPA_HAT},\ b={B_MINBIAS:.5g}\ \mathrm{{fm}}$" "\n"
        + rf"$p_T \leq {PT_MAX_FOR_Y:.0f}\ \mathrm{{GeV}}$"
    )
    ax_u.text(0.03, 0.97, annotate, transform=ax_u.transAxes, va="top", ha="left", fontsize=11)

    decorate(ax_u, xlabel=r"$y$", xlim=(Y_EDGES[0], Y_EDGES[-1]), ylim=(0.0, 1.6))
    decorate(ax_c, xlabel=r"$y$", xlim=(Y_EDGES[0], Y_EDGES[-1]), ylim=(0.0, 1.6))
    ax_u.set_title("S-wave states")
    ax_c.set_title("P-wave states")

    leg_states = ax_u.legend(state_handles, state_labels, loc="upper right", framealpha=0.95, title="States")
    ax_u.add_artist(leg_states)
    ax_u.legend(mode_handles, mode_labels, loc="lower right", framealpha=0.95, title="Mode")

    fig.tight_layout()
    y_pdf = out_fig_dir / "oo5360_raavsy__wReg_vs_noReg.pdf"
    y_png = out_fig_dir / "oo5360_raavsy__wReg_vs_noReg.png"
    fig.savefig(y_pdf, bbox_inches="tight")
    fig.savefig(y_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", y_pdf)

    # Raw matched-trajectory population histograms used to diagnose fluctuations.
    y_hist_edges = Y_EDGES
    pt_hist_edges = np.arange(0.0, float(PT_EDGES[-1]) + 1.0, 1.0, dtype=np.float64)

    fig, (ax_y, ax_pt) = plt.subplots(1, 2, figsize=(12.5, 4.5))

    ax_y.hist(
        noReg.y_all,
        bins=y_hist_edges,
        histtype="step",
        linewidth=1.8,
        linestyle=noReg.linestyle,
        color="#1f77b4",
        label=f"{noReg.description} (n={noReg.n_trajectories})",
    )
    ax_y.hist(
        wReg.y_all,
        bins=y_hist_edges,
        histtype="step",
        linewidth=1.8,
        linestyle=wReg.linestyle,
        color="#d62728",
        label=f"{wReg.description} (n={wReg.n_trajectories})",
    )
    ax_y.set_xlabel(r"$y$")
    ax_y.set_ylabel("Matched physical trajectories")
    ax_y.set_xlim(Y_EDGES[0], Y_EDGES[-1])
    ax_y.grid(alpha=0.2, lw=0.5)
    ax_y.legend(loc="upper center", framealpha=0.95, fontsize=10)

    ax_pt.hist(
        noReg.pt_all,
        bins=pt_hist_edges,
        histtype="step",
        linewidth=1.8,
        linestyle=noReg.linestyle,
        color="#1f77b4",
        label=f"{noReg.description} (n={noReg.n_trajectories})",
    )
    ax_pt.hist(
        wReg.pt_all,
        bins=pt_hist_edges,
        histtype="step",
        linewidth=1.8,
        linestyle=wReg.linestyle,
        color="#d62728",
        label=f"{wReg.description} (n={wReg.n_trajectories})",
    )
    ax_pt.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
    ax_pt.set_ylabel("Matched physical trajectories")
    ax_pt.set_xlim(0.0, max(float(PT_EDGES[-1]), PT_MAX_FOR_Y))
    ax_pt.grid(alpha=0.2, lw=0.5)
    ax_pt.legend(loc="upper right", framealpha=0.95, fontsize=10)

    ax_y.text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$\hat{{\kappa}}={KAPPA_HAT},\ b={B_MINBIAS:.5g}\ \mathrm{{fm}}$" "\n"
        + r"Matched physical-trajectory populations",
        transform=ax_y.transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )

    fig.tight_layout()
    hist_pdf = out_fig_dir / "oo5360_population_histograms__wReg_vs_noReg.pdf"
    hist_png = out_fig_dir / "oo5360_population_histograms__wReg_vs_noReg.png"
    fig.savefig(hist_pdf, bbox_inches="tight")
    fig.savefig(hist_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", hist_pdf)


def _finite_plot_mask(y: np.ndarray, err: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    return np.isfinite(y) & np.isfinite(err)


def _step_band_symmetric(
    ax,
    bin_edges: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    *,
    color: str,
    linestyle: str,
    label: str,
    alpha: float = 0.14,
    linewidth: float = 2.0,
) -> None:
    y = np.asarray(y, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    mask = _finite_plot_mask(y, err)
    if not np.any(mask):
        return
    if not np.all(mask):
        idx = np.where(mask)[0]
        start = idx[0]
        stop = idx[-1] + 1
        y = y[start:stop]
        err = err[start:stop]
        bin_edges = np.asarray(bin_edges, dtype=np.float64)[start : stop + 1]
    x, y_step, y_min, y_max = binned_step_series(np.asarray(bin_edges, dtype=np.float64), y, err)
    ax.fill_between(x, y_min, y_max, step="post", color=color, alpha=alpha, linewidth=0.0)
    ax.step(x, y_step, where="post", color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def _step_band_asymmetric(
    ax,
    bin_edges: np.ndarray,
    y: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    *,
    color: str,
    linestyle: str,
    label: str,
    alpha: float = 0.14,
    linewidth: float = 2.0,
) -> None:
    y = np.asarray(y, dtype=np.float64)
    y_lo = np.asarray(y_lo, dtype=np.float64)
    y_hi = np.asarray(y_hi, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(y_lo) & np.isfinite(y_hi)
    if not np.any(mask):
        return
    if not np.all(mask):
        idx = np.where(mask)[0]
        start = idx[0]
        stop = idx[-1] + 1
        y = y[start:stop]
        y_lo = y_lo[start:stop]
        y_hi = y_hi[start:stop]
        bin_edges = np.asarray(bin_edges, dtype=np.float64)[start : stop + 1]
    x = np.asarray(bin_edges, dtype=np.float64)
    y_step = np.concatenate([y, [y[-1]]])
    y_lo_step = np.concatenate([y_lo, [y_lo[-1]]])
    y_hi_step = np.concatenate([y_hi, [y_hi[-1]]])
    ax.fill_between(x, y_lo_step, y_hi_step, step="post", color=color, alpha=alpha, linewidth=0.0)
    ax.step(x, y_step, where="post", color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def _theory_band(ax, x: np.ndarray, y: np.ndarray, err: np.ndarray, *, color: str, linestyle: str, label: str) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    mask = _finite_plot_mask(y, err)
    if not np.any(mask):
        return
    ax.fill_between(x[mask], y[mask] - err[mask], y[mask] + err[mask], color=color, alpha=0.14, linewidth=0.0)
    ax.plot(x[mask], y[mask], color=color, linestyle=linestyle, linewidth=2.0, label=label)


def _ratio_ylim(*arrays: np.ndarray) -> Tuple[float, float]:
    vmax = 0.0
    for arr in arrays:
        values = np.asarray(arr, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            vmax = max(vmax, float(np.max(finite)))
    upper = max(0.8, min(1.8, 1.15 * vmax + 0.08))
    return 0.0, upper


def _plot_double_ratios(noReg: ModeResult, wReg: ModeResult, out_fig_dir: Path, logger: logging.Logger) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_fig_dir)

    no_reg_pt = {pt_result.key: pt_result for pt_result in noReg.pt_results}
    w_reg_pt = {pt_result.key: pt_result for pt_result in wReg.pt_results}
    mid_no = no_reg_pt["midrapidity"]
    mid_w = w_reg_pt["midrapidity"]

    pt_no = _compute_double_ratio_series(mid_no.pt_centers, mid_no.raa9_pt, mid_no.sem9_pt)
    pt_w = _compute_double_ratio_series(mid_w.pt_centers, mid_w.raa9_pt, mid_w.sem9_pt)
    y_no = _compute_double_ratio_series(noReg.y_centers, noReg.raa9_y, noReg.sem9_y)
    y_w = _compute_double_ratio_series(wReg.y_centers, wReg.raa9_y, wReg.sem9_y)
    int_no = _compute_double_ratio_series(np.array([B_MINBIAS], dtype=np.float64), noReg.raa9_int, noReg.sem9_int)
    int_w = _compute_double_ratio_series(np.array([B_MINBIAS], dtype=np.float64), wReg.raa9_int, wReg.sem9_int)
    exp_data = _load_cms_preliminary_approx()

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

    theory_colors = {"noReg": "#1f77b4", "wReg": "#d62728", "exp": "#111111"}

    fig_pt, axes_pt = plt.subplots(3, 1, figsize=(7.3, 9.0), sharex=True)
    exp_pt_32 = ()
    if exp_data is not None:
        exp_pt_32 = tuple(exp_data.get("vs_pt", {}).get("ratio_3S_2S", ()))
    for ax, (ratio_key, err_key, _, _, label_tex) in zip(axes_pt, DOUBLE_RATIO_SPECS):
        _theory_band(
            ax,
            pt_no.x,
            getattr(pt_no, ratio_key),
            getattr(pt_no, err_key),
            color=theory_colors["noReg"],
            linestyle=noReg.linestyle,
            label=noReg.description,
        )
        _theory_band(
            ax,
            pt_w.x,
            getattr(pt_w, ratio_key),
            getattr(pt_w, err_key),
            color=theory_colors["wReg"],
            linestyle=wReg.linestyle,
            label=wReg.description,
        )
        if ratio_key == "ratio_3S_2S" and exp_pt_32:
            exp_x = np.array([row["pt_center_gev"] for row in exp_pt_32], dtype=np.float64)
            exp_y = np.array([row["value"] for row in exp_pt_32], dtype=np.float64)
            exp_err = np.array([np.hypot(row["stat"], row["syst"]) for row in exp_pt_32], dtype=np.float64)
            ax.errorbar(
                exp_x,
                exp_y,
                yerr=exp_err,
                fmt="s",
                ms=5,
                capsize=2,
                color=theory_colors["exp"],
                label="CMS prelim. approx.",
                zorder=5,
            )
        ax.set_ylabel(label_tex)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_ylim(
            *_ratio_ylim(
                getattr(pt_no, ratio_key),
                getattr(pt_no, ratio_key) + getattr(pt_no, err_key),
                getattr(pt_w, ratio_key),
                getattr(pt_w, ratio_key) + getattr(pt_w, err_key),
            )
        )
    axes_pt[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$\hat{{\kappa}}={KAPPA_HAT},\ b={B_MINBIAS:.5g}\ \mathrm{{fm}}$" "\n"
        + r"$-2.4 \leq y \leq 2.4$",
        transform=axes_pt[0].transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    axes_pt[0].legend(loc="upper right", framealpha=0.95)
    axes_pt[-1].set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
    axes_pt[-1].set_xlim(PT_EDGES[0], PT_EDGES[-1])
    fig_pt.tight_layout()
    pt_pdf = out_fig_dir / "oo5360_double_ratios_vspt__midrapidity__wReg_vs_noReg.pdf"
    pt_png = out_fig_dir / "oo5360_double_ratios_vspt__midrapidity__wReg_vs_noReg.png"
    fig_pt.savefig(pt_pdf, bbox_inches="tight")
    fig_pt.savefig(pt_png, dpi=200, bbox_inches="tight")
    plt.close(fig_pt)
    logger.info("Saved %s", pt_pdf)

    fig_y, axes_y = plt.subplots(3, 1, figsize=(7.3, 9.0), sharex=True)
    for ax, (ratio_key, err_key, _, _, label_tex) in zip(axes_y, DOUBLE_RATIO_SPECS):
        _theory_band(
            ax,
            y_no.x,
            getattr(y_no, ratio_key),
            getattr(y_no, err_key),
            color=theory_colors["noReg"],
            linestyle=noReg.linestyle,
            label=noReg.description,
        )
        _theory_band(
            ax,
            y_w.x,
            getattr(y_w, ratio_key),
            getattr(y_w, err_key),
            color=theory_colors["wReg"],
            linestyle=wReg.linestyle,
            label=wReg.description,
        )
        ax.set_ylabel(label_tex)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_ylim(
            *_ratio_ylim(
                getattr(y_no, ratio_key),
                getattr(y_no, ratio_key) + getattr(y_no, err_key),
                getattr(y_w, ratio_key),
                getattr(y_w, ratio_key) + getattr(y_w, err_key),
            )
        )
    axes_y[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$\hat{{\kappa}}={KAPPA_HAT},\ b={B_MINBIAS:.5g}\ \mathrm{{fm}}$" "\n"
        + rf"$p_T \leq {PT_MAX_FOR_Y:.0f}\ \mathrm{{GeV}}$",
        transform=axes_y[0].transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    axes_y[0].legend(loc="upper right", framealpha=0.95)
    axes_y[-1].set_xlabel(r"$y$")
    axes_y[-1].set_xlim(Y_EDGES[0], Y_EDGES[-1])
    fig_y.tight_layout()
    y_pdf = out_fig_dir / "oo5360_double_ratios_vsy__wReg_vs_noReg.pdf"
    y_png = out_fig_dir / "oo5360_double_ratios_vsy__wReg_vs_noReg.png"
    fig_y.savefig(y_pdf, bbox_inches="tight")
    fig_y.savefig(y_png, dpi=200, bbox_inches="tight")
    plt.close(fig_y)
    logger.info("Saved %s", y_pdf)

    fig_int, ax_int = plt.subplots(figsize=(7.2, 4.8))
    xpos = np.arange(len(DOUBLE_RATIO_SPECS), dtype=np.float64)
    ax_int.errorbar(
        xpos - 0.16,
        np.array([int_no.ratio_2S_1S[0], int_no.ratio_3S_1S[0], int_no.ratio_3S_2S[0]], dtype=np.float64),
        yerr=np.array([int_no.err_2S_1S[0], int_no.err_3S_1S[0], int_no.err_3S_2S[0]], dtype=np.float64),
        fmt="o",
        ms=5,
        capsize=2,
        color=theory_colors["noReg"],
        label=noReg.description,
    )
    ax_int.errorbar(
        xpos + 0.16,
        np.array([int_w.ratio_2S_1S[0], int_w.ratio_3S_1S[0], int_w.ratio_3S_2S[0]], dtype=np.float64),
        yerr=np.array([int_w.err_2S_1S[0], int_w.err_3S_1S[0], int_w.err_3S_2S[0]], dtype=np.float64),
        fmt="o",
        ms=5,
        capsize=2,
        color=theory_colors["wReg"],
        label=wReg.description,
    )
    if exp_data is not None:
        exp_int = exp_data.get("integrated", {})
        exp_y = np.array(
            [
                exp_int["ratio_2S_1S"]["value"],
                exp_int["ratio_3S_1S"]["value"],
                exp_int["ratio_3S_2S"]["value"],
            ],
            dtype=np.float64,
        )
        exp_err = np.array(
            [
                np.hypot(exp_int["ratio_2S_1S"]["stat"], exp_int["ratio_2S_1S"]["syst"]),
                np.hypot(exp_int["ratio_3S_1S"]["stat"], exp_int["ratio_3S_1S"]["syst"]),
                np.hypot(exp_int["ratio_3S_2S"]["stat"], exp_int["ratio_3S_2S"]["syst"]),
            ],
            dtype=np.float64,
        )
        ax_int.errorbar(
            xpos,
            exp_y,
            yerr=exp_err,
            fmt="s",
            ms=5,
            capsize=2,
            color=theory_colors["exp"],
            label="CMS prelim. approx.",
        )
    ax_int.set_xticks(xpos, ["2S/1S", "3S/1S", "3S/2S"])
    ax_int.set_ylabel("Double Ratio")
    ax_int.set_ylim(
        *_ratio_ylim(
            np.array([int_no.ratio_2S_1S[0], int_no.ratio_3S_1S[0], int_no.ratio_3S_2S[0]], dtype=np.float64),
            np.array([int_w.ratio_2S_1S[0], int_w.ratio_3S_1S[0], int_w.ratio_3S_2S[0]], dtype=np.float64),
        )
    )
    ax_int.grid(alpha=0.2, lw=0.5)
    ax_int.text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + r"$-2.4 \leq y \leq 2.4,\ p_T \leq 30\ \mathrm{GeV}$" "\n"
        + r"Single fixed-$b$ point (no Glauber weighting)",
        transform=ax_int.transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    ax_int.legend(loc="upper right", framealpha=0.95)
    fig_int.tight_layout()
    int_pdf = out_fig_dir / "oo5360_double_ratios_integrated__theory_vs_cms_prelim.pdf"
    int_png = out_fig_dir / "oo5360_double_ratios_integrated__theory_vs_cms_prelim.png"
    fig_int.savefig(int_pdf, bbox_inches="tight")
    fig_int.savefig(int_png, dpi=200, bbox_inches="tight")
    plt.close(fig_int)
    logger.info("Saved %s", int_pdf)

    if exp_pt_32:
        fig_cmp, ax_cmp = plt.subplots(figsize=(7.2, 4.8))
        _theory_band(
            ax_cmp,
            pt_no.x,
            pt_no.ratio_3S_2S,
            pt_no.err_3S_2S,
            color=theory_colors["noReg"],
            linestyle=noReg.linestyle,
            label=noReg.description,
        )
        _theory_band(
            ax_cmp,
            pt_w.x,
            pt_w.ratio_3S_2S,
            pt_w.err_3S_2S,
            color=theory_colors["wReg"],
            linestyle=wReg.linestyle,
            label=wReg.description,
        )
        exp_x = np.array([row["pt_center_gev"] for row in exp_pt_32], dtype=np.float64)
        exp_y = np.array([row["value"] for row in exp_pt_32], dtype=np.float64)
        exp_err = np.array([np.hypot(row["stat"], row["syst"]) for row in exp_pt_32], dtype=np.float64)
        ax_cmp.errorbar(
            exp_x,
            exp_y,
            yerr=exp_err,
            fmt="s",
            ms=5,
            capsize=2,
            color=theory_colors["exp"],
            label="CMS prelim. approx.",
        )
        ax_cmp.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
        ax_cmp.set_ylabel(r"$\Upsilon(3S)/\Upsilon(2S)$")
        ax_cmp.set_xlim(PT_EDGES[0], PT_EDGES[-1])
        ax_cmp.set_ylim(
            *_ratio_ylim(
                pt_no.ratio_3S_2S,
                pt_no.ratio_3S_2S + pt_no.err_3S_2S,
                pt_w.ratio_3S_2S,
                pt_w.ratio_3S_2S + pt_w.err_3S_2S,
                exp_y + exp_err,
            )
        )
        ax_cmp.grid(alpha=0.2, lw=0.5)
        ax_cmp.text(
            0.03,
            0.97,
            r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
            + r"$-2.4 \leq y \leq 2.4$" "\n"
            + r"CMS points: approximate digitization",
            transform=ax_cmp.transAxes,
            va="top",
            ha="left",
            fontsize=11,
        )
        ax_cmp.legend(loc="upper right", framealpha=0.95)
        fig_cmp.tight_layout()
        cmp_pdf = out_fig_dir / "oo5360_double_ratio_3S_2S_vspt__theory_vs_cms_prelim.pdf"
        cmp_png = out_fig_dir / "oo5360_double_ratio_3S_2S_vspt__theory_vs_cms_prelim.png"
        fig_cmp.savefig(cmp_pdf, bbox_inches="tight")
        fig_cmp.savefig(cmp_png, dpi=200, bbox_inches="tight")
        plt.close(fig_cmp)
        logger.info("Saved %s", cmp_pdf)


def _plot_single_mode(mode_result: ModeResult, out_fig_dir: Path, logger: logging.Logger) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_fig_dir)

    pt_result = {entry.key: entry for entry in mode_result.pt_results}["midrapidity"]
    pt_ratios = _compute_double_ratio_series(pt_result.pt_centers, pt_result.raa9_pt, pt_result.sem9_pt)
    y_ratios = _compute_double_ratio_series(mode_result.y_centers, mode_result.raa9_y, mode_result.sem9_y)
    int_ratios = _compute_double_ratio_series(
        np.array([B_MINBIAS], dtype=np.float64),
        mode_result.raa9_int,
        mode_result.sem9_int,
    )

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

    fig_pt, ax_pt = plt.subplots(figsize=(7.0, 4.8))
    for idx, _, label_tex, color in STATE_SPECS_MAIN:
        _step_band_symmetric(
            ax_pt,
            PT_EDGES,
            pt_result.raa9_pt[:, idx],
            pt_result.sem9_pt[:, idx],
            color=color,
            linestyle=mode_result.linestyle,
            label=label_tex,
        )
    ax_pt.set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
    ax_pt.set_ylabel(r"$R_{\mathrm{AA}}$")
    ax_pt.set_xlim(PT_EDGES[0], PT_EDGES[-1])
    ax_pt.set_ylim(0.0, 1.6)
    ax_pt.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax_pt.grid(alpha=0.2, lw=0.5)
    ax_pt.text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$\hat{{\kappa}}={KAPPA_HAT},\ b={B_MINBIAS:.5g}\ \mathrm{{fm}}$" "\n"
        + mode_result.description + "\n"
        + r"$-2.4 \leq y \leq 2.4$",
        transform=ax_pt.transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    ax_pt.legend(loc="upper right", framealpha=0.95)
    fig_pt.tight_layout()
    pt_pdf = out_fig_dir / f"oo5360_raavspt__midrapidity__{mode_result.label}.pdf"
    pt_png = out_fig_dir / f"oo5360_raavspt__midrapidity__{mode_result.label}.png"
    fig_pt.savefig(pt_pdf, bbox_inches="tight")
    fig_pt.savefig(pt_png, dpi=200, bbox_inches="tight")
    plt.close(fig_pt)
    logger.info("Saved %s", pt_pdf)

    fig_y, ax_y = plt.subplots(figsize=(7.0, 4.8))
    for idx, _, label_tex, color in STATE_SPECS_MAIN:
        _step_band_symmetric(
            ax_y,
            Y_EDGES,
            mode_result.raa9_y[:, idx],
            mode_result.sem9_y[:, idx],
            color=color,
            linestyle=mode_result.linestyle,
            label=label_tex,
        )
    ax_y.set_xlabel(r"$y$")
    ax_y.set_ylabel(r"$R_{\mathrm{AA}}$")
    ax_y.set_xlim(Y_EDGES[0], Y_EDGES[-1])
    ax_y.set_ylim(0.0, 1.6)
    ax_y.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax_y.grid(alpha=0.2, lw=0.5)
    ax_y.text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$\hat{{\kappa}}={KAPPA_HAT},\ b={B_MINBIAS:.5g}\ \mathrm{{fm}}$" "\n"
        + mode_result.description + "\n"
        + rf"$p_T \leq {PT_MAX_FOR_Y:.0f}\ \mathrm{{GeV}}$",
        transform=ax_y.transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    ax_y.legend(loc="upper right", framealpha=0.95)
    fig_y.tight_layout()
    y_pdf = out_fig_dir / f"oo5360_raavsy__{mode_result.label}.pdf"
    y_png = out_fig_dir / f"oo5360_raavsy__{mode_result.label}.png"
    fig_y.savefig(y_pdf, bbox_inches="tight")
    fig_y.savefig(y_png, dpi=200, bbox_inches="tight")
    plt.close(fig_y)
    logger.info("Saved %s", y_pdf)

    fig_dr_pt, axes_dr_pt = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    for ax, (ratio_key, err_key, _, _, label_tex) in zip(axes_dr_pt, DOUBLE_RATIO_SPECS):
        _step_band_symmetric(
            ax,
            PT_EDGES,
            getattr(pt_ratios, ratio_key),
            getattr(pt_ratios, err_key),
            color="#1f77b4",
            linestyle=mode_result.linestyle,
            label=mode_result.description,
        )
        ax.set_ylabel(label_tex)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_ylim(*_ratio_ylim(getattr(pt_ratios, ratio_key), getattr(pt_ratios, ratio_key) + getattr(pt_ratios, err_key)))
    axes_dr_pt[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + mode_result.description + "\n"
        + r"$-2.4 \leq y \leq 2.4$",
        transform=axes_dr_pt[0].transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    axes_dr_pt[0].legend(loc="upper right", framealpha=0.95)
    axes_dr_pt[-1].set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
    axes_dr_pt[-1].set_xlim(PT_EDGES[0], PT_EDGES[-1])
    fig_dr_pt.tight_layout()
    dr_pt_pdf = out_fig_dir / f"oo5360_double_ratios_vspt__midrapidity__{mode_result.label}.pdf"
    dr_pt_png = out_fig_dir / f"oo5360_double_ratios_vspt__midrapidity__{mode_result.label}.png"
    fig_dr_pt.savefig(dr_pt_pdf, bbox_inches="tight")
    fig_dr_pt.savefig(dr_pt_png, dpi=200, bbox_inches="tight")
    plt.close(fig_dr_pt)
    logger.info("Saved %s", dr_pt_pdf)

    fig_dr_y, axes_dr_y = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    for ax, (ratio_key, err_key, _, _, label_tex) in zip(axes_dr_y, DOUBLE_RATIO_SPECS):
        _step_band_symmetric(
            ax,
            Y_EDGES,
            getattr(y_ratios, ratio_key),
            getattr(y_ratios, err_key),
            color="#1f77b4",
            linestyle=mode_result.linestyle,
            label=mode_result.description,
        )
        ax.set_ylabel(label_tex)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_ylim(*_ratio_ylim(getattr(y_ratios, ratio_key), getattr(y_ratios, ratio_key) + getattr(y_ratios, err_key)))
    axes_dr_y[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + mode_result.description + "\n"
        + rf"$p_T \leq {PT_MAX_FOR_Y:.0f}\ \mathrm{{GeV}}$",
        transform=axes_dr_y[0].transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    axes_dr_y[0].legend(loc="upper right", framealpha=0.95)
    axes_dr_y[-1].set_xlabel(r"$y$")
    axes_dr_y[-1].set_xlim(Y_EDGES[0], Y_EDGES[-1])
    fig_dr_y.tight_layout()
    dr_y_pdf = out_fig_dir / f"oo5360_double_ratios_vsy__{mode_result.label}.pdf"
    dr_y_png = out_fig_dir / f"oo5360_double_ratios_vsy__{mode_result.label}.png"
    fig_dr_y.savefig(dr_y_pdf, bbox_inches="tight")
    fig_dr_y.savefig(dr_y_png, dpi=200, bbox_inches="tight")
    plt.close(fig_dr_y)
    logger.info("Saved %s", dr_y_pdf)

    fig_int, ax_int = plt.subplots(figsize=(6.8, 4.6))
    x = np.arange(3, dtype=np.float64)
    y = np.array([int_ratios.ratio_2S_1S[0], int_ratios.ratio_3S_1S[0], int_ratios.ratio_3S_2S[0]], dtype=np.float64)
    e = np.array([int_ratios.err_2S_1S[0], int_ratios.err_3S_1S[0], int_ratios.err_3S_2S[0]], dtype=np.float64)
    ax_int.errorbar(x, y, yerr=e, fmt="o", ms=5, capsize=2, color="#1f77b4")
    ax_int.set_xticks(x, ["2S/1S", "3S/1S", "3S/2S"])
    ax_int.set_ylabel("Integrated Double Ratio")
    ax_int.set_ylim(*_ratio_ylim(y, y + e))
    ax_int.grid(alpha=0.2, lw=0.5)
    ax_int.text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + mode_result.description + "\n"
        + r"$-2.4 \leq y \leq 2.4,\ p_T \leq 30\ \mathrm{GeV}$",
        transform=ax_int.transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    fig_int.tight_layout()
    int_pdf = out_fig_dir / f"oo5360_double_ratios_integrated__{mode_result.label}.pdf"
    int_png = out_fig_dir / f"oo5360_double_ratios_integrated__{mode_result.label}.png"
    fig_int.savefig(int_pdf, bbox_inches="tight")
    fig_int.savefig(int_png, dpi=200, bbox_inches="tight")
    plt.close(fig_int)
    logger.info("Saved %s", int_pdf)


def _plot_noReg_qgp_vs_qgp_cnm(
    mode_result: ModeResult,
    cnm_package: dict,
    out_fig_dir: Path,
    logger: logging.Logger,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_fig_dir)
    pt_result = {entry.key: entry for entry in mode_result.pt_results}["midrapidity"]

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

    qgp_color = "#1f77b4"
    combo_color = "#111111"

    fig_pt, axes_pt = plt.subplots(3, 1, figsize=(7.4, 9.2), sharex=True)
    for ax, (idx, _, label_tex, _) in zip(axes_pt, STATE_SPECS_MAIN):
        _step_band_symmetric(
            ax,
            PT_EDGES,
            pt_result.raa9_pt[:, idx],
            pt_result.sem9_pt[:, idx],
            color=qgp_color,
            linestyle="--",
            label="QGP only (noReg)",
        )
        _step_band_asymmetric(
            ax,
            cnm_package["pt"]["edges"],
            cnm_package["pt"]["raa9_central"][:, idx],
            cnm_package["pt"]["band_lo"][:, idx],
            cnm_package["pt"]["band_hi"][:, idx],
            color=combo_color,
            linestyle="-",
            label="QGP × CNM",
        )
        ax.set_ylabel(label_tex)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_ylim(0.0, 1.2)
    axes_pt[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$\hat{{\kappa}}={KAPPA_HAT},\ b={B_MINBIAS:.5g}\ \mathrm{{fm}}$" "\n"
        + r"$-2.4 \leq y \leq 2.4$" "\n"
        + r"CNM input tabulated through $p_T=19.5$ GeV",
        transform=axes_pt[0].transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    axes_pt[0].legend(loc="upper right", framealpha=0.95)
    axes_pt[-1].set_xlabel(r"$p_T\ [\mathrm{GeV}]$")
    axes_pt[-1].set_xlim(PT_EDGES[0], PT_EDGES[-1])
    fig_pt.tight_layout()
    pt_pdf = out_fig_dir / "oo5360_raavspt__midrapidity__noReg__qgp_vs_qgp_cnm.pdf"
    pt_png = out_fig_dir / "oo5360_raavspt__midrapidity__noReg__qgp_vs_qgp_cnm.png"
    fig_pt.savefig(pt_pdf, bbox_inches="tight")
    fig_pt.savefig(pt_png, dpi=200, bbox_inches="tight")
    plt.close(fig_pt)
    logger.info("Saved %s", pt_pdf)

    fig_y, axes_y = plt.subplots(3, 1, figsize=(7.4, 9.2), sharex=True)
    for ax, (idx, _, label_tex, _) in zip(axes_y, STATE_SPECS_MAIN):
        _step_band_symmetric(
            ax,
            Y_EDGES,
            mode_result.raa9_y[:, idx],
            mode_result.sem9_y[:, idx],
            color=qgp_color,
            linestyle="--",
            label="QGP only (noReg)",
        )
        _step_band_asymmetric(
            ax,
            cnm_package["y"]["edges"],
            cnm_package["y"]["raa9_central"][:, idx],
            cnm_package["y"]["band_lo"][:, idx],
            cnm_package["y"]["band_hi"][:, idx],
            color=combo_color,
            linestyle="-",
            label="QGP × CNM",
        )
        ax.set_ylabel(label_tex)
        ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_ylim(0.0, 1.2)
    axes_y[0].text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + rf"$\hat{{\kappa}}={KAPPA_HAT},\ b={B_MINBIAS:.5g}\ \mathrm{{fm}}$" "\n"
        + rf"$p_T \leq {PT_MAX_FOR_Y:.0f}\ \mathrm{{GeV}}$",
        transform=axes_y[0].transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    axes_y[0].legend(loc="upper right", framealpha=0.95)
    axes_y[-1].set_xlabel(r"$y$")
    axes_y[-1].set_xlim(Y_EDGES[0], Y_EDGES[-1])
    fig_y.tight_layout()
    y_pdf = out_fig_dir / "oo5360_raavsy__noReg__qgp_vs_qgp_cnm.pdf"
    y_png = out_fig_dir / "oo5360_raavsy__noReg__qgp_vs_qgp_cnm.png"
    fig_y.savefig(y_pdf, bbox_inches="tight")
    fig_y.savefig(y_png, dpi=200, bbox_inches="tight")
    plt.close(fig_y)
    logger.info("Saved %s", y_pdf)

    fig_int, ax_int = plt.subplots(figsize=(7.1, 4.8))
    xpos = np.arange(len(STATE_SPECS_MAIN), dtype=np.float64)
    qgp_vals = np.array([mode_result.raa9_int[idx] for idx, _, _, _ in STATE_SPECS_MAIN], dtype=np.float64)
    qgp_err = np.array([mode_result.sem9_int[idx] for idx, _, _, _ in STATE_SPECS_MAIN], dtype=np.float64)
    pt_vals = np.array(
        [cnm_package["integrated"]["qgp_cnm_pt_projection"]["raa9"][idx] for idx, _, _, _ in STATE_SPECS_MAIN],
        dtype=np.float64,
    )
    pt_lo = np.array(
        [
            cnm_package["integrated"]["qgp_cnm_pt_projection"]["raa9"][idx]
            - cnm_package["integrated"]["qgp_cnm_pt_projection"]["band_lo"][idx]
            for idx, _, _, _ in STATE_SPECS_MAIN
        ],
        dtype=np.float64,
    )
    pt_hi = np.array(
        [
            cnm_package["integrated"]["qgp_cnm_pt_projection"]["band_hi"][idx]
            - cnm_package["integrated"]["qgp_cnm_pt_projection"]["raa9"][idx]
            for idx, _, _, _ in STATE_SPECS_MAIN
        ],
        dtype=np.float64,
    )
    y_vals = np.array(
        [cnm_package["integrated"]["qgp_cnm_y_projection"]["raa9"][idx] for idx, _, _, _ in STATE_SPECS_MAIN],
        dtype=np.float64,
    )
    y_lo = np.array(
        [
            cnm_package["integrated"]["qgp_cnm_y_projection"]["raa9"][idx]
            - cnm_package["integrated"]["qgp_cnm_y_projection"]["band_lo"][idx]
            for idx, _, _, _ in STATE_SPECS_MAIN
        ],
        dtype=np.float64,
    )
    y_hi = np.array(
        [
            cnm_package["integrated"]["qgp_cnm_y_projection"]["band_hi"][idx]
            - cnm_package["integrated"]["qgp_cnm_y_projection"]["raa9"][idx]
            for idx, _, _, _ in STATE_SPECS_MAIN
        ],
        dtype=np.float64,
    )
    ax_int.errorbar(xpos - 0.18, qgp_vals, yerr=qgp_err, fmt="o", ms=5, capsize=2, color=qgp_color, label="QGP only")
    ax_int.errorbar(xpos, pt_vals, yerr=np.vstack([pt_lo, pt_hi]), fmt="s", ms=5, capsize=2, color=combo_color, label="QGP × CNM (pT proj.)")
    ax_int.errorbar(xpos + 0.18, y_vals, yerr=np.vstack([y_lo, y_hi]), fmt="^", ms=5, capsize=2, color="#8c564b", label="QGP × CNM (y proj.)")
    ax_int.set_xticks(xpos, [name for _, name, _, _ in STATE_SPECS_MAIN])
    ax_int.set_ylabel(r"Integrated $R_{\mathrm{AA}}$")
    ax_int.set_ylim(0.0, 1.0)
    ax_int.grid(alpha=0.2, lw=0.5)
    ax_int.text(
        0.03,
        0.97,
        r"$\mathrm{O{+}O}\ \sqrt{s_{NN}}=5.36\ \mathrm{TeV}$" "\n"
        + r"$-2.4 \leq y \leq 2.4,\ p_T \leq 30\ \mathrm{GeV}$" "\n"
        + r"Two QGP$\times$CNM integrated prescriptions from the available 1D CNM tables",
        transform=ax_int.transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )
    ax_int.legend(loc="upper right", framealpha=0.95)
    fig_int.tight_layout()
    int_pdf = out_fig_dir / "oo5360_integrated_raa__noReg__qgp_vs_qgp_cnm.pdf"
    int_png = out_fig_dir / "oo5360_integrated_raa__noReg__qgp_vs_qgp_cnm.png"
    fig_int.savefig(int_pdf, bbox_inches="tight")
    fig_int.savefig(int_png, dpi=200, bbox_inches="tight")
    plt.close(fig_int)
    logger.info("Saved %s", int_pdf)


def _write_noReg_final_note(
    out_root: Path,
    mode_result: ModeResult,
    cnm_package: dict,
    logger: logging.Logger,
) -> None:
    tail_pct = 100.0 * float(cnm_package["pt_tail_fraction_for_integrated"])
    text = f"""# OO 5.36 TeV noReg Final Note

- system: `O+O`, `sqrt(s_NN)=5.36 TeV`
- mode: `QTraj-NLO noReg` only
- fixed impact parameter: `b = {B_MINBIAS:.5f} fm`
- acceptance for `R_AA(pT)`: `-2.4 <= y <= 2.4`, `0 <= pT < 30 GeV`
- acceptance for `R_AA(y)`: `-2.4 <= y <= 2.4`, `pT <= 30 GeV`

## Verified fixed-b averaging

At this single OO impact parameter, no Glauber or centrality weighting is used.

Inside each selected bin:

`<S_i> = sum_a q_a S_i^(a) / sum_a q_a`

For this `noReg` bundle, `qweight = 1` for every matched physical trajectory, so the qweight-weighted mean is exactly the ordinary arithmetic mean over selected trajectories.

The integrated fixed-acceptance six-state mean is:

`1S={mode_result.mean6_int[0]:.8f}`, `2S={mode_result.mean6_int[1]:.8f}`, `1P={mode_result.mean6_int[2]:.8f}`, `3S={mode_result.mean6_int[3]:.8f}`, `2P={mode_result.mean6_int[4]:.8f}`.

After feed-down, the integrated QGP-only inclusive results are:

- `R_AA(1S) = {mode_result.raa9_int[0]:.6f} +/- {mode_result.sem9_int[0]:.6f}`
- `R_AA(2S) = {mode_result.raa9_int[1]:.6f} +/- {mode_result.sem9_int[1]:.6f}`
- `R_AA(3S) = {mode_result.raa9_int[5]:.6f} +/- {mode_result.sem9_int[5]:.6f}`

## QGP × CNM combination

The current OO CNM inputs are common multiplicative Upsilon factors:

- `[outputs/cnm/min_bias/OO_5p36TeV/Upsilon_RAA_cnm_vs_pT_MB_-2.4y2.4_OO_5p36TeV.csv](/mnt/workstation/bottomonia_combined_analysis/outputs/cnm/min_bias/OO_5p36TeV/Upsilon_RAA_cnm_vs_pT_MB_-2.4y2.4_OO_5p36TeV.csv)`
- `[outputs/cnm/min_bias/OO_5p36TeV/Upsilon_RAA_cnm_vs_y_MB_OO_5p36TeV.csv](/mnt/workstation/bottomonia_combined_analysis/outputs/cnm/min_bias/OO_5p36TeV/Upsilon_RAA_cnm_vs_y_MB_OO_5p36TeV.csv)`

So for the current inputs, CNM cancels in `2S/1S`, `3S/1S`, and `3S/2S`.

For `R_AA`, the central `QGP x CNM` value is built from the central CNM curve. The plotted/exported total band combines:

- QGP statistical SEM
- CNM low/high variation

in quadrature.

For the integrated `QGP x CNM` point, the present 1D CNM inputs do not define a unique 2D acceptance-integrated factor. I therefore export both:

- `QGPxCNM_ptProjection`
- `QGPxCNM_yProjection`

The `pT`-projection uses the tabulated CNM curve through `19.5 GeV`; trajectories above that edge are a `{tail_pct:.2f}%` tail of the accepted fixed-acceptance sample and use the last available CNM bin for the integrated estimate only.
"""
    path = out_root / "OO5360_QTRAJ_NOREG_FINAL.md"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def _write_collab_notes(out_root: Path, no_reg: ModeResult, w_reg: ModeResult, logger: logging.Logger) -> None:
    _ensure_dir(out_root)
    data_dir = out_root / "data"
    _ensure_dir(data_dir)
    exp_data = _load_cms_preliminary_approx()
    no_reg_ratios = _compute_double_ratio_series(np.array([B_MINBIAS], dtype=np.float64), no_reg.raa9_int, no_reg.sem9_int)
    w_reg_ratios = _compute_double_ratio_series(np.array([B_MINBIAS], dtype=np.float64), w_reg.raa9_int, w_reg.sem9_int)

    summary = f"""# OO 5.36 TeV QTraj-NLO Fixed-b HNM Summary

This package contains the fixed-impact-parameter QTraj-NLO hot-medium analysis for O+O at `sqrt(s_NN)=5.36 TeV`.

- fixed impact parameter: `b = {B_MINBIAS:.5f} fm`
- `R_AA vs pT`: `-2.4 <= y <= 2.4`, `0 <= pT < 30 GeV`
- `R_AA vs y`: `-2.4 <= y <= 2.4`, `pT <= 30 GeV`
- double ratios: integrated and differential for `2S/1S`, `3S/1S`, `3S/2S`

## Integrated `R_AA`

| mode | `R_AA(1S)` | `SEM(1S)` | `R_AA(2S)` | `SEM(2S)` | `R_AA(3S)` | `SEM(3S)` |
|---|---:|---:|---:|---:|---:|---:|
| `noReg` | {no_reg.raa9_int[0]:.6f} | {no_reg.sem9_int[0]:.6f} | {no_reg.raa9_int[1]:.6f} | {no_reg.sem9_int[1]:.6f} | {no_reg.raa9_int[5]:.6f} | {no_reg.sem9_int[5]:.6f} |
| `wReg` | {w_reg.raa9_int[0]:.6f} | {w_reg.sem9_int[0]:.6f} | {w_reg.raa9_int[1]:.6f} | {w_reg.sem9_int[1]:.6f} | {w_reg.raa9_int[5]:.6f} | {w_reg.sem9_int[5]:.6f} |

## Integrated Double Ratios

| mode | `2S/1S` | `err` | `3S/1S` | `err` | `3S/2S` | `err` |
|---|---:|---:|---:|---:|---:|---:|
| `noReg` | {no_reg_ratios.ratio_2S_1S[0]:.6f} | {no_reg_ratios.err_2S_1S[0]:.6f} | {no_reg_ratios.ratio_3S_1S[0]:.6f} | {no_reg_ratios.err_3S_1S[0]:.6f} | {no_reg_ratios.ratio_3S_2S[0]:.6f} | {no_reg_ratios.err_3S_2S[0]:.6f} |
| `wReg` | {w_reg_ratios.ratio_2S_1S[0]:.6f} | {w_reg_ratios.err_2S_1S[0]:.6f} | {w_reg_ratios.ratio_3S_1S[0]:.6f} | {w_reg_ratios.err_3S_1S[0]:.6f} | {w_reg_ratios.ratio_3S_2S[0]:.6f} | {w_reg_ratios.err_3S_2S[0]:.6f} |

## Uncertainty Definition

The shaded theory bands and quoted errors are statistical SEM uncertainties from the trajectory ensemble, propagated through feed-down and then into the double ratios.

These bands do not include additional model variation from `kappa_hat`, hydro variation, CNM uncertainty, or Glauber/event-weighting uncertainty.

## Single-b Limitation

This OO bundle contains one impact parameter only. It does not produce a centrality or `N_part` scan by itself.

## Important Interpretation

In this OO package, `wReg` means quantum jumps ON in the Lindblad / MCWF evolution. It is not a guaranteed positive thermal-regeneration add-on. Because the jump-enabled evolution changes the full in-medium transition structure, the net inclusive `R_AA` can go down rather than up.
"""
    if exp_data is not None:
        summary += f"""

## Approximate CMS OO Preliminary Comparison

Approximate integrated CMS slide-readout values used in the overlay:

- `2S/1S ≈ {exp_data['integrated']['ratio_2S_1S']['value']:.2f}`
- `3S/1S ≈ {exp_data['integrated']['ratio_3S_1S']['value']:.2f}`
- `3S/2S ≈ {exp_data['integrated']['ratio_3S_2S']['value']:.2f}`
"""

    method = f"""# OO 5.36 TeV QTraj-NLO Method Note

## 1. Input selection

For each mode, read the bundled OO QTraj output:

- `noReg`: `inputs/qtraj_inputs/OxygenOxygen5360/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile.gz`
- `wReg`: `inputs/qtraj_inputs/OxygenOxygen5360/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile.gz`

The file parser builds matched S-wave / P-wave trajectory objects with:

- impact parameter `b`
- transverse momentum `pT`
- rapidity `y`
- six-state survival vector `surv6`
- `qweight`

## 2. Fixed-b analysis

There is one simulated impact parameter in this OO bundle:

- `b = {B_MINBIAS:.5f} fm`

No Glauber averaging or event weighting is applied across different impact parameters.

## 3. Survival averaging inside each selected bin

For each accepted bin, compute the `qweight`-weighted survival mean

`<S_i> = sum_a q_a S_i^(a) / sum_a q_a`

and an effective-weights SEM

`SEM_i = sqrt(Var_i) / sqrt(N_eff)`

with

`Var_i = sum_a q_a (S_i^(a) - <S_i>)^2 / sum_a q_a`

`N_eff = (sum_a q_a)^2 / sum_a q_a^2`

In the present OO bundle:

- `noReg` has uniform `qweight = 1`
- `wReg` has uniform `qweight = 20`

so weighted and unweighted averages are numerically identical here.

## 4. Kinematic selections

- `R_AA vs pT`: keep trajectories with `-2.4 <= y <= 2.4`, then bin in half-open pT bins `[pT_i, pT_(i+1))` from `0` to `30 GeV`
- `R_AA vs y`: keep trajectories with `pT <= 30 GeV`, then bin in half-open rapidity bins `[y_i, y_(i+1))` from `-2.4` to `2.4`
- integrated results: keep trajectories with `-2.4 <= y <= 2.4` and `pT <= 30 GeV`

## 5. Feed-down and inclusive `R_AA`

The weighted six-state survival mean is converted to the nine-state inclusive basis through the standard feed-down matrix:

`surv6 -> feed-down -> R_AA(1S,2S,chi_b(1P),3S,chi_b(2P),...)`

using the pp inclusive baseline `SIGMAS_EXP_OO_5360`.

## 6. Double ratios

At the bin level, the reported double ratios are

- `2S/1S = R_AA(2S) / R_AA(1S)`
- `3S/1S = R_AA(3S) / R_AA(1S)`
- `3S/2S = R_AA(3S) / R_AA(2S)`

with standard propagated uncertainty

`sigma(num/den)^2 = (sigma_num/den)^2 + (num sigma_den / den^2)^2`

## 7. Why `wReg` can be lower than `noReg`

The OO `wReg` file already has lower raw mean survival than `noReg` for the main upsilon states in the selected acceptance:

- `noReg surv6 mean`: `1S={no_reg.mean6_int[0]:.8f}`, `2S={no_reg.mean6_int[1]:.8f}`, `3S={no_reg.mean6_int[3]:.8f}`
- `wReg surv6 mean`: `1S={w_reg.mean6_int[0]:.8f}`, `2S={w_reg.mean6_int[1]:.8f}`, `3S={w_reg.mean6_int[3]:.8f}`

So the lower inclusive `R_AA` in `wReg` is already present in the raw QTraj survival output before any feed-down post-processing. It is not produced by a plotting or binning bug in this OO analysis layer.
"""

    (out_root / "OO5360_QTRAJ_HNM_COLLAB_SUMMARY.md").write_text(summary.strip() + "\n", encoding="utf-8")
    (out_root / "OO5360_QTRAJ_HNM_METHOD.md").write_text(method.strip() + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_root / "OO5360_QTRAJ_HNM_COLLAB_SUMMARY.md")
    logger.info("Wrote %s", out_root / "OO5360_QTRAJ_HNM_METHOD.md")

    if exp_data is not None:
        integrated_csv = (
            "observable,noReg_value,noReg_err,wReg_value,wReg_err,cms_prelim_approx_value,"
            "cms_prelim_approx_stat,cms_prelim_approx_syst,notes\n"
            f"ratio_2S_1S,{no_reg_ratios.ratio_2S_1S[0]:.10g},{no_reg_ratios.err_2S_1S[0]:.10g},"
            f"{w_reg_ratios.ratio_2S_1S[0]:.10g},{w_reg_ratios.err_2S_1S[0]:.10g},"
            f"{exp_data['integrated']['ratio_2S_1S']['value']:.10g},{exp_data['integrated']['ratio_2S_1S']['stat']:.10g},"
            f"{exp_data['integrated']['ratio_2S_1S']['syst']:.10g},"
            "\"Theory errors are propagated SEM; CMS values are approximate digitization\"\n"
            f"ratio_3S_1S,{no_reg_ratios.ratio_3S_1S[0]:.10g},{no_reg_ratios.err_3S_1S[0]:.10g},"
            f"{w_reg_ratios.ratio_3S_1S[0]:.10g},{w_reg_ratios.err_3S_1S[0]:.10g},"
            f"{exp_data['integrated']['ratio_3S_1S']['value']:.10g},{exp_data['integrated']['ratio_3S_1S']['stat']:.10g},"
            f"{exp_data['integrated']['ratio_3S_1S']['syst']:.10g},"
            "\"Theory errors are propagated SEM; CMS values are approximate digitization\"\n"
            f"ratio_3S_2S,{no_reg_ratios.ratio_3S_2S[0]:.10g},{no_reg_ratios.err_3S_2S[0]:.10g},"
            f"{w_reg_ratios.ratio_3S_2S[0]:.10g},{w_reg_ratios.err_3S_2S[0]:.10g},"
            f"{exp_data['integrated']['ratio_3S_2S']['value']:.10g},{exp_data['integrated']['ratio_3S_2S']['stat']:.10g},"
            f"{exp_data['integrated']['ratio_3S_2S']['syst']:.10g},"
            "\"Theory errors are propagated SEM; CMS values are approximate digitization\"\n"
        )
        (data_dir / "oo5360_integrated_summary__theory_vs_cms_prelim.csv").write_text(integrated_csv, encoding="utf-8")
        logger.info("Wrote %s", data_dir / "oo5360_integrated_summary__theory_vs_cms_prelim.csv")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="O+O 5.36 TeV QTraj-NLO (noReg vs wReg) production")
    parser.add_argument("--mode", choices=("noReg", "wReg", "both"), default="both")
    parser.add_argument(
        "--out-root",
        default=None,
        help="Optional output directory root. Default: outputs/qtraj_outputs/LHC/OxygenOxygen5p36TeV/production",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("qtraj_out_analysis.oo5360")

    out_root = Path(args.out_root) if args.out_root else (
        REPO_ROOT / "outputs" / "qtraj_outputs" / "LHC" / "OxygenOxygen5p36TeV" / "production"
    )
    out_data = out_root / "data"
    out_data_noreg = out_root / "modes" / "noReg" / "data"
    out_data_wreg = out_root / "modes" / "wReg" / "data"
    out_fig = out_root / "figures" / "theory_only"
    out_fig_noreg = out_root / "figures" / "noReg"
    out_fig_wreg = out_root / "figures" / "wReg"

    if args.mode in ("noReg", "both"):
        noReg = _analyze_mode("noReg", logger)
        _save_mode_csvs(noReg, out_data, logger)
        _save_mode_csvs(noReg, out_data_noreg, logger)
        _save_double_ratio_csvs(noReg, out_data, logger)
        _save_double_ratio_csvs(noReg, out_data_noreg, logger)
        _write_mode_diagnostics(out_data, "noReg", noReg, logger)
        _write_mode_diagnostics(out_data_noreg, "noReg", noReg, logger)
        _plot_single_mode(noReg, out_fig_noreg, logger)
        noReg_cnm = _compute_noReg_qgp_cnm_package(noReg, logger)
        _save_noReg_qgp_cnm_csvs(noReg_cnm, out_data, logger)
        _save_noReg_qgp_cnm_csvs(noReg_cnm, out_data_noreg, logger)
        _write_noReg_hepdata_tables(noReg, noReg_cnm, out_data, logger)
        _write_noReg_hepdata_tables(noReg, noReg_cnm, out_data_noreg, logger)
        _plot_noReg_qgp_vs_qgp_cnm(noReg, noReg_cnm, out_fig_noreg, logger)
        _write_noReg_final_note(out_root, noReg, noReg_cnm, logger)
    else:
        noReg = None
        noReg_cnm = None

    if args.mode in ("wReg", "both"):
        wReg = _analyze_mode("wReg", logger)
        _save_mode_csvs(wReg, out_data, logger)
        _save_mode_csvs(wReg, out_data_wreg, logger)
        _save_double_ratio_csvs(wReg, out_data, logger)
        _save_double_ratio_csvs(wReg, out_data_wreg, logger)
        _write_mode_diagnostics(out_data, "wReg", wReg, logger)
        _write_mode_diagnostics(out_data_wreg, "wReg", wReg, logger)
        _plot_single_mode(wReg, out_fig_wreg, logger)
    else:
        wReg = None

    if args.mode == "both" and noReg is not None and wReg is not None:
        _write_comparison_diagnostics(out_data, noReg, wReg, logger)
        _plot_overlay(noReg, wReg, out_fig, logger)
        _plot_double_ratios(noReg, wReg, out_fig, logger)
        _write_collab_notes(out_root, noReg, wReg, logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
