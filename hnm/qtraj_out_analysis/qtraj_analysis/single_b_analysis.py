"""
single_b_analysis.py
---------------------
Convenience wrapper for a single impact-parameter (or minimum-bias)
analysis of qtraj outputs — no Glauber model required.

This is useful when:
  - You have one simulation run at a fixed b value.
  - You want a quick sanity check of survival probabilities before
    running the full multi-b pipeline.
  - You are developing / testing new physics setups.

Output
------
Returns a dict with the following keys:

    "surv6_mean"   : np.ndarray (6,)  — raw MCWF survival probs per state
    "surv6_sem"    : np.ndarray (6,)
    "raa9_direct"  : np.ndarray (9,)  — primordial (no feed-down)
    "raa9_incl"    : np.ndarray (9,)  — inclusive (feed-down applied)
    "raa9_direct_sem"  : np.ndarray (9,)
    "raa9_incl_sem"    : np.ndarray (9,)
    "double_ratios"    : DoubleRatioResult  (npart = [Npart or np.nan])
    "state_labels"     : list[str]
    "n_trajectories"   : int
    "b_val"            : float

Optional (computed only if `pt_edges` / `y_edges` are provided):
    "pt_centers"       : np.ndarray (n_pt,)
    "raa9_incl_vs_pt"  : np.ndarray (n_pt, 9)
    "raa9_incl_sem_vs_pt" : np.ndarray (n_pt, 9)
    "y_centers"        : np.ndarray (n_y,)
    "raa9_incl_vs_y"   : np.ndarray (n_y, 9)
    "raa9_incl_sem_vs_y"  : np.ndarray (n_y, 9)
"""

import logging
import numpy as np
from typing import Optional, Tuple

from qtraj_analysis.io import load_qtraj_table, parse_records
from qtraj_analysis.matching import build_observables
from qtraj_analysis.feeddown import (
    apply_feeddown_to_raa6,
    build_feeddown_matrix,
    solve_primordial_sigmas,
    split_hyperfine_6_to_9,
)
from qtraj_analysis.binning import compute_raa_vs_pt, compute_raa_vs_y
from qtraj_analysis.stats import mean_and_sem
from qtraj_analysis.survival_probability import compute_raa_direct, compute_raa_inclusive
from qtraj_analysis.double_ratios import compute_standard_double_ratios
from qtraj_analysis.schema import STATE_LABELS_9, STATE_LABELS_6

logger = logging.getLogger(__name__)


def _apply_feeddown_binned(
    raa6: np.ndarray,
    sem6: np.ndarray,
    *,
    feeddown: np.ndarray,
    sigmas_primordial: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply feed-down bin-by-bin for (nbins, 6) arrays."""
    raa9 = np.full((raa6.shape[0], 9), np.nan, dtype=np.float64)
    sem9 = np.full((raa6.shape[0], 9), np.nan, dtype=np.float64)
    for i in range(raa6.shape[0]):
        if not np.isfinite(raa6[i, 0]):
            continue
        r9, e9 = apply_feeddown_to_raa6(raa6[i], sem6[i], feeddown, sigmas_primordial)
        raa9[i] = r9
        sem9[i] = e9
    return raa9, sem9


def _analyze_filtered_obs(
    obs: list,
    *,
    sigmas_exp: np.ndarray,
    feeddown: np.ndarray,
    nbin: float,
    pt_edges: Optional[np.ndarray],
    y_edges: Optional[np.ndarray],
    y_window_for_pt: Optional[Tuple[float, float]],
    pt_max_for_y: Optional[float],
    logger_: logging.Logger,
) -> dict:
    """Core analysis on an already-filtered observable list."""
    # ── 1. Compute mean survival (primordial 6-state) ─────────────
    surv6_all = np.vstack([o.surv6 for o in obs])   # (n_traj, 6)
    surv6_mean, surv6_sem = mean_and_sem(surv6_all)

    # ── 2. Solve primordial sigmas (pp baseline) ──────────────────
    sigmas_prim = solve_primordial_sigmas(feeddown, np.asarray(sigmas_exp, dtype=np.float64))

    # ── 3. Direct and inclusive R_AA (single point) ───────────────
    raa9_direct, raa9_direct_sem = compute_raa_direct(surv6_mean, surv6_sem)
    raa9_incl, raa9_incl_sem = compute_raa_inclusive(
        surv6_mean, surv6_sem, sigmas_prim, nbin=nbin, feeddown=feeddown,
    )

    # ── 4. Optional: spectra-style binned results ─────────────────
    pt_centers = None
    raa9_incl_vs_pt = None
    raa9_incl_sem_vs_pt = None
    if pt_edges is not None:
        pt_centers, raa6_pt, sem6_pt = compute_raa_vs_pt(
            obs,
            np.asarray(pt_edges, dtype=np.float64),
            y_window=y_window_for_pt,
            logger=logger_,
        )
        raa9_incl_vs_pt, raa9_incl_sem_vs_pt = _apply_feeddown_binned(
            raa6_pt,
            sem6_pt,
            feeddown=feeddown,
            sigmas_primordial=sigmas_prim,
        )

    y_centers = None
    raa9_incl_vs_y = None
    raa9_incl_sem_vs_y = None
    if y_edges is not None:
        obs_y = obs
        if pt_max_for_y is not None and np.isfinite(pt_max_for_y):
            obs_y = [o for o in obs if o.pt <= float(pt_max_for_y)]
        y_centers, raa6_y, sem6_y = compute_raa_vs_y(
            obs_y,
            np.asarray(y_edges, dtype=np.float64),
            logger=logger_,
        )
        raa9_incl_vs_y, raa9_incl_sem_vs_y = _apply_feeddown_binned(
            raa6_y,
            sem6_y,
            feeddown=feeddown,
            sigmas_primordial=sigmas_prim,
        )

    return dict(
        n_trajectories=len(obs),
        surv6_mean=surv6_mean,
        surv6_sem=surv6_sem,
        surv6_labels=STATE_LABELS_6,
        raa9_direct=raa9_direct,
        raa9_direct_sem=raa9_direct_sem,
        raa9_incl=raa9_incl,
        raa9_incl_sem=raa9_incl_sem,
        sigmas_primordial=sigmas_prim,
        pt_centers=pt_centers,
        raa9_incl_vs_pt=raa9_incl_vs_pt,
        raa9_incl_sem_vs_pt=raa9_incl_sem_vs_pt,
        y_centers=y_centers,
        raa9_incl_vs_y=raa9_incl_vs_y,
        raa9_incl_sem_vs_y=raa9_incl_sem_vs_y,
    )


def run_single_b(
    datafile:       str,
    sigmas_exp:     np.ndarray,
    b_val:          Optional[float] = None,
    feeddown:       Optional[np.ndarray] = None,
    nbin:           float = 1.0,
    npart:          float = float("nan"),
    logger_:        Optional[logging.Logger] = None,
    bmb:            Optional[float] = None,
    pt_edges:       Optional[np.ndarray] = None,
    y_edges:        Optional[np.ndarray] = None,
    y_window_for_pt: Optional[Tuple[float, float]] = None,
    pt_max_for_y:   Optional[float] = None,
) -> dict:
    """
    Single impact-parameter analysis.

    Parameters
    ----------
    datafile : str
        Path to qtraj output file (may be .gz).
    sigmas_exp : np.ndarray, shape (9,)
        Experimental pp inclusive cross-section fractions
        (same units / ordering as build_feeddown_matrix output).
    b_val : float or None
        If given, keep only trajectories with this b value.
        If None, use all trajectories.
    feeddown : (9, 9) or None
        Feed-down matrix. Uses build_feeddown_matrix() if None.
    nbin : float
        Binary collision number (N_bin) for this b value.
        Default 1.0 (appropriate for a vacuum/pp reference run).
    npart : float
        N_part for this b value (informational only; may be nan).
    bmb : float or None
        Minimum-bias b to exclude (same as full pipeline).
    pt_edges : np.ndarray or None
        If provided, compute inclusive R_AA vs pT using these bin edges.
    y_edges : np.ndarray or None
        If provided, compute inclusive R_AA vs y using these bin edges.
    y_window_for_pt : (ymin,ymax) or None
        Optional rapidity acceptance cut applied before pT binning.
    pt_max_for_y : float or None
        Optional pT upper cut applied before y binning.

    Returns
    -------
    dict  (see module docstring)
    """
    lg = logger_ or logger

    if feeddown is None:
        feeddown = build_feeddown_matrix()
    F = np.asarray(feeddown, dtype=np.float64)

    # ── 1. Parse file ────────────────────────────────────────────
    table   = load_qtraj_table(datafile, lg)
    records = parse_records(table, lg)
    obs     = build_observables(records, lg)

    # ── 2. Filter by b ──────────────────────────────────────────
    if b_val is not None:
        obs = [o for o in obs if abs(o.b - b_val) < 1e-4]
        lg.info("After b=%.4f filter: %d trajectories.", b_val, len(obs))

    if bmb is not None:
        obs = [o for o in obs if abs(o.b - bmb) > 1e-4]
        lg.info("After bMB=%.4f removal: %d trajectories.", bmb, len(obs))

    if len(obs) == 0:
        raise ValueError(
            f"No trajectories found after filtering "
            f"(b_val={b_val}, bmb={bmb}, n_records={len(records)})."
        )

    # If caller did not specify b_val and the file contains a single discrete b,
    # record it for downstream summaries/outputs.
    if b_val is None:
        b_unique = np.unique(np.round(np.array([o.b for o in obs], dtype=np.float64), 6))
        if b_unique.size == 1:
            b_val = float(b_unique[0])

    core = _analyze_filtered_obs(
        obs,
        sigmas_exp=np.asarray(sigmas_exp, dtype=np.float64),
        feeddown=F,
        nbin=nbin,
        pt_edges=pt_edges,
        y_edges=y_edges,
        y_window_for_pt=y_window_for_pt,
        pt_max_for_y=pt_max_for_y,
        logger_=lg,
    )

    lg.info(
        "Υ(1S) survival: %.4f ± %.4f  (n=%d)",
        core["surv6_mean"][0], core["surv6_sem"][0], core["n_trajectories"],
    )

    # ── 3. Double ratios (single point) ──────────────────────────
    npart_arr = np.array([npart])
    dr = compute_standard_double_ratios(
        npart=npart_arr,
        raa9_mean=core["raa9_incl"][np.newaxis, :],
        raa9_sem=core["raa9_incl_sem"][np.newaxis, :],
    )

    result = dict(
        b_val=b_val if b_val is not None else float("nan"),
        double_ratios=dr,
        state_labels=STATE_LABELS_9,
        **core,
    )
    _print_summary(result)
    return result


def run_b_scan(
    datafile: str,
    sigmas_exp: np.ndarray,
    *,
    feeddown: Optional[np.ndarray] = None,
    bmb: Optional[float] = None,
    b_round: int = 6,
    pt_edges: Optional[np.ndarray] = None,
    y_edges: Optional[np.ndarray] = None,
    y_window_for_pt: Optional[Tuple[float, float]] = None,
    pt_max_for_y: Optional[float] = None,
    logger_: Optional[logging.Logger] = None,
) -> dict:
    """
    Scan a multi-b qtraj file and compute single-b results at each discrete b.

    This is primarily a diagnostic/verification helper:
    - No centrality integration is performed (no `pFunction(c(b))` weighting).
    - Each discrete b point is treated independently.
    """
    lg = logger_ or logger
    if feeddown is None:
        feeddown = build_feeddown_matrix()
    F = np.asarray(feeddown, dtype=np.float64)

    table = read_whitespace_table(datafile, lg)
    records = parse_records(table, lg)
    obs_all = build_observables(records, lg)

    if bmb is not None and np.isfinite(bmb):
        obs_all = [o for o in obs_all if abs(o.b - bmb) > 1e-4]
        lg.info("Removed bMB=%.4f -> remaining %d trajectories.", bmb, len(obs_all))

    groups: dict[float, list] = {}
    for o in obs_all:
        bb = float(np.round(o.b, b_round))
        groups.setdefault(bb, []).append(o)

    bvals = np.array(sorted(groups.keys()), dtype=np.float64)
    results: list[dict] = []
    for b in bvals:
        core = _analyze_filtered_obs(
            groups[b],
            sigmas_exp=np.asarray(sigmas_exp, dtype=np.float64),
            feeddown=F,
            nbin=1.0,
            pt_edges=pt_edges,
            y_edges=y_edges,
            y_window_for_pt=y_window_for_pt,
            pt_max_for_y=pt_max_for_y,
            logger_=lg,
        )
        results.append(dict(b_val=b, **core))

    return dict(
        bvals=bvals,
        results=results,
        state_labels=STATE_LABELS_9,
    )


def _print_summary(res: dict) -> None:
    """Compact stdout summary of a single-b result."""
    b = res["b_val"]
    n = res["n_trajectories"]
    b_tag = f"{b:.3f}" if np.isfinite(b) else "mixed"
    header = f"\n── Single-b Analysis  (b={b_tag} fm,  N_traj={n}) ──"
    print(header)
    print(f"{'State':<20}  {'Survival':>10}  {'RAA_dir':>10}  {'RAA_incl':>10}")
    print("-" * 55)
    raa9d   = res["raa9_direct"]
    raa9i   = res["raa9_incl"]
    from qtraj_analysis.schema import STATE_LABELS_9
    # Print the 5 key states
    idx_map = [
        (0, 0),  # Υ(1S)
        (1, 1),  # Υ(2S)
        (3, 2),  # χ_b1(1P)
        (5, 3),  # Υ(3S)
        (7, 4),  # χ_b1(2P)
    ]  # (9-state idx, surv6 idx)
    for i9, i6 in idx_map:
        s_val = res["surv6_mean"][i6]
        label = STATE_LABELS_9[i9]
        print(f"  {label:<18}  {s_val:10.4f}  {raa9d[i9]:10.4f}  {raa9i[i9]:10.4f}")
    dr = res["double_ratios"]
    print(f"\n  Υ(2S)/Υ(1S) = {dr.ratio_2S_1S[0]:.4f} ± {dr.err_2S_1S[0]:.4f}")
    print(f"  Υ(3S)/Υ(1S) = {dr.ratio_3S_1S[0]:.4f} ± {dr.err_3S_1S[0]:.4f}")
    print(f"  χb1(1P)/1S  = {dr.ratio_chi1_1S[0]:.4f} ± {dr.err_chi1_1S[0]:.4f}")
