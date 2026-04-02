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
"""

import logging
import numpy as np
from typing import Optional

from qtraj_analysis.io import read_whitespace_table, parse_records
from qtraj_analysis.matching import build_observables
from qtraj_analysis.feeddown import (
    build_feeddown_matrix, solve_primordial_sigmas, split_hyperfine_6_to_9,
)
from qtraj_analysis.stats import mean_and_sem
from qtraj_analysis.survival_probability import compute_raa_direct, compute_raa_inclusive
from qtraj_analysis.double_ratios import compute_standard_double_ratios
from qtraj_analysis.schema import STATE_LABELS_9, STATE_LABELS_6

logger = logging.getLogger(__name__)


def run_single_b(
    datafile:       str,
    sigmas_exp:     np.ndarray,
    b_val:          Optional[float] = None,
    feeddown:       Optional[np.ndarray] = None,
    nbin:           float = 1.0,
    npart:          float = float("nan"),
    logger_:        Optional[logging.Logger] = None,
    bmb:            Optional[float] = None,
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

    Returns
    -------
    dict  (see module docstring)
    """
    lg = logger_ or logger

    if feeddown is None:
        feeddown = build_feeddown_matrix()
    F = np.asarray(feeddown, dtype=np.float64)

    # ── 1. Parse file ────────────────────────────────────────────
    table   = read_whitespace_table(datafile, lg)
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

    # ── 3. Compute mean survival ─────────────────────────────────
    surv6_all = np.vstack([o.surv6 for o in obs])   # (n_traj, 6)
    surv6_mean, surv6_sem = mean_and_sem(surv6_all)

    lg.info(
        "Υ(1S) survival: %.4f ± %.4f  (n=%d)",
        surv6_mean[0], surv6_sem[0], len(obs),
    )

    # ── 4. Solve primordial sigmas ───────────────────────────────
    sigmas_prim = solve_primordial_sigmas(F, np.asarray(sigmas_exp, dtype=np.float64))

    # ── 5. Compute direct R_AA (= survival probabilities, 9-state) ─
    raa9_direct, raa9_direct_sem = compute_raa_direct(surv6_mean, surv6_sem)

    # ── 6. Compute inclusive R_AA (feed-down applied) ───────────
    raa9_incl, raa9_incl_sem = compute_raa_inclusive(
        surv6_mean, surv6_sem, sigmas_prim, nbin=nbin, feeddown=F,
    )

    # ── 7. Double ratios ─────────────────────────────────────────
    npart_arr = np.array([npart])
    dr = compute_standard_double_ratios(
        npart    = npart_arr,
        raa9_mean= raa9_incl[np.newaxis, :],
        raa9_sem = raa9_incl_sem[np.newaxis, :],
    )

    result = dict(
        b_val            = b_val if b_val is not None else float("nan"),
        n_trajectories   = len(obs),
        surv6_mean       = surv6_mean,
        surv6_sem        = surv6_sem,
        surv6_labels     = STATE_LABELS_6,
        raa9_direct      = raa9_direct,
        raa9_direct_sem  = raa9_direct_sem,
        raa9_incl        = raa9_incl,
        raa9_incl_sem    = raa9_incl_sem,
        double_ratios    = dr,
        state_labels     = STATE_LABELS_9,
        sigmas_primordial= sigmas_prim,
    )

    _print_summary(result)
    return result


def _print_summary(res: dict) -> None:
    """Compact stdout summary of a single-b result."""
    b = res["b_val"]
    n = res["n_trajectories"]
    header = f"\n── Single-b Analysis  (b={b:.3f} fm,  N_traj={n}) ──"
    print(header)
    print(f"{'State':<20}  {'Survival':>10}  {'RAA_dir':>10}  {'RAA_incl':>10}")
    print("-" * 55)
    labels6 = res["surv6_labels"]
    raa9d   = res["raa9_direct"]
    raa9i   = res["raa9_incl"]
    from qtraj_analysis.schema import STATE_LABELS_9
    # Print the 5 key states
    idx_map = [(0, 0), (1, 1), (3, 2), (5, 5), (7, 6)]  # (9-state idx, surv6-approx)
    for i9, i6 in idx_map:
        s_val  = res["surv6_mean"][min(i6, 5)]
        label  = STATE_LABELS_9[i9]
        print(f"  {label:<18}  {s_val:10.4f}  {raa9d[i9]:10.4f}  {raa9i[i9]:10.4f}")
    dr = res["double_ratios"]
    print(f"\n  Υ(2S)/Υ(1S) = {dr.ratio_2S_1S[0]:.4f} ± {dr.err_2S_1S[0]:.4f}")
    print(f"  Υ(3S)/Υ(1S) = {dr.ratio_3S_1S[0]:.4f} ± {dr.err_3S_1S[0]:.4f}")
    print(f"  χb1(1P)/1S  = {dr.ratio_chi1_1S[0]:.4f} ± {dr.err_chi1_1S[0]:.4f}")
