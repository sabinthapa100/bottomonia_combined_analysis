"""
survival_probability.py
-----------------------
Low-level, reusable functions for computing direct and feed-down
survival probabilities (R_AA) from MCWF ensemble output.

These are the building blocks used by the full pipeline in compute.py
and by standalone single-b analyses.

Terminology
-----------
surv6  : np.ndarray, shape (6,) or (n_b, 6)
    Per-state survival probability from the MCWF ensemble:
    [Υ(1S), Υ(2S), χ_b(1P), Υ(3S), χ_b(2P), Υ(1D)]
    Values ∈ [0, 1].  This is *not yet* R_AA.

surv9  : np.ndarray, shape (9,) or (n_b, 9)
    After split_hyperfine_6_to_9 is applied.

sigmas_prim : np.ndarray, shape (9,)
    Primordial pp cross-section fractions (GeV^{-2} or arb units),
    obtained by solving feeddown^{-1} · sigmas_exp.

feeddown : np.ndarray, shape (9, 9)
    Lower-triangular feed-down matrix (transpose of decay matrix).

R_AA (direct, state i)
    = (N_bin × sigma_prim[i] × surv9[i])
      / (N_bin × sigma_prim[i])
    = surv9[i]                          ← trivially surv9 itself!

R_AA (inclusive, state i)
    = (F @ diag(N_bin × sigma_prim) @ surv9)[i]
      / (F @ (N_bin × sigma_prim))[i]
    where F = feeddown matrix.

For multiple b-values, N_bin(b) is obtained from the Glauber model.
"""

import logging
import numpy as np
from typing import Optional, Tuple

from qtraj_analysis.schema import SurvivalResult, STATE_LABELS_9
from qtraj_analysis.feeddown import split_hyperfine_6_to_9, build_feeddown_matrix

logger = logging.getLogger(__name__)


# ─── Core computation ────────────────────────────────────────────


def compute_raa_direct(
    surv6: np.ndarray,
    surv6_sem: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct (primordial) R_AA — no feed-down applied.

    R_AA_direct[i] = surv9[i]   (survival probability IS the direct R_AA
    when the denominator is normalized to 1 in the pp reference).

    Parameters
    ----------
    surv6 : shape (6,) or (n_b, 6)
    surv6_sem : shape (6,) or (n_b, 6)  — standard error on surv6

    Returns
    -------
    raa9_direct : shape (9,) or (n_b, 9)
    raa9_direct_sem : shape (9,) or (n_b, 9)
    """
    batched = surv6.ndim == 2
    if not batched:
        s6 = surv6[np.newaxis, :]
        e6 = surv6_sem[np.newaxis, :]
    else:
        s6, e6 = surv6, surv6_sem

    n_b = s6.shape[0]
    raa9 = np.vstack([split_hyperfine_6_to_9(s6[i]) for i in range(n_b)])
    sem9 = np.vstack([split_hyperfine_6_to_9(e6[i]) for i in range(n_b)])

    if not batched:
        return raa9[0], sem9[0]
    return raa9, sem9


def compute_raa_inclusive(
    surv6: np.ndarray,
    surv6_sem: np.ndarray,
    sigmas_prim: np.ndarray,
    nbin: np.ndarray,
    feeddown: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inclusive (feed-down corrected) R_AA.

    R_AA_incl = (F @ diag(nbin × σ_prim) @ surv9) / (F @ (nbin × σ_prim))

    Parameters
    ----------
    surv6 : shape (6,) or (n_b, 6)
    surv6_sem : shape (6,) or (n_b, 6)
    sigmas_prim : shape (9,)  — primordial pp cross sections
    nbin : scalar or shape (n_b,)  — binary collision number per b
    feeddown : shape (9, 9) or None  — uses build_feeddown_matrix() if None

    Returns
    -------
    raa9_incl : shape (9,) or (n_b, 9)
    raa9_incl_sem : shape (9,) or (n_b, 9)
    """
    if feeddown is None:
        feeddown = build_feeddown_matrix()

    F = np.asarray(feeddown, dtype=np.float64)
    sigma = np.asarray(sigmas_prim, dtype=np.float64)

    batched = surv6.ndim == 2
    if not batched:
        s6 = surv6[np.newaxis, :]
        e6 = surv6_sem[np.newaxis, :]
        nb = np.asarray([nbin], dtype=np.float64)
    else:
        s6, e6 = surv6, surv6_sem
        nb = np.asarray(nbin, dtype=np.float64)

    n_b = s6.shape[0]

    # Split to 9 states
    surv9 = np.vstack([split_hyperfine_6_to_9(s6[i]) for i in range(n_b)])  # (n_b, 9)
    sem9  = np.vstack([split_hyperfine_6_to_9(e6[i]) for i in range(n_b)])  # (n_b, 9)

    # denominator: F @ (nbin × sigma)  — shape (n_b, 9)
    denom = (F @ (nb[:, None] * sigma[None, :]).T).T   # (n_b, 9)

    # numerator: F @ (nbin × sigma × surv9) — shape (n_b, 9)
    num   = (F @ (nb[:, None] * sigma[None, :] * surv9).T).T

    # R_AA
    raa9_incl = np.divide(num, denom,
                           out=np.full_like(num, np.nan),
                           where=(denom > 0))

    # Error propagation (in quadrature, ignoring off-diagonal covariance)
    # δR_incl[i] ≈ (F @ (nbin × sigma × δsurv9)) / denom[i]
    num_err = (F @ (nb[:, None] * sigma[None, :] * sem9).T).T
    raa9_incl_sem = np.divide(num_err, denom,
                               out=np.zeros_like(num_err),
                               where=(denom > 0))

    if not batched:
        return raa9_incl[0], raa9_incl_sem[0]
    return raa9_incl, raa9_incl_sem


def compute_survival(
    surv6_mean: np.ndarray,
    surv6_sem: np.ndarray,
    sigmas_prim: np.ndarray,
    nbin: np.ndarray,
    feeddown: Optional[np.ndarray] = None,
    bvals: Optional[np.ndarray] = None,
    npart: Optional[np.ndarray] = None,
) -> SurvivalResult:
    """
    Compute both direct and inclusive R_AA, returning a SurvivalResult.

    Parameters
    ----------
    surv6_mean : shape (n_b, 6) or (6,) — mean survival probabilities
    surv6_sem  : shape (n_b, 6) or (6,) — SEM on survival probabilities
    sigmas_prim : shape (9,) — primordial pp cross sections
    nbin : shape (n_b,) or scalar — N_bin(b) from Glauber
    feeddown : (9,9) or None
    bvals : shape (n_b,) or None
    npart : shape (n_b,) or None

    Returns
    -------
    SurvivalResult
    """
    s6 = np.asarray(surv6_mean, dtype=np.float64)
    e6 = np.asarray(surv6_sem, dtype=np.float64)

    raa_dir,  raa_dir_sem  = compute_raa_direct(s6, e6)
    raa_incl, raa_incl_sem = compute_raa_inclusive(s6, e6, sigmas_prim, nbin, feeddown)

    return SurvivalResult(
        raa_direct=raa_dir,
        raa_inclusive=raa_incl,
        raa_direct_sem=raa_dir_sem,
        raa_inclusive_sem=raa_incl_sem,
        bvals=bvals,
        npart=npart,
    )


def print_survival_table(result: SurvivalResult, use_inclusive: bool = True) -> None:
    """Pretty-print survival probabilities to stdout."""
    raa  = result.raa_inclusive  if use_inclusive else result.raa_direct
    serr = result.raa_inclusive_sem if use_inclusive else result.raa_direct_sem
    tag  = "Inclusive" if use_inclusive else "Direct"

    if raa.ndim == 1:
        # single b-point
        print(f"\n{'State':<20}  {'R_AA':>8}  {'SEM':>8}  ({tag})")
        print("-" * 45)
        for label, r, e in zip(STATE_LABELS_9, raa, serr):
            print(f"  {label:<18}  {r:8.4f}  {e:8.4f}")
    else:
        n_b = raa.shape[0]
        b_label = result.bvals if result.bvals is not None else np.arange(n_b)
        print(f"\n{tag} R_AA  ({n_b} b-bins)\n")
        header = f"{'b/Npart':>10}" + "".join(f"  {s[:10]:>10}" for s in STATE_LABELS_9)
        print(header)
        print("-" * len(header))
        for i in range(n_b):
            row_id = f"{b_label[i]:10.3f}"
            vals   = "".join(f"  {raa[i, j]:10.4f}" for j in range(9))
            print(row_id + vals)
