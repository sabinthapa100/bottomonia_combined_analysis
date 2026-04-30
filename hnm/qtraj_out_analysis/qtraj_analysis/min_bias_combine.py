"""
Combine per–impact-parameter R_AA curves into a minimum-bias estimate using weights.

Weights w_i are **physics inputs** (e.g. fractional cross section in each b-bin,
or Monte Carlo sampling weights). They must satisfy w_i >= 0 and sum to a positive
finite number; we normalize to sum(w)=1 internally.

For each kinematic bin index k and state s:

  R_MB[k,s] = sum_i w_i R_i[k,s] / sum_i w_i

Error propagation (optional): if per-slice SEM σ_i[k,s] are uncorrelated between b,

  σ_MB[k,s]^2 = sum_i (w_i^2 σ_i[k,s]^2) / (sum_i w_i)^2

This ignores trajectory-level correlations across b (usually acceptable for first pass).
"""
from __future__ import annotations

import numpy as np


def weighted_average_raa9(
    raa9: np.ndarray,
    weights: np.ndarray,
    sem9: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Combine (n_b, n_bin, 9) or (n_b, 9) arrays along axis 0.

    Parameters
    ----------
    raa9 : ndarray
        Per-b inclusive R_AA after feeddown.
    weights : shape (n_b,)
        Non-negative weights; normalized to sum 1.
    sem9 : optional, same shape as raa9
        Standard errors matching raa9.

    Returns
    -------
    combined : ndarray
        Shape (n_bin, 9) or (9,) after reduction.
    combined_sem : ndarray or None
        Same shape as combined if sem9 was given.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != raa9.shape[0]:
        raise ValueError(f"weights length {w.shape[0]} != first dim of raa9 {raa9.shape[0]}")
    if np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("weights must be non-negative and finite")
    s = float(np.sum(w))
    if s <= 0:
        raise ValueError("sum(weights) must be positive")
    w = w / s

    x = np.asarray(raa9, dtype=np.float64)
    # (n_b, ...) -> weighted sum
    lead = tuple(range(1, x.ndim))
    comb = np.tensordot(w, x, axes=(0, 0))

    comb_sem = None
    if sem9 is not None:
        e = np.asarray(sem9, dtype=np.float64)
        if e.shape != x.shape:
            raise ValueError("sem9 must match raa9 shape")
        # Var(sum w_i X_i) with independent X_i: sum w_i^2 sigma_i^2
        var = np.tensordot(w**2, e**2, axes=(0, 0))
        comb_sem = np.sqrt(np.maximum(var, 0.0))

    return comb, comb_sem


def load_weights_b_pairs(
    path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load two-column whitespace/CSV file: b  weight (one pair per line).

    Lines starting with # are ignored.
    """
    bs: list[float] = []
    ws: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            bs.append(float(parts[0]))
            ws.append(float(parts[1]))
    if not bs:
        raise ValueError(f"No weight rows parsed from {path}")
    return np.asarray(bs, dtype=np.float64), np.asarray(ws, dtype=np.float64)


def align_weights_to_bvals(
    bvals: np.ndarray,
    b_weights: np.ndarray,
    w_weights: np.ndarray,
    *,
    b_round: int = 6,
) -> np.ndarray:
    """
    Map file (b, w) pairs to the order of `bvals` from a scan, by rounded b match.
    Raises if any bvals entry has no matching weight or ambiguous match.
    """
    bvals = np.asarray(bvals, dtype=np.float64).ravel()
    b_weights = np.asarray(b_weights, dtype=np.float64).ravel()
    w_weights = np.asarray(w_weights, dtype=np.float64).ravel()
    if b_weights.shape != w_weights.shape:
        raise ValueError("b_weights and w_weights must have same shape")

    lookup = {round(float(b), b_round): float(w) for b, w in zip(b_weights, w_weights)}
    out = np.zeros(len(bvals), dtype=np.float64)
    for i, b in enumerate(bvals):
        key = round(float(b), b_round)
        if key not in lookup:
            raise KeyError(f"No weight for b={b} (rounded {key}); check weights file.")
        out[i] = lookup[key]
    return out
