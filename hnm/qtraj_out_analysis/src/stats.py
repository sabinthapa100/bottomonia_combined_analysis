import math
import logging
import numpy as np
from typing import Tuple, List

from qtraj_analysis.schema import TrajectoryObs
from qtraj_analysis.glauber import GlauberInterpolator
from qtraj_analysis.feeddown import split_hyperfine_6_to_9

def mean_and_sem(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard error of the mean (SEM).
    Excludes NaNs if present? No, standard np traversal.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x, axis=0)
    if x.shape[0] < 2:
        # sem undefined; return zeros
        sem = np.zeros_like(mu)
    else:
        sem = np.std(x, axis=0, ddof=1) / math.sqrt(x.shape[0])
    return mu, sem


def p_centrality(c: np.ndarray, c0: float = 0.25) -> np.ndarray:
    """
    Mathematica:
      pFunction[c_] = Exp[-c/0.25]/Integrate[Exp[-x/0.25], {x,0,1}]
    """
    c = np.asarray(c, dtype=np.float64)
    # normalization Z = ∫_0^1 exp(-x/c0) dx = c0*(1-exp(-1/c0))
    Z = c0 * (1.0 - np.exp(-1.0 / c0))
    return np.exp(-c / c0) / Z


def weighted_avg_surv9(
    obs: List[TrajectoryObs],
    glauber: GlauberInterpolator,
    selector: np.ndarray,
    logger: logging.Logger,
    c0: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproduces centrality-weighted averaging:
      weight = pFunction(c(b))
      build weighted list with splitHyperfine(surv6)*weight, normalize by mean(weight),
      then take qweight-weighted mean over events.

    We return:
      avg9, sem9   (both length 9)
    """
    chosen = [o for o, ok in zip(obs, selector) if ok]
    if len(chosen) == 0:
        raise ValueError("Selection produced zero trajectories.")

    b = np.array([o.b for o in chosen], dtype=np.float64)
    c = glauber.b_to_c(b)
    w_cent = p_centrality(c, c0=c0)

    # Normalize by mean centrality weight (Mathematica: wtm = Mean[selectWeights...])
    wtm = float(np.mean(w_cent))
    if wtm == 0 or not np.isfinite(wtm):
        raise ValueError("Centrality weight mean is invalid (0 or nan).")

    # Build per-trajectory "features" (9-state split) scaled by w_cent/wtm
    X9 = np.vstack([split_hyperfine_6_to_9(o.surv6) for o in chosen])  # (n,9)
    
    # Scale each trajectory's result by its relative centrality weight
    X9_w = (w_cent[:, None] / wtm) * X9

    # qweight-weighted average (Mathematica: Transpose[...] . qweights/Total[qweights])
    q = np.array([o.qweight for o in chosen], dtype=np.float64)
    qsum = float(np.sum(q))
    if qsum <= 0 or not np.isfinite(qsum):
        raise ValueError("qweight sum is invalid (<=0 or nan). Check your file's qweight column.")

    avg = (X9_w.T @ q) / qsum  # (9,)

    # SEM: Mathematica uses StandardDeviation[wl]/Sqrt[Length[wl]]
    # Their wl is a list with many columns.
    # We approximate SEM across trajectories for the *weighted* quantity X9_w.
    # Note: strictly speaking, weighted SEM is complex. 
    # But matching the previous code (standard deviation of the weighted entries / sqrt(N)):
    _, sem = mean_and_sem(X9_w)

    logger.debug("Weighted avg computed: avg1S=%.4f", avg[0])
    return avg, sem
