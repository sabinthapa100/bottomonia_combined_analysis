import logging
import numpy as np
from typing import List, Tuple, Optional, Dict

from qtraj_analysis.schema import TrajectoryObs, RaaVsBResult
from qtraj_analysis.stats import mean_and_sem

def compute_raa_vs_b(
    obs: List[TrajectoryObs],
    logger: logging.Logger,
    bmb: Optional[float] = None,
    b_round: int = 6,
) -> RaaVsBResult:
    """
    Mathematica:
      bVals = rawData[[All,1,1]]//Union ; bVals=DeleteCases[bVals,bMB]
      selectB[i_] := Select[rawData, #[[1,1]] == bVals[[i]] &]
      rAAavgVsB[i_] := Mean[rAAvsB[i]]
      rAAsdevVsB[i_] := StandardDeviation[rAAvsB[i]]/Sqrt[Length[...]]

    Here we:
      - group by b (rounded)
      - optionally remove bMB
      - compute mean and SEM for each of 6 states
    """
    # gather
    b_list = np.array([o.b for o in obs], dtype=np.float64)
    if bmb is not None and np.isfinite(bmb):
        keep = np.abs(b_list - bmb) > 1e-4  # looser tolerance for float check
        obs = [o for k, o in zip(keep, obs) if k]
        logger.info("Removed bMB=%g -> remaining %d trajectories", bmb, len(obs))

    if not obs:
        logger.warning("No trajectories remaining after bMB filtering.")
        return RaaVsBResult(
            bvals=np.array([]),
            raa6_mean=np.empty((0, 6)),
            raa6_sem=np.empty((0, 6)),
        )

    # group by rounded b
    groups: Dict[float, List[TrajectoryObs]] = {}
    for o in obs:
        bb = float(np.round(o.b, b_round))
        groups.setdefault(bb, []).append(o)

    bvals = np.array(sorted(groups.keys()), dtype=np.float64)
    means = []
    sems = []

    for b in bvals:
        X = np.vstack([o.surv6 for o in groups[b]])  # (n,6)
        mu, se = mean_and_sem(X)
        means.append(mu)
        sems.append(se)

        logger.debug("b=%.6f n=%d mu(1S)=%.4f", b, X.shape[0], mu[0])

    return RaaVsBResult(
        bvals=bvals,
        raa6_mean=np.vstack(means) if means else np.empty((0, 6)),
        raa6_sem=np.vstack(sems) if sems else np.empty((0, 6)),
    )


def binned_step_series(
    bin_edges: np.ndarray,
    y_center: np.ndarray,
    y_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mathematica step-plot trick: Join[yvals, {yvals[[-1]]}] for plotting.
    Returns: x, y_step, y_min, y_max
    """
    if len(bin_edges) != len(y_center) + 1:
        raise ValueError("bin_edges must be length M+1 for y_center length M")

    y_center2 = np.concatenate([y_center, [y_center[-1]]])
    y_min2 = np.concatenate([y_center - y_err, [y_center[-1] - y_err[-1]]])
    y_max2 = np.concatenate([y_center + y_err, [y_center[-1] + y_err[-1]]])

    x = np.asarray(bin_edges, dtype=np.float64)
    return x, y_center2, y_min2, y_max2


def compute_raa_vs_pt(
    obs: List[TrajectoryObs],
    pt_edges: np.ndarray,
    y_window: Optional[Tuple[float, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Groups trajectories into pT bins.
    
    Returns:
      pt_centers: shape (nbins,)
      raa6_mean:  shape (nbins, 6)
      raa6_sem:   shape (nbins, 6)
    """
    if logger is None:
        logger = logging.getLogger("qtraj_analysis.binning")

    # Filter by rapidity
    if y_window is not None:
        y0, y1 = y_window
        obs = [o for o in obs if y0 <= o.y <= y1]
        logger.info("Filtered by y window [%.2f, %.2f] -> %d trajectories", y0, y1, len(obs))

    pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    means = []
    sems = []

    for i in range(len(pt_edges) - 1):
        p0, p1 = pt_edges[i], pt_edges[i + 1]
        bin_obs = [o for o in obs if p0 <= o.pt < p1]
        
        if not bin_obs:
            logger.debug("Bin pT [%.1f, %.1f] is empty.", p0, p1)
            means.append(np.full(6, np.nan))
            sems.append(np.full(6, 0.0))
            continue

        X = np.vstack([o.surv6 for o in bin_obs])
        mu, se = mean_and_sem(X)
        means.append(mu)
        sems.append(se)

    return pt_centers, np.vstack(means), np.vstack(sems)


def compute_raa_vs_y(
    obs: List[TrajectoryObs],
    y_edges: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Groups trajectories into rapidity bins.
    """
    if logger is None:
        logger = logging.getLogger("qtraj_analysis.binning")

    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    means = []
    sems = []

    for i in range(len(y_edges) - 1):
        y0, y1 = y_edges[i], y_edges[i + 1]
        bin_obs = [o for o in obs if y0 <= o.y < y1]
        
        if not bin_obs:
            means.append(np.full(6, np.nan))
            sems.append(np.full(6, 0.0))
            continue

        X = np.vstack([o.surv6 for o in bin_obs])
        mu, se = mean_and_sem(X)
        means.append(mu)
        sems.append(se)

    return y_centers, np.vstack(means), np.vstack(sems)
