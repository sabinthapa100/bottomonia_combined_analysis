import logging
import numpy as np
from typing import List, Dict, Tuple

from qtraj_analysis.schema import Record, TrajectoryObs

def _meta_key(meta: np.ndarray, ndigits: int = 10) -> Tuple[float, ...]:
    """
    Group records that correspond to the same trajectory.
    We round to avoid float-key pitfalls.
    """
    return tuple(np.round(meta.astype(np.float64), ndigits).tolist())


def build_observables(
    records: List[Record],
    logger: logging.Logger,
    L_S: int = 0,
    L_P: int = 1,
) -> List[TrajectoryObs]:
    """
    Match each meta-key to both L=0 and L=1 entries, then build surv6.
    
    State ordering:
      {1S, 2S, 1P, 3S, 2P, 1D}
      
    Construction:
      S_1S = S.v6[0]
      S_2S = S.v6[1]
      S_1P = P.v6[2]   <-- Note: from P-wave record
      S_3S = S.v6[3]
      S_2P = P.v6[4]   <-- Note: from P-wave record
      S_1D = S.v6[5]
    """
    grouped: Dict[Tuple[float, ...], Dict[int, Record]] = {}
    for r in records:
        key = _meta_key(r.meta)
        grouped.setdefault(key, {})
        # if duplicates, keep the first and warn
        if r.L in grouped[key]:
            logger.warning("Duplicate record for key=%s L=%d (keeping first)", key, r.L)
        else:
            grouped[key][r.L] = r

    out: List[TrajectoryObs] = []
    missing_S = 0
    missing_P = 0

    for key, byL in grouped.items():
        rS = byL.get(L_S, None)
        rP = byL.get(L_P, None)
        if rS is None:
            missing_S += 1
            # logger.debug("Missing S-record for key=%s", key)
            continue
        if rP is None:
            missing_P += 1
            # logger.debug("Missing P-record for key=%s", key)
            continue

        # Exact Mathematica mapping:
        surv6 = np.array(
            [rS.v6[0], rS.v6[1], rP.v6[2], rS.v6[3], rP.v6[4], rS.v6[5]],
            dtype=np.float64,
        )

        # qweight: Mathematica uses x[[2,7]] later (optional). We interpret it as qweight.
        # In our Record vec: vec[7] = qweight.
        # Choose from rS by default (should match rP if file consistent).
        qweight = float(rS.qweight)

        out.append(TrajectoryObs(meta=rS.meta, surv6=surv6, qweight=qweight))

    logger.info(
        "Built %d trajectory observables. Missing S=%d, missing P=%d",
        len(out), missing_S, missing_P
    )
    
    if len(out) == 0:
        if len(records) > 0:
            logger.error("Parsed %d records but found 0 matched pairs.", len(records))
            # Just log error, caller decides if it's fatal exception or not, 
            # but task spec says raise error.
            raise ValueError(
                "No matched (L=0,L=1) trajectory pairs found. "
                "This usually means (a) L is not in last column, "
                "(b) meta-keying is wrong for your file, or "
                "(c) L values are not 0 and 1."
            )
        else:
            logger.warning("Input records list was empty.")
            
    return out
