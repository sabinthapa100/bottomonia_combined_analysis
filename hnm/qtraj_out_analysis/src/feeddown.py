import logging
import numpy as np
from typing import Tuple

from qtraj_analysis.schema import RaaVsBResult
from qtraj_analysis.glauber import GlauberInterpolator

def build_feeddown_matrix() -> np.ndarray:
    """
    Exactly your Mathematica decayMatrix (9x9), then feeddownMatrix = Transpose[decayMatrix].
    State ordering:
      0 1S
      1 2S
      2 1P0
      3 1P1
      4 1P2
      5 3S
      6 2P0
      7 2P1
      8 2P2
    """
    decay = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],               # 1S
            [0.2645, 1, 0, 0, 0, 0, 0, 0, 0],          # 2S
            [0.0194, 0, 1, 0, 0, 0, 0, 0, 0],          # chi_b0(1P)
            [0.352, 0, 0, 1, 0, 0, 0, 0, 0],           # chi_b1(1P)
            [0.18, 0, 0, 0, 1, 0, 0, 0, 0],            # chi_b2(1P)
            [0.0657, 0.106, 0, 0, 0, 1, 0, 0, 0],      # 3S
            [0.0038, 0.0138, 0, 0, 0, 0, 1, 0, 0],     # chi_b0(2P)
            [0.1153, 0.181, 0, 0.0091, 0, 0, 0, 1, 0], # chi_b1(2P)
            [0.077, 0.089, 0, 0, 0.0051, 0, 0, 0, 1],  # chi_b2(2P)
        ],
        dtype=np.float64,
    )
    return decay.T  # feeddownMatrix


def split_hyperfine_6_to_9(surv6: np.ndarray) -> np.ndarray:
    """
    Mathematica:
      splitHyperfine[x_] := {x[[1]],x[[2]],x[[3]],x[[3]],x[[3]],x[[4]],x[[5]],x[[5]],x[[5]]}
    with x interpreted as {1S,2S,1P,3S,2P,(1D ignored)}
    """
    x = np.asarray(surv6, dtype=np.float64)
    if x.shape[0] < 5:
        raise ValueError("surv6 must have at least 5 entries {1S,2S,1P,3S,2P,...}.")
    return np.array([x[0], x[1], x[2], x[2], x[2], x[3], x[4], x[4], x[4]], dtype=np.float64)


def solve_primordial_sigmas(feeddown: np.ndarray, sigmas_exp: np.ndarray) -> np.ndarray:
    """
    Mathematica:
      sigmas = Inverse[feeddownMatrix] . sigmasEXP
    """
    A = np.asarray(feeddown, dtype=np.float64)
    b = np.asarray(sigmas_exp, dtype=np.float64)
    if A.shape != (9, 9) or b.shape != (9,):
        raise ValueError("feeddown must be 9x9 and sigmas_exp must be length-9.")
    return np.linalg.solve(A, b)


def compute_raa_with_feeddown_vs_b(
    raa_vs_b: RaaVsBResult,
    glauber: GlauberInterpolator,
    feeddown: np.ndarray,
    sigmas_primordial: np.ndarray,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mathematica logic (conceptually):
      nAAavgVsB[i_] := 0.1 * sigmas * splitHyperfine[spRes[[i]]] * nbinVsb[bVals[[i]]]
      computeRAAwithFeeddown:
        d = feeddown . Table[ nbin(b)*sigmas[j]*0.1 , j=1..9 ]  (denominator)
        avg = feeddown . nAAavgVsB[i]                          (numerator)
        avg/d

    We return:
      Npart(b), RAA9(b,9), and b_used
    
    Warning: RAA9 has shape (nbins, 9)
    """
    b = raa_vs_b.bvals
    nbin = glauber.b_to_nbin(b)  # fm^2 units as in your comment

    # Build primordial survival (6)->(9) for each b
    surv9 = np.vstack([split_hyperfine_6_to_9(s) for s in raa_vs_b.raa6_mean])  # (nb,9)

    # Numerator: feeddown @ (0.1 * nbin * sigmas_primordial * surv9)
    # shape details:
    #   nbin: (nb,)
    #   sigmas_primordial: (9,)
    #   surv9: (nb,9)
    # => yields_prim: (nb,9)
    yields_prim = 0.1 * (nbin[:, None] * sigmas_primordial[None, :] * surv9)
    num_inclusive = (feeddown @ yields_prim.T).T  # (nb,9)

    # Denominator: feeddown @ (0.1 * nbin * sigmas_primordial)
    denom_prim = 0.1 * (nbin[:, None] * sigmas_primordial[None, :])
    denom_inclusive = (feeddown @ denom_prim.T).T  # (nb,9)

    raa9 = np.divide(
        num_inclusive,
        denom_inclusive,
        out=np.full_like(num_inclusive, np.nan),
        where=(denom_inclusive != 0),
    )

    npart = glauber.b_to_npart(b)
    logger.info("Computed feeddown RAA9 vs b for %d b-points.", len(b))
    return npart, raa9, b


def apply_feeddown_to_raa6(
    raa6: np.ndarray,
    sem6: np.ndarray,
    feeddown: np.ndarray,
    sigmas_primordial: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies feed-down to a single survivability vector (length 6).
    Also propagates error (simplified SEM propagation).
    """
    surv9 = split_hyperfine_6_to_9(raa6)
    sem9 = split_hyperfine_6_to_9(sem6)

    # Numerator = M @ (Sigma_prim * surv9)
    num = feeddown @ (sigmas_primordial * surv9)
    # Denominator = M @ Sigma_prim
    den = feeddown @ sigmas_primordial

    raa9 = np.divide(num, den, out=np.full_like(num, np.nan), where=(den != 0))

    # Simple error propagation:
    sem9_inc = np.divide(feeddown @ (sigmas_primordial * sem9), den,
                         out=np.full_like(sem9, 0.0), where=(den != 0))

    return raa9, sem9_inc
