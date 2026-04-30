from __future__ import annotations

import numpy as np

from qtraj_analysis.schema import DoubleRatioResult


def _ratio_with_err(
    num: np.ndarray,
    den: np.ndarray,
    num_err: np.ndarray,
    den_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute r = num/den with standard uncorrelated error propagation.

    r_err^2 = (num_err/den)^2 + (num*den_err/den^2)^2
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    num_err = np.asarray(num_err, dtype=np.float64)
    den_err = np.asarray(den_err, dtype=np.float64)

    r = np.full_like(num, np.nan, dtype=np.float64)
    r_err = np.full_like(num, np.nan, dtype=np.float64)

    ok = np.isfinite(num) & np.isfinite(den) & (den != 0)
    if not np.any(ok):
        return r, r_err

    r[ok] = num[ok] / den[ok]
    r_err[ok] = np.sqrt(
        (num_err[ok] / den[ok]) ** 2 + ((num[ok] * den_err[ok]) / (den[ok] ** 2)) ** 2
    )
    return r, r_err


def compute_standard_double_ratios(
    *,
    npart: np.ndarray,
    raa9_mean: np.ndarray,
    raa9_sem: np.ndarray,
    idx_1s: int = 0,
    idx_2s: int = 1,
    idx_3s: int = 5,
    idx_chi1p: int = 3,  # chi_b1(1P) in the 9-state ordering
    idx_chi2p: int = 7,  # chi_b1(2P) in the 9-state ordering
) -> DoubleRatioResult:
    """
    Compute the standard bottomonium double ratios used throughout the project.

    The default indices match `STATE_LABELS_9` in `schema.py`:
      0: Υ(1S)
      1: Υ(2S)
      3: χ_b1(1P)
      5: Υ(3S)
      7: χ_b1(2P)
    """
    npart = np.asarray(npart, dtype=np.float64)
    raa9_mean = np.asarray(raa9_mean, dtype=np.float64)
    raa9_sem = np.asarray(raa9_sem, dtype=np.float64)

    if raa9_mean.ndim != 2 or raa9_mean.shape[1] != 9:
        raise ValueError("raa9_mean must be shape (n, 9).")
    if raa9_sem.shape != raa9_mean.shape:
        raise ValueError("raa9_sem must have the same shape as raa9_mean.")
    if npart.shape[0] != raa9_mean.shape[0]:
        raise ValueError("npart length must match raa9_mean first dimension.")

    den = raa9_mean[:, idx_1s]
    den_err = raa9_sem[:, idx_1s]

    ratio_2s_1s, err_2s_1s = _ratio_with_err(
        raa9_mean[:, idx_2s], den, raa9_sem[:, idx_2s], den_err
    )
    ratio_3s_1s, err_3s_1s = _ratio_with_err(
        raa9_mean[:, idx_3s], den, raa9_sem[:, idx_3s], den_err
    )
    ratio_chi1_1s, err_chi1_1s = _ratio_with_err(
        raa9_mean[:, idx_chi1p], den, raa9_sem[:, idx_chi1p], den_err
    )
    ratio_chi2_1s, err_chi2_1s = _ratio_with_err(
        raa9_mean[:, idx_chi2p], den, raa9_sem[:, idx_chi2p], den_err
    )

    return DoubleRatioResult(
        npart=npart,
        ratio_2S_1S=ratio_2s_1s,
        ratio_3S_1S=ratio_3s_1s,
        ratio_chi1_1S=ratio_chi1_1s,
        ratio_chi2_1S=ratio_chi2_1s,
        err_2S_1S=err_2s_1s,
        err_3S_1S=err_3s_1s,
        err_chi1_1S=err_chi1_1s,
        err_chi2_1S=err_chi2_1s,
    )

