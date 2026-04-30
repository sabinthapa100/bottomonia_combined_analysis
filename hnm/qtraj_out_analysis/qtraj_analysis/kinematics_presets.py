"""
Shared kinematics and pp baseline for per-b spectra, OO production, and MB combination.

Single source of truth for bin edges and rapidity windows so `run_oo_5360_production`,
`run_per_b_spectra`, and `combine_mb_spectra` stay aligned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# --- Inclusive pp cross sections (nb), ordering:
# 1S, 2S, chi_b0(1P), chi_b1(1P), chi_b2(1P), 3S, chi_b0(2P), chi_b1(2P), chi_b2(2P)
# Canonical LHC 5.02 TeV–style baseline used across bundled HNM workflows
# (see reference_data._SIGMAS_EXP and run_oo_5360_production).
SIGMAS_EXP_LHC_5TEV = np.array(
    [57.6, 19.0, 3.72, 13.69, 16.1, 6.8, 3.27, 12.0, 14.15], dtype=np.float64
)

# Same baseline for PbPb 2.76 / 5.02 TeV and AuAu 200 GeV theory comparisons in this repo
# unless you override --sigmas-exp on the CLI.
SIGMAS_EXP_PBPB_2760 = SIGMAS_EXP_LHC_5TEV
SIGMAS_EXP_PBPB_5023 = SIGMAS_EXP_LHC_5TEV
SIGMAS_EXP_AUAU_200 = SIGMAS_EXP_LHC_5TEV
SIGMAS_EXP_OO_5360 = SIGMAS_EXP_LHC_5TEV

# Default binning for PbPb/AuAu per-b exports (`run_per_b_spectra`).
PT_EDGES = np.arange(0.0, 20.0 + 2.5, 2.5, dtype=np.float64)
Y_EDGES = np.arange(-5.0, 5.0 + 0.5, 0.5, dtype=np.float64)

PT_RAPIDITY_WINDOWS: Tuple[Tuple[str, Tuple[float, float], str], ...] = (
    ("midrapidity", (-1.93, 1.93), r"$-1.93 < y < 1.93$"),
    ("backward", (-5.0, -2.5), r"$-5.0 < y < -2.5$"),
    ("forward", (1.5, 4.0), r"$1.5 < y < 4.0$"),
)

PT_MAX_FOR_Y = 30.0
INTEGRATED_Y_WINDOW = PT_RAPIDITY_WINDOWS[0][1]

# --- O+O 5.36 TeV only (`run_oo_5360_production`): single fixed b, average within each bin.
# pT: 0–30 GeV in 1 GeV bins [0,1), …, [29,30].
# Use the CMS-style midrapidity acceptance for the bundled single-b OO analysis:
#   - R_AA vs pT: |y| <= 2.4
#   - R_AA vs y : y in [-2.4, 2.4], integrating trajectories with pT <= 30 GeV
OO_PT_EDGES = np.arange(0.0, 31.0, 1.0, dtype=np.float64)
OO_Y_EDGES = np.arange(-2.4, 2.4 + 0.4, 0.4, dtype=np.float64)
OO_PT_RAPIDITY_WINDOWS: Tuple[Tuple[str, Tuple[float, float], str], ...] = (
    ("midrapidity", (-2.4, 2.4), r"$-2.4 \leq y \leq 2.4$"),
)
OO_PT_MAX_FOR_Y = 30.0
OO_INTEGRATED_Y_WINDOW = OO_PT_RAPIDITY_WINDOWS[0][1]


@dataclass(frozen=True)
class SystemKinematics:
    """Per-collision-system defaults for per-b / MB analysis."""

    key: str
    sigmas_exp: np.ndarray
    energy_label: str


SYSTEM_KINEMATICS: Dict[str, SystemKinematics] = {
    "pbpb2760": SystemKinematics("pbpb2760", SIGMAS_EXP_PBPB_2760, "2.76 TeV"),
    "pbpb5023": SystemKinematics("pbpb5023", SIGMAS_EXP_PBPB_5023, "5.02 TeV"),
    "auau200": SystemKinematics("auau200", SIGMAS_EXP_AUAU_200, "200 GeV"),
    "oo5360": SystemKinematics("oo5360", SIGMAS_EXP_OO_5360, "5.36 TeV"),
}


def get_system_kinematics(system_key: str) -> SystemKinematics:
    sk = SYSTEM_KINEMATICS.get(system_key)
    if sk is None:
        allowed = ", ".join(sorted(SYSTEM_KINEMATICS))
        raise KeyError(f"Unknown system_key={system_key!r}; expected one of: {allowed}")
    return sk
