# -*- coding: utf-8 -*-
"""
ups_particle.py
===============
Particle definitions for quarkonia: Upsilon (bottomonia) and J/psi (charmonia).

Provides:
  - QuarkoniumSystem  dataclass:  state names, masses, formation times, sigmas, feeddown matrix
  - BOTTOMONIA_SYSTEM : 9-state Upsilon system (as in mathematica/code/primordial_module_bottomonia.py)
  - CHARMONIA_SYSTEM  : 5-state J/psi system

All sigmas are *observed* (inclusive) cross sections in nb or relative units, at the
relevant collision energy; update via the factory functions.

Feed-down matrices F are defined such that:
    sigma_obs = F @ sigma_dir
so sigma_dir = F_inv @ sigma_obs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

# Bottomonia states (9-state basis)
BOTTOM_STATE_NAMES: List[str] = [
    "ups1S",      # Υ(1S)
    "ups2S",      # Υ(2S)
    "chib0_1P",   # χb0(1P)
    "chib1_1P",   # χb1(1P)
    "chib2_1P",   # χb2(1P)
    "ups3S",      # Υ(3S)
    "chib0_2P",   # χb0(2P)
    "chib1_2P",   # χb1(2P)
    "chib2_2P",   # χb2(2P)
]

# Charmonia states (5-state basis)
CHARM_STATE_NAMES: List[str] = [
    "jpsi_1S",    # J/ψ(1S)
    "chic0_1P",   # χc0(1P)
    "chic1_1P",   # χc1(1P)
    "chic2_1P",   # χc2(1P)
    "psi_2S",     # ψ(2S)
]

# Derived convenience sets
BOTTOM_UPSILON_STATES = ["ups1S", "ups2S", "ups3S"]

# ---------------------------------------------------------------------------
# QuarkoniumSystem
# ---------------------------------------------------------------------------

@dataclass
class QuarkoniumSystem:
    """
    Full physical description of a quarkonium family.

    Parameters
    ----------
    name       : 'bottomonia' or 'charmonia'
    state_names: ordered list (must match suppression-column order in datafile)
    masses_GeV : vacuum masses [GeV] for each state (same order)
    tau_form_fm: rest-frame formation times [fm/c] for each state (same order)
    sigma_obs  : observed (inclusive) cross sections [arbitrary units] for each state (same order)
    F          : feed-down matrix F with F @ sigma_dir = sigma_obs
    """
    name:        str
    state_names: List[str]
    masses_GeV:  np.ndarray     # shape (N,)
    tau_form_fm: np.ndarray     # shape (N,)
    sigma_obs:   np.ndarray     # shape (N,)
    F:           np.ndarray     # shape (N, N)

    def __post_init__(self):
        n = len(self.state_names)
        self.masses_GeV  = np.asarray(self.masses_GeV,  dtype=float)
        self.tau_form_fm = np.asarray(self.tau_form_fm, dtype=float)
        self.sigma_obs   = np.asarray(self.sigma_obs,   dtype=float)
        self.F           = np.asarray(self.F,           dtype=float)
        assert self.masses_GeV.shape  == (n,), f"masses_GeV must have {n} entries"
        assert self.tau_form_fm.shape == (n,), f"tau_form_fm must have {n} entries"
        assert self.sigma_obs.shape   == (n,), f"sigma_obs must have {n} entries"
        assert self.F.shape == (n, n),          f"F must be ({n},{n})"
        self._F_inv = np.linalg.inv(self.F)

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @property
    def F_inv(self) -> np.ndarray:
        return self._F_inv

    @property
    def sigma_dir(self) -> np.ndarray:
        """Direct (exclusive) cross sections derived by inverting feed-down."""
        return self._F_inv @ self.sigma_obs

    def index(self, name: str) -> int:
        return self.state_names.index(name)


# ---------------------------------------------------------------------------
# Bottomonia system factory
# ---------------------------------------------------------------------------

def make_bottomonia_system(
    sqrts_pp_GeV: float = 5020.0,
    sigma_obs_override: np.ndarray | None = None,
) -> QuarkoniumSystem:
    """
    9-state Υ system.

    Masses from PDG; formation times from typical TAMU values (2r_Υ convention).
    sigma_obs at 5.02 TeV from Mathematica notebook (raaCalculator-tamu-OO.nb).
    sqrt(s_NN) scaling applied as (sqrt_s / 5020)^0.5 to direct cross sections.

    States (→ column order in bottomonia datafile.gz):
        [Υ(1S), Υ(2S), χb0(1P), χb1(1P), χb2(1P), Υ(3S), χb0(2P), χb1(2P), χb2(2P)]

    Feed-down matrix from Du et al. / Mathematica notebook.
    """
    # Masses [GeV], PDG
    masses = np.array([
        9.4603,   # Υ(1S)
        10.0233,  # Υ(2S)
        9.8594,   # χb0(1P)
        9.8928,   # χb1(1P)
        9.9122,   # χb2(1P)
        10.3552,  # Υ(3S)
        10.2325,  # χb0(2P)
        10.2554,  # χb1(2P)
        10.2686,  # χb2(2P)
    ], dtype=float)

    # Formation times [fm/c] — 2 r_Υ convention (values from TAMU)
    tau_form = np.array([
        0.76,   # Υ(1S)
        0.96,   # Υ(2S) ~ larger radius
        1.1,    # χb0(1P)
        1.1,    # χb1(1P)
        1.1,    # χb2(1P)
        1.16,   # Υ(3S)
        1.2,    # χb0(2P)
        1.2,    # χb1(2P)
        1.2,    # χb2(2P)
    ], dtype=float)

    # Observed sigmas at 5.02 TeV (scale factor applied below)
    # Values: [57.6, 19.0, 3.72, 13.69, 16.1, 6.8, 3.27, 12.0, 14.15] nb
    # (from primordial_module_bottomonia.py / Mathematica notebook)
    sigma_obs_5020 = np.array(
        [57.6, 19.0, 3.72, 13.69, 16.1, 6.8, 3.27, 12.0, 14.15],
        dtype=float
    )
    # Energy scaling (applied to all observed sigmas analogously)
    # Note: the Mathematica code for OO 5.36 TeV actually uses the 5.02 TeV cross sections identically
    # rather than scaling them. So we do NOT scale them by sqrt(5360/5020).
    scale = 1.0
    if sqrts_pp_GeV != 5020.0 and sqrts_pp_GeV != 5360.0:
        scale = math.sqrt(float(sqrts_pp_GeV) / 5020.0)

    sigma_obs = (sigma_obs_override
                 if sigma_obs_override is not None
                 else sigma_obs_5020 * scale)

    # Feed-down matrix F (from primordial_module_bottomonia.py / Mathematica)
    F = np.array([
        [1.0,    0.2645, 0.0194, 0.352,  0.18,   0.0657, 0.0038, 0.1153, 0.077 ],
        [0.0,    1.0,    0.0,    0.0,    0.0,    0.106,  0.0138, 0.181,  0.089 ],
        [0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0   ],
        [0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0091, 0.0   ],
        [0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0051],
        [0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0   ],
        [0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0   ],
        [0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0   ],
        [0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0   ],
    ], dtype=float)

    return QuarkoniumSystem(
        name="bottomonia",
        state_names=BOTTOM_STATE_NAMES,
        masses_GeV=masses,
        tau_form_fm=tau_form,
        sigma_obs=sigma_obs,
        F=F,
    )


# ---------------------------------------------------------------------------
# Charmonia system factory
# ---------------------------------------------------------------------------

def make_charmonia_system(
    sqrts_pp_GeV: float = 5020.0,
    sigma_obs_override: np.ndarray | None = None,
) -> QuarkoniumSystem:
    """
    5-state J/ψ system.

    States (→ column order in charmonia datafile.gz):
        [J/ψ(1S), χc0(1P), χc1(1P), χc2(1P), ψ(2S)]
    """
    # Masses [GeV], PDG
    masses = np.array([
        3.0969,  # J/ψ(1S)
        3.4148,  # χc0(1P)
        3.5107,  # χc1(1P)
        3.5562,  # χc2(1P)
        3.6861,  # ψ(2S)
    ], dtype=float)

    # Formation times [fm/c]
    tau_form = np.array([1.0, 1.5, 1.5, 1.5, 2.5], dtype=float)

    # Observed sigmas at 5.02 TeV [nb] (from primordial_module.py)
    sigma_obs_5020 = np.array([21.91, 0.377, 0.386, 0.355, 3.26], dtype=float)
    scale = math.sqrt(float(sqrts_pp_GeV) / 5020.0) if sqrts_pp_GeV != 5020.0 else 1.0
    sigma_obs = (sigma_obs_override
                 if sigma_obs_override is not None
                 else sigma_obs_5020 * scale)

    # Feed-down matrix F (from primordial_module.py)
    F = np.array([
        [1.0, 0.0141, 0.343, 0.195, 0.615 ],
        [0.0, 1.0,    0.0,   0.0,   0.0977],
        [0.0, 0.0,    1.0,   0.0,   0.0975],
        [0.0, 0.0,    0.0,   1.0,   0.0936],
        [0.0, 0.0,    0.0,   0.0,   1.0   ],
    ], dtype=float)

    return QuarkoniumSystem(
        name="charmonia",
        state_names=CHARM_STATE_NAMES,
        masses_GeV=masses,
        tau_form_fm=tau_form,
        sigma_obs=sigma_obs,
        F=F,
    )
