from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class Record:
    """
    One 2-row record from the datafile:
      meta: vector from first row
      vec:  mapped vector from second row (length 8):
            [v1..v6, L, qweight]
    """
    meta: np.ndarray
    vec: np.ndarray

    @property
    def b(self) -> float:
        return float(self.meta[0])

    @property
    def pt(self) -> float:
        # Mathematica used x[[1,5]] for pT (1-based) -> python idx 4
        return float(self.meta[4]) if self.meta.size > 4 else float("nan")

    @property
    def y(self) -> float:
        # Mathematica used x[[1,7]] for y (1-based) -> python idx 6
        return float(self.meta[6]) if self.meta.size > 6 else float("nan")

    @property
    def L(self) -> int:
        return int(round(float(self.vec[6])))

    @property
    def qweight(self) -> float:
        return float(self.vec[7])

    @property
    def v6(self) -> np.ndarray:
        return self.vec[:6]


@dataclass(frozen=True)
class TrajectoryObs:
    """
    Matched S/P wave trajectory containing 6-state survival vector.
    """
    meta: np.ndarray
    surv6: np.ndarray   # shape (6,) -> {1S, 2S, 1P, 3S, 2P, 1D}
    qweight: float      # typically # of trajectories or event weight

    @property
    def b(self) -> float:
        return float(self.meta[0])

    @property
    def pt(self) -> float:
        return float(self.meta[4]) if self.meta.size > 4 else float("nan")

    @property
    def y(self) -> float:
        return float(self.meta[6]) if self.meta.size > 6 else float("nan")


@dataclass
class RaaVsBResult:
    """
    Container for R_AA vs impact parameter (b) results.
    """
    bvals: np.ndarray               # shape (nbins,)
    raa6_mean: np.ndarray           # shape (nbins, 6)
    raa6_sem: np.ndarray            # shape (nbins, 6)


@dataclass(frozen=True)
class GlauberModel:
    """
    Container for Glauber model interpolation tables.
    """
    bvsc: np.ndarray           # columns: [b, c] or [c, b]
    nbin_vs_b: np.ndarray      # columns: [b, Nbin]
    bvals: np.ndarray          # discrete b-values used in simulation
    npart_vals: np.ndarray     # Npart at those bvals

    # These methods will be attached dynamically or handled by a service,
    # but strictly speaking data classes just hold data.
    # Logic will be in glauber.py, but we keep the structure here.


# ─────────────────────────────────────────────────────────────────
# State index constants (9-state ordering after hyperfine split)
# ─────────────────────────────────────────────────────────────────

STATE_LABELS_9: List[str] = [
    "Upsilon(1S)",
    "Upsilon(2S)",
    "chi_b0(1P)",
    "chi_b1(1P)",
    "chi_b2(1P)",
    "Upsilon(3S)",
    "chi_b0(2P)",
    "chi_b1(2P)",
    "chi_b2(2P)",
]

STATE_LABELS_6: List[str] = [
    "Upsilon(1S)",
    "Upsilon(2S)",
    "chi_b(1P)",
    "Upsilon(3S)",
    "chi_b(2P)",
    "Upsilon(1D)",
]

IDX_1S: int = 0
IDX_2S: int = 1
IDX_CHI1P: int = 2   # First chi_b1(1P) after hyperfine split (index 3 in 9-state)
IDX_3S: int = 5
IDX_CHI2P: int = 7   # chi_b1(2P) after hyperfine split


@dataclass
class SurvivalResult:
    """
    Direct and feed-down survival probabilities / R_AA for all 9 states.

    All arrays are shape (n_bvals, 9) or (9,) for a single b point.
    """
    raa_direct: np.ndarray      # primordial R_AA — no feed-down applied
    raa_inclusive: np.ndarray   # inclusive R_AA  — feed-down applied
    raa_direct_sem: np.ndarray  # SEM on direct R_AA (from MCWF variance)
    raa_inclusive_sem: np.ndarray
    bvals: Optional[np.ndarray] = None   # None for single-b case
    npart: Optional[np.ndarray] = None   # None if Glauber not available


@dataclass
class DoubleRatioResult:
    """
    Double ratios R_AA[state_num] / R_AA[state_den] vs centrality.

    Errors propagated in quadrature from SEM of numerator and denominator.
    """
    npart: np.ndarray           # shape (n_pts,)
    ratio_2S_1S: np.ndarray     # R_AA[Υ2S] / R_AA[Υ1S]
    ratio_3S_1S: np.ndarray     # R_AA[Υ3S] / R_AA[Υ1S]
    ratio_chi1_1S: np.ndarray   # R_AA[χ_b1(1P)] / R_AA[Υ1S]
    ratio_chi2_1S: np.ndarray   # R_AA[χ_b1(2P)] / R_AA[Υ1S]
    err_2S_1S: np.ndarray
    err_3S_1S: np.ndarray
    err_chi1_1S: np.ndarray
    err_chi2_1S: np.ndarray


@dataclass
class FlowResult:
    """
    Elliptic flow coefficient v₂ for each bottomonium state.

    v₂ = ½ (R_AA_in − R_AA_out) / (R_AA_in + R_AA_out)

    state_indices operates on the 9-state basis after feed-down.
    """
    npart: np.ndarray           # shape (n_bins,)
    v2: np.ndarray              # shape (n_bins, 9)
    v2_err: np.ndarray          # shape (n_bins, 9) — propagated SEM
    state_labels: List[str]     # length 9


@dataclass(frozen=True)
class SourceRef:
    """
    Provenance reference for a local theory or experimental source.

    `path` is always repo-relative. `member` is used for tarball members.
    `variable` is used for notebook variables or Mathematica symbols.
    """
    path: str
    member: Optional[str] = None
    variable: Optional[str] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class GridSpec:
    """
    Exact x-grid recorded for a published Mathematica observable.

    `values` stores the canonical Mathematica x-values as exported.
    `interpretation` documents whether those values are centers, edges,
    sample points, or lower-step edges. `bin_edges` is only set when the
    true bin-edge list is known exactly.
    """
    axis: str
    interpretation: str
    values: Tuple[float, ...]
    bin_edges: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class ExperimentalObservableSpec:
    """
    Experimental dataset attached to one published theory observable.
    """
    experiment: str
    state: str
    observable_type: str
    acceptance: str
    sources: Tuple[SourceRef, ...]
    combined_state: bool = False
    upper_limit: bool = False
    note: Optional[str] = None


@dataclass(frozen=True)
class RegistryIssue:
    """
    Known provenance or consistency issue that must be surfaced explicitly.
    """
    code: str
    message: str
    sources: Tuple[SourceRef, ...] = ()


@dataclass(frozen=True)
class TheoryObservableSpec:
    """
    Canonical published qtraj observable with theory and experiment mapping.
    """
    observable_id: str
    system: str
    energy_label: str
    state: str
    observable_type: str
    acceptance: str
    published_figure: Optional[SourceRef]
    mathematica_sources: Tuple[SourceRef, ...]
    datafile_sources: Tuple[SourceRef, ...]
    grid: GridSpec
    experimental_observables: Tuple[ExperimentalObservableSpec, ...]
    issues: Tuple[RegistryIssue, ...] = ()
