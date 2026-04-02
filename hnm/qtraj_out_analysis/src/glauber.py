from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import interp1d
except ImportError:
    interp1d = None

from qtraj_analysis.schema import GlauberModel
from qtraj_analysis.io import read_whitespace_table


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "inputs").exists() and (parent / "hnm").exists():
            return parent
    raise RuntimeError(f"Could not infer repository root from {here}")


REPO_ROOT = _find_repo_root()


@dataclass(frozen=True)
class CanonicalGlauberSpec:
    """
    Canonical HNM Glauber mapping used by the published Mathematica workflows.

    `input_base` must point to the directory that contains `glauber-data/`.
    The `bvals` and `npart_vals` are the exact discrete pairs used in the
    Mathematica notebooks to construct `bvalNpart`.
    """

    system_key: str
    system: str
    energy_label: str
    input_base: str
    source_notebook: str
    bvals: Tuple[float, ...]
    npart_vals: Tuple[float, ...]
    note: Optional[str] = None


CANONICAL_GLAUBER_SPECS: Dict[str, CanonicalGlauberSpec] = {
    "auau200": CanonicalGlauberSpec(
        system_key="auau200",
        system="AuAu",
        energy_label="200 GeV",
        input_base="inputs/qtraj_inputs/AuAu200/input",
        source_notebook="inputs/qtraj_inputs/AuAu200/raaCalculator-trajectories-qtavg-rhic.nb",
        bvals=(
            0.0,
            2.23108,
            4.07936,
            5.76909,
            7.4707,
            8.84676,
            10.0347,
            11.0956,
            12.0634,
            12.9592,
            13.8166,
            15.0803,
        ),
        npart_vals=(
            378.417670385485,
            346.3015741468757,
            289.2703882064798,
            222.07926621062614,
            153.28623190336427,
            102.22223350906017,
            64.52243696535088,
            37.649319445727855,
            19.792881616610448,
            9.205831000899767,
            3.7383741783700617,
            0.9682350180269752,
        ),
    ),
    "pbpb2760": CanonicalGlauberSpec(
        system_key="pbpb2760",
        system="PbPb",
        energy_label="2.76 TeV",
        input_base="inputs/qtraj_inputs/PbPb2760/input",
        source_notebook="inputs/qtraj_inputs/PbPb2760/raaCalculator-trajectories-qtavg-lhc-2tev.nb",
        bvals=(
            0.0,
            2.3154,
            4.23354,
            5.98713,
            7.75305,
            9.18112,
            10.4139,
            11.515,
            12.5194,
            13.4488,
            14.3335,
            15.6079,
        ),
        npart_vals=(
            406.12232235261763,
            374.9882778251423,
            315.8591222901291,
            243.53796282970384,
            168.50283662224285,
            112.39007027735359,
            70.7871426138262,
            41.05093670919672,
            21.292761520545895,
            9.667790836200798,
            3.8095328022884103,
            0.970598659138994,
        ),
        note=(
            "Matches the local 2.76 TeV Mathematica notebook and local Glauber tables. "
            "The local notebook uses the same discrete N_part sequence as the 5.02 TeV bundle."
        ),
    ),
    "pbpb5023": CanonicalGlauberSpec(
        system_key="pbpb5023",
        system="PbPb",
        energy_label="5.023 TeV",
        input_base="inputs/qtraj_inputs/PbPb5023",
        source_notebook="inputs/qtraj_inputs/PbPb5023/raaCalculator-trajectories-qtavg-lhc-5tev.nb",
        bvals=(
            0.0,
            2.32326,
            4.24791,
            6.00746,
            7.77937,
            9.21228,
            10.4493,
            11.5541,
            12.5619,
            13.4945,
            14.3815,
            15.6555,
        ),
        npart_vals=(
            406.12232235261763,
            374.9882778251423,
            315.8591222901291,
            243.53796282970384,
            168.50283662224285,
            112.39007027735359,
            70.7871426138262,
            41.05093670919672,
            21.292761520545895,
            9.667790836200798,
            3.8095328022884103,
            0.970598659138994,
        ),
    ),
    "oo5360": CanonicalGlauberSpec(
        system_key="oo5360",
        system="OO",
        energy_label="5.36 TeV",
        input_base="inputs/qtraj_inputs/OxygenOxygen5360",
        source_notebook="(none — min-bias single-b analysis)",
        bvals=(4.49691,),
        npart_vals=(0.0,),  # interpolated from Glauber table at runtime
        note=(
            "Single min-bias impact parameter for O+O at 5.36 TeV. "
            "N_part is obtained by interpolating the Glauber tables at runtime. "
            "Two modes: noReg (quantum jumps OFF) and wReg (quantum jumps ON)."
        ),
    ),
}


def list_canonical_glauber_systems() -> Tuple[str, ...]:
    return tuple(CANONICAL_GLAUBER_SPECS.keys())


def get_canonical_glauber_spec(system_key: str) -> CanonicalGlauberSpec:
    try:
        return CANONICAL_GLAUBER_SPECS[system_key]
    except KeyError as exc:
        allowed = ", ".join(sorted(CANONICAL_GLAUBER_SPECS))
        raise KeyError(
            f"Unknown Glauber system '{system_key}'. Allowed values: {allowed}"
        ) from exc


def resolve_canonical_input_base(system_key: str) -> Path:
    spec = get_canonical_glauber_spec(system_key)
    return (REPO_ROOT / spec.input_base).resolve()


def infer_canonical_glauber_system(input_base: str | os.PathLike[str]) -> Optional[str]:
    resolved = Path(input_base).resolve()
    for system_key in CANONICAL_GLAUBER_SPECS:
        if resolved == resolve_canonical_input_base(system_key):
            return system_key
    return None


def _validate_b_grid(
    observed_bvals: np.ndarray, expected_bvals: np.ndarray, *, tol: float = 5e-4
) -> None:
    observed = np.asarray(observed_bvals, dtype=np.float64)
    expected = np.asarray(expected_bvals, dtype=np.float64)
    if observed.shape != expected.shape:
        raise ValueError(
            "Observed b-grid does not match canonical Glauber grid length. "
            f"Observed {observed.shape[0]} values, expected {expected.shape[0]}."
        )
    delta = np.max(np.abs(observed - expected)) if observed.size else 0.0
    if delta > tol:
        raise ValueError(
            "Observed b-grid does not match canonical Glauber grid. "
            f"Max |Δb| = {delta:.6g} exceeds tolerance {tol:.6g}.\n"
            f"Observed: {observed}\nExpected: {expected}"
        )


def _ensure_scipy(logger: logging.Logger) -> None:
    if interp1d is None:
        raise RuntimeError(
            "scipy is required for interpolation in this script. "
            "Install it with: pip install scipy"
        )


def load_glauber(
    bvsc_path: str,
    nbinvsb_path: str,
    bvals: np.ndarray,
    npart_vals: np.ndarray,
    logger: logging.Logger,
) -> GlauberModel:
    """
    Load Glauber tables from disk and partial arrays from config.
    """
    logger.info("Loading Glauber data from: %s and %s", bvsc_path, nbinvsb_path)

    # read_whitespace_table returns List[np.ndarray] to support ragged files
    # But glauber tables are rectangular, so we stack them.
    bvsc_list = read_whitespace_table(bvsc_path, logger)
    nbin_list = read_whitespace_table(nbinvsb_path, logger)

    bvsc = np.vstack(bvsc_list)
    nbin = np.vstack(nbin_list)

    if bvsc.shape[1] < 2 or nbin.shape[1] < 2:
        raise ValueError("Glauber tables must have at least 2 columns.")

    return GlauberModel(
        bvsc=bvsc[:, :2],
        nbin_vs_b=nbin[:, :2],
        bvals=np.asarray(bvals, dtype=np.float64),
        npart_vals=np.asarray(npart_vals, dtype=np.float64),
    )


def load_canonical_glauber(
    system_key: str,
    logger: logging.Logger,
    *,
    bvals: Optional[np.ndarray] = None,
    npart_vals: Optional[np.ndarray] = None,
) -> GlauberModel:
    """
    Load the canonical HNM Glauber model for one published system.

    If `bvals` is provided, it must match the canonical notebook b-grid.
    If `npart_vals` is omitted, the canonical notebook values are used.
    """

    spec = get_canonical_glauber_spec(system_key)
    input_base = resolve_canonical_input_base(system_key)
    canonical_bvals = np.asarray(spec.bvals, dtype=np.float64)
    canonical_npart = np.asarray(spec.npart_vals, dtype=np.float64)

    if bvals is None:
        bvals = canonical_bvals
    else:
        bvals = np.asarray(bvals, dtype=np.float64)
        _validate_b_grid(bvals, canonical_bvals)

    if npart_vals is None:
        npart_vals = canonical_npart
    else:
        npart_vals = np.asarray(npart_vals, dtype=np.float64)
        if npart_vals.shape != canonical_npart.shape:
            raise ValueError(
                f"Provided npart_vals length {npart_vals.shape[0]} does not match canonical "
                f"length {canonical_npart.shape[0]} for {system_key}."
            )

    return load_glauber(
        str(input_base / "glauber-data" / "bvscData.tsv"),
        str(input_base / "glauber-data" / "nbinvsbData.tsv"),
        bvals=bvals,
        npart_vals=npart_vals,
        logger=logger,
    )


def load_glauber_from_input_base(
    input_base: str | os.PathLike[str],
    logger: logging.Logger,
    *,
    observed_bvals: Optional[np.ndarray] = None,
    npart_vals: Optional[np.ndarray] = None,
    system_key: Optional[str] = None,
) -> Tuple[GlauberModel, Optional[CanonicalGlauberSpec]]:
    """
    Load a Glauber model from an input base, preferring canonical HNM system specs.

    If the input base matches one of the thesis systems and `npart_vals` is omitted,
    the exact notebook `b -> N_part` mapping is injected automatically.
    """

    resolved_base = Path(input_base).resolve()
    guessed_key = system_key or infer_canonical_glauber_system(resolved_base)

    if guessed_key is not None:
        spec = get_canonical_glauber_spec(guessed_key)
        model = load_canonical_glauber(
            guessed_key,
            logger,
            bvals=observed_bvals,
            npart_vals=npart_vals,
        )
        return model, spec

    if observed_bvals is None or npart_vals is None:
        raise ValueError(
            "Could not infer a canonical HNM Glauber system from input_base, and no explicit "
            "observed_bvals/npart_vals were provided."
        )

    model = load_glauber(
        str(resolved_base / "glauber-data" / "bvscData.tsv"),
        str(resolved_base / "glauber-data" / "nbinvsbData.tsv"),
        bvals=np.asarray(observed_bvals, dtype=np.float64),
        npart_vals=np.asarray(npart_vals, dtype=np.float64),
        logger=logger,
    )
    return model, None


# Note: We are attaching methods to the class logic here, but since GlauberModel is
# defined in schema.py as a frozen dataclass, we can't easily add methods dynamically
# if we want to keep it clean.
#
# Better approach: Functional style or a wrapper helper.
# Given the "reproduce existing code" requirement, let's make a wrapper helper class
# or just standalone functions that take the GlauberModel data.
#
# However, the previous code had methods on the class.
# Let's create a functional interface that can be used easily.


class GlauberInterpolator:
    """
    Wrapper around GlauberModel data to provide interpolation methods.
    """

    def __init__(self, model: GlauberModel):
        self.model = model
        _ensure_scipy(logging.getLogger("qtraj_analysis.glauber"))
        self._prepare_interpolators()

    def _prepare_interpolators(self):
        # b <-> c
        # Matrix is self.model.bvsc
        x0, y0 = self.model.bvsc[:, 0], self.model.bvsc[:, 1]

        # Heuristic: b is typically large (0-20 fm), c is 0-1 (or 0-100)
        # Check max of first column
        if np.nanmax(x0) > 1.5:
            # Formatting is [b, c]
            b_pts, c_pts = x0, y0
        else:
            # Formatting is [c, b]
            c_pts, b_pts = x0, y0

        self._b_to_c_func = interp1d(
            b_pts, c_pts, kind="linear", fill_value="extrapolate"
        )
        self._c_to_b_func = interp1d(
            c_pts, b_pts, kind="linear", fill_value="extrapolate"
        )

        # Nbin(b)
        self._b_to_nbin_func = interp1d(
            self.model.nbin_vs_b[:, 0],
            self.model.nbin_vs_b[:, 1],
            kind="linear",
            fill_value="extrapolate",
        )

        # Npart(b) - from discrete bvals provided in config/CLI
        self._b_to_npart_func = interp1d(
            self.model.bvals,
            self.model.npart_vals,
            kind="linear",
            fill_value="extrapolate",
        )

    def b_to_c(self, b: np.ndarray) -> np.ndarray:
        return self._b_to_c_func(b)

    def c_to_b(self, c: np.ndarray) -> np.ndarray:
        return self._c_to_b_func(c)

    def b_to_nbin(self, b: np.ndarray) -> np.ndarray:
        return self._b_to_nbin_func(b)

    def b_to_npart(self, b: np.ndarray) -> np.ndarray:
        return self._b_to_npart_func(b)
