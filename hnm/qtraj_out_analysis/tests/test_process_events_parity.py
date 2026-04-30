"""Self-consistency tests for the processEvents integration.

For each committed qtraj kappa source, the raw ``datafile.gz`` averaged
in-memory by :func:`qtraj_analysis.io.load_qtraj_table` must produce
numerically identical :class:`TrajectoryObs` records to loading the
committed ``datafile-avg.gz`` (which was produced by the legacy
``processEvents.py`` script).

This proves three things at once:

1. The ``processEvents.py`` refactor into an importable API preserves
   the legacy script's semantics.
2. The committed ``datafile-avg.gz`` files are current w.r.t. their
   raw ``datafile.gz`` siblings (not stale).
3. ``parse_records`` + ``build_observables`` are format-agnostic: any
   downstream computation that ingests the output of ``load_qtraj_table``
   will get the same answer whether the registry points at
   ``datafile.gz`` or ``datafile-avg.gz``.

The test is SLOW (the raw-file averaging pass is ~20-30 seconds per
source on AuAu; ~5-10 seconds on PbPb). It is intentionally not part of
the default ``pytest`` run unless ``QTRAJ_RUN_PARITY=1`` is set in the
environment.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from qtraj_analysis.io import load_qtraj_table, parse_records
from qtraj_analysis.matching import build_observables


# Relative paths from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_KAPPA_SOURCES: Tuple[Tuple[str, str], ...] = (
    (
        "PbPb5023/lhc3d-k3",
        "inputs/qtraj_inputs/PbPb5023/lhc3d-k3",
    ),
    (
        "PbPb5023/lhc3d-k4",
        "inputs/qtraj_inputs/PbPb5023/lhc3d-k4",
    ),
    (
        "PbPb2760/lhc-2.76-3d-k3",
        "inputs/qtraj_inputs/PbPb2760/input/lhc-2.76-3d-k3",
    ),
    (
        "PbPb2760/lhc-2.76-3d-k4",
        "inputs/qtraj_inputs/PbPb2760/input/lhc-2.76-3d-k4",
    ),
    (
        "AuAu200/rhic-3d-kappa4",
        "inputs/qtraj_inputs/AuAu200/input/rhic-3d-kappa4",
    ),
    (
        "AuAu200/rhic-3d-kappa5",
        "inputs/qtraj_inputs/AuAu200/input/rhic-3d-kappa5",
    ),
)


def _pipeline(path: Path, logger: logging.Logger):
    rows = load_qtraj_table(str(path), logger)
    records = parse_records(rows, logger)
    return build_observables(records, logger)


def _by_meta_key(obs_list):
    out = {}
    for o in obs_list:
        key = tuple(np.round(o.meta.astype(np.float64), 10).tolist())
        out[key] = o
    return out


@pytest.mark.skipif(
    os.environ.get("QTRAJ_RUN_PARITY") != "1",
    reason="Set QTRAJ_RUN_PARITY=1 to run the slow processEvents parity suite.",
)
@pytest.mark.parametrize("label,rel_dir", _KAPPA_SOURCES, ids=[s[0] for s in _KAPPA_SOURCES])
def test_raw_vs_avg_equivalence(label: str, rel_dir: str) -> None:
    base = _REPO_ROOT / rel_dir
    raw = base / "datafile.gz"
    avg = base / "datafile-avg.gz"
    assert raw.exists(), f"Missing raw datafile: {raw}"
    assert avg.exists(), f"Missing averaged datafile: {avg}"

    logger = logging.getLogger(f"qtraj_parity.{label}")
    logger.setLevel(logging.WARNING)

    obs_avg = _pipeline(avg, logger)
    obs_raw = _pipeline(raw, logger)

    assert len(obs_avg) == len(obs_raw), (
        f"{label}: observable count mismatch — "
        f"avg={len(obs_avg)} raw={len(obs_raw)}"
    )

    d_avg = _by_meta_key(obs_avg)
    d_raw = _by_meta_key(obs_raw)

    assert set(d_avg.keys()) == set(d_raw.keys()), (
        f"{label}: meta-key set mismatch"
    )

    max_surv_diff = 0.0
    max_qweight_diff = 0.0
    worst_key = None
    for key, left in d_avg.items():
        right = d_raw[key]
        surv_diff = float(np.max(np.abs(left.surv6 - right.surv6)))
        if surv_diff > max_surv_diff:
            max_surv_diff = surv_diff
            worst_key = key
        qw_diff = abs(left.qweight - right.qweight)
        if qw_diff > max_qweight_diff:
            max_qweight_diff = qw_diff

    # Very tight tolerance: the in-memory averaging is the same numpy
    # arithmetic that produced the committed file, so we expect bit-for-
    # bit equality modulo only any str(float) re-encode drift through
    # the committed datafile-avg.gz text file.
    assert max_surv_diff <= 1e-10, (
        f"{label}: surv6 max |diff| = {max_surv_diff:.3e} "
        f"exceeds 1e-10 tolerance. Worst meta-key: {worst_key}"
    )
    assert max_qweight_diff <= 1e-10, (
        f"{label}: qweight max |diff| = {max_qweight_diff:.3e} "
        f"exceeds 1e-10 tolerance."
    )
