"""Tests for datafile_validation and min_bias_combine."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from qtraj_analysis.datafile_validation import validate_datafile
from qtraj_analysis.io import load_qtraj_table, parse_records
from qtraj_analysis.matching import build_observables
from qtraj_analysis.min_bias_combine import weighted_average_raa9


def _write_minimal_datafile(path: Path) -> None:
    """Two matched (L=0,L=1) pairs sharing identical meta; one trajectory."""
    # meta: b, ..., pT at [4], y at [6]; length >= 7
    meta = np.array([2.5, 0.0, 0.0, 0.0, 1.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # S-wave value row: 6 states + qweight + L
    row_s = np.array([0.95, 0.90, 0.85, 0.92, 0.88, 0.75, 1.0, 0.0], dtype=np.float64)
    # P-wave value row
    row_p = np.array([0.95, 0.90, 0.82, 0.92, 0.79, 0.75, 1.0, 1.0], dtype=np.float64)
    lines = []
    for row in (meta, row_s, meta, row_p):
        lines.append(" ".join(str(x) for x in row))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_validate_datafile_ok(tmp_path: Path) -> None:
    p = tmp_path / "mini.txt"
    _write_minimal_datafile(p)
    lg = logging.getLogger("test")
    rep = validate_datafile(str(p), lg)
    assert rep.ok
    assert rep.n_observables == 1
    assert rep.n_unique_b == 1


def test_weighted_average_raa9_shape() -> None:
    raa = np.ones((3, 4, 9))
    raa[0, :, 0] = 0.5
    w = np.array([1.0, 1.0, 2.0])
    comb, sem = weighted_average_raa9(raa, w, np.ones_like(raa) * 0.1)
    assert comb.shape == (4, 9)
    assert sem is not None and sem.shape == (4, 9)


def test_load_qtraj_table_averages_raw_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "mini_raw.txt"
    meta = np.array([2.5, 0.0, 0.0, 0.0, 1.2, 0.0, 0.3], dtype=np.float64)
    row_s1 = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.3, 0.25, 0.0], dtype=np.float64)
    row_s2 = np.array([0.7, 0.6, 0.2, 0.5, 0.4, 0.1, 0.75, 0.0], dtype=np.float64)
    row_p1 = np.array([0.0, 0.0, 0.5, 0.0, 0.7, 0.0, 0.10, 1.0], dtype=np.float64)
    row_p2 = np.array([0.0, 0.0, 0.7, 0.0, 0.9, 0.0, 0.90, 1.0], dtype=np.float64)
    lines = []
    for row in (meta, row_s1, meta, row_s2, meta, row_p1, meta, row_p2):
        lines.append(" ".join(str(x) for x in row))
    p.write_text("\n".join(lines), encoding="utf-8")

    lg = logging.getLogger("test")
    rows = load_qtraj_table(str(p), lg)
    obs = build_observables(parse_records(rows, lg), lg)

    assert len(obs) == 1
    np.testing.assert_allclose(obs[0].surv6, [0.8, 0.7, 0.6, 0.6, 0.8, 0.2])
    assert obs[0].qweight == pytest.approx(2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
