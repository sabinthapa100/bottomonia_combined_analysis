"""
Validate qtraj-nlo / Mathematica-style datafiles for `build_observables`.

Checks:
  - Even row count (2-row records)
  - Value row has enough columns for survival + L + qweight (when applicable)
  - Meta row length for b, pT, y indices used by `schema.TrajectoryObs`
  - Successful construction of at least one matched S/P trajectory
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from qtraj_analysis.io import load_qtraj_table, parse_records, read_whitespace_table
from qtraj_analysis.matching import build_observables


@dataclass
class DatafileValidationReport:
    path: str
    ok: bool
    n_table_rows: int
    n_records: int
    n_observables: int
    meta_len: int
    value_row_lens_sample: List[int]
    b_min: float
    b_max: float
    n_unique_b: int
    messages: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Datafile: {self.path}",
            f"  ok={self.ok}  rows={self.n_table_rows}  records={self.n_records}  "
            f"observables={self.n_observables}  meta_len={self.meta_len}",
            f"  b in [{self.b_min:.6g}, {self.b_max:.6g}]  n_unique_b={self.n_unique_b}",
        ]
        for m in self.messages:
            lines.append(f"  {m}")
        return "\n".join(lines)


def validate_datafile(
    path: str,
    logger: Optional[logging.Logger] = None,
    *,
    require_meta_len: int = 7,
) -> DatafileValidationReport:
    """
    Read path, parse records, build observables. Collect diagnostics.

    `require_meta_len`: minimum meta row length so indices 0 (b), 4 (pT), 6 (y) exist.
    """
    lg = logger or logging.getLogger("qtraj_analysis.datafile_validation")
    messages: List[str] = []

    raw_table = read_whitespace_table(path, lg)
    n_table_rows = len(raw_table)
    if n_table_rows % 2 != 0:
        messages.append(f"FAIL: odd number of data rows ({n_table_rows}); need pairs for meta+value.")
        return DatafileValidationReport(
            path=path,
            ok=False,
            n_table_rows=n_table_rows,
            n_records=0,
            n_observables=0,
            meta_len=0,
            value_row_lens_sample=[],
            b_min=float("nan"),
            b_max=float("nan"),
            n_unique_b=0,
            messages=messages,
        )

    # Value row lengths from the on-disk table (every second row starting at index 1).
    _vrl = [len(raw_table[i]) for i in range(1, len(raw_table), 2)]
    value_row_lens_sample = _vrl[:8]

    logical_table = load_qtraj_table(path, lg)
    if len(logical_table) != n_table_rows:
        messages.append(
            "INFO: raw datafile was averaged in memory before validation "
            f"({n_table_rows} rows on disk -> {len(logical_table)} logical rows)."
        )

    try:
        records = parse_records(logical_table, lg)
    except Exception as exc:
        return DatafileValidationReport(
            path=path,
            ok=False,
            n_table_rows=n_table_rows,
            n_records=0,
            n_observables=0,
            meta_len=0,
            value_row_lens_sample=[],
            b_min=float("nan"),
            b_max=float("nan"),
            n_unique_b=0,
            messages=messages + [f"parse_records failed: {exc}"],
        )

    meta_len = int(len(records[0].meta)) if records else 0
    if meta_len < require_meta_len:
        messages.append(
            f"WARN: meta row length {meta_len} < {require_meta_len}; "
            "pT/y indices may be invalid."
        )

    try:
        obs = build_observables(records, lg)
        n_obs = len(obs)
    except Exception as exc:
        return DatafileValidationReport(
            path=path,
            ok=False,
            n_table_rows=n_table_rows,
            n_records=len(records),
            n_observables=0,
            meta_len=meta_len,
            value_row_lens_sample=value_row_lens_sample,
            b_min=float("nan"),
            b_max=float("nan"),
            n_unique_b=0,
            messages=messages + [f"build_observables failed: {exc}"],
        )

    bs = np.array([o.b for o in obs], dtype=np.float64)
    ub = np.unique(np.round(bs, 6))

    ok = n_obs > 0 and meta_len >= require_meta_len
    if n_obs == 0:
        messages.append("FAIL: zero trajectory observables.")
    elif ok:
        messages.append("OK: datafile is compatible with build_observables.")

    return DatafileValidationReport(
        path=path,
        ok=ok,
        n_table_rows=n_table_rows,
        n_records=len(records),
        n_observables=n_obs,
        meta_len=meta_len,
        value_row_lens_sample=value_row_lens_sample,
        b_min=float(np.min(bs)) if bs.size else float("nan"),
        b_max=float(np.max(bs)) if bs.size else float("nan"),
        n_unique_b=int(ub.size),
        messages=messages,
    )
