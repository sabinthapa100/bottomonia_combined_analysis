"""Quantum-trajectory averaging for qtraj datafiles.

This module exposes the same averaging semantics as the historical
standalone ``processEvents.py`` script that ships inside each
``inputs/qtraj_inputs/<sys>/.../<kappa>/`` directory, but as a proper
importable Python API. The legacy script remains on disk as a provenance
reference and is functionally equivalent to calling
:func:`process_datafile` from this module.

Semantics (must stay byte-equivalent to the legacy script):

* Input ``datafile.gz`` alternates metadata lines and value-row lines.
* Records are grouped by ``metadata_line + str(int(L))`` where ``L`` is
  the last column of the value row. This partitions the quantum
  trajectories by (physical trajectory, angular momentum).
* Within a group, columns ``[:-2]`` (the state-overlap values, dropping
  the raw penultimate tag and ``L``) are averaged via ``np.mean`` /
  ``np.std(ddof=0)/sqrt(N)``.
* The averaged row is written as interleaved ``[mean_i, stderr_i]`` for
  the state columns, followed by ``[N, L]``.
* Groups are emitted in ``sorted(d.keys())`` lexicographic order so the
  output file ordering is deterministic and matches the legacy script.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import math
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np


_DEFAULT_LOGGER = logging.getLogger("qtraj_analysis.processEvents")


def iter_raw_records(raw_path: str | Path) -> Iterator[Tuple[str, List[float]]]:
    """Yield ``(metadata_line, data_row)`` pairs from a raw datafile.gz.

    The metadata line is returned verbatim (including its trailing newline)
    exactly as it came off the wire so downstream consumers can re-emit
    it byte-for-byte. ``data_row`` is a ``list[float]`` parsed from the
    whitespace-separated values on the next line.
    """
    raw_path = str(raw_path)
    opener = gzip.open if raw_path.endswith(".gz") else open
    with opener(raw_path, "rb") as f:
        for metadata_bytes in f:
            metadata_line = metadata_bytes.decode("utf-8")
            try:
                value_bytes = next(f)
            except StopIteration as exc:
                raise ValueError(
                    f"Truncated datafile {raw_path}: metadata line without a "
                    "following value row."
                ) from exc
            data = list(map(float, value_bytes.decode("utf-8").split()))
            yield metadata_line, data


def average_raw_datafile(
    raw_path: str | Path,
    *,
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[str, List[float]]]:
    """Group raw quantum trajectories and average them.

    Parameters
    ----------
    raw_path
        Path to a raw ``datafile.gz`` produced by the qtraj solver.
    logger
        Optional logger for progress messages.

    Returns
    -------
    list of (metadata_line, averaged_row) tuples in ``sorted`` key order.

    The averaged row layout matches the on-disk ``datafile-avg.gz``
    format::

        [mean_1, stderr_1, mean_2, stderr_2, ..., mean_K, stderr_K, N, L]

    where ``K`` is the number of state columns (typically 6) and ``N`` is
    the number of quantum trajectories contributing to the average.
    """
    lg = logger or _DEFAULT_LOGGER
    groups: dict[str, List[List[float]]] = {}
    cnt = 0
    for metadata_line, data in iter_raw_records(raw_path):
        # Legacy key: metadata + str(int(L)). L is the last data column.
        key = metadata_line + str(int(data[-1]))
        groups.setdefault(key, []).append(data)
        cnt += 1
    lg.info(
        "Processed %d trajectory records from %s", cnt, Path(raw_path).name
    )
    lg.info("Found %d unique physical trajectories", len(groups))

    averaged: List[Tuple[str, List[float]]] = []
    for key in sorted(groups.keys()):
        data_array = np.asarray(groups[key], dtype=np.float64)
        num = data_array.shape[0]
        mean_data = np.mean(data_array, axis=0).tolist()[:-2]
        stderr_data = (
            np.std(data_array, axis=0) / math.sqrt(num)
        ).tolist()[:-2]
        combined = [
            val for pair in zip(mean_data, stderr_data) for val in pair
        ]
        # key = metadata_line + str(int(L)); the last char encodes L, and
        # key[:-1] strips it to recover the original metadata line.
        metadata_line = key[:-1]
        L = int(key[-1])
        combined = combined + [float(num), float(L)]
        averaged.append((metadata_line, combined))
    return averaged


def write_averaged_datafile(
    averaged: Iterable[Tuple[str, List[float]]],
    out_path: str | Path,
) -> Path:
    """Write averaged records to a gzipped file.

    The output matches the legacy script byte-for-byte modulo any
    Python ``str(float)`` formatting drift (which is numerically
    equivalent, not textually identical, on modern Python).
    """
    out_path = Path(out_path)
    with gzip.open(str(out_path), "wt") as f:
        for metadata_line, row in averaged:
            f.write(metadata_line)
            f.write("\t".join(map(str, row)))
            f.write("\n")
    return out_path


def process_datafile(
    raw_path: str | Path,
    out_path: str | Path | None = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Average ``raw_path`` and write the result to ``out_path``.

    If ``out_path`` is ``None`` it defaults to a sibling
    ``datafile-avg.gz`` next to ``raw_path``.
    """
    raw_path = Path(raw_path)
    if out_path is None:
        out_path = raw_path.with_name("datafile-avg.gz")
    averaged = average_raw_datafile(raw_path, logger=logger)
    return write_averaged_datafile(averaged, out_path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Average quantum trajectories in a qtraj raw datafile.gz and "
            "write a datafile-avg.gz that reproduces the legacy "
            "processEvents.py script output."
        )
    )
    parser.add_argument("raw_path", help="Path to the raw datafile.gz")
    parser.add_argument(
        "--out",
        help=(
            "Output path. Default: sibling 'datafile-avg.gz' next to "
            "the raw file."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("qtraj_analysis.processEvents")
    out = process_datafile(args.raw_path, args.out, logger=logger)
    logger.info("Wrote averaged datafile -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
