import gzip
import logging
import os
from typing import List, Optional

import numpy as np

from qtraj_analysis.schema import Record

def read_whitespace_table(path: str, logger: logging.Logger) -> List[np.ndarray]:
    """
    Read whitespace-separated numeric table from a file or .gz.
    
    Args:
        path: Path to input file (can be .gz compressed)
        logger: Logger instance
    
    Returns:
        List of numpy arrays, one per data row
    
    Raises:
        DataFileNotFoundError: If file doesn't exist
        DataFileParsingError: If file contains non-numeric data or is empty
        DataFileError: For I/O errors during reading
    """
    from qtraj_analysis.exceptions import (
        DataFileNotFoundError,
        DataFileParsingError,
        DataFileError,
    )
    
    # Validate file exists
    if not os.path.exists(path):
        raise DataFileNotFoundError(
            f"Input file not found",
            context={
                "requested_path": path,
                "absolute_path": os.path.abspath(path),
                "suggestion": "Check file path and ensure file exists"
            }
        )
    
    opener = gzip.open if path.endswith(".gz") else open
    
    try:
        with opener(path, "rt") as f:
            rows: List[List[float]] = []
            
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                
                # Skip comments and blank lines
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split()
                
                try:
                    row_data = [float(x) for x in parts]
                    rows.append(row_data)
                except ValueError as e:
                    # Specific error for parsing issues
                    raise DataFileParsingError(
                        f"Non-numeric token in {os.path.basename(path)}",
                        context={
                            "file": path,
                            "line_number": line_num,
                            "line_content": line[:100],  # Truncate very long lines
                            "num_tokens": len(parts),
                            "parse_error": str(e),
                            "suggestion": "Ensure all data is numeric (whitespace-separated)"
                        }
                    ) from e
    
    except (IOError, OSError, EOFError) as e:
        # I/O errors during reading
        raise DataFileError(
            f"I/O error reading file: {os.path.basename(path)}",
            context={
                "file": path,
                "io_error_type": type(e).__name__,
                "io_error": str(e),
            }
        ) from e
    
    # Check we got data
    if len(rows) == 0:
        raise DataFileParsingError(
            f"No data rows found in {os.path.basename(path)}",
            context={
                "file": path,
                "file_size_bytes": os.path.getsize(path),
                "suggestion": "File may be empty or contain only comments"
            }
        )
    
    logger.info("Read %d data rows from %s", len(rows), os.path.basename(path))
    return [np.array(r, dtype=np.float64) for r in rows]


def _sniff_first_value_row_width(path: str) -> int:
    """Return the column count of the first value row in a qtraj datafile.

    The qtraj datafile format alternates metadata rows and value rows,
    starting with a metadata row. This helper opens the file, skips
    blank/comment lines for the metadata row, and then returns the
    number of whitespace-separated tokens in the value row that
    immediately follows it.

    Used by :func:`load_qtraj_table` to distinguish between raw
    ``datafile.gz`` (8-column value rows: ``[v1..v6, rand_tag, L]``) and
    pre-averaged ``datafile-avg.gz`` (14-column value rows:
    ``[(mean,stderr)x6, N, L]``).
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        saw_meta = False
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not saw_meta:
                saw_meta = True
                continue
            return len(stripped.split())
    raise ValueError(
        f"Could not sniff value row width from {path}: file has no paired "
        "metadata + value rows."
    )


def load_qtraj_table(
    path: str,
    logger: logging.Logger,
    *,
    force_raw: bool = False,
) -> List[np.ndarray]:
    """Load a qtraj datafile as a list of numpy rows in ``read_whitespace_table``
    shape, regardless of whether the on-disk file is raw or pre-averaged.

    Auto-detection policy (overridden by ``force_raw``):

    * Value-row width **>= 12**: file is ``datafile-avg.gz`` format
      (Mathematica/processEvents averaged output). Loaded via
      :func:`read_whitespace_table` with zero extra work. This is the
      fast path and matches the historical behavior for PbPb2760 and
      AuAu200.
    * Value-row width **== 8**: file is a raw ``datafile.gz`` of quantum
      trajectories ``[v1..v6, rand_tag, L]``. Invokes
      :func:`qtraj_analysis.processEvents.average_raw_datafile` to
      average quantum trajectories in memory (equivalent to running
      ``processEvents.py`` to produce ``datafile-avg.gz``), then yields
      the rows in the same 14-column shape the averaged reader would
      produce. This is the path PbPb5023 uses by default.

    Return value is a flat ``list[np.ndarray]`` where entry ``2i``
    is a metadata row and entry ``2i+1`` is the corresponding value row,
    identical to the format :func:`parse_records` expects.
    """
    # Local import to avoid a circular dependency with processEvents which
    # only uses numpy/math/gzip and does not import io.py.
    from qtraj_analysis.processEvents import average_raw_datafile

    if not force_raw:
        width = _sniff_first_value_row_width(path)
        if width >= 12:
            logger.info(
                "load_qtraj_table: detected pre-averaged format (value row "
                "width=%d) for %s",
                width,
                os.path.basename(path),
            )
            try:
                return read_whitespace_table(path, logger)
            except Exception as exc:
                avg_name = os.path.basename(path)
                raw_candidate = os.path.join(os.path.dirname(path), "datafile.gz")
                if avg_name.startswith("datafile-avg") and os.path.exists(raw_candidate):
                    logger.warning(
                        "load_qtraj_table: failed to read %s (%s); falling back "
                        "to raw %s and averaging in memory.",
                        avg_name,
                        type(exc).__name__,
                        os.path.basename(raw_candidate),
                    )
                    path = raw_candidate
                else:
                    raise
        if width != 8:
            raise ValueError(
                f"Unexpected value-row width {width} in {path}; expected 8 "
                "(raw datafile.gz) or >=12 (datafile-avg.gz)."
            )

    logger.info(
        "load_qtraj_table: averaging raw datafile in memory -> %s",
        os.path.basename(path),
    )
    averaged = average_raw_datafile(path, logger=logger)

    rows: List[np.ndarray] = []
    for metadata_line, averaged_row in averaged:
        metadata_values = [float(tok) for tok in metadata_line.split()]
        rows.append(np.asarray(metadata_values, dtype=np.float64))
        rows.append(np.asarray(averaged_row, dtype=np.float64))

    logger.info(
        "load_qtraj_table: averaged %d physical trajectories (%d table rows)",
        len(averaged),
        len(rows),
    )
    return rows


def parse_records(table: List[np.ndarray], logger: logging.Logger) -> List[Record]:
    """
    Parse 2-row records from table into Record objects.
    
    Implements Mathematica:
      rawDataAll = mapData /@ Partition[Import[..., "Table"], 2]
      
    For each record (2 rows):
      row A: meta
      row B: values
      
    Mapping:
      Take entries [0, 2, 4, 6, 8, 10] (1-based odd indices 1,3,5,7,9,11)
      Append last entry (L)
      Append second-to-last entry (Ntraj for averaged qtraj inputs)
         
    Args:
        table: List of numpy arrays from file
        logger: Logger instance
        
    Returns:
        List of Record objects
        
    Raises:
        DataFileFormatError: If table has incorrect structure
    """
    from qtraj_analysis.exceptions import DataFileFormatError
    
    # Validate even number of rows
    if len(table) % 2 != 0:
        raise DataFileFormatError(
            "Data file must have even number of rows (2-row records)",
            context={
                "total_rows": len(table),
                "expected": "even number",
                "actual": f"{len(table)} (odd)",
                "suggestion": "Check if file is truncated or has missing P-wave records"
            }
        )

    recs: List[Record] = []
    for i in range(0, len(table), 2):
        meta = table[i]
        rowB = table[i + 1]

        # Handle different column counts for values row (rowB)
        # 12+ columns: Mathematica style (re, im) pairs OR processEvents.py averaged
        #   format (mean, stderr interleaved) -> take Re/mean parts [0, 2, 4, 6, 8, 10]
        # 8 columns: Raw qtraj values [v1, v2, v3, v4, v5, v6, rand_tag, L]
        if len(rowB) >= 12:
            # Mathematica: Table[x[[2]][[i]], {i,1,12,2}]  (1-based odd) -> python [0,2,4,6,8,10]
            # processEvents.py avg: [mean1,stderr1,...,mean6,stderr6,N,L] -> same extraction
            six = rowB[[0, 2, 4, 6, 8, 10]]
        elif len(rowB) >= 8:
            # Simple format: first 6 are state values
            six = rowB[:6]
        elif len(rowB) >= 6:
            # TAMU format or similar: only states
            six = rowB[:6]
        else:
            raise DataFileFormatError(
                f"Value row has insufficient columns",
                context={
                    "record_index": i // 2,
                    "row_number": i + 2,
                    "expected_min_cols": 6,
                    "actual_cols": len(rowB),
                    "suggestion": "Each value row needs at least 6 columns (states)"
                }
            )

        # L and qweight:
        # For averaged qtraj rows, L is last and the second-to-last entry
        # is Ntraj, which we use as the averaging weight to mirror the
        # Mathematica notebooks.
        # For raw 8-column qtraj rows passed directly, the second-to-last
        # entry is only a raw random-number tag, not a physical weight.
        # The production path should therefore use load_qtraj_table(),
        # which converts raw files to averaged rows first.
        # For 6-column rows, default to L=0, qweight=1.0.
        if len(rowB) >= 8:
            L = rowB[-1]
            qweight = rowB[-2] if len(rowB) >= 12 else 1.0
        else:
            L = 0
            qweight = 1.0
        
        # If qweight is 0, default to 1.0 (some output formats don't include it)
        if qweight == 0:
            qweight = 1.0
        
        vec = np.concatenate([six, [L, qweight]]).astype(np.float64)

        recs.append(Record(meta=meta, vec=vec))

    logger.info("Parsed %d 2-row records.", len(recs))
    return recs
