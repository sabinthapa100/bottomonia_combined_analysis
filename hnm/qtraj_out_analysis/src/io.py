import gzip
import logging
import os
from typing import List

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
    
    except (IOError, OSError) as e:
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
      Append second-to-last entry (qweight/Ntraj)
         
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
        # 12+ columns: Mathematica style (re, im) pairs -> take Re parts [0, 2, 4, 6, 8, 10]
        # 8 columns: Discrete values [v1, v2, v3, v4, v5, v6, L, qweight]
        if len(rowB) >= 12:
            # Mathematica: Table[x[[2]][[i]], {i,1,12,2}]  (1-based odd) -> python [0,2,4,6,8,10]
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
        # For 8+ columns, L is typically last, qweight is second to last.
        # For 6 columns, default to L=0, qweight=1.0.
        if len(rowB) >= 8:
            L = rowB[-1]
            qweight = rowB[-2]
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

