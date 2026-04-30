#!/usr/bin/env python3
"""CLI: validate a qtraj datafile for build_observables compatibility."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qtraj_analysis.datafile_validation import validate_datafile  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("datafile", type=str, help="Path to datafile or datafile.gz")
    p.add_argument(
        "--require-meta-len",
        type=int,
        default=7,
        help="Minimum meta row length (default 7 for b, pT, y indices)",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)
    logger = logging.getLogger("validate_qtraj_datafile")
    rep = validate_datafile(args.datafile, logger, require_meta_len=args.require_meta_len)
    print(rep.summary())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
