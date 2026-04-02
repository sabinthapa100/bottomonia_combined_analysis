#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
QTRAJ_OUT_ANALYSIS_SCRIPTS = REPO_ROOT / "hnm" / "qtraj_out_analysis" / "scripts"
if str(QTRAJ_OUT_ANALYSIS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(QTRAJ_OUT_ANALYSIS_SCRIPTS))

from run_production import SYSTEM_CHOICES, run_export  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parent-level HNM qtraj production runner. "
            "Exports thesis/publication bundles from qtraj_out_analysis into outputs/qtraj_outputs."
        )
    )
    parser.add_argument(
        "--system",
        default="all",
        choices=SYSTEM_CHOICES,
        help="System to export: all, auau200, pbpb2760, or pbpb5023",
    )
    parser.add_argument(
        "--observable-id",
        help="Optional single observable id override",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write data/manifests only",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the selected system's outputs/qtraj_outputs/.../production subtree before exporting",
    )
    parser.add_argument(
        "--kappa",
        nargs="+",
        type=int,
        help="Optional source filter. PbPb accepts 3 4; AuAu accepts 4 5.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/qtraj_outputs",
        help="Repo-relative or absolute output root. Default: outputs/qtraj_outputs",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("hnm.qtraj_production")

    return run_export(
        system_key=args.system,
        observable_id=args.observable_id,
        skip_plots=args.skip_plots,
        clean=args.clean,
        kappa_values=args.kappa,
        output_root=args.output_root,
        logger=logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
