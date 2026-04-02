#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_production import add_common_arguments, run_export  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = add_common_arguments(
        argparse.ArgumentParser(
            description="Export final HNM PbPb 2.76 TeV qtraj reference bundles, plots, and manifests."
        ),
        default_system="pbpb2760",
        allow_system=False,
        allow_kappa=True,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("qtraj_out_analysis.pbpb2760")
    return run_export(
        system_key="pbpb2760",
        observable_id=args.observable_id,
        skip_plots=args.skip_plots,
        kappa_values=args.kappa,
        logger=logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
