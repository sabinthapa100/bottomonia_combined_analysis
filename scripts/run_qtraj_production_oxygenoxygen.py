#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import logging
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = REPO_ROOT / "hnm" / "qtraj_out_analysis" / "scripts" / "run_oo_5360_production.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "qtraj_outputs" / "LHC" / "OxygenOxygen5p36TeV" / "production"


def _load_backend():
    spec = importlib.util.spec_from_file_location("qtraj_out_analysis_run_oo_5360_production", BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load backend module from {BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_output_root(raw: str | None) -> Path:
    if raw is None:
        return DEFAULT_OUTPUT_ROOT
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Root-level production runner for QTraj-NLO HNM O+O 5.36 TeV outputs. "
            "This is a thin wrapper over "
            "hnm/qtraj_out_analysis/scripts/run_oo_5360_production.py."
        )
    )
    parser.add_argument("--mode", choices=("noReg", "wReg", "both"), default="both")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT.relative_to(REPO_ROOT)),
        help="Repo-relative or absolute production output root.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the selected O+O production subtree before exporting.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("scripts.qtraj_production_oxygenoxygen")

    output_root = _resolve_output_root(args.output_root)
    if args.clean and output_root.exists():
        logger.info("Cleaning production outputs -> %s", output_root)
        shutil.rmtree(output_root)

    backend = _load_backend()
    return int(backend.main(["--mode", args.mode, "--out-root", str(output_root)]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
