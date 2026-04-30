#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = REPO_ROOT / "hnm" / "qtraj_out_analysis" / "scripts" / "run_production.py"


def _load_backend():
    spec = importlib.util.spec_from_file_location("qtraj_out_analysis_run_production", BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load backend module from {BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
def _resolve_output_root(raw: str | None) -> str | None:
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path)


def main(argv: list[str] | None = None) -> int:
    backend = _load_backend()

    parser = argparse.ArgumentParser(
        description=(
            "Canonical one-step PbPb/AuAu production runner. Exports the canonical "
            "reference bundles and production figures only."
        )
    )
    backend.add_common_arguments(
        parser,
        default_system="all",
        allow_system=True,
        allow_kappa=True,
    )
    args = parser.parse_args(argv)

    if args.clean and args.observable_id:
        parser.error(
            "--clean cannot be combined with --observable-id in the public PbPb/AuAu runner. "
            "That would delete the whole system output tree and rebuild only one observable."
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("scripts.qtraj_production_pbpbauau")

    status = backend.run_export(
        system_key=args.system,
        observable_id=args.observable_id,
        skip_plots=args.skip_plots,
        clean=args.clean,
        kappa_values=args.kappa,
        output_root=_resolve_output_root(args.output_root),
        logger=logger,
    )
    if status != 0:
        return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
