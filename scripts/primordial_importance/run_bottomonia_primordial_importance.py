#!/usr/bin/env python3
"""Run the weighted bottomonia primordial-only importance-sampling workflow.

This wrapper keeps the primordial-only analysis separate from
scripts/cnm_primordial_importance, which is reserved for CNM x primordial plots.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CNM_PRIM_DIR = REPO_ROOT / "scripts" / "cnm_primordial_importance"
if str(CNM_PRIM_DIR) not in sys.path:
    sys.path.insert(0, str(CNM_PRIM_DIR))

from run_bottomonia_primordial_importance import main as _main  # noqa: E402


def main() -> int:
    argv = list(sys.argv[1:])
    if "--output-dir" not in argv:
        argv += ["--output-dir", str(REPO_ROOT / "outputs" / "primordial_importance")]
    if "--no-cnm" not in argv:
        argv.append("--no-cnm")
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
