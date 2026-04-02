#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qtraj_analysis.observable_registry import (  # noqa: E402
    format_source,
    get_observable_spec,
    iter_registry_issues,
)
from qtraj_analysis.validation import validate_registry_grid_matches_mathematica  # noqa: E402


PBPB5023_REGRESSION_IDS = (
    "pbpb5023_raavsnpart",
    "pbpb5023_raavspt",
    "pbpb5023_raavsy",
    "pbpb5023_ratio21vspt",
    "pbpb5023_ratio32vspt",
)


def _print_observable_summary(observable_id: str) -> None:
    spec = get_observable_spec(observable_id)
    print(f"[OK] {observable_id}")
    print(f"  observable_type: {spec.observable_type}")
    print(f"  state: {spec.state}")
    print(f"  grid interpretation: {spec.grid.interpretation}")
    print(f"  grid values: {list(spec.grid.values)}")
    if spec.grid.bin_edges is not None:
        print(f"  bin edges: {list(spec.grid.bin_edges)}")
    print("  Mathematica sources:")
    for source in spec.mathematica_sources:
        print(f"    - {format_source(source)}")
    print("  Experimental sources:")
    for exp in spec.experimental_observables:
        flags = []
        if exp.combined_state:
            flags.append("combined-state")
        if exp.upper_limit:
            flags.append("upper-limit")
        suffix = "" if not flags else f" [{' '.join(flags)}]"
        print(f"    - {exp.experiment} | {exp.state} | {exp.acceptance}{suffix}")
        for source in exp.sources:
            print(f"      * {format_source(source)}")
    for issue in spec.issues:
        print(f"  issue: {issue.code} :: {issue.message}")
        for source in issue.sources:
            print(f"    * {format_source(source)}")


def main() -> int:
    failed = False
    for observable_id in PBPB5023_REGRESSION_IDS:
        try:
            validate_registry_grid_matches_mathematica(observable_id)
            _print_observable_summary(observable_id)
        except Exception as exc:  # pragma: no cover - CLI-only failure path
            failed = True
            print(f"[FAIL] {observable_id}: {exc}", file=sys.stderr)

    print("\nRegistry-wide provenance issues:")
    for issue in iter_registry_issues():
        print(f"  - {issue.code}: {issue.message}")
        for source in issue.sources:
            print(f"    * {format_source(source)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
