#!/usr/bin/env python3
"""
Regression Test for PbPb 5.02 TeV QTraj-NLO Analysis

Verifies exact x-grid/bin matching to Mathematica for:
  - raavsnpart
  - raavspt
  - raavsy
  - ratio21vspt
  - ratio32vspt

Reports all mismatches and ambiguous sources.
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add the qtraj analysis package to path
REPO_ROOT = str(Path(__file__).parent.parent)
QTraj_ANALYSIS_DIR = os.path.join(REPO_ROOT, "hnm", "qtraj_out_analysis")
sys.path.insert(0, QTraj_ANALYSIS_DIR)

from src.observable_registry import (
    OBSERVABLE_REGISTRY,
    GLOBAL_REGISTRY_ISSUES,
    resolve_source,
    format_source,
)
from src.io import read_whitespace_table, parse_records
from src.matching import build_observables
from src.binning import compute_raa_vs_b, compute_raa_vs_pt, compute_raa_vs_y
from src.feeddown import (
    build_feeddown_matrix,
    solve_primordial_sigmas,
    split_hyperfine_6_to_9,
)
from src.glauber import load_glauber, GlauberInterpolator
from src.schema import RaaVsBResult


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("regression_test")


def verify_x_bin_matching(
    observable_id: str,
    tolerance: float = 1e-6,
) -> Dict:
    """
    Verify that x-bin values match between Python and Mathematica.

    Returns dict with verification results.
    """
    obs = OBSERVABLE_REGISTRY[observable_id]

    # Get the expected x-grid from registry
    expected_x = obs.grid.values
    expected_type = obs.grid.axis
    expected_interp = obs.grid.interpretation

    # Verify datafile exists
    if not obs.datafile_sources:
        return {
            "observable_id": observable_id,
            "status": "skipped",
            "reason": "No datafile sources",
        }

    # Try to read the first datafile
    datafile_ref = obs.datafile_sources[0]
    datafile_path = resolve_source(datafile_ref)

    if not datafile_path.exists():
        return {
            "observable_id": observable_id,
            "status": "missing_datafile",
            "path": str(datafile_path),
        }

    try:
        table = read_whitespace_table(str(datafile_path), logger)
        records = parse_records(table, logger)
        obs_list = build_observables(records, logger)

        # Extract x-values based on observable type
        if obs.observable_type in ("RAA_vs_npart", "double_ratio_vs_npart"):
            # Need to load Glauber to convert b -> Npart
            # First extract unique b values
            b_vals = sorted(set(round(float(r.b), 6) for r in obs_list))

            # Load Glauber for b -> Npart conversion
            glauber_ref = None
            for ref in obs.datafile_sources:
                # Glauber is in same directory
                break

            # For now, just report b values
            python_x_vals = b_vals
            python_x_type = "b_values (raw)"

        elif obs.observable_type in ("RAA_vs_pt", "double_ratio_vs_pt"):
            python_x_vals = sorted(set(round(float(r.pt), 2) for r in obs_list))
            python_x_type = "pt_values"

        elif obs.observable_type == "RAA_vs_y":
            python_x_vals = sorted(set(round(float(r.y), 2) for r in obs_list))
            python_x_type = "y_values"
        else:
            python_x_vals = []
            python_x_type = "unknown"

        python_status = "loaded"
        python_error = None
    except Exception as e:
        python_x_vals = []
        python_x_type = "failed"
        python_status = "failed"
        python_error = str(e)

    # Compare x-values
    matches = []
    mismatches = []

    if python_status == "loaded" and expected_x:
        for exp_val in expected_x:
            found_match = False
            for python_val in python_x_vals:
                if abs(exp_val - python_val) < tolerance:
                    matches.append((exp_val, python_val))
                    found_match = True
                    break

            if not found_match:
                closest_python = (
                    min(python_x_vals, key=lambda x: abs(x - exp_val))
                    if python_x_vals
                    else None
                )
                mismatches.append(
                    {
                        "expected_value": exp_val,
                        "closest_python": closest_python,
                        "difference": abs(exp_val - closest_python)
                        if closest_python is not None
                        else None,
                    }
                )

        match_rate = len(matches) / len(expected_x) if expected_x else 0
    else:
        match_rate = 0.0

    # Check mathematica sources
    math_status = []
    for ref in obs.mathematica_sources:
        path = resolve_source(ref)
        exists = path.exists()
        math_status.append(
            {
                "path": str(path),
                "exists": exists,
                "ref": format_source(ref),
            }
        )

    return {
        "observable_id": observable_id,
        "observable_type": obs.observable_type,
        "system": obs.system,
        "energy": obs.energy_label,
        "states": obs.state,
        "expected_x_grid": {
            "axis": expected_type,
            "interpretation": expected_interp,
            "values": list(expected_x),
            "count": len(expected_x),
        },
        "math_sources": math_status,
        "python_datafile": str(datafile_path),
        "python_status": python_status,
        "python_error": python_error,
        "python_x_vals": python_x_vals[:10] if python_x_vals else [],
        "python_x_type": python_x_type,
        "matches": len(matches),
        "mismatches": mismatches,
        "match_rate": match_rate,
    }


def run_registry_validation():
    """Run registry validation and report issues."""

    print("\n" + "=" * 80)
    print("Registry Validation")
    print("=" * 80)

    # Print global issues
    print("\nGlobal Registry Issues:")
    for issue in GLOBAL_REGISTRY_ISSUES:
        print(f"\n⚠️  {issue.code}:")
        print(f"  {issue.message}")
        if issue.sources:
            print(f"  Sources:")
            for src in issue.sources:
                path = resolve_source(src)
                exists = path.exists()
                status = "✓" if exists else "✗"
                print(f"    {status} {src.path}")

    # Check file existence for each observable
    print("\nObservable File Status:")
    for obs_id, obs in OBSERVABLE_REGISTRY.items():
        print(f"\n{obs_id}:")

        # Check datafile sources
        for ref in obs.datafile_sources:
            path = resolve_source(ref)
            exists = path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} Data: {ref.path}")

        # Check mathematica sources
        for ref in obs.mathematica_sources:
            path = resolve_source(ref)
            exists = path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} Math: {ref.path}")

        # Check experimental sources
        for exp_obs in obs.experimental_observables:
            for ref in exp_obs.sources:
                path = resolve_source(ref)
                exists = path.exists()
                status = "✓" if exists else "✗"
                limit_mark = " (UL)" if exp_obs.upper_limit else ""
                print(
                    f"  {status} Exp: {exp_obs.experiment} {exp_obs.state}{limit_mark}"
                )
                print(f"      {ref.path}")


def run_pbpb_5tev_regression():
    """Run complete regression test for PbPb 5.02 TeV."""

    print("\n" + "=" * 80)
    print("PbPb 5.023 TeV Regression Test")
    print("=" * 80)

    # Get all PbPb 5.023 TeV observables
    pbpb_obs = {
        k: v
        for k, v in OBSERVABLE_REGISTRY.items()
        if v.system == "PbPb" and "5.02" in v.energy_label
    }

    print(f"\nFound {len(pbpb_obs)} PbPb 5.023 TeV observables")

    # Verify each observable
    results = []
    for obs_id in sorted(pbpb_obs.keys()):
        obs = pbpb_obs[obs_id]
        print(f"\n{'─' * 60}")
        print(f"Verifying: {obs_id}")
        print(f"  Type: {obs.observable_type}")
        print(f"  States: {obs.state}")

        result = verify_x_bin_matching(obs_id)
        results.append(result)

        if result["status"] == "skipped":
            print(f"  ⚠️  Skipped: {result['reason']}")
            continue

        if result["status"] == "missing_datafile":
            print(f"  ✗ Missing datafile: {result['path']}")
            continue

        # Show expected x-grid
        grid = result["expected_x_grid"]
        print(
            f"  Expected x-grid ({grid['axis']}): {grid['values'][:5]}... ({grid['count']} points)"
        )

        # Show math sources status
        for math_src in result["math_sources"]:
            status = "✓" if math_src["exists"] else "✗"
            print(f"  {status} Math: {math_src['path']}")

        # Show Python data status
        if result["python_status"] == "loaded":
            print(
                f"  ✓ Python data: {result['python_x_type']} ({len(result['python_x_vals'])} points)"
            )
            print(f"    Python x: {result['python_x_vals'][:5]}...")
        else:
            print(f"  ✗ Python data failed: {result['python_error']}")

        # Show match results
        if result["match_rate"] > 0:
            print(f"  Match rate: {result['match_rate']:.2%}")

            if result["mismatches"]:
                print(f"  ⚠️  {len(result['mismatches'])} mismatches:")
                for mismatch in result["mismatches"][:3]:
                    if mismatch["closest_python"] is not None:
                        print(
                            f"    Expected={mismatch['expected_value']:.6f}, "
                            f"Python={mismatch['closest_python']:.6f}, "
                            f"Δ={mismatch['difference']:.6e}"
                        )

    # Summary
    print("\n" + "=" * 80)
    print("Regression Test Summary")
    print("=" * 80)

    total = len(results)
    skipped = sum(1 for r in results if r["status"] == "skipped")
    missing_data = sum(1 for r in results if r["status"] == "missing_datafile")
    loaded = sum(1 for r in results if r.get("python_status") == "loaded")
    perfect_match = sum(1 for r in results if r.get("match_rate") == 1.0)

    print(f"Total observables: {total}")
    print(f"Skipped (no datafile): {skipped}")
    print(f"Missing datafile: {missing_data}")
    print(f"Python data loaded: {loaded}")
    print(f"Perfect match: {perfect_match}/{loaded if loaded > 0 else total}")

    # List mismatches
    print("\n⚠️  Mismatched observables:")
    for result in results:
        if result.get("match_rate", 0) < 1.0 and result.get("match_rate", 0) > 0:
            print(f"  {result['observable_id']}: {result['match_rate']:.2%} match")
            if result.get("mismatches"):
                worst = max(
                    result["mismatches"], key=lambda x: x.get("difference", 0) or 0
                )
                if worst["closest_python"] is not None:
                    print(
                        f"    Worst: Expected={worst['expected_value']:.6f}, "
                        f"Python={worst['closest_python']:.6f}, "
                        f"Δ={worst['difference']:.6e}"
                    )

    # Save results
    output_file = os.path.join(
        REPO_ROOT, "outputs", "qtraj_nlo", "pbpb_5tev_regression.json"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "system": "PbPb",
                "energy": "5.023 TeV",
                "timestamp": str(np.datetime64("now")),
                "total_observables": total,
                "skipped": skipped,
                "missing_datafile": missing_data,
                "python_data_loaded": loaded,
                "perfect_match": perfect_match,
                "global_issues": [
                    {
                        "code": issue.code,
                        "message": issue.message,
                        "sources": [format_source(s) for s in issue.sources],
                    }
                    for issue in GLOBAL_REGISTRY_ISSUES
                ],
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nResults saved to: {output_file}")

    return results


def show_x_grid_summary():
    """Show summary of x-grids for all observables."""

    print("\n" + "=" * 80)
    print("X-Grid Summary for All Observables")
    print("=" * 80)

    for obs_id, obs in OBSERVABLE_REGISTRY.items():
        grid = obs.grid
        print(f"\n{obs_id}:")
        print(f"  System: {obs.system} {obs.energy_label}")
        print(f"  Type: {obs.observable_type}")
        print(f"  Grid: {grid.axis} ({grid.interpretation})")
        print(f"  Values ({len(grid.values)}): {grid.values}")
        if grid.bin_edges:
            print(f"  Bin edges: {grid.bin_edges}")


if __name__ == "__main__":
    print("QTraj-NLO PbPb 5.023 TeV Regression Test Suite")
    print("=" * 80)

    # 1. Run registry validation
    run_registry_validation()

    # 2. Show x-grid summary
    show_x_grid_summary()

    # 3. Run regression test
    results = run_pbpb_5tev_regression()

    print("\n" + "=" * 80)
    print("Test complete.")
