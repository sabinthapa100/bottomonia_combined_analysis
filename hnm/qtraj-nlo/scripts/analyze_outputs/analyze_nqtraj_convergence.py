#!/usr/bin/env python3
"""
analyze_nqtraj_convergence.py - NQTRAJ convergence study.

Analyzes how survival probability converges with number of quantum trajectories.

Generates:
- Survival vs NQTRAJ with error bars
- Relative error vs NQTRAJ
- Convergence threshold determination

Usage:
    python scripts/analyze_outputs/analyze_nqtraj_convergence.py
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (
    apply_style,
    setup_axes,
    add_physics_note,
    save_fig,
    survival_color,
    survival_label,
    add_cms_label,
    MARKERS,
)

QTRAJ_ROOT = Path(__file__).resolve().parent.parent.parent
CAMPAIGNS_DIR = QTRAJ_ROOT / "campaigns"
ANALYSIS_DIR = QTRAJ_ROOT / "analysis"


def collect_convergence_data():
    """Collect data from Phase 5 convergence study."""
    phase_dir = CAMPAIGNS_DIR / "phase5_nqtraj_convergence"
    if not phase_dir.exists():
        print(f"Phase 5 directory not found: {phase_dir}")
        return {}

    # {(state, nq): [survival_values]}
    results = defaultdict(list)

    for config_dir in sorted(phase_dir.glob("conv_*")):
        # Parse: conv_1S_k4_nq20
        parts = config_dir.name.split("_")
        state = parts[1]
        nq = int(parts[3][2:])

        # Find ratios file directly in config_dir (moved by run_qtraj)
        ratios_file = config_dir / "ratios.tsv"
        if not ratios_file.exists():
            output_dirs = list(config_dir.glob("output-*"))
            if not output_dirs:
                continue
            ratios_file = output_dirs[0] / "ratios.tsv"
            if not ratios_file.exists():
                continue

        with open(ratios_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                values = list(map(float, line.split("\t")))
                if len(values) > 2:
                    results[(state, nq)].append(values[0])

    return results


def plot_convergence(convergence_data, states, output_dir):
    """Plot survival probability vs NQTRAJ for each state."""
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    for state_name in states:
        nq_vals = []
        means = []
        stderrs = []

        for (state, nq), values in sorted(convergence_data.items()):
            if state != state_name:
                continue
            if values:
                nq_vals.append(nq)
                means.append(np.mean(values))
                stderrs.append(np.std(values) / math.sqrt(len(values)))

        if nq_vals:
            color = survival_color(state_name)
            marker = MARKERS.get(nq_vals[0], "o")
            ax.errorbar(
                nq_vals,
                means,
                yerr=stderrs,
                fmt=f"{marker}-",
                color=color,
                linewidth=2,
                markersize=8,
                capsize=5,
                label=survival_label(state_name),
            )

    setup_axes(ax, r"$N_{\rm traj}$", r"Survival Probability $S$")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=11)
    add_physics_note(ax, "PbPb 5 TeV, b=0 fm\n$\\hat{\\kappa}=4$, with jumps, Munich")
    add_cms_label(ax)

    save_fig(fig, "nqtraj_convergence", output_dir)
    plt.close(fig)


def plot_relative_error(convergence_data, states, output_dir):
    """Plot relative error vs NQTRAJ."""
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    for state_name in states:
        nq_vals = []
        rel_errors = []

        for (state, nq), values in sorted(convergence_data.items()):
            if state != state_name:
                continue
            if len(values) > 1:
                mean = np.mean(values)
                if mean > 0:
                    rel_err = np.std(values) / mean
                    nq_vals.append(nq)
                    rel_errors.append(rel_err)

        if nq_vals:
            color = survival_color(state_name)
            ax.plot(
                nq_vals,
                rel_errors,
                color=color,
                linewidth=2,
                marker="o",
                markersize=8,
                label=survival_label(state_name),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    setup_axes(ax, r"$N_{\rm traj}$", r"Relative Error $\sigma/S$")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    add_physics_note(ax, "Convergence Study\nPbPb 5 TeV, b=0 fm")

    # Add 1%, 5%, 10% reference lines
    for threshold in [0.01, 0.05, 0.10]:
        ax.axhline(y=threshold, color="gray", linewidth=1, linestyle="--", alpha=0.5)
        ax.text(
            1.02,
            threshold,
            f"{threshold * 100:.0f}%",
            transform=ax.get_yaxis_transform(),
            fontsize=10,
            verticalalignment="center",
            color="gray",
        )

    save_fig(fig, "nqtraj_relative_error", output_dir)
    plt.close(fig)


def export_convergence_hepdata(convergence_data, output_dir):
    """Export convergence data in HEPData format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "nqtraj_convergence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "state",
                "nqtraj",
                "survival_mean",
                "survival_stderr",
                "relative_error",
                "n_trajectories",
            ]
        )

        for (state, nq), values in sorted(convergence_data.items()):
            if values:
                mean = np.mean(values)
                stderr = (
                    np.std(values) / math.sqrt(len(values)) if len(values) > 1 else 0
                )
                rel_err = stderr / mean if mean > 0 else 0
                writer.writerow(
                    [
                        state,
                        nq,
                        f"{mean:.6f}",
                        f"{stderr:.6f}",
                        f"{rel_err:.6f}",
                        len(values),
                    ]
                )

    print(f"    Exported: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze NQTRAJ convergence")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ANALYSIS_DIR / "nqtraj_convergence_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting convergence data...")
    convergence_data = collect_convergence_data()
    print(f"  Found {len(convergence_data)} state-NQTRAJ combinations")

    if not convergence_data:
        print("  No data found. Run Phase 5 first.")
        return

    states = ["1S", "2S", "1P"]

    print("\nGenerating plots...")
    plot_convergence(convergence_data, states, output_dir)
    plot_relative_error(convergence_data, states, output_dir)

    print("\nExporting HEPData CSV...")
    hepdata_dir = ANALYSIS_DIR / "hepdata_csv" / "convergence"
    export_convergence_hepdata(convergence_data, hepdata_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
