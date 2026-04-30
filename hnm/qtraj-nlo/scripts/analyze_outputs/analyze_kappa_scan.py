#!/usr/bin/env python3
"""
analyze_kappa_scan.py - Kappa dependence analysis.

Analyzes how survival probability depends on kappa-hat.

Generates:
- R_AA vs kappa for each state
- Kappa band plots (uncertainty bands)
- State-by-state kappa sensitivity

Usage:
    python scripts/analyze_outputs/analyze_kappa_scan.py
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
    add_potential_label,
    LINE_STYLES,
)

QTRAJ_ROOT = Path(__file__).resolve().parent.parent.parent
CAMPAIGNS_DIR = QTRAJ_ROOT / "campaigns"
ANALYSIS_DIR = QTRAJ_ROOT / "analysis"


def collect_kappa_data(jump_mode="noJumps", b_val=0.00, nq=100):
    """Collect survival data across kappa values."""
    phase_dir = CAMPAIGNS_DIR / "phase2_munich"
    if not phase_dir.exists():
        return {}

    # {(state, kappa): [survival_values]}
    results = defaultdict(list)

    for kappa_dir in sorted(phase_dir.glob("kappa_*")):
        kappa = int(kappa_dir.name.split("_")[1])

        for jump_dir in kappa_dir.glob("*Jumps*"):
            jm = "withJumps" if "with" in jump_dir.name else "noJumps"
            if jm != jump_mode:
                continue

            for b_dir in sorted(jump_dir.glob("b_*")):
                b = float(b_dir.name.split("b_")[1])
                if abs(b - b_val) > 0.01:
                    continue

                for state_dir in sorted(b_dir.glob("state_*")):
                    state = state_dir.name.split("state_")[1]

                    for nq_dir in sorted(state_dir.glob("nq*")):
                        nq_actual = int(nq_dir.name.split("nq")[1])
                        if nq_actual != nq:
                            continue

                        # Find ratios file directly in nq_dir
                        ratios_file = nq_dir / "ratios.tsv"
                        if not ratios_file.exists():
                            output_dirs = list(nq_dir.glob("output-*"))
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
                                # Column 0 = 1S-like overlap
                                if len(values) > 2:
                                    results[(state, kappa)].append(values[0])

    return results


def plot_kappa_band(kappa_data, states, output_dir):
    """Plot R_AA vs kappa with uncertainty bands for all states."""
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    for state_name in states:
        kappas = []
        means = []
        stderrs = []

        for (state, kappa), values in sorted(kappa_data.items()):
            if state != state_name:
                continue
            if values:
                kappas.append(kappa)
                means.append(np.mean(values))
                stderrs.append(np.std(values) / math.sqrt(len(values)))

        if kappas:
            kappas = np.array(kappas)
            means = np.array(means)
            stderrs = np.array(stderrs)

            color = survival_color(state_name)

            # Fill band
            ax.fill_between(
                kappas, means - stderrs, means + stderrs, color=color, alpha=0.2
            )

            # Line
            ax.plot(
                kappas,
                means,
                color=color,
                linewidth=2.5,
                marker="o",
                markersize=8,
                label=survival_label(state_name),
            )

    setup_axes(
        ax,
        r"$\hat{\kappa}$",
        r"Survival Probability $S$",
        xlim=(1.5, 6.5),
        ylim=(0, 1.1),
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=12)
    add_physics_note(ax, "PbPb 5 TeV, b=0 fm\nMunich Potential, no jumps")
    add_cms_label(ax)

    save_fig(fig, "kappa_band_all_states", output_dir)
    plt.close(fig)


def plot_kappa_sensitivity(kappa_data, states, output_dir):
    """Plot dS/dkappa sensitivity for each state."""
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    for state_name in states:
        kappas = []
        means = []

        for (state, kappa), values in sorted(kappa_data.items()):
            if state != state_name:
                continue
            if values:
                kappas.append(kappa)
                means.append(np.mean(values))

        if len(kappas) > 1:
            kappas = np.array(kappas)
            means = np.array(means)

            # Numerical derivative
            dS_dk = np.gradient(means, kappas)

            color = survival_color(state_name)
            ax.plot(
                kappas[1:-1] if len(kappas) > 2 else kappas,
                dS_dk[1:-1] if len(dS_dk) > 2 else dS_dk,
                color=color,
                linewidth=2,
                marker="o",
                markersize=6,
                label=survival_label(state_name),
            )

    setup_axes(ax, r"$\hat{\kappa}$", r"$dS/d\hat{\kappa}$")
    ax.axhline(y=0, color="black", linewidth=1, linestyle="-", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=11)
    add_physics_note(ax, "Kappa Sensitivity\nPbPb 5 TeV, b=0 fm")

    save_fig(fig, "kappa_sensitivity", output_dir)
    plt.close(fig)


def export_kappa_hepdata(kappa_data, output_dir):
    """Export kappa scan data in HEPData format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "kappa_scan.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["state", "kappa", "survival_mean", "survival_stderr", "n_trajectories"]
        )

        for (state, kappa), values in sorted(kappa_data.items()):
            if values:
                writer.writerow(
                    [
                        state,
                        kappa,
                        f"{np.mean(values):.6f}",
                        f"{np.std(values) / math.sqrt(len(values)):.6f}",
                        len(values),
                    ]
                )

    print(f"    Exported: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze kappa dependence")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--jumps", type=int, choices=[0, 1], default=0)

    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir else ANALYSIS_DIR / "kappa_scan_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    jump_mode = "withJumps" if args.jumps else "noJumps"

    print(f"Collecting kappa scan data (jumps={jump_mode})...")
    kappa_data = collect_kappa_data(jump_mode=jump_mode)
    print(f"  Found {len(kappa_data)} state-kappa combinations")

    if not kappa_data:
        print("  No data found. Run Phase 2 first.")
        return

    states = ["1S", "2S", "3S", "1P", "2P"]

    print("\nGenerating plots...")
    plot_kappa_band(kappa_data, states, output_dir)
    plot_kappa_sensitivity(kappa_data, states, output_dir)

    print("\nExporting HEPData CSV...")
    hepdata_dir = ANALYSIS_DIR / "hepdata_csv" / "kappa_scan"
    export_kappa_hepdata(kappa_data, hepdata_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
