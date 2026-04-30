#!/usr/bin/env python3
"""
analyze_survival.py - Survival probability analysis and PhD-level plots.

Analyzes ratios.tsv files to extract survival probabilities for each
bottomonium state vs proper time, with uncertainty bands.

Generates:
- Survival probability S(tau) vs tau [fm/c] for each state
- Comparison with/without quantum jumps
- Kappa dependence plots
- Impact parameter dependence
- HEPData-compatible CSV exports

Usage:
    python scripts/analyze_outputs/analyze_survival.py --phase 2
    python scripts/analyze_outputs/analyze_survival.py --phase 2 --kappa 4
    python scripts/analyze_outputs/analyze_survival.py --phase 2 --state 1S 2S
"""

import argparse
import csv
import gzip
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for plot_style
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
    GEV_TO_FMC,
)

QTRAJ_ROOT = Path(__file__).resolve().parent.parent.parent
CAMPAIGNS_DIR = QTRAJ_ROOT / "campaigns"
ANALYSIS_DIR = QTRAJ_ROOT / "analysis"

# Column indices in ratios.tsv output (after metadata)
# The ratios are: overlap ratios for each basis state, first random #, initL
# For initType=1 (Gaussian), the columns are:
# col[0] = overlap with basis state 0 (1S-like)
# col[1] = overlap with basis state 1 (2S-like)
# ... etc
# col[-2] = first random number
# col[-1] = initL

# State-to-column mapping (based on basis function ordering)
# nBasis=5: 1S, 2S, 3S, 1P, 2P
STATE_COL_MAP = {
    "1S": 0,
    "2S": 1,
    "3S": 2,
    "1P": 3,
    "2P": 4,
}


def parse_ratios_file(filepath):
    """Parse a ratios.tsv file and return survival data."""
    metadata = []
    data = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                metadata.append(line)
            else:
                values = list(map(float, line.split("\t")))
                data.append(values)

    return metadata, data


def parse_summary_file(filepath):
    """Parse a summary.tsv file for time-evolution data.

    Format: t[1/GeV], norm, color_state, l_val, overlaps[nBasis], <r>/s0, E_vac
    """
    metadata = []
    data = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                metadata.append(line)
            else:
                values = list(map(float, line.split("\t")))
                data.append(values)

    return metadata, data


def extract_survival_from_ratios(data, state_name, n_basis=5):
    """Extract survival probability for a specific state from ratios data.

    Returns list of (survival_prob, ) per trajectory.
    """
    col = STATE_COL_MAP.get(state_name, 0)
    survivals = []

    for row in data:
        if col < len(row) - 2:  # -2 for random# and initL
            survivals.append(row[col])

    return survivals


def extract_time_evolution_from_summary(data, state_name, n_basis=5):
    """Extract time evolution from summary data.

    Returns list of (tau_fm_c, survival_prob) tuples.
    Summary format: t[1/GeV], norm, color, l, overlaps..., <r>/s0, E_vac
    """
    col = STATE_COL_MAP.get(state_name, 0)
    # Columns: 0=t, 1=norm, 2=color, 3=l, 4..4+n_basis-1=overlaps, ...
    overlap_col = 4 + col

    evolution = []
    for row in data:
        if overlap_col < len(row):
            tau_ginv = row[0]
            tau_fmc = tau_ginv * GEV_TO_FMC
            survival = row[overlap_col]
            evolution.append((tau_fmc, survival))

    return evolution


def compute_survival_stats(survival_values):
    """Compute mean and standard error from survival values."""
    if not survival_values:
        return 0.0, 0.0

    arr = np.array(survival_values)
    mean = np.mean(arr)
    stderr = np.std(arr) / math.sqrt(len(arr)) if len(arr) > 1 else 0.0

    return mean, stderr


def collect_phase2_survival(
    kappa_filter=None,
    state_filter=None,
    b_filter=None,
    nq_filter=None,
    jumps_filter=None,
):
    """Collect survival data from Phase 2 runs."""
    phase_dir = CAMPAIGNS_DIR / "phase2_munich"
    if not phase_dir.exists():
        print(f"Phase 2 directory not found: {phase_dir}")
        return {}

    # Structure: {(kappa, jump_mode, b_val, state, nq): [survival_values]}
    results = defaultdict(list)

    for kappa_dir in sorted(phase_dir.glob("kappa_*")):
        kappa = int(kappa_dir.name.split("_")[1])
        if kappa_filter and kappa not in kappa_filter:
            continue

        for jump_dir in kappa_dir.glob("*Jumps*"):
            jump_mode = "withJumps" if "with" in jump_dir.name else "noJumps"
            if jumps_filter is not None:
                expected = "withJumps" if jumps_filter else "noJumps"
                if jump_mode != expected:
                    continue

            for b_dir in sorted(jump_dir.glob("b_*")):
                b_val = float(b_dir.name.split("b_")[1])
                if b_filter and b_val not in b_filter:
                    continue

                for state_dir in sorted(b_dir.glob("state_*")):
                    state_name = state_dir.name.split("state_")[1]
                    if state_filter and state_name not in state_filter:
                        continue

                    for nq_dir in sorted(state_dir.glob("nq*")):
                        nq = int(nq_dir.name.split("nq")[1])
                        if nq_filter and nq not in nq_filter:
                            continue

                        # Find output and ratios file
                        # Files are directly in nq_dir (moved by run_qtraj)
                        ratios_file = nq_dir / "ratios.tsv"
                        if not ratios_file.exists():
                            # Fallback: look in output-* subdirs
                            output_dirs = list(nq_dir.glob("output-*"))
                            if not output_dirs:
                                continue
                            ratios_file = output_dirs[0] / "ratios.tsv"
                            if not ratios_file.exists():
                                continue

                        metadata, data = parse_ratios_file(str(ratios_file))
                        survivals = extract_survival_from_ratios(data, state_name)

                        key = (kappa, jump_mode, b_val, state_name, nq)
                        results[key].extend(survivals)

    return results


def collect_phase2_time_evolution(
    kappa_filter=None,
    state_filter=None,
    b_filter=None,
    nq_filter=None,
    jumps_filter=None,
):
    """Collect time evolution data from Phase 2 summary files."""
    phase_dir = CAMPAIGNS_DIR / "phase2_munich"
    if not phase_dir.exists():
        return {}

    # Structure: {(kappa, jump_mode, b_val, state, nq): [(tau_fmc, survival)]}
    results = defaultdict(list)

    for kappa_dir in sorted(phase_dir.glob("kappa_*")):
        kappa = int(kappa_dir.name.split("_")[1])
        if kappa_filter and kappa not in kappa_filter:
            continue

        for jump_dir in kappa_dir.glob("*Jumps*"):
            jump_mode = "withJumps" if "with" in jump_dir.name else "noJumps"
            if jumps_filter is not None:
                expected = "withJumps" if jumps_filter else "noJumps"
                if jump_mode != expected:
                    continue

            for b_dir in sorted(jump_dir.glob("b_*")):
                b_val = float(b_dir.name.split("b_")[1])
                if b_filter and b_val not in b_filter:
                    continue

                for state_dir in sorted(b_dir.glob("state_*")):
                    state_name = state_dir.name.split("state_")[1]
                    if state_filter and state_name not in state_filter:
                        continue

                    for nq_dir in sorted(state_dir.glob("nq*")):
                        nq = int(nq_dir.name.split("nq")[1])
                        if nq_filter and nq not in nq_filter:
                            continue

                        # Find summary file
                        summary_file = nq_dir / "summary.tsv"
                        if not summary_file.exists():
                            output_dirs = list(nq_dir.glob("output-*"))
                            if output_dirs:
                                summary_file = output_dirs[0] / "summary.tsv"
                            if not summary_file.exists():
                                continue

                        metadata, data = parse_summary_file(str(summary_file))
                        evolution = extract_time_evolution_from_summary(
                            data, state_name
                        )

                        key = (kappa, jump_mode, b_val, state_name, nq)
                        results[key].extend(evolution)

    return results


def plot_survival_vs_kappa(survival_data, state_name, output_dir):
    """Plot survival probability vs kappa for a specific state."""
    apply_style()

    # Group by kappa and jump_mode (use central b=0, nq=100)
    kappa_survival = defaultdict(lambda: {"noJumps": [], "withJumps": []})

    for (kappa, jump_mode, b_val, state, nq), values in survival_data.items():
        if state != state_name or b_val != 0.00 or nq != 100:
            continue
        mean, stderr = compute_survival_stats(values)
        kappa_survival[kappa][jump_mode].append((mean, stderr))

    if not kappa_survival:
        print(f"  No data for {state_name} survival vs kappa")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for jump_mode in ["noJumps", "withJumps"]:
        kappas = []
        means = []
        stderrs = []

        for kappa in sorted(kappa_survival.keys()):
            entries = kappa_survival[kappa][jump_mode]
            if entries:
                mean_vals = [e[0] for e in entries]
                err_vals = [e[1] for e in entries]
                kappas.append(kappa)
                means.append(np.mean(mean_vals))
                stderrs.append(np.sqrt(np.sum(np.array(err_vals) ** 2)) / len(err_vals))

        if kappas:
            color = survival_color("noJumps" if jump_mode == "noJumps" else "withJumps")
            ls = LINE_STYLES[jump_mode]
            ax.errorbar(
                kappas,
                means,
                yerr=stderrs,
                fmt="o-",
                color=color,
                linestyle=ls,
                linewidth=2,
                markersize=8,
                capsize=5,
                label=f"{jump_mode}",
            )

    setup_axes(
        ax, r"$\hat{\kappa}$", r"Survival Probability $S$", xlim=(1, 7), ylim=(0, 1.1)
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=12)
    add_physics_note(ax, f"PbPb 5 TeV, b=0 fm\n{survival_label(state_name)}")
    add_cms_label(ax)
    add_potential_label(ax, "munich")

    save_fig(fig, f"survival_vs_kappa_{state_name}", output_dir)
    plt.close(fig)


def plot_survival_vs_b(survival_data, state_name, kappa=4, nq=100, output_dir=None):
    """Plot survival probability vs impact parameter."""
    apply_style()

    b_survival = {"noJumps": [], "withJumps": []}

    for (k, jump_mode, b_val, state, nq), values in survival_data.items():
        if state != state_name or k != kappa or nq != nq:
            continue
        mean, stderr = compute_survival_stats(values)
        b_survival[jump_mode].append((b_val, mean, stderr))

    if not any(b_survival.values()):
        print(f"  No data for {state_name} survival vs b")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for jump_mode in ["noJumps", "withJumps"]:
        entries = sorted(b_survival[jump_mode], key=lambda x: x[0])
        if entries:
            b_vals = [e[0] for e in entries]
            means = [e[1] for e in entries]
            stderrs = [e[2] for e in entries]

            color = survival_color(jump_mode)
            ls = LINE_STYLES[jump_mode]
            ax.errorbar(
                b_vals,
                means,
                yerr=stderrs,
                fmt="o-",
                color=color,
                linestyle=ls,
                linewidth=2,
                markersize=8,
                capsize=5,
                label=f"{jump_mode}",
            )

    setup_axes(
        ax, r"Impact Parameter $b$ [fm]", r"Survival Probability $S$", ylim=(0, 1.1)
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=12)
    add_physics_note(
        ax, f"PbPb 5 TeV, $\\hat{{\\kappa}}={kappa}$\n{survival_label(state_name)}"
    )
    add_cms_label(ax)
    add_potential_label(ax, "munich")

    save_fig(fig, f"survival_vs_b_{state_name}_k{kappa}", output_dir)
    plt.close(fig)


def plot_all_states_comparison(
    survival_data, kappa=4, b_val=0.00, nq=100, jump_mode="noJumps", output_dir=None
):
    """Plot survival probability for all states in one panel."""
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    for state_name in ["1S", "2S", "3S", "1P", "2P"]:
        values = []
        for (k, jm, b, state, nq), vals in survival_data.items():
            if (
                state == state_name
                and k == kappa
                and b == b_val
                and nq == nq
                and jm == jump_mode
            ):
                values.extend(vals)

        if values:
            mean, stderr = compute_survival_stats(values)
            color = survival_color(state_name)
            ax.bar(
                state_name,
                mean,
                yerr=stderr,
                color=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=1.5,
                capsize=5,
                width=0.6,
                label=survival_label(state_name),
            )

    setup_axes(ax, "State", r"Survival Probability $S$", ylim=(0, 1.15))
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    add_physics_note(
        ax, f"PbPb 5 TeV, b={b_val} fm\n$\\hat{{\\kappa}}={kappa}$, {jump_mode}"
    )
    add_cms_label(ax)
    add_potential_label(ax, "munich")

    save_fig(fig, f"survival_all_states_k{kappa}_b{b_val:.0f}_{jump_mode}", output_dir)
    plt.close(fig)


def plot_jump_comparison(
    survival_data, state_name, kappa=4, b_val=0.00, nq=100, output_dir=None
):
    """Plot with/without jumps comparison."""
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    for jump_mode in ["noJumps", "withJumps"]:
        values = []
        for (k, jm, b, state, nq), vals in survival_data.items():
            if (
                state == state_name
                and k == kappa
                and b == b_val
                and nq == nq
                and jm == jump_mode
            ):
                values.extend(vals)

        if values:
            mean, stderr = compute_survival_stats(values)
            color = survival_color(jump_mode)
            ls = LINE_STYLES[jump_mode]
            ax.bar(
                jump_mode,
                mean,
                yerr=stderr,
                color=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=1.5,
                capsize=5,
                width=0.5,
            )

    setup_axes(ax, "", r"Survival Probability $S$", ylim=(0, 1.15))
    add_physics_note(
        ax,
        f"PbPb 5 TeV, b={b_val} fm\n$\\hat{{\\kappa}}={kappa}$, {survival_label(state_name)}",
    )
    add_cms_label(ax)
    add_potential_label(ax, "munich")

    save_fig(fig, f"jump_comparison_{state_name}_k{kappa}_b{b_val:.0f}", output_dir)
    plt.close(fig)


def export_survival_hepdata(survival_data, output_dir):
    """Export survival data in HEPData-compatible CSV format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-state, per-kappa, per-jump_mode CSV
    for state_name in ["1S", "2S", "3S", "1P", "2P"]:
        for kappa in sorted(set(k for (k, jm, b, s, nq) in survival_data.keys())):
            for jump_mode in ["noJumps", "withJumps"]:
                rows = []
                for (k, jm, b_val, state, nq), values in survival_data.items():
                    if state != state_name or k != kappa or jm != jump_mode:
                        continue
                    mean, stderr = compute_survival_stats(values)
                    rows.append(
                        {
                            "b_fm": b_val,
                            "nqtraj": nq,
                            "survival_mean": mean,
                            "survival_stderr": stderr,
                            "n_trajectories": len(values),
                        }
                    )

                if rows:
                    csv_path = (
                        output_dir / f"survival_{state_name}_k{kappa}_{jump_mode}.csv"
                    )
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "b_fm",
                                "nqtraj",
                                "survival_mean",
                                "survival_stderr",
                                "n_trajectories",
                            ],
                        )
                        writer.writeheader()
                        writer.writerows(rows)
                    print(f"    Exported: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze survival probabilities")
    parser.add_argument("--phase", type=int, default=2)
    parser.add_argument("--kappa", type=int, nargs="+")
    parser.add_argument("--state", type=str, nargs="+")
    parser.add_argument("--b", type=float, nargs="+")
    parser.add_argument("--nq", type=int, nargs="+")
    parser.add_argument("--jumps", type=int, choices=[0, 1])
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir else ANALYSIS_DIR / "survival_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting survival data...")
    survival_data = collect_phase2_survival(
        kappa_filter=args.kappa,
        state_filter=args.state,
        b_filter=args.b,
        nq_filter=args.nq,
        jumps_filter=args.jumps,
    )
    print(f"  Found {len(survival_data)} unique configurations")

    if not survival_data:
        print("  No data found. Run the campaign first.")
        return

    states = args.state if args.state else ["1S", "2S", "3S", "1P", "2P"]

    print("\nGenerating plots...")
    for state_name in states:
        print(f"  {state_name}:")
        plot_survival_vs_kappa(survival_data, state_name, output_dir)
        plot_survival_vs_b(
            survival_data, state_name, kappa=4, nq=100, output_dir=output_dir
        )
        plot_jump_comparison(
            survival_data,
            state_name,
            kappa=4,
            b_val=0.00,
            nq=100,
            output_dir=output_dir,
        )

    print("\nGenerating all-states comparison...")
    for kappa in [2, 3, 4, 5, 6]:
        for jump_mode in ["noJumps", "withJumps"]:
            plot_all_states_comparison(
                survival_data,
                kappa=kappa,
                b_val=0.00,
                nq=100,
                jump_mode=jump_mode,
                output_dir=output_dir,
            )

    print("\nExporting HEPData CSV...")
    hepdata_dir = ANALYSIS_DIR / "hepdata_csv" / "survival"
    export_survival_hepdata(survival_data, hepdata_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
