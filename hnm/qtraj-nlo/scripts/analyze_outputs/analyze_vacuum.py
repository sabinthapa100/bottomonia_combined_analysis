#!/usr/bin/env python3
"""
analyze_vacuum.py - Vacuum eigenstate analysis and validation.

Analyzes vacuum eigenstate runs to verify bottomonium spectrum
matches known PDG values.

Generates:
- Energy level diagram
- |psi_nℓ(r)|^2 for each state
- Comparison with PDG masses
- Vacuum wavefunction tables for HEPData

Usage:
    python scripts/analyze_outputs/analyze_vacuum.py
"""

import csv
import math
import sys
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
    GEV_TO_FMC,
    HBAR_C,
)

QTRAJ_ROOT = Path(__file__).resolve().parent.parent.parent
CAMPAIGNS_DIR = QTRAJ_ROOT / "campaigns"
ANALYSIS_DIR = QTRAJ_ROOT / "analysis"

# PDG bottomonium masses (GeV)
PDG_MASSES = {
    "1S": 9.460,  # Upsilon(1S)
    "2S": 10.023,  # Upsilon(2S)
    "3S": 10.355,  # Upsilon(3S)
    "1P": 9.893,  # chi_b(1P) average
    "2P": 10.260,  # chi_b(2P) average
}

# Reduced mass of bottomonium
M_BOT = 2.365  # GeV
M_BB = 2 * M_BOT  # ~4.73 GeV


def parse_vacuum_ratios(filepath):
    """Parse vacuum ratios.tsv to extract eigenenergies."""
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


def parse_vacuum_snapshot(filepath):
    """Parse vacuum snapshot file for wavefunction."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            data.append([float(x) for x in parts])
    return np.array(data)


def collect_vacuum_data():
    """Collect vacuum eigenstate data from Phase 1."""
    phase_dir = CAMPAIGNS_DIR / "phase1_vacuum"
    if not phase_dir.exists():
        print(f"Phase 1 directory not found: {phase_dir}")
        return {}

    results = {}

    for state_dir in sorted(phase_dir.glob("vacuum_*")):
        state_name = state_dir.name.replace("vacuum_", "")

        # Look for files directly in state_dir (moved by run_qtraj)
        entry = {
            "state": state_name,
            "output_dir": str(state_dir),
            "ratios_file": None,
            "snapshot_files": [],
        }

        ratios = state_dir / "ratios.tsv"
        if ratios.exists():
            entry["ratios_file"] = str(ratios)

        # Also check output-* subdirs
        if not entry["ratios_file"]:
            for out_dir in sorted(state_dir.glob("output-*")):
                ratios = out_dir / "ratios.tsv"
                if ratios.exists():
                    entry["ratios_file"] = str(ratios)
                    entry["output_dir"] = str(out_dir)
                    break

        snapshots = sorted(state_dir.glob("snapshot_*.tsv"))
        if not snapshots:
            for out_dir in sorted(state_dir.glob("output-*")):
                snapshots = sorted(out_dir.glob("snapshot_*.tsv"))
                if snapshots:
                    break

        entry["snapshot_files"] = [str(s) for s in snapshots]

        if entry["ratios_file"] or entry["snapshot_files"]:
            results[state_name] = entry

    return results


def extract_eigenenergies(metadata):
    """Extract eigenenergies from metadata comments."""
    energies = {}
    for line in metadata:
        # Look for energy information in comments
        if "energy" in line.lower() or "eigen" in line.lower():
            parts = line.split()
            for i, p in enumerate(parts):
                try:
                    val = float(p)
                    if -20 < val < 20:  # reasonable energy range
                        energies.setdefault(len(energies), val)
                except ValueError:
                    pass
    return energies


def plot_energy_level_diagram(vacuum_data, output_dir):
    """Plot energy level diagram comparing with PDG."""
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    states = ["1S", "2S", "3S", "1P", "2P"]
    x_positions = np.arange(len(states))

    # Plot PDG values as reference
    pdg_energies = []
    for state in states:
        if state in PDG_MASSES:
            pdg_energies.append(PDG_MASSES[state] - M_BB)  # Binding energy
        else:
            pdg_energies.append(None)

    # Plot computed energies from vacuum runs
    computed_energies = []
    for state in states:
        if state in vacuum_data and vacuum_data[state]["ratios_file"]:
            metadata, data = parse_vacuum_ratios(vacuum_data[state]["ratios_file"])
            energies = extract_eigenenergies(metadata)
            if energies:
                computed_energies.append(list(energies.values())[0])
            else:
                computed_energies.append(None)
        else:
            computed_energies.append(None)

    width = 0.35

    # PDG bars
    pdg_valid = [(i, e) for i, e in enumerate(pdg_energies) if e is not None]
    if pdg_valid:
        idx = [p[0] for p in pdg_valid]
        vals = [p[1] for p in pdg_valid]
        ax.bar(
            [x - width / 2 for x in idx],
            vals,
            width,
            color="lightgray",
            edgecolor="black",
            linewidth=1.5,
            label="PDG",
            alpha=0.7,
        )

    # Computed bars
    comp_valid = [(i, e) for i, e in enumerate(computed_energies) if e is not None]
    if comp_valid:
        idx = [c[0] for c in comp_valid]
        vals = [c[1] for c in comp_valid]
        ax.bar(
            [x + width / 2 for x in idx],
            vals,
            width,
            color="#1f77b4",
            edgecolor="black",
            linewidth=1.5,
            label="qtraj-nlo (vacuum)",
            alpha=0.7,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([survival_label(s) for s in states], fontsize=12)
    setup_axes(ax, "State", r"Binding Energy [GeV]")
    ax.axhline(y=0, color="black", linewidth=1, linestyle="-")
    ax.legend(loc="upper right", framealpha=0.9)
    add_physics_note(ax, "Bottomonium Vacuum Spectrum\nMunich Potential, T=0")

    save_fig(fig, "vacuum_energy_levels", output_dir)
    plt.close(fig)


def plot_vacuum_wavefunctions(vacuum_data, output_dir):
    """Plot vacuum wavefunctions |psi(r)|^2 for each state."""
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    for state_name in ["1S", "2S", "3S", "1P", "2P"]:
        if state_name not in vacuum_data:
            continue

        entry = vacuum_data[state_name]
        if not entry["snapshot_files"]:
            continue

        # Use first snapshot (vacuum is time-independent)
        snap_data = parse_vacuum_snapshot(entry["snapshot_files"][0])
        if len(snap_data) < 2:
            continue

        r_fm = snap_data[:, 0] * GEV_TO_FMC
        psi_sq = snap_data[:, 1] ** 2

        color = survival_color(state_name)
        ax.plot(
            r_fm, psi_sq, color=color, linewidth=2, label=survival_label(state_name)
        )

    setup_axes(ax, r"$r$ [fm]", r"$|\psi(r)|^2$ [1/GeV]", xlim=(0, 5))
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    add_physics_note(ax, "Vacuum Wavefunctions\nT = 0, Munich Potential")

    save_fig(fig, "vacuum_wavefunctions", output_dir)
    plt.close(fig)


def plot_individual_wavefunctions(vacuum_data, output_dir):
    """Plot individual wavefunction panels for each state."""
    apply_style()

    for state_name in ["1S", "2S", "3S", "1P", "2P"]:
        if state_name not in vacuum_data:
            continue

        entry = vacuum_data[state_name]
        if not entry["snapshot_files"]:
            continue

        snap_data = parse_vacuum_snapshot(entry["snapshot_files"][0])
        if len(snap_data) < 2:
            continue

        r_fm = snap_data[:, 0] * GEV_TO_FMC
        psi_sq = snap_data[:, 1] ** 2
        re_psi = snap_data[:, 2]
        im_psi = snap_data[:, 3]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(r_fm, psi_sq, color=survival_color(state_name), linewidth=2)
        setup_axes(ax1, r"$r$ [fm]", r"$|\psi(r)|^2$ [1/GeV]")
        ax1.set_title(survival_label(state_name), fontsize=14)
        ax1.grid(True, alpha=0.3, linestyle="--")

        ax2.plot(r_fm, re_psi, color="#1f77b4", linewidth=2, label="Re")
        ax2.plot(r_fm, im_psi, color="#d62728", linewidth=2, label="Im")
        setup_axes(ax2, r"$r$ [fm]", r"$\psi(r)$")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle="--")

        save_fig(fig, f"vacuum_wf_{state_name}", output_dir)
        plt.close(fig)


def export_vacuum_hepdata(vacuum_data, output_dir):
    """Export vacuum eigenstate data in HEPData format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for state_name in ["1S", "2S", "3S", "1P", "2P"]:
        if state_name not in vacuum_data:
            continue

        entry = vacuum_data[state_name]
        if not entry["snapshot_files"]:
            continue

        snap_data = parse_vacuum_snapshot(entry["snapshot_files"][0])
        if len(snap_data) < 2:
            continue

        csv_path = output_dir / f"vacuum_wavefunction_{state_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["r_fm", "psi_magnitude", "re_psi", "im_psi", "psi_squared"]
            )
            for row in snap_data:
                r_fm = row[0] * GEV_TO_FMC
                writer.writerow(
                    [
                        f"{r_fm:.6f}",
                        f"{row[1]:.8e}",
                        f"{row[2]:.8e}",
                        f"{row[3]:.8e}",
                        f"{row[1] ** 2:.8e}",
                    ]
                )
        print(f"    Exported: {csv_path}")

    # Energy level comparison
    csv_path = output_dir / "vacuum_energy_levels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "pdg_mass_GeV", "pdg_binding_energy_GeV"])
        for state in ["1S", "2S", "3S", "1P", "2P"]:
            if state in PDG_MASSES:
                writer.writerow(
                    [
                        state,
                        f"{PDG_MASSES[state]:.3f}",
                        f"{PDG_MASSES[state] - M_BB:.3f}",
                    ]
                )
    print(f"    Exported: {csv_path}")


def main():
    output_dir = ANALYSIS_DIR / "vacuum_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting vacuum eigenstate data...")
    vacuum_data = collect_vacuum_data()
    print(f"  Found {len(vacuum_data)} states")

    if not vacuum_data:
        print("  No vacuum data found. Run Phase 1 first.")
        print("  python scripts/run_campaign.py --phase 1")
        return

    print("\nGenerating plots...")
    plot_energy_level_diagram(vacuum_data, output_dir)
    plot_vacuum_wavefunctions(vacuum_data, output_dir)
    plot_individual_wavefunctions(vacuum_data, output_dir)

    print("\nExporting HEPData CSV...")
    hepdata_dir = ANALYSIS_DIR / "hepdata_csv" / "vacuum"
    export_vacuum_hepdata(vacuum_data, hepdata_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
