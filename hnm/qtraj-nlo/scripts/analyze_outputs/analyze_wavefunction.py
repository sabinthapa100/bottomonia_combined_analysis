#!/usr/bin/env python3
"""
analyze_wavefunction.py - Wavefunction evolution analysis and PhD-level plots.

Analyzes snapshot_*.tsv files to extract |psi(r,t)|^2 evolution.

Generates:
- |psi(r)|^2 vs r at multiple time snapshots
- 2D colormap of |psi(r,t)|^2
- Real/Imaginary parts evolution
- Probability density animations (frame sequences)
- Comparison with/without jumps
- Octet vs singlet evolution

Usage:
    python scripts/analyze_outputs/analyze_wavefunction.py
    python scripts/analyze_outputs/analyze_wavefunction.py --state 1S
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

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
    GEV_TO_FMC,
)

QTRAJ_ROOT = Path(__file__).resolve().parent.parent.parent
CAMPAIGNS_DIR = QTRAJ_ROOT / "campaigns"
ANALYSIS_DIR = QTRAJ_ROOT / "analysis"


def parse_snapshot_file(filepath):
    """Parse a snapshot_*.tsv file.

    Format: r[1/GeV], |psi|, Re(psi), Im(psi)
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            data.append([float(x) for x in parts])
    return np.array(data)


def extract_snapshot_index(filepath):
    """Extract snapshot index from filename like snapshot_00000.tsv."""
    name = Path(filepath).stem
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return 0


def collect_wavefunction_data(phase4_only=True):
    """Collect all wavefunction snapshot data.

    Returns dict: {config_key: [(snapshot_idx, data_array)]}
    config_key = (potential, state, kappa, jump_mode)
    """
    results = {}

    # Phase 4: dedicated wavefunction runs
    if phase4_only:
        phase_dir = CAMPAIGNS_DIR / "phase4_wavefunctions"
    else:
        phase_dir = CAMPAIGNS_DIR

    # First try direct config dirs (files moved by run_qtraj)
    for config_dir in sorted(phase_dir.glob("wf_*")):
        snapshots = sorted(config_dir.glob("snapshot_*.tsv"))
        if not snapshots:
            continue

        parent_name = config_dir.name
        parts = parent_name.split("_")

        if parent_name.startswith("wf_"):
            potential = parts[1]
            state = parts[2]
            kappa = int(parts[3][1:])
            jump_mode = "_".join(parts[4:]) if len(parts) > 4 else "noJumps"
        else:
            continue

        key = (potential, state, kappa, jump_mode)
        snap_data = []

        for snap_file in snapshots:
            idx = extract_snapshot_index(snap_file)
            data = parse_snapshot_file(str(snap_file))
            if len(data) > 0:
                snap_data.append((idx, data))

        if snap_data:
            snap_data.sort(key=lambda x: x[0])
            results[key] = snap_data

    # Also check output-* subdirs
    for config_dir in sorted(phase_dir.rglob("output-*")):
        snapshots = sorted(config_dir.glob("snapshot_*.tsv"))
        if not snapshots:
            continue

        parent_name = config_dir.parent.name
        parts = parent_name.split("_")

        if parent_name.startswith("wf_"):
            potential = parts[1]
            state = parts[2]
            kappa = int(parts[3][1:])
            jump_mode = "_".join(parts[4:]) if len(parts) > 4 else "noJumps"
        elif parent_name.startswith("vacuum_"):
            potential = "vacuum"
            state = parts[1]
            kappa = 0
            jump_mode = "noJumps"
        else:
            continue

        key = (potential, state, kappa, jump_mode)
        if key in results:
            continue
        snap_data = []

        for snap_file in snapshots:
            idx = extract_snapshot_index(snap_file)
            data = parse_snapshot_file(str(snap_file))
            if len(data) > 0:
                snap_data.append((idx, data))

        if snap_data:
            snap_data.sort(key=lambda x: x[0])
            results[key] = snap_data

    return results


def plot_wavefunction_snapshots(
    snap_data, state_name, potential, kappa, jump_mode, output_dir, n_snapshots=6
):
    """Plot |psi(r)|^2 at multiple time snapshots in a single panel."""
    apply_style()

    if not snap_data:
        return

    n_total = len(snap_data)
    indices = np.linspace(0, n_total - 1, min(n_snapshots, n_total), dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map for different times
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, n_total - 1)

    for i in indices:
        idx, data = snap_data[i]
        r_ginv = data[:, 0]  # r in 1/GeV
        psi_mag = data[:, 1]  # |psi|

        r_fm = r_ginv * GEV_TO_FMC
        tau_ginv = idx * 0.5  # approximate: snapFreq * dt
        tau_fmc = tau_ginv * GEV_TO_FMC

        color = cmap(norm(i))
        ax.plot(
            r_fm,
            psi_mag**2,
            color=color,
            linewidth=1.5,
            label=f"$\\tau$ = {tau_fmc:.1f} fm/c",
        )

    setup_axes(ax, r"$r$ [fm]", r"$|\psi(r)|^2$ [1/GeV]", xlim=(0, 8))
    ax.legend(loc="upper right", fontsize=10, ncol=2, framealpha=0.9)
    add_physics_note(
        ax, f"{survival_label(state_name)}\n$\\hat{{\\kappa}}={kappa}$, {jump_mode}"
    )
    add_cms_label(ax)
    add_potential_label(ax, potential)

    save_fig(
        fig, f"wf_snapshots_{potential}_{state_name}_k{kappa}_{jump_mode}", output_dir
    )
    plt.close(fig)


def plot_wavefunction_2d(
    snap_data, state_name, potential, kappa, jump_mode, output_dir
):
    """Plot 2D colormap of |psi(r,t)|^2 evolution."""
    apply_style()

    if not snap_data:
        return

    # Build 2D array
    r_fm_all = []
    psi_sq_all = []
    tau_fmc_all = []

    for idx, data in snap_data:
        r_ginv = data[:, 0]
        psi_mag = data[:, 1]

        r_fm = r_ginv * GEV_TO_FMC
        tau_fmc = idx * 0.5 * GEV_TO_FMC  # snapFreq * dt * conversion

        r_fm_all.append(r_fm)
        psi_sq_all.append(psi_mag**2)
        tau_fmc_all.append(tau_fmc)

    # Interpolate to regular grid
    r_fm = r_fm_all[0]
    n_r = len(r_fm)
    n_t = len(snap_data)

    psi_grid = np.zeros((n_t, n_r))
    for i, psi_sq in enumerate(psi_sq_all):
        psi_grid[i, :] = psi_sq

    tau_arr = np.array(tau_fmc_all)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for better visibility
    psi_grid_log = np.log10(psi_grid + 1e-15)

    im = ax.pcolormesh(r_fm, tau_arr, psi_grid_log, cmap="viridis", shading="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}|\psi(r,\tau)|^2$", fontsize=14)

    setup_axes(ax, r"$r$ [fm]", r"$\tau$ [fm/c]")
    add_physics_note(
        ax, f"{survival_label(state_name)}\n$\\hat{{\\kappa}}={kappa}$, {jump_mode}"
    )
    add_cms_label(ax)
    add_potential_label(ax, potential)

    save_fig(fig, f"wf_2d_{potential}_{state_name}_k{kappa}_{jump_mode}", output_dir)
    plt.close(fig)


def plot_real_imag_parts(
    snap_data, state_name, potential, kappa, jump_mode, output_dir, snapshot_idx=0
):
    """Plot Re(psi) and Im(psi) for a specific snapshot."""
    apply_style()

    if not snap_data or snapshot_idx >= len(snap_data):
        return

    idx, data = snap_data[snapshot_idx]
    r_ginv = data[:, 0]
    re_psi = data[:, 2]
    im_psi = data[:, 3]

    r_fm = r_ginv * GEV_TO_FMC
    tau_fmc = idx * 0.5 * GEV_TO_FMC

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(r_fm, re_psi, color="#1f77b4", linewidth=2)
    setup_axes(ax1, r"$r$ [fm]", r"${\rm Re}[\psi(r)]$")
    ax1.set_title(f"$\\tau$ = {tau_fmc:.1f} fm/c", fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2.plot(r_fm, im_psi, color="#d62728", linewidth=2)
    setup_axes(ax2, r"$r$ [fm]", r"${\rm Im}[\psi(r)]$")
    ax2.set_title(f"$\\tau$ = {tau_fmc:.1f} fm/c", fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle="--")

    add_physics_note(
        ax1, f"{survival_label(state_name)}\n$\\hat{{\\kappa}}={kappa}$, {jump_mode}"
    )
    add_cms_label(ax2)

    save_fig(
        fig,
        f"wf_reim_{potential}_{state_name}_k{kappa}_{jump_mode}_snap{idx:05d}",
        output_dir,
    )
    plt.close(fig)


def plot_jump_comparison_wavefunction(
    snap_data_nojump,
    snap_data_jump,
    state_name,
    kappa,
    output_dir,
    snapshot_indices=None,
):
    """Compare wavefunction with and without jumps."""
    apply_style()

    if not snap_data_nojump or not snap_data_jump:
        return

    n_snap = min(len(snap_data_nojump), len(snap_data_jump))
    if snapshot_indices is None:
        snapshot_indices = [0, n_snap // 3, 2 * n_snap // 3, n_snap - 1]
    snapshot_indices = [i for i in snapshot_indices if i < n_snap]

    n_panels = len(snapshot_indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for panel_idx, snap_idx in enumerate(snapshot_indices):
        ax = axes[panel_idx]

        # No jumps
        idx_nj, data_nj = snap_data_nojump[snap_idx]
        r_fm_nj = data_nj[:, 0] * GEV_TO_FMC
        psi_sq_nj = data_nj[:, 1] ** 2

        # With jumps
        idx_j, data_j = snap_data_jump[snap_idx]
        r_fm_j = data_j[:, 0] * GEV_TO_FMC
        psi_sq_j = data_j[:, 1] ** 2

        tau_fmc = idx_nj * 0.5 * GEV_TO_FMC

        ax.plot(
            r_fm_nj,
            psi_sq_nj,
            color="#1f77b4",
            linewidth=2,
            linestyle="-",
            label="No jumps",
        )
        ax.plot(
            r_fm_j,
            psi_sq_j,
            color="#d62728",
            linewidth=2,
            linestyle="--",
            label="With jumps",
        )

        setup_axes(ax, r"$r$ [fm]", r"$|\psi(r)|^2$ [1/GeV]")
        ax.set_title(f"$\\tau$ = {tau_fmc:.1f} fm/c", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(
        f"{survival_label(state_name)}, $\\hat{{\\kappa}}={kappa}$, Munich",
        fontsize=14,
        y=1.02,
    )

    save_fig(fig, f"wf_jump_comparison_{state_name}_k{kappa}", output_dir)
    plt.close(fig)


def plot_octet_vs_singlet(
    snap_data_octet, snap_data_singlet, state_name, kappa, jump_mode, output_dir
):
    """Compare octet vs singlet wavefunction evolution."""
    apply_style()

    if not snap_data_octet or not snap_data_singlet:
        return

    n_snap = min(len(snap_data_octet), len(snap_data_singlet))
    indices = [0, n_snap // 2, n_snap - 1]
    indices = [i for i in indices if i < n_snap]

    fig, axes = plt.subplots(1, len(indices), figsize=(6 * len(indices), 5))
    if len(indices) == 1:
        axes = [axes]

    for panel_idx, snap_idx in enumerate(indices):
        ax = axes[panel_idx]

        idx_o, data_o = snap_data_octet[snap_idx]
        idx_s, data_s = snap_data_singlet[snap_idx]

        r_fm_o = data_o[:, 0] * GEV_TO_FMC
        psi_sq_o = data_o[:, 1] ** 2

        r_fm_s = data_s[:, 0] * GEV_TO_FMC
        psi_sq_s = data_s[:, 1] ** 2

        tau_fmc = idx_o * 0.5 * GEV_TO_FMC

        ax.plot(r_fm_s, psi_sq_s, color="#1f77b4", linewidth=2, label="Singlet")
        ax.plot(
            r_fm_o,
            psi_sq_o,
            color="#8c564b",
            linewidth=2,
            linestyle="--",
            label="Octet",
        )

        setup_axes(ax, r"$r$ [fm]", r"$|\psi(r)|^2$ [1/GeV]")
        ax.set_title(f"$\\tau$ = {tau_fmc:.1f} fm/c", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    save_fig(fig, f"wf_octet_vs_singlet_{state_name}_k{kappa}_{jump_mode}", output_dir)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze wavefunction evolution")
    parser.add_argument("--state", type=str, nargs="+")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ANALYSIS_DIR / "wavefunction_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting wavefunction data...")
    wf_data = collect_wavefunction_data(phase4_only=True)
    print(f"  Found {len(wf_data)} configurations with snapshots")

    if not wf_data:
        print("  No wavefunction data found. Run Phase 4 first.")
        print("  python scripts/run_campaign.py --phase 4")
        return

    states_filter = args.state if args.state else None

    for (potential, state, kappa, jump_mode), snap_data in wf_data.items():
        if states_filter and state not in states_filter:
            continue

        print(
            f"\n  {potential} {state} k={kappa} {jump_mode}: {len(snap_data)} snapshots"
        )

        # 1. Wavefunction snapshots at multiple times
        plot_wavefunction_snapshots(
            snap_data, state, potential, kappa, jump_mode, output_dir, n_snapshots=8
        )

        # 2. 2D colormap
        plot_wavefunction_2d(snap_data, state, potential, kappa, jump_mode, output_dir)

        # 3. Real/Imaginary parts (first and last snapshot)
        if len(snap_data) > 1:
            plot_real_imag_parts(
                snap_data,
                state,
                potential,
                kappa,
                jump_mode,
                output_dir,
                snapshot_idx=0,
            )
            plot_real_imag_parts(
                snap_data,
                state,
                potential,
                kappa,
                jump_mode,
                output_dir,
                snapshot_idx=-1,
            )

    # 4. Jump comparisons
    print("\nGenerating jump comparisons...")
    for state in ["1S", "2S", "1P"]:
        for kappa in [4]:
            key_nj = ("munich", state, kappa, "noJumps")
            key_j = ("munich", state, kappa, "withJumps")

            if key_nj in wf_data and key_j in wf_data:
                plot_jump_comparison_wavefunction(
                    wf_data[key_nj], wf_data[key_j], state, kappa, output_dir
                )

    # 5. Octet vs singlet
    print("\nGenerating octet vs singlet comparisons...")
    for state_map in [("1S", "OctS"), ("1P", "OctP")]:
        singlet_state, octet_state = state_map
        for kappa in [4]:
            for jump_mode in ["noJumps", "withJumps"]:
                key_s = ("munich", singlet_state, kappa, jump_mode)
                key_o = ("munich", octet_state, kappa, jump_mode)

                if key_s in wf_data and key_o in wf_data:
                    plot_octet_vs_singlet(
                        wf_data[key_o],
                        wf_data[key_s],
                        singlet_state,
                        kappa,
                        jump_mode,
                        output_dir,
                    )

    print("\nDone!")


if __name__ == "__main__":
    main()
