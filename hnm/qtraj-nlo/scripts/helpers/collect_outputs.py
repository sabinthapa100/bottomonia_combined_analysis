#!/usr/bin/env python3
"""
collect_outputs.py - Collect and organize raw qtraj-nlo outputs.

Scans campaign directories, extracts ratios.tsv and summary.tsv files,
and creates consolidated data files for analysis.

Usage:
    python scripts/helpers/collect_outputs.py --phase 2
    python scripts/helpers/collect_outputs.py --phase 2 --kappa 4
    python scripts/helpers/collect_outputs.py --all
"""

import argparse
import csv
import gzip
import os
import sys
from pathlib import Path

QTRAJ_ROOT = Path(__file__).resolve().parent.parent.parent
CAMPAIGNS_DIR = QTRAJ_ROOT / "campaigns"
ANALYSIS_DIR = QTRAJ_ROOT / "analysis"


def find_output_dirs(phase_dir, pattern="**/output-*"):
    """Find all output directories in a campaign phase."""
    return list(Path(phase_dir).glob(pattern))


def parse_ratios_tsv(filepath):
    """Parse a ratios.tsv file and return metadata + data."""
    metadata_lines = []
    data_lines = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                metadata_lines.append(line)
            else:
                data_lines.append(list(map(float, line.split("\t"))))

    return metadata_lines, data_lines


def parse_summary_tsv(filepath):
    """Parse a summary.tsv file (time-evolution data)."""
    metadata_lines = []
    data_lines = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                metadata_lines.append(line)
            else:
                parts = line.split("\t")
                data_lines.append(list(map(float, parts)))

    return metadata_lines, data_lines


def parse_snapshot_tsv(filepath):
    """Parse a snapshot_*.tsv file (wavefunction data)."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            data.append(list(map(float, parts)))
    return data


def collect_phase2_outputs(
    kappa_filter=None,
    state_filter=None,
    b_filter=None,
    nq_filter=None,
    jumps_filter=None,
):
    """Collect all outputs from Phase 2 (Munich potential)."""
    phase_dir = CAMPAIGNS_DIR / "phase2_munich"
    if not phase_dir.exists():
        print(f"Phase 2 directory not found: {phase_dir}")
        return []

    results = []

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

                        # Find output directory
                        output_dirs = list(nq_dir.glob("output-*"))
                        if not output_dirs:
                            continue

                        out_dir = output_dirs[0]

                        entry = {
                            "phase": 2,
                            "potential": "munich",
                            "kappa": kappa,
                            "jump_mode": jump_mode,
                            "b_val": b_val,
                            "state": state_name,
                            "nqtraj": nq,
                            "output_dir": str(out_dir),
                            "ratios_file": None,
                            "summary_file": None,
                            "snapshot_files": [],
                        }

                        ratios = out_dir / "ratios.tsv"
                        if ratios.exists():
                            entry["ratios_file"] = str(ratios)

                        summary = out_dir / "summary.tsv"
                        if summary.exists():
                            entry["summary_file"] = str(summary)

                        snapshots = sorted(out_dir.glob("snapshot_*.tsv"))
                        entry["snapshot_files"] = [str(s) for s in snapshots]

                        results.append(entry)

    return results


def collect_phase1_outputs():
    """Collect outputs from Phase 1 (vacuum eigenstates)."""
    phase_dir = CAMPAIGNS_DIR / "phase1_vacuum"
    if not phase_dir.exists():
        print(f"Phase 1 directory not found: {phase_dir}")
        return []

    results = []
    for state_dir in sorted(phase_dir.glob("vacuum_*")):
        state_name = state_dir.name.split("vacuum_")[1]
        output_dirs = list(state_dir.glob("output-*"))
        if not output_dirs:
            continue

        out_dir = output_dirs[0]
        entry = {
            "phase": 1,
            "state": state_name,
            "output_dir": str(out_dir),
            "ratios_file": None,
            "summary_file": None,
            "snapshot_files": [],
        }

        ratios = out_dir / "ratios.tsv"
        if ratios.exists():
            entry["ratios_file"] = str(ratios)

        summary = out_dir / "summary.tsv"
        if summary.exists():
            entry["summary_file"] = str(summary)

        snapshots = sorted(out_dir.glob("snapshot_*.tsv"))
        entry["snapshot_files"] = [str(s) for s in snapshots]

        results.append(entry)

    return results


def collect_phase4_outputs():
    """Collect outputs from Phase 4 (wavefunction snapshots)."""
    phase_dir = CAMPAIGNS_DIR / "phase4_wavefunctions"
    if not phase_dir.exists():
        print(f"Phase 4 directory not found: {phase_dir}")
        return []

    results = []
    for config_dir in sorted(phase_dir.glob("wf_*")):
        output_dirs = list(config_dir.glob("output-*"))
        if not output_dirs:
            continue

        out_dir = output_dirs[0]
        # Parse config name: wf_munich_1S_k4_noJumps
        parts = config_dir.name.split("_")
        potential = parts[1]
        state = parts[2]
        kappa = int(parts[3][1:])
        jump_mode = "_".join(parts[4:]) if len(parts) > 4 else "noJumps"

        entry = {
            "phase": 4,
            "potential": potential,
            "state": state,
            "kappa": kappa,
            "jump_mode": jump_mode,
            "output_dir": str(out_dir),
            "ratios_file": None,
            "summary_file": None,
            "snapshot_files": [],
        }

        ratios = out_dir / "ratios.tsv"
        if ratios.exists():
            entry["ratios_file"] = str(ratios)

        summary = out_dir / "summary.tsv"
        if summary.exists():
            entry["summary_file"] = str(summary)

        snapshots = sorted(out_dir.glob("snapshot_*.tsv"))
        entry["snapshot_files"] = [str(s) for s in snapshots]

        results.append(entry)

    return results


def consolidate_ratios(results, output_path):
    """Consolidate all ratios into a single datafile.gz."""
    count = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(str(output_path), "wt") as out:
        for entry in results:
            if entry.get("ratios_file"):
                with open(entry["ratios_file"], "r") as f:
                    content = f.read()
                    out.write(content)
                    if not content.endswith("\n"):
                        out.write("\n")
                    count += 1

    print(f"  Consolidated {count} ratio files -> {output_path}")
    return output_path


def consolidate_summaries(results, output_path):
    """Consolidate all summary files."""
    count = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        for entry in results:
            if entry.get("summary_file"):
                with open(entry["summary_file"], "r") as f:
                    content = f.read()
                    out.write(
                        f"# === {entry.get('state', 'unknown')} b={entry.get('b_val', 'N/A')} ===\n"
                    )
                    out.write(content)
                    if not content.endswith("\n"):
                        out.write("\n")
                    count += 1

    print(f"  Consolidated {count} summary files -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Collect qtraj-nlo outputs")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--all", action="store_true", help="Collect all phases")
    parser.add_argument("--kappa", type=int, nargs="+")
    parser.add_argument("--state", type=str, nargs="+")
    parser.add_argument("--b", type=float, nargs="+")
    parser.add_argument("--nq", type=int, nargs="+")
    parser.add_argument("--jumps", type=int, choices=[0, 1])
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.phase == 1:
        print("Collecting Phase 1 (vacuum) outputs...")
        results = collect_phase1_outputs()
        print(f"  Found {len(results)} configurations")
        if results:
            consolidate_ratios(results, out_dir / "phase1_ratios.tsv.gz")

    if args.all or args.phase == 2:
        print("Collecting Phase 2 (Munich) outputs...")
        results = collect_phase2_outputs(
            kappa_filter=args.kappa,
            state_filter=args.state,
            b_filter=args.b,
            nq_filter=args.nq,
            jumps_filter=args.jumps,
        )
        print(f"  Found {len(results)} configurations")
        if results:
            consolidate_ratios(results, out_dir / "phase2_ratios.tsv.gz")
            consolidate_summaries(results, out_dir / "phase2_summaries.tsv")

    if args.all or args.phase == 4:
        print("Collecting Phase 4 (wavefunction) outputs...")
        results = collect_phase4_outputs()
        print(f"  Found {len(results)} configurations")
        if results:
            consolidate_ratios(results, out_dir / "phase4_ratios.tsv.gz")
            consolidate_summaries(results, out_dir / "phase4_summaries.tsv")


if __name__ == "__main__":
    main()
