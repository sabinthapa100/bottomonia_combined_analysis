#!/usr/bin/env python3
"""
export_hepdata.py - Master HEPData CSV export for all qtraj-nlo results.

Consolidates all analysis results into HEPData-compatible CSV format
with proper metadata for submission.

Usage:
    python scripts/analyze_outputs/export_hepdata.py
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

QTRAJ_ROOT = Path(__file__).resolve().parent.parent.parent
CAMPAIGNS_DIR = QTRAJ_ROOT / "campaigns"
ANALYSIS_DIR = QTRAJ_ROOT / "analysis"
HEPDATA_DIR = ANALYSIS_DIR / "hepdata_csv"

GEV_TO_FMC = 0.1973269804

STATES = ["1S", "2S", "3S", "1P", "2P"]
KAPPA_VALUES = [2, 3, 4, 5, 6]
B_VALUES = [
    0.00,
    2.32,
    4.25,
    6.01,
    7.78,
    9.21,
    10.45,
    11.55,
    11.60,
    12.56,
    12.60,
    13.49,
    13.50,
    14.38,
    14.40,
    15.66,
    15.70,
]

PDG_MASSES = {
    "1S": 9.460,
    "2S": 10.023,
    "3S": 10.355,
    "1P": 9.893,
    "2P": 10.260,
}


def collect_all_survival():
    """Collect survival data from all phases."""
    results = defaultdict(list)

    # Phase 2: Munich
    phase2_dir = CAMPAIGNS_DIR / "phase2_munich"
    if phase2_dir.exists():
        for kappa_dir in sorted(phase2_dir.glob("kappa_*")):
            kappa = int(kappa_dir.name.split("_")[1])
            for jump_dir in kappa_dir.glob("*Jumps*"):
                jm = "withJumps" if "with" in jump_dir.name else "noJumps"
                for b_dir in sorted(jump_dir.glob("b_*")):
                    b = float(b_dir.name.split("b_")[1])
                    for state_dir in sorted(b_dir.glob("state_*")):
                        state = state_dir.name.split("state_")[1]
                        for nq_dir in sorted(state_dir.glob("nq*")):
                            nq = int(nq_dir.name.split("nq")[1])
                            output_dirs = list(nq_dir.glob("output-*"))
                            ratios_file = nq_dir / "ratios.tsv"
                            if not ratios_file.exists() and output_dirs:
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
                                        results[
                                            (state, kappa, jm, b, nq, "munich")
                                        ].append(values[0])

    # Phase 3: KSU
    phase3_dir = CAMPAIGNS_DIR / "phase3_ksu"
    if phase3_dir.exists():
        for pot_dir in phase3_dir.glob("*"):
            if pot_dir.name.startswith("."):
                continue
            pot_name = f"ksu_{pot_dir.name}"
            for b_dir in sorted(pot_dir.glob("b_*")):
                b = float(b_dir.name.split("b_")[1])
                for state_dir in sorted(b_dir.glob("state_*")):
                    state = state_dir.name.split("state_")[1]
                    for nq_dir in sorted(state_dir.glob("nq*")):
                        nq = int(nq_dir.name.split("nq")[1])
                        output_dirs = list(nq_dir.glob("output-*"))
                        ratios_file = nq_dir / "ratios.tsv"
                        if not ratios_file.exists() and output_dirs:
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
                                    results[
                                        (state, 6, "noJumps", b, nq, pot_name)
                                    ].append(values[0])

    return results


def export_master_survival_table(survival_data, output_dir):
    """Export master survival probability table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "master_survival_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "state",
                "kappa",
                "jump_mode",
                "b_fm",
                "nqtraj",
                "potential",
                "survival_mean",
                "survival_stderr",
                "n_trajectories",
            ]
        )

        for (state, kappa, jm, b, nq, pot), values in sorted(survival_data.items()):
            if values:
                mean = np.mean(values)
                stderr = (
                    np.std(values) / math.sqrt(len(values)) if len(values) > 1 else 0
                )
                writer.writerow(
                    [
                        state,
                        kappa,
                        jm,
                        f"{b:.2f}",
                        nq,
                        pot,
                        f"{mean:.6f}",
                        f"{stderr:.6f}",
                        len(values),
                    ]
                )

    print(f"  Exported: {csv_path}")


def export_metadata(output_dir):
    """Export metadata file for HEPData submission."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "title": "Bottomonium Survival Probabilities from qtraj-nlo for PbPb at 5 TeV",
        "collaboration": "Theory",
        "abstract": (
            "Survival probabilities of bottomonium states (1S, 2S, 3S, 1P, 2P) "
            "computed using the qtraj-nlo quantum trajectory method for PbPb collisions "
            "at sqrt(s_NN) = 5 TeV. Results are provided for the Munich potential with "
            "and without quantum jumps, and for KSU potentials (isotropic and anisotropic). "
            "Kappa-hat values from 2 to 6 are scanned, with results for all impact parameters."
        ),
        "keywords": [
            "bottomonium",
            "quark-gluon plasma",
            "quantum trajectories",
            "Lindblad equation",
            "heavy ion collisions",
            "LHC",
        ],
        "physics_parameters": {
            "collision_system": "PbPb",
            "sqrt_s_NN_TeV": 5.0,
            "reduced_mass_GeV": 2.365,
            "alpha_coulomb": 0.6239853,
            "kappa_range": [2, 3, 4, 5, 6],
            "impact_parameters_fm": B_VALUES,
            "states": STATES,
            "potentials": ["Munich", "KSU_isotropic", "KSU_anisotropic"],
            "stepper": "Crank-Nicolson NLO E/T",
            "grid_points": 2048,
            "box_length_inv_GeV": 40,
            "time_step_inv_GeV": 0.001,
            "max_steps": 80000,
        },
        "pdg_reference_masses_GeV": PDG_MASSES,
        "code_version": "qtraj-nlo v2.1",
        "code_reference": "Strickland et al., Comput. Phys. Commun. (2023)",
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Exported: {metadata_path}")


def main():
    print("Exporting HEPData CSV files...")

    print("\nCollecting all survival data...")
    survival_data = collect_all_survival()
    print(f"  Found {len(survival_data)} unique configurations")

    print("\nExporting master survival table...")
    export_master_survival_table(survival_data, HEPDATA_DIR)

    print("\nExporting metadata...")
    export_metadata(HEPDATA_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
