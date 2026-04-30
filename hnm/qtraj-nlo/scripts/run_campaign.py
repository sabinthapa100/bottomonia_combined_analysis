#!/usr/bin/env python3
"""
run_campaign.py - Main campaign runner for qtraj-nlo PbPb 5 TeV analysis.

Generates parameter sets, runs qtraj for each configuration, and organizes outputs.

Usage:
    python scripts/run_campaign.py --phase 2 --dry-run    # preview only
    python scripts/run_campaign.py --phase 2              # execute
    python scripts/run_campaign.py --phase 2 --kappa 4 --b 0.00 --state 1S --nq 100 --jumps  # single config
    python scripts/run_campaign.py --phase 1              # vacuum eigenstates
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# ============================================================
# Constants
# ============================================================
QTRAJ_ROOT = Path(__file__).resolve().parent.parent
QTRAJ_BIN = QTRAJ_ROOT / "qtraj"
DEFAULT_TEMP_FILE = (
    QTRAJ_ROOT / "input" / "temperature" / "PbPb_5_TeV" / "T_center_vs_tau.csv"
)
AUAU_TEMP_FILE = QTRAJ_ROOT / "input" / "temperature" / "AuAu_200_GeV" / "T_center_vs_tau.csv"
BASIS_FILE = QTRAJ_ROOT / "input" / "basisfunctions_ksu_4096.tsv"
OUTPUTS_ROOT = QTRAJ_ROOT / "outputs"
CAMPAIGNS_DIR = OUTPUTS_ROOT / "campaigns"

# Physics constants
GEV_TO_FMC = 0.1973269804  # 1/GeV to fm/c conversion
T0 = 0.550  # GeV
TF = 0.17  # GeV
TMED = 3.0406390839551767  # 0.6 fm/c in 1/GeV
M_BOT = 2.365  # reduced mass GeV
ALPHA = 0.6239853  # coulomb coupling

# States definition
STATES = {
    "1S": {"initN": 1, "initL": 0, "initC": 0},
    "2S": {"initN": 2, "initL": 0, "initC": 0},
    "3S": {"initN": 3, "initL": 0, "initC": 0},
    "1P": {"initN": 2, "initL": 1, "initC": 0},
    "2P": {"initN": 3, "initL": 1, "initC": 0},
    "OctS": {"initN": 1, "initL": 0, "initC": 1},
    "OctP": {"initN": 2, "initL": 1, "initC": 1},
}

# Kappa values for LHC
KAPPA_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0]

# NQTRAJ values for convergence
NQTRAJ_VALUES = [20, 40, 100]


def _format_kappa_for_path(kappa):
    """Format float kappa for stable directory names."""
    if abs(kappa - round(kappa)) < 1e-9:
        return f"{int(round(kappa))}"
    return f"{kappa:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_temperature_file(temp_file):
    """Parse T_center_vs_tau.csv and extract unique b values with their tau-T data."""
    b_data = {}
    with open(temp_file, "r") as f:
        reader = csv.DictReader(f)
        required = {"b_fm", "tau_fm"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Temperature file {temp_file} must include columns: b_fm,tau_fm"
            )
        for row in reader:
            b = float(row["b_fm"])
            tau = float(row["tau_fm"])
            t_col = "T_center_MeV"
            if t_col not in row or row[t_col] in (None, ""):
                if "T_avg_MeV" in row and row["T_avg_MeV"] not in (None, ""):
                    t_col = "T_avg_MeV"
                elif "T_MeV" in row and row["T_MeV"] not in (None, ""):
                    t_col = "T_MeV"
                else:
                    raise ValueError(
                        f"Could not find a usable temperature column in {temp_file}. "
                        "Expected one of: T_center_MeV, T_avg_MeV, T_MeV"
                    )
            T = float(row[t_col])
            if b not in b_data:
                b_data[b] = []
            b_data[b].append((tau, T))
    return b_data


def infer_collision_system(temp_file: Path) -> str:
    p = str(temp_file)
    if "AuAu_200_GeV" in p:
        return "AuAu_200_GeV"
    if "PbPb_5_TeV" in p:
        return "PbPb_5_TeV"
    return "custom"


def create_trajectory_file(b_value, output_path, temp_file):
    """Create a trajectory file for a specific impact parameter.

    qtraj-nlo temperatureEvolution=2 format:
    Line 1: number of metadata lines
    Next N lines: metadata
    Next line: number of data points
    Data lines: tau_fm T_MeV ax ay az Lam
    """
    b_data = parse_temperature_file(temp_file)
    if b_value not in b_data:
        nearest = min(b_data.keys(), key=lambda x: abs(x - b_value))
        if abs(nearest - b_value) > 0.05:
            raise ValueError(
                f"b={b_value} not found in temperature file {temp_file}. "
                f"Nearest available is b={nearest:.5f}"
            )
        b_value = nearest

    data_points = b_data[b_value]
    n_meta = 2
    n_data = len(data_points)

    with open(output_path, "w") as f:
        f.write(f"{n_meta}\n")
        f.write(f"# PbPb 5 TeV, b={b_value:.5f} fm\n")
        f.write(f"# tau(fm/c) T(MeV) ax ay az Lam\n")
        f.write(f"{n_data}\n")
        for tau, T in data_points:
            # For central temperature profiles, set anisotropy to zero
            f.write(f"{tau:.4f} {T:.2f} 0.0 0.0 0.0 1.0\n")


def build_params(
    state_name,
    kappa,
    do_jumps,
    nqtraj,
    potential,
    init_type=1,
    proj_type=1,
    temperature_evolution=2,
    temp_file=None,
    save_wavefunctions=0,
    snap_freq=500,
    dirname_with_seed=2,
    outfile_id="",
    max_steps=80000,
    num_grid=2048,
):
    """Build command line arguments for qtraj."""
    state = STATES[state_name]
    args = [
        str(QTRAJ_BIN),
        f"-initN",
        str(state["initN"]),
        f"-initL",
        str(state["initL"]),
        f"-initC",
        str(state["initC"]),
        f"-initType",
        str(init_type),
        f"-projType",
        str(proj_type),
        f"-potential",
        str(potential),
        f"-kappa",
        str(kappa),
        f"-doJumps",
        str(do_jumps),
        f"-nTrajectories",
        str(nqtraj),
        f"-stepper",
        "2",
        f"-num",
        str(num_grid),
        f"-L",
        "40",
        f"-dt",
        "0.001",
        f"-maxSteps",
        str(max_steps),
        f"-T0",
        str(T0),
        f"-Tf",
        str(TF),
        f"-tmed",
        str(TMED),
        f"-m",
        str(M_BOT),
        f"-alpha",
        str(ALPHA),
        f"-mdfac",
        "1",
        f"-gam",
        "0",
        f"-basisFunctionsFile",
        str(BASIS_FILE),
        f"-saveWavefunctions",
        str(save_wavefunctions),
        f"-snapFreq",
        str(snap_freq),
        f"-snapPts",
        "1024",
        f"-dirnameWithSeed",
        str(dirname_with_seed),
        f"-outputSummaryFile",
        "1",
    ]

    if temperature_evolution == 2 and temp_file:
        args.extend(["-temperatureEvolution", "2", "-temperatureFile", str(temp_file)])
    elif temperature_evolution == 3:
        args.extend(["-temperatureEvolution", "3", "-T0", "0.001", "-Tf", "0.0001"])

    if outfile_id:
        args.extend(["-outfileID", outfile_id])

    return args


def run_qtraj(args, output_dir):
    """Run qtraj from QTRAJ_ROOT, then move output to target directory."""
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        args,
        cwd=str(QTRAJ_ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    # Move output directory to target
    # qtraj creates output-<outfileID> in CWD (QTRAJ_ROOT)
    out_name = f"output-{args[-1]}" if args[-1] != "0" else "output"
    src = QTRAJ_ROOT / out_name
    if src.exists():
        if src.is_dir():
            # Move contents to output_dir
            for item in src.iterdir():
                dest = Path(output_dir) / item.name
                if dest.exists():
                    dest.unlink()
                item.rename(dest)
            src.rmdir()
        else:
            src.rename(Path(output_dir) / src.name)
    return result


def run_phase1_vacuum(dry_run=False):
    """Phase 1: Vacuum eigenstate validation.

    Uses initType=1 (Gaussian) at T~0 to get vacuum wavefunction evolution.
    Also runs initType=200 (pre-computed basis) for comparison.
    """
    print("=" * 70)
    print("PHASE 1: Vacuum Eigenstate Validation")
    print("=" * 70)

    for state_name in ["1S", "2S", "3S", "1P", "2P"]:
        # Approach A: Gaussian at T~0 (fast, gives time evolution)
        outfile_id = f"vacuum_{state_name}_gauss"
        out_dir = CAMPAIGNS_DIR / "phase1_vacuum" / outfile_id

        args = build_params(
            state_name=state_name,
            kappa=0,
            do_jumps=0,
            nqtraj=1,
            potential=0,
            init_type=1,
            proj_type=1,
            temperature_evolution=3,  # constant T~0 = vacuum
            temp_file=None,
            save_wavefunctions=1,
            snap_freq=200,
            dirname_with_seed=2,
            outfile_id=outfile_id,
            max_steps=5000,
            num_grid=512,
        )

        if dry_run:
            print(
                f"  [DRY-A] {state_name} Gaussian: {' '.join(args[:4])} ... -> {out_dir}"
            )
        else:
            print(
                f"  [A] Running {state_name} vacuum (Gaussian, 512 grid, 5000 steps)..."
            )
            result = run_qtraj(args, str(out_dir))
            if result.returncode != 0:
                print(f"    ERROR: {result.stderr[:200]}")
            else:
                print(f"    OK -> {out_dir}")

        # Approach B: Load from pre-computed basis (initType=200)
        outfile_id2 = f"vacuum_{state_name}_basis"
        out_dir2 = CAMPAIGNS_DIR / "phase1_vacuum" / outfile_id2

        args2 = build_params(
            state_name=state_name,
            kappa=0,
            do_jumps=0,
            nqtraj=1,
            potential=0,
            init_type=200,
            proj_type=2,
            temperature_evolution=3,
            temp_file=None,
            save_wavefunctions=1,
            snap_freq=200,
            dirname_with_seed=2,
            outfile_id=outfile_id2,
            max_steps=5000,
            num_grid=512,
        )

        if dry_run:
            print(
                f"  [DRY-B] {state_name} Basis: {' '.join(args2[:4])} ... -> {out_dir2}"
            )
        else:
            print(
                f"  [B] Running {state_name} vacuum (pre-computed basis, 512 grid, 5000 steps)..."
            )
            result = run_qtraj(args2, str(out_dir2))
            if result.returncode != 0:
                print(f"    ERROR: {result.stderr[:200]}")
            else:
                print(f"    OK -> {out_dir2}")


def run_phase2_munich(
    dry_run=False,
    kappa_filter=None,
    kappa_range=None,
    b_filter=None,
    state_filter=None,
    nq_filter=None,
    jumps_filter=None,
    campaigns_root=CAMPAIGNS_DIR,
    temp_file=DEFAULT_TEMP_FILE,
    init_type=1,
    proj_type=1,
):
    """Phase 2: Munich potential main production."""
    print("=" * 70)
    print("PHASE 2: Munich Potential Main Production")
    print("=" * 70)

    if kappa_filter:
        kappas = [float(k) for k in kappa_filter]
    elif kappa_range:
        start, stop, step = kappa_range
        if step <= 0:
            raise ValueError("--kappa-range step must be > 0")
        n_steps = int(round((stop - start) / step)) + 1
        kappas = [round(start + i * step, 10) for i in range(max(0, n_steps))]
    else:
        kappas = list(KAPPA_VALUES)
    states = state_filter if state_filter else ["1S", "2S", "3S"]
    if b_filter is not None:
        b_vals = b_filter
    else:
        b_vals = sorted(parse_temperature_file(temp_file).keys())
    nq_vals = nq_filter if nq_filter else NQTRAJ_VALUES
    jumps_list = [jumps_filter] if jumps_filter is not None else [0, 1]

    total = len(states) * len(kappas) * len(b_vals) * len(nq_vals) * len(jumps_list)
    print(f"  Total configurations: {total}")

    count = 0
    for state_name in states:
        for kappa in kappas:
            for b_val in b_vals:
                for nq in nq_vals:
                    for do_jumps in jumps_list:
                        count += 1
                        b_str = f"b{b_val:.2f}".replace(".", "p")
                        jump_str = "withJumps" if do_jumps else "noJumps"
                        outfile_id = (
                            f"munich_k{kappa}_{state_name}_{jump_str}_{b_str}_nq{nq}"
                        )

                        # Create trajectory file for this b
                        kappa_dir_name = f"kappa_{_format_kappa_for_path(kappa)}"
                        traj_dir = campaigns_root / "phase2_munich" / ".trajectories"
                        os.makedirs(traj_dir, exist_ok=True)
                        traj_file = traj_dir / f"traj_b{b_val:.5f}.txt"

                        if not traj_file.exists():
                            create_trajectory_file(b_val, traj_file, temp_file)

                        out_dir = (
                            campaigns_root
                            / "phase2_munich"
                            / kappa_dir_name
                            / jump_str
                            / f"b_{b_val:.5f}"
                            / f"state_{state_name}"
                            / f"nq{nq}"
                        )

                        args = build_params(
                            state_name=state_name,
                            kappa=kappa,
                            do_jumps=do_jumps,
                            nqtraj=nq,
                            potential=0,  # Munich
                            init_type=init_type,
                            proj_type=proj_type,
                            temperature_evolution=2,
                            temp_file=traj_file,
                            save_wavefunctions=0,
                            dirname_with_seed=2,
                            outfile_id=outfile_id,
                        )

                        if dry_run:
                            print(
                                f"  [{count}/{total}] {state_name} k={kappa} b={b_val} nq={nq} jumps={do_jumps}"
                            )
                        else:
                            print(
                                f"  [{count}/{total}] {state_name} k={kappa} b={b_val} nq={nq} jumps={do_jumps}...",
                                end=" ",
                            )
                            result = run_qtraj(args, str(out_dir))
                            if result.returncode != 0:
                                print(f"ERROR")
                            else:
                                print(f"OK")


def run_phase3_ksu(dry_run=False, b_filter=None, state_filter=None, nq_filter=None):
    """Phase 3: KSU potential production (isotropic + anisotropic)."""
    print("=" * 70)
    print("PHASE 3: KSU Potential Production")
    print("=" * 70)

    states = state_filter if state_filter else ["1S", "2S", "3S", "1P", "2P"]
    b_vals = b_filter if b_filter is not None else B_VALUES
    nq_vals = nq_filter if nq_filter else NQTRAJ_VALUES
    potentials = [(1, "isotropic"), (2, "anisotropic")]

    total = len(states) * len(b_vals) * len(nq_vals) * len(potentials)
    print(f"  Total configurations: {total}")

    count = 0
    for pot_id, pot_name in potentials:
        for state_name in states:
            for b_val in b_vals:
                for nq in nq_vals:
                    count += 1
                    b_str = f"b{b_val:.2f}".replace(".", "p")
                    outfile_id = f"ksu_{pot_name}_{state_name}_{b_str}_nq{nq}"

                    traj_dir = CAMPAIGNS_DIR / "phase3_ksu" / ".trajectories"
                    os.makedirs(traj_dir, exist_ok=True)
                    traj_file = traj_dir / f"traj_b{b_val:.2f}.txt"

                    if not traj_file.exists():
                        create_trajectory_file(b_val, traj_file)

                    out_dir = (
                        CAMPAIGNS_DIR
                        / "phase3_ksu"
                        / pot_name
                        / f"b_{b_val:.2f}"
                        / f"state_{state_name}"
                        / f"nq{nq}"
                    )

                    args = build_params(
                        state_name=state_name,
                        kappa=6,
                        do_jumps=0,  # KSU doesn't support jumps
                        nqtraj=nq,
                        potential=pot_id,
                        init_type=1,
                        proj_type=1,
                        temperature_evolution=2,
                        temp_file=traj_file,
                        save_wavefunctions=0,
                        dirname_with_seed=2,
                        outfile_id=outfile_id,
                    )

                    if dry_run:
                        print(
                            f"  [{count}/{total}] {pot_name} {state_name} b={b_val} nq={nq}"
                        )
                    else:
                        print(
                            f"  [{count}/{total}] {pot_name} {state_name} b={b_val} nq={nq}...",
                            end=" ",
                        )
                        result = run_qtraj(args, str(out_dir))
                        if result.returncode != 0:
                            print(f"ERROR")
                        else:
                            print(f"OK")


def run_phase4_wavefunctions(dry_run=False):
    """Phase 4: Wavefunction evolution snapshots."""
    print("=" * 70)
    print("PHASE 4: Wavefunction Evolution Snapshots")
    print("=" * 70)

    # Representative configs for wavefunction visualization
    configs = [
        # (state, kappa, do_jumps, potential, pot_name)
        ("1S", 4, 0, 0, "munich"),
        ("1S", 4, 1, 0, "munich"),
        ("2S", 4, 0, 0, "munich"),
        ("2S", 4, 1, 0, "munich"),
        ("1P", 4, 0, 0, "munich"),
        ("1P", 4, 1, 0, "munich"),
        ("1S", 6, 0, 1, "ksu_iso"),
        ("1S", 6, 0, 2, "ksu_aniso"),
        ("OctS", 4, 0, 0, "munich"),
        ("OctS", 4, 1, 0, "munich"),
    ]

    # Only central b=0 for wavefunction snapshots
    b_val = 0.00
    traj_dir = CAMPAIGNS_DIR / "phase4_wavefunctions" / ".trajectories"
    os.makedirs(traj_dir, exist_ok=True)
    traj_file = traj_dir / "traj_b0.00.txt"
    if not traj_file.exists():
        create_trajectory_file(b_val, traj_file)

    for i, (state_name, kappa, do_jumps, potential, pot_name) in enumerate(configs):
        jump_str = "withJumps" if do_jumps else "noJumps"
        outfile_id = f"wf_{pot_name}_{state_name}_k{kappa}_{jump_str}"
        out_dir = CAMPAIGNS_DIR / "phase4_wavefunctions" / outfile_id

        args = build_params(
            state_name=state_name,
            kappa=kappa,
            do_jumps=do_jumps,
            nqtraj=100,
            potential=potential,
            init_type=1,
            proj_type=1,
            temperature_evolution=2,
            temp_file=traj_file,
            save_wavefunctions=1,
            snap_freq=200,
            snap_pts=1024,
            dirname_with_seed=2,
            outfile_id=outfile_id,
        )

        if dry_run:
            print(
                f"  [{i + 1}/{len(configs)}] {pot_name} {state_name} k={kappa} jumps={do_jumps}"
            )
        else:
            print(
                f"  [{i + 1}/{len(configs)}] {pot_name} {state_name} k={kappa} jumps={do_jumps}...",
                end=" ",
            )
            result = run_qtraj(args, str(out_dir))
            if result.returncode != 0:
                print(f"ERROR")
            else:
                print(f"OK")


def run_phase5_nqtraj_convergence(dry_run=False):
    """Phase 5: NQTRAJ convergence study."""
    print("=" * 70)
    print("PHASE 5: NQTRAJ Convergence Study")
    print("=" * 70)

    # Use Munich potential, central b=0, kappa=4, with jumps
    b_val = 0.00
    kappa = 4

    traj_dir = CAMPAIGNS_DIR / "phase5_nqtraj_convergence" / ".trajectories"
    os.makedirs(traj_dir, exist_ok=True)
    traj_file = traj_dir / "traj_b0.00.txt"
    if not traj_file.exists():
        create_trajectory_file(b_val, traj_file)

    nq_values = [10, 20, 40, 100, 200, 500]

    for state_name in ["1S", "2S", "1P"]:
        for nq in nq_values:
            outfile_id = f"conv_{state_name}_k{kappa}_nq{nq}"
            out_dir = CAMPAIGNS_DIR / "phase5_nqtraj_convergence" / outfile_id

            args = build_params(
                state_name=state_name,
                kappa=kappa,
                do_jumps=1,
                nqtraj=nq,
                potential=0,
                init_type=1,
                proj_type=1,
                temperature_evolution=2,
                temp_file=traj_file,
                save_wavefunctions=0,
                dirname_with_seed=2,
                outfile_id=outfile_id,
            )

            if dry_run:
                print(f"  {state_name} nq={nq}")
            else:
                print(f"  {state_name} nq={nq}...", end=" ")
                result = run_qtraj(args, str(out_dir))
                if result.returncode != 0:
                    print(f"ERROR")
                else:
                    print(f"OK")


def main():
    parser = argparse.ArgumentParser(
        description="qtraj-nlo campaign runner (outputs managed under qtraj-nlo/outputs)"
    )
    parser.add_argument(
        "--phase", type=int, default=2, choices=[1, 2, 3, 4, 5], help="Phase to run"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    parser.add_argument(
        "--kappa", type=float, nargs="+", help="Filter kappa values (phase 2 only)"
    )
    parser.add_argument(
        "--kappa-range",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Generate kappa grid, e.g. --kappa-range 1.8 3.4 0.2",
    )
    parser.add_argument("--b", type=float, nargs="+", help="Filter impact parameters")
    parser.add_argument(
        "--state",
        type=str,
        nargs="+",
        choices=list(STATES.keys()),
        help="Filter states",
    )
    parser.add_argument("--nq", type=int, nargs="+", help="Filter NQTRAJ values")
    parser.add_argument(
        "--jumps", type=int, choices=[0, 1], help="Filter jumps (0=no, 1=yes)"
    )
    parser.add_argument(
        "--init-type",
        type=int,
        default=1,
        help="qtraj initType (phase 2), e.g. 1 for Gaussian or 200 for basis",
    )
    parser.add_argument(
        "--proj-type",
        type=int,
        default=1,
        help="qtraj projType (phase 2), typically paired with initType",
    )
    parser.add_argument(
        "--temperature-file",
        type=str,
        default=None,
        help="Temperature CSV file (expects b_fm,tau_fm and T_* column)",
    )
    parser.add_argument(
        "--system",
        choices=["pbpb5", "auau200"],
        default="pbpb5",
        help="Collision system used to choose default temperature file",
    )
    parser.add_argument(
        "--campaigns-root",
        type=str,
        default=str(CAMPAIGNS_DIR),
        help="Root directory for campaign outputs (default: qtraj-nlo/outputs/campaigns)",
    )

    args = parser.parse_args()

    # Verify qtraj binary exists
    if not QTRAJ_BIN.exists():
        print(f"ERROR: qtraj binary not found at {QTRAJ_BIN}")
        print("Run 'make' in the qtraj-nlo directory first.")
        sys.exit(1)

    if args.phase == 1:
        run_phase1_vacuum(dry_run=args.dry_run)
    elif args.phase == 2:
        campaigns_root = Path(args.campaigns_root).resolve()
        if args.temperature_file:
            temp_file = Path(args.temperature_file).resolve()
        else:
            temp_file = DEFAULT_TEMP_FILE if args.system == "pbpb5" else AUAU_TEMP_FILE
            temp_file = temp_file.resolve()
        if not temp_file.exists():
            print(f"ERROR: temperature file not found: {temp_file}")
            sys.exit(1)
        if not args.dry_run:
            system_name = infer_collision_system(temp_file)
            print(f"Using system: {system_name}")
            print(f"Temperature file: {temp_file}")
            print(f"Campaign root: {campaigns_root}")
        run_phase2_munich(
            dry_run=args.dry_run,
            kappa_filter=args.kappa,
            kappa_range=args.kappa_range,
            b_filter=args.b,
            state_filter=args.state,
            nq_filter=args.nq,
            jumps_filter=args.jumps,
            campaigns_root=campaigns_root,
            temp_file=temp_file,
            init_type=args.init_type,
            proj_type=args.proj_type,
        )
    elif args.phase == 3:
        run_phase3_ksu(
            dry_run=args.dry_run,
            b_filter=args.b,
            state_filter=args.state,
            nq_filter=args.nq,
        )
    elif args.phase == 4:
        run_phase4_wavefunctions(dry_run=args.dry_run)
    elif args.phase == 5:
        run_phase5_nqtraj_convergence(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
