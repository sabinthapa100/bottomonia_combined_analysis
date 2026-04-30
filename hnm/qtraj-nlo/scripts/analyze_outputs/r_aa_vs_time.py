#!/usr/bin/env python3
"""
R_AA vs time analysis for qtraj-nlo Phase-2 outputs.

Computes:
    R_AA(tau) = S(tau) / S(tau=0)

for 1S/2S/3S across all available impact parameters, with:
- solid line: noJumps
- dashed line: withJumps

Designed to run from any working directory.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

QTRAJ_ROOT = Path(__file__).resolve().parent.parent.parent
CAMPAIGNS_DIR = QTRAJ_ROOT / "outputs" / "campaigns"
OUTPUTS_DIR = QTRAJ_ROOT / "outputs" / "raa_vs_time"
GEV_TO_FMC = 0.1973269804

STATE_COL = {"1S": 4, "2S": 5, "3S": 6}
STATE_LABELS = {
    "1S": r"$\Upsilon(1S)$",
    "2S": r"$\Upsilon(2S)$",
    "3S": r"$\Upsilon(3S)$",
}
JUMP_STYLE = {"noJumps": "-", "withJumps": "--"}


def _parse_kappa_name(name: str):
    raw = name.split("_", 1)[1]
    return float(raw.replace("p", "."))


def _build_kappa_values(kappa_vals, kappa_range):
    if kappa_vals:
        return [float(k) for k in kappa_vals]
    if kappa_range:
        start, stop, step = kappa_range
        if step <= 0:
            raise ValueError("--kappa-range step must be > 0")
        n_steps = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 10) for i in range(max(0, n_steps))]
    return None


def parse_summary(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(list(map(float, line.split("\t"))))
    return rows


def extract_r_aa(rows, state):
    col = STATE_COL[state]
    tau = np.array([r[0] * GEV_TO_FMC for r in rows if len(r) > col], dtype=float)
    surv = np.array([r[col] for r in rows if len(r) > col], dtype=float)
    if len(surv) == 0:
        return None, None
    s0 = surv[0]
    r_aa = surv / s0 if s0 > 0 else surv
    # Physical bounds: survival-probability ratio should stay in [0, 1].
    r_aa = np.clip(r_aa, 0.0, 1.0)
    return tau, r_aa


def collect_data(campaigns_root, nq, states, kappas=None, b_values=None):
    """
    Returns mapping:
      (state, kappa, b, jump_mode) -> [(tau_arr, raa_arr), ...]
    """
    phase_dir = campaigns_root / "phase2_munich"
    if not phase_dir.exists():
        return {}

    data = defaultdict(list)
    for kappa_dir in sorted(phase_dir.glob("kappa_*")):
        kappa = _parse_kappa_name(kappa_dir.name)
        if kappas and not any(abs(kappa - k) < 1e-6 for k in kappas):
            continue

        for jump_dir in sorted(kappa_dir.glob("*Jumps*")):
            jm = "withJumps" if "with" in jump_dir.name else "noJumps"
            for b_dir in sorted(jump_dir.glob("b_*")):
                b = float(b_dir.name.split("b_")[1])
                if b_values and not any(abs(b - bb) < 5e-4 for bb in b_values):
                    continue
                for state_dir in sorted(b_dir.glob("state_*")):
                    state = state_dir.name.split("state_")[1]
                    if state not in states:
                        continue
                    nq_dir = state_dir / f"nq{nq}"
                    summary = nq_dir / "summary.tsv"
                    if not summary.exists():
                        continue
                    rows = parse_summary(summary)
                    tau, r_aa = extract_r_aa(rows, state)
                    if tau is None:
                        continue
                    data[(state, kappa, b, jm)].append((tau, r_aa))
    return data


def _mean_curve(curves):
    tau0 = curves[0][0]
    stack = np.vstack([c[1] for c in curves])
    return tau0, np.mean(stack, axis=0)


def _mean_and_std_curve(curves):
    tau0 = curves[0][0]
    stack = np.vstack([c[1] for c in curves])
    return tau0, np.mean(stack, axis=0), np.std(stack, axis=0)


def _smooth(y, window):
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def _setup_style():
    plt.rcParams.update(
        {
            "figure.figsize": (9, 6),
            "figure.dpi": 150,
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "axes.linewidth": 1.4,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Georgia"],
        }
    )


def plot_state_kappa(data, state, kappa, out_dir, smooth_window=1):
    keys = [k for k in data.keys() if k[0] == state and abs(k[1] - kappa) < 1e-6]
    if not keys:
        return False
    b_vals = sorted({k[2] for k in keys})
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(b_vals)))
    color_by_b = dict(zip(b_vals, colors))

    fig, ax = plt.subplots()
    for b in b_vals:
        for jm in ("noJumps", "withJumps"):
            curves = data.get((state, kappa, b, jm), [])
            if not curves:
                continue
            tau, mean_raa = _mean_curve(curves)
            mean_raa = _smooth(mean_raa, smooth_window)
            ax.plot(
                tau,
                mean_raa,
                color=color_by_b[b],
                linestyle=JUMP_STYLE[jm],
                alpha=0.95,
            )

    ax.set_xlabel(r"$\tau$ [fm/c]")
    ax.set_ylabel(r"$R_{AA}(\tau)$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(direction="in", top=True, right=True)

    b_handles = [Line2D([0], [0], color=color_by_b[b], lw=2.5) for b in b_vals]
    b_labels = [f"b={b:.3f} fm" for b in b_vals]
    leg1 = ax.legend(
        b_handles,
        b_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Impact parameter",
        framealpha=0.9,
    )
    ax.add_artist(leg1)
    style_handles = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-"),
        Line2D([0], [0], color="black", lw=2.5, linestyle="--"),
    ]
    ax.legend(style_handles, ["noJumps", "withJumps"], loc="lower left", framealpha=0.9)

    ax.text(
        0.02,
        0.98,
        f"PbPb 5.023 TeV, Munich potential\n{STATE_LABELS[state]}, kappa={kappa:g}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    out_base = out_dir / f"R_AA_vs_time_{state}_k{str(kappa).replace('.', 'p')}"
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def plot_state_b_band_over_kappa(
    data,
    state,
    b,
    out_dir,
    smooth_window=1,
    min_kappas_for_band=2,
    jump_mode_filter="both",
):
    """
    For fixed (state, b), plot noJumps/withJumps as:
      - central line: mean over selected kappas
      - shaded band: min/max over selected kappas
    """
    kappas = sorted({k[1] for k in data.keys() if k[0] == state and abs(k[2] - b) < 5e-4})
    if len(kappas) < 1:
        return False

    fig, ax = plt.subplots()
    mode_color = {"noJumps": "#1f77b4", "withJumps": "#d62728"}
    modes = (
        ("noJumps", "withJumps")
        if jump_mode_filter == "both"
        else (jump_mode_filter,)
    )

    for jm in modes:
        curves_by_k = []
        for kappa in kappas:
            curves = data.get((state, kappa, b, jm), [])
            if not curves:
                continue
            tau, mean_raa = _mean_curve(curves)
            curves_by_k.append((tau, mean_raa))

        if not curves_by_k:
            continue

        tau_ref = curves_by_k[0][0]
        stack = np.vstack([c[1] for c in curves_by_k if len(c[0]) == len(tau_ref)])
        if stack.size == 0:
            continue

        center = np.mean(stack, axis=0)
        center = _smooth(center, smooth_window)
        center = np.clip(center, 0.0, 1.0)
        color = mode_color[jm]

        if stack.shape[0] >= min_kappas_for_band:
            lo = _smooth(np.min(stack, axis=0), smooth_window)
            hi = _smooth(np.max(stack, axis=0), smooth_window)
            lo = np.clip(lo, 0.0, 1.0)
            hi = np.clip(hi, 0.0, 1.0)
            ax.fill_between(tau_ref, lo, hi, color=color, alpha=0.20)

        ax.plot(tau_ref, center, color=color, linestyle=JUMP_STYLE[jm], label=jm)

    ax.set_xlabel(r"$\tau$ [fm/c]")
    ax.set_ylabel(r"$R_{AA}(\tau)$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="lower left", framealpha=0.9)

    jump_txt = {
        "both": "noJumps (solid) + withJumps (dashed)",
        "noJumps": "noJumps only",
        "withJumps": "withJumps only",
    }[jump_mode_filter]
    ax.text(
        0.02,
        0.98,
        (
            f"PbPb 5.023 TeV, Munich potential\n"
            f"{STATE_LABELS[state]}, b={b:.3f} fm\n"
            f"Band = kappa envelope ({len(kappas)} kappas), {jump_txt}"
        ),
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    out_base = out_dir / f"R_AA_vs_time_band_{state}_b{b:.3f}".replace(".", "p")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def export_csv(data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for (state, kappa, b, jm), curves in sorted(data.items()):
        tau, mean_raa = _mean_curve(curves)
        stem = f"R_AA_vs_time_{state}_k{str(kappa).replace('.', 'p')}_b{b:.3f}_{jm}"
        with open(out_dir / f"{stem}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tau_fm_c", "R_AA"])
            for t, r in zip(tau, mean_raa):
                w.writerow([f"{t:.6f}", f"{r:.8f}"])


def main():
    parser = argparse.ArgumentParser(description="Plot R_AA(tau) for qtraj-nlo outputs")
    parser.add_argument(
        "--campaigns-root",
        type=str,
        default=str(CAMPAIGNS_DIR),
        help="Campaigns root (default: qtraj-nlo/outputs/campaigns)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUTS_DIR),
        help="Output root for plots/csv",
    )
    parser.add_argument("--nq", type=int, default=100, help="nq directory to read")
    parser.add_argument("--state", nargs="+", default=["1S", "2S", "3S"])
    parser.add_argument("--kappa", type=float, nargs="+", default=None)
    parser.add_argument(
        "--kappa-range",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        default=None,
        help="Select kappas by range, e.g. --kappa-range 1.8 3.4 0.4",
    )
    parser.add_argument("--b", type=float, nargs="+", default=None)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Moving-average window for prettier lines (odd like 5/7/9 recommended)",
    )
    parser.add_argument(
        "--plot-kappa-band",
        action="store_true",
        help="Also create fixed-b band plots where shaded region is kappa min/max",
    )
    parser.add_argument(
        "--jump-mode",
        choices=["noJumps", "withJumps", "both"],
        default="both",
        help="Which jump mode(s) to show in kappa-band plots",
    )
    parser.add_argument(
        "--band-only",
        action="store_true",
        help="Only produce band plots (skip per-kappa line plots)",
    )
    args = parser.parse_args()

    _setup_style()
    campaigns_root = Path(args.campaigns_root).resolve()
    output_root = Path(args.output_root).resolve()
    plots_dir = output_root / "plots"
    data_dir = output_root / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_kappas = _build_kappa_values(args.kappa, args.kappa_range)
    data = collect_data(
        campaigns_root=campaigns_root,
        nq=args.nq,
        states=args.state,
        kappas=selected_kappas,
        b_values=args.b,
    )
    if not data:
        print("No matching data found.")
        return

    kappas = sorted({k[1] for k in data.keys()})
    generated = 0
    if not args.band_only:
        for state in args.state:
            for kappa in kappas:
                if plot_state_kappa(
                    data,
                    state,
                    kappa,
                    plots_dir,
                    smooth_window=max(1, args.smooth_window),
                ):
                    generated += 1

    band_generated = 0
    if args.plot_kappa_band and len(kappas) >= 2:
        b_vals = sorted({k[2] for k in data.keys()})
        for state in args.state:
            for b in b_vals:
                if plot_state_b_band_over_kappa(
                    data,
                    state,
                    b,
                    plots_dir,
                    smooth_window=max(1, args.smooth_window),
                    jump_mode_filter=args.jump_mode,
                ):
                    band_generated += 1

    export_csv(data, data_dir)
    if not args.band_only:
        print(f"Generated {generated} plot(s).")
    if args.plot_kappa_band:
        print(f"Generated {band_generated} kappa-band plot(s).")
    print(f"Plots: {plots_dir}")
    print(f"CSV:   {data_dir}")


if __name__ == "__main__":
    main()
