#!/usr/bin/env python3
"""
Investigate OO noReg vs wReg: same kinematics as run_oo_5360_production (mid |y|<=2.4 for R_AA vs pT;
pT<=30 for the R_AA vs y sample).

Writes JSON reporting:
  - N_trajectories per file
  - Mean primordial Υ(1S) survival (surv6[0]) in |y|<=2.4 (full sample, not binned)
  - Per pT-bin: mean surv6[0] mid-rapidity (before feed-down) for noReg vs wReg
  - Per pT-bin: inclusive R_AA(1S) after feed-down (same pipeline as production)
  - Sign of (wReg - noReg) per bin — if negative for survival before feed-down, ordering is in raw MCWF output, not feed-down.

Requires bundled datafiles under inputs/qtraj_inputs/OxygenOxygen5360/ (same paths as run_oo_5360_production).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qtraj_analysis.binning import compute_raa_vs_pt  # noqa: E402
from qtraj_analysis.feeddown import (  # noqa: E402
    apply_feeddown_to_raa6,
    build_feeddown_matrix,
    solve_primordial_sigmas,
)
from qtraj_analysis.io import load_qtraj_table, parse_records  # noqa: E402
from qtraj_analysis.kinematics_presets import (  # noqa: E402
    OO_PT_EDGES as PT_EDGES,
    OO_PT_RAPIDITY_WINDOWS,
    SIGMAS_EXP_OO_5360,
)
from qtraj_analysis.matching import build_observables  # noqa: E402


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "inputs").exists() and (parent / "hnm").exists():
            return parent
    raise RuntimeError("Could not find repo root")


REPO_ROOT = _repo_root()

MODE_CONFIGS = {
    "noReg": "inputs/qtraj_inputs/OxygenOxygen5360/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile.gz",
    "wReg": "inputs/qtraj_inputs/OxygenOxygen5360/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile.gz",
}


def _load_obs(rel: str, logger: logging.Logger):
    path = (REPO_ROOT / rel).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    table = load_qtraj_table(str(path), logger)
    records = parse_records(table, logger)
    return build_observables(records, logger), str(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", "-o", type=str, default="", help="Write JSON to this path (default: stdout)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("diagnose_oo")

    mid = next(w for w in OO_PT_RAPIDITY_WINDOWS if w[0] == "midrapidity")
    y_window = mid[1]

    out: dict = {"mid_rapidity_window": list(y_window), "pt_edges": PT_EDGES.tolist()}
    modes_data = {}

    for mode, rel in MODE_CONFIGS.items():
        try:
            obs, path = _load_obs(rel, logger)
        except FileNotFoundError as e:
            modes_data[mode] = {
                "error": "file_not_found",
                "path": getattr(e, "filename", None) or str(REPO_ROOT / rel),
            }
            continue

        obs_mid = [o for o in obs if y_window[0] <= o.y <= y_window[1]]
        if obs_mid:
            s1 = np.array([o.surv6[0] for o in obs_mid], dtype=np.float64)
            surv1s_global = float(np.mean(s1))
            surv1s_sem = float(np.std(s1, ddof=1) / np.sqrt(len(s1))) if len(s1) > 1 else 0.0
        else:
            surv1s_global = float("nan")
            surv1s_sem = float("nan")

        pt_centers, raa6_pt, sem6_pt = compute_raa_vs_pt(
            obs, PT_EDGES, y_window=y_window, logger=logger
        )
        feeddown = build_feeddown_matrix()
        sigp = solve_primordial_sigmas(feeddown, SIGMAS_EXP_OO_5360)
        raa9_pt = np.full((raa6_pt.shape[0], 9), np.nan, dtype=np.float64)
        for i in range(raa6_pt.shape[0]):
            if np.isnan(raa6_pt[i, 0]):
                continue
            r9, _ = apply_feeddown_to_raa6(raa6_pt[i], sem6_pt[i], feeddown, sigp)
            raa9_pt[i, :] = r9

        modes_data[mode] = {
            "datafile": path,
            "n_trajectories_all": len(obs),
            "n_trajectories_midrapidity": len(obs_mid),
            "mean_primordial_survival_1S_midrapidity": surv1s_global,
            "sem_primordial_survival_1S_midrapidity": surv1s_sem,
            "pt_centers": pt_centers.tolist(),
            "mean_surv6_1S_per_pt_bin": raa6_pt[:, 0].tolist(),
            "inclusive_raa9_1S_per_pt_bin": raa9_pt[:, 0].tolist(),
        }

    out["modes"] = modes_data

    if "noReg" in modes_data and "wReg" in modes_data:
        a, b = modes_data["noReg"], modes_data["wReg"]
        if "mean_surv6_1S_per_pt_bin" in a and "mean_surv6_1S_per_pt_bin" in b:
            sa = np.asarray(a["mean_surv6_1S_per_pt_bin"], float)
            sb = np.asarray(b["mean_surv6_1S_per_pt_bin"], float)
            out["delta_primordial_surv1S_wReg_minus_noReg_per_bin"] = (sb - sa).tolist()
        if "inclusive_raa9_1S_per_pt_bin" in a and "inclusive_raa9_1S_per_pt_bin" in b:
            ia = np.asarray(a["inclusive_raa9_1S_per_pt_bin"], float)
            ib = np.asarray(b["inclusive_raa9_1S_per_pt_bin"], float)
            out["delta_inclusive_raa1S_wReg_minus_noReg_per_bin"] = (ib - ia).tolist()

    if "noReg" in modes_data and "wReg" in modes_data:
        na = modes_data.get("noReg", {})
        wa = modes_data.get("wReg", {})
        if "mean_primordial_survival_1S_midrapidity" in na and "mean_primordial_survival_1S_midrapidity" in wa:
            if np.isfinite(na["mean_primordial_survival_1S_midrapidity"]) and np.isfinite(
                wa["mean_primordial_survival_1S_midrapidity"]
            ):
                out["delta_global_primordial_surv1S_wReg_minus_noReg"] = (
                    wa["mean_primordial_survival_1S_midrapidity"]
                    - na["mean_primordial_survival_1S_midrapidity"]
                )

    out["interpretation"] = (
        "If delta_primordial_surv1S_wReg_minus_noReg is negative at most pT bins, "
        "wReg < noReg is already in the raw survival (MCWF output), not introduced by feed-down. "
        "If primordial deltas are positive but inclusive R_AA flips, inspect feed-down / sigmas_exp."
    )

    text = json.dumps(out, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        logger.info("Wrote %s", args.output)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
