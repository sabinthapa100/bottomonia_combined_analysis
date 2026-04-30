#!/usr/bin/env python3
"""
Run R_AA vs pT (per rapidity window) and R_AA vs y for each discrete impact parameter
using `run_b_scan`. Uses the same bin edges and pp baseline as `kinematics_presets`
(aligned with `run_oo_5360_production`).

Outputs under --output-dir:
  manifest.json
  spectra_raavspt__<window_key>.npz   (bvals, pt_centers, raa9, sem9)
  spectra_raavsy.npz                 (bvals, y_centers, raa9, sem9)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qtraj_analysis.kinematics_presets import (  # noqa: E402
    OO_PT_EDGES,
    OO_PT_MAX_FOR_Y,
    OO_PT_RAPIDITY_WINDOWS,
    OO_Y_EDGES,
    PT_EDGES,
    PT_MAX_FOR_Y,
    PT_RAPIDITY_WINDOWS,
    Y_EDGES,
    get_system_kinematics,
)
from qtraj_analysis.single_b_analysis import run_b_scan  # noqa: E402


def _select_binning(system_key: str) -> tuple[np.ndarray, np.ndarray, float, tuple]:
    if system_key == "oo5360":
        return OO_PT_EDGES, OO_Y_EDGES, OO_PT_MAX_FOR_Y, OO_PT_RAPIDITY_WINDOWS
    return PT_EDGES, Y_EDGES, PT_MAX_FOR_Y, PT_RAPIDITY_WINDOWS


def _nearest_b_index(bvals: np.ndarray, target_b: float, tolerance: float) -> tuple[int, float]:
    diffs = np.abs(np.asarray(bvals, dtype=np.float64) - float(target_b))
    idx = int(np.argmin(diffs))
    chosen_b = float(bvals[idx])
    if float(diffs[idx]) > float(tolerance):
        raise ValueError(
            f"No b value within tolerance: target={target_b:.6f}, "
            f"nearest={chosen_b:.6f}, |db|={float(diffs[idx]):.6f}, tol={tolerance:.6f}"
        )
    return idx, chosen_b


def _write_single_b_csv(
    out: Path,
    *,
    x: np.ndarray,
    raa9: np.ndarray,
    sem9: np.ndarray,
    x_label: str,
) -> None:
    header = (
        f"{x_label},RAA_1S,RAA_2S,RAA_1P0,RAA_1P1,RAA_1P2,RAA_3S,RAA_2P0,RAA_2P1,RAA_2P2,"
        "SEM_1S,SEM_2S,SEM_1P0,SEM_1P1,SEM_1P2,SEM_3S,SEM_2P0,SEM_2P1,SEM_2P2"
    )
    arr = np.column_stack([x, raa9, sem9])
    np.savetxt(out, arr, delimiter=",", header=header, comments="", fmt="%.10g")


def _stack_pt_window(scan: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bvals = scan["bvals"]
    results: List[dict] = scan["results"]
    if not results:
        raise RuntimeError("No b groups found in pT scan result")
    n_b = len(bvals)
    pt0 = results[0]["pt_centers"]
    if pt0 is None:
        raise RuntimeError("No pT spectrum in scan result")
    n_pt = len(pt0)
    raa = np.full((n_b, n_pt, 9), np.nan, dtype=np.float64)
    sem = np.full((n_b, n_pt, 9), np.nan, dtype=np.float64)
    for i, r in enumerate(results):
        if r["raa9_incl_vs_pt"] is None:
            continue
        raa[i, :, :] = r["raa9_incl_vs_pt"]
        sem[i, :, :] = r["raa9_incl_sem_vs_pt"]
    return bvals, pt0, raa, sem


def _stack_y(scan: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bvals = scan["bvals"]
    results: List[dict] = scan["results"]
    if not results:
        raise RuntimeError("No b groups found in y scan result")
    n_b = len(bvals)
    y0 = results[0]["y_centers"]
    if y0 is None:
        raise RuntimeError("No y spectrum in scan result")
    n_y = len(y0)
    raa = np.full((n_b, n_y, 9), np.nan, dtype=np.float64)
    sem = np.full((n_b, n_y, 9), np.nan, dtype=np.float64)
    for i, r in enumerate(results):
        if r["raa9_incl_vs_y"] is None:
            continue
        raa[i, :, :] = r["raa9_incl_vs_y"]
        sem[i, :, :] = r["raa9_incl_sem_vs_y"]
    return bvals, y0, raa, sem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datafile", required=True, help="Path to qtraj output (multi-b or single-b)")
    parser.add_argument(
        "--system-key",
        required=True,
        choices=("pbpb2760", "pbpb5023", "auau200", "oo5360"),
        help="Selects inclusive pp baseline sigmas_exp (see kinematics_presets)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write manifest and .npz spectra",
    )
    parser.add_argument(
        "--bmb",
        type=float,
        default=None,
        help="Optional minimum-bias b to exclude (same as full pipeline)",
    )
    parser.add_argument(
        "--single-b-target",
        type=float,
        default=None,
        help=(
            "If set, also export single-b spectra at the nearest available b "
            "(no centrality weighting)."
        ),
    )
    parser.add_argument(
        "--single-b-tol",
        type=float,
        default=0.2,
        help="Absolute |b-selected - b-target| tolerance for --single-b-target.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("run_per_b_spectra")

    sk = get_system_kinematics(args.system_key)
    pt_edges, y_edges, pt_max_for_y, pt_rapidity_windows = _select_binning(sk.key)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "datafile": str(Path(args.datafile).resolve()),
        "system_key": sk.key,
        "energy_label": sk.energy_label,
        "sigmas_exp": sk.sigmas_exp.tolist(),
        "pt_edges": pt_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "pt_max_for_y": float(pt_max_for_y),
        "rapidity_windows": [{"key": k, "y_lo": lo, "y_hi": hi} for k, (lo, hi), _ in pt_rapidity_windows],
    }
    single_b_payload: Dict[str, Any] = {}

    for key, y_window, _label in pt_rapidity_windows:
        logger.info("run_b_scan pT spectrum: window=%s %s", key, y_window)
        scan = run_b_scan(
            args.datafile,
            sk.sigmas_exp,
            bmb=args.bmb,
            pt_edges=pt_edges,
            y_edges=None,
            y_window_for_pt=y_window,
            pt_max_for_y=None,
            logger_=logger,
        )
        bvals, pt_centers, raa9, sem9 = _stack_pt_window(scan)
        np.savez_compressed(
            out / f"spectra_raavspt__{key}.npz",
            bvals=bvals,
            pt_centers=pt_centers,
            raa9=raa9,
            sem9=sem9,
            window_key=key,
        )
        manifest[f"raavspt__{key}"] = {
            "file": f"spectra_raavspt__{key}.npz",
            "shape_raa9": list(raa9.shape),
        }
        if args.single_b_target is not None:
            ib, b_selected = _nearest_b_index(bvals, args.single_b_target, args.single_b_tol)
            raa9_b = raa9[ib]
            sem9_b = sem9[ib]
            if not np.isfinite(raa9_b).any():
                raise RuntimeError(
                    f"Selected b={b_selected:.6f} has no finite R_AA vs pT values "
                    f"for window={key}; refusing to export NaN-only spectra."
                )
            np.savez_compressed(
                out / f"single_b_raavspt__{key}.npz",
                b_selected=b_selected,
                b_target=float(args.single_b_target),
                pt_centers=pt_centers,
                raa9=raa9_b,
                sem9=sem9_b,
                window_key=key,
            )
            _write_single_b_csv(
                out / f"raavspt__{key}__single_b.csv",
                x=pt_centers,
                raa9=raa9_b,
                sem9=sem9_b,
                x_label="pt",
            )
            single_b_payload.setdefault("b_target", float(args.single_b_target))
            if "b_selected" in single_b_payload and abs(single_b_payload["b_selected"] - b_selected) > 1e-6:
                raise RuntimeError(
                    "Inconsistent selected b across pT windows: "
                    f"{single_b_payload['b_selected']:.6f} vs {b_selected:.6f} (window={key})"
                )
            single_b_payload.setdefault("b_selected", b_selected)
            single_b_payload.setdefault("selection_tolerance", float(args.single_b_tol))
            single_b_payload.setdefault("raavspt", []).append(
                {
                    "window_key": key,
                    "npz": f"single_b_raavspt__{key}.npz",
                    "csv": f"raavspt__{key}__single_b.csv",
                    "n_pt_bins": int(len(pt_centers)),
                }
            )

    logger.info("run_b_scan R_AA vs y")
    scan_y = run_b_scan(
        args.datafile,
        sk.sigmas_exp,
        bmb=args.bmb,
        pt_edges=None,
        y_edges=y_edges,
        y_window_for_pt=None,
        pt_max_for_y=pt_max_for_y,
        logger_=logger,
    )
    bvals_y, y_centers, raa9_y, sem9_y = _stack_y(scan_y)
    np.savez_compressed(
        out / "spectra_raavsy.npz",
        bvals=bvals_y,
        y_centers=y_centers,
        raa9=raa9_y,
        sem9=sem9_y,
    )
    manifest["raavsy"] = {"file": "spectra_raavsy.npz", "shape_raa9": list(raa9_y.shape)}
    if args.single_b_target is not None:
        ib_y, b_selected_y = _nearest_b_index(bvals_y, args.single_b_target, args.single_b_tol)
        raa9_y_b = raa9_y[ib_y]
        sem9_y_b = sem9_y[ib_y]
        if not np.isfinite(raa9_y_b).any():
            raise RuntimeError(
                f"Selected b={b_selected_y:.6f} has no finite R_AA vs y values; "
                "refusing to export NaN-only spectra."
            )
        np.savez_compressed(
            out / "single_b_raavsy.npz",
            b_selected=b_selected_y,
            b_target=float(args.single_b_target),
            y_centers=y_centers,
            raa9=raa9_y_b,
            sem9=sem9_y_b,
        )
        _write_single_b_csv(
            out / "raavsy__single_b.csv",
            x=y_centers,
            raa9=raa9_y_b,
            sem9=sem9_y_b,
            x_label="y",
        )
        single_b_payload.setdefault("b_target", float(args.single_b_target))
        if "b_selected" in single_b_payload and abs(single_b_payload["b_selected"] - b_selected_y) > 1e-6:
            raise RuntimeError(
                "Inconsistent selected b between pT and y scans: "
                f"{single_b_payload['b_selected']:.6f} vs {b_selected_y:.6f}"
            )
        single_b_payload.setdefault("b_selected", b_selected_y)
        single_b_payload.setdefault("selection_tolerance", float(args.single_b_tol))
        single_b_payload["raavsy"] = {
            "npz": "single_b_raavsy.npz",
            "csv": "raavsy__single_b.csv",
            "n_y_bins": int(len(y_centers)),
        }
        logger.info(
            "Single-b export: target b=%.4f -> selected b=%.4f (tol=%.4f)",
            float(args.single_b_target),
            b_selected_y,
            float(args.single_b_tol),
        )
    if single_b_payload:
        manifest["single_b"] = single_b_payload

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
