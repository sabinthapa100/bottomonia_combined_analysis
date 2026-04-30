#!/usr/bin/env python3
"""Combine OO 5.36 TeV CNM x HNM R_AA, bin-by-bin.

Inputs
------
- HNM band directory produced by ``compare_oo_kappa.py``, containing:
    oo5360_integrated__band.csv
    oo5360_raavspt__band.csv
    oo5360_raavsy__band.csv
    oo5360_integrated_dr__band.csv
    oo5360_drvspt__band.csv
    oo5360_drvsy__band.csv
- CNM-only data directory (``outputs/CMS_Collab_OxygenOxygen/CNM_only/data``):
    oo5360_cnm_integrated.csv
    oo5360_cnm_vs_pt.csv
    oo5360_cnm_vs_y.csv

Combination
-----------
Per bin and per state (1S, 2S, 3S):
    R_AA(total) = R_AA(HNM) * R_AA(CNM)
    relative total error = sqrt( (dR_HNM / R_HNM)^2 + (dR_CNM / R_CNM)^2 )
where dR_HNM = 0.5 * (total_max - total_min) from the HNM band file (kappa
model spread in quadrature with trajectory SEM) and dR_CNM = 0.5 *
(band_hi - band_lo) from the CNM file (nPDF + eloss systematic).

Double ratios are invariant under state-independent CNM multiplication; we
therefore copy the HNM DR band CSVs verbatim into the combined output and
annotate them explicitly, rather than re-running a lossy CNM multiplication
that would cancel anyway.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
CNM_DIR = REPO_ROOT / "outputs" / "CMS_Collab_OxygenOxygen" / "CNM_only" / "data"
STATES: Tuple[str, ...] = ("1S", "2S", "3S")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _combine(raa_hnm: float, hnm_lo: float, hnm_hi: float,
             raa_cnm: float, cnm_lo: float, cnm_hi: float) -> Tuple[float, float, float]:
    """Return (central, lo, hi) for R_AA(total) = R_AA(HNM) * R_AA(CNM)."""
    total = raa_hnm * raa_cnm
    d_hnm = 0.5 * (hnm_hi - hnm_lo)
    d_cnm = 0.5 * (cnm_hi - cnm_lo)
    if raa_hnm > 0 and raa_cnm > 0:
        rel = ((d_hnm / raa_hnm) ** 2 + (d_cnm / raa_cnm) ** 2) ** 0.5
    else:
        rel = 0.0
    return total, total * (1.0 - rel), total * (1.0 + rel)


def combine_integrated(hnm_dir: Path, out_dir: Path) -> None:
    hnm_rows = _read_csv(hnm_dir / "oo5360_integrated__band.csv")
    cnm_rows = _read_csv(CNM_DIR / "oo5360_cnm_integrated.csv")
    c = cnm_rows[0]
    raa_cnm = float(c["raa_cnm"]); cnm_lo = float(c["band_lo"]); cnm_hi = float(c["band_hi"])

    hnm_by_state = {r["state"]: r for r in hnm_rows}
    out: Dict[str, object] = {
        "y_min": c["y_min"], "y_max": c["y_max"],
        "pt_min": c["pt_min"], "pt_max": c["pt_max"],
        "cnm_factor": raa_cnm, "cnm_band_lo": cnm_lo, "cnm_band_hi": cnm_hi,
    }
    for st in STATES:
        h = hnm_by_state[st]
        total, lo, hi = _combine(
            float(h["central"]), float(h["total_min"]), float(h["total_max"]),
            raa_cnm, cnm_lo, cnm_hi,
        )
        out[f"raa_{st}"] = total
        out[f"band_lo_{st}"] = lo
        out[f"band_hi_{st}"] = hi
        out[f"hnm_central_{st}"] = float(h["central"])
        out[f"hnm_total_min_{st}"] = float(h["total_min"])
        out[f"hnm_total_max_{st}"] = float(h["total_max"])
    out["notes"] = "R_AA(total)=R_AA(HNM)*R_AA(CNM); relative err = sqrt(HNM^2 + CNM^2) in quadrature."

    fields = [
        "y_min","y_max","pt_min","pt_max","cnm_factor","cnm_band_lo","cnm_band_hi",
        "raa_1S","band_lo_1S","band_hi_1S","hnm_central_1S","hnm_total_min_1S","hnm_total_max_1S",
        "raa_2S","band_lo_2S","band_hi_2S","hnm_central_2S","hnm_total_min_2S","hnm_total_max_2S",
        "raa_3S","band_lo_3S","band_hi_3S","hnm_central_3S","hnm_total_min_3S","hnm_total_max_3S",
        "notes",
    ]
    _write_csv(out_dir / "oo5360_total_raa_integrated.csv", [out], fields)


def combine_vs_pt(hnm_dir: Path, out_dir: Path) -> None:
    hnm_rows = _read_csv(hnm_dir / "oo5360_raavspt__band.csv")
    cnm_rows = _read_csv(CNM_DIR / "oo5360_cnm_vs_pt.csv")
    # CNM rows keyed by pt_center for robustness
    cnm_by_center = {round(float(r["pt_center"]), 3): r for r in cnm_rows}

    out_rows: List[Dict[str, object]] = []
    for h in hnm_rows:
        ptc = round(float(h["x"]), 3)
        c = cnm_by_center.get(ptc)
        if c is None:
            continue
        raa_cnm = float(c["raa_cnm"]); cnm_lo = float(c["band_lo"]); cnm_hi = float(c["band_hi"])
        row: Dict[str, object] = {
            "pt_low": c["pt_low"], "pt_high": c["pt_high"], "pt_center": c["pt_center"],
            "cnm_factor": raa_cnm, "cnm_band_lo": cnm_lo, "cnm_band_hi": cnm_hi,
        }
        for st in STATES:
            total, lo, hi = _combine(
                float(h[f"central_{st}"]), float(h[f"total_min_{st}"]), float(h[f"total_max_{st}"]),
                raa_cnm, cnm_lo, cnm_hi,
            )
            row[f"raa_{st}"] = total
            row[f"band_lo_{st}"] = lo
            row[f"band_hi_{st}"] = hi
        row["notes"] = "R_AA(total)=R_AA(HNM)*R_AA(CNM); band in quadrature"
        out_rows.append(row)

    fields = ["pt_low","pt_high","pt_center","cnm_factor","cnm_band_lo","cnm_band_hi",
              "raa_1S","band_lo_1S","band_hi_1S",
              "raa_2S","band_lo_2S","band_hi_2S",
              "raa_3S","band_lo_3S","band_hi_3S","notes"]
    _write_csv(out_dir / "oo5360_total_raa_vs_pt.csv", out_rows, fields)


def combine_vs_y(hnm_dir: Path, out_dir: Path) -> None:
    hnm_rows = _read_csv(hnm_dir / "oo5360_raavsy__band.csv")
    cnm_rows = _read_csv(CNM_DIR / "oo5360_cnm_vs_y.csv")
    cnm_by_center = {round(float(r["y_center"]), 3): r for r in cnm_rows}

    out_rows: List[Dict[str, object]] = []
    for h in hnm_rows:
        yc = round(float(h["x"]), 3)
        c = cnm_by_center.get(yc)
        if c is None:
            continue
        raa_cnm = float(c["raa_cnm"]); cnm_lo = float(c["band_lo"]); cnm_hi = float(c["band_hi"])
        row: Dict[str, object] = {
            "y_low": c["y_low"], "y_high": c["y_high"], "y_center": c["y_center"],
            "cnm_factor": raa_cnm, "cnm_band_lo": cnm_lo, "cnm_band_hi": cnm_hi,
        }
        for st in STATES:
            total, lo, hi = _combine(
                float(h[f"central_{st}"]), float(h[f"total_min_{st}"]), float(h[f"total_max_{st}"]),
                raa_cnm, cnm_lo, cnm_hi,
            )
            row[f"raa_{st}"] = total
            row[f"band_lo_{st}"] = lo
            row[f"band_hi_{st}"] = hi
        row["notes"] = "R_AA(total)=R_AA(HNM)*R_AA(CNM); band in quadrature"
        out_rows.append(row)

    fields = ["y_low","y_high","y_center","cnm_factor","cnm_band_lo","cnm_band_hi",
              "raa_1S","band_lo_1S","band_hi_1S",
              "raa_2S","band_lo_2S","band_hi_2S",
              "raa_3S","band_lo_3S","band_hi_3S","notes"]
    _write_csv(out_dir / "oo5360_total_raa_vs_y.csv", out_rows, fields)


def copy_dr(hnm_dir: Path, out_dir: Path) -> None:
    """Double ratios are invariant under state-independent CNM — copy HNM DR verbatim."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("oo5360_integrated_dr__band.csv", "oo5360_drvspt__band.csv", "oo5360_drvsy__band.csv"):
        src = hnm_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name.replace("__band.csv", "__band__cnm_invariant.csv"))


def write_manifest(hnm_dir: Path, out_dir: Path) -> None:
    manifest = {
        "description": (
            "OO 5.36 TeV CNM x HNM total R_AA. R_AA(total) = R_AA(HNM) * R_AA(CNM). "
            "Relative band is sqrt(HNM_rel^2 + CNM_rel^2). Double ratios are copied "
            "verbatim from the HNM band (CNM factor is state-independent and cancels)."
        ),
        "hnm_source": str(hnm_dir.relative_to(REPO_ROOT)),
        "cnm_source": str(CNM_DIR.relative_to(REPO_ROOT)),
        "kinematics": {"integrated_y_window": [-2.4, 2.4], "pt_max_for_y": 30.0},
    }
    (out_dir / "oo5360_combined_manifest.json").write_text(json.dumps(manifest, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Combine OO CNM x HNM R_AA bin-by-bin.")
    ap.add_argument("--hnm-dir", required=True, type=Path,
                    help="HNM band directory (e.g. OxygenOxygen5p36TeV_kappa57_final)")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Output directory for the combined CSVs.")
    args = ap.parse_args(argv)

    hnm_dir = args.hnm_dir if args.hnm_dir.is_absolute() else (REPO_ROOT / args.hnm_dir).resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir).resolve()

    combine_integrated(hnm_dir, out_dir)
    combine_vs_pt(hnm_dir, out_dir)
    combine_vs_y(hnm_dir, out_dir)
    copy_dr(hnm_dir, out_dir)
    write_manifest(hnm_dir, out_dir)

    print(f"Wrote CNM x HNM combined outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
