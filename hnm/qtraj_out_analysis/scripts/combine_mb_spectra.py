#!/usr/bin/env python3
"""
Combine per-b R_AA spectra from `run_per_b_spectra` into a minimum-bias curve using weights.

Weights file: two columns per line (whitespace or comma separated)
  b  weight
Lines starting with # are ignored. Weights must be non-negative; normalized internally.

Example:
  # b(fm)  frac_sigma
  2.0  0.1
  5.0  0.2
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

from qtraj_analysis.min_bias_combine import (  # noqa: E402
    align_weights_to_bvals,
    load_weights_b_pairs,
    weighted_average_raa9,
)


def _combine_one_npz(
    path: Path,
    weights: np.ndarray,
    logger: logging.Logger,
) -> Path:
    data = np.load(path, allow_pickle=True)
    bvals = np.asarray(data["bvals"], dtype=np.float64)
    w = align_weights_to_bvals(bvals, weights[:, 0], weights[:, 1])
    raa9 = np.asarray(data["raa9"], dtype=np.float64)
    sem9 = np.asarray(data["sem9"], dtype=np.float64)
    comb, comb_sem = weighted_average_raa9(raa9, w, sem9)
    out_path = path.with_name(path.stem + "_mb" + path.suffix)
    extras = {k: data[k] for k in data.files if k in ("pt_centers", "y_centers", "window_key")}
    np.savez_compressed(
        out_path,
        raa9_mb=comb,
        sem9_mb=comb_sem,
        weights_used=w,
        bvals=bvals,
        **extras,
    )
    logger.info("Wrote %s", out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectra-dir",
        required=True,
        help="Directory containing manifest.json from run_per_b_spectra",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Two-column file: b  weight",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("combine_mb_spectra")

    sdir = Path(args.spectra_dir)
    weights_raw = load_weights_b_pairs(args.weights)
    wmat = np.column_stack([weights_raw[0], weights_raw[1]])

    paths = sorted(sdir.glob("spectra_raavspt__*.npz")) + sorted(sdir.glob("spectra_raavsy.npz"))
    paths = [p for p in paths if "_mb.npz" not in p.name]
    if not paths:
        logger.error("No spectra_raavspt__*.npz or spectra_raavsy.npz under %s", sdir)
        return 1

    out_list = []
    for path in paths:
        out_list.append(str(_combine_one_npz(path, wmat, logger)))

    summary = {"spectra_dir": str(sdir.resolve()), "weights_file": str(Path(args.weights).resolve()), "outputs": out_list}
    (sdir / "combine_mb_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary -> %s", sdir / "combine_mb_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
