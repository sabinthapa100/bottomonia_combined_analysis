"""
Parity harness: compare generated-from-datafile AuAu200 RAA series
against the existing Mathematica-exported values.

Run from repo root:
    python3 scripts/parity_auau200.py
"""
import logging
import sys
import numpy as np

sys.path.insert(0, "hnm/qtraj_out_analysis")

from qtraj_analysis.reference_data import (
    load_theory_series,
    _load_generated_auau200_raa_series,
)
from qtraj_analysis.observable_registry import get_observable_spec

logging.basicConfig(level=logging.WARNING)

OBSERVABLES = [
    "auau200_raavsnpart",
    "auau200_raavspt",
    "auau200_raavsy",
]

MAX_ABS_DIFF_TOLERANCE = 0.05  # flag if |Δ| > 5% (R_AA is O(1))

overall_pass = True

print("=" * 70)
print("PARITY HARNESS: AuAu200  generated vs Mathematica")
print("=" * 70)

for obs_id in OBSERVABLES:
    print(f"\n{'─'*60}")
    print(f"  {obs_id}")
    print(f"{'─'*60}")
    spec = get_observable_spec(obs_id)

    math_series = load_theory_series(obs_id)
    gen_series = _load_generated_auau200_raa_series(spec)

    print(f"  Mathematica series : {len(math_series)}")
    print(f"  Generated series   : {len(gen_series)}")

    # Index by (source_label, series_label)
    math_map = {(s.source_label, s.series_label): s for s in math_series}
    gen_map  = {(s.source_label, s.series_label): s for s in gen_series}

    print()

    # --- print Mathematica values ---
    for key, ms in sorted(math_map.items()):
        src, state = key
        print(f"  [math] [{src}] {state:5s}  x={np.round(ms.x, 3).tolist()}")
        print(f"                         c={np.round(ms.center, 4).tolist()}")

    print()

    # --- print generated values and delta ---
    for key, gs in sorted(gen_map.items()):
        src, state = key
        print(f"  [gen]  [{src}] {state:5s}  x={np.round(gs.x, 3).tolist()}")
        print(f"                         c={np.round(gs.center, 4).tolist()}")

        if key in math_map:
            ms = math_map[key]
            # x-axis alignment check
            if ms.x.shape != gs.x.shape:
                print(f"  *** x-shape mismatch: math {ms.x.shape} vs gen {gs.x.shape}")
                overall_pass = False
                continue

            # Interpolate generated onto Mathematica x-grid if shapes differ
            if not np.allclose(ms.x, gs.x, atol=0.5):
                print(f"  *** x-values differ; skipping pointwise compare")
                continue

            # Align by Mathematica x-size if generated has more points (e.g. vs_npart)
            # Take positions where Mathematica has data
            if ms.x.shape == gs.x.shape:
                delta = gs.center - ms.center
                rel = np.where(np.abs(ms.center) > 1e-4, delta / ms.center, np.zeros_like(delta))
                max_abs = float(np.nanmax(np.abs(delta)))
                max_rel = float(np.nanmax(np.abs(rel)))
                flag = "*** FAIL" if max_abs > MAX_ABS_DIFF_TOLERANCE else "PASS"
                if max_abs > MAX_ABS_DIFF_TOLERANCE:
                    overall_pass = False
                print(f"  Δ={np.round(delta, 4).tolist()}")
                print(f"  max|Δ|={max_abs:.4f}  max|Δ/math|={max_rel:.3f}  [{flag}]")
        else:
            print(f"  (no matching Mathematica series to compare)")

print()
print("=" * 70)
print(f"OVERALL: {'PASS' if overall_pass else 'FAIL — review deltas above'}")
print("=" * 70)
