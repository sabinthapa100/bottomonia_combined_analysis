"""
Parity harness: compare generated-from-datafile PbPb 2760/5023 RAA and ratio
series against the existing Mathematica-exported values.

Run from repo root:
    python3 scripts/parity_pbpb_raa.py
"""

import dataclasses
import logging
import sys
import numpy as np

sys.path.insert(0, "hnm/qtraj_out_analysis")

from qtraj_analysis.observable_registry import (
    OBSERVABLE_REGISTRY,
    get_observable_spec,
    SourceRef,
)
from qtraj_analysis.reference_data import (
    _load_generated_pbpb2760_raa_series,
    _load_generated_pbpb5023_raa_series,
    _load_generated_pbpb5023_ratio_series,
    load_theory_series,
)

logging.basicConfig(level=logging.WARNING)


def _ref(path, **kwargs):
    return SourceRef(path=path, **kwargs)


_MATH_SOURCES = {
    "pbpb2760_raavsnpart": (
        _ref("inputs/qtraj_inputs/PbPb2760/figures/raavsnpart-lhc-2.76-3d-k3.m"),
        _ref("inputs/qtraj_inputs/PbPb2760/figures/raavsnpart-lhc-2.76-3d-k4.m"),
    ),
    "pbpb2760_raavspt": (
        _ref("inputs/qtraj_inputs/PbPb2760/figures/raavspt-lhc-2.76-3d-k3.m"),
        _ref("inputs/qtraj_inputs/PbPb2760/figures/raavspt-lhc-2.76-3d-k4.m"),
    ),
    "pbpb2760_raavsy": (
        _ref("inputs/qtraj_inputs/PbPb2760/figures/raavsy-lhc-2.76-3d-k3.m"),
        _ref("inputs/qtraj_inputs/PbPb2760/figures/raavsy-lhc-2.76-3d-k4.m"),
    ),
    "pbpb5023_raavsnpart": (
        _ref("inputs/qtraj_inputs/PbPb5023/figures/raavsnpart-lhc3d-k3.m"),
        _ref("inputs/qtraj_inputs/PbPb5023/figures/raavsnpart-lhc3d-k4.m"),
    ),
    "pbpb5023_raavspt": (
        _ref("inputs/qtraj_inputs/PbPb5023/figures/raavspt-lhc3d-k3.m"),
        _ref("inputs/qtraj_inputs/PbPb5023/figures/raavspt-lhc3d-k4.m"),
    ),
    "pbpb5023_raavsy": (
        _ref("inputs/qtraj_inputs/PbPb5023/figures/raavsy-lhc3d-k3.m"),
        _ref("inputs/qtraj_inputs/PbPb5023/figures/raavsy-lhc3d-k4.m"),
    ),
    "pbpb5023_ratio21vsnpart": (
        _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio2s1svsnpart-lhc3d-k3.m"),
        _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio2s1svsnpart-lhc3d-k4.m"),
    ),
    "pbpb5023_ratio31vsnpart": (
        _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio3s1svsnpart-lhc3d-k3.m"),
        _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio3s1svsnpart-lhc3d-k4.m"),
    ),
    "pbpb5023_ratio21vspt": (
        _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio-2s1s-vspt-lhc3d-k3.m"),
        _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio-2s1s-vspt-lhc3d-k4.m"),
    ),
    "pbpb5023_ratio32vspt": (
        _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio-3s2s-vspt-lhc3d-k3.m"),
        _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio-3s2s-vspt-lhc3d-k4.m"),
    ),
}

_GEN_FUNC = {
    "pbpb2760_raavsnpart": _load_generated_pbpb2760_raa_series,
    "pbpb2760_raavspt": _load_generated_pbpb2760_raa_series,
    "pbpb2760_raavsy": _load_generated_pbpb2760_raa_series,
    "pbpb5023_raavsnpart": _load_generated_pbpb5023_raa_series,
    "pbpb5023_raavspt": _load_generated_pbpb5023_raa_series,
    "pbpb5023_raavsy": _load_generated_pbpb5023_raa_series,
    "pbpb5023_ratio21vsnpart": _load_generated_pbpb5023_ratio_series,
    "pbpb5023_ratio31vsnpart": _load_generated_pbpb5023_ratio_series,
    "pbpb5023_ratio21vspt": _load_generated_pbpb5023_ratio_series,
    "pbpb5023_ratio32vspt": _load_generated_pbpb5023_ratio_series,
}

MAX_ABS_DIFF_TOLERANCE = 0.05
overall_pass = True

print("=" * 70)
print("PARITY HARNESS: PbPb2760 + PbPb5023  generated vs Mathematica")
print("=" * 70)

for obs_id, math_sources in _MATH_SOURCES.items():
    print(f"\n{'─'*60}")
    print(f"  {obs_id}")
    print(f"{'─'*60}")

    spec = get_observable_spec(obs_id)

    # Temporarily patch the registry to restore old Mathematica sources
    patched_spec = dataclasses.replace(spec, mathematica_sources=math_sources)
    OBSERVABLE_REGISTRY[obs_id] = patched_spec
    try:
        math_series = load_theory_series(obs_id)
    finally:
        OBSERVABLE_REGISTRY[obs_id] = spec  # always restore

    gen_series = _GEN_FUNC[obs_id](spec)

    print(f"  Mathematica series : {len(math_series)}")
    print(f"  Generated series   : {len(gen_series)}")

    math_map = {(s.source_label, s.series_label): s for s in math_series}
    gen_map  = {(s.source_label, s.series_label): s for s in gen_series}

    print()
    for key, ms in sorted(math_map.items()):
        src, state = key
        print(f"  [math] [{src}] {state:6s}  x={np.round(ms.x[:4], 2).tolist()} ...")
        print(f"                         c={np.round(ms.center[:4], 4).tolist()} ...")

    print()
    for key, gs in sorted(gen_map.items()):
        src, state = key
        print(f"  [gen]  [{src}] {state:6s}  x={np.round(gs.x[:4], 2).tolist()} ...")
        print(f"                         c={np.round(gs.center[:4], 4).tolist()} ...")
        if key in math_map:
            ms = math_map[key]
            if ms.center.shape != gs.center.shape:
                print(
                    f"  *** shape mismatch: math {ms.center.shape} vs gen {gs.center.shape}"
                )
                overall_pass = False
                continue
            diff = np.abs(ms.center - gs.center)
            max_diff = float(np.nanmax(diff))
            flag = "PASS" if max_diff < MAX_ABS_DIFF_TOLERANCE else "FAIL ***"
            print(f"  max|Δ| = {max_diff:.6f}  {flag}")
            if max_diff >= MAX_ABS_DIFF_TOLERANCE:
                overall_pass = False
                print(f"  diff vector: {np.round(diff, 6).tolist()}")

print()
print("=" * 70)
print(f"RESULT: {'ALL PASS' if overall_pass else '*** FAILURES DETECTED ***'}")
print("=" * 70)
sys.exit(0 if overall_pass else 1)
