"""
Diagnose the raavspt/raavsy discrepancy by testing two averaging approaches:
  A) flat 0-60% centrality cut (current broken impl)
  B) p_centrality(c) weighting over full range  (what Mathematica likely does)
  C) p_centrality(c) weighting restricted to 0-60%

Run from repo root:
    python3 scripts/diagnose_raavspt.py
"""
import logging
import sys
import numpy as np

sys.path.insert(0, "hnm/qtraj_out_analysis")

from qtraj_analysis.reference_data import (
    load_theory_series,
    _read_generated_observables,
    _SIGMAS_EXP,
    _LOGGER,
)
from qtraj_analysis.observable_registry import get_observable_spec, resolve_source
from qtraj_analysis.feeddown import (
    build_feeddown_matrix,
    solve_primordial_sigmas,
    apply_feeddown_to_raa6,
    split_hyperfine_6_to_9,
)
from qtraj_analysis.glauber import GlauberInterpolator, load_canonical_glauber
from qtraj_analysis.stats import p_centrality, weighted_avg_surv9

logging.basicConfig(level=logging.WARNING)

spec_pt = get_observable_spec("auau200_raavspt")
spec_y  = get_observable_spec("auau200_raavsy")

feeddown = build_feeddown_matrix()
sigmas   = solve_primordial_sigmas(feeddown, _SIGMAS_EXP)
glauber  = GlauberInterpolator(load_canonical_glauber("auau200", _LOGGER))

# Load kappa4 trajectories
source = spec_pt.datafile_sources[0]
obs = _read_generated_observables(source)

# Vectorize
b_arr    = np.asarray([e.b    for e in obs], dtype=np.float64)
pt_arr   = np.asarray([e.pt   for e in obs], dtype=np.float64)
y_arr    = np.asarray([e.y    for e in obs], dtype=np.float64)
q_arr    = np.asarray([e.qweight for e in obs], dtype=np.float64)
surv6_arr = np.vstack([e.surv6 for e in obs])  # (n, 6)
c_arr    = glauber.b_to_c(b_arr)

math_pt = load_theory_series("auau200_raavspt")
math_k4 = {s.series_label: s for s in math_pt if s.source_label == "kappa4"}

print("=" * 70)
print("DIAGNOSIS: raavspt  (kappa4 source)")
print("Mathematica:  1S = {}, 2S = {}".format(
    math_k4["1S"].center.tolist(), math_k4["2S"].center.tolist()))
print()

pt_edges = np.asarray(spec_pt.grid.bin_edges, dtype=np.float64)


def compute_raa_pt(mask_extra, label):
    results_1S = []
    results_2S = []
    for pt_min, pt_max in zip(pt_edges[:-1], pt_edges[1:]):
        mask = (np.abs(y_arr) <= 1.0) & (pt_arr >= pt_min) & (pt_arr < pt_max) & mask_extra
        if not np.any(mask):
            results_1S.append(np.nan)
            results_2S.append(np.nan)
            continue
        S = surv6_arr[mask]
        q = q_arr[mask]
        # cent-probability weight
        c_sel = c_arr[mask]
        w = p_centrality(c_sel)
        wtm = np.mean(w)
        X9 = np.vstack([split_hyperfine_6_to_9(s) for s in S])  # (n,9)
        X9w = (w[:, None] / wtm) * X9
        avg9 = (X9w.T @ q) / q.sum()
        num  = feeddown @ (sigmas * avg9)
        den  = feeddown @ sigmas
        raa9 = num / den
        results_1S.append(float(raa9[0]))
        results_2S.append(float(raa9[1]))
    print(f"  [{label}] 1S = {np.round(results_1S, 4).tolist()}")
    print(f"  [{label}] 2S = {np.round(results_2S, 4).tolist()}")


# A: flat 0-60% cut, NO p_centrality weight (baseline — this is what was wrong)
def compute_raa_pt_flat(mask_extra, label):
    results_1S = []
    results_2S = []
    for pt_min, pt_max in zip(pt_edges[:-1], pt_edges[1:]):
        mask = (np.abs(y_arr) <= 1.0) & (pt_arr >= pt_min) & (pt_arr < pt_max) & mask_extra
        if not np.any(mask):
            results_1S.append(np.nan)
            results_2S.append(np.nan)
            continue
        S = surv6_arr[mask]
        q = q_arr[mask]
        avg6 = (S.T @ q) / q.sum()
        sem6 = np.std(S, axis=0, ddof=1) / np.sqrt(S.shape[0]) if S.shape[0] > 1 else np.zeros(6)
        raa9, _ = apply_feeddown_to_raa6(avg6, sem6, feeddown, sigmas)
        results_1S.append(float(raa9[0]))
        results_2S.append(float(raa9[1]))
    print(f"  [{label}] 1S = {np.round(results_1S, 4).tolist()}")
    print(f"  [{label}] 2S = {np.round(results_2S, 4).tolist()}")


print("A: flat avg, c<=0.6 cut (current broken):")
compute_raa_pt_flat(c_arr <= 0.6, "flat,0-60")
print()
print("B: p_centrality weight, c<=0.6 cut:")
compute_raa_pt(c_arr <= 0.6, "pcent,0-60")
print()
print("C: p_centrality weight, all centrality (no 0-60% cut):")
compute_raa_pt(np.ones(len(obs), dtype=bool), "pcent,all")
print()
print("D: flat avg, all centrality:")
compute_raa_pt_flat(np.ones(len(obs), dtype=bool), "flat,all")


# ---- same for raavsy ----
math_y = load_theory_series("auau200_raavsy")
math_k4_y = {s.series_label: s for s in math_y if s.source_label == "kappa4"}
print()
print("=" * 70)
print("DIAGNOSIS: raavsy  (kappa4 source)")
print("Mathematica:  1S = {}".format(math_k4_y["1S"].center.tolist()))
print()

y_edges = np.asarray(spec_y.grid.bin_edges, dtype=np.float64)


def compute_raa_y(mask_extra, label):
    results_1S = []
    for y_min, y_max in zip(y_edges[:-1], y_edges[1:]):
        mask = (y_arr >= y_min) & (y_arr < y_max) & (pt_arr < 10.0) & mask_extra
        if not np.any(mask):
            results_1S.append(np.nan)
            continue
        S = surv6_arr[mask]
        q = q_arr[mask]
        c_sel = c_arr[mask]
        w = p_centrality(c_sel)
        wtm = np.mean(w)
        X9 = np.vstack([split_hyperfine_6_to_9(s) for s in S])
        X9w = (w[:, None] / wtm) * X9
        avg9 = (X9w.T @ q) / q.sum()
        num  = feeddown @ (sigmas * avg9)
        den  = feeddown @ sigmas
        raa9 = num / den
        results_1S.append(float(raa9[0]))
    print(f"  [{label}] 1S = {np.round(results_1S, 4).tolist()}")


def compute_raa_y_flat(mask_extra, label):
    results_1S = []
    for y_min, y_max in zip(y_edges[:-1], y_edges[1:]):
        mask = (y_arr >= y_min) & (y_arr < y_max) & (pt_arr < 10.0) & mask_extra
        if not np.any(mask):
            results_1S.append(np.nan)
            continue
        S = surv6_arr[mask]
        q = q_arr[mask]
        avg6 = (S.T @ q) / q.sum()
        sem6 = np.std(S, axis=0, ddof=1) / np.sqrt(S.shape[0]) if S.shape[0] > 1 else np.zeros(6)
        raa9, _ = apply_feeddown_to_raa6(avg6, sem6, feeddown, sigmas)
        results_1S.append(float(raa9[0]))
    print(f"  [{label}] 1S = {np.round(results_1S, 4).tolist()}")


print("A: flat avg, c<=0.6:")
compute_raa_y_flat(c_arr <= 0.6, "flat,0-60")
print()
print("B: p_centrality, c<=0.6:")
compute_raa_y(c_arr <= 0.6, "pcent,0-60")
print()
print("C: p_centrality, all centrality:")
compute_raa_y(np.ones(len(obs), dtype=bool), "pcent,all")
print()
print("D: flat avg, all centrality:")
compute_raa_y_flat(np.ones(len(obs), dtype=bool), "flat,all")
