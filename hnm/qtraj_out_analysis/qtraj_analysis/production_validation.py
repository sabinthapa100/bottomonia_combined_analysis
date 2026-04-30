from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from qtraj_analysis.feeddown import (
    apply_feeddown_to_raa6,
    build_feeddown_matrix,
    solve_primordial_sigmas,
)
from qtraj_analysis.glauber import (
    GlauberInterpolator,
    get_canonical_glauber_spec,
    load_canonical_glauber,
)
from qtraj_analysis.io import parse_records, read_whitespace_table
from qtraj_analysis.matching import build_observables
from qtraj_analysis.observable_registry import get_observable_spec
from qtraj_analysis.reference_data import load_theory_series
from qtraj_analysis.stats import weighted_avg_surv9


_SIGMAS_EXP = np.array([57.6, 19.0, 3.72, 13.69, 16.1, 6.8, 3.27, 12.0, 14.15], dtype=np.float64)

_PBPB5023_RATIO32_EXPECTED: Dict[str, List[Tuple[float, float]]] = {
    "k3": [
        (0.0, 0.8548998944316261),
        (4.0, 0.6451196465407647),
        (9.0, 0.7055334562907950),
        (15.0, 0.7230337657782389),
        (30.0, 0.7230337657782389),
    ],
    "k4": [
        (0.0, 0.6738551396377674),
        (4.0, 0.6900649730976400),
        (9.0, 0.7389746342438895),
        (15.0, 0.7188035529280329),
        (30.0, 0.7188035529280329),
    ],
}

_PBPB2760_RATIO21_NPART_ARCHIVE_K3: List[Tuple[float, float]] = [
    (0.9705986591389940, 0.9797199147025983),
    (3.8095328022884103, 0.7470118446192688),
    (9.6677908362007980, 0.6417224575568893),
    (21.2927615205458950, 0.5218580502671659),
    (41.0509367091967200, 0.4213644614949028),
    (70.7871426138262000, 0.3123325730440449),
    (112.3900702773535900, 0.24190568800087558),
    (168.5028366222428500, 0.2020286949612291),
    (243.5379628297038400, 0.14855289709151168),
    (315.8591222901291000, 0.12329024326184523),
    (374.9882778251423000, 0.1058296286067281),
    (406.1223223526176300, 0.10684330601188903),
]

_PBPB2760_RATIO31_NPART_ARCHIVE_K3: List[Tuple[float, float]] = [
    (0.9705986591389940, 0.9772132261002171),
    (3.8095328022884103, 0.6098287786965034),
    (9.6677908362007980, 0.38153313900808666),
    (21.2927615205458950, 0.24682890282562547),
    (41.0509367091967200, 0.21567675165618483),
    (70.7871426138262000, 0.17346891735564082),
    (112.3900702773535900, 0.10247822126764157),
    (168.5028366222428500, 0.06706829725981166),
    (243.5379628297038400, 0.03465036201566919),
    (315.8591222901291000, 0.03444592316068338),
    (374.9882778251423000, 0.02586829615949936),
    (406.1223223526176300, 0.02730216661261922),
]

_PBPB2760_RATIO21_PT_ARCHIVE_K3: List[Tuple[float, float]] = [
    (2.5, 0.25613825630035475),
    (8.5, 0.25691516875256787),
    (21.0, 0.2809150723951365),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_OBS_CACHE: Dict[Tuple[str, str], list] = {}


def _primordial_sigmas() -> np.ndarray:
    return solve_primordial_sigmas(build_feeddown_matrix(), _SIGMAS_EXP)


def _read_lhc_obs(system_key: str, source_label: str, logger: logging.Logger):
    cache_key = (system_key, source_label)
    cached = _OBS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if system_key == "pbpb2760":
        path = (
            _repo_root()
            / "inputs"
            / "qtraj_inputs"
            / "PbPb2760"
            / "input"
            / f"lhc-2.76-3d-{source_label}"
            / "datafile-avg.gz"
        )
    elif system_key == "pbpb5023":
        path = (
            _repo_root()
            / "inputs"
            / "qtraj_inputs"
            / "PbPb5023"
            / f"lhc3d-{source_label}"
            / "datafile-avg.gz"
        )
    else:
        raise ValueError(f"Unsupported LHC system_key: {system_key}")

    table = read_whitespace_table(str(path), logger)
    obs = build_observables(parse_records(table, logger), logger)
    _OBS_CACHE[cache_key] = obs
    return obs


def _read_pbpb2760_obs(source_label: str, logger: logging.Logger):
    return _read_lhc_obs("pbpb2760", source_label, logger)


def _max_abs_diff(expected: List[Tuple[float, float]], observed: List[Tuple[float, float]]) -> float:
    exp = np.asarray(expected, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if exp.shape != obs.shape:
        return float("inf")
    return float(np.max(np.abs(exp - obs))) if exp.size else 0.0


def validate_pbpb5023_ratio32vspt(logger: logging.Logger) -> dict:
    series = load_theory_series("pbpb5023_ratio32vspt")
    report_series = []
    status = "pass"

    for entry in series:
        expected = _PBPB5023_RATIO32_EXPECTED.get(entry.source_label)
        observed = list(zip(entry.x.tolist(), entry.center.tolist()))
        max_abs_diff = _max_abs_diff(expected or [], observed)
        if expected is None or not np.isfinite(max_abs_diff) or max_abs_diff > 1e-12:
            status = "fail"
        report_series.append(
            {
                "source_label": entry.source_label,
                "expected_points": expected,
                "observed_points": observed,
                "max_abs_diff": max_abs_diff,
            }
        )

    return {
        "observable_id": "pbpb5023_ratio32vspt",
        "status": status,
        "tolerance": 1e-12,
        "note": (
            "The large first-bin 3S/2S value is accepted only because the parsed production series "
            "matches the local Mathematica exports exactly."
        ),
        "series": report_series,
    }


def _evaluate_step(bin_edges: np.ndarray, values: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    out = []
    for x in np.asarray(x_eval, dtype=np.float64):
        idx = int(np.searchsorted(bin_edges, x, side="right") - 1)
        idx = max(0, min(idx, len(values) - 1))
        out.append(float(values[idx]))
    return np.asarray(out, dtype=np.float64)


def _distance_to_band(y_obs: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    y_obs = np.asarray(y_obs, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return np.maximum.reduce([lower - y_obs, y_obs - upper, np.zeros_like(y_obs)])


def _theory_lookup(observable_id: str) -> Dict[Tuple[str, str], object]:
    return {
        (series.source_label, series.series_label): series
        for series in load_theory_series(observable_id)
    }


def _build_recomputed_npart_series(
    *,
    system_key: str,
    source_label: str,
    y_abs_max: float,
    pt_max: float,
    logger: logging.Logger,
) -> Dict[str, np.ndarray]:
    obs = _read_lhc_obs(system_key, source_label, logger)
    glauber = GlauberInterpolator(load_canonical_glauber(system_key, logger))
    feeddown = build_feeddown_matrix()
    sigmas = _primordial_sigmas()

    groups: Dict[float, list] = {}
    for entry in obs:
        if abs(entry.y) <= y_abs_max and entry.pt < pt_max:
            groups.setdefault(round(entry.b, 6), []).append(entry)

    x_vals: List[float] = []
    state_values: Dict[str, List[float]] = {"1S": [], "2S": [], "3S": []}
    counts: List[int] = []

    for b in sorted(groups):
        sample = groups[b]
        surv6 = np.vstack([entry.surv6 for entry in sample])
        qweights = np.asarray([entry.qweight for entry in sample], dtype=np.float64)
        avg6 = (surv6.T @ qweights) / qweights.sum()
        sem6 = np.std(surv6, axis=0, ddof=1) / np.sqrt(surv6.shape[0]) if surv6.shape[0] > 1 else np.zeros(6)
        raa9, _ = apply_feeddown_to_raa6(avg6, sem6, feeddown, sigmas)
        x_vals.append(float(glauber.b_to_npart(np.asarray([b], dtype=np.float64))[0]))
        state_values["1S"].append(float(raa9[0]))
        state_values["2S"].append(float(raa9[1]))
        state_values["3S"].append(float(raa9[5]))
        counts.append(len(sample))

    order = np.argsort(np.asarray(x_vals, dtype=np.float64))
    return {
        "x": np.asarray(x_vals, dtype=np.float64)[order],
        "1S": np.asarray(state_values["1S"], dtype=np.float64)[order],
        "2S": np.asarray(state_values["2S"], dtype=np.float64)[order],
        "3S": np.asarray(state_values["3S"], dtype=np.float64)[order],
        "counts": np.asarray(counts, dtype=np.int64)[order],
    }


def _build_recomputed_pt_series(
    *,
    system_key: str,
    source_label: str,
    y_abs_max: float,
    pt_edges: np.ndarray,
    logger: logging.Logger,
) -> Dict[str, np.ndarray]:
    obs = _read_lhc_obs(system_key, source_label, logger)
    glauber = GlauberInterpolator(load_canonical_glauber(system_key, logger))
    x_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    state_values: Dict[str, List[float]] = {"1S": [], "2S": [], "3S": []}
    counts: List[int] = []

    for pt_min, pt_max in zip(pt_edges[:-1], pt_edges[1:]):
        selector = np.asarray(
            [(abs(entry.y) <= y_abs_max) and (pt_min <= entry.pt < pt_max) for entry in obs],
            dtype=bool,
        )
        avg9, _ = weighted_avg_surv9(obs, glauber, selector, logger)
        state_values["1S"].append(float(avg9[0]))
        state_values["2S"].append(float(avg9[1]))
        state_values["3S"].append(float(avg9[5]))
        counts.append(int(selector.sum()))

    return {
        "x": np.asarray(x_centers, dtype=np.float64),
        "bin_edges": np.asarray(pt_edges, dtype=np.float64),
        "1S": np.asarray(state_values["1S"], dtype=np.float64),
        "2S": np.asarray(state_values["2S"], dtype=np.float64),
        "3S": np.asarray(state_values["3S"], dtype=np.float64),
        "counts": np.asarray(counts, dtype=np.int64),
    }


def _series_point_pairs(x: np.ndarray, y: np.ndarray) -> List[Tuple[float, float]]:
    return [(float(xx), float(yy)) for xx, yy in zip(x.tolist(), y.tolist())]


def _experimental_band_metrics(
    *,
    observable_id: str,
    state_label: str,
    reference_k3: object,
    reference_k4: object,
    alternative_k3: Dict[str, np.ndarray],
    alternative_k4: Dict[str, np.ndarray],
) -> Optional[dict]:
    from qtraj_analysis.reference_data import load_experimental_series

    target_state = f"Upsilon({state_label})"
    series = [
        item
        for item in load_experimental_series(observable_id)
        if item.experiment == "CMS" and item.state == target_state
    ]
    if not series:
        return None

    exp = series[0]
    if reference_k3.bin_edges is not None:
        ref_eval_k3 = _evaluate_step(reference_k3.bin_edges, reference_k3.center, exp.x)
        ref_eval_k4 = _evaluate_step(reference_k4.bin_edges, reference_k4.center, exp.x)
        alt_eval_k3 = _evaluate_step(alternative_k3["bin_edges"], alternative_k3[state_label], exp.x)
        alt_eval_k4 = _evaluate_step(alternative_k4["bin_edges"], alternative_k4[state_label], exp.x)
    else:
        ref_eval_k3 = np.interp(exp.x, reference_k3.x, reference_k3.center)
        ref_eval_k4 = np.interp(exp.x, reference_k4.x, reference_k4.center)
        alt_eval_k3 = np.interp(exp.x, alternative_k3["x"], alternative_k3[state_label])
        alt_eval_k4 = np.interp(exp.x, alternative_k4["x"], alternative_k4[state_label])

    ref_lower = np.minimum(ref_eval_k3, ref_eval_k4)
    ref_upper = np.maximum(ref_eval_k3, ref_eval_k4)
    alt_lower = np.minimum(alt_eval_k3, alt_eval_k4)
    alt_upper = np.maximum(alt_eval_k3, alt_eval_k4)

    ref_dist = _distance_to_band(exp.y, ref_lower, ref_upper)
    alt_dist = _distance_to_band(exp.y, alt_lower, alt_upper)

    return {
        "experiment": exp.experiment,
        "acceptance": exp.acceptance,
        "n_points": int(exp.x.shape[0]),
        "reference_band": {
            "mean_abs_distance_to_band": float(np.mean(ref_dist)),
            "max_abs_distance_to_band": float(np.max(ref_dist)),
            "points_inside_band": int(np.sum(ref_dist <= 1e-12)),
            "evaluated_points": _series_point_pairs(exp.x, ref_lower),
            "evaluated_upper": _series_point_pairs(exp.x, ref_upper),
        },
        "alternative_band": {
            "mean_abs_distance_to_band": float(np.mean(alt_dist)),
            "max_abs_distance_to_band": float(np.max(alt_dist)),
            "points_inside_band": int(np.sum(alt_dist <= 1e-12)),
            "evaluated_points": _series_point_pairs(exp.x, alt_lower),
            "evaluated_upper": _series_point_pairs(exp.x, alt_upper),
        },
        "experimental_points": _series_point_pairs(exp.x, exp.y),
    }


def validate_lhc_acceptance_scan(system_key: str, logger: logging.Logger) -> dict:
    if system_key not in {"pbpb2760", "pbpb5023"}:
        raise ValueError("LHC acceptance scan only supports pbpb2760 and pbpb5023")

    spec = get_observable_spec(f"{system_key}_raavsnpart")
    canonical_spec = get_canonical_glauber_spec(system_key)
    glauber_model = load_canonical_glauber(system_key, logger)
    glauber_report = {
        "canonical_input_base": canonical_spec.input_base,
        "canonical_notebook": canonical_spec.source_notebook,
        "bvsc_table": str(((_repo_root() / canonical_spec.input_base) / "glauber-data" / "bvscData.tsv").relative_to(_repo_root())),
        "nbin_table": str(((_repo_root() / canonical_spec.input_base) / "glauber-data" / "nbinvsbData.tsv").relative_to(_repo_root())),
        "notebook_source": spec.datafile_sources[0].path,
        "bvals_max_abs_diff": float(np.max(np.abs(glauber_model.bvals - np.asarray(canonical_spec.bvals, dtype=np.float64)))),
        "npart_max_abs_diff": float(np.max(np.abs(glauber_model.npart_vals - np.asarray(canonical_spec.npart_vals, dtype=np.float64)))),
        "note": "The production validation path uses glauber.py with the canonical notebook b->Npart arrays injected exactly.",
    }

    npart_observable = f"{system_key}_raavsnpart"
    pt_observable = f"{system_key}_raavspt"
    npart_reference = _theory_lookup(npart_observable)
    pt_reference = _theory_lookup(pt_observable)

    pt_edges = np.asarray(get_observable_spec(pt_observable).grid.bin_edges, dtype=np.float64)

    npart_pt_max = 30.0
    if system_key == "pbpb2760":
        npart_pt_max = 40.0

    npart_alt = {
        source_label: _build_recomputed_npart_series(
            system_key=system_key,
            source_label=source_label,
            y_abs_max=2.4,
            pt_max=npart_pt_max,
            logger=logger,
        )
        for source_label in ("k3", "k4")
    }
    pt_alt = {
        source_label: _build_recomputed_pt_series(
            system_key=system_key,
            source_label=source_label,
            y_abs_max=2.4,
            pt_edges=pt_edges,
            logger=logger,
        )
        for source_label in ("k3", "k4")
    }

    observables = []
    for observable_id, reference_lookup, alt_lookup, alt_note in (
        (
            npart_observable,
            npart_reference,
            npart_alt,
            rf"Alternative raw-data recomputation: $|y| < 2.4,\ p_T < {npart_pt_max:.0f}\ \mathrm{{GeV}}$",
        ),
        (
            pt_observable,
            pt_reference,
            pt_alt,
            r"Alternative raw-data recomputation: $0\mathrm{-}100\%$ centrality, $|y| < 2.4$",
        ),
    ):
        spec = get_observable_spec(observable_id)
        series_payload = []
        experiment_metrics = []
        for state_label in ("1S", "2S", "3S"):
            reference_k3 = reference_lookup.get(("k3", state_label))
            reference_k4 = reference_lookup.get(("k4", state_label))
            if reference_k3 is None or reference_k4 is None:
                continue

            alt_k3 = alt_lookup["k3"]
            alt_k4 = alt_lookup["k4"]
            reference_grid = reference_k3.center
            alternative_grid = alt_k3[state_label]
            max_abs_diff_k3 = float(np.max(np.abs(reference_grid - alternative_grid)))
            max_abs_diff_k4 = float(np.max(np.abs(reference_k4.center - alt_k4[state_label])))

            series_payload.append(
                {
                    "state": state_label,
                    "k3_reference_points": _series_point_pairs(reference_k3.x, reference_k3.center),
                    "k3_alternative_points": _series_point_pairs(alt_k3["x"], alt_k3[state_label]),
                    "k4_reference_points": _series_point_pairs(reference_k4.x, reference_k4.center),
                    "k4_alternative_points": _series_point_pairs(alt_k4["x"], alt_k4[state_label]),
                    "max_abs_diff_k3": max_abs_diff_k3,
                    "max_abs_diff_k4": max_abs_diff_k4,
                }
            )

            metrics = _experimental_band_metrics(
                observable_id=observable_id,
                state_label=state_label,
                reference_k3=reference_k3,
                reference_k4=reference_k4,
                alternative_k3=alt_k3,
                alternative_k4=alt_k4,
            )
            if metrics is not None:
                experiment_metrics.append({"state": state_label, **metrics})

        observables.append(
            {
                "observable_id": observable_id,
                "current_production_note": spec.theory_note,
                "alternative_note": alt_note,
                "series": series_payload,
                "cms_comparison": experiment_metrics,
            }
        )

    recommendation = (
        "For strict thesis/source parity, keep the current production export unchanged. "
        "For physics comparison to CMS acceptance, the alternative |y|<2.4 raw-data recomputation "
        "is the more meaningful diagnostic, but it should remain a validation path until it is explicitly promoted."
    )
    if system_key == "pbpb2760":
        recommendation = (
            "PbPb 2.76 TeV lacks a parity-locked double-ratio reference bundle. "
            "Use the CMS-matched raw-data recomputation as a diagnostic only, not as a shipped production replacement, "
            "until the inclusive and ratio recomputation conventions are fully signed off."
        )

    return {
        "system": system_key,
        "status": "info",
        "glauber_consistency": glauber_report,
        "recommendation": recommendation,
        "observables": observables,
    }


def validate_pbpb2760_double_ratio_parity(logger: logging.Logger) -> dict:
    feeddown = build_feeddown_matrix()
    sigmas = _primordial_sigmas()
    model = load_canonical_glauber("pbpb2760", logger)
    glauber = GlauberInterpolator(model)
    obs = _read_pbpb2760_obs("k3", logger)

    groups: Dict[float, list] = {}
    for entry in obs:
        groups.setdefault(round(entry.b, 6), []).append(entry)

    npart_ratio21: List[Tuple[float, float]] = []
    npart_ratio31: List[Tuple[float, float]] = []
    for b in sorted(groups):
        sample = groups[b]
        surv6 = np.vstack([entry.surv6 for entry in sample])
        qweights = np.asarray([entry.qweight for entry in sample], dtype=np.float64)
        avg6 = (surv6.T @ qweights) / qweights.sum()
        sem6 = np.std(surv6, axis=0, ddof=1) / np.sqrt(surv6.shape[0])
        raa9, _ = apply_feeddown_to_raa6(avg6, sem6, feeddown, sigmas)
        npart = float(glauber.b_to_npart(np.asarray([b], dtype=np.float64))[0])
        npart_ratio21.append((npart, float(raa9[1] / raa9[0])))
        npart_ratio31.append((npart, float(raa9[5] / raa9[0])))

    npart_ratio21.sort(key=lambda item: item[0])
    npart_ratio31.sort(key=lambda item: item[0])

    pt_edges = [(0.0, 5.0), (5.0, 12.0), (12.0, 30.0)]
    pt_ratio21: List[Tuple[float, float]] = []
    for ptmin, ptmax in pt_edges:
        selector = np.asarray([(entry.pt >= ptmin) and (entry.pt <= ptmax) for entry in obs], dtype=bool)
        avg9, _ = weighted_avg_surv9(obs, glauber, selector, logger)
        pt_ratio21.append((((ptmin + ptmax) / 2.0), float(avg9[1] / avg9[0])))

    ratio21_npart_diff = _max_abs_diff(_PBPB2760_RATIO21_NPART_ARCHIVE_K3, npart_ratio21)
    ratio31_npart_diff = _max_abs_diff(_PBPB2760_RATIO31_NPART_ARCHIVE_K3, npart_ratio31)
    ratio21_pt_diff = _max_abs_diff(_PBPB2760_RATIO21_PT_ARCHIVE_K3, pt_ratio21)

    return {
        "system": "pbpb2760",
        "status": "pass"
        if max(ratio21_npart_diff, ratio31_npart_diff, ratio21_pt_diff) <= 1e-6
        else "fail",
        "tolerance": 1e-6,
        "archive_reference": "inputs/qtraj_inputs/PbPb2760/archive/raaCalculator-trajectories.nb",
        "note": (
            "PbPb 2.76 TeV double-ratio production observables remain withheld until the "
            "raw-data Python recomputation matches the archived Mathematica notebook numerically."
        ),
        "observables": [
            {
                "observable_id": "pbpb2760_ratio21vsnpart",
                "source_label": "k3",
                "expected_points": _PBPB2760_RATIO21_NPART_ARCHIVE_K3,
                "computed_points": npart_ratio21,
                "max_abs_diff": ratio21_npart_diff,
            },
            {
                "observable_id": "pbpb2760_ratio31vsnpart",
                "source_label": "k3",
                "expected_points": _PBPB2760_RATIO31_NPART_ARCHIVE_K3,
                "computed_points": npart_ratio31,
                "max_abs_diff": ratio31_npart_diff,
            },
            {
                "observable_id": "pbpb2760_ratio21vspt",
                "source_label": "k3",
                "expected_points": _PBPB2760_RATIO21_PT_ARCHIVE_K3,
                "computed_points": pt_ratio21,
                "max_abs_diff": ratio21_pt_diff,
            },
        ],
        "blockers": [
            "The current qweight/raw-data recomputation for k3 does not reproduce the archived notebook ratios.",
            "No separate archived k4 ratio export was found locally, so the [3,4] theory envelope cannot be parity-locked yet.",
        ],
    }
