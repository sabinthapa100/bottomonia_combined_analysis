from __future__ import annotations

import csv
import io
import logging
import math
import re
import tarfile
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from qtraj_analysis.feeddown import (
    apply_feeddown_to_raa6,
    build_feeddown_matrix,
    solve_primordial_sigmas,
    split_hyperfine_6_to_9,
)
from qtraj_analysis.glauber import GlauberInterpolator, load_canonical_glauber
from qtraj_analysis.io import parse_records, read_whitespace_table
from qtraj_analysis.matching import build_observables
from qtraj_analysis.observable_registry import (
    format_source,
    get_observable_spec,
    list_observable_ids,
    resolve_source,
)
from qtraj_analysis.schema import ExperimentalObservableSpec, SourceRef, TheoryObservableSpec
from qtraj_analysis.stats import p_centrality, weighted_avg_surv9


STANDARD_CENTRALITY_LABELS: Tuple[str, ...] = (
    "0%",
    "0-5%",
    "5-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
)

_MATHEMATICA_NUMBER = r"-?(?:\d+\.\d*|\d*\.\d+|\d+)(?:`)?(?:\*\^[+-]?\d+|[eE][+-]?\d+)?"
_TOKEN_RE = re.compile(
    r"\s*("
    r"Around|"
    r"[{}\[\],=]|"
    + _MATHEMATICA_NUMBER
    + r"|"
    r"[A-Za-z][A-Za-z0-9_]*"
    r")"
)
_NB_GRID_RE = re.compile(
    r'GridBox\[\{(?P<body>.*?)\}\s*,\s*GridBoxAlignment',
    re.DOTALL,
)
_NB_ROW_RE = re.compile(
    r'\{"(?P<x>[^"]+?)"\s*,\s*InterpretationBox\[.*?Around\[(?P<y>'
    + _MATHEMATICA_NUMBER
    + r")\s*,\s*(?P<err>"
    + _MATHEMATICA_NUMBER
    + r")\]\]\}",
    re.DOTALL,
)


@dataclass(frozen=True)
class AroundValue:
    value: float
    error: float


@dataclass(frozen=True)
class TheoryBandSeries:
    observable_id: str
    series_label: str
    source_label: str
    source: str
    x: np.ndarray
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    bin_edges: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ExperimentalSeries:
    observable_id: str
    series_label: str
    experiment: str
    state: str
    observable_type: str
    acceptance: str
    source: str
    x: np.ndarray
    y: np.ndarray
    x_low: Optional[np.ndarray]
    x_high: Optional[np.ndarray]
    yerr_low: Optional[np.ndarray]
    yerr_high: Optional[np.ndarray]
    uncertainty_kind: Literal["total", "stat", "sys"] = "total"
    is_minbias: bool = False
    upper_limit: bool = False
    combined_state: bool = False
    note: Optional[str] = None


@dataclass(frozen=True)
class ObservableReferenceBundle:
    observable_id: str
    system: str
    energy_label: str
    observable_type: str
    acceptance: str
    theory_note: str
    category: str
    theory_series: Tuple[TheoryBandSeries, ...]
    experimental_series: Tuple[ExperimentalSeries, ...]
    theory_sources: Tuple[str, ...]
    datafile_sources: Tuple[str, ...]
    issues: Tuple[str, ...]
    minbias_experimental_series: Tuple[ExperimentalSeries, ...] = ()
    centrality_labels: Tuple[str, ...] = ()


_LOGGER = logging.getLogger("qtraj_analysis.reference_data")
_GENERATED_OBS_CACHE: dict[tuple[str, str], list] = {}
_SIGMAS_EXP = np.array([57.6, 19.0, 3.72, 13.69, 16.1, 6.8, 3.27, 12.0, 14.15], dtype=np.float64)
# Index into the length-9 RAA vector (post-feeddown) for each physical state.
_UPSILON_STATE_IDX: dict[str, int] = {"1S": 0, "2S": 1, "3S": 5}


def _clean_number(raw: str) -> float:
    return float(raw.replace("`", "").replace("*^", "e"))


class _MathematicaParser:
    def __init__(self, text: str):
        tokens = _TOKEN_RE.findall(text)
        self.tokens = [tok for tok in tokens if tok.strip()]
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def pop(self) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of Mathematica input")
        self.pos += 1
        return token

    def expect(self, token: str) -> None:
        actual = self.pop()
        if actual != token:
            raise ValueError(f"Expected token '{token}', got '{actual}'")

    def parse_assignment(self) -> Tuple[Optional[str], object]:
        name = None
        if self.peek() and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.peek() or ""):
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1] == "=":
                name = self.pop()
                self.expect("=")
        value = self.parse_value()
        return name, value

    def parse_value(self) -> object:
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of Mathematica input")
        if token == "{":
            return self.parse_list()
        if token == "Around":
            self.pop()
            self.expect("[")
            value = self.parse_number()
            self.expect(",")
            error = self.parse_number()
            self.expect("]")
            return AroundValue(value=value, error=error)
        if re.fullmatch(_MATHEMATICA_NUMBER, token):
            return self.parse_number()
        raise ValueError(f"Unsupported Mathematica token '{token}'")

    def parse_number(self) -> float:
        return _clean_number(self.pop())

    def parse_list(self) -> List[object]:
        self.expect("{")
        values: List[object] = []
        while True:
            if self.peek() == "}":
                self.pop()
                return values
            values.append(self.parse_value())
            if self.peek() == ",":
                self.pop()
                continue
            if self.peek() == "}":
                self.pop()
                return values
            raise ValueError(f"Unexpected token in list: '{self.peek()}'")


def parse_mathematica_assignment(ref: SourceRef | str) -> Tuple[Optional[str], object]:
    source = ref if isinstance(ref, SourceRef) else SourceRef(path=str(ref))
    text = resolve_source(source).read_text()
    parser = _MathematicaParser(text)
    return parser.parse_assignment()


def _is_pair(value: object) -> bool:
    return isinstance(value, list) and len(value) == 2


def _is_numeric_series(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(
        _is_pair(item) and isinstance(item[0], float) and isinstance(item[1], (float, AroundValue))
        for item in value
    )


def _is_band_triplet(value: object) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(_is_numeric_series(item) for item in value)


def _series_contains_around(value: object) -> bool:
    return _is_numeric_series(value) and any(isinstance(item[1], AroundValue) for item in value)


def _series_arrays(series: Sequence[Sequence[object]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_vals: List[float] = []
    centers: List[float] = []
    lowers: List[float] = []
    uppers: List[float] = []

    for x_raw, y_raw in series:
        x_vals.append(float(x_raw))
        if isinstance(y_raw, AroundValue):
            centers.append(float(y_raw.value))
            lowers.append(float(y_raw.value - y_raw.error))
            uppers.append(float(y_raw.value + y_raw.error))
        else:
            centers.append(float(y_raw))
            lowers.append(float(y_raw))
            uppers.append(float(y_raw))

    return (
        np.asarray(x_vals, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
        np.asarray(lowers, dtype=np.float64),
        np.asarray(uppers, dtype=np.float64),
    )


def _source_label(ref: SourceRef) -> str:
    source_path = ref.path.lower()
    for token in ("kappa5", "kappa4", "k4", "k3"):
        if token in source_path:
            return token
    stem = ref.path.split("/")[-1]
    return stem.rsplit(".", 1)[0]


def _default_series_labels(spec: TheoryObservableSpec, count: int) -> Tuple[str, ...]:
    if spec.observable_type.startswith("RAA_"):
        return ("1S", "2S", "3S")[:count]
    if "2S/1S" in spec.state:
        return ("2S/1S",)
    if "3S/1S" in spec.state:
        return ("3S/1S",)
    if "3S/2S" in spec.state:
        return ("3S/2S",)
    return tuple(part.strip() for part in spec.state.split(","))[:count]


def _select_theory_sources(
    spec: TheoryObservableSpec,
    source_labels: Optional[Sequence[str]] = None,
) -> Tuple[SourceRef, ...]:
    if not source_labels:
        return spec.mathematica_sources

    requested = {label.lower() for label in source_labels}
    selected = tuple(
        source for source in spec.mathematica_sources if _source_label(source).lower() in requested
    )
    if not selected:
        raise ValueError(
            f"No Mathematica sources matched {sorted(requested)} for observable '{spec.observable_id}'"
        )
    return selected


def _select_datafile_sources(
    spec: TheoryObservableSpec,
    source_labels: Optional[Sequence[str]] = None,
) -> Tuple[SourceRef, ...]:
    if not source_labels:
        return spec.datafile_sources

    requested = {label.lower() for label in source_labels}
    selected = tuple(
        source for source in spec.datafile_sources if _source_label(source).lower() in requested
    )
    if not selected:
        raise ValueError(
            f"No datafile sources matched {sorted(requested)} for observable '{spec.observable_id}'"
        )
    return selected


def _read_generated_observables(source: SourceRef) -> list:
    cache_key = source.path
    cached = _GENERATED_OBS_CACHE.get((cache_key, "obs"))
    if cached is not None:
        return cached

    table = read_whitespace_table(str(resolve_source(source)), _LOGGER)
    records = parse_records(table, _LOGGER)
    obs = build_observables(records, _LOGGER)
    _GENERATED_OBS_CACHE[(cache_key, "obs")] = obs
    return obs


def _load_generated_pbpb2760_ratio_series(
    spec: TheoryObservableSpec,
    source_labels: Optional[Sequence[str]] = None,
) -> Tuple[TheoryBandSeries, ...]:
    ratio_index = {
        "pbpb2760_ratio21vsnpart": (1, 0),
        "pbpb2760_ratio31vsnpart": (5, 0),
        "pbpb2760_ratio21vspt": (1, 0),
    }
    if spec.observable_id not in ratio_index:
        return ()

    num_idx, den_idx = ratio_index[spec.observable_id]
    selected_sources = _select_datafile_sources(spec, source_labels)
    feeddown = build_feeddown_matrix()
    sigmas = solve_primordial_sigmas(feeddown, _SIGMAS_EXP)
    glauber = GlauberInterpolator(load_canonical_glauber("pbpb2760", _LOGGER))
    label = _default_series_labels(spec, 1)[0]
    out: List[TheoryBandSeries] = []

    for source in selected_sources:
        source_label = _source_label(source)
        source_name = f"generated_from::{format_source(source)}"
        obs = _read_generated_observables(source)

        if spec.observable_type == "double_ratio_vs_npart":
            x_vals: List[float] = []
            ratio_vals: List[float] = []
            groups: dict[float, list] = {}
            for entry in obs:
                if abs(entry.y) <= 2.4 and entry.pt < 40.0:
                    groups.setdefault(round(entry.b, 6), []).append(entry)

            for b in sorted(groups):
                sample = groups[b]
                surv6 = np.vstack([entry.surv6 for entry in sample])
                qweights = np.asarray([entry.qweight for entry in sample], dtype=np.float64)
                avg6 = (surv6.T @ qweights) / qweights.sum()
                sem6 = (
                    np.std(surv6, axis=0, ddof=1) / np.sqrt(surv6.shape[0])
                    if surv6.shape[0] > 1
                    else np.zeros(6, dtype=np.float64)
                )
                raa9, _ = apply_feeddown_to_raa6(avg6, sem6, feeddown, sigmas)
                x_vals.append(float(glauber.b_to_npart(np.asarray([b], dtype=np.float64))[0]))
                ratio_vals.append(float(raa9[num_idx] / raa9[den_idx]))

            x = np.asarray(x_vals, dtype=np.float64)
            center = np.asarray(ratio_vals, dtype=np.float64)
            order = np.argsort(x)
            x = x[order]
            center = center[order]
            out.append(
                TheoryBandSeries(
                    observable_id=spec.observable_id,
                    series_label=label,
                    source_label=source_label,
                    source=source_name,
                    x=x,
                    center=center,
                    lower=center.copy(),
                    upper=center.copy(),
                    bin_edges=None,
                )
            )
            continue

        if spec.observable_type == "double_ratio_vs_pt":
            if spec.grid.bin_edges is None:
                raise ValueError(
                    f"Generated pt-ratio observable '{spec.observable_id}' requires explicit bin edges in registry."
                )
            pt_edges = np.asarray(spec.grid.bin_edges, dtype=np.float64)
            x = 0.5 * (pt_edges[:-1] + pt_edges[1:])
            ratio_vals: List[float] = []
            for pt_min, pt_max in zip(pt_edges[:-1], pt_edges[1:]):
                selector = np.asarray(
                    [(abs(entry.y) <= 2.4) and (pt_min <= entry.pt < pt_max) for entry in obs],
                    dtype=bool,
                )
                if not np.any(selector):
                    ratio_vals.append(float("nan"))
                    continue
                avg9, _ = weighted_avg_surv9(obs, glauber, selector, _LOGGER)
                ratio_vals.append(float(avg9[num_idx] / avg9[den_idx]))
            center = np.asarray(ratio_vals, dtype=np.float64)
            out.append(
                TheoryBandSeries(
                    observable_id=spec.observable_id,
                    series_label=label,
                    source_label=source_label,
                    source=source_name,
                    x=x,
                    center=center,
                    lower=center.copy(),
                    upper=center.copy(),
                    bin_edges=pt_edges,
                )
            )
            continue

        raise ValueError(f"Unsupported generated observable type: {spec.observable_type}")

    return tuple(out)


def _load_generated_pbpb2760_raa_series(
    spec: TheoryObservableSpec,
    source_labels: Optional[Sequence[str]] = None,
) -> Tuple[TheoryBandSeries, ...]:
    _HANDLED_IDS = frozenset({
        "pbpb2760_raavsnpart",
        "pbpb2760_raavspt",
        "pbpb2760_raavsy",
    })
    if spec.observable_id not in _HANDLED_IDS:
        return ()

    selected_sources = _select_datafile_sources(spec, source_labels)
    feeddown = build_feeddown_matrix()
    sigmas = solve_primordial_sigmas(feeddown, _SIGMAS_EXP)
    glauber = GlauberInterpolator(load_canonical_glauber("pbpb2760", _LOGGER))

    requested_states = [
        (s.strip(), _UPSILON_STATE_IDX[s.strip()])
        for s in spec.state.split(",")
        if s.strip() in _UPSILON_STATE_IDX
    ]

    out: List[TheoryBandSeries] = []

    for source in selected_sources:
        source_label = _source_label(source)
        source_name = f"generated_from::{format_source(source)}"
        obs = _read_generated_observables(source)

        b_arr = np.asarray([e.b for e in obs], dtype=np.float64)
        pt_arr = np.asarray([e.pt for e in obs], dtype=np.float64)
        y_arr = np.asarray([e.y for e in obs], dtype=np.float64)
        q_arr = np.asarray([e.qweight for e in obs], dtype=np.float64)
        surv6_arr = np.vstack([e.surv6 for e in obs])

        if spec.observable_type == "RAA_vs_npart":
            base_mask = (np.abs(y_arr) <= 2.4) & (pt_arr < 40.0)
            unique_b = np.unique(np.round(b_arr[base_mask], 6))
            x_vals: List[float] = []
            raa9_list: List[np.ndarray] = []

            for b in unique_b:
                mask = base_mask & (np.abs(b_arr - b) < 5e-5)
                S = surv6_arr[mask]
                q = q_arr[mask]
                avg6 = (S.T @ q) / q.sum()
                sem6 = (
                    np.std(S, axis=0, ddof=1) / np.sqrt(S.shape[0])
                    if S.shape[0] > 1
                    else np.zeros(6, dtype=np.float64)
                )
                raa9, _ = apply_feeddown_to_raa6(avg6, sem6, feeddown, sigmas)
                x_vals.append(float(glauber.b_to_npart(np.asarray([b], dtype=np.float64))[0]))
                raa9_list.append(raa9)

            x = np.asarray(x_vals, dtype=np.float64)
            raa9_arr = np.vstack(raa9_list)
            order = np.argsort(x)
            x = x[order]
            raa9_arr = raa9_arr[order]

            for state_lbl, state_idx in requested_states:
                center = raa9_arr[:, state_idx].copy()
                out.append(
                    TheoryBandSeries(
                        observable_id=spec.observable_id,
                        series_label=state_lbl,
                        source_label=source_label,
                        source=source_name,
                        x=x,
                        center=center,
                        lower=center.copy(),
                        upper=center.copy(),
                        bin_edges=None,
                    )
                )
            continue

        if spec.observable_type == "RAA_vs_pt":
            if spec.grid.bin_edges is None:
                raise ValueError(
                    f"Generated obs '{spec.observable_id}' requires explicit bin_edges in registry."
                )
            pt_edges = np.asarray(spec.grid.bin_edges, dtype=np.float64)
            x = np.asarray(spec.grid.values, dtype=np.float64)

            base_mask = np.abs(y_arr) <= 2.4
            raa9_by_bin: List[np.ndarray] = []
            for pt_min, pt_max in zip(pt_edges[:-1], pt_edges[1:]):
                selector = base_mask & (pt_arr >= pt_min) & (pt_arr < pt_max)
                if not np.any(selector):
                    raa9_by_bin.append(np.full(9, np.nan, dtype=np.float64))
                    continue
                avg9, _ = weighted_avg_surv9(obs, glauber, selector, _LOGGER)
                num = feeddown @ (sigmas * avg9)
                den = feeddown @ sigmas
                raa9 = np.divide(num, den, out=np.full_like(num, np.nan), where=(den != 0))
                raa9_by_bin.append(raa9)

            raa9_arr = np.vstack(raa9_by_bin)
            for state_lbl, state_idx in requested_states:
                center = raa9_arr[:, state_idx].copy()
                out.append(
                    TheoryBandSeries(
                        observable_id=spec.observable_id,
                        series_label=state_lbl,
                        source_label=source_label,
                        source=source_name,
                        x=x,
                        center=center,
                        lower=center.copy(),
                        upper=center.copy(),
                        bin_edges=pt_edges,
                    )
                )
            continue

        if spec.observable_type == "RAA_vs_y":
            if spec.grid.bin_edges is None:
                raise ValueError(
                    f"Generated obs '{spec.observable_id}' requires explicit bin_edges in registry."
                )
            y_edges = np.asarray(spec.grid.bin_edges, dtype=np.float64)
            x = np.asarray(spec.grid.values, dtype=np.float64)

            base_mask = pt_arr < 40.0
            raa9_by_bin_y: List[np.ndarray] = []
            for y_min, y_max in zip(y_edges[:-1], y_edges[1:]):
                selector = base_mask & (y_arr >= y_min) & (y_arr < y_max)
                if not np.any(selector):
                    raa9_by_bin_y.append(np.full(9, np.nan, dtype=np.float64))
                    continue
                avg9, _ = weighted_avg_surv9(obs, glauber, selector, _LOGGER)
                num = feeddown @ (sigmas * avg9)
                den = feeddown @ sigmas
                raa9 = np.divide(num, den, out=np.full_like(num, np.nan), where=(den != 0))
                raa9_by_bin_y.append(raa9)

            raa9_arr = np.vstack(raa9_by_bin_y)
            # x has 12 centers; bin_edges has 13 edges → _step_xy extends automatically.
            for state_lbl, state_idx in requested_states:
                center = raa9_arr[:, state_idx].copy()
                out.append(
                    TheoryBandSeries(
                        observable_id=spec.observable_id,
                        series_label=state_lbl,
                        source_label=source_label,
                        source=source_name,
                        x=x,
                        center=center,
                        lower=center.copy(),
                        upper=center.copy(),
                        bin_edges=y_edges,
                    )
                )
            continue

        raise ValueError(
            f"Unsupported generated observable type for PbPb2760: {spec.observable_type!r}"
        )

    return tuple(out)


def _load_generated_pbpb5023_raa_series(
    spec: TheoryObservableSpec,
    source_labels: Optional[Sequence[str]] = None,
) -> Tuple[TheoryBandSeries, ...]:
    _HANDLED_IDS = frozenset({
        "pbpb5023_raavsnpart",
        "pbpb5023_raavspt",
        "pbpb5023_raavsy",
    })
    if spec.observable_id not in _HANDLED_IDS:
        return ()

    selected_sources = _select_datafile_sources(spec, source_labels)
    feeddown = build_feeddown_matrix()
    sigmas = solve_primordial_sigmas(feeddown, _SIGMAS_EXP)
    glauber = GlauberInterpolator(load_canonical_glauber("pbpb5023", _LOGGER))

    requested_states = [
        (s.strip(), _UPSILON_STATE_IDX[s.strip()])
        for s in spec.state.split(",")
        if s.strip() in _UPSILON_STATE_IDX
    ]

    out: List[TheoryBandSeries] = []

    for source in selected_sources:
        source_label = _source_label(source)
        source_name = f"generated_from::{format_source(source)}"
        obs = _read_generated_observables(source)

        b_arr = np.asarray([e.b for e in obs], dtype=np.float64)
        pt_arr = np.asarray([e.pt for e in obs], dtype=np.float64)
        y_arr = np.asarray([e.y for e in obs], dtype=np.float64)
        q_arr = np.asarray([e.qweight for e in obs], dtype=np.float64)
        surv6_arr = np.vstack([e.surv6 for e in obs])

        if spec.observable_type == "RAA_vs_npart":
            # No y or pt cut: Mathematica selectB filters only by b-value (no y/pt selection)
            base_mask = np.ones(len(obs), dtype=bool)
            unique_b = np.unique(np.round(b_arr[base_mask], 6))
            x_vals: List[float] = []
            raa9_list: List[np.ndarray] = []

            for b in unique_b:
                mask = base_mask & (np.abs(b_arr - b) < 5e-5)
                S = surv6_arr[mask]
                q = q_arr[mask]
                avg6 = (S.T @ q) / q.sum()
                sem6 = (
                    np.std(S, axis=0, ddof=1) / np.sqrt(S.shape[0])
                    if S.shape[0] > 1
                    else np.zeros(6, dtype=np.float64)
                )
                raa9, _ = apply_feeddown_to_raa6(avg6, sem6, feeddown, sigmas)
                x_vals.append(float(glauber.b_to_npart(np.asarray([b], dtype=np.float64))[0]))
                raa9_list.append(raa9)

            x = np.asarray(x_vals, dtype=np.float64)
            raa9_arr = np.vstack(raa9_list)
            order = np.argsort(x)
            x = x[order]
            raa9_arr = raa9_arr[order]

            for state_lbl, state_idx in requested_states:
                center = raa9_arr[:, state_idx].copy()
                out.append(
                    TheoryBandSeries(
                        observable_id=spec.observable_id,
                        series_label=state_lbl,
                        source_label=source_label,
                        source=source_name,
                        x=x,
                        center=center,
                        lower=center.copy(),
                        upper=center.copy(),
                        bin_edges=None,
                    )
                )
            continue

        if spec.observable_type == "RAA_vs_pt":
            if spec.grid.bin_edges is None:
                raise ValueError(
                    f"Generated obs '{spec.observable_id}' requires explicit bin_edges in registry."
                )
            pt_edges = np.asarray(spec.grid.bin_edges, dtype=np.float64)
            x = np.asarray(spec.grid.values, dtype=np.float64)

            base_mask = np.abs(y_arr) <= 2.4
            raa9_by_bin: List[np.ndarray] = []
            for pt_min, pt_max in zip(pt_edges[:-1], pt_edges[1:]):
                selector = base_mask & (pt_arr >= pt_min) & (pt_arr < pt_max)
                if not np.any(selector):
                    raa9_by_bin.append(np.full(9, np.nan, dtype=np.float64))
                    continue
                avg9, _ = weighted_avg_surv9(obs, glauber, selector, _LOGGER)
                num = feeddown @ (sigmas * avg9)
                den = feeddown @ sigmas
                raa9 = np.divide(num, den, out=np.full_like(num, np.nan), where=(den != 0))
                raa9_by_bin.append(raa9)

            raa9_arr = np.vstack(raa9_by_bin)
            for state_lbl, state_idx in requested_states:
                center = raa9_arr[:, state_idx].copy()
                out.append(
                    TheoryBandSeries(
                        observable_id=spec.observable_id,
                        series_label=state_lbl,
                        source_label=source_label,
                        source=source_name,
                        x=x,
                        center=center,
                        lower=center.copy(),
                        upper=center.copy(),
                        bin_edges=pt_edges,
                    )
                )
            continue

        if spec.observable_type == "RAA_vs_y":
            if spec.grid.bin_edges is None:
                raise ValueError(
                    f"Generated obs '{spec.observable_id}' requires explicit bin_edges in registry."
                )
            # x values are bin centers; bin edges are center ± 0.5 to match Mathematica
            # (Mathematica: binwidth=1, bins centered at integer y values)
            y_centers = np.asarray(spec.grid.values, dtype=np.float64)
            x = y_centers

            base_mask = pt_arr < 30.0
            raa9_by_bin_y: List[np.ndarray] = []
            for yc in y_centers:
                # Fold |y|: merge events at +|yc| and -|yc| to match Mathematica's
                # symmetric PbPb computation (same nucleus on both sides → RAA(y) = RAA(-y))
                yabs = float(abs(yc))
                if yabs < 1e-9:
                    selector = base_mask & (np.abs(y_arr) < 0.5)
                else:
                    selector = base_mask & (np.abs(y_arr) >= yabs - 0.5) & (np.abs(y_arr) < yabs + 0.5)
                if not np.any(selector):
                    raa9_by_bin_y.append(np.full(9, np.nan, dtype=np.float64))
                    continue
                avg9, _ = weighted_avg_surv9(obs, glauber, selector, _LOGGER)
                num = feeddown @ (sigmas * avg9)
                den = feeddown @ sigmas
                raa9 = np.divide(num, den, out=np.full_like(num, np.nan), where=(den != 0))
                raa9_by_bin_y.append(raa9)

            raa9_arr = np.vstack(raa9_by_bin_y)
            # pbpb5023_raavsy uses bin_edges for x (13 values) with Mathematica step style.
            # The 13th x-position is the closing edge; append a duplicate last value.
            n_output = len(x)
            n_computed = len(raa9_arr)
            for state_lbl, state_idx in requested_states:
                if n_output == n_computed + 1:
                    center = np.concatenate(
                        [raa9_arr[:, state_idx], [raa9_arr[-1, state_idx]]]
                    )
                else:
                    center = raa9_arr[:, state_idx].copy()
                out.append(
                    TheoryBandSeries(
                        observable_id=spec.observable_id,
                        series_label=state_lbl,
                        source_label=source_label,
                        source=source_name,
                        x=x,
                        center=center,
                        lower=center.copy(),
                        upper=center.copy(),
                        bin_edges=None,
                    )
                )
            continue

        raise ValueError(
            f"Unsupported generated observable type for PbPb5023: {spec.observable_type!r}"
        )

    return tuple(out)


def _load_generated_pbpb5023_ratio_series(
    spec: TheoryObservableSpec,
    source_labels: Optional[Sequence[str]] = None,
) -> Tuple[TheoryBandSeries, ...]:
    ratio_index = {
        "pbpb5023_ratio21vsnpart": (1, 0),
        "pbpb5023_ratio31vsnpart": (5, 0),
        "pbpb5023_ratio21vspt": (1, 0),
        "pbpb5023_ratio32vspt": (5, 1),
    }
    if spec.observable_id not in ratio_index:
        return ()

    num_idx, den_idx = ratio_index[spec.observable_id]
    selected_sources = _select_datafile_sources(spec, source_labels)
    feeddown = build_feeddown_matrix()
    sigmas = solve_primordial_sigmas(feeddown, _SIGMAS_EXP)
    glauber = GlauberInterpolator(load_canonical_glauber("pbpb5023", _LOGGER))
    label = _default_series_labels(spec, 1)[0]
    out: List[TheoryBandSeries] = []

    for source in selected_sources:
        source_label = _source_label(source)
        source_name = f"generated_from::{format_source(source)}"
        obs = _read_generated_observables(source)

        if spec.observable_type == "double_ratio_vs_npart":
            groups: dict[float, list] = {}
            for entry in obs:
                # No y or pt cut: Mathematica selectB filters only by b-value
                groups.setdefault(round(entry.b, 6), []).append(entry)

            x_vals: List[float] = []
            ratio_vals: List[float] = []
            for b in sorted(groups):
                sample = groups[b]
                surv6 = np.vstack([entry.surv6 for entry in sample])
                qweights = np.asarray([entry.qweight for entry in sample], dtype=np.float64)
                avg6 = (surv6.T @ qweights) / qweights.sum()
                sem6 = (
                    np.std(surv6, axis=0, ddof=1) / np.sqrt(surv6.shape[0])
                    if surv6.shape[0] > 1
                    else np.zeros(6, dtype=np.float64)
                )
                raa9, _ = apply_feeddown_to_raa6(avg6, sem6, feeddown, sigmas)
                x_vals.append(float(glauber.b_to_npart(np.asarray([b], dtype=np.float64))[0]))
                ratio_vals.append(float(raa9[num_idx] / raa9[den_idx]))

            x = np.asarray(x_vals, dtype=np.float64)
            center = np.asarray(ratio_vals, dtype=np.float64)
            order = np.argsort(x)
            x = x[order]
            center = center[order]
            out.append(
                TheoryBandSeries(
                    observable_id=spec.observable_id,
                    series_label=label,
                    source_label=source_label,
                    source=source_name,
                    x=x,
                    center=center,
                    lower=center.copy(),
                    upper=center.copy(),
                    bin_edges=None,
                )
            )
            continue

        if spec.observable_type == "double_ratio_vs_pt":
            if spec.grid.bin_edges is None:
                raise ValueError(
                    f"Generated pt-ratio observable '{spec.observable_id}' requires explicit bin edges in registry."
                )
            pt_edges = np.asarray(spec.grid.bin_edges, dtype=np.float64)
            # x uses all bin edges (N+1 values): Mathematica outputs step-function format
            # {pt_left, val}, ..., {pt_right, val_last} where val_last duplicates the last bin
            x = pt_edges
            ratio_vals: List[float] = []
            for pt_min, pt_max in zip(pt_edges[:-1], pt_edges[1:]):
                selector = np.asarray(
                    [(pt_min <= entry.pt < pt_max) for entry in obs],
                    dtype=bool,
                )
                if not np.any(selector):
                    ratio_vals.append(float("nan"))
                    continue
                avg9, _ = weighted_avg_surv9(obs, glauber, selector, _LOGGER)
                # Apply feeddown (consistent with raavspt and Mathematica's computeRAAwithFeeddown)
                num_fd = feeddown @ (sigmas * avg9)
                den_fd = feeddown @ sigmas
                raa9 = np.divide(num_fd, den_fd, where=(den_fd != 0), out=np.zeros_like(num_fd))
                ratio_vals.append(float(raa9[num_idx] / raa9[den_idx]))
            # Duplicate last bin value to produce N+1 output matching Mathematica step-function format
            ratio_vals_extended = ratio_vals + [ratio_vals[-1]]
            center = np.asarray(ratio_vals_extended, dtype=np.float64)
            out.append(
                TheoryBandSeries(
                    observable_id=spec.observable_id,
                    series_label=label,
                    source_label=source_label,
                    source=source_name,
                    x=x,
                    center=center,
                    lower=center.copy(),
                    upper=center.copy(),
                    bin_edges=pt_edges,
                )
            )
            continue

        raise ValueError(f"Unsupported generated observable type: {spec.observable_type}")

    return tuple(out)


def _load_generated_auau200_raa_series(
    spec: TheoryObservableSpec,
    source_labels: Optional[Sequence[str]] = None,
) -> Tuple[TheoryBandSeries, ...]:
    """AuAu 200 GeV RAA observables computed directly from trajectory datafiles.

    Implements the HNM pipeline (qweight-weighted survival → feeddown → RAA) for
    three observable types without relying on pre-exported Mathematica values.
    """
    _HANDLED_IDS = frozenset({
        "auau200_raavsnpart",
        "auau200_raavspt",
        "auau200_raavsy",
    })
    if spec.observable_id not in _HANDLED_IDS:
        return ()

    selected_sources = _select_datafile_sources(spec, source_labels)
    feeddown = build_feeddown_matrix()
    sigmas = solve_primordial_sigmas(feeddown, _SIGMAS_EXP)
    glauber = GlauberInterpolator(load_canonical_glauber("auau200", _LOGGER))

    # States to output, derived from spec.state (e.g. "1S,2S,3S" or "1S,2S").
    requested_states = [
        (s.strip(), _UPSILON_STATE_IDX[s.strip()])
        for s in spec.state.split(",")
        if s.strip() in _UPSILON_STATE_IDX
    ]

    out: List[TheoryBandSeries] = []

    for source in selected_sources:
        source_label = _source_label(source)
        source_name = f"generated_from::{format_source(source)}"
        obs = _read_generated_observables(source)

        # Build flat numpy arrays for vectorised operations across all trajectories.
        b_arr = np.asarray([e.b for e in obs], dtype=np.float64)
        pt_arr = np.asarray([e.pt for e in obs], dtype=np.float64)
        y_arr = np.asarray([e.y for e in obs], dtype=np.float64)
        q_arr = np.asarray([e.qweight for e in obs], dtype=np.float64)
        surv6_arr = np.vstack([e.surv6 for e in obs])  # (n, 6)

        if spec.observable_type == "RAA_vs_npart":
            # STAR acceptance: |y| ≤ 1, pT < 10 GeV.  No centrality cut — Npart IS centrality.
            base_mask = (np.abs(y_arr) <= 1.0) & (pt_arr < 10.0)

            unique_b = np.unique(np.round(b_arr[base_mask], 6))
            x_vals: List[float] = []
            raa9_list: List[np.ndarray] = []

            for b in unique_b:
                mask = base_mask & (np.abs(b_arr - b) < 5e-5)
                S = surv6_arr[mask]
                q = q_arr[mask]
                avg6 = (S.T @ q) / q.sum()
                sem6 = (
                    np.std(S, axis=0, ddof=1) / np.sqrt(S.shape[0])
                    if S.shape[0] > 1
                    else np.zeros(6, dtype=np.float64)
                )
                raa9, _ = apply_feeddown_to_raa6(avg6, sem6, feeddown, sigmas)
                x_vals.append(
                    float(glauber.b_to_npart(np.asarray([b], dtype=np.float64))[0])
                )
                raa9_list.append(raa9)

            x = np.asarray(x_vals, dtype=np.float64)
            raa9_arr = np.vstack(raa9_list)  # (nb, 9)
            order = np.argsort(x)
            x = x[order]
            raa9_arr = raa9_arr[order]

            for state_lbl, state_idx in requested_states:
                center = raa9_arr[:, state_idx].copy()
                out.append(
                    TheoryBandSeries(
                        observable_id=spec.observable_id,
                        series_label=state_lbl,
                        source_label=source_label,
                        source=source_name,
                        x=x,
                        center=center,
                        lower=center.copy(),
                        upper=center.copy(),
                        bin_edges=None,
                    )
                )
            continue

        if spec.observable_type == "RAA_vs_pt":
            if spec.grid.bin_edges is None:
                raise ValueError(
                    f"Generated obs '{spec.observable_id}' requires explicit bin_edges in registry."
                )
            pt_edges = np.asarray(spec.grid.bin_edges, dtype=np.float64)
            x = np.asarray(spec.grid.values, dtype=np.float64)  # upper_edges_without_origin

            # 0-60% centrality (STAR acceptance) with p_centrality(c) weighting —
            # this reproduces the Mathematica pFunction-weighted centrality average.
            c_arr = glauber.b_to_c(b_arr)
            base_mask = (np.abs(y_arr) <= 1.0) & (c_arr <= 0.6)

            raa9_by_bin: List[np.ndarray] = []
            for pt_min, pt_max in zip(pt_edges[:-1], pt_edges[1:]):
                mask = base_mask & (pt_arr >= pt_min) & (pt_arr < pt_max)
                if not np.any(mask):
                    raa9_by_bin.append(np.full(9, np.nan, dtype=np.float64))
                    continue
                S = surv6_arr[mask]
                q = q_arr[mask]
                c_sel = c_arr[mask]
                w = p_centrality(c_sel)          # exp(-c/0.25) / Z
                wtm = float(np.mean(w))
                X9 = np.vstack([split_hyperfine_6_to_9(s) for s in S])  # (n, 9)
                X9w = (w[:, None] / wtm) * X9
                avg9 = (X9w.T @ q) / q.sum()
                num = feeddown @ (sigmas * avg9)
                den = feeddown @ sigmas
                raa9 = np.divide(num, den, out=np.full_like(num, np.nan), where=(den != 0))
                raa9_by_bin.append(raa9)

            raa9_arr = np.vstack(raa9_by_bin)  # (npt, 9)
            for state_lbl, state_idx in requested_states:
                center = raa9_arr[:, state_idx].copy()
                out.append(
                    TheoryBandSeries(
                        observable_id=spec.observable_id,
                        series_label=state_lbl,
                        source_label=source_label,
                        source=source_name,
                        x=x,
                        center=center,
                        lower=center.copy(),
                        upper=center.copy(),
                        bin_edges=pt_edges,
                    )
                )
            continue

        if spec.observable_type == "RAA_vs_y":
            if spec.grid.bin_edges is None:
                raise ValueError(
                    f"Generated obs '{spec.observable_id}' requires explicit bin_edges in registry."
                )
            y_edges = np.asarray(spec.grid.bin_edges, dtype=np.float64)
            x = np.asarray(spec.grid.values, dtype=np.float64)  # bin centers

            # 0-60% centrality, pT < 10 GeV, p_centrality weighted.
            # AuAu is y-symmetric: each bin [y_min,y_max] is symmetrised with its
            # mirror [-y_max,-y_min] before returning, exactly as in the Mathematica code.
            c_arr = glauber.b_to_c(b_arr)
            base_mask = (pt_arr < 10.0) & (c_arr <= 0.6)

            def _raa9_for_ymask(ymask: np.ndarray) -> np.ndarray:
                """p_centrality-weighted 9-state RAA (with feeddown) for a y-selection."""
                mask = base_mask & ymask
                if not np.any(mask):
                    return np.full(9, np.nan, dtype=np.float64)
                S = surv6_arr[mask]
                q = q_arr[mask]
                c_sel = c_arr[mask]
                w = p_centrality(c_sel)
                wtm = float(np.mean(w))
                X9 = np.vstack([split_hyperfine_6_to_9(s) for s in S])  # (n, 9)
                X9w = (w[:, None] / wtm) * X9
                avg9 = (X9w.T @ q) / q.sum()
                num = feeddown @ (sigmas * avg9)
                den = feeddown @ sigmas
                return np.divide(num, den, out=np.full_like(num, np.nan), where=(den != 0))

            raa9_by_bin_y: List[np.ndarray] = []
            y_bin_pairs = list(zip(y_edges[:-1], y_edges[1:]))
            for y_min, y_max in y_bin_pairs:
                # Forward bin
                raa_fwd = _raa9_for_ymask((y_arr >= y_min) & (y_arr < y_max))
                # Mirror bin (symmetric counterpart)
                raa_bwd = _raa9_for_ymask((y_arr >= -y_max) & (y_arr < -y_min))
                # Symmetrise: average forward and backward halves
                raa9_sym = np.where(
                    np.isnan(raa_fwd) | np.isnan(raa_bwd),
                    np.where(np.isnan(raa_fwd), raa_bwd, raa_fwd),
                    0.5 * (raa_fwd + raa_bwd),
                )
                raa9_by_bin_y.append(raa9_sym)

            raa9_arr = np.vstack(raa9_by_bin_y)  # (ny, 9)
            for state_lbl, state_idx in requested_states:
                center = raa9_arr[:, state_idx].copy()
                out.append(
                    TheoryBandSeries(
                        observable_id=spec.observable_id,
                        series_label=state_lbl,
                        source_label=source_label,
                        source=source_name,
                        x=x,
                        center=center,
                        lower=center.copy(),
                        upper=center.copy(),
                        bin_edges=y_edges,
                    )
                )
            continue

        raise ValueError(
            f"Unsupported generated observable type for AuAu200: {spec.observable_type!r}"
        )

    return tuple(out)


def _load_generated_theory_series(
    spec: TheoryObservableSpec,
    *,
    source_labels: Optional[Sequence[str]] = None,
) -> Tuple[TheoryBandSeries, ...]:
    if spec.observable_id.startswith("pbpb2760_ratio"):
        _LOGGER.info(
            "[SOURCE=GENERATED] %s -> _load_generated_pbpb2760_ratio_series()",
            spec.observable_id,
        )
        return _load_generated_pbpb2760_ratio_series(spec, source_labels=source_labels)
    if spec.observable_id.startswith("pbpb2760_raa"):
        _LOGGER.info(
            "[SOURCE=GENERATED] %s -> _load_generated_pbpb2760_raa_series()",
            spec.observable_id,
        )
        return _load_generated_pbpb2760_raa_series(spec, source_labels=source_labels)
    if spec.observable_id.startswith("pbpb5023_raa"):
        _LOGGER.info(
            "[SOURCE=GENERATED] %s -> _load_generated_pbpb5023_raa_series()",
            spec.observable_id,
        )
        return _load_generated_pbpb5023_raa_series(spec, source_labels=source_labels)
    if spec.observable_id.startswith("pbpb5023_ratio"):
        _LOGGER.info(
            "[SOURCE=GENERATED] %s -> _load_generated_pbpb5023_ratio_series()",
            spec.observable_id,
        )
        return _load_generated_pbpb5023_ratio_series(spec, source_labels=source_labels)
    if spec.observable_id.startswith("auau200_raa"):
        _LOGGER.info(
            "[SOURCE=GENERATED] %s -> _load_generated_auau200_raa_series()",
            spec.observable_id,
        )
        return _load_generated_auau200_raa_series(spec, source_labels=source_labels)
    return ()


def load_theory_series(
    observable_id: str,
    *,
    source_labels: Optional[Sequence[str]] = None,
) -> Tuple[TheoryBandSeries, ...]:
    spec = get_observable_spec(observable_id)
    if not spec.mathematica_sources:
        _LOGGER.info(
            "[SOURCE=GENERATED] %s: mathematica_sources=() -> trying generated path",
            observable_id,
        )
        generated = _load_generated_theory_series(spec, source_labels=source_labels)
        if generated:
            return generated
        raise ValueError(
            f"Observable '{observable_id}' defines no Mathematica sources and has no generated theory implementation."
        )

    _LOGGER.info(
        "[SOURCE=MATHEMATICA] %s: loading from %d .m source file(s): %s",
        observable_id,
        len(spec.mathematica_sources),
        [str(s)[:80] for s in spec.mathematica_sources],
    )
    theory_series: List[TheoryBandSeries] = []
    bin_edges = None if spec.grid.bin_edges is None else np.asarray(spec.grid.bin_edges, dtype=np.float64)

    for source in _select_theory_sources(spec, source_labels):
        _, parsed = parse_mathematica_assignment(source)
        source_label = _source_label(source)
        source_name = format_source(source)

        if _is_numeric_series(parsed):
            label = _default_series_labels(spec, 1)[0]
            x_vals, center, lower, upper = _series_arrays(parsed)
            theory_series.append(
                TheoryBandSeries(
                    observable_id=observable_id,
                    series_label=label,
                    source_label=source_label,
                    source=source_name,
                    x=x_vals,
                    center=center,
                    lower=lower,
                    upper=upper,
                    bin_edges=bin_edges,
                )
            )
            continue

        if isinstance(parsed, list) and parsed and all(_is_numeric_series(item) for item in parsed) and any(
            _series_contains_around(item) for item in parsed
        ):
            labels = _default_series_labels(spec, len(parsed))
            for label, series in zip(labels, parsed):
                x_vals, center, lower, upper = _series_arrays(series)
                theory_series.append(
                    TheoryBandSeries(
                        observable_id=observable_id,
                        series_label=label,
                        source_label=source_label,
                        source=source_name,
                        x=x_vals,
                        center=center,
                        lower=lower,
                        upper=upper,
                        bin_edges=bin_edges,
                    )
            )
            continue

        if _is_band_triplet(parsed):
            label = _default_series_labels(spec, 1)[0]
            x_vals, center, _, _ = _series_arrays(parsed[0])
            _, lower, _, _ = _series_arrays(parsed[1])
            _, upper, _, _ = _series_arrays(parsed[2])
            theory_series.append(
                TheoryBandSeries(
                    observable_id=observable_id,
                    series_label=label,
                    source_label=source_label,
                    source=source_name,
                    x=x_vals,
                    center=center,
                    lower=lower,
                    upper=upper,
                    bin_edges=bin_edges,
                )
            )
            continue

        if isinstance(parsed, list) and parsed and all(_is_numeric_series(item) for item in parsed):
            labels = _default_series_labels(spec, len(parsed))
            if len(labels) == len(parsed):
                for label, series in zip(labels, parsed):
                    x_vals, center, lower, upper = _series_arrays(series)
                    theory_series.append(
                        TheoryBandSeries(
                            observable_id=observable_id,
                            series_label=label,
                            source_label=source_label,
                            source=source_name,
                            x=x_vals,
                            center=center,
                            lower=lower,
                            upper=upper,
                            bin_edges=bin_edges,
                        )
                    )
                continue

        if isinstance(parsed, list) and parsed and all(_is_band_triplet(item) for item in parsed):
            labels = _default_series_labels(spec, len(parsed))
            for label, series_triplet in zip(labels, parsed):
                x_vals, center, _, _ = _series_arrays(series_triplet[0])
                _, lower, _, _ = _series_arrays(series_triplet[1])
                _, upper, _, _ = _series_arrays(series_triplet[2])
                theory_series.append(
                    TheoryBandSeries(
                        observable_id=observable_id,
                        series_label=label,
                        source_label=source_label,
                        source=source_name,
                        x=x_vals,
                        center=center,
                        lower=lower,
                        upper=upper,
                        bin_edges=bin_edges,
                    )
                )
            continue

        raise ValueError(
            f"Unsupported Mathematica structure for {observable_id} from {source.path}"
        )

    return tuple(theory_series)


def _read_text_source(ref: SourceRef) -> str:
    path = resolve_source(ref)
    if ref.member:
        with tarfile.open(path, "r:*") as tar:
            member = tar.extractfile(ref.member)
            if member is None:
                raise FileNotFoundError(f"Member {ref.member} not found in {path}")
            return member.read().decode("utf-8")
    return path.read_text()


def _fieldnames_and_rows(csv_rows: Sequence[Sequence[str]]) -> Tuple[List[str], List[dict]]:
    raw_fieldnames = list(csv_rows[0])
    fieldnames: List[str] = []
    seen: dict[str, int] = {}
    for idx, name in enumerate(raw_fieldnames):
        base = name.strip() or f"col{idx}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        fieldnames.append(base if count == 0 else f"{base}__{count}")
    rows = [dict(zip(fieldnames, row)) for row in csv_rows[1:]]
    return fieldnames, rows


def _csv_blocks(text: str) -> List[Tuple[List[str], List[dict]]]:
    blocks: List[List[str]] = []
    current: List[str] = []

    for raw_line in text.splitlines():
        if raw_line.startswith("#:"):
            continue
        if not raw_line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(raw_line)

    if current:
        blocks.append(current)

    parsed_blocks: List[Tuple[List[str], List[dict]]] = []
    for block in blocks:
        csv_rows = list(csv.reader(io.StringIO("\n".join(block))))
        if not csv_rows:
            continue
        parsed_blocks.append(_fieldnames_and_rows(csv_rows))
    return parsed_blocks


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _parse_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    text = text.replace("−", "-").replace("–", "-").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _looks_like_raa_column(name: str) -> bool:
    norm = _normalized(name)
    raw = name.lower()
    return "raa" in norm or ("aa" in raw and ("r" in raw or "\\mathrm{aa}" in raw))


def _choose_x_columns(fieldnames: Sequence[str], observable_type: str) -> Tuple[str, Optional[str], Optional[str]]:
    normalized = {name: _normalized(name) for name in fieldnames}
    raw_lower = {name: name.lower() for name in fieldnames}

    def find_primary(tokens: Sequence[str]) -> Optional[str]:
        for token in tokens:
            for name, norm in normalized.items():
                if token in norm and "low" not in norm and "high" not in norm:
                    return name
                raw = raw_lower[name]
                if token == "npart" and "part" in raw and "centrality" not in raw:
                    if "low" not in raw and "high" not in raw:
                        return name
                if token == "pt" and "p_" in raw and ("t" in raw or "{t}" in raw):
                    if "low" not in raw and "high" not in raw:
                        return name
                if token in ("absrap", "yrap", "rapidity", "rap") and ("rap" in raw or "|y|" in raw):
                    if "low" not in raw and "high" not in raw:
                        return name
        return None

    if observable_type.endswith("_vs_npart"):
        primary = find_primary(("npart",))
        if primary is None:
            primary = find_primary(("centrality",))
    elif observable_type.endswith("_vs_pt"):
        primary = find_primary(("pt",))
    elif observable_type.endswith("_vs_y"):
        primary = find_primary(("absrap", "yrap", "rapidity", "rap"))
    else:
        primary = find_primary(("npart", "pt", "rap"))

    if primary is None:
        if (
            observable_type.endswith("_vs_y")
            and len(fieldnames) >= 3
            and fieldnames[0].startswith("col")
            and "low" in fieldnames[1].lower()
            and "high" in fieldnames[2].lower()
        ):
            return fieldnames[0], fieldnames[1], fieldnames[2]
        if (
            observable_type.endswith("_vs_npart")
            and len(fieldnames) >= 3
            and fieldnames[0].startswith("col")
            and "low" in fieldnames[1].lower()
            and "high" in fieldnames[2].lower()
        ):
            return fieldnames[0], fieldnames[1], fieldnames[2]
        raise ValueError(f"Could not identify x column for observable type '{observable_type}'")

    primary_norm = normalized[primary]
    low = None
    high = None
    for name, norm in normalized.items():
        if primary_norm in norm and "low" in norm:
            low = name
        if primary_norm in norm and "high" in norm:
            high = name
    return primary, low, high


def _choose_y_column(
    fieldnames: Sequence[str],
    rows: Sequence[dict],
    exp_spec: ExperimentalObservableSpec,
    x_columns: Sequence[str],
) -> Tuple[str, int]:
    normalized = {name: _normalized(name) for name in fieldnames}
    excluded = set(name for name in x_columns if name is not None)
    candidates: List[Tuple[int, int, str]] = []

    for idx, name in enumerate(fieldnames):
        norm = normalized[name]
        raw = name.lower()
        if name in excluded:
            continue
        if any(token in norm for token in ("stat", "sys", "low", "high", "global", "unc", "error")):
            continue
        values = np.asarray([_parse_float_or_nan(row.get(name)) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        score = 0
        if exp_spec.observable_type.startswith("RAA"):
            if _looks_like_raa_column(name):
                score += 1000
            if "upsilon" in raw and "raa" in raw:
                score += 250
        elif exp_spec.observable_type.startswith("double_ratio"):
            if "upperbound" in norm:
                score += 1100 if exp_spec.upper_limit else -150
            if "lowerbound" in norm:
                score -= 1100 if exp_spec.upper_limit else 200
            if "ratio" in raw or "ratio" in norm:
                score += 900
            if "pbpb" in norm and "pp" in norm:
                score += 850
            if "upsilon" in raw or "upsilon" in norm or "ny" in norm or "n_" in raw:
                score += 200
        if any(token in raw for token in ("centrality", "npart", "rap", "p_t", "pt [")):
            score -= 800
        median_abs = float(np.nanmedian(np.abs(finite)))
        if exp_spec.observable_type.startswith(("RAA", "double_ratio")):
            if median_abs <= 2.5:
                score += 50
            elif median_abs > 5.0:
                score -= 600
        candidates.append((score, -idx, name))

    if not candidates:
        raise ValueError("Could not identify y column in HEPData CSV")
    score, _, name = max(candidates)
    return name, score


def _choose_error_column(fieldnames: Sequence[str], token: str, sign: str) -> Optional[str]:
    for name in fieldnames:
        raw = name.lower()
        norm = _normalized(name)
        if token not in raw and token not in norm:
            continue
        if sign == "+" and ("+" in raw or "plus" in raw):
            return name
        if sign == "-" and ("-" in raw or "minus" in raw):
            return name
        if sign == "+" and norm.endswith(token):
            return name
        if sign == "-" and norm.endswith(token):
            return name
    return None


def _centrality_to_npart_map(blocks: Sequence[Tuple[List[str], List[dict]]]) -> dict[float, float]:
    mapping: dict[float, float] = {}
    for fieldnames, rows in blocks:
        centrality_name = None
        npart_name = None
        for name in fieldnames:
            raw = name.lower()
            if centrality_name is None and "centrality" in raw and "low" not in raw and "high" not in raw:
                centrality_name = name
            if npart_name is None and "part" in raw and "centrality" not in raw and "low" not in raw and "high" not in raw:
                npart_name = name
        if centrality_name is None or npart_name is None:
            continue
        for row in rows:
            centrality = _parse_float_or_nan(row.get(centrality_name))
            npart = _parse_float_or_nan(row.get(npart_name))
            if np.isfinite(centrality) and np.isfinite(npart):
                mapping[round(float(centrality), 12)] = float(npart)
    return mapping


def _quadrature(*arrays: Optional[np.ndarray]) -> Optional[np.ndarray]:
    finite = [arr for arr in arrays if arr is not None]
    if not finite:
        return None
    total = np.zeros_like(finite[0], dtype=np.float64)
    for arr in finite:
        total += np.square(arr)
    return np.sqrt(total)


def _load_hepdata_series(
    observable_id: str, exp_spec: ExperimentalObservableSpec, ref: SourceRef
) -> Tuple[ExperimentalSeries, ...]:
    text = _read_text_source(ref)
    blocks = _csv_blocks(text)
    if not blocks:
        raise ValueError(f"No CSV blocks found in {format_source(ref)}")

    block_choices: List[Tuple[int, int, List[str], List[dict], str, Optional[str], Optional[str], str]] = []
    for block_index, (fieldnames, rows) in enumerate(blocks):
        try:
            x_name, x_low_name, x_high_name = _choose_x_columns(fieldnames, exp_spec.observable_type)
            y_name, y_score = _choose_y_column(
                fieldnames,
                rows,
                exp_spec,
                [x_name, x_low_name, x_high_name],
            )
        except ValueError:
            continue

        x_probe = np.asarray([_parse_float_or_nan(row.get(x_name)) for row in rows], dtype=np.float64)
        y_probe = np.asarray([_parse_float_or_nan(row.get(y_name)) for row in rows], dtype=np.float64)
        valid_probe = np.isfinite(x_probe) & np.isfinite(y_probe)
        score = int(np.sum(valid_probe)) + y_score
        x_name_lower = x_name.lower()
        if exp_spec.observable_type.endswith("_vs_npart"):
            if "part" in x_name_lower:
                score += 1000
            if "centrality" in x_name_lower:
                score -= 100
        block_choices.append((score, block_index, fieldnames, rows, x_name, x_low_name, x_high_name, y_name))

    if not block_choices:
        raise ValueError(f"Could not parse a usable HEPData block from {format_source(ref)}")

    _, _, fieldnames, rows, x_name, x_low_name, x_high_name, y_name = max(block_choices, key=lambda item: (item[0], -item[1]))

    x = np.asarray([_parse_float_or_nan(row.get(x_name)) for row in rows], dtype=np.float64)
    x_low = None if x_low_name is None else np.asarray([_parse_float_or_nan(row.get(x_low_name)) for row in rows], dtype=np.float64)
    x_high = None if x_high_name is None else np.asarray([_parse_float_or_nan(row.get(x_high_name)) for row in rows], dtype=np.float64)
    y = np.asarray([_parse_float_or_nan(row.get(y_name)) for row in rows], dtype=np.float64)

    stat_plus = _choose_error_column(fieldnames, "stat", "+")
    stat_minus = _choose_error_column(fieldnames, "stat", "-")
    sys_plus = _choose_error_column(fieldnames, "sys", "+")
    sys_minus = _choose_error_column(fieldnames, "sys", "-")

    stat_plus_arr = None if stat_plus is None else np.abs(np.asarray([_parse_float_or_nan(row.get(stat_plus)) for row in rows], dtype=np.float64))
    stat_minus_arr = None if stat_minus is None else np.abs(np.asarray([_parse_float_or_nan(row.get(stat_minus)) for row in rows], dtype=np.float64))
    sys_plus_arr = None if sys_plus is None else np.abs(np.asarray([_parse_float_or_nan(row.get(sys_plus)) for row in rows], dtype=np.float64))
    sys_minus_arr = None if sys_minus is None else np.abs(np.asarray([_parse_float_or_nan(row.get(sys_minus)) for row in rows], dtype=np.float64))

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x_low is not None:
        x_low = x_low[valid]
    if x_high is not None:
        x_high = x_high[valid]
    if stat_plus_arr is not None:
        stat_plus_arr = stat_plus_arr[valid]
    if stat_minus_arr is not None:
        stat_minus_arr = stat_minus_arr[valid]
    if sys_plus_arr is not None:
        sys_plus_arr = sys_plus_arr[valid]
    if sys_minus_arr is not None:
        sys_minus_arr = sys_minus_arr[valid]

    integrated_mask = None
    if (
        exp_spec.observable_type.endswith("_vs_npart")
        and "centrality" in x_name.lower()
        and x_low is not None
        and x_high is not None
        and len(x) > 1
    ):
        min_low = float(np.nanmin(x_low))
        max_high = float(np.nanmax(x_high))
        integrated = np.isclose(x_low, min_low) & np.isclose(x_high, max_high)
        if np.count_nonzero(integrated) == 1:
            if exp_spec.observable_type.startswith("double_ratio_vs_npart"):
                integrated_mask = integrated.copy()
            else:
                keep = ~integrated
                x = x[keep]
                y = y[keep]
                x_low = x_low[keep]
                x_high = x_high[keep]
                if stat_plus_arr is not None:
                    stat_plus_arr = stat_plus_arr[keep]
                if stat_minus_arr is not None:
                    stat_minus_arr = stat_minus_arr[keep]
                if sys_plus_arr is not None:
                    sys_plus_arr = sys_plus_arr[keep]
                if sys_minus_arr is not None:
                    sys_minus_arr = sys_minus_arr[keep]

    if exp_spec.observable_type.endswith("_vs_npart") and "centrality" in x_name.lower():
        centrality_lookup = _centrality_to_npart_map(blocks)
        remapped_x = np.asarray(
            [centrality_lookup.get(round(float(value), 12), float("nan")) for value in x],
            dtype=np.float64,
        )
        if np.all(np.isfinite(remapped_x)):
            x = remapped_x
            x_low = None
            x_high = None

    if exp_spec.upper_limit and len(x) > 0:
        keep_by_x: dict[float, int] = {}
        for idx, x_val in enumerate(x):
            key = round(float(x_val), 12)
            if key not in keep_by_x or y[idx] >= y[keep_by_x[key]]:
                keep_by_x[key] = idx
        keep = np.asarray([keep_by_x[key] for key in sorted(keep_by_x)], dtype=int)
        x = x[keep]
        y = y[keep]
        if x_low is not None:
            x_low = x_low[keep]
        if x_high is not None:
            x_high = x_high[keep]
        if stat_plus_arr is not None:
            stat_plus_arr = stat_plus_arr[keep]
        if stat_minus_arr is not None:
            stat_minus_arr = stat_minus_arr[keep]
        if sys_plus_arr is not None:
            sys_plus_arr = sys_plus_arr[keep]
        if sys_minus_arr is not None:
            sys_minus_arr = sys_minus_arr[keep]
        if integrated_mask is not None:
            integrated_mask = integrated_mask[keep]

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    if x_low is not None:
        x_low = x_low[order]
    if x_high is not None:
        x_high = x_high[order]
    if stat_plus_arr is not None:
        stat_plus_arr = stat_plus_arr[order]
    if stat_minus_arr is not None:
        stat_minus_arr = stat_minus_arr[order]
    if sys_plus_arr is not None:
        sys_plus_arr = sys_plus_arr[order]
    if sys_minus_arr is not None:
        sys_minus_arr = sys_minus_arr[order]
    if integrated_mask is not None:
        integrated_mask = integrated_mask[order]
        if np.any(integrated_mask):
            finite_non_integrated = x[(~integrated_mask) & np.isfinite(x)]
            if finite_non_integrated.size > 0:
                # Draw integrated min-bias point at the right edge, mirroring
                # the dedicated right-side min-bias panel style in paper figures.
                right_side_x = float(np.max(finite_non_integrated) + 25.0)
                x[integrated_mask] = right_side_x
                if x_low is not None:
                    x_low[integrated_mask] = right_side_x
                if x_high is not None:
                    x_high[integrated_mask] = right_side_x
                # Preserve monotonic x ordering after relocating min-bias.
                reorder = np.argsort(x, kind="stable")
                x = x[reorder]
                y = y[reorder]
                if x_low is not None:
                    x_low = x_low[reorder]
                if x_high is not None:
                    x_high = x_high[reorder]
                if stat_plus_arr is not None:
                    stat_plus_arr = stat_plus_arr[reorder]
                if stat_minus_arr is not None:
                    stat_minus_arr = stat_minus_arr[reorder]
                if sys_plus_arr is not None:
                    sys_plus_arr = sys_plus_arr[reorder]
                if sys_minus_arr is not None:
                    sys_minus_arr = sys_minus_arr[reorder]

    # Paper-style: double ratios are rendered with stat and sys bars separately.
    if (
        exp_spec.observable_type.startswith("double_ratio")
        and not exp_spec.upper_limit
        and stat_plus_arr is not None
        and stat_minus_arr is not None
        and sys_plus_arr is not None
        and sys_minus_arr is not None
    ):
        base_kwargs = dict(
            observable_id=observable_id,
            series_label=f"{exp_spec.experiment} {exp_spec.state}",
            experiment=exp_spec.experiment,
            state=exp_spec.state,
            observable_type=exp_spec.observable_type,
            acceptance=exp_spec.acceptance,
            source=format_source(ref),
            x=x,
            y=y,
            x_low=x_low,
            x_high=x_high,
            is_minbias=exp_spec.is_minbias,
            upper_limit=exp_spec.upper_limit,
            combined_state=exp_spec.combined_state,
            note=exp_spec.note,
        )
        return (
            ExperimentalSeries(
                **base_kwargs,
                yerr_low=stat_minus_arr,
                yerr_high=stat_plus_arr,
                uncertainty_kind="stat",
            ),
            ExperimentalSeries(
                **base_kwargs,
                yerr_low=sys_minus_arr,
                yerr_high=sys_plus_arr,
                uncertainty_kind="sys",
            ),
        )

    return (
        ExperimentalSeries(
            observable_id=observable_id,
            series_label=f"{exp_spec.experiment} {exp_spec.state}",
            experiment=exp_spec.experiment,
            state=exp_spec.state,
            observable_type=exp_spec.observable_type,
            acceptance=exp_spec.acceptance,
            source=format_source(ref),
            x=x,
            y=y,
            x_low=x_low,
            x_high=x_high,
            yerr_low=_quadrature(stat_minus_arr, sys_minus_arr),
            yerr_high=_quadrature(stat_plus_arr, sys_plus_arr),
            uncertainty_kind="total",
            is_minbias=exp_spec.is_minbias,
            upper_limit=exp_spec.upper_limit,
            combined_state=exp_spec.combined_state,
            note=exp_spec.note,
        ),
    )


def _note_kv(note: Optional[str], key: str) -> Optional[str]:
    if not note:
        return None
    match = re.search(rf"(?:^|[;,\s]){re.escape(key)}\s*=\s*([A-Za-z0-9_-]+)", note)
    if match:
        return match.group(1)
    return None


def _extract_gridbox_series(
    *,
    observable_id: str,
    exp_spec: ExperimentalObservableSpec,
    ref: SourceRef,
    snippet: str,
) -> ExperimentalSeries:
    matches = list(_NB_GRID_RE.finditer(snippet))
    if not matches:
        raise ValueError(f"Could not extract GridBox for '{ref.variable}' in {ref.path}")
    inferred = (ref.variable or "").lower()
    explicit = _note_kv(ref.note, "uncertainty")
    occurrence = _note_kv(ref.note, "occurrence")
    # Some notebook snippets print stat and sys GridBoxes back-to-back for the
    # same variable occurrence.
    # Keep occurrence selection stable:
    # - occurrence=first -> keep first GridBox (ratio21 3-point ATLAS)
    # - occurrence=last  -> keep last GridBox for explicit sys (ratio31 4-point ATLAS sys)
    # - otherwise        -> retain the historical explicit-sys preference for the last GridBox
    prefer_sys_grid = explicit == "sys" or (explicit is None and "sys" in inferred)
    if prefer_sys_grid and len(matches) > 1:
        if occurrence == "first":
            match = matches[0]
        elif occurrence == "last":
            match = matches[-1]
        else:
            match = matches[-1]
    else:
        match = matches[0]

    body = match.group("body")

    # Two supported notebook renderings:
    # 1) {"x", InterpretationBox[..., Around[y, err]]}
    # 2) {InterpretationBox[..., Around[x, xerr]], InterpretationBox[..., Around[y, err]]}
    rows = list(_NB_ROW_RE.finditer(body))
    if rows:
        x = np.asarray([_clean_number(row.group("x")) for row in rows], dtype=np.float64)
        y = np.asarray([_clean_number(row.group("y")) for row in rows], dtype=np.float64)
        err = np.asarray([abs(_clean_number(row.group("err"))) for row in rows], dtype=np.float64)
        x_low = None
        x_high = None
    else:
        around_re = re.compile(
            r"\{\s*"
            r"InterpretationBox\[.*?Around\[(?P<x>"
            + _MATHEMATICA_NUMBER
            + r")\s*,\s*(?P<xerr>"
            + _MATHEMATICA_NUMBER
            + r")\]\].*?,\s*"
            r"(?:FormBox\[\s*)?"
            r"InterpretationBox\[.*?Around\[(?P<y>"
            + _MATHEMATICA_NUMBER
            + r")\s*,\s*(?P<err>"
            + _MATHEMATICA_NUMBER
            + r")\]\]"
            r"(?:\s*,\s*TraditionalForm\])?"
            r"\s*\}",
            re.DOTALL,
        )
        rows2 = list(around_re.finditer(body))
        if not rows2:
            raise ValueError(f"No Around rows found for '{ref.variable}' in {ref.path}")

        x = np.asarray([_clean_number(row.group("x")) for row in rows2], dtype=np.float64)
        xerr = np.asarray([abs(_clean_number(row.group("xerr"))) for row in rows2], dtype=np.float64)
        y = np.asarray([_clean_number(row.group("y")) for row in rows2], dtype=np.float64)
        err = np.asarray([abs(_clean_number(row.group("err"))) for row in rows2], dtype=np.float64)
        x_low = x - xerr
        x_high = x + xerr

    kind = "total"
    if explicit in {"stat", "sys", "total"}:
        kind = explicit
    elif "stat" in inferred:
        kind = "stat"
    elif "sys" in inferred:
        kind = "sys"

    return ExperimentalSeries(
        observable_id=observable_id,
        series_label=f"{exp_spec.experiment} {exp_spec.state}",
        experiment=exp_spec.experiment,
        state=exp_spec.state,
        observable_type=exp_spec.observable_type,
        acceptance=exp_spec.acceptance,
        source=format_source(ref),
        x=x,
        y=y,
        x_low=x_low,
        x_high=x_high,
        yerr_low=err,
        yerr_high=err,
        uncertainty_kind=kind,  # used for double-ratio stat/sys styling
        upper_limit=exp_spec.upper_limit,
        combined_state=exp_spec.combined_state,
        note=exp_spec.note,
    )


def _load_notebook_series(observable_id: str, exp_spec: ExperimentalObservableSpec, ref: SourceRef) -> ExperimentalSeries:
    text = _read_text_source(SourceRef(path=ref.path))
    marker = f'{ref.variable}",'
    indices: List[int] = []
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            break
        indices.append(idx)
        start = idx + len(marker)
    if not indices:
        raise ValueError(f"Notebook variable '{ref.variable}' not found in {ref.path}")

    candidates: List[ExperimentalSeries] = []
    for idx in indices:
        snippet = text[idx : idx + 20000]
        try:
            candidates.append(
                _extract_gridbox_series(observable_id=observable_id, exp_spec=exp_spec, ref=ref, snippet=snippet)
            )
        except ValueError:
            continue

    if not candidates:
        raise ValueError(f"Could not extract numeric GridBox for '{ref.variable}' in {ref.path}")
    if len(candidates) == 1:
        return candidates[0]

    occurrence = _note_kv(ref.note, "occurrence")
    if occurrence in {"first", "last"}:
        return candidates[0] if occurrence == "first" else candidates[-1]

    summaries = [
        {
            "len": int(series.x.shape[0]),
            "x_min": float(np.min(series.x)),
            "x_max": float(np.max(series.x)),
        }
        for series in candidates
    ]
    raise ValueError(
        "Ambiguous notebook variable extraction for "
        f"'{ref.variable}' in {ref.path}; "
        "add SourceRef.note 'occurrence=first' or 'occurrence=last'. "
        f"Candidates: {summaries}"
    )


def load_experimental_series(observable_id: str) -> Tuple[ExperimentalSeries, ...]:
    spec = get_observable_spec(observable_id)
    series: List[ExperimentalSeries] = []

    for exp_spec in spec.experimental_observables:
        numeric_sources = [
            source
            for source in exp_spec.sources
            if source.member or source.variable or source.path.lower().endswith(".csv")
        ]
        for numeric_source in numeric_sources:
            if numeric_source.member or numeric_source.path.lower().endswith(".csv"):
                series.extend(_load_hepdata_series(observable_id, exp_spec, numeric_source))
            else:
                series.append(_load_notebook_series(observable_id, exp_spec, numeric_source))

    return tuple(series)


def build_reference_bundle(
    observable_id: str,
    *,
    source_labels: Optional[Sequence[str]] = None,
) -> ObservableReferenceBundle:
    spec = get_observable_spec(observable_id)
    selected_mathematica_sources = _select_theory_sources(spec, source_labels) if spec.mathematica_sources else ()
    if selected_mathematica_sources:
        selected_source_labels = {_source_label(source).lower() for source in selected_mathematica_sources}
        selected_datafile_sources = tuple(
            source
            for source in spec.datafile_sources
            if _source_label(source).lower() in selected_source_labels
        )
    else:
        selected_datafile_sources = _select_datafile_sources(spec, source_labels)

    theory_series = load_theory_series(observable_id, source_labels=source_labels)
    all_experimental_series = load_experimental_series(observable_id)
    experimental_series = tuple(s for s in all_experimental_series if not s.is_minbias)
    minbias_experimental_series = tuple(s for s in all_experimental_series if s.is_minbias)
    category = "comparison" if (experimental_series or minbias_experimental_series) else "theory_only"
    centrality_labels = STANDARD_CENTRALITY_LABELS if spec.observable_type.endswith("_vs_npart") else ()
    if selected_mathematica_sources:
        theory_sources = tuple(format_source(source) for source in selected_mathematica_sources)
    else:
        theory_sources = tuple(f"generated_from::{format_source(source)}" for source in selected_datafile_sources)

    return ObservableReferenceBundle(
        observable_id=observable_id,
        system=spec.system,
        energy_label=spec.energy_label,
        observable_type=spec.observable_type,
        acceptance=spec.acceptance,
        theory_note=spec.theory_note,
        category=category,
        theory_series=theory_series,
        experimental_series=experimental_series,
        minbias_experimental_series=minbias_experimental_series,
        theory_sources=theory_sources,
        datafile_sources=tuple(format_source(source) for source in selected_datafile_sources),
        issues=tuple(f"{issue.code}: {issue.message}" for issue in spec.issues),
        centrality_labels=centrality_labels,
    )


def iter_reference_bundles(
    system: Optional[str] = None,
    *,
    source_labels: Optional[Sequence[str]] = None,
) -> Iterable[ObservableReferenceBundle]:
    for observable_id in list_observable_ids(system=system):
        yield build_reference_bundle(observable_id, source_labels=source_labels)
