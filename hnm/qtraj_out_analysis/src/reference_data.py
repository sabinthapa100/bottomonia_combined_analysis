from __future__ import annotations

import csv
import io
import math
import re
import tarfile
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from qtraj_analysis.observable_registry import (
    format_source,
    get_observable_spec,
    list_observable_ids,
    resolve_source,
)
from qtraj_analysis.schema import ExperimentalObservableSpec, SourceRef, TheoryObservableSpec


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
    category: str
    theory_series: Tuple[TheoryBandSeries, ...]
    experimental_series: Tuple[ExperimentalSeries, ...]
    theory_sources: Tuple[str, ...]
    datafile_sources: Tuple[str, ...]
    issues: Tuple[str, ...]
    centrality_labels: Tuple[str, ...] = ()


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


def load_theory_series(
    observable_id: str,
    *,
    source_labels: Optional[Sequence[str]] = None,
) -> Tuple[TheoryBandSeries, ...]:
    spec = get_observable_spec(observable_id)
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
    match = _NB_GRID_RE.search(snippet)
    if match is None:
        raise ValueError(f"Could not extract GridBox for '{ref.variable}' in {ref.path}")

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
    inferred = (ref.variable or "").lower()
    explicit = _note_kv(ref.note, "uncertainty")
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
        numeric_sources = [source for source in exp_spec.sources if source.member or source.variable]
        for numeric_source in numeric_sources:
            if numeric_source.member:
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
    selected_mathematica_sources = _select_theory_sources(spec, source_labels)
    selected_source_labels = {_source_label(source).lower() for source in selected_mathematica_sources}
    selected_datafile_sources = tuple(
        source
        for source in spec.datafile_sources
        if _source_label(source).lower() in selected_source_labels
    )

    theory_series = load_theory_series(observable_id, source_labels=source_labels)
    experimental_series = load_experimental_series(observable_id)
    category = "comparison" if experimental_series else "theory_only"
    centrality_labels = STANDARD_CENTRALITY_LABELS if spec.observable_type.endswith("_vs_npart") else ()

    return ObservableReferenceBundle(
        observable_id=observable_id,
        system=spec.system,
        energy_label=spec.energy_label,
        observable_type=spec.observable_type,
        acceptance=spec.acceptance,
        category=category,
        theory_series=theory_series,
        experimental_series=experimental_series,
        theory_sources=tuple(format_source(source) for source in selected_mathematica_sources),
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
