from __future__ import annotations

import csv
import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from qtraj_analysis.reference_data import (
    ExperimentalSeries,
    ObservableReferenceBundle,
    TheoryBandSeries,
)
from qtraj_analysis.plot_config import (
    config,
    get_x_label,
    get_state_color,
    get_system_annotation,
    get_state_label,
    AXIS_LABELS,
)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "inputs").exists() and (parent / "hnm").exists():
            return parent
    raise RuntimeError(f"Could not infer repository root from {here}")


REPO_ROOT = _find_repo_root()

_SYSTEM_OUTPUT_KEYS = {
    ("AuAu", "200 GeV"): ("RHIC", "AuAu200GeV"),
    ("PbPb", "2.76 TeV"): ("LHC", "PbPb2p76TeV"),
    ("PbPb", "5.02 TeV"): ("LHC", "PbPb5p02TeV"),
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _system_output_key(bundle: ObservableReferenceBundle) -> Tuple[str, str]:
    try:
        return _SYSTEM_OUTPUT_KEYS[(bundle.system, bundle.energy_label)]
    except KeyError as exc:
        raise ValueError(
            f"No output key configured for {(bundle.system, bundle.energy_label)}"
        ) from exc


def _resolve_output_root(output_root: Optional[Path | str]) -> Path:
    if output_root is None:
        return REPO_ROOT / "outputs" / "qtraj_outputs"
    root = Path(output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root.resolve()


def get_output_layout(
    bundle: ObservableReferenceBundle, output_root: Optional[Path | str] = None
) -> Dict[str, Path]:
    collider, system_key = _system_output_key(bundle)
    base = _resolve_output_root(output_root) / collider / system_key / "production"
    category = bundle.category
    return {
        "base": base,
        "data": base / "data" / category,
        "figures": base / "figures" / category,
        "manifests": base / "manifests",
        "validation": base / "validation",
    }


def _series_filename(observable_id: str, label: str, suffix: str) -> str:
    return f"{observable_id}__{_slug(label)}__{suffix}"


def _theory_to_rows(series: TheoryBandSeries) -> np.ndarray:
    x_low = np.full_like(series.x, np.nan)
    x_high = np.full_like(series.x, np.nan)
    if series.bin_edges is not None and len(series.bin_edges) == len(series.x) + 1:
        x_low = series.bin_edges[:-1]
        x_high = series.bin_edges[1:]
    return np.column_stack(
        [series.x, x_low, x_high, series.center, series.lower, series.upper]
    )


def save_theory_series(series: TheoryBandSeries, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (
        _series_filename(
            series.observable_id,
            f"{series.series_label}_{series.source_label}",
            "theory.csv",
        )
    )
    header = "x,x_low,x_high,y_center,y_lower,y_upper"
    np.savetxt(path, _theory_to_rows(series), delimiter=",", header=header, comments="")
    return path


def _combine_theory_envelopes(
    series_list: Iterable[TheoryBandSeries],
) -> Tuple[TheoryBandSeries, ...]:
    envelopes: List[TheoryBandSeries] = []
    for series_label, grouped_series in _group_theory_by_label(series_list).items():
        ordered = list(grouped_series)
        first = ordered[0]
        x = first.x
        bin_edges = first.bin_edges

        for candidate in ordered[1:]:
            if x.shape != candidate.x.shape or not np.allclose(
                x, candidate.x, atol=1e-12, rtol=0.0
            ):
                raise ValueError(
                    f"Cannot combine theory envelope for '{first.observable_id}/{series_label}': mismatched x-grid"
                )
            if bin_edges is None and candidate.bin_edges is not None:
                raise ValueError(
                    f"Cannot combine theory envelope for '{first.observable_id}/{series_label}': inconsistent bin edges"
                )
            if bin_edges is not None:
                if candidate.bin_edges is None or not np.allclose(
                    np.asarray(bin_edges, dtype=np.float64),
                    np.asarray(candidate.bin_edges, dtype=np.float64),
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"Cannot combine theory envelope for '{first.observable_id}/{series_label}': mismatched bin edges"
                    )

        centers = np.vstack([series.center for series in ordered])
        lowers = np.vstack([series.lower for series in ordered])
        uppers = np.vstack([series.upper for series in ordered])
        source_labels = ",".join(series.source_label for series in ordered)

        # Keep strict envelope bounds but allow observable-specific central curves.
        # For PbPb 5.02 TeV 3S/2S vs pT, use k4 as the displayed central line to
        # match the local Mathematica lhc-3d behavior while still showing full k3-k4 spread.
        center_curve = np.mean(centers, axis=0)
        if first.observable_id == "pbpb5023_ratio32vspt":
            preferred = next(
                (series for series in ordered if series.source_label.lower() == "k4"),
                None,
            )
            if preferred is not None:
                center_curve = np.asarray(preferred.center, dtype=np.float64)

        envelopes.append(
            TheoryBandSeries(
                observable_id=first.observable_id,
                series_label=series_label,
                source_label="envelope",
                source=f"Envelope across {source_labels}",
                x=x.copy(),
                center=center_curve,
                lower=np.min(lowers, axis=0),
                upper=np.max(uppers, axis=0),
                bin_edges=None
                if bin_edges is None
                else np.asarray(bin_edges, dtype=np.float64).copy(),
            )
        )
    return tuple(envelopes)


def save_theory_envelope_series(series: TheoryBandSeries, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (
        _series_filename(
            series.observable_id, series.series_label, "theory_envelope.csv"
        )
    )
    header = "x,x_low,x_high,y_center,y_lower,y_upper"
    np.savetxt(path, _theory_to_rows(series), delimiter=",", header=header, comments="")
    return path


def save_experimental_series(series: ExperimentalSeries, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    label = f"{series.experiment}_{series.state}"
    if getattr(series, "uncertainty_kind", "total") != "total":
        label = f"{label}_{series.uncertainty_kind}"
    path = outdir / (_series_filename(series.observable_id, label, "exp.csv"))
    x_low = series.x if series.x_low is None else series.x_low
    x_high = series.x if series.x_high is None else series.x_high
    yerr_low = np.zeros_like(series.y) if series.yerr_low is None else series.yerr_low
    yerr_high = (
        np.zeros_like(series.y) if series.yerr_high is None else series.yerr_high
    )
    arr = np.column_stack([series.x, x_low, x_high, series.y, yerr_low, yerr_high])
    header = "x,x_low,x_high,y,yerr_low,yerr_high"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")
    return path


def _bundle_manifest(
    bundle: ObservableReferenceBundle,
    theory_paths: List[Path],
    exp_paths: List[Path],
    envelope_paths: List[Path],
) -> dict:
    def _path_ref(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    return {
        "observable_id": bundle.observable_id,
        "system": bundle.system,
        "energy_label": bundle.energy_label,
        "observable_type": bundle.observable_type,
        "acceptance": bundle.acceptance,
        "theory_note": bundle.theory_note,
        "category": bundle.category,
        "theory_sources": list(bundle.theory_sources),
        "datafile_sources": list(bundle.datafile_sources),
        "issues": list(bundle.issues),
        "centrality_labels": list(bundle.centrality_labels),
        "theory_files": [_path_ref(path) for path in theory_paths],
        "theory_envelope_files": [_path_ref(path) for path in envelope_paths],
        "experimental_files": [_path_ref(path) for path in exp_paths],
    }


def save_bundle(
    bundle: ObservableReferenceBundle, output_root: Optional[Path | str] = None
) -> Dict[str, Path]:
    layout = get_output_layout(bundle, output_root=output_root)
    theory_dir = layout["data"] / "theory"
    theory_envelope_dir = layout["data"] / "theory_envelopes"
    exp_dir = layout["data"] / "experiment"
    layout["manifests"].mkdir(parents=True, exist_ok=True)

    theory_paths = [
        save_theory_series(series, theory_dir) for series in bundle.theory_series
    ]
    envelope_paths = [
        save_theory_envelope_series(series, theory_envelope_dir)
        for series in _combine_theory_envelopes(bundle.theory_series)
    ]
    exp_paths = [
        save_experimental_series(series, exp_dir)
        for series in bundle.experimental_series
    ]

    manifest_path = layout["manifests"] / f"{bundle.observable_id}.json"
    manifest_path.write_text(
        json.dumps(
            _bundle_manifest(bundle, theory_paths, exp_paths, envelope_paths),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return {
        "manifest": manifest_path,
        "data_dir": layout["data"],
        "figure_dir": layout["figures"],
        "theory_files": theory_paths,
        "theory_envelope_files": envelope_paths,
        "experimental_files": exp_paths,
    }


def _step_xy(
    series: TheoryBandSeries, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if series.bin_edges is None:
        return series.x, values
    if len(series.bin_edges) == len(values) + 1:
        return series.bin_edges, np.concatenate([values, [values[-1]]])
    if len(series.bin_edges) == len(values):
        # Mathematica-style step series: x already includes the final edge and the last y-value is duplicated.
        return series.bin_edges, values
    return series.x, values


def _group_theory_by_label(
    series_list: Iterable[TheoryBandSeries],
) -> Dict[str, List[TheoryBandSeries]]:
    grouped: Dict[str, List[TheoryBandSeries]] = {}
    for series in series_list:
        grouped.setdefault(series.series_label, []).append(series)
    return grouped


def _group_experiment_by_state(
    series_list: Iterable[ExperimentalSeries],
) -> Dict[str, List[ExperimentalSeries]]:
    grouped: Dict[str, List[ExperimentalSeries]] = {}
    for series in series_list:
        label = series.state
        if "2S+3S" in label and "1S" in label:
            label = "3S/1S"
        elif "2S/1S" in label:
            label = "2S/1S"
        elif "3S/1S" in label:
            label = "3S/1S"
        elif "3S/2S" in label:
            label = "3S/2S"
        elif "1S" in label:
            label = "1S"
        elif "2S" in label:
            label = "2S"
        elif "3S" in label:
            label = "3S"
        grouped.setdefault(label, []).append(series)
    return grouped


def _axis_label(bundle: ObservableReferenceBundle) -> str:
    return get_x_label(bundle.observable_type)


def _y_label(bundle: ObservableReferenceBundle) -> str:
    if bundle.observable_type.startswith("RAA"):
        return r"$R_{AA}$"
    observable_id = bundle.observable_id.lower()
    if "ratio21" in observable_id:
        return r"$\frac{[\Upsilon(2S)/\Upsilon(1S)]_{\mathrm{PbPb}}}{[\Upsilon(2S)/\Upsilon(1S)]_{pp}}$"
    if "ratio31" in observable_id:
        return r"$\frac{[\Upsilon(3S)/\Upsilon(1S)]_{\mathrm{PbPb}}}{[\Upsilon(3S)/\Upsilon(1S)]_{pp}}$"
    if "ratio32" in observable_id:
        return r"$\frac{[\Upsilon(3S)/\Upsilon(2S)]_{\mathrm{PbPb}}}{[\Upsilon(3S)/\Upsilon(2S)]_{pp}}$"
    return r"Double ratio"


def _latex_state_label(label: str) -> str:
    return get_state_label(label)


def _theory_legend_label(label: str) -> str:
    return f"QTraj-NLO-{_latex_state_label(label)}"


def _state_color(label: str) -> str:
    return get_state_color(label)


def _marker_for_experiment(experiment: str) -> str:
    return {
        "ALICE": "o",
        "ATLAS": "s",
        "CMS": "D",
        "STAR": "P",
    }.get(experiment, "o")


def _observable_annotation(bundle: ObservableReferenceBundle) -> str:
    # Line 1: System + Energy (bold)
    lines = [get_system_annotation(bundle.system, bundle.energy_label)]

    # Line 2: Kappa range
    kappas: List[int] = []
    for series in bundle.theory_series:
        match = re.search(r"(?:kappa|k)(\d+)", series.source_label.lower())
        if match:
            kappas.append(int(match.group(1)))
    kappas = sorted(set(kappas))
    if kappas:
        if len(kappas) == 1:
            lines.append(rf"$\hat{{\kappa}} = {kappas[0]}$")
        else:
            lines.append(rf"$\hat{{\kappa}} \in [{min(kappas)},{max(kappas)}]$")

    # Line 3: theory-side cuts (single source of truth from registry)
    theory_note = bundle.theory_note.strip().rstrip(".")
    theory_note = re.sub(r"^Theory(?: prediction)?:\s*", "", theory_note, flags=re.IGNORECASE)
    if theory_note:
        lines.append(theory_note)

    return "\n".join(lines)


def _step_has_bins(series: TheoryBandSeries) -> bool:
    return series.bin_edges is not None and len(series.bin_edges) in {
        len(series.x),
        len(series.x) + 1,
    }


def _use_step_style(bundle: ObservableReferenceBundle) -> bool:
    if bundle.observable_type == "RAA_vs_y":
        return bool(bundle.experimental_series)
    return bundle.observable_type in {"RAA_vs_pt", "double_ratio_vs_pt"}


def _draw_theory_series(
    ax,
    series: TheoryBandSeries,
    bundle: ObservableReferenceBundle,
    *,
    label: str,
    linewidth: float,
    alpha: float,
    linestyle: str,
) -> None:
    x_vals, y_vals = _step_xy(series, series.center)
    if _step_has_bins(series) and _use_step_style(bundle):
        ax.step(
            x_vals,
            y_vals,
            where="post",
            label=label,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
        )
    else:
        ax.plot(
            series.x,
            series.center,
            label=label,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
        )


def _draw_theory_band(
    ax,
    series: TheoryBandSeries,
    bundle: ObservableReferenceBundle,
    *,
    label: str,
    alpha: float,
) -> None:
    x_vals, y_lower = _step_xy(series, series.lower)
    _, y_upper = _step_xy(series, series.upper)
    if _step_has_bins(series) and _use_step_style(bundle):
        ax.fill_between(x_vals, y_lower, y_upper, alpha=alpha, step="post", label=label)
    else:
        ax.fill_between(series.x, series.lower, series.upper, alpha=alpha, label=label)


def _panel_ymax(
    theory_series: Iterable[TheoryBandSeries], exp_series: Iterable[ExperimentalSeries]
) -> Optional[float]:
    maxima: List[float] = []
    for series in theory_series:
        finite = series.upper[np.isfinite(series.upper)]
        if finite.size:
            maxima.append(float(np.max(finite)))
    for series in exp_series:
        y = series.y
        if series.yerr_high is not None:
            y = y + series.yerr_high
        finite = y[np.isfinite(y)]
        if finite.size:
            maxima.append(float(np.max(finite)))
    if not maxima:
        return None
    ymax = max(maxima)
    if ymax <= 0.0:
        return None
    return 1.15 * ymax


def _theory_support_xmax(theory_series: Iterable[TheoryBandSeries]) -> Optional[float]:
    maxima: List[float] = []
    for series in theory_series:
        if series.bin_edges is not None:
            finite = np.asarray(series.bin_edges, dtype=np.float64)
        else:
            finite = np.asarray(series.x, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            maxima.append(float(np.max(finite)))
    if not maxima:
        return None
    xmax = max(maxima)
    if not np.isfinite(xmax):
        return None
    return float(xmax)


def _experimental_support_xmax(exp_series: Iterable[ExperimentalSeries]) -> Optional[float]:
    maxima: List[float] = []
    for series in exp_series:
        finite_x = np.asarray(series.x, dtype=np.float64)
        finite_x = finite_x[np.isfinite(finite_x)]
        if finite_x.size:
            maxima.append(float(np.max(finite_x)))
        if series.x_high is not None:
            finite_x_high = np.asarray(series.x_high, dtype=np.float64)
            finite_x_high = finite_x_high[np.isfinite(finite_x_high)]
            if finite_x_high.size:
                maxima.append(float(np.max(finite_x_high)))
    if not maxima:
        return None
    xmax = max(maxima)
    if not np.isfinite(xmax):
        return None
    return float(xmax)


def _plot_combined_bundle(ax, bundle: ObservableReferenceBundle) -> None:
    theory_grouped = _group_theory_by_label(bundle.theory_series)
    theory_envelopes = {
        series.series_label: series
        for series in _combine_theory_envelopes(bundle.theory_series)
    }

    for label, grouped_series in theory_grouped.items():
        color = _state_color(label)
        envelope = theory_envelopes.get(label)
        series = envelope if envelope is not None else grouped_series[0]
        if envelope is not None:
            _draw_theory_band(
                ax, envelope, bundle, label=f"{label} envelope", alpha=0.14
            )
        use_step = _step_has_bins(series) and _use_step_style(bundle)
        if use_step:
            x_vals, y_vals = _step_xy(series, series.center)
            ax.step(
                x_vals,
                y_vals,
                where="post",
                color=color,
                linewidth=2.0,
                alpha=0.95,
            )
        else:
            ax.plot(series.x, series.center, color=color, linewidth=2.0, alpha=0.95)

    if bundle.observable_type.startswith("double_ratio"):
        kind_rank = {"sys": 0, "stat": 1, "total": 2}
        ordered = sorted(
            bundle.experimental_series,
            key=lambda series: kind_rank.get(
                getattr(series, "uncertainty_kind", "total"), 2
            ),
        )
        for exp in ordered:
            marker = _marker_for_experiment(exp.experiment)
            xerr = None
            if exp.x_low is not None and exp.x_high is not None:
                xerr = np.vstack([exp.x - exp.x_low, exp.x_high - exp.x])

            if exp.upper_limit:
                ax.scatter(
                    exp.x,
                    exp.y,
                    marker="v",
                    s=54,
                    color="black",
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=6,
                )
                continue

            yerr = None
            if exp.yerr_low is not None and exp.yerr_high is not None:
                yerr = np.vstack([exp.yerr_low, exp.yerr_high])

            kind = getattr(exp, "uncertainty_kind", "total")
            if kind == "sys":
                ax.errorbar(
                    exp.x,
                    exp.y,
                    xerr=xerr,
                    yerr=yerr,
                    fmt="none",
                    capsize=3.0,
                    color="black",
                    elinewidth=1.6,
                    linestyle="none",
                    alpha=0.45,
                    zorder=4,
                )
            else:
                ax.errorbar(
                    exp.x,
                    exp.y,
                    xerr=xerr,
                    yerr=yerr,
                    fmt=marker,
                    markersize=5.8,
                    capsize=3.0,
                    color="black",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    elinewidth=1.1,
                    linestyle="none",
                    zorder=5,
                )
    else:
        for exp in bundle.experimental_series:
            color = _state_color(exp.state)
            marker = _marker_for_experiment(exp.experiment)
            xerr = None
            if exp.x_low is not None and exp.x_high is not None:
                xerr = np.vstack([exp.x - exp.x_low, exp.x_high - exp.x])

            if exp.upper_limit:
                ax.scatter(
                    exp.x,
                    exp.y,
                    marker="v",
                    s=46,
                    color=color,
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=4,
                )
                continue

            yerr = None
            if exp.yerr_low is not None and exp.yerr_high is not None:
                yerr = np.vstack([exp.yerr_low, exp.yerr_high])

            ax.errorbar(
                exp.x,
                exp.y,
                xerr=xerr,
                yerr=yerr,
                fmt=marker,
                markersize=5.8,
                capsize=3.0,
                color=color,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                linestyle="none",
                zorder=5,
            )


def _apply_publication_style(ax, bundle: ObservableReferenceBundle) -> None:
    ax.set_xlabel(_axis_label(bundle))
    ax.set_ylabel(_y_label(bundle))
    ax.axhline(1.0, color="0.55", linewidth=0.8, linestyle=":")
    ax.grid(alpha=0.18, linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which="minor", length=2.5, width=0.8)
    if bundle.observable_type.endswith("_vs_npart") or bundle.observable_type.endswith(
        "_vs_pt"
    ):
        ax.set_xlim(left=0.0)
    # Use centralized plot config for axis limits
    limits = config.get_limits(
        bundle.observable_type,
        system=bundle.system,
        energy=bundle.energy_label,
    )
    ax.set_xlim(limits.xlim)
    ax.set_ylim(limits.ylim)
    if bundle.observable_type.endswith("_vs_npart"):
        theory_xmax = _theory_support_xmax(bundle.theory_series)
        exp_xmax = _experimental_support_xmax(bundle.experimental_series)
        if exp_xmax is not None:
            right = float(ax.get_xlim()[1])
            x_target = exp_xmax
            if theory_xmax is not None and exp_xmax > theory_xmax:
                x_target = exp_xmax + max(2.0, 0.01 * exp_xmax)
            if x_target > right:
                ax.set_xlim(right=x_target)
    if limits.yticks:
        ax.set_yticks(limits.yticks)

    annotation = _observable_annotation(bundle)
    ann_x = 0.03
    ann_y = 0.97
    ann_ha = "left"
    if bundle.observable_type == "RAA_vs_npart":
        # Keep the annotation off the high-weight (small-N_part) region.
        ann_x = 0.50
        ann_ha = "center"
    ax.text(
        ann_x,
        ann_y,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha=ann_ha,
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.85",
        },
    )


def _add_publication_legends(fig, ax, bundle: ObservableReferenceBundle) -> None:
    from matplotlib.lines import Line2D

    theory_labels = list(_group_theory_by_label(bundle.theory_series).keys())
    experiments = sorted({series.experiment for series in bundle.experimental_series})

    if bundle.observable_type.startswith("RAA"):
        state_handles = [
            Line2D(
                [0],
                [0],
                color=_state_color(label),
                linewidth=2.0,
                label=_theory_legend_label(label),
            )
            for label in theory_labels
        ]
        if state_handles:
            state_legend = ax.legend(
                handles=state_handles,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.985),
                frameon=True,
                framealpha=0.9,
                fontsize=8.5,
                borderaxespad=0.3,
            )
            for text, handle in zip(state_legend.get_texts(), state_handles):
                text.set_color(handle.get_color())
            ax.add_artist(state_legend)

        if experiments:
            exp_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker=_marker_for_experiment(experiment),
                    linestyle="none",
                    markersize=6,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    label=experiment,
                )
                for experiment in experiments
            ]
            loc = "upper center"
            anchor = (0.50, 0.98)
            ncol = min(3, max(1, len(exp_handles)))
            fontsize = 8.0
            if bundle.observable_type == "RAA_vs_npart":
                # Keep the experiment legend inside, away from the annotation and band-heavy region.
                loc = "upper center"
                anchor = (0.52, 0.62)
                ncol = min(3, max(1, len(exp_handles)))
                fontsize = 8.0
            elif bundle.observable_type in {"RAA_vs_pt", "RAA_vs_y"}:
                loc = "upper center"
                anchor = (0.52, 0.98)
                ncol = min(3, max(1, len(exp_handles)))
                fontsize = 7.6

            ax.legend(
                handles=exp_handles,
                loc=loc,
                bbox_to_anchor=anchor,
                frameon=True,
                framealpha=0.9,
                fontsize=fontsize,
                borderaxespad=0.3,
                ncol=ncol,
                columnspacing=0.9,
                handletextpad=0.5,
            )
        return

    # Double ratios: paper-like single legend (QTraj-NLO + experiments).
    handles: List[Line2D] = []
    if theory_labels:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_state_color(theory_labels[0]),
                linewidth=2.0,
                label="QTraj-NLO",
            )
        )
    for experiment in experiments:
        series_for_exp = [
            s for s in bundle.experimental_series if s.experiment == experiment
        ]
        marker = _marker_for_experiment(experiment)
        if series_for_exp and all(
            getattr(s, "upper_limit", False) for s in series_for_exp
        ):
            marker = "v"
        label = experiment
        if any(getattr(s, "combined_state", False) for s in series_for_exp):
            combined_label = None
            for s in series_for_exp:
                state = getattr(s, "state", "")
                if "2S+3S" in state:
                    combined_label = "2S+3S"
                    break
            if combined_label:
                label = f"{experiment} ({combined_label})"
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker=marker,
                linestyle="none",
                markersize=6,
                markerfacecolor="black",
                markeredgecolor="black",
                label=label,
            )
        )

    if handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            frameon=True,
            framealpha=0.9,
            fontsize=8.5,
            borderaxespad=0.3,
        )


def _state_rank(label: str) -> int:
    text = label.lower()
    if "1s" in text:
        return 1
    if "2s" in text:
        return 2
    if "3s" in text:
        return 3
    return 99


def _minbias_label(series_group: Tuple[ExperimentalSeries, ...]) -> str:
    lows: list[float] = []
    highs: list[float] = []
    for series in series_group:
        if series.x_low is None or series.x_high is None or len(series.x_low) == 0:
            continue
        low = float(series.x_low[0])
        high = float(series.x_high[0])
        if np.isfinite(low) and np.isfinite(high):
            lows.append(low)
            highs.append(high)

    if lows and highs:
        low = int(round(min(lows)))
        high = int(round(max(highs)))
        return f"{low}-{high}%"
    return "min-bias"


# Map bundle (system, energy_label) to canonical Glauber system key.
# energy_label values come from the observable registry, not the Glauber spec strings.
_BUNDLE_TO_GLAUBER_KEY: dict = {
    ("AuAu", "200 GeV"): "auau200",
    ("PbPb", "2.76 TeV"): "pbpb2760",
    ("PbPb", "5.02 TeV"): "pbpb5023",
    ("PbPb", "5.023 TeV"): "pbpb5023",
}


def _centrality_bin_edges_percent(bundle: ObservableReferenceBundle) -> np.ndarray:
    """Return centrality bin edges in percent from bundle labels (fallback to standard bins)."""
    parsed_edges: list[float] = []
    for label in bundle.centrality_labels:
        text = str(label).strip()
        if not text.endswith("%"):
            continue
        token = text[:-1]
        if "-" in token:
            lo_text, hi_text = token.split("-", 1)
            try:
                lo = float(lo_text)
                hi = float(hi_text)
            except ValueError:
                continue
            parsed_edges.extend([lo, hi])
        else:
            try:
                parsed_edges.append(float(token))
            except ValueError:
                continue

    if parsed_edges:
        edges = np.array(sorted(set(parsed_edges)), dtype=np.float64)
    else:
        edges = np.array(
            [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            dtype=np.float64,
        )

    if edges[0] > 0.0:
        edges = np.concatenate(([0.0], edges))
    if edges[-1] < 100.0:
        edges = np.concatenate((edges, [100.0]))
    return edges


def _theory_mb_averages(bundle: ObservableReferenceBundle) -> dict:
    """
    Per-state Ncoll-weighted min-bias R_AA from theory.

    Uses the correct hard-probe formula:
        R_AA^MB = sum_k [ R_AA(c_k) * Ncoll(c_k) * Delta c_k ]
                / sum_k [ Ncoll(c_k) * Delta c_k ]
    integrated over the centrality range of the experimental min-bias data.

    The centrality bins are taken from bundle labels (0-5, 5-10, 10-20, ..., 90-100)
    and then clipped to the experiment range (e.g. STAR 0-60, CMS 0-100).
    """
    from collections import defaultdict

    theory_series = bundle.theory_series
    system_key = _BUNDLE_TO_GLAUBER_KEY.get((bundle.system, bundle.energy_label))

    if system_key is None:
        # Unknown system — fall back to simple mean (should not happen for our systems)
        by_label: dict = defaultdict(list)
        for ts in theory_series:
            by_label[ts.series_label].append(ts)
        return {
            label: (
                float(np.mean([np.mean(ts.center) for ts in group])),
                min(float(np.mean(ts.lower)) for ts in group),
                max(float(np.mean(ts.upper)) for ts in group),
            )
            for label, group in by_label.items()
        }

    from qtraj_analysis.glauber import CANONICAL_GLAUBER_SPECS, REPO_ROOT as _GREPO

    spec = CANONICAL_GLAUBER_SPECS[system_key]
    input_base = _GREPO / spec.input_base

    # Canonical b-values (ascending) and Npart (descending as b grows)
    bvals = np.asarray(spec.bvals, dtype=np.float64)
    npart_canonical = np.asarray(spec.npart_vals, dtype=np.float64)  # desc. with b

    # Ncoll(b) from nbinvsbData [col0=b_fm, col1=Ncoll]
    ncoll_tbl = np.loadtxt(input_base / "glauber-data" / "nbinvsbData.tsv")
    ncoll_at_b = np.interp(bvals, ncoll_tbl[:, 0], ncoll_tbl[:, 1])

    # Centrality fraction (0-1) at each b-value via bvscData.
    # The file stores either (b_fm, c_frac) or (c_frac, b_fm); the
    # GlauberInterpolator heuristic distinguishes them by max of col0.
    bvsc_tbl = np.loadtxt(input_base / "glauber-data" / "bvscData.tsv")
    if np.nanmax(bvsc_tbl[:, 0]) > 1.5:
        b_col, c_col = bvsc_tbl[:, 0], bvsc_tbl[:, 1]
    else:
        c_col, b_col = bvsc_tbl[:, 0], bvsc_tbl[:, 1]
    # Normalise to [0,1] fraction in case the file stores 0–100
    c_norm = c_col / max(float(c_col.max()), 1.0)
    cent_at_b = np.interp(bvals, b_col, c_norm)  # centrality fraction at each b-val

    # Determine the centrality upper limit (0-60% for STAR, 0-100% for CMS)
    c_max = 0.0
    for ms in bundle.minbias_experimental_series:
        if ms.x_high is not None and len(ms.x_high) > 0:
            val = float(ms.x_high[0])
            if np.isfinite(val) and val > 0:
                c_max = max(c_max, val / 100.0)
    if c_max <= 0.0:
        c_max = 1.0

    # Build explicit centrality bins (in fraction), then weight by Ncoll * Delta c
    cent_edges_pct = _centrality_bin_edges_percent(bundle)
    cent_edges = cent_edges_pct / 100.0
    cent_edges = cent_edges[cent_edges <= c_max + 1e-9]
    if cent_edges.size < 2:
        return {}
    cent_mid = 0.5 * (cent_edges[:-1] + cent_edges[1:])
    delta_c = cent_edges[1:] - cent_edges[:-1]

    ncoll_at_c = np.interp(cent_mid, cent_at_b, ncoll_at_b)
    bin_weights = ncoll_at_c * delta_c
    w_total = float(np.sum(bin_weights))
    if w_total <= 0.0:
        return {}

    # Group series by label (state), envelope across kappa sources
    by_label = defaultdict(list)
    for ts in theory_series:
        by_label[ts.series_label].append(ts)

    result = {}
    for label, group in by_label.items():
        centers_mb: list = []
        lowers_mb: list = []
        uppers_mb: list = []
        for ts in group:
            # Defensive sort: np.interp requires ascending x-values.
            order = np.argsort(ts.x)
            x_asc = ts.x[order]
            raa_c = np.interp(npart_canonical, x_asc, ts.center[order])
            raa_l = np.interp(npart_canonical, x_asc, ts.lower[order])
            raa_u = np.interp(npart_canonical, x_asc, ts.upper[order])
            # Convert canonical-point theory to centrality bins via interpolation.
            raa_c_mid = np.interp(cent_mid, cent_at_b, raa_c)
            raa_l_mid = np.interp(cent_mid, cent_at_b, raa_l)
            raa_u_mid = np.interp(cent_mid, cent_at_b, raa_u)
            centers_mb.append(float(np.sum(raa_c_mid * bin_weights) / w_total))
            lowers_mb.append(float(np.sum(raa_l_mid * bin_weights) / w_total))
            uppers_mb.append(float(np.sum(raa_u_mid * bin_weights) / w_total))
        result[label] = (
            float(np.mean(centers_mb)),
            min(lowers_mb),
            max(uppers_mb),
        )
    return result


def _plot_minbias_panel(ax, bundle: ObservableReferenceBundle) -> None:
    series_group = tuple(bundle.minbias_experimental_series)
    if not series_group:
        return

    # Theory averaged bands (envelope across kappa sources, unweighted Npart mean)
    theory_avgs = _theory_mb_averages(bundle)
    for label, (center, lower, upper) in sorted(
        theory_avgs.items(), key=lambda kv: _state_rank(kv[0])
    ):
        color = _state_color(label)
        ax.axhspan(lower, upper, alpha=0.25, color=color, zorder=1, linewidth=0)
        ax.axhline(center, color=color, linewidth=0.9, alpha=0.85, zorder=2)

    ordered = sorted(
        series_group,
        key=lambda s: (_state_rank(s.state), s.experiment, s.state),
    )
    offsets = np.linspace(-0.12, 0.12, num=max(1, len(ordered)))

    for offset, exp in zip(offsets, ordered):
        x = np.full_like(exp.y, fill_value=0.5 + float(offset), dtype=np.float64)
        color = _state_color(exp.state)

        if exp.upper_limit:
            ax.scatter(
                x,
                exp.y,
                marker="v",
                s=46,
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
            continue

        yerr = None
        if exp.yerr_low is not None and exp.yerr_high is not None:
            yerr = np.vstack([exp.yerr_low, exp.yerr_high])

        marker = _marker_for_experiment(exp.experiment)
        ax.errorbar(
            x,
            exp.y,
            yerr=yerr,
            fmt=marker,
            markersize=5.8,
            capsize=3.0,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            linestyle="none",
            zorder=6,
        )

    ax.axhline(1.0, color="0.55", linewidth=0.8, linestyle=":")
    ax.set_xlim(0.22, 0.78)
    ax.set_xticks([0.5])
    ax.set_xticklabels([_minbias_label(series_group)])
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which="minor", length=2.5, width=0.8, left=True)
    ax.tick_params(axis="y", left=True, labelleft=False)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Cent.")


def plot_bundle(
    bundle: ObservableReferenceBundle,
    logger: logging.Logger,
    output_root: Optional[Path | str] = None,
) -> Tuple[Path, Path] | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams.update(
            {
                "font.size": 11,
                "axes.labelsize": 13,
                "axes.titlesize": 12,
                "legend.fontsize": 9,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "axes.linewidth": 1.0,
                "savefig.transparent": False,
            }
        )
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot generation")
        return None

    layout = get_output_layout(bundle, output_root=output_root)
    layout["figures"].mkdir(parents=True, exist_ok=True)

    has_minbias_panel = (
        bundle.observable_type == "RAA_vs_npart"
        and len(bundle.minbias_experimental_series) > 0
    )
    if has_minbias_panel:
        fig, (ax, ax_minbias) = plt.subplots(
            1,
            2,
            figsize=(8.8, 5.4),
            gridspec_kw={"width_ratios": [7.5, 1.1]},
            sharey=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        ax_minbias = None

    _plot_combined_bundle(ax, bundle)
    _apply_publication_style(ax, bundle)
    ymax = _panel_ymax(bundle.theory_series, bundle.experimental_series)

    # Use centralized plot config for ylim and only auto-expand when the
    # observable explicitly allows it.
    limits = config.get_limits(
        bundle.observable_type,
        system=bundle.system,
        energy=bundle.energy_label,
    )
    if limits.allow_auto_ymax and ymax is not None and ymax > limits.ylim[1]:
        ax.set_ylim(limits.ylim[0], ymax)
    else:
        ax.set_ylim(limits.ylim)

    if ax_minbias is not None:
        _plot_minbias_panel(ax_minbias, bundle)
        ax_minbias.set_ylim(ax.get_ylim())

    _add_publication_legends(fig, ax, bundle)
    fig.tight_layout()

    pdf_path = layout["figures"] / f"{bundle.observable_id}.pdf"
    png_path = layout["figures"] / f"{bundle.observable_id}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def save_system_summary(
    bundles: Iterable[ObservableReferenceBundle],
    artifacts_by_observable: Dict[str, Dict[str, object]],
    output_root: Optional[Path | str] = None,
) -> Dict[str, Path]:
    bundle_list = list(bundles)
    if not bundle_list:
        raise ValueError("save_system_summary requires at least one exported bundle")

    first_bundle = bundle_list[0]
    layout = get_output_layout(first_bundle, output_root=output_root)
    layout["manifests"].mkdir(parents=True, exist_ok=True)

    summary_json_path = layout["manifests"] / "summary.json"
    summary_csv_path = layout["manifests"] / "summary.csv"

    observables: List[dict] = []
    unique_issues: set[str] = set()

    for bundle in bundle_list:
        artifact = artifacts_by_observable.get(bundle.observable_id, {})
        figure_pdf = artifact.get("figure_pdf")
        figure_png = artifact.get("figure_png")

        def _path_ref(value: object | None) -> str | None:
            if not value:
                return None
            path = Path(value)
            try:
                return str(path.relative_to(REPO_ROOT))
            except ValueError:
                return str(path)

        observable_entry = {
            "observable_id": bundle.observable_id,
            "observable_type": bundle.observable_type,
            "acceptance": bundle.acceptance,
            "theory_note": bundle.theory_note,
            "category": bundle.category,
            "centrality_labels": list(bundle.centrality_labels),
            "issues": list(bundle.issues),
            "theory_sources": list(bundle.theory_sources),
            "datafile_sources": list(bundle.datafile_sources),
            "theory_series_labels": sorted(
                {series.series_label for series in bundle.theory_series}
            ),
            "experimental_series": [
                {
                    "experiment": series.experiment,
                    "state": series.state,
                    "acceptance": series.acceptance,
                    "upper_limit": series.upper_limit,
                    "combined_state": series.combined_state,
                    "note": series.note,
                    "source": series.source,
                }
                for series in bundle.experimental_series
            ],
            "manifest": _path_ref(artifact.get("manifest")),
            "figure_pdf": _path_ref(figure_pdf),
            "figure_png": _path_ref(figure_png),
        }
        observables.append(observable_entry)
        unique_issues.update(bundle.issues)

    summary_payload = {
        "system": first_bundle.system,
        "energy_label": first_bundle.energy_label,
        "observable_count": len(bundle_list),
        "comparison_count": sum(
            bundle.category == "comparison" for bundle in bundle_list
        ),
        "theory_only_count": sum(
            bundle.category == "theory_only" for bundle in bundle_list
        ),
        "issues": sorted(unique_issues),
        "observables": observables,
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )

    with summary_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "observable_id",
                "category",
                "observable_type",
                "theory_series_count",
                "experimental_series_count",
                "issues_count",
                "figure_pdf",
                "figure_png",
            ),
        )
        writer.writeheader()
        for entry, bundle in zip(observables, bundle_list):
            writer.writerow(
                {
                    "observable_id": entry["observable_id"],
                    "category": entry["category"],
                    "observable_type": entry["observable_type"],
                    "theory_series_count": len(bundle.theory_series),
                    "experimental_series_count": len(bundle.experimental_series),
                    "issues_count": len(bundle.issues),
                    "figure_pdf": entry["figure_pdf"] or "",
                    "figure_png": entry["figure_png"] or "",
                }
            )

    return {
        "json": summary_json_path,
        "csv": summary_csv_path,
    }
