from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from qtraj_analysis.reference_data import (
    ExperimentalSeries,
    ObservableReferenceBundle,
    TheoryBandSeries,
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


def get_output_layout(bundle: ObservableReferenceBundle, output_root: Optional[Path | str] = None) -> Dict[str, Path]:
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
    return np.column_stack([series.x, x_low, x_high, series.center, series.lower, series.upper])


def save_theory_series(series: TheoryBandSeries, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (_series_filename(series.observable_id, f"{series.series_label}_{series.source_label}", "theory.csv"))
    header = "x,x_low,x_high,y_center,y_lower,y_upper"
    np.savetxt(path, _theory_to_rows(series), delimiter=",", header=header, comments="")
    return path


def _combine_theory_envelopes(series_list: Iterable[TheoryBandSeries]) -> Tuple[TheoryBandSeries, ...]:
    envelopes: List[TheoryBandSeries] = []
    for series_label, grouped_series in _group_theory_by_label(series_list).items():
        ordered = list(grouped_series)
        first = ordered[0]
        x = first.x
        bin_edges = first.bin_edges

        for candidate in ordered[1:]:
            if x.shape != candidate.x.shape or not np.allclose(x, candidate.x, atol=1e-12, rtol=0.0):
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
        envelopes.append(
            TheoryBandSeries(
                observable_id=first.observable_id,
                series_label=series_label,
                source_label="envelope",
                source=f"Envelope across {source_labels}",
                x=x.copy(),
                center=np.mean(centers, axis=0),
                lower=np.min(lowers, axis=0),
                upper=np.max(uppers, axis=0),
                bin_edges=None if bin_edges is None else np.asarray(bin_edges, dtype=np.float64).copy(),
            )
        )
    return tuple(envelopes)


def save_theory_envelope_series(series: TheoryBandSeries, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (_series_filename(series.observable_id, series.series_label, "theory_envelope.csv"))
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
    yerr_high = np.zeros_like(series.y) if series.yerr_high is None else series.yerr_high
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
    return {
        "observable_id": bundle.observable_id,
        "system": bundle.system,
        "energy_label": bundle.energy_label,
        "observable_type": bundle.observable_type,
        "acceptance": bundle.acceptance,
        "category": bundle.category,
        "theory_sources": list(bundle.theory_sources),
        "datafile_sources": list(bundle.datafile_sources),
        "issues": list(bundle.issues),
        "centrality_labels": list(bundle.centrality_labels),
        "theory_files": [str(path.relative_to(REPO_ROOT)) for path in theory_paths],
        "theory_envelope_files": [str(path.relative_to(REPO_ROOT)) for path in envelope_paths],
        "experimental_files": [str(path.relative_to(REPO_ROOT)) for path in exp_paths],
    }


def save_bundle(bundle: ObservableReferenceBundle, output_root: Optional[Path | str] = None) -> Dict[str, Path]:
    layout = get_output_layout(bundle, output_root=output_root)
    theory_dir = layout["data"] / "theory"
    theory_envelope_dir = layout["data"] / "theory_envelopes"
    exp_dir = layout["data"] / "experiment"
    layout["manifests"].mkdir(parents=True, exist_ok=True)

    theory_paths = [save_theory_series(series, theory_dir) for series in bundle.theory_series]
    envelope_paths = [
        save_theory_envelope_series(series, theory_envelope_dir)
        for series in _combine_theory_envelopes(bundle.theory_series)
    ]
    exp_paths = [save_experimental_series(series, exp_dir) for series in bundle.experimental_series]

    manifest_path = layout["manifests"] / f"{bundle.observable_id}.json"
    manifest_path.write_text(
        json.dumps(_bundle_manifest(bundle, theory_paths, exp_paths, envelope_paths), indent=2, sort_keys=True)
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


def _step_xy(series: TheoryBandSeries, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if series.bin_edges is None:
        return series.x, values
    if len(series.bin_edges) == len(values) + 1:
        return series.bin_edges, np.concatenate([values, [values[-1]]])
    if len(series.bin_edges) == len(values):
        # Mathematica-style step series: x already includes the final edge and the last y-value is duplicated.
        return series.bin_edges, values
    return series.x, values


def _group_theory_by_label(series_list: Iterable[TheoryBandSeries]) -> Dict[str, List[TheoryBandSeries]]:
    grouped: Dict[str, List[TheoryBandSeries]] = {}
    for series in series_list:
        grouped.setdefault(series.series_label, []).append(series)
    return grouped


def _group_experiment_by_state(series_list: Iterable[ExperimentalSeries]) -> Dict[str, List[ExperimentalSeries]]:
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
    mapping = {
        "RAA_vs_npart": r"$N_{\mathrm{part}}$",
        "RAA_vs_pt": r"$p_T$ [GeV]",
        "RAA_vs_y": r"$y$",
        "double_ratio_vs_npart": r"$N_{\mathrm{part}}$",
        "double_ratio_vs_pt": r"$p_T$ [GeV]",
    }
    return mapping.get(bundle.observable_type, bundle.observable_type)


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
    normalized = label.strip()
    mapping = {
        "1S": r"$\Upsilon(1S)$",
        "2S": r"$\Upsilon(2S)$",
        "3S": r"$\Upsilon(3S)$",
        "1P": r"$\chi_b(1P)$",
        "2P": r"$\chi_b(2P)$",
        "1P0": r"$\chi_{b0}(1P)$",
        "1P1": r"$\chi_{b1}(1P)$",
        "1P2": r"$\chi_{b2}(1P)$",
        "2P0": r"$\chi_{b0}(2P)$",
        "2P1": r"$\chi_{b1}(2P)$",
        "2P2": r"$\chi_{b2}(2P)$",
        "Upsilon(1S)": r"$\Upsilon(1S)$",
        "Upsilon(2S)": r"$\Upsilon(2S)$",
        "Upsilon(3S)": r"$\Upsilon(3S)$",
        "Y(1S)": r"$\Upsilon(1S)$",
        "Y(2S)": r"$\Upsilon(2S)$",
        "Y(3S)": r"$\Upsilon(3S)$",
        "Upsilon(2S)/Upsilon(1S)": r"$\Upsilon(2S)/\Upsilon(1S)$",
        "Upsilon(3S)/Upsilon(1S)": r"$\Upsilon(3S)/\Upsilon(1S)$",
        "Upsilon(3S)/Upsilon(2S)": r"$\Upsilon(3S)/\Upsilon(2S)$",
        "2S/1S": r"$\Upsilon(2S)/\Upsilon(1S)$",
        "3S/1S": r"$\Upsilon(3S)/\Upsilon(1S)$",
        "3S/2S": r"$\Upsilon(3S)/\Upsilon(2S)$",
        "(2S/1S)_AA/(2S/1S)_pp": r"$\left(\Upsilon(2S)/\Upsilon(1S)\right)_{\mathrm{AA}} / \left(\Upsilon(2S)/\Upsilon(1S)\right)_{pp}$",
        "(3S/1S)_AA/(3S/1S)_pp": r"$\left(\Upsilon(3S)/\Upsilon(1S)\right)_{\mathrm{AA}} / \left(\Upsilon(3S)/\Upsilon(1S)\right)_{pp}$",
        "(3S/2S)_AA/(3S/2S)_pp": r"$\left(\Upsilon(3S)/\Upsilon(2S)\right)_{\mathrm{AA}} / \left(\Upsilon(3S)/\Upsilon(2S)\right)_{pp}$",
    }
    if normalized in mapping:
        return mapping[normalized]
    if "2S+3S" in normalized and "1S" in normalized:
        return r"$(\Upsilon(2S)+\Upsilon(3S))/\Upsilon(1S)$"
    return normalized


def _theory_legend_label(label: str) -> str:
    return f"QTraj-NLO-{_latex_state_label(label)}"


def _state_color(label: str) -> str:
    normalized = label.replace(" ", "")
    color_map = {
        "1S": "#1f77b4",
        "2S": "#d62728",
        "3S": "#2ca02c",
        "1P": "#9467bd",
        "2P": "#8c564b",
        "1P0": "#9467bd",
        "1P1": "#9467bd",
        "1P2": "#9467bd",
        "2P0": "#8c564b",
        "2P1": "#8c564b",
        "2P2": "#8c564b",
        "Upsilon(1S)": "#1f77b4",
        "Upsilon(2S)": "#d62728",
        "Upsilon(3S)": "#2ca02c",
        "Y(1S)": "#1f77b4",
        "Y(2S)": "#d62728",
        "Y(3S)": "#2ca02c",
        "Upsilon(2S)/Upsilon(1S)": "#d62728",
        "Upsilon(3S)/Upsilon(1S)": "#2ca02c",
        "Upsilon(3S)/Upsilon(2S)": "#8c564b",
        "2S/1S": "#1f77b4",
        "3S/1S": "#1f77b4",
        "3S/2S": "#1f77b4",
        "(2S/1S)_AA/(2S/1S)_pp": "#d62728",
        "(3S/1S)_AA/(3S/1S)_pp": "#2ca02c",
        "(3S/2S)_AA/(3S/2S)_pp": "#8c564b",
    }
    return color_map.get(normalized, "#1f77b4")


def _marker_for_experiment(experiment: str) -> str:
    return {
        "ALICE": "o",
        "ATLAS": "s",
        "CMS": "D",
        "STAR": "P",
    }.get(experiment, "o")


def _observable_annotation(bundle: ObservableReferenceBundle) -> str:
    system_map = {
        ("PbPb", "5.02 TeV"): (r"$\mathrm{Pb{+}Pb}$", r"$\sqrt{s_{NN}} = 5.02\ \mathrm{TeV}$"),
        ("PbPb", "2.76 TeV"): (r"$\mathrm{Pb{+}Pb}$", r"$\sqrt{s_{NN}} = 2.76\ \mathrm{TeV}$"),
        ("AuAu", "200 GeV"): (r"$\mathrm{Au{+}Au}$", r"$\sqrt{s_{NN}} = 200\ \mathrm{GeV}$"),
    }
    lines = list(system_map.get((bundle.system, bundle.energy_label), (bundle.system, bundle.energy_label)))

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

    return "\n".join(lines)


def _step_has_bins(series: TheoryBandSeries) -> bool:
    return series.bin_edges is not None and len(series.bin_edges) in {len(series.x), len(series.x) + 1}


def _use_step_style(bundle: ObservableReferenceBundle) -> bool:
    if bundle.system == "AuAu" and bundle.energy_label == "200 GeV":
        return False
    return bundle.observable_type in {"RAA_vs_pt", "RAA_vs_y", "double_ratio_vs_pt"}


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
        ax.step(x_vals, y_vals, where="post", label=label, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
    else:
        ax.plot(series.x, series.center, label=label, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def _draw_theory_band(ax, series: TheoryBandSeries, bundle: ObservableReferenceBundle, *, label: str, alpha: float) -> None:
    x_vals, y_lower = _step_xy(series, series.lower)
    _, y_upper = _step_xy(series, series.upper)
    if _step_has_bins(series) and _use_step_style(bundle):
        ax.fill_between(x_vals, y_lower, y_upper, alpha=alpha, step="post", label=label)
    else:
        ax.fill_between(series.x, series.lower, series.upper, alpha=alpha, label=label)


def _panel_ymax(theory_series: Iterable[TheoryBandSeries], exp_series: Iterable[ExperimentalSeries]) -> Optional[float]:
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


def _plot_combined_bundle(ax, bundle: ObservableReferenceBundle) -> None:
    theory_grouped = _group_theory_by_label(bundle.theory_series)
    theory_envelopes = {
        series.series_label: series for series in _combine_theory_envelopes(bundle.theory_series)
    }

    for label, grouped_series in theory_grouped.items():
        color = _state_color(label)
        envelope = theory_envelopes.get(label)
        if envelope is not None:
            _draw_theory_band(ax, envelope, bundle, label=f"{label} envelope", alpha=0.14)
            x_vals, y_vals = _step_xy(envelope, envelope.center)
            if _step_has_bins(envelope) and _use_step_style(bundle):
                ax.step(x_vals, y_vals, where="post", color=color, linewidth=2.0, alpha=0.95)
            else:
                ax.plot(envelope.x, envelope.center, color=color, linewidth=2.0, alpha=0.95)
        else:
            first = grouped_series[0]
            x_vals, y_vals = _step_xy(first, first.center)
            if _step_has_bins(first) and _use_step_style(bundle):
                ax.step(x_vals, y_vals, where="post", color=color, linewidth=2.0, alpha=0.95)
            else:
                ax.plot(first.x, first.center, color=color, linewidth=2.0, alpha=0.95)

    if bundle.observable_type.startswith("double_ratio"):
        kind_rank = {"sys": 0, "stat": 1, "total": 2}
        ordered = sorted(
            bundle.experimental_series,
            key=lambda series: kind_rank.get(getattr(series, "uncertainty_kind", "total"), 2),
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
                    color="#b22222",
                    elinewidth=1.6,
                    linestyle="none",
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
    if bundle.observable_type.endswith("_vs_npart") or bundle.observable_type.endswith("_vs_pt"):
        ax.set_xlim(left=0.0)
    if bundle.observable_type.endswith("_vs_npart"):
        xmax = _theory_support_xmax(bundle.theory_series)
        if xmax is not None and xmax > 0.0:
            ax.set_xlim(right=xmax)
    if bundle.observable_type == "RAA_vs_npart":
        ax.set_ylim(0.0, 1.0)
    elif bundle.observable_type == "RAA_vs_pt":
        pt_max = 10.0 if bundle.system == "AuAu" and bundle.energy_label == "200 GeV" else 30.0
        ax.set_xlim(0.0, pt_max)
        ax.set_ylim(0.0, 0.75)
    elif bundle.observable_type == "RAA_vs_y":
        ax.set_xlim(-5.0, 5.0)
        if bundle.system == "AuAu" and bundle.energy_label == "200 GeV":
            ax.set_ylim(0.0, 1.10)
        else:
            ax.set_ylim(0.0, 0.75)
    elif bundle.observable_type == "double_ratio_vs_npart":
        ax.set_ylim(0.0, 1.2)
    elif bundle.observable_type == "double_ratio_vs_pt":
        ax.set_xlim(0.0, 30.0)
        ax.set_ylim(0.0, 1.0)

    annotation = _observable_annotation(bundle)
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
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
        series_for_exp = [s for s in bundle.experimental_series if s.experiment == experiment]
        marker = _marker_for_experiment(experiment)
        if series_for_exp and all(getattr(s, "upper_limit", False) for s in series_for_exp):
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

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    _plot_combined_bundle(ax, bundle)
    _apply_publication_style(ax, bundle)
    ymax = _panel_ymax(bundle.theory_series, bundle.experimental_series)
    if bundle.observable_type == "RAA_vs_npart":
        upper = 1.0
        if ymax is not None and ymax > 0.95:
            upper = min(1.2, max(1.0, 1.05 * ymax))
        ax.set_ylim(0.0, upper)
    elif bundle.observable_type == "RAA_vs_pt":
        upper = 0.75
        if ymax is not None and ymax > 0.72:
            upper = min(0.9, max(0.75, 1.05 * ymax))
        ax.set_ylim(0.0, upper)
    elif bundle.observable_type == "RAA_vs_y":
        if bundle.system == "AuAu" and bundle.energy_label == "200 GeV":
            ax.set_ylim(0.0, 1.10)
        else:
            upper = 0.75
            if ymax is not None and ymax > 0.72:
                upper = min(0.9, max(0.75, 1.05 * ymax))
            ax.set_ylim(0.0, upper)
    elif bundle.observable_type == "double_ratio_vs_npart":
        upper = 1.2
        if ymax is not None and ymax > 1.12:
            upper = min(1.3, max(1.2, 1.05 * ymax))
        ax.set_ylim(0.0, upper)
    elif bundle.observable_type == "double_ratio_vs_pt":
        upper = 1.0
        if ymax is not None and ymax > 0.95:
            upper = min(1.15, max(1.0, 1.05 * ymax))
        ax.set_ylim(0.0, upper)
    elif ymax is not None:
        ax.set_ylim(0.0, ymax)
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

        observable_entry = {
            "observable_id": bundle.observable_id,
            "observable_type": bundle.observable_type,
            "acceptance": bundle.acceptance,
            "category": bundle.category,
            "centrality_labels": list(bundle.centrality_labels),
            "issues": list(bundle.issues),
            "theory_sources": list(bundle.theory_sources),
            "datafile_sources": list(bundle.datafile_sources),
            "theory_series_labels": sorted({series.series_label for series in bundle.theory_series}),
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
            "manifest": str(Path(artifact["manifest"]).relative_to(REPO_ROOT)) if artifact.get("manifest") else None,
            "figure_pdf": str(Path(figure_pdf).relative_to(REPO_ROOT)) if figure_pdf else None,
            "figure_png": str(Path(figure_png).relative_to(REPO_ROOT)) if figure_png else None,
        }
        observables.append(observable_entry)
        unique_issues.update(bundle.issues)

    summary_payload = {
        "system": first_bundle.system,
        "energy_label": first_bundle.energy_label,
        "observable_count": len(bundle_list),
        "comparison_count": sum(bundle.category == "comparison" for bundle in bundle_list),
        "theory_only_count": sum(bundle.category == "theory_only" for bundle in bundle_list),
        "issues": sorted(unique_issues),
        "observables": observables,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")

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
