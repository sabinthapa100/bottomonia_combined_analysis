from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from qtraj_analysis.exceptions import ConfigurationError
from qtraj_analysis.schema import (
    ExperimentalObservableSpec,
    GridSpec,
    RegistryIssue,
    SourceRef,
    TheoryObservableSpec,
)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "inputs").exists() and (parent / "hnm").exists():
            return parent
    raise RuntimeError(f"Could not infer repository root from {here}")


REPO_ROOT = _find_repo_root()

_MATHEMATICA_NUMBER = r"-?(?:\d+\.\d*|\d*\.\d+|\d+)(?:`)?(?:\*\^[+-]?\d+|[eE][+-]?\d+)?"
_MATHEMATICA_POINT = re.compile(r"\{\s*(" + _MATHEMATICA_NUMBER + r")\s*,")


def _ref(
    path: str,
    *,
    member: Optional[str] = None,
    variable: Optional[str] = None,
    note: Optional[str] = None,
) -> SourceRef:
    return SourceRef(path=path, member=member, variable=variable, note=note)


def _grid(
    axis: str,
    interpretation: str,
    values: Iterable[float],
    *,
    bin_edges: Optional[Iterable[float]] = None,
) -> GridSpec:
    return GridSpec(
        axis=axis,
        interpretation=interpretation,
        values=tuple(float(v) for v in values),
        bin_edges=None if bin_edges is None else tuple(float(v) for v in bin_edges),
    )


def resolve_source(ref: SourceRef) -> Path:
    return (REPO_ROOT / ref.path).resolve()


def format_source(ref: SourceRef) -> str:
    parts = [ref.path]
    if ref.member:
        parts.append(f"member={ref.member}")
    if ref.variable:
        parts.append(f"variable={ref.variable}")
    if ref.note:
        parts.append(f"note={ref.note}")
    return " | ".join(parts)


GLOBAL_REGISTRY_ISSUES: Tuple[RegistryIssue, ...] = (
    RegistryIssue(
        code="misplaced_hepdata_bundle",
        message=(
            "inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1672798-v1-csv.tar.gz "
            "is a 5.02 TeV ALICE bundle and must not be treated as canonical 2.76 TeV data."
        ),
        sources=(
            _ref(
                "inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1672798-v1-csv.tar.gz",
            ),
        ),
    ),
    RegistryIssue(
        code="misplaced_cms2019_auau200",
        message=(
            "The CMS2019 TSV files stored under inputs/qtraj_inputs/AuAu200/data "
            "match PbPb 5.02 TeV CMS centrality tables and are misplaced local leftovers."
        ),
        sources=(
            _ref("inputs/qtraj_inputs/AuAu200/data/CMS2019-Y1s-npart.tsv"),
            _ref("inputs/qtraj_inputs/AuAu200/data/CMS2019-Y2s-npart.tsv"),
        ),
    ),
    RegistryIssue(
        code="misplaced_cms2019_pbpb2760",
        message=(
            "The CMS2019 TSV files stored under inputs/qtraj_inputs/PbPb2760/data "
            "match PbPb 5.02 TeV CMS centrality tables and are misplaced local leftovers."
        ),
        sources=(
            _ref("inputs/qtraj_inputs/PbPb2760/data/CMS2019-Y1s-npart.tsv"),
            _ref("inputs/qtraj_inputs/PbPb2760/data/CMS2019-Y2s-npart.tsv"),
        ),
    ),
    RegistryIssue(
        code="atlas_5023_no_local_hepdata_tarball",
        message=(
            "PbPb 5.02 TeV ATLAS bottomonium data are present locally via the paper source bundle "
            "and notebook-extracted numeric arrays, but no original ATLAS HEPData CSV tarball was found."
        ),
        sources=(
            _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
            _ref("inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb"),
        ),
    ),
)


OBSERVABLE_REGISTRY: Dict[str, TheoryObservableSpec] = {
    "auau200_raavsnpart": TheoryObservableSpec(
        observable_id="auau200_raavsnpart",
        system="AuAu",
        energy_label="200 GeV",
        state="1S,2S,3S",
        observable_type="RAA_vs_npart",
        acceptance="Published comparison against STAR acceptance |y| < 1 and pT < 10 GeV.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/200GeV/raavsnpart-rhic-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/AuAu200/figures/raavsnpart-rhic-3d-kappa4.m"),
            _ref("inputs/qtraj_inputs/AuAu200/figures/raavsnpart-rhic-3d-kappa5.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/AuAu200/input/rhic-3d-kappa4/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/AuAu200/input/rhic-3d-kappa5/datafile-avg.gz"),
        ),
        grid=_grid(
            "Npart",
            "centers",
            (
                0.9682350180269752,
                3.7383741783700617,
                9.205831000899767,
                19.792881616610448,
                37.649319445727855,
                64.52243696535088,
                102.22223350906017,
                153.28623190336427,
                222.07926621062614,
                289.2703882064798,
                346.3015741468757,
                378.417670385485,
            ),
        ),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="STAR",
                state="Y(1S)",
                observable_type="RAA_vs_npart",
                acceptance="|y| < 1, pT < 10 GeV",
                sources=(
                    _ref(
                        "inputs/experimental_data/rhic/AuAu200GeV/HEPData-ins2112341-v2-csv.tar.gz",
                        member="HEPData-ins2112341-v2-csv/Figure2.1.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="STAR",
                state="Y(2S)",
                observable_type="RAA_vs_npart",
                acceptance="|y| < 1, pT < 10 GeV",
                sources=(
                    _ref(
                        "inputs/experimental_data/rhic/AuAu200GeV/HEPData-ins2112341-v2-csv.tar.gz",
                        member="HEPData-ins2112341-v2-csv/Figure2.2.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="STAR",
                state="Y(3S)",
                observable_type="RAA_vs_npart",
                acceptance="|y| < 1, pT < 10 GeV",
                sources=(
                    _ref(
                        "inputs/experimental_data/rhic/AuAu200GeV/HEPData-ins2112341-v2-csv.tar.gz",
                        member="HEPData-ins2112341-v2-csv/Figure2.3.csv",
                    ),
                ),
                upper_limit=True,
                note="95% CL upper limit for the 0-60% integrated centrality point.",
            ),
        ),
    ),
    "auau200_raavspt": TheoryObservableSpec(
        observable_id="auau200_raavspt",
        system="AuAu",
        energy_label="200 GeV",
        state="1S,2S",
        observable_type="RAA_vs_pt",
        acceptance="Published comparison against STAR acceptance |y| < 1 and 0-60% centrality.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/200GeV/raavspt-rhic-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/AuAu200/figures/raavspt-rhic-3d-kappa4.m"),
            _ref("inputs/qtraj_inputs/AuAu200/figures/raavspt-rhic-3d-kappa5.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/AuAu200/input/rhic-3d-kappa4/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/AuAu200/input/rhic-3d-kappa5/datafile-avg.gz"),
        ),
        grid=_grid("pT", "upper_edges_without_origin", (2.0, 5.0, 10.0), bin_edges=(0.0, 2.0, 5.0, 10.0)),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="STAR",
                state="Y(1S)",
                observable_type="RAA_vs_pt",
                acceptance="|y| < 1, 0-60% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/rhic/AuAu200GeV/HEPData-ins2112341-v2-csv.tar.gz",
                        member="HEPData-ins2112341-v2-csv/Figure4.1.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="STAR",
                state="Y(2S)",
                observable_type="RAA_vs_pt",
                acceptance="|y| < 1, 0-60% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/rhic/AuAu200GeV/HEPData-ins2112341-v2-csv.tar.gz",
                        member="HEPData-ins2112341-v2-csv/Figure4.2.csv",
                    ),
                ),
            ),
        ),
    ),
    "auau200_raavsy": TheoryObservableSpec(
        observable_id="auau200_raavsy",
        system="AuAu",
        energy_label="200 GeV",
        state="1S,2S,3S",
        observable_type="RAA_vs_y",
        acceptance="Published prediction on the rapidity grid used in the RHIC Mathematica notebook.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/200GeV/raavsy-rhic-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/AuAu200/figures/raavsy-rhic-3d-kappa4.m"),
            _ref("inputs/qtraj_inputs/AuAu200/figures/raavsy-rhic-3d-kappa5.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/AuAu200/input/rhic-3d-kappa4/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/AuAu200/input/rhic-3d-kappa5/datafile-avg.gz"),
        ),
        grid=_grid(
            "y",
            "centers",
            (-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5),
            bin_edges=(-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        ),
        experimental_observables=(),
        issues=(
            RegistryIssue(
                code="no_star_rapidity_dataset",
                message=(
                    "The canonical STAR 2022 HEPData bundle used by the paper does not contain "
                    "rapidity-differential RAA tables. The published AuAu 200 GeV y figure is theory-only."
                ),
                sources=(
                    _ref("inputs/experimental_data/rhic/AuAu200GeV/HEPData-ins2112341-v2-csv.tar.gz"),
                ),
            ),
        ),
    ),
    "pbpb2760_raavsnpart": TheoryObservableSpec(
        observable_id="pbpb2760_raavsnpart",
        system="PbPb",
        energy_label="2.76 TeV",
        state="1S,2S,3S",
        observable_type="RAA_vs_npart",
        acceptance="Published comparison with experiment-dependent acceptance: ALICE forward rapidity and CMS |y| < 2.4.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/2.76TeV/raavsnpart-lhc-2.76-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/PbPb2760/figures/raavsnpart-lhc-2.76-3d-k3.m"),
            _ref("inputs/qtraj_inputs/PbPb2760/figures/raavsnpart-lhc-2.76-3d-k4.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/PbPb2760/input/lhc-2.76-3d-k3/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/PbPb2760/input/lhc-2.76-3d-k4/datafile-avg.gz"),
        ),
        grid=_grid(
            "Npart",
            "centers",
            (
                0.970598659138994,
                3.8095328022884103,
                9.667790836200798,
                21.292761520545895,
                41.05093670919672,
                70.7871426138262,
                112.39007027735359,
                168.50283662224285,
                243.53796282970384,
                315.8591222901291,
                374.9882778251423,
                406.12232235261763,
            ),
        ),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="ALICE",
                state="Upsilon(1S)",
                observable_type="RAA_vs_npart",
                acceptance="2.5 < y < 4, pT > 0 GeV",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1297101-v1-csv.tar.gz",
                        member="HEPData-ins1297101-v1-csv/Table1.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(1S)",
                observable_type="RAA_vs_npart",
                acceptance="|y| < 2.4",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1495866-v2-csv.tar.gz",
                        member="HEPData-ins1495866-v2-csv/Table15.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(2S)",
                observable_type="RAA_vs_npart",
                acceptance="|y| < 2.4",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1495866-v2-csv.tar.gz",
                        member="HEPData-ins1495866-v2-csv/Table16.csv",
                    ),
                ),
            ),
        ),
        issues=(
            RegistryIssue(
                code="no_pbpb2760_differential_3s_npart_dataset",
                message=(
                    "No canonical local 2.76 TeV differential 3S-vs-Npart dataset was found; "
                    "the paper figure carries the 3S theory curve without a matching differential HEPData trace."
                ),
                sources=(
                    _ref("inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1495866-v2-csv.tar.gz"),
                ),
            ),
        ),
    ),
    "pbpb2760_raavspt": TheoryObservableSpec(
        observable_id="pbpb2760_raavspt",
        system="PbPb",
        energy_label="2.76 TeV",
        state="1S,2S,3S",
        observable_type="RAA_vs_pt",
        acceptance="Published comparison with CMS midrapidity pT-differential data.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/2.76TeV/raavspt-lhc-2.76-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/PbPb2760/figures/raavspt-lhc-2.76-3d-k3.m"),
            _ref("inputs/qtraj_inputs/PbPb2760/figures/raavspt-lhc-2.76-3d-k4.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/PbPb2760/input/lhc-2.76-3d-k3/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/PbPb2760/input/lhc-2.76-3d-k4/datafile-avg.gz"),
        ),
        grid=_grid("pT", "lower_edges", (0.0, 5.0, 10.0, 15.0, 20.0, 25.0), bin_edges=(0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(1S)",
                observable_type="RAA_vs_pt",
                acceptance="|y| < 2.4, 0-100% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1495866-v2-csv.tar.gz",
                        member="HEPData-ins1495866-v2-csv/Table11.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(2S)",
                observable_type="RAA_vs_pt",
                acceptance="|y| < 2.4, 0-100% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1495866-v2-csv.tar.gz",
                        member="HEPData-ins1495866-v2-csv/Table12.csv",
                    ),
                ),
            ),
        ),
        issues=(
            RegistryIssue(
                code="no_alice_2760_pt_table",
                message="The canonical 2.76 TeV ALICE local HEPData bundle has no pT-differential Upsilon(1S) RAA table.",
                sources=(
                    _ref("inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1297101-v1-csv.tar.gz"),
                ),
            ),
            RegistryIssue(
                code="no_pbpb2760_differential_3s_pt_dataset",
                message="No canonical local 2.76 TeV differential 3S-vs-pT dataset was found.",
                sources=(
                    _ref("inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1495866-v2-csv.tar.gz"),
                ),
            ),
        ),
    ),
    "pbpb2760_raavsy": TheoryObservableSpec(
        observable_id="pbpb2760_raavsy",
        system="PbPb",
        energy_label="2.76 TeV",
        state="1S,2S,3S",
        observable_type="RAA_vs_y",
        acceptance="Published comparison with ALICE forward rapidity and CMS |y| < 2.4 rapidity slices.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/2.76TeV/raavsy-lhc-2.76-3d.pdf"),
        mathematica_sources=(
            _ref(
                "inputs/qtraj_inputs/PbPb2760/figures/raavsy-lhc-2.76-3d-k3.m",
                note="The Mathematica symbol stored in this file is named raavspt even though the file is the rapidity export.",
            ),
            _ref(
                "inputs/qtraj_inputs/PbPb2760/figures/raavsy-lhc-2.76-3d-k4.m",
                note="The Mathematica symbol stored in this file is named raavspt even though the file is the rapidity export.",
            ),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/PbPb2760/input/lhc-2.76-3d-k3/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/PbPb2760/input/lhc-2.76-3d-k4/datafile-avg.gz"),
        ),
        grid=_grid(
            "y",
            "centers",
            (-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5),
            bin_edges=(-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        ),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="ALICE",
                state="Upsilon(1S)",
                observable_type="RAA_vs_y",
                acceptance="2.5 < y < 4, pT > 0 GeV, 0-90% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1297101-v1-csv.tar.gz",
                        member="HEPData-ins1297101-v1-csv/Table2.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(1S)",
                observable_type="RAA_vs_y",
                acceptance="|y| < 2.4, 0-100% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1495866-v2-csv.tar.gz",
                        member="HEPData-ins1495866-v2-csv/Table13.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(2S)",
                observable_type="RAA_vs_y",
                acceptance="|y| < 2.4, 0-100% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1495866-v2-csv.tar.gz",
                        member="HEPData-ins1495866-v2-csv/Table14.csv",
                    ),
                ),
            ),
        ),
        issues=(
            RegistryIssue(
                code="mathematica_symbol_name_mismatch",
                message=(
                    "The 2.76 TeV rapidity export files are named raavsy-*.m but the internal Mathematica symbol "
                    "is raavspt. Registry parsing uses the file content, not the symbol name."
                ),
                sources=(
                    _ref("inputs/qtraj_inputs/PbPb2760/figures/raavsy-lhc-2.76-3d-k3.m"),
                    _ref("inputs/qtraj_inputs/PbPb2760/figures/raavsy-lhc-2.76-3d-k4.m"),
                ),
            ),
            RegistryIssue(
                code="no_pbpb2760_differential_3s_y_dataset",
                message="No canonical local 2.76 TeV differential 3S-vs-y dataset was found.",
                sources=(
                    _ref("inputs/experimental_data/lhc/PbPb2.76TeV/HEPData-ins1495866-v2-csv.tar.gz"),
                ),
            ),
        ),
    ),
    "pbpb5023_raavsnpart": TheoryObservableSpec(
        observable_id="pbpb5023_raavsnpart",
        system="PbPb",
        energy_label="5.02 TeV",
        state="1S,2S,3S",
        observable_type="RAA_vs_npart",
        acceptance="Published comparison with experiment-dependent acceptance: ALICE forward, ATLAS |y| < 1.5, CMS |y| < 2.4.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/5.02TeV/raavsnpart-lhc-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/figures/raavsnpart-lhc3d-k3.m"),
            _ref("inputs/qtraj_inputs/PbPb5023/figures/raavsnpart-lhc3d-k4.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k3/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k4/datafile-avg.gz"),
        ),
        grid=_grid(
            "Npart",
            "centers",
            (
                0.970598659138994,
                3.8095328022884103,
                9.667790836200798,
                21.292761520545895,
                41.05093670919672,
                70.7871426138262,
                112.39007027735359,
                168.50283662224285,
                243.53796282970384,
                315.8591222901291,
                374.9882778251423,
                406.12232235261763,
            ),
        ),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="ALICE",
                state="Upsilon(1S)",
                observable_type="RAA_vs_npart",
                acceptance="2.5 < y < 4",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1829413-v1-csv.tar.gz",
                        member="HEPData-ins1829413-v1-csv/Table4.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="ALICE",
                state="Upsilon(2S)",
                observable_type="RAA_vs_npart",
                acceptance="2.5 < y < 4",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1829413-v1-csv.tar.gz",
                        member="HEPData-ins1829413-v1-csv/Table5.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="ATLAS",
                state="Upsilon(1S)",
                observable_type="RAA_vs_npart",
                acceptance="|y^{mumu}| < 1.5, pT^{mumu} < 30 GeV, 0-80% centrality",
                sources=(
                    _ref("inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb", variable="ATLASpts1s"),
                    _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
                ),
                note="Local numeric points are notebook-extracted from the ATLAS paper source.",
            ),
            ExperimentalObservableSpec(
                experiment="ATLAS",
                state="Upsilon(2S)",
                observable_type="RAA_vs_npart",
                acceptance="|y^{mumu}| < 1.5, pT^{mumu} < 30 GeV, 0-80% centrality",
                sources=(
                    _ref("inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb", variable="ATLASpts2s"),
                    _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
                ),
                note="Local numeric points are notebook-extracted from the ATLAS paper source.",
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(1S)",
                observable_type="RAA_vs_npart",
                acceptance="|y| < 2.4, pT < 30 GeV",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1674529-v2-csv.tar.gz",
                        member="HEPData-ins1674529-v2-csv/Table19.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(2S)",
                observable_type="RAA_vs_npart",
                acceptance="|y| < 2.4, pT < 30 GeV",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins2648528-v2-csv.tar.gz",
                        member="HEPData-ins2648528-v2-csv/Figure2,leftUPSILON(2S).csv",
                    ),
                ),
                note="The local notebook explicitly documents that PbPb 5.02 2S points were updated to the newer CMS dataset.",
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(3S)",
                observable_type="RAA_vs_npart",
                acceptance="|y| < 2.4, pT < 30 GeV",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins2648528-v2-csv.tar.gz",
                        member="HEPData-ins2648528-v2-csv/Figure2,leftUPSILON(3S).csv",
                    ),
                ),
                note="The local notebook explicitly documents that PbPb 5.02 3S points were updated to the newer CMS dataset.",
            ),
        ),
    ),
    "pbpb5023_ratio21vsnpart": TheoryObservableSpec(
        observable_id="pbpb5023_ratio21vsnpart",
        system="PbPb",
        energy_label="5.02 TeV",
        state="(2S/1S)_AA/(2S/1S)_pp",
        observable_type="double_ratio_vs_npart",
        acceptance="Published comparison against ATLAS and CMS double-ratio data.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/5.02TeV/ratio21vsnpart-lhc-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio2s1svsnpart-lhc3d-k3.m"),
            _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio2s1svsnpart-lhc3d-k4.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k3/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k4/datafile-avg.gz"),
        ),
        grid=_grid(
            "Npart",
            "centers",
            (
                0.970598659138994,
                3.8095328022884103,
                9.667790836200798,
                21.292761520545895,
                41.05093670919672,
                70.7871426138262,
                112.39007027735359,
                168.50283662224285,
                243.53796282970384,
                315.8591222901291,
                374.9882778251423,
                406.12232235261763,
            ),
        ),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(2S)/Upsilon(1S)",
                observable_type="double_ratio_vs_npart",
                acceptance="|y| < 2.4, pT < 30 GeV, pT^mu > 4 GeV",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1605750-v1-csv.tar.gz",
                        member="HEPData-ins1605750-v1-csv/Table1.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="ATLAS",
                state="Upsilon(2S)/Upsilon(1S)",
                observable_type="double_ratio_vs_npart",
                acceptance="|y^{mumu}| < 1.5, pT^{mumu} < 30 GeV, 0-80% centrality",
                sources=(
                    _ref(
                        "inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb",
                        variable="ATLASexpDataStat",
                        note="occurrence=first; uncertainty=stat",
                    ),
                    _ref(
                        "inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb",
                        variable="ATLASexpDataSys",
                        note="occurrence=first; uncertainty=sys",
                    ),
                    _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
                ),
                note="Local numeric points (stat/sys) are extracted from plotMaker-new.nb in the ATLAS paper source bundle.",
            ),
        ),
        issues=(),
    ),
    "pbpb5023_ratio31vsnpart": TheoryObservableSpec(
        observable_id="pbpb5023_ratio31vsnpart",
        system="PbPb",
        energy_label="5.02 TeV",
        state="(3S/1S)_AA/(3S/1S)_pp",
        observable_type="double_ratio_vs_npart",
        acceptance="Published comparison against CMS 3S/1S limits and the ATLAS 2S+3S proxy trace discussed in the paper text.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/5.02TeV/ratio31vsnpart-lhc-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio3s1svsnpart-lhc3d-k3.m"),
            _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio3s1svsnpart-lhc3d-k4.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k3/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k4/datafile-avg.gz"),
        ),
        grid=_grid(
            "Npart",
            "centers",
            (
                0.970598659138994,
                3.8095328022884103,
                9.667790836200798,
                21.292761520545895,
                41.05093670919672,
                70.7871426138262,
                112.39007027735359,
                168.50283662224285,
                243.53796282970384,
                315.8591222901291,
                374.9882778251423,
                406.12232235261763,
            ),
        ),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(3S)/Upsilon(1S)",
                observable_type="double_ratio_vs_npart",
                acceptance="|y| < 2.4, pT < 30 GeV, pT^mu > 4 GeV",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1605750-v1-csv.tar.gz",
                        member="HEPData-ins1605750-v1-csv/Table4.csv",
                    ),
                ),
                upper_limit=True,
                note=None,
            ),
            ExperimentalObservableSpec(
                experiment="ATLAS",
                state="Upsilon(2S+3S)/Upsilon(1S)",
                observable_type="double_ratio_vs_npart",
                acceptance="|y^{mumu}| < 1.5, pT^{mumu} < 30 GeV, 0-80% centrality",
                sources=(
                    _ref(
                        "inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb",
                        variable="ATLASexpDataStat",
                        note="occurrence=last; uncertainty=stat",
                    ),
                    _ref(
                        "inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb",
                        variable="ATLASexpDataSys",
                        note="occurrence=last; uncertainty=sys",
                    ),
                    _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
                ),
                combined_state=True,
                note=(
                    "The paper text states that the ATLAS point shown in this figure is the combined 2S+3S proxy, "
                    "expected to lie between the true 2S/1S and 3S/1S ratios."
                ),
            ),
        ),
        issues=(
            RegistryIssue(
                code="atlas_5023_ratio31_uses_combined_proxy",
                message=(
                    "The published figure uses an ATLAS combined 2S+3S / 1S proxy rather than a true 3S / 1S measurement."
                ),
                sources=(
                    _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
                    _ref("inputs/qtraj_inputs/arXiv-2305.17841v2/qtraj3d.tex"),
                ),
            ),
        ),
    ),
    "pbpb5023_raavspt": TheoryObservableSpec(
        observable_id="pbpb5023_raavspt",
        system="PbPb",
        energy_label="5.02 TeV",
        state="1S,2S,3S",
        observable_type="RAA_vs_pt",
        acceptance="Published comparison with experiment-dependent pT acceptance windows.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/5.02TeV/raavspt-lhc-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/figures/raavspt-lhc3d-k3.m"),
            _ref("inputs/qtraj_inputs/PbPb5023/figures/raavspt-lhc3d-k4.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k3/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k4/datafile-avg.gz"),
        ),
        grid=_grid("pT", "lower_edges", (0.0, 5.0, 10.0, 15.0, 20.0, 25.0), bin_edges=(0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="ALICE",
                state="Upsilon(1S)",
                observable_type="RAA_vs_pt",
                acceptance="2.5 < y < 4, 0-90% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1829413-v1-csv.tar.gz",
                        member="HEPData-ins1829413-v1-csv/Table8.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="ATLAS",
                state="Upsilon(1S)",
                observable_type="RAA_vs_pt",
                acceptance="|y^{mumu}| < 1.5, pT^{mumu} < 30 GeV, 0-80% centrality",
                sources=(
                    _ref("inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb", variable="ATLASpts1spt"),
                    _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
                ),
                note="Local numeric points are notebook-extracted from the ATLAS paper source.",
            ),
            ExperimentalObservableSpec(
                experiment="ATLAS",
                state="Upsilon(2S)",
                observable_type="RAA_vs_pt",
                acceptance="|y^{mumu}| < 1.5, pT^{mumu} < 30 GeV, 0-80% centrality",
                sources=(
                    _ref("inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb", variable="ATLASpts2spt"),
                    _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
                ),
                note="Local numeric points are notebook-extracted from the ATLAS paper source.",
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(1S)",
                observable_type="RAA_vs_pt",
                acceptance="|y| < 2.4, 0-100% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1674529-v2-csv.tar.gz",
                        member="HEPData-ins1674529-v2-csv/Table13.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(2S)",
                observable_type="RAA_vs_pt",
                acceptance="|y| < 2.4, 0-90% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins2648528-v2-csv.tar.gz",
                        member="HEPData-ins2648528-v2-csv/Figure2,rightUPSILON(2S).csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(3S)",
                observable_type="RAA_vs_pt",
                acceptance="|y| < 2.4, 0-90% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins2648528-v2-csv.tar.gz",
                        member="HEPData-ins2648528-v2-csv/Figure2,rightUPSILON(3S).csv",
                    ),
                ),
            ),
        ),
    ),
    "pbpb5023_ratio21vspt": TheoryObservableSpec(
        observable_id="pbpb5023_ratio21vspt",
        system="PbPb",
        energy_label="5.02 TeV",
        state="(2S/1S)_AA/(2S/1S)_pp",
        observable_type="double_ratio_vs_pt",
        acceptance="Published comparison against ATLAS and CMS pT-differential double ratios.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/5.02TeV/ratio21vspt-lhc-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio-2s1s-vspt-lhc3d-k3.m"),
            _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio-2s1s-vspt-lhc3d-k4.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k3/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k4/datafile-avg.gz"),
        ),
        grid=_grid("pT", "bin_edges", (0.0, 5.0, 12.0, 30.0), bin_edges=(0.0, 5.0, 12.0, 30.0)),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(2S)/Upsilon(1S)",
                observable_type="double_ratio_vs_pt",
                acceptance="|y| < 2.4, 0-100% centrality, pT^mu > 4 GeV",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1605750-v1-csv.tar.gz",
                        member="HEPData-ins1605750-v1-csv/Table2.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="ATLAS",
                state="Upsilon(2S)/Upsilon(1S)",
                observable_type="double_ratio_vs_pt",
                acceptance="|y^{mumu}| < 1.5, pT^{mumu} < 30 GeV, 0-80% centrality",
                sources=(
                    _ref(
                        "inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb",
                        variable="ATLASdata2",
                        note="uncertainty=stat",
                    ),
                    _ref(
                        "inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb",
                        variable="ATLASdata1",
                        note="uncertainty=sys",
                    ),
                    _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
                ),
                note="Local numeric points (stat/sys) are extracted from plotMaker-new.nb in the ATLAS paper source bundle.",
            ),
        ),
        issues=(),
    ),
    "pbpb5023_ratio32vspt": TheoryObservableSpec(
        observable_id="pbpb5023_ratio32vspt",
        system="PbPb",
        energy_label="5.02 TeV",
        state="(3S/2S)_AA/(3S/2S)_pp",
        observable_type="double_ratio_vs_pt",
        acceptance="Published comparison against the newer CMS 3S/2S pT-differential double ratio.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/5.02TeV/ratio32vspt-lhc-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio-3s2s-vspt-lhc3d-k3.m"),
            _ref("inputs/qtraj_inputs/PbPb5023/figures/ratio-3s2s-vspt-lhc3d-k4.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k3/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k4/datafile-avg.gz"),
        ),
        grid=_grid("pT", "bin_edges", (0.0, 4.0, 9.0, 15.0, 30.0), bin_edges=(0.0, 4.0, 9.0, 15.0, 30.0)),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(3S)/Upsilon(2S)",
                observable_type="double_ratio_vs_pt",
                acceptance="|y| < 2.4, 0-90% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins2648528-v2-csv.tar.gz",
                        member="HEPData-ins2648528-v2-csv/Figure3,right.csv",
                    ),
                ),
            ),
        ),
    ),
    "pbpb5023_raavsy": TheoryObservableSpec(
        observable_id="pbpb5023_raavsy",
        system="PbPb",
        energy_label="5.02 TeV",
        state="1S,2S,3S",
        observable_type="RAA_vs_y",
        acceptance="Published comparison with experiment-dependent rapidity coverage.",
        published_figure=_ref("inputs/qtraj_inputs/arXiv-2305.17841v2/figures/5.02TeV/raavsy-lhc-3d.pdf"),
        mathematica_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/figures/raavsy-lhc3d-k3.m"),
            _ref("inputs/qtraj_inputs/PbPb5023/figures/raavsy-lhc3d-k4.m"),
        ),
        datafile_sources=(
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k3/datafile-avg.gz"),
            _ref("inputs/qtraj_inputs/PbPb5023/lhc3d-k4/datafile-avg.gz"),
        ),
        # Mathematica exports the y-grid as bin edges including the final edge. The last y-point is duplicated so
        # the series is already step-ready (len(bin_edges) == len(values)).
        grid=_grid(
            "y",
            "bin_edges",
            (-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            bin_edges=(-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        ),
        experimental_observables=(
            ExperimentalObservableSpec(
                experiment="ALICE",
                state="Upsilon(1S)",
                observable_type="RAA_vs_y",
                acceptance="2.5 < y < 4, 0-90% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1829413-v1-csv.tar.gz",
                        member="HEPData-ins1829413-v1-csv/Table9.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="ALICE",
                state="Upsilon(2S)",
                observable_type="RAA_vs_y",
                acceptance="2.5 < y < 4, 0-90% centrality",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1829413-v1-csv.tar.gz",
                        member="HEPData-ins1829413-v1-csv/Table10.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="ATLAS",
                state="Upsilon(1S)",
                observable_type="RAA_vs_y",
                acceptance="|y^{mumu}| < 1.5, pT^{mumu} < 30 GeV, 0-80% centrality",
                sources=(
                    _ref("inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb", variable="data1sATLAS"),
                    _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
                ),
                note="Local numeric points are notebook-extracted from the ATLAS paper source.",
            ),
            ExperimentalObservableSpec(
                experiment="ATLAS",
                state="Upsilon(2S)",
                observable_type="RAA_vs_y",
                acceptance="|y^{mumu}| < 1.5, pT^{mumu} < 30 GeV, 0-80% centrality",
                sources=(
                    _ref("inputs/qtraj_inputs/PbPb5023/plotMaker-new.nb", variable="data2sATLAS"),
                    _ref("inputs/experimental_data/lhc/PbPb5TeV/arXiv-2205.03042v2/ANA-HION-2021-12-PAPER.tex"),
                ),
                note="Local numeric points are notebook-extracted from the ATLAS paper source.",
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(1S)",
                observable_type="RAA_vs_y",
                acceptance="|y| < 2.4",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1674529-v2-csv.tar.gz",
                        member="HEPData-ins1674529-v2-csv/Table16.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(2S)",
                observable_type="RAA_vs_y",
                acceptance="|y| < 2.4",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1674529-v2-csv.tar.gz",
                        member="HEPData-ins1674529-v2-csv/Table17.csv",
                    ),
                ),
            ),
            ExperimentalObservableSpec(
                experiment="CMS",
                state="Upsilon(3S)",
                observable_type="RAA_vs_y",
                acceptance="|y| < 2.4",
                sources=(
                    _ref(
                        "inputs/experimental_data/lhc/PbPb5TeV/HEPData-ins1674529-v2-csv.tar.gz",
                        member="HEPData-ins1674529-v2-csv/Table18.csv",
                    ),
                ),
                upper_limit=True,
                note="Confidence-interval table rather than a direct point measurement.",
            ),
        ),
    ),
}


def get_observable_registry() -> Dict[str, TheoryObservableSpec]:
    return dict(OBSERVABLE_REGISTRY)


def get_observable_spec(observable_id: str) -> TheoryObservableSpec:
    try:
        return OBSERVABLE_REGISTRY[observable_id]
    except KeyError as exc:
        raise ConfigurationError(
            f"Unknown observable id '{observable_id}'",
            context={"known_ids": sorted(OBSERVABLE_REGISTRY)},
        ) from exc


def list_observable_ids(system: Optional[str] = None) -> Tuple[str, ...]:
    if system is None:
        return tuple(sorted(OBSERVABLE_REGISTRY))
    return tuple(
        observable_id
        for observable_id, spec in sorted(OBSERVABLE_REGISTRY.items())
        if spec.system.lower() == system.lower()
    )


def iter_registry_issues(observable_id: Optional[str] = None) -> Tuple[RegistryIssue, ...]:
    issues: List[RegistryIssue] = list(GLOBAL_REGISTRY_ISSUES)
    if observable_id is None:
        for spec in OBSERVABLE_REGISTRY.values():
            issues.extend(spec.issues)
    else:
        issues.extend(get_observable_spec(observable_id).issues)
    return tuple(issues)


def get_mathematica_grid(observable_id: str) -> GridSpec:
    return get_observable_spec(observable_id).grid


def get_mathematica_grid_values(observable_id: str) -> np.ndarray:
    return np.asarray(get_mathematica_grid(observable_id).values, dtype=np.float64)


def get_mathematica_bin_edges(observable_id: str) -> Optional[np.ndarray]:
    grid = get_mathematica_grid(observable_id)
    if grid.bin_edges is None:
        return None
    return np.asarray(grid.bin_edges, dtype=np.float64)


def parse_mathematica_x_values(source: SourceRef | str) -> Tuple[float, ...]:
    ref = source if isinstance(source, SourceRef) else SourceRef(path=source)
    text = resolve_source(ref).read_text()
    values: List[float] = []
    seen = set()
    for raw in _MATHEMATICA_POINT.findall(text):
        value = float(raw.replace("`", "").replace("*^", "e"))
        key = round(value, 12)
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return tuple(values)
