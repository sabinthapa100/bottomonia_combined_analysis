#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qtraj_analysis.observable_registry import list_observable_ids  # noqa: E402
from qtraj_analysis.reference_data import build_reference_bundle  # noqa: E402
from qtraj_analysis.reference_output import plot_bundle, save_bundle, save_system_summary  # noqa: E402


SYSTEM_CHOICES = ("all", "auau200", "pbpb2760", "pbpb5023")
SYSTEM_PREFIXES = {
    "auau200": "auau200_",
    "pbpb2760": "pbpb2760_",
    "pbpb5023": "pbpb5023_",
}
SYSTEM_KAPPA_LABELS = {
    "auau200": {4: "kappa4", 5: "kappa5"},
    "pbpb2760": {3: "k3", 4: "k4"},
    "pbpb5023": {3: "k3", 4: "k4"},
}

SYSTEM_OUTPUT_DIRS = {
    "auau200": ("RHIC", "AuAu200GeV"),
    "pbpb2760": ("LHC", "PbPb2p76TeV"),
    "pbpb5023": ("LHC", "PbPb5p02TeV"),
}


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_system: str = "all",
    allow_system: bool = True,
    allow_kappa: bool = True,
) -> argparse.ArgumentParser:
    if allow_system:
        parser.add_argument(
            "--system",
            default=default_system,
            choices=SYSTEM_CHOICES,
            help="Published HNM system to export",
        )
    parser.add_argument(
        "--observable-id",
        help="Optional single observable id override",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write data/manifests only",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the selected system's outputs/qtraj_outputs/.../production subtree before exporting",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/qtraj_outputs",
        help="Repo-relative or absolute output root. Default: outputs/qtraj_outputs",
    )
    if allow_kappa:
        parser.add_argument(
            "--kappa",
            nargs="+",
            type=int,
            help="Optional source filter. PbPb accepts 3 4; AuAu accepts 4 5.",
        )
    return parser


def _select_ids(system_key: str, observable_id: Optional[str]) -> list[str]:
    if observable_id:
        return [observable_id]
    if system_key == "all":
        return list(list_observable_ids())
    prefix = SYSTEM_PREFIXES[system_key]
    return [observable for observable in list_observable_ids() if observable.startswith(prefix)]


def _source_labels_from_kappa(system_key: str, kappa_values: Optional[Sequence[int]]) -> Optional[list[str]]:
    if not kappa_values:
        return None
    if system_key == "all":
        raise ValueError("--kappa can only be used with a single system export")

    mapping = SYSTEM_KAPPA_LABELS[system_key]
    labels: list[str] = []
    for value in kappa_values:
        if value not in mapping:
            allowed = ", ".join(str(kappa) for kappa in sorted(mapping))
            raise ValueError(f"Unsupported kappa {value} for {system_key}; allowed values: {allowed}")
        labels.append(mapping[value])
    return labels


def run_export(
    *,
    system_key: str = "all",
    observable_id: Optional[str] = None,
    skip_plots: bool = False,
    clean: bool = False,
    kappa_values: Optional[Sequence[int]] = None,
    output_root: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    logger = logger or logging.getLogger("qtraj_out_analysis.production")
    observable_ids = _select_ids(system_key, observable_id)
    source_labels = _source_labels_from_kappa(system_key, kappa_values)

    logger.info("Exporting %d observable bundles", len(observable_ids))
    if source_labels:
        logger.info("Filtering theory sources to: %s", ", ".join(source_labels))

    if clean:
        out_root = Path(output_root or "outputs/qtraj_outputs")
        if not out_root.is_absolute():
            out_root = (REPO_ROOT / out_root).resolve()
        keys = list(SYSTEM_OUTPUT_DIRS.keys()) if system_key == "all" else [system_key]
        for key in keys:
            if key not in SYSTEM_OUTPUT_DIRS:
                continue
            collider, system_dir = SYSTEM_OUTPUT_DIRS[key]
            production_dir = out_root / collider / system_dir / "production"
            if production_dir.exists():
                logger.info("Cleaning production outputs -> %s", production_dir)
                shutil.rmtree(production_dir)

    bundles_by_system: dict[tuple[str, str], list] = defaultdict(list)
    artifacts_by_observable: dict[str, dict[str, object]] = {}

    for current_id in observable_ids:
        bundle = build_reference_bundle(current_id, source_labels=source_labels)
        saved = save_bundle(bundle, output_root=output_root)
        figure_pdf = None
        figure_png = None

        if not skip_plots:
            outputs = plot_bundle(bundle, logger, output_root=output_root)
            if outputs is not None:
                figure_pdf, figure_png = outputs
                logger.info("Saved %s plots -> %s, %s", current_id, figure_pdf, figure_png)

        artifacts_by_observable[current_id] = {
            "manifest": saved["manifest"],
            "figure_pdf": figure_pdf,
            "figure_png": figure_png,
            "theory_files": saved["theory_files"],
            "theory_envelope_files": saved["theory_envelope_files"],
            "experimental_files": saved["experimental_files"],
        }
        bundles_by_system[(bundle.system, bundle.energy_label)].append(bundle)
        logger.info("Saved %s manifest -> %s", current_id, saved["manifest"])

    for system_id, bundle_list in bundles_by_system.items():
        system_artifacts = {
            bundle.observable_id: artifacts_by_observable[bundle.observable_id]
            for bundle in bundle_list
        }
        summary_paths = save_system_summary(bundle_list, system_artifacts, output_root=output_root)
        logger.info(
            "Saved %s %s summary -> %s, %s",
            system_id[0],
            system_id[1],
            summary_paths["json"],
            summary_paths["csv"],
        )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = add_common_arguments(
        argparse.ArgumentParser(
            description="Canonical HNM qtraj production export for AuAu 200 GeV, PbPb 2.76 TeV, and PbPb 5.02 TeV."
        ),
        default_system="all",
        allow_system=True,
        allow_kappa=True,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("qtraj_out_analysis.production")
    return run_export(
        system_key=args.system,
        observable_id=args.observable_id,
        skip_plots=args.skip_plots,
        clean=args.clean,
        kappa_values=args.kappa,
        output_root=args.output_root,
        logger=logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
