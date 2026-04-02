"""
Minimal QTraj Analysis Package
"""
__version__ = "0.2.0"
# Only import what works and is needed for the minimal analyzer
from qtraj_analysis.schema import Record, TrajectoryObs, RaaVsBResult
from qtraj_analysis.io import read_whitespace_table, parse_records
from qtraj_analysis.matching import build_observables
from qtraj_analysis.binning import compute_raa_vs_b, compute_raa_vs_pt, compute_raa_vs_y
from qtraj_analysis.binning_config import BinningConfig

from qtraj_analysis.feeddown import (
    build_feeddown_matrix,
    solve_primordial_sigmas,
    apply_feeddown_to_raa6
)
from qtraj_analysis.glauber import (
    CanonicalGlauberSpec,
    GlauberInterpolator,
    get_canonical_glauber_spec,
    infer_canonical_glauber_system,
    list_canonical_glauber_systems,
    load_canonical_glauber,
    load_glauber,
    load_glauber_from_input_base,
)
from qtraj_analysis.observable_registry import (
    GLOBAL_REGISTRY_ISSUES,
    get_mathematica_bin_edges,
    get_mathematica_grid,
    get_mathematica_grid_values,
    get_observable_registry,
    get_observable_spec,
    iter_registry_issues,
)
from qtraj_analysis.reference_data import (
    STANDARD_CENTRALITY_LABELS,
    ExperimentalSeries,
    ObservableReferenceBundle,
    TheoryBandSeries,
    build_reference_bundle,
    load_experimental_series,
    load_theory_series,
)
from qtraj_analysis.reference_output import (
    get_output_layout,
    plot_bundle,
    save_bundle,
    save_system_summary,
)

__all__ = [
    "Record", "TrajectoryObs", "RaaVsBResult", 
    "read_whitespace_table", "parse_records", "build_observables",
    "compute_raa_vs_b", "compute_raa_vs_pt", "compute_raa_vs_y",
    "build_feeddown_matrix", "solve_primordial_sigmas", "apply_feeddown_to_raa6",
    "CanonicalGlauberSpec", "GlauberInterpolator",
    "list_canonical_glauber_systems", "get_canonical_glauber_spec",
    "infer_canonical_glauber_system", "load_glauber", "load_canonical_glauber",
    "load_glauber_from_input_base",
    "BinningConfig",
    "get_observable_registry", "get_observable_spec",
    "get_mathematica_grid", "get_mathematica_grid_values", "get_mathematica_bin_edges",
    "iter_registry_issues", "GLOBAL_REGISTRY_ISSUES",
    "STANDARD_CENTRALITY_LABELS",
    "TheoryBandSeries", "ExperimentalSeries", "ObservableReferenceBundle",
    "load_theory_series", "load_experimental_series", "build_reference_bundle",
    "get_output_layout", "save_bundle", "plot_bundle", "save_system_summary",
]
