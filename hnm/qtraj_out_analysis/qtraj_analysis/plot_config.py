#!/usr/bin/env python3
"""
Centralized plotting configuration for qtraj analysis.

This module provides a single source of truth for all plot styling parameters:
- Axis limits (xlim, ylim) per observable type and system
- Colors for states, experiments, and kappa values
- Markers for experiments
- Legend positioning
- Font sizes

Usage:
    from qtraj_analysis.plot_config import (
        PLOT_CONFIG,
        get_axis_limits,
        get_state_color,
        get_experiment_marker,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AxisLimits:
    """Axis limits and ticks for a plot."""

    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0.0, 1.0)
    yticks: Optional[list[float]] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    allow_auto_ymax: bool = True


@dataclass
class SystemOverride:
    """System-specific axis limits override."""

    system: str  # e.g., "AuAu", "PbPb"
    energy: str  # e.g., "200 GeV", "5.02 TeV"
    limits: AxisLimits


# ============================================================================
# Default Axis Limits (no system-specific override)
# ============================================================================

DEFAULT_LIMITS: dict[str, AxisLimits] = {
    # R_AA plots
    "RAA_vs_npart": AxisLimits(
        xlim=(0.0, 400.0),
        ylim=(0.0, 1.2),
        ylabel=r"$R_{AA}$",
    ),
    "RAA_vs_pt": AxisLimits(
        xlim=(0.0, 30.0),
        ylim=(0.0, 0.75),
        yticks=[0.0, 0.25, 0.5, 0.75],
        ylabel=r"$R_{AA}$",
    ),
    "RAA_vs_y": AxisLimits(
        xlim=(-5.0, 5.0),
        ylim=(0.0, 0.75),
        yticks=[0.0, 0.25, 0.5, 0.75],
        ylabel=r"$R_{AA}$",
    ),
    # Double ratio plots
    "double_ratio_vs_npart": AxisLimits(
        xlim=(0.0, 400.0),
        ylim=(0.0, 1.2),
    ),
    "double_ratio_vs_pt": AxisLimits(
        xlim=(0.0, 30.0),
        ylim=(0.0, 0.75),
    ),
}

# Axis labels (x-axis)
AXIS_LABELS: dict[str, str] = {
    "RAA_vs_npart": r"$N_{\mathrm{part}}$",
    "RAA_vs_pt": r"$p_T$ [GeV]",
    "RAA_vs_y": r"$y$",
    "double_ratio_vs_npart": r"$N_{\mathrm{part}}$",
    "double_ratio_vs_pt": r"$p_T$ [GeV]",
}


# ============================================================================
# System-Specific Overrides
# ============================================================================

SYSTEM_OVERRIDES: dict[tuple[str, str], dict[str, AxisLimits]] = {
    # AuAu 200 GeV overrides
    ("AuAu", "200 GeV"): {
        "RAA_vs_npart": AxisLimits(
            xlim=(0.0, 380.0),
            ylim=(0.0, 1.2),
        ),
        "RAA_vs_pt": AxisLimits(
            xlim=(0.0, 10.0),
            ylim=(0.0, 0.75),
            yticks=[0.0, 0.25, 0.5, 0.75],
        ),
        "RAA_vs_y": AxisLimits(
            xlim=(-5.0, 5.0),
            ylim=(0.0, 1.2),
            yticks=[0.0, 0.25, 0.5, 0.75],
        ),
    },
    # PbPb 2.76 TeV
    ("PbPb", "2.76 TeV"): {
        "RAA_vs_npart": AxisLimits(
            xlim=(0.0, 408.0),
            ylim=(0.0, 1.30),
            yticks=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
            allow_auto_ymax=False,
        ),
        "RAA_vs_pt": AxisLimits(
            xlim=(0.0, 30.0),
            ylim=(0.0, 0.75),
            yticks=[0.0, 0.25, 0.5, 0.75],
        ),
        "RAA_vs_y": AxisLimits(
            xlim=(-6.0, 6.0),
            ylim=(0.0, 0.85),
            yticks=[0.0, 0.25, 0.5, 0.75],
        ),
        "double_ratio_vs_npart": AxisLimits(
            xlim=(0.0, 430.0),
            ylim=(0.0, 1.2),
        ),
        "double_ratio_vs_pt": AxisLimits(
            xlim=(0.0, 30.0),
            ylim=(0.0, 0.75),
        ),
    },
    # PbPb 5.02 TeV
    ("PbPb", "5.02 TeV"): {
        "RAA_vs_npart": AxisLimits(
            xlim=(0.0, 408.0),
            ylim=(0.0, 1.2),
        ),
        "RAA_vs_pt": AxisLimits(
            xlim=(0.0, 30.0),
            ylim=(0.0, 0.75),
            yticks=[0.0, 0.25, 0.5, 0.75],
        ),
        "RAA_vs_y": AxisLimits(
            xlim=(-5.0, 5.0),
            ylim=(0.0, 0.75),
            yticks=[0.0, 0.25, 0.5, 0.75],
        ),
        "double_ratio_vs_npart": AxisLimits(
            xlim=(0.0, 430.0),
            ylim=(0.0, 1.2),
        ),
        "double_ratio_vs_pt": AxisLimits(
            xlim=(0.0, 30.0),
            ylim=(0.0, 0.75),
        ),
    },
}


# ============================================================================
# Colors
# ============================================================================

STATE_COLORS: dict[str, str] = {
    # States
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
    # Full names
    "Upsilon(1S)": "#1f77b4",
    "Upsilon(2S)": "#d62728",
    "Upsilon(3S)": "#2ca02c",
    "Y(1S)": "#1f77b4",
    "Y(2S)": "#d62728",
    "Y(3S)": "#2ca02c",
    # Ratios
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

# Kappa-specific colors (for theory curves)
KAPPA_COLORS: dict[str, str] = {
    "k3": "#1f77b4",
    "k4": "#ff7f0e",
    "kappa4": "#ff7f0e",
    "kappa5": "#2ca02c",
}

# Experiment colors (for plot_publication.py)
EXPERIMENT_COLORS: dict[str, str] = {
    "alice": "#0066CC",
    "atlas": "#2CA02C",
    "cms": "#D62728",
    "star": "#9467BD",
    "ALICE": "#0066CC",
    "ATLAS": "#2CA02C",
    "CMS": "#D62728",
    "STAR": "#9467BD",
}


# ============================================================================
# Markers
# ============================================================================

EXPERIMENT_MARKERS: dict[str, str] = {
    "alice": "o",
    "atlas": "s",
    "cms": "^",
    "star": "D",
    "ALICE": "o",
    "ATLAS": "s",
    "CMS": "^",
    "STAR": "D",
}


# ============================================================================
# State Display Labels (LaTeX)
# ============================================================================

STATE_LABELS: dict[str, str] = {
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
    "2S/1S": r"$\Upsilon(2S)/\Upsilon(1S)$",
    "3S/1S": r"$\Upsilon(3S)/\Upsilon(1S)$",
    "3S/2S": r"$\Upsilon(3S)/\Upsilon(2S)$",
    "(2S/1S)_AA/(2S/1S)_pp": r"$\left(\Upsilon(2S)/\Upsilon(1S)\right)_{\mathrm{AA}} / \left(\Upsilon(2S)/\Upsilon(1S)\right)_{pp}$",
    "(3S/1S)_AA/(3S/1S)_pp": r"$\left(\Upsilon(3S)/\Upsilon(1S)\right)_{\mathrm{AA}} / \left(\Upsilon(3S)/\Upsilon(1S)\right)_{pp}$",
    "(3S/2S)_AA/(3S/2S)_pp": r"$\left(\Upsilon(3S)/\Upsilon(2S)\right)_{\mathrm{AA}} / \left(\Upsilon(3S)/\Upsilon(2S)\right)_{pp}$",
}


# ============================================================================
# System Annotations
# ============================================================================

SYSTEM_ANNOTATIONS: dict[tuple[str, str], str] = {
    "PbPb": r"$\mathbf{\mathrm{Pb{+}Pb}\ \sqrt{s_{\rm NN}} = 5.02\ TeV}$",
    "PbPb2760": r"$\mathbf{\mathrm{Pb{+}Pb}\ \sqrt{s_{\rm NN}} = 2.76\ TeV}$",
    "AuAu": r"$\mathbf{\mathrm{Au{+}Au}\ \sqrt{s_{\rm NN}} = 200\ GeV}$",
}


def get_system_annotation(system: str, energy: str) -> str:
    """Get formatted system annotation for plot."""
    key = system
    if system == "PbPb":
        if "5.02" in energy:
            key = "PbPb"
        elif "2.76" in energy:
            key = "PbPb2760"
    return SYSTEM_ANNOTATIONS.get(key, f"${system} {energy}$")


# ============================================================================
# Legend Configuration
# ============================================================================

LEGEND_DEFAULTS = {
    "fontsize": 9,
    "framealpha": 0.9,
    "borderaxespad": 0.3,
}


# ============================================================================
# Helper Functions
# ============================================================================


def get_axis_limits(
    observable_type: str,
    system: Optional[str] = None,
    energy: Optional[str] = None,
) -> AxisLimits:
    """
    Get axis limits for an observable, with optional system-specific override.

    Args:
        observable_type: e.g., "RAA_vs_npart", "double_ratio_vs_pt"
        system: e.g., "AuAu", "PbPb" (optional)
        energy: e.g., "200 GeV", "5.02 TeV" (optional)

    Returns:
        AxisLimits with appropriate xlim, ylim, etc.
    """
    # Start with defaults
    limits = DEFAULT_LIMITS.get(observable_type, AxisLimits())

    # Apply system-specific override if available
    if system and energy:
        key = (system, energy)
        if key in SYSTEM_OVERRIDES:
            system_limits = SYSTEM_OVERRIDES[key].get(observable_type)
            if system_limits:
                limits = system_limits

    return limits


def get_x_label(observable_type: str) -> str:
    """Get x-axis label for an observable type."""
    return AXIS_LABELS.get(observable_type, observable_type)


def get_state_color(label: str) -> str:
    """Get color for a state or ratio label."""
    normalized = label.replace(" ", "")
    return STATE_COLORS.get(normalized, "#1f77b4")


def get_kappa_color(kappa_label: str) -> str:
    """Get color for a kappa source label."""
    return KAPPA_COLORS.get(kappa_label.lower(), "#1f77b4")


def get_experiment_color(experiment: str) -> str:
    """Get color for an experiment."""
    return EXPERIMENT_COLORS.get(experiment, "black")


def get_experiment_marker(experiment: str) -> str:
    """Get marker style for an experiment."""
    return EXPERIMENT_MARKERS.get(experiment, "o")


def get_state_label(label: str) -> str:
    """Get LaTeX label for a state."""
    return STATE_LABELS.get(label, label)


# ============================================================================
# Runtime Configuration (can be modified)
# ============================================================================


class PlotConfig:
    """
    Mutable plot configuration that can be modified at runtime.

    Example:
        from qtraj_analysis.plot_config import config
        config.set_ylim("RAA_vs_npart", (0, 1.5))
    """

    def __init__(self):
        self._overrides: dict[str, AxisLimits] = {}
        self._system_overrides: dict[tuple[str, str], dict[str, AxisLimits]] = {}

    def set_ylim(self, observable_type: str, ylim: tuple[float, float]) -> None:
        """Set y-axis limits for an observable type (global override)."""
        if observable_type in DEFAULT_LIMITS:
            current = DEFAULT_LIMITS[observable_type]
            self._overrides[observable_type] = AxisLimits(
                xlim=current.xlim,
                ylim=ylim,
                yticks=current.yticks,
                xlabel=current.xlabel,
                ylabel=current.ylabel,
                allow_auto_ymax=current.allow_auto_ymax,
            )
        else:
            self._overrides[observable_type] = AxisLimits(ylim=ylim)

    def set_xlim(self, observable_type: str, xlim: tuple[float, float]) -> None:
        """Set x-axis limits for an observable type (global override)."""
        if observable_type in DEFAULT_LIMITS:
            current = DEFAULT_LIMITS[observable_type]
            self._overrides[observable_type] = AxisLimits(
                xlim=xlim,
                ylim=current.ylim,
                yticks=current.yticks,
                xlabel=current.xlabel,
                ylabel=current.ylabel,
                allow_auto_ymax=current.allow_auto_ymax,
            )
        else:
            self._overrides[observable_type] = AxisLimits(xlim=xlim)

    def set_limits(
        self,
        observable_type: str,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
    ) -> None:
        """Set both x and y limits for an observable type."""
        current = get_axis_limits(observable_type)
        self._overrides[observable_type] = AxisLimits(
            xlim=xlim if xlim is not None else current.xlim,
            ylim=ylim if ylim is not None else current.ylim,
            yticks=current.yticks,
            xlabel=current.xlabel,
            ylabel=current.ylabel,
            allow_auto_ymax=current.allow_auto_ymax,
        )

    def set_system_limits(
        self,
        system: str,
        energy: str,
        observable_type: str,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
    ) -> None:
        """Set limits for a specific system/energy/observable combination."""
        key = (system, energy)
        if key not in self._system_overrides:
            self._system_overrides[key] = {}
        current = get_axis_limits(observable_type, system, energy)
        self._system_overrides[key][observable_type] = AxisLimits(
            xlim=xlim if xlim is not None else current.xlim,
            ylim=ylim if ylim is not None else current.ylim,
            yticks=current.yticks,
            xlabel=current.xlabel,
            ylabel=current.ylabel,
            allow_auto_ymax=current.allow_auto_ymax,
        )

    def get_limits(
        self,
        observable_type: str,
        system: Optional[str] = None,
        energy: Optional[str] = None,
    ) -> AxisLimits:
        """Get effective axis limits, considering overrides."""
        # Check system-specific override first
        if system and energy:
            key = (system, energy)
            if key in self._system_overrides:
                if observable_type in self._system_overrides[key]:
                    return self._system_overrides[key][observable_type]

        # Check global override
        if observable_type in self._overrides:
            return self._overrides[observable_type]

        # Fall back to default
        return get_axis_limits(observable_type, system, energy)

    def reset(self) -> None:
        """Reset all overrides to defaults."""
        self._overrides.clear()
        self._system_overrides.clear()


# Global config instance
config = PlotConfig()


# ============================================================================
# Convenience: direct access to defaults
# ============================================================================

# For simple use cases, these provide direct access to defaults
PLOT_CONFIG = SYSTEM_OVERRIDES  # For compatibility
