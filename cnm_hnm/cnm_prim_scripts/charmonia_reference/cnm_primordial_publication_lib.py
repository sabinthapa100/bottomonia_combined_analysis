#!/usr/bin/env python3
"""Publication-style CNM x primordial x experimental plotting helpers.

Designed for step-by-step notebooks in `cnm_primordial_notebooks`:
- LHC pPb (5.02 / 8.16 TeV)
- RHIC dAu (200 GeV)

Additive utility module; does not modify legacy workflows.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]

# Make imports robust from notebook or script execution
for p in (
    PROJECT,
    PROJECT / "eloss_code",
    PROJECT / "npdf_code",
    PROJECT / "cnm_combine",
    PROJECT / "cnm_withExperimental",
    PROJECT / "experimental_helpers",
    PROJECT / "primordial_code",
    PROJECT / "primordial_notebooks",
):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from cnm_combine_fast_nuclabs import CNMCombineFast
from cnm_configs import LHC_CONFIG, RHIC_CONFIG
from primordial_module import ReaderConfig, Style, Y_WINDOW_BACKWARD, Y_WINDOW_CENTRAL, Y_WINDOW_FORWARD
from primordial_notebooks.primordial_only_eloss_glauber_test import (
    _load_primordial_combos,
    _tauform_for_form,
    _validate_glauber_maps,
    primordial_vs_cent_from_vs_y,
    primordial_vs_pT_all,
    primordial_vs_y_all,
)
from primordial_code.glauber_bridge import GlauberBridgeConfig, generate_primordial_glauber_maps
from experimental_helpers.hep_cent_data_reader import HEPCentDataReader, _ycms_center
from experimental_helpers.lhc_minbias_data_reader import (
    get_RpA_vs_pT,
    get_RpA_vs_y,
    get_lhcb_digitized_data,
    get_psi2s_8tev_preliminary,
    load_minbias_data,
)
from experimental_helpers.rhic_data_reader import RHICDataReader


STATE_TO_CNM = {
    "jpsi_1S": "Jpsi",
    "chicJ_1P": "1P",
    "psi_2S": "psi2S",
}

STATE_LABELS = {
    "jpsi_1S": r"$J/\psi(1S)$",
    "chicJ_1P": r"$\chi_c(1P)$",
    "psi_2S": r"$\psi(2S)$",
}

M_STATE_GEV = {
    "jpsi_1S": 3.0969,
    "chicJ_1P": 3.5107,
    "psi_2S": 3.6861,
}

# (central, low, high)
ABS_SIGMA_LHC = {
    "jpsi_1S": (0.5, 0.0, 1.0),
    "chicJ_1P": (0.5, 0.0, 1.0),
    "psi_2S": (0.5, 0.0, 1.0),
}
ABS_SIGMA_RHIC = {
    "jpsi_1S": (4.2, 3.0, 5.5),
    "chicJ_1P": (4.2, 3.0, 5.5),
    "psi_2S": (4.2, 3.0, 5.5),
}

THEORY_STYLE = {
    "cnm_total": {"color": "gray", "ls": "-", "label": "CNM"},
    "prim_pert": {"color": "tab:red", "ls": "--", "label": "Prim (Pert)"},
    "prim_npwlc": {"color": "tab:green", "ls": "--", "label": "Prim (NPWLC)"},
    "comb_pert": {"color": "tab:red", "ls": "-", "label": r"CNM$\times$Prim (Pert)"},
    "comb_npwlc": {"color": "tab:green", "ls": "-", "label": r"CNM$\times$Prim (NPWLC)"},
    "cnm_alt": {"color": "#7B2CBF", "ls": "-.", "label": "CNM (alt)"},
}

CNM_COMPONENT_STYLE = {
    "npdf": {"color": "#E69F00", "ls": "-", "label": "nPDF"},
    "eloss": {"color": "#8C564B", "ls": "-", "label": "ELoss"},
    "broad": {"color": "#2B2BEF", "ls": "-", "label": r"$p_T$-Broad"},
    "eloss_broad": {"color": "#222222", "ls": "-", "label": r"ELoss+$p_T$-Broad"},
}

TEXT_STYLE_DEFAULTS = {
    "system_color": "tab:blue",
    "system_fontsize": 11.0,
    "system_weight": "bold",
    "state_color": "black",
    "state_fontsize": 11.0,
    "state_weight": "bold",
    "tag_color": "tab:blue",
    "tag_fontsize": 10.0,
    "tag_weight": "bold",
    "note_fontsize": 9.5,
    "note_box_alpha": 0.82,
}

MINBIAS_TAGS = {"MB", "0-100%", "0-100%(MB)", "0-100% (MB)", "Minimum Bias"}


def apply_style_overrides(
    theory_style_overrides: Optional[Mapping[str, Mapping[str, object]]] = None,
    component_style_overrides: Optional[Mapping[str, Mapping[str, object]]] = None,
    text_style_overrides: Optional[Mapping[str, object]] = None,
) -> None:
    """Minimal runtime style overrides for notebook tuning."""
    if theory_style_overrides:
        for key, vals in theory_style_overrides.items():
            if key in THEORY_STYLE and isinstance(vals, Mapping):
                THEORY_STYLE[key].update(dict(vals))
    if component_style_overrides:
        for key, vals in component_style_overrides.items():
            if key in CNM_COMPONENT_STYLE and isinstance(vals, Mapping):
                CNM_COMPONENT_STYLE[key].update(dict(vals))
    if text_style_overrides:
        TEXT_STYLE_DEFAULTS.update(dict(text_style_overrides))


@dataclass
class SystemSetup:
    system: str  # "LHC" or "RHIC"
    energy: str  # "5.02", "8.16", "200"
    cent_bins: List[Tuple[int, int]]
    y_windows: Dict[str, Tuple[float, float]]
    y_edges: np.ndarray
    pt_edges: np.ndarray
    pt_range_integrated: Tuple[float, float]
    r_label: str


@dataclass
class PrimordialProducts:
    available: bool
    reason: str
    vs_y: dict
    vs_pt: dict
    vs_cent: dict
    tauform: Dict[str, Tuple[float, float]]


@dataclass
class CNMStateProducts:
    state: str
    primary_variant: str  # "no_abs" or "abs"
    abs_sigma_mb: float
    abs_sigma_lo_hi: Tuple[float, float]
    abs_mode: str
    y_cent: np.ndarray
    pt_cent: np.ndarray
    cent_mids: np.ndarray
    cent_edges: np.ndarray
    y_by_variant: Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]
    pt_by_variant: Dict[str, Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]]
    cent_by_variant: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]]


def _resolve_abs_sigma_tuple(
    state: str,
    setup: "SystemSetup",
    abs_sigma_map: Optional[Mapping[str, object]] = None,
) -> Tuple[float, float, float]:
    base = ABS_SIGMA_LHC if setup.system == "LHC" else ABS_SIGMA_RHIC
    sigma_c, sigma_lo, sigma_hi = base[state]
    if abs_sigma_map is None or state not in abs_sigma_map:
        return float(sigma_c), float(sigma_lo), float(sigma_hi)

    raw = abs_sigma_map[state]
    if isinstance(raw, Mapping):
        c = raw.get("central", raw.get("mid", sigma_c))
        lo = raw.get("low", sigma_lo)
        hi = raw.get("high", sigma_hi)
        sigma_c, sigma_lo, sigma_hi = float(c), float(lo), float(hi)
    else:
        vals = tuple(raw)  # type: ignore[arg-type]
        if len(vals) != 3:
            raise ValueError(f"abs_sigma_map['{state}'] must have 3 values: (central, low, high)")
        sigma_c, sigma_lo, sigma_hi = float(vals[0]), float(vals[1]), float(vals[2])

    lo_fix = min(sigma_lo, sigma_hi)
    hi_fix = max(sigma_lo, sigma_hi)
    return float(sigma_c), float(lo_fix), float(hi_fix)


def configure_matplotlib() -> None:
    Style.apply()
    mpl.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )


def canonical_cent(tag: str) -> str:
    s = str(tag).replace("–", "-").replace("%", "").strip()
    if not s or "mb" in s.lower() or "minimum" in s.lower():
        return "MB"
    if "-" not in s:
        return f"{s}%"
    lo, hi = [p.strip() for p in s.split("-", 1)]
    return f"{int(float(lo))}-{int(float(hi))}%"


def iter_ywins(y_wins: Mapping[str, Tuple[float, float]]) -> Iterable[Tuple[str, float, float]]:
    for k, win in y_wins.items():
        y0, y1 = tuple(win)
        yield k, float(y0), float(y1)


def step_from_centers(x_cent: Sequence[float], vals: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    x_cent = np.asarray(x_cent, float)
    vals = np.asarray(vals, float)
    if x_cent.size == 1:
        return np.array([x_cent[0] - 0.5, x_cent[0] + 0.5], float), np.array([vals[0], vals[0]], float)
    dx = np.diff(x_cent)
    if np.allclose(dx, dx[0]):
        edges = np.concatenate(([x_cent[0] - 0.5 * dx[0]], x_cent + 0.5 * dx[0]))
    else:
        edges = np.empty(x_cent.size + 1, float)
        edges[1:-1] = 0.5 * (x_cent[:-1] + x_cent[1:])
        edges[0] = x_cent[0] - 0.5 * (x_cent[1] - x_cent[0])
        edges[-1] = x_cent[-1] + 0.5 * (x_cent[-1] - x_cent[-2])
    return edges, np.concatenate([vals, vals[-1:]])


def align_to_ref(x_src: np.ndarray, vals: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    x_src = np.asarray(x_src, float)
    vals = np.asarray(vals, float)
    x_ref = np.asarray(x_ref, float)
    if x_src.size == x_ref.size and np.allclose(x_src, x_ref):
        return vals
    return np.interp(x_ref, x_src, vals, left=vals[0], right=vals[-1])


def combine_bands(
    cnm_c: np.ndarray,
    cnm_lo: np.ndarray,
    cnm_hi: np.ndarray,
    pr_c: np.ndarray,
    pr_lo: np.ndarray,
    pr_hi: np.ndarray,
    mode: str = "quadrature",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cnm_c = np.asarray(cnm_c, float)
    cnm_lo = np.asarray(cnm_lo, float)
    cnm_hi = np.asarray(cnm_hi, float)
    pr_c = np.asarray(pr_c, float)
    pr_lo = np.asarray(pr_lo, float)
    pr_hi = np.asarray(pr_hi, float)

    rc = cnm_c * pr_c

    if mode == "envelope":
        cands = np.stack([cnm_lo * pr_lo, cnm_lo * pr_hi, cnm_hi * pr_lo, cnm_hi * pr_hi], axis=0)
        return rc, np.min(cands, axis=0), np.max(cands, axis=0)

    # quadrature
    dcnm_lo = np.maximum(0.0, cnm_c - cnm_lo)
    dcnm_hi = np.maximum(0.0, cnm_hi - cnm_c)
    dpr_lo = np.maximum(0.0, pr_c - pr_lo)
    dpr_hi = np.maximum(0.0, pr_hi - pr_c)

    d_lo = np.sqrt((pr_c * dcnm_lo) ** 2 + (cnm_c * dpr_lo) ** 2)
    d_hi = np.sqrt((pr_c * dcnm_hi) ** 2 + (cnm_c * dpr_hi) ** 2)

    return rc, np.maximum(0.0, rc - d_lo), rc + d_hi


def get_system_setup(system: str, energy: str) -> SystemSetup:
    system_u = system.upper()
    if system_u == "LHC":
        if energy not in ("5.02", "8.16"):
            raise ValueError("LHC energy must be '5.02' or '8.16'.")
        cfg = LHC_CONFIG
        cent_bins = [(int(a), int(b)) for (a, b) in cfg["cent_bins_plotting"]]
        y_windows = {
            "backward": (cfg["y_windows"][0][0], cfg["y_windows"][0][1]),
            "central": (cfg["y_windows"][1][0], cfg["y_windows"][1][1]),
            "forward": (cfg["y_windows"][2][0], cfg["y_windows"][2][1]),
        }
        # Keep match with primordial grids.
        y_edges = np.linspace(-5.0, 5.0, 21)
        pt_edges = np.asarray(cfg["pt_edges"], float)
        pt_range = tuple(float(x) for x in cfg["pt_range_integrated"][0])
        return SystemSetup("LHC", energy, cent_bins, y_windows, y_edges, pt_edges, pt_range, r_label=r"$R_{pA}$")

    if system_u == "RHIC":
        if energy != "200":
            raise ValueError("RHIC energy must be '200'.")
        cfg = RHIC_CONFIG
        cent_bins = [(int(a), int(b)) for (a, b) in cfg["cent_bins_plotting"]]
        y_windows = {
            "backward": (cfg["y_windows"][0][0], cfg["y_windows"][0][1]),
            "central": (cfg["y_windows"][1][0], cfg["y_windows"][1][1]),
            "forward": (cfg["y_windows"][2][0], cfg["y_windows"][2][1]),
        }
        y_edges = np.asarray(cfg["y_edges"], float)
        pt_edges = np.asarray(cfg["pt_edges"], float)
        pt_range = tuple(float(x) for x in cfg["pt_range_integrated"][0])
        return SystemSetup("RHIC", energy, cent_bins, y_windows, y_edges, pt_edges, pt_range, r_label=r"$R_{dA}$")

    raise ValueError("system must be 'LHC' or 'RHIC'.")


def _pack_axis_components(
    bands: Mapping[str, Tuple[dict, dict, dict]],
    components: Sequence[str],
    tags: Sequence[str],
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    out: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {c: {} for c in components}
    for comp in components:
        if comp not in bands:
            continue
        rc, lo, hi = bands[comp]
        for tag in tags:
            tcan = canonical_cent(tag)
            key = tag if tag in rc else ("MB" if tcan == "MB" and "MB" in rc else tcan)
            if key in rc:
                out[comp][canonical_cent(tag)] = (
                    np.asarray(rc[key], float),
                    np.asarray(lo[key], float),
                    np.asarray(hi[key], float),
                )
    return out


def _pack_cent_component(
    comp_arr: Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float],
    cent_bins: Sequence[Tuple[int, int]],
) -> Dict[str, Tuple[float, float, float]]:
    vals, lo, hi, mb, mb_lo, mb_hi = comp_arr
    vals = np.asarray(vals, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    out: Dict[str, Tuple[float, float, float]] = {}
    for i, (a, b) in enumerate(cent_bins):
        out[canonical_cent(f"{a}-{b}%")] = (float(vals[i]), float(lo[i]), float(hi[i]))
    out["MB"] = (float(mb), float(mb_lo), float(mb_hi))
    return out


def build_cnm_state_products(
    setup: SystemSetup,
    state: str,
    use_absorption_primary: bool,
    include_cnm_components: bool = False,
    abs_sigma_map: Optional[Mapping[str, object]] = None,
) -> CNMStateProducts:
    if state not in STATE_TO_CNM:
        raise ValueError(f"Unknown state: {state}")

    sigma_c, sigma_lo, sigma_hi = _resolve_abs_sigma_tuple(state, setup, abs_sigma_map=abs_sigma_map)

    no_abs = CNMCombineFast.from_defaults(
        energy=setup.energy,
        family="charmonia",
        particle_state=STATE_TO_CNM[state],
        alpha_s_mode="constant",
        alpha0=0.5,
        cent_bins=setup.cent_bins,
        enable_absorption=False,
        m_state_for_np=M_STATE_GEV[state],
    )

    abs_mode = "avg_TA" if setup.system == "LHC" else "dA_avg_TA"
    with_abs = CNMCombineFast.from_defaults(
        energy=setup.energy,
        family="charmonia",
        particle_state=STATE_TO_CNM[state],
        alpha_s_mode="constant",
        alpha0=0.5,
        cent_bins=setup.cent_bins,
        enable_absorption=True,
        abs_sigma_mb=sigma_c,
        abs_mode=abs_mode,
        m_state_for_np=M_STATE_GEV[state],
    )

    comp_list = ["cnm"]
    if include_cnm_components:
        comp_list = ["npdf", "eloss", "broad", "eloss_broad", "cnm"]

    cent_tags = [canonical_cent(f"{a}-{b}%") for (a, b) in setup.cent_bins] + ["MB"]

    # vs y
    y_cent, _, by_no = no_abs.cnm_vs_y(
        y_edges=setup.y_edges,
        pt_range_avg=setup.pt_range_integrated,
        components=comp_list,
        include_mb=True,
    )
    _, _, by_abs = with_abs.cnm_vs_y(
        y_edges=setup.y_edges,
        pt_range_avg=setup.pt_range_integrated,
        components=comp_list,
        include_mb=True,
    )
    y_no = _pack_axis_components(by_no, comp_list, cent_tags)
    y_abs = _pack_axis_components(by_abs, comp_list, cent_tags)

    # vs pT
    pt_by_variant = {"no_abs": {}, "abs": {}}
    pt_cent_ref = None
    for yname, y0, y1 in iter_ywins(setup.y_windows):
        pt_cent, _, p_no = no_abs.cnm_vs_pT(
            y_window=(y0, y1),
            pt_edges=setup.pt_edges,
            components=comp_list,
            include_mb=True,
        )
        _, _, p_abs = with_abs.cnm_vs_pT(
            y_window=(y0, y1),
            pt_edges=setup.pt_edges,
            components=comp_list,
            include_mb=True,
        )
        if pt_cent_ref is None:
            pt_cent_ref = np.asarray(pt_cent, float)
        pt_by_variant["no_abs"][yname] = _pack_axis_components(p_no, comp_list, cent_tags)
        pt_by_variant["abs"][yname] = _pack_axis_components(p_abs, comp_list, cent_tags)

    # vs centrality
    cent_by_variant: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]] = {"no_abs": {}, "abs": {}}
    for yname, y0, y1 in iter_ywins(setup.y_windows):
        c_no = no_abs.cnm_vs_centrality(
            y_window=(y0, y1),
            pt_range_avg=setup.pt_range_integrated,
            components=comp_list,
            include_mb=True,
        )
        c_abs = with_abs.cnm_vs_centrality(
            y_window=(y0, y1),
            pt_range_avg=setup.pt_range_integrated,
            components=comp_list,
            include_mb=True,
        )

        cent_by_variant["no_abs"][yname] = {}
        cent_by_variant["abs"][yname] = {}
        for comp in comp_list:
            if comp in c_no:
                cent_by_variant["no_abs"][yname][comp] = _pack_cent_component(c_no[comp], setup.cent_bins)
            if comp in c_abs:
                cent_by_variant["abs"][yname][comp] = _pack_cent_component(c_abs[comp], setup.cent_bins)

    cent_mids = np.array([0.5 * (a + b) for (a, b) in setup.cent_bins], float)
    cent_edges = np.array([setup.cent_bins[0][0]] + [b for (_, b) in setup.cent_bins], float)

    primary = "abs" if use_absorption_primary else "no_abs"
    return CNMStateProducts(
        state=state,
        primary_variant=primary,
        abs_sigma_mb=float(sigma_c),
        abs_sigma_lo_hi=(float(sigma_lo), float(sigma_hi)),
        abs_mode=abs_mode,
        y_cent=np.asarray(y_cent, float),
        pt_cent=np.asarray(pt_cent_ref, float),
        cent_mids=cent_mids,
        cent_edges=cent_edges,
        y_by_variant={"no_abs": y_no, "abs": y_abs},
        pt_by_variant=pt_by_variant,
        cent_by_variant=cent_by_variant,
    )


def build_primordial_products(
    setup: SystemSetup,
    formation: str,
    states: Sequence[str],
) -> PrimordialProducts:
    if setup.system != "LHC":
        return PrimordialProducts(
            available=False,
            reason="No RHIC primordial input paths are configured in this repository.",
            vs_y={},
            vs_pt={},
            vs_cent={},
            tauform={},
        )

    if formation not in ("radius", "new", "old"):
        raise ValueError("formation must be radius/new/old")

    sqrts_gev = 8160.0 if setup.energy == "8.16" else 5020.0
    gtag = "8TeV_eloss" if setup.energy == "8.16" else "5TeV_eloss"

    geloof = PROJECT / "primordial_output" / "glauber_data" / gtag
    generate_primordial_glauber_maps(
        geloof,
        cfg=GlauberBridgeConfig(
            roots_gev=sqrts_gev,
            target_a=208,
            system="pA",
            bmax_fm=20.0,
            nb=401,
            include_npart=True,
            verbose=False,
        ),
    )
    _validate_glauber_maps(geloof)

    try:
        combos = _load_primordial_combos(setup.energy, formation, geloof, ReaderConfig(debug=False))
    except Exception as e:
        return PrimordialProducts(
            available=False,
            reason=f"Primordial load failed: {e}",
            vs_y={},
            vs_pt={},
            vs_cent={},
            tauform={},
        )

    y_bins = [(float(setup.y_edges[i]), float(setup.y_edges[i + 1])) for i in range(len(setup.y_edges) - 1)]
    pt_bins = [(float(setup.pt_edges[i]), float(setup.pt_edges[i + 1])) for i in range(len(setup.pt_edges) - 1)]

    prim_y = primordial_vs_y_all(combos, list(states), setup.cent_bins, setup.pt_range_integrated, y_bins)
    prim_pt = primordial_vs_pT_all(combos, list(states), setup.cent_bins, {
        "backward": Y_WINDOW_BACKWARD,
        "central": Y_WINDOW_CENTRAL,
        "forward": Y_WINDOW_FORWARD,
    }, pt_bins)
    prim_cent = primordial_vs_cent_from_vs_y(prim_y, {
        "backward": Y_WINDOW_BACKWARD,
        "central": Y_WINDOW_CENTRAL,
        "forward": Y_WINDOW_FORWARD,
    }, list(states), ("Pert", "NPWLC"))

    return PrimordialProducts(
        available=True,
        reason="ok",
        vs_y=prim_y,
        vs_pt=prim_pt,
        vs_cent=prim_cent,
        tauform=_tauform_for_form(formation),
    )


def extract_prim_y(
    prim: PrimordialProducts,
    state: str,
    model: str,
    cent_tag: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not prim.available:
        return None
    for tag, block in prim.vs_y.get(model, {}).items():
        if canonical_cent(tag) == canonical_cent(cent_tag) and state in block:
            rc, lo, hi, x = block[state]
            return np.asarray(x, float), np.asarray(rc, float), np.asarray(lo, float), np.asarray(hi, float)
    return None


def extract_prim_pt(
    prim: PrimordialProducts,
    state: str,
    model: str,
    yname: str,
    cent_tag: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not prim.available:
        return None
    for tag, block in prim.vs_pt.get(yname, {}).get(model, {}).items():
        if canonical_cent(tag) == canonical_cent(cent_tag) and state in block:
            rc, lo, hi, x = block[state]
            return np.asarray(x, float), np.asarray(rc, float), np.asarray(lo, float), np.asarray(hi, float)
    return None


def extract_prim_cent(
    prim: PrimordialProducts,
    state: str,
    model: str,
    yname: str,
    cent_bins: Sequence[Tuple[int, int]],
) -> Optional[Dict[str, Tuple[float, float, float]]]:
    if not prim.available:
        return None
    block = prim.vs_cent.get(yname, {}).get(model, {})
    if state not in block:
        return None
    cent_mid, rc, lo, hi, rc_mb, lo_mb, hi_mb = block[state]
    cent_mid = np.asarray(cent_mid, float)
    rc = np.asarray(rc, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    out: Dict[str, Tuple[float, float, float]] = {}
    for a, b in cent_bins:
        mid = 0.5 * (a + b)
        idx = int(np.argmin(np.abs(cent_mid - mid)))
        out[canonical_cent(f"{a}-{b}%")] = (float(rc[idx]), float(lo[idx]), float(hi[idx]))
    out["MB"] = (float(rc_mb), float(lo_mb), float(hi_mb))
    return out


def compose_axis_curves(
    cnm_band: Tuple[np.ndarray, np.ndarray, np.ndarray],
    prim_pert: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    prim_npwlc: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    include_primordial: bool = True,
    include_combined: bool = True,
    band_mode: str = "quadrature",
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {"cnm_total": cnm_band}
    if include_primordial and prim_pert is not None and prim_npwlc is not None:
        out["prim_pert"] = prim_pert
        out["prim_npwlc"] = prim_npwlc
    if include_combined and prim_pert is not None and prim_npwlc is not None:
        out["comb_pert"] = combine_bands(*cnm_band, *prim_pert, mode=band_mode)
        out["comb_npwlc"] = combine_bands(*cnm_band, *prim_npwlc, mode=band_mode)
    return out


def _plot_band(
    ax,
    x_cent: np.ndarray,
    band: Tuple[np.ndarray, np.ndarray, np.ndarray],
    style_key: str,
    step: bool = True,
    alpha: float = 0.20,
    x_max_extend: Optional[float] = None,
):
    st = THEORY_STYLE[style_key]
    c, lo, hi = [np.asarray(v, float) for v in band]
    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)
    lo2 = np.minimum(lo2, c)
    hi2 = np.maximum(hi2, c)
    if step:
        xe, yc = step_from_centers(x_cent, c)
        _, yl = step_from_centers(x_cent, lo2)
        _, yh = step_from_centers(x_cent, hi2)
        if x_max_extend is not None and xe.size > 0 and xe[-1] < float(x_max_extend):
            xe = np.append(xe, float(x_max_extend))
            yc = np.append(yc, yc[-1])
            yl = np.append(yl, yl[-1])
            yh = np.append(yh, yh[-1])
        ax.step(xe, yc, where="post", color=st["color"], ls=st["ls"], lw=2.0, label=st["label"])
        ax.fill_between(xe, yl, yh, step="post", color=st["color"], alpha=alpha, lw=0)
    else:
        ax.plot(x_cent, c, color=st["color"], ls=st["ls"], lw=2.0, label=st["label"])
        ax.fill_between(x_cent, lo2, hi2, color=st["color"], alpha=alpha, lw=0)


def _plot_component_band(ax, x_cent: np.ndarray, band: Tuple[np.ndarray, np.ndarray, np.ndarray], comp: str, step: bool = True) -> None:
    if comp not in CNM_COMPONENT_STYLE:
        return
    st = CNM_COMPONENT_STYLE[comp]
    c, lo, hi = [np.asarray(v, float) for v in band]
    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)
    if step:
        xe, yc = step_from_centers(x_cent, c)
        _, yl = step_from_centers(x_cent, lo2)
        _, yh = step_from_centers(x_cent, hi2)
        ax.step(xe, yc, where="post", color=st["color"], ls=st["ls"], lw=1.5, label=st["label"])
        ax.fill_between(xe, yl, yh, step="post", color=st["color"], alpha=0.10, lw=0)
    else:
        ax.plot(x_cent, c, color=st["color"], ls=st["ls"], lw=1.5, label=st["label"])
        ax.fill_between(x_cent, lo2, hi2, color=st["color"], alpha=0.10, lw=0)


def _cnm_line_style(variant: str) -> Dict[str, object]:
    if variant == "abs":
        return {"color": "#4A4A4A", "ls": "-", "label": "CNM (w/ Nucl Abs)", "alpha": 0.20}
    return {"color": "gray", "ls": "--", "label": "CNM (w/out Nucl Abs)", "alpha": 0.12}


def _plot_cnm_band(
    ax,
    x_cent: np.ndarray,
    band: Tuple[np.ndarray, np.ndarray, np.ndarray],
    variant: str,
    label: Optional[str] = None,
    x_max_extend: Optional[float] = None,
) -> None:
    st = _cnm_line_style(variant)
    c, lo, hi = [np.asarray(v, float) for v in band]
    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)
    lo2 = np.minimum(lo2, c)
    hi2 = np.maximum(hi2, c)
    xe, yc = step_from_centers(x_cent, c)
    _, yl = step_from_centers(x_cent, lo2)
    _, yh = step_from_centers(x_cent, hi2)
    if x_max_extend is not None and xe.size > 0 and xe[-1] < float(x_max_extend):
        xe = np.append(xe, float(x_max_extend))
        yc = np.append(yc, yc[-1])
        yl = np.append(yl, yl[-1])
        yh = np.append(yh, yh[-1])
    ax.step(xe, yc, where="post", color=st["color"], ls=st["ls"], lw=2.0, label=(st["label"] if label is None else label))
    ax.fill_between(xe, yl, yh, step="post", color=st["color"], alpha=float(st["alpha"]), lw=0)


def _apply_tick_pruning(ax, row_i: int, col_i: int, nrows: int, ncols: int) -> None:
    # Prevent edge-label collisions for stacked zero-spacing layouts.
    y_prune = None
    if nrows > 1:
        if row_i == 0:
            y_prune = "lower"
        elif row_i == nrows - 1:
            y_prune = "upper"
        else:
            y_prune = "both"
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune=y_prune))
    x_prune = None
    if ncols > 1:
        if col_i == 0:
            x_prune = "upper"
        elif col_i == ncols - 1:
            x_prune = "lower"
        else:
            x_prune = "both"
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune=x_prune))


def style_axis(
    ax,
    xlabel: str,
    ylabel: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    tag: Optional[str] = None,
    note: Optional[str] = None,
    system_tag: Optional[str] = None,
    state_tag: Optional[str] = None,
    note_loc: str = "lower left",
    text_style: Optional[Mapping[str, object]] = None,
    hide_x: bool = False,
    hide_y: bool = False,
) -> None:
    ts = dict(TEXT_STYLE_DEFAULTS)
    if text_style:
        ts.update(dict(text_style))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, pad=2.0)
    if hide_x:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(xlabel)
    if hide_y:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel(ylabel, fontweight="normal")
    if system_tag:
        ax.text(
            0.97,
            0.95,
            system_tag,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=float(ts["system_fontsize"]),
            color=str(ts["system_color"]),
            fontweight=str(ts["system_weight"]),
        )
    if state_tag:
        ax.text(
            0.97,
            0.86,
            state_tag,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=float(ts["state_fontsize"]),
            color=str(ts["state_color"]),
            fontweight=str(ts["state_weight"]),
        )
    if tag:
        ax.text(
            0.03,
            0.95,
            tag,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=float(ts["tag_fontsize"]),
            color=str(ts["tag_color"]),
            fontweight=str(ts["tag_weight"]),
        )
    if note:
        note_loc_norm = note_loc.lower().replace("_", " ").strip()
        x_note, y_note, ha_note, va_note = 0.03, 0.03, "left", "bottom"
        if note_loc_norm == "lower right":
            x_note, y_note, ha_note, va_note = 0.97, 0.03, "right", "bottom"
        elif note_loc_norm == "upper left":
            x_note, y_note, ha_note, va_note = 0.03, 0.97, "left", "top"
        elif note_loc_norm == "upper right":
            x_note, y_note, ha_note, va_note = 0.97, 0.97, "right", "top"
        ax.text(
            x_note,
            y_note,
            note,
            transform=ax.transAxes,
            ha=ha_note,
            va=va_note,
            fontsize=float(ts["note_fontsize"]),
            bbox=dict(facecolor="white", alpha=float(ts["note_box_alpha"]), edgecolor="none", pad=1.4),
        )


def add_global_legend(fig, axes, ncol: int = 3, y_anchor: float = 0.99) -> None:
    seen = set()
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll in seen or (not ll):
                continue
            seen.add(ll)
            handles.append(hh)
            labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y_anchor), frameon=False)


def add_panel_legend(
    axes,
    anchor_index: int = 0,
    loc: str = "upper left",
    ncol: int = 2,
    fontsize: float = 8.2,
    bold_labels: bool = False,
    bbox_to_anchor: Optional[Tuple[float, float]] = None,
) -> None:
    axes_arr = np.atleast_1d(axes).ravel()
    seen = set()
    handles, labels = [], []
    for ax in axes_arr:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if not ll or ll in seen:
                continue
            seen.add(ll)
            handles.append(hh)
            labels.append(ll)
    if not handles:
        return
    idx = max(0, min(anchor_index, len(axes_arr) - 1))
    leg = axes_arr[idx].legend(
        handles,
        labels,
        loc=loc,
        ncol=ncol,
        fontsize=fontsize,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        framealpha=0.84,
        facecolor="white",
        edgecolor="none",
        handlelength=1.8,
        borderpad=0.25,
        labelspacing=0.2,
        columnspacing=0.8,
    )
    if bold_labels:
        for txt in leg.get_texts():
            txt.set_fontweight("bold")


def add_split_panel_legends(
    axes,
    anchor_left: int,
    anchor_right: int,
    loc_left: str = "lower right",
    loc_right: str = "upper right",
    ncol: int = 1,
    fontsize: float = 8.0,
    bold_labels: bool = False,
) -> None:
    axes_arr = np.atleast_1d(axes).ravel()
    seen = set()
    handles, labels = [], []
    for ax in axes_arr:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if not ll or ll in seen:
                continue
            seen.add(ll)
            handles.append(hh)
            labels.append(ll)
    if not handles:
        return
    split_idx = max(1, int(np.ceil(len(handles) / 2.0)))
    groups = [
        (handles[:split_idx], labels[:split_idx], anchor_left, loc_left),
        (handles[split_idx:], labels[split_idx:], anchor_right, loc_right),
    ]
    for hh, ll, idx, loc in groups:
        if not hh:
            continue
        idx_c = max(0, min(idx, len(axes_arr) - 1))
        leg = axes_arr[idx_c].legend(
            hh,
            ll,
            loc=loc,
            ncol=ncol,
            fontsize=fontsize,
            frameon=True,
            framealpha=0.84,
            facecolor="white",
            edgecolor="none",
            handlelength=1.8,
            borderpad=0.25,
            labelspacing=0.2,
            columnspacing=0.8,
        )
        if bold_labels:
            for txt in leg.get_texts():
                txt.set_fontweight("bold")


def _lhc_win_str(rw: Tuple[float, float] | None) -> str:
    if rw is None:
        return "midrapidity"
    y0, y1 = rw
    if y1 < -1.0:
        return "backward"
    if y0 > 1.0:
        return "forward"
    return "midrapidity"


def _filter_lhc_rapidity(df: pd.DataFrame, rw: Tuple[float, float] | None) -> pd.DataFrame:
    if df is None or df.empty or rw is None or "rapidity" not in df.columns:
        return df
    win = _lhc_win_str(rw)
    yc = df["rapidity"].apply(_ycms_center)
    if win == "backward":
        return df[yc < -1.0]
    if win == "forward":
        return df[yc > 1.0]
    return df[(yc >= -1.0) & (yc <= 1.0)]


def _infer_cent_tags_from_ncoll(ncoll_series: pd.Series, energy: str) -> List[str]:
    n = len(np.unique(ncoll_series.dropna()))
    if energy == "8.16":
        if n >= 9:
            return ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-80%", "80-100%"]
        return ["2-10%", "10-20%", "20-40%", "40-60%", "60-80%", "80-90%"]
    # 5.02
    return ["2-10%", "10-20%", "20-40%", "40-60%", "60-80%", "80-100%"] if n >= 6 else ["0-20%", "20-40%", "40-60%", "60-100%"]


def _map_ncoll_to_cent_mid(ncoll_series: pd.Series, energy: str) -> pd.Series:
    tags = _infer_cent_tags_from_ncoll(ncoll_series, energy)
    mids = [0.5 * (float(t.split("-")[0]) + float(t.split("-")[1].replace("%", ""))) for t in tags]
    uniq = np.array(sorted(np.unique(ncoll_series), reverse=True), float)
    n_use = min(len(uniq), len(mids))
    mapping = {uniq[i]: mids[i] for i in range(n_use)}

    def _closest(v: float) -> float:
        key = min(mapping.keys(), key=lambda k: abs(k - v))
        return mapping[key]

    return ncoll_series.apply(_closest)


def overlay_lhc_state(ax, obs_type: str, energy: str, state: str, rap_window: Tuple[float, float] | None = None, cent_tag: str | None = None) -> None:
    if state not in ("jpsi_1S", "psi_2S"):
        return
    state_hep = "Jpsi" if state == "jpsi_1S" else "psi2S"
    marker = "D" if state == "jpsi_1S" else "^"

    if obs_type in ("vs_centrality", "vs_ncoll", "vs_pt_cent"):
        reader = HEPCentDataReader(PROJECT / "input" / "experimental_input")
        tag = "8TeV" if float(energy) > 6 else "5TeV"

        if obs_type in ("vs_centrality", "vs_ncoll"):
            df = reader.rpa_vs_ncoll(tag, state=state_hep)
            df = _filter_lhc_rapidity(df, rap_window)
            if df is None or df.empty:
                return
            err = np.sqrt(df["stat_up"] ** 2 + df["sys_uncorr_up"] ** 2)
            x = df["ncoll"] if obs_type == "vs_ncoll" else _map_ncoll_to_cent_mid(df["ncoll"], energy)
            mfc = "black" if state == "jpsi_1S" else "none"
            ax.errorbar(x, df["value"], yerr=err, fmt=marker, color="black", mfc=mfc, mew=1.2, ms=6.2,
                        capsize=2, label="ALICE data", zorder=22)
            return

        # centrality dependent pT
        if cent_tag is not None:
            groups = reader.rpa_vs_pt(tag, state=state_hep)
            for g in groups:
                if g is None or g.empty:
                    continue
                g = _filter_lhc_rapidity(g, rap_window)
                if g.empty:
                    continue
                dcent = canonical_cent(str(g["centrality"].iloc[0])) if "centrality" in g.columns else None
                if dcent != canonical_cent(cent_tag):
                    continue
                x = g["x_cen"] if "x_cen" in g.columns else 0.5 * (g["x_low"] + g["x_high"])
                err = np.sqrt(g["stat_up"] ** 2 + g["sys_uncorr_up"] ** 2)
                mfc = "black" if state == "jpsi_1S" else "none"
                ax.errorbar(x, g["value"], yerr=err, fmt=marker, color="black", mfc=mfc, mew=1.2, ms=5.7,
                            capsize=2, label="ALICE data", zorder=22)
            return

    # MinBias overlays for vs_y / vs_pt(MB)
    mb = load_minbias_data(verbose=False)
    extras = []
    try:
        extras.append(get_lhcb_digitized_data())
    except Exception:
        pass
    try:
        extras.append(get_psi2s_8tev_preliminary())
    except Exception:
        pass
    if extras:
        mb = pd.concat([mb] + extras, ignore_index=True)

    s_long = "jpsi_1s" if state == "jpsi_1S" else "psi2s_2s"
    if obs_type == "vs_y":
        data = get_RpA_vs_y(mb, state=s_long, energy_tev=float(energy))
    else:
        data = get_RpA_vs_pT(mb, state=s_long, energy_tev=float(energy), rapidity_window=_lhc_win_str(rap_window))

    if data is None or data.empty:
        return
    xcol = "variable"
    if "collaboration" in data.columns:
        for collab, g in data.groupby("collaboration"):
            color = "black" if collab == "ALICE" else "#1f77b4"
            mfc = color if state == "jpsi_1S" else "none"
            ax.errorbar(g[xcol], g["value"], yerr=g["total_err"], fmt=marker, color=color, mfc=mfc,
                        mew=1.2, ms=5.8, capsize=2, label=f"{collab} data", zorder=22)
    else:
        mfc = "black" if state == "jpsi_1S" else "none"
        ax.errorbar(data[xcol], data["value"], yerr=data["total_err"], fmt=marker, color="black", mfc=mfc,
                    mew=1.2, ms=5.8, capsize=2, label="ALICE data", zorder=22)


def _rhic_cent_to_mid(c: str) -> float:
    c = str(c).replace("%", "")
    if c == "60-88":
        return 74.0
    if "-" in c:
        a, b = c.split("-", 1)
        return 0.5 * (float(a) + float(b))
    return 50.0


def _rhic_win_str(rw: Tuple[float, float] | None) -> str:
    if rw is None:
        return "mid"
    y0, y1 = rw
    if y0 < -1.0:
        return "backward"
    if y1 > 1.0:
        return "forward"
    return "mid"


def overlay_rhic_state(ax, obs_type: str, state: str, rap_window: Tuple[float, float] | None = None, cent_tag: str | None = None) -> None:
    if state not in ("jpsi_1S", "psi_2S"):
        return
    rr = RHICDataReader()
    marker = "D" if state == "jpsi_1S" else "^"
    mfc = "black" if state == "jpsi_1S" else "none"

    if obs_type in ("vs_centrality", "vs_ncoll"):
        if state == "jpsi_1S":
            df = rr.get_jpsi_rpa_vs_ncoll()
            if df is None or df.empty:
                return
            x = df["Ncoll"] if obs_type == "vs_ncoll" else df["centrality"].apply(_rhic_cent_to_mid)
            yerr = np.asarray(df.get("sys", 0.0), float)
            ax.errorbar(x, df["value"], yerr=yerr, fmt=marker, color="black", mfc=mfc, mew=1.2, ms=6.2,
                        capsize=2, label="PHENIX data", zorder=22)
        else:
            df = rr.get_psi2s_rpa_vs_ncoll()
            if df is None or df.empty:
                return
            if "Ncoll" not in df.columns:
                return
            x = df["Ncoll"] if obs_type == "vs_ncoll" else df["Ncoll"].apply(lambda n: {15.1: 10.0, 10.2: 30.0, 6.6: 50.0, 3.2: 80.0}[min([15.1, 10.2, 6.6, 3.2], key=lambda k: abs(k - n))])
            yerr = np.asarray(df.get("sys", 0.0), float)
            ax.errorbar(x, df["value"], yerr=yerr, fmt=marker, color="black", mfc=mfc, mew=1.2, ms=6.2,
                        capsize=2, label="PHENIX data", zorder=22)
        return

    # RHIC y and pT published overlays are J/psi only in current reader.
    if state != "jpsi_1S":
        return

    cent_norm = "Minimum Bias"
    ccan = canonical_cent(cent_tag) if cent_tag else "MB"
    if ccan != "MB":
        if ccan == "60-100%":
            cent_norm = "60-88"
        else:
            cent_norm = ccan.replace("%", "")

    if obs_type == "vs_y":
        df = rr.get_jpsi_rpa_vs_y(centrality=cent_norm)
        if df is None or df.empty:
            return
        yerr = np.sqrt(np.asarray(df.get("stat", 0.0), float) ** 2 + np.asarray(df.get("sys", 0.0), float) ** 2)
        ax.errorbar(df["y"], df["value"], yerr=yerr, fmt=marker, color="black", mfc="black", mew=1.2, ms=6.0,
                    capsize=2, label="PHENIX data", zorder=22)
        return

    if obs_type in ("vs_pt", "vs_pt_cent"):
        win = _rhic_win_str(rap_window)
        df = rr.get_jpsi_rpa_vs_pt(centrality=cent_norm, rapidity_window=win)
        if df is not None and not df.empty:
            yerr = np.sqrt(np.asarray(df.get("stat", 0.0), float) ** 2 + np.asarray(df.get("sys", 0.0), float) ** 2)
            ax.errorbar(df["pT"], df["value"], yerr=yerr, fmt=marker, color="black", mfc="black", mew=1.2, ms=6.0,
                        capsize=2, label="PHENIX data", zorder=22)
        if win == "mid" and cent_norm == "Minimum Bias":
            df_star = rr.get_star_jpsi_rpa_vs_pt()
            if df_star is not None and not df_star.empty:
                ax.errorbar(df_star["pT"], df_star["value"], yerr=np.asarray(df_star.get("sys", 0.0), float),
                            fmt="D", color="#FFC107", markeredgecolor="black", mfc="#FFC107", ms=7.0,
                            capsize=2, label="STAR data", zorder=23)


def overlay_experiment(
    ax,
    setup: SystemSetup,
    obs_type: str,
    state: str,
    rap_window: Tuple[float, float] | None = None,
    cent_tag: str | None = None,
) -> None:
    if setup.system == "LHC":
        overlay_lhc_state(ax, obs_type, setup.energy, state, rap_window=rap_window, cent_tag=cent_tag)
    else:
        overlay_rhic_state(ax, obs_type, state, rap_window=rap_window, cent_tag=cent_tag)


def _append_rows(
    rows: List[dict],
    setup: SystemSetup,
    formation: str,
    state: str,
    observable: str,
    y_window: str,
    cent_tag: str,
    xvals: np.ndarray,
    band: Tuple[np.ndarray, np.ndarray, np.ndarray],
    source_tag: str,
) -> None:
    c, lo, hi = [np.asarray(v, float) for v in band]
    for xv, rc, rl, rh in zip(np.asarray(xvals, float), c, lo, hi):
        rows.append(
            {
                "system": "pPb" if setup.system == "LHC" else "dAu",
                "energy": float(setup.energy),
                "formation": formation,
                "state": state,
                "observable": observable,
                "y_window": y_window,
                "centrality_tag": cent_tag,
                "x": float(xv),
                "R": float(rc),
                "R_lo": float(rl),
                "R_hi": float(rh),
                "pt_range": f"{setup.pt_range_integrated[0]}-{setup.pt_range_integrated[1]}",
                "source_tag": source_tag,
            }
        )


def _mb_from_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, Tuple[float, float, float]]:
    out = {}
    for k, band in curves.items():
        c, lo, hi = band
        out[k] = (float(np.asarray(c).reshape(-1)[0]), float(np.asarray(lo).reshape(-1)[0]), float(np.asarray(hi).reshape(-1)[0]))
    return out


def _state_bold_label(state: str) -> str:
    if state == "jpsi_1S":
        return r"$\mathbf{J/\psi(1S)}$"
    if state == "chicJ_1P":
        return r"$\mathbf{\chi_c(1P)}$"
    if state == "psi_2S":
        return r"$\mathbf{\psi(2S)}$"
    return rf"$\mathbf{{{state}}}$"


def _system_label(setup: SystemSetup) -> str:
    if setup.system == "LHC":
        return rf"$\mathbf{{p+Pb}}\ \sqrt{{s_{{NN}}}}={float(setup.energy):.2f}\ \mathrm{{TeV}}$"
    return rf"$\mathbf{{d+Au}}\ \sqrt{{s_{{NN}}}}={float(setup.energy):.0f}\ \mathrm{{GeV}}$"


def _absorption_note(setup: SystemSetup, cnm: CNMStateProducts) -> str:
    lo, hi = cnm.abs_sigma_lo_hi
    mid = cnm.abs_sigma_mb
    return rf"$\sigma_{{abs}}=[{lo:.2g},{mid:.2g},{hi:.2g}]\ \mathrm{{mb}}$"


def _panel_note_text(setup: SystemSetup, state: str, tau: Optional[Tuple[float, float]], cnm: CNMStateProducts) -> str:
    lines = []
    if tau is not None:
        lines.append(rf"$\tau_{{form}}=[{tau[0]:.2g},{tau[1]:.2g}]\ \mathrm{{fm/c}}$")
    if setup.system == "RHIC" or cnm.primary_variant == "abs":
        lines.append(_absorption_note(setup, cnm))
    return "\n".join(lines)


def plot_state_vs_centrality(
    setup: SystemSetup,
    formation: str,
    state: str,
    cnm: CNMStateProducts,
    prim: PrimordialProducts,
    include_primordial: bool = True,
    include_combined: bool = True,
    show_cnm_components: bool = False,
    show_alt_cnm: bool = False,
    band_mode: str = "quadrature",
    include_experiment: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    legend_loc: str = "lower center",
    legend_ncol: int = 2,
    legend_anchor_index: Optional[int] = None,
    legend_fontsize: float = 8.6,
    text_style: Optional[Mapping[str, object]] = None,
) -> Tuple[plt.Figure, List[dict]]:
    rows: List[dict] = []
    fig, axes = plt.subplots(1, len(setup.y_windows), figsize=(13.0, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    xlim_eff = (0.0, 100.0) if xlim is None else xlim

    tau = prim.tauform.get(state) if prim.available else None
    panel_note = _panel_note_text(setup, state, tau, cnm)

    for iax, (yname, y0, y1) in enumerate(iter_ywins(setup.y_windows)):
        ax = axes[iax]

        prim_pert_cent = extract_prim_cent(prim, state, "Pert", yname, setup.cent_bins) if prim.available else None
        prim_npw_cent = extract_prim_cent(prim, state, "NPWLC", yname, setup.cent_bins) if prim.available else None

        xcent = cnm.cent_mids
        cdict = cnm.cent_by_variant[cnm.primary_variant][yname]
        cnm_band = (
            np.array([cdict["cnm"][canonical_cent(f"{a}-{b}%")][0] for (a, b) in setup.cent_bins], float),
            np.array([cdict["cnm"][canonical_cent(f"{a}-{b}%")][1] for (a, b) in setup.cent_bins], float),
            np.array([cdict["cnm"][canonical_cent(f"{a}-{b}%")][2] for (a, b) in setup.cent_bins], float),
        )

        ppert = None
        pnpw = None
        if prim_pert_cent and prim_npw_cent:
            ppert = (
                np.array([prim_pert_cent[canonical_cent(f"{a}-{b}%")][0] for (a, b) in setup.cent_bins], float),
                np.array([prim_pert_cent[canonical_cent(f"{a}-{b}%")][1] for (a, b) in setup.cent_bins], float),
                np.array([prim_pert_cent[canonical_cent(f"{a}-{b}%")][2] for (a, b) in setup.cent_bins], float),
            )
            pnpw = (
                np.array([prim_npw_cent[canonical_cent(f"{a}-{b}%")][0] for (a, b) in setup.cent_bins], float),
                np.array([prim_npw_cent[canonical_cent(f"{a}-{b}%")][1] for (a, b) in setup.cent_bins], float),
                np.array([prim_npw_cent[canonical_cent(f"{a}-{b}%")][2] for (a, b) in setup.cent_bins], float),
            )

        curves = compose_axis_curves(
            cnm_band,
            prim_pert=ppert,
            prim_npwlc=pnpw,
            include_primordial=include_primordial,
            include_combined=include_combined,
            band_mode=band_mode,
        )

        alt_variant = None
        if show_alt_cnm:
            alt = "abs" if cnm.primary_variant == "no_abs" else "no_abs"
            ad = cnm.cent_by_variant[alt][yname]["cnm"]
            alt_band = (
                np.array([ad[canonical_cent(f"{a}-{b}%")][0] for (a, b) in setup.cent_bins], float),
                np.array([ad[canonical_cent(f"{a}-{b}%")][1] for (a, b) in setup.cent_bins], float),
                np.array([ad[canonical_cent(f"{a}-{b}%")][2] for (a, b) in setup.cent_bins], float),
            )
            curves["cnm_alt"] = alt_band
            alt_variant = alt

        for key, band in curves.items():
            if key == "cnm_total":
                _plot_cnm_band(ax, xcent, band, cnm.primary_variant, label="CNM", x_max_extend=float(xlim_eff[1]))
            elif key == "cnm_alt" and alt_variant is not None:
                _plot_cnm_band(ax, xcent, band, alt_variant, x_max_extend=float(xlim_eff[1]))
            else:
                _plot_band(ax, xcent, band, key, step=True, x_max_extend=float(xlim_eff[1]))

            _append_rows(
                rows,
                setup,
                formation,
                state,
                "vs_centrality",
                yname,
                "slice",
                xcent,
                band,
                key,
            )

        if show_cnm_components:
            for comp in ("npdf", "eloss", "broad", "eloss_broad"):
                if comp not in cdict:
                    continue
                comp_band = (
                    np.array([cdict[comp][canonical_cent(f"{a}-{b}%")][0] for (a, b) in setup.cent_bins], float),
                    np.array([cdict[comp][canonical_cent(f"{a}-{b}%")][1] for (a, b) in setup.cent_bins], float),
                    np.array([cdict[comp][canonical_cent(f"{a}-{b}%")][2] for (a, b) in setup.cent_bins], float),
                )
                _plot_component_band(ax, xcent, comp_band, comp)

        # MB horizontal references for selected curves
        mb_src = {}
        mb_cnm = cdict["cnm"]["MB"]
        mb_src["cnm_total"] = mb_cnm
        if "cnm_alt" in curves:
            alt = "abs" if cnm.primary_variant == "no_abs" else "no_abs"
            mb_src["cnm_alt"] = cnm.cent_by_variant[alt][yname]["cnm"]["MB"]
        if prim_pert_cent and prim_npw_cent:
            mb_pert = prim_pert_cent["MB"]
            mb_npw = prim_npw_cent["MB"]
            mb_src["prim_pert"] = mb_pert
            mb_src["prim_npwlc"] = mb_npw
            mb_src["comb_pert"] = tuple(v[0] for v in combine_bands(np.array([mb_cnm[0]]), np.array([mb_cnm[1]]), np.array([mb_cnm[2]]), np.array([mb_pert[0]]), np.array([mb_pert[1]]), np.array([mb_pert[2]]), mode=band_mode))
            mb_src["comb_npwlc"] = tuple(v[0] for v in combine_bands(np.array([mb_cnm[0]]), np.array([mb_cnm[1]]), np.array([mb_cnm[2]]), np.array([mb_npw[0]]), np.array([mb_npw[1]]), np.array([mb_npw[2]]), mode=band_mode))

        for k, (rc, lo, hi) in mb_src.items():
            if k not in curves:
                continue
            if k == "cnm_total":
                st = _cnm_line_style(cnm.primary_variant)
            elif k == "cnm_alt":
                st = _cnm_line_style(alt_variant if alt_variant is not None else ("abs" if cnm.primary_variant == "no_abs" else "no_abs"))
            else:
                st = THEORY_STYLE[k]
            ax.hlines(rc, float(xlim_eff[0]), float(xlim_eff[1]), colors=st["color"], linestyles=st["ls"], linewidth=1.1, alpha=0.75)
            ax.fill_between([float(xlim_eff[0]), float(xlim_eff[1])], [lo, lo], [hi, hi], color=st["color"], alpha=0.06)

        if include_experiment:
            overlay_experiment(ax, setup, "vs_centrality", state, rap_window=(y0, y1))

        style_axis(
            ax,
            xlabel="Centrality [%]",
            ylabel=setup.r_label if iax == 0 else "",
            xlim=xlim_eff,
            ylim=(0.0, 1.6) if ylim is None else ylim,
            tag=rf"${y0:.2f} < y < {y1:.2f}$",
            note=panel_note if (iax == 0 and panel_note) else None,
            system_tag=_system_label(setup) if iax == 0 else None,
            state_tag=_state_bold_label(state) if iax == 0 else None,
            text_style=text_style,
            hide_y=(iax > 0),
        )
        _apply_tick_pruning(ax, 0, iax, 1, len(axes))

    legend_idx = (1 if len(axes) > 1 else 0) if legend_anchor_index is None else legend_anchor_index
    add_panel_legend(axes, anchor_index=legend_idx, loc=legend_loc, ncol=legend_ncol, fontsize=legend_fontsize)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0), pad=0.20, w_pad=0.0)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig, rows


def plot_state_vs_y(
    setup: SystemSetup,
    formation: str,
    state: str,
    cnm: CNMStateProducts,
    prim: PrimordialProducts,
    include_primordial: bool = True,
    include_combined: bool = True,
    show_cnm_components: bool = False,
    show_alt_cnm: bool = False,
    band_mode: str = "quadrature",
    include_experiment: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    legend_location: str = "top",
    legend_repeat: bool = True,
    legend_ncol: int = 3,
    legend_bold: bool = False,
    legend_fontsize: float = 8.2,
    text_style: Optional[Mapping[str, object]] = None,
) -> Tuple[plt.Figure, List[dict]]:
    rows: List[dict] = []
    cent_tags = [canonical_cent(f"{a}-{b}%") for (a, b) in setup.cent_bins] + ["MB"]
    ncols = 3
    nrows = int(np.ceil(len(cent_tags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.1 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    tau = prim.tauform.get(state) if prim.available else None
    panel_note = _panel_note_text(setup, state, tau, cnm)
    yname_for_rows = "all"

    for i, cent_tag in enumerate(cent_tags):
        ax = axes[i]

        p_dict = cnm.y_by_variant[cnm.primary_variant]
        if "cnm" not in p_dict or cent_tag not in p_dict["cnm"]:
            ax.set_visible(False)
            continue
        cnm_band = p_dict["cnm"][cent_tag]

        ppert = None
        pnpw = None
        if prim.available:
            p_pert = extract_prim_y(prim, state, "Pert", cent_tag)
            p_npw = extract_prim_y(prim, state, "NPWLC", cent_tag)
            if p_pert is not None and p_npw is not None:
                x_p, c_p, lo_p, hi_p = p_pert
                x_n, c_n, lo_n, hi_n = p_npw
                ppert = (
                    align_to_ref(x_p, c_p, cnm.y_cent),
                    align_to_ref(x_p, lo_p, cnm.y_cent),
                    align_to_ref(x_p, hi_p, cnm.y_cent),
                )
                pnpw = (
                    align_to_ref(x_n, c_n, cnm.y_cent),
                    align_to_ref(x_n, lo_n, cnm.y_cent),
                    align_to_ref(x_n, hi_n, cnm.y_cent),
                )

        curves = compose_axis_curves(cnm_band, ppert, pnpw, include_primordial=include_primordial, include_combined=include_combined, band_mode=band_mode)

        alt_variant = None
        if show_alt_cnm:
            alt = "abs" if cnm.primary_variant == "no_abs" else "no_abs"
            curves["cnm_alt"] = cnm.y_by_variant[alt]["cnm"][cent_tag]
            alt_variant = alt

        for key, band in curves.items():
            if key == "cnm_total":
                _plot_cnm_band(ax, cnm.y_cent, band, cnm.primary_variant, label="CNM")
            elif key == "cnm_alt" and alt_variant is not None:
                _plot_cnm_band(ax, cnm.y_cent, band, alt_variant)
            else:
                _plot_band(ax, cnm.y_cent, band, key, step=True)
            _append_rows(rows, setup, formation, state, "vs_y", yname_for_rows, cent_tag, cnm.y_cent, band, key)

        if show_cnm_components:
            for comp in ("npdf", "eloss", "broad", "eloss_broad"):
                if comp in p_dict and cent_tag in p_dict[comp]:
                    _plot_component_band(ax, cnm.y_cent, p_dict[comp][cent_tag], comp)

        if include_experiment and cent_tag == "MB":
            overlay_experiment(ax, setup, "vs_y", state)

        row_i = i // ncols
        col_i = i % ncols
        style_axis(
            ax,
            xlabel=r"$y$",
            ylabel=setup.r_label,
            xlim=(float(setup.y_edges.min()), float(setup.y_edges.max())) if xlim is None else xlim,
            ylim=(0.0, 1.6) if ylim is None else ylim,
            tag=cent_tag,
            note=panel_note if (i == 0 and panel_note) else None,
            system_tag=_system_label(setup) if i == 0 else None,
            state_tag=_state_bold_label(state) if i == 0 else None,
            text_style=text_style,
            hide_x=(row_i < nrows - 1),
            hide_y=(col_i > 0),
        )
        _apply_tick_pruning(ax, row_i, col_i, nrows, ncols)

    for j in range(len(cent_tags), len(axes)):
        axes[j].set_visible(False)

    top_center_idx = max(0, min(len(cent_tags) - 1, ncols // 2))
    bottom_center_idx = max(0, min(len(cent_tags) - 1, (nrows - 1) * ncols + (ncols // 2)))
    if legend_location == "bottom":
        anchor_primary = (bottom_center_idx, "lower right")
        anchor_secondary = (top_center_idx, "upper center")
    else:
        anchor_primary = (top_center_idx, "upper center")
        anchor_secondary = (bottom_center_idx, "lower center")
    add_panel_legend(
        axes,
        anchor_index=anchor_primary[0],
        loc=anchor_primary[1],
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        bold_labels=legend_bold,
    )
    if legend_repeat and nrows > 1 and anchor_secondary[0] != anchor_primary[0]:
        add_panel_legend(
            axes,
            anchor_index=anchor_secondary[0],
            loc=anchor_secondary[1],
            ncol=legend_ncol,
            fontsize=max(6.5, legend_fontsize - 0.2),
            bold_labels=legend_bold,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0), pad=0.20, w_pad=0.0, h_pad=0.0)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig, rows


def plot_state_vs_pt(
    setup: SystemSetup,
    formation: str,
    state: str,
    cnm: CNMStateProducts,
    prim: PrimordialProducts,
    include_primordial: bool = True,
    include_combined: bool = True,
    show_cnm_components: bool = False,
    show_alt_cnm: bool = False,
    band_mode: str = "quadrature",
    include_experiment: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    note_loc_first: str = "lower right",
    legend_split: bool = False,
    legend_fontsize: float = 8.0,
    legend_bold: bool = False,
    text_style: Optional[Mapping[str, object]] = None,
) -> Dict[str, Tuple[plt.Figure, List[dict]]]:
    out: Dict[str, Tuple[plt.Figure, List[dict]]] = {}
    tau = prim.tauform.get(state) if prim.available else None
    panel_note = _panel_note_text(setup, state, tau, cnm)

    for yname, y0, y1 in iter_ywins(setup.y_windows):
        rows: List[dict] = []
        cent_tags = [canonical_cent(f"{a}-{b}%") for (a, b) in setup.cent_bins] + ["MB"]
        ncols = 3
        nrows = int(np.ceil(len(cent_tags) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.1 * nrows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for i, cent_tag in enumerate(cent_tags):
            ax = axes[i]
            p_dict = cnm.pt_by_variant[cnm.primary_variant][yname]
            if "cnm" not in p_dict or cent_tag not in p_dict["cnm"]:
                ax.set_visible(False)
                continue
            cnm_band = p_dict["cnm"][cent_tag]

            ppert = None
            pnpw = None
            if prim.available:
                p_pert = extract_prim_pt(prim, state, "Pert", yname, cent_tag)
                p_npw = extract_prim_pt(prim, state, "NPWLC", yname, cent_tag)
                if p_pert is not None and p_npw is not None:
                    x_p, c_p, lo_p, hi_p = p_pert
                    x_n, c_n, lo_n, hi_n = p_npw
                    ppert = (
                        align_to_ref(x_p, c_p, cnm.pt_cent),
                        align_to_ref(x_p, lo_p, cnm.pt_cent),
                        align_to_ref(x_p, hi_p, cnm.pt_cent),
                    )
                    pnpw = (
                        align_to_ref(x_n, c_n, cnm.pt_cent),
                        align_to_ref(x_n, lo_n, cnm.pt_cent),
                        align_to_ref(x_n, hi_n, cnm.pt_cent),
                    )

            curves = compose_axis_curves(cnm_band, ppert, pnpw, include_primordial=include_primordial, include_combined=include_combined, band_mode=band_mode)

            alt_variant = None
            if show_alt_cnm:
                alt = "abs" if cnm.primary_variant == "no_abs" else "no_abs"
                curves["cnm_alt"] = cnm.pt_by_variant[alt][yname]["cnm"][cent_tag]
                alt_variant = alt

            for key, band in curves.items():
                if key == "cnm_total":
                    _plot_cnm_band(ax, cnm.pt_cent, band, cnm.primary_variant, label="CNM")
                elif key == "cnm_alt" and alt_variant is not None:
                    _plot_cnm_band(ax, cnm.pt_cent, band, alt_variant)
                else:
                    _plot_band(ax, cnm.pt_cent, band, key, step=True)
                _append_rows(rows, setup, formation, state, "vs_pt", yname, cent_tag, cnm.pt_cent, band, key)

            if show_cnm_components:
                for comp in ("npdf", "eloss", "broad", "eloss_broad"):
                    if comp in p_dict and cent_tag in p_dict[comp]:
                        _plot_component_band(ax, cnm.pt_cent, p_dict[comp][cent_tag], comp)

            if include_experiment:
                if cent_tag == "MB":
                    overlay_experiment(ax, setup, "vs_pt", state, rap_window=(y0, y1), cent_tag="MB")
                else:
                    overlay_experiment(ax, setup, "vs_pt_cent", state, rap_window=(y0, y1), cent_tag=cent_tag)

            row_i = i // ncols
            col_i = i % ncols
            style_axis(
                ax,
                xlabel=r"$p_T$ [GeV]",
                ylabel=setup.r_label,
                xlim=(float(setup.pt_edges.min()), float(setup.pt_edges.max())) if xlim is None else xlim,
                ylim=(0.0, 1.6) if ylim is None else ylim,
                tag=cent_tag,
                note=panel_note if (i == 0 and panel_note) else None,
                system_tag=_system_label(setup) if i == 0 else None,
                state_tag=_state_bold_label(state) if i == 0 else None,
                note_loc=note_loc_first if i == 0 else "lower left",
                text_style=text_style,
                hide_x=(row_i < nrows - 1),
                hide_y=(col_i > 0),
            )
            _apply_tick_pruning(ax, row_i, col_i, nrows, ncols)
            if i == 0:
                ax.text(
                    0.03,
                    0.84,
                    rf"${y0:.2f} < y < {y1:.2f}$",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    color="tab:blue",
                    fontweight="bold",
                )

        for j in range(len(cent_tags), len(axes)):
            axes[j].set_visible(False)

        legend_anchor = max(0, min(len(cent_tags) - 1, len(axes) - 1))
        if legend_split and len(cent_tags) > 1:
            left_anchor = max(0, min(len(cent_tags) - 1, ncols - 1))
            add_split_panel_legends(
                axes,
                anchor_left=left_anchor,
                anchor_right=legend_anchor,
                loc_left="upper right",
                loc_right="lower right",
                ncol=1,
                fontsize=legend_fontsize,
                bold_labels=legend_bold,
            )
        else:
            add_panel_legend(axes, anchor_index=legend_anchor, loc="lower right", ncol=1, fontsize=legend_fontsize, bold_labels=legend_bold)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0), pad=0.20, w_pad=0.0, h_pad=0.0)
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        out[yname] = (fig, rows)

    return out


def plot_state_vs_pt_mastergrid(
    setup: SystemSetup,
    formation: str,
    state: str,
    cnm: CNMStateProducts,
    prim: PrimordialProducts,
    include_primordial: bool = True,
    include_combined: bool = True,
    show_cnm_components: bool = False,
    show_alt_cnm: bool = False,
    band_mode: str = "quadrature",
    include_experiment: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    note_loc_first: str = "lower right",
    legend_split: bool = False,
    legend_fontsize: float = 8.0,
    legend_bold: bool = False,
    text_style: Optional[Mapping[str, object]] = None,
) -> Tuple[plt.Figure, List[dict]]:
    rows: List[dict] = []
    tau = prim.tauform.get(state) if prim.available else None
    tau_note = None
    if tau is not None:
        tau_note = rf"$\tau_{{form}}=[{tau[0]:.2g},{tau[1]:.2g}]\ \mathrm{{fm}}/c$"

    cent_tags = [canonical_cent(f"{a}-{b}%") for (a, b) in setup.cent_bins] + ["MB"]
    ncols = len(cent_tags)
    nrows = len(setup.y_windows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    ts = TEXT_STYLE_DEFAULTS.copy()
    if text_style:
        ts.update(dict(text_style))

    for irow, (yname, y0, y1) in enumerate(iter_ywins(setup.y_windows)):
        p_dict = cnm.pt_by_variant[cnm.primary_variant][yname]
        for jcol, cent_tag in enumerate(cent_tags):
            ax = axes[irow, jcol]
            if "cnm" not in p_dict or cent_tag not in p_dict["cnm"]:
                ax.set_visible(False)
                continue
            cnm_band = p_dict["cnm"][cent_tag]

            ppert = None
            pnpw = None
            if prim.available:
                p_pert = extract_prim_pt(prim, state, "Pert", yname, cent_tag)
                p_npw = extract_prim_pt(prim, state, "NPWLC", yname, cent_tag)
                if p_pert is not None and p_npw is not None:
                    x_p, c_p, lo_p, hi_p = p_pert
                    x_n, c_n, lo_n, hi_n = p_npw
                    ppert = (
                        align_to_ref(x_p, c_p, cnm.pt_cent),
                        align_to_ref(x_p, lo_p, cnm.pt_cent),
                        align_to_ref(x_p, hi_p, cnm.pt_cent),
                    )
                    pnpw = (
                        align_to_ref(x_n, c_n, cnm.pt_cent),
                        align_to_ref(x_n, lo_n, cnm.pt_cent),
                        align_to_ref(x_n, hi_n, cnm.pt_cent),
                    )

            curves = compose_axis_curves(
                cnm_band,
                ppert,
                pnpw,
                include_primordial=include_primordial,
                include_combined=include_combined,
                band_mode=band_mode,
            )

            alt_variant = None
            if show_alt_cnm:
                alt = "abs" if cnm.primary_variant == "no_abs" else "no_abs"
                curves["cnm_alt"] = cnm.pt_by_variant[alt][yname]["cnm"][cent_tag]
                alt_variant = alt

            for key, band in curves.items():
                if key == "cnm_total":
                    _plot_cnm_band(ax, cnm.pt_cent, band, cnm.primary_variant, label="CNM")
                elif key == "cnm_alt" and alt_variant is not None:
                    _plot_cnm_band(ax, cnm.pt_cent, band, alt_variant)
                else:
                    _plot_band(ax, cnm.pt_cent, band, key, step=True)
                _append_rows(rows, setup, formation, state, "vs_pt", yname, cent_tag, cnm.pt_cent, band, key)

            if show_cnm_components:
                for comp in ("npdf", "eloss", "broad", "eloss_broad"):
                    if comp in p_dict and cent_tag in p_dict[comp]:
                        _plot_component_band(ax, cnm.pt_cent, p_dict[comp][cent_tag], comp)

            if include_experiment:
                if cent_tag == "MB":
                    overlay_experiment(ax, setup, "vs_pt", state, rap_window=(y0, y1), cent_tag="MB")
                else:
                    overlay_experiment(ax, setup, "vs_pt_cent", state, rap_window=(y0, y1), cent_tag=cent_tag)

            row_i = irow
            col_i = jcol
            style_axis(
                ax,
                xlabel=r"$p_T$ [GeV]" if row_i == nrows - 1 else "",
                ylabel=setup.r_label,
                xlim=(float(setup.pt_edges.min()), float(setup.pt_edges.max())) if xlim is None else xlim,
                ylim=(0.0, 1.6) if ylim is None else ylim,
                tag=None,
                note=None,
                system_tag=None,
                state_tag=None,
                note_loc=note_loc_first if (row_i == 0 and col_i == 0) else "lower left",
                text_style=text_style,
                hide_x=(row_i < nrows - 1),
                hide_y=(col_i > 0),
            )
            _apply_tick_pruning(ax, row_i, col_i, nrows, ncols)

            # Centrality tag (top-right)
            ax.text(
                0.95,
                0.95,
                cent_tag if cent_tag != "MB" else "MB",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=float(ts["tag_fontsize"]),
                color=ts["tag_color"],
                fontweight="bold",
            )
            # Rapidity tag (bottom-right) on every panel
            ax.text(
                0.95,
                0.06,
                rf"$y \in [{y0:.2f}, {y1:.2f}]$",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=float(ts["tag_fontsize"]),
                color=ts["tag_color"],
                fontweight="bold",
            )
            # System + state (left-top) and tau (left-bottom) on first panel
            if row_i == 0 and col_i == 0:
                ax.text(
                    0.03,
                    0.95,
                    _system_label(setup),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=float(ts["system_fontsize"]),
                    color=str(ts["system_color"]),
                    fontweight=str(ts["system_weight"]),
                )
                ax.text(
                    0.03,
                    0.86,
                    _state_bold_label(state),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=float(ts["state_fontsize"]),
                    color=str(ts["state_color"]),
                    fontweight=str(ts["state_weight"]),
                )
                if tau_note:
                    ax.text(
                        0.03,
                        0.06,
                        tau_note,
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=float(ts["note_fontsize"]),
                        bbox=dict(facecolor="white", alpha=float(ts["note_box_alpha"]), edgecolor="none", pad=1.0),
                    )

    # Legends: bottom-left on peripheral + MB panels
    def _collect_theory_handles(ax_arr):
        seen = set()
        handles, labels = [], []
        for ax in np.atleast_2d(ax_arr).ravel():
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if not ll or ll in seen:
                    continue
                if any(k in ll for k in ("data", "ALICE", "PHENIX", "STAR")):
                    continue
                seen.add(ll)
                handles.append(hh)
                labels.append(ll)
        return handles, labels

    h_theory, l_theory = _collect_theory_handles(axes)
    if h_theory:
        legend_row = nrows - 1
        periph_col = max(0, ncols - 2)
        mb_col = ncols - 1
        split_idx = max(1, int(np.ceil(len(h_theory) / 2.0)))
        if legend_split and ncols > 1:
            axes[legend_row, periph_col].legend(
                h_theory[:split_idx],
                l_theory[:split_idx],
                loc="lower left",
                fontsize=legend_fontsize,
                frameon=False,
            )
            axes[legend_row, mb_col].legend(
                h_theory[split_idx:],
                l_theory[split_idx:],
                loc="lower left",
                fontsize=legend_fontsize,
                frameon=False,
            )
        else:
            axes[legend_row, mb_col].legend(
                h_theory,
                l_theory,
                loc="lower left",
                fontsize=legend_fontsize,
                frameon=False,
            )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0), pad=0.20, w_pad=0.0, h_pad=0.0)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig, rows


def save_rows_csv(rows: List[dict], out_csv: Path) -> None:
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def save_figure(fig: plt.Figure, out_path: Path, save_pdf: bool = True, save_png: bool = False, dpi: int = 180) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    if save_png:
        fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def summarize_products(cnm_by_state: Mapping[str, CNMStateProducts], prim: PrimordialProducts) -> pd.DataFrame:
    rows = []
    for state, p in cnm_by_state.items():
        rows.append(
            {
                "state": state,
                "primary_variant": p.primary_variant,
                "n_y_bins": int(len(p.y_cent)),
                "n_pt_bins": int(len(p.pt_cent)),
                "n_cent_bins": int(len(p.cent_mids)),
                "primordial_available": prim.available,
            }
        )
    return pd.DataFrame(rows)
