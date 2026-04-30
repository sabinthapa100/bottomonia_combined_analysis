#!/usr/bin/env python3
"""CNM(+/-nucl abs) x Primordial(radius) + experimental overlays for LHC pPb 8.16 TeV.

Additive runner (does not modify legacy notebooks/modules).
Default curves shown:
- CNM without nuclear absorption (gray)
- CNM with nuclear absorption
- Primordial(Pert) x CNM without absorption (blue)
- Primordial(NPWLC) x CNM without absorption (green)
- Primordial(Pert) x CNM with absorption (blue dashed)
- Primordial(NPWLC) x CNM with absorption (green dashed)

Focus: state-by-state outputs for jpsi_1S and psi_2S.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "eloss_code"))
sys.path.insert(0, str(PROJECT / "npdf_code"))
sys.path.insert(0, str(PROJECT / "cnm_combine"))
sys.path.insert(0, str(PROJECT / "cnm_withExperimental"))
sys.path.insert(0, str(PROJECT / "experimental_helpers"))
sys.path.insert(0, str(PROJECT / "primordial_code"))
sys.path.insert(0, str(PROJECT / "primordial_notebooks"))

from cnm_combine_fast_nuclabs import CNMCombineFast
from system_configs import LHCConfig as LHC_CNM_CONFIG
from primordial_module import ReaderConfig, Style, Y_WINDOW_BACKWARD, Y_WINDOW_CENTRAL, Y_WINDOW_FORWARD, make_bins_from_width
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


STATE_ORDER_DEFAULT = ("jpsi_1S", "psi_2S")
STATE_LABELS = {
    "jpsi_1S": r"$J/\psi(1S)$",
    "chicJ_1P": r"$\chi_c(1P)$",
    "psi_2S": r"$\psi(2S)$",
}

STATE_TO_CNM = {
    "jpsi_1S": "Jpsi",
    "chicJ_1P": "1P",
    "psi_2S": "psi2S",
}

M_STATE_GEV = {
    "jpsi_1S": 3.0969,
    "chicJ_1P": 3.5107,
    "psi_2S": 3.6861,
}

# State-dependent absorption knobs (easy to tune)
SIGMA_ABS_BY_STATE_MB = {
    "jpsi_1S": (0.5, 0.0, 1.0),
    "chicJ_1P": (0.5, 0.0, 1.0),
    "psi_2S": (0.5, 0.0, 1.0),
}

# Default visible components
DEFAULT_COMPONENTS = (
    "cnm_wo_abs",
    "cnm_w_abs",
    "comb_wo_abs_pert",
    "comb_wo_abs_npwlc",
    "comb_w_abs_pert",
    "comb_w_abs_npwlc",
)

# Colors: blue and green reserved for Pert/NPWLC as requested.
COMP_STYLE = {
    "cnm_wo_abs": {"color": "gray", "ls": "-", "label": "CNM w/out Nucl Abs"},
    "cnm_w_abs": {"color": "#7B2CBF", "ls": "-", "label": "CNM w/ Nucl Abs"},
    "comb_wo_abs_pert": {"color": "tab:blue", "ls": "-", "label": "CNM w/out Abs x Prim (Pert)"},
    "comb_wo_abs_npwlc": {"color": "tab:green", "ls": "-", "label": "CNM w/out Abs x Prim (NPWLC)"},
    "comb_w_abs_pert": {"color": "tab:blue", "ls": "--", "label": "CNM w/ Abs x Prim (Pert)"},
    "comb_w_abs_npwlc": {"color": "tab:green", "ls": "--", "label": "CNM w/ Abs x Prim (NPWLC)"},
}

Y_WINDOWS = {
    "backward": Y_WINDOW_BACKWARD,
    "central": Y_WINDOW_CENTRAL,
    "forward": Y_WINDOW_FORWARD,
}

MINBIAS_TAGS = {"MB", "0-100%", "0-100%(MB)", "0-100% (MB)", "Minimum Bias"}


def _canonical_cent(tag: str) -> str:
    s = str(tag).replace("–", "-").replace("%", "").strip()
    if "mb" in s.lower() or s in ("0-100 (MB)", "0-100(MB)"):
        return "MB"
    if "-" not in s:
        return "MB" if s.lower() in ("mb", "minimum bias") else f"{s}%"
    lo, hi = [p.strip() for p in s.split("-", 1)]
    return f"{int(float(lo))}-{int(float(hi))}%"


def _iter_ywins(y_wins: Mapping[str, Tuple[float, float]]) -> Iterable[Tuple[str, float, float]]:
    for key, win in y_wins.items():
        y0, y1 = tuple(win)
        yield key, float(y0), float(y1)


def _step_from_centers(x_cent: Sequence[float], vals: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
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


def _band_envelope_product(
    cnm_c: np.ndarray,
    cnm_lo: np.ndarray,
    cnm_hi: np.ndarray,
    pr_c: np.ndarray,
    pr_lo: np.ndarray,
    pr_hi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rc = cnm_c * pr_c
    cands = np.stack(
        [
            cnm_lo * pr_lo,
            cnm_lo * pr_hi,
            cnm_hi * pr_lo,
            cnm_hi * pr_hi,
        ],
        axis=0,
    )
    return rc, np.min(cands, axis=0), np.max(cands, axis=0)


def _style_ax(ax, xlabel: str, ylabel: str, xlim: Tuple[float, float], ylim: Tuple[float, float], tag: str | None = None, note: str | None = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if tag:
        ax.text(0.97, 0.95, tag, transform=ax.transAxes, ha="right", va="top", fontsize=10)
    if note:
        ax.text(0.03, 0.03, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))


def _plot_band(ax, x_cent: np.ndarray, c: np.ndarray, lo: np.ndarray, hi: np.ndarray, style_key: str, step: bool = True, alpha: float = 0.20) -> None:
    st = COMP_STYLE[style_key]
    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)
    lo2 = np.minimum(lo2, c)
    hi2 = np.maximum(hi2, c)
    if step:
        xe, yc = _step_from_centers(x_cent, c)
        _, yl = _step_from_centers(x_cent, lo2)
        _, yh = _step_from_centers(x_cent, hi2)
        ax.step(xe, yc, where="post", color=st["color"], ls=st["ls"], lw=2.0, label=st["label"])
        ax.fill_between(xe, yl, yh, step="post", color=st["color"], alpha=alpha, lw=0)
    else:
        ax.plot(x_cent, c, color=st["color"], ls=st["ls"], lw=2.0, label=st["label"])
        ax.fill_between(x_cent, lo2, hi2, color=st["color"], alpha=alpha, lw=0)


def _lhc_win_str(rw: Tuple[float, float] | None) -> str:
    if rw is None:
        return "midrapidity"
    y0, y1 = rw
    if y1 < -1.0:
        return "backward"
    if y0 > 1.0:
        return "forward"
    return "midrapidity"


def _filter_rapidity_df(df: pd.DataFrame, rap_window: Tuple[float, float] | None) -> pd.DataFrame:
    if df is None or df.empty or rap_window is None or "rapidity" not in df.columns:
        return df
    y0, y1 = rap_window
    w = _lhc_win_str((y0, y1))
    try:
        yc = df["rapidity"].apply(_ycms_center)
        if w == "backward":
            return df[yc < -1.0]
        if w == "forward":
            return df[yc > 1.0]
        return df[(yc >= -1.0) & (yc <= 1.0)]
    except Exception:
        return df


def _infer_cent_tags_from_ncoll(ncoll_series: pd.Series) -> List[str]:
    # Keep parity with existing LHC 8 TeV overlay behavior.
    n = len(np.unique(ncoll_series.dropna()))
    if n >= 9:
        return ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-80%", "80-100%"]
    return ["2-10%", "10-20%", "20-40%", "40-60%", "60-80%", "80-90%"]


def _map_ncoll_to_cent_mid(ncoll_series: pd.Series) -> pd.Series:
    tags = _infer_cent_tags_from_ncoll(ncoll_series)
    mids = [0.5 * (float(t.split("-")[0]) + float(t.split("-")[1].replace("%", ""))) for t in tags]
    uniq = np.array(sorted(np.unique(ncoll_series), reverse=True), float)
    n_use = min(len(uniq), len(tags))
    mids = [0.5 * (float(tags[i].split("-")[0]) + float(tags[i].split("-")[1].replace("%", ""))) for i in range(n_use)]

    mapping = {uniq[i]: mids[i] for i in range(n_use)}

    def _closest(v: float) -> float:
        key = min(mapping.keys(), key=lambda k: abs(k - v))
        return mapping[key]

    return ncoll_series.apply(_closest)


def _overlay_exp_vs_cent(ax, state: str, rap_window: Tuple[float, float]) -> None:
    reader = HEPCentDataReader(PROJECT / "input" / "experimental_input")
    state_hep = "Jpsi" if state == "jpsi_1S" else "psi2S"
    df = reader.rpa_vs_ncoll("8TeV", state=state_hep)
    df = _filter_rapidity_df(df, rap_window)
    if df is None or df.empty:
        return
    err = np.sqrt(df["stat_up"] ** 2 + df["sys_uncorr_up"] ** 2)
    x = _map_ncoll_to_cent_mid(df["ncoll"])
    marker = "D" if state == "jpsi_1S" else "^"
    mfc = "black" if state == "jpsi_1S" else "none"
    ax.errorbar(x, df["value"], yerr=err, fmt=marker, color="black", mfc=mfc, mew=1.2, ms=6.5,
                label="ALICE data", zorder=20, capsize=2)


def _overlay_exp_vs_y(ax, state: str) -> None:
    mb_df = load_minbias_data(verbose=False)
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
        mb_df = pd.concat([mb_df] + extras, ignore_index=True)

    s_long = "jpsi_1s" if state == "jpsi_1S" else "psi2s_2s"
    data = get_RpA_vs_y(mb_df, state=s_long, energy_tev=8.16)
    if data.empty:
        return
    marker = "D" if state == "jpsi_1S" else "^"
    if "collaboration" in data.columns:
        for collab, g in data.groupby("collaboration"):
            color = "black" if collab == "ALICE" else "#1f77b4"
            mfc = color if state == "jpsi_1S" else "none"
            ax.errorbar(g["variable"], g["value"], yerr=g["total_err"], fmt=marker, color=color, mfc=mfc,
                        mew=1.2, ms=6.0, capsize=2, label=f"{collab} data", zorder=20)
    else:
        mfc = "black" if state == "jpsi_1S" else "none"
        ax.errorbar(data["variable"], data["value"], yerr=data["total_err"], fmt=marker, color="black", mfc=mfc,
                    mew=1.2, ms=6.0, capsize=2, label="ALICE data", zorder=20)


def _overlay_exp_vs_pt(ax, state: str, rap_window: Tuple[float, float], cent_tag: str) -> None:
    cent_norm = _canonical_cent(cent_tag)
    if cent_norm == "MB":
        mb_df = load_minbias_data(verbose=False)
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
            mb_df = pd.concat([mb_df] + extras, ignore_index=True)
        s_long = "jpsi_1s" if state == "jpsi_1S" else "psi2s_2s"
        data = get_RpA_vs_pT(mb_df, state=s_long, energy_tev=8.16, rapidity_window=_lhc_win_str(rap_window))
        if data.empty:
            return
        marker = "D" if state == "jpsi_1S" else "^"
        if "collaboration" in data.columns:
            for collab, g in data.groupby("collaboration"):
                color = "black" if collab == "ALICE" else "#1f77b4"
                mfc = color if state == "jpsi_1S" else "none"
                ax.errorbar(g["variable"], g["value"], yerr=g["total_err"], fmt=marker, color=color, mfc=mfc,
                            mew=1.2, ms=5.8, capsize=2, label=f"{collab} data", zorder=20)
        else:
            mfc = "black" if state == "jpsi_1S" else "none"
            ax.errorbar(data["variable"], data["value"], yerr=data["total_err"], fmt=marker, color="black", mfc=mfc,
                        mew=1.2, ms=5.8, capsize=2, label="ALICE data", zorder=20)
        return

    reader = HEPCentDataReader(PROJECT / "input" / "experimental_input")
    state_hep = "Jpsi" if state == "jpsi_1S" else "psi2S"
    groups = reader.rpa_vs_pt("8TeV", state=state_hep)
    if not groups:
        return
    marker = "D" if state == "jpsi_1S" else "^"
    for g in groups:
        if g is None or g.empty:
            continue
        g = _filter_rapidity_df(g, rap_window)
        if g.empty:
            continue
        data_cent = str(g["centrality"].iloc[0]) if "centrality" in g.columns else None
        if data_cent is None:
            continue
        dcan = _canonical_cent(data_cent)
        if dcan != cent_norm:
            continue
        x = g["x_cen"] if "x_cen" in g.columns else 0.5 * (g["x_low"] + g["x_high"])
        err = np.sqrt(g["stat_up"] ** 2 + g["sys_uncorr_up"] ** 2)
        mfc = "black" if state == "jpsi_1S" else "none"
        ax.errorbar(x, g["value"], yerr=err, fmt=marker, color="black", mfc=mfc, mew=1.2, ms=5.8,
                    capsize=2, label="ALICE data", zorder=20)


@dataclass
class PrimordialProducts:
    vs_y: dict
    vs_pt: dict
    vs_cent: dict
    tauform: Dict[str, Tuple[float, float]]


@dataclass
class CNMProducts:
    y_cent: np.ndarray
    pt_cent: np.ndarray
    cent_mids: np.ndarray
    cent_edges: np.ndarray
    vs_y_wo: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    vs_y_w: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    vs_pt_wo: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]
    vs_pt_w: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]
    vs_cent_wo: Dict[str, Dict[str, float]]
    vs_cent_w: Dict[str, Dict[str, float]]


def _build_primordial(form: str, states: Sequence[str], cent_bins: Sequence[Tuple[int, int]], y_edges: np.ndarray, pt_edges: np.ndarray, pt_window: Tuple[float, float]) -> PrimordialProducts:
    generated = PROJECT / "primordial_output" / "glauber_data" / "8TeV_eloss"
    generate_primordial_glauber_maps(
        generated,
        cfg=GlauberBridgeConfig(
            roots_gev=8160.0,
            target_a=208,
            system="pA",
            bmax_fm=20.0,
            nb=401,
            include_npart=True,
            verbose=False,
        ),
    )
    _validate_glauber_maps(generated)

    combos = _load_primordial_combos("8.16", form, generated, ReaderConfig(debug=False))
    y_bins = [(float(y_edges[i]), float(y_edges[i + 1])) for i in range(len(y_edges) - 1)]
    pt_bins = [(float(pt_edges[i]), float(pt_edges[i + 1])) for i in range(len(pt_edges) - 1)]

    prim_y = primordial_vs_y_all(combos, list(states), list(cent_bins), pt_window, y_bins)
    prim_pt = primordial_vs_pT_all(combos, list(states), list(cent_bins), Y_WINDOWS, pt_bins)
    prim_cent = primordial_vs_cent_from_vs_y(prim_y, Y_WINDOWS, list(states), ("Pert", "NPWLC"))
    tauform = _tauform_for_form(form)

    return PrimordialProducts(vs_y=prim_y, vs_pt=prim_pt, vs_cent=prim_cent, tauform=tauform)


def _pack_vs_axis(bands: Mapping[str, Tuple[dict, dict, dict]], cent_tag: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    t = _canonical_cent(cent_tag)
    for comp, (rc, lo, hi) in bands.items():
        k = cent_tag if cent_tag in rc else t
        if k not in rc and t == "MB" and "MB" in rc:
            k = "MB"
        if k not in rc:
            continue
        out[comp] = (np.asarray(rc[k], float), np.asarray(lo[k], float), np.asarray(hi[k], float))
    return out


def _build_cnm_for_state(state: str, cent_bins: Sequence[Tuple[int, int]], y_edges: np.ndarray, pt_edges: np.ndarray, y_windows: Mapping[str, Tuple[float, float]], pt_window: Tuple[float, float]) -> CNMProducts:
    sigma_c, sigma_lo, sigma_hi = SIGMA_ABS_BY_STATE_MB[state]

    cnm_wo = CNMCombineFast.from_defaults(
        energy="8.16",
        family="charmonia",
        particle_state=STATE_TO_CNM[state],
        alpha_s_mode="constant",
        alpha0=0.5,
        cent_bins=cent_bins,
        enable_absorption=False,
        m_state_for_np=M_STATE_GEV[state],
    )

    cnm_w = CNMCombineFast.from_defaults(
        energy="8.16",
        family="charmonia",
        particle_state=STATE_TO_CNM[state],
        alpha_s_mode="constant",
        alpha0=0.5,
        cent_bins=cent_bins,
        enable_absorption=True,
        abs_sigma_mb=sigma_c,
        abs_mode="avg_TA",
        m_state_for_np=M_STATE_GEV[state],
    )

    y_cent, labels_y, bands_y_wo = cnm_wo.cnm_vs_y(
        y_edges=y_edges,
        pt_range_avg=pt_window,
        components=["cnm"],
        include_mb=True,
    )
    _, _, bands_y_w = cnm_w.cnm_vs_y(
        y_edges=y_edges,
        pt_range_avg=pt_window,
        components=["cnm"],
        include_mb=True,
    )

    vs_pt_wo = {}
    vs_pt_w = {}
    pt_cent_ref = None
    for yname, y0, y1 in _iter_ywins(y_windows):
        pt_cent, _, bands_wo = cnm_wo.cnm_vs_pT(
            y_window=(y0, y1),
            pt_edges=pt_edges,
            components=["cnm"],
            include_mb=True,
        )
        _, _, bands_w = cnm_w.cnm_vs_pT(
            y_window=(y0, y1),
            pt_edges=pt_edges,
            components=["cnm"],
            include_mb=True,
        )
        if pt_cent_ref is None:
            pt_cent_ref = np.asarray(pt_cent, float)
        vs_pt_wo[yname] = {t: vals for t, vals in _pack_vs_axis(bands_wo, "MB").items()}
        vs_pt_w[yname] = {t: vals for t, vals in _pack_vs_axis(bands_w, "MB").items()}

        for a, b in cent_bins:
            tag = f"{a}-{b}%"
            vs_pt_wo[yname][tag] = _pack_vs_axis(bands_wo, tag)["cnm"]
            vs_pt_w[yname][tag] = _pack_vs_axis(bands_w, tag)["cnm"]
        vs_pt_wo[yname]["MB"] = _pack_vs_axis(bands_wo, "MB")["cnm"]
        vs_pt_w[yname]["MB"] = _pack_vs_axis(bands_w, "MB")["cnm"]

    # centrality
    vs_cent_wo = {}
    vs_cent_w = {}
    cent_mids = np.array([0.5 * (a + b) for (a, b) in cent_bins], float)
    cent_edges = np.array([cent_bins[0][0]] + [b for (_, b) in cent_bins], float)
    for yname, y0, y1 in _iter_ywins(y_windows):
        cwo = cnm_wo.cnm_vs_centrality(y_window=(y0, y1), pt_range_avg=pt_window, components=["cnm"], include_mb=True)
        cw = cnm_w.cnm_vs_centrality(y_window=(y0, y1), pt_range_avg=pt_window, components=["cnm"], include_mb=True)
        arr_wo = cwo["cnm"]
        arr_w = cw["cnm"]

        def _arr_to_dict(arr):
            vals, lo, hi, mb, mb_lo, mb_hi = arr
            out = {}
            for i, (a, b) in enumerate(cent_bins):
                tag = f"{a}-{b}%"
                out[tag] = (float(vals[i]), float(lo[i]), float(hi[i]))
            out["MB"] = (float(mb), float(mb_lo), float(mb_hi))
            return out

        vs_cent_wo[yname] = _arr_to_dict(arr_wo)
        vs_cent_w[yname] = _arr_to_dict(arr_w)

    out_y_wo = {}
    out_y_w = {}
    for a, b in cent_bins:
        tag = f"{a}-{b}%"
        out_y_wo[tag] = _pack_vs_axis(bands_y_wo, tag)["cnm"]
        out_y_w[tag] = _pack_vs_axis(bands_y_w, tag)["cnm"]
    out_y_wo["MB"] = _pack_vs_axis(bands_y_wo, "MB")["cnm"]
    out_y_w["MB"] = _pack_vs_axis(bands_y_w, "MB")["cnm"]

    return CNMProducts(
        y_cent=np.asarray(y_cent, float),
        pt_cent=np.asarray(pt_cent_ref, float),
        cent_mids=cent_mids,
        cent_edges=cent_edges,
        vs_y_wo=out_y_wo,
        vs_y_w=out_y_w,
        vs_pt_wo=vs_pt_wo,
        vs_pt_w=vs_pt_w,
        vs_cent_wo=vs_cent_wo,
        vs_cent_w=vs_cent_w,
    )


def _extract_prim_y(prim: PrimordialProducts, state: str, model: str, cent_tag: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    tags = prim.vs_y.get(model, {})
    for t, block in tags.items():
        if _canonical_cent(t) == _canonical_cent(cent_tag) and state in block:
            rc, lo, hi, x = block[state]
            return np.asarray(x, float), np.asarray(rc, float), np.asarray(lo, float), np.asarray(hi, float)
    return None


def _extract_prim_pt(prim: PrimordialProducts, state: str, model: str, yname: str, cent_tag: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    block = prim.vs_pt.get(yname, {}).get(model, {})
    for t, by_state in block.items():
        if _canonical_cent(t) == _canonical_cent(cent_tag) and state in by_state:
            rc, lo, hi, x = by_state[state]
            return np.asarray(x, float), np.asarray(rc, float), np.asarray(lo, float), np.asarray(hi, float)
    return None


def _extract_prim_cent(prim: PrimordialProducts, state: str, model: str, yname: str, cent_bins: Sequence[Tuple[int, int]]) -> Dict[str, Tuple[float, float, float]]:
    out = {}
    block = prim.vs_cent.get(yname, {}).get(model, {})
    if state not in block:
        return out
    cent_mid, rc, lo, hi, rc_mb, lo_mb, hi_mb = block[state]
    cent_mid = np.asarray(cent_mid, float)
    rc = np.asarray(rc, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    for a, b in cent_bins:
        mid = 0.5 * (a + b)
        idx = int(np.argmin(np.abs(cent_mid - mid)))
        out[f"{a}-{b}%"] = (float(rc[idx]), float(lo[idx]), float(hi[idx]))
    out["MB"] = (float(rc_mb), float(lo_mb), float(hi_mb))
    return out


def _align_to_ref(x_src: np.ndarray, vals: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    x_src = np.asarray(x_src, float)
    vals = np.asarray(vals, float)
    x_ref = np.asarray(x_ref, float)
    if x_src.size == x_ref.size and np.allclose(x_src, x_ref):
        return vals
    return np.interp(x_ref, x_src, vals, left=vals[0], right=vals[-1])


def _compose_components(
    cnm_c: Tuple[np.ndarray, np.ndarray, np.ndarray],
    cnm_abs_c: Tuple[np.ndarray, np.ndarray, np.ndarray],
    prim_pert: Tuple[np.ndarray, np.ndarray, np.ndarray],
    prim_npwlc: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {
        "cnm_wo_abs": cnm_c,
        "cnm_w_abs": cnm_abs_c,
    }
    out["comb_wo_abs_pert"] = _band_envelope_product(cnm_c[0], cnm_c[1], cnm_c[2], prim_pert[0], prim_pert[1], prim_pert[2])
    out["comb_wo_abs_npwlc"] = _band_envelope_product(cnm_c[0], cnm_c[1], cnm_c[2], prim_npwlc[0], prim_npwlc[1], prim_npwlc[2])
    out["comb_w_abs_pert"] = _band_envelope_product(cnm_abs_c[0], cnm_abs_c[1], cnm_abs_c[2], prim_pert[0], prim_pert[1], prim_pert[2])
    out["comb_w_abs_npwlc"] = _band_envelope_product(cnm_abs_c[0], cnm_abs_c[1], cnm_abs_c[2], prim_npwlc[0], prim_npwlc[1], prim_npwlc[2])
    return out


def _save_table(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _dedupe_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    seen = set()
    h_out, l_out = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h_out.append(h)
        l_out.append(l)
    ax.legend(h_out, l_out, frameon=False, fontsize=8)


def _configure_style() -> None:
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


def _parse_show_components(s: str) -> Tuple[str, ...]:
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    valid = set(COMP_STYLE.keys())
    bad = [t for t in tokens if t not in valid]
    if bad:
        raise ValueError(f"Unknown component(s) in --show-components: {bad}. Valid: {sorted(valid)}")
    return tuple(tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy", default="8.16", choices=["8.16"], help="This runner is fixed to LHC 8.16 TeV.")
    parser.add_argument("--formation", default="radius", choices=["radius", "new", "old"])
    parser.add_argument("--states", default=",".join(STATE_ORDER_DEFAULT), help="Comma list from: jpsi_1S,chicJ_1P,psi_2S")
    parser.add_argument("--observables", default="cent,y,pt", help="Comma list from: cent,y,pt")
    parser.add_argument("--show-components", default=",".join(DEFAULT_COMPONENTS), help="Comma list of component ids.")
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    states = tuple([s.strip() for s in args.states.split(",") if s.strip()])
    allowed_states = set(STATE_LABELS)
    if not states or any(s not in allowed_states for s in states):
        raise ValueError(f"Invalid --states. Allowed: {sorted(allowed_states)}")

    observables = tuple([x.strip() for x in args.observables.split(",") if x.strip()])
    valid_obs = {"cent", "y", "pt"}
    if any(o not in valid_obs for o in observables):
        raise ValueError(f"Invalid --observables. Allowed: {sorted(valid_obs)}")

    show_components = _parse_show_components(args.show_components)

    _configure_style()

    cent_bins = [(int(a), int(b)) for (a, b) in LHC_CNM_CONFIG.cent_bins_plotting]
    y_edges = np.linspace(-5.0, 5.0, 21)
    pt_edges = np.asarray(LHC_CNM_CONFIG.pt_bins, float) if hasattr(LHC_CNM_CONFIG, "pt_bins") else np.arange(0.0, 20.5, 2.5)
    if pt_edges.ndim != 1 or pt_edges.size < 2:
        pt_edges = np.arange(0.0, 20.5, 2.5)
    cfg_ptr = LHC_CNM_CONFIG.pt_range_integrated
    if isinstance(cfg_ptr, (list, tuple)) and len(cfg_ptr) == 2 and not isinstance(cfg_ptr[0], (list, tuple)):
        pt_window = (float(cfg_ptr[0]), float(cfg_ptr[1]))
    else:
        pt_window = tuple(float(x) for x in cfg_ptr[0])

    out_base = PROJECT / "outputs" / "combined_cnm_primordial_experimental" / "LHC_8p16TeV" / args.formation
    for model in ("Pert", "NPWLC"):
        (out_base / model).mkdir(parents=True, exist_ok=True)

    print("[step 1] building primordial products...")
    prim = _build_primordial(args.formation, states, cent_bins, y_edges, pt_edges, pt_window)

    rows_all = []

    for state in states:
        print(f"[step 2] computing CNM for {state} ...")
        cnm = _build_cnm_for_state(state, cent_bins, y_edges, pt_edges, Y_WINDOWS, pt_window)
        tau_lo, tau_hi = prim.tauform[state]
        note_state = rf"p+Pb $\sqrt{{s_{{NN}}}}=8.16$ TeV\n$\tau_{{form}}={tau_lo:.2g}-{tau_hi:.2g}$ fm"

        if "cent" in observables:
            fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharey=True)
            for iax, (yname, y0, y1) in enumerate(_iter_ywins(Y_WINDOWS)):
                ax = axes[iax]
                tag = rf"${y0:.2f} < y < {y1:.2f}$"
                prim_pert = _extract_prim_cent(prim, state, "Pert", yname, cent_bins)
                prim_npwlc = _extract_prim_cent(prim, state, "NPWLC", yname, cent_bins)

                x_cent = cnm.cent_mids
                cnm_c = np.array([cnm.vs_cent_wo[yname][f"{a}-{b}%"][0] for (a, b) in cent_bins], float)
                cnm_lo = np.array([cnm.vs_cent_wo[yname][f"{a}-{b}%"][1] for (a, b) in cent_bins], float)
                cnm_hi = np.array([cnm.vs_cent_wo[yname][f"{a}-{b}%"][2] for (a, b) in cent_bins], float)
                cnm_a_c = np.array([cnm.vs_cent_w[yname][f"{a}-{b}%"][0] for (a, b) in cent_bins], float)
                cnm_a_lo = np.array([cnm.vs_cent_w[yname][f"{a}-{b}%"][1] for (a, b) in cent_bins], float)
                cnm_a_hi = np.array([cnm.vs_cent_w[yname][f"{a}-{b}%"][2] for (a, b) in cent_bins], float)

                p_pert_c = np.array([prim_pert[f"{a}-{b}%"][0] for (a, b) in cent_bins], float)
                p_pert_lo = np.array([prim_pert[f"{a}-{b}%"][1] for (a, b) in cent_bins], float)
                p_pert_hi = np.array([prim_pert[f"{a}-{b}%"][2] for (a, b) in cent_bins], float)
                p_npw_c = np.array([prim_npwlc[f"{a}-{b}%"][0] for (a, b) in cent_bins], float)
                p_npw_lo = np.array([prim_npwlc[f"{a}-{b}%"][1] for (a, b) in cent_bins], float)
                p_npw_hi = np.array([prim_npwlc[f"{a}-{b}%"][2] for (a, b) in cent_bins], float)

                comp = _compose_components(
                    (cnm_c, cnm_lo, cnm_hi),
                    (cnm_a_c, cnm_a_lo, cnm_a_hi),
                    (p_pert_c, p_pert_lo, p_pert_hi),
                    (p_npw_c, p_npw_lo, p_npw_hi),
                )

                for key in show_components:
                    c, lo, hi = comp[key]
                    _plot_band(ax, x_cent, c, lo, hi, key, step=True)
                    for xv, rc, rl, rh in zip(x_cent, c, lo, hi):
                        rows_all.append(
                            {
                                "system": "pPb",
                                "energy": 8.16,
                                "formation": args.formation,
                                "model": "mixed" if "npwlc" in key or "pert" in key else "none",
                                "state": state,
                                "observable": "vs_centrality",
                                "y_window": yname,
                                "centrality_tag": "slice",
                                "x": float(xv),
                                "R": float(rc),
                                "R_lo": float(rl),
                                "R_hi": float(rh),
                                "pt_range": f"{pt_window[0]}-{pt_window[1]}",
                                "y_range": f"{y0}-{y1}",
                                "source_tag": key,
                            }
                        )

                # MB references
                mb_map = {
                    "cnm_wo_abs": cnm.vs_cent_wo[yname]["MB"],
                    "cnm_w_abs": cnm.vs_cent_w[yname]["MB"],
                    "comb_wo_abs_pert": _band_envelope_product(
                        np.array([cnm.vs_cent_wo[yname]["MB"][0]]),
                        np.array([cnm.vs_cent_wo[yname]["MB"][1]]),
                        np.array([cnm.vs_cent_wo[yname]["MB"][2]]),
                        np.array([prim_pert["MB"][0]]),
                        np.array([prim_pert["MB"][1]]),
                        np.array([prim_pert["MB"][2]]),
                    ),
                    "comb_wo_abs_npwlc": _band_envelope_product(
                        np.array([cnm.vs_cent_wo[yname]["MB"][0]]),
                        np.array([cnm.vs_cent_wo[yname]["MB"][1]]),
                        np.array([cnm.vs_cent_wo[yname]["MB"][2]]),
                        np.array([prim_npwlc["MB"][0]]),
                        np.array([prim_npwlc["MB"][1]]),
                        np.array([prim_npwlc["MB"][2]]),
                    ),
                    "comb_w_abs_pert": _band_envelope_product(
                        np.array([cnm.vs_cent_w[yname]["MB"][0]]),
                        np.array([cnm.vs_cent_w[yname]["MB"][1]]),
                        np.array([cnm.vs_cent_w[yname]["MB"][2]]),
                        np.array([prim_pert["MB"][0]]),
                        np.array([prim_pert["MB"][1]]),
                        np.array([prim_pert["MB"][2]]),
                    ),
                    "comb_w_abs_npwlc": _band_envelope_product(
                        np.array([cnm.vs_cent_w[yname]["MB"][0]]),
                        np.array([cnm.vs_cent_w[yname]["MB"][1]]),
                        np.array([cnm.vs_cent_w[yname]["MB"][2]]),
                        np.array([prim_npwlc["MB"][0]]),
                        np.array([prim_npwlc["MB"][1]]),
                        np.array([prim_npwlc["MB"][2]]),
                    ),
                }
                for key in show_components:
                    mb_vals = mb_map[key]
                    if isinstance(mb_vals[0], np.ndarray):
                        rc, lo, hi = float(mb_vals[0][0]), float(mb_vals[1][0]), float(mb_vals[2][0])
                    else:
                        rc, lo, hi = float(mb_vals[0]), float(mb_vals[1]), float(mb_vals[2])
                    st = COMP_STYLE[key]
                    ax.hlines(rc, 0.0, 100.0, colors=st["color"], linestyles=st["ls"], linewidth=1.3, alpha=0.7)
                    ax.fill_between([0.0, 100.0], [lo, lo], [hi, hi], color=st["color"], alpha=0.07)

                _overlay_exp_vs_cent(ax, state, (y0, y1))
                _style_ax(
                    ax,
                    xlabel="Centrality [%]",
                    ylabel=r"$R_{pA}$" if iax == 0 else "",
                    xlim=(0.0, 100.0),
                    ylim=(0.0, 1.6),
                    tag=tag,
                    note=note_state if iax == 0 else None,
                )
                _dedupe_legend(ax)

            fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.2)
            if args.save_pdf:
                fig.savefig(out_base / f"combined_RpA_vs_cent_{state}_8p16TeV_{args.formation}.pdf", bbox_inches="tight")
            plt.close(fig)

        if "y" in observables:
            cent_tags = [f"{a}-{b}%" for (a, b) in cent_bins] + ["MB"]
            ncols = 3
            nrows = int(np.ceil(len(cent_tags) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), sharex=True, sharey=True)
            axes = np.atleast_1d(axes).ravel()
            for i, cent_tag in enumerate(cent_tags):
                ax = axes[i]
                cnm_wo = cnm.vs_y_wo[cent_tag]
                cnm_w = cnm.vs_y_w[cent_tag]
                p_pert = _extract_prim_y(prim, state, "Pert", cent_tag)
                p_npwlc = _extract_prim_y(prim, state, "NPWLC", cent_tag)
                if p_pert is None or p_npwlc is None:
                    continue
                x_p, p_c, p_lo, p_hi = p_pert
                p_c = _align_to_ref(x_p, p_c, cnm.y_cent)
                p_lo = _align_to_ref(x_p, p_lo, cnm.y_cent)
                p_hi = _align_to_ref(x_p, p_hi, cnm.y_cent)
                x_n, n_c, n_lo, n_hi = p_npwlc
                n_c = _align_to_ref(x_n, n_c, cnm.y_cent)
                n_lo = _align_to_ref(x_n, n_lo, cnm.y_cent)
                n_hi = _align_to_ref(x_n, n_hi, cnm.y_cent)

                comp = _compose_components(cnm_wo, cnm_w, (p_c, p_lo, p_hi), (n_c, n_lo, n_hi))
                for key in show_components:
                    c, lo, hi = comp[key]
                    _plot_band(ax, cnm.y_cent, c, lo, hi, key, step=True)
                    for xv, rc, rl, rh in zip(cnm.y_cent, c, lo, hi):
                        rows_all.append(
                            {
                                "system": "pPb",
                                "energy": 8.16,
                                "formation": args.formation,
                                "model": "mixed" if "npwlc" in key or "pert" in key else "none",
                                "state": state,
                                "observable": "vs_y",
                                "y_window": "all" if cent_tag == "MB" else "bin",
                                "centrality_tag": cent_tag,
                                "x": float(xv),
                                "R": float(rc),
                                "R_lo": float(rl),
                                "R_hi": float(rh),
                                "pt_range": f"{pt_window[0]}-{pt_window[1]}",
                                "y_range": "-5-5",
                                "source_tag": key,
                            }
                        )

                if cent_tag == "MB":
                    _overlay_exp_vs_y(ax, state)
                _style_ax(
                    ax,
                    xlabel=r"$y$",
                    ylabel=r"$R_{pA}$",
                    xlim=(-5.0, 5.0),
                    ylim=(0.0, 1.6),
                    tag=cent_tag,
                    note=note_state if i == 0 else None,
                )
                _dedupe_legend(ax)

            for j in range(len(cent_tags), len(axes)):
                axes[j].set_visible(False)

            fig.tight_layout(pad=0.3, w_pad=0.15, h_pad=0.15)
            if args.save_pdf:
                fig.savefig(out_base / f"combined_RpA_vs_y_{state}_8p16TeV_{args.formation}.pdf", bbox_inches="tight")
            plt.close(fig)

        if "pt" in observables:
            for yname, y0, y1 in _iter_ywins(Y_WINDOWS):
                cent_tags = [f"{a}-{b}%" for (a, b) in cent_bins] + ["MB"]
                ncols = 3
                nrows = int(np.ceil(len(cent_tags) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), sharex=True, sharey=True)
                axes = np.atleast_1d(axes).ravel()
                for i, cent_tag in enumerate(cent_tags):
                    ax = axes[i]
                    cnm_wo = cnm.vs_pt_wo[yname][cent_tag]
                    cnm_w = cnm.vs_pt_w[yname][cent_tag]
                    p_pert = _extract_prim_pt(prim, state, "Pert", yname, cent_tag)
                    p_npwlc = _extract_prim_pt(prim, state, "NPWLC", yname, cent_tag)
                    if p_pert is None or p_npwlc is None:
                        continue
                    x_p, p_c, p_lo, p_hi = p_pert
                    p_c = _align_to_ref(x_p, p_c, cnm.pt_cent)
                    p_lo = _align_to_ref(x_p, p_lo, cnm.pt_cent)
                    p_hi = _align_to_ref(x_p, p_hi, cnm.pt_cent)
                    x_n, n_c, n_lo, n_hi = p_npwlc
                    n_c = _align_to_ref(x_n, n_c, cnm.pt_cent)
                    n_lo = _align_to_ref(x_n, n_lo, cnm.pt_cent)
                    n_hi = _align_to_ref(x_n, n_hi, cnm.pt_cent)

                    comp = _compose_components(cnm_wo, cnm_w, (p_c, p_lo, p_hi), (n_c, n_lo, n_hi))
                    for key in show_components:
                        c, lo, hi = comp[key]
                        _plot_band(ax, cnm.pt_cent, c, lo, hi, key, step=True)
                        for xv, rc, rl, rh in zip(cnm.pt_cent, c, lo, hi):
                            rows_all.append(
                                {
                                    "system": "pPb",
                                    "energy": 8.16,
                                    "formation": args.formation,
                                    "model": "mixed" if "npwlc" in key or "pert" in key else "none",
                                    "state": state,
                                    "observable": "vs_pt",
                                    "y_window": yname,
                                    "centrality_tag": cent_tag,
                                    "x": float(xv),
                                    "R": float(rc),
                                    "R_lo": float(rl),
                                    "R_hi": float(rh),
                                    "pt_range": "0-20",
                                    "y_range": f"{y0}-{y1}",
                                    "source_tag": key,
                                }
                            )

                    _overlay_exp_vs_pt(ax, state, (y0, y1), cent_tag)
                    _style_ax(
                        ax,
                        xlabel=r"$p_T$ [GeV]",
                        ylabel=r"$R_{pA}$",
                        xlim=(0.0, 20.0),
                        ylim=(0.0, 1.6),
                        tag=cent_tag,
                        note=note_state if i == 0 else None,
                    )
                    _dedupe_legend(ax)

                for j in range(len(cent_tags), len(axes)):
                    axes[j].set_visible(False)
                fig.tight_layout(pad=0.3, w_pad=0.15, h_pad=0.15)
                if args.save_pdf:
                    fig.savefig(out_base / f"combined_RpA_vs_pT_{state}_{yname}_8p16TeV_{args.formation}.pdf", bbox_inches="tight")
                plt.close(fig)

    if args.save_csv:
        _save_table(rows_all, out_base / f"combined_curves_8p16TeV_{args.formation}.csv")

    print(f"[done] outputs in: {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
