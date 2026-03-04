#!/usr/bin/env python3
"""Primordial 8/5 TeV test pipeline using eloss optical Glauber as map source.

This script is motivated by `primordial_code/primordial_only.ipynb`, but replaces
legacy static `input/glauber_data/...` consumption with maps generated from
`eloss_code.glauber` (optical model) in primordial-compatible TSV format.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


PROJECT = _project_root()
sys.path.insert(0, str(PROJECT / "primordial_code"))
sys.path.insert(0, str(PROJECT / "primordial_notebooks"))
sys.path.insert(0, str(PROJECT))

from primordial_module import (  # noqa: E402
    ReaderConfig,
    Style,
    Y_WINDOW_BACKWARD,
    Y_WINDOW_CENTRAL,
    Y_WINDOW_FORWARD,
    build_ensemble,
    make_bins_from_width,
)
from primordial_code.glauber_bridge import GlauberBridgeConfig, generate_primordial_glauber_maps  # noqa: E402


@dataclass
class PrimordialCombo:
    energy: str
    model: str
    form: str
    ens: object
    runs: Dict[str, object]


STATES = ["jpsi_1S", "chicJ_1P", "psi_2S"]
STATE_LABELS = {
    "jpsi_1S": r"$J/\psi(1S)$",
    "chicJ_1P": r"$\chi_c(1P)$",
    "psi_2S": r"$\psi(2S)$",
}
MODEL_COLORS = {"Pert": "tab:blue", "NPWLC": "tab:green"}
MODEL_LS = {"Pert": "-", "NPWLC": "--"}
MODELS = ("Pert", "NPWLC")

Y_WINDOWS = {
    "forward": Y_WINDOW_FORWARD,
    "central": Y_WINDOW_CENTRAL,
    "backward": Y_WINDOW_BACKWARD,
}

TAUFORM_BINDING = {
    "jpsi_1S": (0.31, 0.62),
    "chicJ_1P": (1.20, 2.40),
    "psi_2S": (4.50, 9.00),
}
TAUFORM_OLD = {
    "jpsi_1S": (1.0, 1.25),
    "chicJ_1P": (1.5, 1.875),
    "psi_2S": (2.5, 3.125),
}
TAUFORM_RADIUS = {
    "jpsi_1S": (1.58, 3.16),
    "chicJ_1P": (2.41, 4.82),
    "psi_2S": (3.30, 6.59),
}


def _paths_for_form(energy: str, form: str) -> Dict[str, Path]:
    # Base prefixes must end with ".../output" so build_ensemble can append "_tau1/_tau2".
    base = PROJECT / "input/primordial/LHC"
    if energy == "8.16":
        if form == "new":
            return {
                "Pert": base / "pPb8TeV/pPb8TeV_Pert_diffT0/output_8pPb_diffT0_new/output",
                "NPWLC": base / "pPb8TeV/pPb8TeV_NPWLC_diffT0/output_8pPb_diffT0_new/output",
            }
        if form == "old":
            return {
                "Pert": base / "pPb8TeV/pPb8TeV_Pert_diffT0/output_8pPb_diffT0_old/output",
                "NPWLC": base / "pPb8TeV/pPb8TeV_NPWLC_diffT0/output_8pPb_diffT0_old/output",
            }
        if form == "radius":
            return {
                "Pert": base / "pPb8TeV/pPb8TeV_Pert_diffT0/output_8pPb_diffT0_radius/output",
                "NPWLC": base / "pPb8TeV/pPb8TeV_NPWLC_diffT0/output_8pPb_diffT0_radius/output",
            }
    if energy == "5.02":
        if form == "new":
            return {
                "Pert": base / "pPb5TeV/pPb5TeV_Pert_diffT0/output_5pPb_diffT0_new/output",
                "NPWLC": base / "pPb5TeV/pPb5TeV_NPWLC_diffT0/output_5pPb_diffT0_new/output",
            }
        if form == "old":
            return {
                "Pert": base / "pPb5TeV/pPb5TeV_Pert_diffT0/output_5pPb_diffT0_old/output",
                "NPWLC": base / "pPb5TeV/pPb5TeV_NPWLC_diffT0/output_5pPb_diffT0_old/output",
            }
        if form == "radius":
            return {
                "Pert": base / "pPb5TeV/pPb5TeV_Pert_diffT0/output_5pPb_diffT0_radius/output",
                "NPWLC": base / "pPb5TeV/pPb5TeV_NPWLC_diffT0/output_5pPb_diffT0_radius/output",
            }
    raise ValueError(f"Unsupported energy/form: {energy}/{form}")


def _tauform_for_form(form: str) -> Dict[str, Tuple[float, float]]:
    if form == "new":
        return TAUFORM_BINDING
    if form == "old":
        return TAUFORM_OLD
    if form == "radius":
        return TAUFORM_RADIUS
    raise ValueError(form)


def _iter_ywins(y_wins: Dict[str, Tuple[float, float]]) -> Iterable[Tuple[str, float, float]]:
    for key, win in y_wins.items():
        y0, y1 = tuple(win)
        yield key, y0, y1


def _maps_from_runs(runs_dict):
    return next(iter(runs_dict.values())).centrality


def _centrality_percent(maps, bvals):
    bvals = np.asarray(bvals, float)
    c = maps.b_to_c(bvals).astype(float)
    if np.isfinite(c).any():
        cmax = float(np.nanmax(c))
        if cmax <= 1.5:
            c *= 100.0
        elif cmax > 150.0:
            c /= (cmax / 100.0)
    good = np.isfinite(c)
    span = (np.nanmax(c[good]) - np.nanmin(c[good])) if good.any() else 0.0
    if (not good.all()) or (span < 5.0):
        bu = np.unique(bvals)
        n = len(bu)
        centers = (np.arange(n) + 0.5) * (100.0 / max(1, n))
        order = np.argsort(bu)
        c = np.interp(bvals, bu[order], centers[order])
    return c


def _aggregate_class(df_center_b, df_band_b, maps, cent_lo, cent_hi, states, xname, weight="nbin"):
    bs = np.sort(df_center_b["b"].unique())
    cents = _centrality_percent(maps, bs)
    sel_b = bs[(cents >= cent_lo) & (cents < cent_hi)]
    if sel_b.size == 0:
        return None, None
    dc = df_center_b[df_center_b["b"].isin(sel_b)].copy()
    db = df_band_b[df_band_b["b"].isin(sel_b)].copy() if df_band_b is not None else None
    if weight == "nbin":
        wmap = dict(zip(sel_b, maps.b_to_nbin(sel_b)))
    else:
        wmap = {b: 1.0 for b in sel_b}
    for b in list(wmap):
        w = wmap[b]
        if (not np.isfinite(w)) or (w <= 0.0):
            wmap[b] = 1.0
    rows = []
    for xv, chunk in dc.groupby(xname, sort=True):
        ws = np.array([wmap[b] for b in chunk["b"]], float)
        denom = ws.sum()
        ws = ws / denom if np.isfinite(denom) and denom > 0 else np.ones_like(ws) / max(1, len(ws))
        row = {xname: float(xv)}
        for s in states:
            row[s] = float(np.sum(ws * chunk[s].to_numpy(float)))
        rows.append(row)
    dfc = pd.DataFrame(rows).sort_values(xname).reset_index(drop=True)
    dfb = None
    if db is not None:
        rowsb = []
        for xv, ch in db.groupby(xname, sort=True):
            rb = {xname: float(xv)}
            for s in states:
                lo = ch.get(f"{s}_lo", pd.Series(dtype=float))
                hi = ch.get(f"{s}_hi", pd.Series(dtype=float))
                if not lo.empty and not hi.empty:
                    rb[f"{s}_lo"] = float(np.nanmin(lo.to_numpy(float)))
                    rb[f"{s}_hi"] = float(np.nanmax(hi.to_numpy(float)))
            rowsb.append(rb)
        dfb = pd.DataFrame(rowsb).sort_values(xname).reset_index(drop=True)
    return dfc, dfb


def _centrality_slice_weights(maps, df_center_b, cent_classes, scheme="nbin"):
    bs = np.sort(df_center_b["b"].unique())
    cents = _centrality_percent(maps, bs)
    if scheme == "nbin":
        w_b = maps.b_to_nbin(bs)
        w_b = np.where((w_b > 0) & np.isfinite(w_b), w_b, 1.0)
    else:
        w_b = np.ones_like(bs, float)
    w_slices = []
    for lo, hi in cent_classes:
        m = (cents >= lo) & (cents < hi)
        w_slices.append(np.sum(w_b[m]))
    w_slices = np.asarray(w_slices, float)
    if not np.any(w_slices > 0):
        w_slices = np.ones_like(w_slices)
    return w_slices / w_slices.sum()


def step_from_centers(x_cent, vals):
    x_cent = np.asarray(x_cent, float)
    vals = np.asarray(vals, float)
    n = x_cent.size
    if n == 0:
        return np.array([]), np.array([])
    if n == 1:
        x_edges = np.array([x_cent[0] - 0.5, x_cent[0] + 0.5], float)
    else:
        dx = np.diff(x_cent)
        if np.allclose(dx, dx[0]):
            x_edges = np.concatenate(([x_cent[0] - 0.5 * dx[0]], x_cent + 0.5 * dx[0]))
        else:
            x_edges = np.empty(n + 1, float)
            x_edges[1:-1] = 0.5 * (x_cent[:-1] + x_cent[1:])
            x_edges[0] = x_cent[0] - 0.5 * (x_cent[1] - x_cent[0])
            x_edges[-1] = x_cent[-1] + 0.5 * (x_cent[-1] - x_cent[-2])
    return x_edges, np.concatenate([vals, vals[-1:]])


def _sorted_cent_tags(model_block):
    all_tags = set(model_block.keys())
    mb = [t for t in all_tags if "MB" in t]
    non_mb = [t for t in all_tags if "MB" not in t]
    non_mb = sorted(non_mb, key=lambda t: float(t.split("–")[0]))
    return non_mb + mb


def _validate_glauber_maps(groot: Path) -> None:
    required = ["bvscData.tsv", "nbinvsbData.tsv", "npartvsbData.tsv"]
    for nm in required:
        p = groot / nm
        if not p.exists():
            raise FileNotFoundError(f"Missing generated map file: {p}")
        arr = np.loadtxt(p, comments="#")
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise RuntimeError(f"Bad map shape in {p}: {arr.shape}")
        if not np.all(np.isfinite(arr[:, :2])):
            raise RuntimeError(f"Non-finite values in {p}")
    bv = np.loadtxt(groot / "bvscData.tsv", comments="#")
    if np.any(np.diff(bv[:, 0]) < -1e-12):
        raise RuntimeError("b grid is not non-decreasing in bvscData.tsv")


def _load_primordial_combos(
    energy: str,
    form: str,
    glauber_root: Path,
    cfg: ReaderConfig,
) -> List[PrimordialCombo]:
    paths = _paths_for_form(energy, form)
    if form == "new":
        tags = ("lower_binding", "upper_binding")
    elif form == "old":
        tags = ("lower_old", "upper_old")
    elif form == "radius":
        tags = ("lower_radius", "upper_radius")
    else:
        raise ValueError(form)
    combos: List[PrimordialCombo] = []
    for model in MODELS:
        base = paths[model]
        try:
            ens, runs = build_ensemble(
                str(base),
                str(glauber_root),
                tags=tags,
                cfg=cfg,
                sqrts_NN=float(energy),
            )
        except FileNotFoundError as e:
            print(f"[WARN] skipping {model}: {e}")
            continue
        if len(runs) == 0:
            raise RuntimeError(f"No runs loaded for {energy}/{form}/{model}")
        combos.append(PrimordialCombo(energy=energy, model=model, form=form, ens=ens, runs=runs))
    if not combos:
        raise RuntimeError(f"No combos loaded for energy={energy}, form={form}")
    return combos


def primordial_vs_y_all(
    combos,
    states,
    cent_classes,
    pt_window,
    y_bins,
):
    results = {c.model: {} for c in combos}
    for combo in combos:
        maps = _maps_from_runs(combo.runs)
        dfC_b, dfB_b = combo.ens.central_and_band_vs_y_per_b(
            pt_window=pt_window,
            y_bins=y_bins,
            with_feeddown=True,
            use_nbin=True,
            flip_y=True,
        )
        if dfC_b.empty:
            raise RuntimeError(f"Empty y-per-b table for {combo.model}")
        slice_w = _centrality_slice_weights(maps, dfC_b, cent_classes, scheme="nbin")
        cent_tags_for_weights = []
        for ic, (lo, hi) in enumerate(cent_classes):
            tag = f"{int(lo)}–{int(hi)}%"
            dfc, dfb = _aggregate_class(dfC_b, dfB_b, maps, lo, hi, states, "y")
            if dfc is None:
                continue
            y_cent = dfc["y"].to_numpy(float)
            entry = {}
            for s in states:
                Rc = dfc[s].to_numpy(float)
                if dfb is not None and f"{s}_lo" in dfb:
                    Rlo = dfb[f"{s}_lo"].to_numpy(float)
                    Rhi = dfb[f"{s}_hi"].to_numpy(float)
                else:
                    Rlo = Rc
                    Rhi = Rc
                entry[s] = (Rc, Rlo, Rhi, y_cent)
            results[combo.model][tag] = entry
            cent_tags_for_weights.append((tag, ic))
        tag_mb = "0–100% (MB)"
        entry_mb = {}
        for s in states:
            Rc_slices, Rlo_slices, Rhi_slices = [], [], []
            y_slices, w_used = [], []
            for tag, ic in cent_tags_for_weights:
                if s not in results[combo.model][tag]:
                    continue
                Rc, Rlo, Rhi, y_cent = results[combo.model][tag][s]
                Rc_slices.append(Rc)
                Rlo_slices.append(Rlo)
                Rhi_slices.append(Rhi)
                y_slices.append(y_cent)
                w_used.append(slice_w[ic])
            if not Rc_slices:
                continue
            W = np.asarray(w_used, float)
            W /= W.sum()
            y_round_lists = [np.round(y_arr, 6) for y_arr in y_slices]
            common_vals = set(y_round_lists[0])
            for yr in y_round_lists[1:]:
                common_vals &= set(yr)
            if not common_vals:
                continue
            y_common = np.array(sorted(common_vals))
            Rc_stack, Rlo_stack, Rhi_stack = [], [], []
            for Rc, Rlo, Rhi, y_cent in zip(Rc_slices, Rlo_slices, Rhi_slices, y_slices):
                y_round = np.round(y_cent, 6)
                idxs = [np.where(y_round == v)[0][0] for v in y_common]
                Rc_stack.append(Rc[idxs])
                Rlo_stack.append(Rlo[idxs])
                Rhi_stack.append(Rhi[idxs])
            Rc_stack = np.stack(Rc_stack, axis=0)
            Rlo_stack = np.stack(Rlo_stack, axis=0)
            Rhi_stack = np.stack(Rhi_stack, axis=0)
            Rc_MB = np.tensordot(W, Rc_stack, axes=(0, 0))
            Rlo_MB = np.tensordot(W, Rlo_stack, axes=(0, 0))
            Rhi_MB = np.tensordot(W, Rhi_stack, axes=(0, 0))
            entry_mb[s] = (Rc_MB, Rlo_MB, Rhi_MB, y_common)
        if entry_mb:
            results[combo.model][tag_mb] = entry_mb
    return results


def primordial_vs_pT_all(
    combos,
    states,
    cent_classes,
    y_wins,
    pt_bins,
):
    out = {}
    for yname, y0, y1 in _iter_ywins(y_wins):
        out[yname] = {c.model: {} for c in combos}
        for combo in combos:
            maps = _maps_from_runs(combo.runs)
            dfC_b, dfB_b = combo.ens.central_and_band_vs_pt_per_b(
                y_window=(y0, y1, yname),
                pt_bins=pt_bins,
                with_feeddown=True,
                use_nbin=True,
            )
            if dfC_b.empty:
                raise RuntimeError(f"Empty pT-per-b table for {combo.model}/{yname}")
            slice_w = _centrality_slice_weights(maps, dfC_b, cent_classes, scheme="nbin")
            cent_tags_for_weights = []
            for ic, (lo, hi) in enumerate(cent_classes):
                tag = f"{int(lo)}–{int(hi)}%"
                dfc, dfb = _aggregate_class(dfC_b, dfB_b, maps, lo, hi, states, "pt")
                if dfc is None:
                    continue
                pT_cent = dfc["pt"].to_numpy(float)
                entry = {}
                for s in states:
                    Rc = dfc[s].to_numpy(float)
                    if dfb is not None and f"{s}_lo" in dfb:
                        Rlo = dfb[f"{s}_lo"].to_numpy(float)
                        Rhi = dfb[f"{s}_hi"].to_numpy(float)
                    else:
                        Rlo = Rc
                        Rhi = Rc
                    entry[s] = (Rc, Rlo, Rhi, pT_cent)
                out[yname][combo.model][tag] = entry
                cent_tags_for_weights.append((tag, ic))
            tag_mb = "0–100% (MB)"
            entry_mb = {}
            for s in states:
                pT_ref = None
                Rc_slices, Rlo_slices, Rhi_slices = [], [], []
                w_used = []
                for tag, ic in cent_tags_for_weights:
                    if s not in out[yname][combo.model].get(tag, {}):
                        continue
                    Rc, Rlo, Rhi, pT_cent = out[yname][combo.model][tag][s]
                    if pT_ref is None:
                        pT_ref = pT_cent
                    Rc_slices.append(Rc)
                    Rlo_slices.append(Rlo)
                    Rhi_slices.append(Rhi)
                    w_used.append(slice_w[ic])
                if not Rc_slices:
                    continue
                W = np.asarray(w_used, float)
                W /= W.sum()
                min_len = min(len(arr) for arr in Rc_slices)
                pT_ref_common = pT_ref[:min_len]
                Rc_stack = np.stack([arr[:min_len] for arr in Rc_slices], axis=0)
                Rlo_stack = np.stack([arr[:min_len] for arr in Rlo_slices], axis=0)
                Rhi_stack = np.stack([arr[:min_len] for arr in Rhi_slices], axis=0)
                Rc_MB = np.tensordot(W, Rc_stack, axes=(0, 0))
                Rlo_MB = np.tensordot(W, Rlo_stack, axes=(0, 0))
                Rhi_MB = np.tensordot(W, Rhi_stack, axes=(0, 0))
                entry_mb[s] = (Rc_MB, Rlo_MB, Rhi_MB, pT_ref_common)
            if entry_mb:
                out[yname][combo.model][tag_mb] = entry_mb
    return out


def primordial_vs_cent_from_vs_y(prim_y_dict, y_wins, states, models):
    out = {}
    for yname, y0, y1 in _iter_ywins(y_wins):
        out[yname] = {}
        for model in models:
            if model not in prim_y_dict:
                continue
            cent_tags = [c for c in prim_y_dict[model].keys() if "MB" not in c]
            mb_tag = [c for c in prim_y_dict[model].keys() if "MB" in c]
            mb_tag = mb_tag[0] if mb_tag else None
            cent_tags = sorted(cent_tags, key=lambda t: float(t.split("–")[0]))
            for s in states:
                cent_mid, Rc_all, Rlo_all, Rhi_all = [], [], [], []
                for ct in cent_tags:
                    if s not in prim_y_dict[model][ct]:
                        continue
                    Rc, Rlo, Rhi, y_cent = prim_y_dict[model][ct][s]
                    mask = (y_cent >= y0) & (y_cent <= y1)
                    if not np.any(mask):
                        continue
                    lo_s, hi_s = ct.split("–")
                    lo = float(lo_s)
                    hi = float(hi_s.strip("%"))
                    cent_mid.append(0.5 * (lo + hi))
                    Rc_all.append(float(np.mean(Rc[mask])))
                    Rlo_all.append(float(np.min(Rlo[mask])))
                    Rhi_all.append(float(np.max(Rhi[mask])))
                if not cent_mid:
                    continue
                Rc_MB = Rlo_MB = Rhi_MB = np.nan
                if mb_tag and s in prim_y_dict[model][mb_tag]:
                    Rc_mb, Rlo_mb, Rhi_mb, y_mb = prim_y_dict[model][mb_tag][s]
                    m = (y_mb >= y0) & (y_mb <= y1)
                    if np.any(m):
                        Rc_MB = float(np.mean(Rc_mb[m]))
                        Rlo_MB = float(np.min(Rlo_mb[m]))
                        Rhi_MB = float(np.max(Rhi_mb[m]))
                out[yname].setdefault(model, {})[s] = (
                    np.array(cent_mid, float),
                    np.array(Rc_all, float),
                    np.array(Rlo_all, float),
                    np.array(Rhi_all, float),
                    Rc_MB,
                    Rlo_MB,
                    Rhi_MB,
                )
    return out


def _save_vs_y_plots_and_csv(
    prim_y,
    outdir: Path,
    energy: str,
    form: str,
    tauform,
    save_pdf: bool,
    save_csv: bool,
    y_lim: Tuple[float, float] = (0.0, 1.0),
):
    alpha_band = 0.22
    dpi = 150
    for state in STATES:
        cent_tags = _sorted_cent_tags(next(iter(prim_y.values())))
        n_cent = len(cent_tags)
        n_cols = 3
        n_rows = int(np.ceil(n_cent / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.0 * n_rows), dpi=dpi, sharex=False, sharey=False)
        axes = np.atleast_1d(axes).ravel()
        for ip, cent_tag in enumerate(cent_tags):
            ax = axes[ip]
            for model in MODELS:
                if model not in prim_y or cent_tag not in prim_y[model] or state not in prim_y[model][cent_tag]:
                    continue
                Rc, Rlo, Rhi, y_cent = prim_y[model][cent_tag][state]
                x_edges, y_c = step_from_centers(y_cent, Rc)
                _, y_lo = step_from_centers(y_cent, Rlo)
                _, y_hi = step_from_centers(y_cent, Rhi)
                lab = model if ip == 0 else None
                col = MODEL_COLORS[model]
                ls = MODEL_LS[model]
                ax.step(x_edges, y_c, where="post", color=col, ls=ls, lw=1.8, label=lab)
                ax.fill_between(x_edges, y_lo, y_hi, step="post", color=col, alpha=alpha_band, linewidth=0.0)
            ax.minorticks_on()
            ax.tick_params(axis="y", which="major", direction="in", right=True)
            ax.tick_params(axis="y", which="minor", direction="in", right=True)
            ax.set_xlim(-5, 5)
            ax.set_ylim(*y_lim)
            ax.set_xlabel(r"$y$")
            if ip % n_cols == 0:
                ax.set_ylabel(r"$R_{pA}$")
            ax.grid(False)
            ax.text(0.02, 0.96, cent_tag, transform=ax.transAxes, ha="left", va="top", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        for k in range(n_cent, len(axes)):
            axes[k].set_visible(False)
        h, l = axes[0].get_legend_handles_labels()
        if h:
            axes[0].legend(h, l, loc="lower right", fontsize=9, frameon=False)
        tau_lo, tau_hi = tauform[state]
        axes[0].text(0.50, 0.93, rf"{STATE_LABELS[state]}, $\sqrt{{s_{{NN}}}}={float(energy):.2f}$ TeV",
                     transform=axes[0].transAxes, ha="center", va="top", fontsize=10)
        if len(axes) > 1:
            axes[1].text(0.50, 0.93, rf"$\tau_{{\rm form}}={tau_lo:.2g}\text{{–}}{tau_hi:.2g}$ fm",
                         transform=axes[1].transAxes, ha="center", va="top", fontsize=10)
        fig.tight_layout()
        tag = f"{energy.replace('.', 'p')}TeV_{form}"
        if save_pdf:
            fig.savefig(outdir / f"primordial_RpA_vs_y_{state}_{tag}.pdf", bbox_inches="tight")
        plt.close(fig)
        if save_csv:
            rows = []
            for model in MODELS:
                if model not in prim_y:
                    continue
                for cent_tag in prim_y[model]:
                    if state not in prim_y[model][cent_tag]:
                        continue
                    Rc, Rlo, Rhi, y_cent = prim_y[model][cent_tag][state]
                    for yv, rc, rlo, rhi in zip(y_cent, Rc, Rlo, Rhi):
                        rows.append({
                            "energy": energy, "form": form, "model": model, "centrality": cent_tag, "state": state,
                            "y": yv, "R": rc, "R_lo": rlo, "R_hi": rhi
                        })
            if rows:
                pd.DataFrame(rows).to_csv(outdir / f"primordial_RpA_vs_y_{state}_{tag}.csv", index=False)


def _save_vs_pt_plots_and_csv(
    prim_pt,
    outdir: Path,
    energy: str,
    form: str,
    tauform,
    pt_window,
    save_pdf: bool,
    save_csv: bool,
    y_lim: Tuple[float, float] = (0.0, 1.0),
):
    alpha_band = 0.22
    dpi = 150
    for state in STATES:
        for yname, y0, y1 in _iter_ywins(Y_WINDOWS):
            if yname not in prim_pt:
                continue
            models_block = prim_pt[yname]
            if not models_block:
                continue
            cent_tags = _sorted_cent_tags(next(iter(models_block.values())))
            n_cent = len(cent_tags)
            n_cols = 3
            n_rows = int(np.ceil(n_cent / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.0 * n_rows), dpi=dpi, sharex=False, sharey=False)
            axes = np.atleast_1d(axes).ravel()
            for ip, cent_tag in enumerate(cent_tags):
                ax = axes[ip]
                for model in MODELS:
                    if model not in models_block or cent_tag not in models_block[model] or state not in models_block[model][cent_tag]:
                        continue
                    Rc, Rlo, Rhi, pT_cent = models_block[model][cent_tag][state]
                    x_edges, y_c = step_from_centers(pT_cent, Rc)
                    _, y_lo = step_from_centers(pT_cent, Rlo)
                    _, y_hi = step_from_centers(pT_cent, Rhi)
                    lab = model if ip == 0 else None
                    col = MODEL_COLORS[model]
                    ls = MODEL_LS[model]
                    ax.step(x_edges, y_c, where="post", color=col, ls=ls, lw=1.8, label=lab)
                    ax.fill_between(x_edges, y_lo, y_hi, step="post", color=col, alpha=alpha_band, linewidth=0.0)
                ax.set_xlim(pt_window[0], pt_window[1])
                ax.set_ylim(*y_lim)
                ax.set_xlabel(r"$p_T$ [GeV]")
                if ip % n_cols == 0:
                    ax.set_ylabel(r"$R_{pA}$")
                ax.grid(False)
                ax.minorticks_on()
                ax.tick_params(axis="y", which="major", direction="in", right=True)
                ax.tick_params(axis="y", which="minor", direction="in", right=True)
                ax.text(0.02, 0.96, cent_tag, transform=ax.transAxes, ha="left", va="top", fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
            for k in range(n_cent, len(axes)):
                axes[k].set_visible(False)
            h, l = axes[0].get_legend_handles_labels()
            if h:
                axes[0].legend(h, l, loc="lower right", fontsize=9, frameon=False)
            tau_lo, tau_hi = tauform[state]
            axes[0].text(0.50, 0.96, rf"{STATE_LABELS[state]}, $\sqrt{{s_{{NN}}}}={float(energy):.2f}$ TeV",
                         transform=axes[0].transAxes, ha="center", va="top", fontsize=10)
            if len(axes) > 1:
                axes[1].text(0.50, 0.96, rf"$\tau_{{\rm form}}={tau_lo:.2g}\text{{–}}{tau_hi:.2g}$ fm",
                             transform=axes[1].transAxes, ha="center", va="top", fontsize=10)
            if len(axes) > 2:
                axes[2].text(0.50, 0.96, rf"${y0:.2f} < y < {y1:.2f}$",
                             transform=axes[2].transAxes, ha="center", va="top", fontsize=10)
            fig.tight_layout()
            tag = f"{energy.replace('.', 'p')}TeV_{form}"
            if save_pdf:
                fig.savefig(outdir / f"primordial_RpA_vs_pT_{state}_{yname}_{tag}.pdf", bbox_inches="tight")
            plt.close(fig)
            if save_csv:
                rows = []
                for model in MODELS:
                    if model not in models_block:
                        continue
                    for cent_tag in models_block[model]:
                        if state not in models_block[model][cent_tag]:
                            continue
                        Rc, Rlo, Rhi, pT_cent = models_block[model][cent_tag][state]
                        for pv, rc, rlo, rhi in zip(pT_cent, Rc, Rlo, Rhi):
                            rows.append({
                                "energy": energy, "form": form, "y_window": yname, "model": model,
                                "centrality": cent_tag, "state": state, "pT": pv, "R": rc, "R_lo": rlo, "R_hi": rhi
                            })
                if rows:
                    pd.DataFrame(rows).to_csv(outdir / f"primordial_RpA_vs_pT_{state}_{yname}_{tag}.csv", index=False)


def _save_vs_cent_plots(
    prim_cent,
    outdir: Path,
    energy: str,
    form: str,
    tauform,
    cent_classes,
    save_pdf: bool,
    y_lim: Tuple[float, float] = (0.0, 1.0),
):
    alpha_band = 0.22
    dpi = 150
    for state in STATES:
        ywins_list = list(_iter_ywins(Y_WINDOWS))
        n_pan = len(ywins_list)
        fig, axes = plt.subplots(1, n_pan, figsize=(5.0 * n_pan, 3.6), dpi=dpi, sharey=False)
        axes = np.atleast_1d(axes)
        for iax, (yname, y0, y1) in enumerate(ywins_list):
            ax = axes[iax]
            if yname not in prim_cent:
                ax.set_visible(False)
                continue
            for model in MODELS:
                if model not in prim_cent[yname] or state not in prim_cent[yname][model]:
                    continue
                cent_mid, Rc, Rlo, Rhi, Rc_MB, Rlo_MB, Rhi_MB = prim_cent[yname][model][state]
                edges = np.array([cent_classes[0][0]] + [hi for (_, hi) in cent_classes], float)
                y_c = np.concatenate([Rc, Rc[-1:]])
                y_lo = np.concatenate([Rlo, Rlo[-1:]])
                y_hi = np.concatenate([Rhi, Rhi[-1:]])
                col = MODEL_COLORS[model]
                ls = MODEL_LS[model]
                lab = model if iax == 0 else None
                ax.step(edges, y_c, where="post", color=col, ls=ls, lw=1.8, label=lab)
                ax.fill_between(edges, y_lo, y_hi, step="post", color=col, alpha=alpha_band, linewidth=0.0)
                if np.isfinite(Rc_MB):
                    x_mb = np.array([0.0, 100.0])
                    ax.fill_between(x_mb, [Rlo_MB, Rlo_MB], [Rhi_MB, Rhi_MB], color="none", hatch="////",
                                    edgecolor=col, linewidth=0.0)
                    ax.hlines(Rc_MB, 0.0, 100.0, colors=col, linestyles="--", linewidth=1.5)
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(*y_lim)
            ax.set_xlabel("Centrality [%]")
            if iax == 0:
                ax.set_ylabel(r"$R_{pA}$")
            ax.grid(False)
            ax.minorticks_on()
            ax.tick_params(axis="y", which="major", direction="in", right=True)
            ax.tick_params(axis="y", which="minor", direction="in", right=True)
            ax.text(0.02, 0.92, f"${y0:.2f} < y < {y1:.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        h, l = axes[0].get_legend_handles_labels()
        if h:
            axes[0].legend(h, l, loc="lower right", frameon=False, fontsize=9)
        tau_lo, tau_hi = tauform[state]
        axes[0].text(0.50, 0.95, rf"{STATE_LABELS[state]}, $\sqrt{{s_{{NN}}}}={float(energy):.2f}$ TeV",
                     transform=axes[0].transAxes, ha="center", va="top", fontsize=10)
        if len(axes) > 1 and axes[1].get_visible():
            axes[1].text(0.50, 0.95, rf"$\tau_{{\rm form}}={tau_lo:.2g}\text{{–}}{tau_hi:.2g}$ fm",
                         transform=axes[1].transAxes, ha="center", va="top", fontsize=10)
        fig.tight_layout()
        tag = f"{energy.replace('.', 'p')}TeV_{form}"
        if save_pdf:
            fig.savefig(outdir / f"primordial_RpA_vs_cent_{state}_{tag}.pdf", bbox_inches="tight")
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy", choices=["8.16", "5.02"], default="8.16")
    p.add_argument("--form", choices=["new", "old", "radius"], default="new")
    p.add_argument("--save-pdf", action="store_true")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    energy = args.energy
    form = args.form
    sqrts_gev = 8160.0 if energy == "8.16" else 5020.0

    outdir = PROJECT / "primordial_output"
    outdir.mkdir(exist_ok=True, parents=True)

    Style.apply()
    mpl.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "axes.unicode_minus": False,
    })

    y_bins = make_bins_from_width(-5.0, 5.0, 0.5)
    pt_bins = make_bins_from_width(0.0, 20.0, 2.5)
    pt_window = (0.0, 15.0)
    cent_classes = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

    print("[step 1] generating primordial-compatible maps from eloss optical glauber...")
    glauber_generated = PROJECT / "primordial_output" / "glauber_data" / ("8TeV_eloss" if energy == "8.16" else "5TeV_eloss")
    generate_primordial_glauber_maps(
        glauber_generated,
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
    _validate_glauber_maps(glauber_generated)
    print(f"[ok] maps in: {glauber_generated}")

    print("[step 2] loading primordial ensembles (tau1/tau2) with new Glauber maps...")
    cfg = ReaderConfig(debug=args.debug)
    combos = _load_primordial_combos(energy, form, glauber_generated, cfg)
    print("[ok] combos:", ", ".join([f"{c.model}:{c.form}" for c in combos]))

    # Step checks on maps in runs
    print("[step 3] validating centrality mapping availability in loaded runs...")
    for c in combos:
        maps = _maps_from_runs(c.runs)
        probe = maps.b_to_nbin(np.array([0.0, 2.0, 4.0], float))
        if not np.all(np.isfinite(probe)):
            raise RuntimeError(f"Non-finite nbin probe for {c.model}: {probe}")
        print(f"[ok] {c.model} nbin(b=0,2,4)={probe[0]:.3f},{probe[1]:.3f},{probe[2]:.3f}")

    print("[step 4] computing R_pA vs y (all centralities + MB)...")
    prim_y = primordial_vs_y_all(combos, STATES, cent_classes, pt_window, y_bins)
    for m in prim_y:
        if not prim_y[m]:
            raise RuntimeError(f"No vs-y results for model={m}")
    print("[ok] vs-y computed")

    print("[step 5] computing R_pA vs pT (all centralities + MB)...")
    prim_pt = primordial_vs_pT_all(combos, STATES, cent_classes, Y_WINDOWS, pt_bins)
    if not prim_pt:
        raise RuntimeError("No vs-pT results")
    print("[ok] vs-pT computed")

    print("[step 6] computing R_pA vs centrality from vs-y...")
    prim_cent = primordial_vs_cent_from_vs_y(prim_y, Y_WINDOWS, STATES, MODELS)
    if not prim_cent:
        raise RuntimeError("No vs-centrality results")
    print("[ok] vs-centrality computed")

    print("[step 7] saving outputs (plots/csv according to flags)...")
    tauform = _tauform_for_form(form)
    _save_vs_y_plots_and_csv(prim_y, outdir, energy, form, tauform, save_pdf=args.save_pdf, save_csv=args.save_csv)
    _save_vs_pt_plots_and_csv(prim_pt, outdir, energy, form, tauform, pt_window, save_pdf=args.save_pdf, save_csv=args.save_csv)
    _save_vs_cent_plots(prim_cent, outdir, energy, form, tauform, cent_classes, save_pdf=args.save_pdf)
    print(f"[done] outputs in: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
