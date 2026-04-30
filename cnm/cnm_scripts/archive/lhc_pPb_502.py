import sys, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import time

# Ensure we can find the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnm_combine.cnm_combine import CNMCombine, cnm_vs_y_to_dataframe, cnm_vs_pT_to_dataframe, cnm_vs_cent_to_dataframe
from eloss_code.eloss_cronin_centrality import (
    plot_RpA_vs_y_components_per_centrality,
    plot_RpA_vs_pT_components_per_centrality,
    plot_RpA_vs_centrality_components_band
)

def run_lhc_production():
    start_time = time.time()
    energy = "5.02"
    outdir = Path(f"../outputs/final/LHC_{energy}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting LHC p+Pb {energy} TeV Production ===")
    print(f"Output directory: {outdir}")

    # 1. Initialize
    # nb_bsamples=20 is a good balance.
    print("Initializing CNMCombine...")
    cnm = CNMCombine.from_defaults(energy, nb_bsamples=20) 
    print(f"System: {cnm.spec.system}, Roots: {cnm.roots} GeV")

    # 2. RpA vs y
    # Range -5.0 to 5.0 for LHC
    y_edges = np.linspace(-5.0, 5.0, 51) 
    pt_range = (0.0, 10.0) # pT integration range for y-dependence (user requested 0-10 for LHC)

    print(f"\n[1/3] Computing RpA vs y (pT integrated {pt_range} GeV)...")
    components = ("npdf", "eloss", "broad", "eloss_broad", "cnm")
    
    y_cent, tags, data_y = cnm.cnm_vs_y(
        y_edges=y_edges,
        pt_range_avg=pt_range,
        components=components,
        include_mb=True
    )

    # Save CSVs
    for comp in components:
        df = cnm_vs_y_to_dataframe(y_cent, tags, data_y, comp)
        df["component"] = comp
        df.to_csv(outdir / f"rpa_vs_y_{comp}.csv", index=False)
    
    # Plotting
    print("  Generating plots for RpA vs y...")
    extra_bands = {k: data_y[k] for k in ["npdf", "cnm", "eloss", "broad", "eloss_broad"]}
    
    fig, _ = plot_RpA_vs_y_components_per_centrality(
        cnm.particle, cnm.roots, cnm.qp, cnm.gl,
        cnm.cent_bins,
        y_edges, pt_range,
        show_components=("loss", "broad", "total", "npdf", "cnm"),
        extra_bands=extra_bands,
        suptitle=f"$J/\\psi$ p+Pb {energy} TeV: $R_{{pPb}}(y)$",
        ncols=3
    )
    plt.savefig(outdir / "plot_rpa_vs_y_all_components.pdf", bbox_inches="tight")
    plt.savefig(outdir / "plot_rpa_vs_y_all_components.png", dpi=150)
    plt.close(fig)


    # 3. RpA vs pT
    # LHC windows from defaults: Backward (-4.46, -2.96), Mid (-1.37, 0.43), Forward (2.03, 3.53)
    p_edges = np.linspace(0.0, 15.0, 31) # 0 to 15 GeV as requested
    # Use default windows for LHC
    windows = cnm.y_windows 

    print(f"\n[2/3] Computing RpA vs pT in {len(windows)} rapidity windows...")
    
    for y0, y1, name in windows:
        print(f"  > Window: {name} ({y0} < y < {y1})")
        pt_cent, tags_pt, data_pt_res = cnm.cnm_vs_pT(
            y_window=(y0, y1, name),
            pt_edges=p_edges,
            components=components,
            include_mb=True
        )

        safe_name = name.replace(" ", "_")
        
        # Save CSVs
        for comp in components:
            df = cnm_vs_pT_to_dataframe(pt_cent, tags_pt, data_pt_res, comp)
            df["component"] = comp
            df.to_csv(outdir / f"rpa_vs_pT_{safe_name}_{comp}.csv", index=False)

        # Plot
        extra_bands_pt = {k: data_pt_res[k] for k in ["npdf", "cnm", "eloss", "broad", "eloss_broad"]}
        fig, _ = plot_RpA_vs_pT_components_per_centrality(
            cnm.particle, cnm.roots, cnm.qp, cnm.gl,
            cnm.cent_bins,
            p_edges, (y0, y1),
            show_components=("loss", "broad", "total", "npdf", "cnm"),
            extra_bands=extra_bands_pt,
            suptitle=f"$J/\\psi$ p+Pb {energy} TeV: $R_{{pPb}}(p_T)$ - {name}",
            ncols=3 
        )
        plt.savefig(outdir / f"plot_rpa_vs_pT_{safe_name}.pdf", bbox_inches="tight")
        plt.savefig(outdir / f"plot_rpa_vs_pT_{safe_name}.png", dpi=150)
        plt.close(fig)


    # 4. RpA vs Centrality
    # Integrate over y-window and pT range (0-10 GeV for LHC)
    print(f"\n[3/3] Computing RpA vs Centrality (pT: 0-10 GeV)...")
    pt_range_cent = (0.0, 10.0)
    cnm.pt_range_avg = pt_range_cent 
    
    for y0, y1, name in windows:
        print(f"  > Window: {name} ({y0} < y < {y1})")
        
        data_cent = cnm.cnm_vs_centrality(
            y_window=(y0, y1, name),
            components=components
        )
        
        safe_name = name.replace(" ", "_")

        # Save CSVs
        for comp in components:
            df = cnm_vs_cent_to_dataframe(cnm.cent_bins, data_cent, comp)
            df["component"] = comp
            df.to_csv(outdir / f"rpa_vs_cent_{safe_name}_{comp}.csv", index=False)

        # Plot
        def _unpack(comp_key):
            rc, rlo, rhi, mb_c, mb_lo, mb_hi = data_cent[comp_key]
            labels = [f"{int(a)}-{int(b)}%" for (a, b) in cnm.cent_bins]
            d_c  = {l: v for l, v in zip(labels, rc)}
            d_lo = {l: v for l, v in zip(labels, rlo)}
            d_hi = {l: v for l, v in zip(labels, rhi)}
            d_mb = (mb_c, mb_lo, mb_hi)
            return d_c, d_lo, d_hi, d_mb

        RL_c, RL_lo, RL_hi, RL_mb = _unpack("eloss")
        RB_c, RB_lo, RB_hi, RB_mb = _unpack("broad")
        RT_c, RT_lo, RT_hi, RT_mb = _unpack("eloss_broad")
        RNP_c, RNP_lo, RNP_hi, RNP_mb = _unpack("npdf")
        RCNM_c, RCNM_lo, RCNM_hi, RCNM_mb = _unpack("cnm")

        fig, _ = plot_RpA_vs_centrality_components_band(
            cnm.cent_bins, [f"{int(a)}-{int(b)}%" for a,b in cnm.cent_bins],
            RL_c, RL_lo, RL_hi, RL_mb,
            RB_c, RB_lo, RB_hi, RB_mb,
            RT_c, RT_lo, RT_hi, RT_mb,
            RNP_c, RNP_lo, RNP_hi, RNP_mb,
            RCNM_c, RCNM_lo, RCNM_hi, RCNM_mb,
            show=("loss", "broad", "total", "npdf", "cnm"),
            system_label=f"p+Pb {energy} TeV",
            note=f"{name}\n${y0} < y < {y1}$",
            ylabel=r"$R_{pPb}$"
        )
        plt.savefig(outdir / f"plot_rpa_vs_cent_{safe_name}.pdf", bbox_inches="tight")
        plt.savefig(outdir / f"plot_rpa_vs_cent_{safe_name}.png", dpi=150)
        plt.close(fig)

    elapsed = time.time() - start_time
    print(f"\n=== Done. Total time: {elapsed:.1f} s ===")

if __name__ == "__main__":
    run_lhc_production()
