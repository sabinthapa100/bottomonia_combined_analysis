import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# 1. Setup Paths & Imports
# ------------------------------------------------------------------
print(f"Executing with python: {sys.executable}")
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "npdf_code"))
sys.path.append(str(ROOT / "eloss_code"))
sys.path.append(str(ROOT))  # To find cnm_combine if needed, or modules

# Import nPDF modules
from npdf_data import NPDFSystem, RpAAnalysis
from gluon_ratio import EPPS21Ratio, GluonEPPSProvider
from glauber import OpticalGlauber, SystemSpec
from npdf_centrality import (
    compute_df49_by_centrality,
    make_centrality_weight_dict,
    bin_rpa_vs_y,
    bin_rpa_vs_pT,
    bin_rpa_vs_centrality,
)

# Import Eloss modules
from particle import Particle
from coupling import alpha_s_provider
import quenching_fast as QF
from eloss_cronin_centrality import (
    plot_RpA_vs_y_components_per_centrality,
    plot_RpA_vs_pT_components_per_centrality,
    plot_RpA_vs_centrality_components_band,
    rpa_band_vs_y,
    rpa_band_vs_pT,
    rpa_band_vs_centrality
)
from cnm_combine.cnm_combine import combine_two_bands_1d

print("=== Step-by-Step RHIC Production Script ===")
print(f"Root dir: {ROOT}")

# ------------------------------------------------------------------
# 2. Configuration (RHIC d+Au 200 GeV)
# ------------------------------------------------------------------
ENERGY = "200"
SQRTS = 200.0
SIGMA_NN = 42.0  # mb
SYSTEM = "dA"    # d+Au

# Centrality
CENT_BINS = [(0, 20), (20, 40), (40, 60), (60, 100)]

# Binning
Y_EDGES = np.linspace(-3.0, 3.0, 31)
P_EDGES = np.linspace(0.0, 10.0, 21)
Y_WINDOWS = [
    (-2.2, -1.2, "Backward"),
    (-0.35, 0.35, "Mid-Rapidity"),
    (1.2, 2.2, "Forward")
]
PT_RANGE_AVG = (0.0, 5.0)

# nPDF Params
NB_BSAMPLES = 20  
Y_SHIFT_FRAC = 1.0

# Eloss Params
Q0_PAIR = (0.05, 0.09) 
P0_SCALE = (0.9, 1.1)
QHAT0_CENTRAL = 0.075

# Output
OUTPUT_DIR = ROOT / "outputs" / "step_by_step_rhic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 3. Initialize Shared Objects
# ------------------------------------------------------------------
print("\n--- Initializing Objects ---")

particle = Particle(family="charmonia", state="avg")
spec = SystemSpec(SYSTEM, SQRTS, A=197, sigma_nn_mb=SIGMA_NN)
gl = OpticalGlauber(spec)
L_mb = gl.leff_minbias_pA()
print(f"System: {SYSTEM} {SQRTS} GeV, L_eff_mb={L_mb:.2f} fm")

print("Loading nPDF data...")
npdf_dir = ROOT / "input" / "npdf" / "dAu200GeV"
sys_npdf = NPDFSystem.from_folder(str(npdf_dir), kick="pp", name="dAu 200 GeV")
ana = RpAAnalysis()
base, r0, M = ana.compute_rpa_members(
    sys_npdf.df_pp, sys_npdf.df_pa, sys_npdf.df_errors,
    join="intersect", lowpt_policy="drop", pt_shift_min=1.0
)

epps_path = ROOT / "input" / "npdf" / "nPDFs"
epps_ratio = EPPS21Ratio(A=197, path=str(epps_path))
gluon = GluonEPPSProvider(epps_ratio, sqrt_sNN_GeV=SQRTS, m_state_GeV="charmonium", y_sign_for_xA=-1)

print("Computing nPDF centrality dependence...")
df49_ctx, _, _, _ = compute_df49_by_centrality(
    base, r0, M, gluon, gl, CENT_BINS, 
    nb_bsamples=NB_BSAMPLES, y_shift_fraction=Y_SHIFT_FRAC
)

qp_base = QF.QuenchParams(
    qhat0=QHAT0_CENTRAL,
    lp_fm=1.5,
    LA_fm=L_mb, LB_fm=L_mb,
    roots_GeV=SQRTS,
    alpha_of_mu=alpha_s_provider(mode="constant", alpha0=0.5),
    system="pA"
)

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def save_band_csv(centers, bands, labels, filename, x_name="x"):
    rows = []
    for tag in labels:
        if tag not in bands: continue
        rc, rlo, rhi = bands[tag]
        for i, x in enumerate(centers):
            rows.append({
                x_name: x,
                "centrality": tag,
                "R_central": rc[i],
                "R_lo": rlo[i],
                "R_hi": rhi[i]
            })
    try:
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Failed to save {filename}: {e}")

def rename_eloss_keys(bands):
    new_bands = {}
    for k, v in bands.items():
        if k == "loss": new_bands["eloss"] = v
        elif k == "total": new_bands["eloss_broad"] = v
        else: new_bands[k] = v
    if "loss" in bands: new_bands["loss"] = bands["loss"] 
    return new_bands

def repack_npdf(res_dict):
    dc, dlo, dhi = {}, {}, {}
    for tag, v in res_dict.items():
        dc[tag] = v['r_central']
        dlo[tag] = v['r_lo']
        dhi[tag] = v['r_hi']
    return (dc, dlo, dhi)

# ------------------------------------------------------------------
# 4. Step-by-Step: RpA vs y
# ------------------------------------------------------------------
print("\n--- Step 1: RpA vs Rapidity ---")

# 1. nPDF
print("  > Computing nPDF...")
wcent = make_centrality_weight_dict(CENT_BINS, c0=0.25)
npdf_res = bin_rpa_vs_y(
    df49_ctx, sys_npdf.df_pp, sys_npdf.df_pa, gluon,
    cent_bins=CENT_BINS, y_edges=Y_EDGES, pt_range_avg=PT_RANGE_AVG,
    wcent_dict=wcent, include_mb=True
)
npdf_band_y = repack_npdf(npdf_res)

# 2. Eloss/Broad
print("  > Computing Eloss & Broadening...")
y_cent, eloss_bands_y, labels_y = rpa_band_vs_y(
    particle, SQRTS, qp_base, gl, CENT_BINS,
    Y_EDGES, PT_RANGE_AVG,
    components=("loss", "broad", "total"),
    q0_pair=Q0_PAIR, p0_scale_pair=P0_SCALE,
    mb_weight_mode="exp",
    Ny_bin=3, Npt_bin=4
)

save_band_csv(y_cent, eloss_bands_y["loss"], labels_y, OUTPUT_DIR / "data_rhic_y_eloss.csv", "y")
save_band_csv(y_cent, eloss_bands_y["total"], labels_y, OUTPUT_DIR / "data_rhic_y_total.csv", "y")

# 4. Plot
all_bands_y = rename_eloss_keys(eloss_bands_y)
all_bands_y["npdf"] = npdf_band_y

print("  > Plotting...")
fig_y, _ = plot_RpA_vs_y_components_per_centrality(
    particle, SQRTS, qp_base, gl, CENT_BINS,
    Y_EDGES, PT_RANGE_AVG,
    show_components=("loss", "broad", "total", "npdf"),
    extra_bands=all_bands_y,
    suptitle=f"$J/\\psi$ d+Au {ENERGY} GeV"
)
fig_y.savefig(OUTPUT_DIR / "plot_rhic_rpa_vs_y.png", dpi=150)
plt.close(fig_y)


# ------------------------------------------------------------------
# 5. Step-by-Step: RpA vs pT
# ------------------------------------------------------------------
print("\n--- Step 2: RpA vs pT ---")

for y0, y1, name in Y_WINDOWS:
    print(f"  > Window: {name}")
    sname = name.replace(" ", "_")
    
    # 1. nPDF
    npdf_res_pt = bin_rpa_vs_pT(
        df49_ctx, sys_npdf.df_pp, sys_npdf.df_pa, gluon,
        cent_bins=CENT_BINS, pt_edges=P_EDGES, y_window=(y0, y1),
        wcent_dict=wcent, include_mb=True
    )
    npdf_band_pt = repack_npdf(npdf_res_pt)
    
    # 2. Eloss/Broad
    pt_cent, eloss_bands_pt, labels_pt = rpa_band_vs_pT(
        particle, SQRTS, qp_base, gl, CENT_BINS,
        P_EDGES, (y0, y1),
        components=("loss", "broad", "total"),
        q0_pair=Q0_PAIR, p0_scale_pair=P0_SCALE,
        mb_weight_mode="exp",
        Ny_bin=3, Npt_bin=4
    )
    
    save_band_csv(pt_cent, eloss_bands_pt["loss"], labels_pt, OUTPUT_DIR / f"data_rhic_pT_{sname}_eloss.csv", "pT")
    save_band_csv(pt_cent, eloss_bands_pt["total"], labels_pt, OUTPUT_DIR / f"data_rhic_pT_{sname}_total.csv", "pT")

    # 4. Plot
    all_bands_pt = rename_eloss_keys(eloss_bands_pt)
    all_bands_pt["npdf"] = npdf_band_pt
    
    fig_pt, _ = plot_RpA_vs_pT_components_per_centrality(
        particle, SQRTS, qp_base, gl, CENT_BINS,
        P_EDGES, (y0, y1),
        show_components=("loss", "broad", "total", "npdf"),
        extra_bands=all_bands_pt,
        suptitle=f"$J/\\psi$ d+Au {ENERGY} GeV ( {name} )"
    )
    fig_pt.savefig(OUTPUT_DIR / f"plot_rhic_rpa_vs_pT_{sname}.png", dpi=150)
    plt.close(fig_pt)


# ------------------------------------------------------------------
# 6. Step-by-Step: RpA vs Centrality
# ------------------------------------------------------------------
print("\n--- Step 3: RpA vs Centrality ---")
pt_range_cent = (0.0, 5.0)

for y0, y1, name in Y_WINDOWS:
    print(f"  > Window: {name}")
    sname = name.replace(" ", "_")
    
    # 1. nPDF
    width_weights = np.array([wcent[f"{int(a)}-{int(b)}%"] for a,b in CENT_BINS])
    npdf_res_cent = bin_rpa_vs_centrality(
        df49_ctx, sys_npdf.df_pp, sys_npdf.df_pa, gluon,
        cent_bins=CENT_BINS, y_window=(y0, y1), pt_range_avg=pt_range_cent,
        width_weights=width_weights
    )
    
    labels = [f"{int(a)}-{int(b)}%" for a,b in CENT_BINS]
    
    RNP_c_arr = npdf_res_cent["r_central"]
    RNP_lo_arr = npdf_res_cent["r_lo"]
    RNP_hi_arr = npdf_res_cent["r_hi"]
    RMB_npdf = (npdf_res_cent["mb_r_central"], npdf_res_cent["mb_r_lo"], npdf_res_cent["mb_r_hi"])
    
    # 2. Eloss/Broad
    (
        res_labels,
        RL_c_dict, RL_lo_dict, RL_hi_dict,
        RB_c_dict, RB_lo_dict, RB_hi_dict,
        RT_c_dict, RT_lo_dict, RT_hi_dict,
        RMB_loss, RMB_broad, RMB_tot
    ) = rpa_band_vs_centrality(
        particle, SQRTS, qp_base, gl, CENT_BINS,
        (y0, y1), pt_range_cent,
        q0_pair=Q0_PAIR, p0_scale_pair=P0_SCALE,
        mb_weight_mode="exp",
        Ny_bin=3, Npt_bin=4
    )
    
    # 3. Combine
    def dict_to_arr(d): return np.array([d[l] for l in labels])
    RT_c_arr = dict_to_arr(RT_c_dict)
    RT_lo_arr = dict_to_arr(RT_lo_dict)
    RT_hi_arr = dict_to_arr(RT_hi_dict)
    
    RCNM_c_arr, RCNM_lo_arr, RCNM_hi_arr = combine_two_bands_1d(
        RT_c_arr, RT_lo_arr, RT_hi_arr,
        RNP_c_arr, RNP_lo_arr, RNP_hi_arr
    )
    RCNM_mb_c, RCNM_mb_lo, RCNM_mb_hi = combine_two_bands_1d(
        np.array([RMB_tot[0]]), np.array([RMB_tot[1]]), np.array([RMB_tot[2]]),
        np.array([RMB_npdf[0]]), np.array([RMB_npdf[1]]), np.array([RMB_npdf[2]])
    )
    RMB_cnm = (RCNM_mb_c[0], RCNM_mb_lo[0], RCNM_mb_hi[0])

    # 5. Plot
    def arr_to_dict(arr): return {l: v for l, v in zip(labels, arr)}
    RNP_c = arr_to_dict(RNP_c_arr)
    RNP_lo = arr_to_dict(RNP_lo_arr)
    RNP_hi = arr_to_dict(RNP_hi_arr)
    
    RCNM_c = arr_to_dict(RCNM_c_arr)
    RCNM_lo = arr_to_dict(RCNM_lo_arr)
    RCNM_hi = arr_to_dict(RCNM_hi_arr)

    fig_cent, _ = plot_RpA_vs_centrality_components_band(
        CENT_BINS, labels,
        RL_c=RL_c_dict, RL_lo=RL_lo_dict, RL_hi=RL_hi_dict, RMB_loss=RMB_loss,
        RB_c=RB_c_dict, RB_lo=RB_lo_dict, RB_hi=RB_hi_dict, RMB_broad=RMB_broad,
        RT_c=RT_c_dict, RT_lo=RT_lo_dict, RT_hi=RT_hi_dict, RMB_tot=RMB_tot,
        RNP_c=RNP_c, RNP_lo=RNP_lo, RNP_hi=RNP_hi, RMB_npdf=RMB_npdf,
        RCNM_c=RCNM_c, RCNM_lo=RCNM_lo, RCNM_hi=RCNM_hi, RMB_cnm=RMB_cnm,
        show=("loss", "broad", "total", "npdf", "cnm"),
        system_label="d+Au 200 GeV",
        note=f"{name}"
    )
    fig_cent.savefig(OUTPUT_DIR / f"plot_rhic_rpa_vs_cent_{sname}.png", dpi=150)
    plt.close(fig_cent)

print("\n=== Done ===")
