#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test to debug where the script hangs.
"""

from pathlib import Path
import sys
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# List of directories to add to sys.path
paths_to_add = [
    "cnm/eloss_code",
    "cnm/cnm_combine",
    "cnm/npdf_code",
]

for d in reversed(paths_to_add):
    p = str(ROOT / d)
    if p not in sys.path:
        sys.path.insert(0, p)

print("✓ Imports setup", flush=True)

from npdf_OO_data import load_OO_dat, build_OO_rpa_grid
print("✓ Loaded npdf_OO_data", flush=True)

from glauber import OpticalGlauber, SystemSpec
print("✓ Loaded glauber", flush=True)

from gluon_ratio import GluonEPPSProvider, EPPS21Ratio
print("✓ Loaded gluon_ratio", flush=True)

from npdf_centrality import compute_df49_by_centrality
print("✓ Loaded npdf_centrality", flush=True)

from particle import Particle
print("✓ Loaded particle", flush=True)

from coupling import alpha_s_provider
print("✓ Loaded coupling", flush=True)

import quenching_fast as QF
print("✓ Loaded quenching_fast", flush=True)

from cnm_combine_fast_nuclabs import CNMCombineFast
print("✓ Loaded cnm_combine_fast_nuclabs", flush=True)

# ── Config ───────────────────────────────────────────────────────────
ENERGIES = ["5.36"]
SQRTS_GEV = {"5.36": 5360.0}
SIG_NN_MB = {"5.36": 68.0}
M_UPSILON = 9.46
M_UPSILON_AVG = 10.01

CENT_BINS = [(0,10),(10,30),(30,50),(50,70),(70,100)]
MB_C0 = 0.25

Y_EDGES = np.arange(-5.0, 5.0 + 1.0, 1.0)
P_EDGES = np.arange(0.1, 25.1, 1.0)
PT_RANGE_AVG = (0.1, 25.0)

Y_WINDOWS = [
    (-5.0, -2.5, r"Backward: $-5.0 < y < -2.5$"),
    (-2.4,  2.4, r"Mid: $-2.4 < y < 2.4$"),
    ( 2.5,  5.0, r"Forward: $2.5 < y < 5.0$"),
]

Q0_PAIR = (0.05, 0.09)
P0_SCALE_PAIR = (0.9, 1.1)

energy = "5.36"
print(f"\n[INFO] Loading O+O @ {energy} TeV ...", flush=True)

SQRT_SNN = SQRTS_GEV[energy]
SIG_NN = SIG_NN_MB[energy]

OO_DAT = ROOT / "inputs" / "npdf" / "OxygenOxygen5360" / "nPDF_OO.dat"
print(f"  Loading data from {OO_DAT}", flush=True)
data = load_OO_dat(str(OO_DAT))
print(f"  ✓ Data loaded, building grid ...", flush=True)
grid = build_OO_rpa_grid(data, pt_max=20.0)
print(f"  ✓ Grid built", flush=True)

print(f"  Building Glauber ...", flush=True)
gl = OpticalGlauber(SystemSpec("AA", SQRT_SNN, A=16, sigma_nn_mb=SIG_NN), nx_pa=64, ny_pa=64, verbose=False)
print(f"  ✓ Glauber built", flush=True)

r0 = grid["r_central"].to_numpy()
M = grid[[f"r_mem_{i:03d}" for i in range(1, 49)]].to_numpy().T
SA_all = np.vstack([r0[None, :], M])
print(f"  ✓ Grid arrays prepared", flush=True)

print(f"  Loading EPPS21 ...", flush=True)
epps_wrapper = EPPS21Ratio(A=16, path=str(ROOT / "inputs" / "npdf" / "nPDFs"))
gluon_provider = GluonEPPSProvider(epps_wrapper, SQRT_SNN, m_state_GeV=M_UPSILON_AVG)
print(f"  ✓ EPPS21 loaded", flush=True)

print(f"  Computing df49 by centrality ...", flush=True)
df49_by_cent, K_by_cent, _, Y_SHIFT = compute_df49_by_centrality(
    grid, r0, M, gluon_provider, gl,
    cent_bins=CENT_BINS, nb_bsamples=5, kind="AA", SA_all=SA_all
)
print(f"  ✓ df49 computed", flush=True)

npdf_ctx = dict(df49_by_cent=df49_by_cent, df_pp=grid, df_pa=grid, gluon=gluon_provider)
particle = Particle(family="bottomonia", state="avg", mass_override_GeV=M_UPSILON)
alpha_s = alpha_s_provider(mode="running", LambdaQCD=0.25)
Lmb = gl.leff_minbias_AA()
print(f"  ✓ Particle and alpha_s initialized", flush=True)

device = "cpu"
qp_base = QF.QuenchParams(
    qhat0=0.075, lp_fm=1.5,
    LA_fm=Lmb, LB_fm=Lmb,
    system="AA", lambdaQCD=0.25, roots_GeV=SQRT_SNN,
    alpha_of_mu=alpha_s, alpha_scale="mT",
    use_hard_cronin=True, mapping="exp", device=device
)
print(f"  ✓ QuenchParams initialized", flush=True)

print(f"  Building CNMCombineFast ...", flush=True)
cnm = CNMCombineFast(
    energy=energy, family="bottomonia", particle_state="avg",
    sqrt_sNN=SQRT_SNN, sigma_nn_mb=SIG_NN,
    cent_bins=CENT_BINS, y_edges=Y_EDGES, p_edges=P_EDGES,
    y_windows=Y_WINDOWS, pt_range_avg=PT_RANGE_AVG, pt_floor_w=1.0,
    weight_mode="flat", y_ref=0.0, cent_c0=MB_C0,
    q0_pair=Q0_PAIR, p0_scale_pair=P0_SCALE_PAIR, nb_bsamples=5,
    y_shift_fraction=1.0, particle=particle,
    npdf_ctx=npdf_ctx, gl=gl, qp_base=qp_base, spec=gl.spec
)
print(f"  ✓ CNMCombineFast initialized", flush=True)

print(f"\n✓ SUCCESS: All loading completed!")
