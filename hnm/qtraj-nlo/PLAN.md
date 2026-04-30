# QTraj-NLO PbPb 5 TeV Campaign Plan

## Goal
PhD-thesis-level bottomonium suppression analysis for PbPb 5 TeV using qtraj-nlo, covering all available potentials, all impact parameters, kappa scanning, quantum jump studies, wavefunction evolution, and vacuum eigenstate validation.

## Physics Parameters

### Collision System
- **System**: PbPb at 5 TeV
- **Temperature file**: `input/temperature/PbPb_5_TeV/T_center_vs_tau.csv`
- **Impact parameters**: ALL b values in file (0.00, 2.32, 4.25, 6.01, 7.78, 9.21, 10.45, 11.55, 11.60, 12.56, 12.60, 13.49, 13.50, 14.38, 14.40, 15.66, 15.70 fm)
- **t0** = 0 (start from vacuum)
- **tmed** = 3.0406390839551767 (0.6 fm/c medium turn-on)

### States
| State | initN | initL | initC |
|-------|-------|-------|-------|
| 1S | 1 | 0 | 0 |
| 2S | 2 | 0 | 0 |
| 3S | 3 | 0 | 0 |
| 1P | 2 | 1 | 0 |
| 2P | 3 | 1 | 0 |
| Octet-S | 1 | 0 | 1 |
| Octet-P | 2 | 1 | 1 |

### Potentials
| # | Name | Jumps? | Description |
|---|------|--------|-------------|
| 0 | Munich | YES | V_Re = -α/r + ½γT³r² (singlet), V_Im = -½κT³r² |
| 1 | Isotropic KSU | NO | Cornell vacuum + Debye-screened medium (internal energy) |
| 2 | Anisotropic KSU | NO | KSU with anisotropic Debye mass (ax,ay,az) |

### Kappa Values (LHC-appropriate)
- **kappa = 2, 3, 4, 5, 6** (central band for LHC PbPb 5 TeV)
- Also test **kappa = -1** (temperature-dependent central fit from lattice)

### Quantum Trajectory Convergence
- **NQTRAJ = 20, 40, 100** (study convergence)

### Jump Study
- **doJumps = 0** (no jumps, pure non-Hermitian evolution)
- **doJumps = 1** (with quantum jumps, MCWF method)

### Grid/Stepper (NLO)
- **stepper = 2** (Crank-Nicholson NLO E/T)
- **num = 2048** grid points
- **L = 40** (1/GeV simulation box)
- **dt = 0.001** (1/GeV time step)
- **maxSteps = 80000**

## Campaign Phases

### Phase 1: Vacuum Eigenstate Validation
- **Goal**: Verify vacuum eigenstates match known bottomonium spectrum
- **Config**: initType=100, projType=1, temperatureEvolution=3 (constant T=0.001 ≈ vacuum), doJumps=0
- **States**: 1S, 2S, 3S, 1P, 2P
- **Output**: Eigenenergies, wavefunctions, comparison with PDG values

### Phase 2: Munich Potential Main Production
- **Goal**: Primary suppression results with Munich potential
- **Matrix**: 5 states × 2 jump configs × 5 kappa values × all b values × 3 NQTRAJ
- **This is the largest phase**: ~5 × 2 × 5 × 17 × 3 = 2550 unique configurations

### Phase 3: KSU Potential Production
- **Goal**: Compare with KSU potentials (no jumps possible)
- **Matrix**: 5 states × 2 potentials (iso/aniso) × all b values × 3 NQTRAJ
- **~5 × 2 × 17 × 3 = 510 configurations**

### Phase 4: Wavefunction Evolution Snapshots
- **Goal**: Track |ψ(r,t)|² evolution for PhD figures
- **Config**: saveWavefunctions=1, snapFreq=500, snapPts=1024
- **Select representative configs**: central b=0, kappa=4, with/without jumps, all states
- **Both Munich and KSU potentials**

### Phase 5: Analysis & Visualization
- Survival probability bands vs time (τ [fm/c])
- State-by-state suppression factors R_AA-like ratios
- Kappa dependence plots
- Jump vs no-jump comparison
- Wavefunction evolution animations/snapshots
- Impact parameter dependence
- Potential comparison plots

### Phase 6: HEPData Export
- CSV files in HEPData format for all results
- Metadata files for submission

## Output Organization

```
qtraj-nlo/
├── campaigns/
│   ├── phase1_vacuum/
│   ├── phase2_munich/
│   │   ├── kappa_2/
│   │   │   ├── noJumps/
│   │   │   │   ├── b_0.00/
│   │   │   │   │   ├── state_1S/
│   │   │   │   │   │   ├── nq20/
│   │   │   │   │   │   ├── nq40/
│   │   │   │   │   │   └── nq100/
│   │   │   │   │   ├── state_2S/
│   │   │   │   │   └── ...
│   │   │   │   └── b_2.32/
│   │   │   └── withJumps/
│   │   ├── kappa_3/
│   │   └── ...
│   ├── phase3_ksu/
│   ├── phase4_wavefunctions/
│   └── phase5_nqtraj_convergence/
├── analysis/
│   ├── survival_plots/
│   ├── wavefunction_plots/
│   ├── kappa_scan_plots/
│   ├── potential_comparison/
│   └── hepdata_csv/
└── scripts/
    ├── run_campaign.py
    ├── helpers/
    │   └── collect_outputs.py
    └── analyze_outputs/
        ├── plot_style.py
        ├── analyze_survival.py
        ├── analyze_wavefunction.py
        ├── analyze_vacuum.py
        ├── analyze_kappa_scan.py
        ├── analyze_nqtraj_convergence.py
        └── export_hepdata.py
```

## Key Figures for PhD Thesis

1. **Vacuum eigenstates**: |ψ_nℓ(r)|² vs r with energy levels
2. **Survival probability bands**: S(τ) vs τ [fm/c] for each state (with/without jumps)
3. **Kappa dependence**: R_AA vs κ̂ for each state at fixed τ
4. **Impact parameter dependence**: R_AA vs b [fm] for each state
5. **Wavefunction evolution**: |ψ(r,τ)|² snapshots at multiple τ values
6. **Potential comparison**: Munich vs KSU vs KSU-aniso survival curves
7. **NQTRAJ convergence**: Error bars vs NQTRAJ for each state
8. **Octet-to-singlet transition**: Jump-induced transitions visualization
