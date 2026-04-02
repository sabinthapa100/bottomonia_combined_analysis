# Bottomonia Combined Analysis

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This repository contains the bottomonium heavy-ion analysis workflow developed for publication and PhD-thesis figure production.

The current stable production path for **qtraj-nlo hot nuclear matter (HNM)** bottomonium observables is:

- analysis engine: [hnm/qtraj_out_analysis](hnm/qtraj_out_analysis)
- parent-level production runner: [hnm/qtraj_production_scripts/run_qtraj_production.py](hnm/qtraj_production_scripts/run_qtraj_production.py)

This path is designed to:

- work from the repository root
- leave `qtraj_nlo` untouched
- use only the modular `qtraj_out_analysis` package for theory and experimental bundle construction
- place outputs in a thesis/publication-oriented tree under `outputs/qtraj_outputs/`

## Current Scope

The stable qtraj HNM production pipeline currently covers the published bottomonium comparison systems used in the local `arXiv-2305.17841v2` bundle:

- `AuAu 200 GeV`
- `PbPb 2.76 TeV`
- `PbPb 5.02 TeV`

Supported published observable families:

- `R_AA` vs `N_part`
- `R_AA` vs `p_T`
- `R_AA` vs `y`
- published double ratios for `PbPb 5.02 TeV`

Current published bundle count:

- `AuAu 200 GeV`: 3 observable bundles
- `PbPb 2.76 TeV`: 3 observable bundles
- `PbPb 5.02 TeV`: 7 observable bundles

Total:

- 13 production bundles

## Important Design Rule

`qtraj_nlo` is treated as an input/upstream source.

Production scripts do **not** modify:

- [hnm/qtraj-nlo](hnm/qtraj-nlo)

All current thesis/publication production logic lives in:

- [hnm/qtraj_out_analysis](hnm/qtraj_out_analysis)
- [hnm/qtraj_production_scripts](hnm/qtraj_production_scripts)

## Quick Start

Minimal runtime requirements for the current HNM production path:

- `python3`
- `numpy`
- `scipy`
- `matplotlib`

Run everything from the repository root:

```bash
python3 hnm/qtraj_production_scripts/run_qtraj_production.py --system all
```

Regenerate from a clean production tree (recommended for final figure runs):

```bash
python3 hnm/qtraj_production_scripts/run_qtraj_production.py --system all --clean
```

Run a single system:

```bash
python3 hnm/qtraj_production_scripts/run_qtraj_production.py --system pbpb5023
python3 hnm/qtraj_production_scripts/run_qtraj_production.py --system pbpb2760
python3 hnm/qtraj_production_scripts/run_qtraj_production.py --system auau200
```

Write data and manifests only:

```bash
python3 hnm/qtraj_production_scripts/run_qtraj_production.py --system all --skip-plots
```

Restrict to one or more theory sources:

```bash
python3 hnm/qtraj_production_scripts/run_qtraj_production.py --system pbpb5023 --kappa 3 4
python3 hnm/qtraj_production_scripts/run_qtraj_production.py --system auau200 --kappa 4 5
```

Override the output root:

```bash
python3 hnm/qtraj_production_scripts/run_qtraj_production.py \
  --system all \
  --output-root outputs/qtraj_outputs
```

You can also call the system-specific runners directly:

```bash
python3 hnm/qtraj_out_analysis/scripts/run_pbpb_5tev_production.py
python3 hnm/qtraj_out_analysis/scripts/run_pbpb_2760_production.py
python3 hnm/qtraj_out_analysis/scripts/run_auau_200gev_production.py
```

## Output Layout

The default production root is:

- [outputs/qtraj_outputs](outputs/qtraj_outputs)

The current collider/system layout is:

```text
outputs/qtraj_outputs/
  RHIC/
    AuAu200GeV/
      production/
        data/
          comparison/
            theory/
            theory_envelopes/
            experiment/
          theory_only/
            theory/
            theory_envelopes/
            experiment/
        figures/
          comparison/
          theory_only/
        manifests/
          summary.json
          summary.csv
          *.json
  LHC/
    PbPb2p76TeV/
      production/
        data/
        figures/
        manifests/
    PbPb5p02TeV/
      production/
        data/
        figures/
        manifests/
```

Examples:

- [AuAu 200 GeV summary](outputs/qtraj_outputs/RHIC/AuAu200GeV/production/manifests/summary.json)
- [PbPb 2.76 TeV summary](outputs/qtraj_outputs/LHC/PbPb2p76TeV/production/manifests/summary.json)
- [PbPb 5.02 TeV summary](outputs/qtraj_outputs/LHC/PbPb5p02TeV/production/manifests/summary.json)

## What The Production Pipeline Uses

The production bundles are assembled from three local source classes:

1. qtraj-nlo theory inputs in:
   - [inputs/qtraj_inputs](inputs/qtraj_inputs)
2. published Mathematica exports `.m` used as canonical published theory references
3. curated local experimental inputs in:
   - [inputs/experimental_data](inputs/experimental_data)

This makes the current production path a **published-reference reproduction pipeline**:

- exact observable registry
- exact local provenance
- exact published Mathematica x-grid mapping
- experimental overlays from local HEPData or local paper/notebook traces when required

Double-ratio overlays follow the paper style:

- `stat` and `sys` uncertainties are rendered separately (black stat bars + red sys bars)

## Glauber Handling

Canonical HNM Glauber handling is centralized in:

- [glauber.py](hnm/qtraj_out_analysis/src/glauber.py)

The module now contains canonical notebook-aligned `b -> N_part` mappings for:

- `auau200`
- `pbpb2760`
- `pbpb5023`

This means production and raw-analysis code do not need per-script hardcoded `npart` arrays anymore.

Useful module entry points:

- `load_canonical_glauber`
- `load_glauber_from_input_base`
- `infer_canonical_glauber_system`

## Main HNM Package Layout

```text
hnm/
  qtraj_out_analysis/
    src/
      io.py
      matching.py
      binning.py
      feeddown.py
      glauber.py
      observable_registry.py
      reference_data.py
      reference_output.py
      validation.py
    scripts/
      run_production.py
      run_pbpb_5tev_production.py
      run_pbpb_2760_production.py
      run_auau_200gev_production.py
      check_pbpb_5023_registry.py
  qtraj_production_scripts/
    run_qtraj_production.py
```

## Supported Systems And Observables

### AuAu 200 GeV

- `R_AA` vs `N_part`
- `R_AA` vs `p_T`
- `R_AA` vs `y`

Notes:

- `R_AA(y)` is currently routed as theory-only because the canonical local STAR bundle does not provide the matching rapidity-differential dataset.

### PbPb 2.76 TeV

- `R_AA` vs `N_part`
- `R_AA` vs `p_T`
- `R_AA` vs `y`

### PbPb 5.02 TeV

- `R_AA` vs `N_part`
- `R_AA` vs `p_T`
- `R_AA` vs `y`
- `(2S/1S)_{AA}/(2S/1S)_{pp}` vs `N_part`
- `(2S/1S)_{AA}/(2S/1S)_{pp}` vs `p_T`
- `(3S/1S)_{AA}/(3S/1S)_{pp}` vs `N_part`
- `(3S/2S)_{AA}/(3S/2S)_{pp}` vs `p_T`

## Accuracy And Validation Status

What is currently verified:

- canonical local source mapping for theory and experiment
- exact local Mathematica grid matching for the verified `PbPb 5.02 TeV` registry observables
- clean, publication-style figure generation from the repo root
- canonical HNM Glauber mapping for the three thesis systems

What the current production outputs represent:

- the published-theory comparison bundles are anchored to the local Mathematica exports used for the paper/reference figures

This is the correct path for final figure production when the goal is to reproduce the local published comparison set exactly.

## Practical Notes

- Work from the repository root.
- Use `hnm/qtraj_production_scripts/run_qtraj_production.py` as the main public entry point.
- Use `hnm/qtraj_out_analysis` as the only HNM qtraj analysis package.
- Do not use stale legacy modules under `hnm/qtraj_out_analysis/src/` that are not part of the production/reference path.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Contact

**Sabin Thapa** – [sthapa3@kent.edu](mailto:sthapa3@kent.edu)
