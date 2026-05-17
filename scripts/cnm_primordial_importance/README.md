# Bottomonia Primordial Importance-Sampling Analysis

This folder is intentionally isolated from the existing publication-ready
bottomonia and charmonia workflows. It is the CNM x primordial workflow.
Use `scripts/primordial_importance/` for primordial-only plots.

It reads TAMU primordial `datafile.gz` files directly and supports both formats:

- old 7-column trajectory metadata rows: uses uniform event weight `1.0`
- new 8-column trajectory metadata rows: uses column 8 as the physical
  importance-sampling weight

The analysis writes only to `outputs/cnm_primordial_importance/` by default.

## Run

```bash
python scripts/cnm_primordial_importance/run_bottomonia_primordial_importance.py
```

For primordial-only outputs from this implementation, either run the dedicated
wrapper in `scripts/primordial_importance/` or pass `--no-cnm`.

For a quick validation run:

```bash
python scripts/cnm_primordial_importance/run_bottomonia_primordial_importance.py --smoke
```

## Outputs

For each configured system and model variant, the workflow writes CSV tables and
figures for:

- `R_AA` vs rapidity in each centrality bin plus an MB panel
- `R_AA` vs `pT` in midrapidity and forward rapidity, in each centrality bin
  plus an MB panel
- `R_AA` vs centrality in midrapidity and forward rapidity

Bottomonia `pT` bins cover `0-20 GeV` for LHC O+O 5.36 TeV and
`0-15 GeV` for RHIC O+O 200 GeV.

Figures use the same centrality-grid convention as the charmonia primordial
workflow: step curves (`where="post"`) with dashed state lines and hatched
uncertainty bands, plus a dedicated legend/energy panel for the `y` and `pT`
centrality grids.

CNM is computed live inside this workflow; it is not read back from previously
written CNM CSV outputs. The LHC O+O 5.36 TeV branch imports the existing
bottomonia OO CNM engine and evaluates total CNM as
`nPDF x ELoss x pT broadening`. The RHIC O+O 200 GeV branch builds the matching
OO200 CNM context from the repo CNM modules and applies the RHIC nuclear
absorption factor, so its total is `nPDF x ELoss x pT broadening x absorption`.
CNM is evaluated in the same centrality bins as primordial:
`0-10`, `10-20`, `20-40`, `40-60`, `60-80`, `80-100`; MB is recomputed with the
workflow MB windows (`0-100%` LHC, `0-80%` RHIC).

The combined band is formed as a central-value product with relative
uncertainties added in quadrature.

Plot convention:

- gray dash-dot: CNM-only factor
- dashed: `TAMU-P Prim` or `TAMU-NP Prim`
- solid: `CNM x TAMU-P Prim` or `CNM x TAMU-NP Prim`
- no plot titles; energy and rapidity-window labels are placed inside panels
- when `1S`, `2S`, and `3S` are shown together they use distinct colors
- when states are separated into subplots, primordial and CNM-multiplied curves
  use the same green with dashed/solid line styles
