# Bottomonia Primordial Importance-Sampling Analysis

This folder runs the weighted primordial-only workflow. It intentionally adds
`--no-cnm` and writes to `outputs/primordial_importance/` by default.

```bash
python scripts/primordial_importance/run_bottomonia_primordial_importance.py
```

Use `scripts/cnm_primordial_importance/` for the CNM x primordial workflow.

The pT grids are system-specific: LHC uses 0-20 GeV and RHIC uses 0-15 GeV.
