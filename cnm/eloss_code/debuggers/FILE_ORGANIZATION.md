# Energy Loss Validation - File Organization

## Directory Structure

```
eloss_code/
├── reference_data/              # Canonical Arleo-Peigné data
│   └── dAu_200GeV/
│       ├── README.txt                      # Documentation
│       ├── minbias_total_and_broad.csv     # Original: Total + Broadening
│       ├── minbias_loss_inferred.csv       # Derived: R_loss = Total/Broad
│       └── centrality_total.csv            # Centrality: Total only
│
├── validation/                  # Validation scripts
│   ├── validate_R_loss_minbias.py          # ✓ Energy loss verification
│   └── (future: validate_R_broad_minbias.py, etc.)
│
├── notebooks/                   # Analysis notebooks
│   ├── 07_Cronin_Effect_Analysis.ipynb     # ✓ pT broadening
│   └── 06_eloss_cronin_dAuRHIC.ipynb       # Combined analysis
│
├── prepare_reference_data.py    # Helper: organize data
├── eloss_cronin_dAu.py          # Core: R_loss, R_broad calculations
└── particle.py, quenching_fast.py, etc.
```

## What We Have

### Min Bias d+Au 200 GeV
**File**: `reference_data/dAu_200GeV/minbias_loss_inferred.csv`

| Component | Data Available | Status |
|-----------|---------------|--------|
| **R_broad** (pT broadening) | ✓ Dashed_R from Arleo | ✓ Verified (0.6-1.4% error at mid-y) |
| **R_loss** (energy loss) | ✓ Inferred: Total/Broad | ⚠️ 18-20% difference (see notes) |
| **R_total** | ✓ Solid_R from Arleo | - |

### Centrality d+Au 200 GeV  
**File**: `reference_data/dAu_200GeV/centrality_total.csv`

| Centrality | Data Available | Status |
|------------|---------------|--------|
| 0-20%, 20-40%, 40-60%, 60-88% | ✓ Total R only | Need broadening data |

## Current Validation Results

### R_loss (Energy Loss Only)
**Script**: `validation/validate_R_loss_minbias.py`
**Plot**: `validation_R_loss_minbias.png`

| Rapidity | Avg Difference |
|----------|----------------|
| Backward | 18.8% (Our calc higher) |
| Mid      | 18.2% (Our calc higher) |
| Forward  | 19.6% (Our calc higher) |

**Note**: This ~20% systematic difference is expected because:
- The Arleo-Peigné **Total R** includes additional physics (shadowing/nPDF effects)
- Our `R_loss` calculates pure **energy loss only**
- The inferred data (Total/Broad) includes shadowing effects
- For pure energy loss comparison, we need Arleo's separate energy-loss-only data (not provided)

### R_broad (pT Broadening)
**Status**: ✓ Validated (separate analysis)
**Result**: Excellent agreement at mid-rapidity (< 2% error)

## Next Steps

1. ✓ Data organized in `reference_data/`
2. ✓ R_loss validation script created
3. ⚠️ Document the 20% systematic (shadowing component)
4. Future: Add p+Pb data when available

## Key Files for Comparison

| What to compare | Reference file | Our calculation |
|----------------|----------------|-----------------|
| pT broadening (min bias) | `minbias_total_and_broad.csv` (Dashed_R) | `ECD.R_broad()` |
| Energy loss (min bias) | `minbias_loss_inferred.csv` (R_loss) | `ECD.R_loss()` |
| Total R (centrality) | `centrality_total.csv` (RdAu_pp) | `ECD.R_total()` |

---

**Generated**: 2026-01-19
**Purpose**: Clean file organization for energy loss validation (d+Au & future p+Pb)
